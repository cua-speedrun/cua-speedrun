"""In-process compute replicas for an evaluator with directly attached GPUs."""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Mapping, Sequence

from cua_speedrun.gateway import Gateway
from cua_speedrun.evaluation_runtime import (
    AgentExecutionError,
    ComputeInfrastructureError,
)
from cua_speedrun.local_runtime import local_agent_python

from .base import AgentResult, ComputeRunnerContext
from .local_sandbox import (
    LOCAL_SANDBOX_BASE_IMAGE,
    LOCAL_SANDBOX_RECIPE,
    LocalSandbox,
    LocalSandboxCache,
    local_sandbox_available,
)


AGENT_KILL_GRACE_SEC = 5.0


def _free_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stop_server(pid_file: Path) -> None:
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.monotonic() + AGENT_KILL_GRACE_SEC
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _device_ids(gpu: str | None, replica_count: int) -> list[str | None]:
    if not gpu:
        return [None] * replica_count
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [
            item.strip()
            for item in visible.split(",")
            if item.strip() and item.strip() not in ("-1", "void")
        ]
    else:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        devices = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    if len(devices) < replica_count:
        raise RuntimeError(
            f"local execution needs {replica_count} visible {gpu} GPU(s), "
            f"but CUDA exposes {len(devices)}: {devices or ['none']}"
        )
    return devices[:replica_count]


@dataclass
class LocalComputeReplica:
    index: int
    runtime_env: Mapping[str, str]
    server_pid_file: Path
    control_dir: Path | None = None
    device_id: str | None = None
    sandbox: LocalSandbox | None = None
    recovery_lock: RLock = field(default_factory=RLock, repr=False)


@dataclass
class LocalAgent:
    process: subprocess.Popen
    stdout_stream: object
    stderr_stream: object


class LocalComputeRunner:
    kind = "local"

    def __init__(self, context: ComputeRunnerContext) -> None:
        self.context = context
        self._runtime_dir: Path | None = None
        self._agent_python: Path | None = None
        self._sandbox_cache: LocalSandboxCache | None = None
        self._init_duration_sec = 0.0

    @property
    def identity(self) -> Mapping[str, object]:
        identity = {"kind": self.kind}
        if self.context.managed_runtime and local_sandbox_available():
            identity.update(
                {
                    "isolation": LOCAL_SANDBOX_RECIPE,
                    "base_image": LOCAL_SANDBOX_BASE_IMAGE,
                }
            )
        return identity

    @property
    def init_duration_sec(self) -> float:
        return self._init_duration_sec

    @property
    def environment_driver(self):
        return None

    def _create_runtime(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix=f"cs_agent_{self.context.run_id}_"))
        try:
            base_python = local_agent_python()
            subprocess.check_call([str(base_python), "-m", "venv", "--clear", str(root)])
            python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if self.context.python_packages:
                subprocess.check_call([
                    str(python), "-m", "pip", "install",
                    "--disable-pip-version-check", *self.context.python_packages,
                ])
            self._runtime_dir = root
            self._agent_python = python
            return python
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise

    def start_replicas(self, shards: Sequence[Sequence[object]]) -> list[LocalComputeReplica]:
        if self.context.managed_runtime and local_sandbox_available():
            return self._start_sandbox_replicas(shards)
        if self._agent_python is None:
            self._agent_python = (
                self._create_runtime()
                if self.context.managed_runtime
                else Path(os.path.abspath(sys.executable))
            )
        python = self._agent_python
        device_ids = _device_ids(self.context.gpu, len(shards))
        replicas: list[LocalComputeReplica] = []
        started = time.monotonic()
        try:
            for index, device_id in enumerate(device_ids, start=1):
                replica_started = time.monotonic()
                compute_dir = (
                    self.context.run_dir
                    if len(shards) == 1
                    else self.context.run_dir / "compute" / f"evaluation_{index}"
                )
                compute_dir.mkdir(parents=True, exist_ok=True)
                init_log = compute_dir / "init.log"
                server_pid_file = (compute_dir / "server.pid").resolve()
                local_port = _free_local_port()
                runtime_env = dict(self.context.base_runtime_env)
                runtime_env.update({
                    "CS_LOCAL_EXECUTION": "1",
                    "CS_SERVER_PID_FILE": str(server_pid_file),
                    "VLLM_PORT": str(local_port),
                    "VLLM_LOG_PATH": str((compute_dir / "model-server.log").resolve()),
                    "PYTHONPATH": os.pathsep.join(filter(None, (
                        str(Path(__file__).resolve().parents[2]),
                        runtime_env.get("PYTHONPATH", ""),
                    ))),
                })
                if self._runtime_dir is not None:
                    runtime_env.update({
                        "VIRTUAL_ENV": str(self._runtime_dir),
                        "PATH": (
                            f"{self._runtime_dir / 'bin'}{os.pathsep}"
                            f"{runtime_env.get('PATH', '')}"
                        ),
                    })
                if device_id is None:
                    runtime_env["CUDA_VISIBLE_DEVICES"] = ""
                    runtime_env["NVIDIA_VISIBLE_DEVICES"] = "void"
                else:
                    runtime_env["CUDA_VISIBLE_DEVICES"] = device_id
                    runtime_env["NVIDIA_VISIBLE_DEVICES"] = device_id
                self.context.events.emit(
                    "init_started", backend="local", execution_mode="local",
                    parallel_evaluation=index, compute_device=device_id,
                )
                with init_log.open("w") as stream:
                    process = subprocess.Popen(
                        [str(python), str(self.context.submission.init_script)],
                        cwd=self.context.submission.submission_dir,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        env=runtime_env,
                    )
                    returncode = process.wait()
                for line in init_log.read_text(errors="replace").splitlines(keepends=True):
                    self.context.events.emit(
                        "init_line", line=line, parallel_evaluation=index
                    )
                if returncode:
                    _stop_server(server_pid_file)
                    raise RuntimeError(
                        f"init.py for local evaluation {index} exited with code "
                        f"{returncode}, see {init_log}"
                    )
                if server_pid_file.is_file():
                    runtime_env["VLLM_URL"] = f"http://127.0.0.1:{local_port}"
                replicas.append(LocalComputeReplica(index, runtime_env, server_pid_file))
                self.context.events.emit(
                    "init_done",
                    duration_sec=time.monotonic() - replica_started,
                    phases={
                        "parallel_evaluation": index,
                        "init_py_sec": time.monotonic() - replica_started,
                    },
                    parallel_evaluation=index,
                )
        except BaseException:
            for replica in replicas:
                _stop_server(replica.server_pid_file)
            raise
        self._init_duration_sec = time.monotonic() - started
        return replicas

    def _start_sandbox_replicas(
        self, shards: Sequence[Sequence[object]]
    ) -> list[LocalComputeReplica]:
        if self._sandbox_cache is None:
            self._sandbox_cache = LocalSandboxCache(
                submission_dir=self.context.submission.submission_dir,
                submission_fingerprint=self.context.submission.fingerprint,
                base_python=local_agent_python(),
                python_packages=self.context.python_packages,
                runtime_env=self.context.base_runtime_env,
                gpu=self.context.gpu,
                run_id=self.context.run_id,
            )
        devices = _device_ids(self.context.gpu, len(shards))
        started = time.monotonic()
        replicas = []
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [
                pool.submit(
                    self._start_sandbox_replica, index, device, len(devices)
                )
                for index, device in enumerate(devices, start=1)
            ]
            failure = None
            for future in futures:
                try:
                    replicas.append(future.result())
                except BaseException as exc:
                    failure = failure or exc
            if failure is not None:
                for replica in replicas:
                    self.close_replica(replica)
                raise failure
        self._init_duration_sec = time.monotonic() - started
        return replicas

    def _start_sandbox_replica(
        self, index: int, device_id: str | None, replica_count: int
    ) -> LocalComputeReplica:
        assert self._sandbox_cache is not None
        started = time.monotonic()
        control = (
            self.context.run_dir
            if replica_count == 1
            else self.context.run_dir / "compute" / f"evaluation_{index}"
        )
        log = control / "init.log"
        log.unlink(missing_ok=True)
        self.context.events.emit(
            "init_started",
            backend="local",
            execution_mode="local",
            parallel_evaluation=index,
            compute_device=device_id,
        )
        sandbox = self._sandbox_cache.start_replica(
            index=index, control=control, log=log, device=device_id
        )
        for line in log.read_text(errors="replace").splitlines(keepends=True):
            self.context.events.emit("init_line", line=line, parallel_evaluation=index)
        duration = time.monotonic() - started
        self.context.events.emit(
            "init_done",
            duration_sec=duration,
            parallel_evaluation=index,
            phases={
                "parallel_evaluation": index,
                "init_py_sec": duration,
                "filesystem_cache": "hit" if sandbox.cache_hit else "stored",
            },
        )
        return LocalComputeReplica(
            index,
            sandbox.runtime_env,
            control / "server.pid",
            control,
            device_id,
            sandbox,
        )

    def gateway_hosts(self) -> tuple[str, str]:
        return "127.0.0.1", "127.0.0.1"

    def ensure_ready(self, replica: LocalComputeReplica) -> None:
        if replica.sandbox is not None and not replica.sandbox.running():
            raise ComputeInfrastructureError(
                f"local compute sandbox {replica.sandbox.name} exited",
                resource_id=replica.sandbox.name,
            )

    def replace_replica(
        self,
        replica: LocalComputeReplica,
        error: ComputeInfrastructureError,
    ) -> None:
        if replica.sandbox is not None and self._sandbox_cache is not None:
            with replica.recovery_lock:
                if error.resource_id and error.resource_id != replica.sandbox.name:
                    return
                assert replica.control_dir is not None
                replica.sandbox.stop()
                replica.sandbox = self._sandbox_cache.start_replica(
                    index=replica.index,
                    control=replica.control_dir,
                    log=replica.control_dir / "init.log",
                    device=replica.device_id,
                )
                replica.runtime_env = replica.sandbox.runtime_env
            return
        raise ComputeInfrastructureError(
            "directly attached compute cannot outlive its evaluator host",
            resource_id=error.resource_id,
        )

    def start_agent(
        self,
        replica: LocalComputeReplica,
        *,
        env_url: str,
        instruction: str,
        task_key: str,
        task_dir: Path,
        timeout_sec: float,
        append_output: bool = False,
    ) -> LocalAgent:
        output_mode = "a" if append_output else "w"
        stdout_stream = (task_dir / "agent.stdout").open(output_mode)
        stderr_stream = (task_dir / "agent.stderr").open(output_mode)
        python = replica.sandbox.agent_python if replica.sandbox else self._agent_python
        assert python is not None
        process = subprocess.Popen(
            [
                str(python),
                str(self.context.submission.agent_script),
                env_url,
                instruction,
            ],
            cwd=self.context.submission.submission_dir,
            stdout=stdout_stream,
            stderr=stderr_stream,
            env=(
                dict(replica.sandbox.agent_host_env)
                if replica.sandbox is not None
                else dict(replica.runtime_env)
            ),
        )
        return LocalAgent(process, stdout_stream, stderr_stream)

    def wait_for_gateway(
        self, replica: LocalComputeReplica, agent: LocalAgent, gateway: Gateway
    ) -> None:
        while True:
            status = gateway.status()
            if status.get("fully_done"):
                return
            if replica.sandbox is not None and not replica.sandbox.running():
                raise ComputeInfrastructureError(
                    f"local compute sandbox {replica.sandbox.name} exited",
                    resource_id=replica.sandbox.name,
                )
            returncode = agent.process.poll()
            if returncode is not None and not status.get("finished"):
                if status.get("continuation_pending"):
                    return
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    status = gateway.status()
                    if status.get("fully_done"):
                        return
                    if status.get("finished") or status.get("continuation_pending"):
                        break
                    time.sleep(0.25)
                if not status.get("finished") and not status.get(
                    "continuation_pending"
                ):
                    raise AgentExecutionError(
                        f"agent.py exited with code {returncode} before "
                        "finishing the task",
                        returncode=returncode,
                    )
                if status.get("continuation_pending"):
                    return
            time.sleep(0.5)

    def stop_agent(
        self, replica: LocalComputeReplica, agent: LocalAgent
    ) -> AgentResult:
        if agent.process.poll() is None:
            agent.process.terminate()
            try:
                agent.process.wait(timeout=AGENT_KILL_GRACE_SEC)
            except subprocess.TimeoutExpired:
                agent.process.kill()
                agent.process.wait()
        agent.stdout_stream.close()
        agent.stderr_stream.close()
        return AgentResult(returncode=int(agent.process.returncode or 0))

    def finish_replica(self, replica: LocalComputeReplica) -> None:
        return None

    def close_replica(self, replica: LocalComputeReplica) -> None:
        if replica.sandbox is not None:
            replica.sandbox.stop()
        else:
            _stop_server(replica.server_pid_file)

    def close(self) -> None:
        if self._runtime_dir is not None:
            shutil.rmtree(self._runtime_dir, ignore_errors=True)


__all__ = ["LocalComputeReplica", "LocalComputeRunner"]
