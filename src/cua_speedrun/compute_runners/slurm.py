"""Slurm compute runner: warm shared replicas or one allocation per instance.

Shared-agent evaluations keep one warm allocation per evaluation replica for
the whole run. Per-task evaluations instead submit one short, task-sized
allocation per instance, which keeps jobs inside scheduler backfill windows.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import shlex
import signal
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from threading import RLock
from typing import Any, Mapping, Sequence

import yaml

from cua_speedrun.evaluation_runtime import (
    AgentExecutionError,
    ComputeInfrastructureError,
    InstanceInfrastructureError,
)
from cua_speedrun.gateway import (
    Gateway,
    VERIFIER_TIMEOUT_SEC,
    monitor_gateway_steps,
)
from cua_speedrun.local_runtime import local_agent_python
from cua_speedrun.remote.control import GatewayControl
from cua_speedrun.runlog import summarize

from .base import AgentResult, ComputeRunnerContext


_VISIBILITY_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
)

_KVM_NODE_CAPABILITY_ERRORS = (
    "gym-anything-local needs /dev/kvm for accelerated local vms",
    "gym-anything-local found /dev/kvm but this user cannot open it",
    "could not use kvm. restore /dev/kvm access for this user",
)


def _is_kvm_node_capability_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _KVM_NODE_CAPABILITY_ERRORS)


def _kvm_exclude_path() -> Path | None:
    """KVM-less nodes are a cluster property; remember them across runs."""
    home = os.environ.get("CUA_SPEEDRUN_HOME", "").strip()
    if not home:
        return None
    return Path(home).expanduser() / "runners" / "slurm" / "kvm-excluded-nodes.json"


def _load_excluded_nodes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = _read_json(path) or {}
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return set()
    return {str(node) for node in nodes if str(node).strip()}


def _save_excluded_nodes(path: Path | None, nodes: set[str]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, {"nodes": sorted(nodes)})
    except OSError:
        # Persistence is an optimization; the in-run set still applies.
        pass


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True))
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _parse_runlog_text(text: str) -> list[dict[str, Any]]:
    # The run log delimiter is exactly newline. JSON strings may legally
    # contain NEL, U+2028, or U+2029, which str.splitlines would treat as
    # line breaks and split an event mid-string.
    events = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def _render(value: str, variables: Mapping[str, str], path: Path) -> str:
    try:
        return Template(value).substitute(variables)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{path}: invalid template placeholder: {exc}") from exc


def _scheduler_time(total_minutes: int) -> str:
    days, remainder = divmod(max(1, total_minutes), 24 * 60)
    hours, minutes = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:00"
    return f"{days}-{clock}" if days else clock


def _replica_time_limit_minutes(shard: Sequence[Any], concurrency: int) -> int:
    lanes = [0.0] * max(1, concurrency)
    for task, _seed in shard:
        lane = min(range(len(lanes)), key=lanes.__getitem__)
        lanes[lane] += float(task.timeout_sec)
    return max(1, math.ceil(max(lanes, default=0.0) / 60.0))


def _load_template(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    list_fields = ("submit_command", "step_command", "cancel_command")
    try:
        valid = (
            raw["schema_version"] == 1
            and raw["runner"] == "slurm"
            and isinstance(raw["name"], str)
            and bool(raw["name"].strip())
            and isinstance(raw["job"], dict)
            and isinstance(raw["script"], str)
            and raw.get("submit_mode", "per-replica") in ("per-replica", "array")
            and all(
                isinstance(raw[field], list)
                and bool(raw[field])
                and all(isinstance(item, str) for item in raw[field])
                for field in list_fields
            )
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ValueError(f"{path}: invalid Slurm runner-template schema")
    return raw


@dataclass
class SlurmComputeReplica:
    index: int
    shard: tuple[Any, ...]
    control_dir: Path
    submission_dir: Path
    log_path: Path
    job_id: str = ""
    cancel_command: tuple[str, ...] = ()
    attempt: int = 0
    ready: bool = False
    seen_init_lines: int = 0
    recovery_lock: RLock = field(default_factory=RLock, repr=False)


@dataclass
class SlurmAgent:
    process: subprocess.Popen
    stdout_stream: Any
    stderr_stream: Any
    task_key: str


@dataclass(frozen=True)
class SlurmPreparedEnvironment:
    handle: str
    scheduler_job_id: str
    task: Any
    seed: int
    task_key: str
    task_dir: Path
    control_url: str
    control_token: str
    agent_url: str
    instruction: str
    prepare_time_sec: float
    environment_info: Mapping[str, Any]
    environment_hostname: str


class SlurmComputeRunner:
    kind = "slurm"

    def __init__(self, context: ComputeRunnerContext, template_path: Path) -> None:
        self.context = context
        self.template_path = template_path.resolve()
        self.template = _load_template(self.template_path)
        # Array mode submits every replica as its own single-element array
        # (--array=1-1%1). Some schedulers grant array submissions their own
        # partition or QoS capacity; the runner keeps its one-submission-per-
        # replica model and only the job address changes, to <master>_1.
        self.array_mode = self.template.get("submit_mode") == "array"
        self._replicas: list[SlurmComputeReplica] = []
        self._closed_job_ids: set[str] = set()
        self._kvm_exclude_path = _kvm_exclude_path()
        self._excluded_nodes: set[str] = _load_excluded_nodes(
            self._kvm_exclude_path
        )
        self._init_duration_sec = 0.0
        self._instance_count = 0
        self._instance_lock = RLock()

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "kind": self.kind,
            "template": self.template["name"],
            "template_sha256": hashlib.sha256(
                self.template_path.read_bytes()
            ).hexdigest(),
        }

    @property
    def init_duration_sec(self) -> float:
        return self._init_duration_sec

    @property
    def environment_driver(self):
        return self

    def _job_values(
        self,
        index: int,
        shard: Sequence[Any],
        control_dir: Path,
        submission_dir: Path,
        log_path: Path,
    ) -> dict[str, str]:
        job = self.template["job"]
        cpu_map = job.get("cpus_per_replica") or {}
        if not isinstance(cpu_map, dict):
            raise ValueError(f"{self.template_path}: cpus_per_replica must be a map")
        mem_map = job.get("mem_per_replica") or {}
        if not isinstance(mem_map, dict):
            raise ValueError(f"{self.template_path}: mem_per_replica must be a map")
        try:
            cpus = int(cpu_map.get(self.context.track_name, cpu_map.get("default")))
            overhead = int(job.get("overhead_minutes", 30))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self.template_path}: invalid CPU or time value") from exc
        mem = str(
            mem_map.get(self.context.track_name, mem_map.get("default")) or ""
        ).strip()
        if cpus < 1 or overhead < 0:
            raise ValueError(f"{self.template_path}: invalid CPU or time value")
        gpu_type = ""
        if self.context.gpu:
            gpu_type = str((job.get("gpu_types") or {}).get(self.context.gpu) or "")
            if not gpu_type:
                raise ValueError(
                    f"{self.template_path}: no scheduler mapping for GPU "
                    f"{self.context.gpu!r}"
                )
        evaluation_minutes = _replica_time_limit_minutes(
            shard, self.context.agents_per_evaluation
        )
        worker = [
            os.path.abspath(sys.executable),
            "-m",
            "cua_speedrun.compute_runners.slurm_worker",
            "--control-dir",
            str(control_dir),
            "--submission-dir",
            str(submission_dir),
            "--base-python",
            str(local_agent_python()),
            # Every replica of the local topology hosts environment VMs, so
            # the worker refuses KVM-less nodes before reporting ready.
            "--require-kvm",
        ]
        if self.context.gpu:
            worker.extend(("--gpu", self.context.gpu))
        for package in self.context.python_packages:
            worker.extend(("--python-package", package))
        worker.extend(("--init-cache-key", self.context.init_cache_key))
        for name in self.context.submission_environment_names:
            worker.extend(("--runtime-environment-name", name))
        for name in sorted(self.context.evaluator_runtime_env):
            worker.extend(("--evaluator-environment-name", name))
        return {
            "evaluation_id": self.context.run_id,
            "replica": str(index),
            "track": self.context.track_name,
            "cpus": str(cpus),
            "mem": mem,
            "gpu_type": gpu_type,
            "evaluation_time_limit": _scheduler_time(evaluation_minutes),
            "job_time": _scheduler_time(evaluation_minutes + overhead),
            "control_dir": str(control_dir),
            "submission_dir": str(submission_dir),
            "log_path": str(log_path),
            "worker_command": shlex.join(worker),
        }

    def _make_replica(
        self, index: int, shard: Sequence[Any]
    ) -> SlurmComputeReplica:
        control_dir = (
            self.context.run_dir / "compute" / f"evaluation_{index}"
        ).resolve()
        submission_dir = control_dir / "submission"
        submission_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.context.submission.init_script, submission_dir / "init.py")
        shutil.copy2(self.context.submission.agent_script, submission_dir / "agent.py")
        log_path = control_dir / "slurm.log"
        return SlurmComputeReplica(
            index=index,
            shard=tuple(shard),
            control_dir=control_dir,
            submission_dir=submission_dir,
            log_path=log_path,
        )

    @staticmethod
    def _clean_replica_state(replica: SlurmComputeReplica) -> None:
        replica.attempt += 1
        replica.ready = False
        replica.seen_init_lines = 0
        shutil.rmtree(
            replica.control_dir / "environment_requests",
            ignore_errors=True,
        )
        (replica.control_dir / "environment_requests").mkdir(
            parents=True,
            exist_ok=True,
        )
        for name in ("heartbeat", "shutdown", "state.json", "server.pid"):
            (replica.control_dir / name).unlink(missing_ok=True)

    def _record_submitted(
        self,
        replica: SlurmComputeReplica,
        job_id: str,
        values: Mapping[str, str],
    ) -> None:
        rendered = {**values, "job_id": job_id}
        cancel_command = tuple(
            _render(item, rendered, self.template_path)
            for item in self.template["cancel_command"]
        )
        replica.job_id = job_id
        replica.cancel_command = cancel_command
        _atomic_json(replica.control_dir / "job.json", {
            "job_id": job_id,
            "runner": self.identity,
            "cancel_command": list(cancel_command),
            "attempt": replica.attempt,
        })
        self.context.events.emit(
            "compute_replica_created",
            backend="slurm",
            runner_template=self.template["name"],
            scheduler_job_id=job_id,
            cancel_command=list(cancel_command),
            attempt=replica.attempt,
            parallel_evaluation=replica.index,
        )

    def _submit_replica(self, replica: SlurmComputeReplica) -> None:
        self._clean_replica_state(replica)
        values = self._job_values(
            replica.index,
            replica.shard,
            replica.control_dir,
            replica.submission_dir,
            replica.log_path,
        )
        command = [
            _render(item, values, self.template_path)
            for item in self.template["submit_command"]
        ]
        if self.context.gpu:
            command.extend(
                _render(item, values, self.template_path)
                for item in self.template.get("gpu_arguments") or ()
            )
        if self._excluded_nodes:
            command.append(f"--exclude={','.join(sorted(self._excluded_nodes))}")
        submit_env = os.environ.copy()
        submit_env.update(self.context.base_runtime_env)
        submit_env.update(self.context.evaluator_runtime_env)
        result = subprocess.run(
            command,
            input=_render(self.template["script"], values, self.template_path),
            text=True,
            cwd=self.context.run_dir.resolve(),
            env=submit_env,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise ComputeInfrastructureError(
                f"Slurm template {self.template['name']!r} failed to submit "
                f"replica {replica.index}: "
                f"{detail or f'exit {result.returncode}'}"
            )
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id:
            raise ComputeInfrastructureError(
                "Slurm submit command did not print a job id"
            )
        if self.array_mode:
            # sbatch --parsable prints the array master id; the single
            # element is addressed as <master>_1 by squeue, scancel, sacct,
            # and the worker's own state reports.
            job_id = f"{job_id}_1"
        self._record_submitted(replica, job_id, values)

    @staticmethod
    def _scheduler_state(job_id: str) -> str:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    @staticmethod
    def _heartbeat_is_fresh(path: Path) -> bool:
        try:
            return time.time() - path.stat().st_mtime <= 30
        except OSError:
            return False

    @staticmethod
    def _sacct_snapshot(job_id: str) -> str:
        try:
            result = subprocess.run(
                ["sacct", "-n", "-P", "-j", job_id,
                 "--format=JobIDRaw,State,ExitCode,Reason,Elapsed,NodeList"],
                text=True, capture_output=True, timeout=15, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"{type(exc).__name__}: {exc}"
        return result.stdout.strip() or result.stderr.strip()

    def _running_replica_error(
        self,
        replica: SlurmComputeReplica,
    ) -> ComputeInfrastructureError | None:
        state = _read_json(replica.control_dir / "state.json") or {}
        state_job_id = str(state.get("scheduler_job_id") or "")
        heartbeat = replica.control_dir / "heartbeat"
        scheduler_state = self._scheduler_state(replica.job_id)
        heartbeat_fresh = self._heartbeat_is_fresh(heartbeat)
        worker_state = str(state.get("status") or "missing")
        if (
            state_job_id == replica.job_id
            and worker_state == "ready"
            and heartbeat_fresh
            and scheduler_state == "RUNNING"
        ):
            return None
        replica.ready = False
        return ComputeInfrastructureError(
            f"Slurm replica {replica.index} is unavailable "
            f"(job={replica.job_id}, state_job={state_job_id or 'none'}, "
            f"worker={worker_state}, "
            f"heartbeat={'fresh' if heartbeat_fresh else 'missing-or-stale'}, "
            f"scheduler={scheduler_state or 'absent'})",
            resource_id=replica.job_id,
        )

    def _agent_infrastructure_error(
        self,
        replica: SlurmComputeReplica,
        agent: SlurmAgent,
        returncode: int,
    ) -> ComputeInfrastructureError | None:
        if returncode < 0 or returncode in {
            128 + signal.SIGKILL,
            128 + signal.SIGTERM,
        }:
            replica.ready = False
            return ComputeInfrastructureError(
                f"Slurm terminated the agent job step with code {returncode}",
                resource_id=replica.job_id,
            )
        try:
            agent.stderr_stream.flush()
            stderr = Path(agent.stderr_stream.name).read_text(errors="replace")
        except (OSError, ValueError):
            stderr = ""
        scheduler_cancelled_step = (
            "slurmstepd: error: *** STEP " in stderr and " CANCELLED AT " in stderr
        )
        pending_allocation = (
            "Unable to confirm allocation for job" in stderr
            and "Job is pending execution" in stderr
        )
        completing_allocation = (
            "Unable to create step for job" in stderr
            and "Job/step already completing or completed" in stderr
        )
        if scheduler_cancelled_step or pending_allocation or completing_allocation:
            replica.ready = False
            return ComputeInfrastructureError(
                f"Slurm terminated the agent job step with code {returncode}",
                resource_id=replica.job_id,
            )
        return self._running_replica_error(replica)

    def _wait_ready(self, replica: SlurmComputeReplica, *, emit: bool) -> None:
        timeout_minutes = int(self.template["job"].get("start_timeout_minutes", 120))
        deadline = time.monotonic() + timeout_minutes * 60
        state_path = replica.control_dir / "state.json"
        while time.monotonic() < deadline:
            if emit and (replica.control_dir / "init.log").is_file():
                lines = (replica.control_dir / "init.log").read_text(
                    errors="replace"
                ).splitlines(keepends=True)
                for line in lines[replica.seen_init_lines:]:
                    self.context.events.emit(
                        "init_line", line=line, parallel_evaluation=replica.index
                    )
                replica.seen_init_lines = len(lines)
            state = _read_json(state_path) or {}
            state_job_id = str(state.get("scheduler_job_id") or "")
            current_state = state_job_id in {"", replica.job_id}
            heartbeat = replica.control_dir / "heartbeat"
            ready_state = (
                current_state
                and state.get("status") == "ready"
                and self._heartbeat_is_fresh(heartbeat)
            )
            scheduler_state = self._scheduler_state(replica.job_id)
            if ready_state and scheduler_state == "RUNNING":
                if emit and not replica.ready:
                    self.context.events.emit(
                        "init_done",
                        duration_sec=float(state.get("init_duration_sec", 0.0)),
                        execution_mode="scheduled",
                        parallel_evaluation=replica.index,
                    )
                replica.ready = True
                self._init_duration_sec = max(
                    self._init_duration_sec,
                    float(state.get("init_duration_sec", 0.0)),
                )
                return
            replica.ready = False
            if current_state and state.get("status") == "failed":
                error = str(state.get("error") or "Slurm replica failed")
                # A KVM-less node is a placement mistake, not a failure of
                # this evaluation: exclude the node and let replacement
                # resubmit the replica elsewhere.
                if _is_kvm_node_capability_error(error):
                    self._exclude_kvmless_node(
                        replica, str(state.get("hostname") or "")
                    )
                    raise ComputeInfrastructureError(
                        error,
                        resource_id=replica.job_id,
                    )
                # Signal-terminated workers were preempted or swept by the
                # scheduler and are replaceable infrastructure. Any other
                # failure (for example the submission's own init.py exiting
                # nonzero) is deterministic and must fail the run, not enter
                # an infrastructure retry loop.
                if "exited with code -15" in error or "exited with code 143" in error:
                    raise ComputeInfrastructureError(
                        error,
                        resource_id=replica.job_id,
                    )
                raise RuntimeError(error)
            if current_state and state.get("status") in {"interrupted", "stopped"}:
                raise ComputeInfrastructureError(
                    str(state.get("error") or f"Slurm job {replica.job_id} stopped"),
                    resource_id=replica.job_id,
                )
            if not scheduler_state:
                raise ComputeInfrastructureError(
                    f"Slurm job {replica.job_id} left the queue before "
                    "the compute replica became ready",
                    resource_id=replica.job_id,
                )
            if scheduler_state in {
                "BOOT_FAIL",
                "CANCELLED",
                "COMPLETED",
                "DEADLINE",
                "FAILED",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "PREEMPTED",
                "TIMEOUT",
            }:
                raise ComputeInfrastructureError(
                    f"Slurm job {replica.job_id} ended in {scheduler_state}",
                    resource_id=replica.job_id,
                )
            time.sleep(2)
        raise ComputeInfrastructureError(
            f"Slurm replica {replica.index} did not become ready within "
            f"{timeout_minutes} minutes",
            resource_id=replica.job_id,
        )

    def start_replicas(
        self, shards: Sequence[Sequence[Any]]
    ) -> list[SlurmComputeReplica]:
        started = time.monotonic()
        self.context.events.emit(
            "init_started",
            backend="slurm",
            execution_mode="scheduled",
            replica_count=len(shards),
        )
        try:
            self._replicas = [
                self._make_replica(index, shard)
                for index, shard in enumerate(shards, start=1)
            ]
            for replica in self._replicas:
                self._submit_replica(replica)
        except BaseException:
            for replica in self._replicas:
                self.close_replica(replica)
            raise
        self._init_duration_sec = time.monotonic() - started
        return list(self._replicas)

    def start_instance_replica(
        self, task: Any, seed: int
    ) -> SlurmComputeReplica:
        """Submit one task-sized allocation for a single isolated instance.

        The allocation's time limit covers exactly one task timeout plus the
        template's overhead budget, so per-task evaluations produce short
        jobs that fit scheduler backfill windows.
        """
        with self._instance_lock:
            self._instance_count += 1
            index = self._instance_count
        safe_task = str(task.task_id).replace("/", "_")
        control_dir = (
            self.context.run_dir
            / "compute"
            / f"instance_{index:04d}_{safe_task}_seed_{seed}"
        ).resolve()
        submission_dir = control_dir / "submission"
        submission_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            self.context.submission.init_script, submission_dir / "init.py"
        )
        shutil.copy2(
            self.context.submission.agent_script, submission_dir / "agent.py"
        )
        replica = SlurmComputeReplica(
            index=index,
            shard=((task, seed),),
            control_dir=control_dir,
            submission_dir=submission_dir,
            log_path=control_dir / "slurm.log",
        )
        self._submit_replica(replica)
        return replica

    def gateway_hosts(self) -> tuple[str, str]:
        advertised = os.environ.get("CS_GATEWAY_ADVERTISE_HOST", "").strip()
        advertised = advertised or socket.getfqdn() or socket.gethostname()
        if advertised in {"localhost", "localhost.localdomain", "127.0.0.1"}:
            raise RuntimeError(
                "Slurm compute jobs need a routable environment-gateway host; "
                "set CS_GATEWAY_ADVERTISE_HOST"
            )
        return "0.0.0.0", advertised

    def ensure_ready(self, replica: SlurmComputeReplica) -> None:
        self._wait_ready(replica, emit=not replica.ready)

    def _exclude_kvmless_node(
        self, replica: SlurmComputeReplica, hostname: str
    ) -> None:
        hostname = hostname.strip()
        if not hostname or hostname in self._excluded_nodes:
            return
        self._excluded_nodes.add(hostname)
        _save_excluded_nodes(self._kvm_exclude_path, self._excluded_nodes)
        self.context.events.emit(
            "compute_node_excluded",
            hostname=hostname,
            reason="dev_kvm_unavailable",
            parallel_evaluation=replica.index,
        )

    @staticmethod
    def _archive_replica_state(replica: SlurmComputeReplica, archive: Path) -> None:
        archive.mkdir(parents=True, exist_ok=True)
        for name in (
            "heartbeat",
            "init.log",
            "job.json",
            "model-server.log",
            "server.pid",
            "slurm.log",
            "state.json",
            "shutdown",
        ):
            source = replica.control_dir / name
            if source.exists():
                source.replace(archive / name)
        requests = replica.control_dir / "environment_requests"
        if requests.exists():
            requests.replace(archive / "environment_requests")
        shutil.rmtree(replica.control_dir / "runtime", ignore_errors=True)

    def replace_replica(
        self,
        replica: SlurmComputeReplica,
        error: ComputeInfrastructureError,
    ) -> None:
        with replica.recovery_lock:
            if error.resource_id and error.resource_id != replica.job_id:
                return
            previous_attempt = replica.attempt
            previous_job_id = replica.job_id
            self._cancel_current_job(
                replica,
                reason=f"infrastructure_replacement: {error}",
            )
            archive = (
                replica.control_dir / "infra_attempts" / f"attempt_{previous_attempt}"
            )
            self._archive_replica_state(replica, archive)
            self.context.events.emit(
                "compute_replica_replaced",
                previous_scheduler_job_id=previous_job_id,
                previous_attempt=previous_attempt,
                error=repr(error),
                parallel_evaluation=replica.index,
            )
            self._submit_replica(replica)

    def _environment_request(
        self,
        replica: SlurmComputeReplica,
        payload: Mapping[str, Any],
        *,
        timeout_sec: float,
    ) -> dict[str, Any]:
        self.ensure_ready(replica)
        scheduler_job_id = replica.job_id
        request_id = secrets.token_hex(12)
        directory = replica.control_dir / "environment_requests"
        request_path = directory / f"{request_id}.request.json"
        response_path = directory / f"{request_id}.response.json"
        _atomic_json(request_path, {
            **dict(payload),
            "scheduler_job_id": scheduler_job_id,
        })
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if replica.job_id != scheduler_job_id:
                raise ComputeInfrastructureError(
                    "Slurm environment request belonged to a replaced replica",
                    resource_id=scheduler_job_id,
                )
            response = _read_json(response_path)
            if response is not None:
                if str(response.get("scheduler_job_id") or "") != scheduler_job_id:
                    raise ComputeInfrastructureError(
                        "Slurm environment response came from a replaced replica",
                        resource_id=scheduler_job_id,
                    )
                if not response.get("ok"):
                    error = str(response.get("error") or "unknown worker error")
                    if _is_kvm_node_capability_error(error):
                        state = _read_json(replica.control_dir / "state.json") or {}
                        hostname = str(state.get("hostname") or "").strip()
                        if (
                            str(state.get("scheduler_job_id") or "")
                            == scheduler_job_id
                        ):
                            self._exclude_kvmless_node(replica, hostname)
                        raise ComputeInfrastructureError(
                            f"Slurm node {hostname or 'unknown'} cannot provide "
                            f"/dev/kvm: {error}",
                            resource_id=scheduler_job_id,
                        )
                    raise InstanceInfrastructureError(
                        "Slurm replica could not prepare its environment: "
                        f"{error}"
                    )
                return response
            infrastructure_error = self._running_replica_error(replica)
            if infrastructure_error is not None:
                raise infrastructure_error
            time.sleep(0.5)
        raise InstanceInfrastructureError(
            f"Slurm replica environment operation did not finish within "
            f"{timeout_sec:g}s"
        )

    def prepare_environment(
        self,
        replica: SlurmComputeReplica,
        *,
        backend_name: str,
        task: Any,
        seed: int,
    ) -> SlurmPreparedEnvironment:
        task_key = f"{task.task_id}/seed_{seed}"
        run_dir = self.context.run_dir.resolve()
        task_dir = run_dir / "tasks" / task.task_id / f"seed_{seed}"
        task_dir.mkdir(parents=True, exist_ok=True)
        handle = f"env_{replica.index}_{secrets.token_hex(10)}"
        run_token = secrets.token_urlsafe(24)
        control_token = secrets.token_urlsafe(24)
        response = self._environment_request(
            replica,
            {
                "operation": "prepare",
                "handle": handle,
                "backend_name": backend_name,
                "run_dir": str(run_dir),
                "run_id": self.context.run_id,
                "seed": seed,
                "run_token": run_token,
                "control_token": control_token,
                "task": {
                    "task_id": task.task_id,
                    "description": task.description,
                    "env": task.env,
                    "timeout_sec": task.timeout_sec,
                    "grace_sec": task.grace_sec,
                    "metadata": task.metadata,
                    "generator": task.generator,
                    "task_dir": str(task.task_dir) if task.task_dir else None,
                },
            },
            timeout_sec=float(
                int(self.template["job"].get("start_timeout_minutes", 120)) * 60
            ),
        )
        return SlurmPreparedEnvironment(
            handle=handle,
            scheduler_job_id=replica.job_id,
            task=task,
            seed=seed,
            task_key=task_key,
            task_dir=task_dir,
            control_url=str(response["control_url"]),
            control_token=control_token,
            agent_url=str(response["agent_url"]),
            instruction=str(response["instruction"]),
            prepare_time_sec=float(response["prepare_time_sec"]),
            environment_info=dict(response.get("environment_info") or {}),
            environment_hostname=str(response["environment_hostname"]),
        )

    def run_prepared_environment(
        self,
        replica: SlurmComputeReplica,
        prepared: SlurmPreparedEnvironment,
        events: Any,
    ) -> Mapping[str, Any]:
        if prepared.scheduler_job_id != replica.job_id:
            raise ComputeInfrastructureError(
                "prepared environment belongs to a replaced Slurm replica",
                resource_id=prepared.scheduler_job_id,
            )
        control = GatewayControl(prepared.control_url, prepared.control_token)
        agent = None
        agent_stopped = False
        try:
            control.arm()
            events.emit("armed", task_key=prepared.task_key)
            instruction = prepared.instruction
            agent_episode = 1
            with monitor_gateway_steps(
                control.status,
                lambda steps: events.emit(
                    "task_progress",
                    task_key=prepared.task_key,
                    num_steps=steps,
                ),
            ):
                while True:
                    agent = self.start_agent(
                        replica,
                        env_url=prepared.agent_url,
                        instruction=instruction,
                        task_key=prepared.task_key,
                        task_dir=prepared.task_dir,
                        timeout_sec=prepared.task.timeout_sec,
                        append_output=agent_episode > 1,
                    )
                    self.wait_for_gateway(replica, agent, control)
                    status = control.status()
                    infrastructure_error = status.get("infrastructure_error")
                    if infrastructure_error:
                        raise InstanceInfrastructureError(
                            str(infrastructure_error)
                        )
                    result = self.stop_agent(
                        replica,
                        agent,
                        reason=(
                            "episode_complete"
                            if status.get("continuation_pending")
                            else "gateway_fully_done"
                        ),
                    )
                    agent_stopped = True
                    episode = status.get("agent_episode", 1)
                    if status.get("continuation_pending"):
                        episode -= 1
                    events.emit(
                        "agent_exit",
                        task_key=prepared.task_key,
                        returncode=result.returncode,
                        agent_episode=episode,
                    )
                    if not status.get("continuation_pending"):
                        break
                    continuation = control.continue_agent()
                    instruction = str(continuation["instruction"])
                    agent_episode = int(continuation["agent_episode"])
                    agent = None
                    agent_stopped = False
        except AgentExecutionError as exc:
            assert agent is not None
            try:
                result = self.stop_agent(
                    replica,
                    agent,
                    reason="agent_execution_error",
                )
                agent_stopped = True
                events.emit(
                    "agent_exit",
                    task_key=prepared.task_key,
                    returncode=result.returncode,
                )
                if control.fail_agent(str(exc)):
                    events.emit(
                        "agent_failed",
                        task_key=prepared.task_key,
                        returncode=exc.returncode,
                        error=str(exc),
                    )
                control.wait_done(
                    budget_sec=(
                        prepared.task.timeout_sec
                        + prepared.task.grace_sec
                        + VERIFIER_TIMEOUT_SEC
                        + 60
                    )
                )
            except Exception as failure_error:
                running_error = self._running_replica_error(replica)
                if running_error is not None:
                    raise running_error from failure_error
                if isinstance(failure_error, InstanceInfrastructureError):
                    raise
                raise InstanceInfrastructureError(
                    "evaluator failed while recording an agent failure: "
                    f"{type(failure_error).__name__}: {failure_error}"
                ) from failure_error
        except Exception as exc:
            running_error = self._running_replica_error(replica)
            if running_error is not None:
                raise running_error from exc
            if isinstance(exc, InstanceInfrastructureError):
                raise
            raise InstanceInfrastructureError(
                "evaluator failed while running the Slurm-local instance: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if agent is not None and not agent_stopped:
                try:
                    self.stop_agent(
                        replica,
                        agent,
                        reason="exception_cleanup",
                    )
                except Exception:
                    pass

        # Fetch the run log through the gateway's control plane rather than
        # the shared filesystem: the node-side writer still holds the file
        # open here, and cross-host visibility of its buffered tail is not
        # guaranteed until the environment closes.
        try:
            row = summarize(_parse_runlog_text(control.runlog()))
        except (ValueError, KeyError) as exc:
            raise InstanceInfrastructureError(
                f"gateway returned an unreadable run log: {exc}"
            ) from exc
        events.emit(
            "task_done",
            task_key=prepared.task_key,
            **{
                key: row.get(key)
                for key in (
                    "passed",
                    "reason",
                    "task_time_sec",
                    "env_time_sec",
                    "agent_time_sec",
                    "num_steps",
                )
            },
        )
        return row

    def close_prepared_environments(
        self,
        replica: SlurmComputeReplica,
        prepared: Sequence[SlurmPreparedEnvironment],
    ) -> None:
        for environment in prepared:
            if environment.scheduler_job_id != replica.job_id:
                continue
            try:
                self._environment_request(
                    replica,
                    {
                        "operation": "close",
                        "handle": environment.handle,
                    },
                    timeout_sec=120.0,
                )
            except ComputeInfrastructureError:
                # A lost allocation already destroyed every process and VM
                # that belonged to it. The original compute error must drive
                # replacement; cleanup must not mask it.
                continue

    def start_agent(
        self,
        replica: SlurmComputeReplica,
        *,
        env_url: str,
        instruction: str,
        task_key: str,
        task_dir: Path,
        timeout_sec: float,
        append_output: bool = False,
    ) -> SlurmAgent:
        self.ensure_ready(replica)
        state = _read_json(replica.control_dir / "state.json") or {}
        runtime = dict(state.get("runtime") or {})
        python = runtime.pop("python", None)
        if not python:
            raise ComputeInfrastructureError(
                "Slurm replica has no agent runtime",
                resource_id=replica.job_id,
            )
        values = {
            "job_id": replica.job_id,
            "replica": str(replica.index),
            "task_key": task_key,
            "submission_dir": str(replica.submission_dir),
            "control_dir": str(replica.control_dir),
        }
        command = [
            _render(item, values, self.template_path)
            for item in self.template["step_command"]
        ] + [str(python), str(replica.submission_dir / "agent.py"), env_url, instruction]
        runtime_env = os.environ.copy()
        runtime_env.update(self.context.base_runtime_env)
        for name in self.context.evaluator_runtime_env:
            runtime_env.pop(name, None)
            runtime_env.pop(f"APPTAINERENV_{name}", None)
        for name in _VISIBILITY_ENV:
            runtime_env.pop(name, None)
        # These values belong inside the already-running Apptainer instance.
        # Exporting them directly to srun makes host-side Slurm use container
        # paths (notably HOME and TMPDIR), so Apptainer cannot find the
        # instance created by the allocation's worker process.
        runtime_env.update({
            f"APPTAINERENV_{name}": str(self.context.base_runtime_env[name])
            for name in self.context.submission_environment_names
        })
        runtime_env.update({
            f"APPTAINERENV_{key}": str(value)
            for key, value in runtime.items()
        })
        output_mode = "a" if append_output else "w"
        stdout_stream = (task_dir / "agent.stdout").open(output_mode)
        stderr_stream = (task_dir / "agent.stderr").open(output_mode)
        process = subprocess.Popen(
            command,
            cwd=replica.submission_dir,
            stdout=stdout_stream,
            stderr=stderr_stream,
            env=runtime_env,
        )
        agent = SlurmAgent(
            process=process,
            stdout_stream=stdout_stream,
            stderr_stream=stderr_stream,
            task_key=task_key,
        )
        state = _read_json(replica.control_dir / "state.json") or {}
        self.context.events.emit(
            "slurm_agent_step_started",
            task_key=task_key,
            local_srun_pid=process.pid,
            scheduler_job_id=replica.job_id,
            worker_hostname=str(state.get("hostname") or ""),
            parallel_evaluation=replica.index,
        )
        return agent

    def wait_for_gateway(
        self, replica: SlurmComputeReplica, agent: SlurmAgent, gateway: Gateway
    ) -> None:
        while True:
            gateway_status = gateway.status()
            if gateway_status.get("fully_done"):
                return
            returncode = agent.process.poll()
            if returncode is not None and not gateway_status.get("finished"):
                if gateway_status.get("continuation_pending"):
                    return
                state = _read_json(replica.control_dir / "state.json") or {}
                self.context.events.emit(
                    "slurm_agent_step_exited_unexpectedly",
                    task_key=agent.task_key,
                    returncode=returncode,
                    local_srun_pid=agent.process.pid,
                    scheduler_job_id=replica.job_id,
                    scheduler_state=self._scheduler_state(replica.job_id),
                    worker_state=str(state.get("status") or "missing"),
                    worker_hostname=str(state.get("hostname") or ""),
                    heartbeat_fresh=self._heartbeat_is_fresh(
                        replica.control_dir / "heartbeat"
                    ),
                    slurm_accounting=self._sacct_snapshot(replica.job_id),
                    parallel_evaluation=replica.index,
                )
                infrastructure_error = self._agent_infrastructure_error(
                    replica,
                    agent,
                    returncode,
                )
                if infrastructure_error is not None:
                    raise infrastructure_error
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    gateway_status = gateway.status()
                    if gateway_status.get("fully_done"):
                        return
                    if gateway_status.get("finished") or gateway_status.get(
                        "continuation_pending"
                    ):
                        break
                    time.sleep(0.25)
                if gateway_status.get("finished"):
                    continue
                if gateway_status.get("continuation_pending"):
                    return
                infrastructure_error = self._agent_infrastructure_error(
                    replica,
                    agent,
                    returncode,
                )
                if infrastructure_error is not None:
                    raise infrastructure_error
                raise AgentExecutionError(
                    f"Slurm agent exited with code {returncode} before "
                    "finishing the task",
                    returncode=returncode,
                )
            state = _read_json(replica.control_dir / "state.json") or {}
            heartbeat = replica.control_dir / "heartbeat"
            if (
                str(state.get("scheduler_job_id") or "") != replica.job_id
                or state.get("status") != "ready"
                or not self._heartbeat_is_fresh(heartbeat)
            ):
                infrastructure_error = self._running_replica_error(replica)
                if infrastructure_error is not None:
                    raise infrastructure_error
            time.sleep(0.5)

    def stop_agent(
        self,
        replica: SlurmComputeReplica,
        agent: SlurmAgent,
        *,
        reason: str = "runner_cleanup",
    ) -> AgentResult:
        if agent.process.poll() is None:
            self.context.events.emit(
                "slurm_agent_stop_requested",
                task_key=agent.task_key,
                reason=reason,
                signal="SIGTERM",
                local_srun_pid=agent.process.pid,
                parallel_evaluation=replica.index,
                scheduler_job_id=replica.job_id,
            )
            agent.process.terminate()
            try:
                agent.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                agent.process.kill()
                agent.process.wait()
        returncode = int(agent.process.returncode or 0)
        agent.stdout_stream.close()
        agent.stderr_stream.close()
        return AgentResult(returncode=returncode)

    def finish_replica(self, replica: SlurmComputeReplica) -> None:
        return None

    def close_replica(self, replica: SlurmComputeReplica) -> None:
        with replica.recovery_lock:
            self._cancel_current_job(replica, reason="runner_close")

    def _cancel_current_job(
        self,
        replica: SlurmComputeReplica,
        *,
        reason: str,
    ) -> None:
        job_id = replica.job_id
        if not job_id or job_id in self._closed_job_ids:
            return
        self._closed_job_ids.add(job_id)
        state = _read_json(replica.control_dir / "state.json") or {}
        self.context.events.emit(
            "slurm_job_cancel_requested",
            reason=reason,
            cancel_command=list(replica.cancel_command),
            parallel_evaluation=replica.index,
            scheduler_job_id=job_id,
            scheduler_state=self._scheduler_state(job_id),
            worker_state=str(state.get("status") or "missing"),
            worker_hostname=str(state.get("hostname") or ""),
        )
        (replica.control_dir / "shutdown").touch()
        result = subprocess.run(
            list(replica.cancel_command),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.context.events.emit(
            "slurm_job_cancel_completed",
            reason=reason,
            returncode=result.returncode,
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip(),
            parallel_evaluation=replica.index,
            scheduler_job_id=job_id,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if not self._scheduler_state(job_id):
                shutil.rmtree(replica.control_dir / "runtime", ignore_errors=True)
                return
            time.sleep(0.5)

    def close(self) -> None:
        return None


__all__ = ["SlurmComputeReplica", "SlurmComputeRunner"]
