"""Keep one Slurm allocation warm after running a submission's init.py."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from cua_speedrun.compute_runners.local_sandbox import LocalSandboxCache
from cua_speedrun.environment_instance import (
    PreparedEnvironmentInstance,
    prepare_environment_instance,
)
from cua_speedrun.envs import get_backend
from cua_speedrun.gateway import Gateway
from cua_speedrun.specs import TaskSpec
from cua_speedrun.submission import Submission


class _WorkerInterrupted(RuntimeError):
    """The allocation or one of its infrastructure processes disappeared."""


def scheduler_job_id() -> str | None:
    """The job id the coordinator uses to address this allocation.

    Array elements are addressed as ``<array_job_id>_<array_task_id>``, the
    form squeue, scancel, and sacct all accept. Plain jobs keep SLURM_JOB_ID.
    """
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID", "").strip()
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID", "").strip()
    if array_job and array_task:
        return f"{array_job}_{array_task}"
    return os.environ.get("SLURM_JOB_ID")


class _NoopEvents:
    def emit(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@dataclass
class _WorkerEnvironment:
    instance: PreparedEnvironmentInstance
    gateway: Gateway


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, default=str))
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _task_from_payload(payload: Mapping[str, Any]) -> TaskSpec:
    raw = dict(payload)
    task_dir = raw.get("task_dir")
    return TaskSpec(
        task_id=str(raw["task_id"]),
        description=str(raw["description"]),
        env=dict(raw["env"]),
        timeout_sec=float(raw["timeout_sec"]),
        grace_sec=float(raw["grace_sec"]),
        metadata=dict(raw.get("metadata") or {}),
        generator=(
            str(raw["generator"]) if raw.get("generator") is not None else None
        ),
        task_dir=Path(task_dir) if task_dir else None,
    )


# Must stay a substring match for slurm.py's KVM capability classifier, so
# the coordinator excludes this node and resubmits elsewhere.
_KVM_UNAVAILABLE = (
    "gym-anything-local needs /dev/kvm for accelerated local VMs on this "
    "node, or this user cannot open it"
)


def _validate_kvm() -> None:
    if os.path.exists("/dev/kvm") and os.access("/dev/kvm", os.R_OK | os.W_OK):
        return
    raise RuntimeError(f"{_KVM_UNAVAILABLE}: {os.uname().nodename}")


def _validate_gpu(required: str | None) -> list[str]:
    if not required:
        return []
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    devices = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    wanted = required.rstrip("!").lower().replace("-", "").replace(" ", "")
    available = [item.lower().replace("-", "").replace(" ", "") for item in devices]
    if probe.returncode or not any(wanted in item for item in available):
        raise RuntimeError(
            f"allocation requires {required!r}, but reports "
            f"{devices or ['no NVIDIA GPU']}"
        )
    return devices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--base-python", type=Path, required=True)
    parser.add_argument("--gpu")
    parser.add_argument("--require-kvm", action="store_true")
    parser.add_argument("--python-package", action="append", default=[])
    parser.add_argument("--init-cache-key", required=True)
    parser.add_argument("--runtime-environment-name", action="append", default=[])
    parser.add_argument("--evaluator-environment-name", action="append", default=[])
    args = parser.parse_args()

    control = args.control_dir.resolve()
    state_path = control / "state.json"
    heartbeat = control / "heartbeat"
    shutdown = control / "shutdown"
    environment_requests = control / "environment_requests"
    environment_requests.mkdir(parents=True, exist_ok=True)
    previous = {}
    try:
        previous = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    generation = int(previous.get("generation", 0)) + 1
    stopping = threading.Event()

    def state(status: str, **payload: Any) -> None:
        _atomic_json(state_path, {
            "status": status,
            "generation": generation,
            "hostname": os.uname().nodename,
            "scheduler_job_id": scheduler_job_id(),
            "scheduler_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
            **payload,
        })

    def stop(_signum, _frame) -> None:
        stopping.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    shutdown.unlink(missing_ok=True)
    started = time.monotonic()
    failed = False
    sandbox = None
    request_threads: list[threading.Thread] = []
    environments: dict[str, _WorkerEnvironment] = {}
    environments_lock = threading.Lock()

    def close_environment(handle: str) -> None:
        with environments_lock:
            environment = environments.pop(handle, None)
        if environment is None:
            return
        cleanup_error = None
        try:
            environment.gateway.shutdown()
        except Exception as exc:
            cleanup_error = exc
        try:
            environment.instance.close()
        except Exception as exc:
            cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            raise cleanup_error

    def handle_environment_request(
        request_path: Path,
        submission: Submission,
    ) -> None:
        response_path = request_path.with_name(
            request_path.name.replace(".request.json", ".response.json")
        )
        try:
            request = json.loads(request_path.read_text())
            expected_job_id = str(request.get("scheduler_job_id") or "")
            current_job_id = str(scheduler_job_id() or "")
            if expected_job_id and expected_job_id != current_job_id:
                return
            operation = str(request["operation"])
            handle = str(request["handle"])
            if operation == "prepare":
                task = _task_from_payload(request["task"])
                backend = get_backend(str(request["backend_name"]))
                instance = prepare_environment_instance(
                    submission,
                    backend,
                    task,
                    int(request["seed"]),
                    Path(request["run_dir"]),
                    str(request["run_id"]),
                    _NoopEvents(),
                )
                instance.log.event(
                    "environment_placement",
                    execution_runner="slurm",
                    hostname=os.uname().nodename,
                    scheduler_job_id=current_job_id,
                )
                gateway = None
                try:
                    gateway = Gateway(
                        adapter=instance.prepared.adapter,
                        log=instance.log,
                        artifacts_dir=instance.task_dir,
                        timeout_sec=task.timeout_sec,
                        grace_sec=task.grace_sec,
                        checker=instance.checker,
                        host="0.0.0.0",
                        advertise_host=os.uname().nodename,
                        run_token=str(request["run_token"]),
                        control_token=str(request["control_token"]),
                        instruction=instance.description,
                    )
                    gateway_url = gateway.start()
                    parsed = urlsplit(gateway_url)
                    if parsed.port is None:
                        raise RuntimeError(
                            "replica environment gateway did not report a port"
                        )
                    with environments_lock:
                        if handle in environments:
                            raise RuntimeError(
                                f"duplicate replica environment handle {handle}"
                            )
                        environments[handle] = _WorkerEnvironment(instance, gateway)
                    gateway = None
                    _atomic_json(response_path, {
                        "ok": True,
                        "scheduler_job_id": current_job_id,
                        "handle": handle,
                        "control_url": f"{parsed.scheme}://{parsed.netloc}",
                        "agent_url": (
                            f"http://127.0.0.1:{parsed.port}/"
                            f"{request['run_token']}"
                        ),
                        "instruction": instance.description,
                        "prepare_time_sec": instance.prepared.prepare_time_sec,
                        "environment_info": instance.prepared.info,
                        "environment_hostname": os.uname().nodename,
                    })
                finally:
                    if gateway is not None:
                        try:
                            gateway.shutdown()
                        except Exception:
                            pass
                        instance.close()
                return
            if operation == "close":
                close_environment(handle)
                _atomic_json(response_path, {
                    "ok": True,
                    "scheduler_job_id": current_job_id,
                    "handle": handle,
                })
                return
            raise ValueError(f"unknown environment operation {operation!r}")
        except BaseException as exc:
            _atomic_json(response_path, {
                "ok": False,
                "scheduler_job_id": scheduler_job_id(),
                "error": repr(exc),
                "error_type": type(exc).__name__,
            })

    try:
        state("initializing")
        if args.require_kvm:
            _validate_kvm()
        devices = _validate_gpu(args.gpu)
        submission = Submission.load(args.submission_dir)
        required_environment_names = (
            args.runtime_environment_name + args.evaluator_environment_name
        )
        missing_environment = sorted(
            name for name in required_environment_names if name not in os.environ
        )
        if missing_environment:
            raise RuntimeError(
                "Slurm did not export required runtime environment variables: "
                + ", ".join(missing_environment)
            )
        runtime_environment = {
            name: os.environ[name] for name in args.runtime_environment_name
        }
        cache = LocalSandboxCache(
            submission_dir=submission.submission_dir,
            submission_fingerprint=submission.fingerprint,
            base_python=args.base_python,
            python_packages=args.python_package,
            runtime_env=runtime_environment,
            gpu=args.gpu,
            run_id=control.parent.parent.name,
            init_cache_key=args.init_cache_key,
        )
        sandbox = cache.start_replica(
            index=int(control.name.rsplit("_", 1)[-1]),
            control=control,
            log=control / "init.log",
            device=os.environ.get("CUDA_VISIBLE_DEVICES") if args.gpu else None,
        )
        if stopping.is_set():
            raise _WorkerInterrupted(
                "Slurm allocation was interrupted during sandbox initialization"
            )
        runtime = {**sandbox.runtime_overrides, "python": str(sandbox.agent_python)}
        state(
            "ready",
            devices=devices,
            init_duration_sec=time.monotonic() - started,
            filesystem_cache_key=sandbox.cache_key,
            filesystem_cache_hit=sandbox.cache_hit,
            runtime=runtime,
        )
        seen_requests: set[str] = set()
        while not stopping.is_set() and not shutdown.exists():
            if not sandbox.running():
                raise _WorkerInterrupted("local compute sandbox exited")
            for request_path in sorted(
                environment_requests.glob("*.request.json")
            ):
                if request_path.name in seen_requests:
                    continue
                seen_requests.add(request_path.name)
                thread = threading.Thread(
                    target=handle_environment_request,
                    args=(request_path, submission),
                    daemon=True,
                )
                request_threads.append(thread)
                thread.start()
            heartbeat.touch()
            time.sleep(1)
    except _WorkerInterrupted as exc:
        failed = True
        state("interrupted", error=repr(exc))
    except BaseException as exc:
        failed = True
        if stopping.is_set():
            state("interrupted", error=repr(_WorkerInterrupted(
                "Slurm allocation was interrupted during sandbox initialization: "
                f"{type(exc).__name__}: {exc}"
            )))
        else:
            state("failed", error=repr(exc))
            raise
    finally:
        stopping.set()
        with environments_lock:
            handles = list(environments)
        for handle in handles:
            try:
                close_environment(handle)
            except Exception:
                pass
        if sandbox is not None:
            sandbox.stop()
        if not failed:
            state("stopped")


if __name__ == "__main__":
    main()
