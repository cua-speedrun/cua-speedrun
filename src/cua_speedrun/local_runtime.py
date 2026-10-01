"""Content-addressed identity of the host used for local execution."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping


LOCAL_RUNTIME_SCHEMA_VERSION = 2
LOCAL_RUNTIME_RECIPE = "local-managed-python-qemu@1"
LOCAL_AGENT_PYTHON_VERSION = "3.10"
LOCAL_RUNTIME_REQUESTED = "requested"
LOCAL_RUNTIME_OBSERVED = "observed"


def _file_identity(command: str) -> str:
    path = shutil.which(command)
    if path is None:
        return f"{command}==missing"
    resolved = Path(path).resolve()
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return f"{command}==sha256:{digest}"


def local_agent_python() -> Path:
    raw = os.environ.get("CS_AGENT_PYTHON", "").strip()
    if not raw:
        raise RuntimeError(
            "local agent Python is not configured; run cua-speedrun install"
        )
    python = Path(raw).expanduser().resolve()
    if not python.is_file():
        raise RuntimeError(
            f"local agent Python is missing at {python}; rerun cua-speedrun install"
        )
    return python


def _agent_python_identity() -> tuple[Path, str, str]:
    python = local_agent_python()
    result = subprocess.run(
        [str(python), "-c", "import platform; print(platform.python_version())"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    observed = result.stdout.strip()
    if result.returncode or not observed.startswith(
        f"{LOCAL_AGENT_PYTHON_VERSION}."
    ):
        detail = (result.stderr or observed or "version probe failed").strip()
        raise RuntimeError(
            "local agent Python does not match the platform runtime "
            f"{LOCAL_AGENT_PYTHON_VERSION}: {python}: {detail}"
        )
    digest = hashlib.sha256(python.resolve().read_bytes()).hexdigest()
    return python, observed, digest


def _qemu_system_command() -> str:
    return (
        "qemu-system-aarch64"
        if platform.system() == "Darwin" and platform.machine() == "arm64"
        else "qemu-system-x86_64"
    )


def _accelerator_identity() -> dict[str, Any]:
    command = shutil.which("nvidia-smi")
    if command is None:
        return {"kind": "none", "devices": []}
    proc = subprocess.run(
        [
            command,
            "--query-gpu=name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    devices = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return {
        "kind": "nvidia",
        "devices": devices,
        "probe_returncode": proc.returncode,
    }


def _osworld_image_identity() -> dict[str, Any]:
    raw = os.environ.get("OSWORLD_QEMU_BASE_IMAGE", "").strip()
    if not raw:
        return {"configured": False}
    path = Path(raw).expanduser().resolve()
    identity: dict[str, Any] = {
        "configured": True,
        "path": str(path),
        "exists": path.is_file(),
    }
    if path.is_file():
        stat = path.stat()
        identity.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    provenance_path = Path(f"{path}.provenance.json")
    if provenance_path.is_file():
        try:
            provenance = json.loads(provenance_path.read_text())
        except (OSError, json.JSONDecodeError):
            provenance = {"invalid": True}
        identity["provenance"] = provenance
    return identity


def current_local_agent_runtime_contract(
    *,
    environment_runner: str,
    accelerator: Mapping[str, Any] | None = None,
    execution_scope: str = "evaluator-host",
) -> dict[str, Any]:
    """Return the exact local host identity captured for a new run."""
    _, agent_python_version, agent_python_sha256 = _agent_python_identity()
    if environment_runner == "QemuNativeRunner":
        launcher_packages = [
            _file_identity(_qemu_system_command()),
            _file_identity("qemu-img"),
        ]
        launcher = {
            "key": "qemu_native",
            "implementation": environment_runner,
        }
    elif environment_runner == "QemuApptainerRunner":
        container_image = os.environ.get(
            "GYM_ANYTHING_QEMU_CONTAINER",
            "docker://ghcr.io/dockur/windows:latest",
        )
        launcher_packages = [_file_identity("apptainer")]
        launcher = {
            "key": "qemu",
            "implementation": environment_runner,
            "container_image": container_image,
        }
    else:
        raise ValueError(
            "local runtime contract needs gym-anything's observed QEMU "
            f"runner, got {environment_runner!r}"
        )
    return {
        "schema_version": LOCAL_RUNTIME_SCHEMA_VERSION,
        "recipe": LOCAL_RUNTIME_RECIPE,
        "base_image": {
            "builder": "uv-managed-python",
            "python_version": LOCAL_AGENT_PYTHON_VERSION,
            "observed_python_version": agent_python_version,
        },
        "system_packages": [
            f"kernel=={platform.system()}-{platform.release()}-{platform.machine()}",
            *launcher_packages,
        ],
        "environment_launcher": launcher,
        "python_packages": ["requests==2.34.2"],
        "agent_python_sha256": agent_python_sha256,
        "agent_isolation": "clean-venv@1",
        "accelerator": dict(accelerator) if accelerator is not None else _accelerator_identity(),
        "osworld_qemu_image": _osworld_image_identity(),
        "execution_scope": execution_scope,
        "runtime_resolution": LOCAL_RUNTIME_OBSERVED,
        "modal_client_version": "not-used",
        "modal_image_builder_version": "not-used",
    }


def requested_local_agent_runtime_contract() -> dict[str, Any]:
    """Portable request stored until this installation's worker claims it.

    Runtime identity is observed at execution rather than page-render time, so
    the contract records the process that actually runs the evaluation.
    """
    return {
        "schema_version": LOCAL_RUNTIME_SCHEMA_VERSION,
        "recipe": LOCAL_RUNTIME_RECIPE,
        "base_image": {
            "builder": "uv-managed-python",
            "python_version": LOCAL_AGENT_PYTHON_VERSION,
            "observed_python_version": "resolved-on-worker",
        },
        "system_packages": ["local-qemu-launcher==resolved-on-worker"],
        "environment_launcher": {
            "key": "resolved-on-worker",
            "implementation": "resolved-on-worker",
        },
        "python_packages": ["requests==2.34.2"],
        "agent_isolation": "clean-venv@1",
        "execution_scope": "evaluator-host",
        "runtime_resolution": LOCAL_RUNTIME_REQUESTED,
        "modal_client_version": "not-used",
        "modal_image_builder_version": "not-used",
    }


def local_runtime_needs_resolution(runtime: Mapping[str, Any]) -> bool:
    return runtime.get("runtime_resolution") == LOCAL_RUNTIME_REQUESTED


def validate_local_agent_runtime_contract(
    runtime: Mapping[str, Any],
    *,
    environment_runner: str,
    accelerator: Mapping[str, Any] | None = None,
    execution_scope: str = "evaluator-host",
) -> None:
    """Fail before user code runs if the queued plan targets another host."""
    expected = current_local_agent_runtime_contract(
        environment_runner=environment_runner,
        accelerator=accelerator,
        execution_scope=execution_scope,
    )
    actual = json.loads(json.dumps(dict(runtime), sort_keys=True))
    if actual != expected:
        raise ValueError(
            "stored local runtime is not reproducible by this worker: "
            f"stored={actual!r}, worker={expected!r}"
        )


def validate_local_gpu(runtime: Mapping[str, Any], gpu: str | None) -> None:
    """Fail closed when a local track's GPU label is not on this host."""
    if not gpu:
        return
    devices = list((runtime.get("accelerator") or {}).get("devices") or ())
    normalized = gpu.rstrip("!").lower().replace("-", "").replace(" ", "")
    available = [
        str(device).lower().replace("-", "").replace(" ", "")
        for device in devices
    ]
    if not any(normalized in device for device in available):
        raise RuntimeError(
            f"local track requires GPU {gpu!r}, but this worker reports "
            f"{devices or ['no NVIDIA GPU']}"
        )


__all__ = [
    "LOCAL_AGENT_PYTHON_VERSION",
    "current_local_agent_runtime_contract",
    "local_agent_python",
    "local_runtime_needs_resolution",
    "requested_local_agent_runtime_contract",
    "validate_local_agent_runtime_contract",
    "validate_local_gpu",
]
