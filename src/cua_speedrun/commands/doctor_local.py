"""Local VM, benchmark, agent, and accelerator capability probes."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from cua_speedrun.local_runtime import (
    LOCAL_AGENT_PYTHON_VERSION,
    local_agent_python,
)

from .dependencies import OSWORLD_RUNTIME_IMPORTS, missing_imports
from .diagnostics import Capability, Check, module_check
from .paths import InstallationPaths


def _virtualization_check() -> Check:
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "kern.hv_support"],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception as exc:
            return Check("virtualization", False, f"cannot probe HVF: {exc}")
        ready = result.returncode == 0 and result.stdout.strip() == "1"
        return Check(
            "virtualization",
            ready,
            "Apple Hypervisor.framework is available"
            if ready
            else "Apple Hypervisor.framework is unavailable",
        )
    if sys.platform != "linux":
        return Check(
            "virtualization",
            False,
            f"unsupported host: {sys.platform} {platform.machine()}",
        )
    descriptor: int | None = None
    try:
        descriptor = os.open("/dev/kvm", os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        return Check(
            "virtualization", False, f"cannot open /dev/kvm read/write: {exc}"
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return Check("virtualization", True, "/dev/kvm opens read/write")


def _runner_check() -> Check:
    try:
        from gym_anything.doctor import get_runner_status
    except Exception as exc:
        return Check("VM runner", False, f"gym-anything diagnostics unavailable: {exc}")
    try:
        status = get_runner_status()
    except Exception as exc:
        return Check("VM runner", False, f"runner discovery failed: {exc}")
    preferred = ("qemu", "qemu_native")
    available = [name for name in preferred if status.get(name, {}).get("available")]
    if available:
        return Check("VM runner", True, "auto-discovered " + ", ".join(available))
    reasons: list[str] = []
    for name in preferred:
        info = status.get(name) or {}
        reason = info.get("reason")
        if reason:
            reasons.append(f"{name}: {reason}")
            continue
        missing = [
            dep
            for dep, dep_info in (info.get("deps") or {}).items()
            if not dep_info.get("installed")
        ]
        if missing:
            reasons.append(f"{name}: missing {', '.join(missing)}")
    return Check(
        "VM runner",
        False,
        "; ".join(reasons) or "no QEMU runner was discovered",
    )


def local_vm_capability() -> Capability:
    return Capability(
        "local-vms",
        "Local environment VMs",
        (
            module_check("environment runtime", "gym_anything"),
            _virtualization_check(),
            _runner_check(),
        ),
    )


def _osworld_provenance_check(paths: InstallationPaths) -> Check:
    image = paths.osworld_image
    sidecar = Path(str(image) + ".provenance.json")
    if not image.is_file():
        return Check("VM image", False, f"missing: {image}")
    if not sidecar.is_file():
        return Check("VM image", False, f"missing provenance: {sidecar}")
    contract_path = paths.resource_root / "benchmarks/osworld-image.json"
    try:
        contract = json.loads(contract_path.read_text())
        provenance = json.loads(sidecar.read_text())
    except (OSError, ValueError) as exc:
        return Check("VM image", False, f"cannot read image contract: {exc}")
    source = contract["official_source"]
    expected = {
        "schema_version": contract["provenance_schema_version"],
        "recipe": contract["recipe"],
        "source_revision": source["revision"],
        "archive_sha256": source["archive_sha256"],
        "source_image_sha256": source["image_sha256"],
        "ssh_prepared": True,
        "tools_prepared": True,
        "guest_auth_prepared": True,
    }
    mismatched = [name for name, value in expected.items() if provenance.get(name) != value]
    if mismatched:
        return Check(
            "VM image", False, "provenance mismatch: " + ", ".join(mismatched)
        )
    return Check("VM image", True, f"{image} with matching provenance")


def osworld_capability(paths: InstallationPaths) -> Capability:
    full_tasks = paths.resource_root / "benchmark-assets/osworld/tasks"
    task_count = sum(1 for path in full_tasks.glob("*/task.yaml") if path.is_file())
    representative = (
        paths.resource_root / "benchmarks/osworld-50/benchmark-source.yaml"
    )
    missing_python = missing_imports(OSWORLD_RUNTIME_IMPORTS)
    return Capability(
        "osworld",
        "Local OSWorld benchmark",
        (
            Check(
                "task catalog",
                task_count == 369,
                f"{task_count}/369 full benchmark task definitions installed",
                required=False,
            ),
            Check(
                "representative catalog",
                representative.is_file(),
                str(representative)
                if representative.is_file()
                else "representative source is missing",
            ),
            Check(
                "verifier dependencies",
                not missing_python,
                "installed"
                if not missing_python
                else "missing Python modules: " + ", ".join(missing_python),
            ),
            _osworld_provenance_check(paths),
        ),
    )


def local_agent_capability(paths: InstallationPaths) -> Capability:
    writable = paths.runs.is_dir() and os.access(paths.runs, os.W_OK | os.X_OK)
    try:
        python = local_agent_python()
        probe = subprocess.run(
            [str(python), "-c", "import platform; print(platform.python_version())"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        observed = probe.stdout.strip()
        python_ready = probe.returncode == 0 and observed.startswith(
            f"{LOCAL_AGENT_PYTHON_VERSION}."
        )
        python_detail = f"{python} ({observed or 'version unavailable'})"
    except Exception as exc:
        python_ready = False
        python_detail = str(exc)
    return Capability(
        "local-agent",
        "Local model/agent process",
        (
            Check("submission Python", python_ready, python_detail),
            Check(
                "run workspace",
                writable,
                f"{paths.runs} {'is writable' if writable else 'is not writable'}",
            ),
        ),
    )


def _gpu_check() -> Check:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        identifiers = [
            item.strip()
            for item in visible.split(",")
            if item.strip() and item.strip().lower() not in {"-1", "none", "void"}
        ]
        if not identifiers:
            return Check("NVIDIA GPU", False, "CUDA_VISIBLE_DEVICES exposes no devices")
        return Check(
            "NVIDIA GPU",
            True,
            f"{len(identifiers)} device(s) exposed by CUDA_VISIBLE_DEVICES",
        )
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return Check("NVIDIA GPU", False, "nvidia-smi not found")
    try:
        result = subprocess.run(
            [binary, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return Check("NVIDIA GPU", False, f"nvidia-smi failed: {exc}")
    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode or not devices:
        detail = (result.stderr or result.stdout or "no visible devices").strip()
        return Check("NVIDIA GPU", False, detail)
    return Check("NVIDIA GPU", True, f"{len(devices)} visible: " + "; ".join(devices))


def local_gpu_capability() -> Capability:
    return Capability("local-gpu", "Local GPU compute", (_gpu_check(),))


__all__ = [
    "local_agent_capability",
    "local_gpu_capability",
    "local_vm_capability",
    "osworld_capability",
]
