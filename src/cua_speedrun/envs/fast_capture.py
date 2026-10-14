"""Fast capture selection and untimed native-QEMU compatibility setup.

Gym-Anything's pinned native runner inherits an Apptainer-only D-Bus
prerequisite check. Replace that check on this instance, not its input or
capture implementation and not the dependency's source files.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import threading
import time
from typing import Any, Mapping


def capture_backend(environment: Mapping[str, str]) -> str:
    value = environment.get("GYM_ANYTHING_QEMU_FAST_IO_BACKEND", "dbus").strip().lower()
    if value not in {"dbus", "qmp"}:
        raise ValueError("GYM_ANYTHING_QEMU_FAST_IO_BACKEND must be dbus or qmp")
    return value


def profiling_enabled(environment: Mapping[str, str]) -> bool:
    value = environment.get("CS_FAST_IO_PROFILE", "false").strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError("CS_FAST_IO_PROFILE must be true, false, 1, or 0")
    return value in {"true", "1"}


def check_native_dbus(guest_arch: str) -> None:
    # The pinned ARM command builder hardcodes -display none; do not silently
    # select a backend its command line cannot start.
    if guest_arch != "x86_64":
        raise RuntimeError("the pinned native D-Bus command builder supports x86_64 only")
    if not shutil.which("dbus-daemon") or importlib.util.find_spec("dbus_next") is None:
        raise RuntimeError("native D-Bus capture requires dbus-daemon and dbus-next")
    result = subprocess.run(
        ["qemu-system-x86_64", "-display", "help"],
        capture_output=True, text=True, timeout=10, check=True,
    )
    if "dbus" not in {line.strip() for line in (result.stdout + result.stderr).splitlines()}:
        raise RuntimeError("installed native QEMU does not provide -display dbus")


def enable_fast_capture(env: Any) -> None:
    from gym_anything.runtime.runners.qemu_native import QemuNativeRunner

    runner = env._runner
    if not runner.supports_fast_io():
        raise RuntimeError(f"{env.runner_name} does not support fast_io=True")
    if isinstance(runner, QemuNativeRunner):
        runner._ensure_dbus_display_container = lambda: check_native_dbus(runner._guest_arch)
    runner.set_fast_io(True)
    env.fast_io = True


class CaptureTimer:
    """Per-thread capture counters; opt-in diagnostics on the actual callable."""

    def __init__(self, capture):
        self._capture = capture
        self._local = threading.local()

    def begin(self) -> dict:
        self._local.metrics = {"capture_sec": 0.0, "capture_calls": 0}
        return self._local.metrics

    def end(self) -> None:
        self._local.metrics = None

    def __call__(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self._capture(*args, **kwargs)
        finally:
            metrics = getattr(self._local, "metrics", None)
            if metrics is not None:
                metrics["capture_sec"] += time.perf_counter() - started
                metrics["capture_calls"] += 1
