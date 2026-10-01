"""Isolate native controllers from desktop wall-clock changes.

Linux namespaces do not isolate CLOCK_REALTIME. Anchor controller UTC to
the sandbox's initial real time plus its unmodified monotonic clock. Load
the guard before Python starts so TLS libraries see the same clock too.
Guest commands use a clean environment and retain the desktop's own date.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def isolated_environment(directory: Path) -> dict[str, str]:
    """Build the controller-only guard and capture UTC before desktop setup."""
    if sys.platform != "linux":
        raise RuntimeError("native controller clock isolation requires Linux")
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("native desktop image needs a C compiler for clock isolation")
    source = Path(__file__).with_suffix(".c")
    library = directory / "native_realtime.so"
    result = subprocess.run(
        [compiler, "-shared", "-fPIC", "-O2", "-Wall", "-Wextra", "-Werror",
         "-o", str(library), str(source)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        raise RuntimeError(f"native controller clock guard compilation failed: {result.stderr.strip()}")
    before = time.monotonic_ns()
    utc = time.time_ns()
    after = time.monotonic_ns()
    environment = dict(os.environ)
    environment["CS_NATIVE_UTC_OFFSET_NS"] = str(utc - (before + after) // 2)
    environment["LD_PRELOAD"] = " ".join(
        part for part in (str(library), environment.get("LD_PRELOAD", "")) if part
    )
    return environment


def isolate_controller_clock() -> None:
    """Re-exec once, before booting a desktop; fail if the guard cannot load."""
    if "CS_NATIVE_UTC_OFFSET_NS" in os.environ:
        try:
            active = ctypes.CDLL(None).cua_native_realtime_active
        except AttributeError as exc:
            raise RuntimeError("native controller clock guard did not load") from exc
        if active() != 1:
            raise RuntimeError("native controller clock guard is inactive")
        return
    directory = Path(tempfile.mkdtemp(prefix="cua-native-clock-"))
    environment = isolated_environment(directory)
    os.execve(
        sys.executable,
        [sys.executable, "-u", "-m", "cua_speedrun.remote.modal_native_env_plane"],
        environment,
    )
