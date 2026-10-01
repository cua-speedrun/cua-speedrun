"""Exercise the compiled clock guard with real child processes on Linux."""

import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from cua_speedrun.envs.modal_native import _guest_process_environment
from cua_speedrun.remote.native_realtime import isolated_environment


def test_guest_commands_do_not_inherit_controller_clock():
    environment = _guest_process_environment()
    assert "LD_PRELOAD" not in environment
    assert "CS_NATIVE_UTC_OFFSET_NS" not in environment


@pytest.mark.skipif(sys.platform != "linux", reason="Linux native controller")
def test_clock_guard_reads_real_utc_and_preserves_monotonic(tmp_path: Path):
    environment = isolated_environment(tmp_path)
    before = time.time()
    code = """
import ctypes, json, subprocess, sys, time
assert ctypes.CDLL(None).cua_native_realtime_active() == 1
libc = ctypes.CDLL(None)
libc.time.restype = ctypes.c_long
class Timeval(ctypes.Structure):
    _fields_ = [('seconds', ctypes.c_long), ('microseconds', ctypes.c_long)]
tv = Timeval()
assert libc.gettimeofday(ctypes.byref(tv), None) == 0
start = time.monotonic()
time.sleep(0.02)
print(json.dumps({'utc': time.time(), 'libc': libc.time(None),
                 'gettimeofday': tv.seconds + tv.microseconds / 1e6,
                 'elapsed': time.monotonic() - start,
                 'child': float(subprocess.check_output(
                     [sys.executable, '-c', 'import time; print(time.time())']))}))
"""
    result = json.loads(subprocess.check_output(
        [sys.executable, "-c", code], env=environment, text=True,
    ))
    after = time.time()
    assert before <= result["utc"] <= after
    assert int(before) <= result["libc"] <= int(after)
    assert before <= result["gettimeofday"] <= after
    assert 0.01 <= result["elapsed"] < 5
    assert before <= result["child"] <= after


@pytest.mark.skipif(sys.platform != "linux", reason="Linux native controller")
def test_clock_guard_rejects_invalid_anchor(tmp_path: Path):
    environment = isolated_environment(tmp_path)
    environment["CS_NATIVE_UTC_OFFSET_NS"] = "invalid"
    result = subprocess.run(
        [sys.executable, "-c", "raise AssertionError('must not start')"],
        env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 127
    assert "Invalid native controller UTC anchor" in result.stderr
