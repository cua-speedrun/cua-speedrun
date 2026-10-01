import subprocess
import sys

import pytest

from cua_speedrun.envs.modal_native import _run_pre_task_command


@pytest.mark.parametrize("returncode", [0, 7])
def test_setup_retains_complete_stdout_and_stderr(returncode, capsys):
    stdout = "stdout-start:" + "x" * 4096 + ":stdout-end"
    stderr = "stderr-start:" + "y" * 4096 + ":stderr-end"
    script = (
        f"import sys; sys.stdout.write({stdout!r}); "
        f"sys.stderr.write({stderr!r}); sys.exit({returncode})"
    )
    command = [sys.executable, "-c", script]
    if returncode:
        with pytest.raises(RuntimeError, match=f"pre_task failed with rc={returncode}"):
            _run_pre_task_command(command, timeout=10)
    else:
        _run_pre_task_command(command, timeout=10)
    logged = capsys.readouterr().out
    assert stdout in logged
    assert stderr in logged
    assert f"pre_task rc={returncode}" in logged
    assert "pre_task stdout:" in logged
    assert "pre_task stderr:" in logged


def test_setup_retains_both_streams_when_command_times_out(capsys):
    stdout = "stdout-before-timeout:" + "x" * 4096
    stderr = "stderr-before-timeout:" + "y" * 4096
    script = (
        f"import sys,time; print({stdout!r},flush=True); "
        f"print({stderr!r},file=sys.stderr,flush=True); time.sleep(30)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        _run_pre_task_command([sys.executable, "-c", script], timeout=1)
    logged = capsys.readouterr().out
    assert stdout in logged
    assert stderr in logged
