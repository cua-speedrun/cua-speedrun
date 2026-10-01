from __future__ import annotations

import subprocess
from pathlib import Path

from cua_speedrun.envs.gym_anything import _posix_privileged_scrub_command


def test_guest_scrub_removes_verifier_material_after_setup(tmp_path: Path) -> None:
    task = tmp_path / "tasks" / "one"
    shared = tmp_path / "tasks" / "_shared"
    task.mkdir(parents=True)
    shared.mkdir()
    for name in (
        "task.json",
        "source.json",
        "verifier.py",
        "validated_pi.json",
        "vlm_checklist.json",
    ):
        (task / name).write_text("privileged")
    (shared / "osworld_verifier.py").write_text("privileged")
    (task / "custom_grader.py").write_text("privileged")
    (task / "setup_task.sh").write_text("setup")
    (task / "export_result.sh").write_text("export")

    subprocess.run(
        [
            "sh",
            "-c",
            _posix_privileged_scrub_command(
                [str(tmp_path / "tasks")], {"custom_grader.py"}
            ),
        ],
        check=True,
    )

    assert not (task / "task.json").exists()
    assert not (task / "source.json").exists()
    assert not (task / "verifier.py").exists()
    assert not (shared / "osworld_verifier.py").exists()
    assert not (task / "custom_grader.py").exists()
    assert (task / "setup_task.sh").read_text() == "setup"
    assert (task / "export_result.sh").read_text() == "export"
