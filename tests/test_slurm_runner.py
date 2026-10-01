from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cua_speedrun.compute_runners.base import ComputeRunnerContext
from cua_speedrun.compute_runners.config import resolve_runner_selection
from cua_speedrun.compute_runners.slurm import (
    SlurmComputeRunner,
    _load_excluded_nodes,
    _load_template,
    _render,
    _save_excluded_nodes,
)
from cua_speedrun.compute_runners.slurm_worker import scheduler_job_id


_TEMPLATE = """\
schema_version: 1
runner: slurm
name: test-template
{mode_line}
job:
  cpus_per_replica:
    default: 8
submit_command:
  - sbatch
  - --parsable
  - --array=1-1%1
step_command:
  - srun
  - --jobid=${{job_id}}
cancel_command:
  - scancel
  - ${{job_id}}
script: |
  exec ${{worker_command}}
"""


def _write(tmp_path: Path, mode_line: str) -> Path:
    path = tmp_path / "template.yaml"
    path.write_text(_TEMPLATE.format(mode_line=mode_line))
    return path


def test_template_schema_accepts_array_submit_mode(tmp_path: Path) -> None:
    loaded = _load_template(_write(tmp_path, "submit_mode: array"))
    assert loaded["submit_mode"] == "array"
    # Absent submit_mode keeps the per-replica default.
    assert "submit_mode" not in _load_template(_write(tmp_path, ""))


def test_template_schema_rejects_unknown_submit_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid Slurm runner-template"):
        _load_template(_write(tmp_path, "submit_mode: parallel"))


@pytest.mark.parametrize(
    "name,gpu,cpus,mem,array_mode",
    [
        ("default", None, 6, "32G", False),
        ("default", "L40S", 6, "32G", False),
        ("cpu-array", None, 8, "16G", True),
    ],
)
def test_shipped_template_renders_complete_submission(
    tmp_path: Path, monkeypatch, name, gpu, cpus, mem, array_mode
) -> None:
    monkeypatch.setenv("CS_AGENT_PYTHON", sys.executable)
    template_path = resolve_runner_selection("slurm", name).template_path
    context = ComputeRunnerContext(
        submission=None,
        run_dir=tmp_path,
        run_id="run1",
        base_runtime_env={},
        submission_environment_names=(),
        init_cache_key="cachekey",
        python_packages=("requests>=2.31",),
        managed_runtime=True,
        gpu=gpu,
        track_name="default",
        agents_per_evaluation=1,
        events=None,
    )
    runner = SlurmComputeRunner(context, template_path)
    assert runner.array_mode is array_mode

    control_dir = tmp_path / "compute" / "evaluation_1"
    shard = [(SimpleNamespace(timeout_sec=300.0), 5)]
    values = runner._job_values(
        1,
        shard,
        control_dir,
        control_dir / "submission",
        control_dir / "slurm.log",
    )

    rendered = [
        _render(item, values, template_path)
        for item in runner.template["submit_command"]
    ]
    if gpu:
        rendered.extend(
            _render(item, values, template_path)
            for item in runner.template["gpu_arguments"]
        )
        assert "--gres=gpu:L40S:1" in rendered
    else:
        assert not any("gres" in item for item in rendered)
    assert ("--array=1-1%1" in rendered) is array_mode
    assert f"--cpus-per-task={cpus}" in rendered
    assert f"--mem={mem}" in rendered
    assert not any(
        item.startswith(("--partition", "--qos", "--account")) for item in rendered
    )

    script = _render(runner.template["script"], values, template_path)
    assert "--control-dir" in values["worker_command"]
    assert values["worker_command"] in script


@pytest.mark.parametrize("name", ["default", "default.yaml", "cpu-array"])
def test_bundled_template_lookup_is_independent_of_working_directory(
    tmp_path: Path, monkeypatch, name
) -> None:
    monkeypatch.chdir(tmp_path)
    selection = resolve_runner_selection("slurm", name)
    expected = (
        Path(__file__).resolve().parents[1]
        / "src/cua_speedrun/compute_runners/templates/slurm"
        / f"{Path(name).stem}.yaml"
    )
    assert selection.template_path == expected
    assert selection.template_path.is_file()


def test_runner_accepts_operator_template_path(tmp_path: Path) -> None:
    path = _write(tmp_path, "")
    assert resolve_runner_selection("slurm", path).template_path == path.resolve()


def test_runlog_parsing_survives_unicode_line_separators() -> None:
    # Verdict feedback can carry characters like U+2028 or emoji whose text
    # crosses str.splitlines' extended line-break set. The parser must split
    # on newline only, or a fetched run log breaks mid-string and the
    # instance is misclassified as infrastructure loss.
    from cua_speedrun.compute_runners.slurm import _parse_runlog_text

    import json as jsonlib

    verdict = {"event": "verdict", "detail": "terminal said done \U0001F605"}
    text = jsonlib.dumps({"event": "header"}, ensure_ascii=False) + "\n"
    text += jsonlib.dumps(verdict, ensure_ascii=False) + "\n"
    events = _parse_runlog_text(text)
    assert [e["event"] for e in events] == ["header", "verdict"]
    assert events[1]["detail"] == "terminal said done \U0001F605"


def test_kvm_exclusions_persist_across_evaluations(tmp_path: Path) -> None:
    path = tmp_path / "runners" / "slurm" / "kvm-excluded-nodes.json"
    assert _load_excluded_nodes(path) == set()
    _save_excluded_nodes(path, {"compute-01", "compute-02"})
    assert _load_excluded_nodes(path) == {"compute-01", "compute-02"}
    # A corrupt file degrades to an empty set instead of failing the run.
    path.write_text("not json")
    assert _load_excluded_nodes(path) == set()


def test_worker_kvm_preflight_message_matches_exclusion_classifier() -> None:
    # The worker's preflight error is the coordinator's only signal that a
    # node lacks /dev/kvm. If the wording drifts out of the classifier's
    # marker set, KVM-less nodes stop being excluded and burn retry budgets
    # as generic infrastructure noise instead.
    from cua_speedrun.compute_runners.slurm import _is_kvm_node_capability_error
    from cua_speedrun.compute_runners.slurm_worker import _KVM_UNAVAILABLE

    assert _is_kvm_node_capability_error(
        repr(RuntimeError(f"{_KVM_UNAVAILABLE}: compute-03"))
    )


def test_array_elements_report_scheduler_addressable_ids(monkeypatch) -> None:
    # Array elements must report <array_job>_<task>, the form squeue,
    # scancel, and sacct accept, or the coordinator would compare against
    # the element's raw SLURM_JOB_ID and declare the replica lost forever.
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "9400001")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    assert scheduler_job_id() == "9400001_7"

    monkeypatch.delenv("SLURM_ARRAY_JOB_ID")
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID")
    assert scheduler_job_id() == "424242"
