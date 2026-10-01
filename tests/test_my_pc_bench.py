"""MyPCBench energy-38 source, grading, and runtime contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import yaml
from sqlalchemy import select

from cua_speedrun.evaluator_environment import (
    agent_environment,
    evaluator_environment,
    validate,
)
from cua_speedrun.remote.modal_native_env import resolve_base_image_id
from cua_speedrun.service.templates_catalog import list_templates
from cua_speedrun.specs import Benchmark

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/my-pc-bench/benchmark-source.yaml"
BUILDER = ROOT / "scripts/build_mypcbench_subset.py"
RUNTIME_INPUTS = {
    "scripts/build_mypcbench_subset.py",
    "benchmarks/mypcbench-image.json",
    "scripts/build_mypcbench_modal_image.py",
    "scripts/mypcbench_shared/canonical/osworld_full_traj_judge.py",
    "scripts/mypcbench_shared/canonical/provenance.json",
    "scripts/mypcbench_shared/mypcbench_setup.py",
    "scripts/mypcbench_shared/mypcbench_verifier.py",
    "scripts/mypcbench_shared/native_delta.sh",
    "scripts/mypcbench_shared/native_image.py",
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    output = tmp_path_factory.mktemp("mypcbench") / "benchmark"
    builder = _load_module(BUILDER, "mypcbench_builder")
    builder.materialize(SOURCE, output)
    return Benchmark.load(output), output


def test_source_materializes_only_the_pinned_tasks(benchmark) -> None:
    loaded, output = benchmark
    source = yaml.safe_load(SOURCE.read_text())

    assert source["name"] == "my-pc-bench"
    assert len(source["tasks"]) == len(loaded.tasks) == 38
    assert RUNTIME_INPUTS == set(source["materializer"]["inputs"])
    assert source["grading"] == {
        "rubric_aggregation": "uniform_mean",
        "pass_condition": "all_rubrics",
    }
    assert all(isinstance(task_id, str) for task_id in source["tasks"])
    task_source = source["source_benchmark"]["tasks"]
    assert source["source_benchmark"]["commit"] in task_source["url"]
    assert task_source["sha256"] == (
        "e36d9c08b8c7b6c84304c84578b64b968da0809fdda51f76795623919fc7a384"
    )
    assert task_source["task_count"] == 184
    selector = source["selection"]["selector"]
    assert selector["commit"] == "b2f69586e413d84445ad2f6450bf9c0487f05c5f"
    assert selector["starts"] == 100
    assert selector["seed_base"] == 7000
    assert selector["rank_threshold"] == 0.95
    selector_path = ROOT / selector["path"]
    assert hashlib.sha256(selector_path.read_bytes()).hexdigest() == selector["sha256"]

    manifest = yaml.safe_load((output / "manifest.yaml").read_text())
    environment = json.loads((output / "environment/env.json").read_text())
    runtime = json.loads((output / "environment/host-runtime.json").read_text())
    evaluator = json.loads(
        (output / "environment/evaluator-environment.json").read_text()
    )
    assert runtime["forward_env"] == ["MYPCBENCH_JUDGE_API_KEY"]
    assert evaluator == {
        "private": ["MYPCBENCH_JUDGE_API_KEY"],
        "required": ["MYPCBENCH_JUDGE_API_KEY"],
    }
    assert manifest["grading"] == source["grading"]
    assert environment["action_settle_ms"] == 2000
    assert all(task.env["action_settle_ms"] == 2000 for task in loaded.tasks)
    assert manifest["version"] == "0.2"
    assert all(task.timeout_sec == 7200 for task in loaded.tasks)
    materialized_task = json.loads(
        (output / "environment/tasks/retrieval-f029/task.json").read_text()
    )
    prompt = materialized_task["natural_language"]["prompt"]
    source_instruction = json.loads(
        (output / "environment/tasks/retrieval-f029/source.json").read_text()
    )["instruction"]
    assert materialized_task["init"]["max_steps"] == 100
    assert materialized_task["init"]["timeout_sec"] == 7200
    assert "Michael Scott" in prompt
    assert "GUI session only" in prompt
    assert "| 3017 | HooliCalendar" in prompt
    assert "sudo password" not in prompt
    assert "Python 3.12" not in prompt
    assert prompt.endswith("Task:\n" + source_instruction)
    assert all(
        task.metadata["mypcbench_rubric_aggregation"] == "uniform_mean"
        for task in loaded.tasks
    )


def test_judge_key_is_required_and_private(benchmark) -> None:
    loaded, _ = benchmark
    for values in ({}, {"MYPCBENCH_JUDGE_API_KEY": "  "}):
        with pytest.raises(ValueError, match="MYPCBENCH_JUDGE_API_KEY"):
            validate(loaded, values)

    values = {
        "MYPCBENCH_JUDGE_API_KEY": "judge-value",
        "AGENT_API_KEY": "agent-value",
    }
    validate(loaded, values)
    assert agent_environment(loaded, values) == {
        "AGENT_API_KEY": "agent-value"
    }
    assert evaluator_environment(loaded, values) == {
        "MYPCBENCH_JUDGE_API_KEY": "judge-value"
    }


def test_admission_rejects_missing_judge_key_before_queue(
    benchmark, tmp_path
) -> None:
    from cua_speedrun.service.db import (
        BenchmarkRow,
        Run,
        SubmissionRow,
        User,
        make_session_factory,
    )
    from cua_speedrun.service.evaluations import (
        EvaluationServiceError,
        queue_evaluation,
    )
    from cua_speedrun.service.store import LocalStore

    _, benchmark_path = benchmark
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'admission.db'}")
    with sessions() as session:
        user = User(handle="mypcbench-admission")
        row = BenchmarkRow(
            name="my-pc-bench",
            version="0.1",
            path=str(benchmark_path),
            task_count=38,
        )
        session.add_all([user, row])
        session.commit()
        user_id, benchmark_id = user.id, row.id

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        for name in ("agent.py", "init.py"):
            zipped.write(ROOT / "agents/codex_cli" / name, name)

    with pytest.raises(
        EvaluationServiceError, match="MYPCBENCH_JUDGE_API_KEY"
    ) as error:
        queue_evaluation(
            session_factory=sessions,
            store=LocalStore(tmp_path / "store"),
            user_id=user_id,
            submission_zip=archive.getvalue(),
            name="missing judge key",
            track_name="not-reached",
            benchmark_id=benchmark_id,
            compute_placement="modal",
            environment_placement="modal-native",
        )

    assert error.value.status_code == 400
    with sessions() as session:
        assert session.scalars(select(Run)).all() == []
        assert session.scalars(select(SubmissionRow)).all() == []


def test_setup_waits_for_post_app_seed_finalization(tmp_path, monkeypatch) -> None:
    setup = _load_module(
        ROOT / "scripts/mypcbench_shared/mypcbench_setup.py",
        "mypcbench_setup_test",
    )
    unit = tmp_path / "mypcbench-canon-patch-post.service"
    unit.write_text("[Service]\n")
    monkeypatch.setattr(setup, "SEED_FINALIZATION_UNIT_FILE", unit)
    replies = iter(
        [
            "ActiveState=active\nSubState=running\nResult=success\n"
            "ExecMainStatus=0\nExecMainStartTimestampMonotonic=10\n",
            "ActiveState=inactive\nSubState=dead\nResult=success\n"
            "ExecMainStatus=0\nExecMainStartTimestampMonotonic=10\n",
        ]
    )
    monkeypatch.setattr(
        setup,
        "_run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=next(replies), stderr=""
        ),
    )
    monkeypatch.setattr(setup.time, "sleep", lambda _seconds: None)
    setup.wait_for_seed_finalization()


def test_setup_rejects_failed_seed_finalization(tmp_path, monkeypatch) -> None:
    setup = _load_module(
        ROOT / "scripts/mypcbench_shared/mypcbench_setup.py",
        "mypcbench_setup_failure_test",
    )
    unit = tmp_path / "mypcbench-canon-patch-post.service"
    unit.write_text("[Service]\n")
    monkeypatch.setattr(setup, "SEED_FINALIZATION_UNIT_FILE", unit)
    monkeypatch.setattr(
        setup,
        "_run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "ActiveState=inactive\nSubState=dead\nResult=success\n"
                "ExecMainStatus=1\nExecMainStartTimestampMonotonic=10\n"
            ),
            stderr="",
        ),
    )
    with pytest.raises(setup.SetupError, match="did not complete successfully"):
        setup.wait_for_seed_finalization()


def test_image_and_judge_provenance_are_pinned(benchmark) -> None:
    _, output = benchmark
    image = json.loads((ROOT / "benchmarks/mypcbench-image.json").read_text())[
        "official_source"
    ]
    source = yaml.safe_load(SOURCE.read_text())["source_benchmark"]
    provenance = json.loads(
        (ROOT / "scripts/mypcbench_shared/canonical/provenance.json").read_text()
    )
    judge = ROOT / "scripts/mypcbench_shared/canonical/osworld_full_traj_judge.py"

    assert source["image_revision"] == image["revision"]
    assert source["image_version"] == image["image_version"]
    assert len(image["image_sha256"]) == 64
    assert hashlib.sha256(judge.read_bytes()).hexdigest() == provenance["sha256"]
    assert provenance["modifications"] == "none"

    contract = json.loads((output / "environment/native-image.json").read_text())
    assert contract["expected_provenance"]["validated"] is True
    assert "builder" not in contract


def test_all_source_rubrics_are_reweighted_uniformly(tmp_path: Path) -> None:
    adapter = _load_module(
        ROOT / "scripts/mypcbench_shared/mypcbench_verifier.py",
        "mypcbench_verifier",
    )
    screenshot = tmp_path / "evidence.png"
    screenshot.write_bytes(b"evidence")
    for count in range(1, 14):
        source = {
            "id": f"task-{count}",
            "instruction": "Complete the task",
            "grading": {
                "rubrics": [
                    {"criterion": str(index), "weight": index + 1}
                    for index in range(count)
                ]
            },
        }
        original = [rubric["weight"] for rubric in source["grading"]["rubrics"]]
        bundle = adapter.build_bundle(
            source,
            {"evidence": [{"screenshot": str(screenshot), "action": "done"}]},
        )
        assert [
            r["weight"] for r in bundle["grading_manifest"]["rubrics"]
        ] == pytest.approx([1 / count] * count, abs=1e-12)
        assert [rubric["weight"] for rubric in source["grading"]["rubrics"]] == original


def test_uniform_result_rejects_weighted_or_inconsistent_output() -> None:
    adapter = _load_module(
        ROOT / "scripts/mypcbench_shared/mypcbench_verifier.py",
        "mypcbench_result_verifier",
    )
    rows = [
        {"weight": 1 / 3, "success": True},
        {"weight": 1 / 3, "success": False},
        {"weight": 1 / 3, "success": False},
    ]
    result = adapter.uniform_result(
        {"score": 33, "passed": False, "rubric_results": rows}, 3
    )
    assert (result["score"], result["passed"]) == (33, False)
    with pytest.raises(adapter.EvaluatorInfrastructure, match="uniform rubric weights"):
        adapter.uniform_result(
            {
                "score": 70,
                "passed": False,
                "rubric_results": [
                    {"weight": 0.7, "success": True},
                    {"weight": 0.2, "success": False},
                    {"weight": 0.1, "success": False},
                ],
            },
            3,
        )
    with pytest.raises(adapter.EvaluatorInfrastructure, match="non-uniform"):
        adapter.uniform_result(
            {"score": 34, "passed": False, "rubric_results": rows}, 3
        )


def test_native_image_rejects_unbuilt_or_unvalidated_snapshot(benchmark) -> None:
    _, output = benchmark

    contract = json.loads((output / "environment/native-image.json").read_text())
    with pytest.raises(RuntimeError, match="not been built"):
        resolve_base_image_id(
            {}, cache_key=contract["cache_key"], expected=contract["expected_provenance"],
        )
    with pytest.raises(ValueError, match="validated"):
        resolve_base_image_id(
            {
                contract["cache_key"]: {
                    **contract["expected_provenance"],
                    "validated": False,
                }
            },
            cache_key=contract["cache_key"], expected=contract["expected_provenance"],
        )


def test_desktop_templates_support_my_pc_bench() -> None:
    templates = list_templates()
    desktop = [item for item in templates if "osworld-50" in item["compatible_benchmarks"]]
    assert desktop
    assert all(
        "my-pc-bench" in item["compatible_benchmarks"] for item in desktop
    )
    assert {"meta", "codex_cli"} <= {item["name"] for item in desktop}


def test_native_judge_reuses_gateway_frames_and_applied_actions(tmp_path: Path) -> None:
    from cua_speedrun.runlog import RunLogWriter

    verifier = _load_module(
        ROOT / "scripts/mypcbench_shared/mypcbench_verifier.py",
        "mypcbench_gateway_evidence",
    )
    log = RunLogWriter(tmp_path / "runlog.jsonl")
    log.event("observe", t_mono_complete=10.0, frame="frame_00000.png")
    log.event("step", t_mono_complete=15.0, actions=[{"rejected_request": True}])
    log.event("observe", t_mono_complete=20.0, frame="frame_00001.png")
    log.close()
    actions = [[{"action": "wait", "time": seconds}] for seconds in range(4)]
    trajectory = verifier.native_trajectory(
        tmp_path,
        {"steps": [
            {"event": "step", "t_mono": timestamp, "action": action}
            for timestamp, action in zip((5.0, 11.0, 12.0, 21.0), actions)
        ]},
    )
    first = str(tmp_path / "frame_00000.png")
    second = str(tmp_path / "frame_00001.png")
    assert trajectory["evidence"] == [
        {"action": actions[0]},
        {"screenshot": first, "action": actions[1]},
        {"screenshot": first, "action": actions[2]},
        {"screenshot": second, "action": actions[3]},
    ]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["runlog.jsonl"]


def test_native_judge_preserves_agent_response_text(tmp_path: Path) -> None:
    from cua_speedrun.runlog import RunLogWriter

    verifier = _load_module(
        ROOT / "scripts/mypcbench_shared/mypcbench_verifier.py",
        "mypcbench_gateway_response_evidence",
    )
    frame = tmp_path / "frame_00000.png"
    frame.write_bytes(b"png")
    log = RunLogWriter(tmp_path / "runlog.jsonl")
    log.event("observe", t_mono_complete=10.0, frame=frame.name)
    log.event(
        "done",
        t_mono_complete=20.0,
        raw_response_text="The requested total is $42.",
    )
    log.close()

    trajectory = verifier.native_trajectory(tmp_path, {"steps": []})
    bundle = verifier.build_bundle(
        {
            "id": "response-task",
            "instruction": "Report the total",
            "grading": {"rubrics": [{"criterion": "Reports $42"}]},
        },
        trajectory,
    )
    assert bundle["artifacts"]["steps"] == [
        {
            "step_num": 0,
            "screenshot": str(frame.resolve()),
            "raw_response_text": "The requested total is $42.",
            "parsed_action_obj": {"action": None},
        }
    ]


def test_native_image_uses_the_existing_resolver(benchmark) -> None:
    _, output = benchmark
    contract = json.loads((output / "environment/native-image.json").read_text())
    record = {**contract["expected_provenance"], "modal_snapshot_image_id": "im-contract-test"}
    assert resolve_base_image_id(
        {contract["cache_key"]: record},
        cache_key=contract["cache_key"], expected=contract["expected_provenance"],
    ) == "im-contract-test"
    assert not (ROOT / "src/cua_speedrun/remote/native_image.py").exists()
    executor = (ROOT / "src/cua_speedrun/remote/run.py").read_text()
    assert "ensure_image" not in executor
    assert "environment_image_provisioning_started" not in executor


def test_judge_scrub_and_guest_environment_do_not_expose_evaluator_keys() -> None:
    import ast
    import inspect
    from cua_speedrun.envs import modal_native

    assert set(modal_native._guest_process_environment()) == {"PATH", "LANG"}
    tree = ast.parse(inspect.getsource(modal_native.scrub_privileged))
    command = next(
        node.value.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "command" for target in node.targets)
    )
    assert "*_judge.py" in command
    assert "echo scrubbed" not in command
    run_call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
    )
    assert any(
        keyword.arg == "check" and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True for keyword in run_call.keywords
    )
