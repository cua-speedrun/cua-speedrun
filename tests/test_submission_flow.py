from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from cua_speedrun.service import api
from cua_speedrun.service.db import (
    BenchmarkRow,
    Run,
    SubmissionRow,
    Track,
    User,
    make_session_factory,
)
from cua_speedrun.service.store import LocalStore
from cua_speedrun.service.worker import _claim_next
from cua_speedrun.eval_algorithms import SHARED_AGENT_VLLM


def _benchmark(root: Path, name: str) -> Path:
    benchmark = root / name
    task = benchmark / "tasks" / "one"
    environment = root / f"{name}-environment"
    task.mkdir(parents=True)
    environment.mkdir()
    (benchmark / "manifest.yaml").write_text(
        f'name: {name}\nversion: "1"\ntasks:\n  - tasks/one\n'
    )
    (task / "task.yaml").write_text(
        "task_id: one\n"
        "description: tiny task\n"
        "env:\n"
        "  kind: gym-anything\n"
        f"  env_dir: {environment}\n"
        "  task_id: one\n"
    )
    (environment / "env.json").write_text('{"name": "test"}')
    return benchmark


def test_starter_submission_cannot_override_selected_contract(
    tmp_path, monkeypatch
) -> None:
    session_factory = make_session_factory(f"sqlite:///{tmp_path / 'platform.db'}")
    benchmark_path = _benchmark(tmp_path, "maintainer-benchmark")
    with session_factory() as session:
        user = User(
            handle="runner",
            quota_tier="dev",
            modal_token_id="ak-test",
            modal_token_secret_enc="encrypted",
        )
        track = Track(name="maintainer-track", gpu="L4")
        benchmark = BenchmarkRow(
            name="maintainer-benchmark",
            version="1",
            path=str(benchmark_path),
        )
        session.add_all([user, track, benchmark])
        session.commit()

    monkeypatch.setattr(api, "session_factory", session_factory)
    monkeypatch.setattr(api, "store", LocalStore(tmp_path / "store"))
    api.app.dependency_overrides[api.current_user] = lambda: user
    try:
        response = TestClient(api.app).post(
            "/api/submissions",
            params={
                "name": "starter-agent",
                "track": track.name,
                "benchmark_id": benchmark.id,
                "template": "qwen3vl",
            },
        )
    finally:
        api.app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    with session_factory() as session:
        submission = session.get(SubmissionRow, response.json()["submission_id"])
        run = session.get(Run, response.json()["run_id"])
        assert submission.track == "maintainer-track"
        assert run.benchmark_id == benchmark.id
        assert run.execution_plan["track"]["name"] == "maintainer-track"
        assert run.execution_plan["benchmark"]["name"] == "maintainer-benchmark"


def test_new_evaluation_page_separates_contract_and_agent(tmp_path) -> None:
    from fastapi import FastAPI

    from cua_speedrun.service.web import register_web

    session_factory = make_session_factory(f"sqlite:///{tmp_path / 'web.db'}")
    with session_factory() as session:
        user = User(handle="runner")
        session.add_all([
            user,
            Track(name="open-l4", gpu="L4"),
            BenchmarkRow(
                name="demo",
                version="1",
                path=str(_benchmark(tmp_path, "demo")),
            ),
        ])
        session.commit()

    app = FastAPI()
    register_web(app, session_factory, lambda _request: user.id)
    response = TestClient(app).get("/submit")

    assert response.status_code == 200
    assert "Step 1 of 2" in response.text
    assert "Step 2 of 2" in response.text
    assert "Maintainer-defined contract" in response.text
    assert "Your two-file submission" in response.text
    assert 'data-wizard-panel="agent"' in response.text
    assert 'name="compute_placement"' in response.text
    assert 'name="environment_placement"' in response.text
    assert 'name="execution_target"' not in response.text
    assert response.text.count(">Local<") >= 2
    assert "Modal" in response.text


def test_fully_local_submission_needs_no_modal_credentials_and_freezes_topology(
    tmp_path, monkeypatch
) -> None:
    session_factory = make_session_factory(f"sqlite:///{tmp_path / 'local.db'}")
    benchmark_path = _benchmark(tmp_path, "local-benchmark")
    with session_factory() as session:
        user = User(handle="local-runner", quota_tier="dev")
        track = Track(
            name="open-l40s-shared",
            gpu="L40S",
            eval_algorithm=SHARED_AGENT_VLLM,
            max_concurrency=2,
            env_pool_size=4,
        )
        benchmark = BenchmarkRow(
            name="local-benchmark", version="1", path=str(benchmark_path)
        )
        session.add_all([user, track, benchmark])
        session.commit()

    monkeypatch.setattr(api, "session_factory", session_factory)
    monkeypatch.setattr(api, "store", LocalStore(tmp_path / "store"))
    api.app.dependency_overrides[api.current_user] = lambda: user
    try:
        response = TestClient(api.app).post(
            "/api/submissions",
            params={
                "name": "local-agent",
                "track": track.name,
                "benchmark_id": benchmark.id,
                "compute_placement": "local",
                "environment_placement": "local",
                "template": "qwen3vl",
            },
        )
    finally:
        api.app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    with session_factory() as session:
        run = session.get(Run, response.json()["run_id"])
        assert run.execution_plan["backend"] == "local"
        assert run.execution_plan["execution"]["topology"] == {
            "key": "local",
            "backend": "local",
            "compute": {
                "key": "local",
                "mode": "local",
                "provider": None,
                "requires_user_credentials": False,
            },
            "environment": {
                "key": "local",
                "mode": "local",
                "provider": None,
                "requires_user_credentials": False,
            },
            "environment_backend": "gym-anything-local",
            "requires_user_credentials": False,
        }
        assert run.execution_plan["track"]["gpu"] == "L40S"
        assert run.execution_plan["track"]["network_policy"] == "host-network"
        assert run.execution_plan["agent_runtime"]["execution_scope"] == "evaluator-host"
        assert run.execution_plan["agent_runtime"]["runtime_resolution"] == "requested"


def test_runner_selection_is_frozen_on_the_run_at_submit(tmp_path) -> None:
    """The CLI's --runner choice must bind whichever executor claims the run.

    A queue worker that wins the claim race used to execute with its own
    process defaults, silently ignoring the submitted flags.
    """
    import io
    import zipfile

    import pytest

    from cua_speedrun.service.evaluations import (
        EvaluationServiceError,
        queue_evaluation,
    )

    session_factory = make_session_factory(f"sqlite:///{tmp_path / 'runner.db'}")
    benchmark_path = _benchmark(tmp_path, "runner-benchmark")
    with session_factory() as session:
        user = User(handle="runner", quota_tier="dev")
        track = Track(
            name="open-l40s-shared",
            gpu="L40S",
            eval_algorithm=SHARED_AGENT_VLLM,
        )
        benchmark = BenchmarkRow(
            name="runner-benchmark", version="1", path=str(benchmark_path)
        )
        session.add_all([user, track, benchmark])
        session.commit()
        user_id, benchmark_id = user.id, benchmark.id

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("init.py", "print('ready')\n")
        archive.writestr("agent.py", "print('agent')\n")

    def submit(**overrides):
        arguments = dict(
            session_factory=session_factory,
            store=LocalStore(tmp_path / "store"),
            user_id=user_id,
            submission_zip=payload.getvalue(),
            name="runner-choice",
            track_name="open-l40s-shared",
            benchmark_id=benchmark_id,
            compute_placement="local",
            environment_placement="local",
        )
        arguments.update(overrides)
        return queue_evaluation(**arguments)

    started = submit(runner="slurm", runner_template="cpu-array")
    with session_factory() as session:
        run = session.get(Run, started["run_id"])
        assert run.runner_selection == {
            "kind": "slurm",
            "template": "cpu-array",
        }

    # An unknown template must fail at submit, not at claim time.
    with pytest.raises(EvaluationServiceError, match="runner template"):
        submit(runner="slurm", runner_template="no-such-template")


def test_workers_claim_only_their_execution_topology(tmp_path) -> None:
    session_factory = make_session_factory(f"sqlite:///{tmp_path / 'routing.db'}")
    with session_factory() as session:
        user = User(handle="runner")
        benchmark = BenchmarkRow(name="demo", version="1", path="/unused")
        session.add_all([user, benchmark])
        session.flush()
        remote_submission = SubmissionRow(
            user_id=user.id, name="remote", storage_ref="remote.zip", track="track"
        )
        local_submission = SubmissionRow(
            user_id=user.id, name="local", storage_ref="local.zip", track="track"
        )
        session.add_all([remote_submission, local_submission])
        session.flush()
        remote_run = Run(
            submission_id=remote_submission.id,
            benchmark_id=benchmark.id,
            stage="queued",
            topology_key="modal-remote",
        )
        local_run = Run(
            submission_id=local_submission.id,
            benchmark_id=benchmark.id,
            stage="queued",
            topology_key="local",
        )
        session.add_all([remote_run, local_run])
        session.commit()

    assert _claim_next(session_factory, "local") == local_run.id
    assert _claim_next(session_factory, "modal-remote") == remote_run.id
