from __future__ import annotations

import json

from cua_speedrun.events import Event
from cua_speedrun.service.db import (
    BenchmarkRow,
    EventRow,
    Run,
    RunTask,
    SubmissionRow,
    User,
    make_session_factory,
)
from cua_speedrun.service.task_cost import (
    COST_SNAPSHOT_PREFIX,
    latest_cost_snapshot,
    read_task_cost_snapshot,
    total_task_costs,
)
from cua_speedrun.service.worker import DBSink


def snapshot(cost_usd: float, usage: dict) -> str:
    return COST_SNAPSHOT_PREFIX + json.dumps({
        "cost_usd": cost_usd,
        "usage": usage,
    })


def test_latest_cost_snapshot_replaces_earlier_cumulative_values() -> None:
    output = "\n".join((
        "ordinary agent output",
        snapshot(0.1, {"input_tokens": 10}),
        COST_SNAPSHOT_PREFIX + "not-json",
        snapshot(0.25, {"input_tokens": 30, "cached_tokens": 8}),
    ))

    assert latest_cost_snapshot(output) == {
        "cost_usd": 0.25,
        "usage": {"input_tokens": 30, "cached_tokens": 8},
    }


def test_read_task_cost_snapshot_stays_inside_run_directory(tmp_path) -> None:
    task_dir = tmp_path / "tasks" / "task-one" / "seed_3"
    task_dir.mkdir(parents=True)
    (task_dir / "agent.stdout").write_text(snapshot(0.4, {"calls": 2}) + "\n")

    assert read_task_cost_snapshot(tmp_path, "task-one/seed_3") == {
        "cost_usd": 0.4,
        "usage": {"calls": 2},
    }
    assert read_task_cost_snapshot(tmp_path, "../outside/seed_3") is None


def test_total_task_costs_sums_only_final_task_snapshots() -> None:
    assert total_task_costs((
        (0.25, {"input_tokens": 30, "cached_tokens": 8}),
        (0.75, {"input_tokens": 70, "output_tokens": 4}),
        (None, None),
    )) == (
        1.0,
        {"input_tokens": 100, "cached_tokens": 8, "output_tokens": 4},
    )


def test_db_sink_persists_latest_task_snapshot(tmp_path) -> None:
    session_factory = make_session_factory(f"sqlite:///{tmp_path / 'platform.db'}")
    run_dir = tmp_path / "runs" / "svc_1"
    task_dir = run_dir / "tasks" / "task-one" / "seed_0"
    task_dir.mkdir(parents=True)
    (task_dir / "agent.stdout").write_text("\n".join((
        snapshot(0.1, {"input_tokens": 10}),
        snapshot(0.3, {"input_tokens": 25, "output_tokens": 4}),
    )))
    with session_factory() as session:
        user = User(handle="runner")
        benchmark = BenchmarkRow(name="demo", version="1", path="/unused")
        session.add_all([user, benchmark])
        session.flush()
        submission = SubmissionRow(
            user_id=user.id,
            name="agent",
            storage_ref="agent.zip",
            track="open-l4",
        )
        session.add(submission)
        session.flush()
        run = Run(
            submission_id=submission.id,
            benchmark_id=benchmark.id,
            stage="running",
            run_dir=str(run_dir),
        )
        session.add(run)
        session.commit()
        run_id = run.id

    sink = DBSink(session_factory, run_id)
    sink(Event(
        kind="task_done",
        ts=1.0,
        task_key="task-one/seed_0",
        payload={"passed": True, "num_steps": 2},
    ))

    with session_factory() as session:
        task = session.query(RunTask).filter_by(run_id=run_id).one()
        event = session.query(EventRow).filter_by(run_id=run_id).one()
        assert task.cost_usd == 0.3
        assert task.usage == {"input_tokens": 25, "output_tokens": 4}
        assert event.payload["cost_usd"] == 0.3
