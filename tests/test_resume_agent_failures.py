from __future__ import annotations

from cua_speedrun.service.db import (
    BenchmarkRow,
    Card,
    EventRow,
    Run,
    RunTask,
    SubmissionRow,
    User,
    make_session_factory,
)
from cua_speedrun.service.evaluations import resume_evaluation


def test_rerun_agent_failures_reopens_unpublished_scored_run(tmp_path) -> None:
    session_factory = make_session_factory(f"sqlite:///{tmp_path / 'platform.db'}")
    with session_factory() as session:
        user = User(handle="runner")
        benchmark = BenchmarkRow(name="demo", version="1", path="/unused")
        session.add_all([user, benchmark])
        session.flush()
        submission = SubmissionRow(
            user_id=user.id,
            name="agent",
            storage_ref="agent.zip",
            track="track",
        )
        session.add(submission)
        session.flush()
        run = Run(
            submission_id=submission.id,
            benchmark_id=benchmark.id,
            stage="card_ready",
            execution_plan={"benchmark": {"task_ids": ["one", "two"]}},
            task_seeds={"one": [11], "two": [22]},
            season_key="old-season",
        )
        session.add(run)
        session.flush()
        session.add_all([
            RunTask(
                run_id=run.id,
                task_key="one/seed_11",
                stage="done",
                passed=False,
                reason="agent_error",
                task_time_sec=1.0,
                agent_time_sec=1.0,
                env_time_sec=0.0,
                num_steps=0,
            ),
            RunTask(
                run_id=run.id,
                task_key="two/seed_22",
                stage="done",
                passed=True,
                reason="done",
                task_time_sec=2.0,
                agent_time_sec=1.0,
                env_time_sec=1.0,
                num_steps=1,
            ),
            Card(run_id=run.id, token="old-card", data={}),
        ])
        session.commit()
        run_id, user_id = run.id, user.id

    resumed = resume_evaluation(
        session_factory,
        run_id,
        user_id,
        rerun_agent_failures=True,
    )

    assert resumed["task_count"] == 1
    with session_factory() as session:
        run = session.get(Run, run_id)
        tasks = {
            task.task_key: task
            for task in session.query(RunTask).filter_by(run_id=run_id)
        }
        event = session.query(EventRow).filter_by(
            run_id=run_id, kind="run_resume_queued"
        ).one()
        assert run.stage == "queued"
        assert run.season_key is None
        assert tasks["one/seed_11"].stage == "queued"
        assert tasks["one/seed_11"].reason is None
        assert tasks["two/seed_22"].stage == "done"
        assert session.query(Card).filter_by(run_id=run_id).one_or_none() is None
        assert event.payload["task_keys"] == ["one/seed_11"]
        assert event.payload["rerun_agent_failures"] is True
