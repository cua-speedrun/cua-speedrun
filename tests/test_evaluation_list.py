from __future__ import annotations

from cua_speedrun.commands.dashboard_client import _print_evaluations
from cua_speedrun.service.db import (
    BenchmarkRow,
    Run,
    RunTask,
    SubmissionRow,
    User,
    make_session_factory,
)
from cua_speedrun.service.evaluations import list_evaluations


def test_evaluation_list_reports_mean_per_task_timings(tmp_path) -> None:
    session_factory = make_session_factory(f"sqlite:///{tmp_path / 'platform.db'}")
    with session_factory() as session:
        user = User(handle="runner")
        benchmark = BenchmarkRow(
            name="demo", version="1", path="/unused", task_count=3
        )
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
            topology_key="modal-remote",
        )
        session.add(run)
        session.flush()
        session.add_all([
            RunTask(
                run_id=run.id,
                task_key="one/seed_1",
                stage="done",
                passed=True,
                task_time_sec=12.0,
                agent_time_sec=7.0,
                env_time_sec=5.0,
                cost_usd=0.25,
                usage={"input_tokens": 100, "cached_tokens": 20},
            ),
            RunTask(
                run_id=run.id,
                task_key="two/seed_2",
                stage="done",
                passed=False,
                task_time_sec=6.0,
                agent_time_sec=2.0,
                env_time_sec=4.0,
                cost_usd=0.75,
                usage={"input_tokens": 200, "output_tokens": 10},
            ),
            RunTask(
                run_id=run.id,
                task_key="three/seed_3",
                stage="running",
            ),
        ])
        session.commit()
        user_id = user.id

    [summary] = list_evaluations(session_factory, user_id)

    assert summary["per_task"] == {
        "time_sec": 9.0,
        "agent_time_sec": 4.5,
        "env_time_sec": 4.5,
    }
    assert summary["cost_usd"] == 1.0
    assert summary["usage"] == {
        "input_tokens": 300,
        "cached_tokens": 20,
        "output_tokens": 10,
    }


def test_evaluation_table_prints_per_task_timing_columns(capsys) -> None:
    _print_evaluations([{
        "run_id": 40,
        "stage": "card_ready",
        "name": "gemini",
        "track": "open-l40s",
        "benchmark": "osworld@1",
        "topology": "modal-remote",
        "progress": {"finished": 48, "total": 48, "passed": 24, "failed": 24},
        "score": {
            "mean_score": 0.73,
            "success_rate": 0.5,
            "total_time_sec": 19_567.0,
        },
        "cost_usd": 12.34,
        "per_task": {
            "time_sec": 407.65,
            "agent_time_sec": 328.41,
            "env_time_sec": 79.24,
        },
        "elapsed_sec": 3_318.0,
    }])

    output = capsys.readouterr().out
    assert "TASK AVG" in output
    assert "AGENT AVG" in output
    assert "ENV AVG" in output
    assert "COST" in output
    assert "$12.34" in output
    assert "73%" in output
    assert "6m48s" in output
    assert "5m28s" in output
    assert "1m19s" in output
