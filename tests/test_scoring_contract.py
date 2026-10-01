from __future__ import annotations

import json
from pathlib import Path

from cua_speedrun.scoring import score


def _write_log(
    root: Path,
    task_id: str,
    *,
    passed: bool,
    duration: float,
    task_score: float | None = None,
) -> None:
    task_dir = root / "tasks" / task_id / "seed_1"
    task_dir.mkdir(parents=True)
    events = [
        {
            "event": "header",
            "task_id": task_id,
            "seed": 1,
            "timeout_sec": 10,
        },
        {"event": "armed", "t_mono": 100.0},
        {"event": "finished", "t_mono": 100.0 + duration, "reason": "done"},
        {"event": "verdict", "passed": passed, "score": task_score},
    ]
    (task_dir / "runlog.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )


def test_default_card_scoring_uses_rules_frozen_in_run_plan(tmp_path: Path) -> None:
    _write_log(tmp_path, "passed", passed=True, duration=3.0)
    _write_log(tmp_path, "failed", passed=False, duration=2.0)
    (tmp_path / "result.json").write_text(json.dumps({
        "run_plan": {
            "scoring": {
                "success_bar": 0.5,
                "failure_costs_timeout": True,
            }
        }
    }))

    result = score(tmp_path)

    assert result["success_rate"] == 0.5
    assert result["mean_score"] == 0.5
    assert result["meets_success_bar"] is True
    assert result["total_time_sec"] == 13.0
    assert result["measured_time_sec"] == 5.0


def test_mean_score_preserves_partial_credit_without_changing_success(tmp_path: Path) -> None:
    _write_log(
        tmp_path,
        "passed",
        passed=True,
        duration=3.0,
        task_score=100.0,
    )
    _write_log(
        tmp_path,
        "partial",
        passed=False,
        duration=2.0,
        task_score=90.0,
    )

    result = score(tmp_path)

    assert result["mean_score"] == 0.95
    assert result["success_rate"] == 0.5
    assert result["num_passed"] == 1
