"""Scoring: a pure function from run logs to numbers.

Scoring never touches an environment and the executor never scores. Timed
measurements come only from run logs; result.json carries the immutable run
plan that selects the scoring rules. Explicit research re-scoring remains a
pure operation over the same stored measurements.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cua_speedrun.runlog import load_runlog, summarize


@dataclass
class ScoringRules:
    success_bar: float = 0.9          # required fraction of passed runs, over ALL runs
    # A failed run contributes its actual measured time, not the full
    # timeout. The fail-fast exploit is already closed by the success bar
    # over ALL runs (failing fast tanks the rate below the bar), so charging
    # the timeout only distorted totals. Set True to restore the old rule;
    # scoring is a pure function over run logs, so either recomputes freely.
    failure_costs_timeout: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success_bar": self.success_bar,
            "failure_costs_timeout": self.failure_costs_timeout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoringRules":
        return cls(
            success_bar=float(data.get("success_bar", 0.9)),
            failure_costs_timeout=bool(data.get("failure_costs_timeout", False)),
        )


def stored_rules(run_dir: Path) -> ScoringRules | None:
    """Return rules frozen in the run's persisted execution contract."""
    try:
        result = json.loads((Path(run_dir) / "result.json").read_text())
    except (OSError, json.JSONDecodeError):
        try:
            result = json.loads((Path(run_dir) / "run_plan.json").read_text())
        except (OSError, json.JSONDecodeError):
            return None
    plan = result.get("run_plan") or {}
    raw = plan.get("scoring") or result.get("scoring_rules")
    return ScoringRules.from_dict(raw) if isinstance(raw, dict) else None


def collect_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for runlog_path in sorted(Path(run_dir).glob("tasks/*/*/runlog.jsonl")):
        rows.append(summarize(load_runlog(runlog_path)))
    return rows


def score(run_dir: Path, rules: ScoringRules | None = None) -> dict[str, Any]:
    # A run's comparison contract is immutable.  Explicit rules are useful
    # for research re-scoring; the default leaderboard/card path honors the
    # rules captured when the run was created.
    rules = rules or stored_rules(run_dir) or ScoringRules()
    rows = collect_rows(run_dir)
    if not rows:
        raise ValueError(f"no run logs found under {run_dir}")

    passed = [r for r in rows if r["passed"]]
    success_rate = len(passed) / len(rows)

    def normalized_task_score(row: dict[str, Any]) -> float:
        raw = row.get("score")
        if raw is None:
            raw = 100.0 if row["passed"] else 0.0
        value = float(raw)
        if not 0.0 <= value <= 100.0:
            raise ValueError(
                f"task {row.get('task_id')} score outside [0, 100]: {value}"
            )
        return value / 100.0

    mean_score = sum(normalized_task_score(row) for row in rows) / len(rows)

    def charged_time(row: dict[str, Any]) -> float | None:
        if row["passed"] or not rules.failure_costs_timeout:
            return row["task_time_sec"]
        events = load_runlog(
            Path(run_dir) / "tasks" / row["task_id"] / f"seed_{row['seed']}" / "runlog.jsonl"
        )
        header = next(e for e in events if e["event"] == "header")
        return float(header["timeout_sec"])

    charged = [t for t in (charged_time(r) for r in rows) if t is not None]
    times_on_success = [r["task_time_sec"] for r in passed if r["task_time_sec"] is not None]

    return {
        "num_runs": len(rows),
        "num_passed": len(passed),
        "mean_score": mean_score,
        "success_rate": success_rate,
        "meets_success_bar": success_rate >= rules.success_bar,
        # total_time_sec is the charged time used for ranking.  Whether a
        # failure costs measured time or the full timeout is frozen in rules.
        # measured_time_sec is always the actual clocked duration.
        "total_time_sec": sum(charged),
        "measured_time_sec": sum(r["task_time_sec"] or 0.0 for r in rows),
        "median_task_time_sec": statistics.median(times_on_success) if times_on_success else None,
        "env_time_sec": sum(r["env_time_sec"] or 0.0 for r in rows),
        "agent_time_sec": sum(r["agent_time_sec"] or 0.0 for r in rows if r["agent_time_sec"]),
        "rules": rules.to_dict(),
        "rows": rows,
    }


def format_table(result: dict[str, Any]) -> str:
    lines = []
    lines.append(f"{'task':<28} {'seed':>6} {'result':<8} {'time':>8} {'steps':>6}")
    lines.append("-" * 62)
    for r in result["rows"]:
        t = f"{r['task_time_sec']:.2f}s" if r["task_time_sec"] is not None else "-"
        lines.append(
            f"{r['task_id']:<28} {r['seed']:>6} "
            f"{'pass' if r['passed'] else 'fail':<8} {t:>8} {r['num_steps']:>6}"
        )
    lines.append("-" * 62)
    bar = "meets" if result["meets_success_bar"] else "BELOW"
    lines.append(
        f"success {result['num_passed']}/{result['num_runs']} "
        f"({result['success_rate']:.0%}, {bar} the bar)   "
        f"score {result['mean_score']:.1%}   "
        f"total {result['total_time_sec']:.2f}s   "
        f"median {result['median_task_time_sec']:.2f}s"
        if result["median_task_time_sec"] is not None
        else (
            f"success {result['num_passed']}/{result['num_runs']}   "
            f"score {result['mean_score']:.1%}"
        )
    )
    return "\n".join(lines)
