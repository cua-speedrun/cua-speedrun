"""Read cumulative task-cost snapshots emitted by model templates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


COST_SNAPSHOT_PREFIX = "__CUA_SPEEDRUN_COST_V1__"


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or value < 0:
        return None
    return value


def parse_cost_snapshot(line: str) -> dict[str, Any] | None:
    """Parse one valid cumulative snapshot, ignoring ordinary agent output."""
    if not line.startswith(COST_SNAPSHOT_PREFIX):
        return None
    try:
        payload = json.loads(line[len(COST_SNAPSHOT_PREFIX):])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    cost_usd = _number(payload.get("cost_usd"))
    if cost_usd is None:
        return None
    raw_usage = payload.get("usage")
    usage = {}
    if isinstance(raw_usage, dict):
        usage = {
            str(key): number
            for key, value in raw_usage.items()
            if (number := _number(value)) is not None
        }
    return {"cost_usd": float(cost_usd), "usage": usage}


def latest_cost_snapshot(output: str) -> dict[str, Any] | None:
    """Return the last valid cumulative snapshot in captured agent stdout."""
    latest = None
    for line in output.splitlines():
        if (snapshot := parse_cost_snapshot(line)) is not None:
            latest = snapshot
    return latest


def read_task_cost_snapshot(
    run_dir: str | Path,
    task_key: str,
) -> dict[str, Any] | None:
    """Read a task's latest snapshot without allowing path traversal."""
    task_id, separator, seed_text = task_key.rpartition("/seed_")
    if not separator:
        return None
    try:
        seed = int(seed_text)
    except ValueError:
        return None
    tasks_root = (Path(run_dir) / "tasks").resolve()
    stdout = (tasks_root / task_id / f"seed_{seed}" / "agent.stdout").resolve()
    try:
        stdout.relative_to(tasks_root)
        return latest_cost_snapshot(stdout.read_text(errors="replace"))
    except (OSError, ValueError):
        return None


def total_task_costs(
    snapshots: Iterable[tuple[float | None, Any]],
) -> tuple[float | None, dict[str, int | float]]:
    """Total one final cost/usage snapshot per task."""
    total = 0.0
    reported = False
    usage: dict[str, int | float] = {}
    for cost_usd, task_usage in snapshots:
        cost = _number(cost_usd)
        if cost is None:
            continue
        reported = True
        total += float(cost)
        if isinstance(task_usage, dict):
            for key, value in task_usage.items():
                number = _number(value)
                if number is not None:
                    usage[str(key)] = usage.get(str(key), 0) + number
    return (total if reported else None), usage
