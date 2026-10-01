"""The run log: the complete record of one run, as JSONL events.

The run log is the system's ground truth. Scoring reads run logs and
nothing else, so scoring rules can change without rerunning anything.
Every timed event carries monotonic-clock stamps from the gateway, and
durations are only ever computed within that one clock.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class RunLogWriter:
    """Append-only JSONL writer, safe to call from gateway handler threads."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8")

    def event(self, kind: str, **fields: Any) -> None:
        record = {"event": kind, "wall_ts": time.time(), **fields}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def load_runlog(path: Path) -> list[dict[str, Any]]:
    events = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce one run's events to the fields scoring needs."""
    by_kind: dict[str, list[dict]] = {}
    for ev in events:
        by_kind.setdefault(ev["event"], []).append(ev)

    header = by_kind.get("header", [{}])[0]
    armed = by_kind.get("armed", [{}])[0]
    finished = by_kind.get("finished", [{}])[0]
    verdict = by_kind.get("verdict", [{}])[0]
    steps = by_kind.get("step", [])
    observes = by_kind.get("observe", [])
    episode_transitions = [
        event
        for event in by_kind.get("done", [])
        if event.get("task_complete") is False
    ]

    t0 = armed.get("t_mono")
    t_end = finished.get("t_mono")
    task_time = (t_end - t0) if (t0 is not None and t_end is not None) else None

    task_score = verdict.get("score")
    # Seeded checkers used 0/1 before task scores were standardized on the
    # verifier ecosystem's 0-100 convention. The header makes that legacy
    # representation unambiguous without changing non-seeded partial scores.
    if header.get("seeded") and task_score in (0, 1, 0.0, 1.0):
        task_score = float(task_score) * 100.0

    env_time = sum(
        ev.get("dur", 0.0) for ev in steps + observes + episode_transitions
    )
    agent_time = (task_time - env_time) if task_time is not None else None

    return {
        "run_id": header.get("run_id"),
        "task_id": header.get("task_id"),
        "seed": header.get("seed"),
        "passed": bool(verdict.get("passed", False)),
        "score": task_score,
        "reason": finished.get("reason"),
        "task_time_sec": task_time,
        "env_time_sec": env_time,
        "agent_time_sec": agent_time,
        "num_steps": len(steps),
        "num_observes": len(observes),
        "verdict_detail": verdict.get("detail"),
    }
