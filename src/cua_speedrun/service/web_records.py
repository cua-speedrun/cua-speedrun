"""Small adapters from stored database rows to web presentation records."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from cua_speedrun.service.db import (
    Card,
    Entry,
    Run,
    SubmissionRow,
    Track,
)
from cua_speedrun.service.frontier import parse_season_key


def result_track(data: dict, season_key: str, fallback: str) -> str:
    run_plan = data.get("run_plan") or {}
    return (
        (run_plan.get("track") or {}).get("name")
        or parse_season_key(season_key).get("track")
        or fallback
    )


def entry_record(entry: Entry, submission: SubmissionRow) -> dict:
    data = dict(entry.data or {})
    run_plan = data.get("run_plan") or {}
    return {
        **data,
        "entry_id": entry.id,
        "entry_name": entry.entry_name,
        "season_key": entry.season_key,
        "track_name": result_track(data, entry.season_key, submission.track),
        "reference_only": bool(
            data.get("reference_only")
            or ((run_plan.get("track") or {}).get("reference_only"))
        ),
        "published_at": entry.published_at,
    }


def task_rows(tasks: Any) -> list[dict]:
    rows = []
    for task in tasks:
        task_id, _, seed = task.task_key.partition("/seed_")
        rows.append({
            "task_key": task.task_key,
            "task_id": task_id,
            "seed": seed,
            "stage": task.stage,
            "passed": task.passed,
            "reason": task.reason,
            "task_time_sec": task.task_time_sec,
            "agent_time_sec": task.agent_time_sec,
            "env_time_sec": task.env_time_sec,
            "num_steps": task.num_steps,
            "cost_usd": task.cost_usd,
            "usage": task.usage,
        })
    return rows


def comparison_facts(
    session: Any, season_key: str, track_name: str
) -> tuple[list[dict], float]:
    """Published peers and frozen ranking facts for one exact boundary."""
    published = session.execute(
        select(Entry, Card, Run, SubmissionRow)
        .join(Card, Entry.card_id == Card.id)
        .join(Run, Card.run_id == Run.id)
        .join(SubmissionRow, Run.submission_id == SubmissionRow.id)
        .where(Entry.season_key == season_key)
    ).all()
    peers = [
        entry_record(entry, submission)
        for entry, _card, _run, submission in published
    ]
    peers = [peer for peer in peers if peer["track_name"] == track_name]
    track = session.execute(
        select(Track).where(Track.name == track_name)
    ).scalar_one_or_none()

    success_bar = None
    for peer in peers:
        success_bar = (peer.get("rules") or {}).get("success_bar")
        if success_bar is None:
            success_bar = (
                ((peer.get("run_plan") or {}).get("scoring") or {}).get(
                    "success_bar"
                )
            )
        if success_bar is not None:
            break
    if success_bar is None:
        success_bar = (
            track.success_bar
            if track is not None and track.success_bar is not None
            else 0.9
        )
    return peers, float(success_bar)
