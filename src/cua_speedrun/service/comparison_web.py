"""Exact-season comparison page registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Query, Request
from sqlalchemy import select

from cua_speedrun.service.db import Card, Entry, Run, RunTask, SubmissionRow
from cua_speedrun.service.frontier import track_label
from cua_speedrun.service.insights import build_comparison, result_contract
from cua_speedrun.service.web_records import (
    comparison_facts,
    entry_record,
    task_rows,
)


def register_comparison_web(
    app: FastAPI,
    session_factory: Any,
    templates: Any,
    base_ctx: Callable[[Request, str], dict],
) -> None:
    @app.get("/compare", include_in_schema=False)
    def compare_page(
        request: Request,
        entry: list[int] = Query(default=[]),
    ):
        selected_ids = list(dict.fromkeys(entry))

        def comparison_error(message: str):
            return templates.TemplateResponse(
                request,
                "compare.html",
                {
                    **base_ctx(request, "frontier"),
                    "comparison": None,
                    "comparison_error": message,
                    "selected_count": len(selected_ids),
                },
                status_code=400,
            )

        if not 2 <= len(selected_ids) <= 4:
            return comparison_error(
                "Select between two and four published entries from one exact season."
            )

        with session_factory() as session:
            selected_rows = session.execute(
                select(Entry, Card, Run, SubmissionRow)
                .join(Card, Entry.card_id == Card.id)
                .join(Run, Card.run_id == Run.id)
                .join(SubmissionRow, Run.submission_id == SubmissionRow.id)
                .where(Entry.id.in_(selected_ids))
            ).all()
            row_by_id = {row[0].id: row for row in selected_rows}
            if len(row_by_id) != len(selected_ids):
                return comparison_error("One or more selected entries no longer exist.")

            ordered_rows = [row_by_id[entry_id] for entry_id in selected_ids]
            records = [
                entry_record(selected_entry, submission)
                for selected_entry, _card, _run, submission in ordered_rows
            ]
            season_keys = {record["season_key"] for record in records}
            track_names = {record["track_name"] for record in records}
            if len(season_keys) != 1 or len(track_names) != 1:
                return comparison_error(
                    "These entries cross a track or season boundary and cannot be compared."
                )

            season_key = records[0]["season_key"]
            track_name = records[0]["track_name"]
            peers, success_bar = comparison_facts(
                session, season_key, track_name
            )
            tasks_by_entry = {}
            for selected_entry, card, _run, _submission in ordered_rows:
                tasks = session.execute(
                    select(RunTask)
                    .where(RunTask.run_id == card.run_id)
                    .order_by(RunTask.task_key)
                ).scalars().all()
                tasks_by_entry[selected_entry.id] = task_rows(tasks)

        try:
            comparison = build_comparison(
                records,
                tasks_by_entry,
                peers,
                success_bar=success_bar,
            )
        except ValueError as exc:
            return comparison_error(str(exc))
        comparison["track_label"] = track_label(track_name)
        comparison["contract"] = result_contract(
            records[0], fallback_track=track_name
        )
        return templates.TemplateResponse(request, "compare.html", {
            **base_ctx(request, "frontier"),
            "comparison": comparison,
            "comparison_error": None,
            "selected_count": len(selected_ids),
        })
