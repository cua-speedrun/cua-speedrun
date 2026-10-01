"""HTML pages: the human face of the same data the JSON API serves.

Server-rendered Jinja2, one CSS file, one small JS file, no build step.
Every page renders from the database, which is itself derived from run
logs, so what the UI shows is always recomputable ground truth.
"""

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

from cua_speedrun.eval_algorithms import (
    list_eval_algorithms,
    resolve_eval_algorithm,
)
from cua_speedrun.execution_placements import list_execution_topologies
from cua_speedrun.service.db import (
    BenchmarkRow,
    Card,
    Entry,
    EventRow,
    Run,
    RunEnvironmentVariable,
    RunTask,
    SavedEnvironmentVariable,
    Season,
    SubmissionRow,
    Track,
    User,
)
from cua_speedrun.service.comparison_web import register_comparison_web
from cua_speedrun.service.frontier import build_frontier_context
from cua_speedrun.service.insights import result_contract, result_insight
from cua_speedrun.service.task_cost import total_task_costs
from cua_speedrun.parallelism import ExecutionScale, max_parallel_evaluations
from cua_speedrun.runplan import RANKING_TIME_AT_SUCCESS_BAR, RunPlan
from cua_speedrun.service.web_records import (
    comparison_facts,
    entry_record,
    result_track,
    task_rows,
)

_HERE = Path(__file__).parent
PROVISIONAL_RUNS_THRESHOLD = 3

STAGE_ORDER = ["queued", "starting", "initializing", "snapshot_done",
               "running", "card_ready"]
TERMINAL_STAGES = {"card_ready", "failed", "rejected", "held", "cancelled"}


def register_web(app: FastAPI, session_factory, session_user_id) -> None:
    templates = Jinja2Templates(directory=str(_HERE / "templates"))
    templates.env.globals["static_asset_versions"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        for path in (_HERE / "static").iterdir()
        if path.suffix in {".css", ".js"}
    }
    app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

    def base_ctx(request: Request, active: str) -> dict:
        user = None
        user_id = session_user_id(request)
        if user_id is not None:
            with session_factory() as session:
                user = session.get(User, user_id)
        # Real identity first: the sign-in button points at GitHub whenever
        # OAuth is configured; the dev login is only the fallback for a
        # machine with no OAuth app (and stays reachable at /login/dev).
        login_url = "/login/github" if os.environ.get("CS_GITHUB_CLIENT_ID") \
            else "/login/dev" if os.environ.get("CS_DEV_LOGIN") == "1" \
            else "/login/github"
        return {"request": request, "active": active, "user": user,
                "login_url": login_url}

    register_comparison_web(app, session_factory, templates, base_ctx)

    @app.get("/", include_in_schema=False)
    def leaderboard_page(
        request: Request,
        track: str | None = None,
        season: str | None = None,
    ):
        with session_factory() as session:
            published_rows = session.execute(
                select(Entry, Card, Run, SubmissionRow)
                .join(Card, Entry.card_id == Card.id)
                .join(Run, Card.run_id == Run.id)
                .join(SubmissionRow, Run.submission_id == SubmissionRow.id)
            ).all()
            track_rows = session.execute(select(Track).order_by(Track.id)).scalars().all()
            stored_seasons = session.execute(select(Season)).scalars().all()

        entry_records = [
            entry_record(entry, submission)
            for entry, _card, _run, submission in published_rows
        ]

        track_specs = [{
            "name": row.name,
            "gpu": row.gpu,
            "eval_algorithm": row.eval_algorithm,
            "agents_per_evaluation": row.agents_per_evaluation,
            "runs_per_task": row.runs_per_task,
            "ranking_rule": row.ranking_rule,
            "success_bar": row.success_bar,
            "reference_only": row.reference_only,
        } for row in track_rows]
        season_specs = {
            row.key: {
                "status": row.status,
                "contract_hash": row.contract_hash,
                "spec": row.spec or {},
            }
            for row in stored_seasons
        }
        frontier = build_frontier_context(
            entry_records,
            track_specs,
            season_specs,
            selected_track=track,
            selected_season=season,
            provisional_threshold=PROVISIONAL_RUNS_THRESHOLD,
        )
        return templates.TemplateResponse(request, "leaderboard.html", {
            **base_ctx(request, "frontier"),
            **frontier,
        })

    @app.get("/entries/{entry_id}", include_in_schema=False)
    def entry_page(request: Request, entry_id: int):
        with session_factory() as session:
            entry = session.get(Entry, entry_id)
            if entry is None:
                raise HTTPException(404)
            card = session.get(Card, entry.card_id)
            run = session.get(Run, card.run_id)
            submission = session.get(SubmissionRow, run.submission_id)
            tasks = session.execute(
                select(RunTask).where(RunTask.run_id == card.run_id)
                .order_by(RunTask.task_key)
            ).scalars().all()
            result = entry_record(entry, submission)
            peers, success_bar = comparison_facts(
                session, entry.season_key, result["track_name"]
            )
        rows = task_rows(tasks)
        insight = result_insight(
            result,
            peers,
            success_bar=success_bar,
            published_entry_id=entry.id,
        )
        alternatives = [
            peer for peer in peers
            if peer["entry_id"] != entry.id and peer.get("total_time_sec") is not None
        ]
        nearest_peer = min(
            alternatives,
            key=lambda peer: abs(
                float(peer["total_time_sec"]) - float(result["total_time_sec"])
            ),
            default=None,
        )
        return templates.TemplateResponse(request, "entry.html", {
            **base_ctx(request, "frontier"),
            "entry": entry,
            "result": result,
            "rows": rows,
            "insight": insight,
            "contract": result_contract(
                result, fallback_track=result["track_name"]
            ),
            "nearest_peer": nearest_peer,
        })

    @app.get("/tracks", include_in_schema=False)
    @app.get("/tracks/{track_name}", include_in_schema=False)
    def tracks_page(request: Request, track_name: str | None = None):
        with session_factory() as session:
            rows = session.execute(
                select(Track).order_by(Track.name)
            ).scalars().all()

        if track_name is not None and track_name not in {row.name for row in rows}:
            raise HTTPException(404, "unknown track")
        from cua_speedrun.service.catalog import default_track_name

        default_name = default_track_name()
        selected_name = track_name or (
            default_name
            if default_name in {row.name for row in rows}
            else rows[0].name if rows else None
        )
        topologies = list_execution_topologies()

        def placement_label(placement) -> str:
            if placement.mode == "local":
                return "Local"
            return str(placement.provider or "Remote").title()

        track_views = []
        for row in rows:
            algorithm = resolve_eval_algorithm(row.eval_algorithm)
            agents = max(1, int(row.agents_per_evaluation or 1))
            module = inspect.getmodule(algorithm.schedule)
            try:
                source = inspect.getsource(module) if module is not None else None
            except (OSError, TypeError):
                source = None
            supported_topologies = [
                {
                    "key": topology.key,
                    "label": (
                        f"{placement_label(topology.compute)} model/agent · "
                        f"{placement_label(topology.environment)} VMs"
                    ),
                }
                for topology in topologies
                if algorithm.supports(topology.runtime_capabilities)
            ]
            track_views.append({
                "name": row.name,
                "active": row.name == selected_name,
                "gpu": row.gpu,
                "agents_per_evaluation": agents,
                "environment_pool_size": (
                    agents * algorithm.default_env_pool_factor
                ),
                "runs_per_task": max(1, int(row.runs_per_task or 1)),
                "ranking_rule": RANKING_TIME_AT_SUCCESS_BAR,
                "success_percent": float(
                    row.success_bar if row.success_bar is not None else 0.9
                ) * 100,
                "failure_charge": (
                    "Full task timeout"
                    if row.failure_costs_timeout
                    else "Measured task time"
                ),
                "seed_policy": row.seed_policy,
                "reference_only": bool(row.reference_only),
                "server_config": dict(row.server_config or {}),
                "algorithm": {
                    "key": algorithm.key,
                    "label": algorithm.label,
                    "aliases": list(algorithm.aliases),
                    "agent_mode": algorithm.agent_mode,
                    "shared_compute": algorithm.shared_agent_sandbox,
                    "environment_pool_factor": algorithm.default_env_pool_factor,
                    "supports_parallel_evaluations": (
                        algorithm.supports_parallel_evaluations
                    ),
                    "required_capabilities": sorted(
                        algorithm.required_runtime_capabilities
                    ),
                    "source_file": (
                        "src/"
                        + algorithm.schedule.__module__.replace(".", "/")
                        + ".py"
                    ),
                    "source": source,
                    "summary": (
                        "One reusable compute replica serves bounded batches "
                        "of single-use environments."
                        if algorithm.shared_agent_sandbox
                        else "Every task receives a fresh isolated compute, "
                        "agent, and environment instance."
                    ),
                },
                "supported_topologies": supported_topologies,
            })

        selected_track = next(
            (track for track in track_views if track["active"]), None
        )
        return templates.TemplateResponse(request, "tracks.html", {
            **base_ctx(request, "tracks"),
            "tracks": track_views,
            "track": selected_track,
        })

    @app.get("/entries/{entry_id}/logs.zip", include_in_schema=False)
    def entry_logs(entry_id: int):
        """Raw run logs for a published entry: the auditability promise.
        Everything on the leaderboard is recomputable from this download."""
        import io
        import zipfile

        from fastapi.responses import Response

        with session_factory() as session:
            entry = session.get(Entry, entry_id)
            if entry is None:
                raise HTTPException(404)
            card = session.get(Card, entry.card_id)
            run = session.get(Run, card.run_id)
        run_dir = Path(run.run_dir or "")
        if not run_dir.is_dir():
            raise HTTPException(410, "run artifacts are no longer available")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(run_dir.rglob("*")):
                if path.is_file() and path.suffix in (
                        ".json", ".jsonl", ".log", ".stdout", ".stderr", ".txt"):
                    zf.write(path, path.relative_to(run_dir))
        return Response(
            buf.getvalue(), media_type="application/zip",
            headers={"Content-Disposition":
                     f'attachment; filename="entry_{entry_id}_logs.zip"'})

    @app.get("/runs/{run_id}", include_in_schema=False)
    def run_page(request: Request, run_id: int):
        user_id = session_user_id(request)
        if user_id is None:
            raise HTTPException(401, "sign in first")
        with session_factory() as session:
            run = session.get(Run, run_id)
            if run is None:
                raise HTTPException(404)
            submission = session.get(SubmissionRow, run.submission_id)
            if submission.user_id != user_id:
                raise HTTPException(403, "not your run")
            tasks = session.execute(
                select(RunTask).where(RunTask.run_id == run_id)
                .order_by(RunTask.task_key)
            ).scalars().all()
            card = session.execute(
                select(Card).where(Card.run_id == run_id)
            ).scalar_one_or_none()
            events = session.execute(
                select(EventRow).where(EventRow.run_id == run_id)
                .order_by(EventRow.id)
            ).scalars().all()
            runtime_environment = session.scalars(
                select(RunEnvironmentVariable)
                .where(RunEnvironmentVariable.run_id == run_id)
                .order_by(RunEnvironmentVariable.name)
            ).all()

        stage_order = list(STAGE_ORDER)
        if run.stage in ("failed", "rejected", "held", "cancelled"):
            stage_order[-1] = run.stage
        now_index = stage_order.index(run.stage) if run.stage in stage_order else 0

        # Per-task logs and sandbox handles, once the run directory exists.
        artifacts = {
            "execution_logs": [],
            "logs": [],
            "sandboxes": [],
            "has_trajectory": False,
        }
        if run.run_dir:
            import json as _json

            run_root = Path(run.run_dir)
            for name in ("init.log", "model-server.log"):
                candidates = [run_root / name]
                candidates.extend(sorted(
                    run_root.glob(f"compute/evaluation_*/{name}")
                ))
                for path in candidates:
                    if not path.is_file():
                        continue
                    relative = path.relative_to(run_root).as_posix()
                    replica = (
                        path.parent.name.replace("evaluation_", "Evaluation ")
                        if path.parent.name.startswith("evaluation_")
                        else "Evaluation"
                    )
                    artifacts["execution_logs"].append({
                        "name": name,
                        "context": replica,
                        "size_kb": max(1, round(path.stat().st_size / 1024)),
                        "url": f"/runs/{run.id}/execution-log/{relative}",
                    })

            log_names = ("env_plane.log", "vllm.log", "agent.stdout",
                         "agent.stderr", "warmup.log", "runlog.jsonl")
            for runlog in sorted(run_root.glob("tasks/*/seed_*/runlog.jsonl")):
                td = runlog.parent
                task_id, seed = td.parent.name, td.name.replace("seed_", "")
                for name in log_names:
                    if (td / name).is_file():
                        artifacts["logs"].append({
                            "task_id": task_id, "seed": seed, "name": name,
                            "url": f"/runs/{run.id}/artifact/{task_id}/{seed}/{name}"})
                if any(td.glob("frame_*.png")):
                    artifacts["has_trajectory"] = True
                meta = td / "meta.json"
                if meta.is_file():
                    try:
                        m = _json.loads(meta.read_text())
                    except Exception:
                        m = {}
                    if m.get("env_sandbox_id") or m.get("agent_sandbox_id"):
                        artifacts["sandboxes"].append({
                            "task": f"{task_id}/seed_{seed}",
                            "env": m.get("env_sandbox_id"),
                            "agent": m.get("agent_sandbox_id")})

        normalized_tasks = task_rows(tasks)
        card_data = dict(card.data or {}) if card else {}
        execution_plan = dict(
            run.execution_plan
            or card_data.get("run_plan")
            or {}
        )
        execution = dict(execution_plan.get("execution") or {})
        topology = dict(execution.get("topology") or {})
        if not topology:
            try:
                from cua_speedrun.execution_placements import (
                    execution_topology_for_backend,
                )

                topology = execution_topology_for_backend(
                    execution_plan.get("backend", "modal-remote")
                ).to_dict()
            except ValueError:
                topology = {
                    "key": "unknown",
                    "compute": {"key": "unknown", "mode": "unknown"},
                    "environment": {"key": "unknown", "mode": "unknown"},
                }
        compute = dict(topology.get("compute") or {})
        environment = dict(topology.get("environment") or {})

        def placement_label(placement: dict) -> str:
            if placement.get("mode") == "local":
                return "Local"
            provider = placement.get("provider")
            return f"Remote · {str(provider).title()}" if provider else "Remote"
        plan_hash = run.plan_hash or card_data.get("run_plan_hash")
        duration_sec = None
        if run.started_at is not None and run.finished_at is not None:
            duration_sec = max(
                0.0, (run.finished_at - run.started_at).total_seconds()
            )
        run_summary = {
            "task_count": len(normalized_tasks),
            "finished_count": sum(
                task["stage"] in ("done", "failed") for task in normalized_tasks
            ),
            "passed_count": sum(task["passed"] is True for task in normalized_tasks),
            "failed_count": sum(task["passed"] is False for task in normalized_tasks),
            "duration_sec": duration_sec,
        }
        run_summary["cost_usd"], run_summary["usage"] = total_task_costs(
            (task.get("cost_usd"), task.get("usage"))
            for task in normalized_tasks
        )
        stored_scale = run.execution_scale or card_data.get("parallelism")
        if stored_scale:
            scale = ExecutionScale.from_dict(stored_scale)
        elif execution_plan:
            scale = ExecutionScale.for_plan(
                RunPlan.from_dict(execution_plan),
                run.parallel_evaluations or 1,
            )
        else:
            # Compatibility for records created before frozen run plans.
            agents = int(
                execution.get("agents_per_evaluation")
                or execution.get("concurrency")
                or 1
            )
            scale = ExecutionScale.for_algorithm(
                execution.get("algorithm") or "per-task-vllm@1",
                run.parallel_evaluations or 1,
                agents,
            )
        parallelism = scale.to_dict()
        contract = result_contract({
            "season_key": run.season_key,
            "run_plan": execution_plan,
            "run_plan_hash": plan_hash,
            "rules": card_data.get("rules") or {},
            "parallelism": card_data.get("parallelism") or parallelism,
        }, fallback_track=submission.track)

        return templates.TemplateResponse(request, "run.html", {
            **base_ctx(request, "me"),
            "run": {
                "run_id": run.id, "stage": run.stage, "error": run.error,
                "season_key": run.season_key,
                "submission": {"name": submission.name, "track": submission.track},
                "tasks": normalized_tasks,
                "card_token": card.token if card else None,
                "result": card_data or None,
                "created_at": run.created_at,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "stop_requested": run.cancel_requested_at is not None,
                "task_seeds": run.task_seeds or {},
                "parallel_evaluations": scale.parallel_evaluations,
            },
            "stage_order": stage_order,
            "now_index": now_index,
            "events": events,
            "artifacts": artifacts,
            "summary": run_summary,
            "contract": contract,
            "execution_topology": {
                **topology,
                "compute_label": placement_label(compute),
                "environment_label": placement_label(environment),
                "label": (
                    f"Model/agent {placement_label(compute)} · "
                    f"VMs {placement_label(environment)}"
                ),
            },
            "runtime_environment": [
                {"name": variable.name, "source": variable.source}
                for variable in runtime_environment
            ],
        })

    @app.get("/cards/{token}", include_in_schema=False)
    def card_page(request: Request, token: str):
        user_id = session_user_id(request)
        with session_factory() as session:
            card = session.execute(
                select(Card).where(Card.token == token)
            ).scalar_one_or_none()
            if card is None:
                raise HTTPException(404)
            run = session.get(Run, card.run_id)
            submission = session.get(SubmissionRow, run.submission_id)
            tasks = session.execute(
                select(RunTask).where(RunTask.run_id == run.id)
                .order_by(RunTask.task_key)
            ).scalars().all()
            published_entry = session.execute(
                select(Entry).where(Entry.card_id == card.id)
            ).scalar_one_or_none()
            data = dict(card.data or {})
            season_key = str(data.get("season_key") or run.season_key or "")
            track_name = result_track(data, season_key, submission.track)
            result = {
                **data,
                "entry_name": (
                    published_entry.entry_name if published_entry else submission.name
                ),
                "season_key": season_key,
                "track_name": track_name,
                "reference_only": bool(
                    data.get("reference_only")
                    or ((data.get("run_plan") or {}).get("track") or {}).get(
                        "reference_only"
                    )
                ),
            }
            peers, success_bar = comparison_facts(
                session, season_key, track_name
            )
        rows = task_rows(tasks)
        insight = result_insight(
            result,
            peers,
            success_bar=success_bar,
            published_entry_id=published_entry.id if published_entry else None,
        )
        return templates.TemplateResponse(request, "card.html", {
            **base_ctx(request, "me"),
            "result": result,
            "rows": rows,
            "token": token,
            "accepted": card.accepted_at is not None,
            "owned": user_id == submission.user_id,
            "published_entry": published_entry,
            "insight": insight,
            "contract": result_contract(result, fallback_track=track_name),
        })

    @app.get("/submit", include_in_schema=False)
    def submit_page(
        request: Request,
        track: str | None = None,
        benchmark: str | None = None,
    ):
        from cua_speedrun.execution_placements import list_execution_topologies
        from cua_speedrun.service.catalog import default_track_name
        from cua_speedrun.service.templates_catalog import list_templates
        user_id = session_user_id(request)
        with session_factory() as session:
            tracks = session.execute(select(Track).order_by(Track.name)).scalars().all()
            benchmarks = session.execute(
                select(BenchmarkRow).where(BenchmarkRow.active).order_by(BenchmarkRow.name, BenchmarkRow.version)
            ).scalars().all()
            saved_environment_variable_names = (
                list(session.scalars(
                    select(SavedEnvironmentVariable.name)
                    .where(SavedEnvironmentVariable.user_id == user_id)
                    .order_by(SavedEnvironmentVariable.name)
                ).all())
                if user_id is not None
                else []
            )
        if tracks and track not in {row.name for row in tracks}:
            preferred = default_track_name()
            track = (
                preferred
                if preferred in {row.name for row in tracks}
                else tracks[0].name
            )
        if benchmarks and benchmark not in {row.name for row in benchmarks}:
            benchmark = benchmarks[0].name
        return templates.TemplateResponse(request, "submit.html", {
            **base_ctx(request, "submit"),
            "tracks": tracks, "benchmarks": benchmarks,
            "templates": list_templates(),
            "saved_environment_variable_names": saved_environment_variable_names,
            "selected_track": track,
            "selected_benchmark": benchmark,
            "execution_topologies": [
                topology.to_dict() for topology in list_execution_topologies()
            ],
            "eval_algorithms": {
                algorithm.key: algorithm.public_dict()
                for algorithm in list_eval_algorithms()
            },
            "max_parallel_evaluations": max_parallel_evaluations(),
        })

    @app.get("/me", include_in_schema=False)
    def me_page(request: Request):
        user_id = session_user_id(request)
        runs = []
        modal_token_id = None
        saved_environment_variable_names: list[str] = []
        if user_id is not None:
            with session_factory() as session:
                u = session.get(User, user_id)
                if u is not None and u.modal_token_id and u.modal_token_secret_enc:
                    modal_token_id = u.modal_token_id
                saved_environment_variable_names = list(session.scalars(
                    select(SavedEnvironmentVariable.name)
                    .where(SavedEnvironmentVariable.user_id == user_id)
                    .order_by(SavedEnvironmentVariable.name)
                ).all())
                rows = session.execute(
                    select(Run, SubmissionRow)
                    .join(SubmissionRow, Run.submission_id == SubmissionRow.id)
                    .where(SubmissionRow.user_id == user_id)
                    .order_by(Run.id.desc())
                ).all()
                runs = [{"run_id": r.id, "stage": r.stage, "name": s.name,
                         "track": s.track, "created_at": str(r.created_at)}
                        for r, s in rows]
        return templates.TemplateResponse(request, "me.html", {
            **base_ctx(request, "me"), "runs": runs,
            "modal_token_id": modal_token_id,
            "saved_environment_variable_names": saved_environment_variable_names,
        })

    @app.get("/methodology", include_in_schema=False)
    def methodology_page(request: Request):
        return templates.TemplateResponse(
            request, "methodology.html", base_ctx(request, "methodology"))

    # -- run artifacts: trajectory frames and logs ------------------------

    def _owned_run(run_id: int, user_id: int | None):
        with session_factory() as session:
            run = session.get(Run, run_id)
            if run is None:
                raise HTTPException(404)
            submission = session.get(SubmissionRow, run.submission_id)
            if submission.user_id != user_id:
                raise HTTPException(403, "not your run")
            return run.run_dir

    _ARTIFACT_OK = (".png", ".log", ".jsonl", ".txt", ".stdout", ".stderr", ".json")

    @app.get("/runs/{run_id}/execution-log/{name:path}",
             include_in_schema=False)
    def execution_log(request: Request, run_id: int, name: str):
        """Owner-only initialization and model-server logs for one run."""
        from fastapi.responses import FileResponse

        user_id = session_user_id(request)
        run_dir = _owned_run(run_id, user_id)
        if not run_dir:
            raise HTTPException(404, "no execution logs for this run yet")
        base = Path(run_dir).resolve()
        target = (base / name).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            raise HTTPException(404)
        if target.suffix.lower() not in (".log", ".stdout", ".stderr", ".txt"):
            raise HTTPException(403, "not a viewable execution log")
        return FileResponse(target, media_type="text/plain")

    @app.get("/runs/{run_id}/artifact/{task_id}/{seed}/{name:path}",
             include_in_schema=False)
    def run_artifact(request: Request, run_id: int, task_id: str, seed: str,
                     name: str):
        from fastapi.responses import FileResponse

        user_id = session_user_id(request)
        run_dir = _owned_run(run_id, user_id)
        if not run_dir:
            raise HTTPException(404, "no artifacts for this run yet")
        base = (Path(run_dir) / "tasks" / task_id / f"seed_{seed}").resolve()
        target = (base / name).resolve()
        if not str(target).startswith(str(base)) or not target.is_file():
            raise HTTPException(404)
        if target.suffix.lower() not in _ARTIFACT_OK:
            raise HTTPException(403, "not a viewable artifact")
        media = "image/png" if target.suffix == ".png" else "text/plain"
        return FileResponse(target, media_type=media)

    @app.get("/runs/{run_id}/tasks/{task_id}/{seed}", include_in_schema=False)
    def task_page(request: Request, run_id: int, task_id: str, seed: str):
        """Task-level diagnostics: verdict, time split, preparation phases,
        logs, and the full step-by-step trajectory for ONE task, linked from
        the run page's grid."""
        import json as _json

        from cua_speedrun.runlog import load_runlog, summarize

        user_id = session_user_id(request)
        if user_id is None:
            raise HTTPException(401, "sign in first")
        run_dir = _owned_run(run_id, user_id)
        if not run_dir:
            raise HTTPException(404, "run artifacts are not available")
        task_dir = Path(run_dir) / "tasks" / task_id / f"seed_{seed}"
        # The path components come from the URL: stay inside the run dir.
        if ".." in task_id or ".." in seed or not task_dir.is_dir():
            raise HTTPException(404, "no such task in this run")

        steps, instruction, verdict_detail, summary = [], "", "", None
        runlog = task_dir / "runlog.jsonl"
        if runlog.is_file():
            events = load_runlog(runlog)
            summary = summarize(events)
            for e in events:
                if e["event"] == "header":
                    instruction = e.get("instruction", "")
                elif e["event"] == "observe":
                    steps.append({"kind": "observe", "frame": e.get("frame"),
                                  "dur": e.get("dur")})
                elif e["event"] == "step":
                    steps.append({"kind": "step", "actions": e.get("actions"),
                                  "dur": e.get("dur")})
                elif e["event"] == "done":
                    steps.append({"kind": "done"})
                elif e["event"] == "verdict":
                    verdict_detail = e.get("detail", "")
                    steps.append({"kind": "verdict", "passed": e.get("passed"),
                                  "detail": verdict_detail})

        phases = {}
        meta_path = task_dir / "meta.json"
        if meta_path.is_file():
            try:
                meta = _json.loads(meta_path.read_text())
                phases = meta.get("phases", {}) or {}
                for key in ("env_region", "agent_region", "placement",
                            "interactivity", "warmup_sec"):
                    if meta.get(key) is not None:
                        phases[key] = meta[key]
            except Exception:
                pass

        logs = [{"name": name,
                 "url": f"/runs/{run_id}/artifact/{task_id}/{seed}/{name}"}
                for name in ("agent.stdout", "agent.stderr", "warmup.log",
                             "env_plane.log", "vllm.log", "runlog.jsonl")
                if (task_dir / name).is_file()]

        return templates.TemplateResponse(request, "task.html", {
            **base_ctx(request, "me"), "run_id": run_id, "task_id": task_id,
            "seed": seed, "instruction": instruction, "steps": steps,
            "summary": summary, "verdict_detail": verdict_detail,
            "phases": phases, "logs": logs,
            "has_frames": any(s.get("frame") for s in steps),
        })

    @app.get("/runs/{run_id}/trajectory", include_in_schema=False)
    def trajectory_page(request: Request, run_id: int):
        from cua_speedrun.runlog import load_runlog

        user_id = session_user_id(request)
        if user_id is None:
            raise HTTPException(401, "sign in first")
        run_dir = _owned_run(run_id, user_id)
        if not run_dir:
            raise HTTPException(404, "no trajectory for this run yet")

        tasks = []
        for runlog in sorted(Path(run_dir).glob("tasks/*/seed_*/runlog.jsonl")):
            task_dir = runlog.parent
            task_id = task_dir.parent.name
            seed = task_dir.name.replace("seed_", "")
            steps, instruction = [], ""
            for e in load_runlog(runlog):
                if e["event"] == "header":
                    instruction = e.get("instruction", "")
                elif e["event"] == "observe":
                    steps.append({"kind": "observe", "frame": e.get("frame"),
                                  "dur": e.get("dur")})
                elif e["event"] == "step":
                    steps.append({"kind": "step", "actions": e.get("actions"),
                                  "dur": e.get("dur")})
                elif e["event"] == "done":
                    steps.append({"kind": "done"})
                elif e["event"] == "verdict":
                    steps.append({"kind": "verdict", "passed": e.get("passed"),
                                  "detail": e.get("detail")})
            agent_logs = [
                {"name": name,
                 "url": f"/runs/{run_id}/artifact/{task_id}/{seed}/{name}"}
                for name in ("agent.stdout", "agent.stderr")
                if (task_dir / name).is_file()
            ]
            tasks.append({"task_id": task_id, "seed": seed,
                          "instruction": instruction, "steps": steps,
                          "agent_logs": agent_logs,
                          "has_frames": any(s.get("frame") for s in steps)})
        return templates.TemplateResponse(request, "trajectory.html", {
            **base_ctx(request, "me"), "run_id": run_id, "tasks": tasks,
        })
