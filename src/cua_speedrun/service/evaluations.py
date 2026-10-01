"""Transport-independent evaluation queue, status, and cancellation services."""

from __future__ import annotations

import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping

from sqlalchemy import func, select

from cua_speedrun.parallelism import ExecutionScale
from cua_speedrun.runlog import load_runlog, summarize
from cua_speedrun.runtime_environment import (
    MAX_ENVIRONMENT_VARIABLES,
    normalize_environment_name,
    normalize_environment_value,
)
from cua_speedrun.service.db import (
    BenchmarkRow,
    Card,
    EventRow,
    Run,
    RunEnvironmentVariable,
    RunTask,
    SavedEnvironmentVariable,
    SubmissionRow,
    Track,
    User,
    utcnow,
)
from cua_speedrun.service.task_cost import (
    read_task_cost_snapshot,
    total_task_costs,
)
from cua_speedrun.service.plans import attach_plan, attach_scale, plan_for_catalog_rows


TERMINAL_STAGES = frozenset({
    "card_ready", "failed", "rejected", "held", "cancelled",
})


class EvaluationServiceError(ValueError):
    """A user-visible evaluation request error with an HTTP-compatible status."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def validate_submission_zip(data: bytes) -> None:
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
    except zipfile.BadZipFile as exc:
        raise EvaluationServiceError(400, "not a zip file") from exc
    if sorted(names) != ["agent.py", "init.py"]:
        raise EvaluationServiceError(
            400,
            "a submission is a zip containing exactly init.py and agent.py "
            f"at the root; got {sorted(names)}",
        )


def ensure_local_user(session_factory, handle: str = "dev") -> int:
    """Return the single-machine operator identity, creating it when needed."""
    with session_factory() as session:
        user = session.execute(
            select(User).where(User.handle == handle)
        ).scalar_one_or_none()
        if user is None:
            user = User(handle=handle, quota_tier="dev")
            session.add(user)
            session.commit()
        return int(user.id)


def benchmark_id_for_name(session_factory, requested: str) -> int:
    name, separator, version = requested.rpartition("@")
    if not separator:
        name, version = requested, ""
    with session_factory() as session:
        rows = session.scalars(
            select(BenchmarkRow).where(BenchmarkRow.name == name, BenchmarkRow.active)
        ).all()
    matches = [row for row in rows if not version or str(row.version) == version]
    if not matches:
        raise EvaluationServiceError(404, f"unknown benchmark {requested!r}")
    if len(matches) > 1:
        versions = ", ".join(sorted(str(row.version) for row in matches))
        raise EvaluationServiceError(
            400,
            f"benchmark {requested!r} has multiple versions ({versions}); "
            "use name@version",
        )
    return int(matches[0].id)


def queue_evaluation(
    *,
    session_factory,
    store,
    user_id: int,
    submission_zip: bytes,
    name: str,
    track_name: str,
    benchmark_id: int,
    compute_placement: str,
    environment_placement: str,
    allocate_gpu: bool = True,
    gpu: str | None = None,
    eval_algorithm: str | None = None,
    parallel_evaluations: int = 1,
    saved_environment_names: Iterable[str] = (),
    evaluation_environment: Mapping[str, str] | None = None,
    required_environment_names: Iterable[str] = (),
    runner: str | None = None,
    runner_template: str | None = None,
) -> dict[str, Any]:
    """Validate and atomically queue one scored evaluation."""
    name = str(name).strip()
    track_name = str(track_name).strip()
    if not name or len(name) > 120:
        raise EvaluationServiceError(
            400, "evaluation name must contain 1 to 120 characters"
        )
    if not track_name:
        raise EvaluationServiceError(400, "choose a track before providing an agent")
    if int(benchmark_id) <= 0:
        raise EvaluationServiceError(400, "choose a benchmark before providing an agent")
    if len(submission_zip) > 1 << 20:
        raise EvaluationServiceError(400, "submission zip larger than 1 MiB")
    validate_submission_zip(submission_zip)

    # Freeze the operator's runner choice on the run itself. Whichever
    # executor claims the run later must honor the selection made here;
    # validating now surfaces an unknown template at submit time.
    runner_selection: dict[str, str] = {}
    if runner or runner_template:
        if compute_placement != "local":
            raise EvaluationServiceError(
                400, "a compute runner only applies to local compute placement"
            )
        from cua_speedrun.compute_runners import resolve_runner_selection

        try:
            resolve_runner_selection(runner, runner_template)
        except (ValueError, FileNotFoundError, OSError) as exc:
            raise EvaluationServiceError(400, str(exc)) from exc
        runner_selection = {"kind": str(runner or "local")}
        if runner_template:
            runner_selection["template"] = str(runner_template)

    try:
        saved_names = sorted({
            normalize_environment_name(item) for item in saved_environment_names
        })
        evaluation_values = {
            normalize_environment_name(key): normalize_environment_value(value)
            for key, value in (evaluation_environment or {}).items()
        }
        required_names = {
            normalize_environment_name(item) for item in required_environment_names
        }
    except ValueError as exc:
        raise EvaluationServiceError(400, str(exc)) from exc
    all_names = set(saved_names) | set(evaluation_values)
    if len(all_names) > MAX_ENVIRONMENT_VARIABLES:
        raise EvaluationServiceError(
            400, f"at most {MAX_ENVIRONMENT_VARIABLES} environment variables may be provided"
        )
    missing_required = sorted(required_names - all_names)
    if missing_required:
        raise EvaluationServiceError(
            400,
            "the selected template requires these environment variables: "
            + ", ".join(missing_required),
        )

    with session_factory() as session:
        user = session.get(User, user_id)
        if user is None:
            raise EvaluationServiceError(401, "unknown user")
        benchmark = session.get(BenchmarkRow, benchmark_id)
        if benchmark is None or not benchmark.active:
            raise EvaluationServiceError(404, "unknown benchmark")
        from cua_speedrun.evaluator_environment import requirements
        from cua_speedrun.specs import Benchmark

        try:
            evaluator_required, _ = requirements(Benchmark.load(Path(benchmark.path)))
        except (OSError, ValueError) as exc:
            raise EvaluationServiceError(409, f"benchmark cannot be loaded: {exc}") from exc
        missing_evaluator = sorted(
            (evaluator_required - all_names)
            | {
                name for name in evaluator_required & evaluation_values.keys()
                if not evaluation_values[name].strip()
            }
        )
        if missing_evaluator:
            raise EvaluationServiceError(
                400, "benchmark requires evaluator credentials before running: "
                + ", ".join(missing_evaluator),
            )
        track = session.execute(
            select(Track).where(Track.name == track_name)
        ).scalar_one_or_none()
        if track is None:
            raise EvaluationServiceError(404, f"unknown track {track_name!r}")
        try:
            plan = plan_for_catalog_rows(
                track,
                benchmark,
                compute_placement or None,
                environment_placement or None,
                allocate_gpu=allocate_gpu,
                gpu=gpu,
                eval_algorithm=eval_algorithm,
            )
            scale = ExecutionScale.for_plan(
                plan,
                parallel_evaluations,
                enforce_admission_limit=True,
            )
        except (OSError, ValueError) as exc:
            raise EvaluationServiceError(
                409, f"evaluation contract cannot be resolved: {exc}"
            ) from exc

        saved_rows = (
            session.scalars(
                select(SavedEnvironmentVariable).where(
                    SavedEnvironmentVariable.user_id == user_id,
                    SavedEnvironmentVariable.name.in_(saved_names),
                )
            ).all()
            if saved_names
            else []
        )
        saved_by_name = {row.name: row for row in saved_rows}
        unknown = sorted(set(saved_names) - set(saved_by_name))
        if unknown:
            raise EvaluationServiceError(
                400,
                "these saved environment variables are not configured on your "
                "account: " + ", ".join(unknown),
            )

        topology = plan.resolved_execution_topology()
        if topology["requires_user_credentials"] and not (
            user.modal_token_id and user.modal_token_secret_enc
        ):
            raise EvaluationServiceError(
                402,
                "no Modal credentials on your account: runs execute in YOUR "
                "Modal workspace and bill your credits",
            )
        from cua_speedrun.service.quota import QuotaError, check_and_reserve

        try:
            check_and_reserve(session, user, track_name)
        except QuotaError as exc:
            raise EvaluationServiceError(429, str(exc)) from exc

        storage_ref = store.put_bytes(submission_zip, suffix=".zip")
        submission = SubmissionRow(
            user_id=user_id,
            name=name,
            storage_ref=storage_ref,
            track=track_name,
            status="queued",
        )
        session.add(submission)
        session.flush()
        run = Run(
            submission_id=submission.id,
            benchmark_id=benchmark_id,
            stage="queued",
            runner_selection=runner_selection,
        )
        attach_plan(run, plan)
        attach_scale(run, scale)
        session.add(run)
        session.flush()

        environment_snapshot = {
            row.name: (row.value_enc, "saved") for row in saved_rows
        }
        if evaluation_values:
            from cua_speedrun.service.usersecrets import encrypt

            environment_snapshot.update({
                variable_name: (encrypt(value), "evaluation")
                for variable_name, value in evaluation_values.items()
            })
        session.add_all([
            RunEnvironmentVariable(
                run_id=run.id,
                name=variable_name,
                value_enc=value_enc,
                source=source,
            )
            for variable_name, (value_enc, source) in sorted(
                environment_snapshot.items()
            )
        ])
        session.commit()
        return {
            "submission_id": submission.id,
            "run_id": run.id,
            "stage": run.stage,
            "execution_topology": topology,
            "gpu": plan.gpu,
            "parallelism": scale.to_dict(),
        }


def run_payload(session, run: Run, user_id: int | None) -> dict[str, Any]:
    submission = session.get(SubmissionRow, run.submission_id)
    if submission is None or submission.user_id != user_id:
        raise EvaluationServiceError(403, "not your run")
    benchmark = session.get(BenchmarkRow, run.benchmark_id)
    stored_tasks = session.scalars(
        select(RunTask).where(RunTask.run_id == run.id).order_by(RunTask.task_key)
    ).all()
    event_bounds = {
        str(task_key): (float(first_ts), float(last_ts))
        for task_key, first_ts, last_ts in session.execute(
            select(
                EventRow.task_key,
                func.min(EventRow.ts),
                func.max(EventRow.ts),
            )
            .where(EventRow.run_id == run.id, EventRow.task_key.is_not(None))
            .group_by(EventRow.task_key)
        ).all()
    }
    armed_at = {
        str(task_key): float(ts)
        for task_key, ts in session.execute(
            select(EventRow.task_key, func.max(EventRow.ts))
            .where(
                EventRow.run_id == run.id,
                EventRow.task_key.is_not(None),
                EventRow.kind == "armed",
            )
            .group_by(EventRow.task_key)
        ).all()
    }
    task_errors: dict[str, str] = {}
    for task_key, kind, payload in session.execute(
        select(EventRow.task_key, EventRow.kind, EventRow.payload)
        .where(
            EventRow.run_id == run.id,
            EventRow.kind.in_((
                "task_started", "env_ready", "task_failed", "agent_failed",
            )),
        )
        .order_by(EventRow.id)
    ).all():
        if task_key is None:
            continue
        if kind in {"task_started", "env_ready"}:
            task_errors.pop(str(task_key), None)
            continue
        error = dict(payload or {}).get("error")
        if error:
            task_errors[str(task_key)] = str(error)
    card = session.execute(
        select(Card).where(Card.run_id == run.id)
    ).scalar_one_or_none()
    environment_variables = session.scalars(
        select(RunEnvironmentVariable)
        .where(RunEnvironmentVariable.run_id == run.id)
        .order_by(RunEnvironmentVariable.name)
    ).all()
    plan = dict(run.execution_plan or {})
    benchmark_plan = dict(plan.get("benchmark") or {})
    execution_plan = dict(plan.get("execution") or {})
    topology_plan = dict(execution_plan.get("topology") or {})
    topology = str(
        run.topology_key
        or topology_plan.get("key")
        or plan.get("backend")
        or "unknown"
    )
    benchmark_name = (
        benchmark.name if benchmark is not None else benchmark_plan.get("name")
    ) or f"benchmark-{run.benchmark_id}"
    benchmark_version = (
        benchmark.version
        if benchmark is not None
        else benchmark_plan.get("version")
    )
    benchmark_label = str(benchmark_name)
    if benchmark_version is not None:
        benchmark_label += f"@{benchmark_version}"

    task_seeds = run.task_seeds if isinstance(run.task_seeds, dict) else {}
    task_ids = list(benchmark_plan.get("task_ids") or task_seeds)
    expected_keys = [
        f"{task_id}/seed_{int(seed)}"
        for task_id in task_ids
        for seed in task_seeds.get(task_id, ())
    ]
    stored_by_key = {task.task_key: task for task in stored_tasks}
    ordered_keys = list(dict.fromkeys([
        *expected_keys,
        *(task.task_key for task in stored_tasks),
    ]))

    tasks = []
    for task_index, task_key in enumerate(ordered_keys, start=1):
        task = stored_by_key.get(task_key)
        first_ts, last_ts = event_bounds.get(task_key, (None, None))
        tasks.append({
            "task_index": task_index,
            "task_key": task_key,
            "stage": task.stage if task is not None else "queued",
            "passed": task.passed if task is not None else None,
            "reason": task.reason if task is not None else None,
            "error": task_errors.get(task_key),
            "task_time_sec": task.task_time_sec if task is not None else None,
            "env_time_sec": task.env_time_sec if task is not None else None,
            "agent_time_sec": task.agent_time_sec if task is not None else None,
            "num_steps": task.num_steps if task is not None else None,
            "cost_usd": task.cost_usd if task is not None else None,
            "usage": task.usage if task is not None else None,
            # A submission's step budget is not currently part of the frozen
            # runtime contract. Keep the field explicit and unknown instead
            # of drawing a misleading percentage from a template default.
            "step_limit": None,
            "started_at": first_ts,
            "armed_at": armed_at.get(task_key),
            "updated_at": last_ts,
        })

    expected = len(expected_keys)
    if not expected:
        catalog_task_count = benchmark.task_count if benchmark is not None else None
        task_count = catalog_task_count or len(task_ids)
        expected = int(task_count) * max(
            1, int(execution_plan.get("runs_per_task") or 1)
        )
    expected = max(expected, len(tasks))
    finished = sum(
        task["stage"] in {"done", "failed", "cancelled"} for task in tasks
    )
    passed = sum(task["passed"] is True for task in tasks)
    verifier_failed = sum(
        task["stage"] == "done"
        and task["passed"] is False
        and task["reason"] != "agent_error"
        for task in tasks
    )
    agent_failed = sum(
        task["stage"] == "done"
        and task["passed"] is False
        and task["reason"] == "agent_error"
        for task in tasks
    )
    infra_failed = sum(task["stage"] == "failed" for task in tasks)
    cancelled = sum(task["stage"] == "cancelled" for task in tasks)
    cost_usd, usage = total_task_costs(
        (task.get("cost_usd"), task.get("usage")) for task in tasks
    )

    elapsed_sec = None
    if run.started_at is not None:
        end = run.finished_at or utcnow()
        if run.started_at.tzinfo is None and end.tzinfo is not None:
            end = end.replace(tzinfo=None)
        elif run.started_at.tzinfo is not None and end.tzinfo is None:
            end = end.replace(tzinfo=run.started_at.tzinfo)
        elapsed_sec = max(0.0, (end - run.started_at).total_seconds())

    card_data = dict(card.data or {}) if card else {}
    stored_scale = dict(run.execution_scale or {})
    return {
        "run_id": run.id,
        "stage": run.stage,
        "stop_requested": run.cancel_requested_at is not None,
        "error": run.error,
        "run_dir": run.run_dir,
        "season_key": run.season_key,
        "execution_scale": stored_scale or None,
        "parallel_evaluations": stored_scale.get(
            "parallel_evaluations", run.parallel_evaluations or 1
        ),
        "submission": {"name": submission.name, "track": submission.track},
        "benchmark": benchmark_label,
        "topology": topology,
        "created_at": run.created_at.isoformat(),
        "started_at": (
            run.started_at.isoformat() if run.started_at is not None else None
        ),
        "finished_at": (
            run.finished_at.isoformat() if run.finished_at is not None else None
        ),
        "elapsed_sec": elapsed_sec,
        "cost_usd": cost_usd,
        "usage": usage,
        "progress": {
            "finished": finished,
            "total": expected,
            "passed": passed,
            "agent_failed": agent_failed,
            "verifier_failed": verifier_failed,
            "infra_failed": infra_failed,
            "cancelled": cancelled,
        },
        "tasks": tasks,
        "environment_variables": [
            {"name": variable.name, "source": variable.source}
            for variable in environment_variables
        ],
        "card_token": card.token if card else None,
        "result": ({
            "num_runs": card_data.get("num_runs"),
            "num_passed": card_data.get("num_passed"),
            "mean_score": card_data.get("mean_score"),
            "success_rate": card_data.get("success_rate"),
            "meets_success_bar": card_data.get("meets_success_bar"),
            "total_time_sec": card_data.get("total_time_sec"),
            "measured_time_sec": card_data.get("measured_time_sec"),
            "agent_time_sec": card_data.get("agent_time_sec"),
            "env_time_sec": card_data.get("env_time_sec"),
        } if card_data else None),
    }


def get_evaluation(session_factory, run_id: int, user_id: int) -> dict[str, Any]:
    with session_factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise EvaluationServiceError(404, "unknown evaluation")
        return run_payload(session, run, user_id)


def list_evaluations(
    session_factory,
    user_id: int,
    *,
    active_only: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return concise progress and score summaries for one user's runs."""
    if limit is not None and limit < 1:
        raise EvaluationServiceError(400, "evaluation list limit must be positive")

    with session_factory() as session:
        query = (
            select(Run, SubmissionRow, BenchmarkRow, Card)
            .join(SubmissionRow, Run.submission_id == SubmissionRow.id)
            .outerjoin(BenchmarkRow, Run.benchmark_id == BenchmarkRow.id)
            .outerjoin(Card, Card.run_id == Run.id)
            .where(SubmissionRow.user_id == user_id)
            .order_by(Run.id.desc())
        )
        if active_only:
            query = query.where(Run.stage.not_in(TERMINAL_STAGES))
        if limit is not None:
            query = query.limit(limit)
        rows = session.execute(query).all()
        run_ids = [run.id for run, _submission, _benchmark, _card in rows]
        tasks_by_run: dict[int, list[RunTask]] = {run_id: [] for run_id in run_ids}
        if run_ids:
            tasks = session.scalars(
                select(RunTask)
                .where(RunTask.run_id.in_(run_ids))
                .order_by(RunTask.id)
            ).all()
            for task in tasks:
                tasks_by_run[task.run_id].append(task)

        now = utcnow()
        summaries = []
        for run, submission, benchmark, card in rows:
            tasks = tasks_by_run[run.id]

            def mean_timing(field: str) -> float | None:
                measured = [
                    float(value)
                    for task in tasks
                    if (value := getattr(task, field)) is not None
                ]
                return sum(measured) / len(measured) if measured else None

            cost_usd, usage = total_task_costs(
                (task.cost_usd, task.usage) for task in tasks
            )

            plan = dict(run.execution_plan or {})
            benchmark_plan = dict(plan.get("benchmark") or {})
            finished = sum(
                task.stage in {"done", "failed", "cancelled"} for task in tasks
            )
            passed = sum(task.passed is True for task in tasks)
            failed = sum(task.passed is False for task in tasks)
            running = sum(
                task.stage not in {"done", "failed", "cancelled"}
                for task in tasks
            )

            task_seeds = run.task_seeds if isinstance(run.task_seeds, dict) else {}
            expected = sum(
                len(seeds) for seeds in task_seeds.values()
                if isinstance(seeds, (list, tuple))
            )
            if not expected:
                execution = dict(plan.get("execution") or {})
                catalog_task_count = benchmark.task_count if benchmark else None
                task_count = catalog_task_count or len(
                    benchmark_plan.get("task_ids") or ()
                )
                expected = int(task_count) * max(
                    1, int(execution.get("runs_per_task") or 1)
                )
            expected = max(expected, len(tasks))

            topology = run.topology_key
            if not topology:
                topology = str(
                    ((plan.get("execution") or {}).get("topology") or {}).get("key")
                    or plan.get("backend")
                    or "unknown"
                )

            elapsed_sec = None
            if run.started_at is not None:
                end = run.finished_at or now
                if run.started_at.tzinfo is None and end.tzinfo is not None:
                    end = end.replace(tzinfo=None)
                elif run.started_at.tzinfo is not None and end.tzinfo is None:
                    end = end.replace(tzinfo=run.started_at.tzinfo)
                elapsed_sec = max(0.0, (end - run.started_at).total_seconds())

            card_data = dict(card.data or {}) if card else {}
            score = None
            if card_data:
                score = {
                    "num_runs": card_data.get("num_runs"),
                    "num_passed": card_data.get("num_passed"),
                    "mean_score": card_data.get("mean_score"),
                    "success_rate": card_data.get("success_rate"),
                    "meets_success_bar": card_data.get("meets_success_bar"),
                    "total_time_sec": card_data.get("total_time_sec"),
                }
            benchmark_name = (
                benchmark.name if benchmark else benchmark_plan.get("name")
            ) or f"benchmark-{run.benchmark_id}"
            benchmark_version = (
                benchmark.version if benchmark else benchmark_plan.get("version")
            )
            benchmark_label = str(benchmark_name)
            if benchmark_version is not None:
                benchmark_label += f"@{benchmark_version}"
            summaries.append({
                "run_id": run.id,
                "name": submission.name,
                "track": submission.track,
                "benchmark": benchmark_label,
                "topology": topology,
                "stage": run.stage,
                "progress": {
                    "finished": finished,
                    "total": expected,
                    "running": running,
                    "passed": passed,
                    "failed": failed,
                },
                "score": score,
                "cost_usd": cost_usd,
                "usage": usage,
                "per_task": {
                    "time_sec": mean_timing("task_time_sec"),
                    "agent_time_sec": mean_timing("agent_time_sec"),
                    "env_time_sec": mean_timing("env_time_sec"),
                },
                "elapsed_sec": elapsed_sec,
                "created_at": run.created_at.isoformat(),
                "started_at": (
                    run.started_at.isoformat() if run.started_at is not None else None
                ),
                "finished_at": (
                    run.finished_at.isoformat() if run.finished_at is not None else None
                ),
            })
        return summaries


def cancel_evaluation(session_factory, run_id: int, user_id: int) -> dict[str, Any]:
    with session_factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise EvaluationServiceError(404, "unknown evaluation")
        submission = session.get(SubmissionRow, run.submission_id)
        if submission is None or submission.user_id != user_id:
            raise EvaluationServiceError(403, "not your run")
        if run.stage in TERMINAL_STAGES:
            if run.stage == "cancelled":
                return {"run_id": run.id, "stage": run.stage}
            raise EvaluationServiceError(409, f"evaluation already {run.stage}")

        now = utcnow()
        if run.cancel_requested_at is None:
            run.cancel_requested_at = now
            session.add(EventRow(
                run_id=run.id,
                ts=time.time(),
                kind="run_cancel_requested",
                task_key=None,
                payload={},
            ))
        if run.stage == "queued":
            run.stage = "cancelled"
            run.finished_at = now
            session.add(EventRow(
                run_id=run.id,
                ts=time.time(),
                kind="run_cancelled",
                task_key=None,
                payload={"before_start": True},
            ))
        session.commit()
        return {
            "run_id": run.id,
            "stage": run.stage,
            "stop_requested": True,
        }


def _complete_task_evidence(
    run_dir: Path | None,
    task_key: str,
) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    task_id, separator, seed_text = task_key.rpartition("/seed_")
    if not separator:
        return None
    try:
        seed = int(seed_text)
        row = summarize(load_runlog(
            run_dir / "tasks" / task_id / f"seed_{seed}" / "runlog.jsonl"
        ))
    except (OSError, ValueError, TypeError, KeyError):
        return None
    complete = (
        row.get("task_id") == task_id
        and row.get("seed") == seed
        and row.get("reason") is not None
        and row.get("reason") != "infrastructure_error"
        and row.get("task_time_sec") is not None
    )
    if not complete:
        return None
    snapshot = read_task_cost_snapshot(run_dir, task_key)
    if snapshot is not None:
        row.update(snapshot)
    return row


def resume_evaluation(
    session_factory,
    run_id: int,
    user_id: int,
    installation_root: Path | None = None,
    *,
    rerun_agent_failures: bool = False,
) -> dict[str, Any]:
    """Queue selected task instances on the same frozen run record."""
    with session_factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise EvaluationServiceError(404, "unknown evaluation")
        submission = session.get(SubmissionRow, run.submission_id)
        if submission is None or submission.user_id != user_id:
            raise EvaluationServiceError(403, "not your run")
        allowed_stages = (
            {"failed", "cancelled", "card_ready"}
            if rerun_agent_failures
            else {"failed", "cancelled"}
        )
        if run.stage not in allowed_stages:
            raise EvaluationServiceError(
                409,
                "only failed or cancelled evaluations can be resumed"
                + (
                    ", or a completed evaluation with "
                    "--rerun-agent-failures"
                    if rerun_agent_failures
                    else ""
                ),
            )
        if not run.execution_plan:
            raise EvaluationServiceError(
                409,
                "this legacy evaluation has no frozen execution plan and cannot "
                "be resumed safely",
            )
        card = session.execute(
            select(Card).where(Card.run_id == run.id)
        ).scalar_one_or_none()
        if card is not None:
            if not rerun_agent_failures:
                raise EvaluationServiceError(409, "a scored evaluation cannot be resumed")
            if card.accepted_at is not None:
                raise EvaluationServiceError(
                    409, "a published evaluation cannot be resumed"
                )
            session.delete(card)

        task_seeds = run.task_seeds if isinstance(run.task_seeds, dict) else {}
        expected_keys = [
            f"{task_id}/seed_{int(seed)}"
            for task_id in (run.execution_plan.get("benchmark") or {}).get(
                "task_ids", task_seeds
            )
            for seed in task_seeds.get(task_id, ())
        ]
        stored_tasks = session.scalars(
            select(RunTask).where(RunTask.run_id == run.id)
        ).all()
        stored_by_key = {task.task_key: task for task in stored_tasks}
        run_dir = Path(run.run_dir) if run.run_dir else None
        if run_dir is not None and not run_dir.is_absolute():
            run_dir = (installation_root or Path.cwd()) / run_dir
        if rerun_agent_failures:
            pending_keys = [
                task_key
                for task_key in expected_keys
                if stored_by_key.get(task_key) is not None
                and stored_by_key[task_key].reason == "agent_error"
            ]
            if not pending_keys:
                raise EvaluationServiceError(
                    409, "evaluation has no recorded agent failures to rerun"
                )
            completed = {
                task_key: evidence
                for task_key in expected_keys
                if task_key not in pending_keys
                and (
                    evidence := _complete_task_evidence(run_dir, task_key)
                ) is not None
            }
        else:
            completed = {
                task_key: evidence
                for task_key in expected_keys
                if (
                    evidence := _complete_task_evidence(run_dir, task_key)
                ) is not None
            }
            pending_keys = [
                task_key
                for task_key in expected_keys
                if task_key not in completed
            ]

        for task_key, evidence in completed.items():
            task = stored_by_key.get(task_key)
            if task is None:
                task = RunTask(run_id=run.id, task_key=task_key)
                session.add(task)
            task.stage = "done"
            task.passed = evidence.get("passed")
            task.reason = evidence.get("reason")
            task.task_time_sec = evidence.get("task_time_sec")
            task.env_time_sec = evidence.get("env_time_sec")
            task.agent_time_sec = evidence.get("agent_time_sec")
            task.num_steps = evidence.get("num_steps")
            task.cost_usd = evidence.get("cost_usd")
            task.usage = evidence.get("usage")
        for task in stored_tasks:
            if task.task_key in pending_keys:
                task.stage = "queued"
                task.passed = None
                task.reason = None
                task.task_time_sec = None
                task.env_time_sec = None
                task.agent_time_sec = None
                task.num_steps = None
                task.cost_usd = None
                task.usage = None

        prior_resumes = session.scalar(
            select(func.count(EventRow.id)).where(
                EventRow.run_id == run.id,
                EventRow.kind == "run_resume_queued",
            )
        ) or 0
        attempt = int(prior_resumes) + 1
        run.stage = "queued"
        run.error = None
        run.finished_at = None
        run.cancel_requested_at = None
        if rerun_agent_failures:
            run.season_key = None
        session.add(EventRow(
            run_id=run.id,
            ts=time.time(),
            kind="run_resume_queued",
            task_key=None,
            payload={
                "attempt": attempt,
                # None means the original run stopped before seeds existed;
                # the worker will draw and persist them exactly once as usual.
                "task_keys": pending_keys if task_seeds else None,
                "task_count": len(pending_keys) if task_seeds else None,
                "rerun_agent_failures": rerun_agent_failures,
            },
        ))
        session.commit()
        return {
            "run_id": run.id,
            "stage": run.stage,
            "resume_attempt": attempt,
            "task_count": len(pending_keys) if task_seeds else None,
            "rerun_agent_failures": rerun_agent_failures,
        }


__all__ = [
    "EvaluationServiceError",
    "TERMINAL_STAGES",
    "benchmark_id_for_name",
    "cancel_evaluation",
    "ensure_local_user",
    "get_evaluation",
    "queue_evaluation",
    "resume_evaluation",
    "run_payload",
    "validate_submission_zip",
]
