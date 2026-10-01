"""The run worker: consumes queued runs and executes their frozen target.

Remote placements use the submitting user's provider credentials; fully local
placements need no user cloud credentials. Each claimed run executes in a fresh child process
so credentials and submission processes never leak into the queue loop.
Both executors emit the same database event stream, run logs, immutable plan,
and private result card.

Run:  .venv/bin/python -m cua_speedrun.service.worker [--once] [--poll 5]
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from sqlalchemy import or_, select, update

from cua_speedrun.events import Event
from cua_speedrun.leaderboard import build_card
from cua_speedrun.service.cancellation import (
    cleanup_remote_run_in_child,
    terminate_run_child,
)
from cua_speedrun.service.db import (
    BenchmarkRow,
    Card,
    EventRow,
    Run,
    RunEnvironmentVariable,
    RunTask,
    SubmissionRow,
    Track,
    make_session_factory,
    utcnow,
)
from cua_speedrun.service.store import LocalStore
from cua_speedrun.service.task_cost import read_task_cost_snapshot

# Executor event kinds that advance the run's coarse stage.
_RUN_STAGE_BY_KIND = {
    "init_started": "initializing",
    "init_done": "snapshot_done",
    "task_started": "running",
}
_RUN_STAGE_ORDER = {
    stage: index
    for index, stage in enumerate((
        "queued",
        "starting",
        "initializing",
        "snapshot_done",
        "running",
        "scoring",
        "card_ready",
    ))
}
# Per-task stage shown in the live grid, keyed by event kind.
_TASK_STAGE_BY_KIND = {
    "task_started": "env_boot",
    "env_created": "env_boot",
    "warmup_started": "warming",
    "warmup_fallback": "warming",
    "warmup_done": "warm",
    "env_ready": "env_ready",
    "armed": "running",
    "agent_exit": "checking",
    "agent_failed": "done",
    "infra_retry": "retrying",
    "task_interrupted": "retrying",
    "task_done": "done",
    "task_failed": "failed",
}
_TERMINAL_RUN_STAGES = frozenset({
    "card_ready", "failed", "rejected", "held", "cancelled",
})


class DBSink:
    """Writes executor events into the events table and folds them into the
    run/run_tasks live-status rows. Runs on executor threads, so it opens a
    short session per event."""

    def __init__(self, session_factory, run_db_id: int):
        self.session_factory = session_factory
        self.run_db_id = run_db_id

    def __call__(self, event: Event) -> None:
        if event.kind == "env_line":
            # The env sandbox's boot log is high-volume line spam; it stays
            # in events.jsonl and the pulled env_plane.log, never in the
            # live-status table.
            return
        with self.session_factory() as session:
            payload = dict(event.payload)
            if event.kind == "task_done" and event.task_key:
                run = session.get(Run, self.run_db_id)
                if run is not None and run.run_dir:
                    snapshot = read_task_cost_snapshot(run.run_dir, event.task_key)
                    if snapshot is not None:
                        payload.update(snapshot)
            # init_line rows are the live log pane's content; init.log on
            # disk stays the verbatim artifact.
            session.add(EventRow(
                run_id=self.run_db_id, ts=event.ts, kind=event.kind,
                task_key=event.task_key, payload=payload,
            ))
            stage = _RUN_STAGE_BY_KIND.get(event.kind)
            if stage:
                run = session.get(Run, self.run_db_id)
                if (
                    run is not None
                    and run.cancel_requested_at is None
                    and _RUN_STAGE_ORDER.get(stage, -1)
                    > _RUN_STAGE_ORDER.get(run.stage, -1)
                ):
                    run.stage = stage
            if event.task_key:
                task = session.execute(
                    select(RunTask).where(RunTask.run_id == self.run_db_id,
                                          RunTask.task_key == event.task_key)
                ).scalar_one_or_none()
                if task is None:
                    task = RunTask(run_id=self.run_db_id, task_key=event.task_key)
                    session.add(task)
                task_stage = _TASK_STAGE_BY_KIND.get(event.kind)
                if task_stage:
                    task.stage = task_stage
                if event.kind == "task_started":
                    task.passed = None
                    task.reason = None
                    task.task_time_sec = None
                    task.env_time_sec = None
                    task.agent_time_sec = None
                    task.num_steps = None
                    task.cost_usd = None
                    task.usage = None
                if event.kind == "armed":
                    task.num_steps = 0
                if event.kind == "task_progress":
                    task.num_steps = int(event.payload.get("num_steps") or 0)
                if event.kind == "agent_failed":
                    task.passed = False
                    task.reason = "agent_error"
                if event.kind == "task_done":
                    p = payload
                    task.passed = p.get("passed")
                    task.reason = p.get("reason")
                    task.task_time_sec = p.get("task_time_sec")
                    task.env_time_sec = p.get("env_time_sec")
                    task.agent_time_sec = p.get("agent_time_sec")
                    task.num_steps = p.get("num_steps")
                    task.cost_usd = p.get("cost_usd")
                    task.usage = p.get("usage")
            session.commit()


def _topology_key(run: Run) -> str:
    if run.topology_key:
        return run.topology_key
    stored = dict(run.execution_plan or {})
    topology = dict((stored.get("execution") or {}).get("topology") or {})
    key = topology.get("key")
    if key:
        return str(key)
    return (
        "local"
        if stored.get("backend") in ("local", "gym-anything-local")
        else "modal-remote"
    )


def _archive_resume_artifacts(
    run_dir: Path,
    task_keys: list[str] | None,
    attempt: int,
) -> Path:
    """Keep failed-attempt evidence while clearing paths that will be reused."""
    archive = run_dir / "resume_attempts" / f"attempt_{attempt}"
    archive.mkdir(parents=True, exist_ok=False)
    (archive / "resume.json").write_text(json.dumps({
        "attempt": attempt,
        "task_keys": task_keys,
    }, indent=2, sort_keys=True))

    for name in (
        "agent-python",
        "compute",
        "harness",
        "init.log",
        "model-server.log",
        "result.json",
        "server.pid",
    ):
        source = run_dir / name
        if source.exists():
            source.replace(archive / name)

    for task_key in task_keys or ():
        task_id, separator, seed = task_key.rpartition("/seed_")
        if not separator:
            raise ValueError(f"invalid resumed task key {task_key!r}")
        source = run_dir / "tasks" / task_id / f"seed_{int(seed)}"
        if not source.exists():
            continue
        target = archive / "tasks" / task_id / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
    return archive


def _claim_run(session_factory, run_id: int) -> bool:
    """Atomically claim one explicitly selected queued run."""
    with session_factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise ValueError(f"unknown run {run_id}")
        topology_key = _topology_key(run)
        claimed = session.execute(
            update(Run)
            .where(Run.id == run_id, Run.stage == "queued")
            .values(
                stage="starting",
                started_at=run.started_at or utcnow(),
                topology_key=topology_key,
            )
        )
        session.commit()
        return claimed.rowcount == 1


def _claim_next(session_factory, accepted_topology: str) -> int | None:
    """Atomically claim the oldest compatible queued run."""
    with session_factory() as session:
        query = select(Run).where(Run.stage == "queued")
        if accepted_topology != "all":
            query = query.where(or_(
                Run.topology_key == accepted_topology,
                Run.topology_key.is_(None),
            ))
        candidates = session.execute(query.order_by(Run.id)).scalars().all()
        for run in candidates:
            topology_key = _topology_key(run)
            if accepted_topology != "all" and topology_key != accepted_topology:
                continue
            claimed = session.execute(
                update(Run)
                .where(Run.id == run.id, Run.stage == "queued")
                .values(
                    stage="starting",
                    started_at=run.started_at or utcnow(),
                    topology_key=topology_key,
                )
            )
            session.commit()
            return run.id if claimed.rowcount == 1 else None
        session.commit()
        return None


def _cancel_requested(session_factory, run_id: int) -> bool:
    with session_factory() as session:
        run = session.get(Run, run_id)
        return run is not None and run.cancel_requested_at is not None


def _run_process_marker(session_factory, run_id: int) -> str | None:
    """Return the unique run id embedded in local runtime process paths."""
    with session_factory() as session:
        run = session.get(Run, run_id)
        return Path(run.run_dir).name if run is not None and run.run_dir else None


def _record_cancelled(session_factory, run_id: int) -> None:
    """Finalize a requested cancellation without discarding partial evidence."""
    with session_factory() as session:
        run = session.get(Run, run_id)
        if run is None or run.stage == "cancelled":
            return
        run.stage = "cancelled"
        run.error = None
        run.finished_at = utcnow()
        session.execute(
            update(RunTask)
            .where(
                RunTask.run_id == run_id,
                RunTask.stage.not_in(("done", "failed", "cancelled")),
            )
            .values(stage="cancelled")
        )
        session.add(EventRow(
            run_id=run_id,
            ts=time.time(),
            kind="run_cancelled",
            task_key=None,
            payload={"before_start": False},
        ))
        session.commit()


def _extract_submission(store: LocalStore, ref: str, workdir: Path) -> Path:
    """Unpack and validate a submission zip: exactly init.py and agent.py
    at the archive root. The web tier never opens these; only the worker
    does, and only to hand paths to sandboxes."""
    target = workdir / "submission"
    target.mkdir()
    with zipfile.ZipFile(store.path(ref)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if sorted(names) != ["agent.py", "init.py"]:
            raise ValueError(
                f"submission must contain exactly init.py and agent.py at the "
                f"root, got: {sorted(names)}"
            )
        zf.extractall(target)
    return target


def _resolve_local_plan_on_worker(
    plan, benchmark_path: Path, runner_selection: dict | None = None
):
    """Validate a local evaluator and freeze its identity before execution.

    A newly queued local plan carries a portable runtime request. The worker
    belonging to this dashboard installation checks the track GPU, KVM/QEMU
    backend, and benchmark image, then replaces that request with the observed
    evaluator contract exactly once.
    """
    from dataclasses import replace

    from cua_speedrun.compute_runners import (
        local_runtime_overrides,
        resolve_runner_selection,
    )
    from cua_speedrun.envs import get_backend
    from cua_speedrun.local_runtime import (
        current_local_agent_runtime_contract,
        local_runtime_needs_resolution,
        validate_local_agent_runtime_contract,
        validate_local_gpu,
    )
    from cua_speedrun.specs import Benchmark

    topology = plan.resolved_execution_topology()
    if topology["key"] != "local":
        return plan

    backend = get_backend(topology["environment_backend"])
    backend.preflight(Benchmark.load(benchmark_path))
    environment_runner = backend.observed_runner_name
    if environment_runner is None:
        raise RuntimeError(
            "gym-anything preflight did not report its selected local runner"
        )
    stored = dict(runner_selection or {})
    selection = resolve_runner_selection(
        stored.get("kind"), stored.get("template")
    )
    overrides = local_runtime_overrides(selection, plan.gpu)
    observed = current_local_agent_runtime_contract(
        environment_runner=environment_runner,
        **overrides,
    )
    validate_local_gpu(observed, plan.gpu)

    if local_runtime_needs_resolution(plan.resolved_agent_runtime()):
        return replace(plan, agent_runtime=observed)

    validate_local_agent_runtime_contract(
        plan.resolved_agent_runtime(),
        environment_runner=environment_runner,
        **overrides,
    )
    return plan


def _execute_claimed_child(
    session_factory,
    run_db_id: int,
) -> bool:
    """Execute and supervise an already-claimed run in an isolated child."""
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cua_speedrun.service.worker",
            "--run-one",
            str(run_db_id),
        ],
    )
    stopped = False
    while child.poll() is None:
        if _cancel_requested(session_factory, run_db_id):
            stopped = True
            terminate_run_child(
                child,
                process_marker=_run_process_marker(session_factory, run_db_id),
            )
            break
        time.sleep(0.25)
    returncode = child.wait()
    if stopped or _cancel_requested(session_factory, run_db_id):
        try:
            cleanup_remote_run_in_child(run_db_id)
        except subprocess.TimeoutExpired:
            print(
                f"run {run_db_id} remote cleanup timed out; "
                "provider sandbox timeouts remain as the fallback",
                flush=True,
            )
        _record_cancelled(session_factory, run_db_id)
        print(f"run {run_db_id} cancelled", flush=True)
        return True
    if returncode != 0:
        with session_factory() as session:
            run = session.get(Run, run_db_id)
            if run is not None and run.stage not in _TERMINAL_RUN_STAGES:
                session.execute(update(Run).where(Run.id == run_db_id).values(
                    stage="failed",
                    error=f"worker child exited with code {returncode}",
                    finished_at=utcnow()))
                session.commit()
        print(f"run {run_db_id} child exited {returncode}", flush=True)
    return True


def process_run(session_factory, run_id: int) -> bool:
    """Claim and execute one explicitly selected queued run."""
    if not _claim_run(session_factory, run_id):
        raise ValueError(f"run {run_id} is not queued")
    return _execute_claimed_child(session_factory, run_id)


def process_one(
    session_factory,
    store: LocalStore,
    runs_root: Path,
    accepted_topology: str = "modal-remote",
) -> bool:
    """Claim one queued run and execute it in a child process. Returns False
    when the queue is empty. The child exists for credential isolation (the
    modal SDK caches its client per process) and so an executor crash can
    never take the queue loop down."""
    run_db_id = _claim_next(session_factory, accepted_topology)
    if run_db_id is None:
        return False
    return _execute_claimed_child(session_factory, run_db_id)


def execute_claimed_run(session_factory, store: LocalStore, runs_root: Path,
                        run_db_id: int) -> bool:
    """Execute one already-claimed run on its frozen execution target."""
    import os

    from cua_speedrun.service.db import User
    from cua_speedrun.service.plans import plan_for_run, scale_for_run
    from cua_speedrun.service.usersecrets import decrypt

    with session_factory() as session:
        run = session.get(Run, run_db_id)
        submission = session.get(SubmissionRow, run.submission_id)
        benchmark = session.get(BenchmarkRow, run.benchmark_id)
        track = session.execute(
            select(Track).where(Track.name == submission.track)
        ).scalar_one_or_none()
        if track is None and not run.execution_plan:
            raise ValueError(
                f"legacy run {run.id} refers to missing track {submission.track!r}"
            )
        owner = session.get(User, submission.user_id)
        encrypted_environment = session.scalars(
            select(RunEnvironmentVariable)
            .where(RunEnvironmentVariable.run_id == run_db_id)
            .order_by(RunEnvironmentVariable.name)
        ).all()
        plan = plan_for_run(run, track, benchmark)
        execution_scale = scale_for_run(run, plan)
        # The operator's runner choice was frozen at submit; every claiming
        # executor must honor it rather than its own process defaults.
        stored_runner = dict(getattr(run, "runner_selection", None) or {})
        resume_event = session.execute(
            select(EventRow)
            .where(
                EventRow.run_id == run_db_id,
                EventRow.kind == "run_resume_queued",
            )
            .order_by(EventRow.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        resume_payload = dict(resume_event.payload or {}) if resume_event else {}
        is_resume = resume_event is not None
        resumed_task_keys = resume_payload.get("task_keys")
        if resumed_task_keys is not None:
            resumed_task_keys = [str(key) for key in resumed_task_keys]
        resume_attempt = int(resume_payload.get("attempt") or 0)
        existing_run_dir = Path(run.run_dir) if run.run_dir else None
        if existing_run_dir is not None and not existing_run_dir.is_absolute():
            existing_run_dir = runs_root.parent / existing_run_dir
        # Legacy queued rows acquire their plan once here. New dashboard and
        # enqueue paths already stored it before the run became visible.
        session.commit()

    topology = plan.resolved_execution_topology()
    token_id = owner.modal_token_id if owner else None
    token_secret = (
        decrypt(owner.modal_token_secret_enc)
        if owner and owner.modal_token_secret_enc else None
    )
    environment_variables: dict[str, str] = {}
    invalid_environment_variables: list[str] = []
    for row in encrypted_environment:
        value = decrypt(row.value_enc)
        if value is None:
            invalid_environment_variables.append(row.name)
        else:
            environment_variables[row.name] = value
    if invalid_environment_variables:
        with session_factory() as session:
            session.execute(update(Run).where(Run.id == run_db_id).values(
                stage="failed",
                error="environment variables could not be decrypted; submit "
                      "the evaluation again: "
                      + ", ".join(sorted(invalid_environment_variables)),
                finished_at=utcnow()))
            session.commit()
        return True
    if topology["requires_user_credentials"] and (not token_id or not token_secret):
        with session_factory() as session:
            session.execute(update(Run).where(Run.id == run_db_id).values(
                stage="failed",
                error="no Modal credentials on the submitting account: "
                      "add your token pair under Evaluations before submitting",
                finished_at=utcnow()))
            session.commit()
        return True
    if topology["requires_user_credentials"]:
        # Remote runs execute in the owner's Modal workspace, never the
        # platform's. This child exits before another user's run starts.
        os.environ["MODAL_TOKEN_ID"] = token_id
        os.environ["MODAL_TOKEN_SECRET"] = token_secret

    sink = DBSink(session_factory, run_db_id)
    try:
        if topology["key"] == "local":
            resolved_plan = _resolve_local_plan_on_worker(
                plan, Path(benchmark.path), stored_runner
            )
            if resolved_plan.contract_hash != plan.contract_hash:
                from cua_speedrun.service.plans import attach_plan

                with session_factory() as session:
                    current_run = session.get(Run, run_db_id)
                    attach_plan(current_run, resolved_plan)
                    session.commit()
                plan = resolved_plan
                topology = plan.resolved_execution_topology()
        with tempfile.TemporaryDirectory(prefix="cs_sub_") as tmp:
            submission_dir = _extract_submission(store, submission.storage_ref, Path(tmp))

            # Persist the exact task-to-seed map before starting the executor;
            # retries recover the same instances.
            with session_factory() as session:
                current_run = session.get(Run, run_db_id)
                task_seeds = current_run.task_seeds or {}
                if not task_seeds:
                    if plan.seed_policy == "scored-without-replacement@1":
                        from cua_speedrun.service.seeds import draw_scored_task_seeds

                        task_seeds = draw_scored_task_seeds(
                            session,
                            benchmark.id,
                            plan.benchmark["task_ids"],
                            run_db_id,
                            runs_per_task=plan.runs_per_task,
                        )
                    else:
                        task_seeds = {
                            task_id: list(range(plan.runs_per_task))
                            for task_id in plan.benchmark["task_ids"]
                        }
                    current_run.task_seeds = task_seeds
                    session.commit()

            if is_resume and resumed_task_keys is None:
                resumed_task_keys = [
                    f"{task_id}/seed_{int(seed)}"
                    for task_id in plan.benchmark["task_ids"]
                    for seed in task_seeds.get(task_id, ())
                ]

            if is_resume and existing_run_dir is not None:
                run_dir_path = existing_run_dir
                run_id = run_dir_path.name
                executor_root = run_dir_path.parent
                archive = _archive_resume_artifacts(
                    run_dir_path,
                    resumed_task_keys,
                    resume_attempt,
                )
                with session_factory() as session:
                    session.add(EventRow(
                        run_id=run_db_id,
                        ts=time.time(),
                        kind="run_resume_started",
                        task_key=None,
                        payload={
                            "attempt": resume_attempt,
                            "task_keys": resumed_task_keys,
                            "archive": str(archive),
                        },
                    ))
                    session.commit()
            else:
                run_id = f"svc_{run_db_id}_{secrets.token_hex(3)}"
                executor_root = runs_root
            # Record the run directory up front, not at card time. Task
            # artifacts land in it as each task finishes, and the task pages
            # resolve the directory through this column; writing it only at
            # completion made every mid-run artifact link 404 even though
            # the files were already on disk.
            with session_factory() as session:
                session.execute(update(Run).where(Run.id == run_db_id)
                                .values(run_dir=str(executor_root / run_id)))
                session.commit()
            # Both Modal topologies run through the remote executor; the run
            # plan's environment backend selects nested QEMU or modal-native.
            if topology["key"] in ("modal-remote", "modal-native"):
                from cua_speedrun.remote.run import run_benchmark_remote

                run_dir = run_benchmark_remote(
                    submission_dir=submission_dir,
                    benchmark_dir=Path(benchmark.path),
                    out_root=executor_root,
                    run_id=run_id,
                    extra_sinks=[sink],
                    run_plan=plan,
                    task_seeds=task_seeds,
                    environment_variables=environment_variables,
                    execution_scale=execution_scale,
                    selected_task_keys=(
                        resumed_task_keys if is_resume else None
                    ),
                    # Snapshot image ids are scoped to the owner's workspace.
                    cache_namespace=token_id,
                )
            elif topology["key"] == "local":
                from cua_speedrun.executor import run_benchmark

                run_dir = run_benchmark(
                    submission_dir=submission_dir,
                    benchmark_dir=Path(benchmark.path),
                    backend_name=topology["environment_backend"],
                    out_root=executor_root,
                    run_id=run_id,
                    extra_sinks=[sink],
                    run_plan=plan,
                    task_seeds=task_seeds,
                    environment_variables=environment_variables,
                    inherit_process_environment=False,
                    execution_scale=execution_scale,
                    compute_runner=stored_runner.get("kind"),
                    runner_template=stored_runner.get("template"),
                    selected_task_keys=(
                        resumed_task_keys if is_resume else None
                    ),
                )
            else:
                raise ValueError(
                    f"unsupported execution topology {topology['key']!r}"
                )

        card_data = build_card(run_dir)
        with session_factory() as session:
            from cua_speedrun.service.publishing import register_season

            completed = session.execute(
                update(Run)
                .where(
                    Run.id == run_db_id,
                    Run.cancel_requested_at.is_(None),
                )
                .values(
                    stage="card_ready",
                    run_dir=str(run_dir),
                    season_key=card_data["season_key"],
                    finished_at=utcnow(),
                )
            )
            if completed.rowcount != 1:
                session.rollback()
                return True
            register_season(
                session,
                key=card_data["season_key"],
                contract_hash=plan.contract_hash,
                spec=plan.to_dict(),
            )
            session.add(Card(run_id=run_db_id,
                             token=secrets.token_urlsafe(24),
                             data=card_data))
            session.commit()
    except Exception as exc:
        with session_factory() as session:
            session.execute(update(Run).where(Run.id == run_db_id).values(
                stage="failed", error=repr(exc), finished_at=utcnow(),
            ))
            session.commit()
        raise
    return True


def run_worker(poll_sec: float = 5.0, once: bool = False,
               accepted_topology: str = "modal-remote",
               db_url: str | None = None,
               store_root: Path = Path("store"),
               runs_root: Path = Path("runs")) -> None:
    session_factory = make_session_factory(db_url)
    from cua_speedrun.service.catalog import sync_catalog

    sync_catalog(session_factory)
    store = LocalStore(store_root)
    print(
        f"worker up (accept={accepted_topology}, db ok, "
        f"store={store.root}, runs={runs_root})",
        flush=True,
    )
    while True:
        try:
            worked = process_one(
                session_factory, store, runs_root, accepted_topology
            )
        except Exception as exc:
            print(f"run failed: {exc!r}", flush=True)
            worked = True  # the failure is recorded; keep draining the queue
        if once:
            return
        if not worked:
            time.sleep(poll_sec)


def _raise_open_file_limit() -> None:
    """Lift the soft fd limit to the hard limit before serving replicas.

    A wide evaluation multiplies descriptors: agent output streams, gateway
    HTTP sessions, request-file polling, and scheduler subprocess pipes per
    replica. The common 1024 soft default exhausts near 100 replicas and
    surfaces as OSError 24 misclassified as instance infrastructure.
    """
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except (ImportError, ValueError, OSError):
        pass


def main() -> None:
    from cua_speedrun.config import load_dotenv

    _raise_open_file_limit()
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true",
                        help="process at most one queued run, then exit")
    parser.add_argument("--poll", type=float, default=5.0)
    from cua_speedrun.execution_placements import list_execution_topologies

    topology_choices = [
        topology.key for topology in list_execution_topologies()
    ] + ["all"]
    parser.add_argument(
        "--accept-topology",
        choices=topology_choices,
        default="modal-remote",
        help="claim only runs for this registered execution topology",
    )
    parser.add_argument("--run-one", type=int, metavar="RUN_ID",
                        help="execute one already-claimed run and exit "
                             "(the per-run child the queue loop spawns)")
    parser.add_argument("--process-run", type=int, metavar="RUN_ID",
                        help="claim, supervise, and execute exactly one queued run")
    args = parser.parse_args()
    if args.run_one is not None and args.process_run is not None:
        parser.error("--run-one and --process-run are mutually exclusive")
    if args.run_one is not None:
        session_factory = make_session_factory()
        store = LocalStore(Path("store"))
        execute_claimed_run(session_factory, store, Path("runs"), args.run_one)
        return
    if args.process_run is not None:
        process_run(make_session_factory(), args.process_run)
        return
    run_worker(
        poll_sec=args.poll,
        once=args.once,
        accepted_topology=args.accept_topology,
    )


if __name__ == "__main__":
    main()
