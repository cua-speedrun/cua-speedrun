"""Enqueue a submission for the worker: the CLI face of the platform queue.

Zips a submission folder into the artifact store, upserts the dev user,
track, and benchmark rows, and creates a queued run.

Run: .venv/bin/python -m cua_speedrun.service.enqueue \
        --submission agents/openai \
        --benchmark benchmarks/osworld-50 --track default
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

from sqlalchemy import select

from cua_speedrun.service.db import (
    BenchmarkRow,
    Run,
    SubmissionRow,
    Track,
    User,
    make_session_factory,
)
from cua_speedrun.service.store import LocalStore
from cua_speedrun.parallelism import ExecutionScale
from cua_speedrun.service.plans import (
    attach_plan,
    attach_scale,
    plan_for_catalog_rows,
)
from cua_speedrun.specs import Benchmark
from cua_speedrun.submission import Submission

def zip_submission(submission_dir: Path) -> bytes:
    sub = Submission.load(submission_dir)  # validates the two-script contract
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, strict_timestamps=False) as zf:
        zf.write(sub.init_script, "init.py")
        zf.write(sub.agent_script, "agent.py")
    return buf.getvalue()


def enqueue(submission_dir: Path, benchmark_dir: Path, track: str,
            name: str | None = None, db_url: str | None = None,
            store_root: Path = Path("store"),
            parallel_evaluations: int = 1) -> int:
    session_factory = make_session_factory(db_url)
    from cua_speedrun.service.catalog import sync_catalog

    sync_catalog(session_factory)
    store = LocalStore(store_root)
    benchmark = Benchmark.load(benchmark_dir)  # validates the manifest
    sub = Submission.load(submission_dir)

    with session_factory() as session:
        user = session.execute(
            select(User).where(User.handle == "dev")
        ).scalar_one_or_none()
        if user is None:
            user = User(handle="dev")
            session.add(user)
            session.flush()

        track_row = session.execute(
            select(Track).where(Track.name == track)
        ).scalar_one_or_none()
        if track_row is None:
            raise ValueError(f"unknown track {track!r}")

        bench = session.execute(
            select(BenchmarkRow).where(BenchmarkRow.name == benchmark.name,
                                       BenchmarkRow.version == benchmark.version)
        ).scalar_one_or_none()
        if bench is None:
            bench = BenchmarkRow(name=benchmark.name, version=benchmark.version,
                                 path=str(Path(benchmark_dir).resolve()),
                                 task_count=len(benchmark.tasks))
            session.add(bench)
            session.flush()

        plan = plan_for_catalog_rows(track_row, bench)
        scale = ExecutionScale.for_plan(
            plan,
            parallel_evaluations,
            enforce_admission_limit=True,
        )
        ref = store.put_bytes(zip_submission(submission_dir), suffix=".zip")
        row = SubmissionRow(
            user_id=user.id,
            name=name or submission_dir.name,
            fingerprint=sub.fingerprint,
            storage_ref=ref,
            track=track,
            status="queued",
        )
        session.add(row)
        session.flush()
        run = Run(
            submission_id=row.id,
            benchmark_id=bench.id,
            stage="queued",
        )
        attach_plan(run, plan)
        attach_scale(run, scale)
        session.add(run)
        session.commit()
        print(f"queued run {run.id}: submission '{row.name}' "
              f"({sub.fingerprint}) on {benchmark.name}@{benchmark.version}, "
              f"track {track}")
        return run.id


def main() -> None:
    from cua_speedrun.config import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--track", default="default")
    parser.add_argument("--name", default=None)
    parser.add_argument("--parallel-evaluations", type=int, default=1)
    args = parser.parse_args()
    enqueue(
        Path(args.submission),
        Path(args.benchmark),
        args.track,
        args.name,
        parallel_evaluations=args.parallel_evaluations,
    )


if __name__ == "__main__":
    main()
