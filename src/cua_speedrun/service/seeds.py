"""Practice-versus-scored seed pool bookkeeping.

A task is a template; a seed instantiates it. Two pools:

- Practice seeds are public and reusable. Users iterate against them freely.
- Scored seeds are private, drawn WITHOUT REPLACEMENT, and retired after one
  use, so no two scored runs ever share a seed. This closes the
  memorize-and-replay hole: because the submission observes everything its
  seed produced, any reused private seed could be harvested and replayed by
  a later submission.

This module owns the draw-and-retire transaction. Practice seeds live in a
fixed low band (0..PRACTICE_MAX) that anyone may run; scored seeds are drawn
from a high band and marked used atomically so two concurrent runs can never
receive the same scored seed.
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from cua_speedrun.service.db import Run, SeedRow, utcnow

PRACTICE_MAX = 999          # seeds 0..999 are the public practice band
SCORED_BASE = 1_000_000     # scored seeds are drawn starting here
SCORED_MAX = 2_147_483_647  # portable signed 32-bit range


def ensure_practice_pool(session, benchmark_id: int, task_id: str,
                         count: int = 100) -> None:
    """Idempotently register the public practice seeds for a task."""
    if not 0 <= count <= PRACTICE_MAX + 1:
        raise ValueError(f"practice seed count must be between 0 and {PRACTICE_MAX + 1}")
    have = set(session.execute(
        select(SeedRow.seed).where(
            SeedRow.benchmark_id == benchmark_id,
            SeedRow.task_id == task_id, SeedRow.pool == "practice")
    ).scalars())
    for seed in range(count):
        if seed not in have:
            session.add(SeedRow(benchmark_id=benchmark_id, task_id=task_id,
                                seed=seed, pool="practice"))


def draw_scored_seed(session, benchmark_id: int, task_id: str,
                     run_id: int) -> int:
    """Stage one fresh scored-seed reservation in the caller's transaction.

    Never returns a seed a prior scored run used. The reservation is a
    single UPDATE that flips an unused row to this run, so two concurrent
    callers cannot claim the same seed (the second sees rowcount 0 and
    retries with the next seed)."""
    while True:
        row = session.execute(
            select(SeedRow).where(
                SeedRow.benchmark_id == benchmark_id,
                SeedRow.task_id == task_id, SeedRow.pool == "scored",
                SeedRow.used_by_run.is_(None))
            .order_by(SeedRow.seed).limit(1)
        ).scalar_one_or_none()
        if row is None:
            # Pool exhausted: mint an unpredictable private instance. A
            # sequential high-water seed would make the next scored task
            # guessable even though it had never been used. The unique index
            # handles the vanishingly unlikely random collision.
            seed = SCORED_BASE + secrets.randbelow(SCORED_MAX - SCORED_BASE + 1)
            try:
                with session.begin_nested():
                    session.add(SeedRow(
                        benchmark_id=benchmark_id,
                        task_id=task_id,
                        seed=seed,
                        pool="scored",
                        used_by_run=run_id,
                        retired_at=utcnow(),
                    ))
                    session.flush()
            except IntegrityError:
                # The unique index is the arbiter for a rare random collision.
                continue
            return int(seed)
        claimed = session.execute(
            update(SeedRow)
            .where(SeedRow.id == row.id, SeedRow.used_by_run.is_(None))
            .values(used_by_run=run_id, retired_at=utcnow())
        )
        session.flush()
        if claimed.rowcount == 1:
            return row.seed
        # Lost the race; loop and try the next unused seed.


def draw_scored_task_seeds(
    session,
    benchmark_id: int,
    task_ids: Sequence[str],
    run_id: int,
    *,
    runs_per_task: int = 1,
) -> dict[str, list[int]]:
    """Allocate a complete, retry-safe scored seed map for one run.

    The complete map and the Run row are committed together by the worker.
    A row-level lock (plus a no-op write lock on SQLite) serializes duplicate
    workers for the same run; a crash rolls the whole allocation back rather
    than silently burning a partial set of private instances.
    """
    if runs_per_task < 1:
        raise ValueError("runs_per_task must be at least 1")
    ordered_ids = list(dict.fromkeys(str(task_id) for task_id in task_ids))
    if len(ordered_ids) != len(task_ids):
        raise ValueError("task_ids must be unique")

    locked = session.execute(
        update(Run).where(Run.id == run_id).values(id=run_id)
    )
    if locked.rowcount != 1:
        raise ValueError(f"unknown run {run_id}")

    existing = session.execute(
        select(SeedRow).where(
            SeedRow.benchmark_id == benchmark_id,
            SeedRow.pool == "scored",
            SeedRow.used_by_run == run_id,
        )
    ).scalars().all()
    allocated = {task_id: [] for task_id in ordered_ids}
    unexpected = sorted({row.task_id for row in existing} - set(ordered_ids))
    if unexpected:
        raise ValueError(
            f"run {run_id} already owns seeds for tasks outside its plan: "
            f"{unexpected}"
        )
    for row in existing:
        if row.task_id in allocated:
            allocated[row.task_id].append(int(row.seed))

    for task_id in ordered_ids:
        seeds = allocated[task_id]
        if len(seeds) > runs_per_task:
            raise ValueError(
                f"run {run_id} already owns {len(seeds)} seeds for "
                f"{task_id!r}, expected {runs_per_task}"
            )
        while len(seeds) < runs_per_task:
            seeds.append(draw_scored_seed(session, benchmark_id, task_id, run_id))
        seeds.sort()
    return allocated


def seed_pool_stats(session, benchmark_id: int) -> dict:
    """Per-task counts for the admin view: practice size, scored retired."""
    rows = session.execute(
        select(SeedRow.task_id, SeedRow.pool,
               func.count(), func.count(SeedRow.used_by_run))
        .where(SeedRow.benchmark_id == benchmark_id)
        .group_by(SeedRow.task_id, SeedRow.pool)
    ).all()
    out: dict[str, dict] = {}
    for task_id, pool, total, used in rows:
        out.setdefault(task_id, {})[pool] = {"total": total, "used": used}
    return out
