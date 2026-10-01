"""Task-to-seed job expansion shared by local and remote executors."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any


def build_task_seed_jobs(
    tasks: Sequence[Any],
    runs_per_task: int,
    seed_base: int,
    task_seeds: Mapping[str, Sequence[int]] | None = None,
) -> list[tuple[Any, int]]:
    if runs_per_task < 1:
        raise ValueError("runs_per_task must be at least 1")
    if task_seeds is None:
        return [
            (task, seed_base + rep)
            for task in tasks
            for rep in range(runs_per_task)
        ]

    tasks_by_id = {task.task_id: task for task in tasks}
    unknown = sorted(set(task_seeds) - set(tasks_by_id))
    if unknown:
        raise ValueError(f"explicit seeds supplied for unknown tasks: {unknown}")

    jobs: list[tuple[Any, int]] = []
    for task in tasks:
        seeds = [int(seed) for seed in task_seeds.get(task.task_id, ())]
        if len(seeds) != runs_per_task:
            raise ValueError(
                f"task {task.task_id!r} needs {runs_per_task} explicit seeds, "
                f"got {len(seeds)}"
            )
        jobs.extend((task, seed) for seed in seeds)
    return jobs


def select_task_seed_jobs(
    jobs: Sequence[tuple[Any, int]],
    task_keys: Collection[str] | None,
) -> list[tuple[Any, int]]:
    """Select an operational subset without changing the frozen seed map."""
    if task_keys is None:
        return list(jobs)
    requested = {str(task_key) for task_key in task_keys}
    available = {
        f"{task.task_id}/seed_{seed}": (task, seed)
        for task, seed in jobs
    }
    unknown = sorted(requested - set(available))
    if unknown:
        raise ValueError(f"selected task instances are not in the run plan: {unknown}")
    return [
        (task, seed)
        for task, seed in jobs
        if f"{task.task_id}/seed_{seed}" in requested
    ]
