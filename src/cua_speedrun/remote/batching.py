"""Small deterministic batching primitive used by bounded eval algorithms."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")


def job_batches(jobs: Sequence[T], size: int) -> Iterator[list[T]]:
    if size < 1:
        raise ValueError("batch size must be at least 1")
    for start in range(0, len(jobs), size):
        yield list(jobs[start:start + size])
