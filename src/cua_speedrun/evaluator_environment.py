"""Benchmark-declared evaluator credentials, isolated from submission code."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping


_PROCESS_ENVIRONMENT_LOCK = threading.RLock()


def requirements(benchmark) -> tuple[set[str], set[str]]:
    required, private = set(), set()
    for task in benchmark.tasks:
        directory = task.env.get("env_dir")
        if not directory:
            continue
        path = Path(directory) / "evaluator-environment.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        if set(data) - {"required", "private"}:
            raise ValueError(f"{path}: unknown evaluator environment fields")
        for field, target in (("required", required), ("private", private)):
            values = data.get(field, [])
            if not isinstance(values, list) or not all(
                isinstance(name, str) and name.isidentifier() and name.isupper()
                for name in values
            ):
                raise ValueError(f"{path}: {field} must contain environment names")
            target.update(values)
    return required, private | required


def validate(benchmark, available: Mapping[str, str]) -> None:
    required, _ = requirements(benchmark)
    missing = sorted(name for name in required if not available.get(name, "").strip())
    if missing:
        raise ValueError(
            "benchmark requires evaluator credentials before running: "
            + ", ".join(missing)
        )


def agent_environment(benchmark, available: Mapping[str, str]) -> dict[str, str]:
    _, private = requirements(benchmark)
    return {name: value for name, value in available.items() if name not in private}


def evaluator_environment(
    benchmark, available: Mapping[str, str]
) -> dict[str, str]:
    """Return only variables declared private to the evaluator plane."""
    _, private = requirements(benchmark)
    return {
        name: str(available[name])
        for name in sorted(private)
        if name in available
    }


@contextmanager
def evaluator_process_environment(
    environment: Mapping[str, str],
) -> Iterator[None]:
    """Expose evaluator variables to one in-process evaluation.

    Local agent processes always receive an explicit, already-filtered
    environment. The lock keeps separate evaluations with different private
    credentials from sharing process state while their backends and verifiers
    run concurrently within each evaluation.
    """
    values = {str(name): str(value) for name, value in environment.items()}
    if not values:
        yield
        return
    with _PROCESS_ENVIRONMENT_LOCK:
        missing = {name for name in values if name not in os.environ}
        previous = {
            name: os.environ[name]
            for name in values
            if name in os.environ
        }
        os.environ.update(values)
        try:
            yield
        finally:
            for name in missing:
                os.environ.pop(name, None)
            os.environ.update(previous)
