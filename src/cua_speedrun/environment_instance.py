"""One prepared environment instance and its evaluator-owned evidence.

Both the in-process runner and scheduled replica workers use this module, so
environment boot, seeded-task resolution, logging, and cleanup stay identical
regardless of where the environment is placed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cua_speedrun
from cua_speedrun.envs.base import Verdict
from cua_speedrun.evaluation_runtime import InstanceInfrastructureError
from cua_speedrun.runlog import RunLogWriter
from cua_speedrun.submission import Submission


@dataclass
class PreparedEnvironmentInstance:
    submission: Submission
    task: Any
    seed: int
    task_key: str
    task_dir: Path
    log: RunLogWriter
    prepared: Any
    description: str
    checker: Callable[[], Verdict] | None
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.prepared.adapter.close()
        except Exception as exc:
            self.log.event("cleanup_error", error=repr(exc))
            raise InstanceInfrastructureError(
                f"environment cleanup failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            self.log.close()


def prepare_environment_instance(
    submission: Submission,
    backend: Any,
    task: Any,
    seed: int,
    run_dir: Path,
    run_id: str,
    events: Any,
) -> PreparedEnvironmentInstance:
    task_key = f"{task.task_id}/seed_{seed}"
    task_dir = run_dir / "tasks" / task.task_id / f"seed_{seed}"
    task_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogWriter(task_dir / "runlog.jsonl")
    prepared = None
    try:
        prepared = backend.prepare(task.env, seed, task_dir)

        # Seeded task: the generator derives the instruction and the
        # privileged expected answer from the seed, and its check runs next to
        # the environment. The answer never enters the agent sandbox.
        generator = task.load_generator()
        checker = None
        if generator is not None:
            spec = generator.generate(seed)
            description = spec["instruction"]
            expected = spec["expected"]

            def checker() -> Verdict:
                result = generator.check(prepared.adapter, expected)
                return Verdict(
                    passed=bool(result["passed"]),
                    score=100.0 if result["passed"] else 0.0,
                    detail=str(result.get("detail", "")),
                )
        else:
            description = prepared.description

        log.event(
            "header",
            run_id=run_id,
            task_id=task.task_id,
            seed=seed,
            backend=backend.name,
            harness_version=cua_speedrun.__version__,
            submission_fingerprint=submission.fingerprint,
            timeout_sec=task.timeout_sec,
            prepare_time_sec=prepared.prepare_time_sec,
            env_info=prepared.info,
            seeded=generator is not None,
            instruction=description,
        )
        events.emit(
            "env_ready",
            task_key=task_key,
            env_boot_sec=prepared.prepare_time_sec,
        )
        return PreparedEnvironmentInstance(
            submission=submission,
            task=task,
            seed=seed,
            task_key=task_key,
            task_dir=task_dir,
            log=log,
            prepared=prepared,
            description=description,
            checker=checker,
        )
    except Exception as exc:
        log.event("harness_error", error=repr(exc))
        if prepared is not None:
            prepared.adapter.close()
        log.close()
        raise


__all__ = ["PreparedEnvironmentInstance", "prepare_environment_instance"]
