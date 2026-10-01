"""Fresh isolated compute/agent resource for every task instance."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
import time
from typing import TYPE_CHECKING

from cua_speedrun.eval_algorithms import EvalAlgorithm
from cua_speedrun.evaluation_runtime import (
    EvaluationRuntime,
    INSTANCE_INFRASTRUCTURE_MAX_ATTEMPTS,
    ISOLATED_INSTANCE,
    InstanceInfrastructureError,
    Job,
    ResultRow,
)

if TYPE_CHECKING:
    from cua_speedrun.parallelism import ExecutionScale


def _run_with_infrastructure_retries(
    runtime: EvaluationRuntime,
    job: Job,
) -> ResultRow:
    for attempt in range(1, INSTANCE_INFRASTRUCTURE_MAX_ATTEMPTS + 1):
        try:
            return runtime.run_isolated(job)
        except InstanceInfrastructureError as exc:
            if attempt == INSTANCE_INFRASTRUCTURE_MAX_ATTEMPTS:
                runtime.instance_infrastructure_failed(
                    None, job, INSTANCE_INFRASTRUCTURE_MAX_ATTEMPTS, exc
                )
                raise
            delay_sec = float(2 ** (attempt - 1))
            runtime.instance_infrastructure_retry(
                None,
                job,
                attempt,
                INSTANCE_INFRASTRUCTURE_MAX_ATTEMPTS,
                delay_sec,
                exc,
            )
            time.sleep(delay_sec)
    raise AssertionError("isolated infrastructure retry loop exhausted")


def schedule(
    runtime: EvaluationRuntime,
    jobs: Sequence[Job],
    scale: ExecutionScale,
) -> list[ResultRow]:
    concurrency = scale.agents_per_evaluation
    if scale.parallel_evaluations != 1:
        raise ValueError("per-task algorithms do not share compute replicas")
    if scale.environment_pool_size_per_evaluation != concurrency:
        raise ValueError(
            "fresh-agent algorithms require env pool size == agent concurrency"
        )

    rows: list[ResultRow] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_run_with_infrastructure_retries, runtime, job)
            for job in jobs
        ]
        rows.extend(future.result() for future in futures)
    return rows


ALGORITHM = EvalAlgorithm(
    key="per-task-vllm@1",
    label="Fresh sandbox / task",
    aliases=("per-task",),
    agent_mode="per-task",
    shared_agent_sandbox=False,
    default_env_pool_factor=1,
    supports_parallel_evaluations=False,
    required_runtime_capabilities=frozenset({ISOLATED_INSTANCE}),
    schedule=schedule,
)
