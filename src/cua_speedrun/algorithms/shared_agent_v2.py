"""Rolling environment pools served by reusable compute replicas."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from queue import Empty, Queue
from threading import BoundedSemaphore, Lock
import time
from typing import Any, TYPE_CHECKING

from cua_speedrun.eval_algorithms import EvalAlgorithm
from cua_speedrun.evaluation_runtime import (
    ComputeInfrastructureError,
    ENVIRONMENT_PREPARE_MAX_ATTEMPTS,
    INSTANCE_INFRASTRUCTURE_MAX_ATTEMPTS,
    INDEPENDENT_ENVIRONMENT_PREPARATION,
    EvaluationRuntime,
    InstanceInfrastructureError,
    Job,
    ResultRow,
    SHARED_COMPUTE,
    prepare_with_backoff,
)

if TYPE_CHECKING:
    from cua_speedrun.parallelism import ExecutionScale


@dataclass(frozen=True)
class _WorkItem:
    job: Job
    attempt: int = 1


class _ReplicaRetired(RuntimeError):
    pass


class _SharedWork:
    def __init__(self, jobs: Sequence[Job]) -> None:
        self._queue: Queue[_WorkItem] = Queue()
        for job in jobs:
            self._queue.put(_WorkItem(job))
        self._remaining = len(jobs)
        self._lock = Lock()

    def claim(self) -> _WorkItem | None:
        try:
            return self._queue.get_nowait()
        except Empty:
            return None

    def retry(self, work: _WorkItem) -> None:
        self._queue.put(work)

    def complete_one(self) -> None:
        with self._lock:
            if self._remaining < 1:
                raise RuntimeError("shared work completion count underflow")
            self._remaining -= 1

    @property
    def complete(self) -> bool:
        with self._lock:
            return self._remaining == 0

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._remaining


def _ensure_compute_ready(
    runtime: EvaluationRuntime,
    replica: Any,
    initial_error: ComputeInfrastructureError | None = None,
) -> None:
    """Recover one compute replica under the algorithm's retry policy."""
    error = initial_error
    for attempt in range(1, INSTANCE_INFRASTRUCTURE_MAX_ATTEMPTS + 1):
        try:
            if error is not None:
                runtime.recover_instance_infrastructure(replica, error)
            runtime.ensure_compute_ready(replica)
            return
        except ComputeInfrastructureError as exc:
            error = exc
            if attempt == INSTANCE_INFRASTRUCTURE_MAX_ATTEMPTS:
                runtime.compute_infrastructure_failed(
                    replica,
                    INSTANCE_INFRASTRUCTURE_MAX_ATTEMPTS,
                    exc,
                )
                raise
            delay_sec = float(2 ** (attempt - 1))
            runtime.compute_infrastructure_retry(
                replica,
                attempt,
                INSTANCE_INFRASTRUCTURE_MAX_ATTEMPTS,
                delay_sec,
                exc,
            )
            time.sleep(delay_sec)
    raise AssertionError("compute infrastructure retry loop exhausted")


def _run_replica(
    runtime: EvaluationRuntime,
    replica: Any,
    pending_jobs: _SharedWork,
    scale: ExecutionScale,
) -> list[ResultRow]:
    concurrency = scale.agents_per_evaluation
    env_pool_size = scale.environment_pool_size_per_evaluation
    if not concurrency <= env_pool_size <= 2 * concurrency:
        raise ValueError(
            "shared-agent-vllm@2 requires agent concurrency <= env pool <= 2C"
        )

    capacity = BoundedSemaphore(env_pool_size)
    overlap = INDEPENDENT_ENVIRONMENT_PREPARATION in runtime.capabilities
    if not overlap:
        try:
            _ensure_compute_ready(runtime, replica)
        except ComputeInfrastructureError:
            runtime.close_compute_replica(replica)
            return []
        except BaseException:
            runtime.close_compute_replica(replica)
            raise
    live_lock = Lock()
    live: dict[int, Any] = {}

    def close_environment(
        number: int,
        prepared: Any,
        *,
        retain_capacity: bool = False,
        suppress_errors: bool = False,
    ) -> None:
        try:
            with live_lock:
                live.pop(number, None)
            runtime.close_environment_batch(
                replica,
                [prepared],
                number,
            )
        except BaseException:
            capacity.release()
            if not suppress_errors:
                raise
            return
        if not retain_capacity:
            capacity.release()

    def prepare_environment(
        number: int,
        job: Job,
        *,
        acquire_capacity: bool = True,
    ) -> tuple[int, Any]:
        if acquire_capacity:
            capacity.acquire()
        prepared = None
        try:
            runtime.begin_environment_batch(replica, (job,), number)

            def retry(attempt, max_attempts, delay_sec, error):
                runtime.environment_prepare_retry(
                    replica,
                    job,
                    number,
                    attempt,
                    max_attempts,
                    delay_sec,
                    error,
                )
                if isinstance(error, ComputeInfrastructureError):
                    try:
                        _ensure_compute_ready(runtime, replica, error)
                    except ComputeInfrastructureError as recovery_error:
                        raise _ReplicaRetired(str(recovery_error)) from recovery_error

            try:
                prepared = prepare_with_backoff(
                    lambda: runtime.prepare_environment(replica, job, number),
                    on_retry=retry,
                )
            except _ReplicaRetired:
                raise
            except Exception as error:
                runtime.environment_prepare_failed(
                    replica,
                    job,
                    number,
                    ENVIRONMENT_PREPARE_MAX_ATTEMPTS,
                    error,
                )
                raise
            with live_lock:
                live[number] = prepared
                try:
                    compute_ready.result()
                except ComputeInfrastructureError as exc:
                    raise _ReplicaRetired(str(exc)) from exc
                runtime.bind_compute(replica, list(live.values()), number)
            return number, prepared
        except BaseException:
            try:
                with live_lock:
                    live.pop(number, None)
                runtime.close_environment_batch(
                    replica,
                    [] if prepared is None else [prepared],
                    number,
                )
            finally:
                capacity.release()
            raise

    def run_environment(
        number: int,
        prepared: Any,
        work: _WorkItem,
    ) -> ResultRow | None:
        try:
            row = runtime.run_prepared(replica, prepared, number)
        except InstanceInfrastructureError as error:
            # Cleanup commonly fails for the same reason as the interrupted
            # instance (for example, its allocation has disappeared).  The
            # original infrastructure error must still be recorded and the
            # task retried.
            close_environment(number, prepared, suppress_errors=True)
            if work.attempt >= INSTANCE_INFRASTRUCTURE_MAX_ATTEMPTS:
                runtime.instance_infrastructure_failed(
                    replica,
                    work.job,
                    INSTANCE_INFRASTRUCTURE_MAX_ATTEMPTS,
                    error,
                )
                raise
            delay_sec = float(2 ** (work.attempt - 1))
            runtime.instance_infrastructure_retry(
                replica,
                work.job,
                work.attempt,
                INSTANCE_INFRASTRUCTURE_MAX_ATTEMPTS,
                delay_sec,
                error,
            )
            time.sleep(delay_sec)
            pending_jobs.retry(_WorkItem(work.job, work.attempt + 1))
            if isinstance(error, ComputeInfrastructureError):
                try:
                    _ensure_compute_ready(runtime, replica, error)
                    with live_lock:
                        if live:
                            runtime.bind_compute(
                                replica,
                                list(live.values()),
                                number,
                            )
                except BaseException as recovery_error:
                    raise _ReplicaRetired(str(recovery_error)) from recovery_error
            return None
        except BaseException:
            close_environment(number, prepared)
            raise
        close_environment(number, prepared)
        return row

    try:
        rows: list[ResultRow] = []
        prepare_failures: list[tuple[str, BaseException]] = []
        run_failures: list[BaseException] = []
        next_number = 1

        def claim_job() -> tuple[int, _WorkItem] | None:
            nonlocal next_number
            work = pending_jobs.claim()
            if work is None:
                return None
            claimed = (next_number, work)
            next_number += 1
            return claimed

        with (
            ThreadPoolExecutor(max_workers=1) as compute_pool,
            ThreadPoolExecutor(max_workers=env_pool_size) as prepare_pool,
            ThreadPoolExecutor(max_workers=concurrency) as agent_pool,
        ):
            compute_ready = Future()
            if overlap:
                compute_ready = compute_pool.submit(_ensure_compute_ready, runtime, replica)
            else:
                compute_ready.set_result(None)
            compute_checked = False
            preparations: dict[Any, tuple[int, _WorkItem]] = {}
            runs: dict[Any, tuple[int, _WorkItem]] = {}
            retired = False
            while preparations or runs or (not retired and not pending_jobs.complete):
                while not retired and len(preparations) + len(runs) < env_pool_size:
                    claimed = claim_job()
                    if claimed is None:
                        break
                    number, work = claimed
                    preparations[
                        prepare_pool.submit(prepare_environment, number, work.job)
                    ] = (number, work)

                active = set(preparations) | set(runs)
                if not compute_checked:
                    active.add(compute_ready)
                if not active:
                    time.sleep(0.1)
                    continue
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    if future is compute_ready:
                        compute_checked = True
                        try:
                            future.result()
                        except ComputeInfrastructureError:
                            retired = True
                        continue
                    if future in preparations:
                        number, work = preparations.pop(future)
                        try:
                            _, prepared = future.result()
                        except _ReplicaRetired:
                            retired = True
                            pending_jobs.retry(work)
                            continue
                        except BaseException as exc:
                            task, seed = work.job
                            prepare_failures.append(
                                (f"{task.task_id}/seed_{seed}", exc)
                            )
                            pending_jobs.complete_one()
                            continue
                        if retired:
                            close_environment(number, prepared)
                            pending_jobs.retry(work)
                            continue
                        runs[
                            agent_pool.submit(
                                run_environment,
                                number,
                                prepared,
                                work,
                            )
                        ] = (number, work)
                        continue

                    runs.pop(future)
                    try:
                        row = future.result()
                        if row is not None:
                            rows.append(row)
                            pending_jobs.complete_one()
                    except _ReplicaRetired:
                        retired = True
                    except BaseException as exc:
                        run_failures.append(exc)
                        pending_jobs.complete_one()

        if not retired:
            runtime.finish_compute(replica)
        if run_failures:
            raise run_failures[0]
        if prepare_failures:
            task_keys = ", ".join(key for key, _ in prepare_failures)
            raise RuntimeError(
                "evaluation incomplete: environment preparation failed after "
                f"{ENVIRONMENT_PREPARE_MAX_ATTEMPTS} attempts for "
                f"{len(prepare_failures)} task instance(s): {task_keys}; "
                "all unrelated instances completed"
            ) from prepare_failures[0][1]
        return rows
    finally:
        for number, prepared in list(live.items()):
            close_environment(number, prepared, suppress_errors=True)
        runtime.close_compute_replica(replica)


def schedule(
    runtime: EvaluationRuntime,
    jobs: Sequence[Job],
    scale: ExecutionScale,
) -> list[ResultRow]:
    replica_count = min(scale.parallel_evaluations, len(jobs))
    if replica_count == 0:
        return []

    # Runners use these balanced shards only to size resource lifetimes. Work
    # ownership remains in the shared queue below, so a ready replica always
    # claims the next unstarted task instead of waiting behind a static shard.
    budget_shards: list[list[Job]] = [[] for _ in range(replica_count)]
    for index, job in enumerate(jobs):
        budget_shards[index % replica_count].append(job)

    replicas = runtime.start_compute_replicas(budget_shards)
    if len(replicas) != len(budget_shards):
        for replica in replicas:
            runtime.close_compute_replica(replica)
        raise RuntimeError(
            f"runtime started {len(replicas)} replicas for "
            f"{len(budget_shards)} resource plans"
        )

    pending_jobs = _SharedWork(jobs)

    rows: list[ResultRow] = []
    submitted = 0
    try:
        with ThreadPoolExecutor(max_workers=len(replicas)) as pool:
            futures = []
            for replica in replicas:
                futures.append(
                    pool.submit(
                        _run_replica,
                        runtime,
                        replica,
                        pending_jobs,
                        scale,
                    )
                )
                submitted += 1
            for future in futures:
                rows.extend(future.result())
        if not pending_jobs.complete:
            raise RuntimeError(
                "evaluation incomplete: all compute replicas retired with "
                f"{pending_jobs.remaining} task instance(s) unfinished"
            )
        return rows
    except BaseException:
        # Submitted replicas close themselves in ``_run_replica``.  This
        # covers an executor-construction or partial-submission failure after
        # the runtime has already started the remaining compute resources.
        for replica in replicas[submitted:]:
            runtime.close_compute_replica(replica)
        raise


ALGORITHM = EvalAlgorithm(
    key="shared-agent-vllm@2",
    label="Shared sandbox",
    aliases=("shared",),
    agent_mode="shared",
    shared_agent_sandbox=True,
    default_env_pool_factor=2,
    supports_parallel_evaluations=True,
    required_runtime_capabilities=frozenset({SHARED_COMPUTE}),
    schedule=schedule,
)
