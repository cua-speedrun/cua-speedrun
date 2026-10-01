"""Provider-neutral operations available to evaluation algorithms.

Algorithms own scheduling.  Runtime implementations own the mechanics of
creating resources on a particular execution topology.  Handles are opaque on
purpose: an algorithm can decide how long a compute replica or environment
lives without knowing whether that resource is a process, VM, or sandbox.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Callable, Protocol, TypeVar


SHARED_COMPUTE = "shared-compute"
ISOLATED_INSTANCE = "isolated-instance"
INDEPENDENT_ENVIRONMENT_PREPARATION = "independent-environment-preparation"

# The local topology can serve both algorithm families: the in-process
# runner provides shared compute replicas, and scheduled runners additionally
# provide one isolated allocation per task instance. Which of the two the
# selected runner actually implements is enforced by the executor at run time.
LOCAL_RUNTIME_CAPABILITIES = frozenset({ISOLATED_INSTANCE, SHARED_COMPUTE})
MODAL_RUNTIME_CAPABILITIES = frozenset({
    ISOLATED_INSTANCE,
    SHARED_COMPUTE,
    INDEPENDENT_ENVIRONMENT_PREPARATION,
})

Job = tuple[Any, int]
ResultRow = dict[str, Any]
Prepared = TypeVar("Prepared")

ENVIRONMENT_PREPARE_MAX_ATTEMPTS = 5
INSTANCE_INFRASTRUCTURE_MAX_ATTEMPTS = 5


class InstanceInfrastructureError(RuntimeError):
    """Evaluator-owned work for one task instance failed.

    Evaluation algorithms retry this class with a fresh single-use
    environment. A submitted agent exiting on a healthy evaluator is instead
    ``AgentExecutionError`` and must not enter this retry path.
    """


class AgentExecutionError(RuntimeError):
    """The submitted agent exited without completing the task.

    This is a scored task failure, not evaluator infrastructure, and therefore
    must never enter an infrastructure retry loop.
    """

    def __init__(self, message: str, *, returncode: int | None = None) -> None:
        super().__init__(message)
        self.returncode = returncode


class ComputeInfrastructureError(InstanceInfrastructureError):
    """A replaceable shared compute resource disappeared.

    ``resource_id`` lets a provider make replacement idempotent when several
    concurrent instances observe the same lost replica.
    """

    def __init__(self, message: str, *, resource_id: str | None = None) -> None:
        super().__init__(message)
        self.resource_id = resource_id


def prepare_with_backoff(
    operation: Callable[[], Prepared],
    *,
    on_retry: Callable[[int, int, float, Exception], None],
) -> Prepared:
    """Prepare an environment up to five times with exponential backoff.

    Preparation is outside the measured task path, so retrying infrastructure
    failures cannot improve an agent's measured time or retry an agent failure.
    """
    for attempt in range(1, ENVIRONMENT_PREPARE_MAX_ATTEMPTS + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == ENVIRONMENT_PREPARE_MAX_ATTEMPTS:
                raise
            delay_sec = float(2 ** (attempt - 1))
            on_retry(
                attempt,
                ENVIRONMENT_PREPARE_MAX_ATTEMPTS,
                delay_sec,
                exc,
            )
            time.sleep(delay_sec)
    raise AssertionError("environment preparation retry loop exhausted")


class EvaluationRuntime(Protocol):
    """The stable runner surface that scheduling functions may use."""

    capabilities: frozenset[str]

    def run_isolated(self, job: Job) -> ResultRow:
        """Run one job with an isolated compute/agent resource."""

    def start_compute_replicas(
        self,
        shards: Sequence[Sequence[Job]],
    ) -> Sequence[Any]:
        """Start one reusable compute resource for every non-empty shard."""

    def ensure_compute_ready(self, replica: Any) -> None:
        """Wait for one compute replica or report replaceable infrastructure."""

    def recover_instance_infrastructure(
        self,
        replica: Any,
        error: InstanceInfrastructureError,
    ) -> None:
        """Replace the failed provider resource named by ``error``.

        The algorithm owns retry count and backoff. The runtime performs one
        provider-specific recovery operation and must make concurrent calls
        for the same failed resource idempotent.
        """

    def compute_infrastructure_retry(
        self,
        replica: Any,
        attempt: int,
        max_attempts: int,
        delay_sec: float,
        error: Exception,
    ) -> None:
        """Report a compute replica that will be replaced and retried."""

    def compute_infrastructure_failed(
        self,
        replica: Any,
        attempts: int,
        error: Exception,
    ) -> None:
        """Report a compute replica that exhausted its recovery budget."""

    def prepare_environment(
        self,
        replica: Any,
        job: Job,
        batch_number: int,
    ) -> Any:
        """Prepare one single-use environment without arming its clock."""

    def environment_prepare_retry(
        self,
        replica: Any,
        job: Job,
        batch_number: int,
        attempt: int,
        max_attempts: int,
        delay_sec: float,
        error: Exception,
    ) -> None:
        """Preserve failed-attempt evidence and report an upcoming retry."""

    def environment_prepare_failed(
        self,
        replica: Any,
        job: Job,
        batch_number: int,
        attempts: int,
        error: Exception,
    ) -> None:
        """Report an environment that exhausted its infrastructure attempts."""

    def instance_infrastructure_retry(
        self,
        replica: Any,
        job: Job,
        attempt: int,
        max_attempts: int,
        delay_sec: float,
        error: Exception,
    ) -> None:
        """Report an instance-scoped infra failure before a fresh retry."""

    def instance_infrastructure_failed(
        self,
        replica: Any,
        job: Job,
        attempts: int,
        error: Exception,
    ) -> None:
        """Report an instance that exhausted its infra retry budget."""

    def begin_environment_batch(
        self,
        replica: Any,
        jobs: Sequence[Job],
        batch_number: int,
    ) -> None:
        """Open provider-side bookkeeping for an environment batch."""

    def bind_compute(
        self,
        replica: Any,
        prepared: Sequence[Any],
        batch_number: int,
    ) -> None:
        """Make a ready environment batch reachable from its compute replica."""

    def run_prepared(
        self,
        replica: Any,
        prepared: Any,
        batch_number: int,
    ) -> ResultRow:
        """Run, verify, and persist one already-prepared instance."""

    def close_environment_batch(
        self,
        replica: Any,
        prepared: Sequence[Any],
        batch_number: int,
    ) -> None:
        """Destroy every environment in a completed or failed batch."""

    def finish_compute(self, replica: Any) -> None:
        """Persist replica-level evidence after all of its jobs finish."""

    def close_compute_replica(self, replica: Any) -> None:
        """Destroy one compute resource, including a partial start."""
