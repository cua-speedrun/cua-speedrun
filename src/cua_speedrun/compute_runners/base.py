"""Provider-neutral model/agent compute-replica interface.

The evaluation algorithm owns sharding and concurrency.  A compute runner
only supplies one reusable replica for each shard and launches agents against
the environment URLs handed to it by the algorithm runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from cua_speedrun.gateway import Gateway
from cua_speedrun.evaluation_runtime import ComputeInfrastructureError
from cua_speedrun.submission import Submission


@dataclass(frozen=True)
class ComputeRunnerContext:
    submission: Submission
    run_dir: Path
    run_id: str
    base_runtime_env: Mapping[str, str]
    submission_environment_names: Sequence[str]
    init_cache_key: str
    python_packages: Sequence[str]
    managed_runtime: bool
    gpu: str | None
    track_name: str
    agents_per_evaluation: int
    events: Any
    evaluator_runtime_env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class ReplicaEnvironmentDriver(Protocol):
    """Environment lifecycle implemented inside one compute replica.

    Scheduled runners use this when ``local`` means local to the allocated
    worker rather than local to the coordinator process.
    """

    def prepare_environment(
        self,
        replica: Any,
        *,
        backend_name: str,
        task: Any,
        seed: int,
    ) -> Any: ...

    def run_prepared_environment(
        self,
        replica: Any,
        prepared: Any,
        events: Any,
    ) -> Mapping[str, Any]: ...

    def close_prepared_environments(
        self,
        replica: Any,
        prepared: Sequence[Any],
    ) -> None: ...


class ComputeRunner(Protocol):
    kind: str
    context: ComputeRunnerContext

    @property
    def identity(self) -> Mapping[str, Any]: ...

    @property
    def init_duration_sec(self) -> float: ...

    @property
    def environment_driver(self) -> ReplicaEnvironmentDriver | None: ...

    def start_replicas(self, shards: Sequence[Sequence[Any]]) -> Sequence[Any]: ...

    def gateway_hosts(self) -> tuple[str, str]:
        """Return ``(bind_host, advertised_host)`` for environment gateways."""
        ...

    def ensure_ready(self, replica: Any) -> None:
        """Wait untimed until a reusable replica can accept another agent."""
        ...

    def replace_replica(
        self,
        replica: Any,
        error: ComputeInfrastructureError,
    ) -> None:
        """Perform one idempotent replacement of a lost compute resource."""
        ...

    def start_agent(
        self,
        replica: Any,
        *,
        env_url: str,
        instruction: str,
        task_key: str,
        task_dir: Path,
        timeout_sec: float,
        append_output: bool = False,
    ) -> Any: ...

    def wait_for_gateway(self, replica: Any, agent: Any, gateway: Gateway) -> None: ...

    def stop_agent(self, replica: Any, agent: Any) -> AgentResult: ...

    def finish_replica(self, replica: Any) -> None: ...

    def close_replica(self, replica: Any) -> None: ...


__all__ = [
    "AgentResult",
    "ComputeRunner",
    "ComputeRunnerContext",
    "ReplicaEnvironmentDriver",
]
