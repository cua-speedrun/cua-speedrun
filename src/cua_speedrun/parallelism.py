"""One immutable execution-scale contract for every platform layer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from cua_speedrun.eval_algorithms import resolve_eval_algorithm


DEFAULT_PARALLEL_EVALUATIONS = 1
EXECUTION_SCALE_SCHEMA_VERSION = 1


def max_parallel_evaluations() -> int:
    """Operator quota for one run; ``-1`` means unlimited."""
    try:
        value = int(os.environ.get("CS_MAX_PARALLEL_EVALUATIONS", "-1"))
    except ValueError as exc:
        raise ValueError("CS_MAX_PARALLEL_EVALUATIONS must be an integer") from exc
    if value != -1 and value < 1:
        raise ValueError(
            "CS_MAX_PARALLEL_EVALUATIONS must be -1 (unlimited) or at least 1"
        )
    return value


def validate_parallel_evaluations(value: int) -> int:
    """Apply the installation's admission limit to a new evaluation."""
    parallel = int(value)
    maximum = max_parallel_evaluations()
    if parallel < 1:
        raise ValueError("parallel_evaluations must be at least 1")
    if maximum != -1 and parallel > maximum:
        raise ValueError(
            f"parallel_evaluations must be between 1 and {maximum}"
        )
    return parallel


@dataclass(frozen=True)
class ExecutionScale:
    """The complete operational scale selected before an evaluation runs.

    Only ``parallel_evaluations`` is user-owned. ``agents_per_evaluation``
    comes from the frozen track, and the environment pool comes from its eval
    algorithm. Derived totals are properties, so no layer can independently
    reinterpret the same run.
    """

    parallel_evaluations: int
    agents_per_evaluation: int
    environment_pool_size_per_evaluation: int
    SCHEMA_VERSION: ClassVar[int] = EXECUTION_SCALE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "parallel_evaluations",
            "agents_per_evaluation",
            "environment_pool_size_per_evaluation",
        ):
            try:
                normalized = int(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be an integer") from exc
            object.__setattr__(self, name, normalized)
        if self.parallel_evaluations < 1:
            raise ValueError("parallel_evaluations must be at least 1")
        if self.agents_per_evaluation < 1:
            raise ValueError("agents_per_evaluation must be at least 1")
        if (
            self.environment_pool_size_per_evaluation
            < self.agents_per_evaluation
        ):
            raise ValueError(
                "environment pool per evaluation cannot be smaller than "
                "agents_per_evaluation"
            )

    @property
    def agent_concurrency(self) -> int:
        return self.parallel_evaluations * self.agents_per_evaluation

    @property
    def environment_concurrency(self) -> int:
        return self.agent_concurrency

    @property
    def env_pool_size(self) -> int:
        return (
            self.parallel_evaluations
            * self.environment_pool_size_per_evaluation
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "parallel_evaluations": self.parallel_evaluations,
            "agents_per_evaluation": self.agents_per_evaluation,
            "agent_concurrency": self.agent_concurrency,
            "environment_concurrency": self.environment_concurrency,
            "env_pool_size": self.env_pool_size,
            "env_pool_size_per_evaluation": (
                self.environment_pool_size_per_evaluation
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionScale":
        version = int(data.get("schema_version", cls.SCHEMA_VERSION))
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"unsupported execution-scale schema {version}; "
                f"expected {cls.SCHEMA_VERSION}"
            )
        scale = cls(
            parallel_evaluations=int(data["parallel_evaluations"]),
            agents_per_evaluation=int(data["agents_per_evaluation"]),
            environment_pool_size_per_evaluation=int(
                data["env_pool_size_per_evaluation"]
            ),
        )
        canonical = scale.to_dict()
        for field in (
            "agent_concurrency",
            "environment_concurrency",
            "env_pool_size",
        ):
            if field in data and int(data[field]) != canonical[field]:
                raise ValueError(
                    f"execution-scale {field} does not match its inputs: "
                    f"stored {data[field]!r}, derived {canonical[field]}"
                )
        return scale

    @classmethod
    def from_value(
        cls, value: "ExecutionScale | Mapping[str, Any]"
    ) -> "ExecutionScale":
        return value if isinstance(value, cls) else cls.from_dict(value)

    @classmethod
    def for_plan(
        cls,
        plan: Any,
        parallel_evaluations: int = DEFAULT_PARALLEL_EVALUATIONS,
        *,
        enforce_admission_limit: bool = False,
    ) -> "ExecutionScale":
        parallel = (
            validate_parallel_evaluations(parallel_evaluations)
            if enforce_admission_limit
            else int(parallel_evaluations)
        )
        scale = cls(
            parallel_evaluations=parallel,
            agents_per_evaluation=int(
                plan.agents_per_evaluation or plan.concurrency
            ),
            environment_pool_size_per_evaluation=int(plan.env_pool_size),
        )
        return scale.validate_for_plan(plan)

    @classmethod
    def for_algorithm(
        cls,
        eval_algorithm: str,
        parallel_evaluations: int,
        agents_per_evaluation: int,
        *,
        enforce_admission_limit: bool = False,
    ) -> "ExecutionScale":
        """Resolve a direct CLI run before it has a frozen ``RunPlan``."""
        algorithm = resolve_eval_algorithm(eval_algorithm)
        parallel = (
            validate_parallel_evaluations(parallel_evaluations)
            if enforce_admission_limit
            else int(parallel_evaluations)
        )
        agents = int(agents_per_evaluation)
        scale = cls(
            parallel_evaluations=parallel,
            agents_per_evaluation=agents,
            environment_pool_size_per_evaluation=(
                agents * algorithm.default_env_pool_factor
            ),
        )
        if parallel != 1 and not algorithm.supports_parallel_evaluations:
            raise ValueError(
                "selected evaluation algorithm does not support parallel evaluations"
            )
        return scale

    def validate_for_plan(self, plan: Any) -> "ExecutionScale":
        expected_agents = int(
            plan.agents_per_evaluation or plan.concurrency
        )
        if self.agents_per_evaluation != expected_agents:
            raise ValueError(
                "execution scale conflicts with the frozen run plan: "
                f"{self.agents_per_evaluation} agents per evaluation != "
                f"{expected_agents}"
            )
        if self.environment_pool_size_per_evaluation != int(plan.env_pool_size):
            raise ValueError(
                "execution scale conflicts with the frozen run plan: "
                f"environment pool {self.environment_pool_size_per_evaluation} "
                f"!= {plan.env_pool_size}"
            )
        algorithm = resolve_eval_algorithm(plan.eval_algorithm)
        if (
            self.parallel_evaluations != 1
            and not algorithm.supports_parallel_evaluations
        ):
            raise ValueError(
                "selected evaluation algorithm does not support parallel evaluations"
            )
        return self
