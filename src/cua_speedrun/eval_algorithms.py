"""Discovery and contracts for named, versioned evaluation algorithms."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TYPE_CHECKING

from cua_speedrun.evaluation_runtime import (
    EvaluationRuntime,
    Job,
    ResultRow,
)

if TYPE_CHECKING:
    from cua_speedrun.parallelism import ExecutionScale


Schedule = Callable[
    [EvaluationRuntime, Sequence[Job], "ExecutionScale"],
    list[ResultRow],
]


@dataclass(frozen=True)
class EvalAlgorithm:
    key: str
    label: str
    aliases: tuple[str, ...]
    agent_mode: str
    shared_agent_sandbox: bool
    default_env_pool_factor: int
    supports_parallel_evaluations: bool
    required_runtime_capabilities: frozenset[str]
    schedule: Schedule

    def supports(self, runtime_capabilities: Collection[str]) -> bool:
        return self.required_runtime_capabilities.issubset(runtime_capabilities)

    def run(
        self,
        runtime: EvaluationRuntime,
        jobs: Sequence[Job],
        scale: "ExecutionScale",
    ) -> list[ResultRow]:
        missing = self.required_runtime_capabilities - runtime.capabilities
        if missing:
            raise ValueError(
                f"runtime does not implement {self.key}: missing "
                f"{', '.join(sorted(missing))}"
            )
        return self.schedule(runtime, jobs, scale)

    def public_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "aliases": list(self.aliases),
            "supports_parallel_evaluations": self.supports_parallel_evaluations,
            "required_runtime_capabilities": sorted(
                self.required_runtime_capabilities
            ),
        }


PER_TASK_VLLM = "per-task-vllm@1"
SHARED_AGENT_VLLM = "shared-agent-vllm@2"


@lru_cache(maxsize=1)
def _registry() -> tuple[Mapping[str, EvalAlgorithm], Mapping[str, str]]:
    import cua_speedrun.algorithms as package

    algorithms: dict[str, EvalAlgorithm] = {}
    aliases: dict[str, str] = {}
    modules = sorted(
        pkgutil.iter_modules(package.__path__, package.__name__ + "."),
        key=lambda item: item.name,
    )
    for module_info in modules:
        if module_info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        module = importlib.import_module(module_info.name)
        algorithm = getattr(module, "ALGORITHM", None)
        if not isinstance(algorithm, EvalAlgorithm):
            raise TypeError(
                f"{module_info.name} must export ALGORITHM: EvalAlgorithm"
            )
        if algorithm.key in algorithms or algorithm.key in aliases:
            raise ValueError(f"duplicate evaluation algorithm {algorithm.key!r}")
        algorithms[algorithm.key] = algorithm
        for alias in algorithm.aliases:
            if alias in aliases or alias in algorithms:
                raise ValueError(f"duplicate evaluation algorithm alias {alias!r}")
            aliases[alias] = algorithm.key
    return algorithms, aliases


def resolve_eval_algorithm(key: str | None) -> EvalAlgorithm:
    algorithms, aliases = _registry()
    requested = (key or PER_TASK_VLLM).strip()
    normalized = aliases.get(requested, requested)
    try:
        return algorithms[normalized]
    except KeyError as exc:
        known = ", ".join(sorted((*algorithms, *aliases)))
        raise ValueError(f"unknown eval algorithm {key!r}; known: {known}") from exc


def list_eval_algorithms() -> list[EvalAlgorithm]:
    return list(_registry()[0].values())


def list_eval_algorithm_choices() -> list[str]:
    algorithms, aliases = _registry()
    return sorted((*algorithms, *aliases))
