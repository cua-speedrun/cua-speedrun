"""Swappable compute replicas used by local evaluation topologies."""

from .config import (
    RunnerSelection,
    local_runtime_overrides,
    resolve_runner_selection,
)
from .local import LocalComputeRunner
from .slurm import SlurmComputeRunner

__all__ = [
    "LocalComputeRunner",
    "RunnerSelection",
    "SlurmComputeRunner",
    "local_runtime_overrides",
    "resolve_runner_selection",
]
