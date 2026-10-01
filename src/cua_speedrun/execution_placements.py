"""Independent placement contracts for model/agent compute and env VMs.

Hardware requirements belong to tracks. Placement only says whether each
execution plane runs on the evaluator host or through a remote provider.
Supported combinations are registered as execution topologies; adding a new
engine does not add a track or change scoring code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cua_speedrun.eval_algorithms import list_eval_algorithms
from cua_speedrun.evaluation_runtime import (
    LOCAL_RUNTIME_CAPABILITIES,
    MODAL_RUNTIME_CAPABILITIES,
)


LOCAL_PLACEMENT = "local"
MODAL_PLACEMENT = "modal"
MODAL_NATIVE_PLACEMENT = "modal-native"


@dataclass(frozen=True)
class ExecutionPlacement:
    key: str
    mode: str
    provider: str | None
    requires_user_credentials: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionTopology:
    key: str
    backend: str
    compute: ExecutionPlacement
    environment: ExecutionPlacement
    environment_backend: str
    runtime_capabilities: frozenset[str]

    @property
    def requires_user_credentials(self) -> bool:
        return (
            self.compute.requires_user_credentials
            or self.environment.requires_user_credentials
        )

    @property
    def eval_algorithms(self) -> tuple[str, ...]:
        return tuple(
            algorithm.key
            for algorithm in list_eval_algorithms()
            if algorithm.supports(self.runtime_capabilities)
        )

    def contract_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "backend": self.backend,
            "compute": self.compute.to_dict(),
            "environment": self.environment.to_dict(),
            "environment_backend": self.environment_backend,
            "requires_user_credentials": self.requires_user_credentials,
        }

    def to_dict(self) -> dict[str, Any]:
        """Public discovery data; mutable capabilities stay out of contracts."""
        return {
            **self.contract_dict(),
            "runtime_capabilities": sorted(self.runtime_capabilities),
            "eval_algorithms": list(self.eval_algorithms),
        }


_PLACEMENTS = {
    LOCAL_PLACEMENT: ExecutionPlacement(
        key=LOCAL_PLACEMENT,
        mode="local",
        provider=None,
        requires_user_credentials=False,
    ),
    MODAL_PLACEMENT: ExecutionPlacement(
        key=MODAL_PLACEMENT,
        mode="remote",
        provider="modal",
        requires_user_credentials=True,
    ),
    # The OSWorld desktop booted directly on a Modal sandbox kernel, without
    # KVM or nested QEMU (Modal removed /dev/kvm from its VM sandboxes).
    MODAL_NATIVE_PLACEMENT: ExecutionPlacement(
        key=MODAL_NATIVE_PLACEMENT,
        mode="remote",
        provider="modal",
        requires_user_credentials=True,
    ),
}


_TOPOLOGIES = {
    (LOCAL_PLACEMENT, LOCAL_PLACEMENT): ExecutionTopology(
        key="local",
        backend="local",
        compute=_PLACEMENTS[LOCAL_PLACEMENT],
        environment=_PLACEMENTS[LOCAL_PLACEMENT],
        environment_backend="gym-anything-local",
        runtime_capabilities=LOCAL_RUNTIME_CAPABILITIES,
    ),
    (MODAL_PLACEMENT, MODAL_PLACEMENT): ExecutionTopology(
        key="modal-remote",
        backend="modal-remote",
        compute=_PLACEMENTS[MODAL_PLACEMENT],
        environment=_PLACEMENTS[MODAL_PLACEMENT],
        environment_backend="gym-anything-modal",
        runtime_capabilities=MODAL_RUNTIME_CAPABILITIES,
    ),
    (MODAL_PLACEMENT, MODAL_NATIVE_PLACEMENT): ExecutionTopology(
        key="modal-native",
        backend="modal-native",
        compute=_PLACEMENTS[MODAL_PLACEMENT],
        environment=_PLACEMENTS[MODAL_NATIVE_PLACEMENT],
        environment_backend="modal-native",
        runtime_capabilities=MODAL_RUNTIME_CAPABILITIES,
    ),
}


def list_execution_placements() -> list[ExecutionPlacement]:
    return list(_PLACEMENTS.values())


def get_execution_placement(key: str) -> ExecutionPlacement:
    try:
        return _PLACEMENTS[key.strip().lower()]
    except (AttributeError, KeyError) as exc:
        known = ", ".join(sorted(_PLACEMENTS))
        raise ValueError(f"unknown execution placement {key!r}; known: {known}") from exc


def list_execution_topologies() -> list[ExecutionTopology]:
    return list(_TOPOLOGIES.values())


def get_execution_topology(
    compute_key: str, environment_key: str
) -> ExecutionTopology:
    compute = get_execution_placement(compute_key)
    environment = get_execution_placement(environment_key)
    try:
        return _TOPOLOGIES[(compute.key, environment.key)]
    except KeyError as exc:
        raise ValueError(
            "that placement combination has no registered executor yet: "
            f"model/agent={compute.key}, environment-vm={environment.key}"
        ) from exc


def execution_topology_for_backend(backend: str) -> ExecutionTopology:
    normalized = "local" if backend == "gym-anything-local" else backend
    for topology in _TOPOLOGIES.values():
        if topology.backend == normalized:
            return topology
    raise ValueError(f"no execution topology owns backend {backend!r}")
