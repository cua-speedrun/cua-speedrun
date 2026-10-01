"""Resolve mutable catalog rows into immutable core RunPlans."""

from __future__ import annotations

from pathlib import Path

from cua_speedrun.execution_placements import (
    LOCAL_PLACEMENT,
    MODAL_PLACEMENT,
    get_execution_topology,
)
from cua_speedrun.local_runtime import requested_local_agent_runtime_contract
from cua_speedrun.parallelism import ExecutionScale
from cua_speedrun.remote.agent_runtime import current_agent_runtime_contract
from cua_speedrun.runplan import RunPlan, build_run_plan


def plan_for_catalog_rows(
    track,
    benchmark,
    compute_placement: str | None = None,
    environment_placement: str | None = None,
    allocate_gpu: bool = True,
    gpu: str | None = None,
    eval_algorithm: str | None = None,
) -> RunPlan:
    if gpu is not None:
        gpu = gpu.strip()
        if not gpu:
            raise ValueError("GPU type must not be empty")
        if not allocate_gpu:
            raise ValueError("GPU type cannot be combined with disabled GPU allocation")
    topology = get_execution_topology(
        compute_placement or MODAL_PLACEMENT,
        environment_placement or MODAL_PLACEMENT,
    )
    # Submission code may use a local model, an external API, both, or neither.
    # Networking is therefore an execution capability, not an agent class or
    # a reason to create a provider-specific track.
    network_policy = "host-network"
    agent_runtime = (
        requested_local_agent_runtime_contract()
        if topology.compute.key == LOCAL_PLACEMENT
        else current_agent_runtime_contract()
    )
    agents_per_evaluation = max(1, int(track.agents_per_evaluation or 2))
    return build_run_plan(
        track_name=track.name,
        benchmark_dir=Path(benchmark.path),
        gpu=(gpu if gpu is not None else track.gpu) if allocate_gpu else None,
        network_policy=network_policy,
        api_domain_allowlist=[],
        reference_only=bool(track.reference_only),
        ranking_rule=track.ranking_rule or "time-at-success-bar@1",
        eval_algorithm=eval_algorithm or track.eval_algorithm or "per-task-vllm@1",
        agents_per_evaluation=agents_per_evaluation,
        runs_per_task=max(1, int(track.runs_per_task or 1)),
        success_bar=float(track.success_bar if track.success_bar is not None else 0.9),
        failure_costs_timeout=bool(track.failure_costs_timeout),
        seed_policy=track.seed_policy or "scored-without-replacement@1",
        server_config=track.server_config or {},
        agent_runtime=agent_runtime,
        execution_topology=topology.contract_dict(),
        backend=topology.backend,
    )


def attach_plan(run, plan: RunPlan) -> None:
    run.execution_plan = plan.to_dict()
    run.topology_key = plan.resolved_execution_topology()["key"]
    run.plan_hash = plan.contract_hash


def attach_scale(run, scale: ExecutionScale) -> None:
    """Freeze one canonical scale document on a queued run."""
    run.execution_scale = scale.to_dict()
    # Transitional mirror for installations and queued rows created before
    # the versioned document existed. Runtime code reads the document first.
    run.parallel_evaluations = scale.parallel_evaluations


def scale_for_run(run, plan: RunPlan) -> ExecutionScale:
    """Load the canonical scale, upgrading a pre-ExecutionScale run once."""
    if run.execution_scale:
        return ExecutionScale.from_dict(run.execution_scale).validate_for_plan(
            plan
        )
    scale = ExecutionScale.for_plan(
        plan, run.parallel_evaluations or 1
    )
    attach_scale(run, scale)
    return scale


def plan_for_run(run, track, benchmark) -> RunPlan:
    if run.execution_plan:
        plan = RunPlan.from_dict(run.execution_plan)
        if run.plan_hash and run.plan_hash != plan.contract_hash:
            raise ValueError(
                f"run {run.id} execution plan hash mismatch: "
                f"stored {run.plan_hash}, computed {plan.contract_hash}"
            )
        return plan
    plan = plan_for_catalog_rows(track, benchmark)
    attach_plan(run, plan)
    return plan
