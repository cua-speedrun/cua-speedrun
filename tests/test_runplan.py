from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cua_speedrun.eval_algorithms import SHARED_AGENT_VLLM
from cua_speedrun.execution_placements import (
    get_execution_topology,
    list_execution_placements,
)
from cua_speedrun.local_runtime import current_local_agent_runtime_contract
from cua_speedrun.parallelism import ExecutionScale
from cua_speedrun.leaderboard import season_key
from cua_speedrun.runplan import RunPlan, benchmark_contract
from cua_speedrun.service.plans import attach_plan, plan_for_run


def _plan(**changes) -> RunPlan:
    values = {
        "track_name": "open-l4",
        "benchmark": {
            "name": "tiny",
            "version": "1",
            "content_hash": "b" * 64,
            "task_ids": ["one"],
        },
        "harness": {
            "cua_speedrun_version": "0.2.0",
            "measurement_hash": "h" * 64,
            "gym_anything_version": "1",
            "gym_anything_revision": "r1",
            "gym_anything_hash": "g" * 64,
        },
        "concurrency": 2,
        "env_pool_size": 2,
    }
    values.update(changes)
    return RunPlan(**values)


def _write_benchmark(root: Path, marker: str) -> Path:
    benchmark = root / "benchmark"
    task = benchmark / "tasks" / "one"
    env = root / "environment"
    task.mkdir(parents=True)
    env.mkdir()
    (benchmark / "manifest.yaml").write_text(
        'name: tiny\nversion: "1"\ntasks:\n  - tasks/one\n'
    )
    (task / "task.yaml").write_text(
        "task_id: one\n"
        "description: tiny task\n"
        "env:\n"
        "  kind: gym-anything\n"
        f"  env_dir: {env}\n"
        "  task_id: one\n"
    )
    (env / "env.json").write_text(marker)
    return benchmark


def test_run_plan_is_canonical_immutable_and_content_addressed() -> None:
    first = _plan()
    second = RunPlan.from_dict(first.to_dict())

    assert first.canonical_json == second.canonical_json
    assert first.contract_hash == second.contract_hash
    with pytest.raises(TypeError):
        first.benchmark["name"] = "changed"  # type: ignore[index]

    assert replace(first, success_bar=0.95).contract_hash != first.contract_hash
    assert replace(first, failure_costs_timeout=True).contract_hash != first.contract_hash
    assert replace(first, seed_policy="practice-fixed@1").contract_hash != first.contract_hash
    changed_runtime = {
        **first.resolved_agent_runtime(),
        "recipe": "different-runtime@1",
    }
    assert replace(first, agent_runtime=changed_runtime).contract_hash != first.contract_hash
    local_topology = get_execution_topology("local", "local")
    local = replace(
        first,
        track_name="open-l40s-shared",
        backend=local_topology.backend,
        gpu="L40S",
        eval_algorithm=SHARED_AGENT_VLLM,
        env_pool_size=4,
        execution_topology=local_topology.to_dict(),
        agent_runtime=current_local_agent_runtime_contract(
            environment_runner="QemuNativeRunner"
        ),
    )
    assert local.contract_hash != first.contract_hash
    assert local.season()["hardware"].startswith("L40S")
    assert first.to_dict()["agent_runtime"]["recipe"].endswith("@1")
    configured = replace(
        first,
        server_config={
            "environment": {"VLLM_MODEL": "org/model"},
            "extra_pip": ["flash-attn==1.0"],
            "region": "us-east",
        },
    )
    assert configured.contract_hash != first.contract_hash
    assert configured.server_environment()["VLLM_MODEL"] == "org/model"
    assert "CS_SERVER_CONFIG" in configured.server_environment()
    with pytest.raises(ValueError, match="cannot contain secret"):
        replace(first, server_config={"environment": {"MODEL_API_KEY": "secret"}})
    with pytest.raises(ValueError, match="must use exact == versions"):
        replace(first, server_config={"extra_pip": ["flash-attn>=1"]})


def test_shared_algorithm_has_a_hard_two_c_pool_bound() -> None:
    derived = _plan(
        eval_algorithm=SHARED_AGENT_VLLM,
        agents_per_evaluation=3,
    )
    assert derived.env_pool_size == 6


def test_compute_and_environment_placements_are_independent_inputs() -> None:
    assert {placement.key for placement in list_execution_placements()} == {
        "local", "modal", "modal-native"
    }
    local = get_execution_topology("local", "local")
    remote = get_execution_topology("modal", "modal")
    native = get_execution_topology("modal", "modal-native")
    assert local.compute.key == "local"
    assert local.environment.key == "local"
    assert remote.compute.key == "modal"
    assert remote.environment.key == "modal"
    assert native.compute.key == "modal"
    assert native.environment.key == "modal-native"
    with pytest.raises(ValueError, match="no registered executor yet"):
        get_execution_topology("local", "modal")
    with pytest.raises(ValueError, match="no registered executor yet"):
        get_execution_topology("modal", "local")
    with pytest.raises(ValueError, match="no registered executor yet"):
        get_execution_topology("modal-native", "modal-native")
    with pytest.raises(ValueError, match="no registered executor yet"):
        get_execution_topology("local", "modal-native")


def test_no_preload_pool_is_frozen_and_distinct_from_shared_default() -> None:
    shared = _plan(eval_algorithm=SHARED_AGENT_VLLM, agents_per_evaluation=3)
    no_preload = replace(shared, eval_algorithm="shared-no-preload")
    assert shared.env_pool_size == 6
    assert no_preload.env_pool_size == 3
    assert no_preload.eval_algorithm == "shared-agent-no-preload@1"
    assert no_preload.contract_hash != shared.contract_hash
    restored = RunPlan.from_dict(no_preload.to_dict())
    assert restored.canonical_json == no_preload.canonical_json
    scale = ExecutionScale.for_plan(restored, parallel_evaluations=4)
    assert scale.env_pool_size == scale.agent_concurrency == 12
    with pytest.raises(ValueError, match="environment pool"):
        ExecutionScale.for_plan(shared, parallel_evaluations=4).validate_for_plan(restored)


def test_benchmark_contract_hashes_environment_payload(tmp_path: Path) -> None:
    benchmark = _write_benchmark(tmp_path, '{"marker": 1}')
    before = benchmark_contract(benchmark)
    (tmp_path / "environment" / "env.json").write_text('{"marker": 2}')
    after = benchmark_contract(benchmark)
    assert before["content_hash"] != after["content_hash"]


def test_season_key_versions_the_execution_contract() -> None:
    key = season_key(_plan().season())
    assert "track=open-l4" in key
    assert "algorithm=per-task-vllm@1" in key
    assert f"contract={_plan().contract_hash[:16]}" in key


def test_run_uses_stored_plan_after_catalog_track_changes() -> None:
    run = SimpleNamespace(id=42, execution_plan={}, plan_hash=None)
    original = _plan(gpu="L4", success_bar=0.9)
    attach_plan(run, original)
    mutated_track = SimpleNamespace(gpu="H200", success_bar=1.0)

    recovered = plan_for_run(run, mutated_track, benchmark=None)

    assert recovered.contract_hash == original.contract_hash
    assert recovered.gpu == "L4"
    assert recovered.success_bar == 0.9
