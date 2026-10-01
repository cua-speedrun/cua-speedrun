from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from cua_speedrun.compute_runners import LocalComputeRunner, SlurmComputeRunner
from cua_speedrun.eval_algorithms import (
    PER_TASK_VLLM,
    SHARED_AGENT_VLLM,
    list_eval_algorithms,
    resolve_eval_algorithm,
)
from cua_speedrun.execution_placements import get_execution_topology
from cua_speedrun.local_runtime import (
    current_local_agent_runtime_contract,
    local_runtime_needs_resolution,
    requested_local_agent_runtime_contract,
    validate_local_agent_runtime_contract,
)
from cua_speedrun.remote.agent_runtime import (
    current_agent_runtime_contract,
    validate_agent_runtime_contract,
)
from cua_speedrun.parallelism import ExecutionScale
from cua_speedrun.remote.batching import job_batches
from cua_speedrun.remote.network_policy import api_domains_for_policy
from cua_speedrun.remote.snapshot_cache import compute_key
from cua_speedrun.task_jobs import build_task_seed_jobs


@dataclass
class Task:
    task_id: str


def test_explicit_task_seed_map_is_the_executed_job_list() -> None:
    tasks = [Task("alpha"), Task("beta")]
    jobs = build_task_seed_jobs(
        tasks,
        runs_per_task=2,
        seed_base=0,
        task_seeds={"alpha": [1_000_004, 1_000_009], "beta": [1_000_010, 1_000_012]},
    )
    assert [(task.task_id, seed) for task, seed in jobs] == [
        ("alpha", 1_000_004),
        ("alpha", 1_000_009),
        ("beta", 1_000_010),
        ("beta", 1_000_012),
    ]
    with pytest.raises(ValueError, match="needs 2 explicit seeds"):
        build_task_seed_jobs(tasks, 2, 0, {"alpha": [1], "beta": [2, 3]})


def test_algorithm_registry_dispatches_real_executor_objects() -> None:
    algorithms = {algorithm.key: algorithm for algorithm in list_eval_algorithms()}
    assert set(algorithms) == {
        PER_TASK_VLLM, SHARED_AGENT_VLLM, "shared-agent-no-preload@1"
    }
    assert all(
        callable(algorithm.schedule) for algorithm in algorithms.values()
    )


def test_local_topology_serves_both_algorithm_families() -> None:
    local = get_execution_topology("local", "local")
    assert set(local.eval_algorithms) == {
        PER_TASK_VLLM, SHARED_AGENT_VLLM, "shared-agent-no-preload@1"
    }

    # The shared algorithm keeps its rolling pool: environments are prepared
    # ahead at up to 2C, never gated into lock-step waves.
    shared = resolve_eval_algorithm(SHARED_AGENT_VLLM)
    assert shared.default_env_pool_factor == 2

    per_task = resolve_eval_algorithm(PER_TASK_VLLM)
    assert per_task.required_runtime_capabilities == {"isolated-instance"}


@pytest.mark.parametrize("parallel,agents", [(1, 1), (8, 1), (3, 4)])
def test_no_preload_reuses_shared_scheduler_with_one_c_pool(parallel, agents) -> None:
    shared = resolve_eval_algorithm("shared")
    no_preload = resolve_eval_algorithm("shared-no-preload")
    assert no_preload.key == "shared-agent-no-preload@1"
    assert no_preload.schedule is shared.schedule
    assert no_preload.agent_mode == "shared"
    assert no_preload.shared_agent_sandbox
    assert no_preload.supports_parallel_evaluations
    assert no_preload.required_runtime_capabilities == shared.required_runtime_capabilities
    scale = ExecutionScale.for_algorithm(no_preload.key, parallel, agents)
    assert scale.env_pool_size == scale.agent_concurrency == parallel * agents
    assert ExecutionScale.from_dict(scale.to_dict()) == scale
    assert ExecutionScale.for_algorithm("shared", parallel, agents).env_pool_size == 2 * scale.env_pool_size
    for placements in (("local", "local"), ("modal", "modal"), ("modal", "modal-native")):
        assert no_preload.key in get_execution_topology(*placements).eval_algorithms


def test_only_scheduled_runners_provide_isolated_instances() -> None:
    assert not hasattr(LocalComputeRunner, "start_instance_replica")
    assert hasattr(SlurmComputeRunner, "start_instance_replica")


def test_batched_pool_never_exceeds_declared_size() -> None:
    batches = list(job_batches(list(range(11)), 4))
    assert [len(batch) for batch in batches] == [4, 4, 3]
    assert [item for batch in batches for item in batch] == list(range(11))


def test_network_policy_is_explicit_and_normalized() -> None:
    assert api_domains_for_policy("gateway-only", []) == ()
    assert api_domains_for_policy("host-network", []) == ()
    assert api_domains_for_policy(
        "gateway+api", ["API.Example.com.", "api.example.com"]
    ) == ("api.example.com",)
    with pytest.raises(ValueError, match="cannot declare API domains"):
        api_domains_for_policy("gateway-only", ["api.example.com"])
    with pytest.raises(ValueError, match="bare domains"):
        api_domains_for_policy("gateway+api", ["https://api.example.com/v1"])


def test_server_environment_versions_the_snapshot_cache(tmp_path: Path) -> None:
    (tmp_path / "init.py").write_text("print('ready')")
    (tmp_path / "agent.py").write_text("pass")
    first = compute_key(tmp_path, "L4", [], {"VLLM_MODEL": "org/one"})
    second = compute_key(tmp_path, "L4", [], {"VLLM_MODEL": "org/two"})
    assert first != second


def test_agent_runtime_versions_the_snapshot_cache(tmp_path: Path) -> None:
    (tmp_path / "init.py").write_text("print('ready')")
    (tmp_path / "agent.py").write_text("pass")
    first = compute_key(
        tmp_path,
        "L40S",
        [],
        {},
        {"recipe": "modal-py311@1", "system_packages": ["ffmpeg==1"]},
    )
    second = compute_key(
        tmp_path,
        "L40S",
        [],
        {},
        {"recipe": "modal-py311@2", "system_packages": ["ffmpeg==2"]},
    )
    assert first != second


def test_agent_runtime_recipe_is_exact_and_worker_checked() -> None:
    runtime = current_agent_runtime_contract()
    validate_agent_runtime_contract(runtime)
    assert runtime["base_image"]["observed_python_version"] == "3.11.12"
    assert runtime["system_packages"] == ["ffmpeg=7:5.1.9-0+deb12u1"]
    assert all("==" in package for package in runtime["python_packages"])

    changed = {**runtime, "recipe": "unknown@1"}
    with pytest.raises(ValueError, match="not reproducible"):
        validate_agent_runtime_contract(changed)


def test_local_runtime_is_observed_by_the_claiming_worker() -> None:
    request = requested_local_agent_runtime_contract()
    assert local_runtime_needs_resolution(request) is True
    assert request["base_image"]["observed_python_version"] == "resolved-on-worker"

    observed = current_local_agent_runtime_contract(
        environment_runner="QemuNativeRunner"
    )
    assert local_runtime_needs_resolution(observed) is False
    validate_local_agent_runtime_contract(
        observed, environment_runner="QemuNativeRunner"
    )


def test_local_runtime_does_not_require_optional_ffmpeg() -> None:
    request = requested_local_agent_runtime_contract()
    assert request["recipe"] == "local-managed-python-qemu@1"
    assert "ffmpeg==required" not in request["system_packages"]
