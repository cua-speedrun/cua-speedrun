from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from gym_anything.registry import load_environment_task_splits


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "benchmarks/cua-world-offline/benchmark-source.yaml"
CUA_WORLD_ROOT = ROOT / "third_party/gym-anything/benchmarks/cua_world"
_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "import_cua_world_long", ROOT / "scripts/import_cua_world_long.py"
)
assert _BUILDER_SPEC is not None and _BUILDER_SPEC.loader is not None
_BUILDER = importlib.util.module_from_spec(_BUILDER_SPEC)
_BUILDER_SPEC.loader.exec_module(_BUILDER)


def test_offline_source_matches_the_pinned_registry_and_descriptions() -> None:
    source = yaml.safe_load(SOURCE_PATH.read_text())
    selected = [
        (entry["env_name"], entry["task_name"])
        for entry in source["tasks"]
    ]
    splits = load_environment_task_splits(CUA_WORLD_ROOT)
    canonical = [
        (env_name, task_name)
        for env_name, env_splits in sorted(splits.items())
        for task_name in env_splits.get("long_horizon", [])
    ]

    assert source["source_benchmark"]["commit"] == _BUILDER.GYM_ANYTHING_COMMIT
    assert source["selection"]["selected_task_count"] == 143
    assert source["selection"]["platform_counts"] == {'windows': 0, 'linux': 143, 'android': 0}
    assert selected == [entry for entry in canonical if entry in set(selected)]
    assert len({entry["id"] for entry in source["tasks"]}) == 143

    selected_environments = {env_name for env_name, _ in selected}
    actual_fetch = sorted(
        path.parent.parent.name
        for path in (CUA_WORLD_ROOT / "environments").glob(
            "*/scripts/fetch_data.sh"
        )
        if path.parent.parent.name in selected_environments
    )
    actual_manual = sorted(
        path.parent.name
        for path in (CUA_WORLD_ROOT / "environments").glob(
            "*/MANUAL_DOWNLOAD.md"
        )
        if path.parent.name in selected_environments
    )
    assert source["asset_setup"]["fetch_script_environments"] == actual_fetch
    assert source["asset_setup"]["manual_download_environments"] == actual_manual

    for entry in source["tasks"]:
        task_path = (
            CUA_WORLD_ROOT
            / "environments"
            / entry["env_name"]
            / "tasks"
            / entry["task_name"]
            / "task.json"
        )
        task = json.loads(task_path.read_text())
        assert entry["description"] == str(task["description"]).strip()


def test_long_task_protocol_uses_uniform_limits_and_default_cache() -> None:
    source = yaml.safe_load(SOURCE_PATH.read_text())
    entry = source["tasks"][0]
    source_task = json.loads(
        (
            CUA_WORLD_ROOT
            / "environments"
            / entry["env_name"]
            / "tasks"
            / entry["task_name"]
            / "task.json"
        ).read_text()
    )
    task = _BUILDER._task_yaml(
        entry,
        benchmark_name="cua-world-offline",
        fetch_script_environments=set(
            source["asset_setup"]["fetch_script_environments"]
        ),
        manual_download_environments=set(
            source["asset_setup"]["manual_download_environments"]
        ),
    )

    assert task["timeout_sec"] == 21600
    assert task["description"] == entry["description"]
    assert "max_steps" not in task
    assert "verifier_timeout_sec" not in task
    assert task["env"]["use_cache"] is True
    assert task["env"]["cache_level"] == "default"
    assert task["env"]["max_steps"] == 500
    assert task["env"]["prepare_entrypoint"] == (
        "cua_speedrun.envs.cua_world_runtime:prepare"
    )
    assert "defer_episode_limits" not in task["env"]
    assert "verifier" not in task["env"]
    assert "fail_on_error" not in source_task["hooks"]


def test_materializer_rewrites_only_its_environment_root(tmp_path: Path) -> None:
    env_dir = tmp_path / "example_env"
    env_dir.mkdir()
    (env_dir / "env.json").write_text(json.dumps({
        "mounts": [{
            "source": "benchmarks/cua_world/environments/example_env/scripts"
        }],
        "command": "benchmarks/cua_world/environments/other_env/start.sh",
    }))

    _BUILDER._make_environment_paths_portable(env_dir, "example_env")

    config = json.loads((env_dir / "env.json").read_text())
    assert config["mounts"][0]["source"] == "scripts"
    assert config["command"] == (
        "benchmarks/cua_world/environments/other_env/start.sh"
    )
