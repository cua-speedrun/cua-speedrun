from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    ROOT / "benchmarks" / "osworld-offline" / "benchmark-source.yaml"
)
OSWORLD_ROOT = ROOT / "benchmark-assets" / "osworld"
_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_osworld_subset",
    ROOT / "scripts" / "build_osworld_subset.py",
)
assert _BUILDER_SPEC is not None and _BUILDER_SPEC.loader is not None
_BUILDER = importlib.util.module_from_spec(_BUILDER_SPEC)
_BUILDER_SPEC.loader.exec_module(_BUILDER)


def test_offline_source_preserves_task_membership() -> None:
    spec = _BUILDER._read_spec(SOURCE_PATH)
    selection = spec["selection"]
    tasks = spec["tasks"]

    assert spec["name"] == "osworld-offline"
    assert spec["version"] == "0.1"
    assert selection["selected_task_count"] == 295
    assert selection["task_ids_sha256"] == "c5964dbe0b19aa92c01992596c217d999bde292d1db34136a64a155b13b983a1"
    assert len(tasks) == 295
    assert len({task["id"] for task in tasks}) == 295


def test_offline_tasks_match_the_pinned_osworld_sources_in_canonical_order() -> None:
    spec = yaml.safe_load(SOURCE_PATH.read_text())
    selected = spec["tasks"]
    selected_ids = [task["id"] for task in selected]
    expected_hashes = {
        task["id"]: task["source_sha256"]
        for task in selected
    }

    source_manifest = yaml.safe_load((OSWORLD_ROOT / "manifest.yaml").read_text())
    source_ids: list[str] = []
    actual_hashes: dict[str, str] = {}
    for relative in source_manifest["tasks"]:
        task_dir = OSWORLD_ROOT / relative
        task = yaml.safe_load((task_dir / "task.yaml").read_text())
        task_id = task["task_id"]
        source_ids.append(task_id)
        source_json = (
            OSWORLD_ROOT
            / "environment"
            / "tasks"
            / task_dir.name
            / "source.json"
        )
        actual_hashes[task_id] = hashlib.sha256(source_json.read_bytes()).hexdigest()

    selected_set = set(selected_ids)
    assert selected_ids == [task_id for task_id in source_ids if task_id in selected_set]
    assert expected_hashes == {
        task_id: actual_hashes[task_id]
        for task_id in selected_ids
    }
