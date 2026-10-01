from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    ROOT
    / "benchmarks"
    / "osworld-50"
    / "benchmark-source.yaml"
)
OSWORLD_ROOT = ROOT / "benchmark-assets" / "osworld"
_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_osworld_subset",
    ROOT / "scripts" / "build_osworld_subset.py",
)
assert _BUILDER_SPEC is not None and _BUILDER_SPEC.loader is not None
_BUILDER = importlib.util.module_from_spec(_BUILDER_SPEC)
_BUILDER_SPEC.loader.exec_module(_BUILDER)


def test_osworld_50_preserves_task_membership() -> None:
    spec = _BUILDER._read_spec(SOURCE_PATH)
    selection = spec["selection"]
    tasks = spec["tasks"]

    assert spec["name"] == "osworld-50"
    assert spec["version"] == "0.1"
    assert selection["selected_task_count"] == 50
    assert selection["ordering"] == "canonical source benchmark manifest"
    assert selection["task_ids_sha256"] == (
        "8a577f2e475111d89c9840783fa431cd6f31655f071bdcc68fbc361a2ce2d22d"
    )
    assert len(tasks) == 50
    assert len({task["id"] for task in tasks}) == 50


def test_osworld_50_tasks_match_pinned_sources_in_canonical_order() -> None:
    spec = yaml.safe_load(SOURCE_PATH.read_text())
    selected = spec["tasks"]
    selected_ids = [task["id"] for task in selected]
    expected_hashes = {task["id"]: task["source_sha256"] for task in selected}

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
        task_id: actual_hashes[task_id] for task_id in selected_ids
    }


def test_osworld_50_materializes_all_tasks_and_setup_patch(tmp_path: Path) -> None:
    output = tmp_path / "osworld-50"
    _BUILDER.materialize(SOURCE_PATH, output)

    manifest = yaml.safe_load((output / "manifest.yaml").read_text())
    assert manifest["name"] == "osworld-50"
    assert manifest["version"] == "0.1"
    assert len(manifest["tasks"]) == 50
    assert len(list((output / "tasks").glob("*/task.yaml"))) == 50
    for name in ("osworld_setup.py", "osworld_verifier.py"):
        assert (output / "environment" / "tasks" / "_shared" / name).read_bytes() == (
            ROOT / "scripts" / "osworld_shared" / name
        ).read_bytes()

    patched_task = "multi_apps__48d05431-6cd5-4e76-82eb-12b60d823f7d"
    assert (
        output / "environment" / "tasks" / patched_task / "setup-patch.json"
    ).read_bytes() == (
        ROOT / "scripts" / "osworld_task_patches" / f"{patched_task}.json"
    ).read_bytes()
