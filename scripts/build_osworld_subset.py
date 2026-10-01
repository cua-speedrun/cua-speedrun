#!/usr/bin/env python3
"""Materialize an ordered OSWorld subset from the installed full benchmark."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def _repository_root(source_path: Path) -> Path:
    for candidate in source_path.parents:
        if (
            (candidate / "catalog" / "tracks.yaml").is_file()
            and (candidate / "benchmarks").is_dir()
            and (candidate / "scripts").is_dir()
        ):
            return candidate
    raise ValueError(f"{source_path}: cannot locate the cua-speedrun resource root")


def _read_spec(source_path: Path) -> dict[str, Any]:
    spec = yaml.safe_load(source_path.read_text()) or {}
    source = spec.get("source_benchmark")
    selection = spec.get("selection")
    tasks = spec.get("tasks")
    if not isinstance(source, dict) or not isinstance(selection, dict):
        raise ValueError(f"{source_path}: source_benchmark and selection are required")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{source_path}: tasks must be a non-empty list")
    ids = [str(task.get("id", "")) for task in tasks if isinstance(task, dict)]
    if len(ids) != len(tasks) or any(not task_id for task_id in ids):
        raise ValueError(f"{source_path}: every task needs a non-empty id")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source_path}: duplicate task id")
    actual = hashlib.sha256(
        "".join(f"{task_id}\n" for task_id in ids).encode()
    ).hexdigest()
    expected = str(selection.get("task_ids_sha256", ""))
    if actual != expected:
        raise ValueError(
            f"{source_path}: ordered task digest mismatch: expected {expected}, got {actual}"
        )
    return spec


def materialize(source_path: Path, out: Path) -> Path:
    """Build a self-contained benchmark package containing the selected tasks."""
    source_path = Path(source_path).resolve()
    out = Path(out).resolve()
    spec = _read_spec(source_path)
    root = _repository_root(source_path)
    source_spec = spec["source_benchmark"]
    source_root = (root / str(source_spec["path"])).resolve()
    if not source_root.is_relative_to(root.resolve()):
        raise ValueError(f"{source_path}: source benchmark escapes the resource root")

    source_manifest_path = source_root / "manifest.yaml"
    source_manifest = yaml.safe_load(source_manifest_path.read_text()) or {}
    expected_identity = (
        str(source_spec["name"]),
        str(source_spec["version"]),
        str(source_spec["commit"]),
    )
    actual_identity = (
        str(source_manifest.get("name", "")),
        str(source_manifest.get("version", "")),
        str(source_manifest.get("source_commit", "")),
    )
    if actual_identity != expected_identity:
        raise ValueError(
            f"{source_manifest_path}: expected source {expected_identity}, got {actual_identity}"
        )

    task_dirs: dict[str, Path] = {}
    for task_yaml in sorted((source_root / "tasks").glob("*/task.yaml")):
        task = yaml.safe_load(task_yaml.read_text()) or {}
        task_id = str(task.get("task_id", ""))
        if task_id in task_dirs:
            raise ValueError(f"{source_manifest_path}: duplicate task id {task_id!r}")
        task_dirs[task_id] = task_yaml.parent

    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    environment = out / "environment"
    environment.mkdir()
    shutil.copy2(source_root / "environment" / "env.json", environment / "env.json")
    shutil.copytree(
        source_root / "environment" / "tasks" / "_shared",
        environment / "tasks" / "_shared",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    for name in ("osworld_setup.py", "osworld_verifier.py"):
        shutil.copy2(
            root / "scripts" / "osworld_shared" / name,
            environment / "tasks" / "_shared" / name,
        )

    manifest_tasks: list[str] = []
    for selected in spec["tasks"]:
        task_id = str(selected["id"])
        task_dir = task_dirs.get(task_id)
        if task_dir is None:
            raise ValueError(
                f"{source_manifest_path}: selected task is missing: {task_id}"
            )
        name = task_dir.name
        env_task = source_root / "environment" / "tasks" / name
        source_json = env_task / "source.json"
        actual_hash = hashlib.sha256(source_json.read_bytes()).hexdigest()
        expected_hash = str(selected.get("source_sha256", ""))
        if actual_hash != expected_hash:
            raise ValueError(
                f"{source_json}: expected SHA-256 {expected_hash}, got {actual_hash}"
            )

        copied_task = out / "tasks" / name
        shutil.copytree(task_dir, copied_task)
        task_yaml = copied_task / "task.yaml"
        task = yaml.safe_load(task_yaml.read_text()) or {}
        task["env"]["env_dir"] = "${BENCHMARK_DIR}/environment"
        task_yaml.write_text(yaml.safe_dump(task, sort_keys=False, allow_unicode=True))
        copied_env_task = environment / "tasks" / name
        shutil.copytree(
            env_task,
            copied_env_task,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
        patch = root / "scripts/osworld_task_patches" / f"{name}.json"
        if patch.is_file():
            shutil.copy2(patch, copied_env_task / "setup-patch.json")
        manifest_tasks.append(f"tasks/{name}")

    manifest = {
        "name": str(spec["name"]),
        "version": str(spec["version"]),
        "description": str(spec.get("description", "")),
        "source_benchmark": f"{actual_identity[0]}@{actual_identity[1]}",
        "source_commit": actual_identity[2],
        "selection": spec["selection"],
        "tasks": manifest_tasks,
    }
    (out / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
    )
    print(f"materialized {len(manifest_tasks)} OSWorld tasks in {out}")
    return out


__all__ = ["materialize"]
