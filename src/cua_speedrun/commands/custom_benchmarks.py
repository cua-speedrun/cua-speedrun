"""Validate and register immutable copies of user-authored benchmarks."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import shutil
import tempfile

import yaml

from cua_speedrun.benchmark_sources import benchmark_catalog_paths
from cua_speedrun.runplan import benchmark_contract
from cua_speedrun.specs import Benchmark

from .paths import InstallationPaths


def validate_benchmark(benchmark: Benchmark) -> None:
    root = benchmark.benchmark_dir.resolve()
    if not benchmark.tasks:
        raise ValueError("a benchmark must contain at least one task")
    if not benchmark.name or "@" in benchmark.name or not benchmark.version:
        raise ValueError("provide a benchmark name without @ and a nonempty version")
    for task in benchmark.tasks:
        if not task.task_dir.resolve().is_relative_to(root):
            raise ValueError(f"{task.task_id}: task folders must be inside the benchmark")
        if task.timeout_sec <= 0 or task.grace_sec < 0:
            raise ValueError(f"{task.task_id}: invalid timeout or grace period")
        if task.generator:
            generator = (task.task_dir / task.generator).resolve()
            if not generator.is_relative_to(root) or not generator.is_file():
                raise ValueError(f"{task.task_id}: generator must be a file inside the benchmark")
            compile(generator.read_bytes(), str(generator), "exec")
        if "env_dir" in task.env:
            environment = Path(task.env["env_dir"]).resolve()
            if not environment.is_relative_to(root) or not environment.is_dir():
                raise ValueError(f"{task.task_id}: env_dir must use ${{BENCHMARK_DIR}} and point inside the benchmark")
            if not any((environment / name).is_file() for name in ("env.json", "env.yaml", "env.yml")):
                raise ValueError(f"{task.task_id}: environment definition is missing")
            raw = yaml.safe_load((task.task_dir / "task.yaml").read_text())
            if "${BENCHMARK_DIR}" not in str(raw["env"]["env_dir"]):
                raise ValueError(f"{task.task_id}: use ${{BENCHMARK_DIR}} in env_dir to keep the benchmark portable")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"benchmark folders must be self-contained; replace symlink {path}")


def register_benchmark(benchmark: Benchmark, paths: InstallationPaths) -> Benchmark:
    validate_benchmark(benchmark)
    digest = benchmark_contract(benchmark.benchmark_dir)["content_hash"]
    name = "local-" + digest
    root = paths.resource_root
    target = root / "benchmarks" / name
    registry = root / "catalog/local-benchmarks.yaml"
    with (root / "catalog/local-benchmarks.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for existing in benchmark_catalog_paths(root):
            definition = existing / "benchmark-source.yaml"
            if not definition.is_file():
                definition = existing / "manifest.yaml"
            data = yaml.safe_load(definition.read_text())
            if (str(data["name"]), str(data["version"])) == (benchmark.name, benchmark.version):
                if existing == target:
                    return Benchmark.load(target)
                raise ValueError(f"{benchmark.name}@{benchmark.version} is already registered; give the changed benchmark a new version")
        if not target.exists():
            staging = Path(tempfile.mkdtemp(prefix=".benchmark-", dir=target.parent))
            try:
                shutil.copytree(benchmark.benchmark_dir, staging, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns(".git", ".env", "__pycache__", "*.pyc", ".DS_Store"))
                staging.rename(target)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        data = yaml.safe_load(registry.read_text()) if registry.is_file() else {"benchmarks": []}
        data["benchmarks"].append(name)
        temporary = registry.with_suffix(".tmp")
        temporary.write_text(yaml.safe_dump(data, sort_keys=False))
        os.replace(temporary, registry)
    return Benchmark.load(target)
