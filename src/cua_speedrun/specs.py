"""Task and benchmark definitions.

A task is a folder with a task.yaml. A benchmark is a folder with a
manifest.yaml that lists task folders. The executor never interprets the
`env` block of a task; it hands it to the chosen environment backend.
That is what keeps the core general: new environment kinds require a new
backend, never a core change.
"""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TIMEOUT_SEC = 300.0
DEFAULT_GRACE_SEC = 1.5


def _expand_benchmark_dir(value: Any, benchmark_dir: Path) -> Any:
    """Resolve the portable benchmark-package placeholder after hashing.

    Materialized files retain ``${BENCHMARK_DIR}``, keeping their bytes and
    content hash independent of the operator's cache location. Backends still
    receive ordinary absolute paths at execution time.
    """
    if isinstance(value, dict):
        return {
            key: _expand_benchmark_dir(item, benchmark_dir)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_expand_benchmark_dir(item, benchmark_dir) for item in value]
    if isinstance(value, str):
        return value.replace("${BENCHMARK_DIR}", str(benchmark_dir.resolve()))
    return value


@dataclass
class TaskSpec:
    """One task template. `env` is backend-specific and opaque to the core.

    A seeded task also names a `generator`: a Python file in the task folder
    exposing generate(seed) -> {"instruction": str, "expected": Any} and
    check(state_text, expected) -> {"passed": bool, "detail": str}. The
    generator derives the instruction and the privileged expected answer from
    the seed, and the check runs host-side, so the answer never touches the
    desktop. This is the deterministic-given-seed anti-memorization mechanism.
    """

    task_id: str
    description: str
    env: dict[str, Any]
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    grace_sec: float = DEFAULT_GRACE_SEC
    metadata: dict[str, Any] = field(default_factory=dict)
    generator: str | None = None
    task_dir: Path | None = None

    @classmethod
    def load(
        cls, task_dir: Path, benchmark_dir: Path | None = None
    ) -> "TaskSpec":
        path = task_dir / "task.yaml"
        raw = yaml.safe_load(path.read_text())
        for key in ("task_id", "description", "env"):
            if key not in raw:
                raise ValueError(f"{path}: missing required field '{key}'")
        if "kind" not in raw["env"]:
            raise ValueError(f"{path}: env block needs a 'kind' field")
        return cls(
            task_id=raw["task_id"],
            description=raw["description"],
            env=_expand_benchmark_dir(raw["env"], benchmark_dir or task_dir),
            timeout_sec=float(raw.get("timeout_sec", DEFAULT_TIMEOUT_SEC)),
            grace_sec=float(raw.get("grace_sec", DEFAULT_GRACE_SEC)),
            metadata=raw.get("metadata", {}),
            generator=raw.get("generator"),
            task_dir=task_dir,
        )

    def load_generator(self):
        """Import and return the task's generator module, or None."""
        if not self.generator or self.task_dir is None:
            return None
        gen_path = self.task_dir / self.generator
        spec = importlib.util.spec_from_file_location(
            f"cua_speedrun_gen_{self.task_id}", gen_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for fn in ("generate", "check"):
            if not hasattr(module, fn):
                raise ValueError(f"{gen_path}: generator must define {fn}()")
        return module


@dataclass
class Benchmark:
    """A named, versioned list of tasks."""

    name: str
    version: str
    tasks: list[TaskSpec]
    benchmark_dir: Path

    @classmethod
    def load(cls, benchmark_dir: Path) -> "Benchmark":
        from cua_speedrun.benchmark_sources import resolve_benchmark_path

        benchmark_dir = resolve_benchmark_path(benchmark_dir)
        path = benchmark_dir / "manifest.yaml"
        raw = yaml.safe_load(path.read_text())
        for key in ("name", "version", "tasks"):
            if key not in raw:
                raise ValueError(f"{path}: missing required field '{key}'")
        tasks = [
            TaskSpec.load(benchmark_dir / rel, benchmark_dir=benchmark_dir)
            for rel in raw["tasks"]
        ]
        ids = [t.task_id for t in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path}: duplicate task_id in manifest")
        return cls(
            name=raw["name"],
            version=str(raw["version"]),
            tasks=tasks,
            benchmark_dir=benchmark_dir,
        )


def content_hash(paths: list[Path]) -> str:
    """A stable hash over file contents, used to fingerprint submissions."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]
