"""Resolve compact benchmark sources into immutable local packages.

Benchmark source directories contain ``benchmark-source.yaml`` instead of a
materialized ``manifest.yaml`` tree.  The source names a maintainer-owned
Python materializer and every file that affects its output.  Their combined
digest selects an operator-local cache directory, so generated task trees do
not belong in Git and identical sources resolve to identical package bytes on
every machine.

Materializers are deliberately outside the core package.  Adding a benchmark
therefore remains data plus a replaceable builder; it does not add benchmark
branches to the executor, dashboard, or scoring code.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml


SOURCE_FILENAME = "benchmark-source.yaml"
SOURCE_SCHEMA_VERSION = 1
MATERIALIZATION_ENGINE_VERSION = 1


def benchmark_catalog_paths(root: Path) -> list[Path]:
    """Read the public benchmark registry without materializing any tasks."""
    data = yaml.safe_load((root / "catalog/benchmarks.yaml").read_text()) or {}
    names = data.get("benchmarks")
    local = root / "catalog/local-benchmarks.yaml"
    if local.is_file() and isinstance(names, list):
        local_names = (yaml.safe_load(local.read_text()) or {}).get("benchmarks", [])
        if not isinstance(local_names, list):
            raise ValueError("local-benchmarks.yaml must list benchmark directory names")
        names = names + local_names
    if not isinstance(names, list) or not all(
        isinstance(name, str) and name and Path(name).name == name
        and name not in (".", "..") for name in names
    ):
        raise ValueError("catalog/benchmarks.yaml must list benchmark directory names")
    if len(names) != len(set(names)):
        raise ValueError("catalog/benchmarks.yaml contains duplicate benchmarks")
    base = (root / "benchmarks").resolve()
    paths = [(base / name).resolve() for name in names]
    for path in paths:
        if path.parent != base or not path.is_dir():
            raise ValueError(f"invalid catalog benchmark directory: {path}")
        if not any((path / name).is_file() for name in ("manifest.yaml", SOURCE_FILENAME)):
            raise FileNotFoundError(f"benchmark definition is missing: {path}")
    return paths


def _load_source(path: Path) -> tuple[Path, dict[str, Any]]:
    source_path = path / SOURCE_FILENAME if path.is_dir() else path
    if source_path.name != SOURCE_FILENAME or not source_path.is_file():
        raise FileNotFoundError(f"benchmark source does not exist: {source_path}")
    raw = yaml.safe_load(source_path.read_text()) or {}
    if raw.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError(
            f"{source_path}: schema_version must be {SOURCE_SCHEMA_VERSION}"
        )
    for key in ("name", "version", "tasks", "materializer"):
        if key not in raw:
            raise ValueError(f"{source_path}: missing required field {key!r}")
    if not isinstance(raw["tasks"], list) or not raw["tasks"]:
        raise ValueError(f"{source_path}: tasks must be a non-empty list")
    materializer = raw["materializer"]
    if not isinstance(materializer, dict):
        raise ValueError(f"{source_path}: materializer must be a mapping")
    for key in ("path", "function", "inputs"):
        if key not in materializer:
            raise ValueError(
                f"{source_path}: materializer is missing required field {key!r}"
            )
    return source_path.resolve(), raw


def benchmark_source_metadata(path: Path) -> dict[str, Any]:
    """Return catalog-safe metadata without downloading or executing code."""
    source_path, raw = _load_source(Path(path))
    return {
        "name": str(raw["name"]),
        "version": str(raw["version"]),
        "task_count": len(raw["tasks"]),
        "path": str(source_path.parent),
    }


def _repository_root(source_path: Path) -> Path:
    if source_path.is_file():
        raw = yaml.safe_load(source_path.read_text()) or {}
        script = (raw.get("materializer") or {}).get("path")
        if isinstance(script, str) and (source_path.parent / script).is_file():
            return source_path.parent
    for candidate in source_path.parents:
        if (candidate / "pyproject.toml").is_file() or (
            (candidate / "catalog" / "tracks.yaml").is_file()
            and (candidate / "benchmarks").is_dir()
            and (candidate / "scripts").is_dir()
        ):
            return candidate
    return source_path.parent


def _declared_inputs(
    source_path: Path, raw: dict[str, Any], root: Path
) -> list[tuple[str, Path]]:
    values = list(raw["materializer"]["inputs"])
    materializer_path = str(raw["materializer"]["path"])
    if materializer_path not in values:
        values.append(materializer_path)
    resolved: list[tuple[str, Path]] = [(SOURCE_FILENAME, source_path)]
    for value in sorted({str(item) for item in values}):
        path = (root / value).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"{source_path}: materializer input escapes repository: {value}")
        if not path.is_file():
            raise FileNotFoundError(
                f"{source_path}: materializer input does not exist: {value}"
            )
        resolved.append((value, path))
    return resolved


def _source_digest(source_path: Path, raw: dict[str, Any], root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(f"engine:{MATERIALIZATION_ENGINE_VERSION}".encode())
    digest.update(b"\0")
    for label, path in _declared_inputs(source_path, raw, root):
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _cache_root() -> Path:
    configured = os.environ.get("CS_BENCHMARK_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache" / "cua-speedrun" / "benchmarks"


def _load_materializer(path: Path, function_name: str):
    module_name = f"cua_speedrun_benchmark_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark materializer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ValueError(f"{path}: materializer function {function_name!r} is missing")
    return function


def materialize_benchmark(path: Path) -> Path:
    """Materialize one compact source into its content-addressed cache.

    Builders write into a unique staging directory.  Renaming that directory
    makes publication atomic; concurrent builders may duplicate download work
    but can never expose a partial benchmark package.
    """
    source_path, raw = _load_source(Path(path))
    root = _repository_root(source_path)
    from cua_speedrun.benchmark_preparation import prepare_data

    prepare_data(source_path.parent, root)
    digest = _source_digest(source_path, raw, root)
    target = (
        _cache_root()
        / str(raw["name"])
        / str(raw["version"])
        / digest
    )
    if (target / "manifest.yaml").is_file():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=target.parent))
    try:
        materializer = raw["materializer"]
        materializer_path = (root / str(materializer["path"])).resolve()
        function = _load_materializer(materializer_path, str(materializer["function"]))
        function(source_path=source_path, out=staging)
        if not (staging / "manifest.yaml").is_file():
            raise RuntimeError(
                f"benchmark materializer did not create manifest.yaml: {materializer_path}"
            )
        provenance = {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "name": str(raw["name"]),
            "version": str(raw["version"]),
            "source_digest": digest,
            "materializer": str(materializer["path"]),
        }
        (staging / ".materialization.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        try:
            staging.replace(target)
        except OSError:
            if not (target / "manifest.yaml").is_file():
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def resolve_benchmark_path(path: Path) -> Path:
    """Return a materialized package for either a package or source path."""
    candidate = Path(path).expanduser().resolve()
    if (candidate / "manifest.yaml").is_file():
        return candidate
    source = candidate / SOURCE_FILENAME if candidate.is_dir() else candidate
    if source.name == SOURCE_FILENAME and source.is_file():
        return materialize_benchmark(source)
    raise FileNotFoundError(
        f"benchmark needs manifest.yaml or {SOURCE_FILENAME}: {candidate}"
    )
