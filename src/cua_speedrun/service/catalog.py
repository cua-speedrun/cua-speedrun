"""Synchronize maintainer data and materialized benchmarks into the DB.

Tracks and benchmark choices live in catalog data files. Benchmarks provide
an executable manifest or a compact ``benchmark-source.yaml`` that resolves to
an immutable operator-local package. Adding a benchmark remains data plus an
optional replaceable builder, never an API or UI change. Existing
result-bearing contracts are never rewritten: runs retain their copied
RunPlan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select

from cua_speedrun.benchmark_sources import (
    SOURCE_FILENAME,
    benchmark_catalog_paths,
    benchmark_source_metadata,
)
from cua_speedrun.eval_algorithms import resolve_eval_algorithm
from cua_speedrun.resources import resource_root
from cua_speedrun.service.db import BenchmarkRow, SubmissionRow, Track
from cua_speedrun.specs import Benchmark


ROOT = resource_root()
TRACK_CATALOG = ROOT / "catalog" / "tracks.yaml"


def _track_specs(path: Path = TRACK_CATALOG) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text()) or {}
    tracks = data.get("tracks") or []
    if not isinstance(tracks, list):
        raise ValueError("catalog/tracks.yaml must contain a tracks list")
    specs = [dict(spec) for spec in tracks]
    for spec in specs:
        spec["eval_algorithm"] = resolve_eval_algorithm(
            spec.get("eval_algorithm")
        ).key
    return specs


def default_track_name(path: Path = TRACK_CATALOG) -> str | None:
    data = yaml.safe_load(path.read_text()) or {}
    value = data.get("default_track")
    if value is None:
        return None
    name = str(value).strip()
    known = {spec["name"] for spec in _track_specs(path)}
    if name not in known:
        raise ValueError(
            f"catalog default_track {name!r} is not present in tracks"
        )
    return name


def sync_catalog(session_factory) -> dict[str, int]:
    """Upsert configured tracks and available benchmark packages/sources."""
    counts = {"tracks_added": 0, "tracks_updated": 0, "tracks_removed": 0,
              "benchmarks_added": 0, "benchmarks_updated": 0,
              "benchmarks_retired": 0}
    with session_factory() as session:
        root = resource_root()
        track_specs = _track_specs(root / "catalog/tracks.yaml")
        active_track_names = {spec["name"] for spec in track_specs}
        for row in session.execute(select(Track)).scalars().all():
            if row.name not in active_track_names:
                # Runs carry a complete copied RunPlan, so removing a retired
                # catalog choice cannot rewrite historical evidence.
                session.delete(row)
                counts["tracks_removed"] += 1
        for spec in track_specs:
            row = session.execute(
                select(Track).where(Track.name == spec["name"])
            ).scalar_one_or_none()
            if row is None:
                session.add(Track(**spec))
                counts["tracks_added"] += 1
            else:
                used = session.execute(
                    select(SubmissionRow.id)
                    .where(SubmissionRow.track == row.name)
                    .limit(1)
                ).scalar_one_or_none()
                changed = any(getattr(row, key) != value for key, value in spec.items())
                if changed and used is None:
                    for key, value in spec.items():
                        setattr(row, key, value)
                    counts["tracks_updated"] += 1

        benchmark_specs: dict[tuple[str, str], dict[str, Any]] = {}
        paths = benchmark_catalog_paths(root)
        manifests = [path / "manifest.yaml" for path in paths]
        for manifest in filter(Path.is_file, manifests):
            benchmark = Benchmark.load(manifest.parent)
            spec = {
                "name": benchmark.name,
                "version": benchmark.version,
                "path": str(manifest.parent.resolve()),
                "task_count": len(benchmark.tasks),
            }
            benchmark_specs[(benchmark.name, benchmark.version)] = spec
        sources = [path / SOURCE_FILENAME for path in paths]
        for source in filter(Path.is_file, sources):
            spec = benchmark_source_metadata(source)
            key = (spec["name"], spec["version"])
            if key in benchmark_specs:
                raise ValueError(
                    f"duplicate materialized and source benchmark {key[0]}@{key[1]}"
                )
            benchmark_specs[key] = spec

        for row in session.scalars(select(BenchmarkRow)):
            if row.active and (row.name, row.version) not in benchmark_specs:
                # Keep IDs used by existing runs, but remove retired choices.
                row.active = False
                counts["benchmarks_retired"] += 1

        for spec in benchmark_specs.values():
            row = session.execute(
                select(BenchmarkRow).where(
                    BenchmarkRow.name == spec["name"],
                    BenchmarkRow.version == spec["version"],
                )
            ).scalar_one_or_none()
            if row is None:
                session.add(BenchmarkRow(
                    name=spec["name"],
                    version=spec["version"],
                    path=spec["path"],
                    task_count=spec["task_count"],
                ))
                counts["benchmarks_added"] += 1
            elif (
                row.path != spec["path"]
                or row.task_count != spec["task_count"]
                or not row.active
            ):
                row.path = spec["path"]
                row.task_count = spec["task_count"]
                row.active = True
                counts["benchmarks_updated"] += 1
        session.commit()
    return counts
