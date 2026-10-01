from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from sqlalchemy import select

from cua_speedrun.benchmark_sources import (
    benchmark_source_metadata,
    materialize_benchmark,
)
from cua_speedrun.parallelism import ExecutionScale
from cua_speedrun.service.catalog import default_track_name, sync_catalog
from cua_speedrun.service.db import BenchmarkRow, Run, SubmissionRow, Track, User, make_session_factory
from cua_speedrun.service.evaluations import EvaluationServiceError, benchmark_id_for_name


def test_catalog_sync_registers_dashboard_tracks_and_benchmark_sources(
    tmp_path,
) -> None:
    session_factory = make_session_factory(f"sqlite:///{tmp_path / 'catalog.db'}")

    first = sync_catalog(session_factory)
    second = sync_catalog(session_factory)

    assert first["tracks_added"] == 1
    assert second == {
        "tracks_added": 0,
        "tracks_updated": 0,
        "tracks_removed": 0,
        "benchmarks_added": 0,
        "benchmarks_updated": 0,
        "benchmarks_retired": 0,
    }
    with session_factory() as session:
        track = session.scalars(select(Track)).one()
        benchmarks = {row.name: row.task_count for row in session.scalars(
            select(BenchmarkRow).where(BenchmarkRow.active)
        )}

    assert track.name == "default"
    assert track.gpu is None
    assert track.eval_algorithm == "shared-agent-vllm@2"
    assert track.network_policy == "host-network"
    assert track.agents_per_evaluation == 1
    assert ExecutionScale.for_algorithm(track.eval_algorithm, 8, 1).env_pool_size == 16
    assert benchmarks == {
        "cua-world-26": 26, "osworld-50": 50, "osworld2-52": 52,
        "my-pc-bench": 38, "cua-world-offline": 143,
        "osworld-offline": 295, "osworld2-offline": 63,
    }
    assert default_track_name() == "default"


def test_catalog_retires_old_choices_without_changing_existing_runs(tmp_path) -> None:
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'retired.db'}")
    with sessions() as session:
        user = User(handle="catalog-test")
        retired = BenchmarkRow(name="retired-benchmark", version="1", path="/unused")
        session.add_all([user, retired, Track(name="old-track")])
        session.flush()
        submission = SubmissionRow(user_id=user.id, name="completed", storage_ref="/unused", track="old-track")
        session.add(submission)
        session.flush()
        run = Run(submission_id=submission.id, benchmark_id=retired.id, stage="done", execution_plan={"benchmark": {"name": retired.name}})
        session.add(run)
        session.commit()
        retired_id, run_id = retired.id, run.id

    first = sync_catalog(sessions)
    assert first["benchmarks_retired"] == 1
    assert first["tracks_removed"] == 1
    assert sync_catalog(sessions)["benchmarks_retired"] == 0
    with sessions() as session:
        assert session.get(BenchmarkRow, retired_id).active is False
        run = session.get(Run, run_id)
        assert run.benchmark_id == retired_id
        assert run.execution_plan == {"benchmark": {"name": "retired-benchmark"}}
        assert session.get(SubmissionRow, run.submission_id).track == "old-track"
        assert session.scalars(select(Track)).one().name == "default"
        assert len(session.scalars(select(BenchmarkRow).where(BenchmarkRow.active)).all()) == 7
    with pytest.raises(EvaluationServiceError, match="unknown benchmark"):
        benchmark_id_for_name(sessions, "retired-benchmark")
    assert benchmark_id_for_name(sessions, "osworld-50") > 0


def test_catalog_reactivates_a_restored_benchmark(tmp_path) -> None:
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'restored.db'}")
    sync_catalog(sessions)
    identity = benchmark_id_for_name(sessions, "osworld-50")
    with sessions() as session:
        session.get(BenchmarkRow, identity).active = False
        session.commit()
    assert sync_catalog(sessions)["benchmarks_updated"] == 1
    assert benchmark_id_for_name(sessions, "osworld-50") == identity


def test_generic_benchmark_source_materializes_into_content_addressed_cache(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "repo"
    source_dir = root / "benchmarks" / "demo"
    scripts = root / "scripts"
    source_dir.mkdir(parents=True)
    scripts.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (scripts / "input.txt").write_text("contract-v1\n")
    (scripts / "builder.py").write_text(
        """from pathlib import Path
import yaml

def materialize(source_path, out):
    out = Path(out)
    (out / 'tasks' / 'one').mkdir(parents=True, exist_ok=True)
    (out / 'manifest.yaml').write_text(yaml.safe_dump({
        'name': 'demo', 'version': '1', 'tasks': ['tasks/one']}))
    (out / 'tasks' / 'one' / 'task.yaml').write_text(yaml.safe_dump({
        'task_id': 'one', 'description': 'fixture', 'env': {'kind': 'fixture'}}))
"""
    )
    spec = {
        "schema_version": 1,
        "name": "demo",
        "version": "1",
        "materializer": {
            "path": "scripts/builder.py",
            "function": "materialize",
            "inputs": ["scripts/builder.py", "scripts/input.txt"],
        },
        "tasks": [{"id": "one"}],
    }
    source_path = source_dir / "benchmark-source.yaml"
    source_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    cache = tmp_path / "cache"
    monkeypatch.setenv("CS_BENCHMARK_CACHE", str(cache))

    assert benchmark_source_metadata(source_dir)["task_count"] == 1
    first = materialize_benchmark(source_dir)
    assert (first / "manifest.yaml").is_file()
    assert materialize_benchmark(source_dir) == first

    (scripts / "input.txt").write_text("contract-v2\n")
    second = materialize_benchmark(source_dir)
    assert second != first
    assert (second / "manifest.yaml").is_file()
