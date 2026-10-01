"""Submission options resolve against the real catalog without executing agents."""

import argparse

import pytest
from sqlalchemy import select

from cua_speedrun.commands.dashboard_client import register_dashboard_client_commands
from cua_speedrun.commands.local_evaluations import LocalEvaluations
from cua_speedrun.commands.paths import InstallationPaths
from cua_speedrun.parallelism import ExecutionScale
from cua_speedrun.service.catalog import sync_catalog
from cua_speedrun.service.db import BenchmarkRow, Run, Track, make_session_factory
from cua_speedrun.service.plans import attach_plan, plan_for_catalog_rows, plan_for_run
from cua_speedrun.service.evaluations import ensure_local_user
from cua_speedrun.service.store import LocalStore
from cua_speedrun.service.templates_catalog import template_zip


def _parse(*options):
    parser = argparse.ArgumentParser()
    register_dashboard_client_commands(parser.add_subparsers())
    return parser.parse_args([
        "submit", "--template", "qwen3vl", "--benchmark", "osworld-50", *options,
    ])


def test_submit_options_do_not_require_a_track():
    defaults = _parse()
    assert defaults.track is None
    assert defaults.gpu is None
    assert defaults.agent_mode is None
    selected = _parse("--gpu", "L40S", "--agent-mode", "shared-no-preload")
    assert selected.gpu == "L40S"
    assert selected.agent_mode == "shared-no-preload"
    assert selected.allocate_gpu is True
    assert _parse("--no-gpu").allocate_gpu is False


def test_submit_rejects_conflicting_gpu_options():
    with pytest.raises(SystemExit):
        _parse("--gpu", "L40S", "--no-gpu")
    with pytest.raises(SystemExit):
        _parse("--agent-mode", "unknown")


def test_run_options_are_frozen_without_changing_the_catalog(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'catalog.db'}")
    sync_catalog(sessions)
    with sessions() as session:
        track = session.scalars(select(Track)).one()
        benchmark = session.scalars(
            select(BenchmarkRow).where(BenchmarkRow.name == "osworld-50")
        ).one()
        default = plan_for_catalog_rows(track, benchmark)
        selected = plan_for_catalog_rows(
            track, benchmark, gpu="L40S", eval_algorithm="shared-no-preload",
        )
        assert default.gpu is None
        assert default.eval_algorithm == "shared-agent-vllm@2"
        assert selected.gpu == "L40S"
        assert selected.eval_algorithm == "shared-agent-no-preload@1"
        assert selected.agents_per_evaluation == 1
        assert selected.contract_hash != default.contract_hash
        assert selected.season() != default.season()
        assert ExecutionScale.for_plan(default, 4).env_pool_size == 8
        assert ExecutionScale.for_plan(selected, 4).env_pool_size == 4
        assert track.gpu is None
        assert track.eval_algorithm == "shared-agent-vllm@2"
        run = Run()
        attach_plan(run, selected)
        assert plan_for_run(run, track, benchmark).canonical_json == selected.canonical_json
        with pytest.raises(ValueError, match="disabled GPU"):
            plan_for_catalog_rows(track, benchmark, gpu="L40S", allocate_gpu=False)
        with pytest.raises(ValueError, match="must not be empty"):
            plan_for_catalog_rows(track, benchmark, gpu="  ")
        with pytest.raises(ValueError, match="unknown eval algorithm"):
            plan_for_catalog_rows(track, benchmark, eval_algorithm="unknown")


def test_local_submission_freezes_gpu_and_mode_before_execution(tmp_path):
    sessions = make_session_factory(f"sqlite:///{tmp_path / 'queue.db'}")
    sync_catalog(sessions)
    service = LocalEvaluations(
        paths=InstallationPaths.resolve(tmp_path),
        session_factory=sessions,
        store=LocalStore(tmp_path / "store"),
        user_id=ensure_local_user(sessions),
    )
    queued = service.submit(
        submission_zip=template_zip("qwen3vl"), name="options-check",
        track="default", benchmark="osworld-50", compute="local",
        environment="local", allocate_gpu=True, gpu="L40S",
        eval_algorithm="shared-no-preload", parallel_evaluations=4,
        saved_environment_names=(), evaluation_environment={},
    )
    with sessions() as session:
        run = session.get(Run, queued["run_id"])
        assert run.stage == "queued"
        assert run.execution_plan["track"]["gpu"] == "L40S"
        assert run.execution_plan["execution"]["algorithm"] == "shared-agent-no-preload@1"
        assert run.execution_scale["env_pool_size"] == 4
        assert session.scalars(select(Track)).one().gpu is None
    service.cancel(queued["run_id"])
