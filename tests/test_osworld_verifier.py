"""Contracts for the pinned, canonical OSWorld verifier path."""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts" / "osworld_shared" / "osworld_verifier.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("_test_osworld_verifier", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_verifier()
FAIL = {"action_type": "FAIL"}
CLICK = {"action_type": "CLICK", "x": 10, "y": 20}
TYPE = {"action_type": "TYPE", "text": "hello"}


def _traj(*action_values):
    steps = [{"event": "session", "session_id": "s"}]
    steps.extend(
        {"event": "step", "ts": 0.0, "idx": index, "action": value}
        for index, value in enumerate(action_values)
    )
    steps.append({"event": "finalize", "ts": 0.0})
    return {"steps": steps}


def test_logged_action_batches_become_canonical_action_history() -> None:
    trajectory = _traj([CLICK, TYPE], [FAIL], [])
    assert VERIFIER._extract_action_history(trajectory) == [CLICK, TYPE, FAIL]


def test_trailing_finalize_step_does_not_mask_infeasible_action() -> None:
    trajectory = _traj([CLICK], [FAIL], [])
    assert VERIFIER._extract_action_history(trajectory)[-1] == FAIL


def test_verifier_calls_upstream_desktop_env_evaluate_directly() -> None:
    evaluate_source = inspect.getsource(VERIFIER._evaluate)
    assert "DesktopEnv.evaluate(env)" in evaluate_source


def test_verifier_does_not_reimplement_osworld_scoring() -> None:
    removed_local_semantics = {
        "_LocalController",
        "_LocalSetupController",
        "_LocalEnv",
        "_get_state",
        "_metric_score",
        "_run_setup_item",
        "_score_after_setup",
        "_patch_getters",
        "_last_action_is_fail",
    }
    assert removed_local_semantics.isdisjoint(vars(VERIFIER))


def test_verifier_exceptions_are_not_converted_to_score_zero() -> None:
    tree = ast.parse(VERIFIER_PATH.read_text())
    check = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "check_with_source"
    )
    assert not any(isinstance(node, ast.Try) for node in ast.walk(check))


def test_desktop_env_source_is_immutable_and_content_addressed() -> None:
    source_spec = yaml.safe_load(
        (ROOT / "benchmarks/osworld-50/benchmark-source.yaml").read_text()
    )
    assert VERIFIER.OSWORLD_COMMIT == "315a7603173feadf1b8a85cbc006c93ffe1dc1a1"
    assert VERIFIER.OSWORLD_COMMIT == source_spec["source_benchmark"]["commit"]
    assert len(VERIFIER.OSWORLD_DESKTOP_ENV_SHA256) == 64
    source = VERIFIER_PATH.read_text()
    assert "_desktop_env_digest(root)" in source
    assert "DesktopEnv.__new__(DesktopEnv)" in source
