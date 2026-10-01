"""Proxy boundary checks without booting or simulating a task environment."""

import ast
import importlib.util
from pathlib import Path

import pytest
import requests

from cua_speedrun.client import Computer
from cua_speedrun.runtime_environment import normalize_environment_name


SOURCE = Path(__file__).resolve().parents[1] / "agents/codex_cli/agent.py"
spec = importlib.util.spec_from_file_location("codex_http_agent", SOURCE)
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


@pytest.mark.parametrize("codex_steps,cs_steps,expected", [
    (None, None, 100),
    (None, "200", 200),
    ("500", None, 500),
    ("500", "200", 500),
    ("", "200", 200),
])
def test_forwardable_step_budget(monkeypatch, codex_steps, cs_steps, expected):
    assert normalize_environment_name("CODEX_MAX_STEPS") == "CODEX_MAX_STEPS"
    for name, value in (("CODEX_MAX_STEPS", codex_steps), ("CS_MAX_STEPS", cs_steps)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    configured = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configured)
    assert configured.MAX_STEPS == expected


@pytest.fixture
def proxy():
    # No request in these tests reaches Computer: only local HTTP boundaries
    # are exercised. This is not an environment or model evaluation.
    gateway = agent.ActionGateway(Computer("http://127.0.0.1:1"), max_steps=0)
    url = gateway.start()
    try:
        yield gateway, url, {"X-Gateway-Token": gateway.token}
    finally:
        gateway.stop()


@pytest.mark.parametrize("method,path", [("GET", "/observe"), ("POST", "/step"), ("POST", "/done")])
def test_authentication_required(proxy, method, path):
    gateway, url, _ = proxy
    response = requests.request(method, url + path, timeout=5)
    assert response.status_code == 403
    assert gateway.steps == 0
    assert gateway.fatal_error is None


@pytest.mark.parametrize("payload", [None, [], {}, {"actions": {}}, {"actions": "click"}])
def test_step_requires_actions_list(proxy, payload):
    gateway, url, headers = proxy
    response = requests.post(url + "/step", json=payload, headers=headers, timeout=5)
    assert response.status_code == 400
    assert gateway.steps == 0
    assert gateway.fatal_error is None


def test_unknown_route_and_exhausted_budget(proxy):
    gateway, url, headers = proxy
    assert requests.post(url + "/act", headers=headers, timeout=5).status_code == 404
    response = requests.post(url + "/step", headers=headers, json={"actions": []}, timeout=5)
    assert response.status_code == 409
    assert response.headers["X-Steps-Remaining"] == "0"
    assert gateway.fatal_error is None


def test_step_forwarding_has_no_action_translation():
    """Structural check: exactly one raw client call, no per-action parser."""
    tree = ast.parse(SOURCE.read_text())
    handler = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "do_POST")
    calls = [n for n in ast.walk(handler) if isinstance(n, ast.Call) and ast.unparse(n.func) == "gateway.computer.step"]
    assert len(calls) == 1
    assert ast.unparse(calls[0].args[0]) == "actions"
    assert any(isinstance(n, ast.Assign) and ast.unparse(n) == "actions = body['actions']" for n in ast.walk(handler))
    assert not any(isinstance(n, (ast.For, ast.ListComp)) for n in ast.walk(handler))
    assert not hasattr(agent, "_point")
    assert not hasattr(agent, "ACT_SCRIPT")


def test_prompt_describes_native_contract(proxy):
    gateway, _, _ = proxy
    text = agent.prompt("Test instruction", gateway)
    for term in ("GET /observe", "POST /step", "POST /done", "native resolution", "keys_down", "keys_up", "right_click_drag", "middle_click", "not an allowlist", "Test instruction"):
        assert term in text


def test_failed_cli_does_not_finalize_environment():
    tree = ast.parse(SOURCE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    guards = [n for n in ast.walk(main) if isinstance(n, ast.If) and any(
        isinstance(call, ast.Call) and ast.unparse(call.func) == "gateway.ensure_done"
        for statement in n.body for call in ast.walk(statement)
    )]
    assert len(guards) == 1
    assert ast.unparse(guards[0].test) == "returncode == 0 and gateway.fatal_error is None"


def test_subscription_secret_is_not_inherited_by_child(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_AUTH_JSON", '{"tokens":{"test":"not-a-credential"}}')
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-credential")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid")
    env = agent.child_environment(tmp_path, "http://127.0.0.1:1", "local-proxy-token")
    assert not {"CODEX_AUTH_JSON", "OPENAI_API_KEY", "OPENAI_BASE_URL"} & env.keys()


def test_invalid_subscription_cache_is_rejected(monkeypatch):
    monkeypatch.setenv("CODEX_AUTH_JSON", "not-json")
    with pytest.raises(SystemExit, match="valid credential-cache JSON"):
        agent.authentication()
