from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from cua_speedrun.client import Computer


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"claude_code_{name}_test", ROOT / "agents" / "claude_code" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gateway():
    agent = load_script("agent")
    # An unsupported URL scheme keeps protocol-boundary checks off the network.
    return agent.ActionGateway(Computer("unconfigured://desktop"), max_steps=100)


@pytest.mark.parametrize("kind", ["terminate", "done", "finish"])
@pytest.mark.parametrize("status", ["success", "failure"])
def test_terminal_status_translation(gateway, kind, status):
    actions, terminal = gateway._translate({"action": kind, "status": status})
    assert terminal
    assert actions == ([{"action_type": "FAIL"}] if status == "failure" else [])


@pytest.mark.parametrize("status,endpoint", [("failure", "/step"), ("success", "/done")])
def test_terminal_command_reaches_correct_client_endpoint(gateway, status, endpoint):
    response = gateway.command(json.dumps({"action": "terminate", "status": status}))
    assert endpoint in response["error"]
    assert not gateway.finished


@pytest.mark.parametrize("clicks", [-7, -1, 0, 1, 3, 8])
def test_scroll_uses_wheel_clicks(gateway, clicks):
    assert gateway._translate({"action": "scroll", "clicks": clicks}) == (
        [{"mouse": {"scroll": clicks}}], False,
    )


@pytest.mark.parametrize("fields", [
    {"pixels": 1000}, {"pixels": 1000, "clicks": 3},
    {"clicks": 1.5}, {"clicks": True}, {"clicks": "3"}, {},
])
def test_scroll_rejects_pixels_and_invalid_click_counts(gateway, fields):
    with pytest.raises(ValueError, match="integer 'clicks' in mouse-wheel units"):
        gateway._translate({"action": "scroll", **fields})


def test_scroll_preserves_pointer_coordinate_scaling(gateway):
    gateway.ratio_x = gateway.ratio_y = 1.5
    gateway.native_w, gateway.native_h = 1920, 1080
    assert gateway._translate({
        "action": "scroll", "clicks": -3, "coordinate": [640, 360],
    }) == ([{"mouse": {"move": [960, 540]}}, {"mouse": {"scroll": -3}}], False)


def test_action_error_is_preserved_in_response(gateway):
    agent = load_script("agent")
    message = "Action timed out after 30s. The batch may have partially executed."
    error = agent._action_error({"ok": False, "error": message})
    assert error == message
    assert gateway._response(error=error)["error"] == message
    assert agent._action_error({"ok": True}) is None
    assert agent._action_error({"ok": False})
    assert agent._action_error({"error": message}) == message


def test_screenshot_command_requests_a_new_observation(gateway):
    response = gateway.command('{"action": "screenshot"}')
    assert "/observe" in response["error"]
    assert gateway.steps == 0


def test_observations_have_distinct_ids_without_consuming_actions(gateway):
    frames = []
    for color in ("red", "blue"):
        raw = BytesIO()
        Image.new("RGB", (8, 6), color).save(raw, format="PNG")
        gateway._set_observation(raw.getvalue())
        frames.append(gateway._response())
    assert [frame["observation_id"] for frame in frames] == [1, 2]
    assert all(frame["step"] == 0 and frame["budget_remaining"] == 100 for frame in frames)
    assert frames[0]["screenshot_b64"] != frames[1]["screenshot_b64"]
    with Image.open(BytesIO(base64.b64decode(frames[1]["screenshot_b64"]))) as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)


def test_prompt_uses_scroll_clicks_and_explicit_failure(gateway):
    agent = load_script("agent")
    text = agent.prompt("task", gateway)
    assert "scroll with integer clicks in mouse-wheel units" in text
    assert '"action": "scroll", "clicks": 3' in text
    assert "scroll with pixels" not in text
    assert 'If the task is infeasible, terminate with status "failure".' in text
    compile(agent.ACT_SCRIPT, "act", "exec")


def test_act_client_preserves_images_and_displays_errors(gateway, tmp_path):
    agent = load_script("agent")
    url = gateway.start()
    try:
        for observation_id, color in enumerate(("red", "blue"), start=1):
            raw = BytesIO()
            Image.new("RGB", (8, 6), color).save(raw, format="PNG")
            gateway._set_observation(raw.getvalue())
            result = subprocess.run(
                [sys.executable, "-c", agent.ACT_SCRIPT,
                 '{"action": "scroll", "pixels": 1000}'],
                cwd=tmp_path,
                env={**os.environ, "CS_ACT_GATEWAY": url, "CS_ACT_TOKEN": gateway.token},
                text=True, capture_output=True, timeout=15, check=True,
            )
            assert "error: scroll requires integer 'clicks'" in result.stdout
            assert f"screenshot: obs/frame_{observation_id:04d}.png" in result.stdout
            assert "budget_remaining: 100" in result.stdout
        with Image.open(tmp_path / "obs/frame_0001.png") as first:
            assert first.getpixel((0, 0)) == (255, 0, 0)
        with Image.open(tmp_path / "obs/frame_0002.png") as second:
            assert second.getpixel((0, 0)) == (0, 0, 255)
    finally:
        gateway.stop()


def test_oauth_isolated_from_api_credentials(monkeypatch, tmp_path):
    agent = load_script("agent")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-test-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-test-value")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    first = agent.child_environment(tmp_path / "first", "http://localhost", "gateway")
    second = agent.child_environment(tmp_path / "second", "http://localhost", "gateway")
    assert first["CLAUDE_CODE_OAUTH_TOKEN"] == second["CLAUDE_CODE_OAUTH_TOKEN"]
    assert "ANTHROPIC_API_KEY" not in first
    assert "ANTHROPIC_BASE_URL" not in first
    assert first["HOME"] != second["HOME"]
    assert first["CLAUDE_CONFIG_DIR"] != second["CLAUDE_CONFIG_DIR"]


def test_api_auth_is_preserved(monkeypatch, tmp_path):
    agent = load_script("agent")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-test-value")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    env = agent.child_environment(tmp_path, "http://localhost", "gateway")
    assert env["ANTHROPIC_API_KEY"] == "api-test-value"
    assert env["ANTHROPIC_BASE_URL"] == "https://example.invalid"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env


def test_missing_auth_fails_before_cli(monkeypatch, tmp_path):
    agent = load_script("agent")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY"):
        agent.child_environment(tmp_path, "http://localhost", "gateway")


def test_explicit_effort_and_oauth_flags(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("CLAUDE_CODE_EFFORT", "medium")
    agent = load_script("agent")
    command = agent.cli_command(Path("/bin/claude"), oauth=True)
    assert command[command.index("--model") + 1] == "claude-sonnet-5-5"
    assert command[command.index("--effort") + 1] == "medium"
    assert "--bare" not in command
    assert "--strict-mcp-config" in command
    assert command[command.index("--setting-sources") + 1] == ""
    assert "--bare" in agent.cli_command(Path("/bin/claude"), oauth=False)
    monkeypatch.setenv("CLAUDE_CODE_EFFORT", "meduim")
    with pytest.raises(ValueError, match="invalid CLAUDE_CODE_EFFORT"):
        agent.cli_command(Path("/bin/claude"), oauth=True)


def test_installers_do_not_receive_credentials(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-test-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-test-value")
    env = load_script("init").install_environment()
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_optional_credential_metadata(tmp_path):
    from cua_speedrun.service.templates_catalog import agent_metadata

    metadata = agent_metadata(ROOT / "agents" / "claude_code")
    assert not metadata.get("required_environment_variables")
    assert set(metadata["optional_environment_variables"]) == {
        "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"
    }
    (tmp_path / "agent.json").write_text(json.dumps({"optional_environment_variables": "bad"}))
    with pytest.raises(ValueError, match="must be a list of names"):
        agent_metadata(tmp_path)
