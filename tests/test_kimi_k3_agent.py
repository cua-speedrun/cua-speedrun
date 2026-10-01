from __future__ import annotations

import importlib.util
import io
import json
import runpy
import sys
import types
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = ROOT / "agents" / "kimi_k3" / "agent.py"
INIT_PATH = ROOT / "agents" / "kimi_k3" / "init.py"
ENV_VARS = (
    "OPENROUTER_API_KEY",
    "CS_MAX_STEPS",
    "KIMI_MODEL",
    "KIMI_BASE_URL",
    "KIMI_MAX_STEPS",
    "KIMI_MAX_IMAGE_HISTORY_LENGTH",
    "KIMI_MAX_TOKENS",
    "KIMI_TEMPERATURE",
    "KIMI_TOP_P",
    "KIMI_REASONING_EFFORT",
    "KIMI_HTTP_TIMEOUT",
    "KIMI_HTTP_RETRIES",
    "KIMI_PREDICT_RETRIES",
    "KIMI_RETRY_SLEEP",
    "KIMI_ENV_HTTP_TIMEOUT",
    "KIMI_WAIT_SECONDS",
    "KIMI_CLIENT_PASSWORD",
)


def import_agent(monkeypatch: pytest.MonkeyPatch, env: dict[str, str] | None = None):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    module_name = f"kimi_k3_agent_test_{id(monkeypatch)}"
    spec = importlib.util.spec_from_file_location(module_name, AGENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def tiny_png(width: int = 8, height: int = 6) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (width, height), (20, 40, 60)).save(output, format="PNG")
    return output.getvalue()


def response(code: str, action: str = "Do it", reasoning: str = "Inspect the UI") -> dict:
    return {
        "role": "assistant",
        "reasoning": reasoning,
        "content": f"## Action:\n{action}\n## Code:\n```python\n{code}\n```",
    }


def tool_response(
    code: str,
    action: str = "Do it",
    reasoning: str = "Inspect the UI",
) -> dict:
    return {
        "role": "assistant",
        "content": action,
        "reasoning": reasoning,
        "tool_calls": [
            {
                "id": "computer_action_0",
                "type": "function",
                "function": {
                    "name": "computer_action",
                    "arguments": json.dumps({"description": action, "code": code}),
                },
            }
        ],
    }


def test_kimi_k3_defaults_and_openrouter_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)
    assert agent.MODEL == "moonshotai/kimi-k3"
    assert agent.API_URL == "https://openrouter.ai/api/v1/chat/completions"
    assert agent.MAX_STEPS == 100
    assert agent.MAX_IMAGE_HISTORY_LENGTH == 3
    assert agent.MAX_TOKENS == 16384
    assert agent.HTTP_RETRIES == 5
    assert agent.TEMPERATURE == 1.0
    assert agent.TOP_P == 1.0
    assert agent.REASONING_EFFORT == "max"
    payload = agent.build_payload([{"role": "user", "content": "hello"}])
    assert payload["reasoning"] == {"effort": "max", "exclude": False}
    assert payload["provider"] == {
        "only": ["moonshotai/mxfp4"],
        "allow_fallbacks": False,
    }
    assert payload["tool_choice"] == "required"
    assert payload["parallel_tool_calls"] is False
    assert [tool["function"]["name"] for tool in payload["tools"]] == [
        "computer_action",
        "computer_wait",
        "computer_terminate",
    ]
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 1.0


def test_system_prompt_is_upstream_prompt_with_real_password(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)
    messages = agent.build_messages("task", tiny_png(), [])
    prompt = messages[0]["content"]
    assert "You are a GUI agent." in prompt
    assert "The passoword of the computer is password." in prompt
    assert '"name": "computer.wait"' in prompt
    assert '"name": "computer.terminate"' in prompt


def test_history_compacts_boundary_and_preserves_recent_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)
    history = [
        {
            "screenshot": tiny_png(),
            "message": tool_response(
                "pyautogui.click(2, 2)",
                action=f"action {index}",
                reasoning=f"reason {index}",
            ),
            "thought": f"reason {index}",
            "action": f"action {index}",
        }
        for index in range(4)
    ]
    messages = agent.build_messages("task", tiny_png(), history)
    historical_images = [
        message
        for message in messages[:-1]
        if message.get("role") == "user" and isinstance(message.get("content"), list)
    ]
    # Exact upstream boundary: max_image_history_length=3 retains two images.
    assert len(historical_images) == 2
    assert any("# Step 2:" in str(message.get("content")) for message in messages)
    recent_assistant = messages[-3]
    assert recent_assistant["role"] == "assistant"
    assert recent_assistant["reasoning"] == "reason 3"
    assert recent_assistant["tool_calls"][0]["function"]["name"] == "computer_action"
    assert messages[-2] == {
        "role": "tool",
        "tool_call_id": "computer_action_0",
        "content": "The action was executed. Inspect the next screenshot for the result.",
    }


def test_response_parser_accepts_native_computer_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch)
    parsed = agent.parse_response(
        tool_response("pyautogui.click(0.5, 0.25)"),
        (1920, 1080),
    )
    assert parsed == {
        "thought": "Inspect the UI",
        "action": "Do it",
        "original_code": "pyautogui.click(0.5, 0.25)",
        "code": "pyautogui.click(960, 270)",
    }


@pytest.mark.parametrize(
    ("name", "arguments", "token"),
    [
        ("computer_wait", {}, "WAIT"),
        ("computer_terminate", {"status": "success"}, "DONE"),
        ("computer_terminate", {"status": "failure"}, "FAIL"),
    ],
)
def test_response_parser_native_control_tools(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    arguments: dict,
    token: str,
) -> None:
    agent = import_agent(monkeypatch)
    message = {
        "content": "",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }
    assert agent.parse_response(message, (100, 100))["code"] == token


def test_response_parser_accepts_openrouter_reasoning_and_scales_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch)
    parsed = agent.parse_response(
        response("pyautogui.click(0.5, 0.25)"),
        (1920, 1080),
    )
    assert parsed["thought"] == "Inspect the UI"
    assert parsed["action"] == "Do it"
    assert parsed["code"] == "pyautogui.click(960, 270)"


@pytest.mark.parametrize(
    ("status", "token"),
    [("success", "DONE"), ("failure", "FAIL")],
)
def test_response_parser_terminal_status(
    monkeypatch: pytest.MonkeyPatch, status: str, token: str
) -> None:
    agent = import_agent(monkeypatch)
    message = {
        "content": (
            "## Action:\nStop\n## Code:\n```code\n"
            f'{{"name":"computer.terminate","status":"{status}"}}\n```'
        )
    }
    assert agent.parse_response(message, (100, 100))["code"] == token


def test_literal_pyautogui_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)
    cursor: list[int] = []
    actions = agent.translate_pyautogui(
        """pyautogui.click(50, 25)
pyautogui.hotkey('ctrl', 'a')
pyautogui.write('hello\\nworld')
pyautogui.scroll(3)
pyautogui.dragTo(80, 40, duration=1)""",
        100,
        50,
        cursor,
    )
    assert actions[0] == {"mouse": {"left_click": [50, 25]}}
    assert {"keyboard": {"keys": ["ctrl", "a"]}} in actions
    assert {"keyboard": {"text": "hello"}} in actions
    assert {"keyboard": {"keys": ["Return"]}} in actions
    assert {"mouse": {"scroll": -3}} in actions
    assert actions[-1] == {"mouse": {"left_click_drag": [[50, 25], [80, 40]]}}
    assert cursor == [80, 40]


def test_translator_rejects_arbitrary_python(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)
    with pytest.raises(agent.ParseError, match=r"only pyautogui\.\* and time\.sleep"):
        agent.translate_pyautogui("open('/tmp/x', 'w').write('bad')", 100, 100, [])


def test_catalog_declares_kimi_openrouter_key() -> None:
    from cua_speedrun.service.templates_catalog import list_templates

    template = next(item for item in list_templates() if item["name"] == "kimi_k3")
    assert template["required_environment_variables"] == ["OPENROUTER_API_KEY"]
    assert "osworld-50" in template["compatible_benchmarks"]


def test_init_credential_check_does_not_offer_computer_tools(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict = {}

    def fake_build_payload(messages: list[dict]) -> dict:
        return {
            "messages": messages,
            "tools": [{"type": "function"}],
            "tool_choice": "required",
            "reasoning": {"effort": "max"},
            "max_tokens": 4096,
        }

    def fake_request(payload: dict) -> dict:
        seen.update(payload)
        return {"role": "assistant", "content": None}

    fake_agent = types.ModuleType("agent")
    fake_agent.MODEL = "moonshotai/kimi-k3"
    fake_agent.build_payload = fake_build_payload
    fake_agent.kimi_request = fake_request
    monkeypatch.setitem(sys.modules, "agent", fake_agent)

    runpy.run_path(str(INIT_PATH), run_name="__main__")

    assert "tools" not in seen
    assert "tool_choice" not in seen
    assert seen["reasoning"] == {"effort": "low", "exclude": False}
    assert seen["max_tokens"] == 256
    assert "credential check ok: ready" in capsys.readouterr().out


def test_invalid_native_action_is_returned_as_tool_error_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(
        monkeypatch,
        {"OPENROUTER_API_KEY": "test-key", "KIMI_MAX_STEPS": "2"},
    )
    requests_seen: list[dict] = []
    replies = iter(
        (
            tool_response(
                "values = ['A', 'B']\nfor value in values:\n    pyautogui.write(value)"
            ),
            tool_response("pyautogui.click(2, 2)"),
            {
                "role": "assistant",
                "content": "Done",
                "tool_calls": [
                    {
                        "id": "computer_terminate_0",
                        "type": "function",
                        "function": {
                            "name": "computer_terminate",
                            "arguments": json.dumps({"status": "success"}),
                        },
                    }
                ],
            },
        )
    )

    def fake_request(payload: dict, reporter) -> dict:
        requests_seen.append(payload)
        return next(replies)

    class FakeComputer:
        instance = None

        def __init__(self, *_args, **_kwargs) -> None:
            self.steps: list[list[dict]] = []
            self.done_called = False
            FakeComputer.instance = self

        def observe(self) -> dict:
            return {"png": tiny_png()}

        def step(self, actions: list[dict]) -> None:
            self.steps.append(actions)

        def wait(self, _seconds: float) -> None:
            raise AssertionError("wait was not expected")

        def done(self) -> None:
            self.done_called = True

    monkeypatch.setattr(agent, "kimi_request", fake_request)
    monkeypatch.setattr(agent, "Computer", FakeComputer)

    agent.run("http://environment.invalid", "click once")

    retry_messages = requests_seen[1]["messages"]
    assert retry_messages[-2]["role"] == "assistant"
    assert retry_messages[-2]["tool_calls"][0]["id"] == "computer_action_0"
    assert retry_messages[-1]["role"] == "tool"
    assert retry_messages[-1]["tool_call_id"] == "computer_action_0"
    assert "use only direct sequential pyautogui" in retry_messages[-1]["content"]
    assert FakeComputer.instance.steps == [[{"mouse": {"left_click": [2, 2]}}]]
    assert FakeComputer.instance.done_called is True


def test_openrouter_cost_reporter_emits_cumulative_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = import_agent(monkeypatch, {"OPENROUTER_API_KEY": "test-key"})
    responses = iter((
        {
            "choices": [{"finish_reason": "stop", "message": response("DONE")}],
            "usage": {
                "cost": 0.1,
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        },
        {
            "choices": [{"finish_reason": "stop", "message": response("DONE")}],
            "usage": {
                "cost": 0.25,
                "prompt_tokens": 20,
                "completion_tokens": 3,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        },
    ))

    class FakeResponse:
        status_code = 200
        text = ""

        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def json(self) -> dict:
            return self.payload

    monkeypatch.setattr(
        agent.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(next(responses)),
    )
    reporter = agent.CostReporter()

    agent.kimi_request({}, reporter)
    agent.kimi_request({}, reporter)

    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(agent.COST_SNAPSHOT_PREFIX)
    ]
    assert len(lines) == 2
    latest = json.loads(lines[-1][len(agent.COST_SNAPSHOT_PREFIX):])
    assert latest == {
        "cost_usd": 0.35,
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 5,
            "prompt_tokens_details.cached_tokens": 4,
            "completion_tokens_details.reasoning_tokens": 2,
        },
    }
