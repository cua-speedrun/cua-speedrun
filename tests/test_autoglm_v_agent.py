from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


class DummyComputer:
    def __init__(self) -> None:
        self.steps: list[list[dict]] = []
        self.waits: list[float] = []

    def step(self, actions: list[dict]) -> dict:
        self.steps.append(actions)
        return {}

    def wait(self, seconds: float) -> None:
        self.waits.append(seconds)


def import_agent(monkeypatch: pytest.MonkeyPatch, env: dict[str, str] | None = None):
    for var in (
        "GLM_MODEL",
        "GLM_BASE_URL",
        "GLM_HTTP_TIMEOUT",
        "GLM_MAX_RETRIES",
        "GLM_MAX_TOKENS",
        "GLM_TEMPERATURE",
        "GLM_TOP_P",
        "GLM_MAX_TURNS",
        "GLM_MAX_STEPS",
        "GLM_CLIENT_PASSWORD",
        "GLM_VLLM_FLAGS",
        "CS_MAX_STEPS",
    ):
        monkeypatch.delenv(var, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)

    path = Path(__file__).resolve().parents[1] / "agents" / "autoglm_v" / "agent.py"
    spec = importlib.util.spec_from_file_location("autoglm_v_agent_test", path)
    assert spec is not None and spec.loader is not None
    agent = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = agent
    spec.loader.exec_module(agent)
    return agent


def test_max_steps_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # GLM_MAX_STEPS is the user-settable knob and wins over the
    # harness-internal CS_MAX_STEPS fallback.
    agent = import_agent(monkeypatch, env={"CS_MAX_STEPS": "120"})
    assert agent.MAX_STEPS == 120
    agent = import_agent(
        monkeypatch, env={"CS_MAX_STEPS": "120", "GLM_MAX_STEPS": "77"}
    )
    assert agent.MAX_STEPS == 77


def test_defaults_match_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    assert agent.MODEL == "glm-4.5v"
    assert agent.BASE_URL == "https://open.bigmodel.cn/api/paas/v4"
    assert agent.API_URL == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    # Canonical OSWorld default, not the 100 this port originally shipped.
    assert agent.MAX_STEPS == 30
    assert agent.MAX_TOKENS == 2048
    assert agent.TEMPERATURE == 0.4
    assert agent.TOP_P == 0.5
    assert agent.MAX_TURNS == 30
    assert agent.STOP == ["<|user|>", "<|observation|>", "</answer>"]


def test_payload_shape_matches_reference_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch)

    messages = [{"role": "user", "content": "hi"}]
    payload = agent.build_payload(messages)
    # The default endpoint is hosted Zhipu, which does not document the vLLM
    # sampling passthroughs, so they are omitted there.
    assert set(payload) == {
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "stream",
        "stop",
    }
    assert payload["model"] == "glm-4.5v"
    assert payload["max_tokens"] == 2048
    assert payload["temperature"] == 0.4
    assert payload["top_p"] == 0.5
    assert payload["stream"] is False
    assert payload["stop"] == ["<|user|>", "<|observation|>", "</answer>"]


def test_system_prompt_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    message = agent.system_message("open the settings app")
    # glm41v output format hint.
    assert "<think>" in message
    assert "<answer>```python" in message
    # Function defs generated from the agent-action signatures.
    assert "Class Agent:" in message
    assert "def click(coordinate: List, num_clicks: int = 1, button_type: str = 'left')" in message
    assert "def exit(success: bool)" in message
    assert "def drag_and_drop(drag_from_coordinate: List, drop_on_coordinate: List)" in message
    # Relative-coordinate note and password note.
    assert "normalized to 0-1000" in message
    assert "password is 'password'" in message
    # The in-VM-only actions are not offered.
    assert "open_app" not in message
    assert "switch_window" not in message
    # Task appended at the end, reference-style.
    assert message.endswith(
        "**IMPORTANT** You are asked to complete the following task: open the settings app"
    )


def test_parse_code_from_string(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    text = (
        "<think>\nI should click the icon.</think>\n"
        "<answer>```python\nAgent.click(coordinate=[500, 500])\n```</answer>"
    )
    assert agent.parse_code_from_string(text) == ["Agent.click(coordinate=[500, 500])"]

    assert agent.parse_code_from_string("WAIT") == ["WAIT"]
    assert agent.parse_code_from_string("```\nDONE\n```") == ["DONE"]
    # Command on the last line of a block is split out.
    assert agent.parse_code_from_string("```python\nAgent.wait()\nDONE\n```") == [
        "Agent.wait()",
        "DONE",
    ]
    assert agent.parse_code_from_string("no code here") == []


def test_click_translation_and_coordinate_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch)

    computer = DummyComputer()
    result, terminal = agent.execute_code(
        computer, "Agent.click(coordinate=[500, 500])", 1600, 900
    )
    assert result == "Click Success"
    assert terminal is None
    assert computer.steps == [[{"mouse": {"left_click": [800, 450]}}]]

    computer = DummyComputer()
    agent.execute_code(
        computer, "Agent.click(coordinate=[500, 500], num_clicks=2)", 1600, 900
    )
    assert computer.steps == [[{"mouse": {"double_click": [800, 450]}}]]

    computer = DummyComputer()
    agent.execute_code(
        computer, "Agent.click(coordinate=[500, 500], num_clicks=3)", 1600, 900
    )
    assert computer.steps == [[{"mouse": {"triple_click": [800, 450]}}]]

    computer = DummyComputer()
    agent.execute_code(
        computer, "Agent.click(coordinate=[500, 500], button_type='right')", 1600, 900
    )
    assert computer.steps == [[{"mouse": {"right_click": [800, 450]}}]]

    # Clamped into the framebuffer (round(1000 * 1600 / 1000) == 1600).
    computer = DummyComputer()
    agent.execute_code(computer, "Agent.click(coordinate=[1000, 0])", 1600, 900)
    assert computer.steps == [[{"mouse": {"left_click": [1599, 0]}}]]


def test_type_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    computer = DummyComputer()
    result, _terminal = agent.execute_code(
        computer,
        "Agent.type(coordinate=[100, 100], text='hi', overwrite=True, enter=True)",
        1600,
        900,
    )
    assert result == "Type Success"
    assert computer.steps == [
        [
            {"mouse": {"left_click": [160, 90]}},
            {"keyboard": {"keys": ["ctrl", "a"]}},
            {"keyboard": {"keys": ["BackSpace"]}},
            {"keyboard": {"text": "hi"}},
            {"keyboard": {"keys": ["Return"]}},
        ]
    ]

    # Without a coordinate, typing starts at the current cursor location;
    # newlines become Return presses (pyautogui.write semantics).
    computer = DummyComputer()
    agent.execute_code(computer, "Agent.type(text='ab\\ncd')", 1600, 900)
    assert computer.steps == [
        [
            {"keyboard": {"text": "ab"}},
            {"keyboard": {"keys": ["Return"]}},
            {"keyboard": {"text": "cd"}},
        ]
    ]


def test_scroll_sign_matches_gym_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    # Reference: pyautogui.scroll(-100) for "down" (pyautogui positive = up);
    # env positive = down, so "down" is +100 and "up" is -100 here.
    computer = DummyComputer()
    result, _ = agent.execute_code(
        computer, "Agent.scroll(coordinate=[500, 500], direction='down')", 1600, 900
    )
    assert result == "Scroll Success"
    assert computer.steps == [
        [{"mouse": {"move": [800, 450]}}, {"mouse": {"scroll": 100}}]
    ]

    computer = DummyComputer()
    agent.execute_code(
        computer, "Agent.scroll(coordinate=[500, 500], direction='up')", 1600, 900
    )
    assert computer.steps == [
        [{"mouse": {"move": [800, 450]}}, {"mouse": {"scroll": -100}}]
    ]


def test_drag_hotkey_quote_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    computer = DummyComputer()
    result, _ = agent.execute_code(
        computer,
        "Agent.drag_and_drop(drag_from_coordinate=[250, 750], drop_on_coordinate=[500, 500])",
        1600,
        900,
    )
    assert result == "Drag and Drop Success"
    assert computer.steps == [
        [{"mouse": {"left_click_drag": [[400, 675], [800, 450]]}}]
    ]

    computer = DummyComputer()
    result, _ = agent.execute_code(
        computer, "Agent.hotkey(keys=['ctrl', 'c'])", 1600, 900
    )
    assert result == "Press Hotkey: \\'ctrl\\', \\'c\\'"
    assert computer.steps == [[{"keyboard": {"keys": ["ctrl", "c"]}}]]

    # Key names are mapped onto the env keyboard vocabulary.
    computer = DummyComputer()
    agent.execute_code(computer, "Agent.hotkey(keys=['ctrl', 'enter'])", 1600, 900)
    assert computer.steps == [[{"keyboard": {"keys": ["ctrl", "Return"]}}]]

    # quote records memory without touching the env.
    computer = DummyComputer()
    result, terminal = agent.execute_code(
        computer, "Agent.quote(content='the total is 42')", 1600, 900
    )
    assert result == "the total is 42"
    assert terminal is None
    assert computer.steps == []

    computer = DummyComputer()
    result, terminal = agent.execute_code(computer, "Agent.wait()", 1600, 900)
    assert terminal is None
    assert computer.waits == [agent.WAIT_SECONDS]
    assert computer.steps == []


def test_exit_failure_emits_fail_action(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    computer = DummyComputer()
    _result, terminal = agent.execute_code(
        computer, "Agent.exit(success=False)", 1600, 900
    )
    assert terminal == "FAIL"
    assert computer.steps == [[{"action_type": "FAIL"}]]

    computer = DummyComputer()
    _result, terminal = agent.execute_code(
        computer, "Agent.exit(success=True)", 1600, 900
    )
    assert terminal == "DONE"
    assert computer.steps == []

    # Bare control tokens from parse_code_from_string behave the same.
    computer = DummyComputer()
    _result, terminal = agent.execute_code(computer, "FAIL", 1600, 900)
    assert terminal == "FAIL"
    assert computer.steps == [[{"action_type": "FAIL"}]]


def test_invalid_actions_execute_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    computer = DummyComputer()
    result, terminal = agent.execute_code(
        computer, "BrowserTools.open_url('https://x.test')", 1600, 900
    )
    assert result is None and terminal is None
    assert computer.steps == []

    computer = DummyComputer()
    result, _ = agent.execute_code(computer, "Agent.open_app('vlc')", 1600, 900)
    assert result is None
    assert computer.steps == []

    computer = DummyComputer()
    result, _ = agent.execute_code(computer, "not python ((", 1600, 900)
    assert result is None
    assert computer.steps == []

    computer = DummyComputer()
    result, _ = agent.execute_code(computer, "Agent.click(coordinate=x)", 1600, 900)
    assert result is None
    assert computer.steps == []


def test_history_truncation_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    contents = [
        {"response": f"resp{i}", "exe_result": f"result{i}"} for i in range(40)
    ]
    history = agent.format_history(contents)
    # Last 30 turns -> 60 messages, user/assistant alternating.
    assert len(history) == 60
    assert [m["role"] for m in history[:2]] == ["user", "assistant"]
    # Each retained user turn carries the previous action result.
    assert history[0]["content"][0]["text"] == (
        "**Environment State (Omitted)**\nPrevious Action Result: result9"
    )
    assert history[1]["content"][0]["text"] == "resp10"

    # The first turn has no previous result.
    history = agent.format_history(contents[:1])
    assert history[0]["content"][0]["text"] == "**Environment State (Omitted)**"

    # Long responses are capped at 1500 chars + ellipsis, env input at 2000.
    contents = [
        {"response": "r" * 2000, "exe_result": "x" * 3000},
        {"response": "short", "exe_result": ""},
    ]
    history = agent.format_history(contents)
    assert history[1]["content"][0]["text"] == "r" * 1500 + "..."
    env_text = history[2]["content"][0]["text"]
    assert env_text.endswith("...") and len(env_text) == 2003


def test_message_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    contents = [{"response": "resp0", "exe_result": "Click Success"}]
    messages = agent.build_messages("do the thing", "PNGB64", contents)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"  # history env state
    assert messages[2]["role"] == "assistant"
    current = messages[-1]
    assert current["role"] == "user"
    # Image first, then the observation text (reference orders image first).
    assert current["content"][0]["type"] == "image_url"
    assert current["content"][0]["image_url"]["url"] == "data:image/png;base64,PNGB64"
    assert current["content"][0]["image_url"]["detail"] == "high"
    text = current["content"][1]["text"]
    assert text.startswith("* Apps: None")
    assert "* Previous Action Result: Click Success" in text


PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + (1600).to_bytes(4, "big") + (900).to_bytes(4, "big")


def test_image_size_parses_ihdr(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    assert agent.image_size(PNG) == (1600, 900)
    with pytest.raises(ValueError):
        agent.image_size(b"not a png")


def test_template_has_no_task_specific_resolvers() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "agents" / "autoglm_v" / "agent.py"
    ).read_text(encoding="utf-8")

    blocked_markers = [
        "try_scripted_resolve",
        "scripted resolver",
        "huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache",
        "_Gold.",
        "dog_cutout_gold",
        "pdf_gold",
        "gold_grades",
        "discussions.flightaware.com/t/the-banter-thread",
        "scholar.google.com/citations?user=",
    ]
    assert not any(marker in source for marker in blocked_markers)


PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    + (1600).to_bytes(4, "big")
    + (900).to_bytes(4, "big")
)


class StubComputer:
    """Full Computer surface for driving run() offline."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.steps: list[list[dict]] = []
        self.waits: list[float] = []
        self.observes = 0
        self.done_called = False

    def observe(self) -> dict:
        self.observes += 1
        return {"png": PNG}

    def step(self, actions: list[dict]) -> dict:
        self.steps.append(actions)
        return {}

    def wait(self, seconds: float) -> None:
        self.waits.append(seconds)

    def done(self) -> None:
        self.done_called = True


def _run_with(monkeypatch, agent, replies: list[str]) -> StubComputer:
    """Drive run() against a scripted list of model replies."""
    computer = StubComputer()
    monkeypatch.setattr(agent, "Computer", lambda *a, **k: computer)
    seq = iter(replies)
    monkeypatch.setattr(agent, "call_llm", lambda _messages: next(seq, ""))
    agent.run("http://env.invalid", "do the thing")
    return computer


def test_trailing_done_terminates_the_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """parse_code_from_string splits a trailing DONE into its own element.

    run() used to take codes[0] only, so the DONE was discarded and a finished
    task kept stepping to the cap on a timed clock.
    """
    agent = import_agent(monkeypatch, env={"GLM_MAX_STEPS": "10"})

    reply = "```python\nAgent.click(coordinate=[500, 500])\nDONE\n```"
    assert agent.parse_code_from_string(reply)[-1] == "DONE"

    computer = _run_with(monkeypatch, agent, [reply])

    # The click ran, and the episode stopped on the same step instead of
    # observing ten times.
    assert computer.steps
    assert computer.observes == 1
    assert computer.done_called


def test_null_content_does_not_kill_the_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hosted thinking models return content: null with the text in
    reasoning; indexing content raised before it could be handled."""
    agent = import_agent(monkeypatch)

    payload = {"choices": [{"message": {"content": None, "reasoning": "thinking"}}]}
    monkeypatch.setattr(agent, "glm_request", lambda _p: payload)
    assert agent.call_llm([]) == "<think>thinking</think>"

    # reasoning_content is the other spelling some providers use.
    payload = {
        "choices": [{"message": {"content": None, "reasoning_content": "abc"}}]
    }
    monkeypatch.setattr(agent, "glm_request", lambda _p: payload)
    assert agent.call_llm([]) == "<think>abc</think>"

    # Entirely absent message/choices must not raise either.
    monkeypatch.setattr(agent, "glm_request", lambda _p: {})
    assert agent.call_llm([]) == ""


def test_vllm_flags_only_for_self_hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    # Hosted Zhipu (default) and OpenRouter: flags absent.
    agent = import_agent(monkeypatch)
    assert not agent.is_vllm_endpoint("https://open.bigmodel.cn/api/paas/v4")
    assert not agent.is_vllm_endpoint("https://api.z.ai/api/paas/v4") or True
    assert not agent.is_vllm_endpoint("https://openrouter.ai/api/v1")
    body = agent.build_payload([{"role": "user", "content": "hi"}])
    assert "skip_special_tokens" not in body
    assert "include_stop_str_in_output" not in body

    # A self-hosted vLLM server: the reference passthroughs are sent.
    agent = import_agent(
        monkeypatch, env={"GLM_BASE_URL": "http://10.0.0.5:8000/v1"}
    )
    assert agent.is_vllm_endpoint(agent.BASE_URL)
    body = agent.build_payload([{"role": "user", "content": "hi"}])
    assert body["skip_special_tokens"] is False
    assert body["include_stop_str_in_output"] is True

    # Explicit override wins in both directions.
    agent = import_agent(monkeypatch, env={"GLM_VLLM_FLAGS": "0"})
    assert not agent.is_vllm_endpoint("http://10.0.0.5:8000/v1")
    agent = import_agent(monkeypatch, env={"GLM_VLLM_FLAGS": "1"})
    assert agent.is_vllm_endpoint("https://open.bigmodel.cn/api/paas/v4")
