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


def import_agent(monkeypatch: pytest.MonkeyPatch):
    for var in (
        "META_MODEL",
        "META_MODE",
        "META_BASE_URL",
        "META_MAX_STEPS",
        "META_HTTP_TIMEOUT",
        "META_ENV_HTTP_TIMEOUT",
        "META_MAX_RETRIES",
        "META_IMAGE_MAX",
        "META_IMAGE_KEEP",
        "META_CONTEXT_LIMIT",
        "META_CLIENT_PASSWORD",
        "META_SESSION_ID",
        "CS_MAX_STEPS",
    ):
        monkeypatch.delenv(var, raising=False)

    path = Path(__file__).resolve().parents[1] / "agents" / "meta" / "agent.py"
    spec = importlib.util.spec_from_file_location("meta_agent_test", path)
    assert spec is not None and spec.loader is not None
    agent = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = agent
    spec.loader.exec_module(agent)
    return agent


def test_tool_schema_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    assert agent.MODEL == "super_nova_ext"
    assert agent.BASE_URL == "https://api.ai.meta.com/v1"
    assert agent.MAX_STEPS == 100
    assert agent.IMAGE_MAX == 30
    assert agent.IMAGE_KEEP == 5

    tool = agent.COMPUTER_TOOL
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "computer"
    assert tool["function"]["parameters"]["properties"]["action"]["enum"] == [
        "click",
        "double_click",
        "right_click",
        "move",
        "drag",
        "type",
        "key",
        "scroll",
        "wait",
        "screenshot",
        "terminate",
    ]


def test_payload_has_no_sampling_params(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    payload = agent.build_payload([{"role": "user", "content": "hi"}])
    assert set(payload) == {"model", "messages", "tools", "tool_choice"}
    assert payload["tool_choice"] == "auto"
    assert payload["tools"] == [agent.COMPUTER_TOOL]

    probe = agent.build_payload([{"role": "user", "content": "hi"}], include_tools=False)
    assert set(probe) == {"model", "messages"}


def test_action_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    computer = DummyComputer()
    outcome = agent.convert_and_execute(
        computer, "computer", {"action": "click", "x": 500, "y": 500}, 1600, 900
    )
    assert not outcome.terminal
    assert computer.steps == [[{"mouse": {"left_click": [800, 450]}}]]

    computer = DummyComputer()
    agent.convert_and_execute(
        computer,
        "computer",
        {"action": "drag", "x": 250, "y": 750, "to_x": 500, "to_y": 500},
        1600,
        900,
    )
    assert computer.steps == [
        [{"mouse": {"left_click_drag": [[400, 675], [800, 450]]}}]
    ]

    computer = DummyComputer()
    agent.convert_and_execute(
        computer, "computer", {"action": "double_click", "x": 0, "y": 1000}, 1600, 900
    )
    assert computer.steps == [[{"mouse": {"double_click": [0, 899]}}]]

    computer = DummyComputer()
    agent.convert_and_execute(
        computer, "computer", {"action": "key", "keys": ["cmd", "l"]}, 1600, 900
    )
    assert computer.steps == [[{"keyboard": {"keys": ["ctrl", "l"]}}]]

    computer = DummyComputer()
    agent.convert_and_execute(
        computer, "computer", {"action": "key", "keys": ["enter"]}, 1600, 900
    )
    assert computer.steps == [[{"keyboard": {"keys": ["Return"]}}]]

    computer = DummyComputer()
    agent.convert_and_execute(
        computer, "computer", {"action": "type", "text": "ab\ncd"}, 1600, 900
    )
    assert computer.steps == [
        [
            {"keyboard": {"text": "ab"}},
            {"keyboard": {"keys": ["Return"]}},
            {"keyboard": {"text": "cd"}},
        ]
    ]

    computer = DummyComputer()
    agent.convert_and_execute(
        computer, "computer", {"action": "type", "text": "it’s — fine"}, 1600, 900
    )
    assert computer.steps == [[{"keyboard": {"text": "it's - fine"}}]]


def test_scroll_sign_matches_gym_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    computer = DummyComputer()
    agent.convert_and_execute(
        computer,
        "computer",
        {"action": "scroll", "scroll_direction": "down", "amount": 5, "x": 500, "y": 500},
        1600,
        900,
    )
    assert computer.steps == [
        [{"mouse": {"move": [800, 450]}}, {"mouse": {"scroll": 5}}]
    ]

    computer = DummyComputer()
    agent.convert_and_execute(
        computer, "computer", {"action": "scroll", "scroll_direction": "up"}, 1600, 900
    )
    assert computer.steps == [[{"mouse": {"scroll": -3}}]]


def test_invalid_arguments_execute_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    computer = DummyComputer()
    outcome = agent.convert_and_execute(
        computer, "computer", {"action": "click", "x": None, "y": 200}, 1600, 900
    )
    assert not outcome.executed
    assert computer.steps == []

    computer = DummyComputer()
    outcome = agent.convert_and_execute(
        computer, "computer", {"action": "fly"}, 1600, 900
    )
    assert not outcome.executed
    assert computer.steps == []

    computer = DummyComputer()
    outcome = agent.convert_and_execute(computer, "browser", {"action": "click"}, 1600, 900)
    assert not outcome.executed
    assert "Unknown tool" in outcome.result_text
    assert computer.steps == []

    assert agent.clamp_xy(float("nan"), 5, 1600, 900) == (None, None)
    assert agent.clamp_xy(2000, -50, 1600, 900) == (1599, 0)


def test_terminate_failure_emits_fail_action(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    computer = DummyComputer()
    outcome = agent.convert_and_execute(
        computer, "computer", {"action": "terminate", "status": "failure"}, 1600, 900
    )
    assert outcome.terminal is True
    assert computer.steps == [[{"action_type": "FAIL"}]]

    computer = DummyComputer()
    outcome = agent.convert_and_execute(
        computer, "computer", {"action": "terminate", "status": "success"}, 1600, 900
    )
    assert outcome.terminal is True
    assert computer.steps == []


def test_wait_uses_computer_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    computer = DummyComputer()
    agent.convert_and_execute(
        computer, "computer", {"action": "wait", "duration": 90}, 1600, 900
    )
    assert computer.waits == [30.0]
    assert computer.steps == []

    computer = DummyComputer()
    agent.convert_and_execute(computer, "computer", {"action": "wait"}, 1600, 900)
    assert computer.waits == [1.0]


PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + (1600).to_bytes(4, "big") + (900).to_bytes(4, "big")


def test_image_sawtooth_preserves_task_message(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    history = agent.History()
    history.messages.append({"role": "system", "content": "sys"})
    history.messages.append({"role": "user", "content": "Task: do the thing"})
    for i in range(agent.IMAGE_MAX):
        history.append_image_user_message(f"shot {i}", b"png-bytes")

    image_parts = [
        part
        for msg in history.messages
        if isinstance(msg.get("content"), list)
        for part in msg["content"]
        if part.get("type") == "image_url"
    ]
    assert len(image_parts) == agent.IMAGE_KEEP
    assert history.messages[1] == {"role": "user", "content": "Task: do the thing"}
    replaced = history.messages[2]["content"]
    assert replaced[0] == {"type": "text", "text": "shot 0"}
    assert replaced[1] == {"type": "text", "text": agent.IMAGE_PLACEHOLDER}


def test_compact_tool_outputs_keeps_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    history = agent.History()
    history.append_tool_result("a", "first")
    history.append_tool_result("b", "second")
    history.append_tool_result("c", "third")
    history.compact_tool_outputs()

    assert history.messages[0]["content"] == agent.TOOL_OUTPUT_PLACEHOLDER
    assert history.messages[1]["content"] == agent.TOOL_OUTPUT_PLACEHOLDER
    assert history.messages[2]["content"] == "third"


def test_error_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    assert agent.should_retry_status(429) is True
    assert agent.should_retry_status(503) is True
    assert agent.should_retry_status(401) is False
    assert agent.is_context_overflow(400, "prompt is too long") is True
    assert agent.is_context_overflow(400, "maximum token count exceeded") is True
    assert agent.is_context_overflow(422, "context") is False
    assert agent.is_content_policy(400, "violates content policy") is True
    assert agent.is_content_policy(400, "bad argument") is False


def test_image_size_parses_ihdr(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    assert agent.image_size(PNG) == (1600, 900)
    with pytest.raises(ValueError):
        agent.image_size(b"not a png")


def test_pyautogui_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    text = "I will click the button.\n```python\npyautogui.click(500, 500)\n```"
    assert agent.extract_pyautogui_actions(text) == ["pyautogui.click(500, 500)"]

    # Last fenced block wins; control token split out alongside code.
    text = (
        "```python\npyautogui.press('enter')\n```\nthen\n"
        "```python\npyautogui.click(10, 10)\nDONE\n```"
    )
    assert agent.extract_pyautogui_actions(text) == ["pyautogui.click(10, 10)", "DONE"]

    # A ```pyautogui fence must not lose its first characters to the tag regex.
    text = "```pyautogui\npyautogui.scroll(-3)\n```"
    assert agent.extract_pyautogui_actions(text) == ["pyautogui.scroll(-3)"]

    assert agent.extract_pyautogui_actions("WAIT") == ["WAIT"]
    assert agent.extract_pyautogui_actions("thinking about it...") == []

    # Printed control tokens count as the token (observed in live runs).
    text = "```python\npyautogui.click(5, 5)\nprint('DONE')\n```"
    assert agent.extract_pyautogui_actions(text) == ["pyautogui.click(5, 5)", "DONE"]


def test_pyautogui_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    # Normalized coordinates scale exactly like the tool mode.
    segments, err = agent.translate_pyautogui_block(
        "import pyautogui\npyautogui.click(500, 500)", 1600, 900
    )
    assert err is None
    assert segments == [{"actions": [{"mouse": {"left_click": [800, 450]}}]}]

    # Pixel-space coordinates (>1000) pass through unrescaled.
    segments, _ = agent.translate_pyautogui_block("pyautogui.click(1200, 700)", 1600, 900)
    assert segments == [{"actions": [{"mouse": {"left_click": [1200, 700]}}]}]

    # scroll: sign inverted for the env, clicks clamped to 30.
    segments, _ = agent.translate_pyautogui_block("pyautogui.scroll(-300)", 1600, 900)
    assert segments == [{"actions": [{"mouse": {"scroll": 30}}]}]

    # write transliterates and splits newlines with Return presses.
    segments, _ = agent.translate_pyautogui_block(
        "pyautogui.write('it’s\\nok')", 1600, 900
    )
    assert segments == [
        {
            "actions": [
                {"keyboard": {"text": "it's"}},
                {"keyboard": {"keys": ["Return"]}},
                {"keyboard": {"text": "ok"}},
            ]
        }
    ]

    # hotkey maps the model's key vocabulary (cmd -> ctrl).
    segments, _ = agent.translate_pyautogui_block("pyautogui.hotkey('cmd', 'l')", 1600, 900)
    assert segments == [{"actions": [{"keyboard": {"keys": ["ctrl", "l"]}}]}]

    # Positional clicks arg (x, y, clicks, interval, button) maps correctly.
    segments, _ = agent.translate_pyautogui_block("pyautogui.click(500, 500, 2)", 1600, 900)
    assert segments == [{"actions": [{"mouse": {"double_click": [800, 450]}}]}]
    segments, _ = agent.translate_pyautogui_block(
        "pyautogui.click(500, 500, 1, 0.0, 'right')", 1600, 900
    )
    assert segments == [{"actions": [{"mouse": {"right_click": [800, 450]}}]}]

    # pyautogui.sleep is an alias for time.sleep.
    segments, _ = agent.translate_pyautogui_block("pyautogui.sleep(1.5)", 1600, 900)
    assert segments == [{"wait": 1.5}]

    # for-loops over literal ranges unroll (the source agent runs them in-guest).
    segments, _ = agent.translate_pyautogui_block(
        "for i in range(3):\n    pyautogui.press('down')", 1600, 900
    )
    assert segments == [
        {"actions": [{"keyboard": {"keys": ["Down"]}}] * 3}
    ]
    _, err = agent.translate_pyautogui_block(
        "for i in range(3):\n    pyautogui.click(i, 0)", 1600, 900
    )
    assert err is not None and "loop variable" in err
    _, err = agent.translate_pyautogui_block(
        "for i in range(2):\n    for j in range(2):\n        pyautogui.press('down')",
        1600,
        900,
    )
    assert err is not None and "nested" in err

    # hscroll becomes shift+wheel, positive = right (no sign inversion).
    segments, _ = agent.translate_pyautogui_block("pyautogui.hscroll(5)", 1600, 900)
    assert segments == [
        {
            "actions": [
                {"keyboard": {"key_down": "shift"}},
                {"mouse": {"scroll": 5}},
                {"keyboard": {"key_up": "shift"}},
            ]
        }
    ]

    # time.sleep splits the block into step/wait segments.
    segments, _ = agent.translate_pyautogui_block(
        "pyautogui.click(0, 0)\nimport time\ntime.sleep(2)\npyautogui.press('enter')",
        1600,
        900,
    )
    assert segments == [
        {"actions": [{"mouse": {"left_click": [0, 0]}}]},
        {"wait": 2.0},
        {"actions": [{"keyboard": {"keys": ["Return"]}}]},
    ]

    # dragTo anchored by a prior moveTo; unanchored dragTo is rejected.
    segments, _ = agent.translate_pyautogui_block(
        "pyautogui.moveTo(250, 750)\npyautogui.dragTo(500, 500)", 1600, 900
    )
    assert segments == [
        {
            "actions": [
                {"mouse": {"move": [400, 675]}},
                {"mouse": {"left_click_drag": [[400, 675], [800, 450]]}},
            ]
        }
    ]
    _, err = agent.translate_pyautogui_block("pyautogui.dragTo(500, 500)", 1600, 900)
    assert err is not None and "prior moveTo" in err

    # Unsupported constructs are rejected, never silently dropped.
    _, err = agent.translate_pyautogui_block("for i in range(3):\n    pass", 1600, 900)
    assert err is not None
    _, err = agent.translate_pyautogui_block("pyautogui.screenshot('x.png')", 1600, 900)
    assert err is not None
    _, err = agent.translate_pyautogui_block("pyautogui.click(x, y)", 1600, 900)
    assert err is not None
    _, err = agent.translate_pyautogui_block("not python ((", 1600, 900)
    assert err is not None and "not valid Python" in err


def test_responses_payload_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    payload = agent.responses_payload([{"role": "user", "content": []}])
    # Exactly the keys the source's _call_llm sends -- in particular NO
    # `reasoning` (no effort/summary) and no sampling params.
    assert set(payload) == {
        "model",
        "input",
        "store",
        "include",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
    }
    assert "reasoning" not in payload
    assert payload["store"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is False
    assert "instructions" not in payload
    assert "previous_response_id" not in payload


def test_responses_tools_match_flattened_computer_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch)

    # gui mode's toolset: the single `computer` tool, flattened for the
    # Responses API exactly as the source's _flatten_tool does it.
    assert agent.RESPONSES_TOOLS == [
        {
            "type": "function",
            "name": "computer",
            "description": agent.COMPUTER_TOOL["function"]["description"],
            "parameters": agent.COMPUTER_TOOL["function"]["parameters"],
        }
    ]
    # terminate goes through the computer tool's own action/status, not a
    # separate stop tool.
    props = agent.RESPONSES_TOOLS[0]["parameters"]["properties"]
    assert "terminate" in props["action"]["enum"]
    assert props["status"]["enum"] == ["success", "failure"]


def test_responses_surplus_function_calls_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch)

    history = agent.ResponsesHistory()
    history.append_output_items(
        [
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "abc"},
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "clicking"}],
                "status": "completed",
            },
            {"type": "function_call", "call_id": "c1", "name": "computer", "arguments": "{}"},
            {"type": "function_call", "call_id": "c2", "name": "computer", "arguments": "{}"},
            {"type": "function_call", "call_id": "c3", "name": "computer", "arguments": "{}"},
        ]
    )
    calls = [i for i in history.items if i.get("type") == "function_call"]
    assert [c["call_id"] for c in calls] == ["c1"]
    # Reasoning and message items are all echoed back, in order.
    assert [i.get("type") or i.get("role") for i in history.items] == [
        "reasoning",
        "message",
        "function_call",
    ]


def test_responses_trailing_reasoning_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    history = agent.ResponsesHistory()
    history.append_output_items(
        [{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "abc"}]
    )
    assert history.items[-1] == {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "(continuing)"}],
    }

    # A reasoning item followed by a function_call needs no guard.
    history = agent.ResponsesHistory()
    history.append_output_items(
        [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "function_call", "call_id": "c1", "name": "computer", "arguments": "{}"},
        ]
    )
    assert history.items[-1]["type"] == "function_call"


def test_resp_item_drops_none_values(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    item = {"type": "function_call", "call_id": "c1", "name": "computer", "id": None, "status": None}
    assert agent.resp_item(item) == {
        "type": "function_call",
        "call_id": "c1",
        "name": "computer",
    }


def test_responses_image_sawtooth(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    history = agent.ResponsesHistory()
    history.items.append(
        {"role": "system", "content": [{"type": "input_text", "text": "sys"}]}
    )
    history.items.append(
        {"role": "user", "content": [{"type": "input_text", "text": "Task: do the thing"}]}
    )
    for i in range(agent.IMAGE_MAX):
        history.append_image_user_message(f"shot {i}", b"png-bytes")

    image_parts = [
        part
        for item in history.items
        if isinstance(item.get("content"), list)
        for part in item["content"]
        if part.get("type") == "input_image"
    ]
    # Sawtooth: reaching IMAGE_MAX (30) trims to IMAGE_KEEP (5).
    assert len(image_parts) == agent.IMAGE_KEEP
    assert len(history.image_item_indices) == agent.IMAGE_KEEP
    # The task message is untouched, and stale messages keep sibling text.
    assert history.items[1]["content"] == [
        {"type": "input_text", "text": "Task: do the thing"}
    ]
    assert history.items[2]["content"] == [
        {"type": "input_text", "text": "shot 0"},
        {"type": "input_text", "text": agent.IMAGE_PLACEHOLDER},
    ]


def test_responses_compact_tool_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    history = agent.ResponsesHistory()
    history.items.append({"type": "reasoning", "id": "rs_1", "summary": []})
    history.append_tool_result("a", "first")
    history.append_tool_result("b", "second")
    history.append_tool_result("c", "third")
    history.compact_tool_outputs()

    assert history.items[0] == {"type": "reasoning", "id": "rs_1", "summary": []}
    assert history.items[1]["output"] == agent.TOOL_OUTPUT_PLACEHOLDER
    assert history.items[2]["output"] == agent.TOOL_OUTPUT_PLACEHOLDER
    assert history.items[3]["output"] == "third"


def test_template_has_no_task_specific_resolvers() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "agents" / "meta" / "agent.py"
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
