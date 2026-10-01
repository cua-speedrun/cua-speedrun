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
        "MINIMAX_MODEL",
        "MINIMAX_BASE_URL",
        "MINIMAX_HTTP_TIMEOUT",
        "MINIMAX_HTTP_MAX_RETRIES",
        "MINIMAX_MAX_LLM_RETRIES",
        "MINIMAX_MAX_TOKENS",
        "MINIMAX_TEMPERATURE",
        "MINIMAX_ONLY_N_MOST_RECENT_IMAGES",
        "MINIMAX_IMAGE_TRUNCATION_THRESHOLD",
        "MINIMAX_CLIENT_PASSWORD",
        "MINIMAX_MAX_STEPS",
        "CS_MAX_STEPS",
    ):
        monkeypatch.delenv(var, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)

    path = Path(__file__).resolve().parents[1] / "agents" / "minimax_m3" / "agent.py"
    spec = importlib.util.spec_from_file_location("minimax_m3_agent_test", path)
    assert spec is not None and spec.loader is not None
    agent = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = agent
    spec.loader.exec_module(agent)
    return agent


def test_max_steps_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # MINIMAX_MAX_STEPS is the user-settable knob and wins over the
    # harness-internal CS_MAX_STEPS fallback.
    agent = import_agent(monkeypatch, env={"CS_MAX_STEPS": "120"})
    assert agent.MAX_STEPS == 120
    agent = import_agent(
        monkeypatch, env={"CS_MAX_STEPS": "120", "MINIMAX_MAX_STEPS": "77"}
    )
    assert agent.MAX_STEPS == 77


def test_defaults_match_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    assert agent.MODEL == "MiniMax-M3"
    assert agent.BASE_URL == "https://api.minimax.io/anthropic"
    assert agent.API_URL == "https://api.minimax.io/anthropic/v1/messages"
    assert agent.MAX_STEPS == 100
    assert agent.MAX_TOKENS == 8192
    assert agent.TEMPERATURE == 0.6
    assert agent.MAX_LLM_RETRIES == 2
    assert agent.ONLY_N_MOST_RECENT_IMAGES == 10
    assert agent.IMAGE_TRUNCATION_THRESHOLD == 20
    assert agent.STOP_SEQUENCES == [
        "</tool_call>",
        "Perform the next action. Perform",
    ]
    # The sudo password must match this platform's guest image
    # (benchmarks/osworld-image.json "ssh_password"), not the upstream
    # reference's, or every sudo task fails.
    assert agent.CLIENT_PASSWORD == "password"
    # The M3 system prompt is templated with date + password, [0, 1000]
    # normalized coordinates.
    prompt = agent.system_prompt()
    assert "password" in prompt
    assert "[INFEASIBLE]" in prompt
    assert "normalized integer values in [0, 1000]" in prompt


def test_payload_shape_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    body = agent.build_request_body([{"role": "user", "content": "hi"}])
    # Exactly what the reference _build_request_body sends with its defaults:
    # temperature + stop_sequences, and NO top_p / thinking.
    assert set(body) == {
        "model",
        "messages",
        "max_tokens",
        "system",
        "temperature",
        "stop_sequences",
    }
    assert body["model"] == "MiniMax-M3"
    assert body["max_tokens"] == 8192
    assert body["temperature"] == 0.6
    assert body["stop_sequences"] == [
        "</tool_call>",
        "Perform the next action. Perform",
    ]
    assert "top_p" not in body


def test_auth_header_autodetection(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-token")
    headers = agent.anthropic_headers()
    assert headers["x-api-key"] == "sk-test-token"
    assert "Authorization" not in headers

    monkeypatch.setenv("MINIMAX_API_KEY", "eyJhbGciOi.jwt.token")
    headers = agent.anthropic_headers()
    assert headers["Authorization"] == "Bearer eyJhbGciOi.jwt.token"
    assert "x-api-key" not in headers


def tool_call(action: str, **kwargs) -> str:
    import json

    args = {"action": action, **kwargs}
    return (
        "Action: do it\n<tool_call>\n"
        + json.dumps({"name": "computer", "arguments": args})
        + "\n</tool_call>"
    )


def parse_and_execute(agent, computer, response: str, width=1600, height=900, cursor=None):
    _instruction, items = parse(agent, response, width, height, cursor)
    return agent.execute_items(computer, items)


def parse(agent, response: str, width=1600, height=900, cursor=None):
    return agent.parse_m3_response(response, width, height, cursor)


def test_action_translation_and_coordinate_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch)

    computer = DummyComputer()
    parse_and_execute(
        agent, computer, tool_call("left_click", coordinate=[500, 500])
    )
    assert computer.steps == [[{"mouse": {"left_click": [800, 450]}}]]

    # x=1000 stays inside the framebuffer.
    computer = DummyComputer()
    parse_and_execute(
        agent, computer, tool_call("double_click", coordinate=[1000, 0])
    )
    assert computer.steps == [[{"mouse": {"double_click": [1599, 0]}}]]

    # Modifier text wraps the click in keys_down / keys_up.
    computer = DummyComputer()
    parse_and_execute(
        agent,
        computer,
        tool_call("left_click", coordinate=[500, 500], text="ctrl+shift"),
    )
    assert computer.steps == [
        [
            {"keyboard": {"keys_down": ["ctrl", "shift"]}},
            {"mouse": {"left_click": [800, 450]}},
            {"keyboard": {"keys_up": ["shift", "ctrl"]}},
        ]
    ]

    # key combos go through the reference key_conversion plus env mapping.
    computer = DummyComputer()
    parse_and_execute(agent, computer, tool_call("key", text="ctrl+s"))
    assert computer.steps == [[{"keyboard": {"keys": ["ctrl", "s"]}}]]

    computer = DummyComputer()
    parse_and_execute(agent, computer, tool_call("key", text="Enter"))
    assert computer.steps == [[{"keyboard": {"keys": ["Return"]}}]]

    # type splits newlines into Return presses (pyautogui semantics).
    computer = DummyComputer()
    parse_and_execute(agent, computer, tool_call("type", text="ab\ncd"))
    assert computer.steps == [
        [
            {"keyboard": {"text": "ab"}},
            {"keyboard": {"keys": ["Return"]}},
            {"keyboard": {"text": "cd"}},
        ]
    ]

    # left_click_drag with an explicit start.
    computer = DummyComputer()
    parse_and_execute(
        agent,
        computer,
        tool_call(
            "left_click_drag", coordinate=[500, 500], start_coordinate=[250, 750]
        ),
    )
    assert computer.steps == [
        [{"mouse": {"left_click_drag": [[400, 675], [800, 450]]}}]
    ]

    # ... and without a start it anchors on the tracked cursor position.
    computer = DummyComputer()
    cursor: list = []
    parse_and_execute(
        agent, computer, tool_call("left_click", coordinate=[250, 750]), cursor=cursor
    )
    parse_and_execute(
        agent, computer, tool_call("left_click_drag", coordinate=[500, 500]), cursor=cursor
    )
    assert computer.steps == [
        [{"mouse": {"left_click": [400, 675]}}],
        [{"mouse": {"left_click_drag": [[400, 675], [800, 450]]}}],
    ]

    computer = DummyComputer()
    parse_and_execute(agent, computer, tool_call("mouse_move", coordinate=[0, 1000]))
    assert computer.steps == [[{"mouse": {"move": [0, 899]}}]]


def test_scroll_sign_matches_gym_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    # pyautogui positive = up; the env positive = down. "down" keeps the
    # positive amount, "up" negates it.
    computer = DummyComputer()
    parse_and_execute(
        agent,
        computer,
        tool_call(
            "scroll", coordinate=[500, 500], scroll_direction="down", scroll_amount=5
        ),
    )
    assert computer.steps == [
        [{"mouse": {"move": [800, 450]}}, {"mouse": {"scroll": 5}}]
    ]

    computer = DummyComputer()
    parse_and_execute(
        agent,
        computer,
        tool_call(
            "scroll", coordinate=[500, 500], scroll_direction="up", scroll_amount=5
        ),
    )
    assert computer.steps == [
        [{"mouse": {"move": [800, 450]}}, {"mouse": {"scroll": -5}}]
    ]

    # Horizontal scrolling is shift+wheel, positive = right.
    computer = DummyComputer()
    parse_and_execute(
        agent,
        computer,
        tool_call(
            "scroll", coordinate=[500, 500], scroll_direction="left", scroll_amount=3
        ),
    )
    assert computer.steps == [
        [
            {"mouse": {"move": [800, 450]}},
            {"keyboard": {"keys_down": ["shift"]}},
            {"mouse": {"scroll": -3}},
            {"keyboard": {"keys_up": ["shift"]}},
        ]
    ]


def test_terminate_failure_and_infeasible_emit_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch)

    computer = DummyComputer()
    terminal = parse_and_execute(
        agent, computer, tool_call("terminate", status="failure")
    )
    assert terminal == "FAIL"
    assert computer.steps == [[{"action_type": "FAIL"}]]

    computer = DummyComputer()
    terminal = parse_and_execute(agent, computer, tool_call("fail"))
    assert terminal == "FAIL"
    assert computer.steps == [[{"action_type": "FAIL"}]]

    # The [INFEASIBLE] token short-circuits the whole response to FAIL.
    computer = DummyComputer()
    instruction, items = parse(agent, "I believe this is [INFEASIBLE] to do.")
    assert instruction == "[INFEASIBLE]"
    assert items == ["FAIL"]
    terminal = agent.execute_items(computer, items)
    assert terminal == "FAIL"
    assert computer.steps == [[{"action_type": "FAIL"}]]

    computer = DummyComputer()
    terminal = parse_and_execute(agent, computer, tool_call("done"))
    assert terminal == "DONE"
    assert computer.steps == []

    computer = DummyComputer()
    terminal = parse_and_execute(
        agent, computer, tool_call("terminate", status="success")
    )
    assert terminal == "DONE"
    assert computer.steps == []


def test_wait_and_invalid_actions_are_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    computer = DummyComputer()
    parse_and_execute(agent, computer, tool_call("wait", duration=3))
    assert computer.waits == [0.5]
    assert computer.steps == []

    computer = DummyComputer()
    parse_and_execute(agent, computer, tool_call("screenshot"))
    assert computer.waits == [0.1]
    assert computer.steps == []

    # Invalid arguments -> no-op sleep (never a crash, never an env action).
    computer = DummyComputer()
    parse_and_execute(agent, computer, tool_call("mouse_move"))
    assert computer.waits == [0.1]
    assert computer.steps == []

    # No response, no actions.
    assert parse(agent, "") == ("", [])
    assert parse(agent, "just talking, no tool call")[1] == []


def test_failed_llm_call_does_not_poison_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed call must not append an empty assistant turn.

    The Messages API rejects an empty assistant turn with 400, and 400 is not
    retryable, so appending one made every later step 400 and append another --
    burning the whole step budget on a timed clock. The episode must end
    instead.
    """
    agent = import_agent(monkeypatch, env={"MINIMAX_MAX_STEPS": "20"})

    observes = {"n": 0}

    class StubComputer:
        def __init__(self, *_args, **_kwargs) -> None:
            self.done_called = False

        def observe(self) -> dict:
            observes["n"] += 1
            return {"png": PNG}

        def step(self, actions: list[dict]) -> dict:
            return {}

        def wait(self, seconds: float) -> None:
            pass

        def done(self) -> None:
            self.done_called = True

    def always_fails(_body: dict) -> dict:
        raise RuntimeError("upstream 503")

    monkeypatch.setattr(agent, "Computer", StubComputer)
    monkeypatch.setattr(agent, "anthropic_request", always_fails)

    agent.run("http://env.invalid", "do the thing")

    # One observation, then the episode stops -- it does not spin to the cap.
    assert observes["n"] == 1


def test_hold_key_presses_waits_and_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """hold_key must release. Emitting only keys_down latches the modifier for
    the rest of the episode and corrupts every later keystroke."""
    agent = import_agent(monkeypatch)

    computer = DummyComputer()
    parse_and_execute(
        agent, computer, tool_call("hold_key", text="shift", duration=2)
    )
    assert computer.steps == [
        [{"keyboard": {"keys_down": ["shift"]}}],
        [{"keyboard": {"keys_up": ["shift"]}}],
    ]
    assert computer.waits == [2.0]

    # Multiple keys release in reverse order, matching the modifier-wrapped
    # click path.
    computer = DummyComputer()
    parse_and_execute(
        agent, computer, tool_call("hold_key", text="ctrl+shift", duration=1)
    )
    assert computer.steps == [
        [{"keyboard": {"keys_down": ["ctrl", "shift"]}}],
        [{"keyboard": {"keys_up": ["shift", "ctrl"]}}],
    ]


def test_hold_duration_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)
    assert agent.hold_duration(None) == 1.0
    assert agent.hold_duration("not a number") == 1.0
    assert agent.hold_duration(2) == 2.0
    # A hallucinated magnitude cannot eat the task's wall clock.
    assert agent.hold_duration(9999) == 10.0
    assert agent.hold_duration(0) == 0.1


def test_bpe_artifact_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    computer = DummyComputer()
    parse_and_execute(
        agent, computer, tool_call("left_ click", coordinate=[500, 500])
    )
    assert computer.steps == [[{"mouse": {"left_click": [800, 450]}}]]

    computer = DummyComputer()
    parse_and_execute(
        agent, computer, tool_call("left_lick", coordinate=[500, 500])
    )
    assert computer.steps == [[{"mouse": {"left_click": [800, 450]}}]]

    # Truncated/short action tokens degrade to a no-op sleep.
    computer = DummyComputer()
    parse_and_execute(agent, computer, tool_call("left_"))
    assert computer.waits == [0.1]
    assert computer.steps == []


def test_stop_sequence_recovery_and_bare_json(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    # The "</tool_call>" stop sequence eats the closing tag: still parsed.
    computer = DummyComputer()
    response = (
        "Action: click it\n<tool_call>\n"
        '{"name": "computer", "arguments": {"action": "left_click", "coordinate": [500, 500]}}'
    )
    parse_and_execute(agent, computer, response)
    assert computer.steps == [[{"mouse": {"left_click": [800, 450]}}]]

    # Bare JSON without the wrapper is also accepted.
    computer = DummyComputer()
    response = '{"name": "computer", "arguments": {"action": "done"}}'
    assert parse_and_execute(agent, computer, response) == "DONE"


def test_wrap_for_history(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    # Missing closing tag is restored.
    wrapped = agent.wrap_for_history("Action: x\n<tool_call>\n{\"name\": \"computer\"}")
    assert wrapped.endswith("</tool_call>")

    # Bare JSON tool calls get re-wrapped for the chat template.
    bare = '{"name": "computer", "arguments": {"action": "done"}}'
    wrapped = agent.wrap_for_history(f"Action: finish\n{bare}")
    assert "<tool_call>" in wrapped and "</tool_call>" in wrapped
    assert bare in wrapped

    # Already well-formed text passes through untouched.
    text = "Action: x\n<tool_call>\n{}\n</tool_call>"
    assert agent.wrap_for_history(text) == text


def test_image_truncation_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    screenshots = [f"img{i}" for i in range(31)]
    responses = [f"resp{i}" for i in range(30)]
    messages = agent.build_messages("do the thing", screenshots, responses)

    # k=30, K=10, T=20 -> remove 20: placeholders for tool results 1..20,
    # images for 21..30, plus the always-kept initial screenshot.
    assert len(messages) == 1 + 2 * 30
    image_count = sum(
        1
        for msg in messages
        if isinstance(msg.get("content"), list)
        for part in msg["content"]
        if part.get("type") == "image"
    )
    assert image_count == 11
    placeholder_count = sum(
        1
        for msg in messages
        if msg.get("content")
        == [{"type": "text", "text": agent.TOOL_RESULT_PLACEHOLDER}]
    )
    assert placeholder_count == 20

    # The initial user message keeps its image AND the instruction text.
    first = messages[0]
    assert first["role"] == "user"
    assert first["content"][0]["type"] == "image"
    assert first["content"][0]["source"]["data"] == "img0"
    assert first["content"][1] == {"type": "text", "text": "do the thing"}

    # Below the K+T boundary nothing is dropped (sawtooth, not a hard cap).
    screenshots = [f"img{i}" for i in range(30)]
    responses = [f"resp{i}" for i in range(29)]
    messages = agent.build_messages("do the thing", screenshots, responses)
    image_count = sum(
        1
        for msg in messages
        if isinstance(msg.get("content"), list)
        for part in msg["content"]
        if part.get("type") == "image"
    )
    assert image_count == 30


def test_response_text_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    data = {
        "content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "Action: go"},
        ]
    }
    text = agent.response_text_of(data)
    assert text.startswith("<mm:think>hmm</mm:think>")
    assert "Action: go" in text


PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + (1600).to_bytes(4, "big") + (900).to_bytes(4, "big")


def test_image_size_parses_ihdr(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    assert agent.image_size(PNG) == (1600, 900)
    with pytest.raises(ValueError):
        agent.image_size(b"not a png")


def test_error_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)

    assert agent.should_retry_status(429) is True
    assert agent.should_retry_status(503) is True
    assert agent.should_retry_status(401) is False
    assert agent.should_retry_status(400) is False


def test_template_has_no_task_specific_resolvers() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "agents" / "minimax_m3" / "agent.py"
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
