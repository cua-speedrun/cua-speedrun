from __future__ import annotations

import importlib.util
import sys
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image


def import_agent(monkeypatch: pytest.MonkeyPatch, model: str = "claude-sonnet-4-6"):
    monkeypatch.setenv("CLAUDE_MODEL", model)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    path = Path(__file__).resolve().parents[1] / "agents" / "claude" / "agent.py"
    name = f"claude_agent_test_{model.replace('-', '_')}_{id(monkeypatch)}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    agent = importlib.util.module_from_spec(spec)
    sys.modules[name] = agent
    spec.loader.exec_module(agent)
    return agent


def png(width: int = 1920, height: int = 1080, color: str = "white") -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), color).save(output, format="PNG")
    return output.getvalue()


def test_current_tool_payload_and_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch)

    assert agent.MODEL == "claude-sonnet-4-6"
    assert agent.computer_tool_version() == (
        "computer_20251124",
        "computer-use-2025-11-24",
    )
    payload = agent.request_payload([{"role": "user", "content": "task"}], (1280, 720))
    assert payload["tools"] == [
        {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": 1280,
            "display_height_px": 720,
            "enable_zoom": True,
        },
        {
            "name": "complete",
            "description": "Signal that the visible desktop state fully satisfies the task.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "infeasible",
            "description": "Signal that the task is infeasible.",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]
    assert "MUST call the\ninfeasible tool" in payload["system"]
    assert payload["output_config"] == {"effort": "medium"}
    assert agent.anthropic_headers()["anthropic-beta"] == "computer-use-2025-11-24"

    legacy = import_agent(monkeypatch, "claude-sonnet-4-5")
    assert legacy.computer_tool_version() == (
        "computer_20250124",
        "computer-use-2025-01-24",
    )
    legacy_payload = legacy.request_payload(
        [{"role": "user", "content": "task"}], (1280, 720)
    )
    assert legacy_payload["tools"][0]["type"] == "computer_20250124"
    assert "enable_zoom" not in legacy_payload["tools"][0]
    assert "output_config" not in legacy_payload


def test_resize_preserves_aspect_ratio_and_coordinates_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch)

    assert agent.display_size(1920, 1080) == (1280, 720)
    assert agent.display_size(1024, 768) == (960, 720)
    assert agent.image_size(agent.resize_png(png(), (1280, 720))) == (1280, 720)
    assert agent.scaled_point([640, 360], (1920, 1080), (1280, 720)) == [960, 540]
    assert agent.scaled_point([-10, 9999], (1920, 1080), (1280, 720)) == [0, 1079]


def test_translate_anthropic_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch)
    native = (1920, 1080)
    display = (1280, 720)

    actions, cursor = agent.translate_action(
        {"action": "left_click", "coordinate": [640, 360], "text": "CTRL+SHIFT"},
        native,
        display,
    )
    assert actions == [
        {"keyboard": {"keys_down": ["ctrl", "shift"]}},
        {"mouse": {"left_click": [960, 540]}},
        {"keyboard": {"keys_up": ["shift", "ctrl"]}},
    ]
    assert cursor == [960, 540]

    actions, cursor = agent.translate_action(
        {"action": "left_click_drag", "coordinate": [800, 400]},
        native,
        display,
        cursor,
    )
    assert actions == [{"mouse": {"left_click_drag": [[960, 540], [1200, 600]]}}]
    assert cursor == [1200, 600]

    actions, _ = agent.translate_action(
        {
            "action": "scroll",
            "coordinate": [100, 200],
            "scroll_direction": "left",
            "scroll_amount": 3,
        },
        native,
        display,
    )
    assert actions == [
        {"mouse": {"move": [150, 300]}},
        {"keyboard": {"keys_down": ["shift"]}},
        {"mouse": {"scroll": -3}},
        {"keyboard": {"keys_up": ["shift"]}},
    ]

    actions, _ = agent.translate_action(
        {"action": "hold_key", "text": "ALT", "duration": 2}, native, display
    )
    assert actions == [
        {"keyboard": {"keys_down": ["alt"]}},
        {"action": "wait", "time": 2.0},
        {"keyboard": {"keys_up": ["alt"]}},
    ]

    actions, _ = agent.translate_action(
        {"action": "type", "text": "first\nsecond"}, native, display
    )
    assert actions == [
        {"keyboard": {"text": "first"}},
        {"keyboard": {"keys": ["Return"]}},
        {"keyboard": {"text": "second"}},
    ]


def test_parse_turn_is_atomic_and_tracks_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch)
    calls = [
        {
            "type": "tool_use",
            "id": "one",
            "name": "computer",
            "input": {"action": "mouse_move", "coordinate": [100, 100]},
        },
        {
            "type": "tool_use",
            "id": "two",
            "name": "computer",
            "input": {"action": "left_click_drag", "coordinate": [200, 200]},
        },
    ]
    parsed, cursor = agent.parse_turn(calls, (1920, 1080), (1280, 720), None)
    assert parsed[1][1] == [{"mouse": {"left_click_drag": [[150, 150], [300, 300]]}}]
    assert cursor == [300, 300]

    bad = [
        *calls,
        {
            "type": "tool_use",
            "id": "bad",
            "name": "computer",
            "input": {"action": "nope"},
        },
    ]
    with pytest.raises(ValueError, match="unsupported computer action"):
        agent.parse_turn(bad, (1920, 1080), (1280, 720), None)

    bad_zoom = [
        {
            "type": "tool_use",
            "id": "zoom",
            "name": "computer",
            "input": {"action": "zoom", "region": [10, 10, 10, 20]},
        }
    ]
    with pytest.raises(ValueError, match="non-zero area"):
        agent.parse_turn(bad_zoom, (1920, 1080), (1280, 720), None)


def test_image_history_pruning_and_cache_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch)
    messages = [{"role": "user", "content": "task"}]
    for index in range(14):
        messages.append(
            {
                "role": "user",
                "content": [agent.screenshot_result(str(index), png(10, 10), (10, 10))],
            }
        )
        agent.add_cache_marker(messages)

    agent.prune_old_images(messages, keep=7, threshold=7)
    images = [
        block
        for message in messages
        for result in (
            message.get("content") if isinstance(message.get("content"), list) else []
        )
        if isinstance(result, dict) and result.get("type") == "tool_result"
        for block in result.get("content", [])
        if isinstance(block, dict) and block.get("type") == "image"
    ]
    markers = [
        block
        for message in messages
        for block in (
            message.get("content") if isinstance(message.get("content"), list) else []
        )[-1:]
        if isinstance(block, dict) and "cache_control" in block
    ]
    assert len(images) == 7
    assert len(markers) == 4


def test_run_returns_screenshot_after_each_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch)

    class DummyComputer:
        instance = None

        def __init__(self, env_url: str, timeout_sec: float) -> None:
            del env_url, timeout_sec
            self.steps: list[list[dict]] = []
            self.observations = 0
            self.done_calls = 0
            DummyComputer.instance = self

        def observe(self):
            self.observations += 1
            return {
                "png": png(color="white" if self.observations == 1 else "blue"),
                "meta": {},
            }

        def step(self, actions):
            self.steps.append(actions)
            return {"done": False}

        def done(self):
            self.done_calls += 1

    requests_seen = []
    responses = iter(
        [
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "click-1",
                        "name": "computer",
                        "input": {"action": "left_click", "coordinate": [100, 200]},
                    },
                    {
                        "type": "tool_use",
                        "id": "type-1",
                        "name": "computer",
                        "input": {"action": "type", "text": "hello"},
                    },
                ],
                "stop_reason": "tool_use",
            },
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "complete-1",
                        "name": "complete",
                        "input": {},
                    }
                ],
                "stop_reason": "tool_use",
            },
        ]
    )

    def fake_request(payload):
        requests_seen.append(payload)
        return next(responses)

    monkeypatch.setattr(agent, "Computer", DummyComputer)
    monkeypatch.setattr(agent, "anthropic_request", fake_request)
    agent.run("http://env", "Do the task")

    computer = DummyComputer.instance
    assert computer is not None
    assert computer.steps == [
        [{"mouse": {"left_click": [150, 300]}}],
        [{"keyboard": {"text": "hello"}}],
    ]
    assert computer.observations == 3
    assert computer.done_calls == 1
    initial_content = requests_seen[0]["messages"][0]["content"]
    assert initial_content[0] == {"type": "text", "text": "Do the task"}
    assert initial_content[1]["type"] == "image"
    initial_png = agent.base64.b64decode(initial_content[1]["source"]["data"])
    assert agent.image_size(initial_png) == (1280, 720)
    results = requests_seen[1]["messages"][-2]["content"]
    assert [result["tool_use_id"] for result in results] == ["click-1", "type-1"]
    assert all(result["content"][-1]["type"] == "image" for result in results)


def test_infeasible_tool_call_emits_terminal_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch)

    class DummyComputer:
        instance = None

        def __init__(self, env_url: str, timeout_sec: float) -> None:
            del env_url, timeout_sec
            self.steps: list[list[dict]] = []
            self.done_calls = 0
            DummyComputer.instance = self

        def observe(self):
            return {"png": png(), "meta": {}}

        def step(self, actions):
            self.steps.append(actions)
            return {"done": True}

        def done(self):
            self.done_calls += 1

    response = {
        "content": [
            {
                "type": "tool_use",
                "id": "infeasible-1",
                "name": "infeasible",
                "input": {},
            }
        ],
        "stop_reason": "tool_use",
    }

    monkeypatch.setattr(agent, "Computer", DummyComputer)
    monkeypatch.setattr(agent, "anthropic_request", lambda payload: response)
    agent.run("http://env", "Do the impossible task")

    computer = DummyComputer.instance
    assert computer is not None
    assert computer.steps == [[{"action_type": "FAIL"}]]
    assert computer.done_calls == 1


def test_text_only_response_requires_structured_terminal_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch)

    class DummyComputer:
        instance = None

        def __init__(self, env_url: str, timeout_sec: float) -> None:
            del env_url, timeout_sec
            self.steps: list[list[dict]] = []
            self.done_calls = 0
            DummyComputer.instance = self

        def observe(self):
            return {"png": png(), "meta": {}}

        def step(self, actions):
            self.steps.append(actions)
            return {"done": True}

        def done(self):
            self.done_calls += 1

    requests_seen = []
    responses = iter(
        [
            {
                "content": [
                    {"type": "text", "text": "I cannot do this and need help."}
                ],
                "stop_reason": "end_turn",
            },
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "infeasible-1",
                        "name": "infeasible",
                        "input": {},
                    }
                ],
                "stop_reason": "tool_use",
            },
        ]
    )

    def fake_request(payload):
        requests_seen.append(payload)
        return next(responses)

    monkeypatch.setattr(agent, "Computer", DummyComputer)
    monkeypatch.setattr(agent, "anthropic_request", fake_request)
    agent.run("http://env", "Do the impossible task")

    computer = DummyComputer.instance
    assert computer is not None
    assert computer.steps == [[{"action_type": "FAIL"}]]
    assert computer.done_calls == 1
    reminder_texts = [
        block.get("text", "")
        for message in requests_seen[1]["messages"]
        for block in (
            message.get("content") if isinstance(message.get("content"), list) else []
        )
        if isinstance(block, dict)
    ]
    assert any(
        "Do not end this desktop task with only text" in text
        for text in reminder_texts
    )


def test_catalog_exposes_claude_template() -> None:
    from cua_speedrun.service.templates_catalog import list_templates

    claude = next(item for item in list_templates() if item["name"] == "claude")
    assert claude["required_environment_variables"] == ["ANTHROPIC_API_KEY"]
    assert "osworld-50" in claude["compatible_benchmarks"]
