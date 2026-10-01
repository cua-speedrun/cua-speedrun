"""Check translation contracts against the real SDK; no environment doubles."""

from __future__ import annotations

import asyncio
import argparse
import importlib.util
import inspect
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("yutori_n2_agent", ROOT / "agents/yutori_n2/agent.py")
assert spec is not None and spec.loader is not None
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


def test_yutori_template_is_discoverable_and_valid():
    from cua_speedrun.commands.benchmark import validate_agent
    from cua_speedrun.service.templates_catalog import list_templates, template_dir

    entry = next(item for item in list_templates() if item["name"] == "yutori_n2")
    assert entry["gpu"] is None
    assert entry["required_environment_variables"] == ["YUTORI_API_KEY"]
    assert entry["optional_environment_variables"] == [
        "YUTORI_REASONING_EFFORT", "YUTORI_MAX_STEPS", "YUTORI_ENV_TIMEOUT_SEC",
    ]
    validate_agent(template_dir("yutori_n2"))


def test_hosted_yutori_inputs_forward_settings_and_keep_credentials_separate(monkeypatch):
    from cua_speedrun.commands import register_operator_commands
    from cua_speedrun.hosted.inputs import pack_inputs

    parser = argparse.ArgumentParser()
    register_operator_commands(parser.add_subparsers(dest="command"))
    args = parser.parse_args([
        "benchmark", "--host", "modal", "--agent", "yutori_n2", "--dataset", "osworld-50",
    ])
    monkeypatch.delenv("YUTORI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="YUTORI_API_KEY"):
        pack_inputs(args)
    values = {
        "YUTORI_API_KEY": "packaging-check-only",
        "YUTORI_REASONING_EFFORT": "low",
        "YUTORI_MAX_STEPS": "500",
        "YUTORI_ENV_TIMEOUT_SEC": "600",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    request, archive, credentials = pack_inputs(args)
    assert request["task_count"] == 50
    assert request["argv"][request["argv"].index("--agent") + 1] == "yutori_n2"
    assert {name: credentials[name] for name in values} == values
    assert values["YUTORI_API_KEY"] not in json.dumps(request)
    assert values["YUTORI_API_KEY"].encode() not in archive


def test_reasoning_default_and_sweep_values(monkeypatch):
    monkeypatch.delenv("YUTORI_REASONING_EFFORT", raising=False)
    assert agent.reasoning_effort() is None
    for effort in ("none", "low", "medium", "xhigh"):
        monkeypatch.setenv("YUTORI_REASONING_EFFORT", effort)
        assert agent.reasoning_effort() == effort
    monkeypatch.setenv("YUTORI_REASONING_EFFORT", "typo")
    with pytest.raises(ValueError):
        agent.reasoning_effort()


def test_real_sdk_primitives_bind_to_adapter():
    sdk = pytest.importorskip("yutori.navigator.n2_actions")
    cases = {
        "left_click": {"coordinates": [250, 500], "modifier": "ctrl"},
        "right_click": {"coordinates": [250, 500]},
        "middle_click": {"coordinates": [250, 500]},
        "double_click": {"coordinates": [250, 500]},
        "triple_click": {"coordinates": [250, 500]},
        "mouse_move": {"coordinates": [250, 500]},
        "mouse_down": {}, "mouse_up": {},
        "drag": {"start_coordinates": [250, 500], "coordinates": [500, 750]},
        "scroll": {"coordinates": [250, 500], "direction": "down", "amount": 3},
        "key_press": {"key": "ctrl+s down enter"},
        "type": {"text": "text"}, "wait": {"duration": 1.25},
        "hold_key": {"key": "shift", "duration": 0.25}, "screenshot": {},
    }
    for name, arguments in cases.items():
        for action in sdk.translate_n2_action(
            name, arguments, 1280, 720, allow_click_modifiers=True, allow_scroll_modifiers=True,
        ):
            if action["type"] == "screenshot":
                continue  # The SDK captures once after the batch.
            method = getattr(agent.SpeedrunComputer, action["type"])
            kwargs = {key: value for key, value in action.items() if key != "type"}
            inspect.signature(method).bind(None, **kwargs)


def test_real_sdk_coordinates_keys_and_scroll_units():
    sdk = pytest.importorskip("yutori.navigator.n2_actions")
    from cua_speedrun.envs._pyautogui_keyboard import key_names

    click = sdk.translate_n2_action("left_click", {"coordinates": [250, 500]}, 1280, 720)[0]
    assert (click["x"], click["y"]) == (320, 360)
    presses = sdk.translate_n2_action("key_press", {"key": "super pageup slash"}, 1280, 720)
    assert [key_names(action["keys"]) for action in presses] == [["Super_L"], ["Prior"], ["/"]]
    assert key_names([agent.environment_key("printscreen")]) == ["Print"]
    for direction, expected in (("down", 3), ("up", -3)):
        model_action = {"direction": direction, "amount": 3}
        assert agent.wheel_notches(model_action) == expected
    with pytest.raises(ValueError):
        agent.wheel_notches({"direction": "left", "amount": 3})


def test_modified_gesture_uses_held_input_and_reverse_release():
    click = {"mouse": {"left_click": [320, 360]}}
    actions = agent.modified_actions(click, ["ctrl", "shift"])
    assert actions == [
        {"keyboard": {"keys_down": ["ctrl", "shift"]}}, click,
        {"keyboard": {"keys_up": ["shift", "ctrl"]}},
    ]


def test_real_sdk_batch_is_queued_as_one_gateway_batch():
    sdk = pytest.importorskip("yutori.navigator.n2_actions")
    from cua_speedrun.client import Computer

    adapter = agent.SpeedrunComputer(Computer("http://127.0.0.1:1"))
    model_actions = [
        ("left_click", {"coordinates": [250, 500]}),
        ("left_click", {"coordinates": [500, 500]}),
        ("left_click", {"coordinates": [750, 500]}),
        ("key_press", {"key": "ctrl+shift+t"}),
    ]

    async def queue_batch():
        for name, arguments in model_actions:
            for action in sdk.translate_n2_action(name, arguments, 1920, 1080):
                method = getattr(adapter, action.pop("type"))
                await method(**action)

    asyncio.run(queue_batch())
    assert adapter.pending_actions == [
        {"mouse": {"left_click": [480, 540]}},
        {"mouse": {"left_click": [960, 540]}},
        {"mouse": {"left_click": [1440, 540]}},
        {"keyboard": {"keys": ["ctrl", "shift", "t"]}},
    ]
