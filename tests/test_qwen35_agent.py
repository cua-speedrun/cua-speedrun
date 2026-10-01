"""The qwen35 response parser preserves modifier holds and drag coordinates."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_agent():
    path = Path(__file__).resolve().parents[1] / "agents" / "qwen35" / "agent.py"
    spec = importlib.util.spec_from_file_location("qwen35_agent_test", path)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    return agent


_AGENT = _load_agent()
# Dimensions equal to the model's coordinate grid make scaling the identity,
# so tests can assert on the coordinates they wrote.
_DIM = int(_AGENT.GRID_MAX)


def _tool_call(action: str, params: dict[str, str]) -> str:
    parts = [f"<parameter=action>\n{action}\n</parameter>"]
    for name, value in params.items():
        parts.append(f"<parameter={name}>\n{value}\n</parameter>")
    body = "\n".join(parts)
    return (
        "Action: test action\n\n<tool_call>\n<function=computer_use>\n"
        f"{body}\n</function>\n</tool_call>"
    )


def _parse(response: str):
    return _AGENT.parse_response(response, _DIM, _DIM)


def test_click_with_keys_holds_the_modifier_around_the_click() -> None:
    parsed = _parse(_tool_call(
        "left_click", {"coordinate": "[100, 200]", "keys": '["shift"]'}
    ))
    assert parsed["actions"] == [
        {"keyboard": {"keys_down": ["shift"]}},
        {"mouse": {"left_click": [100, 200]}},
        {"keyboard": {"keys_up": ["shift"]}},
    ]


def test_scroll_with_keys_wraps_the_whole_gesture() -> None:
    parsed = _parse(_tool_call(
        "scroll",
        {"coordinate": "[50, 60]", "pixels": "-4", "keys": '["ctrl"]'},
    ))
    actions = parsed["actions"]
    assert actions[0] == {"keyboard": {"keys_down": ["ctrl"]}}
    assert actions[-1] == {"keyboard": {"keys_up": ["ctrl"]}}
    kinds = [next(iter(a.get("mouse", a.get("keyboard")))) for a in actions]
    assert kinds == ["keys_down", "move", "scroll", "keys_up"]


def test_multiple_held_keys_release_in_reverse_order() -> None:
    parsed = _parse(_tool_call(
        "left_click", {"coordinate": "[10, 10]", "keys": '["ctrl", "shift"]'}
    ))
    assert parsed["actions"][0] == {"keyboard": {"keys_down": ["ctrl", "shift"]}}
    assert parsed["actions"][-1] == {"keyboard": {"keys_up": ["shift", "ctrl"]}}


def test_plain_key_action_keeps_keys_as_the_chord() -> None:
    parsed = _parse(_tool_call("key", {"keys": '["ctrl", "s"]'}))
    assert parsed["actions"] == [{"keyboard": {"keys": ["ctrl", "s"]}}]


def test_click_without_keys_is_unchanged() -> None:
    parsed = _parse(_tool_call("left_click", {"coordinate": "[100, 200]"}))
    assert parsed["actions"] == [{"mouse": {"left_click": [100, 200]}}]


def test_hold_around_an_empty_gesture_is_dropped() -> None:
    # A scroll with no coordinate and no pixels appends nothing; the held
    # modifier must not leak as a bare press-release.
    parsed = _parse(_tool_call("scroll", {"pixels": "0", "keys": '["ctrl"]'}))
    assert parsed["actions"] == []


def test_single_coordinate_drag_parses_as_a_one_point_drag() -> None:
    parsed = _parse(_tool_call("left_click_drag", {"coordinate": "[300, 400]"}))
    assert parsed["actions"] == [{"mouse": {"left_click_drag": [[300, 400]]}}]


def test_two_point_drag_keeps_both_points() -> None:
    parsed = _parse(_tool_call(
        "left_click_drag",
        {"coordinate": "[1, 2]", "coordinate2": "[3, 4]"},
    ))
    assert parsed["actions"] == [
        {"mouse": {"left_click_drag": [[1, 2], [3, 4]]}}
    ]


def test_every_tool_call_in_a_turn_executes() -> None:
    response = (
        _tool_call("mouse_move", {"coordinate": "[10, 20]"})
        + "\n"
        + _tool_call("left_click_drag", {"coordinate": "[30, 40]"})
    )
    parsed = _parse(response)
    assert parsed["actions"] == [
        {"mouse": {"move": [10, 20]}},
        {"mouse": {"left_click_drag": [[30, 40]]}},
    ]
