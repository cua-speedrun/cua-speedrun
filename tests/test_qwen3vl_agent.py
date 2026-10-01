import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "qwen3vl_agent", Path(__file__).resolve().parents[1] / "agents/qwen3vl/agent.py"
)
AGENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGENT)


def call(**arguments):
    return "<tool_call>" + json.dumps({"name": "computer_use", "arguments": arguments}) + "</tool_call>"


def test_drag_from_current_pointer_after_move():
    response = call(action="mouse_move", coordinate=[226, 474]) + call(
        action="left_click_drag", coordinate2=[293, 474]
    )
    assert AGENT.parse_response(response, (1.92, 1.08))["actions"] == [
        {"mouse": {"move": [433, 511]}},
        {"mouse": {"left_click_drag": [[562, 511]]}},
    ]


def test_drag_with_explicit_start_and_end():
    result = AGENT.parse_response(call(action="drag", coordinate=[10, 20], coordinate2=[30, 40]), (2, 2))
    assert result["actions"] == [{"mouse": {"left_click_drag": [[20, 40], [60, 80]]}}]


@pytest.mark.parametrize("arguments", [
    {"action": "left_click"}, {"action": "left_click", "coordinate": [1]},
    {"action": "left_click", "coordinate": ["bad", 2]},
    {"action": "drag"}, {"action": "scroll", "pixels": "bad"},
])
def test_invalid_action_does_not_crash_agent(arguments):
    result = AGENT.parse_response(call(**arguments), (1, 1))
    assert result["actions"] == []
    assert not result["is_terminal"]
    assert result["wait_time"] == 1.0
