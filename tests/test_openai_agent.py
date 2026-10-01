from __future__ import annotations

import importlib.util
import json
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


def import_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model: str = "gpt-5.4-mini",
):
    monkeypatch.setenv("OPENAI_MODEL", model)
    monkeypatch.setenv("OPENAI_COST_LEDGER", str(tmp_path / "openai-cost.json"))
    monkeypatch.delenv("OPENAI_COST_RESERVE_USD", raising=False)
    monkeypatch.delenv("OPENAI_COST_LIMIT_USD", raising=False)
    path = Path(__file__).resolve().parents[1] / "agents" / "openai" / "agent.py"
    spec = importlib.util.spec_from_file_location(f"openai_agent_test_{id(tmp_path)}", path)
    assert spec is not None and spec.loader is not None
    agent = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = agent
    spec.loader.exec_module(agent)
    return agent


def test_model_prices_and_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = import_agent(monkeypatch, tmp_path)

    assert agent.MODEL == "gpt-5.4-mini"
    assert agent.MAX_STEPS == 100
    assert agent.MAX_OUTPUT_TOKENS == 16384
    assert agent.REASONING_EFFORT == "xhigh"
    assert agent.COMPUTER_TIMEOUT == 300
    assert agent.COST_RESERVE_USD == 0.50
    assert agent.MODEL_PRICING == {
        "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
        "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
        "gpt-5.5": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
        "gpt-5.6-sol": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
        "gpt-5.6": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
        "gpt-5.6-terra": {"input": 2.00, "cached_input": 0.20, "output": 12.00},
        "gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
    }
    assert agent.DEFAULT_RESERVES["gpt-5.6-terra"] == 5.00
    assert agent.DEFAULT_RESERVES["gpt-5.6-luna"] == 1.00
    assert agent.model_family("gpt-5.4-mini-2026-03-17") == "gpt-5.4-mini"
    assert agent.model_family("gpt-5.6-terra-2026-06-01") == "gpt-5.6-terra"


def test_payload_uses_ga_computer_tool_and_original_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)

    payload = agent.first_payload("Do the thing", b"png")

    assert payload["tools"] == [
        {"type": "computer"},
        {
            "type": "function",
            "name": "infeasible",
            "description": "Signal that the task is infeasible.",
            "parameters": {"type": "object", "properties": {}},
        },
    ]
    assert "MUST call the infeasible tool" in payload["instructions"]
    assert payload["parallel_tool_calls"] is False
    assert payload["reasoning"] == {"effort": "xhigh", "summary": "concise"}
    image = payload["input"][0]["content"][1]
    assert image["type"] == "input_image"
    assert image["detail"] == "original"
    assert image["image_url"].startswith("data:image/png;base64,")

    continued = agent.continue_payload(
        "resp_1", "call_1", b"next", [{"id": "safe_1", "code": "x"}]
    )
    assert continued["previous_response_id"] == "resp_1"
    output = continued["input"][0]
    assert output["type"] == "computer_call_output"
    assert output["call_id"] == "call_1"
    assert output["output"]["type"] == "computer_screenshot"
    assert output["output"]["detail"] == "original"
    assert output["acknowledged_safety_checks"][0]["id"] == "safe_1"

    restarted = agent.incomplete_payload("Do the thing", b"latest")
    assert "previous_response_id" not in restarted
    assert "Do the thing" in restarted["input"][0]["content"][0]["text"]
    assert restarted["input"][0]["content"][1]["detail"] == "original"


def test_cached_token_cost_and_reasoning_not_double_counted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)
    usage = {
        "input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 400},
        "output_tokens": 200,
        "output_tokens_details": {"reasoning_tokens": 150},
    }

    cost, tokens = agent.estimate_cost(usage)

    assert cost == pytest.approx((600 * 0.75 + 400 * 0.075 + 200 * 4.50) / 1e6)
    assert tokens == {"input": 1000, "cached_input": 400, "output": 200, "reasoning": 150}


def test_action_translation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = import_agent(monkeypatch, tmp_path)

    actions, wait = agent.translate_action(
        {"type": "click", "x": 1601, "y": -2, "button": "right", "keys": ["CTRL"]},
        1600,
        900,
    )
    assert wait is None
    assert actions == [
        {"keyboard": {"key_down": "ctrl"}},
        {"mouse": {"right_click": [1599, 0]}},
        {"keyboard": {"key_up": "ctrl"}},
    ]

    actions, _ = agent.translate_action(
        {
            "type": "drag",
            "path": [
                {"x": 10, "y": 20},
                {"x": 20, "y": 30},
                {"x": 30, "y": 40},
            ],
        },
        1600,
        900,
    )
    assert actions == [
        {"mouse": {"move": [10, 20]}},
        {"mouse": {"buttons": {"left_down": True}}},
        {"mouse": {"move": [20, 30]}},
        {"mouse": {"move": [30, 40]}},
        {"mouse": {"buttons": {"left_up": True}}},
    ]

    actions, _ = agent.translate_action(
        {"type": "scroll", "x": 500, "y": 400, "scroll_y": 250, "scroll_x": -121},
        1600,
        900,
    )
    assert actions == [
        {"mouse": {"move": [500, 400]}},
        {"mouse": {"scroll": 3}},
        {"keyboard": {"key_down": "shift"}},
        {"mouse": {"scroll": -2}},
        {"keyboard": {"key_up": "shift"}},
    ]

    actions, _ = agent.translate_action(
        {
            "type": "scroll",
            "x": 20,
            "y": 30,
            "scroll_y": -120,
            "scroll_x": 120,
            "keys": ["CTRL", "SHIFT"],
        },
        1600,
        900,
    )
    assert actions == [
        {"keyboard": {"key_down": "ctrl"}},
        {"keyboard": {"key_down": "shift"}},
        {"mouse": {"move": [20, 30]}},
        {"mouse": {"scroll": -1}},
        {"mouse": {"scroll": 1}},
        {"keyboard": {"key_up": "shift"}},
        {"keyboard": {"key_up": "ctrl"}},
    ]

    actions, _ = agent.translate_action(
        {"type": "click", "x": 5, "y": 6, "button": "wheel"}, 1600, 900
    )
    assert actions == [{"mouse": {"middle_click": [5, 6]}}]

    for unsupported_button in ("back", "forward"):
        with pytest.raises(ValueError, match="unsupported mouse button"):
            agent.translate_action(
                {
                    "type": "click",
                    "x": 5,
                    "y": 6,
                    "button": unsupported_button,
                },
                1600,
                900,
            )

    actions, _ = agent.translate_action(
        {"type": "keypress", "keys": ["CTRL", "L"]}, 1600, 900
    )
    assert actions == [{"keyboard": {"keys": ["ctrl", "L"]}}]

    actions, wait = agent.translate_action({"type": "wait", "ms": 2500}, 1600, 900)
    assert actions == []
    assert wait == 2.5


def test_execute_batches_actions_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)
    computer = DummyComputer()

    result = agent.execute_actions(
        computer,
        [
            {"type": "move", "x": 3, "y": 4},
            {"type": "type", "text": "hello"},
            {"type": "wait", "ms": 500},
            {"type": "screenshot"},
        ],
        1600,
        900,
    )

    assert computer.steps == [[
        {"mouse": {"move": [3, 4]}},
        {"keyboard": {"text": "hello"}},
        {"action": "wait", "time": 0.5},
        {"action": "screenshot"},
    ]]
    assert computer.waits == []
    assert result == {}


def test_run_stops_when_environment_finalizes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)

    class FinalizingComputer(DummyComputer):
        def __init__(self, env_url: str, timeout_sec: float) -> None:
            super().__init__()
            self.observe_count = 0
            self.done_count = 0

        def observe(self) -> dict:
            self.observe_count += 1
            png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (1600).to_bytes(
                4, "big"
            ) + (900).to_bytes(4, "big")
            return {"png": png, "meta": {}}

        def step(self, actions: list[dict]) -> dict:
            self.steps.append(actions)
            return {"done": True}

        def done(self) -> None:
            self.done_count += 1

    computer = FinalizingComputer("http://example", 300)
    requests: list[dict] = []

    def fake_request(payload: dict, tracker: object, label: str) -> dict:
        requests.append(payload)
        return {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "call_1",
                    "actions": [
                        {"type": "move", "x": 3, "y": 4},
                        {"type": "click", "x": 3, "y": 4, "button": "left"},
                    ],
                }
            ],
        }

    monkeypatch.setattr(agent, "Computer", lambda *args, **kwargs: computer)
    monkeypatch.setattr(agent, "responses_request", fake_request)

    agent.run("http://example", "Do the thing")

    assert len(requests) == 1
    assert computer.observe_count == 1
    assert computer.steps == [[
        {"mouse": {"move": [3, 4]}},
        {"mouse": {"left_click": [3, 4]}},
    ]]
    assert computer.done_count == 1


def test_infeasible_function_call_emits_terminal_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)
    computer = DummyComputer()
    response = {
        "output": [
            {
                "type": "function_call",
                "name": "infeasible",
                "arguments": "{}",
                "call_id": "call_fail",
            }
        ]
    }

    call = agent.function_calls(response)[0]
    agent.execute_function_call(computer, call)

    assert computer.steps == [[{"action_type": "FAIL"}]]


def test_cost_reservation_and_stale_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)
    agent.COST_LEDGER.write_text(
        json.dumps(
            {
                "actual_usd": 99.4,
                "reserved_usd": 0.5,
                "reservations": {
                    "dead": {"amount_usd": 0.5, "pid": 999_999_999, "created_at": 0}
                },
            }
        ),
        encoding="utf-8",
    )

    reservation_id = agent.CostTracker().reserve("test")
    state = json.loads(agent.COST_LEDGER.read_text(encoding="utf-8"))
    assert "dead" not in state["reservations"]
    assert reservation_id in state["reservations"]
    assert state["reserved_usd"] == 0.5

    with pytest.raises(agent.CostLimitReached):
        agent.CostTracker().reserve("would-overflow")


def test_over_reservation_response_is_recorded_before_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)
    tracker = agent.CostTracker()
    reservation_id = tracker.reserve("large-response")

    with pytest.raises(RuntimeError, match="exceeded the concurrency reservation"):
        tracker.release(reservation_id, {"input_tokens": 1_000_000})

    state = json.loads(agent.COST_LEDGER.read_text(encoding="utf-8"))
    assert state["actual_usd"] == pytest.approx(0.75)
    assert state["calls"] == 1
    assert state["reserved_usd"] == 0.0
    assert state["reservations"] == {}


def test_over_cap_response_is_recorded_and_blocks_new_reservations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)
    agent.COST_LIMIT_USD = 0.60
    tracker = agent.CostTracker()
    reservation_id = tracker.reserve("over-cap-response")

    with pytest.raises(agent.CostLimitReached, match="pushed total cost"):
        tracker.release(reservation_id, {"input_tokens": 1_000_000})

    state = json.loads(agent.COST_LEDGER.read_text(encoding="utf-8"))
    assert state["actual_usd"] == pytest.approx(0.75)
    assert state["calls"] == 1
    assert state["reserved_usd"] == 0.0
    assert state["reservations"] == {}
    with pytest.raises(agent.CostLimitReached):
        tracker.reserve("blocked-after-cap")


def test_response_parsing_and_no_task_specific_resolvers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)
    response = {
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "computer_call",
                "call_id": "c1",
                "actions": [{"type": "click", "x": 1, "y": 2}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "done"}],
            },
        ]
    }
    assert agent.computer_calls(response)[0]["call_id"] == "c1"
    assert agent.response_text(response) == "done"

    source = (Path(__file__).resolve().parents[1] / "agents" / "openai" / "agent.py").read_text()
    blocked = ["try_scripted_resolve", "_Gold.", "dog_cutout_gold", "gold_grades"]
    assert not any(marker in source for marker in blocked)
