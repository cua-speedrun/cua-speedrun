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
    template: str = "gemini35",
):
    monkeypatch.setenv("GEMINI_COST_LEDGER", str(tmp_path / "gemini-cost.json"))
    monkeypatch.delenv("GEMINI_EXCLUDED_PREDEFINED_FUNCTIONS", raising=False)
    monkeypatch.delenv("GEMINI_COST_RESERVE_USD", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_INPUT_RATE_PER_MILLION", raising=False)
    monkeypatch.delenv("GEMINI_OUTPUT_RATE_PER_MILLION", raising=False)
    monkeypatch.delenv("GEMINI_AUTO_ACK_SAFETY", raising=False)
    monkeypatch.delenv("GEMINI_DISABLED_SAFETY_POLICIES", raising=False)
    monkeypatch.delenv("GEMINI_TEMPERATURE", raising=False)
    monkeypatch.delenv("GEMINI_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("GEMINI_THINKING_LEVEL", raising=False)

    path = Path(__file__).resolve().parents[1] / "agents" / template / "agent.py"
    spec = importlib.util.spec_from_file_location(f"{template}_agent_test", path)
    assert spec is not None and spec.loader is not None
    agent = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = agent
    spec.loader.exec_module(agent)
    return agent


@pytest.mark.parametrize(
    (
        "template",
        "model",
        "input_rate",
        "output_rate",
        "environment",
        "thinking_level",
        "max_output_tokens",
        "excluded_functions",
    ),
    [
        (
            "gemini35",
            "gemini-3.5-flash",
            1.50,
            9.00,
            "desktop",
            "medium",
            16384,
            None,
        ),
        (
            "gemini3_flash_preview",
            "gemini-3-flash-preview",
            0.50,
            3.00,
            "desktop",
            "high",
            None,
            ["bash"],
        ),
    ],
)
def test_templates_use_expected_desktop_inference_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    template: str,
    model: str,
    input_rate: float,
    output_rate: float,
    environment: str,
    thinking_level: str,
    max_output_tokens: int | None,
    excluded_functions: list[str] | None,
) -> None:
    agent = import_agent(monkeypatch, tmp_path, template)

    computer_tool = agent.tools()[0]

    assert agent.MODEL == model
    assert agent.REQUEST_TIMEOUT == 600
    assert agent.INPUT_RATE_PER_MILLION == input_rate
    assert agent.OUTPUT_RATE_PER_MILLION == output_rate
    assert agent.MAX_STEPS == 100
    assert computer_tool["type"] == "computer_use"
    assert computer_tool["environment"] == environment
    if excluded_functions is None:
        assert "excluded_predefined_functions" not in computer_tool
    else:
        assert computer_tool["excluded_predefined_functions"] == excluded_functions
    # The benchmark VM is disposable and isolated (no real accounts,
    # purchases, or user data), so every documented safety-policy category
    # is disabled on purpose: leaving them on turns task-required actions
    # into confirmation prompts and measures refusals instead of ability.
    assert set(computer_tool["disabled_safety_policies"]) == {
        "financial_transactions",
        "sensitive_data_modification",
        "communication_tool",
        "account_creation",
        "data_modification",
        "user_consent_management",
        "legal_terms_and_agreements",
    }
    # Same reason: an unattended run has nobody to answer a safety
    # confirmation, so leaving them unacknowledged just stalls the episode.
    assert agent.AUTO_ACK_SAFETY is True
    assert agent.THINKING_LEVEL == thinking_level
    generation_config = agent.base_payload()["generation_config"]
    assert generation_config["thinking_level"] == thinking_level
    if max_output_tokens is None:
        assert not hasattr(agent, "MAX_OUTPUT_TOKENS")
        assert generation_config == {"thinking_level": thinking_level}
    else:
        assert agent.MAX_OUTPUT_TOKENS == max_output_tokens
        assert generation_config["max_output_tokens"] == max_output_tokens

    desktop_core = {
        "click",
        "double_click",
        "triple_click",
        "middle_click",
        "right_click",
        "mouse_down",
        "mouse_up",
        "move",
        "type",
        "drag_and_drop",
        "wait",
        "press_key",
        "key_down",
        "key_up",
        "hotkey",
        "take_screenshot",
        "scroll",
    }
    assert agent.SUPPORTED_DESKTOP_COMMANDS == desktop_core


def test_action_translation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = import_agent(monkeypatch, tmp_path)

    assert agent.COST_RESERVE_USD == 2.0
    assert agent.MISSING_USAGE_FALLBACK_USD == agent.COST_RESERVE_USD
    assert agent.normalize_key("Super_L") == "super"
    assert agent.normalize_key("Alt_R") == "alt"
    assert agent.normalize_key("Shift_L") == "shift"
    assert agent.normalize_key("Control_R") == "ctrl"
    assert agent.normalize_key("page_down") == "pagedown"

    computer = DummyComputer()
    result = agent.execute_call(
        computer,
        {
            "name": "scroll",
            "id": "s1",
            "arguments": {
                "x": 500,
                "y": 500,
                "direction": "down",
                "magnitude_in_wheel_clicks": 20,
            },
        },
        1600,
        900,
    )
    assert result.result["actions"] == [
        {"mouse": {"move": [800, 450]}},
        {"mouse": {"scroll": 20}},
    ]

    computer = DummyComputer()
    result = agent.execute_call(
        computer,
        {
            "name": "scroll",
            "id": "s2",
            "arguments": {
                "x": 500,
                "y": 500,
                "direction": "left",
                "magnitude_in_pixels": 240,
            },
        },
        1600,
        900,
    )
    assert result.result["actions"] == [
        {"keyboard": {"key_down": "shift"}},
        {"mouse": {"move": [800, 450]}},
        {"mouse": {"scroll": -2}},
        {"keyboard": {"key_up": "shift"}},
    ]

    computer = DummyComputer()
    result = agent.execute_call(
        computer,
        {
            "name": "type",
            "id": "t1",
            "arguments": {"text": "abc", "press_enter": True},
        },
        1600,
        900,
    )
    assert result.result["actions"] == [
        {"keyboard": {"text": "abc"}},
        {"keyboard": {"keys": ["Return"]}},
    ]

    computer = DummyComputer()
    result = agent.execute_call(
        computer,
        {
            "name": "type",
            "id": "t2",
            "arguments": {"text": "abc"},
        },
        1600,
        900,
    )
    assert result.result["actions"] == [
        {"keyboard": {"text": "abc"}},
    ]

    computer = DummyComputer()
    result = agent.execute_call(
        computer,
        {
            "name": "drag_and_drop",
            "id": "d1",
            "arguments": {
                "start_x": 250,
                "start_y": 750,
                "end_x": 500,
                "end_y": 500,
            },
        },
        1600,
        900,
    )
    assert result.result["actions"] == [
        {"mouse": {"left_click_drag": [[400, 675], [800, 450]]}},
    ]

    computer = DummyComputer()
    result = agent.execute_call(
        computer,
        {"name": "key_down", "id": "k1", "arguments": {"key": "Shift_L"}},
        1600,
        900,
    )
    assert result.result["actions"] == [{"keyboard": {"key_down": "shift"}}]

    computer = DummyComputer()
    result = agent.execute_call(
        computer,
        {"name": "key_up", "id": "k2", "arguments": {"key": "Shift_L"}},
        1600,
        900,
    )
    assert result.result["actions"] == [{"keyboard": {"key_up": "shift"}}]

    computer = DummyComputer()
    result = agent.execute_call(
        computer,
        {"name": "navigate", "id": "n1", "arguments": {"url": "https://example.com"}},
        1600,
        900,
    )
    assert result.result["ok"] is False
    assert "unsupported desktop action" in result.result["error"]
    assert computer.steps == []


def test_cost_reservation_respects_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A cap only exists when one is configured; test the mechanism, not a
    # default. GEMINI_COST_LIMIT_USD is unset (-1) by default, which the
    # template maps to unlimited.
    monkeypatch.setenv("GEMINI_COST_LIMIT_USD", "25")
    agent = import_agent(monkeypatch, tmp_path)
    agent.COST_LEDGER.write_text(
        json.dumps({"actual_usd": 30.0, "reserved_usd": 0.0, "reservations": {}}),
        encoding="utf-8",
    )

    with pytest.raises(agent.CostLimitReached):
        agent.CostTracker().reserve("test")


def test_cost_limit_is_unlimited_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # -1 means "no cap"; nothing pinned that contract before, so a change
    # to the default could have silently started failing runs mid-flight.
    monkeypatch.delenv("GEMINI_COST_LIMIT_USD", raising=False)
    agent = import_agent(monkeypatch, tmp_path)
    assert agent.COST_LIMIT_USD == float("inf")
    agent.COST_LEDGER.write_text(
        json.dumps({"actual_usd": 10_000.0, "reserved_usd": 0.0, "reservations": {}}),
        encoding="utf-8",
    )
    agent.CostTracker().reserve("test")  # must not raise


@pytest.mark.parametrize("template", ["gemini35", "gemini3_flash_preview"])
def test_malformed_tool_call_is_retriable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, template: str
) -> None:
    agent = import_agent(monkeypatch, tmp_path, template)

    assert agent.should_retry_error(400, '{"code":"malformed_tool_call"}') is True
    assert agent.should_retry_error(400, '{"code":"invalid_argument"}') is False
    assert agent.should_retry_error(429, "rate limit") is True


@pytest.mark.parametrize("template", ["gemini35", "gemini3_flash_preview"])
def test_incomplete_continue_payload_includes_screenshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, template: str
) -> None:
    agent = import_agent(monkeypatch, tmp_path, template)

    payload = agent.continue_after_incomplete_payload("abc123", b"png-bytes")

    assert payload["previous_interaction_id"] == "abc123"
    assert payload["input"][0]["type"] == "text"
    assert "previous response ended" in payload["input"][0]["text"]
    assert payload["input"][1]["type"] == "image"
    if template == "gemini3_flash_preview":
        assert payload["input"][1]["resolution"] == "high"
    else:
        assert "resolution" not in payload["input"][1]


@pytest.mark.parametrize("template", ["gemini35", "gemini3_flash_preview"])
def test_gemini_templates_use_upstream_first_turn_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, template: str
) -> None:
    agent = import_agent(monkeypatch, tmp_path, template)

    payload = agent.first_payload("Example task")

    assert payload["input"] == [{"type": "text", "text": "Example task"}]
    assert agent.tools()[1] == {
        "type": "function",
        "name": "infeasible",
        "description": "Signal that the task is infeasible.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    }
    assert "task_complete" not in agent.SYSTEM_INSTRUCTION
    assert "MUST use the 'take_screenshot' tool" in agent.SYSTEM_INSTRUCTION
    if template == "gemini3_flash_preview":
        assert "bash" not in agent.SYSTEM_INSTRUCTION.lower()


@pytest.mark.parametrize("template", ["gemini35", "gemini3_flash_preview"])
def test_infeasible_emits_osworld_fail_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, template: str
) -> None:
    agent = import_agent(monkeypatch, tmp_path, template)
    computer = DummyComputer()

    result = agent.execute_call(
        computer,
        {
            "name": "infeasible",
            "id": "infeasible1",
            "arguments": {},
        },
        1600,
        900,
    )

    assert result.terminal is True
    assert result.result == {"ok": True}
    assert computer.steps == [[{"action_type": "FAIL"}]]


def test_gemini3_bash_is_never_executed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path, "gemini3_flash_preview")
    computer = DummyComputer()

    result = agent.execute_call(
        computer,
        {
            "name": "bash",
            "id": "bash1",
            "arguments": {"command": "touch /tmp/should-not-exist"},
        },
        1600,
        900,
    )

    assert result.result == {
        "ok": False,
        "error": (
            "shell execution is disabled; use visible desktop actions, including "
            "opening and typing into a terminal through the GUI when needed"
        ),
    }
    assert computer.steps == []


@pytest.mark.parametrize(
    ("name", "arguments", "expected_actions"),
    [
        (
            "click",
            {"coordinate": [250, 500]},
            [{"mouse": {"left_click": [400, 450]}}],
        ),
        (
            "click_at",
            {"x": 250, "y": 500},
            [{"mouse": {"left_click": [400, 450]}}],
        ),
        (
            "mouse_click",
            {"coordinate": [250, 500], "button": "right"},
            [{"mouse": {"right_click": [400, 450]}}],
        ),
        (
            "mouse_move",
            {"point": {'"x"': 250, '"y"': 500}},
            [{"mouse": {"move": [400, 450]}}],
        ),
        (
            "double_click_at",
            {"x": 250, "y": 500},
            [{"mouse": {"double_click": [400, 450]}}],
        ),
        (
            "hover_at",
            {"x": 250, "y": 500},
            [{"mouse": {"move": [400, 450]}}],
        ),
        (
            "type_text",
            {"text": "hello"},
            [{"keyboard": {"text": "hello"}}],
        ),
        (
            "type_text_at",
            {"x": 250, "y": 500, "text": "hello"},
            [
                {"mouse": {"left_click": [400, 450]}},
                {"keyboard": {"keys": ["ctrl", "a"]}},
                {"keyboard": {"text": "hello"}},
            ],
        ),
        (
            "type_at",
            {"x": 250, "y": 500, "text": "hello"},
            [
                {"mouse": {"left_click": [400, 450]}},
                {"keyboard": {"text": "hello"}},
            ],
        ),
        (
            "key",
            {"name": "enter"},
            [{"keyboard": {"keys": ["Return"]}}],
        ),
        (
            "key_combination",
            {"key_combination": "ctrl+s"},
            [{"keyboard": {"keys": ["ctrl", "s"]}}],
        ),
        (
            "press_key_combination",
            {"combination": "ctrl+shift+x"},
            [{"keyboard": {"keys": ["ctrl", "shift", "x"]}}],
        ),
        (
            "keypress_at",
            {"keysymbol": "t", "modifiers": ["control", "alt"]},
            [{"keyboard": {"keys": ["ctrl", "alt", "t"]}}],
        ),
        (
            "type_key_combination",
            {"key_combination": "ctrl+d"},
            [{"keyboard": {"keys": ["ctrl", "d"]}}],
        ),
        (
            "scroll_at",
            {"x": 500, "y": 500, "direction": "down", "magnitude": 240},
            [
                {"mouse": {"move": [800, 450]}},
                {"mouse": {"scroll": 2}},
            ],
        ),
        (
            "mouse_scroll",
            {"direction": "down", "amount": 10},
            [{"mouse": {"scroll": 10}}],
        ),
        (
            "mouse_drag",
            {
                "source_x": 250,
                "source_y": 500,
                "destination_x": 500,
                "destination_y": 750,
            },
            [{"mouse": {"left_click_drag": [[400, 450], [800, 675]]}}],
        ),
        (
            "drag",
            {"x": 250, "y": 500, "to_x": 500, "to_y": 750},
            [{"mouse": {"left_click_drag": [[400, 450], [800, 675]]}}],
        ),
        (
            "drag_and_drop",
            {
                "source_x": 250,
                "source_y": 500,
                "target_x": 500,
                "target_y": 750,
            },
            [{"mouse": {"left_click_drag": [[400, 450], [800, 675]]}}],
        ),
        (
            "open_url",
            {"url": "https://example.com"},
            [
                {"keyboard": {"keys": ["ctrl", "l"]}},
                {"keyboard": {"text": "https://example.com"}},
                {"keyboard": {"keys": ["Return"]}},
            ],
        ),
        (
            "open_application",
            {"application": "libreoffice"},
            [
                {"keyboard": {"keys": ["super"]}},
                {"keyboard": {"text": "libreoffice"}},
                {"keyboard": {"keys": ["Return"]}},
            ],
        ),
        (
            "open_terminal",
            {},
            [{"keyboard": {"keys": ["ctrl", "alt", "t"]}}],
        ),
    ],
)
def test_gemini3_ui_action_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    arguments: dict,
    expected_actions: list[dict],
) -> None:
    agent = import_agent(monkeypatch, tmp_path, "gemini3_flash_preview")
    computer = DummyComputer()

    result = agent.execute_call(
        computer,
        {"name": name, "id": "alias-test", "arguments": arguments},
        1600,
        900,
    )

    assert result.result == {"ok": True, "actions": expected_actions}
    assert computer.steps == [expected_actions]


@pytest.mark.parametrize(
    "name",
    [
        "bash",
        "bash_run",
        "execute_shell",
        "run_bash",
        "run_bash_command",
        "run_command",
        "run_terminal_command",
        "run_tool",
    ],
)
def test_gemini3_shell_aliases_are_never_executed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    agent = import_agent(monkeypatch, tmp_path, "gemini3_flash_preview")
    computer = DummyComputer()

    result = agent.execute_call(
        computer,
        {
            "name": name,
            "id": "shell1",
            "arguments": {"command": "touch /tmp/should-not-exist"},
        },
        1600,
        900,
    )

    assert result.result["ok"] is False
    assert result.result["error"].startswith("shell execution is disabled")
    assert computer.steps == []


@pytest.mark.parametrize("template", ["gemini35", "gemini3_flash_preview"])
def test_gemini_retries_request_exceptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, template: str
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    agent = import_agent(monkeypatch, tmp_path, template)
    agent.MAX_RETRIES = 3
    monkeypatch.setattr(agent.time, "sleep", lambda _seconds: None)

    class Response:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {"id": "ok", "usage": {}}

    attempts = 0

    def post(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise agent.requests.ReadTimeout("timed out")
        return Response()

    monkeypatch.setattr(agent.requests, "post", post)

    result = agent.gemini_request({}, agent.CostTracker(), "test")

    assert result["id"] == "ok"
    assert attempts == 3


def test_gemini3_wait_accepts_minutes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path, "gemini3_flash_preview")
    computer = DummyComputer()

    result = agent.execute_call(
        computer,
        {"name": "wait", "id": "wait1", "arguments": {"minutes": 0.5}},
        1600,
        900,
    )

    assert computer.waits == [30.0]
    assert result.result["wait_seconds"] == 30.0


def test_gemini3_invalid_coordinates_are_returned_to_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path, "gemini3_flash_preview")
    computer = DummyComputer()

    result = agent.execute_call(
        computer,
        {"name": "click", "id": "click1", "arguments": {"coordinates": [1, 2]}},
        1600,
        900,
    )

    assert result.result == {
        "ok": False,
        "error": agent.INVALID_TOOL_ARGUMENTS,
    }
    assert computer.steps == []


@pytest.mark.parametrize("template", ["gemini35", "gemini3_flash_preview"])
def test_gemini_run_propagates_agent_failures_without_declaring_done(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, template: str
) -> None:
    agent = import_agent(monkeypatch, tmp_path, template)
    done = False

    class Computer:
        def __init__(self, _env_url: str) -> None:
            pass

        @staticmethod
        def observe() -> dict:
            return {"png": b"not-needed"}

        def done(self) -> None:
            nonlocal done
            done = True

    monkeypatch.setattr(agent, "Computer", Computer)
    monkeypatch.setattr(agent, "image_size", lambda _png: (1600, 900))
    monkeypatch.setattr(
        agent,
        "gemini_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("API unavailable")),
    )

    with pytest.raises(RuntimeError, match="API unavailable"):
        agent.run("http://environment", "task")

    assert done is False


@pytest.mark.parametrize("template", ["gemini35", "gemini3_flash_preview"])
def test_template_has_no_task_specific_resolvers(template: str) -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "agents"
        / template
        / "agent.py"
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


def test_stale_cost_reservations_are_released(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)
    agent.COST_LEDGER.write_text(
        json.dumps(
            {
                "actual_usd": 28.0,
                "reserved_usd": 1.0,
                "reservations": {
                    "dead": {
                        "amount_usd": 1.0,
                        "pid": 999_999_999,
                        "label": "old",
                        "created_at": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    reservation_id = agent.CostTracker().reserve("test")
    data = json.loads(agent.COST_LEDGER.read_text(encoding="utf-8"))

    assert "dead" not in data["reservations"]
    assert reservation_id in data["reservations"]
    assert data["reserved_usd"] == agent.COST_RESERVE_USD


@pytest.mark.parametrize("template", ["gemini35", "gemini3_flash_preview"])
def test_safety_policies_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, template: str
) -> None:
    agent = import_agent(monkeypatch, tmp_path, template)
    tool = agent.tools()[0]
    assert tool["type"] == "computer_use"
    assert "data_modification" in tool["disabled_safety_policies"]
    assert "financial_transactions" in tool["disabled_safety_policies"]


@pytest.mark.parametrize("template", ["gemini35", "gemini3_flash_preview"])
def test_safety_acknowledgement_is_top_level_in_function_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, template: str
) -> None:
    # The API reads safety_acknowledgement only as a top-level field of the
    # function_result; buried inside the serialized text body it is ignored
    # and the follow-up request is rejected with HTTP 400.
    agent = import_agent(monkeypatch, tmp_path, template)
    acked = agent.ExecutedCall(
        name="click",
        call_id="c1",
        summary="click",
        result={"ok": True, "safety_acknowledgement": True},
    )
    plain = agent.ExecutedCall(
        name="click", call_id="c2", summary="click", result={"ok": True}
    )
    results = agent.build_function_results([acked, plain], b"png-bytes")
    assert results[0]["safety_acknowledgement"] is True
    assert "safety_acknowledgement" not in results[1]
