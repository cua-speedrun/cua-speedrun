from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def import_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("GEMINI_COST_LEDGER", str(tmp_path / "gemini-new-cost.json"))
    for name in (
        "GEMINI_MODEL",
        "GEMINI_INPUT_RATE_PER_MILLION",
        "GEMINI_CACHED_INPUT_RATE_PER_MILLION",
        "GEMINI_OUTPUT_RATE_PER_MILLION",
        "GEMINI_MAX_OUTPUT_TOKENS",
        "GEMINI_THINKING_LEVEL",
    ):
        monkeypatch.delenv(name, raising=False)
    path = ROOT / "agents/gemini/agent.py"
    spec = importlib.util.spec_from_file_location("gemini_agent_test", path)
    assert spec is not None and spec.loader is not None
    agent = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = agent
    spec.loader.exec_module(agent)
    return agent


def image(index: int) -> dict[str, str]:
    return {
        "type": "image",
        "data": f"screenshot-{index}",
        "mime_type": "image/png",
    }


@pytest.mark.parametrize("call", [
    {"name": "click", "arguments": {"x": 100, "y": 100}},
    {"name": "wait", "arguments": {"seconds": 1}},
])
def test_action_failure_is_returned_to_model(monkeypatch, tmp_path, call):
    agent = import_agent(monkeypatch, tmp_path)

    class Computer:
        def step(self, actions):
            return {"ok": False, "error": "Action timed out after 90s."}

    result = agent.execute_call(Computer(), call, 1920, 1080)
    assert result.result["ok"] is False
    assert result.result["error"] == "Action timed out after 90s."
    assert result.summary == "Action timed out after 90s."


def realistic_history(count: int) -> list[dict]:
    steps: list[dict] = [
        {
            "type": "user_input",
            "content": [{"type": "text", "text": "Complete the desktop task."}],
        }
    ]
    for index in range(count):
        steps.extend(
            [
                {
                    "type": "thought",
                    "summary": [{"type": "text", "text": f"intent-{index}"}],
                    "signature": f"signed-thought-{index}",
                },
                {
                    "type": "function_call",
                    "name": "click" if index % 2 else "take_screenshot",
                    "id": f"call-{index}",
                    "arguments": {"x": index, "y": index, "intent": f"act-{index}"},
                },
                {
                    "type": "function_result",
                    "name": "click" if index % 2 else "take_screenshot",
                    "call_id": f"call-{index}",
                    "result": [
                        {
                            "type": "text",
                            "text": json.dumps({"ok": True, "turn": index}),
                        },
                        image(index),
                    ],
                },
            ]
        )
    return steps


def test_gemini_38_configuration_matches_current_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)

    assert agent.MODEL == "gemini-3.8-flash"
    assert agent.THINKING_LEVEL == "medium"
    assert agent.MAX_OUTPUT_TOKENS == 65536
    assert agent.REQUEST_TIMEOUT == 600
    assert agent.INPUT_RATE_PER_MILLION == 0.75
    assert agent.CACHED_INPUT_RATE_PER_MILLION == 0.075
    assert agent.OUTPUT_RATE_PER_MILLION == 3.75
    assert agent.MAX_ACTIVE_SCREENSHOTS == 20
    assert agent.SCREENSHOTS_PER_FOLD == 10

    config = agent.base_payload()["generation_config"]
    assert config == {
        "max_output_tokens": 65536,
        "thinking_level": "medium",
        "tool_choice": "auto",
    }
    assert not {"temperature", "top_p", "top_k", "candidate_count"} & config.keys()


def test_invalid_gemini_38_thinking_level_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "minimal")
    monkeypatch.setenv("GEMINI_COST_LEDGER", str(tmp_path / "cost.json"))
    path = ROOT / "agents/gemini/agent.py"
    spec = importlib.util.spec_from_file_location("gemini_invalid_effort", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(ValueError, match="low, medium, or high"):
        spec.loader.exec_module(module)


def test_folding_preserves_interleaved_non_image_objects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)
    history = realistic_history(21)

    folded = agent.fold_oldest_screenshots(history, 10)

    assert agent.count_screenshots(history) == 21
    assert agent.count_screenshots(folded) == 11
    assert json.dumps(folded).count(agent.COLLAPSED_SCREENSHOT_TEXT) == 10
    assert [step for step in folded if step["type"] == "thought"] == [
        step for step in history if step["type"] == "thought"
    ]
    assert [step for step in folded if step["type"] == "function_call"] == [
        step for step in history if step["type"] == "function_call"
    ]
    original_text = [
        part
        for step in history
        if step["type"] == "function_result"
        for part in step["result"]
        if part["type"] == "text"
    ]
    folded_text = [
        part
        for step in folded
        if step["type"] == "function_result"
        for part in step["result"]
        if part["type"] == "text" and part["text"] != agent.COLLAPSED_SCREENSHOT_TEXT
    ]
    assert folded_text == original_text


def test_interaction_history_walks_stored_chain_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)
    stored = {
        "root": {"id": "root", "steps": [{"type": "user_input", "content": []}]},
        "middle": {
            "id": "middle",
            "previous_interaction_id": "root",
            "steps": [{"type": "function_call", "id": "one"}],
        },
        "latest": {
            "id": "latest",
            "previous_interaction_id": "middle",
            "steps": [{"type": "function_result", "call_id": "one"}],
        },
    }
    requested: list[str] = []

    def get(interaction_id: str) -> dict:
        requested.append(interaction_id)
        return stored[interaction_id]

    monkeypatch.setattr(agent, "gemini_get", get)

    assert agent.interaction_history("latest") == [
        {"type": "user_input", "content": []},
        {"type": "function_call", "id": "one"},
        {"type": "function_result", "call_id": "one"},
    ]
    assert requested == ["latest", "middle", "root"]


def test_twenty_first_screenshot_reroots_with_eleven_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)
    history = realistic_history(20)
    monkeypatch.setattr(agent, "interaction_history", lambda _interaction_id: history)
    next_input = [
        {
            "type": "function_result",
            "name": "click",
            "call_id": "call-20",
            "result": [{"type": "text", "text": '{"ok": true}'}, image(20)],
        }
    ]

    payload, active = agent.bounded_continue_payload("latest", next_input, 20)

    assert "previous_interaction_id" not in payload
    assert payload["store"] is True
    assert active == 11
    assert agent.count_screenshots(payload["input"]) == 11
    assert json.dumps(payload["input"]).count(agent.COLLAPSED_SCREENSHOT_TEXT) == 10
    assert any(
        step.get("signature") == "signed-thought-19" for step in payload["input"]
    )


def test_first_twenty_screenshots_remain_stateful(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)
    monkeypatch.setattr(
        agent,
        "interaction_history",
        lambda _interaction_id: pytest.fail(
            "history must not be fetched before folding"
        ),
    )

    payload, active = agent.bounded_continue_payload("turn-19", [image(19)], 19)

    assert payload["previous_interaction_id"] == "turn-19"
    assert active == 20


def test_cost_uses_cached_input_and_emits_cumulative_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = import_agent(monkeypatch, tmp_path)
    tracker = agent.CostTracker()
    usage = {
        "total_input_tokens": 1_000,
        "total_cached_tokens": 600,
        "total_output_tokens": 200,
        "total_thought_tokens": 300,
        "total_tool_use_tokens": 100,
    }
    expected = (400 * 0.75 + 600 * 0.075 + 100 * 0.75 + 500 * 3.75) / 1_000_000

    tracker.release(tracker.reserve("one"), usage)
    tracker.release(tracker.reserve("two"), usage)

    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(agent.COST_SNAPSHOT_PREFIX)
    ]
    snapshot = json.loads(lines[-1][len(agent.COST_SNAPSHOT_PREFIX) :])
    assert snapshot["cost_usd"] == pytest.approx(2 * expected)
    assert snapshot["usage"] == {
        "requests": 2,
        "input_tokens": 2_000,
        "cached_input_tokens": 1_200,
        "uncached_input_tokens": 800,
        "output_tokens": 400,
        "thought_tokens": 600,
        "tool_use_tokens": 200,
    }


def test_template_is_listed() -> None:
    from cua_speedrun.service.templates_catalog import list_templates

    template = next(item for item in list_templates() if item["name"] == "gemini")
    assert template["required_environment_variables"] == ["GEMINI_API_KEY"]
    assert "Gemini 3.8 Flash" in template["description"]


def test_init_omits_deprecated_sampling_parameters() -> None:
    source = (ROOT / "agents/gemini/init.py").read_text(encoding="utf-8")
    assert "temperature" not in source
    assert "top_p" not in source
    assert "top_k" not in source


def test_agent_always_marks_environment_done(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)

    class BrokenComputer:
        done_called = False

        def __init__(self, _env_url: str) -> None:
            pass

        def observe(self):
            raise RuntimeError("request failed")

        def done(self) -> None:
            type(self).done_called = True

    monkeypatch.setattr(agent, "Computer", BrokenComputer)

    agent.run("http://environment", "task")

    assert BrokenComputer.done_called is True


def test_agent_does_not_finalize_after_transport_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = import_agent(monkeypatch, tmp_path)

    class DisconnectedComputer:
        done_called = False

        def __init__(self, _env_url: str) -> None:
            pass

        def observe(self):
            raise agent.requests.ConnectionError("gateway disconnected")

        def done(self) -> None:
            type(self).done_called = True

    monkeypatch.setattr(agent, "Computer", DisconnectedComputer)

    with pytest.raises(agent.requests.ConnectionError, match="gateway disconnected"):
        agent.run("http://environment", "task")

    assert DisconnectedComputer.done_called is False
