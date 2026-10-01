from __future__ import annotations

import importlib.util
import io
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


# Every env var the two templates actually read (kept in sync with the
# os.environ.get calls in agents/glm5v_turbo/agent.py and
# agents/minimax_m3/agent.py).
_ENV_VARS = (
    "OPENROUTER_API_KEY",
    "CS_MAX_STEPS",
    "GLM_MODEL",
    "GLM_BASE_URL",
    "GLM_API_KEY",
    "GLM_MAX_STEPS",
    "GLM_HTTP_TIMEOUT",
    "GLM_MAX_RETRIES",
    "GLM_TEMPERATURE",
    "GLM_TOP_P",
    "GLM_MAX_TOKENS",
    "GLM_THINKING",
    "GLM_HISTORY_IMAGES",
    "GLM_MAX_CONSECUTIVE_PARSE_FAILURES",
    "GLM_CLIENT_PASSWORD",
    "GLM_ENV_HTTP_TIMEOUT",
    "GLM_ENABLE_LEFT_CLICK_HOLD",
    "GLM_TASK_ID",
    "GLM_PROVIDER_ORDER",
    "GLM_PROVIDER_ALLOW_FALLBACKS",
    "MINIMAX_PROVIDER_ORDER",
    "MINIMAX_PROVIDER_ALLOW_FALLBACKS",
    "MINIMAX_MODEL",
    "MINIMAX_BASE_URL",
    "MINIMAX_API_KEY",
    "MINIMAX_HTTP_TIMEOUT",
    "MINIMAX_HTTP_MAX_RETRIES",
    "MINIMAX_MAX_LLM_RETRIES",
    "MINIMAX_TEMPERATURE",
    "MINIMAX_MAX_TOKENS",
    "MINIMAX_ONLY_N_MOST_RECENT_IMAGES",
    "MINIMAX_IMAGE_TRUNCATION_THRESHOLD",
    "MINIMAX_CLIENT_PASSWORD",
    "MINIMAX_ENV_HTTP_TIMEOUT",
    "M3_IMAGE_FORMAT",
    "M3_IMAGE_QUALITY",
)


def import_agent(
    monkeypatch: pytest.MonkeyPatch, template: str, env: dict[str, str] | None = None
):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    path = Path(__file__).resolve().parents[1] / "agents" / template / "agent.py"
    spec = importlib.util.spec_from_file_location(f"{template}_agent_test", path)
    assert spec is not None and spec.loader is not None
    agent = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = agent
    spec.loader.exec_module(agent)
    return agent


def tiny_png() -> bytes:
    from PIL import Image

    image = Image.new("RGB", (8, 8), (200, 30, 30))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


# --------------------------------------------------------------------------
# GLM (agents/glm5v_turbo)
# --------------------------------------------------------------------------


def test_glm_defaults_and_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "glm5v_turbo")

    # GLM-5.2 is text-only on OpenRouter; the GUI-capable vision variant
    # (the OSWorld-V2 GLM eval agent's own default model) is used instead.
    assert agent.MODEL == "z-ai/glm-5v-turbo"
    assert agent.BASE_URL == "https://openrouter.ai/api/v1"
    assert agent.API_URL == "https://openrouter.ai/api/v1/chat/completions"
    # Reference max_trajectory_length default.
    assert agent.MAX_STEPS == 50
    assert agent.HISTORY_IMAGES == 4

    # The action space rides in the prompt text (GLM-V convention), with
    # thousandths coordinates.
    for token in (
        "{left,right,middle}_click", "left_drag", "start_box", "scroll", "WAIT", "DONE",
    ):
        assert token in agent.ACTION_SPACE
    assert "0-999" in agent.ACTION_SPACE
    assert "thousandths (0-999)" in agent.PROMPT_HEAD
    assert "{task}" in agent.PROMPT_HEAD
    assert "{memory}" in agent.PROMPT_TAIL


def test_glm_payload_matches_vendor_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "glm5v_turbo")

    payload = agent.build_payload([{"role": "user", "content": "hi"}])
    # Exactly the sampling the OSWorld-V2 glm_eval_agent sends -- nothing more.
    assert set(payload) == {
        "model", "messages", "temperature", "top_p", "max_tokens", "thinking", "stream",
    }
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 1.0
    assert payload["max_tokens"] == 8192
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["stream"] is False


def test_glm_action_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "glm5v_turbo")

    computer = DummyComputer()
    outcome = agent.convert_and_execute(
        computer, "left_click(start_box='[500,500]', element_info='OK')", 1600, 900
    )
    assert not outcome.terminal
    assert computer.steps == [[{"mouse": {"left_click": [800, 450]}}]]

    computer = DummyComputer()
    agent.convert_and_execute(
        computer,
        "left_drag(start_box='[250,750]', start_element_info='a', "
        "end_box='[500,500]', end_element_info='b')",
        1600,
        900,
    )
    assert computer.steps == [
        [{"mouse": {"left_click_drag": [[400, 675], [800, 450]]}}]
    ]

    computer = DummyComputer()
    agent.convert_and_execute(
        computer, "left_double_click(start_box='[0,999]', element_info='x')", 1600, 900
    )
    assert computer.steps == [[{"mouse": {"double_click": [0, 899]}}]]

    computer = DummyComputer()
    agent.convert_and_execute(computer, "key(keys='ctrl+c')", 1600, 900)
    assert computer.steps == [[{"keyboard": {"keys": ["ctrl", "c"]}}]]

    computer = DummyComputer()
    agent.convert_and_execute(computer, "key(keys='enter')", 1600, 900)
    assert computer.steps == [[{"keyboard": {"keys": ["Return"]}}]]

    computer = DummyComputer()
    agent.convert_and_execute(computer, "type(content='ab\ncd')", 1600, 900)
    assert computer.steps == [
        [
            {"keyboard": {"text": "ab"}},
            {"keyboard": {"keys": ["Return"]}},
            {"keyboard": {"text": "cd"}},
        ]
    ]

    computer = DummyComputer()
    agent.convert_and_execute(computer, "type(content='it’s — fine')", 1600, 900)
    assert computer.steps == [[{"keyboard": {"text": "it's - fine"}}]]

    # hold_keys wraps the click in key_down/key_up with the reference's
    # 0.1s sleeps (upstream _wrap_with_hold_keys) as env wait actions.
    computer = DummyComputer()
    agent.convert_and_execute(
        computer,
        "left_click(start_box='[500,500]', hold_keys='ctrl', element_info='row')",
        1600,
        900,
    )
    assert computer.steps == [
        [
            {"keyboard": {"key_down": "ctrl"}},
            {"action": "wait", "time": 0.1},
            {"mouse": {"left_click": [800, 450]}},
            {"action": "wait", "time": 0.1},
            {"keyboard": {"key_up": "ctrl"}},
        ]
    ]


def test_glm_scroll_sign_matches_gym_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "glm5v_turbo")

    # Reference emits pyautogui.scroll(-step) for "down"; the env convention
    # is inverted (positive = down).
    computer = DummyComputer()
    agent.convert_and_execute(
        computer,
        "scroll(start_box='[500,500]', direction='down', step=5, element_info='page')",
        1600,
        900,
    )
    assert computer.steps == [
        [{"mouse": {"move": [800, 450]}}, {"mouse": {"scroll": 5}}]
    ]

    computer = DummyComputer()
    agent.convert_and_execute(
        computer,
        "scroll(start_box='[500,500]', direction='up', element_info='page')",
        1600,
        900,
    )
    assert computer.steps == [
        [{"mouse": {"move": [800, 450]}}, {"mouse": {"scroll": -5}}]
    ]


def test_glm_terminal_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "glm5v_turbo")

    computer = DummyComputer()
    outcome = agent.convert_and_execute(computer, "FAIL()", 1600, 900)
    assert outcome.terminal is True
    assert computer.steps == [[{"action_type": "FAIL"}]]

    computer = DummyComputer()
    outcome = agent.convert_and_execute(computer, "DONE()", 1600, 900)
    assert outcome.terminal is True
    assert computer.steps == []

    computer = DummyComputer()
    outcome = agent.convert_and_execute(computer, "WAIT()", 1600, 900)
    assert outcome.terminal is False
    assert computer.waits == [5.0]


def test_glm_invalid_arguments_execute_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "glm5v_turbo")

    computer = DummyComputer()
    outcome = agent.convert_and_execute(computer, "left_click(element_info='no box')", 1600, 900)
    assert not outcome.executed
    assert computer.steps == []

    computer = DummyComputer()
    outcome = agent.convert_and_execute(computer, "fly(start_box='[1,1]')", 1600, 900)
    assert not outcome.executed
    assert computer.steps == []

    assert agent.clamp_xy(float("nan"), 5, 1600, 900) == (None, None)
    assert agent.clamp_xy(2000, -50, 1600, 900) == (1599, 0)


def test_glm_parse_response(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "glm5v_turbo")

    text = (
        "I will click the OK button.\n"
        "Action: <|begin_of_box|>left_click(start_box='[500,500]', "
        "element_info='OK')<|end_of_box|>\n"
        "Memory:\n[{\"key\": \"value\"}]"
    )
    parsed = agent.parse_response(text)
    assert parsed["action"].startswith("left_click(start_box='[500,500]'")
    assert parsed["memory"] == '[{"key": "value"}]'
    assert "click the OK button" in parsed["thought"]

    # Fallback: no box markers survived the serving stack.
    parsed = agent.parse_response(
        "Click it.\nleft_click(start_box='[100,200]', element_info='x')\nMemory:\n[]"
    )
    assert parsed["action"].startswith("left_click(start_box='[100,200]'")
    assert parsed["memory"] == "[]"

    # Bare terminals normalize to the reference's canonical DONE() form.
    parsed = agent.parse_response("All done. <|begin_of_box|>DONE<|end_of_box|>")
    assert parsed["action"] == "DONE()"

    assert agent.parse_response("just thinking out loud")["action"] is None


def test_glm_history_keeps_last_four_screenshots(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "glm5v_turbo")

    png = tiny_png()
    history = agent.GlmHistory()
    for i in range(6):
        history.add(f"thought {i}", f"key(keys='f{i}')", png)

    # Raw bytes retained only for the last HISTORY_IMAGES steps.
    assert [shot is None for shot in history.screenshots] == [
        True, True, False, False, False, False,
    ]

    message = agent.build_user_message("do the thing", history, png)
    assert message["role"] == "user"
    images = [part for part in message["content"] if part["type"] == "image_url"]
    # 4 shrunk history screenshots + the current full-size screenshot.
    assert len(images) == agent.HISTORY_IMAGES + 1
    text = "".join(part["text"] for part in message["content"] if part["type"] == "text")
    assert "do the thing" in text
    assert text.count("(Omitted in context.)") == 2
    assert "step 6" in text


# --------------------------------------------------------------------------
# GLM upstream-parity tests
# --------------------------------------------------------------------------


def test_glm_prompts_verbatim_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prompt text must stay byte-identical to the vendor reference.

    Hashes are sha256 of the upstream constants in
    xlang-ai/OSWorld-V2 mm_agents/glm_prompts.py (main @ 2026-07-21):
      * EVAL_ACTION_SPACE_UNIFIED_TRIPLE (left_click_hold is a separate
        task-specific constant upstream and is deliberately excluded)
      * EVAL_USER_INSERT_HEAD
      * EVAL_USER_INSERT_TAIL.format(memory="[]", user_response_block="")
        (the template additionally templates the sudo password; rendered
        with the upstream default "password" they must match exactly)
    """
    import hashlib

    agent = import_agent(monkeypatch, "glm5v_turbo")

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    assert digest(agent.ACTION_SPACE) == (
        "6aaa1a157a5e8281f39f437cd8714e58da91e760f1063fe5736ed5774fe37e60"
    )
    assert digest(agent.PROMPT_HEAD) == (
        "015d2005bb6be722840a36c9a1ddb44f4ed4c3b31e8b7903e79202a6e34291c7"
    )
    rendered_tail = agent.PROMPT_TAIL.format(
        memory="[]", client_password="password", user_response_block=""
    )
    assert digest(rendered_tail) == (
        "982ea3370b231d476bb047aea4082278d1e2c52936dee94d57d12bd0c9771536"
    )
    # Check the required action arguments and user-response prompt block.
    assert "'required': ['start_box', 'element_info']" in agent.ACTION_SPACE
    assert "- If a latest user response is provided" in agent.PROMPT_TAIL


def test_glm_coordinate_scaling_matches_upstream_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """clamp_xy must reproduce the reference's exact float expression.

    Upstream (glm_eval_agent._action2pyautogui):
        int(x * (img_width / 1000))
    Writing it as int(x / 1000 * width) differs by one pixel for some
    inputs (e.g. x=175 at width=720: 126 vs 125), so the exact expression
    order is load-bearing.
    """
    agent = import_agent(monkeypatch, "glm5v_turbo")

    for width, height in ((1920, 1080), (1280, 720), (1366, 768), (800, 600)):
        for v in range(0, 1000):
            up_x = int(v * (width / 1000))  # vendored upstream expression
            up_y = int(v * (height / 1000))
            assert agent.clamp_xy(v, v, width, height) == (
                min(up_x, width - 1),
                min(up_y, height - 1),
            ), f"divergence at v={v} {width}x{height}"
    # Template-only robustness on inputs upstream would crash on.
    assert agent.clamp_xy(1000, 1000, 1920, 1080) == (1919, 1079)
    assert agent.clamp_xy(-5, 999.9, 1920, 1080) == (0, 1079)


def test_glm_parser_balanced_trimming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Action calls are trimmed at the balanced closing paren, so trailing
    prose cannot leak into string params (upstream _find_last_action_by_parsing)."""
    agent = import_agent(monkeypatch, "glm5v_turbo")

    # Trailing prose after the call must not be part of the action.
    parsed = agent.parse_response(
        "<think>t</think>I will click.\n"
        "left_click(start_box='[500,300]', element_info='OK') "
        "That should work (hopefully).\nMemory:\n[]"
    )
    assert parsed["action"] == "left_click(start_box='[500,300]', element_info='OK')"

    # type(content=...) followed by prose containing an apostrophe: the old
    # rfind-based slice corrupted the typed text to "hello') Now let".
    parsed = agent.parse_response(
        "<think>t</think>type(content='hello') Now let's wait and see.\nMemory:\n[]"
    )
    assert parsed["action"] == "type(content='hello')"

    # Unclosed box: the action must not swallow the Memory block.
    parsed = agent.parse_response(
        "<think>t</think>go <|begin_of_box|>scroll(start_box='[500,500]', "
        "direction='down', step=3, element_info='page')\nMemory:\n[]"
    )
    assert parsed["action"] == (
        "scroll(start_box='[500,500]', direction='down', step=3, element_info='page')"
    )

    # An action name inside a previous action's string args is "covered" and
    # must not be picked over the real (earlier-starting) call.
    parsed = agent.parse_response(
        "<think>t</think><|begin_of_box|>type(content='later run "
        "left_click(start_box=(1,2)) manually')<|end_of_box|>\nMemory:\n[]"
    )
    assert parsed["action"].startswith("type(content='later run")
    assert parsed["action"].endswith("manually')")


def test_glm_parser_terminal_token_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal tokens match case-insensitively and normalize to upper-case
    with parens (upstream _normalize_action_call + IGNORECASE terminal regex)."""
    agent = import_agent(monkeypatch, "glm5v_turbo")

    assert (
        agent.parse_response("<think>t</think>finish <|begin_of_box|>done()<|end_of_box|>\nMemory:\n[]")["action"]
        == "DONE()"
    )
    assert (
        agent.parse_response("<think>t</think>this is impossible. fail()\nMemory:\n[]")["action"]
        == "FAIL()"
    )
    assert (
        agent.parse_response("<think>t</think>Nothing to do yet, WAIT\nMemory:\n[]")["action"]
        == "WAIT()"
    )
    # convert_and_execute accepts the normalized form.
    computer = DummyComputer()
    outcome = agent.convert_and_execute(computer, "DONE()", 1600, 900)
    assert outcome.terminal


def test_glm_no_full_text_fallback_when_box_markers_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream only rescues box-less responses (_wrap_last_action_with_box_
    if_needed); when markers exist but contain no action, it refuses a
    full-response fallback to avoid template-text pollution."""
    agent = import_agent(monkeypatch, "glm5v_turbo")

    text = (
        "<think>t</think>Earlier I did left_click(start_box='[1,2]', "
        "element_info='old'). <|begin_of_box|>no action here<|end_of_box|>\nMemory:\n[]"
    )
    assert agent.parse_response(text)["action"] is None


def test_glm_history_thought_drops_action_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reference removes the action call from the thought before writing
    the history line (clean_thought_text), so it is not duplicated."""
    agent = import_agent(monkeypatch, "glm5v_turbo")

    png = tiny_png()
    history = agent.GlmHistory()
    action = "left_click(start_box='[500,300]', element_info='OK')"
    history.add(f"I will click the OK button.\n{action}", action, png)
    message = agent.build_user_message("task", history, png)
    text = "".join(
        part["text"] for part in message["content"] if part["type"] == "text"
    )
    assert "Thought: I will click the OK button.\nAction: left_click" in text
    assert text.count(action) == 1  # only the Action: line


def test_glm_memory_strips_special_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "glm5v_turbo")

    parsed = agent.parse_response(
        "<think>t</think><|begin_of_box|>WAIT()<|end_of_box|>\n"
        'Memory:\n[{"k": "v"}]<|user|>'
    )
    assert parsed["memory"] == '[{"k": "v"}]'


def test_glm_key_repeated_tail_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream _build_hotkey_command: a repeated tail key means the combo is
    pressed multiple times, not one chord with a duplicated key."""
    agent = import_agent(monkeypatch, "glm5v_turbo")

    computer = DummyComputer()
    agent.convert_and_execute(computer, "key(keys='down+down')", 1600, 900)
    assert computer.steps == [
        [{"keyboard": {"keys": ["Down"]}}, {"keyboard": {"keys": ["Down"]}}]
    ]

    computer = DummyComputer()
    agent.convert_and_execute(computer, "key(keys='ctrl+tab+tab')", 1600, 900)
    assert computer.steps == [
        [{"keyboard": {"keys": ["ctrl", "Tab"]}}, {"keyboard": {"keys": ["ctrl", "Tab"]}}]
    ]

    # No repetition: single chord, unchanged.
    computer = DummyComputer()
    agent.convert_and_execute(computer, "key(keys='ctrl+shift+t')", 1600, 900)
    assert computer.steps == [[{"keyboard": {"keys": ["ctrl", "shift", "t"]}}]]


def test_glm_cmd_maps_to_super(monkeypatch: pytest.MonkeyPatch) -> None:
    """pyautogui maps 'command'/'cmd' to Super_L on X11, so the reference's
    pass-through presses the Super key -- not ctrl."""
    agent = import_agent(monkeypatch, "glm5v_turbo")

    assert agent.map_key("cmd") == "super"
    assert agent.map_key("command") == "super"


# --------------------------------------------------------------------------
# MiniMax M3 upstream-parity tests
# --------------------------------------------------------------------------


def test_m3_system_prompt_verbatim_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """sha256 of M3_SYSTEM_PROMPT_TEMPLATE from OSWorld-V2 mm_agents/m3/
    prompts.py (main @ 2026-07-21). The template must carry it verbatim."""
    import hashlib

    agent = import_agent(monkeypatch, "minimax_m3")
    assert hashlib.sha256(agent.M3_SYSTEM_PROMPT_TEMPLATE.encode()).hexdigest() == (
        "9318fb5ebd444231d10c7302fb182f92e324a536616e8927754a4eb49ae4e94e"
    )


def test_m3_provider_pin_injected_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(
        monkeypatch,
        "minimax_m3",
        env={
            "MINIMAX_PROVIDER_ORDER": "minimax, novita",
            "MINIMAX_BASE_URL": "https://openrouter.ai/api",
        },
    )
    body = agent.build_request_body([{"role": "user", "content": "hi"}])
    # OpenRouter provider-routing preference: slugs in order, fallbacks OFF
    # (pin hard) by default. Whitespace around comma-separated slugs is trimmed.
    assert body["provider"] == {"order": ["minimax", "novita"], "allow_fallbacks": False}
    # The reference fields are still all present and unchanged.
    assert set(body) == {
        "model", "messages", "max_tokens", "system", "temperature",
        "stop_sequences", "provider",
    }


def test_m3_provider_pin_allow_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(
        monkeypatch,
        "minimax_m3",
        env={
            "MINIMAX_PROVIDER_ORDER": "minimax",
            "MINIMAX_PROVIDER_ALLOW_FALLBACKS": "1",
            "MINIMAX_BASE_URL": "https://openrouter.ai/api",
        },
    )
    body = agent.build_request_body([{"role": "user", "content": "hi"}])
    assert body["provider"] == {"order": ["minimax"], "allow_fallbacks": True}


def test_m3_provider_pin_not_sent_to_native_minimax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``provider`` is an OpenRouter extension and is off-spec on the native
    MiniMax endpoint, so it must never be sent there even when pinned."""
    agent = import_agent(
        monkeypatch,
        "minimax_m3",
        env={"MINIMAX_PROVIDER_ORDER": "minimax"},
    )
    assert agent.BASE_URL == "https://api.minimax.io/anthropic"
    body = agent.build_request_body([{"role": "user", "content": "hi"}])
    assert "provider" not in body
    # Byte-identical to the body from before the provider feature existed.
    assert set(body) == {
        "model", "messages", "max_tokens", "system", "temperature",
        "stop_sequences",
    }


def test_m3_openrouter_host_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "minimax_m3", env={})
    assert agent.is_openrouter_endpoint("https://openrouter.ai/api")
    assert agent.is_openrouter_endpoint("https://OpenRouter.ai/api/v1")
    assert not agent.is_openrouter_endpoint("https://api.minimax.io/anthropic")
    assert not agent.is_openrouter_endpoint("https://api.minimaxi.com/anthropic")


def test_m3_provider_pin_absent_for_blank_env(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(
        monkeypatch, "minimax_m3", env={"MINIMAX_PROVIDER_ORDER": "  ,  "}
    )
    body = agent.build_request_body([{"role": "user", "content": "hi"}])
    assert "provider" not in body


def test_glm_provider_pin_injected_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(
        monkeypatch, "glm5v_turbo", env={"GLM_PROVIDER_ORDER": "z-ai"}
    )
    payload = agent.build_payload([{"role": "user", "content": "hi"}])
    assert payload["provider"] == {"order": ["z-ai"], "allow_fallbacks": False}


def test_glm_provider_pin_absent_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "glm5v_turbo")
    payload = agent.build_payload([{"role": "user", "content": "hi"}])
    assert "provider" not in payload


def test_m3_coordinate_scaling_matches_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reference parser (coordinate_type='relative'):
    int(x * original_width / 1000); the template adds a screen clamp."""
    agent = import_agent(monkeypatch, "minimax_m3")

    for width, height in ((1920, 1080), (1280, 720), (999, 601)):
        for v in range(0, 1001, 7):
            resp = (
                '<tool_call>\n{"name": "computer", "arguments": '
                '{"action": "mouse_move", "coordinate": [%d, %d]}}\n</tool_call>' % (v, v)
            )
            _, items = agent.parse_m3_response(resp, width, height)
            got = items[0]["actions"][0]["mouse"]["move"]
            up = [int(v * width / 1000), int(v * height / 1000)]  # vendored upstream
            expect = [min(max(up[0], 0), width - 1), min(max(up[1], 0), height - 1)]
            assert got == expect, f"divergence at v={v} {width}x{height}"


def tool_call(action: str, **kwargs) -> str:
    import json

    args = {"action": action, **kwargs}
    return (
        "<tool_call>\n"
        + json.dumps({"name": "computer", "arguments": args})
        + "\n</tool_call>"
    )


def test_m3_parser_core_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "minimax_m3")

    instr, items = agent.parse_m3_response(
        "Action: Click OK\n" + tool_call("left_click", coordinate=[500, 300]),
        1920, 1080,
    )
    assert instr == "Click OK"
    assert items == [{"actions": [{"mouse": {"left_click": [960, 324]}}]}]

    # Stop-sequence recovery: the serving stack eats "</tool_call>".
    _, items = agent.parse_m3_response(
        'Action: x\n<tool_call>\n{"name": "computer", "arguments": '
        '{"action": "left_click", "coordinate": [500, 300]}}',
        1920, 1080,
    )
    assert items == [{"actions": [{"mouse": {"left_click": [960, 324]}}]}]

    # Bare JSON fallback (wrapper stripped upstream).
    _, items = agent.parse_m3_response(
        '{"name": "computer", "arguments": {"action": "key", "text": "ctrl+s"}}',
        1920, 1080,
    )
    assert items == [{"actions": [{"keyboard": {"keys": ["ctrl", "s"]}}]}]

    # Two tool calls in one turn stay in emission order.
    _, items = agent.parse_m3_response(
        tool_call("left_click", coordinate=[100, 100])
        + "\n"
        + tool_call("key", text="Return"),
        1920, 1080,
    )
    assert items == [
        {"actions": [{"mouse": {"left_click": [192, 108]}}]},
        {"actions": [{"keyboard": {"keys": ["Return"]}}]},
    ]


def test_m3_parser_bpe_artifact_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "minimax_m3")

    _, items = agent.parse_m3_response(
        tool_call("left_ click", coordinate=[10, 10]), 1920, 1080
    )
    assert items == [{"actions": [{"mouse": {"left_click": [19, 10]}}]}]

    _, items = agent.parse_m3_response(
        tool_call("left_lick", coordinate=[10, 10]), 1920, 1080
    )
    assert items == [{"actions": [{"mouse": {"left_click": [19, 10]}}]}]

    # Truncated action tail -> no-op sleep, mirroring upstream.
    _, items = agent.parse_m3_response(tool_call("left_"), 1920, 1080)
    assert items == [{"wait": 0.1}]


def test_m3_infeasible_and_terminals(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "minimax_m3")

    instr, items = agent.parse_m3_response(
        "This cannot be done. [INFEASIBLE]", 1920, 1080
    )
    assert (instr, items) == ("[INFEASIBLE]", ["FAIL"])

    _, items = agent.parse_m3_response(
        tool_call("terminate", status="failure"), 1920, 1080
    )
    assert items == ["FAIL"]
    _, items = agent.parse_m3_response(
        tool_call("terminate", status="success"), 1920, 1080
    )
    assert items == ["DONE"]
    _, items = agent.parse_m3_response(tool_call("done"), 1920, 1080)
    assert items == ["DONE"]


def test_m3_scroll_signs_match_gym_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reference emits pyautogui.scroll(+amount) for up / -amount for down;
    the env convention is inverted (positive = down)."""
    agent = import_agent(monkeypatch, "minimax_m3")

    _, items = agent.parse_m3_response(
        tool_call("scroll", coordinate=[500, 500], scroll_direction="down", scroll_amount=3),
        1920, 1080,
    )
    assert items == [
        {"actions": [{"mouse": {"move": [960, 540]}}, {"mouse": {"scroll": 3}}]}
    ]

    _, items = agent.parse_m3_response(
        tool_call("scroll", scroll_direction="up", scroll_amount=2), 1920, 1080
    )
    assert items == [{"actions": [{"mouse": {"scroll": -2}}]}]

    # Horizontal: shift+wheel emulation, positive = right.
    _, items = agent.parse_m3_response(
        tool_call("scroll", scroll_direction="right", scroll_amount=2), 1920, 1080
    )
    assert items == [
        {
            "actions": [
                {"keyboard": {"keys_down": ["shift"]}},
                {"mouse": {"scroll": 2}},
                {"keyboard": {"keys_up": ["shift"]}},
            ]
        }
    ]


def test_m3_modifier_click_and_clickless_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch, "minimax_m3")

    _, items = agent.parse_m3_response(
        tool_call("left_click", coordinate=[500, 500], text="ctrl+shift"), 1920, 1080
    )
    assert items == [
        {
            "actions": [
                {"keyboard": {"keys_down": ["ctrl", "shift"]}},
                {"mouse": {"left_click": [960, 540]}},
                {"keyboard": {"keys_up": ["shift", "ctrl"]}},
            ]
        }
    ]

    # Reference clicks at the current position when no coordinate is given;
    # the env emulates that with button events at the cursor.
    _, items = agent.parse_m3_response(tool_call("left_click"), 1920, 1080)
    assert items == [
        {
            "actions": [
                {"mouse": {"buttons": {"left_down": True}}},
                {"mouse": {"buttons": {"left_up": True}}},
            ]
        }
    ]

    # right_click without coordinate: pyautogui.rightClick() upstream; the
    # env's buttons channel supports right_down/right_up.
    _, items = agent.parse_m3_response(tool_call("right_click"), 1920, 1080)
    assert items == [
        {
            "actions": [
                {"mouse": {"buttons": {"right_down": True}}},
                {"mouse": {"buttons": {"right_up": True}}},
            ]
        }
    ]

    # middle_click without coordinate has no env equivalent -> no-op sleep.
    _, items = agent.parse_m3_response(tool_call("middle_click"), 1920, 1080)
    assert items == [{"wait": 0.1}]


def test_m3_drag_uses_tracked_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "minimax_m3")

    _, items = agent.parse_m3_response(
        tool_call("left_click_drag", start_coordinate=[100, 100], coordinate=[300, 300]),
        1920, 1080,
    )
    assert items == [
        {"actions": [{"mouse": {"left_click_drag": [[192, 108], [576, 324]]}}]}
    ]

    # Without start_coordinate the reference drags from the current mouse
    # position; the template tracks the cursor across steps for that.
    cursor: list = []
    agent.parse_m3_response(
        tool_call("mouse_move", coordinate=[100, 100]), 1920, 1080, cursor
    )
    _, items = agent.parse_m3_response(
        tool_call("left_click_drag", coordinate=[300, 300]), 1920, 1080, cursor
    )
    assert items == [
        {"actions": [{"mouse": {"left_click_drag": [[192, 108], [576, 324]]}}]}
    ]

    # Unknown cursor -> no-op (upstream would drag from wherever the OS
    # cursor happens to be; the env cannot query it).
    _, items = agent.parse_m3_response(
        tool_call("left_click_drag", coordinate=[300, 300]), 1920, 1080, []
    )
    assert items == [{"wait": 0.1}]


def test_m3_type_and_key_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "minimax_m3")

    _, items = agent.parse_m3_response(
        tool_call("type", text="ab\ncd"), 1920, 1080
    )
    assert items == [
        {
            "actions": [
                {"keyboard": {"text": "ab"}},
                {"keyboard": {"keys": ["Return"]}},
                {"keyboard": {"text": "cd"}},
            ]
        }
    ]

    # Reference key_conversion chain: super -> command -> env super;
    # page_down -> pagedown; escape -> esc -> env Escape.
    assert agent.map_key("super") == "super"
    assert agent.map_key("page_down") == "pagedown"
    assert agent.map_key("escape") == "Escape"
    assert agent.map_key("enter") == "Return"


def test_m3_call_user_keeps_episode_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reference converts CALL_USER to an empty action list (its runner
    asks the user and continues); it must NOT terminate the episode or send
    the env FAIL action."""
    agent = import_agent(monkeypatch, "minimax_m3")

    _, items = agent.parse_m3_response(tool_call("call_user"), 1920, 1080)
    assert items == ["CALL_USER"]

    computer = DummyComputer()
    terminal = agent.execute_items(computer, items)
    assert terminal is None
    assert computer.steps == []


def test_m3_wrap_for_history(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "minimax_m3")

    # Stop sequence ate the closing tag -> restored.
    wrapped = agent.wrap_for_history(
        'Action: x\n<tool_call>\n{"name": "computer", "arguments": {"action": "done"}}'
    )
    assert wrapped.endswith("</tool_call>")

    # Bare JSON re-wrapped so history renders the trained format.
    wrapped = agent.wrap_for_history(
        '  {"name": "computer", "arguments": {"action": "done"}}'
    )
    assert "<tool_call>" in wrapped and "</tool_call>" in wrapped

    assert agent.wrap_for_history("just words") == "just words"
    assert agent.wrap_for_history("") == ""


def test_m3_image_truncation_math(monkeypatch: pytest.MonkeyPatch) -> None:
    """README worked example: keep_min=10, chunk=20 -> images sawtooth
    between 11 and 30 (incl. the always-kept initial screenshot)."""
    agent = import_agent(monkeypatch, "minimax_m3")

    def counts(k: int) -> tuple[int, int]:
        msgs = agent.build_messages(
            "TASK", [f"s{i}" for i in range(k + 1)], [f"r{i}" for i in range(k)]
        )
        images = sum(
            1
            for m in msgs
            if isinstance(m["content"], list)
            for c in m["content"]
            if isinstance(c, dict) and c.get("type") == "image"
        )
        placeholders = sum(
            1
            for m in msgs
            if isinstance(m["content"], list)
            for c in m["content"]
            if isinstance(c, dict) and c.get("text") == "Tool result: Success"
        )
        return images, placeholders

    assert counts(0) == (1, 0)
    assert counts(9) == (10, 0)
    assert counts(29) == (30, 0)  # peak before the first drop
    assert counts(30) == (11, 20)  # first chunked drop
    assert counts(49) == (30, 20)
    assert counts(50) == (11, 40)


def test_m3_message_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "minimax_m3")

    msgs = agent.build_messages("TASK", ["s0", "s1", "s2"], ["r0", "r1"])
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    first = msgs[0]["content"]
    assert first[0]["type"] == "image"
    assert first[1] == {"type": "text", "text": "TASK"}
    assert msgs[1]["content"] == "r0"
    # Follow-up user turns are image-only (tool result implicit).
    assert [c["type"] for c in msgs[2]["content"]] == ["image"]


# --------------------------------------------------------------------------
# Agent-loop behavior
# --------------------------------------------------------------------------


class FakeComputer:
    """Env double for run()-loop tests: fixed observation, records actions."""

    def __init__(self, png: bytes) -> None:
        self.png = png
        self.steps: list[list[dict]] = []
        self.waits: list[float] = []
        self.done_called = False

    def observe(self) -> dict:
        return {"png": self.png, "meta": {}}

    def step(self, actions: list[dict]) -> dict:
        self.steps.append(actions)
        return {}

    def wait(self, seconds: float) -> None:
        self.waits.append(seconds)

    def done(self) -> None:
        self.done_called = True


def glm_reply(text: str, reasoning: str | None = None) -> dict:
    message: dict = {"content": text}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {"choices": [{"message": message}]}


def run_glm_with_replies(monkeypatch, agent, replies: list[dict]):
    """Drive agent.run() with canned LLM replies; returns (computer, calls)."""
    png = tiny_png()
    computer = FakeComputer(png)
    calls: list[list] = []

    def fake_request(messages):
        calls.append(messages)
        index = min(len(calls) - 1, len(replies) - 1)
        return replies[index]

    monkeypatch.setattr(agent, "glm_request", fake_request)
    monkeypatch.setattr(agent, "Computer", lambda *a, **k: computer)
    agent.run("http://env", "task")
    return computer, calls


def test_glm_reasoning_field_reconstructed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream call_llm stitches the reply's separate reasoning field back
    as <think>...</think>{content}; OpenRouter serves it as
    message.reasoning (some providers reasoning_content)."""
    agent = import_agent(monkeypatch, "glm5v_turbo")

    data = glm_reply(
        "<|begin_of_box|>WAIT()<|end_of_box|>\nMemory:\n[]", reasoning="THOUGHTS"
    )
    text = agent.response_text_of(data)
    assert text.startswith("<think>THOUGHTS</think>")
    parsed = agent.parse_response(text)
    assert parsed["action"] == "WAIT()"
    # The reasoning stays out of the post-think thought/history text.
    assert "THOUGHTS" not in parsed["thought"]

    # reasoning_content variant.
    data = {
        "choices": [
            {"message": {"content": "body\nMemory:\n[]", "reasoning_content": "R"}}
        ]
    }
    assert agent.response_text_of(data) == "<think>R</think>body\nMemory:\n[]"

    # No reasoning at all: upstream still wraps with an empty think block.
    assert agent.response_text_of(glm_reply("X")) == "<think></think>X"


def test_glm_scroll_step_unclamped_and_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixes: no 1..30 clamp on step (upstream uses the emitted value) and
    non-down directions fall through to the upstream 'else' (scroll up) --
    no shift+wheel horizontal extension."""
    agent = import_agent(monkeypatch, "glm5v_turbo")

    computer = DummyComputer()
    agent.convert_and_execute(
        computer,
        "scroll(start_box='[500,500]', direction='down', step=50, element_info='p')",
        1600,
        900,
    )
    assert computer.steps == [
        [{"mouse": {"move": [800, 450]}}, {"mouse": {"scroll": 50}}]
    ]

    for direction in ("up", "left", "right"):
        computer = DummyComputer()
        agent.convert_and_execute(
            computer,
            f"scroll(start_box='[500,500]', direction='{direction}', element_info='p')",
            1600,
            900,
        )
        # Upstream: only "down" negates; everything else is pyautogui
        # scroll-up (+5) == env -5. No shift press, no horizontal wheel.
        assert computer.steps == [
            [{"mouse": {"move": [800, 450]}}, {"mouse": {"scroll": -5}}]
        ], direction


def test_glm_key_repeat_detected_on_raw_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream _build_hotkey_command compares the RAW lowercased tokens;
    'enter+return' maps to the same env key but is NOT a repeat upstream."""
    agent = import_agent(monkeypatch, "glm5v_turbo")

    computer = DummyComputer()
    agent.convert_and_execute(computer, "key(keys='enter+return')", 1600, 900)
    assert computer.steps == [[{"keyboard": {"keys": ["Return", "Return"]}}]]

    computer = DummyComputer()
    agent.convert_and_execute(computer, "key(keys='return+return')", 1600, 900)
    assert computer.steps == [
        [{"keyboard": {"keys": ["Return"]}}, {"keyboard": {"keys": ["Return"]}}]
    ]


def test_glm_left_click_hold_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream appends EVAL_ACTION_SPACE_LEFT_CLICK_HOLD (and accepts the
    action) only for one hardcoded task id; here the gate is
    GLM_ENABLE_LEFT_CLICK_HOLD=1 or GLM_TASK_ID matching that id."""
    import hashlib

    # Default: disabled -- not advertised, not translated.
    agent = import_agent(monkeypatch, "glm5v_turbo")
    assert "left_click_hold" not in agent.action_space_text()
    assert "left_click_hold" not in agent.VALID_ACTIONS
    computer = DummyComputer()
    outcome = agent.convert_and_execute(
        computer, "left_click_hold(start_box='[500,500]')", 1600, 900
    )
    assert not outcome.executed

    # Enabled via env switch: verbatim upstream appendix + press/hold/release.
    agent = import_agent(
        monkeypatch, "glm5v_turbo", env={"GLM_ENABLE_LEFT_CLICK_HOLD": "1"}
    )
    # sha256 of EVAL_ACTION_SPACE_LEFT_CLICK_HOLD from glm_eval_agent.py
    # (main @ 2026-07-21; __HOLD_SECONDS__ -> 8, .strip() applied upstream).
    assert hashlib.sha256(
        agent.ACTION_SPACE_LEFT_CLICK_HOLD.encode()
    ).hexdigest() == (
        "ed30af6377cff1dcc9b78e70040cf7e77281edd5c380959b916896e330b33152"
    )
    # Combined space matches upstream _get_action_space rendering exactly.
    assert hashlib.sha256(agent.action_space_text().encode()).hexdigest() == (
        "8ccda93cd3984742fce688d0dc7a04355c5450b68ba7f082662aea77b090ba0b"
    )
    assert "left_click_hold" in agent.VALID_ACTIONS
    computer = DummyComputer()
    outcome = agent.convert_and_execute(
        computer, "left_click_hold(start_box='[500,500]', element_info='x')", 1600, 900
    )
    assert outcome.executed
    assert computer.steps == [
        [
            {"mouse": {"move": [800, 450]}},
            {"mouse": {"buttons": {"left_down": True}}},
            {"action": "wait", "time": 8.0},
            {"mouse": {"buttons": {"left_up": True}}},
        ]
    ]

    # Enabled via the upstream task id.
    agent = import_agent(
        monkeypatch,
        "glm5v_turbo",
        env={"GLM_TASK_ID": "47543840-672a-467d-80df-8f7c3b9788c9"},
    )
    assert agent.LEFT_CLICK_HOLD_ENABLED


def test_glm_no_resample_on_actionless_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream records a clean-parse-but-no-action reply as a no-op history
    step and does NOT resample -- one LLM call per step."""
    agent = import_agent(monkeypatch, "glm5v_turbo")

    replies = [
        glm_reply("I need to look around first.\nMemory:\n[]"),
        glm_reply("Still thinking.\nMemory:\n[]"),
        glm_reply("<|begin_of_box|>DONE()<|end_of_box|>\nMemory:\n[]"),
    ]
    computer, calls = run_glm_with_replies(monkeypatch, agent, replies)
    # Exactly one call per step (the old behavior resampled 3x per step).
    assert len(calls) == 3
    # No env FAIL was sent; DONE ends the episode without executing anything.
    assert computer.steps == []
    assert computer.done_called


def test_glm_consecutive_invalid_actions_emit_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untranslatable actions count toward the consecutive cap; at the
    threshold the env FAIL action fires (upstream returns the FAIL command
    when consecutive_parse_failures hits the max)."""
    agent = import_agent(monkeypatch, "glm5v_turbo")

    # Parses fine (balanced call) but has no start_box -> untranslatable.
    bad = glm_reply(
        "<|begin_of_box|>left_click(element_info='nowhere')<|end_of_box|>\nMemory:\n[]"
    )
    computer, calls = run_glm_with_replies(monkeypatch, agent, [bad])
    assert len(calls) == agent.MAX_CONSECUTIVE_PARSE_FAILURES
    assert computer.steps == [[{"action_type": "FAIL"}]]


def test_glm_max_steps_emits_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream converts the step into FAIL once max_trajectory_length is
    reached; the run loop mirrors that when the cap exhausts."""
    agent = import_agent(monkeypatch, "glm5v_turbo")
    monkeypatch.setattr(agent, "MAX_STEPS", 2)

    click = glm_reply(
        "<|begin_of_box|>left_click(start_box='[500,500]', "
        "element_info='x')<|end_of_box|>\nMemory:\n[]"
    )
    computer, calls = run_glm_with_replies(monkeypatch, agent, [click])
    assert len(calls) == 2
    assert computer.steps[-1] == [{"action_type": "FAIL"}]
    # The two click steps executed before the cap fired.
    assert len(computer.steps) == 3


def test_glm_max_steps_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = import_agent(monkeypatch, "glm5v_turbo", env={"CS_MAX_STEPS": "120"})
    assert agent.MAX_STEPS == 120
    agent = import_agent(
        monkeypatch, "glm5v_turbo", env={"CS_MAX_STEPS": "120", "GLM_MAX_STEPS": "77"}
    )
    assert agent.MAX_STEPS == 77


def test_m3_system_date_frozen_per_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream resolves default_system_date once per task (init/reset); the
    template freezes it at startup instead of recomputing per request."""
    agent = import_agent(monkeypatch, "minimax_m3")

    monkeypatch.setattr(agent, "SYSTEM_DATE", "Friday, May 29, 2026")
    first = agent.system_prompt()
    assert "The current date is Friday, May 29, 2026." in first
    # Stable across calls: no per-request datetime.today().
    assert agent.system_prompt() == first


def test_m3_image_format_default_png_and_jpeg_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch, "minimax_m3")
    assert agent.IMAGE_FORMAT == "PNG"
    assert agent.MEDIA_TYPE == "image/png"
    png = tiny_png()
    import base64 as b64mod

    assert agent.encode_screenshot(png) == b64mod.b64encode(png).decode("ascii")

    # JPEG opt-in (PIL is importable in the test env).
    agent = import_agent(monkeypatch, "minimax_m3", env={"M3_IMAGE_FORMAT": "JPEG"})
    assert agent.IMAGE_FORMAT == "JPEG"
    assert agent.MEDIA_TYPE == "image/jpeg"
    encoded = b64mod.b64decode(agent.encode_screenshot(png))
    assert encoded[:2] == b"\xff\xd8"  # JPEG SOI marker
    assert agent.image_block("abc")["source"]["media_type"] == "image/jpeg"


def test_m3_resample_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reference predict(): resample when a non-empty response parses to
    zero actions, up to max_llm_retries extra attempts."""
    agent = import_agent(monkeypatch, "minimax_m3")

    png = tiny_png()
    computer = FakeComputer(png)
    replies = [
        {"content": [{"type": "text", "text": "rambling, no tool call"}]},
        {
            "content": [
                {
                    "type": "text",
                    "text": 'Action: finish\n<tool_call>\n{"name": "computer", '
                    '"arguments": {"action": "done"}}\n</tool_call>',
                }
            ]
        },
    ]
    calls: list = []

    def fake_request(body):
        calls.append(body)
        return replies[min(len(calls) - 1, len(replies) - 1)]

    monkeypatch.setattr(agent, "anthropic_request", fake_request)
    monkeypatch.setattr(agent, "Computer", lambda *a, **k: computer)
    agent.run("http://env", "task")
    # One step: first sample parsed to zero actions, second sample was DONE.
    assert len(calls) == 2
    assert computer.steps == []
    assert computer.done_called


def test_m3_left_press_and_mouse_button_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = import_agent(monkeypatch, "minimax_m3")

    # left_press: press, hold 1s (upstream sleep(1)), release.
    _, items = agent.parse_m3_response(
        tool_call("left_press", coordinate=[500, 500]), 1920, 1080
    )
    assert items == [
        {
            "actions": [
                {"mouse": {"move": [960, 540]}},
                {"mouse": {"buttons": {"left_down": True}}},
            ]
        },
        {"wait": 1.0},
        {"actions": [{"mouse": {"buttons": {"left_up": True}}}]},
    ]

    # left_mouse_down at a coordinate moves then presses; left_mouse_up
    # without a coordinate releases in place (upstream mouseDown/mouseUp).
    _, items = agent.parse_m3_response(
        tool_call("left_mouse_down", coordinate=[400, 400]), 1920, 1080
    )
    assert items == [
        {
            "actions": [
                {"mouse": {"move": [768, 432]}},
                {"mouse": {"buttons": {"left_down": True}}},
            ]
        }
    ]
    _, items = agent.parse_m3_response(tool_call("left_mouse_up"), 1920, 1080)
    assert items == [{"actions": [{"mouse": {"buttons": {"left_up": True}}}]}]


# --------------------------------------------------------------------------
# Shared guardrails
# --------------------------------------------------------------------------


@pytest.mark.parametrize("template", ["glm5v_turbo", "minimax_m3"])
def test_template_has_no_task_specific_resolvers(template: str) -> None:
    source = (
        Path(__file__).resolve().parents[1] / "agents" / template / "agent.py"
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
