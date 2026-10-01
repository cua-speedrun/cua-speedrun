"""MiniMax M3 computer-use agent for OSWorld-style desktop tasks.

Contract: python agent.py <env_url> <task_description>

Port of the official OSWorld M3 agent (xlang-ai/OSWorld mm_agents/m3 at commit
b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf -- the agent behind MiniMax's official
70.06% OSWorld-Verified result) onto the cua-speedrun Computer client.

Faithfully ported from the reference:
  * Anthropic Messages API transport (MiniMax serves an Anthropic-compatible
    endpoint; the reference drives it through ANTHROPIC_BASE_URL). Default
    base URL is MiniMax's international platform,
    https://api.minimax.io/anthropic (use
    MINIMAX_BASE_URL=https://api.minimaxi.com/anthropic for the China
    platform). Auth header style is auto-detected by token prefix exactly as
    the reference does: "sk-" tokens go out as ``x-api-key``, anything else as
    ``Authorization: Bearer``.
  * The M3 system prompt (date + client password templated), verbatim.
  * ``<tool_call>{"name": "computer", "arguments": {...}}</tool_call>`` action
    parsing including the reference's BPE-artifact recovery ("_ " squeeze,
    "_lick" -> "_click", truncated-tail no-op) and bare-JSON fallback.
  * [0, 1000] normalized coordinates, rescaled to the observed screenshot size
    read from the PNG IHDR chunk (coordinate_type="relative").
  * Image truncation: keep the most recent ONLY_N_MOST_RECENT_IMAGES
    screenshots, dropping older ones in chunks of IMAGE_TRUNCATION_THRESHOLD
    and replacing them with the "Tool result: Success" placeholder; the
    initial screenshot is always kept.
  * Sampling exactly as the reference sets it: temperature 0.6, max_tokens
    8192, stop_sequences ["</tool_call>", "Perform the next action. Perform"].
    No top_p (the reference default is None).
  * The predict-level retry ladder: resample when the LLM call raises or when
    a non-empty response parses to zero actions, up to MINIMAX_MAX_LLM_RETRIES
    extra attempts (default 2 -> 3 total), plus HTTP-level retries mirroring
    the Anthropic SDK's max_retries=4 / timeout=180 configuration.
  * ``wrap_for_history``: assistant turns are stored with the <tool_call>
    wrapper restored so the chat template sees the trained format.
  * "[INFEASIBLE]" token and terminate/fail actions -> the env FAIL action.
  * CALL_USER -> no-op that keeps the episode alive (the reference converts
    it to an empty action list; this harness has no ask-user channel).

Environment adaptations:
  * The agent sandbox ships requests+pyyaml only (no PIL), so the default is
    raw observed PNG -- the reference's M3_IMAGE_FORMAT=PNG passthrough path
    (no resize either way). Setting M3_IMAGE_FORMAT=JPEG opts into the
    reference's JPEG-90 transcoding when PIL is importable.
  * pyautogui code strings are replaced by the env's gym-anything action
    dicts; the per-action translation preserves the reference semantics,
    including the scroll sign flip (pyautogui positive = up, env positive =
    down) and no-op sleeps for invalid/unsupported actions.
  * Coordinates use the reference's exact int(x * width / 1000) then a
    screen clamp; in-range values are grid-verified identical, the clamp
    only affects out-of-range values (which in-guest pyautogui clamps too).
  * left_click_drag without start_coordinate drags from a cursor position
    tracked across steps (the reference drags from the live OS cursor,
    which the env cannot query); unknown cursor -> no-op.
  * Clickless clicks run at the current cursor: left/double/triple as
    button down/up pairs (script-sequential, so double/triple-click timing
    is best-effort), right via the env's right-button events; the env has
    no middle-button channel, so clickless middle_click is a no-op.

Model id: MINIMAX_MODEL (default "MiniMax-M3", the MiniMax platform id; the
reference leaves the id to ANTHROPIC_MODEL). API key: MINIMAX_API_KEY.
"""

from __future__ import annotations

import base64
import json
import os
import struct
import sys
import time
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

import requests

from cua_speedrun.client import Computer

MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M3")
BASE_URL = os.environ.get(
    "MINIMAX_BASE_URL", "https://api.minimax.io/anthropic"
).rstrip("/")
API_URL = f"{BASE_URL}/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
# Step cap. MINIMAX_MAX_STEPS is the user-settable knob; CS_MAX_STEPS is a
# harness-internal fallback (kept for parity with the other templates but not
# user-forwardable, so it stays a no-op in normal submissions).
MAX_STEPS = int(
    os.environ.get("MINIMAX_MAX_STEPS") or os.environ.get("CS_MAX_STEPS") or "100"
)
# The reference constructs the Anthropic SDK client with max_retries=4 and
# timeout=180.0; mirror that at the raw-HTTP level.
REQUEST_TIMEOUT = float(os.environ.get("MINIMAX_HTTP_TIMEOUT", "180"))
HTTP_MAX_RETRIES = int(os.environ.get("MINIMAX_HTTP_MAX_RETRIES", "4"))
# predict()-level resample ladder (reference: M3_MAX_LLM_RETRIES, default 2).
MAX_LLM_RETRIES = int(os.environ.get("MINIMAX_MAX_LLM_RETRIES", "2"))
MAX_TOKENS = int(os.environ.get("MINIMAX_MAX_TOKENS", "8192"))
TEMPERATURE = float(os.environ.get("MINIMAX_TEMPERATURE", "0.6"))
STOP_SEQUENCES = ["</tool_call>", "Perform the next action. Perform"]
# Image truncation knobs (reference: only_n_most_recent_images /
# image_truncation_threshold).
ONLY_N_MOST_RECENT_IMAGES = int(os.environ.get("MINIMAX_ONLY_N_MOST_RECENT_IMAGES", "10"))
IMAGE_TRUNCATION_THRESHOLD = int(os.environ.get("MINIMAX_IMAGE_TRUNCATION_THRESHOLD", "20"))
CLIENT_PASSWORD = os.environ.get("MINIMAX_CLIENT_PASSWORD", "password")
# Env-gateway HTTP timeout (the client default of 120s is tight for heavy envs).
ENV_HTTP_TIMEOUT = float(os.environ.get("MINIMAX_ENV_HTTP_TIMEOUT", "600"))
GRID = 1000.0


def is_openrouter_endpoint(base_url: str) -> bool:
    """True when ``base_url`` points at OpenRouter.

    ``provider`` is an OpenRouter routing extension. The native MiniMax
    Anthropic endpoint does not document it, so it is only attached when the
    request actually goes to OpenRouter; everywhere else the body stays exactly
    as it was before the provider-pin feature existed. Matched on host rather
    than on the literal string so a path or scheme variation still routes.
    """
    host = (urlsplit(base_url).hostname or "").lower()
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def _provider_preference() -> dict[str, Any] | None:
    """OpenRouter provider-routing preference from env vars.

    ``MINIMAX_PROVIDER_ORDER`` is a comma-separated list of OpenRouter provider
    slugs tried in order (e.g. "minimax" to pin the first-party endpoint, or
    "minimax,novita"). By default fallbacks are DISABLED so the request pins
    hard to the listed providers; set ``MINIMAX_PROVIDER_ALLOW_FALLBACKS`` truthy
    to allow OpenRouter to fall back to others. Only applied when the request
    is routed through OpenRouter (MINIMAX_BASE_URL=https://openrouter.ai/api);
    see ``is_openrouter_endpoint`` — on the native MiniMax endpoint the field
    is off-spec and is never sent, so the body there is byte-identical to what
    it was before this feature existed. Returns None when unset so the body is
    untouched in every case. minimax/minimax-m3 is served on
    OpenRouter by 9 providers (Novita, Venice, GMICloud, Minimax, AtlasCloud,
    Together, Parasail, DeepInfra, Morph), so pinning removes serving variance.
    """
    raw = os.environ.get("MINIMAX_PROVIDER_ORDER", "").strip()
    if not raw:
        return None
    order = [p.strip() for p in raw.split(",") if p.strip()]
    if not order:
        return None
    allow = os.environ.get("MINIMAX_PROVIDER_ALLOW_FALLBACKS", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    return {"order": order, "allow_fallbacks": allow}


PROVIDER_PREFERENCE = _provider_preference()
# The reference resolves the system date ONCE per task (default_system_date
# at __init__/reset); this process runs exactly one episode, so freeze it at
# startup instead of recomputing per request.
SYSTEM_DATE = datetime.today().strftime("%A, %B %d, %Y")

# On-wire screenshot encoding. The agent sandbox has no PIL, so the default
# is raw-PNG passthrough (the reference's M3_IMAGE_FORMAT=PNG path). Setting
# M3_IMAGE_FORMAT=JPEG opts into the reference's default JPEG-90 transcoding
# when PIL happens to be importable; otherwise it falls back to PNG.
IMAGE_FORMAT = (os.environ.get("M3_IMAGE_FORMAT") or "PNG").strip().upper()
IMAGE_QUALITY = int(os.environ.get("M3_IMAGE_QUALITY", "90"))
if IMAGE_FORMAT == "JPEG":
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("M3_IMAGE_FORMAT=JPEG but PIL is unavailable; using PNG", file=sys.stderr)
        IMAGE_FORMAT = "PNG"
MEDIA_TYPE = "image/jpeg" if IMAGE_FORMAT == "JPEG" else "image/png"

TOOL_RESULT_PLACEHOLDER = "Tool result: Success"

# Verbatim from mm_agents/m3/prompts.py (ubuntu platform; the macOS keyboard
# suffix is not applicable to this benchmark's guests).
M3_SYSTEM_PROMPT_TEMPLATE = """<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using x86_64 architecture with internet access.
* To open browser, please just click on the Chrome icon.  Note, Chrome is what is installed on your system.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* DO NOT ask users for clarification during task execution. DO NOT stop to request more information from users. Always take action using available tools.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* TASK FEASIBILITY — this is a frequent failure point, read carefully. Some tasks are intentionally impossible. You may declare a task infeasible at any point: immediately after the first screenshot, or later after attempting actions and hitting a hard barrier. A task is infeasible when it cannot be completed due to:
  - Missing required applications or dependencies that cannot be installed
  - Insufficient permissions or system limitations
  - Contradictory, fictional, or impossible requirements
  - The task requires a specific application, but you have verified that application does not provide the required feature or capability
  - Any other fundamental barrier that makes completion impossible
  When you conclude a task is infeasible, you MUST output the literal token "[INFEASIBLE]" (with the square brackets) in that same turn. This exact token is the ONLY signal the system recognizes for an impossible task.
  CRITICAL — when a task is infeasible, do NOT do any of the following instead:
  - Do NOT emit the `done` action. `done` means "I have successfully completed the task" and will be graded as WRONG for an impossible task.
  - Do NOT only explain in prose that it is "not possible" / "does not exist" / "cannot be done". Prose refusals are NOT detected — only the literal "[INFEASIBLE]" token counts.
  - Do NOT propose workarounds or alternatives, and do NOT ask the user what they prefer. Decide yourself, output "[INFEASIBLE]", and stop.
  Only declare a task infeasible when you are genuinely confident it is impossible. Do NOT give up on a task that is merely difficult, slow, or unfamiliar — try reasonable approaches first.
* The current date is {date_str}.
* Home directory of this Ubuntu system is '/home/user'.
* If you need a password for sudo, the password of the computer is '{client_password}'.
* All `coordinate` values in `computer` tool calls are normalized integer values in [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner of the screen, regardless of the underlying screen resolution.
</SYSTEM_CAPABILITY>

<TOOLS>
You have access to the `computer` tool for interacting with the desktop GUI. Output one tool call per turn.

For each tool call, return a json object with name and arguments inside <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "computer", "arguments": {{"action": "<action>", ...}}}}
</tool_call>

Supported `action` values include:
* `key`        — args: {{"text": "<key combo, e.g. ctrl+s>"}}
* `type`       — args: {{"text": "<text to type>"}}
* `mouse_move` — args: {{"coordinate": [x, y]}}
* `left_click` / `right_click` / `middle_click` / `double_click` / `triple_click`
                 — args: {{"coordinate": [x, y]}}, optional {{"text": "<modifier keys>"}}
* `left_click_drag` — args: {{"coordinate": [x, y]}}, optional {{"start_coordinate": [x, y]}}
* `scroll`     — args: {{"coordinate": [x, y], "scroll_direction": "up|down|left|right", "scroll_amount": <int>}}
* `wait`       — args: {{"duration": <seconds>}}
* `screenshot` — args: {{}}
* `hold_key`   — args: {{"text": "<keys>", "duration": <seconds>}}
* `left_mouse_down` / `left_mouse_up` — args: {{"coordinate": [x, y]}}
* `done`       — args: {{}}: declare the task has been completed successfully. Output this as the final tool call after you are confident the task is fully done.

Rules:
- Output exactly one short imperative `Action: <text>` line followed by exactly one <tool_call>...</tool_call> block.
- Coordinates are normalized integer values in [0, 1000], where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner of the screen.
- When the task has been completed and there is nothing more to do, your FINAL turn must end with a <tool_call> using action `done` to signal completion.
- For impossible/infeasible tasks, your response MUST contain the literal `[INFEASIBLE]` token — not the `done` action, and not a prose-only explanation.
</TOOLS>"""

# Reference: mm_agents/m3/parser.py _VALID_ACTIONS.
VALID_ACTIONS = frozenset({
    "click", "left click", "right click",
    "left_click", "right_click", "double_click", "middle_click",
    "left_press", "triple_click",
    "left_mouse_down", "left_mouse_up",
    "mouse_move", "left_click_drag",
    "hold_key", "key", "type",
    "scroll",
    "wait", "screenshot", "fail", "done", "call_user",
    "terminate", "cursor_position",
    "zoom", "zoom_in",
})

# The reference emits pyautogui key names; this maps them (after the
# reference's own key_conversion) onto the vocabulary the gym-anything
# keyboard backend accepts.
KEY_CONVERSION = {
    # Verbatim from the reference parser's `key` branch.
    "page_down": "pagedown",
    "page_up": "pageup",
    "super_l": "win",
    "super": "command",
    "escape": "esc",
}
ENV_KEY_MAP = {
    "enter": "Return",
    "return": "Return",
    "kp_enter": "Return",
    "esc": "Escape",
    "escape": "Escape",
    "tab": "Tab",
    "backspace": "BackSpace",
    "bksp": "BackSpace",
    "delete": "Delete",
    "del": "Delete",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "home": "Home",
    "end": "End",
    "space": "space",
    "pageup": "pageup",
    "pgup": "pageup",
    "prior": "pageup",
    "pagedown": "pagedown",
    "pgdn": "pagedown",
    "next": "pagedown",
    "insert": "insert",
    "ins": "insert",
    "win": "super",
    "cmd": "super",
    "command": "super",
    "meta": "super",
    "super": "super",
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "option": "alt",
    "shift": "shift",
    "capslock": "capslock",
}


def api_key() -> str:
    key = os.environ.get("MINIMAX_API_KEY")
    if not key:
        raise RuntimeError("set MINIMAX_API_KEY for the MiniMax API")
    return key


def anthropic_headers() -> dict[str, str]:
    """Auth-style auto-detection, exactly as the reference: tokens starting
    with "sk-" go via ``x-api-key``; anything else (e.g. a JWT) goes via
    ``Authorization: Bearer``."""
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": ANTHROPIC_VERSION,
    }
    key = api_key()
    if key.startswith("sk-"):
        headers["x-api-key"] = key
    else:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def system_prompt() -> str:
    return M3_SYSTEM_PROMPT_TEMPLATE.format(
        date_str=SYSTEM_DATE, client_password=CLIENT_PASSWORD
    )


def encode_screenshot(png: bytes) -> str:
    """Base64 body for the on-wire image (media type in MEDIA_TYPE)."""
    if IMAGE_FORMAT == "JPEG":
        import io

        from PIL import Image

        image = Image.open(io.BytesIO(png))
        if image.mode != "RGB":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=IMAGE_QUALITY, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    return base64.b64encode(png).decode("ascii")


def image_size(png: bytes) -> tuple[int, int]:
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
        raise ValueError("observation is not a valid PNG with an IHDR chunk")
    return struct.unpack(">II", png[16:24])


def image_block(b64_png: str) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": MEDIA_TYPE, "data": b64_png},
    }


def user_message_with_image(b64_png: str, text: str | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = [image_block(b64_png)]
    if text:
        content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def tool_result_placeholder_message() -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "text", "text": TOOL_RESULT_PLACEHOLDER}],
    }


def build_messages(
    instruction: str, screenshots: list[str], responses: list[str]
) -> list[dict[str, Any]]:
    """Reference `_build_messages` layout:

      1. user: screenshot 0 + instruction text
      2. for i in 0..k-1: assistant response[i]; user screenshot[i+1] or the
         "Tool result: Success" placeholder when dropped by truncation
    Truncation: keep the most recent K screenshots, drop older ones in chunks
    of T; the initial screenshot is always kept.
    """
    K = ONLY_N_MOST_RECENT_IMAGES
    T = IMAGE_TRUNCATION_THRESHOLD
    k = len(responses)
    remove = max(0, k - K)
    remove -= remove % T
    if remove > 0:
        print(
            f"[ImageTruncation] k={k} K={K} T={T} -> drop oldest {remove} of {k} "
            f"tool_result images, keep {k - remove} + initial = {k - remove + 1}",
            file=sys.stderr,
        )

    messages: list[dict[str, Any]] = [
        user_message_with_image(screenshots[0], instruction)
    ]
    for i in range(k):
        messages.append({"role": "assistant", "content": responses[i]})
        tool_result_idx = i + 1
        if tool_result_idx <= remove:
            messages.append(tool_result_placeholder_message())
        else:
            messages.append(user_message_with_image(screenshots[tool_result_idx]))
    return messages


def build_request_body(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Sampling exactly as the reference sets it: temperature and
    stop_sequences, no top_p (its reference default is None)."""
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "system": system_prompt(),
        "temperature": TEMPERATURE,
        "stop_sequences": list(STOP_SEQUENCES),
    }
    if PROVIDER_PREFERENCE is not None and is_openrouter_endpoint(BASE_URL):
        body["provider"] = PROVIDER_PREFERENCE
    return body


def should_retry_status(status_code: int) -> bool:
    return status_code in {408, 409, 429} or status_code >= 500


def anthropic_request(body: dict[str, Any]) -> dict[str, Any]:
    """One Messages-API call with SDK-equivalent retries (max_retries=4)."""
    last_error = ""
    for attempt in range(1, HTTP_MAX_RETRIES + 2):
        try:
            resp = requests.post(
                API_URL,
                headers=anthropic_headers(),
                json=body,
                timeout=REQUEST_TIMEOUT,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = f"connection error: {exc}"
            print(f"minimax request attempt {attempt}: {last_error}", file=sys.stderr)
            time.sleep(min(2.0 * attempt, 30.0))
            continue
        if resp.status_code < 400:
            return resp.json()
        last_error = f"HTTP {resp.status_code} {resp.text[:500]}"
        if not should_retry_status(resp.status_code):
            break
        print(f"minimax request attempt {attempt}: {last_error}", file=sys.stderr)
        time.sleep(min(2.0 * attempt, 30.0))
    raise RuntimeError(
        f"MiniMax request failed after {HTTP_MAX_RETRIES + 1} attempts: {last_error}"
    )


def response_text_of(data: dict[str, Any]) -> str:
    """Concatenate text blocks; a thinking block is wrapped as
    ``<mm:think>...</mm:think>`` and prepended (reference `_call_llm`)."""
    text_parts: list[str] = []
    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(str(block.get("text", "")))
        elif btype == "thinking":
            thinking = block.get("thinking", "")
            if thinking and "<mm:think>" not in "".join(text_parts):
                text_parts.insert(0, f"<mm:think>{thinking}</mm:think>\n")
    return "".join(text_parts)


def map_key(key: str) -> str:
    lowered = str(key).strip().lower()
    lowered = KEY_CONVERSION.get(lowered, lowered)
    return ENV_KEY_MAP.get(lowered, lowered)


def split_modifiers(text: str) -> list[str]:
    return [map_key(k) for k in text.split("+") if k.strip()]


def hold_duration(raw: Any) -> float:
    """Seconds to hold a key for ``hold_key``, clamped to a sane range.

    The tool schema advertises ``duration``; a missing or unparseable value
    falls back to one second, and the upper bound keeps a hallucinated number
    from eating the task's wall clock.
    """
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return 1.0
    return max(0.1, min(10.0, seconds))


def type_segments(text: str) -> list[dict[str, Any]]:
    """pyautogui types '\\n' as an Enter press; split lines accordingly."""
    lines = text.split("\n")
    actions: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if line:
            actions.append({"keyboard": {"text": line}})
        if index < len(lines) - 1:
            actions.append({"keyboard": {"keys": ["Return"]}})
    return actions


def tool_action_to_env(
    action: str,
    args: dict[str, Any],
    scaled_xy,
    cursor: list | None = None,
) -> list[Any]:
    """Translate one computer-use tool action into env segments.

    Returns a list whose elements are either the special tokens
    "FAIL"/"DONE"/"CALL_USER" or segment dicts: {"actions": [env dicts]} /
    {"wait": seconds}. Raises ValueError on invalid arguments, mirroring the
    reference translator (the caller turns that into a no-op sleep).
    """
    segments: list[Any] = []
    actions: list[dict[str, Any]] = []

    def flush() -> None:
        if actions:
            segments.append({"actions": list(actions)})
            actions.clear()

    def set_cursor(point: list[int]) -> None:
        if cursor is not None:
            cursor[:] = list(point)

    action_conversion = {"left click": "click", "right click": "right_click"}
    action = action_conversion.get(action, action)
    if action == "click":
        action = "left_click"

    text = args.get("text")
    coordinate = args.get("coordinate")
    start_coordinate = args.get("start_coordinate")
    scroll_direction = args.get("scroll_direction")
    scroll_amount = args.get("scroll_amount")

    if coordinate is not None:
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
            raise ValueError(f"{coordinate} must be a tuple of length 2")
        coordinate = list(scaled_xy(coordinate[0], coordinate[1]))
    if start_coordinate is not None:
        if not isinstance(start_coordinate, (list, tuple)) or len(start_coordinate) != 2:
            raise ValueError(f"{start_coordinate} must be a tuple of length 2")
        start_coordinate = list(scaled_xy(start_coordinate[0], start_coordinate[1]))

    if action in ("left_mouse_down", "left_mouse_up"):
        if text:
            actions.append({"keyboard": {"keys_down": split_modifiers(text)}})
        if coordinate is not None:
            actions.append({"mouse": {"move": coordinate}})
            set_cursor(coordinate)
        state = "left_down" if action == "left_mouse_down" else "left_up"
        actions.append({"mouse": {"buttons": {state: True}}})
        if text:
            actions.append({"keyboard": {"keys_up": list(reversed(split_modifiers(text)))}})
    elif action == "hold_key":
        if not isinstance(text, str):
            raise ValueError(f"{text} must be a string")
        # Press, hold for the requested duration, then release. Without the
        # release the modifier stays latched for the rest of the episode and
        # corrupts every later keystroke.
        held = split_modifiers(text)
        actions.append({"keyboard": {"keys_down": held}})
        flush()
        segments.append({"wait": hold_duration(args.get("duration"))})
        actions.append({"keyboard": {"keys_up": list(reversed(held))}})
    elif action in ("mouse_move", "left_click_drag"):
        if coordinate is None:
            raise ValueError(f"coordinate is required for {action}")
        if text is not None:
            raise ValueError(f"text is not accepted for {action}")
        if action == "mouse_move":
            actions.append({"mouse": {"move": coordinate}})
        else:
            start = start_coordinate
            if start is None and cursor is not None and len(cursor) == 2:
                start = list(cursor)
            if start is None:
                raise ValueError(
                    "left_click_drag needs start_coordinate or a known cursor position"
                )
            actions.append({"mouse": {"left_click_drag": [start, coordinate]}})
        set_cursor(coordinate)
    elif action in ("key", "type"):
        if text is None:
            raise ValueError(f"text is required for {action}")
        if coordinate is not None:
            raise ValueError(f"coordinate is not accepted for {action}")
        if not isinstance(text, str):
            raise ValueError(f"{text} must be a string")
        if action == "key":
            actions.append({"keyboard": {"keys": split_modifiers(text)}})
        else:
            actions.extend(type_segments(text))
    elif action == "scroll":
        if text is not None:
            actions.append({"keyboard": {"keys_down": split_modifiers(text)}})
        if not isinstance(scroll_amount, int) or isinstance(scroll_amount, bool):
            raise ValueError(f"scroll_amount must be an int, got {scroll_amount!r}")
        if coordinate is not None:
            actions.append({"mouse": {"move": coordinate}})
            set_cursor(coordinate)
        # pyautogui sign convention: positive scrolls up / right. The env's
        # convention is positive = down, so the vertical sign flips; horizontal
        # scrolling is shift+wheel with positive = right.
        if scroll_direction in ("up", "down"):
            clicks = -scroll_amount if scroll_direction == "up" else scroll_amount
            actions.append({"mouse": {"scroll": clicks}})
        elif scroll_direction in ("left", "right"):
            clicks = scroll_amount if scroll_direction == "right" else -scroll_amount
            actions.append({"keyboard": {"keys_down": ["shift"]}})
            actions.append({"mouse": {"scroll": clicks}})
            actions.append({"keyboard": {"keys_up": ["shift"]}})
        else:
            raise ValueError(f"invalid scroll_direction: {scroll_direction!r}")
        if text is not None:
            actions.append({"keyboard": {"keys_up": list(reversed(split_modifiers(text)))}})
    elif action in ("left_click", "right_click", "double_click", "middle_click",
                    "left_press", "triple_click"):
        mods = split_modifiers(text) if text else []
        if mods:
            actions.append({"keyboard": {"keys_down": mods}})
        if action == "left_press":
            if coordinate is not None:
                actions.append({"mouse": {"move": coordinate}})
                set_cursor(coordinate)
            actions.append({"mouse": {"buttons": {"left_down": True}}})
            flush()
            segments.append({"wait": 1.0})
            actions.append({"mouse": {"buttons": {"left_up": True}}})
        elif coordinate is not None:
            actions.append({"mouse": {action: coordinate}})
            set_cursor(coordinate)
        else:
            # No coordinate: click at the current position (the reference
            # emits pyautogui.click()/rightClick() there). The env has no
            # position-free click, so emit button events at the cursor; the
            # env's buttons channel supports left and right only, so
            # middle_click without a coordinate stays a no-op.
            if action == "right_click":
                actions.append({"mouse": {"buttons": {"right_down": True}}})
                actions.append({"mouse": {"buttons": {"right_up": True}}})
            else:
                presses = {"left_click": 1, "double_click": 2, "triple_click": 3}.get(action)
                if presses is None:
                    raise ValueError(f"{action} without coordinate is not supported")
                for _ in range(presses):
                    actions.append({"mouse": {"buttons": {"left_down": True}}})
                    actions.append({"mouse": {"buttons": {"left_up": True}}})
        if mods:
            actions.append({"keyboard": {"keys_up": list(reversed(mods))}})
    elif action == "wait":
        flush()
        segments.append({"wait": 0.5})
    elif action == "fail":
        flush()
        segments.append("FAIL")
    elif action == "done":
        flush()
        segments.append("DONE")
    elif action == "call_user":
        flush()
        segments.append("CALL_USER")
    elif action == "terminate":
        status = str(args.get("status", "success")).lower()
        flush()
        segments.append("FAIL" if status in ("failure", "fail") else "DONE")
    elif action in ("screenshot", "cursor_position", "zoom", "zoom_in"):
        flush()
        segments.append({"wait": 0.1})
    else:
        raise ValueError(f"Invalid action: {action}")

    flush()
    return segments


def parse_m3_response(
    response: str,
    width: int,
    height: int,
    cursor: list | None = None,
) -> tuple[str, list[Any]]:
    """Parse one M3 response into ``(low_level_instruction, items)``.

    ``items`` elements are special-token strings ("FAIL"/"DONE"/"CALL_USER")
    or segment dicts ({"actions": [...]} / {"wait": seconds}), in emission
    order. Ported from the reference parser: [INFEASIBLE] handling, Action:
    line pickup, <tool_call> block walking with stop-sequence recovery, bare
    JSON fallback, BPE-artifact fixes, invalid-action no-ops.
    """
    if not response or not response.strip():
        return "", []
    if "[INFEASIBLE]" in response:
        return "[INFEASIBLE]", ["FAIL"]

    def scaled_xy(x: Any, y: Any) -> tuple[int, int]:
        # coordinate_type="relative": [0, 1000] normalized -> pixels, clamped
        # into the screen (the reference's int(x * width / 1000), kept inside
        # the framebuffer so x=1000 cannot land one pixel out of range).
        px = int(float(x) * width / GRID)
        py = int(float(y) * height / GRID)
        return min(max(px, 0), width - 1), min(max(py, 0), height - 1)

    low_level_instruction = ""
    items: list[Any] = []

    def emit(json_str: str) -> None:
        try:
            tool_call = json.loads(json_str)
        except json.JSONDecodeError as exc:
            print(f"failed to parse tool_call JSON: {exc}", file=sys.stderr)
            return
        if not isinstance(tool_call, dict) or tool_call.get("name") != "computer":
            print(f"expected name='computer', skipping: {json_str[:120]}", file=sys.stderr)
            return
        args = tool_call.get("arguments") or tool_call.get("input") or {}
        if not isinstance(args, dict):
            return
        action = args.get("action")
        if not action:
            print(f"tool_call missing 'action': {json_str[:120]}", file=sys.stderr)
            return

        # BPE tokenizer artifact recovery (reference quirks).
        if isinstance(action, str) and "_ " in action:
            action = action.replace("_ ", "_")
            args = dict(args, action=action)
        args = {
            (k.replace("_ ", "_") if isinstance(k, str) else k): v
            for k, v in args.items()
        }
        if isinstance(action, str) and "_lick" in action and action not in VALID_ACTIONS:
            candidate = action.replace("_lick", "_click")
            if candidate in VALID_ACTIONS:
                action = candidate
                args = dict(args, action=candidate)
        if (isinstance(action, str)
                and action not in VALID_ACTIONS
                and (action.endswith("_") or len(action) < 3)):
            print(f"truncated/short action {action!r} -> no-op", file=sys.stderr)
            items.append({"wait": 0.1})
            return

        try:
            segments = tool_action_to_env(action, args, scaled_xy, cursor)
        except ValueError as exc:
            print(f"invalid action {action!r}: {exc} -> no-op", file=sys.stderr)
            items.append({"wait": 0.1})
            return
        items.extend(segments)

    inside = False
    buffer: list[str] = []
    for line in response.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("action:"):
            if not low_level_instruction:
                low_level_instruction = stripped.split(":", 1)[-1].strip()
            continue
        if stripped.startswith("<tool_call>"):
            inside = True
            continue
        if stripped.startswith("</tool_call>"):
            if buffer:
                emit("\n".join(buffer))
                buffer = []
            inside = False
            continue
        if inside:
            buffer.append(stripped)
            continue
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                emit(stripped)

    # Stop-sequence recovery: the upstream eats "</tool_call>".
    if inside and buffer:
        emit("\n".join(buffer))

    return low_level_instruction, items


def wrap_for_history(raw_text: str) -> str:
    """Verbatim port of the reference: restore the <tool_call> wrapper so past
    assistant turns render in the trained format."""
    out = raw_text or ""
    if "<tool_call>" in out:
        if "</tool_call>" not in out:
            out = out.rstrip() + "\n</tool_call>"
        return out
    new_lines = []
    for line in out.split("\n"):
        stripped = line.strip()
        if (stripped.startswith("{")
                and stripped.endswith("}")
                and '"name"' in stripped
                and '"arguments"' in stripped):
            indent = line[: len(line) - len(line.lstrip())]
            new_lines.append(f"{indent}<tool_call>\n{stripped}\n</tool_call>")
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def execute_items(computer: Computer, items: list[Any]) -> str | None:
    """Run parsed items against the env. Returns the terminal token
    ("DONE"/"FAIL") if one fired, else None. A terminate-failure (and
    [INFEASIBLE]) emits the env FAIL action. CALL_USER is a NO-OP that keeps
    the episode alive -- the reference converts it to an empty action list so
    its runner can ask the user; this harness has no user, so the model just
    gets the next screenshot (its prompt already forbids asking users)."""
    for item in items:
        if item == "DONE":
            return "DONE"
        if item == "FAIL":
            computer.step([{"action_type": "FAIL"}])
            return "FAIL"
        if item == "CALL_USER":
            print("CALL_USER emitted; no user available -> no-op", file=sys.stderr)
            continue
        if isinstance(item, dict) and "wait" in item:
            computer.wait(float(item["wait"]))
        elif isinstance(item, dict) and item.get("actions"):
            computer.step(item["actions"])
    return None


def run(env_url: str, task: str) -> None:
    computer = Computer(env_url, timeout_sec=ENV_HTTP_TIMEOUT)
    screenshots: list[str] = []
    responses: list[str] = []
    cursor: list = []
    try:
        obs = computer.observe()
        for step in range(MAX_STEPS):
            width, height = image_size(obs["png"])
            screenshots.append(encode_screenshot(obs["png"]))
            body = build_request_body(build_messages(task, screenshots, responses))

            # predict()-level retry ladder: resample on transport failure or
            # when a non-empty response parses to zero actions.
            response_text = ""
            items: list[Any] = []
            for attempt in range(MAX_LLM_RETRIES + 1):
                try:
                    data = anthropic_request(body)
                    response_text = response_text_of(data)
                except Exception as exc:
                    print(
                        f"step {step}: LLM call failed (attempt {attempt + 1}): {exc}",
                        file=sys.stderr,
                    )
                    response_text = ""
                    if attempt < MAX_LLM_RETRIES:
                        continue
                    break
                print(
                    f"step {step}: M3 output (attempt {attempt + 1}): "
                    f"{response_text[:400]!r}",
                    file=sys.stderr,
                )
                _instruction, items = parse_m3_response(
                    response_text, width, height, cursor
                )
                if not items and response_text.strip():
                    print(
                        f"step {step}: parsed 0 actions from a non-empty response"
                        + ("; retrying" if attempt < MAX_LLM_RETRIES else "; no-op"),
                        file=sys.stderr,
                    )
                    if attempt < MAX_LLM_RETRIES:
                        continue
                break

            if not response_text.strip():
                # Never append an empty assistant turn: the Messages API
                # rejects one with 400, 400 is not retryable, and the next
                # step would append another — spinning the episode to the
                # step cap on a timed clock. Stop instead.
                print(
                    f"step {step}: no usable response after "
                    f"{MAX_LLM_RETRIES + 1} attempts; ending episode",
                    file=sys.stderr,
                )
                break

            responses.append(wrap_for_history(response_text))

            terminal = execute_items(computer, items)
            if terminal is not None:
                print(f"step {step}: terminal token {terminal}", file=sys.stderr)
                break

            obs = computer.observe()
    except Exception as exc:
        print(f"agent failed: {exc!r}", file=sys.stderr, flush=True)
    finally:
        try:
            computer.done()
        except Exception as exc:
            print(f"computer.done failed: {exc!r}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
