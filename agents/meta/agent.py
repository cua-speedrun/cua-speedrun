"""Meta super_nova_ext GUI agent for OSWorld-style desktop tasks.

Contract: python agent.py <env_url> <task_description>

Port of the Meta "gui" mode agent (screenshot observation, one `computer`
function tool with 0-1000 normalized coordinates) onto the cua-speedrun
Computer client. The model is served by Meta's OpenAI-compatible relay; the
default mode drives it through the stateless Responses API so encrypted
reasoning items carry the hidden chain-of-thought across turns, with Chat
Completions ("gui") and pyautogui-code ("gui_pyautogui") modes selectable via
META_MODE. Per vendor requirements the request carries no sampling parameters
(API defaults only) and exactly one tool call per assistant turn is honored.

The hybrid mode of the source agent (Playwright-MCP over CDP plus a bash
sandbox) is intentionally not ported: the agent sandbox only sees the
observe/step/done gateway, with no CDP path into the env's browser.
"""

from __future__ import annotations

import ast
import base64
import json
import math
import os
import re
import struct
import sys
import time
import unicodedata
import uuid
from typing import Any

import requests

from cua_speedrun.client import Computer

MODEL = os.environ.get("META_MODEL", "super_nova_ext")
# "responses" (the default): the gui-mode `computer` tool contract over the
# stateless Responses API — encrypted reasoning items are replayed each turn
# so the hidden chain-of-thought survives across steps.
# "gui": the same contract over Chat Completions.
# "gui_pyautogui": metacua's OSWorld backend — the model writes raw pyautogui
# code, which is AST-translated into the env's structured actions (agents can
# only reach the observe/step/done gateway, never an exec path).
MODE = os.environ.get("META_MODE", "responses")
BASE_URL = os.environ.get("META_BASE_URL", "https://api.ai.meta.com/v1").rstrip("/")
API_URL = f"{BASE_URL}/chat/completions"
MAX_STEPS = int(os.environ.get("META_MAX_STEPS") or os.environ.get("CS_MAX_STEPS", "100"))
REQUEST_TIMEOUT = float(os.environ.get("META_HTTP_TIMEOUT", "600"))
MAX_RETRIES = int(os.environ.get("META_MAX_RETRIES", "10"))
GRID = 1000.0
IMAGE_MAX = int(os.environ.get("META_IMAGE_MAX", "30"))
IMAGE_KEEP = int(os.environ.get("META_IMAGE_KEEP", "5"))
CONTEXT_LIMIT = int(os.environ.get("META_CONTEXT_LIMIT", "256000"))
TRIM_RATIO = 0.8
CLIENT_PASSWORD = os.environ.get("META_CLIENT_PASSWORD", "password")
# Env-gateway HTTP timeout. The client default of 120s is too tight for
# heavyweight environments (MyPCBench app pages can hold a step/observe past
# it), and a timeout here kills the whole episode.
ENV_HTTP_TIMEOUT = float(os.environ.get("META_ENV_HTTP_TIMEOUT", "600"))
SESSION_ID = os.environ.get("META_SESSION_ID") or f"cua-speedrun--{uuid.uuid4().hex}"

IMAGE_PLACEHOLDER = "(earlier screenshot removed to save context)"
TOOL_OUTPUT_PLACEHOLDER = "[Tool output is cleared due to context compaction]"

SYSTEM_PROMPT = """You are a computer-use agent operating an Ubuntu desktop at {width}x{height} resolution through the `computer` tool.

You are given a task to complete on this desktop. The relevant application may already be open, or may need to be launched: press the Super key, type the app name, then press Enter.

Rules:
- Respond with exactly ONE `computer` tool call per turn. After each action you will receive a screenshot of the resulting screen state.
- Coordinates are normalized to a 0-1000 space: (0, 0) is the top-left corner of the screen and (1000, 1000) the bottom-right, regardless of resolution.
- When the task is fully complete, call `computer` with action="terminate" and status="success". If the task is impossible, use status="failure".

Helpful tips:
- My computer password is "{client_password}" when sudo is needed.
- Stick to the website or application already opened for the task when possible.
- Check the screenshot after each action and wait when the UI is still loading.
- You can act without asking for confirmation.
"""

NUDGE_MESSAGE = (
    "Your previous reply did not contain a tool call. Respond with exactly one "
    "`computer` tool call (use action=\"terminate\" if the task is finished or "
    "impossible)."
)

COMPUTER_TOOL = {
    "type": "function",
    "function": {
        "name": "computer",
        "description": (
            "Control the desktop GUI. Coordinates are normalized to 0-1000: "
            "(0,0)=top-left, (1000,1000)=bottom-right of the screen. "
            "Exactly one action per call."
        ),
        "parameters": {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "click",
                        "double_click",
                        "right_click",
                        "move",
                        "drag",
                        "type",
                        "key",
                        "scroll",
                        "wait",
                        "screenshot",
                        "terminate",
                    ],
                    "description": (
                        "click/double_click/right_click/move need x,y. "
                        "drag needs x,y (start) and to_x,to_y (end). "
                        "type needs text (typed into the focused element). "
                        "key needs keys (e.g. ['ctrl','l'] or ['enter']). "
                        "scroll needs scroll_direction and optionally amount, x, y. "
                        "wait pauses briefly (optionally duration seconds). "
                        "screenshot just returns a fresh screenshot. "
                        "terminate ends the task and needs status."
                    ),
                },
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "to_x": {"type": "integer"},
                "to_y": {"type": "integer"},
                "text": {"type": "string"},
                "keys": {"type": "array", "items": {"type": "string"}},
                "scroll_direction": {"type": "string", "enum": ["up", "down"]},
                "amount": {
                    "type": "integer",
                    "description": "Scroll clicks (default 3).",
                },
                "duration": {
                    "type": "number",
                    "description": "Seconds to wait (wait action only).",
                },
                "status": {"type": "string", "enum": ["success", "failure"]},
            },
        },
    },
}

# Model key name -> env key vocabulary (the names gemini35's tests verify the
# gym-anything keyboard backend accepts). The model is RL-aligned on macOS, so
# cmd/option emissions are a live risk even with an Ubuntu system prompt;
# cmd -> ctrl is the behaviour-preserving choice on Linux (the model reaches
# for cmd+L / cmd+F meaning "the browser shortcut", which on Linux is ctrl).
# Punctuation NAMES (minus/period/...) map to the literal characters.
KEY_MAPPING = {
    "alt": "alt",
    "apostrophe": "'",
    "arrowdown": "Down",
    "arrowleft": "Left",
    "arrowright": "Right",
    "arrowup": "Up",
    "backslash": "\\",
    "backspace": "BackSpace",
    "backtick": "`",
    "bksp": "BackSpace",
    "capslock": "capslock",
    "cmd": "ctrl",
    "comma": ",",
    "command": "ctrl",
    "ctl": "ctrl",
    "ctrl": "ctrl",
    "control": "ctrl",
    "del": "Delete",
    "delete": "Delete",
    "dot": ".",
    "down": "Down",
    "downarrow": "Down",
    "end": "End",
    "enter": "Return",
    "equal": "=",
    "equals": "=",
    "esc": "Escape",
    "escape": "Escape",
    "forwarddelete": "Delete",
    "forward-delete": "Delete",
    "fwddelete": "Delete",
    "grave": "`",
    "home": "Home",
    "insert": "insert",
    "keypadenter": "Return",
    "kpenter": "Return",
    "left": "Left",
    "leftarrow": "Left",
    "leftbracket": "[",
    "meta": "super",
    "minus": "-",
    "opt": "alt",
    "option": "alt",
    "pagedn": "pagedown",
    "pagedown": "pagedown",
    "pageup": "pageup",
    "period": ".",
    "plus": "+",
    "quote": "'",
    "return": "Return",
    "right": "Right",
    "rightarrow": "Right",
    "rightbracket": "]",
    "semicolon": ";",
    "shift": "shift",
    "slash": "/",
    "space": "space",
    "spacebar": "space",
    "super": "super",
    "tab": "Tab",
    "tilde": "`",
    "up": "Up",
    "uparrow": "Up",
    "win": "super",
}

# The task sets' non-ASCII is essentially latin punctuation; the guest keyboard
# backend cannot type non-ASCII, so transliterate rather than silently no-op.
TRANSLITERATE = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    " ": " ", "…": "...", "£": "GBP ", "€": "EUR ",
    "×": "x", "·": "-",
}


def api_key() -> str:
    key = os.environ.get("META_API_KEY")
    if not key:
        raise RuntimeError("set META_API_KEY for the Meta API")
    return key


def system_prompt(width: int, height: int) -> str:
    return SYSTEM_PROMPT.format(
        width=width, height=height, client_password=CLIENT_PASSWORD
    )


def build_payload(messages: list[dict[str, Any]], include_tools: bool = True) -> dict[str, Any]:
    # No sampling params, ever: Meta requires API defaults.
    payload: dict[str, Any] = {"model": MODEL, "messages": messages}
    if include_tools:
        payload["tools"] = [COMPUTER_TOOL]
        payload["tool_choice"] = "auto"
    return payload


# Transient transport failures, matching what the source agent retries: its
# openai client maps ANY exception raised while sending the request -- including
# mid-stream protocol errors (httpx.RemoteProtocolError) and response decoding
# failures -- to APIConnectionError, which its retry ladder catches. requests
# surfaces those same failures as ChunkedEncodingError/ContentDecodingError,
# which are NOT ConnectionError subclasses, so list them explicitly.
RETRYABLE_EXCEPTIONS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ContentDecodingError,
)


def is_context_overflow(status_code: int, body: str) -> bool:
    text = body.lower()
    return status_code == 400 and (
        "context" in text
        or "too long" in text
        or ("maximum" in text and "token" in text)
    )


def is_content_policy(status_code: int, body: str) -> bool:
    text = body.lower()
    return status_code == 400 and "content" in text and ("policy" in text or "filter" in text)


def should_retry_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


class History:
    """Chat history with the vendor-mandated image/context hygiene.

    Screenshots ride on user messages (the relay rejects images on tool
    messages). The task instruction lives in its own text-only message so
    image cleanup can never touch the goal.
    """

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.image_msg_indices: list[int] = []
        self.last_prompt_tokens = 0

    def append_image_user_message(self, text: str, png: bytes) -> None:
        encoded = base64.b64encode(png).decode("ascii")
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            }
        )
        self.image_msg_indices.append(len(self.messages) - 1)
        if len(self.image_msg_indices) >= IMAGE_MAX:
            self.cleanup_images(keep=IMAGE_KEEP)

    def cleanup_images(self, keep: int) -> None:
        """Replace stale screenshots with a placeholder, keeping the last `keep`.

        Replaces ONLY the image parts of each message and preserves sibling
        text parts.
        """
        if len(self.image_msg_indices) <= keep:
            return
        stale = self.image_msg_indices[:-keep] if keep > 0 else self.image_msg_indices[:]
        for idx in stale:
            content = self.messages[idx].get("content")
            if not isinstance(content, list):
                continue
            self.messages[idx]["content"] = [
                {"type": "text", "text": IMAGE_PLACEHOLDER}
                if part.get("type") == "image_url"
                else part
                for part in content
            ]
        self.image_msg_indices = self.image_msg_indices[-keep:] if keep > 0 else []
        print(f"cleaned {len(stale)} old screenshots, keeping {keep}", file=sys.stderr)

    def compact_tool_outputs(self) -> None:
        """Clear tool outputs earliest-to-latest, keeping the most recent one."""
        tool_indices = [
            i
            for i, msg in enumerate(self.messages)
            if msg.get("role") == "tool" and msg.get("content") != TOOL_OUTPUT_PLACEHOLDER
        ]
        for idx in tool_indices[:-1]:
            self.messages[idx]["content"] = TOOL_OUTPUT_PLACEHOLDER

    def maybe_compact(self) -> None:
        if self.last_prompt_tokens > CONTEXT_LIMIT * TRIM_RATIO:
            print(
                f"prompt_tokens={self.last_prompt_tokens} exceeds threshold: compacting",
                file=sys.stderr,
            )
            self.compact_tool_outputs()

    def append_assistant_message(self, message: dict[str, Any], tool_call: dict[str, Any] | None) -> None:
        entry: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
        if tool_call is not None:
            entry["tool_calls"] = [
                {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"].get("arguments") or "{}",
                    },
                }
            ]
        self.messages.append(entry)

    def append_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )


def meta_request(history: History, include_tools: bool = True) -> dict[str, Any]:
    """One chat-completions call with the ported retry/recovery ladder."""
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
        "x-session-id": SESSION_ID,
    }
    last_error = ""
    shrunk_once = False
    policy_attempts = 0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                API_URL,
                headers=headers,
                json=build_payload(history.messages, include_tools),
                timeout=REQUEST_TIMEOUT,
            )
        except RETRYABLE_EXCEPTIONS as exc:
            last_error = f"connection error: {exc}"
            print(f"meta request attempt {attempt}: {last_error}", file=sys.stderr)
            time.sleep(min(5.0 * attempt, 30.0))
            continue
        if resp.status_code < 400:
            data = resp.json()
            usage = data.get("usage") or {}
            if usage.get("prompt_tokens"):
                history.last_prompt_tokens = int(usage["prompt_tokens"])
                print(
                    f"usage: prompt={usage.get('prompt_tokens')} "
                    f"completion={usage.get('completion_tokens')}",
                    file=sys.stderr,
                )
            return data
        body = resp.text[:2000]
        last_error = f"HTTP {resp.status_code} {body[:500]}"
        if not shrunk_once and is_context_overflow(resp.status_code, body):
            print(f"context overflow; forcing history shrink: {last_error}", file=sys.stderr)
            shrunk_once = True
            history.cleanup_images(keep=2)
            history.compact_tool_outputs()
            continue
        if is_content_policy(resp.status_code, body):
            # One screenshot or one tool text tripping the filter must not kill
            # the task: recover in two escalating steps before giving up.
            policy_attempts += 1
            if policy_attempts == 1:
                print("content-policy refusal; dropping images", file=sys.stderr)
                history.cleanup_images(keep=1)
                continue
            if policy_attempts == 2:
                print("content-policy refusal persists; clearing tool outputs", file=sys.stderr)
                for msg in history.messages:
                    if msg.get("role") == "tool":
                        msg["content"] = TOOL_OUTPUT_PLACEHOLDER
                history.cleanup_images(keep=0)
                continue
            break
        if not should_retry_status(resp.status_code):
            break
        print(f"meta request attempt {attempt}: {last_error}", file=sys.stderr)
        time.sleep(min(5.0 * attempt, 30.0))
    raise RuntimeError(f"Meta request failed after {MAX_RETRIES} attempts: {last_error}")


def image_size(png: bytes) -> tuple[int, int]:
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
        raise ValueError("observation is not a valid PNG with an IHDR chunk")
    return struct.unpack(">II", png[16:24])


def clamp_xy(x: Any, y: Any, width: int, height: int) -> tuple[int | None, int | None]:
    """Convert 0-1000 normalized model coordinates to clamped absolute pixels."""
    try:
        nx = min(max(float(x), 0.0), GRID)
        ny = min(max(float(y), 0.0), GRID)
    except (TypeError, ValueError):
        return None, None
    # json.loads accepts a bare NaN token and min/max propagate it, so guard
    # explicitly: one bad action must not crash the task.
    if not (math.isfinite(nx) and math.isfinite(ny)):
        return None, None
    cx = min(int(round(nx / GRID * width)), width - 1)
    cy = min(int(round(ny / GRID * height)), height - 1)
    return cx, cy


def transliterate(text: str) -> str:
    """Best-effort ASCII rendering of `text`; unmappable chars are dropped."""
    if text.isascii():
        return text
    out = []
    for ch in text:
        if ch.isascii():
            out.append(ch)
            continue
        if ch in TRANSLITERATE:
            out.append(TRANSLITERATE[ch])
            continue
        base = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
        if base:
            out.append(base)
        else:
            print(f"dropping untypeable character {ch!r}", file=sys.stderr)
    return "".join(out)


def map_key(key: Any) -> str:
    text = str(key).strip()
    return KEY_MAPPING.get(text.lower(), text.lower())


def type_actions(text: str) -> list[dict[str, Any]]:
    lines = text.split("\n")
    actions: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if line:
            actions.append({"keyboard": {"text": line}})
        if index < len(lines) - 1:
            actions.append({"keyboard": {"keys": ["Return"]}})
    return actions


class Outcome:
    def __init__(self, result_text: str, terminal: bool = False, executed: bool = True):
        self.result_text = result_text
        self.terminal = terminal
        self.executed = executed


INVALID_RESULT = (
    "Invalid or unsupported action/arguments; nothing was executed. "
    "The current screenshot follows."
)


def convert_and_execute(
    computer: Computer, name: str, args: dict[str, Any], width: int, height: int
) -> Outcome:
    if name != "computer":
        return Outcome(
            f"Unknown tool '{name}'; nothing was executed. Use the `computer` "
            "tool. The current screenshot follows.",
            executed=False,
        )
    action = str(args.get("action") or "").lower()

    if action == "terminate":
        if args.get("status") != "success":
            computer.step([{"action_type": "FAIL"}])
        return Outcome("Task terminated.", terminal=True)

    actions: list[dict[str, Any]] = []

    if action in ("click", "double_click", "right_click", "move"):
        x, y = clamp_xy(args.get("x"), args.get("y"), width, height)
        if x is None or y is None:
            return Outcome(INVALID_RESULT, executed=False)
        mouse_key = {
            "click": "left_click",
            "double_click": "double_click",
            "right_click": "right_click",
            "move": "move",
        }[action]
        actions.append({"mouse": {mouse_key: [x, y]}})
    elif action == "drag":
        x, y = clamp_xy(args.get("x"), args.get("y"), width, height)
        tx, ty = clamp_xy(args.get("to_x"), args.get("to_y"), width, height)
        if x is None or y is None or tx is None or ty is None:
            return Outcome(INVALID_RESULT, executed=False)
        actions.append({"mouse": {"left_click_drag": [[x, y], [tx, ty]]}})
    elif action == "type":
        text = args.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        text = transliterate(text)
        if not text:
            return Outcome("Nothing to type. The current screenshot follows.")
        actions.extend(type_actions(text))
    elif action == "key":
        keys = args.get("keys") or []
        if isinstance(keys, str):
            keys = [keys]
        if not keys:
            return Outcome(INVALID_RESULT, executed=False)
        actions.append({"keyboard": {"keys": [map_key(k) for k in keys]}})
    elif action == "scroll":
        direction = str(args.get("scroll_direction") or "down").lower()
        try:
            amount = max(1, min(int(args.get("amount") or 3), 30))
        except (TypeError, ValueError):
            amount = 3
        # gym-anything scroll sign: positive = down (opposite of pyautogui).
        clicks = -amount if direction == "up" else amount
        x, y = clamp_xy(args.get("x"), args.get("y"), width, height)
        if x is not None and y is not None:
            actions.append({"mouse": {"move": [x, y]}})
        actions.append({"mouse": {"scroll": clicks}})
    elif action == "wait":
        try:
            secs = float(args.get("duration") or 0)
        except (TypeError, ValueError):
            secs = 0.0
        computer.wait(min(secs, 30.0) if secs > 0 else 1.0)
        return Outcome("Waited. The new screenshot follows.")
    elif action == "screenshot":
        return Outcome("The current screenshot follows.")
    else:
        return Outcome(INVALID_RESULT, executed=False)

    computer.step(actions)
    return Outcome("Action executed. The new screenshot follows.")


# --------------------------------------------------------------------------
# gui_pyautogui mode: the model writes pyautogui code; we AST-translate it.
# --------------------------------------------------------------------------

GUI_PYAUTOGUI_SYSTEM_PROMPT = """You are an autonomous agent operating a real Ubuntu computer to complete the user's task.

At each step you receive a screenshot of the screen ({width}x{height} pixels) and you control the computer with pyautogui. The relevant application may already be open, or may need to be launched.

OUTPUT FORMAT:
- Return your next action as a single ```python code block of pyautogui calls.
- Coordinates are normalized integers in [0, 1000]: (0, 0) is the top-left of the screen and (1000, 1000) the bottom-right, regardless of the pixel resolution. Read them off the screenshot; do not guess.
- Take one small step at a time, then wait for the next screenshot before continuing.
- You may write multiple lines and use time.sleep(seconds) between them when the UI needs a moment to update.
- Do NOT use pyautogui.screenshot() or pyautogui.locateCenterOnScreen(). Do NOT define variables or functions; nothing persists between steps. Use only literal numbers and strings as arguments.

NOTES:
- Use the 'ctrl' modifier for shortcuts. pyautogui.hotkey('ctrl', 'l') focuses the address bar; pyautogui.hotkey('ctrl', 'f') finds text on the page.
- Click a text field before typing. Use pyautogui.write('text') to type and pyautogui.press('enter') to confirm.
- Prefer a site's own search box, navigation, and menus over typing URLs. Do not guess or invent URLs.
- Scroll with pyautogui.scroll(-3) to go down 3 wheel clicks and pyautogui.scroll(3) to go up.
- The password for sudo is "{client_password}".

SPECIAL RESPONSES (return exactly one, alone in a code block):
- WAIT  - nothing to do yet; wait and re-observe.
- DONE  - the task is complete.
- FAIL  - the task cannot be completed.

Stop when you finish.

Think briefly about what you see, then act."""

PYAUTOGUI_NUDGE_MESSAGE = (
    "Your previous reply did not contain a ```python code block. Respond with a "
    "single ```python code block of pyautogui calls, or exactly one of WAIT, "
    "DONE, or FAIL alone in a code block."
)

PYAUTOGUI_SYNTAX_MESSAGE = (
    "Your previous code block was not executed; the screen is unchanged. "
    "Reason: {reason}. Send a corrected ```python code block of pyautogui "
    "calls using only the documented functions with literal arguments."
)

# The language tag must be anchored to end-of-line: with an unanchored optional
# tag, a fence opened as ```pyautogui matches a "py" alternative and captures
# "autogui\n..." as code.
_PY_FENCE_RE = re.compile(
    r"```[ \t]*[A-Za-z0-9_+.-]*[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE
)
CONTROL_TOKENS = ("WAIT", "DONE", "FAIL")


def extract_pyautogui_actions(text: str) -> list[str]:
    """Pull the action(s) out of a pyautogui-syntax reply.

    The action is the LAST fenced block (the model may reason first, then
    act); WAIT/DONE/FAIL are control tokens, split out line by line so a
    final action and DONE in one block become two env actions in order.
    """
    if not text:
        return []
    blocks = _PY_FENCE_RE.findall(text)
    code = (blocks[-1] if blocks else text).strip().strip("`").strip()
    if not code:
        return []
    control: str | None = None
    code_lines: list[str] = []
    for line in code.splitlines():
        bare = line.strip().strip("`").strip().rstrip(".")
        if bare.upper() in CONTROL_TOKENS and not bare.startswith("#"):
            control = bare.upper()
            continue
        # Models also emit print('DONE') and friends; accept the printed token.
        printed = re.fullmatch(
            r"print\(\s*['\"](WAIT|DONE|FAIL)['\"]\s*\)", bare, re.IGNORECASE
        )
        if printed:
            control = printed.group(1).upper()
            continue
        if line.strip():
            code_lines.append(line)
    if not code_lines:
        return [control] if control else []
    if not blocks and control is None:
        # No fence and no control token: the model was just talking.
        return []
    actions = ["\n".join(code_lines)]
    if control:
        actions.append(control)
    return actions


_BUTTON_ACTIONS = {
    "left": "left_click",
    "right": "right_click",
    "middle": "middle_click",
}
_CLICK_FNS = {
    "click": "left",
    "rightClick": "right",
    "middleClick": "middle",
    "doubleClick": "double",
    "tripleClick": "triple",
}


class _Cursor:
    """Virtual cursor so dragTo/relative motion have a start point."""

    def __init__(self) -> None:
        self.position: tuple[int, int] | None = None


def _literal(node: Any) -> float | None:
    """Numeric literal, handling -5 (UnaryOp(USub, 5))."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        return -float(node.operand.value)
    return None


def _str_literal(node: Any) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def translate_pyautogui_block(
    code: str, width: int, height: int
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Translate one pyautogui code block into env action segments.

    Returns (segments, None) on success or (None, reason) when the block
    cannot be executed. Segments are {"actions": [env dicts]} and
    {"wait": seconds}, in program order. Coordinate handling mirrors the
    source agent: points are 0-1000 normalized (clamped, scaled, capped at
    span-1); a call whose coordinates exceed 1000 is assumed to already be
    in pixels and passed through; relative offsets scale by span/1000 but
    are never clamped; scroll clicks clamp to +-30.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return None, f"not valid Python ({exc.msg})"

    def scale_point(x: float, y: float) -> tuple[int, int]:
        if abs(x) > 1000 or abs(y) > 1000:
            # Already pixel space; pass through (clamped to the screen).
            return (
                min(max(int(round(x)), 0), width - 1),
                min(max(int(round(y)), 0), height - 1),
            )
        px = min(max(x, 0.0), 1000.0) / 1000.0 * width
        py = min(max(y, 0.0), 1000.0) / 1000.0 * height
        return min(int(round(px)), width - 1), min(int(round(py)), height - 1)

    def scale_offset(dx: float, dy: float) -> tuple[int, int]:
        return int(round(dx / 1000.0 * width)), int(round(dy / 1000.0 * height))

    segments: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    cursor = _Cursor()

    class _Reject(Exception):
        pass

    def expand(stmts: list[ast.stmt], depth: int):
        """Yield executable statements, unrolling for-loops over literal
        ranges (the source agent executes loops natively in the guest)."""
        for stmt in stmts:
            if isinstance(stmt, ast.For):
                if depth >= 1:
                    raise _Reject("nested loops are not supported")
                it = stmt.iter
                if not (
                    isinstance(it, ast.Call)
                    and isinstance(it.func, ast.Name)
                    and it.func.id == "range"
                ):
                    raise _Reject("only for-loops over range(...) are supported")
                bounds = [_literal(a) for a in it.args]
                if not bounds or any(b is None for b in bounds):
                    raise _Reject("range bounds must be literal numbers")
                try:
                    count = len(range(*(int(b) for b in bounds[:3])))
                except (TypeError, ValueError) as exc:
                    raise _Reject("invalid range bounds") from exc
                name = stmt.target.id if isinstance(stmt.target, ast.Name) else None
                for sub in stmt.body:
                    for node in ast.walk(sub):
                        if isinstance(node, ast.Name) and node.id == name:
                            raise _Reject(
                                "the loop variable must not be used in the body"
                            )
                for _ in range(min(count, 40)):
                    yield from expand(stmt.body, depth + 1)
            else:
                yield stmt

    def flush() -> None:
        if actions:
            segments.append({"actions": list(actions)})
            actions.clear()

    def arg(call: ast.Call, index: int, name: str) -> Any:
        if index < len(call.args):
            return call.args[index]
        for kw in call.keywords:
            if kw.arg == name:
                return kw.value
        return None

    def xy(call: ast.Call) -> tuple[int, int] | None | str:
        nx, ny = arg(call, 0, "x"), arg(call, 1, "y")
        if nx is None and ny is None:
            return None
        vx, vy = _literal(nx), _literal(ny)
        if vx is None or vy is None:
            return "coordinates must be numeric literals"
        return scale_point(vx, vy)

    try:
        expanded = list(expand(tree.body, 0))
    except _Reject as exc:
        return None, str(exc)
    for stmt in expanded:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(stmt, ast.Assign) and all(
            isinstance(t, ast.Attribute) for t in stmt.targets
        ):
            continue  # pyautogui.FAILSAFE = False and friends
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return None, f"unsupported statement: {ast.unparse(stmt)[:80]}"
        call = stmt.value
        if not isinstance(call.func, ast.Attribute) or not isinstance(
            call.func.value, ast.Name
        ):
            return None, f"unsupported call: {ast.unparse(call)[:80]}"
        module, fn = call.func.value.id, call.func.attr

        if fn == "sleep" and module in ("time", "pyautogui"):
            # pyautogui.sleep is a documented alias for time.sleep.
            secs = _literal(arg(call, 0, "seconds"))
            if secs is None:
                return None, f"{module}.sleep needs a literal number"
            flush()
            segments.append({"wait": min(max(float(secs), 0.0), 30.0)})
            continue
        if module != "pyautogui":
            return None, f"unsupported module: {module}.{fn}"

        if fn in _CLICK_FNS or fn == "moveTo":
            point = xy(call)
            if isinstance(point, str):
                return None, point
            if point is None:
                if cursor.position is None:
                    return None, f"{fn}() without coordinates or a prior moveTo"
                point = cursor.position
            cursor.position = point
            if fn == "moveTo":
                actions.append({"mouse": {"move": list(point)}})
                continue
            kind = _CLICK_FNS[fn]
            if fn == "click":
                # pyautogui.click(x, y, clicks, interval, button)
                button = _str_literal(arg(call, 4, "button")) or "left"
                clicks = _literal(arg(call, 2, "clicks")) or 1
                if button == "left" and clicks >= 3:
                    kind = "triple"
                elif button == "left" and clicks == 2:
                    kind = "double"
                else:
                    kind = button
            action_name = {
                "double": "double_click",
                "triple": "triple_click",
            }.get(kind) or _BUTTON_ACTIONS.get(kind)
            if action_name is None:
                return None, f"unsupported mouse button: {kind}"
            actions.append({"mouse": {action_name: list(point)}})
        elif fn in ("dragTo", "drag", "dragRel", "moveRel", "move"):
            if cursor.position is None:
                return None, f"{fn}() needs a prior moveTo/click to anchor the cursor"
            vx, vy = _literal(arg(call, 0, "x")), _literal(arg(call, 1, "y"))
            if vx is None or vy is None:
                return None, f"{fn} coordinates must be numeric literals"
            if fn == "dragTo":
                target = scale_point(vx, vy)
            else:
                dx, dy = scale_offset(vx, vy)
                target = (cursor.position[0] + dx, cursor.position[1] + dy)
            target = (
                min(max(target[0], 0), width - 1),
                min(max(target[1], 0), height - 1),
            )
            if fn in ("moveRel", "move"):
                actions.append({"mouse": {"move": list(target)}})
            else:
                actions.append(
                    {"mouse": {"left_click_drag": [list(cursor.position), list(target)]}}
                )
            cursor.position = target
        elif fn in ("scroll", "vscroll", "hscroll"):
            clicks = _literal(arg(call, 0, "clicks"))
            if clicks is None:
                return None, f"{fn} needs a literal clicks argument"
            clicks = max(-30.0, min(30.0, clicks))
            px, py = _literal(arg(call, 1, "x")), _literal(arg(call, 2, "y"))
            if px is not None and py is not None:
                point = scale_point(px, py)
                cursor.position = point
                actions.append({"mouse": {"move": list(point)}})
            if fn == "hscroll":
                # Horizontal scroll = shift + wheel; pyautogui positive = right.
                actions.append({"keyboard": {"key_down": "shift"}})
                actions.append({"mouse": {"scroll": int(round(clicks))}})
                actions.append({"keyboard": {"key_up": "shift"}})
            else:
                # pyautogui: positive scrolls up; env convention: positive = down.
                actions.append({"mouse": {"scroll": -int(round(clicks))}})
        elif fn in ("write", "typewrite"):
            text = _str_literal(arg(call, 0, "message"))
            if text is None:
                return None, f"{fn} needs a literal string"
            actions.extend(type_actions(transliterate(text)))
        elif fn == "press":
            key_node = arg(call, 0, "keys")
            presses = int(_literal(arg(call, 1, "presses")) or 1)
            keys: list[str] = []
            if _str_literal(key_node) is not None:
                keys = [_str_literal(key_node)]
            elif isinstance(key_node, (ast.List, ast.Tuple)):
                keys = [k for k in (_str_literal(e) for e in key_node.elts) if k]
            if not keys:
                return None, "press needs a literal key name"
            for _ in range(max(1, min(presses, 20))):
                for key in keys:
                    actions.append({"keyboard": {"keys": [map_key(key)]}})
        elif fn == "hotkey":
            keys = [k for k in (_str_literal(a) for a in call.args) if k]
            if not keys:
                return None, "hotkey needs literal key names"
            actions.append({"keyboard": {"keys": [map_key(k) for k in keys]}})
        elif fn in ("keyDown", "keyUp"):
            key = _str_literal(arg(call, 0, "key"))
            if key is None:
                return None, f"{fn} needs a literal key name"
            field = "keys_down" if fn == "keyDown" else "keys_up"
            actions.append({"keyboard": {field: [map_key(key)]}})
        elif fn in ("mouseDown", "mouseUp"):
            point = xy(call)
            if isinstance(point, str):
                return None, point
            if point is not None:
                cursor.position = point
                actions.append({"mouse": {"move": list(point)}})
            button = _str_literal(arg(call, 2, "button")) or "left"
            if button not in ("left", "right"):
                return None, f"unsupported {fn} button: {button}"
            state = f"{button}_down" if fn == "mouseDown" else f"{button}_up"
            actions.append({"mouse": {"buttons": {state: True}}})
        else:
            return None, f"unsupported pyautogui call: {fn}"

    flush()
    if not segments:
        return None, "the block contained no executable action"
    return segments, None


def run_pyautogui(env_url: str, task: str) -> None:
    computer = Computer(env_url, timeout_sec=ENV_HTTP_TIMEOUT)
    history = History()
    try:
        obs = computer.observe()
        width, height = image_size(obs["png"])
        history.messages.append(
            {
                "role": "system",
                "content": GUI_PYAUTOGUI_SYSTEM_PROMPT.format(
                    width=width, height=height, client_password=CLIENT_PASSWORD
                ),
            }
        )
        history.messages.append({"role": "user", "content": f"Task: {task}"})
        history.append_image_user_message(
            "Here is the current screenshot of the screen.", obs["png"]
        )

        for step in range(MAX_STEPS):
            history.maybe_compact()
            terminal = False
            executed = False
            for _nudge in range(3):
                data = meta_request(history, include_tools=False)
                message = data["choices"][0]["message"]
                content = str(message.get("content") or "")
                history.messages.append({"role": "assistant", "content": content})
                items = extract_pyautogui_actions(content)
                if not items:
                    print(f"step {step}: no code block; nudging", file=sys.stderr)
                    history.messages.append(
                        {"role": "user", "content": PYAUTOGUI_NUDGE_MESSAGE}
                    )
                    continue
                # Validate every code item before executing anything.
                translated: list[tuple[str, Any]] = []
                reason = None
                for item in items:
                    if item in CONTROL_TOKENS:
                        translated.append(("control", item))
                        continue
                    segments, reason = translate_pyautogui_block(item, width, height)
                    if reason is not None:
                        break
                    translated.append(("segments", segments))
                if reason is not None:
                    print(f"step {step}: rejected block: {reason}", file=sys.stderr)
                    history.messages.append(
                        {
                            "role": "user",
                            "content": PYAUTOGUI_SYNTAX_MESSAGE.format(reason=reason),
                        }
                    )
                    continue
                for kind, value in translated:
                    if kind == "control":
                        print(f"step {step}: control {value}", file=sys.stderr)
                        if value == "WAIT":
                            computer.wait(2.0)
                            continue
                        if value == "FAIL":
                            computer.step([{"action_type": "FAIL"}])
                        terminal = True
                        break
                    for segment in value:
                        if "wait" in segment:
                            computer.wait(segment["wait"])
                        elif segment["actions"]:
                            print(
                                f"step {step}: exec "
                                f"{json.dumps(segment['actions'], ensure_ascii=False)[:400]}",
                                file=sys.stderr,
                            )
                            computer.step(segment["actions"])
                executed = True
                break
            if not executed:
                print(f"step {step}: no action after nudges; ending task", file=sys.stderr)
                break
            if terminal:
                break

            obs = computer.observe()
            width, height = image_size(obs["png"])
            history.append_image_user_message(
                "Here is the screenshot after the action.", obs["png"]
            )
    except Exception as exc:
        print(f"agent failed: {exc!r}", file=sys.stderr, flush=True)
    finally:
        try:
            computer.done()
        except Exception as exc:
            print(f"computer.done failed: {exc!r}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# responses mode (default): the Responses API stateless variant of the gui
# mode, a 1:1 port of the source agent's `gui` mode over its Responses-API
# layer. Chat Completions does not preserve the model's hidden
# chain-of-thought across turns, which measurably hurts multi-step
# performance; here each turn replays the full item history including
# encrypted reasoning items (store=false,
# include=["reasoning.encrypted_content"]). No sampling params and no
# reasoning overrides are ever sent: API defaults only, per vendor
# requirements.
# --------------------------------------------------------------------------

RESPONSES_URL = f"{BASE_URL}/responses"


def flatten_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Chat-Completions nests the schema under "function"; the Responses API
    wants it flat. Accepts either shape (source: MetaAgent._flatten_tool)."""
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        fn = tool["function"]
        return {
            "type": "function",
            "name": fn.get("name"),
            "description": fn.get("description"),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        }
    return tool


# gui mode's toolset: the single `computer` function tool (with the built-in
# terminate action), exactly as the source builds it for the Responses API.
RESPONSES_TOOLS = [flatten_tool(COMPUTER_TOOL)]


def responses_payload(input_items: list[dict[str, Any]]) -> dict[str, Any]:
    # No sampling params and no `reasoning` field, ever: Meta requires API
    # defaults (the source's _call_llm sends exactly these keys).
    return {
        "model": MODEL,
        "input": input_items,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "tools": RESPONSES_TOOLS,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    }


class ContextExceeded(RuntimeError):
    pass


def _image_item(png: bytes) -> dict[str, Any]:
    """Responses-API image part. (Chat Completions used {"type":"image_url",
    "image_url":{"url":...}}; the Responses API takes a flat string url.)"""
    encoded = base64.b64encode(png).decode("ascii")
    return {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"}


def resp_item(item: dict[str, Any]) -> dict[str, Any]:
    """A response output item (reasoning / message / function_call) as a plain
    dict suitable for feeding straight back into the next request's `input`.

    Dropping None-valued keys matters: the API rejects nulls on some fields."""
    return {k: v for k, v in item.items() if v is not None}


class ResponsesHistory:
    """Responses-API input-item history with the source's image/context hygiene.

    Screenshots ride on user messages (never on function_call_output items).
    The task instruction lives in its own text-only message so image cleanup
    can never touch the goal.
    """

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.image_item_indices: list[int] = []
        self.last_prompt_tokens = 0

    def append_image_user_message(self, text: str, png: bytes) -> None:
        self.items.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": text}, _image_item(png)],
            }
        )
        self.image_item_indices.append(len(self.items) - 1)
        # Meta guidance: upon reaching IMAGE_MAX images, keep only the latest
        # IMAGE_KEEP (30 -> 5 sawtooth).
        if len(self.image_item_indices) >= IMAGE_MAX:
            self.cleanup_images(keep=IMAGE_KEEP)

    def cleanup_images(self, keep: int) -> None:
        """Replace stale screenshots with a placeholder, keeping the last `keep`.

        Replaces ONLY the image parts of each message and preserves sibling
        text parts.
        """
        if len(self.image_item_indices) <= keep:
            return
        stale = self.image_item_indices[:-keep] if keep > 0 else self.image_item_indices[:]
        for idx in stale:
            content = self.items[idx].get("content")
            if not isinstance(content, list):
                continue
            self.items[idx]["content"] = [
                {"type": "input_text", "text": IMAGE_PLACEHOLDER}
                if part.get("type") == "input_image"
                else part
                for part in content
            ]
        self.image_item_indices = self.image_item_indices[-keep:] if keep > 0 else []
        print(f"cleaned {len(stale)} old screenshots, keeping {keep}", file=sys.stderr)

    def compact_tool_outputs(self) -> None:
        """Clear tool outputs earliest-to-latest, keeping the most recent one.

        NB: `reasoning` items are never touched -- carrying them across turns
        is the whole point of the Responses API here.
        """
        tool_indices = [
            i
            for i, item in enumerate(self.items)
            if item.get("type") == "function_call_output"
            and item.get("output") != TOOL_OUTPUT_PLACEHOLDER
        ]
        for idx in tool_indices[:-1]:
            self.items[idx]["output"] = TOOL_OUTPUT_PLACEHOLDER

    def maybe_compact(self) -> None:
        if self.last_prompt_tokens > CONTEXT_LIMIT * TRIM_RATIO:
            print(
                f"input_tokens={self.last_prompt_tokens} exceeds threshold: compacting",
                file=sys.stderr,
            )
            # Meta's prescribed lever for gui mode: clear history tool outputs
            # earliest->latest, keeping the latest. Reasoning is NEVER trimmed
            # (Meta: "we keep all of the reasoning"). If context is still
            # exceeded, the request fails and the task is deemed failed --
            # Meta's recommended behavior for context overflow.
            self.compact_tool_outputs()

    def append_output_items(self, output: list[dict[str, Any]]) -> None:
        """Echo back EVERY output item -- reasoning included.

        The reasoning items carry the model's encrypted hidden chain-of-thought;
        replaying them in the next request is what preserves reasoning state
        across turns (Chat Completions had no way to do this).

        One-call-per-turn: we only emit a function_call_output for the FIRST
        function_call. So we must NOT replay any extra parallel function_call
        items -- an unanswered call_id is orphaned in the next request and
        returns HTTP 400. Reasoning and message items are all kept; only
        surplus function_calls are dropped."""
        seen_call = False
        for item in output:
            if item.get("type") == "function_call":
                if seen_call:
                    print(
                        "dropping surplus parallel function_call to avoid orphan",
                        file=sys.stderr,
                    )
                    continue
                seen_call = True
            self.items.append(resp_item(item))
        # Responses API rule: a replayed `reasoning` item must be followed by an
        # assistant message or function_call before any new user/system message,
        # else HTTP 400. super_nova_ext always emits reasoning FIRST (followed by
        # a message/function_call), so this normally never fires -- but guard the
        # trailing-reasoning case so a fresh user obs can't sit right after it.
        if self.items and self.items[-1].get("type") == "reasoning":
            self.items.append(
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "(continuing)"}],
                }
            )

    def append_tool_result(self, call_id: str, content: str) -> None:
        self.items.append(
            {"type": "function_call_output", "call_id": call_id, "output": content}
        )


def responses_request(history: ResponsesHistory) -> dict[str, Any]:
    """One Responses-API call with the source's retry/recovery ladder."""
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
        "x-session-id": SESSION_ID,
    }
    last_error = ""
    # Must be its own flag, NOT derived from last_error: the retryable branches
    # also set last_error, so a single transient 5xx/rate-limit blip anywhere in
    # this call would permanently disable the context-overflow recovery -- i.e.
    # switch off the safety net exactly on the long, blip-prone episodes that
    # need it.
    shrunk_once = False
    policy_attempts = 0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                RESPONSES_URL,
                headers=headers,
                json=responses_payload(history.items),
                timeout=REQUEST_TIMEOUT,
            )
        except RETRYABLE_EXCEPTIONS as exc:
            last_error = f"connection error: {exc}"
            print(f"responses request attempt {attempt}: {last_error}", file=sys.stderr)
            time.sleep(min(5.0 * attempt, 30.0))
            continue
        if resp.status_code < 400:
            data = resp.json()
            usage = data.get("usage") or {}
            if usage.get("input_tokens"):
                history.last_prompt_tokens = int(usage["input_tokens"])
                print(
                    f"usage: input={usage.get('input_tokens')} "
                    f"output={usage.get('output_tokens')}",
                    file=sys.stderr,
                )
            return data
        body = resp.text[:2000]
        last_error = f"HTTP {resp.status_code} {body[:500]}"
        if is_context_overflow(resp.status_code, body):
            # Context overflow: shrink aggressively, once.
            if not shrunk_once:
                print(f"context overflow; forcing history shrink: {last_error}", file=sys.stderr)
                shrunk_once = True
                history.cleanup_images(keep=2)
                history.compact_tool_outputs()
                continue
            # Reasoning is never truncated (vendor guidance), so once the
            # prescribed levers are spent the task is deemed failed.
            raise ContextExceeded(last_error)
        if is_content_policy(resp.status_code, body):
            # A content-policy refusal is terminal for THIS message but not for
            # the task: it is one screenshot or one tool text tripping the
            # filter. Recover in two escalating steps before giving up.
            policy_attempts += 1
            if policy_attempts == 1:
                print("content-policy refusal; dropping images", file=sys.stderr)
                history.cleanup_images(keep=1)
                continue
            if policy_attempts == 2:
                print("content-policy refusal persists; clearing tool outputs", file=sys.stderr)
                for item in history.items:
                    if item.get("type") == "function_call_output":
                        item["output"] = TOOL_OUTPUT_PLACEHOLDER
                history.cleanup_images(keep=0)
                continue
            break
        if not should_retry_status(resp.status_code):
            break
        print(f"responses request attempt {attempt}: {last_error}", file=sys.stderr)
        time.sleep(min(5.0 * attempt, 30.0))
    raise RuntimeError(f"Responses API failed after {MAX_RETRIES} attempts: {last_error}")


def run_responses(env_url: str, task: str) -> None:
    computer = Computer(env_url, timeout_sec=ENV_HTTP_TIMEOUT)
    history = ResponsesHistory()
    try:
        obs = computer.observe()
        width, height = image_size(obs["png"])
        history.items.append(
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt(width, height)}],
            }
        )
        # The instruction gets its own text-only message, never fused with a
        # screenshot: image cleanup must have no way to touch the task goal.
        history.items.append(
            {"role": "user", "content": [{"type": "input_text", "text": f"Task: {task}"}]}
        )
        history.append_image_user_message(
            "Here is the current screenshot of the screen.", obs["png"]
        )

        for step in range(MAX_STEPS):
            history.maybe_compact()
            call: dict[str, Any] | None = None
            for _nudge in range(3):
                data = responses_request(history)
                output = [
                    item for item in (data.get("output") or []) if isinstance(item, dict)
                ]
                calls = [item for item in output if item.get("type") == "function_call"]
                if len(calls) > 1:
                    print(
                        f"model returned {len(calls)} parallel tool calls; using the first",
                        file=sys.stderr,
                    )
                call = calls[0] if calls else None
                history.append_output_items(output)
                if call is not None:
                    break
                print(f"step {step}: no tool call; nudging", file=sys.stderr)
                history.items.append(
                    {"role": "user", "content": [{"type": "input_text", "text": NUDGE_MESSAGE}]}
                )
            if call is None:
                print(f"step {step}: no tool call after nudges; ending task", file=sys.stderr)
                break

            name = str(call.get("name") or "")
            args = parse_arguments({"function": {"arguments": call.get("arguments")}})
            print(
                f"step {step}: call {name} {json.dumps(args, ensure_ascii=False)[:500]}",
                file=sys.stderr,
            )
            outcome = convert_and_execute(computer, name, args, width, height)
            history.append_tool_result(
                str(call.get("call_id") or call.get("id") or ""), outcome.result_text
            )
            if outcome.terminal:
                break

            obs = computer.observe()
            width, height = image_size(obs["png"])
            history.append_image_user_message(
                "Here is the screenshot after the action.", obs["png"]
            )
    except ContextExceeded as exc:
        print(f"context window exceeded; failing task: {exc}", file=sys.stderr, flush=True)
        try:
            computer.step([{"action_type": "FAIL"}])
        except Exception:
            pass
    except Exception as exc:
        print(f"agent failed: {exc!r}", file=sys.stderr, flush=True)
    finally:
        try:
            computer.done()
        except Exception as exc:
            print(f"computer.done failed: {exc!r}", file=sys.stderr, flush=True)


def first_tool_call(message: dict[str, Any]) -> dict[str, Any] | None:
    tool_calls = message.get("tool_calls") or []
    if len(tool_calls) > 1:
        print(
            f"model returned {len(tool_calls)} parallel tool calls; using the first",
            file=sys.stderr,
        )
    return tool_calls[0] if tool_calls else None


def parse_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(tool_call["function"].get("arguments") or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        print(
            f"unparseable tool arguments: {tool_call['function'].get('arguments')!r}",
            file=sys.stderr,
        )
        return {}


def run(env_url: str, task: str) -> None:
    computer = Computer(env_url, timeout_sec=ENV_HTTP_TIMEOUT)
    history = History()
    try:
        obs = computer.observe()
        width, height = image_size(obs["png"])
        history.messages.append(
            {"role": "system", "content": system_prompt(width, height)}
        )
        # The instruction gets its own text-only message, never fused with a
        # screenshot: image cleanup must have no way to touch the task goal.
        history.messages.append({"role": "user", "content": f"Task: {task}"})
        history.append_image_user_message(
            "Here is the current screenshot of the screen.", obs["png"]
        )

        for step in range(MAX_STEPS):
            history.maybe_compact()
            tool_call = None
            for _nudge in range(3):
                data = meta_request(history)
                message = data["choices"][0]["message"]
                tool_call = first_tool_call(message)
                history.append_assistant_message(message, tool_call)
                if tool_call is not None:
                    break
                print(f"step {step}: no tool call; nudging", file=sys.stderr)
                history.messages.append({"role": "user", "content": NUDGE_MESSAGE})
            if tool_call is None:
                print(f"step {step}: no tool call after nudges; ending task", file=sys.stderr)
                break

            name = tool_call["function"]["name"]
            args = parse_arguments(tool_call)
            print(
                f"step {step}: call {name} {json.dumps(args, ensure_ascii=False)[:500]}",
                file=sys.stderr,
            )
            outcome = convert_and_execute(computer, name, args, width, height)
            history.append_tool_result(tool_call["id"], outcome.result_text)
            if outcome.terminal:
                break

            obs = computer.observe()
            width, height = image_size(obs["png"])
            history.append_image_user_message(
                "Here is the screenshot after the action.", obs["png"]
            )
    except Exception as exc:
        print(f"agent failed: {exc!r}", file=sys.stderr, flush=True)
    finally:
        try:
            computer.done()
        except Exception as exc:
            print(f"computer.done failed: {exc!r}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    if MODE == "gui_pyautogui":
        run_pyautogui(sys.argv[1], sys.argv[2])
    elif MODE == "gui":
        run(sys.argv[1], sys.argv[2])
    else:
        run_responses(sys.argv[1], sys.argv[2])
