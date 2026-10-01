"""GLM GUI agent (Zhipu / z.ai) for OSWorld-style desktop tasks, via OpenRouter.

Contract: python agent.py <env_url> <task_description>

The default model is `z-ai/glm-5v-turbo`. Set GLM_MODEL to use another
vision-capable model with the same action protocol.

Protocol: a port of the official GLM eval agent from
https://github.com/xlang-ai/OSWorld-V2 (mm_agents/glm_eval_agent.py +
mm_agents/glm_prompts.py, the GLM-V / AutoGLM GUI-agent convention):

- Stateless single-round prompting: every request is ONE user message that
  contains the task, the action space, the full text history of previous
  steps, the last <=4 screenshots shrunk to 50% x 50%, the model-maintained
  Memory block, and the current full-size screenshot.
- Plain-text function-call action space, VERBATIM
  EVAL_ACTION_SPACE_UNIFIED_TRIPLE: left/right/middle_click, hover,
  left_double_click, triple_click, left_drag, key, type, scroll, WAIT,
  DONE -- coordinates are thousandths (0-999), scaled with the reference's
  exact int(x * (width / 1000)) expression. The parser additionally accepts
  the FAIL terminal, exactly as the reference parser does even though the
  prompt text does not advertise it.
- The action is read from the model's <|begin_of_box|>...<|end_of_box|>
  markers via a port of the reference's balanced-paren action scanner
  (_find_last_action_by_parsing + right-boundary fallback, including the
  type(content=...) closing heuristics and terminal-token normalization);
  when no box markers survive the serving stack at all, the last
  recognizable action call in the text is used, mirroring the reference's
  _wrap_last_action_with_box_if_needed.
- The reply's separate reasoning field (OpenRouter message.reasoning, some
  providers message.reasoning_content) is stitched back in front of the
  content as <think>...</think> before parsing, exactly as the reference's
  call_llm builds final_answer.
- A clean parse that yields no action call is recorded as a no-op history
  step (not resampled), matching the reference; untranslatable actions
  count toward the consecutive-parse-failure cap and convert into the env
  FAIL action at the threshold. Exhausting the step cap (default 50, the
  reference's max_trajectory_length; GLM_MAX_STEPS / CS_MAX_STEPS
  override) also emits FAIL.
- left_click_hold: the reference appends this action (and accepts it) only
  for one hardcoded task id resolved from runner kwargs; this harness has
  no task-id channel, so the same behavior is gated on
  GLM_ENABLE_LEFT_CLICK_HOLD=1 (or GLM_TASK_ID equal to the upstream id).
- Sampling parameters match the vendor reference exactly (and are sent for
  that reason): temperature=1.0, top_p=1.0, max_tokens=8192, and
  thinking={"type": "enabled"} (z.ai's thinking switch, forwarded verbatim
  by OpenRouter to the provider).

Execution happens through the cua-speedrun Computer client with
gym-anything action dicts; the reference's pyautogui strings are translated
1:1 (including the scroll-sign inversion: env positive = down; hold_keys
sleeps ride as env wait control actions).

Environment adaptations:
- Coordinates are clamped into the screen after the reference's exact
  scaling expression. In-range inputs are grid-verified identical; the
  clamp only changes out-of-range values, which in-guest pyautogui clamps
  anyway.
- type() simulates typing line-by-line with Return between lines and
  transliterates non-ASCII (dropping untypeable characters, logged); the
  reference pastes via the guest clipboard (xsel), which this env substrate
  does not expose. Non-ASCII beyond the transliteration table is lost.
- Untranslatable-action steps are recorded in the text history (the
  reference drops them from history); the model sees what it emitted.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import struct
import sys
import time
import unicodedata
from typing import Any

import requests

from cua_speedrun.client import Computer

MODEL = os.environ.get("GLM_MODEL", "z-ai/glm-5v-turbo")
BASE_URL = os.environ.get("GLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
API_URL = f"{BASE_URL}/chat/completions"
# Reference default: max_trajectory_length=50 (glm_eval_agent.py); hitting
# the cap converts the step into FAIL.
MAX_STEPS = int(
    os.environ.get("GLM_MAX_STEPS") or os.environ.get("CS_MAX_STEPS") or "50"
)
REQUEST_TIMEOUT = float(os.environ.get("GLM_HTTP_TIMEOUT", "180"))
MAX_RETRIES = int(os.environ.get("GLM_MAX_RETRIES", "10"))
GRID = 1000.0
# Vendor reference sampling (glm_eval_agent.py InsertEvalAgent defaults).
TEMPERATURE = float(os.environ.get("GLM_TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("GLM_TOP_P", "1.0"))
MAX_TOKENS = int(os.environ.get("GLM_MAX_TOKENS", "8192"))
THINKING_ENABLED = os.environ.get("GLM_THINKING", "enabled").lower() not in {
    "0", "false", "disabled", "no",
}
# The reference sends the last 4 history screenshots at 50% x 50%.
HISTORY_IMAGES = int(os.environ.get("GLM_HISTORY_IMAGES", "4"))
MAX_CONSECUTIVE_PARSE_FAILURES = int(
    os.environ.get("GLM_MAX_CONSECUTIVE_PARSE_FAILURES", "5")
)
# Upstream gates left_click_hold on one hardcoded task id resolved from
# runner kwargs (LEFT_CLICK_HOLD_TASK_ID in glm_eval_agent.py). This harness
# passes no task id to agents, so the gate is an env switch -- or GLM_TASK_ID
# matching the upstream id exactly, for runners that export one.
LEFT_CLICK_HOLD_TASK_ID = "47543840-672a-467d-80df-8f7c3b9788c9"
LEFT_CLICK_HOLD_DURATION_SECONDS = 8.0
LEFT_CLICK_HOLD_ENABLED = (
    os.environ.get("GLM_ENABLE_LEFT_CLICK_HOLD", "").strip().lower()
    in {"1", "true", "yes"}
    or os.environ.get("GLM_TASK_ID", "").strip() == LEFT_CLICK_HOLD_TASK_ID
)
CLIENT_PASSWORD = os.environ.get("GLM_CLIENT_PASSWORD", "password")
# Env-gateway HTTP timeout: the client default of 120s is too tight for
# heavyweight environments; a timeout here kills the whole episode.
ENV_HTTP_TIMEOUT = float(os.environ.get("GLM_ENV_HTTP_TIMEOUT", "600"))


def _provider_preference(order_env: str, allow_fallbacks_env: str) -> dict[str, Any] | None:
    """Build an OpenRouter provider-routing preference from env vars.

    ``order_env`` holds a comma-separated list of OpenRouter provider slugs
    (e.g. "z-ai" or "minimax,novita"); when set, OpenRouter is asked to try
    them in that order. ``allow_fallbacks_env`` (default off -> pin hard) may
    be set truthy to permit OpenRouter to fall back to other providers. Only
    meaningful when the request goes through OpenRouter; harmless otherwise
    since it is only added when the order env is non-empty. Returns None when
    no order is configured so the request body is untouched.
    """
    raw = os.environ.get(order_env, "").strip()
    if not raw:
        return None
    order = [p.strip() for p in raw.split(",") if p.strip()]
    if not order:
        return None
    allow = os.environ.get(allow_fallbacks_env, "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    return {"order": order, "allow_fallbacks": allow}


# Optional OpenRouter provider pin. z-ai/glm-5v-turbo is served ONLY by Z.AI
# on OpenRouter today, so this is a no-op safeguard there, but the knob lets a
# run pin the provider (and disable silent fallback) if that changes.
PROVIDER_PREFERENCE = _provider_preference(
    "GLM_PROVIDER_ORDER", "GLM_PROVIDER_ALLOW_FALLBACKS"
)

BOX_BEGIN = "<|begin_of_box|>"
BOX_END = "<|end_of_box|>"

# EVAL_ACTION_SPACE_UNIFIED_TRIPLE from OSWorld-V2 mm_agents/glm_prompts.py,
# VERBATIM (the task-specific left_click_hold extension is a separate
# upstream constant appended only for one hardcoded task id, so it is not
# part of this text). Plain text, not an API tools param: GLM's GUI
# convention describes the action space in the prompt.
ACTION_SPACE = """
### {left,right,middle}_click

Call rule: `{left,right,middle}_click(start_box='[x,y]', hold_keys='', element_info='')`
{
    'name': ['left_click', 'right_click', 'middle_click'],
    'description': 'Perform a left/right/middle mouse click at the specified coordinates on the screen.',
    'parameters': {
        'type': 'object',
        'properties': {
            'start_box': {
                'type': 'array',
                'items': {
                    'type': 'integer'
                },
                'description': 'Coordinates [x,y] where to perform the click, normalized to 0-999 range.'
            },
            'hold_keys': {
                'type': 'string',
                'description': "Keyboard key or key combination to hold while performing the click. Use '+' to separate keys in combinations (e.g., 'ctrl', 'shift', 'ctrl+shift'). If not provided or 'None', no key is held."
            },
            'element_info': {
                'type': 'string',
                'description': 'Text description of the UI element being clicked.'
            }
        },
        'required': ['start_box', 'element_info']
    }
}

### hover

Call rule: `hover(start_box='[x,y]', hold_keys='', element_info='')`
{
    'name': 'hover',
    'description': 'Move the mouse pointer to the specified coordinates without performing any click action.',
    'parameters': {
        'type': 'object',
        'properties': {
            'start_box': {
                'type': 'array',
                'items': {
                    'type': 'integer'
                },
                'description': 'Coordinates [x,y] where to move the mouse pointer, normalized to 0-999 range.'
            },
            'hold_keys': {
                'type': 'string',
                'description': "Keyboard key or key combination to hold while performing the click. Use '+' to separate keys in combinations (e.g., 'ctrl', 'shift', 'ctrl+shift'). If not provided or 'None', no key is held."
            },
            'element_info': {
                'type': 'string',
                'description': 'Text description of the UI element being hovered over.'
            }
        },
        'required': ['start_box', 'element_info']
    }
}

### left_{double,triple}_click

Call rule: `left_{double,triple}_click(start_box='[x,y]', hold_keys='', element_info='')`
{
    'name': 'left_{double,triple}_click',
    'description': 'Perform a left mouse double-click at the specified coordinates on the screen.',
    'parameters': {
        'type': 'object',
        'properties': {
            'start_box': {
                'type': 'array',
                'items': {
                    'type': 'integer'
                },
                'description': 'Coordinates [x,y] where to perform the double/triple-click, normalized to 0-999 range.'
            },
            'hold_keys': {
                'type': 'string',
                'description': "Keyboard key or key combination to hold while performing the click. Use '+' to separate keys in combinations (e.g., 'ctrl', 'shift', 'ctrl+shift'). If not provided or 'None', no key is held."
            },
            'element_info': {
                'type': 'string',
                'description': 'Text description of the UI element being double/triple-clicked.'
            }
        },
        'required': ['start_box', 'element_info']
    }
}

### left_drag

Call rule: `left_drag(start_box='[x1,y1]', start_element_info='', end_box='[x2,y2]', end_element_info='', hold_keys='')`
{
    'name': 'left_drag',
    'description': 'Drag the mouse from starting coordinates to ending coordinates while holding the left mouse button.',
    'parameters': {
        'type': 'object',
        'properties': {
            'start_box': {
                'type': 'array',
                'items': {
                    'type': 'integer'
                },
                'description': 'Starting coordinates [x1,y1] for the drag operation, normalized to 0-999 range.'
            },
            'start_element_info': {
                'type': 'string',
                'description': 'Text description of the UI element being dragged from.'
            },
            'end_box': {
                'type': 'array',
                'items': {
                    'type': 'integer'
                },
                'description': 'Ending coordinates [x2,y2] for the drag operation, normalized to 0-999 range.'
            },
            'end_element_info': {
                'type': 'string',
                'description': 'Text description of the UI element being dragged to.'
            },
            'hold_keys': {
                'type': 'string',
                'description': "Keyboard key or key combination to hold while performing the click. Use '+' to separate keys in combinations (e.g., 'ctrl', 'shift', 'ctrl+shift'). If not provided or 'None', no key is held."
            }
        },
        'required': ['start_box', 'start_element_info', 'end_box', 'end_element_info']
    }
}

### key

Call rule: `key(keys='')`
{
    'name': 'key',
    'description': 'Simulate pressing a single key or combination of keys on the keyboard.',
    'parameters': {
        'type': 'object',
        'properties': {
            'keys': {
                'type': 'string',
                'description': 'The key or key combination to press. Use '+' to separate keys in combinations (e.g., 'ctrl+c', 'alt+tab').'
            }
        },
        'required': ['keys']
    }
}

### type

Call rule: `type(content='')`
{
    'name': 'type',
    'description': 'Type text content into the currently focused text input field. This action only performs typing and does not handle field activation or clearing.',
    'parameters': {
        'type': 'object',
        'properties': {
            'content': {
                'type': 'string',
                'description': 'The text content to be typed into the active text field.'
            }
        },
        'required': ['content']
    }
}

### scroll

Call rule: `scroll(start_box='[x,y]', direction='', step=5, hold_keys='', element_info='')`
{
    'name': 'scroll',
    'description': 'Scroll an element at the specified coordinates in the specified direction by a given number of wheel steps.',
    'parameters': {
        'type': 'object',
        'properties': {
            'start_box': {
                'type': 'array',
                'items': {
                    'type': 'integer'
                },
                'description': 'Coordinates [x,y] of the element or area to scroll, normalized to 0-999 range.'
            },
            'direction': {
                'type': 'string',
                'enum': ['down', 'up'],
                'description': 'The direction to scroll: 'down' or 'up'.'
            },
            'step': {
                'type': 'integer',
                'default': 5,
                'description': 'Number of wheel steps to scroll, default is 5.'
            },
            'hold_keys': {
                'type': 'string',
                'description': "Keyboard key or key combination to hold while performing the click. Use '+' to separate keys in combinations (e.g., 'ctrl', 'shift', 'ctrl+shift'). If not provided or 'None', no key is held."
            },
            'element_info': {
                'type': 'string',
                'description': 'Text description of the UI element being scrolled.'
            }
        },
        'required': ['start_box', 'direction', 'element_info']
    }
}

### WAIT

Call rule: `WAIT()`
{
    'name': 'WAIT',
    'description': 'Wait for 5 seconds before proceeding to the next action.',
    'parameters': {
        'type': 'object',
        'properties': {},
        'required': []
    }
}

### DONE

Call rule: `DONE()`
{
    'name': 'DONE',
    'description': 'Indicate that the current task has been completed successfully and no further actions are needed.',
    'parameters': {
        'type': 'object',
        'properties': {},
        'required': []
    }
}"""

# EVAL_ACTION_SPACE_LEFT_CLICK_HOLD from glm_eval_agent.py, VERBATIM with
# __HOLD_SECONDS__ substituted (8) and .strip() applied, exactly as upstream
# builds the constant. Appended to the action space only when the
# left_click_hold gate is on (upstream: one hardcoded task id).
ACTION_SPACE_LEFT_CLICK_HOLD = """### left_click_hold

Call rule: `left_click_hold(start_box='[x,y]', hold_keys='', element_info='')`
{
    'name': 'left_click_hold',
    'description': 'Move to the target position, press and hold the left mouse button for 8 seconds, then release.',
    'parameters': {
        'type': 'object',
        'properties': {
            'start_box': {
                'type': 'array',
                'items': {
                    'type': 'integer'
                },
                'description': 'Coordinates [x,y] where to press and hold, normalized to 0-999 range.'
            },
            'hold_keys': {
                'type': 'string',
                'description': "Keyboard key or key combination to hold while performing the click. Use '+' to separate keys in combinations (e.g., 'ctrl', 'shift', 'ctrl+shift'). If not provided or 'None', no key is held."
            },
            'element_info': {
                'type': 'string',
                'description': 'Optional text description of the UI element being click-held.'
            }
        },
        'required': ['start_box']
    }
}"""


def action_space_text() -> str:
    """Reference _get_action_space: base space, plus the left_click_hold
    extension when the task gate is on."""
    if not LEFT_CLICK_HOLD_ENABLED:
        return ACTION_SPACE
    return f"{ACTION_SPACE.rstrip()}\n\n{ACTION_SPACE_LEFT_CLICK_HOLD}"


# EVAL_USER_INSERT_HEAD / EVAL_USER_INSERT_TAIL from glm_prompts.py, with the
# sudo password templated instead of hardcoded. The tail is otherwise
# verbatim; user_response_block is always formatted as "" because this
# harness has no ask-user channel (upstream renders the same empty block
# when no user response exists).
PROMPT_HEAD = """You are a GUI Agent, and your primary task is to respond accurately to user requests or questions. In addition to directly answering the user's queries, you can also use tools or perform GUI operations directly until you fulfill the user's request or provide a correct answer. You should carefully read and understand the images and questions provided by the user, and engage in thinking and reflection when appropriate. The coordinates involved are all represented in thousandths (0-999).

# Task:
{task}

# Task Platform
Ubuntu

# Action Space
{action_space}

# Historical Actions and Current Memory
History:"""

PROMPT_TAIL = """
Memory:
{memory}
# Output Format
Plain text explanation with action(param='...')
Memory:
[{{"key": "value"}}, ...]

# Some Additional Notes
- I'll give you the most recent 4 history screenshots(shrunked to 50%*50%) along with the historical action steps.
- If a latest user response is provided, use it to continue after your previous question to the user.
- You should put the key information you *have to remember* in a seperated memory part and I'll give it to you in the next round. The content in this part should be a dict list. If you no longer need some given information, you should remove it from the memory. Even if you don't need to remember anything, you should also output an empty list.
- My computer's password is "{client_password}", feel free to use it when you need sudo rights.

{user_response_block}
Current Screenshot:
"""

VALID_ACTIONS = (
    "left_triple_click",
    "left_double_click",
    "triple_click",
    "middle_click",
    "right_click",
    "left_click",
    "left_drag",
    "hover",
    "scroll",
    "key",
    "type",
) + (("left_click_hold",) if LEFT_CLICK_HOLD_ENABLED else ())
CONTROL_ACTIONS = ("WAIT", "DONE", "FAIL")


def _choice_pattern(actions: tuple[str, ...]) -> str:
    """Reference _build_action_choice_pattern: longest-first alternation."""
    return "|".join(re.escape(a) for a in sorted(actions, key=len, reverse=True))


# Reference parser patterns (_find_last_action_by_parsing /
# _find_last_action_by_right_boundary): the parsing pass matches every valid
# action (control tokens need parens there); the right-boundary fallback
# matches call actions plus bare/case-insensitive terminal tokens.
_PARSE_CALL_RE = re.compile("(" + _choice_pattern(VALID_ACTIONS + CONTROL_ACTIONS) + r")\s*\(")
_FALLBACK_CALL_RE = re.compile("(" + _choice_pattern(VALID_ACTIONS) + r")\s*\(")
_TERMINAL_RE = re.compile(
    r"\b(?:" + "|".join(CONTROL_ACTIONS) + r")\s*(?:\(\s*\))?\b", re.IGNORECASE
)
_TYPE_CLOSING_SUFFIXES = ("\"')", "')", "\")")

# Model key name -> env key vocabulary accepted by the gym-anything keyboard
# backend. Punctuation NAMES map to the literal characters.
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
    # pyautogui maps 'command'/'cmd' to Super_L on X11, so the reference's
    # pass-through sends the Super key; mirror that (env 'super' -> win key).
    "cmd": "super",
    "comma": ",",
    "command": "super",
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
    "page_down": "pagedown",
    "pageup": "pageup",
    "page_up": "pageup",
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
    "super_l": "super",
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
    " ": " ", "…": "...", "£": "GBP ", "€": "EUR ",
    "×": "x", "·": "-",
}


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GLM_API_KEY")
    if not key:
        raise RuntimeError("set OPENROUTER_API_KEY (or GLM_API_KEY) for the GLM API")
    return key


def build_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Chat Completions payload with EXACTLY the vendor reference's sampling.

    glm_eval_agent.py sends temperature/top_p/max_tokens/thinking/stream and
    nothing else; we mirror that (thinking is z.ai's reasoning switch and is
    forwarded to the provider by OpenRouter).
    """
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "enabled" if THINKING_ENABLED else "disabled"},
        "stream": False,
    }
    if PROVIDER_PREFERENCE is not None:
        payload["provider"] = PROVIDER_PREFERENCE
    return payload


def is_context_overflow(status_code: int, body: str) -> bool:
    text = body.lower()
    return status_code == 400 and (
        "context" in text
        or "too long" in text
        or ("maximum" in text and "token" in text)
    )


def should_retry_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


class ContextOverflow(RuntimeError):
    pass


def glm_request(messages: list[dict[str, Any]]) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
    }
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                API_URL,
                headers=headers,
                json=build_payload(messages),
                timeout=REQUEST_TIMEOUT,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = f"connection error: {exc}"
            print(f"glm request attempt {attempt}: {last_error}", file=sys.stderr)
            time.sleep(min(5.0 * attempt, 30.0))
            continue
        if resp.status_code < 400:
            data = resp.json()
            usage = data.get("usage") or {}
            if usage:
                print(
                    f"usage: prompt={usage.get('prompt_tokens')} "
                    f"completion={usage.get('completion_tokens')}",
                    file=sys.stderr,
                )
            return data
        body = resp.text[:2000]
        last_error = f"HTTP {resp.status_code} {body[:500]}"
        if is_context_overflow(resp.status_code, body):
            raise ContextOverflow(last_error)
        if not should_retry_status(resp.status_code):
            break
        print(f"glm request attempt {attempt}: {last_error}", file=sys.stderr)
        time.sleep(min(5.0 * attempt, 30.0))
    raise RuntimeError(f"GLM request failed after {MAX_RETRIES} attempts: {last_error}")


def image_size(png: bytes) -> tuple[int, int]:
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
        raise ValueError("observation is not a valid PNG with an IHDR chunk")
    return struct.unpack(">II", png[16:24])


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def shrink_png(png: bytes) -> bytes:
    """History screenshots go out at 50% x 50% (vendor reference)."""
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(png))
        shrunk = image.resize((max(1, image.width // 2), max(1, image.height // 2)))
        out = io.BytesIO()
        shrunk.save(out, format="PNG")
        return out.getvalue()
    except Exception as exc:  # PIL missing or malformed image: send full size.
        print(f"history screenshot shrink failed ({exc!r}); sending full size", file=sys.stderr)
        return png


class GlmHistory:
    """Text history for every step; raw screenshots for the last few only."""

    def __init__(self) -> None:
        self.thoughts: list[str] = []
        self.actions: list[str] = []
        self.screenshots: list[bytes | None] = []
        self.memory: str = "[]"

    def add(self, thought: str, action: str, png: bytes) -> None:
        self.thoughts.append(thought)
        self.actions.append(action)
        self.screenshots.append(png)
        for idx in range(len(self.screenshots) - HISTORY_IMAGES):
            self.screenshots[idx] = None

    def drop_images(self) -> None:
        self.screenshots = [None] * len(self.screenshots)


def build_user_message(task: str, history: GlmHistory, current_png: bytes) -> dict[str, Any]:
    """One interleaved user message per turn (glm_eval_agent convention).

    Head (task + action space) -> per-step text history with the last
    HISTORY_IMAGES screenshots shrunk to 50% -> tail (memory + notes) ->
    current full-size screenshot.
    """
    content: list[dict[str, Any]] = []
    text = PROMPT_HEAD.format(task=task, action_space=action_space_text())
    steps = len(history.actions)
    if steps == 0:
        text += "\nNone"
    for idx in range(steps):
        action = history.actions[idx]
        # Reference _build_interleaved_messages: the raw thought still
        # contains the action call text; remove it so the history line does
        # not duplicate the action (clean_thought_text upstream).
        raw_thought = history.thoughts[idx]
        thought = raw_thought.replace(action, "").strip() if raw_thought else "None"
        png = history.screenshots[idx]
        if png is None:
            text += (
                f"\nstep {idx + 1}: Screenshot:(Omitted in context.) "
                f"Thought: {thought}\nAction: {action}"
            )
        else:
            text += f"\nstep {idx + 1}: Screenshot:"
            content.append({"type": "text", "text": text})
            content.append(
                {"type": "image_url", "image_url": {"url": data_url(shrink_png(png))}}
            )
            text = f" Thought: {thought}\nAction: {action}"
    text += PROMPT_TAIL.format(
        memory=history.memory,
        client_password=CLIENT_PASSWORD,
        # This harness has no ask-user channel; upstream renders the same
        # empty block whenever there is no latest user response.
        user_response_block="",
    )
    content.append({"type": "text", "text": text})
    content.append({"type": "image_url", "image_url": {"url": data_url(current_png)}})
    return {"role": "user", "content": content}


def extract_parentheses_content(text: str) -> str | None:
    start = text.find("(")
    end = text.rfind(")")
    if start != -1 and end != -1 and end > start:
        return text[start + 1 : end]
    return None


def _normalize_action_call(action: str | None) -> str | None:
    """Reference _normalize_action_call: uppercase terminal tokens (always
    with parens), lowercase call names, arguments preserved."""
    action = (action or "").strip()
    if not action:
        return None
    match = re.match(r"^\s*([A-Za-z_]+)\s*(\((.*)\))?\s*$", action, re.DOTALL)
    if not match:
        return action
    name = match.group(1)
    call_suffix = match.group(2)
    if name.upper() in CONTROL_ACTIONS:
        return f"{name.upper()}{call_suffix if call_suffix is not None else '()'}"
    if call_suffix is None:
        return name.lower()
    return f"{name.lower()}{call_suffix}"


def _is_likely_quote_closer(text: str, quote_idx: int) -> bool:
    """Reference heuristic for malformed outputs like society's: treat a quote
    as a closer only when the next non-space char is a normal delimiter."""
    next_idx = quote_idx + 1
    while next_idx < len(text) and text[next_idx].isspace():
        next_idx += 1
    if next_idx >= len(text):
        return True
    return text[next_idx] in {",", ")", "]", "}"}


def _find_last_closing_suffix(segment: str, closings: tuple[str, ...]) -> int | None:
    best_pos = -1
    best_len = 0
    for closing in closings:
        pos = segment.rfind(closing)
        if pos > best_pos or (pos == best_pos and len(closing) > best_len):
            best_pos = pos
            best_len = len(closing)
    if best_pos == -1:
        return None
    return best_pos + best_len


def _find_type_action_end(text: str, start_idx: int) -> int | None:
    """Reference _find_type_action_end: locate the closing ') of a
    type(content=...) call whose content may embed unescaped quotes/parens.
    A trailing key(keys='Return'/'Enter') is excluded from the search first
    so it is not swallowed into the type content."""
    segment = text[start_idx:]
    lower_segment = segment.lower()
    key_match = re.search(
        r"key\s*\(\s*keys\s*=\s*['\"](?:return|enter)['\"]\s*\)", lower_segment
    )
    search_segment = segment[: key_match.start()] if key_match else segment
    closing_idx = _find_last_closing_suffix(search_segment, _TYPE_CLOSING_SUFFIXES)
    if closing_idx is not None:
        return start_idx + closing_idx
    if key_match:
        closing_idx = _find_last_closing_suffix(segment, _TYPE_CLOSING_SUFFIXES)
        if closing_idx is not None:
            return start_idx + closing_idx
    return None


def _find_last_action_by_parsing(text: str) -> str | None:
    """Reference _find_last_action_by_parsing: forward scan for the last
    balanced action call, quote-aware, skipping matches covered by an earlier
    action's arguments; type(content=...) closes via _find_type_action_end."""
    last_action: str | None = None
    covered_end = -1
    for match in _PARSE_CALL_RE.finditer(text):
        start_idx = match.start()
        if start_idx < covered_end:
            continue
        action_name = match.group(1)
        if action_name == "type":
            after_paren = text[match.end() : match.end() + 20]
            if after_paren.lstrip().startswith("content="):
                end_idx = _find_type_action_end(text, start_idx)
                if end_idx:
                    last_action = text[start_idx:end_idx]
                    covered_end = end_idx
                    continue
        current_idx = match.end() - 1
        balance = 0
        in_quote = False
        quote_char: str | None = None
        while current_idx < len(text):
            char = text[current_idx]
            if char == "\\":
                current_idx += 2
                continue
            if char in ("'", '"'):
                if not in_quote:
                    in_quote = True
                    quote_char = char
                elif char == quote_char:
                    if action_name == "type" or _is_likely_quote_closer(text, current_idx):
                        in_quote = False
                        quote_char = None
            if not in_quote:
                if char == "(":
                    balance += 1
                elif char == ")":
                    balance -= 1
                    if balance == 0:
                        end_idx = current_idx + 1
                        last_action = text[start_idx:end_idx]
                        covered_end = end_idx
                        break
            current_idx += 1
    return last_action


def _find_last_action_by_right_boundary(text: str) -> str | None:
    """Reference _find_last_action_by_right_boundary: last call action up to
    the final ')', or the last (case-insensitive, possibly bare) terminal."""
    stripped = (text or "").strip()
    if not stripped:
        return None
    call_matches = list(_FALLBACK_CALL_RE.finditer(stripped))
    terminal_matches = list(_TERMINAL_RE.finditer(stripped))
    last_call = call_matches[-1] if call_matches else None
    last_terminal = terminal_matches[-1] if terminal_matches else None
    if last_call and (not last_terminal or last_call.start() >= last_terminal.start()):
        end_idx = stripped.rfind(")")
        if end_idx == -1 or end_idx < last_call.start():
            return None
        return _normalize_action_call(stripped[last_call.start() : end_idx + 1])
    if last_terminal:
        return _normalize_action_call(last_terminal.group(0))
    return None


def find_action(candidate: str) -> str | None:
    """Last recognizable action call (or control token) in `candidate`,
    trimmed at the balanced closing paren (reference _find_last_action_span)."""
    action = _find_last_action_by_parsing(candidate)
    if action is not None:
        return _normalize_action_call(action)
    return _find_last_action_by_right_boundary(candidate)


def response_text_of(data: dict[str, Any]) -> str:
    """Reference call_llm: the reply's separate reasoning field is stitched
    back in front of the content as <think>...</think> before parsing
    (final_answer = f"<think>{reasoning_content}</think>{output_content}").
    OpenRouter serves GLM reasoning as message.reasoning; some providers use
    message.reasoning_content -- read both. Without this the parser and the
    text history never see the model's thought text."""
    message = (data.get("choices") or [{}])[0].get("message") or {}
    content = message.get("content")
    content = content if isinstance(content, str) else ("" if content is None else str(content))
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    return f"<think>{reasoning}</think>{content}"


def parse_response(text: str) -> dict[str, Any]:
    """Extract {action, thought, memory} from a GLM reply.

    The reference reads the action strictly from <|begin_of_box|> markers;
    we fall back to the last action-shaped call in the text so a serving
    stack that strips special tokens does not zero the episode.
    """
    action: str | None = None
    boxed = re.findall(
        re.escape(BOX_BEGIN) + r"(.*?)" + re.escape(BOX_END), text, re.DOTALL
    )
    begin_idx = text.rfind(BOX_BEGIN)
    if begin_idx != -1:
        tail = text[begin_idx + len(BOX_BEGIN) :]
        end_idx = tail.find(BOX_END)
        action = find_action(tail[:end_idx] if end_idx != -1 else tail)
    if action is None:
        for content in reversed(boxed):
            action = find_action(content)
            if action is not None:
                break
    body = text
    if "</think>" in body:
        body = body.split("</think>", 1)[1]
    if action is None and BOX_BEGIN not in text and BOX_END not in text:
        # Mirrors the reference's _wrap_last_action_with_box_if_needed, which
        # only rescues responses that contain NO box markers at all (when
        # markers exist but hold no action, upstream deliberately refuses a
        # full-response fallback to avoid template-text pollution).
        section = body.replace("</answer>", "").replace("<answer>", "")
        action = find_action(section.split("Memory:", 1)[0])

    thought_match = re.search(r"^(.*?)Memory:", body, re.DOTALL)
    thought = (thought_match.group(1) if thought_match else body).strip()
    thought = thought.replace(BOX_BEGIN, "").replace(BOX_END, "").strip()

    memory_match = re.search(r"Memory:(.*?)$", body, re.DOTALL)
    memory = (
        memory_match.group(1).replace("<|user|>", "").replace("</answer>", "").strip()
        if memory_match
        else "[]"
    )
    return {"action": action, "thought": thought, "memory": memory or "[]"}


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


def clamp_xy(x: Any, y: Any, width: int, height: int) -> tuple[int | None, int | None]:
    """Convert 0-999 thousandths coordinates to clamped absolute pixels."""
    try:
        nx = min(max(float(x), 0.0), GRID)
        ny = min(max(float(y), 0.0), GRID)
    except (TypeError, ValueError):
        return None, None
    if not (math.isfinite(nx) and math.isfinite(ny)):
        return None, None
    # Exactly the reference's float expression, int(x * (width / 1000)) --
    # writing it as int(x / 1000 * width) diverges by one pixel on some
    # values (e.g. x=175 at width=720) because float rounding differs.
    cx = min(int(nx * (width / GRID)), width - 1)
    cy = min(int(ny * (height / GRID)), height - 1)
    return cx, cy


def _parse_box(params: str, name: str) -> tuple[float, float] | None:
    match = re.search(rf"{name}\s*=\s*['\"]?\[\s*([^\]]*?)\s*\]['\"]?", params)
    if not match:
        # Tolerate the older '(x, y)' form used by some GLM checkpoints.
        match = re.search(rf"{name}\s*=\s*['\"]?\(\s*([^)]*?)\s*\)['\"]?", params)
    if not match:
        return None
    parts = [p.strip() for p in match.group(1).split(",") if p.strip()]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _string_param(params: str, name: str) -> str | None:
    match = re.search(rf"{name}\s*=\s*(['\"])", params, re.DOTALL)
    if not match:
        return None
    quote = match.group(1)
    start = match.end()
    end = params.rfind(quote)
    raw = params[start:] if end < start else params[start:end]
    try:
        import ast

        return str(ast.literal_eval(f"{quote}{raw}{quote}"))
    except Exception:
        return raw


def _hold_keys(params: str) -> list[str]:
    match = re.search(r"hold_keys\s*=\s*(['\"])(.*?)\1", params)
    if not match or not match.group(2) or match.group(2).lower() == "none":
        return []
    return [map_key(k) for k in match.group(2).split("+") if k.strip()]


class Outcome:
    def __init__(self, result_text: str, terminal: bool = False, executed: bool = True):
        self.result_text = result_text
        self.terminal = terminal
        self.executed = executed


INVALID_RESULT = "Invalid or unsupported action/arguments; nothing was executed."

_CLICK_ACTIONS = {
    "left_click": "left_click",
    "right_click": "right_click",
    "middle_click": "middle_click",
    "left_double_click": "double_click",
    "triple_click": "triple_click",
    "left_triple_click": "triple_click",
    "hover": "move",
}


def convert_and_execute(
    computer: Computer, action_text: str, width: int, height: int
) -> Outcome:
    """Translate one GLM action string into env actions and run it."""
    control = re.fullmatch(r"(WAIT|DONE|FAIL)\s*(?:\(\s*\))?", action_text.strip())
    if control:
        token = control.group(1)
        if token == "WAIT":
            computer.wait(5.0)
            return Outcome("Waited 5 seconds.")
        if token == "FAIL":
            computer.step([{"action_type": "FAIL"}])
            return Outcome("Task declared infeasible.", terminal=True)
        return Outcome("Task declared complete.", terminal=True)

    name_match = re.match(r"\s*([A-Za-z_]+)\s*\(", action_text)
    if not name_match:
        return Outcome(INVALID_RESULT, executed=False)
    name = name_match.group(1)
    params = extract_parentheses_content(action_text) or ""
    hold = _hold_keys(params)

    actions: list[dict[str, Any]] = []
    if name in _CLICK_ACTIONS:
        box = _parse_box(params, "start_box")
        if box is None:
            return Outcome(INVALID_RESULT, executed=False)
        x, y = clamp_xy(box[0], box[1], width, height)
        if x is None or y is None:
            return Outcome(INVALID_RESULT, executed=False)
        actions.append({"mouse": {_CLICK_ACTIONS[name]: [x, y]}})
    elif name == "left_drag":
        start = _parse_box(params, "start_box")
        end = _parse_box(params, "end_box")
        if start is None or end is None:
            return Outcome(INVALID_RESULT, executed=False)
        x1, y1 = clamp_xy(start[0], start[1], width, height)
        x2, y2 = clamp_xy(end[0], end[1], width, height)
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return Outcome(INVALID_RESULT, executed=False)
        actions.append({"mouse": {"left_click_drag": [[x1, y1], [x2, y2]]}})
    elif name == "key":
        keys = _string_param(params, "keys")
        if not keys:
            return Outcome(INVALID_RESULT, executed=False)
        raw = [k.strip().lower() for k in keys.split("+") if k.strip()]
        if not raw:
            return Outcome(INVALID_RESULT, executed=False)
        # Reference _build_hotkey_command: a repeated tail key means "press
        # this combo N times", not one chord (e.g. 'down+down' is two Down
        # presses; 'ctrl+tab+tab' is ctrl+tab twice). Upstream detects the
        # repeat on the RAW lowercased tokens, before any key-name mapping.
        tail_repeat = 1
        for key in reversed(raw[:-1]):
            if key == raw[-1]:
                tail_repeat += 1
            else:
                break
        combo = [map_key(k) for k in raw[: len(raw) - tail_repeat] + [raw[-1]]]
        for _ in range(tail_repeat):
            actions.append({"keyboard": {"keys": combo}})
    elif name == "type":
        content = _string_param(params, "content")
        if content is None:
            return Outcome(INVALID_RESULT, executed=False)
        content = transliterate(content)
        if not content:
            return Outcome("Nothing to type.")
        actions.extend(type_actions(content))
    elif name == "scroll":
        box = _parse_box(params, "start_box")
        direction_match = re.search(r"direction\s*=\s*['\"](.*?)['\"]", params)
        direction = (direction_match.group(1) if direction_match else "down").lower()
        step_match = re.search(r"step\s*=\s*['\"]?(\d+)", params)
        step = int(step_match.group(1)) if step_match else 5
        if box is not None:
            x, y = clamp_xy(box[0], box[1], width, height)
            if x is not None and y is not None:
                actions.append({"mouse": {"move": [x, y]}})
        # Reference: pyautogui.scroll(-step) for "down", +step for any other
        # direction (the prompt enum only offers down/up). Env sign is the
        # opposite of pyautogui's (env positive = down), so this flips.
        clicks = step if direction == "down" else -step
        actions.append({"mouse": {"scroll": clicks}})
    elif name == "left_click_hold" and LEFT_CLICK_HOLD_ENABLED:
        # Reference _build_click_hold_command: moveTo; mouseDown;
        # sleep(8.0); mouseUp. The wait rides in-list as the env's control
        # action so the whole gesture stays one step.
        box = _parse_box(params, "start_box")
        if box is None:
            return Outcome(INVALID_RESULT, executed=False)
        x, y = clamp_xy(box[0], box[1], width, height)
        if x is None or y is None:
            return Outcome(INVALID_RESULT, executed=False)
        actions.append({"mouse": {"move": [x, y]}})
        actions.append({"mouse": {"buttons": {"left_down": True}}})
        actions.append({"action": "wait", "time": LEFT_CLICK_HOLD_DURATION_SECONDS})
        actions.append({"mouse": {"buttons": {"left_up": True}}})
    else:
        return Outcome(INVALID_RESULT, executed=False)

    if hold:
        # Reference _wrap_with_hold_keys: keyDown*; sleep(0.1); action;
        # sleep(0.1); keyUp* -- the sleeps ride as env wait control actions.
        actions = (
            [{"keyboard": {"key_down": k}} for k in hold]
            + [{"action": "wait", "time": 0.1}]
            + actions
            + [{"action": "wait", "time": 0.1}]
            + [{"keyboard": {"key_up": k}} for k in reversed(hold)]
        )
    computer.step(actions)
    return Outcome("Action executed.")


def run(env_url: str, task: str) -> None:
    computer = Computer(env_url, timeout_sec=ENV_HTTP_TIMEOUT)
    history = GlmHistory()
    consecutive_parse_failures = 0
    terminated = False
    try:
        obs = computer.observe()
        width, height = image_size(obs["png"])

        for step in range(MAX_STEPS):
            messages = [build_user_message(task, history, obs["png"])]
            try:
                data = glm_request(messages)
            except ContextOverflow as exc:
                print(f"context overflow; dropping history images: {exc}", file=sys.stderr)
                history.drop_images()
                messages = [build_user_message(task, history, obs["png"])]
                data = glm_request(messages)
            content = response_text_of(data)
            parsed = parse_response(content)

            if parsed["action"] is None:
                # Reference predict(): a clean parse with no action call is a
                # recorded no-op step (the model may be asking a question),
                # NOT a resample; only parser exceptions retry upstream, and
                # the consecutive-failure counter resets here.
                print(
                    f"step {step}: no action parsed; recording no-op step; "
                    f"tail={content[-200:]!r}",
                    file=sys.stderr,
                )
                history.add(parsed["thought"] or content, "", obs["png"])
                history.memory = parsed["memory"]
                consecutive_parse_failures = 0
                obs = computer.observe()
                width, height = image_size(obs["png"])
                continue

            action = str(parsed["action"])
            print(f"step {step}: action {action[:300]}", file=sys.stderr)
            outcome = convert_and_execute(computer, action, width, height)
            history.add(parsed["thought"], action, obs["png"])
            history.memory = parsed["memory"]
            if not outcome.executed:
                # Mirrors the reference's parse-retry exhaustion accounting:
                # untranslatable actions count toward the consecutive cap and
                # convert into FAIL at the threshold (upstream returns the
                # FAIL command once consecutive_parse_failures hits the max).
                print(f"step {step}: {outcome.result_text}", file=sys.stderr)
                consecutive_parse_failures += 1
                if consecutive_parse_failures >= MAX_CONSECUTIVE_PARSE_FAILURES:
                    print(
                        "too many consecutive parse failures; marking task as FAIL",
                        file=sys.stderr,
                    )
                    computer.step([{"action_type": "FAIL"}])
                    terminated = True
                    break
            else:
                consecutive_parse_failures = 0
            if outcome.terminal:
                terminated = True
                break

            obs = computer.observe()
            width, height = image_size(obs["png"])

        if not terminated:
            # Reference: exceeding max_trajectory_length converts the step
            # into the FAIL command (glm_eval_agent.py predict()).
            print("max steps reached; marking task as FAIL", file=sys.stderr)
            computer.step([{"action_type": "FAIL"}])
    except Exception as exc:
        print(f"agent failed: {exc!r}", file=sys.stderr, flush=True)
    finally:
        try:
            computer.done()
        except Exception as exc:
            print(f"computer.done failed: {exc!r}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
