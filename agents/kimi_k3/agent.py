"""Kimi K3 GUI agent for OSWorld-style desktop tasks via OpenRouter.

Contract: python agent.py <env_url> <task_description>

The prompting, screenshot history, response parser, coordinate projection,
retry counts, and terminal conventions are ported from OSWorld's
``mm_agents/kimi/kimi_agent.py`` at commit
091f5ef1d5544bc74953c77875d5feb5bed30108.  Kimi K3-specific inference
defaults follow Moonshot's K3 documentation: maximum reasoning effort,
temperature 1.0, and top_p 1.0 for agentic workloads.

Necessary harness adaptations are deliberately narrow:

* requests go to OpenRouter with model ``moonshotai/kimi-k3``;
* recent assistant messages preserve OpenRouter reasoning fields for K3's
  preserved-thinking history requirement;
* the real guest password (``password``) replaces the upstream image's
  password in the otherwise verbatim system prompt; and
* K3's native tool calls replace the older K2.5 text/code-block action format;
* model-written pyautogui is parsed as literal AST and translated into the
  Computer client's structured GUI actions. It is never executed as Python,
  and unsupported code is resampled just like an upstream parse failure.
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
from typing import Any

import requests

from cua_speedrun.client import Computer


MODEL = os.environ.get("KIMI_MODEL", "moonshotai/kimi-k3")
BASE_URL = os.environ.get("KIMI_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
API_URL = f"{BASE_URL}/chat/completions"
MAX_STEPS = int(os.environ.get("KIMI_MAX_STEPS") or os.environ.get("CS_MAX_STEPS", "100"))
MAX_IMAGE_HISTORY_LENGTH = int(os.environ.get("KIMI_MAX_IMAGE_HISTORY_LENGTH", "3"))
MAX_TOKENS = int(os.environ.get("KIMI_MAX_TOKENS", "16384"))
TEMPERATURE = float(os.environ.get("KIMI_TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("KIMI_TOP_P", "1.0"))
REASONING_EFFORT = os.environ.get("KIMI_REASONING_EFFORT", "max")
HTTP_TIMEOUT = float(os.environ.get("KIMI_HTTP_TIMEOUT", "1200"))
HTTP_RETRIES = int(os.environ.get("KIMI_HTTP_RETRIES", "5"))
PREDICT_RETRIES = int(os.environ.get("KIMI_PREDICT_RETRIES", "5"))
RETRY_SLEEP = float(os.environ.get("KIMI_RETRY_SLEEP", "5"))
ENV_HTTP_TIMEOUT = float(os.environ.get("KIMI_ENV_HTTP_TIMEOUT", "600"))
WAIT_SECONDS = float(os.environ.get("KIMI_WAIT_SECONDS", "20"))
CLIENT_PASSWORD = os.environ.get("KIMI_CLIENT_PASSWORD", "password")
COST_SNAPSHOT_PREFIX = "__CUA_SPEEDRUN_COST_V1__"

COMPUTER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "computer_action",
            "description": (
                "Execute one or more pyautogui statements to interact with the "
                "visible desktop. Use literal arguments only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Brief description of the GUI action",
                    },
                    "code": {
                        "type": "string",
                        "description": (
                            "Sequential Python statements using only pyautogui "
                            "and time.sleep"
                        ),
                    },
                },
                "required": ["description", "code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_wait",
            "description": "Wait for the desktop to finish loading or processing.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_terminate",
            "description": "Terminate the task only when it is complete or impossible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["success", "failure"],
                    },
                    "answer": {"type": "string"},
                },
                "required": ["status"],
                "additionalProperties": False,
            },
        },
    },
]


INSTRUCTION_TEMPLATE = "# Task Instruction:\n{instruction}\n\nPlease generate the next move according to the screenshot, task instruction and previous steps (if provided).\n"
STEP_TEMPLATE = "# Step {step_num}:\n"

# Verbatim OSWorld Kimi thinking prompt except for substituting the password
# that actually unlocks cua-speedrun's canonical OSWorld guest.
SYSTEM_PROMPT_THINKING = """
You are a GUI agent. You are given an instruction, a screenshot of the screen and your previous interactions with the computer. You need to perform a series of actions to complete the task. The passoword of the computer is {password}.

For each step, provide your response in this format:
{thought}
## Action:
{action}
## Code:
{code}

In the code section, the code should be either pyautogui code or one of the following functions wrapped in the code block:
- {"name": "computer.wait", "description": "Make the computer wait for 20 seconds for installation, running code, etc.", "parameters": {"type": "object", "properties": {}, "required": []}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}, "answer": {"type": "string", "description": "The answer of the task"}}, "required": ["status"]}}
""".strip()

THOUGHT_HISTORY_TEMPLATE_THINKING = "◁think▷{thought}◁/think▷## Action:\n{action}\n"


class ParseError(ValueError):
    """The model response did not satisfy the upstream action convention."""


def _usage_numbers(usage: dict[str, Any]) -> dict[str, int | float]:
    numbers: dict[str, int | float] = {}
    for key, value in usage.items():
        if key == "cost" or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0:
            numbers[key] = value
        elif isinstance(value, dict):
            for detail_key, detail_value in value.items():
                if isinstance(detail_value, bool):
                    continue
                if (
                    isinstance(detail_value, (int, float))
                    and math.isfinite(float(detail_value))
                    and detail_value >= 0
                ):
                    numbers[f"{key}.{detail_key}"] = detail_value
    return numbers


class CostReporter:
    """Emit the latest cumulative OpenRouter cost and usage for this task."""

    def __init__(self) -> None:
        self.cost_usd = 0.0
        self.usage: dict[str, int | float] = {}

    def record(self, usage: dict[str, Any] | None) -> None:
        if not isinstance(usage, dict):
            return
        cost = usage.get("cost")
        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or cost < 0
        ):
            return
        self.cost_usd += float(cost)
        for key, value in _usage_numbers(usage).items():
            self.usage[key] = self.usage.get(key, 0) + value
        print(
            COST_SNAPSHOT_PREFIX
            + json.dumps({"cost_usd": self.cost_usd, "usage": self.usage}),
            flush=True,
        )


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is required by the kimi_k3 template")
    return key


def encode_image(image_content: bytes) -> str:
    return base64.b64encode(image_content).decode("ascii")


def image_size(png: bytes) -> tuple[int, int]:
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
        raise ValueError("observation is not a valid PNG with an IHDR chunk")
    return struct.unpack(">II", png[16:24])


def _reasoning_text(message: dict[str, Any]) -> str:
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str):
        return reasoning.strip()
    details = message.get("reasoning_details")
    if isinstance(details, list):
        parts: list[str] = []
        for detail in details:
            if not isinstance(detail, dict):
                continue
            for key in ("text", "summary", "content"):
                value = detail.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
                    break
        return "\n".join(parts)
    return ""


def _assistant_message_for_history(message: dict[str, Any]) -> dict[str, Any]:
    """Keep the complete assistant payload fields OpenRouter may require."""
    kept = {"role": "assistant", "content": message.get("content") or ""}
    for key in (
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "tool_calls",
        "refusal",
    ):
        if message.get(key) is not None:
            kept[key] = message[key]
    return kept


def _tool_results_for_history(
    message: dict[str, Any],
    content: str = "The action was executed. Inspect the next screenshot for the result.",
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict) or not call.get("id"):
            continue
        results.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": content,
            }
        )
    return results


def build_messages(
    instruction: str,
    screenshot: bytes,
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the upstream Kimi history with K3 reasoning preservation.

    OSWorld compacts older turns into one textual assistant message and pairs
    only its recent turns with screenshots.  Recent assistant responses are
    kept intact here so K3 receives its prior reasoning fields as required.
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_THINKING.replace("{password}", CLIENT_PASSWORD),
        }
    ]
    older_text: list[str] = []
    history_count = len(history)
    for index, item in enumerate(history):
        history_text = STEP_TEMPLATE.format(step_num=index + 1) + THOUGHT_HISTORY_TEMPLATE_THINKING.format(
            thought=item.get("thought", ""),
            action=item.get("action", ""),
        )
        # Preserve the upstream boundary exactly: with a length-three visual
        # history, the final two completed turns carry screenshots while the
        # boundary turn joins the compacted text block.
        if index > history_count - MAX_IMAGE_HISTORY_LENGTH:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + encode_image(item["screenshot"])
                            },
                        }
                    ],
                }
            )
            messages.append(_assistant_message_for_history(item["message"]))
            messages.extend(_tool_results_for_history(item["message"]))
        else:
            older_text.append(history_text)
            if index == history_count - MAX_IMAGE_HISTORY_LENGTH:
                messages.append({"role": "assistant", "content": "\n".join(older_text)})

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + encode_image(screenshot)},
                },
                {
                    "type": "text",
                    "text": INSTRUCTION_TEMPLATE.format(instruction=instruction),
                },
            ],
        }
    )
    return messages


def build_payload(messages: list[dict[str, Any]], temperature: float = TEMPERATURE) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": messages,
        "tools": COMPUTER_TOOLS,
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "max_tokens": MAX_TOKENS,
        "top_p": TOP_P,
        "temperature": temperature,
        "reasoning": {"effort": REASONING_EFFORT, "exclude": False},
        "provider": {
            "only": ["moonshotai/mxfp4"],
            "allow_fallbacks": False,
        },
    }


def kimi_request(
    payload: dict[str, Any],
    cost_reporter: CostReporter | None = None,
) -> dict[str, Any]:
    """Call OpenRouter with the template's bounded completion retry."""
    last_error = "Kimi request did not run"
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
    }
    for attempt in range(HTTP_RETRIES):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=HTTP_TIMEOUT,
            )
            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}: {response.text[:1000]}"
            else:
                data = response.json()
                if cost_reporter is not None:
                    cost_reporter.record(data.get("usage"))
                choice = (data.get("choices") or [{}])[0]
                if choice.get("finish_reason") in {"stop", "tool_calls"} and isinstance(
                    choice.get("message"), dict
                ):
                    return choice["message"]
                last_error = f"finish_reason={choice.get('finish_reason')!r}"
        except Exception as exc:
            last_error = repr(exc)
        print(
            f"Kimi request attempt {attempt + 1}/{HTTP_RETRIES} failed: {last_error}",
            file=sys.stderr,
            flush=True,
        )
        if attempt + 1 < HTTP_RETRIES:
            time.sleep(RETRY_SLEEP)
    raise RuntimeError(f"Kimi request failed after {HTTP_RETRIES} attempts: {last_error}")


def project_coordinate_to_absolute_scale(
    pyautogui_code_relative_coordinates: str,
    screen_width: int,
    screen_height: int,
    coordinate_type: str = "relative",
) -> str:
    """Verbatim coordinate rule from OSWorld's Kimi agent.

    ``coordinate_type`` is retained for source compatibility; upstream does
    not branch on it.  A pair where both values are <= 1 is relative, and any
    other pair is already absolute pixels.
    """
    del coordinate_type

    def projection(x: Any, y: Any) -> tuple[int, int]:
        if float(x) <= 1.0 and float(y) <= 1.0:
            return int(round(float(x) * screen_width)), int(round(float(y) * screen_height))
        return int(round(float(x))), int(round(float(y)))

    pattern = r"(pyautogui\.\w+\([^\)]*\))"
    new_code = pyautogui_code_relative_coordinates
    function_parameters = {
        "click": ["x", "y", "clicks", "interval", "button", "duration", "pause"],
        "rightClick": ["x", "y", "duration", "tween", "pause"],
        "middleClick": ["x", "y", "duration", "tween", "pause"],
        "doubleClick": ["x", "y", "interval", "button", "duration", "pause"],
        "tripleClick": ["x", "y", "interval", "button", "duration", "pause"],
        "moveTo": ["x", "y", "duration", "tween", "pause"],
        "dragTo": ["x", "y", "duration", "button", "mouseDownUp", "pause"],
    }
    for full_call in re.findall(pattern, pyautogui_code_relative_coordinates):
        match = re.match(r"(pyautogui\.\w+)\((.*)\)", full_call, re.DOTALL)
        if not match:
            continue
        func_name, args_str = match.groups()
        try:
            parsed = ast.parse(f"func({args_str})").body[0].value
            positional = parsed.args
            keywords = parsed.keywords
        except SyntaxError:
            return pyautogui_code_relative_coordinates
        param_names = function_parameters.get(func_name.split(".")[-1], [])
        args: dict[str, Any] = {}
        try:
            for index, arg in enumerate(positional):
                if index < len(param_names):
                    args[param_names[index]] = ast.literal_eval(arg)
            for keyword in keywords:
                if keyword.arg:
                    args[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError):
            return pyautogui_code_relative_coordinates
        if "x" not in args or "y" not in args:
            continue
        try:
            args["x"], args["y"] = projection(args["x"], args["y"])
        except (TypeError, ValueError):
            continue
        reconstructed: list[str] = []
        for name in param_names:
            if name not in args:
                break
            reconstructed.append(repr(args[name]))
        used = set(param_names[: len(reconstructed)])
        for keyword in keywords:
            if keyword.arg and keyword.arg not in used:
                reconstructed.append(f"{keyword.arg}={args[keyword.arg]!r}")
        new_code = new_code.replace(full_call, f"{func_name}({', '.join(reconstructed)})")
    return new_code


def parse_response(message: dict[str, Any], screen_size: tuple[int, int]) -> dict[str, str]:
    """Port OSWorld's thinking-mode response parser."""
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        if len(tool_calls) != 1 or not isinstance(tool_calls[0], dict):
            raise ParseError("response must contain exactly one computer tool call")
        function = tool_calls[0].get("function")
        if not isinstance(function, dict):
            raise ParseError("tool call has no function")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ParseError("tool call arguments are not valid JSON") from exc
        if not isinstance(arguments, dict):
            raise ParseError("tool call arguments are not an object")

        name = function.get("name")
        if name == "computer_action":
            action = arguments.get("description")
            original_code = arguments.get("code")
            if not isinstance(action, str) or not action.strip():
                raise ParseError("computer_action has no description")
            if not isinstance(original_code, str) or not original_code.strip():
                raise ParseError("computer_action has no code")
            code = project_coordinate_to_absolute_scale(
                original_code.strip(),
                screen_width=screen_size[0],
                screen_height=screen_size[1],
            )
        elif name == "computer_wait":
            action, original_code, code = "Wait", "computer.wait()", "WAIT"
        elif name == "computer_terminate":
            status = arguments.get("status")
            if status not in {"success", "failure"}:
                raise ParseError("computer_terminate has no valid status")
            action = str(arguments.get("answer") or f"Terminate with {status}")
            original_code = json.dumps(arguments)
            code = "DONE" if status == "success" else "FAIL"
        else:
            raise ParseError(f"unsupported computer tool: {name!r}")
        return {
            "thought": _reasoning_text(message),
            "action": action.strip(),
            "original_code": original_code,
            "code": code,
        }

    content = message.get("content")
    if not isinstance(content, str):
        raise ParseError(f"response content is not text: {content!r}")
    input_string = content.lstrip()
    thought = _reasoning_text(message)
    action_marker = re.search(r"^##\s*Action\b", input_string, flags=re.MULTILINE)
    if action_marker:
        input_string = input_string[action_marker.start() :]
    action_match = re.search(
        r"^\s*##\s*Action\s*:?\s*[\n\r]+(.*?)(?=^\s*##|\Z)",
        input_string,
        re.DOTALL | re.MULTILINE,
    )
    action = action_match.group(1).strip() if action_match else ""
    code_blocks = re.findall(
        r"```(?:code|python)?\s*(.*?)\s*```",
        input_string,
        re.DOTALL | re.IGNORECASE,
    )
    if not code_blocks:
        raise ParseError(f"no code blocks found: {input_string}")
    original_code = code_blocks[-1].strip()
    lowered = original_code.lower()
    if "computer.wait" in lowered:
        code = "WAIT"
    elif "computer.terminate" in lowered:
        if "failure" in lowered or "fail" in lowered:
            code = "FAIL"
        elif "success" in lowered:
            code = "DONE"
        else:
            raise ParseError("computer.terminate has no success/failure status")
    else:
        code = project_coordinate_to_absolute_scale(
            original_code,
            screen_width=screen_size[0],
            screen_height=screen_size[1],
        )
    if not action or not code:
        raise ParseError("response is missing the Action or Code section")
    return {
        "thought": thought,
        "action": action,
        "original_code": original_code,
        "code": code,
    }


KEY_MAP = {
    "alt": "alt",
    "backspace": "BackSpace",
    "capslock": "capslock",
    "cmd": "ctrl",
    "command": "ctrl",
    "ctrl": "ctrl",
    "del": "Delete",
    "delete": "Delete",
    "down": "Down",
    "end": "End",
    "enter": "Return",
    "esc": "Escape",
    "escape": "Escape",
    "home": "Home",
    "insert": "insert",
    "left": "Left",
    "option": "alt",
    "pagedown": "pagedown",
    "pageup": "pageup",
    "return": "Return",
    "right": "Right",
    "shift": "shift",
    "space": "space",
    "super": "super",
    "tab": "Tab",
    "up": "Up",
    "win": "super",
}


def map_key(value: Any) -> str:
    text = str(value).strip()
    return KEY_MAP.get(text.lower(), text)


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError) as exc:
        raise ParseError("pyautogui arguments must be literals") from exc


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        if call.func.value.id in {"pyautogui", "time"}:
            return f"{call.func.value.id}.{call.func.attr}"
    raise ParseError("only pyautogui.* and time.sleep calls are supported")


def _arguments(call: ast.Call) -> tuple[list[Any], dict[str, Any]]:
    args = [_literal(arg) for arg in call.args]
    kwargs = {kw.arg: _literal(kw.value) for kw in call.keywords if kw.arg}
    return args, kwargs


def _value(args: list[Any], kwargs: dict[str, Any], index: int, name: str, default: Any = None) -> Any:
    return args[index] if index < len(args) else kwargs.get(name, default)


def _point(x: Any, y: Any, width: int, height: int) -> list[int]:
    if x is None or y is None:
        raise ParseError("both x and y coordinates are required")
    x_num, y_num = float(x), float(y)
    if x_num <= 1.0 and y_num <= 1.0:
        x_num, y_num = round(x_num * width), round(y_num * height)
    point = [int(round(x_num)), int(round(y_num))]
    return [min(max(point[0], 0), width - 1), min(max(point[1], 0), height - 1)]


def _type_actions(text: str) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if line:
            actions.append({"keyboard": {"text": line}})
        if index < len(lines) - 1:
            actions.append({"keyboard": {"keys": ["Return"]}})
    return actions


def translate_pyautogui(
    code: str,
    width: int,
    height: int,
    cursor: list[int],
) -> list[dict[str, Any]]:
    """Translate literal pyautogui statements without executing model code."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ParseError(f"invalid Python code: {exc.msg}") from exc
    actions: list[dict[str, Any]] = []

    def add_click(button: str, point: list[int] | None, clicks: int = 1, interval: float = 0) -> None:
        if clicks < 1:
            raise ParseError("clicks must be positive")
        for click_index in range(clicks):
            if point is not None:
                key = {"left": "left_click", "right": "right_click", "middle": "middle_click"}.get(button)
                if key is None:
                    raise ParseError(f"unsupported mouse button: {button!r}")
                actions.append({"mouse": {key: point}})
                cursor[:] = point
            else:
                field = {"left": "left", "right": "right", "middle": "middle"}.get(button)
                if field is None:
                    raise ParseError(f"unsupported mouse button: {button!r}")
                actions.append({"mouse": {"buttons": {f"{field}_down": True}}})
                actions.append({"mouse": {"buttons": {f"{field}_up": True}}})
            if interval and click_index + 1 < clicks:
                actions.append({"action": "wait", "time": float(interval)})

    for statement in tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            names = {alias.name for alias in statement.names}
            if names <= {"pyautogui", "time"}:
                continue
            raise ParseError("only pyautogui/time imports are accepted")
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ParseError("only sequential pyautogui calls are supported")
        call = statement.value
        name = _call_name(call)
        args, kwargs = _arguments(call)
        short = name.split(".", 1)[1]

        if name == "time.sleep" or short in {"sleep", "pause"}:
            seconds = float(_value(args, kwargs, 0, "seconds", 0.5))
            actions.append({"action": "wait", "time": max(seconds, 0.0)})
        elif short in {"write", "typewrite"}:
            actions.extend(_type_actions(str(_value(args, kwargs, 0, "message", ""))))
        elif short == "hotkey":
            keys = [map_key(value) for value in args]
            if not keys:
                raise ParseError("pyautogui.hotkey requires keys")
            actions.append({"keyboard": {"keys": keys}})
        elif short == "press":
            keys_value = _value(args, kwargs, 0, "keys")
            keys = list(keys_value) if isinstance(keys_value, (list, tuple)) else [keys_value]
            presses = int(_value(args, kwargs, 1, "presses", 1))
            for _ in range(presses):
                for key in keys:
                    actions.append({"keyboard": {"keys": [map_key(key)]}})
        elif short in {"keyDown", "keyUp"}:
            key = map_key(_value(args, kwargs, 0, "key"))
            field = "keys_down" if short == "keyDown" else "keys_up"
            actions.append({"keyboard": {field: [key]}})
        elif short in {"moveTo", "moveRel"}:
            x = _value(args, kwargs, 0, "x")
            y = _value(args, kwargs, 1, "y")
            if short == "moveRel":
                if len(cursor) != 2:
                    raise ParseError("moveRel requires a known cursor position")
                point = [cursor[0] + int(x), cursor[1] + int(y)]
                point = [min(max(point[0], 0), width - 1), min(max(point[1], 0), height - 1)]
            else:
                point = _point(x, y, width, height)
            actions.append({"mouse": {"move": point}})
            cursor[:] = point
        elif short in {"dragTo", "dragRel"}:
            button = str(_value(args, kwargs, 3, "button", "left")).lower()
            if button != "left":
                raise ParseError("the environment supports only left-button drag")
            x = _value(args, kwargs, 0, "x")
            y = _value(args, kwargs, 1, "y")
            if short == "dragRel":
                if len(cursor) != 2:
                    raise ParseError("dragRel requires a known cursor position")
                destination = [cursor[0] + int(x), cursor[1] + int(y)]
                destination = [
                    min(max(destination[0], 0), width - 1),
                    min(max(destination[1], 0), height - 1),
                ]
            else:
                destination = _point(x, y, width, height)
            points = [list(cursor), destination] if len(cursor) == 2 else [destination]
            actions.append({"mouse": {"left_click_drag": points}})
            cursor[:] = destination
        elif short in {"scroll", "hscroll"}:
            amount = int(_value(args, kwargs, 0, "clicks", 0))
            x = _value(args, kwargs, 1, "x")
            y = _value(args, kwargs, 2, "y")
            if x is not None or y is not None:
                point = _point(x, y, width, height)
                actions.append({"mouse": {"move": point}})
                cursor[:] = point
            if short == "hscroll":
                actions.extend(
                    [
                        {"keyboard": {"keys_down": ["shift"]}},
                        {"mouse": {"scroll": amount}},
                        {"keyboard": {"keys_up": ["shift"]}},
                    ]
                )
            else:
                # pyautogui positive is up; the environment's positive is down.
                actions.append({"mouse": {"scroll": -amount}})
        elif short in {"mouseDown", "mouseUp"}:
            x = _value(args, kwargs, 0, "x")
            y = _value(args, kwargs, 1, "y")
            button = str(_value(args, kwargs, 2, "button", "left")).lower()
            if x is not None or y is not None:
                point = _point(x, y, width, height)
                actions.append({"mouse": {"move": point}})
                cursor[:] = point
            if button not in {"left", "right", "middle"}:
                raise ParseError(f"unsupported mouse button: {button!r}")
            state = "down" if short == "mouseDown" else "up"
            actions.append({"mouse": {"buttons": {f"{button}_{state}": True}}})
        elif short in {"click", "rightClick", "middleClick", "doubleClick", "tripleClick"}:
            x = _value(args, kwargs, 0, "x")
            y = _value(args, kwargs, 1, "y")
            point = None if x is None and y is None else _point(x, y, width, height)
            defaults = {
                "rightClick": ("right", 1),
                "middleClick": ("middle", 1),
                "doubleClick": (str(_value(args, kwargs, 3, "button", "left")).lower(), 2),
                "tripleClick": (str(_value(args, kwargs, 3, "button", "left")).lower(), 3),
            }
            if short == "click":
                button = str(_value(args, kwargs, 4, "button", "left")).lower()
                clicks = int(_value(args, kwargs, 2, "clicks", 1))
                interval = float(_value(args, kwargs, 3, "interval", 0))
            else:
                button, clicks = defaults[short]
                interval = float(_value(args, kwargs, 2, "interval", 0))
            add_click(button, point, clicks, interval)
        else:
            raise ParseError(f"unsupported pyautogui action: {short}")

    if not actions:
        raise ParseError("pyautogui code produced no actions")
    return actions


def execute_code(
    computer: Computer,
    code: str,
    width: int,
    height: int,
    cursor: list[int],
) -> str | None:
    if code == "WAIT":
        computer.wait(WAIT_SECONDS)
        return None
    if code == "DONE":
        return "DONE"
    if code == "FAIL":
        computer.step([{"action_type": "FAIL"}])
        return "FAIL"
    computer.step(translate_pyautogui(code, width, height, cursor))
    return None


def run(env_url: str, task: str) -> None:
    computer = Computer(env_url, timeout_sec=ENV_HTTP_TIMEOUT)
    cost_reporter = CostReporter()
    history: list[dict[str, Any]] = []
    cursor: list[int] = []
    try:
        observation = computer.observe()
        for step in range(MAX_STEPS):
            width, height = image_size(observation["png"])
            messages = build_messages(task, observation["png"], history)
            message: dict[str, Any] | None = None
            parsed: dict[str, str] | None = None

            for attempt in range(PREDICT_RETRIES):
                candidate: dict[str, Any] | None = None
                try:
                    request_temperature = TEMPERATURE if attempt == 0 else max(0.2, TEMPERATURE)
                    candidate = kimi_request(
                        build_payload(messages, request_temperature),
                        cost_reporter,
                    )
                    print(
                        f"step {step} Kimi response attempt {attempt + 1}: "
                        + json.dumps(candidate, ensure_ascii=False),
                        file=sys.stderr,
                        flush=True,
                    )
                    candidate_parsed = parse_response(candidate, (width, height))
                    if candidate_parsed["code"] not in {"WAIT", "DONE", "FAIL"}:
                        # Validate before accepting the turn.  Unsupported model
                        # code is equivalent to the upstream parser rejecting it.
                        translate_pyautogui(candidate_parsed["code"], width, height, list(cursor))
                    message, parsed = candidate, candidate_parsed
                    break
                except Exception as exc:
                    print(
                        f"step {step} prediction attempt {attempt + 1}/{PREDICT_RETRIES} failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if candidate is not None and candidate.get("tool_calls"):
                        messages.append(_assistant_message_for_history(candidate))
                        messages.extend(
                            _tool_results_for_history(
                                candidate,
                                content=(
                                    "Invalid tool call arguments. Follow the computer tool "
                                    "semantics: use only direct sequential pyautogui or "
                                    "time.sleep calls with literal arguments. Do not use "
                                    "variables, assignments, loops, comprehensions, or "
                                    "function definitions."
                                ),
                            )
                        )

            if message is None or parsed is None:
                print(f"step {step}: maximum prediction retries reached -> FAIL", file=sys.stderr)
                computer.step([{"action_type": "FAIL"}])
                break

            history.append(
                {
                    "screenshot": observation["png"],
                    "message": message,
                    "thought": parsed["thought"],
                    "action": parsed["action"],
                }
            )

            code = parsed["code"]
            if len(history) >= MAX_STEPS and code not in {"DONE", "FAIL"}:
                print(f"reached maximum steps {MAX_STEPS}; forcing FAIL", file=sys.stderr)
                code = "FAIL"
            terminal = execute_code(computer, code, width, height, cursor)
            if terminal is not None:
                print(f"step {step}: terminal token {terminal}", file=sys.stderr)
                break
            observation = computer.observe()
    except Exception as exc:
        print(f"agent failed: {exc!r}", file=sys.stderr, flush=True)
    finally:
        try:
            computer.done()
        except Exception as exc:
            print(f"computer.done failed: {exc!r}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
