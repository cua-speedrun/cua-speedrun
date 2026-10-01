"""Qwen3.5 OSWorld agent.

Contract: python agent.py <env_url> <task_description>

This adapts OSWorld-V2's qwen35vl_agent.py to cua-speedrun's Computer
client. It keeps the Qwen3.5 XML function-call prompt, 1000x1000 relative
coordinate convention, resized screenshots, and long-history screenshot
folding, then translates parsed tool calls into gym-anything action dicts.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import sys
import time
from datetime import date, datetime
from typing import Any

import requests
from PIL import Image

from cua_speedrun.client import Computer

VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B")
MAX_STEPS = int(os.environ.get("CS_MAX_STEPS", "100"))
MAX_TOKENS = int(
    os.environ.get("CS_MAX_OUTPUT_TOKENS", os.environ.get("CS_MAX_TOKENS", "2048"))
)
TEMPERATURE = float(os.environ.get("CS_TEMPERATURE", "1.0"))
TOP_P = float(os.environ.get("CS_TOP_P", "0.95"))
HISTORY_N = int(os.environ.get("CS_HISTORY_N", "100"))
IMAGE_MAX = int(os.environ.get("CS_IMAGE_MAX", "20"))
FOLD_SIZE = int(os.environ.get("CS_FOLD_SIZE", "10"))
CONNECT_TIMEOUT = float(os.environ.get("OSWORLD_HTTP_CONNECT_TIMEOUT", "10"))
READ_TIMEOUT = float(os.environ.get("OSWORLD_HTTP_READ_TIMEOUT", "180"))
MAX_RETRY_TIMES = int(os.environ.get("OSWORLD_MAX_RETRY_TIMES", "5"))
OBSERVE_IMAGE_RETRIES = int(os.environ.get("OSWORLD_OBSERVE_IMAGE_RETRIES", "3"))

GRID_MAX = 999.0
SCROLL_STEP_LIMIT = 10
COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."


class ContextLengthError(RuntimeError):
    """Raised when vLLM rejects a prompt for exceeding the served context."""


def _format_prompt_date(value: Any = None) -> str:
    if value is None:
        return datetime.today().strftime("%A, %B %d, %Y")
    if isinstance(value, datetime):
        return value.strftime("%A, %B %d, %Y")
    if isinstance(value, date):
        return value.strftime("%A, %B %d, %Y")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return datetime.today().strftime("%A, %B %d, %Y")
        for parser in (datetime.fromisoformat,):
            try:
                return parser(text).strftime("%A, %B %d, %Y")
            except ValueError:
                pass
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt).strftime("%A, %B %d, %Y")
            except ValueError:
                pass
        return text
    return str(value)


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def smart_resize(
    *,
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 4 * 32 * 32,
    max_pixels: int = 16 * 16 * 4 * 12800,
) -> tuple[int, int]:
    """Local equivalent of Qwen-VL's smart_resize helper."""
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be >= factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {height}/{width}")

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return int(h_bar), int(w_bar)


def process_image(image_bytes: bytes) -> tuple[str, int, int, int, int]:
    image = Image.open(io.BytesIO(image_bytes))
    original_width, original_height = image.size
    resized_height, resized_width = smart_resize(
        height=original_height,
        width=original_width,
        factor=32,
        max_pixels=16 * 16 * 4 * 12800,
    )
    image = image.resize((resized_width, resized_height))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return (
        base64.b64encode(buffer.getvalue()).decode("utf-8"),
        original_width,
        original_height,
        resized_width,
        resized_height,
    )


def observe_processed_image(computer: Computer) -> tuple[str, int, int, int, int]:
    last_exc: Exception | None = None
    attempts = max(1, OBSERVE_IMAGE_RETRIES)
    for attempt in range(1, attempts + 1):
        obs = computer.observe()
        try:
            return process_image(obs["png"])
        except (OSError, ValueError) as exc:
            last_exc = exc
            print(
                f"invalid screenshot from observe attempt {attempt}/{attempts}: {exc!r}",
                file=sys.stderr,
            )
            if attempt < attempts:
                time.sleep(0.5)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("observe did not return a processable screenshot")


def _py_string(text: Any) -> str:
    return json.dumps("" if text is None else str(text), ensure_ascii=False)


def _wrap_tool_response(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return (
        [{"type": "text", "text": "<tool_response>\n"}]
        + parts
        + [{"type": "text", "text": "\n</tool_response>"}]
    )


def _tools_def(processed_width: int, processed_height: int) -> dict[str, Any]:
    description_prompt = "\n".join(
        [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            "* The screen's resolution is 1000x1000.",
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ]
    )
    action_description_prompt = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Drag from `coordinate` to `coordinate2`.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `scroll`: Performs a vertical mouse-wheel scroll. Pass `pixels` as signed wheel steps: negative scrolls down, positive scrolls up, and the magnitude must be between 1 and 10.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question."""

    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": description_prompt,
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "description": action_description_prompt,
                        "enum": [
                            "key",
                            "type",
                            "mouse_move",
                            "left_click",
                            "left_click_drag",
                            "right_click",
                            "middle_click",
                            "double_click",
                            "triple_click",
                            "scroll",
                            "wait",
                            "terminate",
                            "answer",
                        ],
                    },
                    "keys": {"type": "array", "description": "Required only by `action=key`."},
                    "text": {
                        "type": "string",
                        "description": "Required by `action=type` and `action=answer`.",
                    },
                    "coordinate": {"type": "array", "description": "(x, y) coordinates."},
                    "coordinate2": {
                        "type": "array",
                        "description": "Drag-end (x, y) coordinates; required by `action=left_click_drag`.",
                    },
                    "pixels": {
                        "type": "number",
                        "description": "Signed wheel steps: negative scrolls down, positive scrolls up; use a magnitude from 1 to 10.",
                    },
                    "time": {"type": "number", "description": "Seconds to wait."},
                    "status": {
                        "type": "string",
                        "description": "Task status for terminate.",
                        "enum": ["success", "failure"],
                    },
                },
            },
        },
    }


def _system_prompt(processed_width: int, processed_height: int) -> str:
    return (
        "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        "<tools>\n"
        + json.dumps(_tools_def(processed_width, processed_height), ensure_ascii=False)
        + "\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\n"
        "value_1\n"
        "</parameter>\n"
        "<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\n"
        "that can span\n"
        "multiple lines\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
        f"- The current date is {_format_prompt_date()}.\n"
        f"- Collapsed screenshots appear as text: {COLLAPSED_SCREENSHOT_TEXT}\n"
        "</IMPORTANT>\n\n"
        "# Response format\n\n"
        "Response format for every step:\n"
        "1) Action: a short imperative describing what to do in the UI.\n"
        "2) A single <tool_call>...</tool_call> block.\n\n"
        "Rules:\n"
        "- Output exactly in the order: Action, <tool_call>.\n"
        "- Be brief: one sentence for Action.\n"
        "- Do not output anything else outside those parts.\n"
        "- If finishing, use action=terminate in the tool call."
    )


def _update_folding_state(
    total_screenshots: int,
    folded_prefix_k: int,
    image_max: int = IMAGE_MAX,
    fold_size: int = FOLD_SIZE,
) -> int:
    while (total_screenshots - folded_prefix_k) > image_max:
        folded_prefix_k += fold_size
    if folded_prefix_k > total_screenshots:
        folded_prefix_k = total_screenshots
    return folded_prefix_k


def _folded_prefix_for(total_screenshots: int, image_max: int, fold_size: int) -> int:
    return _update_folding_state(total_screenshots, 0, image_max=image_max, fold_size=fold_size)


def build_messages(
    task: str,
    screenshots: list[str],
    responses: list[str],
    action_history: list[str],
    processed_width: int,
    processed_height: int,
    history_n: int = HISTORY_N,
    image_max: int = IMAGE_MAX,
    fold_size: int = FOLD_SIZE,
) -> list[dict[str, Any]]:
    total_steps = len(screenshots)
    history_n = max(1, int(history_n))
    image_max = max(1, int(image_max))
    fold_size = max(1, int(fold_size))
    folded_prefix_k = _folded_prefix_for(total_steps, image_max=image_max, fold_size=fold_size)
    start_step = max(1, total_steps - history_n)
    previous_actions = [
        f"Step {i + 1}: {action_history[i]}"
        for i in range(0, min(start_step - 1, len(action_history)))
    ]
    previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"
    instruction_prompt = (
        "\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
        f"Instruction: {task}\n\n"
        "Previous actions:\n"
        f"{previous_actions_str}"
    )

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": _system_prompt(processed_width, processed_height)}],
        }
    ]

    for step_num in range(start_step, total_steps + 1):
        is_first_turn = step_num == start_step
        is_collapsed = step_num <= folded_prefix_k

        if is_collapsed:
            parts = [{"type": "text", "text": COLLAPSED_SCREENSHOT_TEXT}]
            if is_first_turn:
                user_content = [{"type": "text", "text": instruction_prompt}]
            else:
                user_content = _wrap_tool_response(parts)
        else:
            img_url = f"data:image/png;base64,{screenshots[step_num - 1]}"
            if is_first_turn:
                user_content = [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": instruction_prompt},
                ]
            else:
                user_content = _wrap_tool_response(
                    [{"type": "image_url", "image_url": {"url": img_url}}]
                )
        messages.append({"role": "user", "content": user_content})

        if step_num <= total_steps - 1 and (step_num - 1) < len(responses):
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": responses[step_num - 1]}],
                }
            )

    return messages


def _is_context_length_error(response: requests.Response) -> bool:
    if response.status_code != 400:
        return False
    text = response.text.lower()
    context_markers = (
        "context length",
        "maximum context",
        "maximum context length",
        "input length",
    )
    return any(marker in text for marker in context_markers)


def _response_excerpt(response: requests.Response, limit: int = 500) -> str:
    text = response.text.replace("\n", " ").strip()
    return text[:limit]


def ask(messages: list[dict[str, Any]]) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRY_TIMES + 1):
        try:
            resp = requests.post(
                f"{VLLM_URL}/v1/chat/completions",
                json=payload,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            if _is_context_length_error(resp):
                raise ContextLengthError(_response_excerpt(resp))
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if content is None:
                return ""
            if isinstance(content, list):
                return "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return str(content)
        except requests.RequestException as exc:
            last_exc = exc
            detail = ""
            if getattr(exc, "response", None) is not None:
                detail = f" body={_response_excerpt(exc.response)}"
            print(
                f"model call failed attempt {attempt}/{MAX_RETRY_TIMES}: {exc!r}{detail}",
                file=sys.stderr,
            )
            time.sleep(min(5.0 * attempt, 30.0))
    if last_exc is not None:
        raise last_exc
    return ""


def _parse_json_tool_call(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and parsed.get("name") == "computer_use":
        args = parsed.get("arguments", {})
        return args if isinstance(args, dict) else None
    if isinstance(parsed, dict) and parsed.get("action"):
        return parsed
    return None


def _parse_xml_tool_call(xml_content: str) -> dict[str, Any] | None:
    json_call = _parse_json_tool_call(xml_content)
    if json_call is not None:
        return json_call

    params: dict[str, Any] = {}
    func_match = re.search(r"<function=([^>]+)>", xml_content)
    if not func_match or func_match.group(1) != "computer_use":
        return None

    for match in re.finditer(
        r"<parameter=([^>]+)>(.*?)</parameter>",
        xml_content,
        re.DOTALL,
    ):
        name = match.group(1)
        value = match.group(2)
        if name == "text":
            if value.startswith("\r\n"):
                value = value[2:]
            elif value.startswith("\n"):
                value = value[1:]
            if value.endswith("\r\n"):
                value = value[:-2]
            elif value.endswith("\n"):
                value = value[:-1]
        else:
            value = value.strip()
        if value.startswith("[") or value.startswith("{"):
            try:
                params[name] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        params[name] = value
    return params


# Mouse-family actions where a `keys` parameter means "hold these modifiers
# while performing the action" rather than a chord of its own.
_MOUSE_ACTIONS = {
    "left_click", "click", "right_click", "middle_click", "double_click",
    "triple_click", "left_click_drag", "drag", "scroll", "mouse_move",
}


def _parse_keys(raw_keys: Any) -> list[str]:
    if isinstance(raw_keys, str):
        try:
            raw_keys = json.loads(raw_keys)
        except Exception:
            raw_keys = [raw_keys]
    if isinstance(raw_keys, list):
        return [str(key).strip() for key in raw_keys if str(key).strip()]
    if raw_keys is None:
        return []
    return [str(raw_keys).strip()]


def _parse_coordinate(raw_coord: Any) -> tuple[float, float] | None:
    if isinstance(raw_coord, str):
        try:
            raw_coord = json.loads(raw_coord)
        except Exception:
            return None
    if isinstance(raw_coord, list) and len(raw_coord) >= 2:
        try:
            return float(raw_coord[0]), float(raw_coord[1])
        except Exception:
            return None
    return None


def _scale_coordinate(
    coord: tuple[float, float],
    original_width: int,
    original_height: int,
) -> list[int]:
    x, y = coord
    return [int(x * original_width / GRID_MAX), int(y * original_height / GRID_MAX)]


def parse_response(
    response: str,
    original_width: int,
    original_height: int,
) -> dict[str, Any]:
    if not response or not response.strip():
        return {"actions": [], "conclusion": "empty response", "is_terminal": False}
    if "</think>" in response:
        response = response.split("</think>", 1)[1]

    low_level_instruction = ""
    for line in response.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("action:"):
            low_level_instruction = stripped.split(":", 1)[-1].strip()
            break

    actions: list[dict[str, Any]] = []
    is_terminal = False
    wait_time: float | None = None

    def scaled(params: dict[str, Any], key: str = "coordinate") -> list[int] | None:
        coord = _parse_coordinate(params.get(key))
        if coord is None:
            return None
        return _scale_coordinate(coord, original_width, original_height)

    def process_params(params: dict[str, Any]) -> None:
        nonlocal is_terminal, wait_time
        action = str(params.get("action", "")).strip()
        if not action:
            return

        # Qwen emits `keys` on mouse actions to mean "hold these while
        # performing the action" (shift+click to extend a selection,
        # ctrl+scroll to zoom). Emit an explicit hold around the mouse
        # action; the plain `key` action keeps `keys` as the chord itself.
        mouse_hold = (
            _parse_keys(params.get("keys", []))
            if action in _MOUSE_ACTIONS
            else []
        )
        if mouse_hold:
            actions.append({"keyboard": {"keys_down": mouse_hold}})
        hold_marker = len(actions)

        if action == "key":
            keys = _parse_keys(params.get("keys", []))
            if keys:
                actions.append({"keyboard": {"keys": keys}})
        elif action == "type":
            actions.append({"keyboard": {"text": str(params.get("text", ""))}})
        elif action == "mouse_move":
            point = scaled(params)
            if point:
                actions.append({"mouse": {"move": point}})
        elif action in {"left_click", "click"}:
            point = scaled(params)
            if point:
                actions.append({"mouse": {"left_click": point}})
            else:
                actions.append({"mouse": {"buttons": {"left_down": True, "left_up": True}}})
        elif action == "right_click":
            point = scaled(params)
            if point:
                actions.append({"mouse": {"right_click": point}})
            else:
                actions.append({"mouse": {"buttons": {"right_down": True, "right_up": True}}})
        elif action == "middle_click":
            point = scaled(params)
            if point:
                actions.append({"mouse": {"middle_click": point}})
        elif action == "double_click":
            point = scaled(params)
            if point:
                actions.append({"mouse": {"double_click": point}})
        elif action == "triple_click":
            point = scaled(params)
            if point:
                actions.append({"mouse": {"triple_click": point}})
        elif action in {"left_click_drag", "drag"}:
            start = scaled(params)
            end = scaled(params, "coordinate2")
            if start and end:
                actions.append({"mouse": {"left_click_drag": [start, end]}})
            elif start:
                # Qwen computer_use emits left_click_drag with a single
                # coordinate: drag from the current cursor to that point. The
                # two-point branch above dropped these entirely.
                actions.append({"mouse": {"left_click_drag": [start]}})
        elif action == "scroll":
            try:
                requested_steps = int(float(params.get("pixels", 0)))
            except Exception:
                requested_steps = 0
            bounded_steps = max(
                -SCROLL_STEP_LIMIT,
                min(SCROLL_STEP_LIMIT, requested_steps),
            )
            point = scaled(params)
            if point:
                actions.append({"mouse": {"move": point}})
            if bounded_steps:
                # Qwen uses negative=down; Computer's canonical action uses
                # positive=down and lets each runner perform its own mapping.
                actions.append({"mouse": {"scroll": -bounded_steps}})
        elif action == "wait":
            try:
                wait_time = float(params.get("time", 1.0))
            except Exception:
                wait_time = 1.0
        elif action in {"terminate", "answer"}:
            if (
                action == "terminate"
                and str(params.get("status", "success")).strip().lower() == "failure"
            ):
                actions.append({"action_type": "FAIL"})
            is_terminal = True

        if mouse_hold:
            if len(actions) == hold_marker:
                # The action produced nothing to hold around; drop the press
                # so a bare modifier tap never reaches the environment.
                actions.pop()
            else:
                actions.append(
                    {"keyboard": {"keys_up": list(reversed(mouse_hold))}}
                )

    for tool_call_match in re.finditer(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL):
        params = _parse_xml_tool_call(tool_call_match.group(1))
        if params:
            process_params(params)

    if not actions and not is_terminal:
        match = re.search(r"(\{\"name\"\s*:\s*\"computer_use\".*\})", response, re.DOTALL)
        if match:
            params = _parse_json_tool_call(match.group(1))
            if params:
                process_params(params)

    parse_diagnostic = None
    if not actions and not is_terminal and wait_time is None:
        parse_diagnostic = "no executable action parsed from model response"

    if not low_level_instruction:
        if is_terminal:
            low_level_instruction = "Task completed"
        elif wait_time is not None:
            low_level_instruction = "Waiting"
        elif actions:
            low_level_instruction = "Performing action"
        else:
            low_level_instruction = "cannot parse; waiting"
            wait_time = 1.0

    return {
        "actions": actions,
        "conclusion": low_level_instruction,
        "is_terminal": is_terminal,
        "wait_time": wait_time,
        "parse_diagnostic": parse_diagnostic,
    }


def _context_variants() -> list[tuple[int, int, int]]:
    candidates = [
        (HISTORY_N, IMAGE_MAX, FOLD_SIZE),
        (min(HISTORY_N, 60), min(IMAGE_MAX, 12), min(FOLD_SIZE, 6)),
        (min(HISTORY_N, 24), min(IMAGE_MAX, 8), min(FOLD_SIZE, 4)),
        (min(HISTORY_N, 8), min(IMAGE_MAX, 4), min(FOLD_SIZE, 2)),
    ]
    variants: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for history_n, image_max, fold_size in candidates:
        variant = (max(1, history_n), max(1, image_max), max(1, fold_size))
        if variant not in seen:
            variants.append(variant)
            seen.add(variant)
    return variants


def _is_environment_conflict(exc: requests.HTTPError) -> bool:
    response = getattr(exc, "response", None)
    return response is not None and response.status_code == 409


def run(env_url: str, task: str) -> None:
    print(f"using vllm url: {VLLM_URL}", flush=True)
    computer = Computer(env_url)
    screenshots: list[str] = []
    responses: list[str] = []
    action_history: list[str] = []
    model_error: Exception | None = None

    for step in range(MAX_STEPS):
        try:
            (
                screenshot_b64,
                original_width,
                original_height,
                processed_width,
                processed_height,
            ) = observe_processed_image(computer)
        except requests.HTTPError as exc:
            if _is_environment_conflict(exc):
                print("environment returned 409 on observe; task is already closed", file=sys.stderr)
                break
            raise
        screenshots.append(screenshot_b64)

        reply = ""
        last_context_exc: ContextLengthError | None = None
        try:
            for history_n, image_max, fold_size in _context_variants():
                messages = build_messages(
                    task,
                    screenshots,
                    responses,
                    action_history,
                    processed_width,
                    processed_height,
                    history_n=history_n,
                    image_max=image_max,
                    fold_size=fold_size,
                )
                try:
                    reply = ask(messages)
                    if (history_n, image_max, fold_size) != (HISTORY_N, IMAGE_MAX, FOLD_SIZE):
                        print(
                            "context retry succeeded with "
                            f"history_n={history_n}, image_max={image_max}, fold_size={fold_size}",
                            file=sys.stderr,
                        )
                    break
                except ContextLengthError as exc:
                    last_context_exc = exc
                    print(
                        "context length rejected with "
                        f"history_n={history_n}, image_max={image_max}, fold_size={fold_size}: {exc}",
                        file=sys.stderr,
                    )
            else:
                if last_context_exc is not None:
                    raise last_context_exc
                raise RuntimeError("no context variants available")
        except Exception as exc:
            print(f"model call failed: {exc!r}", file=sys.stderr)
            model_error = exc
            break

        responses.append(reply or "")
        print(
            f"step {step} raw ({len(reply) if reply else 0} chars): {(reply or '')!r}",
            file=sys.stderr,
        )

        parsed = parse_response(reply or "", original_width, original_height)
        action_history.append(parsed["conclusion"])
        if parsed.get("parse_diagnostic"):
            print(f"step {step} tool-call diagnostic: {parsed['parse_diagnostic']}", file=sys.stderr)
        print(
            f"step {step}: {parsed['conclusion']} -> {parsed['actions']}",
            file=sys.stderr,
        )

        if parsed["is_terminal"]:
            if parsed["actions"]:
                computer.step(parsed["actions"])
            break

        try:
            if parsed.get("wait_time") is not None:
                computer.wait(float(parsed["wait_time"]))
            elif parsed["actions"]:
                computer.step(parsed["actions"])
            else:
                computer.wait(1.0)
        except requests.HTTPError as exc:
            if _is_environment_conflict(exc):
                print("environment returned 409; task is already closed", file=sys.stderr)
                break
            raise

    try:
        computer.done()
    except requests.HTTPError as exc:
        if _is_environment_conflict(exc):
            print("environment returned 409 on done; task is already closed", file=sys.stderr)
        else:
            raise
    if model_error is not None:
        raise RuntimeError("Qwen3.5 model call failed") from model_error


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
