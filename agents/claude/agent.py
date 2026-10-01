"""Claude computer-use agent for OSWorld-style desktop tasks.

Contract: python agent.py <env_url> <task_description>

This is a cua-speedrun port of gym-anything's ``ClaudeFixedAgent``. It uses
Anthropic's native computer-use tool, returns a screenshot after every tool
action, rescales coordinates from the image sent to Claude, and bounds old
screenshot history while preserving the Messages API tool-result contract.
"""

from __future__ import annotations

import base64
import json
import math
import os
import sys
import time
from io import BytesIO
from typing import Any

import requests
from PIL import Image

from cua_speedrun.client import Computer

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
API_URL = os.environ.get(
    "ANTHROPIC_MESSAGES_URL", "https://api.anthropic.com/v1/messages"
)
ANTHROPIC_VERSION = "2023-06-01"
MAX_STEPS = int(os.environ.get("CS_MAX_STEPS", "100"))
MAX_OUTPUT_TOKENS = int(os.environ.get("CLAUDE_MAX_OUTPUT_TOKENS", "16384"))
EFFORT = os.environ.get("CLAUDE_EFFORT", "medium").strip().lower()
REQUEST_TIMEOUT = float(os.environ.get("CLAUDE_REQUEST_TIMEOUT", "240"))
COMPUTER_TIMEOUT = float(os.environ.get("CLAUDE_COMPUTER_TIMEOUT", "300"))
MAX_RETRIES = int(os.environ.get("CLAUDE_MAX_RETRIES", "5"))
IMAGES_TO_KEEP = int(os.environ.get("CLAUDE_IMAGES_TO_KEEP", "7"))
MIN_REMOVAL_THRESHOLD = int(os.environ.get("CLAUDE_IMAGE_REMOVAL_THRESHOLD", "7"))
MAX_IMAGE_WIDTH = int(os.environ.get("CLAUDE_IMAGE_WIDTH", "1280"))
MAX_IMAGE_HEIGHT = int(os.environ.get("CLAUDE_IMAGE_HEIGHT", "720"))

_NEW_COMPUTER_MODELS = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
)

SYSTEM_PROMPT = """You control a sandboxed Ubuntu GNOME desktop for a computer-use benchmark task.
Use the computer tool to complete the task in the visible desktop. Do not ask
the user for help. The account password is "password" if an application
asks for it. Applications may already be open; otherwise launch them from the
GNOME overview. Prefer the intended visible application for GUI, document,
browser, image, email, and editor tasks. Inspect the screenshot after every
action, allow loading interfaces time to settle, and scroll or zoom before
concluding that content is absent. Do not assume an action succeeded without
checking its result. When possible, emit a sequence of compatible computer
tool calls in one response. You can declare a task infeasible at any point,
including after attempting it. Carefully evaluate whether the task can be
completed with the current system, available applications, permissions, and
task requirements. If it cannot be completed because a required application
or dependency is unavailable and cannot be installed, permissions or system
limitations prevent it, requirements are contradictory or impossible, or
another fundamental barrier makes completion impossible, you MUST call the
infeasible tool. When the visible state fully satisfies the task, you MUST call
the complete tool. Never end the task with only a text response: use the
computer tool to continue working, the complete tool for success, or the
infeasible tool for a fundamental blocker. This is an isolated benchmark VM;
the task instruction authorizes in-scope actions."""


def api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("set ANTHROPIC_API_KEY for the Claude API")
    return key


def computer_tool_version(model: str = MODEL) -> tuple[str, str]:
    """Return the tool type and beta header supported by ``model``."""
    if any(tag in model for tag in _NEW_COMPUTER_MODELS):
        defaults = ("computer_20251124", "computer-use-2025-11-24")
    else:
        defaults = ("computer_20250124", "computer-use-2025-01-24")
    return (
        os.environ.get("CLAUDE_COMPUTER_TOOL_TYPE", defaults[0]),
        os.environ.get("CLAUDE_COMPUTER_BETA", defaults[1]),
    )


def supports_effort(model: str = MODEL) -> bool:
    return any(tag in model for tag in _NEW_COMPUTER_MODELS)


def anthropic_headers(model: str = MODEL) -> dict[str, str]:
    _, beta = computer_tool_version(model)
    return {
        "content-type": "application/json",
        "x-api-key": api_key(),
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": beta,
    }


def should_retry(status: int) -> bool:
    return status in {408, 409, 429} or status >= 500


def anthropic_request(payload: dict[str, Any]) -> dict[str, Any]:
    last_error = "request was not attempted"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                API_URL,
                headers=anthropic_headers(str(payload.get("model") or MODEL)),
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = f"connection error: {exc}"
        else:
            if response.status_code < 400:
                data = response.json()
                print(
                    "claude usage: "
                    + json.dumps(data.get("usage") or {}, sort_keys=True),
                    file=sys.stderr,
                    flush=True,
                )
                return data
            last_error = f"HTTP {response.status_code}: {response.text[:1000]}"
            if not should_retry(response.status_code):
                break
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    time.sleep(min(30.0, max(0.0, float(retry_after))))
                    continue
                except ValueError:
                    pass
        if attempt < MAX_RETRIES:
            time.sleep(min(2.0**attempt, 30.0))
    raise RuntimeError(
        f"Claude request failed after {MAX_RETRIES} attempts: {last_error}"
    )


def image_size(png: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(png)) as image:
        return image.size


def display_size(width: int, height: int) -> tuple[int, int]:
    """Fit a screenshot inside the reference 1280x720 box without distortion."""
    scale = min(1.0, MAX_IMAGE_WIDTH / width, MAX_IMAGE_HEIGHT / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_png(png: bytes, size: tuple[int, int]) -> bytes:
    with Image.open(BytesIO(png)) as image:
        if image.size == size:
            return png
        image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()


def image_content(png: bytes) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(png).decode("ascii"),
        },
    }


def initial_user_message(
    task: str, png: bytes, size: tuple[int, int]
) -> dict[str, Any]:
    """Send the task and initial desktop state in the first model turn."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": task},
            image_content(resize_png(png, size)),
        ],
    }


def screenshot_result(
    tool_id: str, png: bytes, size: tuple[int, int]
) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": [
            {"type": "text", "text": "Here is the screenshot after the action."},
            image_content(resize_png(png, size)),
        ],
        "is_error": False,
    }


def prune_old_images(
    messages: list[dict[str, Any]],
    keep: int = IMAGES_TO_KEEP,
    threshold: int = MIN_REMOVAL_THRESHOLD,
) -> None:
    """Remove old tool-result screenshots in chunks, as ClaudeFixedAgent does."""
    if keep <= 0 or threshold <= 0:
        return
    results = [
        block
        for message in messages
        for block in (
            message.get("content") if isinstance(message.get("content"), list) else []
        )
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    total = sum(
        1
        for result in results
        for block in (
            result.get("content") if isinstance(result.get("content"), list) else []
        )
        if isinstance(block, dict) and block.get("type") == "image"
    )
    remove = total - keep
    remove -= remove % threshold
    if remove <= 0:
        return
    for result in results:
        content = result.get("content")
        if not isinstance(content, list):
            continue
        retained = []
        for block in content:
            if remove > 0 and isinstance(block, dict) and block.get("type") == "image":
                remove -= 1
            else:
                retained.append(block)
        result["content"] = retained


def add_cache_marker(messages: list[dict[str, Any]]) -> None:
    """Mark the newest prefix for caching and retain at most four markers."""
    content = messages[-1].get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
        messages[-1]["content"] = content
    if isinstance(content, list) and content and isinstance(content[-1], dict):
        content[-1]["cache_control"] = {"type": "ephemeral"}
    seen = 0
    for message in reversed(messages):
        blocks = message.get("content")
        if (
            not isinstance(blocks, list)
            or not blocks
            or not isinstance(blocks[-1], dict)
        ):
            continue
        if "cache_control" not in blocks[-1]:
            continue
        seen += 1
        if seen > 4:
            blocks[-1].pop("cache_control", None)


def request_payload(
    messages: list[dict[str, Any]], display: tuple[int, int], model: str = MODEL
) -> dict[str, Any]:
    tool_type, _ = computer_tool_version(model)
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "tools": [
            {
                "type": tool_type,
                "name": "computer",
                "display_width_px": display[0],
                "display_height_px": display[1],
                **({"enable_zoom": True} if tool_type == "computer_20251124" else {}),
            },
            {
                "name": "complete",
                "description": (
                    "Signal that the visible desktop state fully satisfies the task."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "infeasible",
                "description": "Signal that the task is infeasible.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ],
    }
    if EFFORT and supports_effort(model):
        payload["output_config"] = {"effort": EFFORT}
    return payload


def normalize_key(key: Any) -> str:
    raw = str(key).strip()
    compact = raw.lower().replace("_", "").replace("-", "").replace(" ", "")
    aliases = {
        "enter": "Return",
        "return": "Return",
        "esc": "Escape",
        "escape": "Escape",
        "ctrl": "ctrl",
        "control": "ctrl",
        "cmd": "super",
        "command": "super",
        "win": "super",
        "super": "super",
        "alt": "alt",
        "option": "alt",
        "shift": "shift",
        "tab": "Tab",
        "backspace": "BackSpace",
        "delete": "Delete",
        "space": "space",
        "spacebar": "space",
        "pagedown": "pagedown",
        "pageup": "pageup",
        "arrowleft": "Left",
        "left": "Left",
        "arrowright": "Right",
        "right": "Right",
        "arrowup": "Up",
        "up": "Up",
        "arrowdown": "Down",
        "down": "Down",
    }
    return aliases.get(compact, raw)


def keys(text: Any) -> list[str]:
    if isinstance(text, str):
        values = text.split("+")
    elif isinstance(text, list):
        values = text
    else:
        raise TypeError("key text must be a string or list")
    normalized = [normalize_key(value) for value in values if str(value).strip()]
    if not normalized:
        raise ValueError("at least one key is required")
    return normalized


def scaled_point(
    value: Any,
    native: tuple[int, int],
    display: tuple[int, int],
) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"invalid coordinate: {value!r}")
    x = round(float(value[0]) * native[0] / display[0])
    y = round(float(value[1]) * native[1] / display[1])
    return [max(0, min(native[0] - 1, x)), max(0, min(native[1] - 1, y))]


def type_actions(text: str) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if line:
            actions.append({"keyboard": {"text": line}})
        if index < len(lines) - 1:
            actions.append({"keyboard": {"keys": ["Return"]}})
    return actions


def modifier_wrap(actions: list[dict[str, Any]], modifier: Any) -> list[dict[str, Any]]:
    if not modifier:
        return actions
    held = keys(modifier)
    return [
        {"keyboard": {"keys_down": held}},
        *actions,
        {"keyboard": {"keys_up": list(reversed(held))}},
    ]


def translate_action(
    action: dict[str, Any],
    native: tuple[int, int],
    display: tuple[int, int],
    cursor: list[int] | None = None,
) -> tuple[list[dict[str, Any]], list[int] | None]:
    """Translate one Anthropic computer action and return its resulting cursor."""
    kind = str(action.get("action") or "")
    coordinate = action.get("coordinate")
    point = (
        scaled_point(coordinate, native, display) if coordinate is not None else None
    )
    next_cursor = list(cursor) if cursor is not None else None
    if point is not None:
        next_cursor = point

    if kind == "zoom":
        region = action.get("region")
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            raise ValueError("zoom requires region [x1, y1, x2, y2]")
        first = scaled_point(region[:2], native, display)
        second = scaled_point(region[2:], native, display)
        if first[0] == second[0] or first[1] == second[1]:
            raise ValueError("zoom region must have non-zero area")
        return [], next_cursor
    if kind in {"screenshot", "cursor_position"}:
        return [], next_cursor
    if kind == "key":
        return [
            {"keyboard": {"keys": keys(action.get("text", action.get("keys")))}}
        ], next_cursor
    if kind == "type":
        text = action.get("text")
        if not isinstance(text, str):
            raise ValueError("type requires string text")
        return type_actions(text), next_cursor
    if kind == "hold_key":
        held = keys(action.get("text", action.get("keys")))
        duration = max(0.1, min(10.0, float(action.get("duration", 1.0))))
        return [
            {"keyboard": {"keys_down": held}},
            {"action": "wait", "time": duration},
            {"keyboard": {"keys_up": list(reversed(held))}},
        ], next_cursor
    if kind == "wait":
        duration = max(0.1, min(30.0, float(action.get("duration", 1.0))))
        return [{"action": "wait", "time": duration}], next_cursor
    if kind == "mouse_move":
        if point is None:
            raise ValueError("mouse_move requires coordinate")
        return [{"mouse": {"move": point}}], next_cursor
    if kind in {"left_mouse_down", "left_mouse_up"}:
        actions = [] if point is None else [{"mouse": {"move": point}}]
        field = "left_down" if kind.endswith("down") else "left_up"
        actions.append({"mouse": {"buttons": {field: True}}})
        return modifier_wrap(actions, action.get("text")), next_cursor
    if kind == "left_click_drag":
        if point is None:
            raise ValueError("left_click_drag requires coordinate")
        start_value = action.get("start_coordinate")
        start = (
            scaled_point(start_value, native, display)
            if start_value is not None
            else cursor
        )
        drag_points = [list(start), point] if start is not None else [point]
        return modifier_wrap(
            [{"mouse": {"left_click_drag": drag_points}}], action.get("text")
        ), next_cursor
    if kind == "scroll":
        direction = str(action.get("scroll_direction") or "")
        amount = action.get("scroll_amount")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError("scroll_amount must be a non-negative integer")
        actions = [] if point is None else [{"mouse": {"move": point}}]
        if direction == "up":
            actions.append({"mouse": {"scroll": -amount}})
        elif direction == "down":
            actions.append({"mouse": {"scroll": amount}})
        elif direction in {"left", "right"}:
            horizontal = amount if direction == "right" else -amount
            actions.extend(
                [
                    {"keyboard": {"keys_down": ["shift"]}},
                    {"mouse": {"scroll": horizontal}},
                    {"keyboard": {"keys_up": ["shift"]}},
                ]
            )
        else:
            raise ValueError(f"invalid scroll_direction: {direction!r}")
        return modifier_wrap(actions, action.get("text")), next_cursor
    clicks = {
        "left_click": "left_click",
        "right_click": "right_click",
        "middle_click": "middle_click",
        "double_click": "double_click",
        "triple_click": "triple_click",
    }
    if kind in clicks:
        if point is None:
            raise ValueError(f"{kind} requires coordinate")
        return modifier_wrap(
            [{"mouse": {clicks[kind]: point}}], action.get("text")
        ), next_cursor
    raise ValueError(f"unsupported computer action: {kind!r}")


def zoom_image(
    png: bytes,
    region: Any,
    native: tuple[int, int],
    display: tuple[int, int],
) -> bytes:
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        raise ValueError("zoom requires region [x1, y1, x2, y2]")
    first = scaled_point(region[:2], native, display)
    second = scaled_point(region[2:], native, display)
    left, right = sorted((first[0], second[0]))
    top, bottom = sorted((first[1], second[1]))
    if left == right or top == bottom:
        raise ValueError("zoom region must have non-zero area")
    with Image.open(BytesIO(png)) as image:
        cropped = image.crop((left, top, right + 1, bottom + 1)).convert("RGB")
        scale = min(
            1.0,
            1568 / max(cropped.size),
            math.sqrt(1_150_000 / (cropped.width * cropped.height)),
        )
        if scale < 1.0:
            cropped = cropped.resize(
                (
                    max(1, round(cropped.width * scale)),
                    max(1, round(cropped.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        output = BytesIO()
        cropped.save(output, format="PNG")
        return output.getvalue()


def tool_uses(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        block
        for block in (response.get("content") or [])
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


def execute_infeasible_call(computer: Computer, call: dict[str, Any]) -> dict[str, Any]:
    if call.get("name") != "infeasible":
        raise RuntimeError(f"unsupported function call: {call.get('name')!r}")
    print("model declared task infeasible", file=sys.stderr, flush=True)
    return computer.step([{"action_type": "FAIL"}])


def terminal_reminder() -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Do not end this desktop task with only text. Continue with the "
                    "computer tool, call complete if the visible desktop state fully "
                    "satisfies the task, or call infeasible if a fundamental blocker "
                    "makes the task impossible."
                ),
            }
        ],
    }


def parse_turn(
    calls: list[dict[str, Any]],
    native: tuple[int, int],
    display: tuple[int, int],
    cursor: list[int] | None,
) -> tuple[list[tuple[dict[str, Any], list[dict[str, Any]]]], list[int] | None]:
    """Validate every call before any is executed (atomic-turn semantics)."""
    parsed = []
    next_cursor = list(cursor) if cursor is not None else None
    for call in calls:
        if call.get("name") != "computer":
            raise ValueError(f"unsupported tool: {call.get('name')!r}")
        tool_input = call.get("input")
        if not isinstance(tool_input, dict):
            raise TypeError("computer tool input must be an object")
        actions, next_cursor = translate_action(
            tool_input, native, display, next_cursor
        )
        if not call.get("id"):
            raise ValueError("computer tool call is missing an id")
        parsed.append((call, actions))
    return parsed, next_cursor


def parse_error_results(
    calls: list[dict[str, Any]], error: Exception
) -> list[dict[str, Any]]:
    text = (
        f"Tool input could not be parsed: {type(error).__name__}: {error}. "
        "No actions in this response were executed; retry with valid tool calls."
    )
    return [
        {
            "type": "tool_result",
            "tool_use_id": str(call.get("id") or "missing-tool-id"),
            "content": [{"type": "text", "text": text}],
            "is_error": True,
        }
        for call in calls
    ]


def run(env_url: str, task: str) -> None:
    computer = Computer(env_url, timeout_sec=COMPUTER_TIMEOUT)
    messages: list[dict[str, Any]] = []
    cursor: list[int] | None = None
    try:
        observation = computer.observe()
        native = image_size(observation["png"])
        display = display_size(*native)
        messages.append(initial_user_message(task, observation["png"], display))

        for step in range(MAX_STEPS):
            prune_old_images(messages)
            add_cache_marker(messages)
            response = anthropic_request(request_payload(messages, display))
            content = response.get("content")
            if not isinstance(content, list):
                raise TypeError("Claude response content must be a list")
            messages.append({"role": "assistant", "content": content})
            calls = tool_uses(response)
            text = " ".join(
                str(block.get("text") or "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            print(
                f"step {step}: stop={response.get('stop_reason')} tools={len(calls)} "
                f"text={text[:300]!r}",
                file=sys.stderr,
                flush=True,
            )
            terminal_calls = [
                call
                for call in calls
                if call.get("name") in {"complete", "infeasible"}
            ]
            if terminal_calls:
                if len(calls) != 1:
                    raise RuntimeError(
                        "a terminal tool must be the only tool call in a response"
                    )
                terminal_call = terminal_calls[0]
                if terminal_call.get("name") == "infeasible":
                    execute_infeasible_call(computer, terminal_call)
                else:
                    print("model declared task complete", file=sys.stderr, flush=True)
                break
            if not calls:
                print(
                    f"step {step}: text-only response rejected; requesting a "
                    "structured terminal decision",
                    file=sys.stderr,
                    flush=True,
                )
                messages.append(terminal_reminder())
                continue
            try:
                parsed, next_cursor = parse_turn(calls, native, display, cursor)
            except (KeyError, TypeError, ValueError) as exc:
                print(f"step {step}: atomic tool parse failure: {exc}", file=sys.stderr)
                messages.append(
                    {"role": "user", "content": parse_error_results(calls, exc)}
                )
                continue

            results = []
            terminal = False
            for call, actions in parsed:
                tool_input = call["input"]
                print(
                    "computer action: " + json.dumps(tool_input, ensure_ascii=False),
                    file=sys.stderr,
                    flush=True,
                )
                if actions:
                    step_result = computer.step(actions)
                    if step_result.get("done"):
                        terminal = True
                        break
                observation = computer.observe()
                native = image_size(observation["png"])
                new_display = display_size(*native)
                if new_display != display:
                    raise RuntimeError(
                        f"display changed during task: declared {display}, observed {new_display}"
                    )
                result_png = observation["png"]
                if tool_input.get("action") == "zoom":
                    result_png = zoom_image(
                        result_png, tool_input.get("region"), native, display
                    )
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": str(call["id"]),
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Here is the requested zoomed region.",
                                },
                                image_content(result_png),
                            ],
                            "is_error": False,
                        }
                    )
                else:
                    results.append(
                        screenshot_result(str(call["id"]), result_png, display)
                    )
            if terminal:
                print(
                    "environment finalized after model action",
                    file=sys.stderr,
                    flush=True,
                )
                break
            cursor = next_cursor
            messages.append({"role": "user", "content": results})
        else:
            print(f"maximum model call count reached ({MAX_STEPS})", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - task failures must still finalize the clock.
        print(f"agent failed: {exc!r}", file=sys.stderr, flush=True)
    finally:
        try:
            computer.done()
        except Exception as exc:  # noqa: BLE001 - best-effort finalization report.
            print(f"computer.done failed: {exc!r}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
