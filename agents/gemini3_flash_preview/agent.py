"""Gemini 3 Flash Preview Computer Use agent for OSWorld-style desktop tasks.

Contract: python agent.py <env_url> <task_description>

This template uses the Gemini Interactions API `computer_use` tool with the
desktop environment. It executes returned function calls through the
cua-speedrun Computer client, sends screenshots back as `function_result`
objects, and records token/cost usage in a shared JSON ledger so parallel task
agents observe one budget cap.
"""

from __future__ import annotations

import base64
import json
import math
import os
import struct
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from cua_speedrun.client import Computer

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
API_URL = os.environ.get(
    "GEMINI_INTERACTIONS_URL",
    "https://generativelanguage.googleapis.com/v1beta/interactions",
)
MAX_STEPS = int(os.environ.get("CS_MAX_STEPS", "100"))
# A model stuck emitting incomplete responses without actions burns a full
# screenshot-sized request per cycle; give up after this many in a row.
MAX_INCOMPLETE_STREAK = int(os.environ.get("CS_MAX_INCOMPLETE_STREAK", "5"))
REQUEST_TIMEOUT = float(os.environ.get("GEMINI_REQUEST_TIMEOUT", "600"))
MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "5"))
GRID = 1000.0
INVALID_TOOL_ARGUMENTS = (
    "invalid tool call arguments; please follow the computer use tool semantics provided to you"
)

INPUT_RATE_PER_MILLION = float(os.environ.get("GEMINI_INPUT_RATE_PER_MILLION", "0.50"))
OUTPUT_RATE_PER_MILLION = float(os.environ.get("GEMINI_OUTPUT_RATE_PER_MILLION", "3.00"))
# -1 (or any negative value, or unset) disables the cap entirely; the
# ledger still records spend for reporting.
_raw_cost_limit = float(os.environ.get("GEMINI_COST_LIMIT_USD", "-1"))
COST_LIMIT_USD = _raw_cost_limit if _raw_cost_limit >= 0 else float("inf")
COST_LEDGER = Path(
    os.environ.get("GEMINI_COST_LEDGER", "/tmp/cua-speedrun-gemini3-flash-preview-cost.json")
)
COST_RESERVE_USD = float(os.environ.get("GEMINI_COST_RESERVE_USD", "2.00"))
MISSING_USAGE_FALLBACK_USD = float(
    os.environ.get("GEMINI_MISSING_USAGE_FALLBACK_USD", str(COST_RESERVE_USD))
)

THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL", "high")
PROMPT_INJECTION_DETECTION = os.environ.get(
    "GEMINI_PROMPT_INJECTION_DETECTION", ""
).lower() in {"1", "true", "yes"}
AUTO_ACK_SAFETY = os.environ.get("GEMINI_AUTO_ACK_SAFETY", "1").lower() in {
    "1",
    "true",
    "yes",
}
EXCLUDED_PREDEFINED_FUNCTIONS = [
    part.strip()
    for part in os.environ.get("GEMINI_EXCLUDED_PREDEFINED_FUNCTIONS", "bash").split(",")
    if part.strip()
]
# The benchmark VM is disposable and isolated: no real accounts, purchases,
# or user data exist, so confirmation prompts only burn steps and requests.
# Every documented policy category is disabled by default; set the variable
# to a comma list (or "none") to re-enable specific categories.
DISABLED_SAFETY_POLICIES = [
    part.strip().lower()
    for part in os.environ.get(
        "GEMINI_DISABLED_SAFETY_POLICIES",
        "financial_transactions,sensitive_data_modification,communication_tool,"
        "account_creation,data_modification,user_consent_management,"
        "legal_terms_and_agreements",
    ).split(",")
    if part.strip() and part.strip().lower() != "none"
]
SUPPORTED_DESKTOP_COMMANDS = {
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
UNSUPPORTED_DESKTOP_COMMANDS: set[str] = set()

# Gemini 3 Flash Preview sometimes emits legacy or equivalent UI action names
# even with the desktop tool configured. Normalize only direct UI operations;
# command-execution aliases remain blocked below.
DESKTOP_ACTION_ALIASES = {
    "open_url": "navigate",
    "navigate": "navigate",
    "open_application": "open_application",
    "open_terminal": "open_terminal",
    "open_web_browser": "open_web_browser",
    "search": "search",
    "go_back": "go_back",
    "go_forward": "go_forward",
    "wait_5_seconds": "wait_5_seconds",
    "click_at": "click",
    "mouse_click": "click",
    "double_click_at": "double_click",
    "hover_at": "move",
    "mouse_move": "move",
    "type_text": "type",
    "type_text_at": "type_at",
    "type_at": "type_at",
    "key": "press_key",
    "key_combination": "hotkey",
    "press_key_combination": "hotkey",
    "type_key_combination": "hotkey",
    "keypress_at": "keypress_at",
    "scroll_at": "scroll",
    "scroll_document": "scroll",
    "mouse_scroll": "scroll",
    "mouse_drag": "drag_and_drop",
    "drag": "drag_and_drop",
}
SHELL_COMMANDS = {
    "bash",
    "bash_run",
    "execute_shell",
    "run_bash",
    "run_bash_command",
    "run_command",
    "run_terminal_command",
    "run_tool",
}


SYSTEM_INSTRUCTION = """
* You are utilising an Ubuntu virtual machine using x86_64 architecture with internet access.
* To open browser, please just click on the Chrome icon.  Note, Chrome is what is installed on your system.
* DO NOT ask users for clarification during task execution. DO NOT stop to request more information from users. Always take action using available tools.
* TASK FEASIBILITY: You can declare a task infeasible at any point during execution - whether at the beginning after taking a screenshot, or later after attempting some actions and discovering barriers. Carefully evaluate whether the task is feasible given the current system state, available applications, and task requirements. If you determine that a task cannot be completed due to:
  - Missing required applications or dependencies that cannot be installed
  - Insufficient permissions or system limitations
  - Contradictory or impossible requirements
  - Any other fundamental barriers that make completion impossible
  Then you MUST use the infeasible tool call.
* Home directory of this Ubuntu system is '/home/user'.
* If you need a password for sudo, the password of the computer is 'password'.
* Before you make an initial plan, you MUST use the 'take_screenshot' tool to take a screenshot of the current screen. The screenshot returned will show important context needed to decide what to do next.
"""


class CostLimitReached(RuntimeError):
    pass


class InvalidToolArguments(ValueError):
    pass


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("set GEMINI_API_KEY or GOOGLE_API_KEY for the Gemini API")
    return key


@contextmanager
def locked_cost_state():
    import fcntl

    COST_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    lock_path = COST_LEDGER.with_suffix(COST_LEDGER.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if COST_LEDGER.exists():
            try:
                state = json.loads(COST_LEDGER.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                state = {}
        else:
            state = {}
        state.setdefault("actual_usd", 0.0)
        state.setdefault("reserved_usd", 0.0)
        state.setdefault("calls", 0)
        state.setdefault("input_tokens", 0)
        state.setdefault("output_tokens", 0)
        state.setdefault("thought_tokens", 0)
        state.setdefault("tool_use_tokens", 0)
        state.setdefault("missing_usage_calls", 0)
        state.setdefault("reservations", {})
        yield state
        tmp_path = COST_LEDGER.with_suffix(COST_LEDGER.suffix + ".tmp")
        tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(COST_LEDGER)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


class CostTracker:
    def reserve(self, label: str) -> str:
        reservation_id = f"{os.getpid()}-{uuid.uuid4().hex}"
        with locked_cost_state() as state:
            cleanup_stale_reservations(state)
            current = float(state["actual_usd"]) + float(state["reserved_usd"])
            if current + COST_RESERVE_USD > COST_LIMIT_USD:
                raise CostLimitReached(
                    f"Gemini cost cap would be exceeded: "
                    f"${current:.4f} reserved/actual + ${COST_RESERVE_USD:.4f} > "
                    f"${COST_LIMIT_USD:.2f}"
                )
            state["reserved_usd"] = float(state["reserved_usd"]) + COST_RESERVE_USD
            state["reservations"][reservation_id] = {
                "amount_usd": COST_RESERVE_USD,
                "pid": os.getpid(),
                "label": label,
                "created_at": time.time(),
            }
        return reservation_id

    def release(self, reservation_id: str, usage: dict[str, Any] | None, error: bool = False) -> float:
        cost, tokens = estimate_cost(usage)
        if usage is None and not error:
            cost = MISSING_USAGE_FALLBACK_USD
        with locked_cost_state() as state:
            reservation = state["reservations"].pop(reservation_id, None)
            if reservation is not None:
                state["reserved_usd"] = max(
                    0.0,
                    float(state["reserved_usd"]) - float(reservation.get("amount_usd", 0.0)),
                )
            if not error:
                state["actual_usd"] = float(state["actual_usd"]) + cost
                state["calls"] = int(state["calls"]) + 1
                state["input_tokens"] = int(state["input_tokens"]) + tokens["input"]
                state["output_tokens"] = int(state["output_tokens"]) + tokens["output"]
                state["thought_tokens"] = int(state["thought_tokens"]) + tokens["thought"]
                state["tool_use_tokens"] = int(state["tool_use_tokens"]) + tokens["tool_use"]
                if usage is None:
                    state["missing_usage_calls"] = int(state["missing_usage_calls"]) + 1
                print(
                    "gemini cost: "
                    f"call=${cost:.5f}, total=${float(state['actual_usd']):.5f}/"
                    f"${COST_LIMIT_USD:.2f}, tokens="
                    f"in:{tokens['input']} out:{tokens['output']} "
                    f"thought:{tokens['thought']} tool:{tokens['tool_use']}",
                    file=sys.stderr,
                    flush=True,
                )
                if float(state["actual_usd"]) >= COST_LIMIT_USD:
                    print("gemini cost cap reached after this call", file=sys.stderr, flush=True)
            return cost


def pid_is_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_stale_reservations(state: dict[str, Any]) -> None:
    reservations = state.get("reservations")
    if not isinstance(reservations, dict):
        state["reservations"] = {}
        state["reserved_usd"] = 0.0
        return
    stale = [
        key
        for key, reservation in reservations.items()
        if not isinstance(reservation, dict) or not pid_is_alive(reservation.get("pid"))
    ]
    if not stale:
        return
    released = 0.0
    for key in stale:
        reservation = reservations.pop(key, {})
        if isinstance(reservation, dict):
            released += float(reservation.get("amount_usd", 0.0) or 0.0)
    state["reserved_usd"] = max(0.0, float(state.get("reserved_usd", 0.0)) - released)


def estimate_cost(usage: dict[str, Any] | None) -> tuple[float, dict[str, int]]:
    if not usage:
        return 0.0, {"input": 0, "output": 0, "thought": 0, "tool_use": 0}
    input_tokens = int(usage.get("total_input_tokens") or 0)
    output_tokens = int(usage.get("total_output_tokens") or 0)
    thought_tokens = int(usage.get("total_thought_tokens") or 0)
    tool_use_tokens = int(usage.get("total_tool_use_tokens") or 0)
    billed_output = output_tokens + thought_tokens + tool_use_tokens
    cost = (
        input_tokens * INPUT_RATE_PER_MILLION
        + billed_output * OUTPUT_RATE_PER_MILLION
    ) / 1_000_000.0
    return cost, {
        "input": input_tokens,
        "output": output_tokens,
        "thought": thought_tokens,
        "tool_use": tool_use_tokens,
    }


def extract_usage(interaction: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(interaction.get("usage"), dict):
        return interaction["usage"]
    metadata = interaction.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("total_usage"), dict):
        return metadata["total_usage"]
    for step in reversed(interaction.get("steps") or []):
        if isinstance(step, dict) and isinstance(step.get("usage"), dict):
            return step["usage"]
        step_meta = step.get("metadata") if isinstance(step, dict) else None
        if isinstance(step_meta, dict) and isinstance(step_meta.get("total_usage"), dict):
            return step_meta["total_usage"]
    return None


def should_retry_error(status_code: int, body: str) -> bool:
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    if status_code == 400 and (
        "malformed_tool_call" in body or "Model generated undefined function" in body
    ):
        return True
    return False


def gemini_request(payload: dict[str, Any], tracker: CostTracker, label: str) -> dict[str, Any]:
    reservation_id = tracker.reserve(label)
    try:
        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    API_URL,
                    headers={"x-goog-api-key": api_key(), "Content-Type": "application/json"},
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < MAX_RETRIES:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                break
            if resp.status_code < 400:
                data = resp.json()
                print(
                    "gemini response: " + json.dumps(data, ensure_ascii=False),
                    file=sys.stderr,
                    flush=True,
                )
                tracker.release(reservation_id, extract_usage(data), error=False)
                return data
            last_error = f"HTTP {resp.status_code} {resp.text[:1000]}"
            if not should_retry_error(resp.status_code, resp.text):
                break
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Gemini request failed after {MAX_RETRIES} attempts: {last_error}")
    except BaseException:
        tracker.release(reservation_id, None, error=True)
        raise


def screenshot_part(png: bytes) -> dict[str, str]:
    return {
        "type": "image",
        "data": base64.b64encode(png).decode("ascii"),
        "mime_type": "image/png",
        "resolution": "high",
    }


def tools() -> list[dict[str, Any]]:
    computer_tool: dict[str, Any] = {
        "type": "computer_use",
        "environment": "desktop",
    }
    if PROMPT_INJECTION_DETECTION:
        computer_tool["enable_prompt_injection_detection"] = True
    if EXCLUDED_PREDEFINED_FUNCTIONS:
        computer_tool["excluded_predefined_functions"] = EXCLUDED_PREDEFINED_FUNCTIONS
    if DISABLED_SAFETY_POLICIES:
        computer_tool["disabled_safety_policies"] = DISABLED_SAFETY_POLICIES
    return [
        computer_tool,
        {
            "type": "function",
            "name": "infeasible",
            "description": "Signal that the task is infeasible.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    ]


def base_payload() -> dict[str, Any]:
    return {
        "model": MODEL,
        "system_instruction": SYSTEM_INSTRUCTION,
        "tools": tools(),
        "store": True,
        "generation_config": {
            "thinking_level": THINKING_LEVEL,
        },
    }


def first_payload(task: str) -> dict[str, Any]:
    payload = base_payload()
    payload["input"] = [{"type": "text", "text": task}]
    return payload


def continue_payload(previous_interaction_id: str, function_results: list[dict[str, Any]]) -> dict[str, Any]:
    payload = base_payload()
    payload["previous_interaction_id"] = previous_interaction_id
    payload["input"] = function_results
    return payload


def continue_after_incomplete_payload(previous_interaction_id: str, png: bytes) -> dict[str, Any]:
    payload = base_payload()
    payload["previous_interaction_id"] = previous_interaction_id
    payload["input"] = [
        {
            "type": "text",
            "text": (
                "Your previous response ended before issuing a tool call. "
                "Continue from the current desktop screenshot by calling the next "
                "computer_use action, or give your final response if the task is done."
            ),
        },
        screenshot_part(png),
    ]
    return payload


def function_calls(interaction: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for step in interaction.get("steps") or []:
        if isinstance(step, dict) and step.get("type") == "function_call":
            calls.append(step)
    return calls


def model_text(interaction: dict[str, Any]) -> str:
    chunks: list[str] = []
    for step in interaction.get("steps") or []:
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        for block in step.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                chunks.append(str(block.get("text", "")))
    return "\n".join(part for part in chunks if part)


def image_size(png: bytes) -> tuple[int, int]:
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
        raise ValueError("observation is not a valid PNG with an IHDR chunk")
    return struct.unpack(">II", png[16:24])


def scale_xy(args: dict[str, Any], width: int, height: int, x_key: str = "x", y_key: str = "y") -> list[int]:
    x_value = args.get(x_key)
    y_value = args.get(y_key)
    if x_key == "x" and y_key == "y" and (x_value is None or y_value is None):
        coordinate = args.get("coordinate")
        if isinstance(coordinate, (list, tuple)) and len(coordinate) >= 2:
            x_value, y_value = coordinate[:2]
        else:
            point = args.get("point")
            if isinstance(point, dict):
                x_value = point.get("x", point.get('"x"', x_value))
                y_value = point.get("y", point.get('"y"', y_value))
    if x_value is None or y_value is None:
        raise InvalidToolArguments(INVALID_TOOL_ARGUMENTS)
    try:
        x = max(0.0, min(GRID - 1, float(x_value)))
        y = max(0.0, min(GRID - 1, float(y_value)))
    except (TypeError, ValueError) as exc:
        raise InvalidToolArguments(INVALID_TOOL_ARGUMENTS) from exc
    return [int(x / GRID * width), int(y / GRID * height)]


def scroll_amount(args: dict[str, Any], direction: str) -> int:
    if "magnitude_in_wheel_clicks" in args:
        raw_steps = int(math.ceil(abs(float(args.get("magnitude_in_wheel_clicks") or 1))))
        wheel_steps = max(1, min(30, raw_steps))
    elif "amount" in args:
        raw_steps = int(math.ceil(abs(float(args.get("amount") or 1))))
        wheel_steps = max(1, min(30, raw_steps))
    else:
        magnitude = int(float(args.get("magnitude_in_pixels", args.get("magnitude", 300)) or 300))
        wheel_steps = max(1, min(10, math.ceil(abs(magnitude) / 120)))
    if direction in {"up", "left"}:
        return -wheel_steps
    return wheel_steps


def normalize_key(key: Any) -> str:
    text = str(key).strip()
    lowered = text.lower().replace(" ", "")
    compact = lowered.replace("_", "").replace("-", "")
    aliases = {
        "enter": "Return",
        "return": "Return",
        "esc": "Escape",
        "escape": "Escape",
        "cmd": "super",
        "command": "super",
        "win": "super",
        "winl": "super",
        "winr": "super",
        "winleft": "super",
        "winright": "super",
        "windows": "super",
        "superl": "super",
        "superr": "super",
        "superleft": "super",
        "superright": "super",
        "metal": "super",
        "metar": "super",
        "metaleft": "super",
        "metaright": "super",
        "ctrl": "ctrl",
        "control": "ctrl",
        "ctrll": "ctrl",
        "ctrlr": "ctrl",
        "ctrlleft": "ctrl",
        "ctrlright": "ctrl",
        "controll": "ctrl",
        "controlr": "ctrl",
        "controlleft": "ctrl",
        "controlright": "ctrl",
        "alt": "alt",
        "altl": "alt",
        "altr": "alt",
        "altleft": "alt",
        "altright": "alt",
        "option": "alt",
        "optionl": "alt",
        "optionr": "alt",
        "optionleft": "alt",
        "optionright": "alt",
        "shift": "shift",
        "shiftl": "shift",
        "shiftr": "shift",
        "shiftleft": "shift",
        "shiftright": "shift",
        "tab": "Tab",
        "backspace": "BackSpace",
        "delete": "Delete",
        "del": "Delete",
        "space": "space",
        "spacebar": "space",
        "pagedown": "pagedown",
        "pgdn": "pagedown",
        "next": "pagedown",
        "pageup": "pageup",
        "pgup": "pageup",
        "prior": "pageup",
        "home": "Home",
        "end": "End",
        "insert": "insert",
        "ins": "insert",
        "capslock": "capslock",
        "arrowleft": "Left",
        "left": "Left",
        "arrowright": "Right",
        "right": "Right",
        "arrowup": "Up",
        "up": "Up",
        "arrowdown": "Down",
        "down": "Down",
    }
    return aliases.get(compact, aliases.get(lowered, text))


def normalize_keys(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.replace("Control", "ctrl").replace("Command", "super")
        parts = [part for part in raw.replace("+", " ").split() if part]
    elif isinstance(raw, list):
        parts = raw
    else:
        parts = [raw]
    return [normalize_key(part) for part in parts if str(part).strip()]


@dataclass
class ExecutedCall:
    name: str
    call_id: str
    summary: str
    result: dict[str, Any]
    terminal: bool = False


def safety_ack(args: dict[str, Any], result: dict[str, Any]) -> bool:
    safety = args.get("safety_decision")
    if not isinstance(safety, dict):
        return True
    decision = str(safety.get("decision", "")).lower()
    result["safety_decision"] = safety
    if decision in {"blocked", "block"}:
        result["ok"] = False
        result["error"] = "safety decision blocked the action"
        return False
    if decision == "require_confirmation":
        if not AUTO_ACK_SAFETY:
            result["ok"] = False
            result["error"] = "safety confirmation required"
            return False
        result["safety_acknowledgement"] = True
    return True


def _execute_call(computer: Computer, call: dict[str, Any], width: int, height: int) -> ExecutedCall:
    name = str(call.get("name", ""))
    action_name = DESKTOP_ACTION_ALIASES.get(name, name)
    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    call_id = str(call.get("id") or call.get("call_id") or uuid.uuid4().hex)
    result: dict[str, Any] = {"ok": True}

    if name == "infeasible":
        computer.step([{"action_type": "FAIL"}])
        return ExecutedCall(
            name=name,
            call_id=call_id,
            summary="task declared infeasible",
            result={"ok": True},
            terminal=True,
        )

    if not safety_ack(args, result):
        return ExecutedCall(
            name=name,
            call_id=call_id,
            summary=result.get("error", "safety stop"),
            result=result,
            terminal=True,
        )

    actions: list[dict[str, Any]] = []
    wait_seconds: float | None = None

    if name in SHELL_COMMANDS:
        result["ok"] = False
        result["error"] = (
            "shell execution is disabled; use visible desktop actions, including opening and "
            "typing into a terminal through the GUI when needed"
        )
    elif action_name not in SUPPORTED_DESKTOP_COMMANDS and action_name not in {
        "navigate",
        "open_application",
        "open_terminal",
        "open_web_browser",
        "search",
        "go_back",
        "go_forward",
        "wait_5_seconds",
        "type_at",
        "keypress_at",
    }:
        result["ok"] = False
        result["error"] = f"unsupported desktop action: {name}"
    elif action_name == "click":
        button = str(args.get("button", "left")).lower()
        click_action = {
            "right": "right_click",
            "middle": "middle_click",
        }.get(button, "left_click")
        actions.append({"mouse": {click_action: scale_xy(args, width, height)}})
    elif action_name == "double_click":
        actions.append({"mouse": {"double_click": scale_xy(args, width, height)}})
    elif action_name == "triple_click":
        actions.append({"mouse": {"triple_click": scale_xy(args, width, height)}})
    elif action_name == "right_click":
        actions.append({"mouse": {"right_click": scale_xy(args, width, height)}})
    elif action_name == "middle_click":
        actions.append({"mouse": {"middle_click": scale_xy(args, width, height)}})
    elif action_name == "move":
        actions.append({"mouse": {"move": scale_xy(args, width, height)}})
    elif action_name == "mouse_down":
        actions.extend([
            {"mouse": {"move": scale_xy(args, width, height)}},
            {"mouse": {"buttons": {"left_down": True}}},
        ])
    elif action_name == "mouse_up":
        actions.extend([
            {"mouse": {"move": scale_xy(args, width, height)}},
            {"mouse": {"buttons": {"left_up": True}}},
        ])
    elif action_name == "drag_and_drop":
        if "start_x" in args and "start_y" in args:
            start_keys = ("start_x", "start_y")
        elif "source_x" in args and "source_y" in args:
            start_keys = ("source_x", "source_y")
        else:
            start_keys = ("x", "y")
        if "end_x" in args and "end_y" in args:
            end_keys = ("end_x", "end_y")
        elif "destination_x" in args and "destination_y" in args:
            end_keys = ("destination_x", "destination_y")
        elif "target_x" in args and "target_y" in args:
            end_keys = ("target_x", "target_y")
        else:
            end_keys = ("to_x", "to_y")
        start = scale_xy(args, width, height, *start_keys)
        end = scale_xy(args, width, height, *end_keys)
        actions.append({"mouse": {"left_click_drag": [start, end]}})
    elif action_name == "type_at":
        actions.append({"mouse": {"left_click": scale_xy(args, width, height)}})
        if bool(args.get("clear_before_typing", name == "type_text_at")):
            actions.append({"keyboard": {"keys": ["ctrl", "a"]}})
        text = str(args.get("text", ""))
        if text:
            actions.append({"keyboard": {"text": text}})
        if bool(args.get("press_enter", False)):
            actions.append({"keyboard": {"keys": ["Return"]}})
    elif action_name == "type":
        text = str(args.get("text", ""))
        if text:
            actions.append({"keyboard": {"text": text}})
        if bool(args.get("press_enter", False)):
            actions.append({"keyboard": {"keys": ["Return"]}})
    elif action_name == "press_key":
        keys = normalize_keys(args.get("key", args.get("name", args.get("text"))))
        if keys:
            actions.append({"keyboard": {"keys": keys}})
    elif action_name == "keypress_at":
        raw_keys = [*args.get("modifiers", []), args.get("keysymbol")]
        keys = normalize_keys([key for key in raw_keys if key])
        if keys:
            actions.append({"keyboard": {"keys": keys}})
    elif action_name == "key_down":
        key = normalize_key(args.get("key", ""))
        if key:
            actions.append({"keyboard": {"key_down": key}})
    elif action_name == "key_up":
        key = normalize_key(args.get("key", ""))
        if key:
            actions.append({"keyboard": {"key_up": key}})
    elif action_name == "hotkey":
        raw_keys = args.get(
            "keys",
            args.get("combination", args.get("key_combination")),
        )
        keys = normalize_keys(raw_keys)
        if keys:
            actions.append({"keyboard": {"keys": keys}})
    elif action_name == "navigate":
        actions.extend(
            [
                {"keyboard": {"keys": ["ctrl", "l"]}},
                {"keyboard": {"text": str(args.get("url", ""))}},
                {"keyboard": {"keys": ["Return"]}},
            ]
        )
    elif action_name == "open_application":
        application = str(args.get("application", args.get("name", ""))).strip()
        actions.extend(
            [
                {"keyboard": {"keys": ["super"]}},
                {"keyboard": {"text": application}},
                {"keyboard": {"keys": ["Return"]}},
            ]
        )
    elif action_name == "open_terminal":
        actions.append({"keyboard": {"keys": ["ctrl", "alt", "t"]}})
    elif action_name == "open_web_browser":
        actions.extend(
            [
                {"keyboard": {"keys": ["super"]}},
                {"keyboard": {"text": "google chrome"}},
                {"keyboard": {"keys": ["Return"]}},
            ]
        )
    elif action_name == "search":
        actions.extend(
            [
                {"keyboard": {"keys": ["ctrl", "l"]}},
                {"keyboard": {"text": "https://www.google.com"}},
                {"keyboard": {"keys": ["Return"]}},
            ]
        )
    elif action_name == "go_back":
        actions.append({"keyboard": {"keys": ["alt", "Left"]}})
    elif action_name == "go_forward":
        actions.append({"keyboard": {"keys": ["alt", "Right"]}})
    elif action_name in UNSUPPORTED_DESKTOP_COMMANDS:
        key = normalize_key(args.get("key", ""))
        if key:
            result["ok"] = False
            result["error"] = (
                f"{name} hold semantics are not supported by this backend; "
                "the action is excluded from Gemini's predefined desktop functions"
            )
    elif action_name == "scroll":
        direction = str(args.get("direction", "down")).lower()
        if direction in {"left", "right"}:
            actions.append({"keyboard": {"key_down": "shift"}})
            if "x" in args and "y" in args:
                actions.append({"mouse": {"move": scale_xy(args, width, height)}})
            actions.append({"mouse": {"scroll": scroll_amount(args, direction)}})
            actions.append({"keyboard": {"key_up": "shift"}})
        else:
            wheel_steps = scroll_amount(args, direction)
            if "x" in args and "y" in args:
                actions.append({"mouse": {"move": scale_xy(args, width, height)}})
            actions.append({"mouse": {"scroll": wheel_steps}})
    elif action_name == "wait":
        wait_seconds = (
            float(args["seconds"])
            if "seconds" in args
            else float(args.get("minutes", 1 / 60)) * 60
        )
    elif action_name == "wait_5_seconds":
        wait_seconds = 5.0
    elif action_name == "take_screenshot":
        result["note"] = "screenshot attached"

    if actions:
        computer.step(actions)
        result["actions"] = actions
        summary = f"{name} -> {actions}"
    elif wait_seconds is not None:
        computer.wait(wait_seconds)
        result["wait_seconds"] = wait_seconds
        summary = f"wait {wait_seconds}s"
    else:
        summary = result.get("error") or result.get("note") or name
    return ExecutedCall(name=name, call_id=call_id, summary=summary, result=result)


def execute_call(computer: Computer, call: dict[str, Any], width: int, height: int) -> ExecutedCall:
    try:
        return _execute_call(computer, call, width, height)
    except InvalidToolArguments:
        name = str(call.get("name", ""))
        call_id = str(call.get("id") or call.get("call_id") or uuid.uuid4().hex)
        return ExecutedCall(
            name=name,
            call_id=call_id,
            summary=INVALID_TOOL_ARGUMENTS,
            result={"ok": False, "error": INVALID_TOOL_ARGUMENTS},
        )


def build_function_results(executed: list[ExecutedCall], png: bytes) -> list[dict[str, Any]]:
    image = screenshot_part(png)
    out = []
    for item in executed:
        entry: dict[str, Any] = {
            "type": "function_result",
            "name": item.name,
            "call_id": item.call_id,
            "result": [
                {"type": "text", "text": json.dumps(item.result, ensure_ascii=False)},
                image,
            ],
        }
        # The API requires the acknowledgement as a top-level field of the
        # function_result; inside the serialized text body it is invisible
        # to the server and the follow-up request is rejected with HTTP 400.
        if item.result.get("safety_acknowledgement"):
            entry["safety_acknowledgement"] = True
        out.append(entry)
    return out


def run(env_url: str, task: str) -> None:
    tracker = CostTracker()
    computer = Computer(env_url)
    interaction: dict[str, Any] | None = None

    try:
        obs = computer.observe()
        width, height = image_size(obs["png"])
        interaction = gemini_request(first_payload(task), tracker, "initial")

        incomplete_streak = 0
        for step in range(MAX_STEPS):
            text = model_text(interaction)
            calls = function_calls(interaction)
            print(
                f"step {step}: interaction={interaction.get('id')} "
                f"status={interaction.get('status')} calls={len(calls)} text={text[:200]!r}",
                file=sys.stderr,
                flush=True,
            )
            if not calls:
                if str(interaction.get("status")) == "incomplete":
                    incomplete_streak += 1
                    if incomplete_streak >= MAX_INCOMPLETE_STREAK:
                        print(
                            f"step {step}: {incomplete_streak} incomplete "
                            "responses in a row without a function call; "
                            "ending task",
                            file=sys.stderr,
                            flush=True,
                        )
                        break
                    print(
                        f"step {step}: incomplete response without function call; continuing",
                        file=sys.stderr,
                        flush=True,
                    )
                    obs = computer.observe()
                    width, height = image_size(obs["png"])
                    interaction_id = str(interaction.get("id") or "")
                    if not interaction_id:
                        print("interaction response had no id; ending task", file=sys.stderr)
                        break
                    interaction = gemini_request(
                        continue_after_incomplete_payload(interaction_id, obs["png"]),
                        tracker,
                        f"step-{step}-incomplete",
                    )
                    continue
                print(f"step {step}: no function call; ending task", file=sys.stderr)
                break

            incomplete_streak = 0
            executed: list[ExecutedCall] = []
            terminal = False
            for call in calls:
                print(
                    "step "
                    f"{step}: call {call.get('name')} "
                    f"args={json.dumps(call.get('arguments', {}), ensure_ascii=False)[:500]}",
                    file=sys.stderr,
                    flush=True,
                )
                item = execute_call(computer, call, width, height)
                executed.append(item)
                print(f"step {step}: {item.summary}", file=sys.stderr, flush=True)
                terminal = terminal or item.terminal
                if terminal:
                    break

            if terminal:
                break

            obs = computer.observe()
            width, height = image_size(obs["png"])
            interaction_id = str(interaction.get("id") or "")
            if not interaction_id:
                print("interaction response had no id; ending task", file=sys.stderr)
                break
            interaction = gemini_request(
                continue_payload(interaction_id, build_function_results(executed, obs["png"])),
                tracker,
                f"step-{step}",
            )
    except CostLimitReached as exc:
        print(f"cost limit reached: {exc}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"agent failed: {exc!r}", file=sys.stderr, flush=True)
        raise
    computer.done()


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
