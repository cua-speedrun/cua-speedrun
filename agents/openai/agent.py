"""OpenAI Responses API Computer Use agent for OSWorld desktop tasks.

Contract: python agent.py <env_url> <task_description>

The model calls the GA Responses API ``computer`` tool. Returned pixel-space
actions are translated to the cua-speedrun Computer schema and executed as one
environment step per computer call, followed by an original-resolution
screenshot. A locked JSON ledger accounts for every API response and enforces
one shared cost cap across init and concurrent task agents.
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
from pathlib import Path
from typing import Any

import requests

from cua_speedrun.client import Computer


MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")
API_URL = os.environ.get("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")
MAX_STEPS = int(os.environ.get("CS_MAX_STEPS", "100"))
MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "16384"))
REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "xhigh")
REQUEST_TIMEOUT = float(os.environ.get("OPENAI_REQUEST_TIMEOUT", "240"))
COMPUTER_TIMEOUT = float(os.environ.get("OPENAI_COMPUTER_TIMEOUT", "300"))
MAX_RETRIES = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))

# Standard API prices in USD per million tokens. Keep this table explicit so a
# run artifact records the effective rates instead of depending on a mutable
# remote pricing page. Versioned snapshots match their model family prefix.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.5": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.6-sol": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.6": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.6-terra": {"input": 2.00, "cached_input": 0.20, "output": 12.00},
    "gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
}

# A reservation prevents concurrent workers from all starting requests just
# below the cap. Defaults cover the maximum configured response plus a full
# context at the selected model's rate; test runs may override it explicitly.
DEFAULT_RESERVES = {
    "gpt-5.4-mini": 0.50,
    "gpt-5.4": 6.00,
    "gpt-5.5": 12.00,
    "gpt-5.6-sol": 12.00,
    "gpt-5.6": 12.00,
    "gpt-5.6-terra": 5.00,
    "gpt-5.6-luna": 1.00,
}

COST_LIMIT_USD = float(os.environ.get("OPENAI_COST_LIMIT_USD", "100.0"))
COST_LEDGER = Path(
    os.environ.get("OPENAI_COST_LEDGER", "/tmp/cua-speedrun-gpt54-cost.json")
)
MISSING_USAGE_FALLBACK_USD = float(
    os.environ.get("OPENAI_MISSING_USAGE_FALLBACK_USD", "0.50")
)


SYSTEM_INSTRUCTION = """You control a sandboxed Ubuntu GNOME desktop for an OSWorld benchmark task.
Use the computer tool to complete the task in the visible desktop. Do not ask
the user for help. The account password is "password" if sudo is needed.
Prefer the intended visible application for GUI, document, browser, image,
email, and editor tasks. Use terminal shortcuts only when the task requests
command-line work, filesystem operations, local app configuration, or a native
application command is clearly the most reliable path. Do not install packages
or download applications unless the task explicitly requires it. Applications
may already be open; otherwise launch them from the GNOME overview. Inspect the
latest screenshot after acting, wait for loading UIs, and scroll or zoom out
before concluding content is absent. Bundle compatible actions in one turn.
You can declare a task infeasible at any point, including after attempting it.
Carefully evaluate whether the task can be completed with the current system,
available applications, permissions, and task requirements. If it cannot be
completed because a required application or dependency is unavailable and
cannot be installed, permissions or system limitations prevent it,
requirements are contradictory or impossible, or another fundamental barrier
makes completion impossible, you MUST call the infeasible tool.
When the visible state fully satisfies the task, stop calling the computer tool
and briefly state that the task is complete. This is an isolated benchmark VM;
the task instruction is the user's authorization for in-scope actions."""


class CostLimitReached(RuntimeError):
    pass


def api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("set OPENAI_API_KEY for the OpenAI API")
    return key


def model_family(model: str) -> str:
    """Resolve aliases and dated snapshots to the pricing family."""
    if model in MODEL_PRICING:
        return model
    for family in sorted(MODEL_PRICING, key=len, reverse=True):
        if model.startswith(family + "-"):
            return family
    raise ValueError(
        f"no pricing configured for {model!r}; set OPENAI_MODEL to one of "
        + ", ".join(sorted(MODEL_PRICING))
    )


def pricing_for_model(model: str = MODEL) -> dict[str, float]:
    return MODEL_PRICING[model_family(model)]


def default_reserve_usd(model: str = MODEL) -> float:
    return DEFAULT_RESERVES[model_family(model)]


COST_RESERVE_USD = float(
    os.environ.get("OPENAI_COST_RESERVE_USD", str(default_reserve_usd()))
)


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
            except (json.JSONDecodeError, OSError):
                state = {}
        else:
            state = {}
        state.setdefault("actual_usd", 0.0)
        state.setdefault("reserved_usd", 0.0)
        state.setdefault("calls", 0)
        state.setdefault("input_tokens", 0)
        state.setdefault("cached_input_tokens", 0)
        state.setdefault("output_tokens", 0)
        state.setdefault("reasoning_tokens", 0)
        state.setdefault("missing_usage_calls", 0)
        state.setdefault("models", {})
        state.setdefault("reservations", {})
        state["cost_limit_usd"] = COST_LIMIT_USD
        yield state
        tmp_path = COST_LEDGER.with_suffix(COST_LEDGER.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tmp_path.replace(COST_LEDGER)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


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
    released = 0.0
    for key, reservation in list(reservations.items()):
        if isinstance(reservation, dict) and pid_is_alive(reservation.get("pid")):
            continue
        removed = reservations.pop(key, {})
        if isinstance(removed, dict):
            released += float(removed.get("amount_usd", 0.0) or 0.0)
    state["reserved_usd"] = max(
        0.0, float(state.get("reserved_usd", 0.0)) - released
    )


def usage_tokens(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    input_details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    return {
        "input": int(usage.get("input_tokens") or 0),
        "cached_input": int(input_details.get("cached_tokens") or 0),
        "output": int(usage.get("output_tokens") or 0),
        # Included in output_tokens, retained separately for diagnostics.
        "reasoning": int(output_details.get("reasoning_tokens") or 0),
    }


def estimate_cost(
    usage: dict[str, Any] | None, model: str = MODEL
) -> tuple[float, dict[str, int]]:
    tokens = usage_tokens(usage)
    rates = pricing_for_model(model)
    cached = min(tokens["input"], tokens["cached_input"])
    uncached = tokens["input"] - cached
    input_multiplier = 1.0
    output_multiplier = 1.0
    family = model_family(model)
    if tokens["input"] > 272_000 and family not in {"gpt-5.4-mini"}:
        input_multiplier = 2.0
        output_multiplier = 1.5
    cost = (
        (uncached * rates["input"] + cached * rates["cached_input"])
        * input_multiplier
        + tokens["output"] * rates["output"] * output_multiplier
    ) / 1_000_000.0
    return cost, tokens


class CostTracker:
    def reserve(self, label: str) -> str:
        reservation_id = f"{os.getpid()}-{uuid.uuid4().hex}"
        with locked_cost_state() as state:
            cleanup_stale_reservations(state)
            committed = float(state["actual_usd"]) + float(state["reserved_usd"])
            if committed + COST_RESERVE_USD > COST_LIMIT_USD:
                raise CostLimitReached(
                    f"cost cap would be exceeded: ${committed:.5f} committed + "
                    f"${COST_RESERVE_USD:.5f} reserve > ${COST_LIMIT_USD:.2f}"
                )
            state["reserved_usd"] = float(state["reserved_usd"]) + COST_RESERVE_USD
            state["reservations"][reservation_id] = {
                "amount_usd": COST_RESERVE_USD,
                "pid": os.getpid(),
                "label": label,
                "model": MODEL,
                "created_at": time.time(),
            }
        return reservation_id

    def release(
        self,
        reservation_id: str,
        usage: dict[str, Any] | None,
        *,
        error: bool = False,
    ) -> float:
        cost, tokens = estimate_cost(usage)
        if usage is None and not error:
            cost = MISSING_USAGE_FALLBACK_USD
        violation: Exception | None = None
        with locked_cost_state() as state:
            reservation = state["reservations"].pop(reservation_id, None)
            reserved_amount = float(
                reservation.get("amount_usd", 0.0) if reservation is not None else 0.0
            )
            if reservation is not None:
                state["reserved_usd"] = max(
                    0.0,
                    float(state["reserved_usd"]) - reserved_amount,
                )
            if error:
                return 0.0
            state["actual_usd"] = float(state["actual_usd"]) + cost
            state["calls"] = int(state["calls"]) + 1
            state["input_tokens"] = int(state["input_tokens"]) + tokens["input"]
            state["cached_input_tokens"] = (
                int(state["cached_input_tokens"]) + tokens["cached_input"]
            )
            state["output_tokens"] = int(state["output_tokens"]) + tokens["output"]
            state["reasoning_tokens"] = (
                int(state["reasoning_tokens"]) + tokens["reasoning"]
            )
            if usage is None:
                state["missing_usage_calls"] = int(state["missing_usage_calls"]) + 1
            per_model = state["models"].setdefault(
                MODEL,
                {"actual_usd": 0.0, "calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0},
            )
            per_model["actual_usd"] = float(per_model["actual_usd"]) + cost
            per_model["calls"] = int(per_model["calls"]) + 1
            per_model["input_tokens"] = int(per_model["input_tokens"]) + tokens["input"]
            per_model["cached_input_tokens"] = int(per_model["cached_input_tokens"]) + tokens["cached_input"]
            per_model["output_tokens"] = int(per_model["output_tokens"]) + tokens["output"]
            total = float(state["actual_usd"])
            if total > COST_LIMIT_USD:
                violation = CostLimitReached(
                    f"actual response pushed total cost to ${total:.5f}, "
                    f"above the ${COST_LIMIT_USD:.2f} cap"
                )
            elif cost > reserved_amount:
                violation = RuntimeError(
                    f"actual request cost ${cost:.5f} exceeded the concurrency "
                    f"reservation ${reserved_amount:.5f}; increase "
                    "OPENAI_COST_RESERVE_USD before running"
                )
        print(
            f"openai cost: model={MODEL} call=${cost:.6f} "
            f"total=${total:.6f}/${COST_LIMIT_USD:.2f} "
            f"tokens=in:{tokens['input']} cached:{tokens['cached_input']} "
            f"out:{tokens['output']} reasoning:{tokens['reasoning']}",
            file=sys.stderr,
            flush=True,
        )
        if violation is not None:
            raise violation
        return cost


def should_retry_error(status_code: int) -> bool:
    return status_code in {408, 409, 429, 500, 502, 503, 504}


def responses_request(
    payload: dict[str, Any], tracker: CostTracker, label: str
) -> dict[str, Any]:
    reservation_id = tracker.reserve(label)
    try:
        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {api_key()}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            request_id = response.headers.get("x-request-id", "")
            if response.status_code < 400:
                data = response.json()
                print(
                    f"openai response: request_id={request_id} "
                    + json.dumps(data, ensure_ascii=False),
                    file=sys.stderr,
                    flush=True,
                )
                tracker.release(reservation_id, data.get("usage"), error=False)
                return data
            last_error = (
                f"HTTP {response.status_code} request_id={request_id} "
                f"{response.text[:2000]}"
            )
            if not should_retry_error(response.status_code):
                break
            retry_after = response.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after else min(2**attempt, 30)
            except ValueError:
                delay = min(2**attempt, 30)
            time.sleep(max(0.0, min(delay, 60.0)))
        raise RuntimeError(f"OpenAI request failed: {last_error}")
    except BaseException:
        # The API does not bill failed HTTP requests, and a successful response
        # is charged before it is returned above.
        with locked_cost_state() as state:
            if reservation_id in state.get("reservations", {}):
                reservation = state["reservations"].pop(reservation_id)
                state["reserved_usd"] = max(
                    0.0,
                    float(state["reserved_usd"])
                    - float(reservation.get("amount_usd", 0.0)),
                )
        raise


def screenshot_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def tools() -> list[dict[str, Any]]:
    return [
        {"type": "computer"},
        {
            "type": "function",
            "name": "infeasible",
            "description": "Signal that the task is infeasible.",
            "parameters": {"type": "object", "properties": {}},
        },
    ]


def base_payload() -> dict[str, Any]:
    return {
        "model": MODEL,
        "instructions": SYSTEM_INSTRUCTION,
        "tools": tools(),
        "parallel_tool_calls": False,
        "reasoning": {"effort": REASONING_EFFORT, "summary": "concise"},
        "truncation": "auto",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


def first_payload(task: str, png: bytes) -> dict[str, Any]:
    payload = base_payload()
    payload["input"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Complete this OSWorld desktop task in the sandboxed VM.\n\nTask: " + task,
                },
                {"type": "input_image", "image_url": screenshot_url(png), "detail": "original"},
            ],
        }
    ]
    return payload


def continue_payload(
    previous_response_id: str,
    call_id: str,
    png: bytes,
    pending_safety_checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = base_payload()
    payload["previous_response_id"] = previous_response_id
    output: dict[str, Any] = {
        "type": "computer_call_output",
        "call_id": call_id,
        "output": {
            "type": "computer_screenshot",
            "image_url": screenshot_url(png),
            "detail": "original",
        },
    }
    # OSWorld is an isolated benchmark and the task itself supplies the user's
    # authorization, matching the upstream reference agent's behavior.
    if pending_safety_checks:
        output["acknowledged_safety_checks"] = pending_safety_checks
    payload["input"] = [output]
    return payload


def incomplete_payload(task: str, png: bytes) -> dict[str, Any]:
    """Restart an action turn after reasoning exhausted its output budget.

    The GA computer tool accepts user image input only at the start of a new
    response chain; after ``previous_response_id`` it expects a
    ``computer_call_output``. An incomplete reasoning-only response has no
    computer call to answer, so recovery must start a fresh chain with the
    original task and latest screenshot.
    """
    payload = base_payload()
    payload["input"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "The previous reasoning turn reached its output limit before issuing an action. "
                        "Continue this OSWorld task from the current screenshot and issue the next "
                        "computer action, or finish if the task is complete.\n\nTask: " + task
                    ),
                },
                {"type": "input_image", "image_url": screenshot_url(png), "detail": "original"},
            ],
        }
    ]
    return payload


def computer_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in response.get("output") or []
        if isinstance(item, dict) and item.get("type") == "computer_call"
    ]


def function_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in response.get("output") or []
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]


def response_text(response: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                chunks.append(str(part.get("text", "")))
    return "\n".join(chunks)


def image_size(png: bytes) -> tuple[int, int]:
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
        raise ValueError("observation is not a valid PNG")
    return struct.unpack(">II", png[16:24])


def normalize_key(key: Any) -> str:
    raw = str(key).strip()
    compact = raw.lower().replace("_", "").replace("-", "").replace(" ", "")
    aliases = {
        "enter": "Return", "return": "Return", "esc": "Escape", "escape": "Escape",
        "ctrl": "ctrl", "control": "ctrl", "cmd": "super", "command": "super",
        "win": "super", "super": "super", "alt": "alt", "option": "alt",
        "shift": "shift", "tab": "Tab", "backspace": "BackSpace", "delete": "Delete",
        "space": "space", "spacebar": "space", "pagedown": "pagedown", "pageup": "pageup",
        "home": "Home", "end": "End", "insert": "insert", "capslock": "capslock",
        "arrowleft": "Left", "left": "Left", "arrowright": "Right", "right": "Right",
        "arrowup": "Up", "up": "Up", "arrowdown": "Down", "down": "Down",
    }
    for suffix in ("left", "right", "l", "r"):
        if compact in {"ctrl" + suffix, "control" + suffix}:
            return "ctrl"
        if compact in {"alt" + suffix, "option" + suffix}:
            return "alt"
        if compact == "shift" + suffix:
            return "shift"
        if compact in {"super" + suffix, "win" + suffix, "meta" + suffix}:
            return "super"
    return aliases.get(compact, raw)


def point(value: Any, width: int, height: int) -> list[int]:
    if isinstance(value, dict):
        x, y = value.get("x"), value.get("y")
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        x, y = value[0], value[1]
    else:
        raise ValueError(f"invalid point: {value!r}")
    return [
        max(0, min(width - 1, int(float(x)))),
        max(0, min(height - 1, int(float(y)))),
    ]


def xy(action: dict[str, Any], width: int, height: int) -> list[int]:
    return point({"x": action.get("x"), "y": action.get("y")}, width, height)


def modifier_actions(keys: Any, *, down: bool) -> list[dict[str, Any]]:
    normalized = [normalize_key(key) for key in (keys or [])]
    if not down:
        normalized.reverse()
    field = "key_down" if down else "key_up"
    return [{"keyboard": {field: key}} for key in normalized]


def scroll_steps(pixels: Any) -> int:
    value = float(pixels or 0)
    if value == 0:
        return 0
    return int(math.copysign(max(1, min(30, math.ceil(abs(value) / 120))), value))


def translate_action(
    action: dict[str, Any], width: int, height: int
) -> tuple[list[dict[str, Any]], float | None]:
    """Translate one GA computer action to env actions and optional wait."""
    kind = str(action.get("type", ""))
    keys = action.get("keys") or []
    out: list[dict[str, Any]] = []
    if kind in {"click", "double_click"}:
        button = str(action.get("button", "left")).lower()
        click_names = {
            "left": "left_click",
            "right": "right_click",
            "wheel": "middle_click",
            "middle": "middle_click",
        }
        if button not in click_names:
            raise ValueError(f"unsupported mouse button: {button}")
        out.extend(modifier_actions(keys, down=True))
        if kind == "double_click" and button == "left":
            out.append({"mouse": {"double_click": xy(action, width, height)}})
        else:
            click = click_names[button]
            out.append({"mouse": {click: xy(action, width, height)}})
            if kind == "double_click":
                out.append({"mouse": {click: xy(action, width, height)}})
        out.extend(modifier_actions(keys, down=False))
    elif kind == "move":
        out.extend(modifier_actions(keys, down=True))
        out.append({"mouse": {"move": xy(action, width, height)}})
        out.extend(modifier_actions(keys, down=False))
    elif kind == "drag":
        path = action.get("path") or []
        points = [point(p, width, height) for p in path]
        if len(points) < 2:
            raise ValueError("drag requires at least two path points")
        out.extend(modifier_actions(keys, down=True))
        out.append({"mouse": {"move": points[0]}})
        out.append({"mouse": {"buttons": {"left_down": True}}})
        out.extend({"mouse": {"move": drag_point}} for drag_point in points[1:])
        out.append({"mouse": {"buttons": {"left_up": True}}})
        out.extend(modifier_actions(keys, down=False))
    elif kind == "scroll":
        normalized_keys = [normalize_key(key) for key in keys]
        out.extend(modifier_actions(keys, down=True))
        if action.get("x") is not None and action.get("y") is not None:
            out.append({"mouse": {"move": xy(action, width, height)}})
        vertical = scroll_steps(action.get("scroll_y", action.get("delta_y", 0)))
        horizontal = scroll_steps(action.get("scroll_x", action.get("delta_x", 0)))
        if vertical:
            out.append({"mouse": {"scroll": vertical}})
        if horizontal:
            if "shift" not in normalized_keys:
                out.append({"keyboard": {"key_down": "shift"}})
            out.append({"mouse": {"scroll": horizontal}})
            if "shift" not in normalized_keys:
                out.append({"keyboard": {"key_up": "shift"}})
        out.extend(modifier_actions(keys, down=False))
    elif kind == "type":
        out.append({"keyboard": {"text": str(action.get("text", ""))}})
    elif kind == "keypress":
        pressed = action.get("keys") or ([action["key"]] if action.get("key") else [])
        normalized = [normalize_key(key) for key in pressed]
        if normalized:
            out.append({"keyboard": {"keys": normalized}})
    elif kind == "wait":
        milliseconds = float(action.get("ms", 1000) or 1000)
        return [], max(0.1, min(30.0, milliseconds / 1000.0))
    elif kind == "screenshot":
        return [], None
    else:
        raise ValueError(f"unsupported computer action: {kind}")
    return out, None


def execute_actions(
    computer: Computer, actions: list[dict[str, Any]], width: int, height: int
) -> dict[str, Any]:
    env_batch: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        env_actions, wait_seconds = translate_action(action, width, height)
        if wait_seconds is not None:
            env_actions.append({"action": "wait", "time": wait_seconds})
        elif action.get("type") == "screenshot":
            env_actions.append({"action": "screenshot"})
        print(
            f"computer action {index + 1}/{len(actions)}: "
            f"{json.dumps(action, ensure_ascii=False)} -> "
            f"{json.dumps(env_actions, ensure_ascii=False)}",
            file=sys.stderr,
            flush=True,
        )
        env_batch.extend(env_actions)
    return computer.step(env_batch)


def execute_function_call(
    computer: Computer, call: dict[str, Any]
) -> dict[str, Any]:
    name = str(call.get("name") or "")
    if name != "infeasible":
        raise RuntimeError(f"unsupported function call: {name!r}")
    print("model declared task infeasible", file=sys.stderr, flush=True)
    return computer.step([{"action_type": "FAIL"}])


def run(env_url: str, task: str) -> None:
    tracker = CostTracker()
    computer = Computer(env_url, timeout_sec=COMPUTER_TIMEOUT)
    try:
        observation = computer.observe()
        width, height = image_size(observation["png"])
        response = responses_request(first_payload(task, observation["png"]), tracker, "initial")

        for step in range(MAX_STEPS):
            calls = computer_calls(response)
            custom_calls = function_calls(response)
            text = response_text(response)
            print(
                f"step {step}: response={response.get('id')} status={response.get('status')} "
                f"computer_calls={len(calls)} function_calls={len(custom_calls)} "
                f"text={text[:500]!r}",
                file=sys.stderr,
                flush=True,
            )
            if custom_calls:
                if len(custom_calls) != 1 or calls:
                    raise RuntimeError(
                        "expected one tool call, got "
                        f"{len(calls)} computer and {len(custom_calls)} function calls"
                    )
                execute_function_call(computer, custom_calls[0])
                break
            if not calls:
                if response.get("status") == "incomplete":
                    if step + 1 >= MAX_STEPS:
                        print(
                            f"maximum model call count reached ({MAX_STEPS})",
                            file=sys.stderr,
                            flush=True,
                        )
                        break
                    observation = computer.observe()
                    width, height = image_size(observation["png"])
                    response = responses_request(
                        incomplete_payload(task, observation["png"]),
                        tracker,
                        f"step-{step}-incomplete",
                    )
                    continue
                break
            if len(calls) != 1:
                raise RuntimeError(f"expected one computer_call, got {len(calls)}")
            call = calls[0]
            actions = call.get("actions")
            if actions is None and call.get("action") is not None:
                actions = [call["action"]]
            if not isinstance(actions, list):
                raise RuntimeError("computer_call actions are missing")
            step_result = execute_actions(computer, actions, width, height)
            if step_result.get("done"):
                print("environment finalized after model action", file=sys.stderr, flush=True)
                break
            if step + 1 >= MAX_STEPS:
                print(
                    f"maximum model call count reached ({MAX_STEPS})",
                    file=sys.stderr,
                    flush=True,
                )
                break
            observation = computer.observe()
            width, height = image_size(observation["png"])
            response_id = str(response.get("id") or "")
            call_id = str(call.get("call_id") or "")
            if not response_id or not call_id:
                raise RuntimeError("computer response is missing response/call id")
            response = responses_request(
                continue_payload(
                    response_id,
                    call_id,
                    observation["png"],
                    call.get("pending_safety_checks") or [],
                ),
                tracker,
                f"step-{step}",
            )
    except CostLimitReached as exc:
        print(f"cost limit reached: {exc}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"agent failed: {exc!r}", file=sys.stderr, flush=True)
    finally:
        try:
            computer.done()
        except Exception as exc:
            print(f"computer.done failed: {exc!r}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
