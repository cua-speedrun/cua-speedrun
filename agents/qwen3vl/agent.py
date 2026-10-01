"""agent.py: a Qwen3-VL computer-use agent. Runs once per task, timed.

Contract: python agent.py <env_url> <task_description>

Ported from gym-anything's Qwen3VLFixedAgent (agents/agents/qwen3vlfixed.py):
the osworld-aligned design. The model is prompted with a `computer_use`
tool-call schema, emits coordinates in a 1000x1000 grid, and keeps a short
window of previous screenshots and responses as multi-turn context. This
file is self-contained (only cua_speedrun.client + requests + Pillow) so it
runs unchanged in the agent sandbox.

The model server is started by init.py; this script only talks to localhost.
"""

import base64
import io
import json
import os
import sys
from collections import Counter

import requests
from PIL import Image

from cua_speedrun.client import Computer

# VLLM_URL can override the local model-server endpoint.
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
MAX_STEPS = int(os.environ.get("CS_MAX_STEPS", "100"))
HISTORY_N = 4
TEMPERATURE = float(os.environ.get("CS_TEMPERATURE", "1.0"))
TOP_P = 0.95
TOP_K = 20
MAX_TOKENS = 1500

# The model reasons in a 1000x1000 grid; coordinates are scaled to the real
# screen. GRID is that reference edge.
GRID = 1000.0


# ---- prompt (osworld-aligned computer_use tool) ---------------------------

_TOOLS_DEF = {
    "type": "function",
    "function": {
        "name_for_human": "computer_use",
        "name": "computer_use",
        "description": (
            "Use a mouse and keyboard to interact with a computer, and take "
            "screenshots.\n"
            "* This is an interface to a desktop GUI. You do not have access to a "
            "terminal or applications menu. You must click on desktop icons to "
            "start applications.\n"
            "* Some applications may take time to start or process actions, so you "
            "may need to wait and take successive screenshots to see the results "
            "of your actions.\n"
            "* The screen's resolution is 1000x1000.\n"
            "* Whenever you intend to click on an element like an icon, you should "
            "consult a screenshot to determine the coordinates of the element "
            "before moving the cursor.\n"
            "* Make sure to click any buttons, links, icons with the cursor tip in "
            "the center of the element."),
        "parameters": {
            "properties": {
                "action": {
                    "description": (
                        "The action to perform. The available actions are:\n"
                        "* `key`: Performs key down presses on the arguments in "
                        "order, then releases in reverse order.\n"
                        "* `type`: Type a string of text on the keyboard.\n"
                        "* `mouse_move`: Move the cursor to (x, y).\n"
                        "* `left_click`: Click the left mouse button at (x, y).\n"
                        "* `left_click_drag`: Click and drag to (x, y).\n"
                        "* `right_click`: Right-click at (x, y).\n"
                        "* `double_click`: Double-click at (x, y).\n"
                        "* `scroll`: Scroll the mouse wheel.\n"
                        "* `wait`: Wait a number of seconds.\n"
                        "* `terminate`: Terminate the task and report status."),
                    "enum": ["key", "type", "mouse_move", "left_click",
                             "left_click_drag", "right_click", "double_click",
                             "scroll", "wait", "terminate"],
                    "type": "string",
                },
                "keys": {"description": "Required only by `action=key`.", "type": "array"},
                "text": {"description": "Required only by `action=type`.", "type": "string"},
                "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                "coordinate2": {"description": "The x2,y2 for a drag end. Required by `left_click_drag`.", "type": "array"},
                "pixels": {"description": "Signed pixels: negative scrolls down, positive scrolls up.", "type": "number"},
                "time": {"description": "The seconds to wait.", "type": "number"},
                "status": {"description": "Task status.", "type": "string",
                           "enum": ["success", "failure"]},
            },
            "required": ["action"],
            "type": "object",
        },
    },
}

SYSTEM = (
    "# Tools\n\nYou may call one or more functions to assist with the user "
    "query.\n\nYou are provided with function signatures within "
    "<tools></tools> XML tags:\n<tools>\n" + json.dumps(_TOOLS_DEF) + "\n</tools>"
    "\n\nFor each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n\n'
    "# Response format\n\nResponse format for every step:\n"
    "1) Action: a short imperative describing what to do in the UI.\n"
    "2) A single <tool_call>...</tool_call> block containing only the JSON.\n\n"
    "Rules:\n- Output exactly in the order: Action, <tool_call>.\n"
    "- Be brief: one sentence for Action.\n- Do not output anything else.\n"
    "- If finishing, use action=terminate in the tool call.")


# ---- coordinate scaling + response parsing (ported) -----------------------

def _scale(x, y, ratio):
    return int(x * ratio[0]), int(y * ratio[1])


def parse_response(response, ratio):
    """Parse a Qwen3-VL <tool_call> response into gym-anything action dicts
    and metadata. `ratio` scales the model's 1000-grid coordinates to pixels."""
    try:
        return _parse_response(response, ratio)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return {"actions": [], "conclusion": f"invalid action: {exc}",
                "is_terminal": False, "wait_time": 1.0}


def _parse_response(response, ratio):
    if not response or not isinstance(response, str):
        return {"actions": [], "conclusion": "empty response", "is_terminal": False}

    if "</think>" in response:
        response = response.split("</think>")[1]

    tool_calls = response.split("<tool_call>")[1:]
    if len(tool_calls) > 1:
        parsed = [parse_response("<tool_call>" + call, ratio) for call in tool_calls]
        actions = [action for item in parsed for action in item["actions"]]
        return {
            "actions": actions,
            "conclusion": " then ".join(item["conclusion"] for item in parsed),
            "is_terminal": not actions and any(item["is_terminal"] for item in parsed),
            "wait_time": None,
        }

    if "<tool_call>" in response and "</tool_call>" in response:
        action = response.split("<tool_call>")[-1].split("</tool_call>")[0]
    else:
        try:
            action = ('{"name": "computer_use"'
                      + response.split('{"name": "computer_use"')[1].split("}}")[0]
                      + "}}")
        except Exception:
            return {"actions": [], "conclusion": "cannot parse; waiting",
                    "is_terminal": False, "wait_time": 1.0}

    conclusion = ""
    for line in response.split("\n"):
        if "Action:" in line:
            conclusion = line.split("Action:")[-1].strip()

    try:
        parsed = json.loads(action.strip("\n"))
        aj = parsed.get("arguments", parsed)
        kind = aj["action"]
    except Exception as exc:
        return {"actions": [], "conclusion": f"parse error: {exc}",
                "is_terminal": False, "wait_time": 1.0}

    meta = {"conclusion": conclusion or kind, "is_terminal": False, "wait_time": None}

    if kind == "key":
        actions = [{"keyboard": {"keys": aj.get("keys", [])}}]
    elif kind == "type":
        actions = []
        if aj.get("clear"):
            actions.append({"keyboard": {"keys": ["ctrl", "a"]}})
        actions.append({"keyboard": {"text": aj.get("text", "")}})
        if aj.get("enter"):
            actions.append({"keyboard": {"keys": ["Return"]}})
    elif kind == "mouse_move":
        x, y = _scale(*aj["coordinate"], ratio)
        actions = [{"mouse": {"move": [x, y]}}]
    elif kind in ("left_click", "click"):
        x, y = _scale(*aj["coordinate"], ratio)
        actions = [{"mouse": {"left_click": [x, y]}}]
    elif kind == "right_click":
        x, y = _scale(*aj["coordinate"], ratio)
        actions = [{"mouse": {"right_click": [x, y]}}]
    elif kind == "double_click":
        x, y = _scale(*aj["coordinate"], ratio)
        actions = [{"mouse": {"double_click": [x, y]}}]
    elif kind in ("left_click_drag", "drag"):
        points = [list(_scale(*aj[name], ratio))
                  for name in ("coordinate", "coordinate2") if name in aj]
        if not points:
            raise ValueError("drag needs a destination")
        actions = [{"mouse": {"left_click_drag": points}}]
    elif kind == "scroll":
        pixels = float(aj.get("pixels", aj.get("scroll", 0)))
        amount = 0 if pixels == 0 else (-1 if pixels > 0 else 1) * max(
            1, round(abs(pixels) / 100)
        )
        amount = max(-10, min(10, amount))
        if "coordinate" in aj:
            x, y = _scale(*aj["coordinate"], ratio)
            actions = [{"mouse": {"move": [x, y]}}, {"mouse": {"scroll": amount}}]
        else:
            actions = [{"mouse": {"scroll": amount}}]
    elif kind == "wait":
        actions, meta["wait_time"] = [], aj.get("time", 1.0)
    elif kind == "terminate":
        actions, meta["is_terminal"] = [], True
    else:
        actions = []

    return {"actions": actions, **meta}


# ---- agent loop -----------------------------------------------------------

def ask(messages):
    resp = requests.post(
        f"{VLLM_URL}/v1/chat/completions",
        json={"model": MODEL, "messages": messages, "max_tokens": MAX_TOKENS,
              "temperature": TEMPERATURE, "top_p": TOP_P,
              # vLLM's OpenAI-compatible server accepts top_k as a native
              # top-level extension (extra_body is an OpenAI-SDK-only concept).
              "top_k": TOP_K},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def build_messages(task, screenshots, responses, history):
    """Instruction + a window of the last HISTORY_N (screenshot, response)
    pairs + the current screenshot, mirroring the reference agent."""
    prev = "\n".join(f"Step {i+1}: {h}" for i, h in enumerate(history)) or "None"
    instruction = (f"Please generate the next move according to the UI "
                   f"screenshot, instruction and previous actions.\n\n"
                   f"Instruction: {task}\n\nPrevious actions:\n{prev}")

    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}]
    n = min(HISTORY_N, len(responses))
    if n > 0:
        hist_shots = screenshots[-n - 1:-1]
        hist_resp = responses[-n:]
        for idx in range(n):
            content = [{"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{hist_shots[idx]}"}}]
            if idx == 0:
                content.append({"type": "text", "text": instruction})
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant",
                             "content": [{"type": "text", "text": hist_resp[idx]}]})
        messages.append({"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{screenshots[-1]}"}}]})
    else:
        messages.append({"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{screenshots[-1]}"}},
            {"type": "text", "text": instruction}]})
    return messages


def run(env_url, task):
    computer = Computer(env_url)
    screenshots, responses, history = [], [], []
    action_signatures = []

    for step in range(MAX_STEPS):
        obs = computer.observe()
        png = obs["png"]
        w, h = Image.open(io.BytesIO(png)).size
        ratio = (w / GRID, h / GRID)
        screenshots.append(base64.b64encode(png).decode())

        try:
            reply = ask(build_messages(task, screenshots, responses, history))
        except Exception as exc:
            print(f"model call failed: {exc!r}", file=sys.stderr)
            break
        responses.append(reply)
        # The raw reply is the evidence when parsing fails; without it a
        # failed run cannot tell a format mismatch from an empty response.
        print(f"step {step} raw ({len(reply) if reply else 0} chars): "
              f"{(reply or '')!r}", file=sys.stderr)

        parsed = parse_response(reply, ratio)
        history.append(parsed["conclusion"])
        print(f"step {step}: {parsed['conclusion']} -> {parsed['actions']}",
              file=sys.stderr)

        if parsed["is_terminal"]:
            break
        signature = json.dumps(parsed["actions"], sort_keys=True)
        action_signatures.append(signature)
        if _repeating_cycle(action_signatures):
            print("loop detected; ending task so verifier can score state",
                  file=sys.stderr)
            break
        if parsed.get("wait_time"):
            computer.wait(parsed["wait_time"])
        elif parsed["actions"]:
            computer.step(parsed["actions"])

    computer.done()


def _repeating_cycle(signatures):
    if len(signatures) < 12:
        return False
    recent = signatures[-12:]
    if len(set(recent)) == 1:
        return True
    for size in (2, 3, 4):
        cycle = recent[-size:]
        if cycle * (len(recent) // size) == recent:
            return True
    return Counter(recent).most_common(1)[0][1] >= 9


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
