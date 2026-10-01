"""GLM AutoGLM-V computer-use agent for OSWorld-style desktop tasks.

Contract: python agent.py <env_url> <task_description>

Port of the official OSWorld AutoGLM-V agent (xlang-ai/OSWorld
mm_agents/autoglm_v plus its runner scripts/python/run_multienv_autoglm_v.py,
at commit b7db4d8c85d9e95e0b1db44de5bec954cf37f0cf; autoglm_v is the current
vision variant -- mm_agents/autoglm is the older a11y-tree-only scaffold)
onto the cua-speedrun Computer client.

Model and endpoint
------------------
The reference runner drives a self-hosted vLLM checkpoint (default model name
"autoglm-os") through an OpenAI-compatible /chat/completions endpoint chosen
by OPENAI_BASE_URL. Its prompt format (``glm41v_format``: <think>...</think>
<answer>```python ...```</answer>) is the GLM-4.1V/GLM-4.5V vision family
format, so this template defaults to Zhipu's hosted vision model
``glm-4.5v`` -- the newest hosted model this scaffold's format supports.

About "GLM 5.2": ``glm-5.2`` is Zhipu's text-only agentic-coding line, not a
vision model; this computer-use scaffold requires a vision model (screenshot
observations), so glm-4.5v remains the default. To try another id anyway,
set GLM_MODEL (e.g. GLM_MODEL=glm-5.2) -- everything else is unchanged.

Endpoint: GLM_BASE_URL, default https://open.bigmodel.cn/api/paas/v4 (Zhipu
open platform; the international mirror is https://api.z.ai/api/paas/v4).
API key: GLM_API_KEY (ZHIPUAI_API_KEY also accepted).

Faithfully ported from the reference:
  * The procedural-memory prompt: setup prompt with the dynamic observation
    list, the ``Class Agent:`` function-definition block generated from the
    agent-action signatures/docstrings via inspect, and the note prompt with
    the 0-1000 relative-coordinate hint and client password -- all with the
    glm41v <think>/<answer> output format.
  * ``parse_code_from_string``: fenced ```python``` one-liner extraction with
    WAIT/DONE/FAIL command splitting.
  * History: text-only "**Environment State (Omitted)**" user turns carrying
    the previous action result (2000-char cap), assistant responses capped at
    1500 chars, last 30 turns.
  * Sampling exactly as the reference runner sets it: temperature 0.4,
    top_p 0.5, max_tokens 2048, stop ["<|user|>", "<|observation|>",
    "</answer>"], stream False.
  * Step budget 30, the canonical OSWorld default (GLM_MAX_STEPS /
    CS_MAX_STEPS override).
  * Retry: constant 0.1 s backoff on rate limits and connection errors (the
    reference uses backoff.constant with no cap; this port caps attempts at
    GLM_MAX_RETRIES, default 20). A failed generation yields an empty
    response, which is recorded as "Invalid action" and the episode continues
    -- same as the reference's predict().
  * Coordinates are 0-1000 normalized and rescaled with round() to the
    observed screenshot size, read from the PNG IHDR chunk.
  * ``Agent.exit(success=False)`` (and FAIL) -> the env FAIL action.

Deliberate deviations (environment constraints, not behavior choices):
  * The gateway only exposes screenshots and GUI actions, so the OSWorld
    obs extras (window list, current app, app info, in-VM execution results)
    are reported as "None" and the previous-action result is synthesized from
    the same "... Success" strings the reference's generated pyautogui code
    prints.
  * The reference's vLLM sampling passthroughs (skip_special_tokens False,
    include_stop_str_in_output True) are sent only to a self-hosted vLLM
    endpoint. They are not documented for hosted Zhipu or OpenRouter, so on
    those hosts the body omits them entirely; ``GLM_VLLM_FLAGS`` forces the
    decision either way. See ``is_vllm_endpoint``.
  * App-specific tool packages (CalcTools etc.), ``Agent.open_app`` (spawns a
    shell command) and ``Agent.switch_window`` (wmctrl) need an in-VM exec
    path that the sandbox does not have; they are omitted from the generated
    function list so the model is never offered them.
  * Screenshots are sent at the observed resolution rather than the
    reference's 1280x720. The agent sandbox installs only ``requests``
    (src/cua_speedrun/remote/agent_runtime.py PYTHON_PACKAGES), so there is no
    imaging library to resample with, and ``Computer.observe()`` takes no size
    argument, so the gateway cannot deliver a scaled frame either. Rather than
    ship a resize that cannot run, the frames go out full-size: this costs
    input tokens and is the one fidelity gap that is not closed. Coordinates
    are resolution-free (0-1000), so accuracy is unaffected.
"""

import ast
import base64
import inspect
import json
import os
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import requests

from cua_speedrun.client import Computer

MODEL = os.environ.get("GLM_MODEL", "glm-4.5v")
BASE_URL = os.environ.get(
    "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
).rstrip("/")
API_URL = f"{BASE_URL}/chat/completions"
# Step cap. GLM_MAX_STEPS is the user-settable knob; CS_MAX_STEPS is a
# harness-internal fallback (kept for parity with the other templates but not
# user-forwardable, so it stays a no-op in normal submissions).
MAX_STEPS = int(
    os.environ.get("GLM_MAX_STEPS") or os.environ.get("CS_MAX_STEPS") or "30"
)
REQUEST_TIMEOUT = float(os.environ.get("GLM_HTTP_TIMEOUT", "60"))
# Reference: @backoff.on_exception(backoff.constant, (RateLimitError,
# APIConnectionError), interval=0.1) -- unbounded; capped here.
MAX_RETRIES = int(os.environ.get("GLM_MAX_RETRIES", "20"))
RETRY_INTERVAL = 0.1
MAX_TOKENS = int(os.environ.get("GLM_MAX_TOKENS", "2048"))
TEMPERATURE = float(os.environ.get("GLM_TEMPERATURE", "0.4"))
TOP_P = float(os.environ.get("GLM_TOP_P", "0.5"))
STOP = ["<|user|>", "<|observation|>", "</answer>"]
MAX_TURNS = int(os.environ.get("GLM_MAX_TURNS", "30"))
CLIENT_PASSWORD = os.environ.get("GLM_CLIENT_PASSWORD", "password")
# Env-gateway HTTP timeout (the client default of 120s is tight for heavy envs).
ENV_HTTP_TIMEOUT = float(os.environ.get("GLM_ENV_HTTP_TIMEOUT", "600"))
GRID = 1000.0

WAIT_SECONDS = 2.0

# pyautogui-style key names (what the model emits) -> gym-anything keyboard
# vocabulary.
ENV_KEY_MAP = {
    "enter": "Return",
    "return": "Return",
    "esc": "Escape",
    "escape": "Escape",
    "tab": "Tab",
    "backspace": "BackSpace",
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
    "pagedown": "pagedown",
    "pgdn": "pagedown",
    "insert": "insert",
    "win": "super",
    "winleft": "super",
    "super": "super",
    "cmd": "super",
    "command": "super",
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "option": "alt",
    "shift": "shift",
    "capslock": "capslock",
}


def api_key() -> str:
    key = os.environ.get("GLM_API_KEY") or os.environ.get("ZHIPUAI_API_KEY")
    if not key:
        raise RuntimeError("set GLM_API_KEY (or ZHIPUAI_API_KEY) for the GLM API")
    return key


def agent_action(func):
    func.is_agent_action = True
    return func


class GroundingAgent:
    """The GUI subset of the reference GroundingAgent. Signatures and
    docstrings are verbatim (they are rendered into the system prompt);
    the bodies live in the translator below, which emits env action dicts
    instead of pyautogui code."""

    @classmethod
    @agent_action
    def click(
        cls,
        coordinate: List,
        num_clicks: int = 1,
        button_type: str = "left",
    ):
        """
        Click on the element

        Args:
            coordinate (List): [x, y], coordinate of the element to click on
            num_clicks (int): number of times to click the element
            button_type (str): which mouse button to press ("left", "middle", or "right")
        """

    @classmethod
    @agent_action
    def type(
        cls,
        coordinate: Optional[List] = None,
        text: str = "",
        overwrite: bool = False,
        enter: bool = False,
    ):
        """
        Type text into the element

        Args:
            coordinate (List): [x, y], coordinate of the element to type into. If None, typing starts at current cursor location
            text (str): the text to type
            overwrite (bool): True to overwrite existing text, False otherwise
            enter (bool): True to press enter after typing, False otherwise
        """

    @classmethod
    @agent_action
    def drag_and_drop(cls, drag_from_coordinate: List, drop_on_coordinate: List):
        """
        Drag element1 and drop it on element2

        Args:
            drag_from_coordinate (List): [x, y], coordinate of element to drag
            drop_on_coordinate (List): [x, y], coordinate of element to drop on
        """

    @classmethod
    @agent_action
    def scroll(cls, coordinate: List, direction: str):
        """
        Scroll the element in the specified direction

        Args:
            coordinate (List): [x, y], coordinate of the element to scroll in
            direction (str): the direction to scroll ("up" or "down")
        """

    @classmethod
    @agent_action
    def hotkey(cls, keys: List):
        """
        Press a hotkey combination

        Args:
            keys (List): the keys to press in combination (e.g. ['ctrl', 'c'] for copy, ['prtsc'] for screenshot)
        """

    @classmethod
    @agent_action
    def quote(cls, content: str):
        """
        Quote information from the current page for memory

        Args:
            content (str): text summarized or copied from the page for later operation
        """

    @classmethod
    @agent_action
    def wait(cls):
        """
        Wait for a while

        """

    @classmethod
    @agent_action
    def exit(cls, success: bool):
        """
        End the current task

        Args:
            success (bool): True if successfully finish a task, False otherwise
        """


# Verbatim prompt scaffolding from the reference procedural_memory.py.
SETUP_PROMPT = """You are a GUI operation agent. You will be given a task and your action history, with current observation ({observation_list}). You should help me control the computer, output the best action step by step to accomplish the task.
You should first generate a plan, reflect on the current observation, then generate actions to complete the task in python-style pseudo code using the predefined functions.

* Output Format:
{format_hint}"""

FUNC_DEF_TEMPLATE = """* Available Functions:
```python
{class_content}
```"""

NOTE_PROMPT = """* Note:
- Your code should only be wrapped in ```python```.
- Only **ONE-LINE-OF-CODE** at a time.
- Each code block is context independent, and variables from the previous round cannot be used in the next round.
{relative_coordinate_hint}- Return with `Agent.exit(success=True)` immediately after the task is completed.
- The computer's environment is Linux, e.g., Desktop path is '/home/user/Desktop'
- My computer's password is '{client_password}', feel free to use it when you need sudo rights"""

GLM41V_FORMAT_HINT = (
    "<think>\n{**YOUR-PLAN-AND-THINKING**}</think>\n"
    "<answer>```python\n{**ONE-LINE-OF-CODE**}\n```</answer>"
)

RELATIVE_COORDINATE_HINT = (
    "- The coordinate [x, y] should be normalized to 0-1000, which usually "
    "should be the center of a specific target element.\n"
)


def construct_procedural_memory() -> Tuple[str, str, str]:
    """Reference Prompt.construct_procedural_memory with with_image=True,
    with_atree=False, relative_coordinate=True, glm41v_format=True and no
    per-app tool package (the sandbox has no in-VM exec path)."""
    agent_class_content = "Class Agent:"
    for attr_name in dir(GroundingAgent):
        attr = getattr(GroundingAgent, attr_name)
        if callable(attr) and hasattr(attr, "is_agent_action"):
            signature = inspect.signature(attr)
            agent_class_content += f"""
    def {attr_name}{signature}:
        '''{attr.__doc__}'''
    """

    func_def_prompt = FUNC_DEF_TEMPLATE.format(class_content=agent_class_content.strip())

    observation_list = "screenshot, current app name, app info, last action result"
    setup_prompt_formatted = SETUP_PROMPT.format(
        observation_list=observation_list, format_hint=GLM41V_FORMAT_HINT
    )
    note_prompt_formatted = NOTE_PROMPT.format(
        relative_coordinate_hint=RELATIVE_COORDINATE_HINT,
        client_password=CLIENT_PASSWORD,
    )
    return setup_prompt_formatted, func_def_prompt, note_prompt_formatted


def system_message(instruction: str) -> str:
    setup_prompt, func_def_prompt, note_prompt = construct_procedural_memory()
    message = setup_prompt + "\n\n" + func_def_prompt + "\n\n" + note_prompt
    message += "\n\n**IMPORTANT** You are asked to complete the following task: {}".format(
        instruction
    )
    return message


def observation_text(last_result: str) -> str:
    """The reference per-turn observation text. The OSWorld-only obs extras
    (window list / current app / app info) are not exposed by this gateway
    and are reported as None."""
    last_result = last_result.strip() if last_result else "None"
    last_result = last_result[:2000] + "..." if len(last_result) > 2000 else last_result
    return (
        "* Apps: None\n\n* Current App: None\n\n* App Info: None\n\n"
        "* Previous Action Result: {}".format(last_result if last_result else "None")
    )


def format_history(contents: List[Dict[str, Any]], max_turns: int = MAX_TURNS) -> List[Dict[str, Any]]:
    """Verbatim port of AutoGLMAgent.format_history."""
    history = []
    for ix in range(len(contents)):
        if ix == 0:
            env_input = "**Environment State (Omitted)**"
        else:
            env_input = (
                "**Environment State (Omitted)**\n"
                f"Previous Action Result: {contents[ix - 1]['exe_result']}"
            )

        env_input = env_input[:2000] + "..." if len(env_input) > 2000 else env_input
        response = (
            contents[ix]["response"][:1500] + "..."
            if len(contents[ix]["response"]) > 1500
            else contents[ix]["response"]
        )
        history.append({"role": "user", "content": [{"type": "text", "text": env_input}]})
        history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

    return history[-max_turns * 2:]


def build_messages(
    instruction: str,
    png_b64: str,
    contents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Reference AutoGLMAgent.prepare: system + history + current observation
    (image first, then the observation text)."""
    last_result = contents[-1]["exe_result"] if contents else ""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_message(instruction)}
    ]
    messages.extend(format_history(contents))
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{png_b64}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": observation_text(last_result)},
            ],
        }
    )
    return messages


def is_vllm_endpoint(base_url: str) -> bool:
    """True when the request goes to a self-hosted vLLM-style server.

    ``skip_special_tokens`` and ``include_stop_str_in_output`` are vLLM
    sampling passthroughs. The reference runner serves the model on vLLM, but
    they are not documented for the hosted providers, so they are only sent to
    a self-hosted endpoint. Known hosted hosts are excluded by host match;
    anything else is assumed to be the operator's own vLLM. ``GLM_VLLM_FLAGS``
    forces the decision either way.
    """
    forced = os.environ.get("GLM_VLLM_FLAGS", "").strip().lower()
    if forced in {"1", "true", "yes", "on"}:
        return True
    if forced in {"0", "false", "no", "off"}:
        return False
    host = (urlsplit(base_url).hostname or "").lower()
    hosted = ("bigmodel.cn", "openrouter.ai", "zhipuai.cn")
    return not any(host == h or host.endswith("." + h) for h in hosted)


def build_payload(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The request body the reference runner posts.

    Model and sampling are the reference's exactly; the two vLLM passthrough
    flags are attached only for a self-hosted vLLM endpoint (see
    ``is_vllm_endpoint``) because hosted Zhipu does not document them.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": False,
        "stop": list(STOP),
    }
    if is_vllm_endpoint(BASE_URL):
        payload["skip_special_tokens"] = False
        payload["include_stop_str_in_output"] = True
    return payload


def glm_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """One chat-completions call with the reference's constant-interval retry
    on rate limits / connection errors."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key()}",
    }
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                API_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = f"connection error: {exc}"
            print(f"glm request attempt {attempt}: {last_error}", file=sys.stderr)
            time.sleep(RETRY_INTERVAL)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = f"HTTP {resp.status_code} {resp.text[:500]}"
            print(f"glm request attempt {attempt}: {last_error}", file=sys.stderr)
            time.sleep(RETRY_INTERVAL)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"GLM request failed after {MAX_RETRIES} attempts: {last_error}")


def call_llm(messages: List[Dict[str, Any]]) -> str:
    """Return the reply text, tolerating a null ``content``.

    The default model is a thinking model, and hosted thinking models routinely
    return ``content: null`` with the text in a separate ``reasoning`` /
    ``reasoning_content`` field. Indexing the raw content therefore has to be
    defensive, and the reasoning has to be stitched back in front of the
    content the way the reference's call_llm builds final_answer, or the parser
    never sees the model's thought text. Mirrors
    ``agents/glm5v_turbo/agent.py::response_text_of``.
    """
    result = glm_request(build_payload(messages))
    message = (result.get("choices") or [{}])[0].get("message") or {}
    content = message.get("content")
    content = content if isinstance(content, str) else ("" if content is None else str(content))
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    return f"<think>{reasoning}</think>{content}" if reasoning else content


def parse_code_from_string(input_string: str) -> List[str]:
    """Verbatim port of the reference parser."""
    import re

    if input_string.strip() in ["WAIT", "DONE", "FAIL"]:
        return [input_string.strip()]

    pattern = r"```(?:\w+\s+)?(.*?)```"
    matches = re.findall(pattern, input_string, re.DOTALL)

    codes = []

    for match in matches:
        match = match.strip()
        commands = ["WAIT", "DONE", "FAIL"]

        if match in commands:
            codes.append(match.strip())
        elif match.split("\n")[-1] in commands:
            if len(match.split("\n")) > 1:
                codes.append("\n".join(match.split("\n")[:-1]))
            codes.append(match.split("\n")[-1])
        else:
            codes.append(match)

    return codes


def map_key(key: Any) -> str:
    lowered = str(key).strip().lower()
    return ENV_KEY_MAP.get(lowered, lowered)


def scale_xy(coordinate: Any, width: int, height: int) -> List[int]:
    """0-1000 normalized -> pixels; the reference rounds
    (round(x * 1920 / 1000)) against its fixed screen size, here against the
    observed screenshot size, clamped into the framebuffer."""
    if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
        raise ValueError(f"{coordinate!r} must be [x, y]")
    x, y = coordinate
    px = round(float(x) * width / GRID)
    py = round(float(y) * height / GRID)
    return [min(max(px, 0), width - 1), min(max(py, 0), height - 1)]


def type_segments(text: str) -> List[Dict[str, Any]]:
    """pyautogui.write types '\\n' as an Enter press; split lines accordingly."""
    lines = text.split("\n")
    actions: List[Dict[str, Any]] = []
    for index, line in enumerate(lines):
        if line:
            actions.append({"keyboard": {"text": line}})
        if index < len(lines) - 1:
            actions.append({"keyboard": {"keys": ["Return"]}})
    return actions


def parse_agent_call(code: str) -> Tuple[str, list, dict]:
    """Parse one ``Agent.method(...)`` pseudo-code line into (name, args,
    kwargs) with literal arguments only. The reference eval()s the line
    against GroundingAgent; this port parses it instead (agents have no exec
    path)."""
    try:
        tree = ast.parse(code.strip(), mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"not valid Python: {exc.msg}")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        raise ValueError("expected a single Agent.<action>(...) call")
    call = tree.body[0].value
    if not isinstance(call, ast.Call):
        raise ValueError("expected a single Agent.<action>(...) call")
    func = call.func
    if not (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "Agent"
    ):
        raise ValueError("only Agent.<action>(...) calls are supported")
    try:
        args = [ast.literal_eval(a) for a in call.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg}
    except (ValueError, SyntaxError):
        raise ValueError("arguments must be literals")
    return func.attr, args, kwargs


class Grounded:
    """One grounded action: env segments plus the result string the
    reference's generated pyautogui code would have printed (fed back to the
    model as the previous-action result), and an optional control token."""

    def __init__(
        self,
        segments: Optional[List[Dict[str, Any]]] = None,
        result: str = "",
        token: Optional[str] = None,
    ):
        self.segments = segments or []
        self.result = result
        self.token = token


def ground_click(width: int, height: int, coordinate: List, num_clicks: int = 1,
                 button_type: str = "left") -> Grounded:
    point = scale_xy(coordinate, width, height)
    actions: List[Dict[str, Any]] = []
    if button_type == "left":
        if num_clicks >= 3:
            actions.append({"mouse": {"triple_click": point}})
        elif num_clicks == 2:
            actions.append({"mouse": {"double_click": point}})
        else:
            actions.append({"mouse": {"left_click": point}})
    elif button_type in ("right", "middle"):
        key = f"{button_type}_click"
        for _ in range(max(1, int(num_clicks))):
            actions.append({"mouse": {key: point}})
    else:
        raise ValueError(f"unsupported button_type: {button_type!r}")
    return Grounded([{"actions": actions}], "Click Success")


def ground_type(width: int, height: int, coordinate: Optional[List] = None,
                text: str = "", overwrite: bool = False, enter: bool = False) -> Grounded:
    actions: List[Dict[str, Any]] = []
    if coordinate is not None:
        actions.append({"mouse": {"left_click": scale_xy(coordinate, width, height)}})
    if overwrite:
        actions.append({"keyboard": {"keys": ["ctrl", "a"]}})
        actions.append({"keyboard": {"keys": ["BackSpace"]}})
    actions.extend(type_segments(str(text)))
    if enter:
        actions.append({"keyboard": {"keys": ["Return"]}})
    return Grounded([{"actions": actions}], "Type Success")


def ground_drag_and_drop(width: int, height: int, drag_from_coordinate: List,
                         drop_on_coordinate: List) -> Grounded:
    start = scale_xy(drag_from_coordinate, width, height)
    end = scale_xy(drop_on_coordinate, width, height)
    return Grounded(
        [{"actions": [{"mouse": {"left_click_drag": [start, end]}}]}],
        "Drag and Drop Success",
    )


def ground_scroll(width: int, height: int, coordinate: List, direction: str) -> Grounded:
    point = scale_xy(coordinate, width, height)
    # Reference: pyautogui.scroll(100) for "up", pyautogui.scroll(-100)
    # otherwise; pyautogui positive = up while the env positive = down, so
    # the env sign flips: up -> -100, anything else -> +100.
    clicks = -100 if direction == "up" else 100
    return Grounded(
        [{"actions": [{"mouse": {"move": point}}, {"mouse": {"scroll": clicks}}]}],
        "Scroll Success",
    )


def ground_hotkey(width: int, height: int, keys: List) -> Grounded:
    if not isinstance(keys, (list, tuple)) or not keys:
        raise ValueError("hotkey needs a non-empty key list")
    mapped = [map_key(k) for k in keys]
    # Reproduce the printed result of the reference's generated code.
    quoted = [f"'{k}'" for k in keys]
    key_str = ", ".join(quoted).replace("'", "\\'")
    return Grounded(
        [{"actions": [{"keyboard": {"keys": mapped}}]}],
        f"Press Hotkey: {key_str}",
    )


def ground_quote(width: int, height: int, content: str) -> Grounded:
    return Grounded([], str(content))


def ground_wait(width: int, height: int) -> Grounded:
    return Grounded([], "", token="WAIT")


def ground_exit(width: int, height: int, success: bool) -> Grounded:
    return Grounded([], "", token="DONE" if success else "FAIL")


GROUNDERS = {
    "click": ground_click,
    "type": ground_type,
    "drag_and_drop": ground_drag_and_drop,
    "scroll": ground_scroll,
    "hotkey": ground_hotkey,
    "quote": ground_quote,
    "wait": ground_wait,
    "exit": ground_exit,
}


def translate_agent_call(code: str, width: int, height: int) -> Grounded:
    """Ground one pseudo-code line. Raises ValueError when the line is not a
    supported Agent call (the caller records "Invalid action", matching the
    reference's failed-parse path)."""
    name, args, kwargs = parse_agent_call(code)
    grounder = GROUNDERS.get(name)
    if grounder is None:
        raise ValueError(f"unsupported Agent action: {name}")
    try:
        return grounder(width, height, *args, **kwargs)
    except TypeError as exc:
        raise ValueError(f"bad arguments for Agent.{name}: {exc}")


def execute_code(
    computer: Computer, code: str, width: int, height: int
) -> Tuple[Optional[str], Optional[str]]:
    """Translate and run one action line. Returns (exe_result, terminal):
    exe_result None means the line could not be grounded ("Invalid action");
    terminal is "DONE"/"FAIL" when the episode should end. FAIL (and
    Agent.exit(success=False)) emits the env FAIL action."""
    if code in ("WAIT", "DONE", "FAIL"):
        token = code
    else:
        try:
            grounded = translate_agent_call(code, width, height)
        except ValueError as exc:
            print(f"failed to ground action: {exc}", file=sys.stderr)
            return None, None
        token = grounded.token
        if token is None:
            for segment in grounded.segments:
                if segment.get("actions"):
                    computer.step(segment["actions"])
            return grounded.result, None

    if token == "WAIT":
        computer.wait(WAIT_SECONDS)
        return "", None
    if token == "FAIL":
        computer.step([{"action_type": "FAIL"}])
        return "", "FAIL"
    return "", "DONE"


def image_size(png: bytes) -> Tuple[int, int]:
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
        raise ValueError("observation is not a valid PNG with an IHDR chunk")
    return struct.unpack(">II", png[16:24])


def run(env_url: str, task: str) -> None:
    computer = Computer(env_url, timeout_sec=ENV_HTTP_TIMEOUT)
    contents: List[Dict[str, Any]] = []
    try:
        obs = computer.observe()
        for step in range(MAX_STEPS):
            width, height = image_size(obs["png"])
            png_b64 = base64.b64encode(obs["png"]).decode("ascii")
            messages = build_messages(task, png_b64, contents)

            try:
                response = call_llm(messages)
            except Exception as exc:
                # Reference predict(): a failed generation becomes an empty
                # response and the step is recorded as a parse error.
                print(f"step {step}: LLM call failed: {exc}", file=sys.stderr)
                response = ""
            print(f"step {step}: response: {response[:400]!r}", file=sys.stderr)

            codes = parse_code_from_string(response)
            print(f"step {step}: actions: {codes!r}", file=sys.stderr)

            terminal = None
            if not codes:
                exe_result = "Invalid action"
            else:
                # parse_code_from_string deliberately splits a trailing
                # DONE/WAIT/FAIL into its own element, so honour every element:
                # taking only codes[0] discarded the terminal token and the
                # episode ran to the step cap after finishing the task.
                exe_result = "Invalid action"
                for code in codes:
                    exe_result, terminal = execute_code(
                        computer, code, width, height
                    )
                    if exe_result is None:
                        exe_result = "Invalid action"
                    if terminal is not None:
                        break

            contents.append({"response": response, "exe_result": exe_result})
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
