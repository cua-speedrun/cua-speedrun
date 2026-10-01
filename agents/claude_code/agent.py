"""Run Claude Code as a computer-use agent through a tiny ``act`` command.

The CLI runs inside cua-speedrun's agent sandbox, never inside the task VM.
It can affect the task only by POSTing one action at a time to the authenticated
localhost gateway below; that gateway uses the public ``Computer`` interface.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from cua_speedrun.client import Computer


DISPLAY_MAX_LONG_SIDE = 1280
MAX_STEPS = int(os.environ.get("CS_MAX_STEPS", "100"))
CLI_TIMEOUT_SEC = float(os.environ.get("CLAUDE_CODE_TIMEOUT_SEC", "3600"))
MODEL = os.environ.get("CLAUDE_CODE_MODEL", "claude-opus-4-6")

ROOT = Path(__file__).resolve().parent
NODE_BIN = ROOT / ".cli_runtime" / "node-v22.23.1" / "bin"
CLI_BIN = ROOT / ".cli_runtime" / "claude-2.1.207" / "bin"


def _point(
    value: Any,
    ratio_x: float,
    ratio_y: float,
    native_w: int,
    native_h: int,
) -> list[int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("coordinate must be [x, y]")
    x, y = value
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError("coordinate values must be numbers")
    return [
        min(native_w - 1, max(0, round(x * ratio_x))),
        min(native_h - 1, max(0, round(y * ratio_y))),
    ]


def _action_error(result: dict[str, Any]) -> str | None:
    error = result.get("error")
    if error:
        return str(error)
    if result.get("ok") is False:
        return "The environment rejected the action without an error message."
    return None


class ActionGateway:
    """Translate the CLI's small JSON vocabulary into Computer actions."""

    def __init__(self, computer: Computer, max_steps: int):
        self.computer = computer
        self.max_steps = max_steps
        self.token = secrets.token_urlsafe(32)
        self.steps = 0
        self.finished = False
        self.fatal_error: BaseException | None = None
        self.transcript: list[dict[str, Any]] = []
        self._screenshot_b64 = ""
        self.observation_id = 0
        self.native_w = 0
        self.native_h = 0
        self.display_w = 0
        self.display_h = 0
        self.ratio_x = 1.0
        self.ratio_y = 1.0
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def prepare(self) -> None:
        self._set_observation(self.computer.observe()["png"])

    def _set_observation(self, raw: bytes) -> None:
        with Image.open(BytesIO(raw)) as source:
            image = source.convert("RGB")
            self.native_w, self.native_h = image.size
            scale = min(
                1.0,
                DISPLAY_MAX_LONG_SIDE / max(self.native_w, self.native_h),
            )
            self.display_w = max(1, round(self.native_w * scale))
            self.display_h = max(1, round(self.native_h * scale))
            if image.size != (self.display_w, self.display_h):
                image = image.resize(
                    (self.display_w, self.display_h), Image.Resampling.LANCZOS
                )
            output = BytesIO()
            image.save(output, format="PNG")
        self.ratio_x = self.native_w / self.display_w
        self.ratio_y = self.native_h / self.display_h
        self._screenshot_b64 = base64.b64encode(output.getvalue()).decode("ascii")
        self.observation_id += 1

    def _finish(self) -> None:
        if not self.finished:
            self.computer.done()
            self.finished = True

    def ensure_done(self) -> None:
        with self._lock:
            self._finish()

    def _response(self, *, error: str | None = None) -> dict[str, Any]:
        return {
            "step": self.steps,
            "budget_remaining": max(0, self.max_steps - self.steps),
            "done": self.finished or self.steps >= self.max_steps,
            "error": error,
            "observation_id": self.observation_id,
            "screenshot_b64": self._screenshot_b64,
        }

    def _translate(self, action: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
        kind = str(action.get("action", "")).strip().lower()
        if not kind:
            raise ValueError("action JSON needs a non-empty 'action' field")
        if kind == "key":
            keys = action.get("keys")
            if not isinstance(keys, list) or not keys:
                raise ValueError("key requires a non-empty 'keys' list")
            return [{"keyboard": {"keys": [str(key) for key in keys]}}], False
        if kind == "type":
            actions: list[dict[str, Any]] = []
            if action.get("clear"):
                actions.append({"keyboard": {"keys": ["ctrl", "a"]}})
            actions.append({"keyboard": {"text": str(action.get("text", ""))}})
            if action.get("enter"):
                actions.append({"keyboard": {"keys": ["Return"]}})
            return actions, False
        if kind == "mouse_move":
            point = _point(
                action.get("coordinate"), self.ratio_x, self.ratio_y,
                self.native_w, self.native_h,
            )
            return [{"mouse": {"move": point}}], False
        clicks = {
            "click": "left_click",
            "left_click": "left_click",
            "right_click": "right_click",
            "double_click": "double_click",
            "triple_click": "triple_click",
        }
        if kind in clicks:
            point = _point(
                action.get("coordinate"), self.ratio_x, self.ratio_y,
                self.native_w, self.native_h,
            )
            return [{"mouse": {clicks[kind]: point}}], False
        if kind in {"drag", "left_click_drag"}:
            start = _point(
                action.get("coordinate"), self.ratio_x, self.ratio_y,
                self.native_w, self.native_h,
            )
            end = _point(
                action.get("coordinate2"), self.ratio_x, self.ratio_y,
                self.native_w, self.native_h,
            )
            return [{"mouse": {"left_click_drag": [start, end]}}], False
        if kind == "scroll":
            clicks = action.get("clicks")
            if "pixels" in action or type(clicks) is not int:
                raise ValueError(
                    "scroll requires integer 'clicks' in mouse-wheel units, "
                    "not pixels (negative up, positive down; e.g. 3)"
                )
            actions = []
            if "coordinate" in action:
                point = _point(
                    action["coordinate"], self.ratio_x, self.ratio_y,
                    self.native_w, self.native_h,
                )
                actions.append({"mouse": {"move": point}})
            actions.append({"mouse": {"scroll": clicks}})
            return actions, False
        if kind == "wait":
            seconds = action.get("time", 1.0)
            if not isinstance(seconds, (int, float)) or not 0 <= seconds <= 30:
                raise ValueError("wait time must be a number from 0 to 30 seconds")
            return [{"action": "wait", "time": float(seconds)}], False
        if kind in {"terminate", "done", "finish"}:
            status = str(action.get("status", "success")).lower()
            if status not in {"success", "failure"}:
                raise ValueError("terminate status must be 'success' or 'failure'")
            return ([{"action_type": "FAIL"}] if status == "failure" else []), True
        raise ValueError(f"unsupported action {kind!r}")

    def command(self, raw: str) -> dict[str, Any]:
        with self._lock:
            if self.finished:
                return self._response(error="episode is already finished")
            try:
                action = json.loads(raw)
                if not isinstance(action, dict):
                    raise ValueError("action JSON must be an object")
                if str(action.get("action", "")).strip().lower() == "screenshot":
                    self.prepare()
                    return self._response()
                if self.steps >= self.max_steps:
                    self._finish()
                    return self._response(error="action budget exhausted")
                env_actions, terminal = self._translate(action)
                error = _action_error(self.computer.step(env_actions)) if env_actions else None
                if not terminal:
                    self.steps += 1
                entry = {"step": self.steps, "command": action, "actions": env_actions}
                if terminal:
                    entry["terminal"] = True
                if error:
                    entry["error"] = error
                self.transcript.append(entry)
                if terminal and error is None:
                    self._finish()
                else:
                    self.prepare()
                    if self.steps >= self.max_steps:
                        self._finish()
                return self._response(error=error)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                self.transcript.append(
                    {"step": self.steps, "raw_command": raw, "error": str(exc)}
                )
                return self._response(error=str(exc))
            except BaseException as exc:
                self.fatal_error = exc
                try:
                    self._finish()
                except BaseException:
                    pass
                return self._response(error=f"environment gateway failed: {exc}")

    def start(self) -> str:
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                pass

            def do_POST(self) -> None:
                if self.path != "/act":
                    self.send_error(404)
                    return
                if self.headers.get("X-Gateway-Token") != gateway.token:
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 <= length <= 65536:
                        raise ValueError("request is too large")
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    command = payload["command"]
                    if not isinstance(command, str):
                        raise ValueError("command must be a JSON string")
                    result = gateway.command(command)
                    body = json.dumps(result).encode()
                    self.send_response(200)
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    body = json.dumps({"error": str(exc), "done": False}).encode()
                    self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{port}/act"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


ACT_SCRIPT = r'''#!/usr/bin/env python3
import base64, json, os, sys, urllib.request
if len(sys.argv) != 2:
    print("usage: act '<json action>'", file=sys.stderr)
    raise SystemExit(2)
request = urllib.request.Request(
    os.environ["CS_ACT_GATEWAY"],
    data=json.dumps({"command": sys.argv[1]}).encode(),
    headers={"Content-Type": "application/json",
             "X-Gateway-Token": os.environ["CS_ACT_TOKEN"]},
)
try:
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
except Exception as exc:
    print(f"act: gateway request failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
if payload.get("screenshot_b64"):
    os.makedirs("obs", exist_ok=True)
    path = f"obs/frame_{int(payload['observation_id']):04d}.png"
    with open(path, "wb") as output:
        output.write(base64.b64decode(payload["screenshot_b64"]))
    print(f"screenshot: {path} (view this image before your next action)")
if payload.get("error"):
    print(f"error: {payload['error']}")
print(f"budget_remaining: {payload.get('budget_remaining', 0)}")
if payload.get("done"):
    print("EPISODE DONE: do not issue another action.")
'''


def prompt(task: str, gateway: ActionGateway) -> str:
    return f"""You are operating a remote computer to complete a task. The computer is
in a separate VM. Your only interface to it is the `act` command already on
PATH. Do not use networking or files to try to reach the task directly.

Call `act` with exactly one JSON action, then view the screenshot path it
prints before choosing the next action. Begin with a screenshot:

    act '{{"action": "screenshot"}}'
    act '{{"action": "left_click", "coordinate": [640, 360]}}'
    act '{{"action": "type", "text": "hello"}}'

The screenshots are exactly {gateway.display_w}x{gateway.display_h} pixels.
Coordinates use that displayed image: [0, 0] is the top left.

Available actions:
- screenshot
- left_click, right_click, double_click, or triple_click with coordinate [x, y]
- mouse_move with coordinate [x, y]
- drag with coordinate [x1, y1] and coordinate2 [x2, y2]
- type with text, plus optional clear and enter booleans
- key with keys such as ["ctrl", "s"] for one simultaneous chord
- scroll with integer clicks in mouse-wheel units (negative up, positive down)
  and optional coordinate; for example {{"action": "scroll", "clicks": 3}}
- wait with time in seconds
- terminate with status "success" or "failure"

You have {gateway.max_steps} non-observation actions. When the task is complete,
call `act '{{"action": "terminate", "status": "success"}}'` and stop.
If the task is infeasible, terminate with status "failure".

Task:
{task}
"""


def child_environment(workdir: Path, gateway_url: str, token: str) -> dict[str, str]:
    env = {
        "PATH": os.pathsep.join(
            (str(CLI_BIN), str(NODE_BIN), "/usr/local/bin", "/usr/bin", "/bin")
        ),
        "HOME": str(workdir / "home"),
        "TMPDIR": str(workdir / "tmp"),
        "CLAUDE_CONFIG_DIR": str(workdir / "claude-config"),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1",
        "IS_SANDBOX": "1",
        "CS_ACT_GATEWAY": gateway_url,
        "CS_ACT_TOKEN": token,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        env["CLAUDE_CODE_OAUTH_TOKEN"] = os.environ["CLAUDE_CODE_OAUTH_TOKEN"]
    elif os.environ.get("ANTHROPIC_API_KEY"):
        env["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
        if os.environ.get("ANTHROPIC_BASE_URL"):
            env["ANTHROPIC_BASE_URL"] = os.environ["ANTHROPIC_BASE_URL"]
    else:
        raise ValueError("set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY")
    return env


def cli_command(cli: Path, *, oauth: bool) -> list[str]:
    command = [str(cli)]
    if oauth:
        command.extend([
            "--setting-sources", "",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        ])
    else:
        command.append("--bare")
    command.extend([
        "--no-session-persistence", "--no-chrome", "--tools", "Bash,Read",
        "--verbose", "--output-format", "stream-json",
        "--permission-mode", "bypassPermissions", "--print", "--model", MODEL,
    ])
    effort = os.environ.get("CLAUDE_CODE_EFFORT", "").strip().lower()
    if effort:
        if effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("invalid CLAUDE_CODE_EFFORT: use low, medium, high, xhigh, or max")
        command.extend(["--effort", effort])
    if os.environ.get("CLAUDE_MAX_BUDGET_USD"):
        command.extend(["--max-budget-usd", os.environ["CLAUDE_MAX_BUDGET_USD"]])
    return command


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: agent.py <env_url> <task_description>")
    if not (os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")):
        raise SystemExit("set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY")
    cli = CLI_BIN / "claude"
    if not cli.is_file():
        raise SystemExit("Claude Code is missing; init.py did not complete")
    command = cli_command(cli, oauth=bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")))

    computer = Computer(sys.argv[1])
    gateway = ActionGateway(computer, MAX_STEPS)
    returncode = 1
    try:
        gateway.prepare()
        gateway_url = gateway.start()
        with tempfile.TemporaryDirectory(prefix="cs_claude_code_") as tmp:
            workdir = Path(tmp)
            for directory in (
                workdir / "home",
                workdir / "tmp",
                workdir / "claude-config",
            ):
                directory.mkdir()
            act = workdir / "act"
            act.write_text(ACT_SCRIPT)
            act.chmod(0o700)
            env = child_environment(workdir, gateway_url, gateway.token)
            env["PATH"] = f"{workdir}{os.pathsep}{env['PATH']}"
            result = subprocess.run(
                command,
                input=prompt(sys.argv[2], gateway),
                text=True,
                cwd=workdir,
                env=env,
                timeout=CLI_TIMEOUT_SEC,
                check=False,
            )
            returncode = result.returncode
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Claude Code timed out after {CLI_TIMEOUT_SEC:.0f}s"
        ) from exc
    finally:
        try:
            gateway.ensure_done()
        finally:
            gateway.stop()
        for item in gateway.transcript:
            print(json.dumps({"cli_harness": item}, sort_keys=True), file=sys.stderr)

    if gateway.fatal_error is not None:
        raise RuntimeError("the environment action gateway failed") from gateway.fatal_error
    if returncode != 0:
        raise RuntimeError(f"Claude Code exited with status {returncode}")


if __name__ == "__main__":
    main()
