"""Run Codex CLI as a computer-use agent through a pass-through HTTP proxy.

The CLI runs inside cua-speedrun's agent sandbox, never inside the task VM.
It can affect the task by sending raw action batches to the authenticated
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
from pathlib import Path
from typing import Any

from cua_speedrun.client import Computer


MAX_STEPS = int(os.environ.get("CODEX_MAX_STEPS") or os.environ.get("CS_MAX_STEPS", "100"))
CLI_TIMEOUT_SEC = float(os.environ.get("CODEX_CLI_TIMEOUT_SEC", "3600"))
MODEL = os.environ.get("CODEX_MODEL", "gpt-5.1")

ROOT = Path(__file__).resolve().parent
NODE_BIN = ROOT / ".cli_runtime" / "node-v22.23.1" / "bin"
CLI_BIN = ROOT / ".cli_runtime" / "codex-0.153.2" / "bin"


class ActionGateway:
    """Authenticated HTTP access to the public Computer contract.

    Action dictionaries and step responses are passed through unchanged.
    This proxy deliberately knows nothing about individual action types.
    """

    def __init__(self, computer: Computer, max_steps: int):
        self.computer = computer
        self.max_steps = max_steps
        self.token = secrets.token_urlsafe(32)
        self.steps = 0
        self.finished = False
        self.fatal_error: Exception | None = None
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def ensure_done(self) -> None:
        with self._lock:
            if not self.finished:
                self.computer.done()
                self.finished = True

    def start(self) -> str:
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                pass

            def reply(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Steps-Remaining", str(max(0, gateway.max_steps - gateway.steps)))
                self.send_header("X-Episode-Done", str(gateway.finished).lower())
                self.end_headers()
                self.wfile.write(body)

            def authorized(self) -> bool:
                if self.headers.get("X-Gateway-Token") != gateway.token:
                    self.reply(403, {"error": "invalid gateway token"})
                    return False
                return True

            def do_GET(self) -> None:
                if not self.authorized():
                    return
                if self.path != "/observe":
                    self.reply(404, {"error": "unknown endpoint"})
                    return
                try:
                    obs = gateway.computer.observe()
                    payload = {
                        "png_b64": base64.b64encode(obs["png"]).decode("ascii"),
                        "meta": obs["meta"],
                    }
                except Exception as exc:
                    gateway.fatal_error = exc
                    self.reply(502, {"error": str(exc)})
                    return
                self.reply(200, payload)

            def do_POST(self) -> None:
                if not self.authorized():
                    return
                if self.path not in {"/step", "/done"}:
                    self.reply(404, {"error": "unknown endpoint"})
                    return
                actions = None
                if self.path == "/step":
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if length <= 0:
                            raise ValueError("step requires a JSON body")
                        body = json.loads(self.rfile.read(length))
                        if not isinstance(body, dict) or not isinstance(body.get("actions"), list):
                            raise ValueError("body must contain an actions list")
                        actions = body["actions"]
                    except (ValueError, TypeError) as exc:
                        self.reply(400, {"error": str(exc)})
                        return
                # Serialize requests without transforming or splitting batches.
                with gateway._lock:
                    try:
                        if gateway.finished:
                            status, result = 409, {"error": "episode is already finished"}
                        elif self.path == "/done":
                            gateway.computer.done()
                            gateway.finished = True
                            status, result = 200, {"ok": True}
                        elif gateway.steps >= gateway.max_steps:
                            status, result = 409, {"error": "step budget exhausted"}
                        else:
                            result = gateway.computer.step(actions)
                            gateway.steps += 1
                            status = 200
                            if gateway.steps >= gateway.max_steps:
                                gateway.computer.done()
                                gateway.finished = True
                    except Exception as exc:
                        gateway.fatal_error = exc
                        status, result = 502, {"error": str(exc)}
                self.reply(status, result)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{port}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def prompt(task: str, gateway: ActionGateway) -> str:
    return f"""You are operating a remote computer in a separate VM.
Your only interface to that computer is the HTTP proxy at $CS_COMPUTER_URL.
Authenticate every request with the header X-Gateway-Token: $CS_COMPUTER_TOKEN.
Your local shell and files are in the agent sandbox, not the task VM.
Do not bypass the proxy to access the task computer.

HTTP contract:
- GET /observe returns {{"png_b64": "<base64 PNG>", "meta": {{...}}}}.
  Decode png_b64 to a local PNG and view it before choosing your next action.
  Screenshots are native resolution. Use their actual pixel coordinates;
  the proxy does not resize, scale, clamp, or otherwise change coordinates.
- POST /step accepts {{"actions": [<action dictionaries>]}}.
  The entire list is sent unchanged as ONE Computer.step call, in order.
  Its JSON response is returned unchanged, including ok, info, and error.
  Observe explicitly after acting; step does not append a screenshot.
- POST /done finishes the task and triggers verification.
Response headers X-Steps-Remaining and X-Episode-Done report your budget.
You have {gateway.max_steps} step requests; each batch counts once.
Malformed JSON does not count. A forwarded step returning ok:false does count.
If an action times out or its outcome is unknown, observe before deciding
what to do next; do not blindly repeat a potentially executed action.

Action schema (support depends on the selected environment/runner):
- {{"mouse": {{"left_click": [x, y]}}}}. Other click keys:
  right_click, middle_click, double_click, triple_click.
- {{"mouse": {{"move": [x, y]}}}}
- {{"mouse": {{"left_click_drag": [[x1, y1], [x2, y2]]}}}};
  right_click_drag has the same form. Runners supporting a one-point drag
  also accept [[x2, y2]] to drag from the current pointer position.
- {{"mouse": {{"buttons": {{"left_down": true}}}}}}; also left_up,
  right_down, right_up. Use separate actions to hold and release buttons.
- {{"mouse": {{"scroll": 5}}}}; positive scrolls down, negative up.
  This is the environment's scroll value, not a pixel distance.
- {{"keyboard": {{"text": "text to type"}}}}
- {{"keyboard": {{"keys": ["ctrl", "s"]}}}} for a simultaneous key chord.
- {{"keyboard": {{"keys_down": ["shift"]}}}} and
  {{"keyboard": {{"keys_up": ["shift"]}}}} for holding/releasing keys.
- {{"action": "wait", "time": 1.0}}; the environment also accepts
  "type" instead of "action" and "seconds" instead of "time".
- {{"action": "screenshot"}} is an environment control action within a batch;
  use GET /observe to retrieve the screenshot.
Action dictionaries may combine fields, and batches may contain multiple
dictionaries. These examples are not an allowlist: any other action fields
supported by the environment are forwarded unchanged as well.

Example step request from your local shell:
curl -sS -H "X-Gateway-Token: $CS_COMPUTER_TOKEN" \\
  -H 'Content-Type: application/json' \\
  --data '{{"actions":[{{"mouse":{{"left_click":[640,360]}}}}]}}' \\
  "$CS_COMPUTER_URL/step"

To retrieve an observation:
curl -sS -H "X-Gateway-Token: $CS_COMPUTER_TOKEN" \\
  "$CS_COMPUTER_URL/observe" -o observation.json
python -c 'import base64,json; from pathlib import Path; d=json.loads(Path("observation.json").read_text()); Path("screen.png").write_bytes(base64.b64decode(d["png_b64"]))'
Then view screen.png using your image-viewing tool.

When finished, POST /done and stop. A textual final answer alone does not
perform an environment action.

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
        "CODEX_HOME": str(workdir / "codex-home"),
        "CS_COMPUTER_URL": gateway_url,
        "CS_COMPUTER_TOKEN": token,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }
    if not os.environ.get("CODEX_AUTH_JSON"):
        if os.environ.get("OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        if os.environ.get("OPENAI_BASE_URL"):
            env["OPENAI_BASE_URL"] = os.environ["OPENAI_BASE_URL"]
    return env


def authentication() -> dict[str, Any]:
    """Load a per-evaluation secret; never stage auth during image creation."""
    if os.environ.get("CODEX_AUTH_JSON"):
        try:
            payload = json.loads(os.environ["CODEX_AUTH_JSON"])
        except ValueError:
            raise SystemExit("CODEX_AUTH_JSON must be valid credential-cache JSON") from None
        if not isinstance(payload, dict) or not payload.get("tokens"):
            raise SystemExit("CODEX_AUTH_JSON must contain Codex subscription tokens")
        return payload
    if os.environ.get("OPENAI_API_KEY"):
        return {"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}
    raise SystemExit("Provide CODEX_AUTH_JSON or OPENAI_API_KEY")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: agent.py <env_url> <task_description>")
    auth = authentication()
    cli = CLI_BIN / "codex"
    if not cli.is_file():
        raise SystemExit("Codex CLI is missing; init.py did not complete")

    computer = Computer(sys.argv[1])
    gateway = ActionGateway(computer, MAX_STEPS)
    returncode = 1
    try:
        gateway_url = gateway.start()
        with tempfile.TemporaryDirectory(prefix="cs_codex_cli_") as tmp:
            workdir = Path(tmp)
            for directory in (
                workdir / "home", workdir / "tmp", workdir / "codex-home"
            ):
                directory.mkdir(mode=0o700)
            env = child_environment(workdir, gateway_url, gateway.token)
            auth_file = workdir / "codex-home" / "auth.json"
            auth_file.write_text(json.dumps(auth))
            auth_file.chmod(0o600)
            preflight = subprocess.run(
                [str(cli), "login", "status"], cwd=workdir, env=env,
                capture_output=True, text=True, timeout=30, check=False,
            )
            if preflight.returncode:
                raise RuntimeError("Codex authentication preflight failed")
            command = [
                str(cli), "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--model", MODEL,
                "--json",
                "-",
            ]
            if os.environ.get("CODEX_REASONING_EFFORT"):
                command[2:2] = [
                    "-c", "model_reasoning_effort="
                    + json.dumps(os.environ["CODEX_REASONING_EFFORT"]),
                ]
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
        raise RuntimeError(f"Codex CLI timed out after {CLI_TIMEOUT_SEC:.0f}s") from exc
    finally:
        try:
            if returncode == 0 and gateway.fatal_error is None:
                gateway.ensure_done()
        finally:
            gateway.stop()

    if gateway.fatal_error is not None:
        raise RuntimeError("the environment action gateway failed") from gateway.fatal_error
    if returncode != 0:
        raise RuntimeError(f"Codex CLI exited with status {returncode}")


if __name__ == "__main__":
    main()
