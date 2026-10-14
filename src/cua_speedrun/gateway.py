"""The gateway: the agent's only door into the environment, and the clock.

One gateway serves one run. Agent-facing operations, under the run token:

    GET  /{run_token}/observe      -> {"png_b64", "meta"}
    POST /{run_token}/step         -> {"ok": true}     body: {"actions": [...]}
    POST /{run_token}/done         -> {"ok": true}

Executor-facing control operations, under the control token:

    GET  /_ctl/{control_token}/health   -> {"ready": bool}
    POST /_ctl/{control_token}/arm      -> {"armed": true, "t0": ...}
    GET  /_ctl/{control_token}/status   -> {"armed", "finished", "reason",
                                            "fully_done", "verdict"?}
    GET  /_ctl/{control_token}/runlog   -> the run log as JSONL text
    POST /_ctl/{control_token}/agent_failed -> {"accepted": bool}
    POST /_ctl/{control_token}/shutdown -> {"ok": true}

Every timed event is stamped on the gateway's single monotonic clock. The
clock starts on arm(), immediately before the agent launches, and stops
when /done arrives (or the timeout). A fixed grace period later the checker
runs exactly once. There is deliberately no reset, snapshot, or shell here.

The gateway runs adjacent to the environment. Locally that is this process;
on Modal it is a process inside the env sandbox, so the clock is env-side
and only the agent-to-gateway hop crosses the network. The executor drives
it either in-process (start/arm/join) or over the control endpoints; both
use the same background finisher, so behaviour is identical.
"""

from __future__ import annotations

import base64
import json
import secrets
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterator

from PIL import Image

from cua_speedrun.envs.base import EnvAdapter, Observation, Verdict
from cua_speedrun.fast_io import CONTENT_TYPE, encode_observation
from cua_speedrun.images import image_mime_type, image_suffix
from cua_speedrun.runlog import RunLogWriter


VERIFIER_TIMEOUT_SEC = 5 * 60
ENVIRONMENT_OPERATION_TIMEOUT_SEC = 90.0


class _AgentRequestError(ValueError):
    """The agent sent a request that does not satisfy the gateway contract."""


class _EnvironmentOperationError(RuntimeError):
    """An evaluator-owned environment operation failed or stopped responding."""


class Gateway:
    def __init__(
        self,
        adapter: EnvAdapter,
        log: RunLogWriter,
        artifacts_dir: Path,
        timeout_sec: float,
        grace_sec: float,
        checker: Callable[[], Verdict] | None = None,
        verifier_timeout_sec: float = VERIFIER_TIMEOUT_SEC,
        environment_operation_timeout_sec: float = ENVIRONMENT_OPERATION_TIMEOUT_SEC,
        host: str = "127.0.0.1",
        advertise_host: str | None = None,
        port: int = 0,
        run_token: str | None = None,
        control_token: str | None = None,
        instruction: str = "",
    ):
        self.adapter = adapter
        self.log = log
        self.fast_io = bool(getattr(adapter, "fast_io", False))
        if self.fast_io:
            self.log.enable_async()
            self.log.event("io_contract", version="fast-io-v3", wire_content_type=CONTENT_TYPE,
                           timing="gateway-response-interval-union")
        self.artifacts_dir = artifacts_dir
        self.runlog_path = log.path
        self.timeout_sec = timeout_sec
        self.grace_sec = grace_sec
        self.verifier_timeout_sec = verifier_timeout_sec
        self.environment_operation_timeout_sec = environment_operation_timeout_sec
        # The seed-resolved task instruction, served to the executor over the
        # control plane so it can hand it to the agent (which is in a
        # different sandbox in the remote topology).
        self.instruction = instruction
        # The checker decides pass/fail after the run. It defaults to the
        # adapter's own verifier (gym-anything's), but a seeded task supplies
        # a host-side checker that compares to a seed-derived answer the
        # desktop never saw.
        self.checker = checker or adapter.finalize
        self.advertise_host = advertise_host
        # Tokens may be preset so a remote executor knows them before the
        # gateway starts; otherwise they are generated.
        self.token = run_token or secrets.token_urlsafe(16)
        self.control_token = control_token or secrets.token_urlsafe(16)

        self._t0: float | None = None
        self._t_end: float | None = None
        self._finish_reason: str | None = None
        self._finished = threading.Event()   # done or timeout reached
        self._fully_done = threading.Event()  # checker has run, verdict set
        self._armed = threading.Event()
        self._lock = threading.Lock()
        self._action_lock = threading.Lock()
        self._frame_idx = 0
        self._num_steps = 0
        self._next_operation_id = 0
        self._active_operations: dict[int, tuple[str, float, float]] = {}
        self._responses = threading.Condition()
        self._active_responses = 0
        # A benchmark may divide one task into several independent agent
        # episodes on the same live environment.  Existing adapters never
        # enter this path, so their one-agent/one-done lifecycle is unchanged.
        self._multi_episode = bool(getattr(adapter, "multi_episode", False))
        self._agent_episode = 1
        self._continuation_pending = False
        self._transitioning_episode = False
        self._episode_lock = threading.Lock()
        self.verdict: Verdict | None = None
        self._infrastructure_error: str | None = None
        self._agent_error: str | None = None
        self._finisher: threading.Thread | None = None
        self._operation_watchdog: threading.Thread | None = None

        self._server = ThreadingHTTPServer((host, port), self._make_handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # -- lifecycle, called by the executor (local, in-process) --------------

    def start(self) -> str:
        self._thread.start()
        host, port = self._server.server_address
        advertised = self.advertise_host or host
        return f"http://{advertised}:{port}/{self.token}"

    def arm(self) -> None:
        """Start the clock and the background finisher. Idempotent. Called
        immediately before the agent launches, so there is no free thinking
        time between the two."""
        with self._lock:
            if self._armed.is_set():
                return
            self._t0 = time.monotonic()
            self._armed.set()
        self.log.event("armed", t_mono=self._t0)
        self._finisher = threading.Thread(target=self._run_finisher, daemon=True)
        self._finisher.start()
        self._operation_watchdog = threading.Thread(
            target=self._watch_environment_operations,
            daemon=True,
        )
        self._operation_watchdog.start()

    def _watch_environment_operations(self) -> None:
        while not self._finished.wait(timeout=0.25):
            now = time.monotonic()
            with self._lock:
                overdue = next((
                    (name, now - started, limit)
                    for name, started, limit in self._active_operations.values()
                    if name != "action step" and now - started > limit
                ), None)
            if overdue is not None:
                name, elapsed, limit = overdue
                self._record_infrastructure_error(
                    f"environment {name} did not finish within "
                    f"{limit:g}s "
                    f"(still running after {elapsed:.1f}s)"
                )
                return

    def _run_finisher(self) -> None:
        try:
            remaining = self.timeout_sec - (time.monotonic() - self._t0)
            if not self._finished.wait(timeout=max(0.0, remaining)):
                with self._lock:
                    if not self._finished.is_set():
                        self._t_end = self._t0 + self.timeout_sec
                        self._finish_reason = "timeout"
                        self._finished.set()
            self.log.event("finished", t_mono=self._t_end, reason=self._finish_reason)
            if self._infrastructure_error is not None:
                return
            if self._agent_error is not None:
                self.verdict = Verdict(
                    passed=False,
                    score=0.0,
                    detail=self._agent_error,
                )
                self.log.event(
                    "verdict",
                    passed=False,
                    score=0.0,
                    detail=self._agent_error,
                    checker_time_sec=0.0,
                    grace_sec=0.0,
                )
                return

            time.sleep(self.grace_sec)
            if self._infrastructure_error is not None:
                return

            result: dict[str, Any] = {}
            checker_done = threading.Event()

            def run_checker() -> None:
                try:
                    # A timed-out action may still be mutating the environment.
                    # The existing verifier deadline bounds this wait too.
                    with self._action_lock:
                        if self.fast_io:
                            self.adapter.flush_recording()
                            self.log.flush()
                        if self._multi_episode:
                            # Do not evaluate during a phase transition.
                            with self._episode_lock:
                                result["verdict"] = self.checker()
                        else:
                            result["verdict"] = self.checker()
                except BaseException as exc:
                    result["error"] = repr(exc)
                finally:
                    checker_done.set()

            t_check = time.monotonic()
            threading.Thread(target=run_checker, daemon=True).start()
            if not checker_done.wait(timeout=self.verifier_timeout_sec):
                self._record_infrastructure_error(
                    f"verifier did not finish within {self.verifier_timeout_sec:g}s"
                )
                return
            if "error" in result:
                self._record_infrastructure_error(
                    f"verifier raised an exception: {result['error']}"
                )
                return

            self.verdict = result["verdict"]
            self.log.event(
                "verdict",
                passed=self.verdict.passed,
                score=self.verdict.score,
                detail=self.verdict.detail,
                checker_time_sec=time.monotonic() - t_check,
                grace_sec=self.grace_sec,
            )
        except BaseException as exc:
            self._record_infrastructure_error(
                f"gateway finisher failed: {type(exc).__name__}: {exc}"
            )
        finally:
            if self.fast_io:
                try:
                    # /done can finish the checker before its own HTTP reply.
                    # Do not expose fully_done until every accepted response
                    # and its exact archived frame have been recorded.
                    with self._responses:
                        drained = self._responses.wait_for(
                            lambda: self._active_responses == 0,
                            timeout=self.environment_operation_timeout_sec,
                        )
                    if not drained:
                        raise TimeoutError("gateway responses did not finish")
                    self.adapter.flush_recording()
                    self.log.flush()
                except Exception as exc:
                    self._record_infrastructure_error(f"recording drain failed: {exc}")
            if self._infrastructure_error is not None:
                try:
                    self.log.event(
                        "infrastructure_error",
                        error=self._infrastructure_error,
                    )
                    self.log.flush()
                except Exception:
                    pass
            self._fully_done.set()

    def _record_infrastructure_error(self, message: str) -> None:
        with self._lock:
            if self._infrastructure_error is None:
                self._infrastructure_error = message
            if not self._finished.is_set():
                self._t_end = time.monotonic()
                self._finish_reason = "infrastructure_error"
                self._finished.set()

    def fail_agent(self, message: str) -> bool:
        """Record an evaluator-observed submission failure.

        Only the control plane and the in-process executor can call this. The
        agent-facing token cannot manufacture a retry or choose its verdict.
        Returns false when the task already reached a terminal condition.
        """
        with self._lock:
            if self._finished.is_set():
                return False
            self._agent_error = message
            self._t_end = time.monotonic()
            self._finish_reason = "agent_error"
            self._finished.set()
        self.log.event("agent_failure", error=message)
        return True

    def join(self, timeout: float | None = None) -> str:
        """Block until the checker has run. Returns the finish reason."""
        self._fully_done.wait(timeout=timeout)
        return self._finish_reason or "done"

    # Back-compat convenience for the local executor: arm was already called.
    def wait(self) -> str:
        return self.join()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self.fast_io:
            self.log.flush()

    def status(self) -> dict[str, Any]:
        st = {
            "ready": True,
            "armed": self._armed.is_set(),
            "finished": self._finished.is_set(),
            "reason": self._finish_reason,
            "fully_done": self._fully_done.is_set(),
            "num_steps": self._num_steps,
        }
        if self.verdict is not None:
            st["verdict"] = {
                "passed": self.verdict.passed,
                "score": self.verdict.score,
                "detail": self.verdict.detail,
            }
        if self._infrastructure_error is not None:
            st["infrastructure_error"] = self._infrastructure_error
        if self._agent_error is not None:
            st["agent_error"] = self._agent_error
        if self._multi_episode:
            st.update({
                "agent_episode": self._agent_episode,
                "continuation_pending": self._continuation_pending,
                "transitioning_episode": self._transitioning_episode,
            })
        return st

    def continue_agent(self) -> dict[str, Any]:
        """Release the next independent agent episode after its setup.

        Only the evaluator control plane calls this.  The environment adapter
        has already evaluated the preceding episode and prepared the next one.
        """
        with self._episode_lock:
            if self._finished.is_set():
                raise ValueError("task is already finished")
            if not self._continuation_pending:
                raise ValueError("no agent continuation is pending")
            self._continuation_pending = False
            instruction = self.instruction
            episode = self._agent_episode
        self.log.event(
            "agent_episode_started",
            episode=episode,
            instruction=instruction,
        )
        return {"instruction": instruction, "agent_episode": episode}

    # -- request handling ----------------------------------------------------

    def _running(self) -> bool:
        return (
            self._armed.is_set()
            and not self._finished.is_set()
            and not self._continuation_pending
            and not self._transitioning_episode
        )

    def _timed_out(self) -> bool:
        return self._t0 is not None and (time.monotonic() - self._t0) > self.timeout_sec

    def _environment_operation(
        self,
        name: str,
        operation: Callable[[], Any],
        *,
        timeout_sec: float | None = None,
    ) -> Any:
        limit = (
            self.environment_operation_timeout_sec
            if timeout_sec is None
            else timeout_sec
        )
        with self._lock:
            operation_id = self._next_operation_id
            self._next_operation_id += 1
            self._active_operations[operation_id] = (name, time.monotonic(), limit)
        try:
            value = operation()
        except BaseException as error:
            message = (
                f"environment {name} failed: "
                f"{type(error).__name__}: {error}"
            )
            self._record_infrastructure_error(message)
            raise _EnvironmentOperationError(message) from error
        finally:
            with self._lock:
                self._active_operations.pop(operation_id, None)
        if self._infrastructure_error is not None:
            raise _EnvironmentOperationError(self._infrastructure_error)
        return value

    def _handle_observe(self, *, binary: bool = False) -> dict[str, Any]:
        operation = self.adapter.observe if binary else self.adapter.observe_png
        obs = self._environment_operation("observation", operation)
        return self._record_observation(obs, binary=binary)

    def _record_observation(self, obs: Observation, *, binary: bool) -> dict[str, Any]:
        try:
            data = obs.image
            mime_type = image_mime_type(data)
            meta = obs.meta
            # Old agents explicitly use the PNG JSON interface. The v2
            # binary path returns the original JPEG without any transcoding.
            if not binary and mime_type != "image/png":
                data = obs.png
                meta = {**meta, "format": "png", "source_encoding": meta.get("encoding"),
                        "encoding": {"format": "png", "compress_level": 1,
                                     "lossless": False, "converted_from": "jpeg"}}
                mime_type = "image/png"
            with Image.open(BytesIO(data)) as image:
                if Image.MIME[image.format] != mime_type:
                    raise OSError("image format does not match its MIME type")
                image.verify()
        except (AttributeError, TypeError, OSError, SyntaxError, ValueError) as exc:
            message = f"environment returned an invalid screenshot: {exc}"
            self._record_infrastructure_error(message)
            raise _EnvironmentOperationError(message) from exc
        with self._lock:
            frame_path = self.artifacts_dir / f"frame_{self._frame_idx:05d}{image_suffix(mime_type)}"
            self._frame_idx += 1
        self.log.artifact(frame_path, data)
        return {
            **({"_image": data} if binary else {"png_b64": base64.b64encode(data).decode()}),
            "meta": meta,
            "_frame": frame_path.name,
            "_timings": obs.timings,
            "_image_info": {"mime_type": mime_type, "encoding": meta.get("encoding")},
        }

    def _handle_step(self, body: dict[str, Any], *, observe: bool = False) -> dict[str, Any]:
        actions = body.get("actions", [])
        if not isinstance(actions, list):
            raise _AgentRequestError("'actions' must be a list")
        if not self._action_lock.acquire(blocking=False):
            return {
                "ok": False,
                "error": "Previous action is still running; no new action was executed.",
                "_actions": actions,
            }
        result: dict[str, Any] = {}
        done = threading.Event()

        def run_action() -> None:
            try:
                # A failed step is an action result, not evidence of VM death.
                if observe:
                    result["info"], result["observation"] = self.adapter.step_observe(actions)
                else:
                    result["info"] = self.adapter.step(actions)
            except BaseException as exc:
                message = (
                    f"Action failed: {type(exc).__name__}: {exc}. "
                    "The batch may have partially executed; observe before retrying."
                )
                result["error"] = message
                # Retain the error even if the HTTP action deadline passed.
                self.log.event("action_error", error=message, actions=actions)
            finally:
                self._action_lock.release()
                done.set()

        threading.Thread(target=run_action, daemon=True).start()
        if not done.wait(self.environment_operation_timeout_sec):
            message = (
                f"Action timed out after {self.environment_operation_timeout_sec:g}s. "
                "It may have partially executed and may still be running; observe before retrying."
            )
            self.log.event("action_timeout", error=message)
            return {"ok": False, "error": message, "_actions": actions}
        if "error" in result:
            return {"ok": False, "error": result["error"], "_actions": actions}
        payload = {"ok": True, "info": result["info"], "_actions": actions}
        if observe:
            payload.update(self._record_observation(result["observation"], binary=True))
        return payload

    def _handle_done(self) -> dict[str, Any]:
        if self._multi_episode:
            with self._action_lock, self._episode_lock:
                self._transitioning_episode = True
                try:
                    instruction = self._environment_operation(
                        "episode transition",
                        self.adapter.advance_episode,
                        timeout_sec=self.verifier_timeout_sec,
                    )
                    # The task clock can expire while a phase setup is in
                    # flight. The finisher will evaluate the resulting state;
                    # never release another agent after that deadline.
                    if self._finished.is_set():
                        instruction = None
                    if instruction is not None:
                        if not isinstance(instruction, str) or not instruction.strip():
                            message = (
                                "environment returned an invalid next-episode "
                                "instruction"
                            )
                            self._record_infrastructure_error(message)
                            raise _EnvironmentOperationError(message)
                        completed_episode = self._agent_episode
                        self._agent_episode += 1
                        self.instruction = instruction
                        self._continuation_pending = True
                        self.log.event(
                            "agent_episode_finished",
                            episode=completed_episode,
                            task_complete=False,
                        )
                        return {
                            "ok": True,
                            "task_complete": False,
                            "agent_episode": completed_episode,
                        }
                finally:
                    self._transitioning_episode = False
        with self._lock:
            if not self._finished.is_set():
                self._t_end = time.monotonic()
                self._finish_reason = "done"
                self._finished.set()
        if self._multi_episode:
            self.log.event(
                "agent_episode_finished",
                episode=self._agent_episode,
                task_complete=True,
            )
            return {"ok": True, "task_complete": True}
        return {"ok": True}

    def _make_handler(self):
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1" if gateway.fast_io else "HTTP/1.0"
            disable_nagle_algorithm = gateway.fast_io
            timeout = 120 if gateway.fast_io else None

            def log_message(self, *args):  # silence default stderr access log
                pass

            def _reply(self, code: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload).encode()
                self._reply_bytes(code, data, "application/json")

            def _reply_text(self, code: int, text: str) -> None:
                data = text.encode()
                self._reply_bytes(code, data, "text/plain; charset=utf-8")

            def _reply_bytes(self, code: int, data: bytes, content_type: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                if code >= 400 or (
                    self.headers.get("Content-Length", "0") != "0"
                    and not getattr(self, "_body_read", False)
                ) or self.headers.get("Transfer-Encoding"):
                    # Never interpret an unread body as the next request.
                    self.close_connection = True
                    self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)

            def _read_body(self) -> dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    if length < 0 or length > 16 * 1024 * 1024:
                        raise ValueError("invalid request body length")
                    body = json.loads(self.rfile.read(length) or b"{}")
                    self._body_read = True
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise _AgentRequestError("request body must be valid JSON") from exc
                if not isinstance(body, dict):
                    raise _AgentRequestError("request body must be a JSON object")
                return body

            # -- executor-facing control plane ------------------------------

            def _route_control(self, op: str, method: str) -> None:
                if method == "GET" and op == "health":
                    self._reply(200, {"ready": True})
                elif method == "GET" and op == "info":
                    info = {
                        "ready": True,
                        "instruction": gateway.instruction,
                        "run_token": gateway.token,
                    }
                    if gateway._multi_episode:
                        info.update({
                            "multi_episode": True,
                            "agent_episode": gateway._agent_episode,
                        })
                    self._reply(200, info)
                elif method == "POST" and op == "arm":
                    gateway.arm()
                    self._reply(200, {"armed": True, "t0": gateway._t0})
                elif method == "GET" and op == "status":
                    self._reply(200, gateway.status())
                elif method == "GET" and op == "runlog":
                    try:
                        gateway.log.flush()
                        text = Path(gateway.runlog_path).read_text()
                    except OSError:
                        text = ""
                    self._reply_text(200, text)
                elif method == "POST" and op == "agent_failed":
                    body = self._read_body()
                    error = str(body.get("error") or "agent exited before completion")
                    accepted = gateway.fail_agent(error)
                    self._reply(200, {"accepted": accepted})
                elif method == "POST" and op == "continue_agent":
                    try:
                        self._reply(200, gateway.continue_agent())
                    except ValueError as exc:
                        self._reply(409, {"error": str(exc)})
                elif method == "POST" and op == "shutdown":
                    self._reply(200, {"ok": True})
                    threading.Thread(target=gateway.shutdown, daemon=True).start()
                else:
                    self._reply(404, {"error": f"unknown control op '{op}'"})

            # -- agent-facing run plane -------------------------------------

            def _route_run(self, op: str, method: str) -> None:
                if not gateway.fast_io:
                    self._route_run_inner(op, method)
                    return
                with gateway._responses:
                    if gateway._finished.is_set():
                        self._reply(409, {"error": "run is not active"})
                        return
                    gateway._active_responses += 1
                arrived = time.monotonic()
                try:
                    self._route_run_inner(op, method)
                finally:
                    sent = time.monotonic()
                    try:
                        gateway.log.event(
                            "gateway_response", op=op,
                            t_mono_arrive=arrived, t_mono_sent=sent,
                            dur=sent - arrived,
                        )
                    except Exception as exc:
                        gateway._record_infrastructure_error(f"response recording failed: {exc}")
                    finally:
                        with gateway._responses:
                            gateway._active_responses -= 1
                            gateway._responses.notify_all()

            def _route_run_inner(self, op: str, method: str) -> None:
                if not gateway._running() or gateway._timed_out():
                    self._reply(409, {"error": "run is not active"})
                    return
                t_arrive = time.monotonic()
                binary = gateway.fast_io and (
                    op == "step-observe" or self.headers.get("Accept") == CONTENT_TYPE
                )
                try:
                    if method == "GET" and op == "observe":
                        payload = gateway._handle_observe(binary=binary)
                        if binary:
                            payload["ok"] = True
                        kind, fields = "observe", {"frame": payload.pop("_frame")}
                        fields.update(payload.pop("_image_info"))
                        timings = payload.pop("_timings")
                        if timings:
                            fields["io_timings"] = timings
                    elif method == "POST" and (op == "step" or (gateway.fast_io and op == "step-observe")):
                        payload = gateway._handle_step(self._read_body(), observe=op == "step-observe")
                        kind, fields = "step", {"actions": payload.pop("_actions")}
                        if "_frame" in payload:
                            fields["frame"] = payload.pop("_frame")
                            fields["combined_observation"] = True
                            fields.update(payload.pop("_image_info"))
                            timings = payload.pop("_timings")
                            if timings:
                                fields["io_timings"] = timings
                    elif method == "POST" and op == "done":
                        payload = gateway._handle_done()
                        kind, fields = "done", {}
                        if gateway._multi_episode:
                            fields = {
                                "task_complete": payload.get(
                                    "task_complete", True
                                ),
                                "agent_episode": payload.get(
                                    "agent_episode", gateway._agent_episode
                                ),
                            }
                    else:
                        self._reply(404, {"error": f"unknown operation '{op}'"})
                        return
                except _AgentRequestError as exc:
                    gateway.log.event("agent_request_error", op=op, error=repr(exc))
                    self._reply(400, {"error": str(exc)})
                    return
                except Exception as exc:  # surfaced to the agent, and logged
                    gateway.log.event("env_error", op=op, error=repr(exc))
                    if op in {"observe", "step", "step-observe"}:
                        gateway._record_infrastructure_error(
                            f"environment {op} failed: {type(exc).__name__}: {exc}"
                        )
                    self._reply(500, {"error": repr(exc)})
                    return
                t_complete = time.monotonic()
                gateway.log.event(
                    kind,
                    t_mono_arrive=t_arrive,
                    t_mono_complete=t_complete,
                    dur=t_complete - t_arrive,
                    **fields,
                )
                if fields.get("combined_observation"):
                    # The capture is part of step's interval, not a second
                    # timed operation. Keep observation counts/frame readers.
                    gateway.log.event(
                        "observe", frame=fields["frame"], combined=True,
                        mime_type=fields["mime_type"], encoding=fields["encoding"],
                        t_mono_arrive=t_complete, t_mono_complete=t_complete, dur=0.0,
                    )
                if binary and op in {"observe", "step-observe"}:
                    image = payload.pop("_image", b"")
                    self._reply_bytes(200, encode_observation(image, payload), CONTENT_TYPE)
                else:
                    self._reply(200, payload)
                # This counter exists only for untimed status reporting. It is
                # deliberately advanced after the response is sent so the TUI
                # cannot add work to the measured environment operation.
                if kind == "step":
                    with gateway._lock:
                        gateway._num_steps += 1

            def _route(self, method: str) -> None:
                self._body_read = False
                if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length", [])) > 1:
                    self._reply(400, {"error": "use one Content-Length and no Transfer-Encoding"})
                    return
                parts = self.path.strip("/").split("/")
                if len(parts) == 3 and parts[0] == "_ctl":
                    if parts[1] != gateway.control_token:
                        self._reply(403, {"error": "bad control token"})
                        return
                    self._route_control(parts[2], method)
                    return
                if len(parts) == 2 and parts[0] == gateway.token:
                    self._route_run(parts[1], method)
                    return
                self._reply(403, {"error": "bad token"})

            def do_GET(self):
                self._route("GET")

            def do_POST(self):
                self._route("POST")

        return Handler


@contextmanager
def monitor_gateway_steps(
    status: Callable[[], dict[str, Any]],
    on_progress: Callable[[int], None],
    *,
    poll_sec: float = 0.5,
) -> Iterator[None]:
    """Report live gateway step counts without entering the timed path.

    Local executors read ``Gateway.status`` directly and remote executors use
    the same control-plane status endpoint. Monitoring failures are ignored:
    progress reporting must never change whether an evaluation succeeds.
    """
    stopped = threading.Event()
    last_steps = 0

    def sample() -> None:
        nonlocal last_steps
        try:
            steps = int(status().get("num_steps") or 0)
        except Exception:
            return
        if steps > last_steps:
            last_steps = steps
            try:
                on_progress(steps)
            except Exception:
                pass

    def poll() -> None:
        while not stopped.wait(poll_sec):
            sample()

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=max(1.0, poll_sec * 2))
        sample()
