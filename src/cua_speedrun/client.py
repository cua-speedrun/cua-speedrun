"""The agent-side client. This is the whole world an agent.py sees.

    from cua_speedrun.client import Computer

    computer = Computer(env_url)
    obs = computer.observe()
    computer.click(640, 400)
    computer.type_text("hello")
    computer.done()

Action dicts follow gym-anything's schema, and the convenience methods
below just build them. `step` accepts a raw list of action dicts, so an
agent can send a planned batch in one round trip.
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any

import requests

from cua_speedrun.fast_io import CONTENT_TYPE, decode_observation

# Connection-level failures are retried a bounded number of times before
# they surface, so a single tunnel blip does not kill the agent. What is
# safe to retry depends on the operation:
#
# - observe and done are idempotent, so any ConnectionError (refused,
#   reset, connect timeout) can be retried.
# - step is NOT idempotent, and a ConnectionError can also mean the
#   connection dropped while reading the response, after the gateway
#   already executed the actions. Only ConnectTimeout guarantees the
#   request never reached the gateway, so step retries only on that. If
#   the response is still lost, step recovers by observing the desktop;
#   it never sends the actions a second time.
#
# Read timeouts and HTTP errors are never retried. The retry pauses sit
# on the timed clock, so the budget is deliberately small: it absorbs a
# transient blip, not an outage.
_CONNECT_RETRIES = 2
_CONNECT_RETRY_DELAY_SEC = 1.0


class Computer:
    def __init__(self, env_url: str, timeout_sec: float = 120.0, *, fast_io: bool | None = None):
        self.env_url = env_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self._session = requests.Session()
        self.fast_io = (
            os.environ.get("GYM_ANYTHING_FAST_IO", "").strip().lower() in {"true", "1"}
            if fast_io is None else fast_io
        )
        self.last_request_timing: dict[str, Any] | None = None
        self._profile_transport = self.fast_io and os.environ.get(
            "CS_FAST_IO_PROFILE", "false"
        ).lower() in {"true", "1"}
        self._last_socket = None

    def _request(
        self,
        method: str,
        path: str,
        retry_on: type[Exception] = requests.exceptions.ConnectionError,
        **kwargs: Any,
    ) -> requests.Response:
        started = time.perf_counter()
        profile: dict[str, Any] = {}

        def received_headers(response, *args, **hook_kwargs):
            # Requests dispatches this hook before consuming the response body.
            headers_at = time.perf_counter()
            profile["headers_ready_sec"] = headers_at - started
            profile["requests_headers_sec"] = response.elapsed.total_seconds()
            connection = getattr(response.raw, "_connection", None)
            sock = getattr(connection, "sock", None)
            if sock is not None:
                try:
                    signature = (id(sock), sock.getsockname(), sock.getpeername())
                    profile["connection_reused"] = signature == self._last_socket
                    self._last_socket = signature
                except OSError:
                    pass
            profile["http_version"] = response.raw.version
            profile["hook_sec"] = time.perf_counter() - headers_at

        if self._profile_transport:
            kwargs["hooks"] = {"response": received_headers}
        for attempt in range(_CONNECT_RETRIES + 1):
            try:
                resp = self._session.request(
                    method,
                    f"{self.env_url}/{path}",
                    timeout=self.timeout_sec,
                    **kwargs,
                )
            except requests.exceptions.ConnectionError as exc:
                if attempt == _CONNECT_RETRIES or not isinstance(exc, retry_on):
                    raise
                time.sleep(_CONNECT_RETRY_DELAY_SEC * (attempt + 1))
                continue
            received_at = time.perf_counter()
            resp.raise_for_status()
            if self.fast_io:
                self.last_request_timing = {
                    "operation": path,
                    "round_trip_sec": received_at - started,
                    "attempts": attempt + 1,
                    "response_bytes": len(resp.content),
                }
                if profile:
                    profile["body_receive_sec"] = (
                        received_at - started - profile["headers_ready_sec"] - profile["hook_sec"]
                    )
                    self.last_request_timing["transport_profile"] = profile
            return resp
        raise AssertionError("unreachable")

    # -- core operations ---------------------------------------------------

    def observe(self) -> dict[str, Any]:
        """Legacy PNG interface. Use observe_image for JPEG-aware fast I/O."""
        payload = self._request("GET", "observe").json()
        return {"png": base64.b64decode(payload["png_b64"]), "meta": payload["meta"]}

    def observe_image(self) -> dict[str, Any]:
        """Returns image bytes, their MIME type, and unchanged metadata."""
        if self.fast_io:
            return decode_observation(self.observe_wire())
        payload = self.observe()
        return {"image": payload["png"], "mime_type": "image/png", "meta": payload["meta"]}

    def _observation_wire(self, method: str, path: str, **kwargs: Any) -> bytes:
        response = self._request(
            method, path, headers={"Accept": CONTENT_TYPE},
            allow_redirects=False, **kwargs,
        )
        if 300 <= response.status_code < 400:
            raise ValueError("gateway redirected an observation request; action was not replayed")
        if response.headers.get("Content-Type") != CONTENT_TYPE:
            raise ValueError("gateway does not support binary observations")
        return response.content

    def observe_wire(self) -> bytes:
        """For pass-through proxies: no decoding or re-encoding of the body."""
        return self._observation_wire("GET", "observe")

    def step_observe_wire(self, actions: list[dict[str, Any]]) -> bytes:
        # A lost response surfaces as an error. Never replay, fall back to
        # another step, or report a later observation as the action's frame.
        return self._observation_wire(
            "POST", "step-observe", retry_on=requests.exceptions.ConnectTimeout,
            json={"actions": actions},
        )

    def step_observe(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        """One request, one action batch, its screenshot, and backend metadata.

        On an unknown action outcome, call observe separately; do not replay.
        """
        return decode_observation(self.step_observe_wire(actions))

    def close(self) -> None:
        self._session.close()

    def step(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            response = self._request(
                "POST",
                "step",
                retry_on=requests.exceptions.ConnectTimeout,
                json={"actions": actions},
            )
        except requests.exceptions.ConnectionError:
            # The gateway may already have executed the actions. Never replay
            # them: recover the current state and let the agent's next turn
            # decide what, if anything, still needs to be done.
            self.observe()
            return {
                "ok": True,
                "info": {
                    "connection_recovered": True,
                    "step_outcome": "unknown",
                },
            }
        return response.json()

    def done(self) -> None:
        self._request("POST", "done")

    # -- convenience action builders ----------------------------------------

    def click(self, x: int, y: int) -> None:
        self.step([{"mouse": {"left_click": [x, y]}}])

    def type_text(self, text: str) -> None:
        self.step([{"keyboard": {"text": text}}])

    def keys(self, keys: list[str]) -> None:
        self.step([{"keyboard": {"keys": keys}}])

    def wait(self, seconds: float) -> None:
        self.step([{"action": "wait", "time": seconds}])
