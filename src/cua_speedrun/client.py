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
import time
from typing import Any

import requests

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
    def __init__(self, env_url: str, timeout_sec: float = 120.0):
        self.env_url = env_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self._session = requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        retry_on: type[Exception] = requests.exceptions.ConnectionError,
        **kwargs: Any,
    ) -> requests.Response:
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
            resp.raise_for_status()
            return resp
        raise AssertionError("unreachable")

    # -- core operations ---------------------------------------------------

    def observe(self) -> dict[str, Any]:
        """Returns {"png": bytes, "meta": dict}."""
        payload = self._request("GET", "observe").json()
        return {"png": base64.b64decode(payload["png_b64"]), "meta": payload["meta"]}

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
