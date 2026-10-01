"""GatewayControl: the executor's client for a gateway's control plane.

The executor is never on the timed path. It uses this to arm the run, poll
for completion, and fetch the run log, whether the gateway is in-process on
localhost or inside a Modal sandbox reached over a tunnel.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from cua_speedrun.evaluation_runtime import InstanceInfrastructureError


class GatewayControl:
    def __init__(self, base_url: str, control_token: str, timeout_sec: float = 30.0):
        # base_url is scheme://host[:port], with no trailing path.
        self.base = base_url.rstrip("/")
        self.token = control_token
        self.timeout_sec = timeout_sec
        self._s = requests.Session()

    def _url(self, op: str) -> str:
        return f"{self.base}/_ctl/{self.token}/{op}"

    def _request(self, method: str, op: str, **kwargs: Any) -> requests.Response:
        try:
            response = self._s.request(
                method,
                self._url(op),
                timeout=self.timeout_sec,
                **kwargs,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            raise InstanceInfrastructureError(
                f"gateway control operation {op!r} failed: {exc}"
            ) from exc

    def _json(self, method: str, op: str, **kwargs: Any) -> dict[str, Any]:
        response = self._request(method, op, **kwargs)
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise InstanceInfrastructureError(
                f"gateway control operation {op!r} returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise InstanceInfrastructureError(
                f"gateway control operation {op!r} returned a non-object response"
            )
        return payload

    def wait_healthy(self, budget_sec: float, poll: float = 2.0) -> bool:
        deadline = time.time() + budget_sec
        while time.time() < deadline:
            try:
                r = self._s.get(self._url("health"), timeout=self.timeout_sec)
                if r.status_code == 200 and r.json().get("ready"):
                    return True
            except requests.RequestException:
                pass
            time.sleep(poll)
        return False

    def info(self) -> dict[str, Any]:
        return self._json("GET", "info")

    def arm(self) -> dict[str, Any]:
        return self._json("POST", "arm")

    def status(self) -> dict[str, Any]:
        return self._json("GET", "status")

    def fail_agent(self, error: str) -> bool:
        response = self._json(
            "POST", "agent_failed", json={"error": str(error)}
        )
        return bool(response.get("accepted"))

    def continue_agent(self) -> dict[str, Any]:
        """Start the next agent episode prepared by the environment."""
        return self._json("POST", "continue_agent")

    def wait_done(self, budget_sec: float, poll: float = 1.0) -> dict[str, Any]:
        deadline = time.time() + budget_sec
        last: dict[str, Any] = {}
        while time.time() < deadline:
            last = self.status()
            if last.get("fully_done"):
                if last.get("infrastructure_error"):
                    raise InstanceInfrastructureError(
                        str(last["infrastructure_error"])
                    )
                return last
            time.sleep(poll)
        raise InstanceInfrastructureError(
            f"gateway did not report a terminal result within {budget_sec:g}s"
        )

    def runlog(self) -> str:
        # Decode explicitly: the run log is UTF-8 bytes, and without a charset
        # in the Content-Type header requests would fall back to latin-1,
        # scrambling multibyte characters into spurious control characters.
        return self._request("GET", "runlog").content.decode("utf-8", "replace")

    def shutdown(self) -> None:
        try:
            self._request("POST", "shutdown")
        except InstanceInfrastructureError:
            pass
