"""The agent-side client's bounded connect retries.

Retries must absorb connect-level blips
without ever risking a double-executed step: only errors that prove the
request never reached the gateway are retried on non-idempotent calls.
"""

from __future__ import annotations

import base64

import pytest
import requests

import cua_speedrun.client as client_module
from cua_speedrun.client import Computer


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, outcomes: list):
        # Each entry is either an Exception to raise or a payload to return.
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str]] = []

    def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(outcome)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)


def _computer(outcomes: list) -> tuple[Computer, _FakeSession]:
    computer = Computer("http://env.invalid/token")
    session = _FakeSession(outcomes)
    computer._session = session
    return computer, session


def test_observe_retries_connection_errors() -> None:
    payload = {"png_b64": base64.b64encode(b"png").decode(), "meta": {}}
    computer, session = _computer(
        [requests.exceptions.ConnectionError("blip"), payload]
    )
    assert computer.observe()["png"] == b"png"
    assert len(session.calls) == 2


def test_observe_gives_up_after_the_retry_budget() -> None:
    computer, session = _computer([requests.exceptions.ConnectionError("down")] * 3)
    with pytest.raises(requests.exceptions.ConnectionError):
        computer.observe()
    assert len(session.calls) == 3


def test_step_retries_only_connect_timeouts() -> None:
    computer, session = _computer(
        [requests.exceptions.ConnectTimeout("queue"), {"done": False}]
    )
    assert computer.step([{"mouse": {"left_click": [1, 2]}}]) == {"done": False}
    assert len(session.calls) == 2


def test_step_recovers_a_generic_connection_error_without_replaying() -> None:
    # A reset mid-response may mean the gateway already executed the step;
    # recover through an observation without double-executing the action.
    payload = {"png_b64": base64.b64encode(b"recovered").decode(), "meta": {}}
    computer, session = _computer(
        [requests.exceptions.ConnectionError("reset"), payload]
    )
    result = computer.step([{"mouse": {"left_click": [1, 2]}}])

    assert result == {
        "ok": True,
        "info": {"connection_recovered": True, "step_outcome": "unknown"},
    }
    assert session.calls == [
        ("POST", "http://env.invalid/token/step"),
        ("GET", "http://env.invalid/token/observe"),
    ]


def test_step_surfaces_error_when_recovery_observation_fails() -> None:
    computer, session = _computer(
        [requests.exceptions.ConnectionError("step reset")]
        + [requests.exceptions.ConnectionError("gateway down")] * 3
    )

    with pytest.raises(requests.exceptions.ConnectionError, match="gateway down"):
        computer.step([{"mouse": {"left_click": [1, 2]}}])

    assert [method for method, _url in session.calls] == ["POST", "GET", "GET", "GET"]


def test_read_timeouts_are_never_retried() -> None:
    computer, session = _computer([requests.exceptions.ReadTimeout("slow")])
    with pytest.raises(requests.exceptions.ReadTimeout):
        computer.observe()
    assert len(session.calls) == 1
