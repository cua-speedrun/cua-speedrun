from __future__ import annotations

from io import BytesIO
import subprocess
import threading
import time

import pytest
import requests
from PIL import Image

from cua_speedrun.client import Computer
from cua_speedrun.envs.base import Observation, Verdict
from cua_speedrun.gateway import Gateway
from cua_speedrun.remote import run as remote_run
from cua_speedrun.runlog import RunLogWriter, load_runlog, summarize


def _png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (2, 2)).save(output, format="PNG")
    return output.getvalue()


class _MultiEpisodeAdapter:
    multi_episode = True

    def __init__(self) -> None:
        self.transitions = 0
        self.finalizations = 0

    def observe(self) -> Observation:
        return Observation(_png(), {})

    def step(self, _actions):
        return {}

    def advance_episode(self) -> str | None:
        self.transitions += 1
        return "second instruction" if self.transitions == 1 else None

    def finalize(self) -> Verdict:
        self.finalizations += 1
        return Verdict(True, 100.0, "all episodes complete")

    def close(self) -> None:
        return None


class _SingleEpisodeAdapter:
    def observe(self) -> Observation:
        return Observation(_png(), {})

    def step(self, _actions):
        return {}

    def finalize(self) -> Verdict:
        return Verdict(True, 100.0, "complete")

    def close(self) -> None:
        return None


@pytest.mark.parametrize("error", [
    ValueError("unsupported key"),
    RuntimeError("guest action exited with status 1"),
    subprocess.TimeoutExpired("guest action", 90),
])
@pytest.mark.parametrize("after_timeout", [False, True])
def test_step_exception_is_an_action_failure(tmp_path, error, after_timeout) -> None:
    release = threading.Event()
    attempted = []

    class Adapter(_SingleEpisodeAdapter):
        def step(self, actions):
            attempted.append(actions)
            if len(attempted) == 1:
                assert release.wait(5)
                raise error
            return {"executed": True}

    log = RunLogWriter(tmp_path / "runlog.jsonl")
    gateway = Gateway(Adapter(), log, tmp_path, timeout_sec=10, grace_sec=0,
                      environment_operation_timeout_sec=0.02 if after_timeout else 2)
    computer = Computer(gateway.start())
    gateway.arm()
    actions = [{"keyboard": {"keys": ["unsupported"]}}]
    recovery = [{"keyboard": {"keys": ["escape"]}}]
    if not after_timeout:
        release.set()
    try:
        response = computer.step(actions)
        assert response["ok"] is False
        if after_timeout:
            assert "timed out" in response["error"]
            release.set()
            with gateway._action_lock:
                pass
        else:
            assert f"{type(error).__name__}: {error}" in response["error"]
        assert "infrastructure_error" not in gateway.status()
        assert not gateway.status()["finished"]
        assert computer.observe()["png"].startswith(b"\x89PNG")
        assert computer.step(recovery) == {"ok": True, "info": {"executed": True}}
        computer.done()
        gateway.join(timeout=2)

        assert attempted == [actions, recovery]  # No automatic replay.
        assert gateway.verdict.passed
        assert "infrastructure_error" not in gateway.status()
        assert gateway.status()["num_steps"] == 2
        events = load_runlog(log.path)
        assert summarize(events)["num_steps"] == 2
        errors = [event for event in events if event["event"] == "action_error"]
        assert len(errors) == 1
        assert f"{type(error).__name__}: {error}" in errors[0]["error"]
    finally:
        release.set()
        gateway.shutdown()
        log.close()


@pytest.mark.parametrize("operation", ["observe", "finalize"])
def test_environment_failure_is_still_infrastructure_failure(tmp_path, operation) -> None:
    class Adapter(_SingleEpisodeAdapter):
        def observe(self):
            if operation == "observe":
                raise RuntimeError("environment unavailable")
            return super().observe()

        def finalize(self):
            raise RuntimeError("verifier unavailable")

    log = RunLogWriter(tmp_path / "runlog.jsonl")
    gateway = Gateway(Adapter(), log, tmp_path, timeout_sec=10, grace_sec=0)
    computer = Computer(gateway.start())
    gateway.arm()
    try:
        if operation == "observe":
            with pytest.raises(requests.HTTPError):
                computer.observe()
        else:
            computer.done()
        gateway.join(timeout=2)
        assert gateway.status()["finished"]
        assert "unavailable" in gateway.status()["infrastructure_error"]
        assert gateway.verdict is None
    finally:
        gateway.shutdown()
        log.close()


@pytest.mark.parametrize("recover", [False, True])
def test_action_timeout_is_recoverable_and_verification_waits(tmp_path, recover) -> None:
    release = threading.Event()
    checked = threading.Event()

    class Adapter(_SingleEpisodeAdapter):
        def step(self, actions):
            assert release.wait(5)
            return {"executed": True}

        def finalize(self):
            checked.set()
            return super().finalize()

    log = RunLogWriter(tmp_path / "runlog.jsonl")
    gateway = Gateway(Adapter(), log, tmp_path, timeout_sec=10, grace_sec=0,
                      environment_operation_timeout_sec=0.02)
    url = gateway.start()
    gateway.arm()
    try:
        response = requests.post(url + "/step", json={"actions": []}).json()
        assert response["ok"] is False
        assert "timed out" in response["error"]
        time.sleep(0.3)  # Cover a watchdog tick after the action deadline.
        assert "infrastructure_error" not in gateway.status()
        assert not gateway.status()["finished"]
        response = requests.post(url + "/step", json={"actions": []}).json()
        assert response["ok"] is False
        assert "no new action" in response["error"]
        if recover:
            release.set()
            with gateway._action_lock:
                pass
            response = requests.post(url + "/step", json={"actions": []}).json()
            assert response == {"ok": True, "info": {"executed": True}}
        requests.post(url + "/done").raise_for_status()
        if not recover:
            assert not checked.wait(0.05)
        release.set()
        gateway.join(timeout=2)
        assert checked.is_set()
        assert gateway.verdict.passed
        assert "infrastructure_error" not in gateway.status()
        assert gateway.status()["num_steps"] == (3 if recover else 2)
    finally:
        release.set()
        gateway.shutdown()
        log.close()


@pytest.fixture
def multi_gateway(tmp_path):
    adapter = _MultiEpisodeAdapter()
    log = RunLogWriter(tmp_path / "runlog.jsonl")
    gateway = Gateway(
        adapter,
        log,
        tmp_path,
        timeout_sec=30,
        grace_sec=0,
        instruction="first instruction",
    )
    url = gateway.start()
    gateway.arm()
    try:
        yield gateway, adapter, Computer(url), log
    finally:
        gateway.shutdown()
        log.close()


def test_gateway_runs_multiple_agent_episodes_without_finishing_early(
    multi_gateway,
) -> None:
    gateway, adapter, computer, _log = multi_gateway

    computer.done()
    status = gateway.status()
    assert status["finished"] is False
    assert status["continuation_pending"] is True
    assert status["agent_episode"] == 2
    with pytest.raises(requests.HTTPError):
        computer.observe()

    continuation = gateway.continue_agent()
    assert continuation == {
        "instruction": "second instruction",
        "agent_episode": 2,
    }
    assert computer.observe()["png"].startswith(b"\x89PNG")
    computer.done()
    gateway.join(timeout=2)

    assert gateway.status()["finished"] is True
    assert gateway.verdict == Verdict(True, 100.0, "all episodes complete")
    assert adapter.transitions == 2
    assert adapter.finalizations == 1


def test_single_episode_gateway_contract_is_unchanged(tmp_path) -> None:
    log = RunLogWriter(tmp_path / "runlog.jsonl")
    gateway = Gateway(
        _SingleEpisodeAdapter(),
        log,
        tmp_path,
        timeout_sec=30,
        grace_sec=0,
        instruction="ordinary instruction",
    )
    url = gateway.start()
    gateway.arm()
    try:
        info = requests.get(
            f"{url.rsplit('/', 1)[0]}/_ctl/{gateway.control_token}/info",
            timeout=2,
        ).json()
        response = requests.post(f"{url}/done", timeout=2)
        response.raise_for_status()
        gateway.join(timeout=2)

        assert info == {
            "ready": True,
            "instruction": "ordinary instruction",
            "run_token": gateway.token,
        }
        assert response.json() == {"ok": True}
        assert not {
            "multi_episode",
            "agent_episode",
            "continuation_pending",
        } & gateway.status().keys()
        done_event = next(
            event for event in load_runlog(log.path) if event["event"] == "done"
        )
        assert "task_complete" not in done_event
        assert "agent_episode" not in done_event
    finally:
        gateway.shutdown()
        log.close()


def test_runlog_counts_only_intermediate_episode_transition_as_env_time(
    multi_gateway,
) -> None:
    gateway, _adapter, computer, log = multi_gateway

    computer.done()
    gateway.continue_agent()
    computer.done()
    gateway.join(timeout=2)
    events = load_runlog(log.path)
    row = summarize(events)
    done_events = [event for event in events if event["event"] == "done"]

    assert [event["task_complete"] for event in done_events] == [False, True]
    assert row["env_time_sec"] == pytest.approx(done_events[0]["dur"])


def test_timeout_during_episode_transition_never_releases_another_agent(
    tmp_path,
) -> None:
    transition_started = threading.Event()
    release_transition = threading.Event()

    class SlowTransitionAdapter(_MultiEpisodeAdapter):
        def advance_episode(self) -> str:
            transition_started.set()
            assert release_transition.wait(timeout=2)
            return "second instruction"

    adapter = SlowTransitionAdapter()
    log = RunLogWriter(tmp_path / "runlog.jsonl")
    gateway = Gateway(
        adapter,
        log,
        tmp_path,
        timeout_sec=0.05,
        grace_sec=0,
        instruction="first instruction",
    )
    url = gateway.start()
    gateway.arm()
    responses = []
    request = threading.Thread(
        target=lambda: responses.append(requests.post(f"{url}/done", timeout=2))
    )
    request.start()
    try:
        assert transition_started.wait(timeout=1)
        time.sleep(0.1)
        release_transition.set()
        request.join(timeout=2)
        gateway.join(timeout=2)

        assert len(responses) == 1
        responses[0].raise_for_status()
        assert responses[0].json() == {"ok": True, "task_complete": True}
        assert gateway.status()["reason"] == "timeout"
        assert gateway.status()["continuation_pending"] is False
        assert adapter.finalizations == 1
    finally:
        release_transition.set()
        request.join(timeout=2)
        gateway.shutdown()
        log.close()


def test_modal_executor_restarts_agent_with_each_phase_instruction(
    monkeypatch,
) -> None:
    calls = []

    class Control:
        episode = 1
        continuation_pending = False
        finished = False

        def info(self):
            return {"agent_episode": self.episode}

        def status(self):
            return {
                "agent_episode": self.episode,
                "continuation_pending": self.continuation_pending,
                "finished": self.finished,
            }

        def continue_agent(self):
            assert self.continuation_pending
            self.continuation_pending = False
            return {
                "instruction": "second instruction",
                "agent_episode": self.episode,
            }

        def fail_agent(self, _error):
            raise AssertionError("a completed episode is not an agent failure")

    class Events:
        def emit(self, *_args, **_kwargs):
            return None

    control = Control()

    def exec_agent(_agent, _url, instruction, **kwargs):
        calls.append((instruction, kwargs["task_id"]))
        if control.episode == 1:
            control.episode = 2
            control.continuation_pending = True
        else:
            control.finished = True
        return 0, f"stdout {instruction}\n", f"stderr {instruction}\n"

    monkeypatch.setattr(remote_run, "exec_agent", exec_agent)

    code, stdout, stderr = remote_run._run_agent_episodes(
        control,
        object(),
        "http://environment/run-token",
        "first instruction",
        task_id="task",
        timeout_sec=600,
        events=Events(),
        task_key="task/seed_0",
    )

    assert calls == [
        ("first instruction", "task_episode_1"),
        ("second instruction", "task_episode_2"),
    ]
    assert code == 0
    assert stdout == "stdout first instruction\nstdout second instruction\n"
    assert stderr == "stderr first instruction\nstderr second instruction\n"
