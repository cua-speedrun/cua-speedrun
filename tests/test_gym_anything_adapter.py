from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cua_speedrun.envs import gym_anything
from cua_speedrun.envs.gym_anything import GymAnythingAdapter, GymAnythingBackend


class DummyRunner:
    def __init__(self) -> None:
        self.injected: list[dict[str, Any]] = []
        self.pyautogui_commands: list[str] = []

    def inject_action(self, action: dict[str, Any]) -> None:
        self.injected.append(action)

    def _normalize_key_name(self, key: str) -> str:
        return {"Shift_L": "shift"}.get(key, key.lower())

    def _run_pyautogui(self, commands: list[str]) -> None:
        self.pyautogui_commands.extend(commands)


class DummyEnv:
    def __init__(self) -> None:
        self._runner = DummyRunner()

    def step(
        self,
        actions: list[dict[str, Any]],
        wait_between_actions: float = 0.0,
        settle_sec: float = 0.0,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        for action in actions:
            self._runner.inject_action(action)
        return {}, 0.0, False, {}


class DummyEnvWithoutSettle(DummyEnv):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def step(
        self,
        actions: list[dict[str, Any]],
        wait_between_actions: float = 0.0,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.calls += 1
        for action in actions:
            self._runner.inject_action(action)
        return {}, 0.0, False, {}


class DummyQemuRunner:
    def __init__(self, checkpoint_dir: Path) -> None:
        self.base_qcow2 = Path("/tmp/base_ubuntu_gnome.qcow2")
        self.env_hash = "defaultenv"
        self.env_checkpoint = checkpoint_dir / "checkpoint_defaultenv.qcow2"
        self._ssh_user = "ga"
        self._ssh_password = "password123"
        self.ssh_port = 2222
        self.is_android = False
        self.is_windows = False
        self.ssh_commands: list[str] = []
        self.sftp_from: list[tuple[str, str]] = []
        self.sftp_to: list[tuple[str, str]] = []
        self.original_copies: list[tuple[str, str, str]] = []

    def _run_ssh_cmd(self, port: int, cmd: str) -> str:
        self.ssh_commands.append(cmd)
        return cmd

    def _ssh_command(self, cmd: str) -> str:
        self.ssh_commands.append(cmd)
        return cmd

    def _sftp_copy_from(self, container_src: str, host_dst: str) -> None:
        self.sftp_from.append((container_src, host_dst))

    def _sftp_copy_to(self, host_src: str, container_dst: str) -> None:
        self.sftp_to.append((host_src, container_dst))

    def copy_from(self, container_src: str, host_dst: str) -> None:
        self.original_copies.append(("from", container_src, host_dst))

    def copy_to(self, host_src: str, container_dst: str) -> None:
        self.original_copies.append(("to", host_src, container_dst))


class DummyQemuEnv:
    def __init__(self, runner: DummyQemuRunner) -> None:
        self._runner = runner


class DummyQemuRunnerWithoutBase:
    pass


class DummyPreparedEnv:
    def __init__(self) -> None:
        self.runner_name = "QemuApptainerRunner"
        self._runner = DummyRunner()
        self.env_spec = SimpleNamespace(recording=SimpleNamespace(output_dir=None))
        self.task_spec = SimpleNamespace(natural_language="Do the task", description="")
        self.max_steps = 37
        self.events: list[str] = []
        self.limits: list[dict[str, Any]] = []

    def reset(self, **kwargs: Any) -> None:
        self.events.append("reset")

    def set_episode_limits(self, **kwargs: Any) -> None:
        self.events.append("limits")
        self.limits.append(kwargs)

    def capture_observation(self) -> dict[str, Any]:
        png = base64.b64encode(
            b"\x89PNG\r\n\x1a\n" + b"unused by the header-only prepare check"
        ).decode()
        return {"screen": {"png_b64": png}}

    def close(self) -> None:
        pass


def test_gym_anything_adapter_executes_keyboard_hold_actions() -> None:
    env = DummyEnv()
    adapter = GymAnythingAdapter(env, settle_sec=0.0)

    adapter.step(
        [
            {"keyboard": {"key_down": "Shift_L"}},
            {"mouse": {"move": [10, 20]}},
            {"keyboard": {"key_up": "Shift_L"}},
        ]
    )

    assert env._runner.pyautogui_commands == [
        'pyautogui.keyDown("shift")',
        'pyautogui.keyUp("shift")',
    ]
    assert env._runner.injected == [{"mouse": {"move": [10, 20]}}]


def test_gym_anything_adapter_falls_back_without_settle_sec() -> None:
    env = DummyEnvWithoutSettle()
    adapter = GymAnythingAdapter(env, settle_sec=0.0)

    adapter.step([{"mouse": {"move": [1, 2]}}])
    adapter.step([{"mouse": {"move": [3, 4]}}])

    assert env.calls == 2
    assert env._runner.injected == [
        {"mouse": {"move": [1, 2]}},
        {"mouse": {"move": [3, 4]}},
    ]


def test_gym_anything_adapter_preserves_final_verifier_report(monkeypatch: Any) -> None:
    env = DummyEnv()
    adapter = GymAnythingAdapter(env, settle_sec=0.0)
    results = iter(
        [
            ({}, 0.0, True, {"verifier": {"passed": True, "score": 100}}),
            ({}, 0.0, True, {"verifier": None}),
        ]
    )
    monkeypatch.setattr(adapter, "_step_env", lambda actions: next(results))

    adapter.step([{"mouse": {"left_click": [1, 2]}}])
    adapter.step([{"mouse": {"left_click": [3, 4]}}])
    verdict = adapter.finalize()

    assert verdict.passed is True
    assert verdict.score == 100


def test_gym_anything_adapter_surfaces_verifier_errors(monkeypatch: Any) -> None:
    env = DummyEnv()
    adapter = GymAnythingAdapter(env, settle_sec=0.0)
    monkeypatch.setattr(
        adapter,
        "_step_env",
        lambda actions: (
            {},
            0.0,
            True,
            {"verifier": {"error": "verifier error: canonical endpoint unavailable"}},
        ),
    )

    adapter.step([{"mouse": {"left_click": [1, 2]}}])

    with pytest.raises(RuntimeError, match="canonical endpoint unavailable"):
        adapter.finalize()


def test_backend_disables_reset_time_timeout_but_preserves_max_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = DummyPreparedEnv()
    monkeypatch.setattr("gym_anything.from_config", lambda *args, **kwargs: env)
    monkeypatch.setattr(
        gym_anything, "scrub_guest_privileged_material", lambda prepared: []
    )
    backend = GymAnythingBackend(runner="qemu")
    monkeypatch.setattr(backend, "_apply_qemu_env_overrides", lambda *args: None)

    backend.prepare(
        {"env_dir": str(tmp_path), "task_id": "task"}, seed=7, workdir=tmp_path
    )

    assert env.events[:2] == ["reset", "limits"]
    assert env.limits == [{"max_steps": 37, "timeout_sec": None}]


def test_backend_applies_benchmark_max_steps_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = DummyPreparedEnv()
    monkeypatch.setattr("gym_anything.from_config", lambda *args, **kwargs: env)
    monkeypatch.setattr(
        gym_anything, "scrub_guest_privileged_material", lambda prepared: []
    )
    backend = GymAnythingBackend(runner="qemu")
    monkeypatch.setattr(backend, "_apply_qemu_env_overrides", lambda *args: None)

    backend.prepare(
        {"env_dir": str(tmp_path), "task_id": "task", "max_steps": 500},
        seed=7,
        workdir=tmp_path,
    )

    assert env.limits == [{"max_steps": 500, "timeout_sec": None}]


def test_backend_forwards_generic_config_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = DummyPreparedEnv()
    calls: list[dict[str, Any]] = []

    def from_config(*_args: Any, **kwargs: Any) -> DummyPreparedEnv:
        calls.append(kwargs)
        return env

    monkeypatch.setattr("gym_anything.from_config", from_config)
    monkeypatch.setattr(
        gym_anything, "scrub_guest_privileged_material", lambda prepared: []
    )
    backend = GymAnythingBackend(runner="qemu")
    monkeypatch.setattr(backend, "_apply_qemu_env_overrides", lambda *args: None)

    backend.prepare(
        {
            "env_dir": str(tmp_path),
            "task_id": "task",
            "config_overrides": {"resources": {"cpu": 8, "mem_gb": 32}},
        },
        seed=7,
        workdir=tmp_path,
    )

    assert calls == [
        {
            "task_id": "task",
            "overrides": {"resources": {"cpu": 8, "mem_gb": 32}},
        }
    ]


def test_backend_expands_only_the_per_run_qemu_overlay(
    tmp_path: Path,
) -> None:
    base = tmp_path / "official-base.qcow2"
    base.write_bytes(b"base")
    calls: list[list[str]] = []

    class Runner:
        def __init__(self) -> None:
            self.base_qcow2 = base
            self.env_hash = "env"
            self.env_checkpoint = tmp_path / "checkpoint_env.qcow2"

        def _run_qemu_img(self, args: list[str]) -> SimpleNamespace:
            calls.append(args)
            if args[0] == "info":
                return SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(
                        {"virtual-size": 30 * 1024**3, "format": "qcow2"}
                    ),
                    stderr="",
                )
            Path(args[-2]).write_bytes(b"overlay")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    runner = Runner()
    GymAnythingBackend()._apply_qemu_volume_size(
        SimpleNamespace(_runner=runner),
        100,
        tmp_path,
    )

    expanded = tmp_path / "qemu-base-100g.qcow2"
    assert runner.base_qcow2 == expanded
    assert base.read_bytes() == b"base"
    assert calls[1][-2:] == [str(expanded), "100G"]
    assert runner._cs_requested_qemu_volume_size_gb == 100


def test_backend_applies_configured_qemu_image_and_ssh(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv("GYM_ANYTHING_QEMU_X11_DISPLAY", raising=False)
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    image_path = tmp_path / "osworld_ubuntu.qcow2"
    image_path.write_bytes(b"fake qcow2")
    (env_dir / "env.json").write_text(
        json.dumps(
            {
                "qemu_base_image": str(image_path),
                "qemu_base_format": "qcow2",
                "qemu_require_base_image": True,
                "qemu_x11_display": ":0",
                "ssh": {"user": "user", "password": "password"},
            }
        ),
        encoding="utf-8",
    )
    runner = DummyQemuRunner(tmp_path)
    env = DummyQemuEnv(runner)

    GymAnythingBackend()._apply_qemu_env_overrides(env, str(env_dir))

    assert runner.base_qcow2 == image_path.resolve()
    assert runner._cs_configured_qemu_base_image == str(image_path.resolve())
    assert runner._cs_effective_qemu_base_image == str(image_path.resolve())
    assert runner._ssh_user == "user"
    assert runner._ssh_password == "password"
    assert runner.env_hash.startswith("defaultenv_")
    assert runner.env_checkpoint == tmp_path / f"checkpoint_{runner.env_hash}.qcow2"
    assert os.environ["GYM_ANYTHING_QEMU_X11_DISPLAY"] == ":0"

    runner._run_ssh_cmd(2222, "sudo mkdir -p /workspace/tasks")
    runner._run_ssh_cmd(2222, "sudo chown ga:ga /workspace/tasks")
    runner._ssh_command("sudo whoami")
    runner._ssh_command("DISPLAY=:1 touch /home/ga/task_pre_task.log")
    host_src = tmp_path / "host.txt"
    host_src.write_text("x", encoding="utf-8")
    runner.copy_from("/tmp/remote.png", str(tmp_path / "remote.png"))
    runner.copy_to(str(host_src), "/tmp/host.txt")

    assert runner.ssh_commands == [
        "printf '%s\\n' password | sudo -S -p '' mkdir -p /workspace/tasks",
        "printf '%s\\n' password | sudo -S -p '' chown user:user /workspace/tasks",
        "printf '%s\\n' password | sudo -S -p '' whoami",
        "DISPLAY=:0 touch /home/user/task_pre_task.log",
    ]
    assert runner.sftp_from == [("/tmp/remote.png", str(tmp_path / "remote.png"))]
    assert runner.sftp_to == [(str(host_src), "/tmp/host.txt")]
    assert runner.original_copies == []


def test_backend_refuses_required_qemu_image_without_path(tmp_path: Path) -> None:
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "env.json").write_text(
        json.dumps({"id": "osworld.full@0.1", "qemu_require_base_image": True}),
        encoding="utf-8",
    )

    backend = GymAnythingBackend()

    with pytest.raises(RuntimeError, match="requires qemu_base_image"):
        backend._configured_qemu_base_image(
            str(env_dir), backend._load_env_config(str(env_dir))
        )


def test_backend_refuses_unenforceable_configured_qemu_image(tmp_path: Path) -> None:
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    image_path = tmp_path / "osworld_ubuntu.qcow2"
    image_path.write_bytes(b"fake qcow2")
    (env_dir / "env.json").write_text(
        json.dumps(
            {
                "qemu_base_image": str(image_path),
                "qemu_base_format": "qcow2",
                "qemu_require_base_image": True,
            }
        ),
        encoding="utf-8",
    )
    env = DummyQemuEnv(DummyQemuRunnerWithoutBase())

    with pytest.raises(RuntimeError, match="does not expose base_qcow2"):
        GymAnythingBackend()._apply_qemu_env_overrides(env, str(env_dir))
