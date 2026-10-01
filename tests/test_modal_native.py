from __future__ import annotations

import inspect
import ast
import base64
import json
import shlex
import subprocess

import pytest

from cua_speedrun.envs.base import EnvAdapter, Observation, Verdict
from cua_speedrun.envs import modal_native
from cua_speedrun.envs.modal_native import ModalNativeAdapter
from cua_speedrun.remote import osworld_modal_base as base


def _python_c_script(command: str) -> str:
    argv = shlex.split(command)
    assert argv[:2] == ["python3", "-c"]
    return argv[2]


def _keyboard_payload(command: str) -> dict:
    tree = ast.parse(_python_c_script(command))
    encoded = tree.body[-1].value.args[0].args[0].args[0]
    return json.loads(base64.b64decode(ast.literal_eval(encoded)))


def test_base_provenance_mirrors_the_qcow2_source_pins() -> None:
    # The modal-native image is booted from the same pinned rootfs, so its
    # source pins must equal the qcow2 image contract exactly.
    provenance = base.build_provenance(image_id="im-abc123", tools_installed={"scrot": "1.7-1"})
    assert provenance["source_revision"] == base.OSWORLD_IMAGE_CONTRACT["source_revision"]
    assert provenance["archive_sha256"] == base.OSWORLD_IMAGE_CONTRACT["archive_sha256"]
    assert provenance["source_image_sha256"] == base.OSWORLD_IMAGE_CONTRACT["source_image_sha256"]
    assert provenance["schema_version"] == base.OSWORLD_IMAGE_CONTRACT["provenance_schema_version"]


def test_expected_block_has_the_ten_equality_checked_keys() -> None:
    block = base.expected_provenance_block()
    assert set(block) == set(base.PROVENANCE_EXPECTED_KEYS)
    assert len(base.PROVENANCE_EXPECTED_KEYS) == 10
    # These are exactly the keys the qcow2 preflight equality-checks.
    assert block["recipe"] == base.RECIPE
    assert block["ssh_user"] == "user"
    assert block["nopasswd_sudo"] is True


def test_delta_fingerprint_is_stable_and_content_addressed() -> None:
    # Deterministic across calls.
    assert base.delta_fingerprint() == base.delta_fingerprint()
    # The cache key folds in the fingerprint, so identical inputs reuse a build.
    assert base.delta_fingerprint() in base.cache_key()
    assert base.cache_key().startswith(f"base:{base.RECIPE}:")


def test_delta_fingerprint_tracks_the_recipe_bytes(monkeypatch) -> None:
    original = base.delta_fingerprint()
    monkeypatch.setattr(base, "UBUNTU_SNAPSHOT", "20991231T000000Z")
    assert base.delta_fingerprint() != original


def test_rendered_delta_resolves_every_placeholder() -> None:
    rendered = base.rendered_delta()
    assert "__" not in rendered  # all placeholders filled
    assert base.UBUNTU_SNAPSHOT in rendered
    # The desktop is the image's own GDM autologin session on the dummy Xorg
    # driver; the delta must not mask GDM and must not bolt on a VNC session.
    assert "xserver-xorg-video-dummy" in rendered
    assert "/etc/X11/xorg.conf" in rendered
    assert "gdm3.service" not in rendered
    assert "vnc" not in rendered.lower()
    assert "huggingface-hub" in rendered


def test_guest_service_that_is_ready_is_not_restarted(monkeypatch) -> None:
    monkeypatch.setattr(modal_native, "_guest_command_succeeds", lambda *_: True)

    def unexpected_run(*args, **kwargs):
        raise AssertionError(f"ready service was mutated: {args!r} {kwargs!r}")

    monkeypatch.setattr(modal_native.subprocess, "run", unexpected_run)
    modal_native._ensure_guest_service(
        "123", "database.service", ("healthcheck",), wait_sec=0,
    )


def test_guest_service_recovers_after_reset_and_restart(monkeypatch) -> None:
    ready = iter((False, True))
    monkeypatch.setattr(
        modal_native, "_guest_command_succeeds", lambda *_: next(ready),
    )
    calls = []

    def successful_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(modal_native.subprocess, "run", successful_run)
    modal_native._ensure_guest_service(
        "123", "database.service", ("healthcheck",), wait_sec=30,
    )

    assert calls == [
        modal_native._ns("123")
        + ["systemctl", "reset-failed", "database.service"],
        modal_native._ns("123")
        + ["systemctl", "restart", "database.service"],
    ]


def test_guest_service_that_never_recovers_fails_preparation(monkeypatch) -> None:
    monkeypatch.setattr(modal_native, "_guest_command_succeeds", lambda *_: False)
    monkeypatch.setattr(modal_native, "_service_diagnostics", lambda *_: "failed status")

    def successful_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(modal_native.subprocess, "run", successful_run)
    with pytest.raises(
        RuntimeError,
        match="required guest service database.service did not become ready",
    ):
        modal_native._ensure_guest_service(
            "123", "database.service", ("healthcheck",), wait_sec=0,
        )


def test_validate_base_provenance_accepts_a_matching_record() -> None:
    provenance = base.build_provenance(image_id="im-01ABC", tools_installed={})
    base.validate_base_provenance(provenance, base.expected_provenance_block())


def test_validate_base_provenance_rejects_a_wrong_contract() -> None:
    provenance = base.build_provenance(image_id="im-01ABC", tools_installed={})
    provenance["source_revision"] = "deadbeef"
    with pytest.raises(ValueError, match="wrong contract"):
        base.validate_base_provenance(provenance, base.expected_provenance_block())


def test_validate_base_provenance_requires_a_modal_image_id() -> None:
    provenance = base.build_provenance(image_id="", tools_installed={})
    with pytest.raises(ValueError, match="modal_snapshot_image_id"):
        base.validate_base_provenance(provenance, base.expected_provenance_block())


def test_adapter_implements_the_env_adapter_surface() -> None:
    adapter = ModalNativeAdapter(desktop_user="user")
    assert isinstance(adapter, EnvAdapter)
    # The Gateway-facing and verifier-facing methods exist and are callable.
    for name in ("observe", "step", "finalize", "close", "exec_read", "copy_from_env"):
        assert callable(getattr(adapter, name))
    # finalize returns a Verdict even with no checker (safe default).
    verdict = adapter.finalize()
    assert isinstance(verdict, Verdict)
    assert verdict.passed is False


def test_adapter_maps_actions_without_touching_the_desktop(monkeypatch) -> None:
    # Pointer actions use xdotool; keyboard actions use the shared PyAutoGUI
    # script. Capture both instead of running a real desktop.
    calls: list[str] = []
    adapter = ModalNativeAdapter(desktop_user="user", settle_sec=0.0)
    monkeypatch.setattr(adapter, "_run_user", lambda cmd, timeout=60: calls.append(cmd))
    adapter.step([
        {"mouse": {"left_click": [10, 20]}},
        {"keyboard": {"text": "hello"}},
        {"keyboard": {"keys": ["ctrl", "s"]}},
    ])
    assert calls[0] == "xdotool mousemove 10 20 click 1"
    assert _keyboard_payload(calls[1]) == {"text": "hello"}
    assert _keyboard_payload(calls[2]) == {"keys": ["ctrl", "s"]}


def test_launch_env_carries_the_env_plane_contract() -> None:
    import json as _json

    from cua_speedrun.remote.modal_native_env import build_launch_env

    env = build_launch_env(
        env_spec={"env_dir": "/local/env", "task_id": "autolock",
                  "action_settle_ms": 400},
        seed=0, timeout_sec=300.0, grace_sec=1.5,
        gateway_port=8390, run_token="r", control_token="c",
        task_label="osworld_autolock", desktop_user="user",
    )
    spec = _json.loads(env["CS_ENV_SPEC"])
    # The env block travels verbatim; only env_dir is rewritten to the
    # in-sandbox mount. The backend applies its own defaults for the rest.
    assert spec["env_dir"] == "/envs/env"
    assert spec["task_id"] == "autolock"
    assert spec["action_settle_ms"] == 400
    assert env["CS_ENV_BACKEND"] == "modal-native"
    assert env["PYTHONPATH"] == "/opt/cs"
    assert env["CS_RUN_TOKEN"] == "r" and env["CS_CONTROL_TOKEN"] == "c"
    assert env["CS_DESKTOP_USER"] == "user"


def test_resolve_base_image_validates_provenance_and_id() -> None:
    from cua_speedrun.remote.modal_native_env import resolve_base_image_id

    class _Cache:
        def __init__(self, record):
            self._record = record

        def get(self, _key):
            return self._record

    good = base.build_provenance(image_id="im-01VALID", tools_installed={})
    assert resolve_base_image_id(_Cache(good)) == "im-01VALID"

    with pytest.raises(RuntimeError, match="no OSWorld modal-native base image"):
        resolve_base_image_id(_Cache(None))

    wrong = base.build_provenance(image_id="im-01VALID", tools_installed={})
    wrong["recipe"] = "someone-elses-recipe@9"
    with pytest.raises(ValueError, match="wrong contract"):
        resolve_base_image_id(_Cache(wrong))


def test_observation_and_verdict_shapes_are_the_backend_contract() -> None:
    # The Gateway validates PNG bytes and reads Verdict fields; keep the shapes.
    obs = Observation(png=b"\x89PNG\r\n\x1a\n", meta={"resolution": [1920, 1080]})
    assert obs.png.startswith(b"\x89PNG")
    verdict = Verdict(passed=True, score=100.0, detail="ok")
    assert (verdict.passed, verdict.score, verdict.detail) == (True, 100.0, "ok")


def test_modal_native_topology_is_registered() -> None:
    from cua_speedrun.execution_placements import get_execution_topology

    topology = get_execution_topology("modal", "modal-native")
    assert topology.key == "modal-native"
    assert topology.backend == "modal-native"
    assert topology.environment_backend == "modal-native"
    assert topology.requires_user_credentials is True
    assert topology.compute.provider == "modal"
    assert topology.environment.provider == "modal"
    # Same runtime capabilities as modal-remote.
    assert set(topology.eval_algorithms) == {
        "per-task-vllm@1",
        "shared-agent-vllm@2",
        "shared-agent-no-preload@1",
    }


def test_modal_native_backend_is_a_distinct_season(tmp_path) -> None:
    from cua_speedrun.execution_placements import get_execution_topology
    from cua_speedrun.runplan import build_run_plan

    benchmark = tmp_path / "benchmark"
    task = benchmark / "tasks" / "one"
    env = tmp_path / "environment"
    task.mkdir(parents=True)
    env.mkdir()
    (benchmark / "manifest.yaml").write_text(
        'name: tiny\nversion: "1"\ntasks:\n  - tasks/one\n'
    )
    (task / "task.yaml").write_text(
        "task_id: one\n"
        "description: tiny task\n"
        "env:\n"
        "  kind: gym-anything\n"
        f"  env_dir: {env}\n"
        "  task_id: one\n"
    )
    (env / "env.json").write_text("{}")

    plans = {}
    for environment_key in ("modal", "modal-native"):
        topology = get_execution_topology("modal", environment_key)
        plans[environment_key] = build_run_plan(
            track_name="cpu-scripted",
            benchmark_dir=benchmark,
            eval_algorithm="per-task-vllm@1",
            execution_topology=topology.contract_dict(),
            backend=topology.backend,
        )
    assert plans["modal-native"].backend == "modal-native"
    resolved = plans["modal-native"].resolved_execution_topology()
    assert resolved["environment_backend"] == "modal-native"
    # A different backend is a different contract, hence a different season.
    assert (
        plans["modal-native"].contract_hash != plans["modal"].contract_hash
    )
    assert plans["modal-native"].season() != plans["modal"].season()


def test_remote_executor_supports_both_modal_backends() -> None:
    from cua_speedrun.remote.run import BACKEND, SUPPORTED_BACKENDS

    assert BACKEND == "modal-remote"
    assert SUPPORTED_BACKENDS == ("modal-remote", "modal-native")


def test_environment_sandbox_dispatch_routes_by_backend(monkeypatch, tmp_path) -> None:
    from cua_speedrun.remote import run as run_module

    calls: dict[str, dict] = {}

    def _native(**kwargs):
        calls["native"] = kwargs
        return "native-sandbox"

    def _qemu(**kwargs):
        calls["qemu"] = kwargs
        return "qemu-sandbox"

    monkeypatch.setattr(run_module, "create_modal_native_env_sandbox", _native)
    monkeypatch.setattr(run_module, "create_env_sandbox", _qemu)
    common = dict(
        env_local_dir=tmp_path,
        env_spec={"env_dir": "e", "task_id": "t", "use_cache": True},
        seed=3, timeout_sec=60.0, grace_sec=1.5,
        generator_local=None, task_label="t",
        region=None, sandbox_timeout_sec=1234,
    )
    assert run_module._create_environment_sandbox("modal-native", **common) == "native-sandbox"
    # The env block travels verbatim to both creators; QEMU-only knobs live
    # inside it rather than as separate launcher parameters.
    assert calls["native"]["env_spec"]["task_id"] == "t"
    assert "use_cache" not in calls["native"]
    assert "cache_level" not in calls["native"]
    assert calls["native"]["seed"] == 3
    assert calls["native"]["sandbox_timeout_sec"] == 1234

    assert run_module._create_environment_sandbox("gym-anything-modal", **common) == "qemu-sandbox"
    assert calls["qemu"]["env_spec"]["use_cache"] is True

    with pytest.raises(ValueError, match="seeded generators"):
        run_module._create_environment_sandbox(
            "modal-native", **{**common, "generator_local": tmp_path / "g.py"}
        )


def test_shared_environment_timeout_uses_one_task_budget(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    from cua_speedrun.remote import run as run_module

    captured: dict[str, object] = {}

    def _create_environment(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            sandbox=SimpleNamespace(object_id="env-1"),
            base_url="https://env.invalid",
        )

    class _Events:
        def emit(self, *args, **kwargs) -> None:
            return None

    monkeypatch.setattr(
        run_module, "_create_environment_sandbox", _create_environment
    )
    monkeypatch.setattr(run_module, "tail_sandbox_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_module, "wait_env_healthy", lambda environment: None)

    events = _Events()
    runtime = run_module._RemoteEvaluationRuntime(
        agent_image=None,
        run_dir=tmp_path,
        run_id="run",
        region=None,
        events=events,
        execution_scale=SimpleNamespace(),
        network_policy="host",
        api_domains=(),
        eval_algorithm="shared-agent-vllm@2",
    )
    replica = run_module._RemoteComputeReplica(
        index=1,
        events=run_module._ReplicaEvents(events, 1),
        total_timeout=135_600,
    )
    task = SimpleNamespace(
        task_id="osworld_test",
        task_dir=tmp_path,
        env={"env_dir": str(tmp_path)},
        generator=None,
        timeout_sec=3600.0,
        grace_sec=1.5,
    )

    runtime.prepare_environment(replica, (task, 7), 1)

    assert captured["sandbox_timeout_sec"] == 5701


def test_worker_topology_key_resolves_modal_native() -> None:
    from types import SimpleNamespace

    from cua_speedrun.service.worker import _topology_key

    frozen = SimpleNamespace(topology_key="modal-native", execution_plan=None)
    assert _topology_key(frozen) == "modal-native"
    derived = SimpleNamespace(
        topology_key=None,
        execution_plan={"execution": {"topology": {"key": "modal-native"}}},
    )
    assert _topology_key(derived) == "modal-native"


def test_env_plane_pip_covers_the_pyproject_osworld_extra() -> None:
    import re
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    extra = data["project"]["optional-dependencies"]["osworld"]

    def norm(requirement: str) -> str:
        name = re.split(r"[~<>=!\[;]", requirement, 1)[0]
        return name.strip().lower().replace("_", "-")

    missing = {norm(r) for r in extra} - {norm(r) for r in base.ENV_PLANE_PIP}
    # The env-plane runs the OSWorld verifiers host-side, so the base snapshot
    # needs everything a local worker gets from the osworld extra.
    assert not missing, f"ENV_PLANE_PIP is missing osworld extra packages: {missing}"


def test_env_plane_installs_osworld_system_executables() -> None:
    # compare_image_text shells out to `file`; Python dependencies alone are
    # insufficient for the canonical OSWorld evaluator stack.
    assert "file" in base.ENV_PLANE_APT


def test_env_plane_runtime_is_part_of_the_image_identity(monkeypatch) -> None:
    # Changing the baked runtime must change the fingerprint, hence the cache
    # key, hence force a rebuild: it cannot be layered on after the snapshot.
    original = base.delta_fingerprint()
    monkeypatch.setattr(base, "ENV_PLANE_PIP", base.ENV_PLANE_PIP + ("extra-package",))
    assert base.delta_fingerprint() != original


def test_env_plane_system_packages_are_part_of_the_image_identity(monkeypatch) -> None:
    original = base.delta_fingerprint()
    monkeypatch.setattr(base, "ENV_PLANE_APT", base.ENV_PLANE_APT + ("extra-system-package",))
    assert base.delta_fingerprint() != original


def test_warm_osworld_evaluators_is_a_noop_without_shared(tmp_path) -> None:
    from cua_speedrun.envs.modal_native import warm_osworld_evaluators

    (tmp_path / "tasks").mkdir(parents=True)
    warm_osworld_evaluators(str(tmp_path))  # nothing to warm; must not raise


def test_warm_osworld_evaluators_imports_the_evaluator_package(tmp_path) -> None:
    import sys

    from cua_speedrun.envs.modal_native import warm_osworld_evaluators

    shared = tmp_path / "tasks" / "_shared"
    shared.mkdir(parents=True)
    stub_root = tmp_path / "stub"
    pkg = stub_root / "desktop_env" / "evaluators"
    pkg.mkdir(parents=True)
    (stub_root / "desktop_env" / "__init__.py").write_text("")
    (stub_root / "desktop_env" / "desktop_env.py").write_text(
        "class DesktopEnv:\n"
        "    pass\n"
    )
    (pkg / "__init__.py").write_text("")
    (pkg / "getters.py").write_text("")
    (pkg / "metrics.py").write_text("")
    (shared / "osworld_verifier.py").write_text(
        "import sys\n"
        "def ensure_osworld_evaluators():\n"
        f"    sys.path.insert(0, {str(stub_root)!r})\n"
    )
    for name in [n for n in sys.modules if n.startswith("desktop_env")]:
        del sys.modules[name]
    try:
        warm_osworld_evaluators(str(tmp_path))
    finally:
        sys.path = [p for p in sys.path if p != str(stub_root)]
        for name in [n for n in sys.modules if n.startswith("desktop_env")]:
            del sys.modules[name]


def test_adapter_normalizes_agent_key_names_to_keysyms(monkeypatch) -> None:
    # Lowercase agent vocabulary must be normalized exactly as the
    # gym-anything QEMU runner does.
    calls: list[str] = []
    adapter = ModalNativeAdapter(desktop_user="user", settle_sec=0.0)
    monkeypatch.setattr(adapter, "_run_user", lambda cmd, timeout=60: calls.append(cmd))
    adapter.step([
        {"keyboard": {"keys": ["enter"]}},
        {"keyboard": {"keys": ["esc"]}},
        {"keyboard": {"keys": ["ctrl", "shift", "pagedown"]}},
        {"keyboard": {"keys": "ctrl+enter"}},
        {"keyboard": {"keys": ["XF86AudioPlay"]}},
    ])
    from cua_speedrun.envs._pyautogui_keyboard import key_names

    names = [key_names(_keyboard_payload(call)["keys"]) for call in calls]
    assert names == [["Return"], ["Escape"], ["Control_L", "Shift_L", "Next"],
                     ["Control_L", "Return"], ["XF86AudioPlay"]]


def test_adapter_supports_right_button_drag(monkeypatch) -> None:
    calls: list[str] = []
    adapter = ModalNativeAdapter(desktop_user="user", settle_sec=0.0)
    monkeypatch.setattr(adapter, "_run_user", lambda cmd, timeout=60: calls.append(cmd))
    adapter.step([
        {"mouse": {"right_click_drag": [[10, 20], [30, 40]]}},
    ])
    assert calls == ["xdotool mousemove 10 20 mousedown 3 mousemove 30 40 mouseup 3"]


def test_keyboard_script_preserves_multiline_unicode_and_extended_keys() -> None:
    from cua_speedrun.envs.modal_native import _keyboard_script

    keyboard = {
        "text": "naïve\n\t東京 👨‍👩‍👧",
        "keys": ["insert", "kp_enter", "kp_add", "menu", "caps_lock"],
    }
    script = _keyboard_script(keyboard)
    assert _keyboard_payload(f"python3 -c {shlex.quote(script)}") == keyboard
    assert "self.pg.write(" in script and "self.pg.hotkey(" in script


def test_adapter_provides_copy_to_env(monkeypatch) -> None:
    # Keep the generic checker transfer surface symmetric.
    adapter = ModalNativeAdapter(desktop_user="user")
    assert callable(adapter.copy_to_env)
    captured = {}
    def fake_run_root(command, timeout=60, stdin=None):
        captured["command"] = command
        captured["stdin"] = stdin
        import subprocess
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
    monkeypatch.setattr(adapter, "_run_root", fake_run_root)
    import base64, tempfile, os
    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(b"gold-bytes")
        host = fh.name
    try:
        adapter.copy_to_env(host, "/home/user/gold.bin")
    finally:
        os.unlink(host)
    assert "base64 -d > " in captured["command"]
    assert "/home/user/gold.bin" in captured["command"]
    assert base64.b64decode(captured["stdin"]) == b"gold-bytes"


def test_env_info_carries_both_copy_directions() -> None:
    # Guard against dropping either transfer direction from the verifier env.
    import inspect
    from cua_speedrun.envs import modal_native as plane
    src = inspect.getsource(plane.load_osworld_checker)
    assert '"copy_from_env": adapter.copy_from_env' in src
    assert '"copy_to_env": adapter.copy_to_env' in src
    assert "traj=adapter.trajectory" in src


def test_osworld_checker_receives_modal_native_action_trajectory(
    tmp_path, monkeypatch
) -> None:
    from cua_speedrun.envs.modal_native import (
        ModalNativeAdapter,
        load_osworld_checker,
    )

    task_dir = tmp_path / "tasks" / "infeasible-task"
    task_dir.mkdir(parents=True)
    (task_dir / "verifier.py").write_text(
        "def check(traj, env_info, task_info):\n"
        "    actions = [action for step in traj['steps'] "
        "for action in step['action']]\n"
        "    passed = actions[-1] == {'action_type': 'FAIL'}\n"
        "    return {'passed': passed, 'score': 100 if passed else 0}\n"
    )

    adapter = ModalNativeAdapter(settle_sec=0)
    monkeypatch.setattr(adapter, "_inject", lambda action: None)
    adapter.step([{"action_type": "FAIL"}])

    verdict = load_osworld_checker(
        adapter, str(tmp_path), "infeasible-task"
    )()
    assert verdict.passed is True
    assert verdict.score == 100.0


def test_adapter_single_coordinate_drag(monkeypatch) -> None:
    # Qwen computer_use emits left_click_drag with ONE coordinate (drag from
    # current cursor). The two-point form must still work; the one-point form
    # must drag from the current position (no initial mousemove).
    calls: list[str] = []
    adapter = ModalNativeAdapter(desktop_user="user", settle_sec=0.0)
    monkeypatch.setattr(adapter, "_run_user", lambda cmd, timeout=60: calls.append(cmd))
    adapter.step([
        {"mouse": {"left_click_drag": [[10, 20], [30, 40]]}},
        {"mouse": {"left_click_drag": [[300, 400]]}},
    ])
    joined = "\n".join(calls)
    assert "mousemove 10 20 mousedown 1 mousemove 30 40 mouseup 1" in joined
    assert "mousedown 1 mousemove 300 400 mouseup 1" in joined


def test_template_emits_single_coordinate_drag() -> None:
    # The qwen35 parser must not drop a drag that lacks coordinate2.
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "agents/qwen35/agent.py"
    spec = importlib.util.spec_from_file_location("qwen35_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    xml = (
        "Action: drag it\n<tool_call><function=computer_use>"
        "<parameter=action>left_click_drag</parameter>"
        "<parameter=coordinate>[500, 500]</parameter>"
        "</function></tool_call>"
    )
    parsed = mod.parse_response(xml, original_width=1000, original_height=1000)
    drags = [a for a in parsed["actions"] if "left_click_drag" in a.get("mouse", {})]
    assert len(drags) == 1, "single-coordinate drag was dropped"
    assert len(drags[0]["mouse"]["left_click_drag"]) == 1


def test_adapter_holds_modifiers_mouse_buttons_and_triple_clicks(monkeypatch) -> None:
    # Both singular and plural key holds are part of the template action
    # vocabulary. Mouse button holds preserve multi-point drag paths.
    calls: list[str] = []
    adapter = ModalNativeAdapter(desktop_user="user", settle_sec=0.0)
    monkeypatch.setattr(adapter, "_run_user", lambda cmd, timeout=60: calls.append(cmd))
    adapter.step([
        {"keyboard": {"key_down": "ctrl"}},
        {"mouse": {"buttons": {"left_down": True}}},
        {"mouse": {"move": [5, 6]}},
        {"mouse": {"buttons": {"left_up": True}}},
        {"keyboard": {"key_up": "ctrl"}},
        {"keyboard": {"keys_down": ["shift"]}},
        {"mouse": {"left_click": [10, 20]}},
        {"keyboard": {"keys_up": ["shift"]}},
        {"mouse": {"triple_click": [30, 40]}},
    ])
    assert _keyboard_payload(calls[0]) == {"key_down": "ctrl"}
    assert calls[1:4] == [
        "xdotool mousedown 1",
        "xdotool mousemove 5 6",
        "xdotool mouseup 1",
    ]
    assert _keyboard_payload(calls[4]) == {"key_up": "ctrl"}
    assert _keyboard_payload(calls[5]) == {"keys_down": ["shift"]}
    assert calls[6] == "xdotool mousemove 10 20 click 1"
    assert _keyboard_payload(calls[7]) == {"keys_up": ["shift"]}
    assert calls[8] == "xdotool mousemove 30 40 click --repeat 3 1"


def test_observation_includes_the_mouse_pointer(monkeypatch) -> None:
    # OSWorld's server composites the cursor into every Linux screenshot
    # (capture_screen_with_cursor), and the QEMU path gets it from the VNC
    # framebuffer. scrot omits the pointer unless -p is passed, which left
    # this backend's agent unable to see where the mouse was.
    calls: list[str] = []
    adapter = ModalNativeAdapter(desktop_user="user", settle_sec=0.0)
    monkeypatch.setattr(adapter, "_run_user", lambda cmd, timeout=60: calls.append(cmd))
    monkeypatch.setattr(
        "builtins.open",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no capture in test")),
    )
    try:
        adapter.observe()
    except FileNotFoundError:
        pass
    assert calls and "scrot" in calls[0]
    assert " -p " in calls[0], f"screenshot must include the pointer: {calls[0]}"
