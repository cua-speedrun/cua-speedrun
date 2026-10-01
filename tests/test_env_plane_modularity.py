"""Env planes forward task configuration; backends prepare environments."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cua_speedrun.envs import get_backend
from cua_speedrun.envs.base import PreparedEnv


def _code_names(module) -> set[str]:
    """Every identifier the module's CODE references, docstrings and
    comments excluded, so documentation may explain what the backend does
    without tripping the check."""
    import io
    import tokenize

    source = Path(module.__file__).read_text()
    return {
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.NAME
    }


def test_registry_covers_every_env_plane_backend() -> None:
    qemu = get_backend("gym-anything-qemu-native")
    assert qemu.require_runner == "QemuNativeRunner"
    avd = get_backend("gym-anything-avd-native")
    assert avd.require_runner == "AVDNativeRunner"
    native = get_backend("modal-native")
    assert type(native).__name__ == "ModalNativeBackend"
    assert native.name == "modal-native"


def test_prepared_env_checker_defaults_to_the_adapter() -> None:
    import dataclasses

    field_defaults = {
        field.name: field.default for field in dataclasses.fields(PreparedEnv)
    }
    assert field_defaults["checker"] is None


def test_qemu_env_plane_contains_no_preparation_logic() -> None:
    from cua_speedrun.remote import env_plane

    names = _code_names(env_plane)
    for marker in (
        "from_config",            # environment construction
        "_apply_qemu_env_overrides",
        "scrub_guest_privileged_material",
        "LocalRunner",            # runner guards live in the backend
        "_accel_type",
        "reset",
        "recording",
        "GymAnythingAdapter",
        "GymAnythingBackend",
    ):
        assert marker not in names, (
            f"env_plane grew preparation logic ({marker!r}); it belongs in "
            "GymAnythingBackend.prepare so local and remote cannot drift"
        )


def test_modal_native_env_plane_contains_no_preparation_logic() -> None:
    from cua_speedrun.remote import modal_native_env_plane

    names = _code_names(modal_native_env_plane)
    for marker in (
        "boot_desktop",
        "run_pre_task_setup",
        "scrub_privileged",
        "warm_osworld_evaluators",
        "load_osworld_checker",
        "ModalNativeAdapter",
        "Gateway",
    ):
        assert marker not in names, (
            f"modal_native_env_plane grew preparation logic ({marker!r}); it "
            "belongs in ModalNativeBackend.prepare"
        )


def test_env_spec_payload_forwards_the_task_env_block_verbatim() -> None:
    from cua_speedrun.remote.modal_env import _env_spec_payload

    block = {
        "kind": "gym-anything",
        "env_dir": "${BENCHMARK_DIR}/environment",
        "task_id": "chrome__abc",
        "use_cache": False,
        "action_settle_ms": 450,
    }
    payload = json.loads(_env_spec_payload(block))
    assert payload["env_dir"] == "/envs/env"
    assert {k: v for k, v in payload.items() if k != "env_dir"} == {
        k: v for k, v in block.items() if k != "env_dir"
    }
    # No launcher-side defaults: keys the task did not set stay absent, so
    # the backend's own defaults (shared with the local path) apply.
    assert "cache_level" not in payload


def test_modal_env_sandbox_honors_guest_resources_and_task_overrides(
    tmp_path: Path,
) -> None:
    from cua_speedrun.remote.modal_env import _environment_resources

    (tmp_path / "env.json").write_text(
        json.dumps({"resources": {"cpu": 2, "mem_gb": 4}})
    )
    assert _environment_resources(tmp_path, {}) == (5, 12 * 1024)
    assert _environment_resources(
        tmp_path,
        {"config_overrides": {"resources": {"cpu": 8, "mem_gb": 32}}},
    ) == (9, 35 * 1024)


def test_environment_host_system_packages_are_strictly_opt_in(
    tmp_path: Path,
) -> None:
    from cua_speedrun.remote.modal_env import _host_runtime

    assert _host_runtime(tmp_path) == {}

    (tmp_path / "host-runtime.json").write_text(
        json.dumps(
            {
                "python_version": "3.12",
                "apt_packages": ["ffmpeg", "imagemagick"],
            }
        )
    )
    assert _host_runtime(tmp_path)["apt_packages"] == [
        "ffmpeg",
        "imagemagick",
    ]


def test_environment_host_system_packages_reject_invalid_names(
    tmp_path: Path,
) -> None:
    from cua_speedrun.remote.modal_env import _host_runtime

    (tmp_path / "host-runtime.json").write_text(
        json.dumps({"apt_packages": ["ffmpeg", "bad package"]})
    )
    with pytest.raises(ValueError, match="apt_packages"):
        _host_runtime(tmp_path)


def test_environment_host_packages_do_not_change_undeclared_image_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from gym_anything.runtime.runners.modal_runner import _SANDBOX_APT
    from cua_speedrun.remote import modal_env, osworld_modal_base

    apt_calls = []

    class Image:
        def apt_install(self, *packages, **_kwargs):
            apt_calls.append(packages)
            return self

        def pip_install(self, *_packages, **_kwargs):
            return self

        def add_local_dir(self, *_args, **_kwargs):
            return self

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            Image=SimpleNamespace(
                debian_slim=lambda **_kwargs: Image(),
            )
        ),
    )

    modal_env._build_image(tmp_path, {})

    assert apt_calls == [tuple(_SANDBOX_APT) + osworld_modal_base.ENV_PLANE_APT]


def test_modal_native_backend_reads_launcher_knobs(monkeypatch) -> None:
    from cua_speedrun.envs.modal_native import ModalNativeBackend

    monkeypatch.setenv("CS_DESKTOP_USER", "osworlduser")
    monkeypatch.setenv("CS_DESKTOP_SETTLE_SEC", "7")
    backend = ModalNativeBackend()
    assert backend.desktop_user == "osworlduser"
    assert backend.desktop_settle_sec == 7
    # Explicit arguments win, so tests and callers can bypass the process
    # environment entirely.
    explicit = ModalNativeBackend(
        desktop_user="u", desktop_settle_sec=0
    )
    assert explicit.desktop_user == "u"
    assert explicit.desktop_settle_sec == 0
