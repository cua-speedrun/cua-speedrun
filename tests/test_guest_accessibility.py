"""The harness must preserve OSWorld's image-owned AT-SPI lifecycle."""

from __future__ import annotations

import inspect
from pathlib import Path

from cua_speedrun.envs import gym_anything, modal_native


def test_gym_backend_does_not_mutate_accessibility_after_reset() -> None:
    source = inspect.getsource(gym_anything)
    prepare = inspect.getsource(gym_anything.GymAnythingBackend.prepare)

    assert "enable_guest_accessibility" not in source
    assert "audit_osworld_accessibility_boot" not in source
    assert "toolkit-accessibility" not in source
    assert "cs_a11y_keeper" not in source
    assert "pyatspi.Registry.getDesktop" not in source
    assert "env.reset(" in prepare
    assert "scrub_guest_privileged_material(env)" in prepare


def test_modal_native_boot_does_not_mutate_accessibility() -> None:
    source = inspect.getsource(modal_native.boot_desktop)

    assert "toolkit-accessibility" not in source
    assert "cs_a11y_keeper" not in source
    assert "pyatspi" not in source


def test_modal_native_builder_gates_on_the_service_shipped_by_the_image() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "build_osworld_modal_base.py").read_text()

    assert "systemctl is-active osworld.service" in source
    assert "systemctl is-active osworld_server.service" not in source
    assert "http://127.0.0.1:5000/platform" in source
