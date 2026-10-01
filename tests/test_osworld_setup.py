from __future__ import annotations

import getpass
import importlib.util
import os
import pwd
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from cua_speedrun.service.templates_catalog import list_templates


ROOT = Path(__file__).resolve().parents[1]


def _load_shared_module(name: str):
    path = ROOT / "scripts" / "osworld_shared" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    if name == "osworld_setup":
        account = pwd.getpwuid(os.getuid())
        env = {
            "OSWORLD_DESKTOP_USER": account.pw_name,
            "OSWORLD_DESKTOP_HOME": account.pw_dir,
            "OSWORLD_X11_DISPLAY": ":99",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            spec.loader.exec_module(module)
    else:
        spec.loader.exec_module(module)
    return module


def test_setup_discovers_the_real_desktop_account_and_maps_legacy_homes() -> None:
    setup = _load_shared_module("osworld_setup")
    configured_user = getpass.getuser()
    with mock.patch.dict(
        os.environ,
        {
            "OSWORLD_DESKTOP_USER": configured_user,
            "OSWORLD_DESKTOP_HOME": "/srv/test-desktop",
            "OSWORLD_X11_DISPLAY": ":0",
        },
        clear=True,
    ):
        user, uid, home, display, dbus = setup._discover_desktop_context()

    assert (user, uid, home, display) == (
        configured_user,
        os.getuid(),
        "/srv/test-desktop",
        ":0",
    )
    assert dbus == f"unix:path=/run/user/{os.getuid()}/bus"

    setup.HOME = "/srv/desktop"
    assert setup._map_path("/home/user/Desktop/a") == "/srv/desktop/Desktop/a"
    assert setup._map_path("/home/ga/Desktop/a") == "/srv/desktop/Desktop/a"


def test_setup_command_failures_are_fatal() -> None:
    setup = _load_shared_module("osworld_setup")
    failed = SimpleNamespace(returncode=23, stdout="download failed\n")
    with mock.patch.object(setup.subprocess, "run", return_value=failed):
        with pytest.raises(setup.SetupError, match="exited 23"):
            setup._run_root("false")


def test_setup_waits_for_slow_application_windows_before_activation() -> None:
    setup = _load_shared_module("osworld_setup")
    with mock.patch.object(setup, "_run_as_ga") as run:
        setup._run_item({
            "type": "activate_window",
            "parameters": {"window_name": "Visual Studio Code"},
        })

    command = run.call_args.args[0]
    assert "seq 1 40" in command
    assert "windowactivate --sync" in command
    assert run.call_args.kwargs["timeout"] == 25


def test_qwen_starter_is_available_for_osworld_50() -> None:
    qwen = next(item for item in list_templates() if item["name"] == "qwen3vl")
    assert "osworld-50" in qwen["compatible_benchmarks"]
    init_source = (ROOT / "agents/qwen3vl/init.py").read_text()
    # The cu129 wheel path is selected by an exact runtime key, so the key
    # the installer branches on must stay pinned here. Only the 3.10 local
    # runtime takes that path; every other runtime falls back to plain pip.
    assert 'LOCAL_CU129_KEY = "linux-x86_64-py3.10"' in init_source
    assert "if runtime_key != LOCAL_CU129_KEY:" in init_source


def test_download_creates_desktop_owned_directories_for_relative_paths(tmp_path):
    setup = _load_shared_module("osworld_setup")
    source = tmp_path / "source.txt"
    source.write_text("Task input\n")
    desktop_home = tmp_path / "desktop"
    desktop_home.mkdir()
    setup.HOME = str(desktop_home)
    setup._download([{"url": source.as_uri(), "path": "Downloads/nested/input.txt"}])
    downloaded = desktop_home / "Downloads/nested/input.txt"
    assert downloaded.read_bytes() == source.read_bytes()
    for path in (downloaded, downloaded.parent, downloaded.parent.parent):
        assert path.stat().st_uid == os.getuid()
