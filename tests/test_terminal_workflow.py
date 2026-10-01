"""Exercise the real terminal widgets, CLI, and local setup; no cloud runs."""

import argparse
import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import pytest
from textual.widgets import Checkbox, Input, OptionList, RichLog, Select, Static

from cua_speedrun.commands import register_operator_commands
from cua_speedrun.commands.paths import InstallationPaths
from cua_speedrun.commands.preparation_tui import Preparation
from cua_speedrun.commands.wizard import Configuration, Credentials, Wizard, benchmark_argv, catalog_entries


ROOT = Path(__file__).resolve().parents[1]


def args(*values):
    parser = argparse.ArgumentParser()
    register_operator_commands(parser.add_subparsers(dest="command"))
    return parser.parse_args(values)


def cli_environment():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CS_", "CUA_SPEEDRUN_", "MODAL_", "GYM_ANYTHING_", "OSWORLD_"))}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "third_party/gym-anything/src")])
    return env


def run_cli(*values, cwd):
    return subprocess.run([sys.executable, "-m", "cua_speedrun.cli", *values],
                          cwd=cwd, env=cli_environment(), capture_output=True, text=True, timeout=30)


@contextmanager
def terminal_cli(*values, cwd):
    master, slave = os.openpty()
    env = {**cli_environment(), "TERM": "xterm-256color", "COLUMNS": "100", "LINES": "40"}
    process = subprocess.Popen(
        [sys.executable, "-m", "cua_speedrun.cli", *values], cwd=cwd, env=env,
        stdin=slave, stdout=slave, stderr=slave,
    )
    os.close(slave)
    try:
        yield process, master
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        os.close(master)


def terminal_output_until(process, master, marker):
    output = b""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if select.select([master], [], [], .1)[0]:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            output += chunk
            if marker in output:
                break
        elif process.poll() is not None:
            break
    return output.decode("utf-8", errors="replace")


@pytest.mark.parametrize("host", ["local", "modal"])
@pytest.mark.parametrize("flags", [(), ("--no-input",), ("--json",)])
def test_complete_command_skips_wizard_in_real_terminal(tmp_path, host, flags):
    # Invalid input stops before any cloud resource or desktop is requested.
    with terminal_cli(
        "benchmark", "--agent", "missing-agent", "--dataset", "osworld-50",
        "--host", host, "--home", str(tmp_path / "install"), *flags, cwd=tmp_path,
    ) as (process, master):
        output = terminal_output_until(process, master, b"error")
        assert process.wait(timeout=5) == 1
        assert "unknown agent" in output or "Modal credentials are missing" in output
        assert "Choose an agent template" not in output


@pytest.mark.parametrize("options", [(), ("--agent", "claude_code"), ("--dataset", "osworld-50")])
def test_missing_options_open_wizard_in_real_terminal(tmp_path, options):
    with terminal_cli("benchmark", "--home", str(tmp_path), *options, cwd=tmp_path) as (process, master):
        output = terminal_output_until(process, master, b"Choose an agent template")
        assert "Choose an agent template" in output
        assert process.poll() is None


def test_catalog_needs_no_install_or_download(tmp_path):
    agents, datasets = catalog_entries(InstallationPaths(tmp_path / "new-install"))
    assert next(a for a in agents if a["name"] == "qwen3vl")["gpu"] == "L40S"
    assert next(d for d in datasets if d["name"] == "osworld-50")["tasks"] == 50
    assert len(datasets) == 7
    assert not (tmp_path / "new-install").exists()


def test_wizard_keeps_cli_dataset_aliases(tmp_path):
    app = Wizard(args("benchmark", "--agent", "qwen3vl", "--dataset", "osworld50@0.1", "--home", str(tmp_path)))
    assert app.args.dataset == "osworld-50"


def test_keyboard_navigation_defaults_and_editing(tmp_path):
    async def exercise():
        app = Wizard(args("benchmark", "--agent", "qwen3vl", "--dataset", "osworld-50", "--home", str(tmp_path)))
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.press("enter", "enter")
            await pilot.pause()
            assert isinstance(app.screen, Configuration)
            assert app.screen.read_config().gpu == "L40S"
            assert app.screen.read_config().environment == "modal-native"
            app.screen.query_one("#host", Select).value = "modal"
            assert app.screen.read_config().host == "modal"
            app.screen.query_one("#parallel", Input).value = "0"
            await pilot.click("#next")
            assert app.return_value is None
            assert "at least 1" in str(app.screen.query_one(".error", Static).render())
            app.screen.query_one("#parallel", Input).value = "3"
            app.screen.query_one("#gpu", Input).value = "CPU"
            app.screen.query_one("#preload", Checkbox).value = False
            app.screen.query_one("#name", Input).value = "My evaluation"
            await pilot.pause(.3)
            await pilot.click("#next")
        selected = app.return_value
        assert selected.parallel_evaluations == 3
        assert selected.no_gpu and selected.gpu is None
        assert selected.no_preload
        parsed = args(*benchmark_argv(selected))
        assert parsed.json and parsed.background and parsed.no_input
        for name in ("agent", "dataset", "gpu", "no_gpu", "compute", "environment", "host", "parallel_evaluations", "name", "no_preload"):
            assert getattr(parsed, name) == getattr(selected, name)
    asyncio.run(exercise())


def test_small_terminal_and_back_navigation(tmp_path):
    async def exercise():
        app = Wizard(args("benchmark", "--agent", "qwen3vl", "--home", str(tmp_path)))
        async with app.run_test(size=(65, 24)) as pilot:
            await pilot.press("enter", "enter")
            await pilot.pause()
            assert isinstance(app.screen, Configuration)
            assert app.screen.has_class("narrow")
            assert app.screen.query_one("#next").region.bottom <= 24
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen.kind == "dataset"
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen.kind == "agent"
            await pilot.press("ctrl+c")
        assert app.return_value is None
    asyncio.run(exercise())


def test_credential_editor_masks_values_and_keeps_them_off_argv(tmp_path):
    async def exercise():
        app = Wizard(args("benchmark", "--agent", "qwen3vl", "--dataset", "osworld-50", "--home", str(tmp_path)))
        async with app.run_test(size=(100, 42)) as pilot:
            await pilot.press("enter", "enter")
            await pilot.pause()
            await pilot.click("#credentials")
            await pilot.pause()
            assert isinstance(app.screen, Credentials)
            app.screen.query_one("#key-name", Input).value = "MODEL_NAME"
            app.screen.query_one("#key-value", Input).value = "private-model-choice"
            assert "private-model-choice" not in app.export_screenshot()
            await pilot.click("#add")
            await pilot.click("#next")
            await pilot.pause()
            selected = app.screen.read_config()
            assert selected._environment_values == {"MODEL_NAME": "private-model-choice"}
            assert "MODEL_NAME" in selected.env
            assert "private-model-choice" not in benchmark_argv(selected)
        assert not (tmp_path / "config.env").exists()
    asyncio.run(exercise())


@pytest.mark.parametrize("options", [("--json", "benchmark"), ("benchmark", "--json"), ("--json", "benchmark", "--bad-option")])
def test_json_errors_are_parseable_and_never_prompt(tmp_path, options):
    result = run_cli(*options, cwd=tmp_path)
    assert result.returncode != 0
    assert json.loads(result.stdout)["type"] == "error"
    assert "\x1b" not in result.stdout


def test_json_setup_and_catalog_in_fresh_installation(tmp_path):
    home = str(tmp_path / "installation")
    result = run_cli("--json", "setup", "--home", home, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["type"] == "ready"
    assert "[ok]" not in result.stdout
    result = run_cli("--json", "catalog", "--home", home, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["benchmarks"]) == 7
    assert (Path(home) / "config.env").stat().st_mode & 0o777 == 0o600


def test_preparation_screen_runs_real_setup(tmp_path):
    async def exercise():
        app = Preparation(["setup", "--json", "--no-input", "--home", str(tmp_path / "install")], "Set up")
        async with app.run_test(size=(90, 30)):
            await app.workers.wait_for_complete()
        assert app.return_value["type"] == "ready"
        assert (tmp_path / "install/platform.db").is_file()
        assert app.process.returncode == 0
    asyncio.run(exercise())


def test_preparation_errors_remain_visible(tmp_path):
    async def exercise():
        app = Preparation(["benchmark", "--json", "--home", str(tmp_path)], "Benchmark")
        async with app.run_test(size=(90, 30)) as pilot:
            await app.workers.wait_for_complete()
            assert app.return_value is None
            assert app.result["type"] == "error"
            assert "failed" in str(app.query_one("#phase", Static).render())
            assert app.query_one(RichLog).display
            await pilot.press("d")
            assert not app.query_one(RichLog).display
            await pilot.click("#details")
            assert app.query_one(RichLog).display
            await pilot.click("#stop")
        assert app.process.returncode == 1
    asyncio.run(exercise())
