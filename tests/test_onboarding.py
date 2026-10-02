"""Public onboarding checks using the real catalog, scripts, and local database."""

import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest
import yaml

from cua_speedrun.benchmark_sources import benchmark_catalog_paths
from cua_speedrun.commands import register_operator_commands
from cua_speedrun.commands.benchmark import dataset_path, validate_agent
from cua_speedrun.commands.custom_benchmarks import register_benchmark, validate_benchmark
from cua_speedrun.commands.install_assets import install_resources
from cua_speedrun.commands.paths import InstallationPaths
from cua_speedrun.service.templates_catalog import agent_metadata, list_templates, template_zip
from cua_speedrun.specs import Benchmark


ROOT = Path(__file__).resolve().parents[1]


def parser():
    result = argparse.ArgumentParser()
    register_operator_commands(result.add_subparsers(dest="command"))
    return result


def test_simple_command_and_advanced_overrides():
    args = parser().parse_args(["benchmark", "--dataset", "osworld50", "--agent", "qwen3vl"])
    assert args.dataset == "osworld50"
    assert args.agent == "qwen3vl"
    assert args.compute == "modal"
    assert args.parallel_evaluations == 1
    args = parser().parse_args([
        "benchmark", "--dataset", "./tasks", "--agent", "./agent",
        "--gpu", "L4", "--parallel-evaluations", "3", "--no-preload", "--background",
    ])
    assert (args.gpu, args.parallel_evaluations, args.no_preload, args.background) == ("L4", 3, True, True)


def test_existing_submit_command_is_unchanged():
    args = parser().parse_args(["submit", "--template", "claude", "--benchmark", "osworld-50"])
    assert args.template == "claude"
    assert args.gpu is None
    assert args.compute == args.environment == "modal"


def test_all_bundled_agents_are_valid_and_have_metadata():
    agents = list_templates()
    names = {item["name"] for item in agents}
    assert names == {
        "autoglm_v", "claude", "claude_code", "codex_cli", "gemini",
        "gemini35", "gemini3_flash_preview", "glm5v_turbo", "jev", "kimi_k3",
        "meta", "minimax_m3", "openai", "qwen35", "qwen3vl", "yutori_n2",
    }
    assert names == {path.name for path in (ROOT / "agents").iterdir() if path.is_dir()}
    for item in agents:
        path = ROOT / "agents" / item["name"]
        validate_agent(path)
        assert agent_metadata(path)["description"]
        with zipfile.ZipFile(io.BytesIO(template_zip(item["name"]))) as archive:
            assert set(archive.namelist()) == {"init.py", "agent.py"}
            for script in archive.namelist():
                assert archive.read(script) == (path / script).read_bytes()
    assert agent_metadata(ROOT / "agents/qwen3vl")["gpu"] == "L40S"
    assert agent_metadata(ROOT / "agents/claude")["gpu"] is None


def test_setup_archives_old_agent_names_and_preserves_custom_copies(tmp_path):
    paths = InstallationPaths(tmp_path / "installation")
    renames = {
        "gemini_new": "gemini", "gpt54": "openai",
        "glm_cua": "autoglm_v", "qwen35vl": "qwen35",
    }
    for old, new in renames.items():
        shutil.copytree(ROOT / "agents" / new, paths.resource_root / "agents" / old)
    edited = paths.resource_root / "agents/gemini_new/agent.py"
    edited.write_text(edited.read_text() + "\n# Local customization.\n")
    old_content = edited.read_bytes()
    custom = paths.resource_root / "agents/my-agent"
    shutil.copytree(ROOT / "agents/claude", custom)

    install_resources(paths)
    install_resources(paths)

    installed = paths.resource_root / "agents"
    assert not any((installed / name).exists() for name in renames)
    assert {path.name for path in installed.iterdir()} == {
        path.name for path in (ROOT / "agents").iterdir() if path.is_dir()
    } | {"my-agent"}
    for name in renames.values():
        for filename in ("init.py", "agent.py", "agent.json"):
            assert (installed / name / filename).read_bytes() == (
                ROOT / "agents" / name / filename
            ).read_bytes()
    assert (custom / "agent.py").read_bytes() == (ROOT / "agents/claude/agent.py").read_bytes()
    backups = list((paths.resource_root.parent / "renamed-agents").iterdir())
    assert len(backups) == 4
    saved = next(path / "gemini_new/agent.py" for path in backups if path.name.startswith("gemini_new-"))
    assert saved.read_bytes() == old_content


def test_dataset_aliases_and_missing_names():
    assert dataset_path("osworld50").name == "osworld-50"
    assert dataset_path("osworld-50@0.1").name == "osworld-50"
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        dataset_path("not-a-benchmark")


def test_setup_needs_no_assets_model_or_credentials(tmp_path):
    installation = tmp_path / "installation"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CS_", "CUA_SPEEDRUN_", "MODAL_", "GYM_ANYTHING_", "OSWORLD_", "XDG_"))}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "third_party/gym-anything/src")])
    for _ in range(2):
        result = subprocess.run([
            sys.executable, "-m", "cua_speedrun.cli", "setup", "--no-input", "--home", str(installation),
        ], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
        assert "Ready." in result.stdout
    paths = InstallationPaths(installation)
    assert paths.database.is_file()
    assert paths.config_file.stat().st_mode & 0o777 == 0o600
    assert not paths.agent_runtime.exists()
    assert not paths.osworld_image.exists()
    assert not (paths.resource_root / "benchmark-assets").exists()
    assert len(benchmark_catalog_paths(paths.resource_root)) == 7
    assert {path.parent.name for path in (paths.resource_root / "agents").glob("*/agent.json")} == {
        path.parent.name for path in (ROOT / "agents").glob("*/agent.json")
    }


def test_register_real_benchmark_snapshot(tmp_path):
    # Use the actual OSWorld task package; no environment is started.
    spec = importlib.util.spec_from_file_location("osworld_builder", ROOT / "scripts/build_osworld_subset.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    source = tmp_path / "my-benchmark"
    builder.materialize(ROOT / "benchmarks/osworld-50/benchmark-source.yaml", source)
    manifest = source / "manifest.yaml"
    data = yaml.safe_load(manifest.read_text())
    data["name"] = "my-osworld-tasks"
    manifest.write_text(yaml.safe_dump(data))
    original = Benchmark.load(source)
    validate_benchmark(original)
    paths = InstallationPaths(tmp_path / "installation")
    paths.create_directories()
    install_resources(paths)
    registered = register_benchmark(original, paths)
    assert registered.name == original.name
    assert registered.benchmark_dir != source
    assert len(registered.tasks) == 50
    assert all(str(registered.benchmark_dir) in task.env["env_dir"] for task in registered.tasks)
    assert register_benchmark(original, paths).benchmark_dir == registered.benchmark_dir
    assert len(benchmark_catalog_paths(paths.resource_root)) == 8
    data["description"] = "Changed task set"
    manifest.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="new version"):
        register_benchmark(Benchmark.load(source), paths)
    assert yaml.safe_load((registered.benchmark_dir / "manifest.yaml").read_text())["description"] != data["description"]


def test_preparation_scripts_exist_and_are_packaged():
    for path in benchmark_catalog_paths(ROOT):
        data = yaml.safe_load((path / "benchmark-source.yaml").read_text())
        for key, steps in data.get("prepare", {}).items():
            if key == "default_environment":
                assert steps in {"modal", "modal-native", "local"}
                continue
            if key == "parallel":
                assert isinstance(steps, list)
                assert set(steps) <= {"modal", "modal-native", "local"}
                continue
            if key == "forward_env":
                from cua_speedrun.runtime_environment import normalize_environment_name

                assert isinstance(steps, list)
                assert all(normalize_environment_name(name) == name for name in steps)
                continue
            for step in steps:
                assert (ROOT / step["script"]).is_file()


def test_setup_preserves_huggingface_login_location(tmp_path):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CS_", "CUA_SPEEDRUN_", "GYM_ANYTHING_", "HF_", "XDG_"))}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "third_party/gym-anything/src")])
    code = """
import os
from pathlib import Path
from cua_speedrun.commands.paths import InstallationPaths, configure_process
expected = str(Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'huggingface' / 'token')
configure_process(InstallationPaths.resolve('installation'))
from huggingface_hub.constants import HF_TOKEN_PATH
assert HF_TOKEN_PATH == expected, (HF_TOKEN_PATH, expected)
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)
    env["XDG_CACHE_HOME"] = str(tmp_path / "user-cache")
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)


def test_upgrade_refreshes_resources_and_preserves_user_files(tmp_path):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CS_", "CUA_SPEEDRUN_", "MODAL_", "GYM_ANYTHING_", "OSWORLD_", "XDG_"))}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "third_party/gym-anything/src")])
    code = """
import json
from pathlib import Path
from cua_speedrun import __version__
from cua_speedrun.commands.paths import InstallationPaths
from cua_speedrun.commands.setup import initialize
from cua_speedrun.resources import bundled_resource_root
p = InstallationPaths.resolve('installation')
initialize(p)
config = p.config_file.read_bytes()
custom = p.resource_root / 'agents/my-agent'
custom.mkdir()
(custom / 'agent.py').write_text('print(42)\\n')
record = json.loads(p.install_record.read_text())
record['cua_speedrun_version'] = '0.0.0'
p.install_record.write_text(json.dumps(record))
(p.resource_root / 'agents/qwen3vl/agent.py').write_text('outdated')
for name in ('agent.py', 'init.py', 'agent.json'):
    (p.resource_root / 'agents/claude_code' / name).write_text('outdated')
initialize(p)
assert json.loads(p.install_record.read_text())['cua_speedrun_version'] == __version__
assert p.config_file.read_bytes() == config
assert (custom / 'agent.py').read_text() == 'print(42)\\n'
assert (p.resource_root / 'agents/qwen3vl/agent.py').read_text() != 'outdated'
for name in ('agent.py', 'init.py', 'agent.json'):
    relative = Path('agents/claude_code') / name
    assert (p.resource_root / relative).read_bytes() == (bundled_resource_root() / relative).read_bytes()
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, check=True)
