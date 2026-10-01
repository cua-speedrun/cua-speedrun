"""Hosted input, persistence and CLI checks without starting environments."""

import argparse
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import zipfile

import pytest

from cua_speedrun.commands import register_operator_commands
from cua_speedrun.commands.wizard import benchmark_argv
from cua_speedrun.hosted import evaluation_id, initial_status
from cua_speedrun.hosted.inputs import pack_inputs, unpack_inputs
from cua_speedrun.hosted.worker import snapshot


def parser():
    value = argparse.ArgumentParser()
    register_operator_commands(value.add_subparsers(dest="command"))
    return value


def test_hosted_flags_and_existing_local_ids():
    args = parser().parse_args(['benchmark', '--host', 'modal', '--agent', 'claude_code', '--dataset', 'osworld50'])
    assert parser().parse_args(benchmark_argv(args)).host == 'modal'
    for command in ('status', 'export', 'cancel'):
        assert parser().parse_args([command, '3']).run_id == 3
        assert parser().parse_args([command, 'm-0123456789abcdef']).run_id == 'm-0123456789abcdef'
    assert parser().parse_args(['evaluations', '--host', 'modal']).host == 'modal'


@pytest.mark.parametrize('value', ['0', '-1', '../run', 'm-1234', 'm-' + '0' * 16 + '/other'])
def test_invalid_run_ids(value):
    with pytest.raises(argparse.ArgumentTypeError):
        evaluation_id(value)


@pytest.mark.parametrize('name', ['../outside', '/etc/passwd', 'benchmark/../../outside', 'other/file'])
def test_input_archive_cannot_escape(name, tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as file:
        file.writestr(name, 'content')
    with pytest.raises(ValueError, match='invalid hosted input'):
        unpack_inputs(buffer.getvalue(), tmp_path)
    assert not list(tmp_path.iterdir())


def test_real_custom_agent_bundle_and_public_benchmark(tmp_path):
    agent = Path(__file__).resolve().parents[1] / 'agents/claude_code'
    args = parser().parse_args(['benchmark', '--host', 'modal', '--agent', str(agent), '--dataset', 'osworld50'])
    request, bundle, variables = pack_inputs(args)
    assert request['task_count'] == 50
    assert request['argv'][request['argv'].index('--host') + 1] == 'local'
    assert request['argv'][request['argv'].index('--environment') + 1] == 'modal-native'
    assert all(name not in json.dumps(request) for name in variables.values() if name)
    unpack_inputs(bundle, tmp_path)
    assert (tmp_path / 'agent/agent.py').read_bytes() == (agent / 'agent.py').read_bytes()
    assert set(p.name for p in (tmp_path / 'agent').iterdir()) == {'init.py', 'agent.py', 'agent.json'}


def test_checkpoint_is_consistent_and_excludes_configuration(tmp_path):
    home, destination = tmp_path / 'home', tmp_path / 'saved'
    home.mkdir()
    destination.mkdir()
    (home / 'config.env').write_text('CS_SECRET_KEY=unit-test-only\n')
    (home / 'runs/example').mkdir(parents=True)
    (home / 'runs/example/result.json').write_text('{}')
    with sqlite3.connect(home / 'platform.db') as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE completed (id INTEGER)')
        db.execute('INSERT INTO completed VALUES (1)')
        db.commit()
        snapshot(home, destination)
        with sqlite3.connect(destination / 'state.sqlite') as saved:
            assert saved.execute('SELECT id FROM completed').fetchall() == [(1,)]
    assert not (destination / 'config.env').exists()
    assert (destination / 'runs/example/result.json').read_text() == '{}'


def test_initial_status_uses_existing_dashboard_shape():
    status = initial_status({'run_id': 'm-0123456789abcdef', 'name': 'Claude',
                             'benchmark': 'osworld-50', 'task_count': 50, 'created_at': 0})
    assert status['benchmark'] == 'osworld-50'
    assert status['stage'] == 'preparing'
    assert status['progress']['total'] == 50


def test_bundled_inputs_need_no_local_downloads(tmp_path):
    program = '''
import argparse
import json
from cua_speedrun.commands import register_operator_commands
from cua_speedrun.commands.paths import InstallationPaths
from cua_speedrun.commands.setup import initialize
from cua_speedrun.hosted.inputs import pack_inputs
paths = InstallationPaths.resolve()
initialize(paths)
parser = argparse.ArgumentParser()
register_operator_commands(parser.add_subparsers(dest="command"))
counts = {}
for dataset in ("osworld-50", "osworld-offline", "osworld2-52", "osworld2-offline",
                "cua-world-26", "cua-world-offline", "my-pc-bench"):
    args = parser.parse_args(["benchmark", "--host", "modal", "--agent", "claude_code", "--dataset", dataset])
    request, archive, variables = pack_inputs(args)
    counts[dataset] = request["task_count"]
    if dataset == "my-pc-bench":
        assert "MYPCBENCH_JUDGE_API_KEY" in variables
assert not (paths.resource_root / "benchmark-assets").exists()
print(json.dumps(counts))
'''
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(('CS_', 'CUA_SPEEDRUN_', 'GYM_ANYTHING_', 'MYPCBENCH_'))}
    environment.update(CUA_SPEEDRUN_HOME=str(tmp_path), MYPCBENCH_JUDGE_API_KEY='packaging-check-only')
    result = subprocess.run([sys.executable, '-c', program], env=environment,
                            capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(result.stdout.splitlines()[-1]) == {
        'osworld-50': 50, 'osworld-offline': 295, 'osworld2-52': 52,
        'osworld2-offline': 63, 'cua-world-26': 26, 'cua-world-offline': 143,
        'my-pc-bench': 38,
    }


def test_preparation_credentials_stay_out_of_agent_inputs(tmp_path):
    program = '''
import argparse, json, os
from cua_speedrun.commands import register_operator_commands
from cua_speedrun.commands.paths import InstallationPaths
from cua_speedrun.commands.setup import initialize
from cua_speedrun.hosted.inputs import pack_inputs
initialize(InstallationPaths.resolve())
parser = argparse.ArgumentParser()
register_operator_commands(parser.add_subparsers(dest="command"))
for dataset in ("osworld2-52", "osworld2-offline", "osworld-50"):
    args = parser.parse_args(["benchmark", "--host", "modal", "--agent", "claude_code", "--dataset", dataset])
    request, archive, credentials = pack_inputs(args)
    assert ("HF_TOKEN" in credentials) == dataset.startswith("osworld2")
    assert "HF_TOKEN" not in request["argv"]
    assert os.environ["HF_TOKEN"] not in json.dumps(request)
    assert os.environ["HF_TOKEN"].encode() not in archive
'''
    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith(('CS_', 'CUA_SPEEDRUN_', 'GYM_ANYTHING_'))}
    environment.update(CUA_SPEEDRUN_HOME=str(tmp_path), HF_TOKEN='packaging-check-only')
    subprocess.run([sys.executable, '-c', program], env=environment, check=True, timeout=30,
                   capture_output=True, text=True)
