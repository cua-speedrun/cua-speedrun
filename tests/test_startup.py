"""Offline checks for preparation; the cold-start acceptance target is 180 s."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from rich.console import Console
import yaml

from cua_speedrun.benchmark_preparation import independent_environment_preparation
from cua_speedrun.commands.preparation_view import render_preparation
from cua_speedrun.commands.status_tui import StatusViewState, _handle_key
from cua_speedrun.remote import snapshot_cache
from cua_speedrun.startup import PREFIX, PreparationProgress
from cua_speedrun.evaluation_runtime import (
    INDEPENDENT_ENVIRONMENT_PREPARATION, LOCAL_RUNTIME_CAPABILITIES, MODAL_RUNTIME_CAPABILITIES,
)


ROOT = Path(__file__).resolve().parents[1]


def test_bundled_tasks_preserve_pinned_source_hashes():
    spec = importlib.util.spec_from_file_location("osworld_importer", ROOT / "scripts/import_osworld_eval.py")
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    names, sources = importer.task_sources()
    assert len(names) == 369
    for benchmark in ("osworld-50", "osworld-offline"):
        definition = yaml.safe_load((ROOT / "benchmarks" / benchmark / "benchmark-source.yaml").read_text())
        expected = {item["id"]: item["source_sha256"] for item in definition["tasks"]}
        actual = {}
        for name in names:
            domain, uid = name.split("__")
            task_id = importer.task_id(domain, uid)
            if task_id in expected:
                actual[task_id] = hashlib.sha256((json.dumps(sources[name], indent=2) + "\n").encode()).hexdigest()
        assert actual == expected


def test_fresh_osworld_materialization_needs_no_network(tmp_path):
    program = '''
import json, runpy, sys, time
from cua_speedrun.commands.paths import InstallationPaths
from cua_speedrun.commands.setup import initialize
from cua_speedrun.specs import Benchmark
paths = InstallationPaths.resolve()
initialize(paths)
def offline(event, args):
    if event == "socket.connect":
        raise RuntimeError("task materialization must not use the network")
sys.addaudithook(offline)
started = time.monotonic()
runpy.run_path(str(paths.resource_root / "scripts/import_osworld_eval.py"), run_name="__main__")
benchmark = Benchmark.load(paths.resource_root / "benchmarks/osworld-50")
print(json.dumps({"tasks": len(benchmark.tasks), "seconds": time.monotonic() - started}))
'''
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("CS_", "CUA_SPEEDRUN_"))}
    environment["CUA_SPEEDRUN_HOME"] = str(tmp_path / "installation")
    result = subprocess.run([sys.executable, "-c", program], env=environment,
                            capture_output=True, text=True, check=True, timeout=30)
    measured = json.loads(result.stdout.splitlines()[-1])
    assert measured["tasks"] == 50
    assert measured["seconds"] < 10


def test_parallel_preparation_is_explicit_and_dependency_aware():
    assert independent_environment_preparation(ROOT / "benchmarks/osworld-50", "modal-native")
    assert not independent_environment_preparation(ROOT / "benchmarks/osworld-50", "modal")
    assert not independent_environment_preparation(ROOT / "benchmarks/osworld2-52", "modal-native")
    assert not independent_environment_preparation(ROOT / "benchmarks/my-pc-bench", "modal-native")
    assert independent_environment_preparation(ROOT / "benchmarks/cua-world-26", "modal-native")
    assert not independent_environment_preparation(ROOT / "benchmarks/cua-world-26", "modal")
    assert INDEPENDENT_ENVIRONMENT_PREPARATION in MODAL_RUNTIME_CAPABILITIES
    assert INDEPENDENT_ENVIRONMENT_PREPARATION not in LOCAL_RUNTIME_CAPABILITIES


def test_osworld2_preparation_overlaps_independent_asset_and_image_downloads():
    from cua_speedrun.benchmark_preparation import preparation_options

    for name in ("osworld2-52", "osworld2-offline"):
        options = preparation_options(ROOT / "benchmarks" / name)
        assert options["parallel"] == ["modal-native"]
        assert options["forward_env"] == ["HF_TOKEN"]
        assert len(options["modal-native"]) == 2


def test_concurrent_cache_writes_preserve_every_entry(tmp_path):
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(lambda i: snapshot_cache.store(tmp_path, f"key-{i}", f"image-{i}", f"run-{i}"), range(60)))
    for i in range(60):
        assert snapshot_cache.lookup(tmp_path, f"key-{i}")["image_id"] == f"image-{i}"
    assert snapshot_cache.lookup(tmp_path, "missing") is None


def test_hosted_cache_survives_different_controller_roots(tmp_path):
    environment = {**os.environ, "CS_AGENT_SNAPSHOT_CACHE_DIR": str(tmp_path / "shared")}
    program = '''
from pathlib import Path
from cua_speedrun.remote import snapshot_cache
import sys
root = Path(sys.argv[1])
if sys.argv[2] == "store":
    snapshot_cache.store(root, "same-contract", "image-id", "prior-run")
else:
    assert snapshot_cache.lookup(root, "same-contract")["image_id"] == "image-id"
    assert snapshot_cache.lookup(root, "different-contract") is None
'''
    for folder, action in (("first", "store"), ("second", "lookup")):
        subprocess.run([sys.executable, "-c", program, str(tmp_path / folder), action], env=environment, check=True)


def test_preparation_details_are_opt_in_and_events_survive_log_volume():
    progress = PreparationProgress()
    now = time.time()
    for key, label in (("controller", "Start hosted controller"), ("desktop", "Prepare desktop image")):
        progress.feed(PREFIX + json.dumps({"key": key, "label": label, "state": "running", "at": now}))
    for _ in range(300):
        progress.feed("Copying blob sha256:diagnostic-only")
    progress.feed(PREFIX + json.dumps({"key": "controller", "label": "Start hosted controller",
                                      "state": "done", "at": now + 2, "elapsed_sec": 2}))
    payload = progress.payload()
    assert len(payload["preparation_steps"]) == 2
    assert len(payload["preparation"]) == 200
    def rendered(details):
        stream = io.StringIO()
        Console(file=stream, width=80, color_system=None).print(render_preparation(payload, details=details))
        return stream.getvalue()
    assert "Start hosted controller" in rendered(False)
    assert "Prepare desktop image" in rendered(False)
    assert "sha256" not in rendered(False)
    assert "sha256" in rendered(True)
    state = StatusViewState()
    assert _handle_key(state, "enter", payload, 10)
    assert state.details
    assert _handle_key(state, "enter", payload, 10)
    assert not state.details
