#!/usr/bin/env python3
"""Materialize the pinned 38-task MyPCBench benchmark.

The compact source names 38 tasks from MyPCBench's public canonical graded
task file. This builder downloads that file from the pinned commit, verifies
its SHA-256, selects the specified tasks, and emits an executable
gym-anything package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "scripts" / "mypcbench_shared"
MYPCBENCH_IMAGE_CONTRACT = json.loads(
    (ROOT / "benchmarks" / "mypcbench-image.json").read_text(encoding="utf-8")
)
DEFAULT_TIMEOUT_SEC = 7200
MAX_TASK_SOURCE_BYTES = 8 * 1024 * 1024


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode())


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def _read_spec(source_path: Path) -> dict[str, Any]:
    spec = yaml.safe_load(source_path.read_text()) or {}
    source = spec.get("source_benchmark")
    selection = spec.get("selection")
    tasks = spec.get("tasks")
    if not isinstance(source, dict) or not isinstance(selection, dict):
        raise ValueError(f"{source_path}: source_benchmark and selection are required")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{source_path}: tasks must be a non-empty list")
    ids = [str(task) for task in tasks if isinstance(task, str)]
    if len(ids) != len(tasks) or any(not task_id.strip() for task_id in ids):
        raise ValueError(f"{source_path}: tasks must be non-empty string IDs")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source_path}: duplicate task id")
    actual = hashlib.sha256(
        "".join(f"{task_id}\n" for task_id in ids).encode()
    ).hexdigest()
    expected = str(selection.get("task_ids_sha256", ""))
    if actual != expected:
        raise ValueError(
            f"{source_path}: ordered task digest mismatch: expected {expected}, got {actual}"
        )
    return spec


def _download_task_records(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source = spec["source_benchmark"]
    task_source = source.get("tasks")
    if not isinstance(task_source, dict):
        raise ValueError("source_benchmark.tasks must be a mapping")
    url = str(task_source.get("url") or "")
    expected = str(task_source.get("sha256") or "")
    expected_count = int(task_source.get("task_count") or 0)
    commit = str(source.get("commit") or "")
    if not url.startswith("https://") or commit not in url:
        raise ValueError("MyPCBench task URL must use HTTPS and pin the source commit")
    request = urllib.request.Request(url, headers={"User-Agent": "cua-speedrun"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(MAX_TASK_SOURCE_BYTES + 1)
    if len(payload) > MAX_TASK_SOURCE_BYTES:
        raise ValueError("MyPCBench task source exceeds the download limit")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(
            f"MyPCBench task source expected SHA-256 {expected}, got {actual}"
        )
    records = json.loads(payload)
    if (
        not isinstance(records, list)
        or len(records) != expected_count
        or any(not isinstance(record, dict) for record in records)
    ):
        raise ValueError("MyPCBench task source has an invalid task inventory")
    by_id = {
        str(record.get("id")): record
        for record in records
        if isinstance(record.get("id"), str) and record["id"]
    }
    if len(by_id) != len(records):
        raise ValueError("MyPCBench task source has missing or duplicate task IDs")
    return by_id


def _environment_contract(spec: dict[str, Any]) -> dict[str, Any]:
    guest = MYPCBENCH_IMAGE_CONTRACT["guest"]
    source = MYPCBENCH_IMAGE_CONTRACT["official_source"]
    return {
        "id": f"{spec['name']}@{spec['version']}",
        "version": str(spec["version"]),
        "description": (
            "MyPCBench persona desktop (Ubuntu 24.04 GNOME, 17 seeded pre-logged-in "
            f"web apps plus LibreOffice) for {spec['name']}."
        ),
        "base": "ubuntu-gnome-systemd",
        "resources": {"cpu": 4, "mem_gb": 8, "gpu": 0, "net": True},
        "observation": [{"type": "rgb_screen", "resolution": [1280, 800]}],
        "action": [{"type": "mouse"}, {"type": "keyboard"}],
        # Match MyPCBench's canonical VM runner, whose step_delay defaults to
        # two seconds. The delay remains inside the environment-owned clock.
        "action_settle_ms": 2000,
        "mounts": [{"source": "tasks", "target": "/workspace/tasks", "mode": "ro"}],
        "vnc": {
            "enable": True,
            "host_port": -1,
            "view_only": True,
            "password": "password",
        },
        "qemu_base_image": "${MYPCBENCH_QEMU_BASE_IMAGE}",
        "qemu_base_format": "qcow2",
        "qemu_require_base_image": True,
        # The persona image requires UEFI; boot with the OVMF sidecars.
        "qemu_firmware": "uefi",
        "qemu_base_image_provenance": {
            "schema_version": MYPCBENCH_IMAGE_CONTRACT["provenance_schema_version"],
            "recipe": MYPCBENCH_IMAGE_CONTRACT["recipe"],
            "source_revision": source["revision"],
            "source_image_sha256": source["image_sha256"],
            "ssh_prepared": True,
            "tools_prepared": True,
            "nopasswd_sudo": True,
            "ssh_user": guest["ssh_user"],
            "guest_auth_prepared": True,
        },
        "qemu_x11_display": ":0",
        "ssh": {
            "user": guest["ssh_user"],
            "password": guest["ssh_password"],
        },
    }


def _verifier_wrapper() -> str:
    return """from pathlib import Path
import importlib.util

_shared = Path(__file__).resolve().parents[1] / "_shared" / "mypcbench_verifier.py"
_spec = importlib.util.spec_from_file_location("_mypcbench_verifier", _shared)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def check(traj, env_info, task_info):
    return _mod.check_with_source(
        Path(__file__).with_name("source.json"), traj, env_info, task_info
    )
"""


def materialize(source_path: Path, out: Path | None = None) -> Path:
    """Build the locked benchmark package at ``out``."""
    if out is None:
        raise ValueError("materialize requires an output directory")
    source_path = Path(source_path).resolve()
    out = Path(out).resolve()
    spec = _read_spec(source_path)
    records = _download_task_records(spec)
    missing = [task_id for task_id in spec["tasks"] if task_id not in records]
    if missing:
        raise ValueError(f"MyPCBench task source is missing selected IDs: {missing}")
    selected = [(task_id, records[task_id]) for task_id in spec["tasks"]]
    runtime = spec.get("runtime") or {}
    max_steps = int(runtime.get("max_steps") or 0)
    default_timeout = int(runtime.get("default_timeout_sec") or DEFAULT_TIMEOUT_SEC)
    if max_steps <= 0:
        raise ValueError("MyPCBench max_steps must be a positive integer")
    if default_timeout <= 0:
        raise ValueError("MyPCBench runtime timeout configuration is invalid")
    agent_context = spec.get("agent_context") or {}
    if agent_context.get("capability") != "cua_only":
        raise ValueError("MyPCBench agent context must declare cua_only")
    context_prompt = str(agent_context.get("prompt") or "").strip()
    if not context_prompt:
        raise ValueError("MyPCBench agent context prompt is required")

    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    upstream = spec["source_benchmark"]
    manifest = {
        "name": str(spec["name"]),
        "version": str(spec["version"]),
        "source": str(upstream["repository"]),
        "source_commit": str(upstream["commit"]),
        "image_version": str(upstream.get("image_version", "")),
        "grading": dict(
            spec.get("grading")
            or {
                "rubric_aggregation": "uniform_mean",
                "pass_condition": "all_rubrics",
            }
        ),
        "tasks": [f"tasks/{task_id}" for task_id, _ in selected],
    }
    _write_yaml(out / "manifest.yaml", manifest)

    environment = out / "environment"
    environment_contract = _environment_contract(spec)
    _write_json(environment / "env.json", environment_contract)
    _write_json(
        environment / "evaluator-environment.json",
        spec["evaluator_environment"],
    )
    if spec.get("native_image"):
        import importlib.util

        module_spec = importlib.util.spec_from_file_location(
            "mypcbench_native_recipe", SHARED / "native_image.py"
        )
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        _write_json(environment / "native-image.json", module.contract(ROOT))
    # The generic Modal environment launcher consumes this allowlist. Judge
    # credentials belong to the evaluator process, never the agent/guest.
    _write_json(
        environment / "host-runtime.json",
        {
            "forward_env": sorted(set(spec["evaluator_environment"]["required"])
                                  | set(spec["evaluator_environment"]["private"])),
        },
    )
    # Keep build-only image helpers and attribution files out of the runtime
    # environment. The evaluator needs only these four pinned files.
    runtime_shared = environment / "tasks" / "_shared"
    runtime_shared.mkdir(parents=True)
    for name in ("mypcbench_setup.py", "mypcbench_verifier.py"):
        shutil.copy2(SHARED / name, runtime_shared / name)
    canonical = runtime_shared / "canonical"
    canonical.mkdir()
    for name in ("osworld_full_traj_judge.py", "provenance.json"):
        shutil.copy2(SHARED / "canonical" / name, canonical / name)

    for name, source in selected:
        timeout_sec = default_timeout
        _write_yaml(
            out / "tasks" / name / "task.yaml",
            {
                "task_id": f"mypcbench_{name.replace('-', '_')}",
                "description": (
                    f"MyPCBench {source.get('category', 'task')} {name} "
                    "(instruction comes from the environment task spec)"
                ),
                "timeout_sec": timeout_sec,
                "env": {
                    "kind": "gym-anything",
                    "env_dir": "${BENCHMARK_DIR}/environment",
                    "task_id": name,
                    "use_cache": False,
                    "action_settle_ms": environment_contract["action_settle_ms"],
                },
                "metadata": {
                    "mypcbench_id": name,
                    "mypcbench_category": source.get("category"),
                    "mypcbench_difficulty": source.get("difficulty"),
                    "mypcbench_apps_involved": source.get("apps_involved"),
                    "mypcbench_rubric_aggregation": "uniform_mean",
                    "mypcbench_selection": spec["selection"]["method"],
                },
            },
        )

        env_task = environment / "tasks" / name
        # Firstboot of the seeded image can take minutes before the app stack
        # answers, and the 17-app warmup retries on top of that (all untimed).
        # fail_on_error is opt-in and defaults to permissive: without it a
        # failed warmup runs the episode against a dead app stack.
        hooks: dict[str, Any] = {
            "pre_task": (
                "python3 /workspace/tasks/_shared/mypcbench_setup.py "
                f"/workspace/tasks/{name}/source.json"
            ),
            "pre_task_timeout": 1200,
            "fail_on_error": True,
        }
        rubrics = (source.get("grading") or {}).get("rubrics") or []
        _write_json(
            env_task / "task.json",
            {
                "id": f"{name}@1",
                "env_id": environment_contract["id"],
                "difficulty": source.get("difficulty") or "unknown",
                "natural_language": {
                    "prompt": (
                        f"{context_prompt}\n\nTask:\n"
                        f"{source.get('instruction') or ''}"
                    )
                },
                "init": {
                    "timeout_sec": timeout_sec,
                    "max_steps": max_steps,
                    "reward_type": "sparse",
                },
                "hooks": hooks,
                "success": {
                    "mode": "program",
                    "spec": {"program": "verifier.py::check"},
                },
                "metadata": {
                    "mypcbench_id": name,
                    "mypcbench_category": source.get("category"),
                    "mypcbench_grading_type": (source.get("grading") or {}).get("type"),
                    "mypcbench_rubric_count": len(rubrics),
                    "mypcbench_rubric_aggregation": "uniform_mean",
                    "mypcbench_selection": spec["selection"]["method"],
                },
            },
        )
        _write_json(env_task / "source.json", source)
        (env_task / "verifier.py").write_text(_verifier_wrapper())

    print(f"materialized {len(selected)} MyPCBench tasks in {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out is not None:
        path = materialize(args.source, args.out)
    else:
        from cua_speedrun.benchmark_sources import materialize_benchmark

        path = materialize_benchmark(args.source)
    print(path)


if __name__ == "__main__":
    main()
