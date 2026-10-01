#!/usr/bin/env python3
"""Materialize the pinned official OSWorld 2.0 benchmark release."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml


CODE_COMMIT = "2b9b7b4eb73243d557bdbf2998fe18d8e18e19c6"
CODE_ARCHIVE_URL = (
    "https://codeload.github.com/xlang-ai/OSWorld-V2/tar.gz/"
    "refs/tags/v2026.06.24"
)
CODE_ARCHIVE_SHA256 = "7e9c5c1f79e346373ed8ca6745c465ba5a3fe61eae72c480959447bcb8582986"
TASK_REPO = "xlangai/osworld_v2_tasks"
TASK_REVISION = "v2026.06.24"
TASK_HASH_MANIFEST = "manifests/task_hashes.json"
TASK_HASH_MANIFEST_SHA256 = "3312a7df40dbd004c300804f71c57d5a23a083d6c675082fcc34c60a37f9a76c"
IMAGE_ARCHIVE_SHA256 = "eb737ae70b49849e24af407de6a518439a23de05a8497096a948334ce0a909aa"
MAX_STEPS = 500
TASK_TIMEOUT_SEC = 39600
DEFAULT_VOLUME_SIZE_GB = 40
INSTANCE_RESOURCES = {
    None: {"cpu": 2, "mem_gb": 4},
    "t3.medium": {"cpu": 2, "mem_gb": 4},
    "t3.large": {"cpu": 2, "mem_gb": 8},
    "t3.xlarge": {"cpu": 4, "mem_gb": 16},
    "t3.2xlarge": {"cpu": 8, "mem_gb": 32},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resource_root(source_path: Path) -> Path:
    for candidate in source_path.parents:
        if (candidate / "scripts/osworld2_adapter.py").is_file():
            return candidate
    raise RuntimeError(f"cannot locate OSWorld2 resources above {source_path}")


def _download_code_archive(destination: Path) -> None:
    cache = Path.home() / ".cache/cua-speedrun/sources"
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"osworld2-{CODE_COMMIT}.tar.gz"
    if cached.is_file() and _sha256(cached) == CODE_ARCHIVE_SHA256:
        shutil.copy2(cached, destination)
        return

    for attempt in range(1, 4):
        descriptor, temporary_name = tempfile.mkstemp(prefix="osworld2.", dir=cache)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            request = urllib.request.Request(
                CODE_ARCHIVE_URL, headers={"User-Agent": "cua-speedrun"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            actual = _sha256(temporary)
            if actual != CODE_ARCHIVE_SHA256:
                raise RuntimeError(
                    f"OSWorld2 code archive checksum mismatch: {actual}"
                )
            temporary.replace(cached)
            shutil.copy2(cached, destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(attempt * 2)


def _safe_extract_code(archive: Path, destination: Path) -> Path:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError(
                    f"OSWorld2 archive member escapes extraction root: {member.name}"
                )
            if member.issym() or member.islnk():
                raise RuntimeError(
                    f"OSWorld2 archive unexpectedly contains a link: {member.name}"
                )
        bundle.extractall(destination)

    candidates = [
        path
        for path in destination.iterdir()
        if (path / "desktop_env/task_base.py").is_file()
        and (path / "task_loader.py").is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError("OSWorld2 archive has an unexpected root layout")
    return candidates[0]


def _download_tasks(destination: Path, expected_ids: list[str]) -> Path:
    try:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.utils import HfHubHTTPError, LocalTokenNotFoundError
    except ImportError as exc:
        raise RuntimeError(
            "OSWorld2 materialization needs huggingface-hub>=0.35.0"
        ) from exc

    expected_names = [f"task_{task_id}.py" for task_id in expected_ids]
    release_names = [f"task_{number:03d}.py" for number in range(1, 109)]
    try:
        remote_files = HfApi(token=True).list_repo_files(
            repo_id=TASK_REPO,
            repo_type="dataset",
            revision=TASK_REVISION,
        )
        root_tasks = sorted(
            name
            for name in remote_files
            if "/" not in name and name.startswith("task_") and name.endswith(".py")
        )
        if root_tasks != release_names:
            raise RuntimeError(
                f"{TASK_REPO}@{TASK_REVISION} does not contain exactly the "
                "official 108 task files"
            )
        snapshot = Path(
            snapshot_download(
                repo_id=TASK_REPO,
                repo_type="dataset",
                revision=TASK_REVISION,
                allow_patterns=[*expected_names, TASK_HASH_MANIFEST],
                local_dir=destination,
                token=True,
            )
        )
    except LocalTokenNotFoundError as exc:
        raise RuntimeError(
            "OSWorld2 requires a Hugging Face login. Accept access at "
            "https://huggingface.co/datasets/xlangai/osworld_v2_tasks, "
            "then run `uvx --from huggingface_hub hf auth login`."
        ) from exc
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status in {401, 403, 404}:
            raise RuntimeError(
                "OSWorld2 tasks are gated. Accept access at "
                "https://huggingface.co/datasets/xlangai/osworld_v2_tasks, "
                "then run `uvx --from huggingface_hub hf auth login`."
            ) from exc
        raise

    missing = [name for name in expected_names if not (snapshot / name).is_file()]
    if missing:
        raise RuntimeError(f"downloaded OSWorld2 snapshot is missing: {missing}")
    return snapshot


def _hash_value(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.removeprefix("sha256:").lower()
        if len(text) == 64 and all(char in "0123456789abcdef" for char in text):
            return text
    if isinstance(value, dict):
        for key in ("sha256", "hash", "digest"):
            found = _hash_value(value.get(key))
            if found:
                return found
    return None


def _task_hashes(manifest: Any) -> dict[str, str]:
    candidates: list[Any] = [manifest]
    if isinstance(manifest, dict):
        candidates.extend(
            manifest.get(key) for key in ("tasks", "files", "hashes")
        )
    for candidate in candidates:
        result: dict[str, str] = {}
        if isinstance(candidate, dict):
            for name, value in candidate.items():
                digest = _hash_value(value)
                if digest and str(name).endswith(".py"):
                    result[Path(str(name)).name] = digest
        elif isinstance(candidate, list):
            for item in candidate:
                if not isinstance(item, dict):
                    continue
                name = next(
                    (item.get(key) for key in ("path", "name", "file") if item.get(key)),
                    None,
                )
                digest = _hash_value(item)
                if name and digest and str(name).endswith(".py"):
                    result[Path(str(name)).name] = digest
        if result:
            return result
    raise RuntimeError("official OSWorld2 task hash manifest has an unknown schema")


def _verify_tasks(snapshot: Path, expected_ids: list[str]) -> None:
    manifest_path = snapshot / TASK_HASH_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError("downloaded OSWorld2 tasks omit the release hash manifest")
    actual_manifest_sha = _sha256(manifest_path)
    if actual_manifest_sha != TASK_HASH_MANIFEST_SHA256:
        raise RuntimeError(
            "OSWorld2 task hash manifest checksum mismatch: "
            f"{actual_manifest_sha}"
        )
    hashes = _task_hashes(json.loads(manifest_path.read_text()))
    expected_names = [f"task_{task_id}.py" for task_id in expected_ids]
    release_names = [f"task_{number:03d}.py" for number in range(1, 109)]
    if sorted(hashes) != release_names:
        raise RuntimeError(
            "OSWorld2 task hash manifest does not cover exactly the release tasks"
        )
    for name in expected_names:
        actual = _sha256(snapshot / name)
        if actual != hashes[name]:
            raise RuntimeError(f"OSWorld2 task checksum mismatch for {name}: {actual}")


def _task_runtime_metadata(path: Path) -> dict[str, Any]:
    """Read literal runtime requirements without importing gated task code."""
    tree = ast.parse(path.read_text(), filename=str(path))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            ast.unparse(base).rsplit(".", 1)[-1] in {"BaseTask", "MultiPhaseTask"}
            for base in node.bases
        )
    ]
    if len(classes) != 1:
        raise RuntimeError(f"expected one OSWorld2 task class in {path}")
    values: dict[str, Any] = {}
    for statement in classes[0].body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            value = statement.value
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in {
                "instance_type",
                "volume_size",
            }:
                continue
            try:
                values[target.id] = ast.literal_eval(value)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"OSWorld2 task runtime requirement {target.id} must be "
                    f"literal: {path}"
                ) from exc
    instance_type = values.get("instance_type")
    if instance_type not in INSTANCE_RESOURCES:
        raise RuntimeError(
            f"unsupported OSWorld2 instance_type {instance_type!r}: {path}"
        )
    return {
        "instance_type": instance_type,
        "volume_size": values.get("volume_size"),
        "resources": INSTANCE_RESOURCES[instance_type],
    }


def _environment_config() -> dict[str, Any]:
    return {
        "id": "osworld2.2026.06.24",
        "version": "2026.06.24",
        "description": "Official OSWorld2 Ubuntu environment and Python task runtime.",
        "base": "ubuntu-gnome-systemd_highres",
        "resources": {"cpu": 2, "mem_gb": 4, "gpu": 0, "net": True},
        "observation": [{"type": "rgb_screen", "resolution": [1920, 1080]}],
        "action": [{"type": "mouse"}, {"type": "keyboard"}],
        "mounts": [],
        "ssh": {"user": "user", "password": "osworld-public-evaluation"},
        "vnc": {
            "enable": False,
            "host_port": -1,
            "view_only": True,
            "password": "password",
        },
        "qemu_base_image": "${GYM_ANYTHING_QEMU_CACHE}/osworld2_ubuntu.qcow2",
        "qemu_base_format": "qcow2",
        "qemu_require_base_image": True,
        "qemu_base_image_provenance": {
            "schema_version": 1,
            "recipe": "cua-speedrun-osworld2-qcow2@1",
            "release": "osworld-v2-2026.06.24",
            "artifact_tag": "v2026.06.24",
            "artifact_path": "osworld-v2-ubuntu-x86.qcow2.zip",
            "archive_sha256": IMAGE_ARCHIVE_SHA256,
        },
        "qemu_x11_display": ":0",
    }


def _gym_task(task_id: str) -> dict[str, Any]:
    return {
        "id": f"task_{task_id}@2026.06.24",
        "env_id": "osworld2.2026.06.24",
        "difficulty": "unknown",
        "natural_language": {
            "prompt": f"OSWorld2 task {task_id}; instruction resolves during canonical setup."
        },
        "init": {
            "timeout_sec": TASK_TIMEOUT_SEC,
            "max_steps": MAX_STEPS,
            "reward_type": "sparse",
        },
        "success": {"mode": "program", "spec": {"program": "verifier.py::check"}},
    }


def _task_yaml(
    task_id: str,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = runtime or {
        "instance_type": None,
        "volume_size": None,
        "resources": INSTANCE_RESOURCES[None],
    }
    volume_size = runtime["volume_size"]
    if volume_size is None:
        volume_size = DEFAULT_VOLUME_SIZE_GB
    return {
        "task_id": f"osworld2_{task_id}",
        "description": (
            f"OSWorld2 task {task_id} (the exact gated instruction is resolved "
            "by the environment before timing starts)"
        ),
        "timeout_sec": TASK_TIMEOUT_SEC,
        "env": {
            "kind": "gym-anything",
            "env_dir": "${BENCHMARK_DIR}/environment",
            "task_id": task_id,
            "use_cache": False,
            "max_steps": MAX_STEPS,
            "config_overrides": {"resources": runtime["resources"]},
            "qemu_volume_size_gb": volume_size,
            "adapter_entrypoint": "osworld2_adapter.py:prepare",
            "native_adapter_entrypoint": "osworld2_adapter.py:prepare",
            "osworld2_root": "osworld2",
            "osworld2_task_file": f"task_class/task_{task_id}.py",
            "osworld2_assets_dir": "${GYM_ANYTHING_QEMU_CACHE}/osworld2-assets",
            "website_host_suffix": "web.hku.icu",
            "enable_proxy": True,
            "require_a11y_tree": False,
            "action_settle_ms": 3000,
        },
        "metadata": {
            "benchmark": "osworld2",
            "osworld2_id": task_id,
            "release": "osworld-v2-2026.06.24",
            "code_commit": CODE_COMMIT,
            "task_revision": TASK_REVISION,
            "instance_type": runtime["instance_type"],
            "volume_size_gb": volume_size,
        },
    }


def _host_runtime_config() -> dict[str, Any]:
    return {
        "python_version": "3.12",
        "pyproject": "osworld2/pyproject.toml",
        # Official task evaluators invoke these host-side executables.
        # ffmpeg also supplies ffprobe.
        "apt_packages": ["ffmpeg", "imagemagick"],
        "forward_env": [
            "WEBSITE_HOST_SUFFIX",
            "GITLAB_URL",
            "GITLAB_PRIVATE_TOKEN",
            "OSWORLD_EVAL_MODEL_PROVIDER",
            "OSWORLD_EVAL_MODEL_NAME",
            "OSWORLD_EVAL_MODEL_API_KEY",
            "OSWORLD_EVAL_MODEL_API_KEY_ENV",
            "OSWORLD_EVAL_MODEL_BASE_URL",
            "OSWORLD_EVAL_MODEL_TEMPERATURE",
            "OSWORLD_EVAL_MODEL_MAX_OUTPUT_TOKENS",
            "OSWORLD_EVAL_MODEL_RETRY_ATTEMPTS",
            "OSWORLD_EVAL_MODEL_RETRY_DELAY",
            "OSWORLD_EVAL_MODEL_IMAGE_DETAIL",
            "OSWORLD_EVAL_MODEL_REASONING_EFFORT",
            "OSWORLD_EVAL_MODEL_DEBUG",
            "OSWORLD_USER_SIM_PROVIDER",
            "OSWORLD_USER_SIM_API_KEY",
            "OSWORLD_USER_SIM_API_KEY_ENV",
            "OSWORLD_USER_SIM_MODEL",
            "OSWORLD_USER_SIM_BASE_URL",
            "OSWORLD_USER_SIM_TEMPERATURE",
            "OSWORLD_USER_SIM_MAX_TOKENS",
            "OSWORLD_USER_SIM_DEBUG",
            "OSWORLD2_PROXY_CONFIG_JSON",
        ],
    }


def materialize(*, source_path: Path, out: Path) -> None:
    source_path = Path(source_path).resolve()
    source = yaml.safe_load(source_path.read_text()) or {}
    task_ids = [str(item) for item in source.get("tasks") or ()]
    release_ids = [f"{number:03d}" for number in range(1, 109)]
    if (
        not task_ids
        or len(task_ids) != len(set(task_ids))
        or any(task_id not in release_ids for task_id in task_ids)
    ):
        raise ValueError(
            "OSWorld2 source tasks must be a unique subset of IDs 001 through 108"
        )

    root = _resource_root(source_path)
    with tempfile.TemporaryDirectory(prefix="osworld2.") as temporary_text:
        temporary = Path(temporary_text)
        archive = temporary / "osworld2.tar.gz"
        _download_code_archive(archive)
        code_root = _safe_extract_code(archive, temporary / "code")
        task_snapshot = _download_tasks(temporary / "tasks", task_ids)
        _verify_tasks(task_snapshot, task_ids)
        task_runtime = {
            task_id: _task_runtime_metadata(
                task_snapshot / f"task_{task_id}.py"
            )
            for task_id in task_ids
        }

        environment = out / "environment"
        shutil.copytree(code_root, environment / "osworld2")
        task_class = environment / "task_class"
        task_class.mkdir(parents=True)
        for task_id in task_ids:
            shutil.copy2(task_snapshot / f"task_{task_id}.py", task_class)
        shutil.copy2(root / "scripts/osworld2_adapter.py", environment)
        recipe_spec = importlib.util.spec_from_file_location(
            "osworld2_native_recipe", root / "scripts/osworld2_native.py"
        )
        recipe_module = importlib.util.module_from_spec(recipe_spec)
        recipe_spec.loader.exec_module(recipe_module)
        (environment / "native-image.json").write_text(
            json.dumps(recipe_module.contract(root), indent=2) + "\n"
        )

        (environment / "env.json").write_text(
            json.dumps(_environment_config(), indent=2) + "\n"
        )
        (environment / "host-runtime.json").write_text(
            json.dumps(_host_runtime_config(), indent=2)
            + "\n"
        )
        (environment / "evaluator-environment.json").write_text(
            json.dumps({"private": _host_runtime_config()["forward_env"]}, indent=2)
            + "\n"
        )

        manifest = {
            "name": str(source["name"]),
            "version": str(source["version"]),
            "source": CODE_ARCHIVE_URL,
            "source_commit": CODE_COMMIT,
            "tasks": [],
        }
        for task_id in task_ids:
            gym_task_dir = environment / "tasks" / task_id
            gym_task_dir.mkdir(parents=True)
            (gym_task_dir / "task.json").write_text(
                json.dumps(_gym_task(task_id), indent=2) + "\n"
            )
            (gym_task_dir / "verifier.py").write_text(
                "def check(*_args, **_kwargs):\n"
                "    raise RuntimeError('OSWorld2 must use its canonical task evaluator')\n"
            )

            task_rel = f"tasks/{task_id}"
            task_dir = out / task_rel
            task_dir.mkdir(parents=True)
            (task_dir / "task.yaml").write_text(
                yaml.safe_dump(
                    _task_yaml(task_id, task_runtime[task_id]),
                    sort_keys=False,
                )
            )
            manifest["tasks"].append(task_rel)

        (out / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False)
        )
        (out / "SOURCE.json").write_text(
            json.dumps(
                {
                    "release": "osworld-v2-2026.06.24",
                    "code_repository": "https://github.com/xlang-ai/OSWorld-V2",
                    "code_commit": CODE_COMMIT,
                    "code_archive_sha256": CODE_ARCHIVE_SHA256,
                    "task_repository": TASK_REPO,
                    "task_revision": TASK_REVISION,
                    "task_hash_manifest_sha256": TASK_HASH_MANIFEST_SHA256,
                    "task_count": len(task_ids),
                    "image_archive_sha256": IMAGE_ARCHIVE_SHA256,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(
        "This is a benchmark materializer; load benchmarks/osworld2-offline or benchmarks/osworld2-52 instead."
    )
