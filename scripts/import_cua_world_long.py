#!/usr/bin/env python3
"""Materialize the pinned CUA-World-Long split for cua-speedrun."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from cua_speedrun.remote.registry_images import image_definition


GYM_ANYTHING_COMMIT = "d5e3f88bdc66fcc79c037c4e5da48554535e52bb"
ARCHIVE_URL = (
    "https://codeload.github.com/cmu-l3/gym-anything/tar.gz/"
    + GYM_ANYTHING_COMMIT
)
ARCHIVE_SHA256 = "29aefe3285860706b97bc61f7be6c9cab313f536773e2768abefdb40cb4ce293"
# CUA-World Long's published model-side protocol is uniform across tasks;
# task.json limits describe the underlying environments, not this benchmark.
MAX_STEPS = 500
AGENT_TIMEOUT_SEC = 21600


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _repository_root(source_path: Path) -> Path:
    for candidate in source_path.parents:
        if (candidate / "pyproject.toml").is_file() or (
            (candidate / "catalog").is_dir()
            and (candidate / "scripts").is_dir()
        ):
            return candidate
    raise RuntimeError(f"cannot locate resource root above {source_path}")


def _archive_from_local_checkout(
    root: Path, destination: Path, environments: set[str] | None = None
) -> bool:
    checkout = root / "third_party/gym-anything"
    if not checkout.exists() or shutil.which("git") is None:
        return False
    probe = subprocess.run(
        ["git", "-C", str(checkout), "cat-file", "-e", f"{GYM_ANYTHING_COMMIT}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if probe.returncode:
        return False
    subprocess.run(
        [
            "git", "-C", str(checkout), "archive", "--format=tar.gz",
            f"--output={destination}", GYM_ANYTHING_COMMIT,
            *[
                f"benchmarks/cua_world/environments/{name}"
                for name in sorted(environments or ())
            ],
        ],
        check=True,
    )
    return True


def _download_archive(destination: Path) -> None:
    cache = Path.home() / ".cache/cua-speedrun/sources"
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"gym-anything-{GYM_ANYTHING_COMMIT}.tar.gz"
    if cached.is_file() and _sha256(cached) == ARCHIVE_SHA256:
        shutil.copy2(cached, destination)
        return
    for attempt in range(1, 4):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="gym-anything.", dir=cache
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            request = urllib.request.Request(
                ARCHIVE_URL, headers={"User-Agent": "cua-speedrun"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            actual = _sha256(temporary)
            if actual != ARCHIVE_SHA256:
                raise RuntimeError(
                    f"gym-anything archive checksum mismatch: {actual}"
                )
            temporary.replace(cached)
            shutil.copy2(cached, destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(attempt * 2)


def _safe_extract(
    archive: Path, destination: Path, environments: set[str] | None = None
) -> Path:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = []
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError(
                    f"gym-anything archive member escapes extraction root: {member.name}"
                )
            # Both git-archive and GitHub archives carry this path; the latter
            # adds one repository root component. No unselected files are used.
            _, separator, relative = member.name.partition(
                "benchmarks/cua_world/environments/"
            )
            if environments is None or (
                separator and relative.split("/", 1)[0] in environments
            ):
                members.append(member)
        bundle.extractall(destination, members=members)
    candidates = [
        path
        for path in destination.iterdir()
        if (path / "benchmarks/cua_world/environments").is_dir()
    ]
    if (destination / "benchmarks/cua_world/environments").is_dir():
        candidates.append(destination)
    if len(candidates) != 1:
        raise RuntimeError("gym-anything archive has an unexpected root layout")
    return candidates[0]


def _task_yaml(
    entry: dict[str, Any],
    *,
    benchmark_name: str,
    instruction_prefix: str = "",
    fetch_script_environments: set[str] | None = None,
    manual_download_environments: set[str] | None = None,
    required_devices: list[str] | None = None,
    native_image: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env_name = str(entry["env_name"])
    task_name = str(entry["task_name"])
    asset_setup = []
    if env_name in (fetch_script_environments or set()):
        asset_setup.append("fetch_script")
    if env_name in (manual_download_environments or set()):
        asset_setup.append("manual_download")
    return {
        "task_id": str(entry["id"]),
        "description": (
            f"{instruction_prefix}\n\n{entry['description']}"
            if instruction_prefix else str(entry["description"])
        ),
        "timeout_sec": AGENT_TIMEOUT_SEC,
        "env": {
            "kind": "gym-anything",
            "env_dir": (
                "${BENCHMARK_DIR}/gym-anything/benchmarks/"
                f"cua_world/environments/{env_name}"
            ),
            "task_id": task_name,
            "use_cache": True,
            "cache_level": "default",
            "max_steps": MAX_STEPS,
            "prepare_entrypoint": "cua_speedrun.envs.cua_world_runtime:prepare",
            **({"native_image": native_image} if native_image else {}),
            **({"required_devices": required_devices} if required_devices else {}),
        },
        "metadata": {
            "benchmark": benchmark_name,
            "source_environment": env_name,
            "source_task": task_name,
            "platform": str(entry["platform"]),
            "source_commit": GYM_ANYTHING_COMMIT,
            "asset_setup": asset_setup,
        },
    }


def _make_environment_paths_portable(env_dir: Path, env_name: str) -> None:
    """Make copied CUA-World paths relative to their environment root."""
    config_path = env_dir / "env.json"
    config = json.loads(config_path.read_text())
    prefix = f"benchmarks/cua_world/environments/{env_name}/"

    def rewrite(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str) and value.startswith(prefix):
            return value[len(prefix):]
        return value

    config_path.write_text(json.dumps(rewrite(config), indent=2) + "\n")


def _apply_setup_patch(env_dir: Path, patch_path: Path) -> dict[str, Any]:
    """Apply declared setup/export patches, never an upstream checkout edit."""
    patch = yaml.safe_load(patch_path.read_text())
    if patch.get("schema_version") != 1 or not patch.get("replacements"):
        raise ValueError(f"invalid setup patch: {patch_path}")
    changed = set()
    for replacement in patch["replacements"]:
        relative = Path(replacement["path"])
        target = (env_dir / relative).resolve()
        if (
            relative.is_absolute()
            or not target.is_relative_to(env_dir.resolve())
            or target.suffix != ".sh"
            or not (
                relative.parts[0] == "scripts"
                or (relative.parts[0] == "tasks" and target.name in {
                    "setup_task.sh", "export_result.sh",
                })
            )
        ):
            raise ValueError(f"patch target is not a setup/export script: {relative}")
        original = target.read_text()
        before, after = replacement["old"], replacement["new"]
        count = int(replacement.get("count", 1))
        if not before or count < 1 or original.count(before) != count:
            raise ValueError(f"setup patch does not exactly match {relative}")
        target.write_text(original.replace(before, after))
        changed.add(relative.as_posix())

    # Gym hashes hook commands, not the mounted scripts. A shell comment salts
    # only this environment's checkpoints, without changing hook exit status.
    setup_hash = _tree_sha256(env_dir)
    config_path = env_dir / "env.json"
    config = json.loads(config_path.read_text())
    config["hooks"]["pre_start"] += f"\n# cua-speedrun-setup-sha256={setup_hash}"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return {
        "patch_sha256": _sha256(patch_path),
        "setup_sha256": setup_hash,
        "effective_environment_sha256": _tree_sha256(env_dir),
        "changed_scripts": sorted(changed),
    }


def materialize(*, source_path: Path, out: Path) -> None:
    source_path = Path(source_path).resolve()
    source = yaml.safe_load(source_path.read_text()) or {}
    protocol = source.get("protocol") or {}
    instruction_prefix = str(protocol.get("instruction_prefix") or "").strip()
    entries = list(source.get("tasks") or ())
    expected_task_count = int(
        (source.get("selection") or {}).get("selected_task_count") or 0
    )
    if expected_task_count <= 0 or len(entries) != expected_task_count:
        raise ValueError(
            f"expected {expected_task_count} CUA-World-Long tasks, "
            f"found {len(entries)}"
        )
    asset_setup = dict(source.get("asset_setup") or {})
    fetch_script_environments = {
        str(name) for name in asset_setup.get("fetch_script_environments") or ()
    }
    manual_download_environments = {
        str(name) for name in asset_setup.get("manual_download_environments") or ()
    }
    selected_environments = {str(entry["env_name"]) for entry in entries}
    undeclared = (
        fetch_script_environments | manual_download_environments
    ) - selected_environments
    if undeclared:
        raise ValueError(
            "asset setup names unselected environments: "
            + ", ".join(sorted(undeclared))
        )

    root = _repository_root(source_path)
    setup_patches = dict(source.get("setup_patches") or {})
    if set(setup_patches) - selected_environments:
        raise ValueError("setup patches name unselected environments")
    declared_inputs = set((source.get("materializer") or {}).get("inputs") or ())
    for patch in setup_patches.values():
        if patch not in declared_inputs or not (root / patch).resolve().is_relative_to(root):
            raise ValueError(f"setup patch must be a declared repository input: {patch}")
    patch_provenance = {}
    with tempfile.TemporaryDirectory(prefix="cua-world-long.") as temporary_text:
        temporary = Path(temporary_text)
        archive = temporary / "gym-anything.tar.gz"
        local_archive = _archive_from_local_checkout(
            root, archive, selected_environments
        )
        if not local_archive:
            _download_archive(archive)
        extracted = _safe_extract(
            archive, temporary / "source", selected_environments
        )
        source_environments = (
            extracted / "benchmarks/cua_world/environments"
        )
        destination_environments = (
            out / "gym-anything/benchmarks/cua_world/environments"
        )

        manifest = {
            "name": str(source["name"]),
            "version": str(source["version"]),
            "source": ARCHIVE_URL,
            "source_commit": GYM_ANYTHING_COMMIT,
            "tasks": [],
        }
        for entry in entries:
            env_name = str(entry["env_name"])
            task_name = str(entry["task_name"])
            source_env = source_environments / env_name
            if (
                env_name in fetch_script_environments
                and not (source_env / "scripts/fetch_data.sh").is_file()
            ):
                raise FileNotFoundError(
                    f"missing declared asset fetch script for {env_name}"
                )
            if (
                env_name in manual_download_environments
                and not (source_env / "MANUAL_DOWNLOAD.md").is_file()
            ):
                raise FileNotFoundError(
                    f"missing declared manual-download guide for {env_name}"
                )
            checklist = source_env / "tasks" / task_name / "vlm_checklist.json"
            if not checklist.is_file():
                raise FileNotFoundError(
                    f"missing VLM checklist for {env_name}/{task_name}"
                )
            actual_hash = _tree_sha256(source_env)
            if actual_hash != entry["environment_sha256"]:
                raise RuntimeError(
                    f"source hash mismatch for {env_name}: {actual_hash}"
                )
            destination_env = destination_environments / env_name
            shutil.copytree(source_env, destination_env)
            _make_environment_paths_portable(destination_env, env_name)
            if env_name in setup_patches:
                patch_path = setup_patches[env_name]
                patch_provenance[env_name] = {
                    "patch": patch_path,
                    "source_environment_sha256": actual_hash,
                    **_apply_setup_patch(destination_env, root / patch_path),
                }

            # Benchmark policy belongs in the copied native task configuration.
            verifier = protocol.get("verifier")
            if isinstance(verifier, dict) or instruction_prefix:
                task_path = destination_env / "tasks" / task_name / "task.json"
                task = json.loads(task_path.read_text())
                if isinstance(verifier, dict):
                    task["success"] = verifier
                if instruction_prefix:
                    task["description"] = f"{instruction_prefix}\n\n{task['description']}"
                    # Gym prefers natural_language when an upstream task provides it.
                    nl = task.get("natural_language")
                    if isinstance(nl, dict) and nl.get("prompt"):
                        nl["prompt"] = f"{instruction_prefix}\n\n{nl['prompt']}"
                    elif isinstance(nl, str) and nl.strip():
                        task["natural_language"] = f"{instruction_prefix}\n\n{nl}"
                task_path.write_text(json.dumps(task, indent=2) + "\n")
            if source.get("host_runtime"):
                (destination_env / "host-runtime.json").write_text(
                    json.dumps(source["host_runtime"], indent=2) + "\n"
                )
            if env_name in patch_provenance:
                patch_provenance[env_name]["effective_environment_sha256"] = (
                    _tree_sha256(destination_env)
                )

            task_rel = f"tasks/{env_name}__{task_name}"
            task_dir = out / task_rel
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "task.yaml").write_text(
                yaml.safe_dump(
                    _task_yaml(
                        entry,
                        benchmark_name=str(source["name"]),
                        instruction_prefix=instruction_prefix,
                        fetch_script_environments=fetch_script_environments,
                        manual_download_environments=manual_download_environments,
                        required_devices=(source.get("required_devices") or {}).get(env_name),
                        native_image=(
                            image_definition(root / source["native_image"]["manifest"], source["native_image"]["name"])
                            if source.get("native_image") else None
                        ),
                    ),
                    sort_keys=False,
                    allow_unicode=True,
                )
            )
            manifest["tasks"].append(task_rel)

        (out / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
        )
        (out / "SOURCE.json").write_text(
            json.dumps(
                {
                    "repository": "https://github.com/cmu-l3/gym-anything",
                    "commit": GYM_ANYTHING_COMMIT,
                    "archive_sha256": ARCHIVE_SHA256,
                    "split": "long_horizon",
                    "task_count": len(entries),
                    **({"setup_patches": patch_provenance} if patch_provenance else {}),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


if __name__ == "__main__":
    raise SystemExit(
        "This is a benchmark materializer; load benchmarks/cua-world-long instead."
    )
