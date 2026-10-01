"""Portable resources and generated/downloaded machine assets."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from cua_speedrun.resources import bundled_gym_anything_root, bundled_resource_root
from cua_speedrun.benchmark_sources import benchmark_catalog_paths

from .paths import InstallationPaths


_BENCHMARK_RESOURCES = tuple(
    path.name for path in benchmark_catalog_paths(bundled_resource_root())
)

_GYM_ENVIRONMENTS = ("preset_gnome_systemd",)
_OSWORLD_TIMEOUT_SEC = 11 * 60 * 60

_RENAMED_AGENTS = {
    "gemini_new": "gemini",
    "gpt54": "openai",
    "glm_cua": "autoglm_v",
    "qwen35vl": "qwen35",
}


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )


def install_resources(paths: InstallationPaths) -> None:
    source = bundled_resource_root()
    destination = paths.resource_root
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("catalog", "agents", "scripts"):
        _copy_tree(source / name, destination / name)
    for old_name, new_name in _RENAMED_AGENTS.items():
        old = destination / "agents" / old_name
        if not old.is_dir() or not (source / "agents" / new_name).is_dir():
            continue
        # Keep edited copies recoverable without listing duplicate templates.
        backup_root = destination.parent / "renamed-agents"
        backup_root.mkdir(parents=True, exist_ok=True)
        backup = Path(tempfile.mkdtemp(prefix=f"{old_name}-", dir=backup_root))
        shutil.move(str(old), str(backup / old_name))
        print(f"[ok] Renamed agent {old_name} to {new_name}; old copy: {backup / old_name}")
    benchmark_destination = destination / "benchmarks"
    benchmark_destination.mkdir(parents=True, exist_ok=True)
    for name in _BENCHMARK_RESOURCES:
        _copy_tree(source / "benchmarks" / name, benchmark_destination / name)
    for path in sorted((source / "benchmarks").iterdir()):
        if not path.is_file() or path.suffix not in {".json", ".LICENSE"}:
            continue
        shutil.copy2(path, benchmark_destination / path.name)

    gym_source = bundled_gym_anything_root()
    env_source = gym_source / "benchmarks/cua_world/environments"
    env_destination = paths.gym_anything_root / "benchmarks/cua_world/environments"
    for name in _GYM_ENVIRONMENTS:
        _copy_tree(env_source / name, env_destination / name)
    print(f"[ok] Maintained resources: {destination}")


def _full_osworld_ready(paths: InstallationPaths) -> bool:
    task_root = paths.resource_root / "benchmark-assets/osworld/tasks"
    environment_tasks = paths.resource_root / "benchmark-assets/osworld/environment/tasks"
    if not (paths.resource_root / "benchmark-assets/osworld/manifest.yaml").is_file():
        return False
    task_paths = [path for path in task_root.glob("*/task.yaml") if path.is_file()]
    if len(task_paths) != 369:
        return False
    try:
        shared_source = paths.resource_root / "scripts/osworld_shared"
        shared_installed = environment_tasks / "_shared"
        for name in ("osworld_setup.py", "osworld_verifier.py"):
            if (shared_source / name).read_bytes() != (
                shared_installed / name
            ).read_bytes():
                return False
        patch_root = paths.resource_root / "scripts/osworld_task_patches"
        expected_patches = list(patch_root.glob("*.json"))
        installed_patches = list(environment_tasks.glob("*/setup-patch.json"))
        if len(expected_patches) != len(installed_patches):
            return False
        for patch in expected_patches:
            installed = environment_tasks / patch.stem / "setup-patch.json"
            if patch.read_bytes() != installed.read_bytes():
                return False
        for task_path in task_paths:
            task = yaml.safe_load(task_path.read_text()) or {}
            if task.get("timeout_sec") != _OSWORLD_TIMEOUT_SEC:
                return False
            env_task_path = (
                paths.resource_root
                / "benchmark-assets/osworld/environment/tasks"
                / task_path.parent.name
                / "task.json"
            )
            env_task = json.loads(env_task_path.read_text())
            env_timeout = (env_task.get("init") or {}).get("timeout_sec")
            if env_timeout != _OSWORLD_TIMEOUT_SEC:
                return False
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return False
    return True


def install_full_osworld(paths: InstallationPaths) -> None:
    if _full_osworld_ready(paths):
        print("[ok] Full OSWorld task catalog (369 tasks)")
        return
    print("[prepare] Full OSWorld task catalog (369 bundled task definitions)")
    subprocess.run(
        [sys.executable, str(paths.resource_root / "scripts/import_osworld_eval.py")],
        cwd=paths.resource_root,
        env=os.environ.copy(),
        check=True,
    )


def _kvm_openable() -> bool:
    if sys.platform != "linux":
        return False
    descriptor: int | None = None
    try:
        descriptor = os.open("/dev/kvm", os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
        return True
    except OSError:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _image_prep_command(paths: InstallationPaths) -> list[str] | None:
    scripts = paths.resource_root / "scripts"
    native_commands = (
        "awk",
        "curl",
        "flock",
        "python3",
        "qemu-img",
        "sha256sum",
        "unzip",
        "apt-get",
        "dpkg-deb",
        "virt-customize",
    )
    if all(shutil.which(command) for command in native_commands):
        return [
            "bash",
            str(scripts / "prepare_osworld_qcow2.sh"),
            "--output",
            str(paths.osworld_image),
        ]
    if shutil.which("apptainer"):
        return [
            "bash",
            str(scripts / "prepare_osworld_qcow2_apptainer.sh"),
            "--output",
            str(paths.osworld_image),
        ]
    return None


def prepare_osworld_image(
    paths: InstallationPaths, *, force_attempt: bool, skip: bool
) -> str:
    sidecar = Path(str(paths.osworld_image) + ".provenance.json")
    if paths.osworld_image.is_file() and sidecar.is_file():
        print(f"[ok] OSWorld VM image: {paths.osworld_image}")
        return "ready"
    if skip:
        print("[skip] OSWorld VM image (--skip-osworld-image)")
        return "skipped"
    if not force_attempt and not _kvm_openable():
        print("[skip] OSWorld VM image (this host cannot open /dev/kvm)")
        return "hardware-unavailable"
    command = _image_prep_command(paths)
    if command is None:
        print(
            "[skip] OSWorld VM image (install native QEMU/libguestfs tools or "
            "Apptainer, then rerun install)"
        )
        return "runner-unavailable"
    print("[prepare] Pinned OSWorld VM image")
    subprocess.run(command, cwd=paths.resource_root, env=os.environ.copy(), check=True)
    return "ready"


__all__ = [
    "install_full_osworld",
    "install_resources",
    "prepare_osworld_image",
]
