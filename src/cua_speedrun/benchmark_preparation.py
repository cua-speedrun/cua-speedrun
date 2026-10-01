"""Run benchmark-declared preparation before an evaluation starts."""

from __future__ import annotations

import fcntl
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from string import Formatter

import yaml


def _definition(folder: Path) -> dict:
    source = folder / "benchmark-source.yaml"
    if not source.is_file():
        source = folder / "manifest.yaml"
    return yaml.safe_load(source.read_text()) or {}


def preparation_options(folder: Path) -> dict:
    options = _definition(folder).get("prepare") or {}
    if not isinstance(options, dict):
        raise ValueError(f"{folder}: prepare must be a mapping")
    return options


def independent_environment_preparation(folder: Path, environment: str) -> bool:
    """Whether image preparation can precede materializing the task folder."""
    steps = preparation_options(folder).get(environment, [])
    image = _definition(folder).get("native_image")
    published = environment == "modal-native" and isinstance(image, dict) and bool(image)
    return bool(steps or published) and all(step.get("independent", False) for step in steps) and all(
        field != "benchmark"
        for step in steps
        for argument in step.get("args", [])
        for _, field, _, _ in Formatter().parse(argument)
    )


def _run_steps(steps: list, root: Path, values: dict[str, str], cache: Path, *, parallel: bool = False) -> None:
    if not isinstance(steps, list):
        raise ValueError("preparation steps must be a list")
    if parallel and len(steps) > 1:
        with ThreadPoolExecutor(max_workers=min(len(steps), 4)) as pool:
            list(pool.map(lambda step: _run_steps([step], root, values, cache), steps))
        return
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("script"), str):
            raise ValueError("each preparation step needs a script")
        script = (root / step["script"]).resolve()
        if not script.is_relative_to(root.resolve()) or not script.is_file():
            raise ValueError(f"preparation script must be inside the benchmark resource tree: {script}")
        if script.suffix not in {".py", ".sh"}:
            raise ValueError("preparation scripts must be Python or shell files")
        args = step.get("args", [])
        outputs = step.get("outputs", [])
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise ValueError("preparation args must be a list of strings")
        if not isinstance(outputs, list) or not all(isinstance(item, str) for item in outputs):
            raise ValueError("preparation outputs must be a list of paths")
        output_paths = [(root / item).resolve() for item in outputs]
        if any(not path.is_relative_to(root.resolve()) for path in output_paths):
            raise ValueError("preparation outputs must be inside the resource tree")
        command = [sys.executable if script.suffix == ".py" else "bash", str(script)]
        command.extend(item.format_map(values) for item in args)
        key = hashlib.sha256(json.dumps(command).encode()).hexdigest()
        cache.mkdir(parents=True, exist_ok=True)
        with (cache / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if output_paths and all(path.exists() for path in output_paths):
                continue
            print(f"[prepare] {script.name}", flush=True)
            subprocess.run(command, cwd=root, check=True)
            if output_paths and not all(path.exists() for path in output_paths):
                raise RuntimeError(f"{script.name} did not produce its declared outputs")


def prepare_data(folder: Path, root: Path) -> None:
    from cua_speedrun.benchmark_sources import _cache_root

    _run_steps(preparation_options(folder).get("data", []), root,
               {"benchmark": str(folder), "resources": str(root)}, _cache_root() / "preparation")


def prepare_environment(source: Path, benchmark: Path, environment: str, paths) -> None:
    from cua_speedrun.benchmark_sources import _repository_root

    definition = source / "benchmark-source.yaml"
    root = _repository_root(definition) if definition.is_file() else source
    options = preparation_options(source)
    published = _definition(source).get("native_image")
    if environment == "modal-native" and isinstance(published, dict) and published:
        import modal
        from cua_speedrun.remote.registry_images import image_definition

        manifest = (root / published["manifest"]).resolve()
        if not manifest.is_relative_to(root.resolve()):
            raise ValueError("desktop image manifest must be inside the resource tree")
        image = image_definition(manifest, published["name"])
        app = modal.App.lookup("cua-speedrun-native-desktops", create_if_missing=True)
        print("[prepare] Importing prebuilt desktop", flush=True)
        runtime = _definition(source).get("host_runtime") or {}
        with ThreadPoolExecutor(max_workers=1) as pool:
            controller = None
            if runtime.get("native_runner") == "gym-anything":
                from cua_speedrun.remote.modal_env import prepare_native_controller

                controller = pool.submit(prepare_native_controller, runtime)
            with modal.enable_output():
                modal.Image.from_registry(image["reference"]).build(app)
            if controller is not None:
                controller.result()
    if environment == "local":
        from cua_speedrun.commands.dependencies import install_runtime_dependencies

        install_runtime_dependencies(paths)
    _run_steps(options.get(environment, []), root, {
        "benchmark": str(benchmark),
        "resources": str(root),
        "cache": str(paths.cache),
        "osworld_image": str(paths.osworld_image),
    }, paths.cache / "preparation", parallel=environment in options.get("parallel", []))
