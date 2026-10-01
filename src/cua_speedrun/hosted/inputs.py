"""Portable hosted inputs; credentials travel separately from file bundles."""

from __future__ import annotations

import io
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

import yaml

from cua_speedrun.benchmark_preparation import preparation_options
from cua_speedrun.commands.benchmark import dataset_path, validate_agent, environment_names
from cua_speedrun.commands.custom_benchmarks import validate_benchmark
from cua_speedrun.service.templates_catalog import agent_metadata, template_dir
from cua_speedrun.specs import Benchmark


def pack_inputs(args) -> tuple[dict, bytes, dict[str, str]]:
    source = Path(args.agent).expanduser()
    custom_agent = source.is_dir()
    if not custom_agent:
        source = template_dir(args.agent)
        if source is None:
            raise ValueError(f"unknown agent {args.agent!r}")
    validate_agent(source)
    metadata = agent_metadata(source)
    dataset = dataset_path(args.dataset)
    custom_dataset = Path(args.dataset).expanduser().is_dir()
    definition = dataset / ("benchmark-source.yaml" if (dataset / "benchmark-source.yaml").exists() else "manifest.yaml")
    data = yaml.safe_load(definition.read_text())
    environment = args.environment or preparation_options(dataset).get("default_environment", "modal")
    if args.compute != "modal" or environment == "local" or args.runner or args.runner_template:
        raise ValueError("a Modal-hosted controller requires Modal agent and desktop placement")
    if args.parallel_evaluations < 1:
        raise ValueError("parallel evaluations must be at least 1")
    benchmark = None
    if custom_dataset or definition.name == "manifest.yaml":
        benchmark = Benchmark.load(dataset)
        if custom_dataset:
            validate_benchmark(benchmark)
        from cua_speedrun.evaluator_environment import validate
        validate(benchmark, os.environ)
    names = environment_names(benchmark, metadata, args.env)
    evaluator = data.get("evaluator_environment") or {}
    required = set(args.env) | set(metadata.get("required_environment_variables", []))
    required.update(evaluator.get("required", []))
    names.update(required)
    names.update(evaluator.get("private", []))
    names.update((data.get("host_runtime") or {}).get("forward_env", []))
    preparation_names = set(preparation_options(dataset).get("forward_env", []))
    names.update(preparation_names)
    missing = [n for n in required if not os.environ.get(n)]
    if missing:
        raise ValueError("missing environment variables: " + ", ".join(sorted(missing)))
    from cua_speedrun.runtime_environment import normalize_environment_name
    for name in names:
        normalize_environment_name(name)
    credentials = {n: os.environ[n] for n in names if os.environ.get(n)}
    if "HF_TOKEN" in preparation_names and "HF_TOKEN" not in credentials:
        from huggingface_hub import get_token

        token = get_token()
        if token:
            credentials["HF_TOKEN"] = token

    argv = ["benchmark", "--host", "local", "--json", "--no-input", "--background",
            "--agent", "/work/inputs/agent" if custom_agent else args.agent,
            "--dataset", "/work/inputs/benchmark" if custom_dataset else args.dataset,
            "--compute", "modal", "--environment", environment,
            "--parallel-evaluations", str(args.parallel_evaluations)]
    for key in ("gpu", "name"):
        if getattr(args, key, None):
            argv += ["--" + key, getattr(args, key)]
    for key in ("no_gpu", "no_preload"):
        if getattr(args, key, False):
            argv.append("--" + key.replace("_", "-"))
    for name in args.env:
        argv += ["--env", name]

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        if custom_agent:
            for name in ("init.py", "agent.py", "agent.json"):
                if (source / name).is_file():
                    archive.write(source / name, "agent/" + name)
        if custom_dataset:
            for path in sorted(dataset.rglob("*")):
                relative = path.relative_to(dataset)
                if any(p in {".git", "__pycache__"} or p.startswith(".env") for p in relative.parts):
                    continue
                if path.is_symlink():
                    raise ValueError(f"benchmark must not contain symlinks: {relative}")
                if path.is_file():
                    archive.write(path, "benchmark/" + relative.as_posix())
    request = {"argv": argv, "name": args.name or source.name,
               "benchmark": str(data["name"]), "task_count": len(data["tasks"]),
               "environment": environment}
    return request, buffer.getvalue(), credentials


def unpack_inputs(content: bytes, destination: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if (path.is_absolute() or ".." in path.parts or not path.parts
                    or path.parts[0] not in {"agent", "benchmark"}
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError("invalid hosted input archive")
        archive.extractall(destination)
