"""Public benchmark command backed by the existing scored submission path."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from cua_speedrun.benchmark_sources import benchmark_catalog_paths
from cua_speedrun.resources import resource_root
from cua_speedrun.specs import Benchmark
from cua_speedrun.submission import Submission

from .paths import InstallationPaths, add_home_argument


def register_benchmark_command(subparsers) -> None:
    parser = subparsers.add_parser("benchmark", help="evaluate an agent on a dataset")
    add_home_argument(parser)
    parser.add_argument("--dataset", help="benchmark name or folder")
    parser.add_argument("--agent", help="bundled agent name or folder")
    parser.add_argument("--json", action="store_true", help="emit JSON records; never prompt")
    parser.add_argument("--no-input", action="store_true", help="never prompt for missing agent or dataset")
    parser.add_argument("--gpu", help="override the agent's default GPU")
    parser.add_argument("--no-gpu", action="store_true", help="use CPU")
    parser.add_argument("--compute", choices=("modal", "local"), default="modal")
    parser.add_argument("--environment", choices=("modal", "modal-native", "local"))
    parser.add_argument("--parallel-evaluations", type=int, default=1)
    parser.add_argument("--no-preload", action="store_true", help="disable environment preloading")
    parser.add_argument("--env", action="append", default=[], metavar="NAME")
    parser.add_argument("--background", action="store_true", help="return the evaluation ID once launched")
    parser.add_argument("--name")
    parser.add_argument("--host", choices=("local", "modal"), default="local",
                        help="evaluation controller location (default: this machine)")
    parser.add_argument("--runner", choices=("local", "slurm"))
    parser.add_argument("--runner-template")
    parser.set_defaults(_operator_handler=run_benchmark)

    validate = subparsers.add_parser("validate", help="check an agent or benchmark without running it")
    validate.add_argument("--agent", type=Path, help="agent folder")
    validate.add_argument("--dataset", type=Path, help="benchmark folder")
    validate.set_defaults(_operator_handler=run_validate)


def validate_agent(path: Path) -> None:
    submission = Submission.load(path)
    for script in (submission.init_script, submission.agent_script):
        try:
            compile(script.read_bytes(), str(script), "exec")
        except SyntaxError as exc:
            raise ValueError(f"{script}:{exc.lineno}: {exc.msg}") from exc
    from cua_speedrun.service.templates_catalog import agent_metadata

    agent_metadata(path)


def dataset_path(name: str) -> Path:
    candidate = Path(name).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    requested, _, version = name.partition("@")
    # Accept osworld50 as well as the catalog spelling osworld-50.
    normalize = lambda value: value.lower().replace("-", "").replace("_", "")
    matches = []
    import yaml

    for path in benchmark_catalog_paths(resource_root()):
        definition = path / "benchmark-source.yaml"
        if not definition.is_file():
            definition = path / "manifest.yaml"
        metadata = yaml.safe_load(definition.read_text())
        if normalize(str(metadata["name"])) == normalize(requested) and (
            not version or str(metadata["version"]) == version
        ):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"dataset {name!r} is unknown or ambiguous; use cua-speedrun catalog")
    return matches[0]


def run_benchmark(args: argparse.Namespace) -> int:
    from .output import diagnostics, interactive
    if args.gpu and args.no_gpu:
        raise ValueError("--gpu and --no-gpu cannot be combined")
    if (not args.agent or not args.dataset) and interactive(args):
        from .wizard import run_benchmark_wizard

        return run_benchmark_wizard(args)
    if not args.agent or not args.dataset:
        raise ValueError("provide --agent and --dataset, or run cua-speedrun benchmark in a terminal")
    if interactive(args):
        from .wizard import run_benchmark_preparation

        return run_benchmark_preparation(args, exit_on_error=True)
    if args.host == "modal":
        from cua_speedrun.hosted.launch import run_hosted_benchmark
        return run_hosted_benchmark(args)
    from .dashboard_client import run_submit

    with diagnostics(getattr(args, "json", False)):
        submitted = _submission_args(args)
    return run_submit(submitted)


def _submission_args(args: argparse.Namespace) -> argparse.Namespace:
    from .dashboard_client import register_dashboard_client_commands
    from .setup import initialize, modal_credentials
    from cua_speedrun.service.templates_catalog import agent_metadata, template_dir
    from cua_speedrun.benchmark_preparation import prepare_environment, preparation_options, independent_environment_preparation
    from cua_speedrun.startup import phase

    paths = InstallationPaths.resolve(args.home)
    print("[prepare] Initializing installation", flush=True)
    initialize(paths)
    source = Path(args.agent).expanduser()
    bundled = not source.is_dir()
    if bundled:
        source = template_dir(args.agent)
        if source is None:
            raise ValueError(f"unknown agent {args.agent!r}; use a name from catalog or an agent folder")
    validate_agent(source)
    metadata = agent_metadata(source)
    if args.gpu and args.no_gpu:
        raise ValueError("--gpu and --no-gpu cannot be combined")
    if args.parallel_evaluations < 1:
        raise ValueError("parallel evaluations must be at least 1")
    if (args.runner or args.runner_template) and args.compute != "local":
        raise ValueError("--runner and --runner-template require --compute local")
    for name in args.env:
        from cua_speedrun.runtime_environment import normalize_environment_name

        normalize_environment_name(name)
        if not os.environ.get(name):
            raise ValueError(f"environment variable is missing: {name}")
    gpu = None if args.no_gpu else args.gpu or metadata.get("gpu")
    dataset = dataset_path(args.dataset)
    custom = Path(args.dataset).expanduser().is_dir()
    if custom and not (dataset / "manifest.yaml").is_file():
        raise ValueError("custom dataset folders need manifest.yaml; select bundled datasets by name")
    options = preparation_options(dataset)
    environment = args.environment or options.get("default_environment", "modal")
    if environment not in {"local", "modal", "modal-native"}:
        raise ValueError(f"invalid default_environment in dataset: {environment!r}")
    if args.compute == "modal" or environment.startswith("modal"):
        token_id, token_secret = modal_credentials()
        if not token_id or not token_secret:
            raise ValueError("Modal credentials are missing; run cua-speedrun setup")
        os.environ.update(MODAL_TOKEN_ID=token_id, MODAL_TOKEN_SECRET=token_secret)
    missing = [key for key in metadata.get("required_environment_variables", []) if not os.environ.get(key)]
    if missing:
        raise ValueError("set the agent credentials before running: " + ", ".join(missing))
    def load_tasks():
        with phase("tasks", "Load benchmark tasks"):
            return Benchmark.load(dataset)

    def prepare_desktop(folder):
        with phase("desktop", "Prepare desktop image"):
            prepare_environment(dataset, folder, environment, paths)

    independent = independent_environment_preparation(dataset, environment)
    if independent:
        with ThreadPoolExecutor(max_workers=2) as preparation:
            desktop = preparation.submit(prepare_desktop, dataset)
            benchmark = load_tasks()
            desktop.result()
    else:
        benchmark = load_tasks()
    from cua_speedrun.evaluator_environment import validate

    validate(benchmark, os.environ)
    if custom:
        from .custom_benchmarks import register_benchmark

        benchmark = register_benchmark(benchmark, paths)
    if args.compute == "local":
        from .dependencies import install_local_agent_python

        install_local_agent_python(paths)
    if not independent:
        prepare_desktop(benchmark.benchmark_dir)

    argv = ["submit", "--home", str(paths.home), "--benchmark", f"{benchmark.name}@{benchmark.version}",
            "--compute", args.compute, "--environment", environment,
            "--parallel-evaluations", str(args.parallel_evaluations)]
    argv += ["--template", args.agent] if bundled else ["--submission", str(source)]
    if gpu:
        argv += ["--gpu", gpu]
    elif args.no_gpu:
        argv += ["--no-gpu"]
    if args.no_preload:
        argv += ["--agent-mode", "shared-no-preload"]
    if args.background:
        argv += ["--background"]
    if getattr(args, "json", False):
        argv += ["--json"]
    names = environment_names(benchmark, metadata, args.env)
    argv += ["--no-auto-api-keys"]
    for name in sorted(names):
        if not os.environ.get(name):
            continue
        argv += ["--env", name]
    for option in ("name", "runner", "runner_template"):
        if getattr(args, option):
            argv += ["--" + option.replace("_", "-"), getattr(args, option)]
    parser = argparse.ArgumentParser()
    register_dashboard_client_commands(parser.add_subparsers(dest="command"))
    print("[prepare] Starting evaluation", flush=True)
    return parser.parse_args(argv)


def environment_names(benchmark, metadata, requested) -> set[str]:
    names = set(requested) | set(metadata.get("required_environment_variables", []))
    names.update(metadata.get("optional_environment_variables", []))
    from cua_speedrun.evaluator_environment import requirements

    if benchmark is not None:
        names.update(requirements(benchmark)[1])
    for task in benchmark.tasks if benchmark is not None else ():
        directory = task.env.get("env_dir")
        if not directory:
            continue
        runtime_path = Path(directory) / "host-runtime.json"
        if runtime_path.is_file():
            names.update(json.loads(runtime_path.read_text()).get("forward_env", []))
    for name in list(names):
        if name.endswith("_API_KEY_ENV") and os.environ.get(name):
            names.add(os.environ[name])
    return names


def run_validate(args: argparse.Namespace) -> int:
    if args.agent is None and args.dataset is None:
        raise ValueError("provide --agent or --dataset")
    if args.agent is not None:
        validate_agent(args.agent.expanduser().resolve())
        print("Agent: init.py and agent.py are valid Python")
    if args.dataset is not None:
        from .custom_benchmarks import validate_benchmark

        benchmark = Benchmark.load(args.dataset)
        validate_benchmark(benchmark)
        print(f"Dataset: {benchmark.name}@{benchmark.version}, {len(benchmark.tasks)} tasks")
    return 0
