"""CLI surface over either a local installation or a remote dashboard."""

from __future__ import annotations

import argparse
import getpass
import io
import json
import os
import re
import sys
import tempfile
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

from cua_speedrun.runtime_environment import normalize_environment_name
from cua_speedrun.eval_algorithms import list_eval_algorithm_choices

from .dashboard_api import (
    DashboardClient,
    add_connection_arguments,
    dashboard_client,
    has_saved_dashboard,
    register_dashboard_auth_commands,
)
from .local_evaluations import LocalEvaluations
from .paths import InstallationPaths
from cua_speedrun.hosted import evaluation_id, is_hosted


TERMINAL_STAGES = frozenset({
    "card_ready", "failed", "rejected", "held", "cancelled",
})


def _use_local_installation(args: argparse.Namespace) -> bool:
    if args.dashboard or os.environ.get("CUA_SPEEDRUN_DASHBOARD"):
        return False
    paths = InstallationPaths.resolve(args.home)
    if paths.install_record.is_file():
        return True
    return not has_saved_dashboard(paths)


def register_dashboard_client_commands(
    subparsers: argparse._SubParsersAction,
) -> None:
    register_dashboard_auth_commands(subparsers)

    catalog = subparsers.add_parser(
        "catalog", help="list tracks, benchmarks, and starter agents"
    )
    add_connection_arguments(catalog)
    catalog.add_argument("--json", action="store_true")
    catalog.set_defaults(_operator_handler=run_catalog)

    submit = subparsers.add_parser(
        "submit",
        help="run a scored evaluation from this installation",
        description="""Queue a scored evaluation using the installed catalog,
database, private seeds, worker, and result-card pipeline.""",
        epilog="""examples:
  cua-speedrun submit --template qwen35 --gpu L40S \\
    --benchmark osworld-50 --compute local --environment local

  cua-speedrun submit --init ./init.py --agent ./agent.py \\
    --benchmark osworld-50 \\
    --compute local --environment local --background

  cua-speedrun submit --template qwen35 --gpu L40S \\
    --benchmark osworld-50 --compute local --environment local \\
    --runner slurm --runner-template default --background

The foreground form follows progress until completion. Background mode prints
only the evaluation ID and returns; inspect it with "cua-speedrun status ID"
or "cua-speedrun evaluations --active". A compute runner creates one reusable
model/agent replica per --parallel-evaluations; its template is loaded directly
from the supplied path or the compute_runners/templates/<runner> directory.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_connection_arguments(submit)
    submit.add_argument("--json", action="store_true", help="emit JSON records without a terminal UI")
    source = submit.add_argument_group("agent source (choose exactly one)")
    source.add_argument(
        "--submission", type=Path, help="folder containing init.py and agent.py"
    )
    source.add_argument(
        "--template", help="name of an installed starter agent"
    )
    source.add_argument(
        "--init", dest="init_file", type=Path,
        help="path to init.py; use together with --agent",
    )
    source.add_argument(
        "--agent", dest="agent_file", type=Path,
        help="path to agent.py; use together with --init",
    )
    submit.add_argument("--name", help="evaluation name; defaults to the agent name")
    submit.add_argument("--track", help="track name; optional when the catalog has one track")
    submit.add_argument(
        "--benchmark",
        required=True,
        help="benchmark name, or name@version when more than one version exists",
    )
    submit.add_argument(
        "--compute", choices=("modal", "local"), default="modal",
        help="model/agent placement (default: modal)",
    )
    submit.add_argument(
        "--environment", choices=("modal", "modal-native", "local"), default="modal",
        help="environment-VM placement (default: modal); modal-native boots "
             "the OSWorld desktop directly on a Modal sandbox kernel, no KVM",
    )
    gpu_options = submit.add_mutually_exclusive_group()
    gpu_options.add_argument(
        "--gpu", help="model/agent GPU type (e.g. L40S); default: track setting (CPU)",
    )
    gpu_options.add_argument(
        "--no-gpu", action="store_false", dest="allocate_gpu",
        help="use CPU even if the track has a default GPU",
    )
    submit.set_defaults(allocate_gpu=True)
    submit.add_argument(
        "--agent-mode", choices=list_eval_algorithm_choices(),
        help="evaluation algorithm key or alias; default: track setting",
    )
    submit.add_argument(
        "--parallel-evaluations", type=int, default=1,
        help="number of isolated evaluations to run in parallel",
    )
    submit.add_argument(
        "--saved-env", action="append", default=[], metavar="NAME",
        help="select a saved account variable; repeat for multiple names",
    )
    submit.add_argument(
        "--env", action="append", default=[], metavar="NAME",
        help="forward this process variable only to this evaluation; repeatable",
    )
    submit.add_argument(
        "--no-auto-api-keys", action="store_true",
        help="do not automatically forward non-platform *_API_KEY variables",
    )
    submit.add_argument(
        "--background", "--detach", dest="detach", action="store_true",
        help=(
            "print only the evaluation ID and return while it runs in the background "
            "(--detach is a compatibility alias)"
        ),
    )
    submit.add_argument(
        "--runner",
        default=None,
        choices=("local", "slurm"),
        help=(
            "model/agent compute-replica provider; omitted, the executing "
            "installation's configured default applies"
        ),
    )
    submit.add_argument(
        "--runner-template",
        metavar="NAME_OR_PATH",
        help=(
            "live operator template for a templated runner; Slurm creates "
            "one scheduler job per parallel evaluation"
        ),
    )
    submit.add_argument(
        "--poll", type=float, default=2.0, help=argparse.SUPPRESS
    )
    submit.set_defaults(_operator_handler=run_submit)

    status = subparsers.add_parser(
        "status",
        help="follow an evaluation in a live terminal view",
        description=(
            "Show task completion, live task stages, step counts, and results. "
            "An active evaluation opens a live terminal view when stdin and "
            "stdout are interactive; piped output is a single snapshot."
        ),
    )
    add_connection_arguments(status)
    status.add_argument("run_id", type=evaluation_id)
    status_output = status.add_mutually_exclusive_group()
    status_output.add_argument("--json", action="store_true")
    status_output.add_argument(
        "--once", action="store_true",
        help="print one non-interactive snapshot and exit",
    )
    status.add_argument(
        "--poll", type=float, default=1.0, metavar="SECONDS",
        help="live refresh interval (default: 1.0)",
    )
    status.set_defaults(_operator_handler=run_status)

    evaluations = subparsers.add_parser(
        "evaluations",
        help="list evaluation progress, scores, and status",
        description="""Show recent evaluations from this installation. Live rows
show task progress and partial pass/fail counts; final rows also show the
scored success rate and ranking time.""",
        epilog="""examples:
  cua-speedrun evaluations
  cua-speedrun evaluations --active
  cua-speedrun evaluations --all
  cua-speedrun evaluations --json
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_connection_arguments(evaluations)
    scope = evaluations.add_mutually_exclusive_group()
    scope.add_argument(
        "--active", action="store_true", help="show only active evaluations"
    )
    scope.add_argument(
        "--all", action="store_true", help="show the complete evaluation history"
    )
    evaluations.add_argument("--json", action="store_true")
    evaluations.add_argument("--host", choices=("local", "modal"), default="local")
    evaluations.set_defaults(_operator_handler=run_evaluations)

    export = subparsers.add_parser(
        "export",
        help="export an evaluation's trajectories and recorded evidence",
        description=(
            "Create a portable ZIP containing every recorded trajectory frame, "
            "run log, agent/environment log, verdict, timing, event, and frozen "
            "run contract for a stopped evaluation."
        ),
        epilog="""examples:
  cua-speedrun export 13
  cua-speedrun export 13 --output collaborator-audit.zip
  cua-speedrun export 13 --dashboard https://speedrun.example.org
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_connection_arguments(export)
    export.add_argument("run_id", type=evaluation_id, help="evaluation ID")
    export.add_argument(
        "--output",
        type=Path,
        help="archive path (default: USER-NAME-ID.zip in the current directory)",
    )
    export.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output archive",
    )
    export.set_defaults(_operator_handler=run_export)

    cancel = subparsers.add_parser(
        "cancel", help="stop a queued or running evaluation"
    )
    add_connection_arguments(cancel)
    cancel.add_argument("run_id", type=evaluation_id)
    cancel.set_defaults(_operator_handler=run_cancel)

    resume = subparsers.add_parser(
        "resume",
        help="continue a failed or cancelled evaluation without rerunning results",
        description=(
            "Reuse the evaluation's frozen submission, track, benchmark, seeds, "
            "environment variables, and scale. Only task instances without "
            "complete evidence are scheduled again by default. Use "
            "--rerun-agent-failures to rerun only recorded agent errors, on the "
            "same evaluation ID."
        ),
        epilog="""examples:
  cua-speedrun resume 11
  cua-speedrun resume 11 --rerun-agent-failures
  cua-speedrun resume 11 --background
  cua-speedrun resume 11 --runner slurm --runner-template default
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_connection_arguments(resume)
    resume.add_argument("run_id", type=int)
    resume.add_argument(
        "--rerun-agent-failures",
        action="store_true",
        help="rerun only task instances recorded as agent errors",
    )
    resume.add_argument(
        "--background", "--detach", dest="detach", action="store_true",
        help="print the evaluation ID and return while it resumes",
    )
    resume.add_argument(
        "--runner",
        choices=("local", "slurm"),
        help="recorded local compute runner (normally detected automatically)",
    )
    resume.add_argument(
        "--runner-template",
        metavar="NAME_OR_PATH",
        help="recorded runner profile (normally detected automatically)",
    )
    resume.add_argument("--poll", type=float, default=2.0, help=argparse.SUPPRESS)
    resume.set_defaults(_operator_handler=run_resume)


def _remote_catalog(client: DashboardClient) -> dict[str, Any]:
    return {
        "tracks": client.json("GET", "/api/tracks"),
        "benchmarks": client.json("GET", "/api/benchmarks"),
        "templates": client.json("GET", "/api/templates"),
    }


def run_catalog(args: argparse.Namespace) -> int:
    if _use_local_installation(args):
        catalog = LocalEvaluations.open(args.home).catalog()
    else:
        _paths, client = dashboard_client(args, require_auth=False)
        catalog = _remote_catalog(client)
    if args.json:
        print(json.dumps(catalog, indent=2, sort_keys=True))
        return 0
    print("Tracks")
    for track in catalog["tracks"]:
        gpu = track.get("gpu") or "no GPU"
        print(
            f"  {track['name']:<24} {gpu:<8} "
            f"{track['eval_algorithm']} · "
            f"{track['agents_per_evaluation']} agents/evaluation"
        )
    print("\nBenchmarks")
    for benchmark in catalog["benchmarks"]:
        count = benchmark.get("task_count") or "?"
        print(
            f"  {benchmark['name']}@{benchmark['version']} · {count} tasks"
        )
    print("\nStarter agents")
    for template in catalog["templates"]:
        print(f"  {template['name']}")
    return 0


def _benchmark_id(
    client: DashboardClient, requested: str
) -> int:
    rows = client.json("GET", "/api/benchmarks")
    exact_name = requested
    version = None
    if "@" in requested:
        exact_name, version = requested.rsplit("@", 1)
    matches = [
        row for row in rows
        if row["name"] == exact_name
        and (version is None or str(row["version"]) == version)
    ]
    if not matches:
        known = ", ".join(
            f"{row['name']}@{row['version']}" for row in rows
        )
        raise ValueError(f"unknown benchmark {requested!r}; known: {known}")
    if len(matches) > 1:
        versions = ", ".join(str(row["version"]) for row in matches)
        raise ValueError(
            f"benchmark {requested!r} has multiple versions ({versions}); "
            "use name@version"
        )
    return int(matches[0]["id"])


def _submission_zip(path: Path) -> bytes:
    path = path.expanduser().resolve()
    return _submission_files_zip(path / "init.py", path / "agent.py")


def _submission_files_zip(init_file: Path, agent_file: Path) -> bytes:
    files = {
        "init.py": init_file.expanduser().resolve(),
        "agent.py": agent_file.expanduser().resolve(),
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise ValueError("submission file(s) not found: " + ", ".join(missing))
    if files["init.py"] == files["agent.py"]:
        raise ValueError("--init and --agent must refer to different files")
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, strict_timestamps=False,
    ) as archive:
        for archive_name, file_path in files.items():
            archive.write(file_path, archive_name)
    return output.getvalue()


def _evaluation_environment(args: argparse.Namespace) -> list[dict[str, str]]:
    names = list(args.env)
    if not args.no_auto_api_keys:
        names.extend(
            name for name in os.environ
            if name.endswith("_API_KEY")
            and not name.startswith(("CS_", "MODAL_"))
            and name not in names
        )
    normalized = list(dict.fromkeys(
        normalize_environment_name(name) for name in names
    ))
    missing = [name for name in normalized if not os.environ.get(name)]
    if missing:
        raise ValueError(
            "environment variable(s) are missing: "
            + ", ".join(sorted(missing))
        )
    return [{"name": name, "value": os.environ[name]} for name in normalized]


def _progress_line(payload: dict[str, Any]) -> str:
    progress = payload.get("progress") or {}
    if progress.get("total"):
        return (
            f"{payload['stage']} · {int(progress.get('finished') or 0)}/"
            f"{int(progress['total'])} tasks · "
            f"{int(progress.get('passed') or 0)} passed"
        )
    tasks = payload.get("tasks") or []
    finished = [task for task in tasks if task.get("stage") in {"done", "failed"}]
    passed = sum(task.get("passed") is True for task in finished)
    progress = f" · {len(finished)}/{len(tasks)} tasks · {passed} passed" if tasks else ""
    return f"{payload['stage']}{progress}"


def _fit(value: Any, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def _duration(value: Any) -> str:
    if value is None:
        return "—"
    seconds = max(0, int(round(float(value))))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _print_evaluations(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No evaluations found.")
        return
    id_width = max(5, max(len(str(row["run_id"])) for row in rows))
    header = (
        f"{'ID':>{id_width}}  {'STATUS':<11}  {'NAME':<18}  "
        f"{'TRACK / BENCHMARK':<42}  {'EXECUTION':<12}  "
        f"{'TASKS':>7}  {'RESULT':>9}  {'SCORE':>13}  {'COST':>10}  "
        f"{'TASK AVG':>9}  {'AGENT AVG':>9}  {'ENV AVG':>9}  "
        f"{'ELAPSED':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        progress = row.get("progress") or {}
        result = f"{progress.get('passed', 0)}P/{progress.get('failed', 0)}F"
        tasks = f"{progress.get('finished', 0)}/{progress.get('total', 0)}"
        score = row.get("score") or {}
        displayed_score = score.get("mean_score")
        if displayed_score is None:
            displayed_score = score.get("success_rate")
        if displayed_score is None:
            score_text = "—"
        else:
            score_text = f"{float(displayed_score):.0%}"
            if score.get("total_time_sec") is not None:
                score_text += f"/{_duration(score['total_time_sec'])}"
        per_task = row.get("per_task") or {}
        cost = row.get("cost_usd")
        cost_text = "—" if cost is None else f"${float(cost):,.2f}"
        stage = "complete" if row.get("stage") == "card_ready" else row.get("stage", "?")
        contract = f"{row.get('track', '?')} / {row.get('benchmark', '?')}"
        print(
            f"{str(row['run_id']):>{id_width}}  {_fit(stage, 11):<11}  "
            f"{_fit(row.get('name', '?'), 18):<18}  "
            f"{_fit(contract, 42):<42}  "
            f"{_fit(row.get('topology', '?'), 12):<12}  "
            f"{tasks:>7}  {result:>9}  {_fit(score_text, 13):>13}  "
            f"{cost_text:>10}  "
            f"{_duration(per_task.get('time_sec')):>9}  "
            f"{_duration(per_task.get('agent_time_sec')):>9}  "
            f"{_duration(per_task.get('env_time_sec')):>9}  "
            f"{_duration(row.get('elapsed_sec')):>8}"
        )


def _follow(client: DashboardClient, run_id: int, poll: float) -> int:
    previous = None
    while True:
        payload = client.json("GET", f"/api/runs/{run_id}")
        line = _progress_line(payload)
        if line != previous:
            print(line, flush=True)
            previous = line
        if payload["stage"] in TERMINAL_STAGES:
            if payload.get("error"):
                print(f"error: {payload['error']}")
            if payload.get("card_token"):
                print(
                    "result: "
                    f"{client.dashboard}/cards/{payload['card_token']}"
                )
            return 0 if payload["stage"] == "card_ready" else 1
        time.sleep(max(0.2, poll))


def _follow_local(service: LocalEvaluations, run_id: int, poll: float) -> int:
    previous = None
    while True:
        payload = service.status(run_id)
        line = _progress_line(payload)
        if line != previous:
            print(line, flush=True)
            previous = line
        if payload["stage"] in TERMINAL_STAGES:
            if payload.get("error"):
                print(f"error: {payload['error']}")
            if payload.get("run_dir"):
                print(f"result: {payload['run_dir']}")
            return 0 if payload["stage"] == "card_ready" else 1
        time.sleep(max(0.2, poll))


def run_submit(args: argparse.Namespace) -> int:
    from .output import diagnostics, emit
    with diagnostics(getattr(args, "json", False)):
        result = _submit(args)
    if not getattr(args, "json", False):
        return result
    emit(args._queued)
    if args.detach:
        return 0
    previous = None
    try:
        while True:
            with diagnostics():
                payload = args._fetch_status()
            serialized = json.dumps(payload, sort_keys=True)
            if serialized != previous:
                emit({"type": "status", **payload})
                previous = serialized
            if payload["stage"] in TERMINAL_STAGES:
                return 0 if payload["stage"] == "card_ready" else 1
            time.sleep(max(0.2, args.poll))
    except KeyboardInterrupt:
        emit({"type": "detached", "run_id": args._queued["run_id"]})
        return 130


def _submit(args: argparse.Namespace) -> int:
    has_direct_files = args.init_file is not None or args.agent_file is not None
    if (args.init_file is None) != (args.agent_file is None):
        raise ValueError("--init and --agent must be provided together")
    source_count = sum((
        args.template is not None,
        args.submission is not None,
        has_direct_files,
    ))
    if source_count != 1:
        raise ValueError(
            "choose exactly one agent source: --template NAME, --submission DIR, "
            "or --init FILE --agent FILE"
        )

    local = _use_local_installation(args)
    if (args.runner or args.runner_template) and args.compute != "local":
        raise ValueError(
            "a compute runner only applies when --compute local"
        )
    if not local and (args.runner is not None or args.runner_template):
        raise ValueError(
            "--runner and --runner-template are chosen by the installation "
            "that owns the database; they cannot be set through --dashboard"
        )
    service = LocalEvaluations.open(args.home) if local else None
    client = None
    if local:
        catalog = service.catalog()
    else:
        _paths, client = dashboard_client(args)
        catalog = _remote_catalog(client)
    tracks = {row["name"] for row in catalog["tracks"]}
    if args.track is None:
        if len(tracks) != 1:
            raise ValueError("choose --track from: " + ", ".join(sorted(tracks)))
        args.track = next(iter(tracks))
    if args.track not in tracks:
        raise ValueError(
            f"unknown track {args.track!r}; known: {', '.join(sorted(tracks))}"
        )
    if args.parallel_evaluations < 1:
        raise ValueError("parallel evaluations must be at least 1")

    if args.template:
        source_name = args.template
    elif args.submission:
        source_name = args.submission.expanduser().resolve().name
    else:
        source_name = args.agent_file.expanduser().resolve().parent.name
    evaluation_environment = _evaluation_environment(args)
    archive = None
    required_environment_names: list[str] = []
    if args.template:
        if local:
            from cua_speedrun.service.templates_catalog import (
                template_required_environment_variables,
                template_zip,
            )

            archive = template_zip(args.template)
            if archive is None:
                raise ValueError(f"unknown template {args.template!r}")
            required_environment_names = (
                template_required_environment_variables(args.template)
            )
    elif args.submission:
        archive = _submission_zip(args.submission)
    else:
        archive = _submission_files_zip(args.init_file, args.agent_file)

    if local:
        started = service.submit(
            submission_zip=archive,
            name=args.name or source_name,
            track=args.track,
            benchmark=args.benchmark,
            compute=args.compute,
            environment=args.environment,
            allocate_gpu=args.allocate_gpu,
            gpu=args.gpu,
            eval_algorithm=args.agent_mode,
            parallel_evaluations=args.parallel_evaluations,
            saved_environment_names=args.saved_env,
            evaluation_environment={
                item["name"]: item["value"] for item in evaluation_environment
            },
            required_environment_names=required_environment_names,
            runner=args.runner,
            runner_template=(
                str(args.runner_template) if args.runner_template else None
            ),
        )
        run_id = int(started["run_id"])
        try:
            # The runner selection is frozen on the run at submit; start()
            # reads it back, so a racing queue worker uses the same choice.
            launch = service.start(run_id)
        except Exception:
            service.cancel(run_id)
            raise
    else:
        benchmark_id = _benchmark_id(client, args.benchmark)
        data = {
            "name": args.name or source_name,
            "track": args.track,
            "benchmark_id": str(benchmark_id),
            "compute_placement": args.compute,
            "environment_placement": args.environment,
            "allocate_gpu": str(args.allocate_gpu).lower(),
            "parallel_evaluations": str(args.parallel_evaluations),
            "saved_environment_variables": json.dumps(
                [normalize_environment_name(name) for name in args.saved_env]
            ),
            "evaluation_environment_variables": json.dumps(
                evaluation_environment
            ),
        }
        files = None
        if args.gpu is not None:
            data["gpu"] = args.gpu
        if args.agent_mode is not None:
            data["eval_algorithm"] = args.agent_mode
        if args.template:
            data["template"] = args.template
        else:
            files = {"file": ("submission.zip", archive, "application/zip")}

        started = client.json(
            "POST",
            "/api/submissions",
            data=data,
            files=files,
            timeout=120,
        )
        run_id = int(started["run_id"])

    if getattr(args, "json", False):
        # Submission diagnostics were redirected; JSON uses the original channel.
        args._queued = {"type": "queued", "run_id": run_id,
                        "home": str(service.paths.home) if local else None,
                        "dashboard": None if local else client.dashboard}
        args._fetch_status = (lambda: service.status(run_id)) if local else (
            lambda: client.json("GET", f"/api/runs/{run_id}"))
        return 0
    if args.detach:
        print(run_id)
        return 0

    print(f"Evaluation {run_id} queued")
    if local:
        runner_detail = (
            f" · coordinator pid {launch.external_id}"
            if launch.external_id else ""
        )
        print(f"runner: {launch.runner}{runner_detail}")
        print(f"worker log: {launch.log_path}")
    else:
        print(f"dashboard: {client.dashboard}/runs/{run_id}")

    print(
        "contract: "
        f"{args.track} · {args.benchmark} · "
        f"{started['execution_topology']['key']} · "
        f"{started['parallelism']['parallel_evaluations']} parallel"
    )
    try:
        return (
            _follow_local(service, run_id, args.poll)
            if local
            else _follow(client, run_id, args.poll)
        )
    except KeyboardInterrupt:
        print(
            f"\nDetached; evaluation {run_id} is still running. "
            f"Stop it with `cua-speedrun cancel {run_id}`."
        )
        return 130


def run_status(args: argparse.Namespace) -> int:
    local = _use_local_installation(args)
    hosted = is_hosted(args.run_id)
    if hosted:
        from cua_speedrun.hosted.client import HostedEvaluations
        from .output import diagnostics
        with diagnostics():
            service = HostedEvaluations.open(args.home)
        def fetch() -> dict[str, Any]:
            return service.status(args.run_id)
        client = None
    elif local:
        service = LocalEvaluations.open(args.home)

        def fetch() -> dict[str, Any]:
            return service.status(args.run_id)

        client = None
    else:
        _paths, client = dashboard_client(args)

        def fetch() -> dict[str, Any]:
            return client.json("GET", f"/api/runs/{args.run_id}")

    payload = fetch()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.poll <= 0:
        raise ValueError("--poll must be greater than zero")

    from rich.console import Console

    from .status_tui import print_status_snapshot, run_status_tui

    def result_location(current: dict[str, Any]) -> str | None:
        if hosted:
            return f"cua-speedrun export {args.run_id}" if current.get("stage") in TERMINAL_STAGES else None
        if not current.get("card_token"):
            return None
        if local:
            return str(current.get("run_dir") or current["card_token"])
        return f"{client.dashboard}/cards/{current['card_token']}"

    console = Console()
    interactive = (
        not args.once
        and payload.get("stage") not in TERMINAL_STAGES
        and sys.stdin.isatty()
        and sys.stdout.isatty()
        and console.is_terminal
    )
    if not interactive:
        print_status_snapshot(
            payload,
            result_location=result_location(payload),
            console=Console(no_color=True) if args.once else console,
        )
        return 0

    try:
        payload, detached = run_status_tui(
            fetch,
            payload,
            poll_sec=args.poll,
            result_location=result_location,
            console=console,
        )
    except KeyboardInterrupt:
        detached = True
    if detached:
        console.print(
            f"Detached; evaluation {args.run_id} is still running. "
            f"Stop it with [bold]cua-speedrun cancel {args.run_id}[/bold]."
        )
    else:
        print_status_snapshot(
            payload,
            result_location=result_location(payload),
            console=console,
        )
    return 0


def run_evaluations(args: argparse.Namespace) -> int:
    limit = None if args.active or args.all else 20
    if args.host == "modal":
        from cua_speedrun.hosted.client import HostedEvaluations
        from .output import diagnostics
        with diagnostics():
            rows = HostedEvaluations.open(args.home).evaluations(active_only=args.active, limit=limit)
    elif _use_local_installation(args):
        rows = LocalEvaluations.open(args.home).evaluations(
            active_only=args.active,
            limit=limit,
        )
    else:
        _paths, client = dashboard_client(args)
        query = []
        if args.active:
            query.append("active=true")
        if limit is not None:
            query.append(f"limit={limit}")
        suffix = "?" + "&".join(query) if query else ""
        rows = client.json("GET", "/api/me/runs" + suffix)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        _print_evaluations(rows)
    return 0


def _filename_component(value: str | None, fallback: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value or "").encode(
        "ascii", "ignore"
    ).decode()
    safe_value = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_value).strip("-._")
    return safe_value or fallback


def _export_filename(
    name: str | None,
    run_id: int,
    *,
    user: str | None = None,
) -> str:
    if user is None:
        try:
            user = getpass.getuser()
        except OSError:
            pass
    safe_user = _filename_component(user, "user")
    safe_name = _filename_component(name, "evaluation")
    return f"{safe_user}-{safe_name}-{run_id}.zip"


def _export_output_path(
    args: argparse.Namespace, evaluation_name: str | None = None
) -> Path:
    supplied = args.output or Path(
        _export_filename(evaluation_name, args.run_id)
    )
    output = supplied.expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output = output.absolute()
    if output.is_dir():
        raise ValueError(f"export output is a directory: {output}")
    if output.exists() and not args.overwrite:
        raise ValueError(
            f"export output already exists: {output}; pass --overwrite to replace it"
        )
    if not output.parent.is_dir():
        raise ValueError(f"export output directory does not exist: {output.parent}")
    return output


def _temporary_output(output: Path) -> Path:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    return Path(temporary)


def run_export(args: argparse.Namespace) -> int:
    """Export locally or download from the installation that owns the run."""
    if is_hosted(args.run_id):
        from cua_speedrun.hosted.client import HostedEvaluations
        from .output import diagnostics
        with diagnostics():
            service = HostedEvaluations.open(args.home)
        payload = service.status(args.run_id)
        output = _export_output_path(args, (payload.get("submission") or {}).get("name"))
        temporary = _temporary_output(output)
        try:
            service.export(args.run_id, temporary)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Exported evaluation {args.run_id} to {output}")
        return 0
    local = _use_local_installation(args)
    service = LocalEvaluations.open(args.home) if local else None
    client = None
    evaluation_name = None
    if args.output is None:
        if local:
            payload = service.status(args.run_id)
        else:
            _paths, client = dashboard_client(args)
            payload = client.json("GET", f"/api/runs/{args.run_id}")
        evaluation_name = (payload.get("submission") or {}).get("name")

    output = _export_output_path(args, evaluation_name)
    temporary = _temporary_output(output)
    try:
        if local:
            from cua_speedrun.service.run_export import (
                evaluation_run_directory,
                write_run_archive,
            )

            run_dir = evaluation_run_directory(
                service.session_factory,
                args.run_id,
                service.user_id,
                installation_root=service.paths.home,
            )
            file_count = write_run_archive(run_dir, temporary)
        else:
            if client is None:
                _paths, client = dashboard_client(args)
            response = client.request(
                "GET",
                f"/api/runs/{args.run_id}/export",
                stream=True,
                timeout=600,
            )
            with temporary.open("wb") as archive:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        archive.write(chunk)
            file_count = None
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    detail = f" · {file_count} files" if file_count is not None else ""
    print(
        f"Exported evaluation {args.run_id} to {output}{detail} · "
        f"{output.stat().st_size / (1024 * 1024):.1f} MiB"
    )
    return 0


def run_cancel(args: argparse.Namespace) -> int:
    if is_hosted(args.run_id):
        from cua_speedrun.hosted.client import HostedEvaluations
        from .output import diagnostics
        with diagnostics():
            payload = HostedEvaluations.open(args.home).cancel(args.run_id)
    elif _use_local_installation(args):
        payload = LocalEvaluations.open(args.home).cancel(args.run_id)
    else:
        _paths, client = dashboard_client(args)
        payload = client.json("POST", f"/api/runs/{args.run_id}/cancel")
    print(f"Evaluation {args.run_id}: {payload['stage']} (stop requested)")
    return 0


def run_resume(args: argparse.Namespace) -> int:
    local = _use_local_installation(args)
    if not local and (args.runner is not None or args.runner_template):
        raise ValueError(
            "runner selection belongs to the installation owning the dashboard"
        )
    if local:
        service = LocalEvaluations.open(args.home)
        resumed = service.resume(
            args.run_id,
            rerun_agent_failures=args.rerun_agent_failures,
        )
        try:
            launch = service.start(
                args.run_id,
                runner=args.runner,
                runner_template=args.runner_template,
            )
        except Exception:
            service.cancel(args.run_id)
            raise
        client = None
    else:
        _paths, client = dashboard_client(args)
        resumed = client.json(
            "POST",
            f"/api/runs/{args.run_id}/resume",
            params={"rerun_agent_failures": args.rerun_agent_failures},
        )
        launch = None

    if args.detach:
        print(args.run_id)
        return 0

    count = resumed.get("task_count")
    if args.rerun_agent_failures:
        detail = f"{count} agent-failure tasks"
    else:
        detail = "all unfinished tasks" if count is None else f"{count} unfinished tasks"
    print(f"Evaluation {args.run_id} resumed · {detail}")
    if launch is not None:
        print(f"runner: {launch.runner} · worker log: {launch.log_path}")
    else:
        print(f"dashboard: {client.dashboard}/runs/{args.run_id}")
    try:
        return (
            _follow_local(service, args.run_id, args.poll)
            if local
            else _follow(client, args.run_id, args.poll)
        )
    except KeyboardInterrupt:
        print(
            f"\nDetached; evaluation {args.run_id} is still running. "
            f"Stop it with `cua-speedrun cancel {args.run_id}`."
        )
        return 130


__all__ = ["register_dashboard_client_commands"]
