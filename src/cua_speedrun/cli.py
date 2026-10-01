"""Command line interface.

    cua-speedrun run --submission agents/qwen3vl \
        --benchmark benchmarks/osworld-50 --backend gym-anything
    cua-speedrun score runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from functools import partial


from cua_speedrun.config import load_dotenv as _load_dotenv
from cua_speedrun.eval_algorithms import list_eval_algorithm_choices
from cua_speedrun.runtime_environment import normalize_environment_name


def _environment_variables_from_process(
    names: list[str] | None,
) -> dict[str, str]:
    """Select runtime variables for a direct CLI run.

    Dashboard workers pass the submitting user's encrypted run snapshot
    explicitly instead, so an operator's environment is never substituted.
    """
    selected = list(names or ())
    selected.extend(
        name
        for name in os.environ
        if name.endswith("_API_KEY")
        and not name.startswith(("CS_", "MODAL_"))
        and name not in selected
    )
    try:
        selected = [normalize_environment_name(name) for name in selected]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    missing = [name for name in selected if not os.environ.get(name)]
    if missing:
        raise SystemExit(
            "environment variable(s) are missing: "
            + ", ".join(sorted(missing))
        )
    return {name: os.environ[name] for name in selected}


def _benchmark_path(value: str) -> Path:
    """Resolve a benchmark name from the active installation or accept a path."""
    supplied = Path(value).expanduser()
    if supplied.exists() or supplied.is_absolute() or len(supplied.parts) > 1:
        return supplied

    from cua_speedrun.resources import resource_root

    installed = resource_root() / "benchmarks" / value
    return installed if installed.exists() else supplied


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args, json_errors=False, **kwargs):
        self.json_errors = json_errors
        super().__init__(*args, **kwargs)

    def error(self, message):
        if self.json_errors:
            from cua_speedrun.commands.output import emit
            emit({"type": "error", "error": message})
            self.exit(2)
        super().error(message)


def main(argv: list[str] | None = None) -> int:
    from cua_speedrun.commands import (
        dispatch_operator_command,
        register_operator_commands,
    )

    argv = list(sys.argv[1:] if argv is None else argv)
    json_requested = "--json" in argv
    parser = _Parser(
        json_errors=json_requested,
        prog="cua-speedrun",
        description=(
            "Install, run, and inspect scored computer-use agent evaluations."
        ),
        epilog="""common workflow:
  cua-speedrun setup
  cua-speedrun benchmark --dataset osworld-50 --agent qwen3vl
  cua-speedrun catalog
  cua-speedrun evaluations --active
  cua-speedrun status EVALUATION_ID
  cua-speedrun export EVALUATION_ID
  cua-speedrun cancel EVALUATION_ID

benchmark and submit create scored evaluations in the installed service. run is the
standalone practice path. Run "cua-speedrun help COMMAND" for command details.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", dest="global_json", action="store_true",
                        help="machine-readable output for setup, benchmark, submit, catalog, status, evaluations, and score")
    sub = parser.add_subparsers(dest="command", required=True,
                               parser_class=partial(_Parser, json_errors=json_requested))

    register_operator_commands(sub)

    run_p = sub.add_parser(
        "run", help="run an unscored standalone practice evaluation"
    )
    run_p.add_argument("--submission", required=True, help="folder with init.py and agent.py")
    run_p.add_argument("--benchmark", required=True, help="folder with manifest.yaml")
    run_p.add_argument("--backend", default="gym-anything-local",
                       help="gym-anything-local | gym-anything-qemu-native | "
                            "gym-anything-qemu-apptainer")
    run_p.add_argument("--remote", action="store_true",
                       help="run the design-correct distributed topology: only the "
                            "executor is local; env+gateway and agent each in their "
                            "own Modal sandbox")
    run_p.add_argument("--gpu", default=None,
                       help="GPU label. With --remote this selects the Modal "
                            "agent sandbox GPU (e.g. L40S); locally it is "
                            "recorded in the season key (use CUDA_VISIBLE_DEVICES "
                            "to choose local devices).")
    run_p.add_argument("--region", default=None,
                       help="pin both Modal sandboxes to this region (remote); "
                            "without it the agent sandbox is pinned to whatever "
                            "region the env sandbox actually landed in")
    run_p.add_argument("--fresh-init", action="store_true",
                       help="ignore the cached init snapshot and re-run init.py "
                            "(remote); use when init.py fetches something that "
                            "changed outside the submission files")
    run_p.add_argument("--agent-mode", default=None,
                       choices=list_eval_algorithm_choices(),
                       help="evaluation algorithm key or alias. The selected "
                            "versioned algorithm is baked into the season")
    run_p.add_argument(
        "--env",
        action="append",
        default=None,
        metavar="NAME",
        help="name of an existing process/.env variable to expose to init.py "
             "and agent.py; repeat for multiple variables. Non-platform "
             "*_API_KEY values are included automatically",
    )
    run_p.add_argument(
        "--out",
        default=None,
        help="output root directory (default: CUA_SPEEDRUN_HOME/runs after install)",
    )
    run_p.add_argument(
        "--agents-per-evaluation",
        "--concurrency",
        dest="agents_per_evaluation",
        type=int,
        default=4,
        help="agents served by each isolated compute replica "
             "(--concurrency is retained as a compatibility alias)",
    )
    run_p.add_argument(
        "--parallel-evaluations",
        type=int,
        default=1,
        help="number of isolated compute replicas; each runs the selected "
             "agents per evaluation",
    )
    run_p.add_argument(
        "--runner",
        choices=("local", "slurm"),
        default="local",
        help="model/agent compute-replica provider (default: local)",
    )
    run_p.add_argument(
        "--runner-template",
        metavar="NAME_OR_PATH",
        help="live operator template required by a templated runner",
    )
    run_p.add_argument("--runs-per-task", type=int, default=1)
    run_p.add_argument("--seed", type=int, default=0, help="base seed")
    run_p.add_argument("--task", action="append", default=None,
                       help="run only this task_id from the benchmark; may be "
                            "provided multiple times")

    score_p = sub.add_parser("score", help="score a finished run directory")
    score_p.add_argument("run_dir")
    score_p.add_argument(
        "--success-bar",
        type=float,
        default=None,
        help="research re-score override; default uses the run's frozen rules",
    )
    score_p.add_argument("--json", action="store_true", help="print JSON instead of a table")

    card_p = sub.add_parser("card", help="show the private result card for a run")
    card_p.add_argument("run_dir")

    pub_p = sub.add_parser("publish", help="publish a run's result to the leaderboard")
    pub_p.add_argument("run_dir")
    pub_p.add_argument("--name", required=True, help="entry name for the leaderboard")

    sub.add_parser("leaderboard", help="show the ranked leaderboard")

    help_topics = tuple(sub.choices)
    help_p = sub.add_parser(
        "help", help="show general help or help for one command"
    )
    help_p.add_argument(
        "topic", nargs="?", choices=help_topics, metavar="COMMAND"
    )

    args = parser.parse_args(argv)
    args.json = getattr(args, "json", False) or args.global_json
    if args.json and args.command not in {
        "setup", "benchmark", "submit", "catalog", "status", "evaluations", "score",
    }:
        parser.error(f"--json is not supported by {args.command}")
    if args.command == "help":
        (sub.choices[args.topic] if args.topic else parser).print_help()
        return 0
    operator_result = dispatch_operator_command(args)
    if operator_result is not None:
        return operator_result
    from cua_speedrun.commands.paths import InstallationPaths, configure_process

    installation = InstallationPaths.resolve()
    installation_ready = installation.install_record.is_file()
    if installation_ready:
        configure_process(installation)
    else:
        _load_dotenv()

    if args.command == "run":
        from cua_speedrun.parallelism import (
            ExecutionScale,
            validate_parallel_evaluations,
        )
        from cua_speedrun.scoring import format_table, score

        parallel_evaluations = validate_parallel_evaluations(
            args.parallel_evaluations
        )
        environment_variables = _environment_variables_from_process(args.env)
        selected_algorithm = (
            args.agent_mode
            or os.environ.get("CS_AGENT_MODE")
            or ("per-task" if args.remote else "shared")
        )
        if args.remote and (args.runner != "local" or args.runner_template):
            raise SystemExit(
                "--runner applies to local compute; --remote already selects "
                "the Modal compute provider"
            )
        scale = ExecutionScale.for_algorithm(
            selected_algorithm,
            parallel_evaluations,
            args.agents_per_evaluation,
        )
        out_root = (
            Path(args.out)
            if args.out
            else installation.runs if installation_ready else Path("runs")
        )
        benchmark_dir = _benchmark_path(args.benchmark)
        if args.remote:
            from cua_speedrun.remote.run import run_benchmark_remote

            run_dir = run_benchmark_remote(
                submission_dir=Path(args.submission),
                benchmark_dir=benchmark_dir,
                out_root=out_root,
                runs_per_task=args.runs_per_task,
                seed_base=args.seed,
                gpu=args.gpu,
                region=args.region,
                use_init_cache=not args.fresh_init,
                agent_mode=args.agent_mode,
                task_ids=args.task,
                environment_variables=environment_variables,
                execution_scale=scale,
            )
        else:
            from cua_speedrun.executor import run_benchmark

            run_dir = run_benchmark(
                submission_dir=Path(args.submission),
                benchmark_dir=benchmark_dir,
                backend_name=args.backend,
                out_root=out_root,
                runs_per_task=args.runs_per_task,
                seed_base=args.seed,
                task_ids=args.task,
                gpu=args.gpu,
                environment_variables=environment_variables,
                execution_scale=scale,
                eval_algorithm=selected_algorithm,
                compute_runner=args.runner,
                runner_template=args.runner_template,
            )
        print()
        print(format_table(score(run_dir)))
        print(f"\nrun directory: {run_dir}")
        return 0

    if args.command == "score":
        from cua_speedrun.scoring import ScoringRules, format_table, score, stored_rules

        rules = None
        if args.success_bar is not None:
            rules = stored_rules(Path(args.run_dir)) or ScoringRules()
            rules.success_bar = args.success_bar
        result = score(Path(args.run_dir), rules)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(format_table(result))
        return 0

    if args.command == "card":
        from cua_speedrun.leaderboard import build_card, format_card

        print(format_card(build_card(Path(args.run_dir))))
        return 0

    if args.command == "publish":
        from cua_speedrun.leaderboard import ENTRIES_FILE, format_card, publish

        card = publish(Path(args.run_dir), entry_name=args.name)
        print(format_card(card, entry_name=args.name))
        print(f"\npublished to {ENTRIES_FILE}")
        return 0

    if args.command == "leaderboard":
        from cua_speedrun.leaderboard import format_leaderboard, load_entries

        print(format_leaderboard(load_entries()))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
