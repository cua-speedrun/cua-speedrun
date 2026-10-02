"""Remote executor: run a benchmark on the distributed topology.

Only this process (the executor) is local. init.py runs once in a Modal
sandbox and is snapshotted; then per task instance, the environment plus
gateway run in one Modal sandbox and the agent (cloned from the init
snapshot) runs in another. Submission code has outbound internet access and
decides whether to run a local model, call external APIs, use both, or use
neither. The executor mirrors the local outputs (per-task run logs +
result.json) so scoring, cards, and the leaderboard work unchanged.

Progress is emitted as structured events (see cua_speedrun.events): the
console sink renders the familiar CLI lines, events.jsonl lands in the run
directory, and the platform worker attaches a database sink for live
status. Timing truth stays in the gateway's run logs; events only report.

Per-task order of operations, which is what keeps the timing honest. Isolated
tasks use one of two placements, chosen by whether the submission needs a GPU:

- CPU (env-first): create the env sandbox, then spawn + warm the agent
  sandbox pinned to the env's actual region. Env boot and warmup overlap.
- GPU (gpu-first): acquire the GPU first, then create the env in its region.
  Env boot overlaps the submission's warmup.

A shared agent pool (shared-bounded) starts each replica's agent first and
creates every environment it serves in that agent's region.

In both placements everything above is untimed; only when both sides are
ready does the executor arm the clock and deliver the go-signal that
reveals the task. The agent-to-gateway hop stays inside one region either way
(explicit --region pins strictly). Any external model-API latency remains on
the timed path.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import secrets
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence
from urllib.parse import urlparse

import cua_speedrun
from cua_speedrun.eval_algorithms import (
    resolve_eval_algorithm,
)
from cua_speedrun.evaluation_runtime import (
    ComputeInfrastructureError,
    InstanceInfrastructureError,
    MODAL_RUNTIME_CAPABILITIES,
)
from cua_speedrun.events import ConsoleSink, EventBus, JsonlSink, Sink
from cua_speedrun.gateway import VERIFIER_TIMEOUT_SEC, monitor_gateway_steps
from cua_speedrun.parallelism import (
    DEFAULT_PARALLEL_EVALUATIONS,
    ExecutionScale,
)
from cua_speedrun.remote.control import GatewayControl
from cua_speedrun.remote import snapshot_cache
from cua_speedrun.remote.modal_agent import (
    AgentWarmupError,
    agent_image_from_cache,
    exec_agent,
    grant_gateway_access,
    init_and_snapshot,
    pull_agent_logs,
    spawn_agent_sandbox,
)
from cua_speedrun.remote.modal_env import (
    create_env_sandbox,
    pull_env_artifacts,
    read_region,
    tail_sandbox_file,
    wait_env_healthy,
)
from cua_speedrun.remote.modal_native_env import create_modal_native_env_sandbox
from cua_speedrun.runlog import load_runlog, summarize
from cua_speedrun.runplan import (
    SEED_PRACTICE_V1,
    RunPlan,
    benchmark_contract,
    build_run_plan,
    harness_contract,
)
from cua_speedrun.specs import Benchmark
from cua_speedrun.submission import Submission
from cua_speedrun.task_jobs import build_task_seed_jobs, select_task_seed_jobs

BACKEND = "modal-remote"
# Backends this executor can run. Both live on Modal; they differ only in how
# the environment sandbox is created (nested QEMU via gym-anything versus the
# OSWorld rootfs booted directly on the sandbox kernel).
SUPPORTED_BACKENDS = ("modal-remote", "modal-native")

# Untimed budgets cover sandbox queueing, boot, and submission warmup.
CPU_WARMUP_TIMEOUT_SEC = 1200
GPU_WARMUP_TIMEOUT_SEC = 3600


def _record_agent_outcome(
    control: GatewayControl,
    returncode: int,
    events: Any,
    task_key: str,
) -> None:
    """Turn an agent exit without ``done`` into a scored submission failure."""
    status = control.status()
    infrastructure_error = status.get("infrastructure_error")
    if infrastructure_error:
        raise InstanceInfrastructureError(str(infrastructure_error))
    if status.get("finished") or status.get("continuation_pending"):
        return
    error = f"agent.py exited with code {returncode} before finishing the task"
    if control.fail_agent(error):
        events.emit(
            "agent_failed",
            task_key=task_key,
            returncode=returncode,
            error=error,
        )


def _run_agent_episodes(
    control: GatewayControl,
    agent: Any,
    run_url: str,
    instruction: str,
    *,
    task_id: str,
    timeout_sec: int,
    events: Any,
    task_key: str,
) -> tuple[int, str, str]:
    """Run every evaluator-requested agent episode on one environment.

    This helper is entered only when the gateway explicitly advertises a
    multi-episode benchmark. Ordinary tasks retain their existing one-shot
    execution path.
    """
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    episode = int(control.info().get("agent_episode") or 1)
    while True:
        code, agent_out, agent_err = exec_agent(
            agent,
            run_url,
            instruction,
            task_id=f"{task_id}_episode_{episode}",
            timeout_sec=timeout_sec,
            on_event=lambda kind, **payload: events.emit(
                kind,
                task_key=task_key,
                agent_episode=episode,
                **payload,
            ),
        )
        stdout_parts.append(agent_out)
        stderr_parts.append(agent_err)
        events.emit(
            "agent_exit",
            task_key=task_key,
            code=code,
            agent_episode=episode,
        )
        _record_agent_outcome(control, code, events, task_key)
        status = control.status()
        if not status.get("continuation_pending"):
            return code, "".join(stdout_parts), "".join(stderr_parts)
        continuation = control.continue_agent()
        instruction = str(continuation["instruction"])
        episode = int(continuation["agent_episode"])


def run_benchmark_remote(
    submission_dir: Path,
    benchmark_dir: Path,
    out_root: Path,
    concurrency: int | None = None,
    runs_per_task: int = 1,
    seed_base: int = 0,
    gpu: str | None = None,
    region: str | None = None,
    extra_pip: list[str] | None = None,
    run_id: str | None = None,
    extra_sinks: Sequence[Sink] = (),
    use_init_cache: bool = True,
    agent_mode: str | None = None,
    cache_namespace: str | None = None,
    task_ids: list[str] | None = None,
    run_plan: RunPlan | dict[str, Any] | None = None,
    task_seeds: Mapping[str, Sequence[int]] | None = None,
    environment_variables: Mapping[str, str] | None = None,
    execution_scale: ExecutionScale | Mapping[str, Any] | None = None,
    parallel_evaluations: int | None = None,
    selected_task_keys: Collection[str] | None = None,
) -> Path:
    submission = Submission.load(submission_dir)
    benchmark = Benchmark.load(Path(benchmark_dir))

    # A dashboard run arrives with the exact contract captured at submission
    # time. The CLI creates the same object locally, using public fixed seeds.
    # In either case all execution values below come from the plan, never from
    # mutable track rows or environment switches once the run has started.
    supplied_plan = run_plan is not None
    supplied_scale = (
        ExecutionScale.from_value(execution_scale)
        if execution_scale is not None
        else None
    )
    if run_plan is None:
        if seed_base < 0 or seed_base + runs_per_task - 1 > 999:
            raise ValueError(
                "CLI runs use the public practice seed band 0..999; choose "
                "a seed/runs-per-task range inside that band"
            )
        legacy_mode = (
            agent_mode or os.environ.get("CS_AGENT_MODE") or "per-task"
        ).lower()
        algorithm_key = resolve_eval_algorithm(legacy_mode).key
        agents_per_evaluation = (
            supplied_scale.agents_per_evaluation
            if supplied_scale is not None
            else int(concurrency if concurrency is not None else 2)
        )
        run_plan = build_run_plan(
            track_name="cli",
            benchmark_dir=Path(benchmark_dir),
            gpu=gpu,
            network_policy="host-network",
            eval_algorithm=algorithm_key,
            agents_per_evaluation=agents_per_evaluation,
            runs_per_task=runs_per_task,
            seed_policy=SEED_PRACTICE_V1,
            task_ids=task_ids,
            server_config={
                "extra_pip": list(extra_pip or ()),
                "region": region,
            },
        )
    elif isinstance(run_plan, dict):
        run_plan = RunPlan.from_dict(run_plan)
    plan = run_plan
    algorithm = resolve_eval_algorithm(plan.eval_algorithm)
    if supplied_scale is not None:
        scale = supplied_scale.validate_for_plan(plan)
        if (
            parallel_evaluations is not None
            and int(parallel_evaluations) != scale.parallel_evaluations
        ):
            raise ValueError(
                "parallel_evaluations conflicts with execution_scale"
            )
        if (
            concurrency is not None
            and int(concurrency) != scale.agents_per_evaluation
        ):
            raise ValueError("concurrency conflicts with execution_scale")
    else:
        scale = ExecutionScale.for_plan(
            plan,
            (
                DEFAULT_PARALLEL_EVALUATIONS
                if parallel_evaluations is None
                else int(parallel_evaluations)
            ),
        )
    scale_payload = scale.to_dict()
    planned_extra_pip = list(plan.server_config.get("extra_pip", ()))
    planned_region = plan.server_config.get("region")

    if supplied_plan:
        current_benchmark = benchmark_contract(Path(benchmark_dir))
        for field in ("name", "version", "content_hash"):
            if current_benchmark.get(field) != plan.benchmark.get(field):
                raise ValueError(
                    f"benchmark {field} changed after this run was planned: "
                    f"planned {plan.benchmark.get(field)!r}, "
                    f"current {current_benchmark.get(field)!r}"
                )
        current_harness = harness_contract(plan.eval_algorithm)
        if selected_task_keys is None and current_harness != plan.to_dict()["harness"]:
            raise ValueError(
                "measurement harness changed after this run was planned; "
                "submit again to create a new season"
            )
        if plan.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"remote executor cannot execute backend {plan.backend!r}"
            )
        if (
            agent_mode is not None
            and resolve_eval_algorithm(agent_mode).key != algorithm.key
        ):
            raise ValueError(
                f"agent_mode {agent_mode!r} conflicts with frozen algorithm "
                f"{plan.eval_algorithm!r}"
            )
        if gpu is not None and gpu != plan.gpu:
            raise ValueError(
                f"gpu {gpu!r} conflicts with frozen gpu {plan.gpu!r}"
            )
        if extra_pip is not None and list(extra_pip) != planned_extra_pip:
            raise ValueError("extra_pip conflicts with the frozen server config")
        if region is not None and region != planned_region:
            raise ValueError("region conflicts with the frozen server config")
        if task_ids is not None and list(task_ids) != list(plan.benchmark["task_ids"]):
            raise ValueError("task_ids conflict with the frozen run plan")

    tasks_by_id = {task.task_id: task for task in benchmark.tasks}
    planned_task_ids = list(plan.benchmark["task_ids"])
    missing = sorted(set(planned_task_ids) - set(tasks_by_id))
    if missing:
        raise ValueError(f"run plan names unknown benchmark tasks: {missing}")
    benchmark.tasks = [tasks_by_id[task_id] for task_id in planned_task_ids]
    concurrency = scale.agents_per_evaluation
    runs_per_task = plan.runs_per_task
    gpu = plan.gpu
    extra_pip = planned_extra_pip
    region = planned_region
    server_environment = plan.server_environment()
    private_environment = {
        str(name): str(value)
        for name, value in (environment_variables or {}).items()
    }
    # Maintainer-owned server configuration is frozen in the run plan and wins
    # over a same-named account or per-evaluation variable.
    runtime_environment = {**private_environment, **server_environment}
    from cua_speedrun.evaluator_environment import agent_environment, validate

    validate(benchmark, runtime_environment)
    evaluator_runtime_environment = dict(runtime_environment)
    runtime_environment = agent_environment(benchmark, runtime_environment)
    private_environment = agent_environment(benchmark, private_environment)
    # Runtime variables can influence init.py and therefore the filesystem
    # snapshot, but values must never appear in the on-disk cache index. Hash
    # only their values for cache identity and pass plaintext solely to Modal.
    cache_environment = {
        **server_environment,
        **{
            f"private:{name}": "sha256:"
            + hashlib.sha256(value.encode()).hexdigest()
            for name, value in private_environment.items()
        },
    }
    network_policy = plan.network_policy
    # Retained only so an already-frozen legacy plan can still execute exactly
    # as recorded. New plans always use host-network and an empty allowlist.
    api_domains = plan.api_domain_allowlist

    if run_id is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{stamp}_{secrets.token_hex(3)}"
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=selected_task_keys is not None)

    # Validate and persist the exact job list before any expensive sandbox is
    # created. A crashed run still leaves the contract it attempted, and an
    # invalid scored-seed map fails before consuming Modal resources.
    all_jobs = build_task_seed_jobs(
        benchmark.tasks,
        runs_per_task,
        seed_base,
        task_seeds=task_seeds,
    )
    resolved_task_seeds: dict[str, list[int]] = {}
    for task, seed in all_jobs:
        resolved_task_seeds.setdefault(task.task_id, []).append(seed)
    jobs = select_task_seed_jobs(all_jobs, selected_task_keys)
    (run_dir / "run_plan.json").write_text(json.dumps({
        "run_plan_hash": plan.contract_hash,
        "run_plan": plan.to_dict(),
        "parallelism": scale_payload,
        "task_seeds": resolved_task_seeds,
    }, indent=2, sort_keys=True))

    environment_backend = str(
        plan.resolved_execution_topology().get("environment_backend")
        or "gym-anything-modal"
    )

    events = EventBus([ConsoleSink(), JsonlSink(run_dir / "events.jsonl"), *extra_sinks])
    events.emit(
        "run_started",
        run_id=run_id,
        benchmark=benchmark.name,
        version=benchmark.version,
        n_tasks=len(jobs),
        runs_per_task=runs_per_task,
        backend=plan.backend,
        eval_algorithm=plan.eval_algorithm,
        run_plan_hash=plan.contract_hash,
        env_pool_size=scale.environment_pool_size_per_evaluation,
        parallelism=scale_payload,
    )

    # Init once, untimed, then snapshot. This is the agent machine's warm
    # state. Task sandboxes also have outbound internet, so submission code
    # can combine local services and external APIs without a framework mode.
    #
    # The snapshot is reused across runs: byte-identical inputs (submission
    # files, harness source, gpu, extra pip) hit the cache and skip the
    # whole prologue; the cached image is probe-validated before use, so an
    # expired snapshot falls back to a fresh init instead of failing.
    init_phases: dict[str, Any] = {}
    init_duration = 0.0

    def prepare_agent_image():
        nonlocal init_duration
        agent_image = None
        # cache_namespace separates snapshot ids that live in different Modal
        # workspaces (runs execute under the submitting user's credentials):
        # image ids are workspace-scoped, so without it users would evict each
        # other's cache entries on every alternating run.
        agent_runtime = plan.resolved_agent_runtime()
        cache_key = snapshot_cache.compute_key(
            submission.submission_dir,
            gpu,
            extra_pip,
            cache_environment,
            agent_runtime,
        )
        if cache_namespace:
            cache_key += f"|ns={cache_namespace}"
        if use_init_cache:
            cached = snapshot_cache.lookup(out_root, cache_key)
            if cached is not None:
                agent_image = agent_image_from_cache(
                    cached["image_id"], gpu, runtime_environment
                )
                if agent_image is None:
                    events.emit("init_cache_invalid", image_id=cached["image_id"])
                else:
                    events.emit("init_cache_hit", image_id=cached["image_id"],
                                from_run=cached.get("run_id"),
                                created_at=cached.get("created_at"))
                    init_phases["cache"] = "hit"
                    init_phases["snapshot_image_id"] = cached["image_id"]

        if agent_image is None:
            events.emit("init_started")
            init_t0 = time.monotonic()
            init_phases["cache"] = "miss"
            # Phase durations for the init leg (sandbox create incl. image
            # build, queue+boot, init.py, snapshot), collected from the phase
            # events so result.json answers "what was slow" directly.
            _INIT_PHASE_FIELDS = {
                # create_sec includes the image build; queue_boot_sec is the GPU
                # queue plus container boot (see init_and_snapshot).
                "init_sandbox_created": ("sandbox_create_sec", "create_sec"),
                "init_sandbox_started": ("queue_boot_sec", "queue_boot_sec"),
                "init_py_done": ("init_py_sec", "init_py_sec"),
                "snapshot_done": ("snapshot_sec", "snapshot_sec"),
            }

            def _init_event(kind: str, **payload: Any) -> None:
                events.emit(kind, **payload)
                mapping = _INIT_PHASE_FIELDS.get(kind)
                if mapping:
                    init_phases[mapping[0]] = payload.get(mapping[1])
                if kind == "snapshot_done":
                    init_phases["snapshot_image_id"] = payload.get("image_id")

            agent_image = init_and_snapshot(
                submission.submission_dir, gpu=gpu, extra_pip=extra_pip,
                agent_runtime=agent_runtime,
                runtime_env=runtime_environment,
                log_path=run_dir / "init.log",
                on_line=lambda line: events.emit("init_line", line=line),
                on_event=_init_event,
            )
            init_duration = time.monotonic() - init_t0
            events.emit("init_done", duration_sec=init_duration, phases=init_phases)
            image_id = init_phases.get("snapshot_image_id")
            if use_init_cache and image_id:
                snapshot_cache.store(out_root, cache_key, image_id, run_id)
                events.emit("init_cache_stored", image_id=image_id)
        else:
            init_duration = 0.0
        return agent_image

    # The desktop boot is independent of agent installation and its snapshot.
    with ThreadPoolExecutor(max_workers=1) as preparation:
        runtime = _RemoteEvaluationRuntime(
            agent_image=preparation.submit(prepare_agent_image),
            run_dir=run_dir,
            run_id=run_id,
            region=region,
            events=events,
            execution_scale=scale,
            network_policy=network_policy,
            api_domains=api_domains,
            eval_algorithm=algorithm.key,
            environment_backend=environment_backend,
            environment_runtime=evaluator_runtime_environment,
        )
        try:
            rows = algorithm.run(runtime, jobs, scale)
        except BaseException as exc:
            events.emit("run_failed", error=repr(exc), run_plan_hash=plan.contract_hash)
            raise

    if selected_task_keys is not None:
        from cua_speedrun.scoring import collect_rows

        rows = collect_rows(run_dir)

    result = {
        "run_id": run_id,
        "benchmark": {"name": benchmark.name, "version": benchmark.version},
        "submission_fingerprint": submission.fingerprint,
        "backend": plan.backend,
        "harness_version": cua_speedrun.__version__,
        "agent_mode": algorithm.agent_mode,
        "eval_algorithm": plan.eval_algorithm,
        "interactivity": concurrency if algorithm.shared_agent_sandbox else None,
        "parallelism": scale_payload,
        "season": plan.season(),
        "run_plan": plan.to_dict(),
        "run_plan_hash": plan.contract_hash,
        "scoring_rules": plan.scoring_rules(),
        "task_seeds": resolved_task_seeds,
        "init_duration_sec": init_duration,
        "init_phases": init_phases,
        "rows": rows,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    events.emit("run_done", run_dir=str(run_dir))
    return run_dir


def _create_environment_sandbox(
    environment_backend: str,
    *,
    env_local_dir: Path,
    env_spec: Mapping[str, Any],
    seed: int,
    timeout_sec: float,
    grace_sec: float,
    generator_local: Path | None,
    task_label: str,
    region: str | None,
    sandbox_timeout_sec: int = 3600,
    runtime_env: Mapping[str, str] | None = None,
):
    """Create the env sandbox for the plan's environment backend.

    Both creators return the same EnvSandbox handle, so everything downstream
    (GatewayControl, arm, exec_agent, wait_done, pull_env_artifacts) stays
    backend-agnostic. ``env_spec`` is the task's env block, forwarded
    verbatim to the in-sandbox Backend so preparation semantics (defaults
    included) are the local backend's, not a launcher copy."""
    if environment_backend == "modal-native":
        if generator_local is not None:
            raise ValueError(
                "the modal-native environment does not run seeded generators"
            )
        return create_modal_native_env_sandbox(
            env_local_dir=env_local_dir,
            env_spec=env_spec,
            seed=seed,
            timeout_sec=timeout_sec,
            grace_sec=grace_sec,
            task_label=task_label,
            region=region,
            sandbox_timeout_sec=sandbox_timeout_sec,
            runtime_env=runtime_env,
        )
    return create_env_sandbox(
        env_local_dir=env_local_dir,
        env_spec=env_spec,
        seed=seed,
        timeout_sec=timeout_sec,
        grace_sec=grace_sec,
        generator_local=generator_local,
        task_label=task_label,
        region=region,
        sandbox_timeout_sec=sandbox_timeout_sec,
        runtime_env=runtime_env,
    )


def _run_one_remote(
    agent_image,
    task,
    seed: int,
    run_dir: Path,
    run_id: str,
    region: str | None,
    events: EventBus,
    network_policy: str,
    api_domains: Sequence[str] = (),
    environment_backend: str = "gym-anything-modal",
    environment_runtime: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    task_key = f"{task.task_id}/seed_{seed}"
    task_dir = run_dir / "tasks" / task.task_id / f"seed_{seed}"
    task_dir.mkdir(parents=True, exist_ok=True)

    env_dir = Path(os.path.expanduser(os.path.expandvars(str(task.env["env_dir"]))))
    generator_local = (task.task_dir / task.generator) if task.generator else None

    es = None
    agent_sb = None
    warm: dict[str, Any] = {}
    warm_thread: threading.Thread | None = None
    # meta.json is built progressively and written in the finally block, so a
    # failed or timed-out task still leaves its regions, sandbox ids, and
    # phase timings behind for diagnosis.
    meta: dict[str, Any] = {"requested_region": region}
    phases: dict[str, Any] = {}
    warmup_log_fh = None
    env_log_thread = None
    stop_env_log = threading.Event()

    # A heartbeat names the current phase every 30s, so a stalled task is
    # visible long before its timeout fires.
    current = {"phase": "env_create", "since": time.monotonic()}
    stop_heartbeat = threading.Event()

    def _set_phase(name: str) -> None:
        current["phase"] = name
        current["since"] = time.monotonic()

    def _heartbeat() -> None:
        while not stop_heartbeat.wait(30.0):
            events.emit("phase_heartbeat", task_key=task_key,
                        phase=current["phase"],
                        elapsed_sec=round(time.monotonic() - current["since"], 1))

    threading.Thread(target=_heartbeat, daemon=True).start()

    t_task0 = time.monotonic()
    try:
        events.emit("task_started", task_key=task_key, task_id=task.task_id, seed=seed)
        warmup_log_fh = open(task_dir / "warmup.log", "w")

        def _on_warmup_line(line: str) -> None:
            try:
                warmup_log_fh.write(line)
                warmup_log_fh.flush()
            except ValueError:
                pass  # file closed by the task thread after a failure
            events.emit("warmup_line", task_key=task_key, line=line)

        def _spawn(spawn_region: str | None, gateway_host: str | None,
                   on_started=None):
            return spawn_agent_sandbox(
                agent_image,
                gateway_host,
                network_policy=network_policy,
                api_domains=api_domains,
                region=spawn_region,
                task_timeout_sec=task.timeout_sec,
                warmup_timeout_sec=(GPU_WARMUP_TIMEOUT_SEC if agent_image.gpu
                                    else CPU_WARMUP_TIMEOUT_SEC),
                # Captured immediately so the finally block can terminate a
                # sandbox whose warmup thread never returned.
                on_sandbox=lambda raw: warm.__setitem__("sandbox_early", raw),
                on_line=_on_warmup_line,
                on_event=lambda kind, **p: events.emit(
                    kind, task_key=task_key, **p),
                on_started=on_started,
            )

        def _create_env(env_region: str | None):
            t0 = time.monotonic()
            created = _create_environment_sandbox(
                environment_backend,
                env_local_dir=env_dir,
                env_spec=task.env,
                seed=seed,
                timeout_sec=task.timeout_sec,
                grace_sec=task.grace_sec,
                generator_local=generator_local,
                task_label=task.task_id,
                region=env_region,
                sandbox_timeout_sec=max(
                    3600,
                    int(
                        task.timeout_sec
                        + task.grace_sec
                        + VERIFIER_TIMEOUT_SEC
                        + 1800
                    ),
                ),
                runtime_env=environment_runtime,
            )
            phases["env_create_sec"] = round(time.monotonic() - t0, 3)
            meta["env_sandbox_id"] = getattr(created.sandbox, "object_id", None)
            events.emit("env_created", task_key=task_key,
                        sandbox_id=meta["env_sandbox_id"])
            return created

        if agent_image.gpu:
            # Acquire GPU capacity first, then place the environment nearby.
            # An explicit --region remains binding.
            meta["placement"] = "gpu-first"
            region_known = threading.Event()

            def _on_agent_started(actual_region: str | None) -> None:
                warm["agent_region_early"] = actual_region
                region_known.set()

            def _warm_agent() -> None:
                try:
                    events.emit("warmup_started", task_key=task_key,
                                region=region or "unpinned")
                    warm["result"] = _spawn(region, None, _on_agent_started)
                except BaseException as exc:  # re-raised on the task thread
                    warm["error"] = exc
                finally:
                    region_known.set()  # unblock the waiter on failure too

            warm_thread = threading.Thread(target=_warm_agent, daemon=True)
            warm_thread.start()
            _set_phase("waiting for GPU sandbox (capacity queue)")
            deadline = time.monotonic() + GPU_WARMUP_TIMEOUT_SEC + 60
            while not region_known.wait(5.0):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"GPU sandbox never started within "
                        f"{GPU_WARMUP_TIMEOUT_SEC + 60}s")
            if "error" in warm:
                raise warm["error"]

            # GPU acquired: bring the env to its region while the submitted
            # init.py warms; env boot and agent warmup overlap.
            _set_phase("prepare (env boot + agent warmup)")
            env_region = region or warm.get("agent_region_early")
            try:
                es = _create_env(env_region)
            except Exception as exc:
                if region is not None:
                    raise  # an explicit --region stays strict
                events.emit("env_region_fallback", task_key=task_key,
                            region=env_region, error=repr(exc))
                es = _create_env(None)
            env_log_thread = tail_sandbox_file(
                es.sandbox, "/tmp/cs_run/env_plane.log",
                lambda line: events.emit("env_line", task_key=task_key, line=line),
                stop=stop_env_log)
            t0 = time.monotonic()
            wait_env_healthy(es)
            phases["env_boot_sec"] = round(time.monotonic() - t0, 3)
            events.emit("env_ready", task_key=task_key,
                        env_boot_sec=phases["env_boot_sec"])
            warm_thread.join()
            if "error" in warm:
                raise warm["error"]
            agent_sb = warm["result"]
            meta["env_region"] = read_region(es)
            gateway_host = urlparse(es.base_url).hostname
            grant_gateway_access(
                agent_sb,
                gateway_host,
                network_policy=network_policy,
                api_domains=api_domains,
            )
            events.emit("agent_network_granted", task_key=task_key,
                        host=gateway_host, network_policy=network_policy)
        else:
            # Env-first placement (CPU agents): CPU sandboxes schedule in
            # seconds anywhere, so boot the env and pin the agent to the
            # env's actual region; env boot and agent warmup overlap.
            meta["placement"] = "env-first"
            es = _create_env(region)
            gateway_host = urlparse(es.base_url).hostname
            env_log_thread = tail_sandbox_file(
                es.sandbox, "/tmp/cs_run/env_plane.log",
                lambda line: events.emit("env_line", task_key=task_key, line=line),
                stop=stop_env_log)
            _set_phase("prepare (env boot + agent warmup)")

            def _warm_agent() -> None:
                try:
                    env_region = region or read_region(es)
                    warm["env_region"] = env_region
                    events.emit("warmup_started", task_key=task_key,
                                region=env_region)
                    try:
                        warm["result"] = _spawn(env_region, gateway_host)
                    except Exception as exc:
                        # Auto co-location is best effort: the env sandbox
                        # can land in a region with no capable worker. Fall
                        # back to unpinned placement and record both
                        # regions. An explicit --region stays strict.
                        if region is None and env_region and \
                                "worker type supports" in str(exc):
                            events.emit("warmup_fallback", task_key=task_key,
                                        region=env_region)
                            warm["result"] = _spawn(None, gateway_host)
                        else:
                            raise
                except BaseException as exc:  # re-raised on the task thread
                    warm["error"] = exc

            warm_thread = threading.Thread(target=_warm_agent, daemon=True)
            warm_thread.start()
            t0 = time.monotonic()
            wait_env_healthy(es)
            phases["env_boot_sec"] = round(time.monotonic() - t0, 3)
            events.emit("env_ready", task_key=task_key,
                        env_boot_sec=phases["env_boot_sec"])
            warm_thread.join()
            if "error" in warm:
                raise warm["error"]
            agent_sb = warm["result"]
            meta["env_region"] = warm.get("env_region")

        phases["warmup_sec"] = round(agent_sb.warmup_sec, 3)
        meta["agent_region"] = agent_sb.region
        meta["warmup_sec"] = agent_sb.warmup_sec
        meta["agent_sandbox_id"] = getattr(agent_sb.sandbox, "object_id", None)
        events.emit("warmup_done", task_key=task_key,
                    warmup_sec=agent_sb.warmup_sec, agent_region=agent_sb.region)
        phases["prepare_total_sec"] = round(time.monotonic() - t_task0, 3)

        ctl = GatewayControl(es.base_url, es.control_token)
        gateway_info = ctl.info()
        instruction = gateway_info["instruction"]
        run_url = f"{es.base_url}/{es.run_token}"

        # Everything is warm; the task is revealed only past this line.
        _set_phase("running")
        ctl.arm()
        events.emit("armed", task_key=task_key)
        t0 = time.monotonic()
        with monitor_gateway_steps(
            ctl.status,
            lambda steps: events.emit(
                "task_progress", task_key=task_key, num_steps=steps
            ),
        ):
            if gateway_info.get("multi_episode"):
                code, agent_out, agent_err = _run_agent_episodes(
                    ctl,
                    agent_sb,
                    run_url,
                    instruction,
                    task_id="t0",
                    timeout_sec=int(task.timeout_sec) + 60,
                    events=events,
                    task_key=task_key,
                )
            else:
                code, agent_out, agent_err = exec_agent(
                    agent_sb, run_url, instruction,
                    timeout_sec=int(task.timeout_sec) + 60,
                    on_event=lambda kind, **p: events.emit(
                        kind, task_key=task_key, **p
                    ),
                )
            phases["agent_wall_sec"] = round(time.monotonic() - t0, 3)
            (task_dir / "agent.stdout").write_text(agent_out)
            (task_dir / "agent.stderr").write_text(agent_err)
            if not gateway_info.get("multi_episode"):
                events.emit("agent_exit", task_key=task_key, code=code)
            meta["agent_exit_code"] = code
            _record_agent_outcome(ctl, code, events, task_key)

            _set_phase("finishing (grace + checker)")
            t0 = time.monotonic()
            ctl.wait_done(
                budget_sec=(
                    task.timeout_sec
                    + task.grace_sec
                    + VERIFIER_TIMEOUT_SEC
                    + 60
                )
            )
            phases["finish_wait_sec"] = round(time.monotonic() - t0, 3)
        (task_dir / "runlog.jsonl").write_text(ctl.runlog())

        # Pull the trajectory (frames + episode) and logs out of the sandboxes
        # before they are torn down, so they land with the run's artifacts.
        _set_phase("pulling artifacts")
        t0 = time.monotonic()
        pulled, pull_error = pull_env_artifacts(es, task_dir)
        pull_agent_logs(agent_sb, task_dir)
        phases["artifact_pull_sec"] = round(time.monotonic() - t0, 3)
        meta["artifacts"] = pulled
        if pull_error:
            meta["artifacts_error"] = pull_error
        events.emit(
            "artifacts_pulled",
            task_key=task_key,
            count=len(pulled),
            error=pull_error,
        )
    except AgentWarmupError as exc:
        # The submission's init.py rerun failed deterministically. Mirror the
        # local runtime, where this class of failure propagates out of
        # run_isolated unwrapped and fails the run: retrying cannot change a
        # deterministic failure, so it must not enter the infrastructure
        # retry loop.
        meta["error"] = repr(exc)
        raise
    except Exception as exc:
        infrastructure_error = (
            exc
            if isinstance(exc, InstanceInfrastructureError)
            else InstanceInfrastructureError(
                f"evaluator failed while running the isolated instance: "
                f"{type(exc).__name__}: {exc}"
            )
        )
        meta["error"] = repr(infrastructure_error)
        events.emit(
            "task_interrupted",
            task_key=task_key,
            error=repr(infrastructure_error),
        )
        if infrastructure_error is exc:
            raise
        raise infrastructure_error from exc
    finally:
        stop_heartbeat.set()
        if warmup_log_fh is not None:
            try:
                warmup_log_fh.close()
            except OSError:
                pass
        meta["phases"] = phases
        try:
            (task_dir / "meta.json").write_text(
                json.dumps(meta, indent=2, default=str))
        except OSError:
            pass
        events.emit("task_phases", task_key=task_key, **phases)
        # Stop the env log tail before terminating the sandbox: its polling
        # execs stop cleanly, so nothing races interpreter shutdown.
        stop_env_log.set()
        if env_log_thread is not None:
            env_log_thread.join(timeout=10)
        # Reap the warm thread's sandbox too: when the env dies while the
        # agent is still warming, agent_sb is never assigned and the spawned
        # sandbox previously leaked, one per retry attempt.
        if warm_thread is not None and warm_thread.is_alive():
            warm_thread.join(timeout=15)
        warm_result = warm.get("result")
        warm_handle = (
            warm_result.sandbox if warm_result is not None
            else warm.get("sandbox_early")
        )
        if agent_sb is not None and warm_result is agent_sb:
            warm_handle = None
        for handle in (agent_sb.sandbox if agent_sb else None,
                       warm_handle,
                       es.sandbox if es else None):
            if handle is not None:
                try:
                    handle.terminate()
                except Exception:
                    pass

    row = summarize(load_runlog(task_dir / "runlog.jsonl"))
    events.emit("task_done", task_key=task_key, **{
        k: row.get(k) for k in
        ("passed", "reason", "task_time_sec", "env_time_sec",
         "agent_time_sec", "num_steps")
    })
    return row


class _ReplicaEvents:
    def __init__(self, parent: EventBus, replica_index: int) -> None:
        self.parent = parent
        self.replica_index = replica_index

    def emit(
        self,
        kind: str,
        task_key: str | None = None,
        **payload: Any,
    ) -> None:
        self.parent.emit(
            kind,
            task_key=task_key,
            parallel_evaluation=self.replica_index,
            **payload,
        )


@dataclass
class _RemoteComputeReplica:
    index: int
    events: _ReplicaEvents
    total_timeout: float
    shared: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, int] = field(
        default_factory=lambda: {"batch": 0, "ready": 0, "size": 0}
    )
    stop_heartbeat: threading.Event = field(default_factory=threading.Event)
    warm_thread: threading.Thread | None = None
    heartbeat_thread: threading.Thread | None = None
    recovery_lock: Any = field(default_factory=threading.Lock, repr=False)
    # The region the replica's agent actually runs in; its environments are
    # created there so the timed agent-to-gateway hop stays in one region.
    region: str | None = None
    region_known: threading.Event = field(default_factory=threading.Event)


class _RemoteEvaluationRuntime:
    """Modal implementation of the provider-neutral algorithm operations."""

    capabilities = MODAL_RUNTIME_CAPABILITIES

    def __init__(
        self,
        *,
        agent_image,
        run_dir: Path,
        run_id: str,
        region: str | None,
        events: EventBus,
        execution_scale: ExecutionScale,
        network_policy: str,
        api_domains: Sequence[str],
        eval_algorithm: str,
        environment_backend: str = "gym-anything-modal",
        environment_runtime: Mapping[str, str] | None = None,
    ) -> None:
        self._agent_image = agent_image
        self.run_dir = run_dir
        self.run_id = run_id
        self.region = region
        self.events = events
        self.scale = execution_scale
        self.network_policy = network_policy
        self.api_domains = api_domains
        self.eval_algorithm = eval_algorithm
        self.environment_backend = environment_backend
        self.environment_runtime = dict(environment_runtime or {})

    @property
    def agent_image(self):
        image = self._agent_image
        return image.result() if isinstance(image, Future) else image

    def run_isolated(self, job):
        task, seed = job
        return _run_one_remote(
            self.agent_image,
            task,
            seed,
            self.run_dir,
            self.run_id,
            self.region,
            self.events,
            self.network_policy,
            self.api_domains,
            environment_backend=self.environment_backend,
            environment_runtime=self.environment_runtime,
        )

    def _start_replica(self, index, jobs):
        replica = _RemoteComputeReplica(
            index=index,
            events=_ReplicaEvents(self.events, index),
            total_timeout=sum(task.timeout_sec for task, _ in jobs) + 600,
        )

        def heartbeat() -> None:
            while not replica.stop_heartbeat.wait(30.0):
                progress = replica.progress
                replica.events.emit(
                    "phase_heartbeat",
                    phase=(
                        f"preparing batch {progress['batch']}: "
                        f"{progress['ready']}/{progress['size']} envs ready, "
                        f"warmup "
                        f"{'done' if 'sb' in replica.shared else 'running'}"
                    ),
                    elapsed_sec=0.0,
                )

        replica.heartbeat_thread = threading.Thread(
            target=heartbeat, daemon=True
        )
        replica.heartbeat_thread.start()
        self._launch_replica(replica)
        return replica

    def _launch_replica(self, replica):
        replica.shared.clear()
        replica.region_known.clear()

        def started(actual_region: str | None) -> None:
            replica.region = actual_region
            replica.region_known.set()

        def warm() -> None:
            try:
                replica.events.emit(
                    "warmup_started", region=self.region or "unpinned"
                )
                replica.shared["sb"] = spawn_agent_sandbox(
                    self.agent_image,
                    None,
                    network_policy=self.network_policy,
                    api_domains=self.api_domains,
                    region=self.region,
                    task_timeout_sec=replica.total_timeout,
                    warmup_timeout_sec=(
                        GPU_WARMUP_TIMEOUT_SEC
                        if self.agent_image.gpu
                        else CPU_WARMUP_TIMEOUT_SEC
                    ),
                    on_line=lambda line: replica.events.emit(
                        "warmup_line", line=line
                    ),
                    on_event=lambda kind, **payload: replica.events.emit(
                        kind, **payload
                    ),
                    on_started=started,
                )
            except BaseException as exc:
                replica.shared["error"] = exc
            finally:
                # Never leave an environment waiting on a warmup that failed.
                replica.region_known.set()
        replica.warm_thread = threading.Thread(target=warm, daemon=True)
        replica.warm_thread.start()

    def start_compute_replicas(self, shards):
        replicas = []
        try:
            for index, jobs in enumerate(shards, start=1):
                replicas.append(self._start_replica(index, jobs))
            return replicas
        except BaseException:
            for replica in replicas:
                self.close_compute_replica(replica)
            raise

    def ensure_compute_ready(self, replica):
        if replica.warm_thread is not None:
            replica.warm_thread.join()
        if "error" in replica.shared:
            raise replica.shared["error"]
        agent = replica.shared.get("sb")
        if agent is None:
            raise ComputeInfrastructureError(
                "Modal compute replica has no live agent sandbox"
            )
        if agent.sandbox.poll() is not None:
            resource_id = getattr(agent.sandbox, "object_id", None)
            raise ComputeInfrastructureError(
                "Modal compute replica exited",
                resource_id=resource_id,
            )

    def recover_instance_infrastructure(
        self,
        replica,
        error: InstanceInfrastructureError,
    ):
        if not isinstance(error, ComputeInfrastructureError):
            return
        with replica.recovery_lock:
            agent = replica.shared.get("sb")
            current_id = (
                getattr(agent.sandbox, "object_id", None)
                if agent is not None else None
            )
            if error.resource_id and current_id != error.resource_id:
                return
            if agent is not None:
                try:
                    agent.sandbox.terminate()
                except Exception:
                    pass
            self._launch_replica(replica)

    def compute_infrastructure_retry(
        self,
        replica,
        attempt,
        max_attempts,
        delay_sec,
        error,
    ):
        replica.events.emit(
            "infra_retry",
            phase="compute_replica",
            attempt=attempt,
            next_attempt=attempt + 1,
            max_attempts=max_attempts,
            delay_sec=delay_sec,
            error=repr(error),
        )

    def compute_infrastructure_failed(
        self, replica, attempts, error
    ):
        replica.events.emit(
            "compute_replica_failed",
            phase="compute_replica",
            attempts=attempts,
            error=repr(error),
        )

    def begin_environment_batch(self, replica, jobs, batch_number):
        replica.progress.update(
            batch=batch_number,
            ready=0,
            size=len(jobs),
        )
        task, seed = jobs[0]
        replica.events.emit(
            "task_started",
            task_key=f"{task.task_id}/seed_{seed}",
            task_id=task.task_id,
            seed=seed,
        )

    def prepare_environment(self, replica, job, batch_number):
        task, seed = job
        task_key = f"{task.task_id}/seed_{seed}"
        task_dir = self.run_dir / "tasks" / task.task_id / f"seed_{seed}"
        task_dir.mkdir(parents=True, exist_ok=True)
        env_dir = Path(
            os.path.expanduser(os.path.expandvars(str(task.env["env_dir"])))
        )
        generator_local = (
            task.task_dir / task.generator if task.generator else None
        )
        started = time.monotonic()
        environment = None
        env_log_stop = threading.Event()
        env_log_thread = None
        try:
            def create(region):
                return _create_environment_sandbox(
                    self.environment_backend,
                    env_local_dir=env_dir,
                    env_spec=task.env,
                    seed=seed,
                    timeout_sec=task.timeout_sec,
                    grace_sec=task.grace_sec,
                    generator_local=generator_local,
                    task_label=task.task_id,
                    region=region,
                    sandbox_timeout_sec=max(
                        3600,
                        int(
                            task.timeout_sec
                            + task.grace_sec
                            + VERIFIER_TIMEOUT_SEC
                            + 1800
                        ),
                    ),
                    runtime_env=self.environment_runtime,
                )
            region = self.region
            if region is None and replica.warm_thread is not None:
                # Untimed: wait until the agent is running and follow it.
                replica.region_known.wait(GPU_WARMUP_TIMEOUT_SEC + 60)
            if region is None:
                region = replica.region
            try:
                environment = create(region)
            except Exception as exc:
                if self.region is not None or region is None:
                    raise  # an explicit --region stays strict
                replica.events.emit("env_region_fallback", task_key=task_key,
                                    region=region, error=repr(exc))
                environment = create(None)
            replica.events.emit(
                "env_created",
                task_key=task_key,
                sandbox_id=getattr(environment.sandbox, "object_id", None),
            )
            env_log_thread = tail_sandbox_file(
                environment.sandbox,
                "/tmp/cs_run/env_plane.log",
                lambda line: replica.events.emit(
                    "env_line", task_key=task_key, line=line
                ),
                stop=env_log_stop,
            )
            wait_env_healthy(environment)
        except BaseException:
            env_log_stop.set()
            if env_log_thread is not None:
                env_log_thread.join(timeout=10)
            if environment is not None:
                try:
                    environment.sandbox.terminate()
                except Exception:
                    pass
            raise
        boot_sec = round(time.monotonic() - started, 3)
        replica.progress["ready"] += 1
        env_region = read_region(environment)
        replica.events.emit(
            "env_ready", task_key=task_key, env_boot_sec=boot_sec,
            env_region=env_region,
        )
        return {
            "task": task,
            "seed": seed,
            "task_key": task_key,
            "task_dir": task_dir,
            "es": environment,
            "env_region": env_region,
            "env_boot_sec": boot_sec,
            "env_log_stop": env_log_stop,
            "env_log_thread": env_log_thread,
        }

    def environment_prepare_retry(
        self,
        replica,
        job,
        batch_number,
        attempt,
        max_attempts,
        delay_sec,
        error,
    ):
        task, seed = job
        # Preserve the failed attempt's evidence before the retry overwrites
        # it, exactly as the local runtime does.
        task_dir = self.run_dir / "tasks" / task.task_id / f"seed_{seed}"
        archive = task_dir / f"_infra_attempt_{attempt}"
        archive.mkdir(parents=True, exist_ok=True)
        for child in list(task_dir.iterdir()):
            if child.name.startswith("_infra_attempt_"):
                continue
            child.replace(archive / child.name)
        replica.events.emit(
            "infra_retry",
            task_key=f"{task.task_id}/seed_{seed}",
            phase="environment_prepare",
            attempt=attempt,
            next_attempt=attempt + 1,
            max_attempts=max_attempts,
            delay_sec=delay_sec,
            error=repr(error),
        )

    def environment_prepare_failed(
        self,
        replica,
        job,
        batch_number,
        attempts,
        error,
    ):
        task, seed = job
        replica.events.emit(
            "task_failed",
            task_key=f"{task.task_id}/seed_{seed}",
            phase="environment_prepare",
            attempts=attempts,
            error=repr(error),
        )

    def instance_infrastructure_retry(
        self,
        replica,
        job,
        attempt,
        max_attempts,
        delay_sec,
        error,
    ):
        task, seed = job
        events = self.events if replica is None else replica.events
        # Archive the interrupted attempt on both the isolated and the
        # shared path, as the local runtime does: without this a shared-mode
        # retry overwrites the previous attempt's artifacts, and a stale
        # runlog from the failed attempt could be summarized as the result.
        task_dir = self.run_dir / "tasks" / task.task_id / f"seed_{seed}"
        archive = task_dir / f"_instance_infra_attempt_{attempt}"
        archive.mkdir(parents=True, exist_ok=True)
        for child in list(task_dir.iterdir()):
            if child.name.startswith("_instance_infra_attempt_"):
                continue
            child.replace(archive / child.name)
        events.emit(
            "infra_retry",
            task_key=f"{task.task_id}/seed_{seed}",
            phase="instance_run",
            attempt=attempt,
            next_attempt=attempt + 1,
            max_attempts=max_attempts,
            delay_sec=delay_sec,
            error=repr(error),
        )

    def instance_infrastructure_failed(
        self, replica, job, attempts, error
    ):
        task, seed = job
        events = self.events if replica is None else replica.events
        events.emit(
            "task_failed",
            task_key=f"{task.task_id}/seed_{seed}",
            phase="instance_run",
            attempts=attempts,
            error=repr(error),
        )

    def bind_compute(self, replica, prepared, batch_number):
        self.ensure_compute_ready(replica)
        agent = replica.shared["sb"]
        replica.events.emit(
            "all_envs_ready",
            count=len(prepared),
            batch=batch_number,
            warmup_sec=agent.warmup_sec,
            agent_region=agent.region,
        )
        hosts = [urlparse(item["es"].base_url).hostname for item in prepared]
        grant_gateway_access(
            agent,
            hosts,
            network_policy=self.network_policy,
            api_domains=self.api_domains,
        )
        replica.events.emit(
            "agent_network_granted",
            host=f"{len(hosts)} batch gateways",
            network_policy=self.network_policy,
        )

    def run_prepared(self, replica, prepared, batch_number):
        task = prepared["task"]
        task_key = prepared["task_key"]
        task_dir = prepared["task_dir"]
        environment = prepared["es"]
        self.ensure_compute_ready(replica)
        agent = replica.shared["sb"]
        meta: dict[str, Any] = {
            "requested_region": self.region,
            "placement": "shared-bounded",
            "parallel_evaluation": replica.index,
            "eval_algorithm": self.eval_algorithm,
            "interactivity": self.scale.agents_per_evaluation,
            "env_pool_size": self.scale.environment_pool_size_per_evaluation,
            "env_batch": batch_number,
            "env_boot_sec": prepared["env_boot_sec"],
            "env_sandbox_id": getattr(environment.sandbox, "object_id", None),
            "agent_sandbox_id": getattr(agent.sandbox, "object_id", None),
            "agent_region": agent.region,
            "env_region": prepared.get("env_region"),
        }
        try:
            control = GatewayControl(
                environment.base_url, environment.control_token
            )
            gateway_info = control.info()
            instruction = gateway_info["instruction"]
            run_url = f"{environment.base_url}/{environment.run_token}"
            control.arm()
            replica.events.emit("armed", task_key=task_key)
            started = time.monotonic()
            with monitor_gateway_steps(
                control.status,
                lambda steps: replica.events.emit(
                    "task_progress", task_key=task_key, num_steps=steps
                ),
            ):
                dispatch_id = f"replica_{replica.index}_batch_{batch_number}"
                if gateway_info.get("multi_episode"):
                    code, agent_out, agent_err = _run_agent_episodes(
                        control,
                        agent,
                        run_url,
                        instruction,
                        task_id=dispatch_id,
                        timeout_sec=int(task.timeout_sec) + 60,
                        events=replica.events,
                        task_key=task_key,
                    )
                else:
                    code, agent_out, agent_err = exec_agent(
                        agent,
                        run_url,
                        instruction,
                        # The supervisor de-duplicates go-file names for its
                        # lifetime. A retried task needs an attempt-scoped ID.
                        task_id=dispatch_id,
                        timeout_sec=int(task.timeout_sec) + 60,
                        on_event=lambda kind, **payload: replica.events.emit(
                            kind, task_key=task_key, **payload
                        ),
                    )
                meta["agent_wall_sec"] = round(time.monotonic() - started, 3)
                meta["agent_exit_code"] = code
                (task_dir / "agent.stdout").write_text(agent_out)
                (task_dir / "agent.stderr").write_text(agent_err)
                if not gateway_info.get("multi_episode"):
                    replica.events.emit("agent_exit", task_key=task_key, code=code)
                _record_agent_outcome(
                    control, code, replica.events, task_key
                )
                control.wait_done(
                    budget_sec=(
                        task.timeout_sec
                        + task.grace_sec
                        + VERIFIER_TIMEOUT_SEC
                        + 60
                    )
                )
            (task_dir / "runlog.jsonl").write_text(control.runlog())
            meta["artifacts"], pull_error = pull_env_artifacts(
                environment, task_dir
            )
            if pull_error:
                meta["artifacts_error"] = pull_error
            replica.events.emit(
                "artifacts_pulled",
                task_key=task_key,
                count=len(meta["artifacts"]),
                error=pull_error,
            )
        except Exception as exc:
            meta["error"] = repr(exc)
            if isinstance(exc, InstanceInfrastructureError):
                replica.events.emit(
                    "task_interrupted", task_key=task_key, error=repr(exc)
                )
                raise
            if environment.sandbox.poll() is not None:
                replica.events.emit(
                    "task_interrupted", task_key=task_key, error=repr(exc)
                )
                raise InstanceInfrastructureError(
                    "Modal environment sandbox exited during the instance"
                ) from exc
            if agent.sandbox.poll() is not None:
                replica.events.emit(
                    "task_interrupted", task_key=task_key, error=repr(exc)
                )
                raise ComputeInfrastructureError(
                    "Modal compute replica exited during the instance",
                    resource_id=getattr(agent.sandbox, "object_id", None),
                ) from exc
            infrastructure_error = InstanceInfrastructureError(
                f"evaluator failed while running the remote instance: "
                f"{type(exc).__name__}: {exc}"
            )
            replica.events.emit(
                "task_interrupted",
                task_key=task_key,
                error=repr(infrastructure_error),
            )
            raise infrastructure_error from exc
        finally:
            try:
                (task_dir / "meta.json").write_text(
                    json.dumps(meta, indent=2, default=str)
                )
            except OSError:
                pass
        row = summarize(load_runlog(task_dir / "runlog.jsonl"))
        replica.events.emit(
            "task_done",
            task_key=task_key,
            **{
                key: row.get(key)
                for key in (
                    "passed",
                    "reason",
                    "task_time_sec",
                    "env_time_sec",
                    "agent_time_sec",
                    "num_steps",
                )
            },
        )
        return row

    def close_environment_batch(self, replica, prepared, batch_number):
        for item in prepared:
            item["env_log_stop"].set()
            if item["env_log_thread"] is not None:
                item["env_log_thread"].join(timeout=10)
            try:
                item["es"].sandbox.terminate()
            except Exception:
                pass

    def finish_compute(self, replica):
        compute_dir = self.run_dir / "compute" / f"evaluation_{replica.index}"
        compute_dir.mkdir(parents=True, exist_ok=True)
        pull_agent_logs(replica.shared["sb"], compute_dir)

    def close_compute_replica(self, replica):
        replica.stop_heartbeat.set()
        if replica.warm_thread is not None:
            replica.warm_thread.join()
        if "sb" in replica.shared:
            try:
                replica.shared["sb"].sandbox.terminate()
            except Exception:
                pass
