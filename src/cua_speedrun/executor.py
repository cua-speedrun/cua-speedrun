"""The executor: prepares machines, launches runs, collects logs.

The executor is never on the timed path. It arms the gateway, starts the
agent process, and waits. Task runs share nothing, so the benchmark fans
out over a thread pool at the selected agents-per-evaluation width.

The local implementation places both execution planes on the selected local
runner. For the in-process runner that is the evaluator host; for a scheduled
runner the gateway, KVM/QEMU environment, model server, and agent live inside
the allocated worker. Dashboard runs arrive with the same frozen RunPlan,
private seed map, event sinks, and result format as the Modal executor.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import secrets
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

import cua_speedrun
from cua_speedrun.compute_runners import (
    LocalComputeRunner,
    SlurmComputeRunner,
    local_runtime_overrides,
    resolve_runner_selection,
)
from cua_speedrun.compute_runners.base import ComputeRunner, ComputeRunnerContext
from cua_speedrun.envs import get_backend
from cua_speedrun.environment_instance import (
    PreparedEnvironmentInstance as _LocalPreparedJob,
    prepare_environment_instance as _prepare_one,
)
from cua_speedrun.eval_algorithms import (
    SHARED_AGENT_VLLM,
    resolve_eval_algorithm,
)
from cua_speedrun.evaluation_runtime import (
    AgentExecutionError,
    ComputeInfrastructureError,
    ISOLATED_INSTANCE,
    InstanceInfrastructureError,
    LOCAL_RUNTIME_CAPABILITIES,
)
from cua_speedrun.events import ConsoleSink, EventBus, JsonlSink, Sink
from cua_speedrun.evaluator_environment import (
    agent_environment,
    evaluator_environment,
    evaluator_process_environment,
    validate as validate_evaluator_environment,
)
from cua_speedrun.local_runtime import (
    validate_local_agent_runtime_contract,
    validate_local_gpu,
)
from cua_speedrun.parallelism import (
    DEFAULT_PARALLEL_EVALUATIONS,
    ExecutionScale,
)
from cua_speedrun.gateway import Gateway, monitor_gateway_steps
from cua_speedrun.runlog import load_runlog, summarize
from cua_speedrun.remote.snapshot_cache import compute_key as compute_init_cache_key
from cua_speedrun.runplan import RunPlan, benchmark_contract, harness_contract
from cua_speedrun.specs import Benchmark
from cua_speedrun.submission import Submission
from cua_speedrun.task_jobs import build_task_seed_jobs, select_task_seed_jobs

_SAFE_SERVICE_ENVIRONMENT_NAMES = {
    "CURL_CA_BUNDLE",
    "HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "LD_LIBRARY_PATH",
    "LOGNAME",
    "PATH",
    "PIP_CACHE_DIR",
    "REQUESTS_CA_BUNDLE",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TMP",
    "TMPDIR",
    "TZ",
    "USER",
    "UV_CACHE_DIR",
    "XDG_CACHE_HOME",
}
_SAFE_SERVICE_ENVIRONMENT_PREFIXES = (
    "CUDA_",
    "HIP_",
    "HSA_",
    "MKL_",
    "NCCL_",
    "NVIDIA_",
    "OMP_",
    "OPENBLAS_",
    "ROCM_",
    "ROCR_",
    "UCX_",
)


def _process_environment(*, inherit_all: bool) -> dict[str, str]:
    if inherit_all:
        return os.environ.copy()
    return {
        name: value
        for name, value in os.environ.items()
        if name in _SAFE_SERVICE_ENVIRONMENT_NAMES
        or name.startswith(_SAFE_SERVICE_ENVIRONMENT_PREFIXES)
    }


def run_benchmark(
    submission_dir: Path,
    benchmark_dir: Path,
    backend_name: str,
    out_root: Path,
    concurrency: int | None = None,
    runs_per_task: int = 1,
    seed_base: int = 0,
    task_ids: list[str] | None = None,
    gpu: str | None = None,
    run_id: str | None = None,
    extra_sinks: Sequence[Sink] = (),
    run_plan: RunPlan | dict[str, Any] | None = None,
    task_seeds: Mapping[str, Sequence[int]] | None = None,
    environment_variables: Mapping[str, str] | None = None,
    inherit_process_environment: bool = True,
    execution_scale: ExecutionScale | Mapping[str, Any] | None = None,
    parallel_evaluations: int | None = None,
    eval_algorithm: str | None = None,
    compute_runner: str | None = None,
    runner_template: str | Path | None = None,
    selected_task_keys: Collection[str] | None = None,
) -> Path:
    """Run one submission against one benchmark. Returns the run directory."""
    submission = Submission.load(submission_dir)
    benchmark = Benchmark.load(Path(benchmark_dir))
    supplied_plan = run_plan is not None
    if isinstance(run_plan, dict):
        run_plan = RunPlan.from_dict(run_plan)
    plan = run_plan
    runner_selection = resolve_runner_selection(compute_runner, runner_template)
    # The local topology advertises both capabilities; the in-process runner
    # only provides shared compute replicas, while a scheduled runner also
    # provides one isolated allocation per task instance.
    runtime_capabilities = (
        LOCAL_RUNTIME_CAPABILITIES
        if runner_selection.kind == "slurm"
        else LOCAL_RUNTIME_CAPABILITIES - frozenset({ISOLATED_INSTANCE})
    )
    algorithm = resolve_eval_algorithm(
        plan.eval_algorithm if plan is not None
        else eval_algorithm or SHARED_AGENT_VLLM
    )
    if (
        plan is not None
        and eval_algorithm is not None
        and resolve_eval_algorithm(eval_algorithm).key != plan.eval_algorithm
    ):
        raise ValueError("eval_algorithm conflicts with the frozen run plan")
    backend = get_backend(backend_name)
    supplied_scale = (
        ExecutionScale.from_value(execution_scale)
        if execution_scale is not None
        else None
    )
    if supplied_scale is not None:
        scale = (
            supplied_scale.validate_for_plan(plan)
            if plan is not None
            else supplied_scale
        )
        if (
            concurrency is not None
            and int(concurrency) != scale.agents_per_evaluation
        ):
            raise ValueError("concurrency conflicts with execution_scale")
        if (
            parallel_evaluations is not None
            and int(parallel_evaluations) != scale.parallel_evaluations
        ):
            raise ValueError(
                "parallel_evaluations conflicts with execution_scale"
            )
    elif plan is not None:
        scale = ExecutionScale.for_plan(
            plan,
            DEFAULT_PARALLEL_EVALUATIONS
            if parallel_evaluations is None
            else int(parallel_evaluations),
        )
        if (
            concurrency is not None
            and int(concurrency) != scale.agents_per_evaluation
        ):
            raise ValueError("concurrency conflicts with the frozen run plan")
    else:
        agents = int(concurrency if concurrency is not None else 4)
        scale = ExecutionScale.for_algorithm(
            algorithm.key,
            (
                DEFAULT_PARALLEL_EVALUATIONS
                if parallel_evaluations is None
                else int(parallel_evaluations)
            ),
            agents,
        )
    scale_payload = scale.to_dict()

    if plan is not None:
        current_benchmark = benchmark_contract(Path(benchmark_dir))
        for field in ("name", "version", "content_hash"):
            if current_benchmark.get(field) != plan.benchmark.get(field):
                raise ValueError(
                    f"benchmark {field} changed after this run was planned: "
                    f"planned {plan.benchmark.get(field)!r}, "
                    f"current {current_benchmark.get(field)!r}"
                )
        if (
            selected_task_keys is None
            and harness_contract(plan.eval_algorithm) != plan.to_dict()["harness"]
        ):
            raise ValueError(
                "measurement harness changed after this run was planned; "
                "submit again to create a new season"
            )
        topology = plan.resolved_execution_topology()
        if topology["key"] != "local":
            raise ValueError(
                f"local executor cannot execute topology {topology['key']!r}"
            )
        if topology["compute"]["mode"] != "local":
            raise ValueError(
                "local executor requires local model/agent placement"
            )
        if topology["environment"]["mode"] != "local":
            raise ValueError(
                "local executor requires local environment-VM placement"
            )
        if topology["environment_backend"] != backend.name:
            raise ValueError(
                f"local topology requires environment backend "
                f"{topology['environment_backend']!r}, got {backend.name!r}"
            )
        if not algorithm.supports(runtime_capabilities):
            raise ValueError(
                f"the {runner_selection.kind!r} compute runner does not "
                f"implement {plan.eval_algorithm!r}; per-task evaluations on "
                "the local topology need a scheduled compute runner"
            )
        if gpu is not None and gpu != plan.gpu:
            raise ValueError(f"gpu {gpu!r} conflicts with frozen gpu {plan.gpu!r}")
        if task_ids is not None and list(task_ids) != list(plan.benchmark["task_ids"]):
            raise ValueError("task_ids conflict with the frozen run plan")
        task_ids = list(plan.benchmark["task_ids"])
        runs_per_task = plan.runs_per_task
        gpu = plan.gpu

    if task_ids:
        tasks_by_id = {task.task_id: task for task in benchmark.tasks}
        missing = sorted(set(task_ids) - set(tasks_by_id))
        if missing:
            raise ValueError(
                "unknown task_id(s) for benchmark "
                f"{benchmark.name}: {', '.join(missing)}"
            )
        benchmark.tasks = [tasks_by_id[task_id] for task_id in task_ids]
    preflight = getattr(backend, "preflight", None)
    if callable(preflight):
        preflight(benchmark)
    if plan is not None:
        environment_runner = getattr(backend, "observed_runner_name", None)
        if environment_runner is None:
            raise RuntimeError(
                "gym-anything preflight did not report its selected local runner"
            )
        validate_local_agent_runtime_contract(
            plan.resolved_agent_runtime(),
            environment_runner=environment_runner,
            **local_runtime_overrides(runner_selection, plan.gpu),
        )
        validate_local_gpu(plan.resolved_agent_runtime(), plan.gpu)

    if run_id is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{stamp}_{secrets.token_hex(3)}"
    run_dir = Path(out_root) / run_id
    prior_runner: dict[str, Any] = {}
    prior_plan_path = run_dir / "run_plan.json"
    if selected_task_keys is not None and prior_plan_path.is_file():
        try:
            prior_runner = dict(
                json.loads(prior_plan_path.read_text()).get("compute_runner") or {}
            )
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(
                "cannot resume: the stored compute-runner record is unreadable"
            ) from exc
    run_dir.mkdir(parents=True, exist_ok=selected_task_keys is not None)

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
    json_sink = JsonlSink(run_dir / "events.jsonl")
    events = EventBus([ConsoleSink(), json_sink, *extra_sinks])
    private_environment = {
        str(name): str(value)
        for name, value in (environment_variables or {}).items()
    }
    server_environment = plan.server_environment() if plan is not None else {}
    submission_environment = {**private_environment, **server_environment}
    validate_evaluator_environment(benchmark, submission_environment)
    evaluator_runtime_env = evaluator_environment(
        benchmark, submission_environment
    )
    submission_environment = agent_environment(
        benchmark, submission_environment
    )
    private_environment = agent_environment(benchmark, private_environment)
    base_runtime_env = _process_environment(
        inherit_all=inherit_process_environment
    )
    # Maintainer-owned server configuration is frozen in the run plan and
    # cannot be replaced by account or per-evaluation variables.
    base_runtime_env.update(submission_environment)
    base_runtime_env = agent_environment(benchmark, base_runtime_env)
    agent_runtime = (
        plan.resolved_agent_runtime()
        if plan is not None
        else {"python_packages": ["requests>=2.31"]}
    )
    cache_environment = {
        **agent_environment(benchmark, server_environment),
        **{
            f"private:{name}": "sha256:"
            + hashlib.sha256(value.encode()).hexdigest()
            for name, value in private_environment.items()
        },
    }
    init_cache_key = compute_init_cache_key(
        submission.submission_dir,
        gpu,
        list(plan.server_config.get("extra_pip") or []) if plan is not None else [],
        cache_environment,
        agent_runtime,
    )
    runner_context = ComputeRunnerContext(
        submission=submission,
        run_dir=run_dir,
        run_id=run_id,
        base_runtime_env=base_runtime_env,
        submission_environment_names=tuple(sorted(submission_environment)),
        evaluator_runtime_env=evaluator_runtime_env,
        init_cache_key=init_cache_key,
        python_packages=(
            tuple(agent_runtime["python_packages"])
            if plan is not None else ("requests>=2.31",)
        ),
        managed_runtime=supplied_plan,
        gpu=gpu,
        track_name=plan.track_name if plan is not None else backend_name,
        agents_per_evaluation=scale.agents_per_evaluation,
        events=events,
    )
    runner: ComputeRunner
    if runner_selection.kind == "local":
        runner = LocalComputeRunner(runner_context)
    else:
        assert runner_selection.template_path is not None
        runner = SlurmComputeRunner(
            runner_context, runner_selection.template_path
        )
    runner_identity = dict(runner.identity)
    if prior_runner and runner_identity != prior_runner:
        raise ValueError(
            "the compute runner changed since this evaluation started; "
            "restore the recorded runner before resuming"
        )
    if plan is not None:
        (run_dir / "run_plan.json").write_text(json.dumps({
            "run_plan_hash": plan.contract_hash,
            "run_plan": plan.to_dict(),
            "parallelism": scale_payload,
            "compute_runner": runner_identity,
            "task_seeds": resolved_task_seeds,
        }, indent=2, sort_keys=True))
    events.emit(
        "run_started",
        run_id=run_id,
        benchmark=benchmark.name,
        version=benchmark.version,
        n_tasks=len(jobs),
        runs_per_task=runs_per_task,
        backend=plan.backend if plan is not None else backend.name,
        compute_runner=runner_identity,
        eval_algorithm=algorithm.key,
        run_plan_hash=plan.contract_hash if plan else None,
        env_pool_size=scale.environment_pool_size_per_evaluation,
        parallelism=scale_payload,
    )
    runtime = _LocalEvaluationRuntime(
        submission=submission,
        backend=backend,
        compute_runner=runner,
        run_dir=run_dir,
        run_id=run_id,
        events=events,
        capabilities=runtime_capabilities,
    )
    try:
        if runner.kind == "local":
            with evaluator_process_environment(evaluator_runtime_env):
                rows = algorithm.run(runtime, jobs, scale)
        else:
            rows = algorithm.run(runtime, jobs, scale)
    except BaseException as exc:
        events.emit("run_failed", error=repr(exc))
        runner.close()
        json_sink.close()
        raise

    init_duration = runtime.init_duration_sec
    runner.close()

    if selected_task_keys is not None:
        from cua_speedrun.scoring import collect_rows

        rows = collect_rows(run_dir)

    hardware = f"{platform.system()}-{platform.machine()}-{os.cpu_count()}cpu"
    if gpu:
        hardware += f"-{gpu}"
    result = {
        "run_id": run_id,
        "benchmark": {"name": benchmark.name, "version": benchmark.version},
        "submission_fingerprint": submission.fingerprint,
        "backend": plan.backend if plan is not None else backend.name,
        "harness_version": cua_speedrun.__version__,
        "gpu": gpu,
        "compute_runner": runner_identity,
        # The season key: every leaderboard record belongs to exactly one
        # frozen combination of these. Any change opens a new season and old
        # records freeze, so times are only ever compared within a season.
        "season": plan.season() if plan is not None else {
            "benchmark": f"{benchmark.name}@{benchmark.version}",
            "harness": cua_speedrun.__version__,
            "backend": backend.name,
            "hardware": hardware,
        },
        "init_duration_sec": init_duration,
        "run_plan": plan.to_dict() if plan is not None else None,
        "run_plan_hash": plan.contract_hash if plan is not None else None,
        "scoring_rules": plan.scoring_rules() if plan is not None else None,
        "parallelism": scale_payload,
        "task_seeds": resolved_task_seeds,
        "rows": rows,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    events.emit("run_done", run_dir=str(run_dir))
    json_sink.close()
    return run_dir


def _run_prepared(
    job: _LocalPreparedJob,
    events: EventBus,
    compute_runner: ComputeRunner,
    replica: Any,
) -> dict[str, Any]:
    gateway = None
    agent = None
    agent_stopped = False
    try:
        bind_host, advertised_host = compute_runner.gateway_hosts()
        gateway = Gateway(
            adapter=job.prepared.adapter,
            log=job.log,
            artifacts_dir=job.task_dir,
            timeout_sec=job.task.timeout_sec,
            grace_sec=job.task.grace_sec,
            checker=job.checker,
            host=bind_host,
            advertise_host=advertised_host,
        )
        env_url = gateway.start()
        # The clock starts here, immediately before the agent launches,
        # so there is no free thinking time between the two.
        gateway.arm()
        events.emit("armed", task_key=job.task_key)
        instruction = job.description
        with monitor_gateway_steps(
            gateway.status,
            lambda steps: events.emit(
                "task_progress", task_key=job.task_key, num_steps=steps
            ),
        ):
            agent_episode = 1
            while True:
                agent = compute_runner.start_agent(
                    replica,
                    env_url=env_url,
                    instruction=instruction,
                    task_key=job.task_key,
                    task_dir=job.task_dir,
                    timeout_sec=job.task.timeout_sec,
                    append_output=agent_episode > 1,
                )
                compute_runner.wait_for_gateway(replica, agent, gateway)
                infrastructure_error = gateway.status().get("infrastructure_error")
                if infrastructure_error:
                    raise InstanceInfrastructureError(str(infrastructure_error))
                result = compute_runner.stop_agent(replica, agent)
                agent_stopped = True
                status = gateway.status()
                episode = status.get("agent_episode", 1)
                if status.get("continuation_pending"):
                    episode -= 1
                job.log.event(
                    "agent_exit",
                    returncode=result.returncode,
                    agent_episode=episode,
                )
                events.emit(
                    "agent_exit",
                    task_key=job.task_key,
                    returncode=result.returncode,
                    agent_episode=episode,
                )
                if not status.get("continuation_pending"):
                    break
                continuation = gateway.continue_agent()
                instruction = str(continuation["instruction"])
                agent_episode = int(continuation["agent_episode"])
                agent = None
                agent_stopped = False
    except AgentExecutionError as exc:
        assert gateway is not None and agent is not None
        try:
            result = compute_runner.stop_agent(replica, agent)
            agent_stopped = True
            job.log.event("agent_exit", returncode=result.returncode)
            events.emit(
                "agent_exit", task_key=job.task_key, returncode=result.returncode
            )
            if gateway.fail_agent(str(exc)):
                events.emit(
                    "agent_failed",
                    task_key=job.task_key,
                    returncode=exc.returncode,
                    error=str(exc),
                )
            gateway.wait()
            infrastructure_error = gateway.status().get("infrastructure_error")
            if infrastructure_error:
                raise InstanceInfrastructureError(str(infrastructure_error))
        except Exception as failure_error:
            error = (
                failure_error
                if isinstance(failure_error, InstanceInfrastructureError)
                else InstanceInfrastructureError(
                    "evaluator failed while recording an agent failure: "
                    f"{type(failure_error).__name__}: {failure_error}"
                )
            )
            job.log.event("harness_error", error=repr(error))
            events.emit(
                "task_interrupted", task_key=job.task_key, error=repr(error)
            )
            if error is failure_error:
                raise
            raise error from failure_error
    except Exception as exc:
        infrastructure_error = (
            exc
            if isinstance(exc, InstanceInfrastructureError)
            else InstanceInfrastructureError(
                f"evaluator failed while running the instance: "
                f"{type(exc).__name__}: {exc}"
            )
        )
        job.log.event("harness_error", error=repr(infrastructure_error))
        events.emit(
            "task_interrupted",
            task_key=job.task_key,
            error=repr(infrastructure_error),
        )
        if infrastructure_error is exc:
            raise
        raise infrastructure_error from exc
    finally:
        if agent is not None and not agent_stopped:
            try:
                compute_runner.stop_agent(replica, agent)
            except Exception:
                pass
        cleanup_error = None
        if gateway is not None:
            try:
                gateway.shutdown()
            except Exception as exc:
                cleanup_error = InstanceInfrastructureError(
                    f"gateway cleanup failed: {type(exc).__name__}: {exc}"
                )
        try:
            job.close()
        except InstanceInfrastructureError as exc:
            cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            raise cleanup_error

    row = summarize(load_runlog(job.task_dir / "runlog.jsonl"))
    events.emit("task_done", task_key=job.task_key, **{
        key: row.get(key) for key in (
            "passed", "reason", "task_time_sec", "env_time_sec",
            "agent_time_sec", "num_steps",
        )
    })
    return row


class _LocalEvaluationRuntime:
    """Local process/VM implementation of the algorithm runtime contract."""

    def __init__(
        self,
        *,
        submission: Submission,
        backend: Any,
        compute_runner: ComputeRunner,
        run_dir: Path,
        run_id: str,
        events: EventBus,
        capabilities: frozenset[str] = LOCAL_RUNTIME_CAPABILITIES,
    ) -> None:
        self.submission = submission
        self.backend = backend
        self.compute_runner = compute_runner
        self.run_dir = run_dir
        self.run_id = run_id
        self.events = events
        self.capabilities = capabilities
        self.init_duration_sec = 0.0
        self.environment_driver = compute_runner.environment_driver
        self._replica_events: dict[int, Any] = {}

    def _events_for(self, replica: Any):
        if replica is None:
            return self.events
        if replica.index not in self._replica_events:
            parent = self.events
            replica_index = replica.index

            class _ReplicaEvents:
                def emit(
                    self,
                    kind: str,
                    task_key: str | None = None,
                    **payload: Any,
                ) -> None:
                    payload.setdefault("parallel_evaluation", replica_index)
                    parent.emit(kind, task_key=task_key, **payload)

            self._replica_events[replica.index] = _ReplicaEvents()
        return self._replica_events[replica.index]

    def run_isolated(self, job):
        """One fresh scheduled allocation runs one task instance, then ends.

        The allocation hosts the compute sandbox, the environment VM, and the
        gateway together, so its scheduler time limit covers exactly one
        instance. Any infrastructure failure surfaces as
        ``InstanceInfrastructureError`` and the algorithm retries with a
        brand-new allocation.
        """
        start_instance = getattr(
            self.compute_runner, "start_instance_replica", None
        )
        if start_instance is None or self.environment_driver is None:
            raise ValueError(
                "the in-process local runner does not implement isolated "
                "instances; per-task evaluations on the local topology need "
                "a scheduled compute runner"
            )
        task, seed = job
        replica = start_instance(task, seed)
        events = self._events_for(replica)
        try:
            events.emit(
                "task_started",
                task_key=f"{task.task_id}/seed_{seed}",
                task_id=task.task_id,
                seed=seed,
            )
            self.ensure_compute_ready(replica)
            prepared = self.prepare_environment(replica, job, 1)
            try:
                row = self.run_prepared(replica, prepared, 1)
            except BaseException:
                # The original error owns retry classification; a cleanup
                # failure on an allocation that is already lost must not
                # replace it.
                try:
                    self.close_environment_batch(replica, [prepared], 1)
                except Exception:
                    pass
                raise
            self.close_environment_batch(replica, [prepared], 1)
            return row
        finally:
            try:
                self.compute_runner.close_replica(replica)
            except Exception as exc:
                events.emit(
                    "compute_replica_close_failed",
                    error=repr(exc),
                    parallel_evaluation=replica.index,
                )

    def start_compute_replicas(self, shards):
        replicas = self.compute_runner.start_replicas(shards)
        self.init_duration_sec = self.compute_runner.init_duration_sec
        return replicas

    def ensure_compute_ready(self, replica):
        self.compute_runner.ensure_ready(replica)
        self.init_duration_sec = self.compute_runner.init_duration_sec

    def recover_instance_infrastructure(
        self,
        replica,
        error: InstanceInfrastructureError,
    ):
        if isinstance(error, ComputeInfrastructureError):
            self.compute_runner.replace_replica(replica, error)

    def compute_infrastructure_retry(
        self,
        replica,
        attempt,
        max_attempts,
        delay_sec,
        error,
    ):
        self._events_for(replica).emit(
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
        self._events_for(replica).emit(
            "compute_replica_failed",
            phase="compute_replica",
            attempts=attempts,
            error=repr(error),
        )

    def prepare_environment(self, replica, job, batch_number):
        task, seed = job
        if self.environment_driver is not None:
            prepared = self.environment_driver.prepare_environment(
                replica,
                backend_name=self.backend.name,
                task=task,
                seed=seed,
            )
            self._events_for(replica).emit(
                "env_ready",
                task_key=f"{task.task_id}/seed_{seed}",
                env_boot_sec=prepared.prepare_time_sec,
                environment_host=prepared.environment_hostname,
            )
            return prepared
        return _prepare_one(
            self.submission,
            self.backend,
            task,
            seed,
            self.run_dir,
            self.run_id,
            self._events_for(replica),
        )

    def begin_environment_batch(self, replica, jobs, batch_number):
        task, seed = jobs[0]
        task_key = f"{task.task_id}/seed_{seed}"
        self._events_for(replica).emit(
            "task_started",
            task_key=task_key,
            task_id=task.task_id,
            seed=seed,
        )

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
        task_key = f"{task.task_id}/seed_{seed}"
        task_dir = self.run_dir / "tasks" / task.task_id / f"seed_{seed}"
        archive = task_dir / f"_infra_attempt_{attempt}"
        archive.mkdir(parents=True, exist_ok=True)
        for child in list(task_dir.iterdir()):
            if child.name.startswith("_infra_attempt_"):
                continue
            child.replace(archive / child.name)
        self._events_for(replica).emit(
            "infra_retry",
            task_key=task_key,
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
        self._events_for(replica).emit(
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
        task_key = f"{task.task_id}/seed_{seed}"
        task_dir = self.run_dir / "tasks" / task.task_id / f"seed_{seed}"
        archive = task_dir / f"_instance_infra_attempt_{attempt}"
        archive.mkdir(parents=True, exist_ok=True)
        for child in list(task_dir.iterdir()):
            if child.name.startswith("_instance_infra_attempt_"):
                continue
            child.replace(archive / child.name)
        self._events_for(replica).emit(
            "infra_retry",
            task_key=task_key,
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
        self._events_for(replica).emit(
            "task_failed",
            task_key=f"{task.task_id}/seed_{seed}",
            phase="instance_run",
            attempts=attempts,
            error=repr(error),
        )

    def bind_compute(self, replica, prepared, batch_number):
        return None

    def run_prepared(self, replica, prepared, batch_number):
        if self.environment_driver is not None:
            return self.environment_driver.run_prepared_environment(
                replica,
                prepared,
                self._events_for(replica),
            )
        self.compute_runner.ensure_ready(replica)
        return _run_prepared(
            prepared,
            self._events_for(replica),
            self.compute_runner,
            replica,
        )

    def close_environment_batch(self, replica, prepared, batch_number):
        if self.environment_driver is not None:
            self.environment_driver.close_prepared_environments(
                replica,
                prepared,
            )
            return
        for job in prepared:
            job.close()

    def finish_compute(self, replica):
        self.compute_runner.finish_replica(replica)

    def close_compute_replica(self, replica):
        self.compute_runner.close_replica(replica)
