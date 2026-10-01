"""Immutable, content-addressed execution contract for one evaluation.

The dashboard resolves a track and benchmark into a RunPlan before a run is
executed.  The exact plan is stored on the database row and in result.json;
its canonical hash is part of the season identity.  A resource, algorithm,
benchmark, seed-policy, or scoring change therefore opens a new season without
depending on somebody remembering to bump a display name.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import cua_speedrun
from cua_speedrun.eval_algorithms import resolve_eval_algorithm
from cua_speedrun.execution_placements import (
    execution_topology_for_backend,
    get_execution_topology,
)
from cua_speedrun.remote.agent_runtime import current_agent_runtime_contract
from cua_speedrun.remote.network_policy import api_domains_for_policy
from cua_speedrun.specs import Benchmark

RUN_PLAN_SCHEMA_VERSION = 6
SUPPORTED_RUN_PLAN_SCHEMAS = (1, 2, 3, 4, 5, RUN_PLAN_SCHEMA_VERSION)
RANKING_TIME_AT_SUCCESS_BAR = "time-at-success-bar@1"
SEED_SCORED_V1 = "scored-without-replacement@1"
SEED_PRACTICE_V1 = "practice-fixed@1"



def _source_layout() -> tuple[Path, Path]:
    package = Path(__file__).resolve().parent
    for candidate in (package, *package.parents):
        source = candidate / "src" / "cua_speedrun"
        if (candidate / "pyproject.toml").is_file() and source.is_dir():
            return candidate, source
    # A wheel/install has no repository root; hash the installed package
    # itself so content addressing still works outside an editable checkout.
    return package.parent, package


_ROOT, _CS_SRC = _source_layout()
_MEASUREMENT_PATHS = (
    "__init__.py",
    "algorithms/__init__.py",
    "cli.py",
    "client.py",
    "compute_runners",
    "eval_algorithms.py",
    "evaluation_runtime.py",
    "environment_instance.py",
    "execution_placements.py",
    "executor.py",
    "gateway.py",
    "leaderboard.py",
    "local_runtime.py",
    "parallelism.py",
    "runlog.py",
    "runplan.py",
    "runtime_environment.py",
    "scoring.py",
    "specs.py",
    "submission.py",
    "task_jobs.py",
    "envs",
    "remote",
    "service/seeds.py",
)
_IGNORED_CONTRACT_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _contract_files(root: Path):
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if (
            _IGNORED_CONTRACT_PARTS.intersection(path.parts)
            or path.suffix == ".pyc"
            or path.name == ".DS_Store"
        ):
            continue
        yield path


def _hash_roots(roots: list[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for label, root in sorted(roots, key=lambda item: item[0]):
        digest.update(label.encode())
        digest.update(b"\0")
        if not root.exists():
            digest.update(b"<missing>")
            continue
        for path in _contract_files(root):
            relative = path.name if root.is_file() else path.relative_to(root).as_posix()
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def benchmark_contract(benchmark_dir: Path) -> dict[str, Any]:
    benchmark = Benchmark.load(Path(benchmark_dir))
    # A compact benchmark source may resolve into an operator-local cache.
    # Hash the stable materialized package, never the source checkout path.
    benchmark_dir = benchmark.benchmark_dir.resolve()
    roots: list[tuple[str, Path]] = [("benchmark", benchmark_dir)]
    seen = {benchmark_dir}
    for task in benchmark.tasks:
        task_dir = Path(task.task_dir).resolve() if task.task_dir else None
        if (
            task_dir is not None
            and task_dir not in seen
            and not task_dir.is_relative_to(benchmark_dir)
        ):
            if not task_dir.exists():
                raise FileNotFoundError(
                    f"benchmark task directory does not exist: {task_dir}"
                )
            seen.add(task_dir)
            roots.append((f"task:{task.task_id}", task_dir))
        raw = task.env.get("env_dir") if isinstance(task.env, dict) else None
        if not raw:
            continue
        expanded_text = os.path.expanduser(os.path.expandvars(str(raw)))
        expanded_path = Path(expanded_text)
        if not expanded_path.is_absolute() and (_ROOT / expanded_path).exists():
            expanded_path = _ROOT / expanded_path
        expanded = expanded_path.resolve()
        if not expanded.exists():
            raise FileNotFoundError(
                f"benchmark task {task.task_id!r} environment does not exist: "
                f"{raw!r} resolved to {expanded}"
            )
        if expanded in seen or expanded.is_relative_to(benchmark_dir):
            continue
        seen.add(expanded)
        roots.append((f"environment:{task.task_id}", expanded))
    return {
        "name": benchmark.name,
        "version": benchmark.version,
        "content_hash": _hash_roots(roots),
        "task_ids": [task.task_id for task in benchmark.tasks],
    }


def harness_contract(eval_algorithm: str) -> dict[str, Any]:
    algorithm = resolve_eval_algorithm(eval_algorithm)
    algorithm_source = inspect.getsourcefile(algorithm.schedule)
    if algorithm_source is None:
        raise RuntimeError(
            f"cannot locate source for evaluation algorithm {algorithm.key!r}"
        )
    roots = [(name, _CS_SRC / name) for name in _MEASUREMENT_PATHS]
    roots.append((f"algorithm:{algorithm.key}", Path(algorithm_source).resolve()))
    vendored_gym_root = _ROOT / "third_party" / "gym-anything"
    if vendored_gym_root.is_dir():
        gym_root = vendored_gym_root
        gym_source = gym_root / "src" / "gym_anything"
    else:
        gym_spec = importlib.util.find_spec("gym_anything")
        gym_source = (
            Path(gym_spec.origin).resolve().parent
            if gym_spec is not None and gym_spec.origin
            else _ROOT / "<missing-gym-anything>"
        )
    gym_hash = _hash_roots([("gym-anything", gym_source)])
    try:
        gym_version = importlib.metadata.version("gym-anything")
    except importlib.metadata.PackageNotFoundError:
        # Published cua-speedrun wheels carry the pinned gym-anything package
        # directly, without pretending it is a separately installed
        # distribution. Its exact bytes are already contract-hashed.
        gym_version = f"bundled-sha256:{gym_hash[:16]}"
    return {
        "cua_speedrun_version": cua_speedrun.__version__,
        "measurement_hash": _hash_roots(roots),
        "gym_anything_version": gym_version,
        "gym_anything_hash": gym_hash,
    }


@dataclass(frozen=True)
class RunPlan:
    track_name: str
    benchmark: Mapping[str, Any]
    harness: Mapping[str, Any]
    backend: str = "modal-remote"
    gpu: str | None = None
    network_policy: str = "host-network"
    api_domain_allowlist: tuple[str, ...] = ()
    reference_only: bool = False
    ranking_rule: str = RANKING_TIME_AT_SUCCESS_BAR
    eval_algorithm: str = "per-task-vllm@1"
    # Schema 5+'s only track-owned parallelism value. The legacy concurrency
    # fields remain in memory solely so schema 1-4 plans round-trip exactly.
    agents_per_evaluation: int | None = None
    concurrency: int = 2
    env_pool_size: int = 2
    runs_per_task: int = 1
    success_bar: float = 0.9
    failure_costs_timeout: bool = False
    seed_policy: str = SEED_SCORED_V1
    server_config: Mapping[str, Any] = field(default_factory=dict)
    agent_runtime: Mapping[str, Any] = field(
        default_factory=current_agent_runtime_contract
    )
    # Schema 3 used one combined target. Kept only so an already-stored
    # schema-3 plan can round-trip byte-for-byte.
    execution_target: Mapping[str, Any] = field(
        default_factory=dict
    )
    execution_topology: Mapping[str, Any] = field(
        default_factory=lambda: execution_topology_for_backend(
            "modal-remote"
        ).contract_dict()
    )
    schema_version: int = RUN_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_RUN_PLAN_SCHEMAS:
            raise ValueError(
                f"unsupported run-plan schema {self.schema_version}; "
                f"supported schemas are {SUPPORTED_RUN_PLAN_SCHEMAS}"
            )
        algorithm = resolve_eval_algorithm(self.eval_algorithm)
        object.__setattr__(self, "benchmark", _freeze(self.benchmark))
        object.__setattr__(self, "harness", _freeze(self.harness))
        object.__setattr__(self, "server_config", _freeze(self.server_config))
        object.__setattr__(self, "agent_runtime", _freeze(self.agent_runtime))
        object.__setattr__(self, "execution_target", _freeze(self.execution_target))
        object.__setattr__(self, "execution_topology", _freeze(self.execution_topology))
        try:
            json.dumps(_thaw(self.server_config), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("server_config must be JSON-serializable") from exc
        try:
            json.dumps(_thaw(self.agent_runtime), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("agent_runtime must be JSON-serializable") from exc
        try:
            json.dumps(_thaw(self.execution_target), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("execution_target must be JSON-serializable") from exc
        try:
            json.dumps(_thaw(self.execution_topology), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("execution_topology must be JSON-serializable") from exc
        if self.schema_version >= 2:
            required_runtime_fields = {
                "schema_version",
                "recipe",
                "base_image",
                "system_packages",
                "python_packages",
                "modal_client_version",
                "modal_image_builder_version",
            }
            missing_runtime_fields = sorted(
                required_runtime_fields - set(self.agent_runtime)
            )
            if missing_runtime_fields:
                raise ValueError(
                    "agent_runtime is missing required fields: "
                    f"{missing_runtime_fields}"
                )
        if self.schema_version == 3:
            required_target_fields = {
                "key", "mode", "provider", "site", "backend",
                "requires_user_credentials",
            }
            missing_target_fields = sorted(
                required_target_fields - set(self.execution_target)
            )
            if missing_target_fields:
                raise ValueError(
                    "execution_target is missing required fields: "
                    f"{missing_target_fields}"
                )
            if self.execution_target["backend"] != self.backend:
                raise ValueError(
                    "execution target backend does not match the run backend: "
                    f"{self.execution_target['backend']!r} != {self.backend!r}"
                )
            if self.execution_target["mode"] not in ("local", "remote"):
                raise ValueError("execution target mode must be local or remote")
        if self.schema_version >= 4:
            required_topology_fields = {
                "key", "backend", "compute", "environment",
                "environment_backend", "requires_user_credentials",
            }
            # Schema 6 keeps the topology's discoverable algorithm list out
            # of the frozen contract. Adding an unrelated compatible plugin
            # must not open a new season for existing algorithms.
            if self.schema_version < 6:
                required_topology_fields.add("eval_algorithms")
            missing_topology_fields = sorted(
                required_topology_fields - set(self.execution_topology)
            )
            if missing_topology_fields:
                raise ValueError(
                    "execution_topology is missing required fields: "
                    f"{missing_topology_fields}"
                )
            if self.execution_topology["backend"] != self.backend:
                raise ValueError(
                    "execution topology backend does not match the run backend: "
                    f"{self.execution_topology['backend']!r} != {self.backend!r}"
                )
            for plane in ("compute", "environment"):
                placement = self.execution_topology[plane]
                required_placement_fields = {
                    "key", "mode", "provider", "requires_user_credentials",
                }
                missing = sorted(required_placement_fields - set(placement))
                if missing:
                    raise ValueError(
                        f"execution_topology.{plane} is missing fields: {missing}"
                    )
                if placement["mode"] not in ("local", "remote"):
                    raise ValueError(
                        f"execution_topology.{plane}.mode must be local or remote"
                    )
            registered_topology = get_execution_topology(
                str(self.execution_topology["compute"]["key"]),
                str(self.execution_topology["environment"]["key"]),
            )
            planned_topology = _thaw(self.execution_topology)
            planned_identity = {
                key: value for key, value in planned_topology.items()
                if key not in ("eval_algorithms", "runtime_capabilities")
            }
            if planned_identity != registered_topology.contract_dict():
                raise ValueError(
                    "execution_topology does not match its registered executor"
                )
            if not algorithm.supports(
                registered_topology.runtime_capabilities
            ):
                raise ValueError(
                    f"execution topology {self.execution_topology['key']!r} does "
                    f"not implement {self.eval_algorithm!r}"
                )
        environment = self.server_config.get("environment", {})
        if not isinstance(environment, Mapping):
            raise ValueError("server_config.environment must be an object")
        if any(
            not isinstance(value, (str, int, float, bool))
            for value in environment.values()
        ):
            raise ValueError(
                "server_config.environment values must be strings, numbers, or booleans"
            )
        sensitive_keys = [
            key for key in environment
            if str(key).upper().endswith(
                ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_CREDENTIALS")
            )
        ]
        if sensitive_keys:
            raise ValueError(
                "server_config is stored in public run artifacts and cannot "
                f"contain secret environment keys: {sorted(sensitive_keys)}"
            )
        extra_pip = self.server_config.get("extra_pip", ())
        if not isinstance(extra_pip, (list, tuple)) or any(
            not isinstance(package, str) or not package.strip()
            for package in extra_pip
        ):
            raise ValueError("server_config.extra_pip must be a list of packages")
        unpinned_extra_pip = [
            package for package in extra_pip if "==" not in package
        ]
        if unpinned_extra_pip:
            raise ValueError(
                "server_config.extra_pip packages must use exact == versions: "
                f"{unpinned_extra_pip}"
            )
        region = self.server_config.get("region")
        if region is not None and (not isinstance(region, str) or not region.strip()):
            raise ValueError("server_config.region must be a non-empty string or null")
        object.__setattr__(self, "eval_algorithm", algorithm.key)
        network_policy = self.network_policy.strip().lower()
        domains = api_domains_for_policy(network_policy, self.api_domain_allowlist)
        object.__setattr__(self, "network_policy", network_policy)
        object.__setattr__(self, "api_domain_allowlist", domains)
        if self.schema_version >= 5:
            agents_per_evaluation = int(
                self.agents_per_evaluation
                if self.agents_per_evaluation is not None
                else self.concurrency
            )
            if agents_per_evaluation < 1:
                raise ValueError("agents_per_evaluation must be at least 1")
            object.__setattr__(
                self, "agents_per_evaluation", agents_per_evaluation
            )
            # These aliases keep the existing executors able to replay both
            # old and new plans. They are derived and are not serialized in a
            # schema-5+ comparison contract.
            object.__setattr__(self, "concurrency", agents_per_evaluation)
            object.__setattr__(
                self,
                "env_pool_size",
                agents_per_evaluation * algorithm.default_env_pool_factor,
            )
        if self.concurrency < 1:
            raise ValueError("run-plan concurrency must be at least 1")
        if self.env_pool_size < self.concurrency:
            raise ValueError("env_pool_size cannot be smaller than concurrency")
        max_pool_size = self.concurrency * algorithm.default_env_pool_factor
        if self.env_pool_size > max_pool_size:
            raise ValueError(
                f"{algorithm.key} bounds env_pool_size at {max_pool_size} "
                f"for concurrency {self.concurrency}, got {self.env_pool_size}"
            )
        if self.runs_per_task < 1:
            raise ValueError("runs_per_task must be at least 1")
        if not self.benchmark.get("task_ids"):
            raise ValueError("run plan must contain at least one benchmark task")
        if not 0.0 <= self.success_bar <= 1.0:
            raise ValueError("success_bar must be between 0 and 1")
        if self.ranking_rule not in ("time-at-success-bar", RANKING_TIME_AT_SUCCESS_BAR):
            raise ValueError(f"unsupported ranking rule {self.ranking_rule!r}")
        object.__setattr__(self, "ranking_rule", RANKING_TIME_AT_SUCCESS_BAR)
        if self.seed_policy not in (SEED_SCORED_V1, SEED_PRACTICE_V1):
            raise ValueError(f"unsupported seed policy {self.seed_policy!r}")
        if self.seed_policy == SEED_PRACTICE_V1 and self.runs_per_task > 1000:
            raise ValueError("practice-fixed@1 has only the public seed band 0..999")

    def to_dict(self) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "track": {
                "name": self.track_name,
                "gpu": self.gpu,
                "network_policy": self.network_policy,
                "api_domain_allowlist": list(self.api_domain_allowlist),
                "reference_only": self.reference_only,
                "ranking_rule": self.ranking_rule,
                "server_config": _thaw(self.server_config),
            },
            "benchmark": _thaw(self.benchmark),
            "harness": _thaw(self.harness),
            "backend": self.backend,
            "execution": {
                "algorithm": self.eval_algorithm,
                "runs_per_task": self.runs_per_task,
            },
            "scoring": {
                "success_bar": self.success_bar,
                "failure_costs_timeout": self.failure_costs_timeout,
                "aggregation": "total-time@1",
            },
            "seeds": {"policy": self.seed_policy},
        }
        if self.schema_version >= 2:
            data["agent_runtime"] = _thaw(self.agent_runtime)
        if self.schema_version >= 5:
            data["execution"]["agents_per_evaluation"] = (
                self.agents_per_evaluation
            )
        else:
            data["execution"]["concurrency"] = self.concurrency
            data["execution"]["env_pool_size"] = self.env_pool_size
        if self.schema_version == 3:
            data["execution"]["target"] = _thaw(self.execution_target)
        if self.schema_version >= 4:
            data["execution"]["topology"] = _thaw(self.execution_topology)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunPlan":
        schema_version = int(data.get("schema_version", 1))
        track = data["track"]
        execution = data["execution"]
        scoring = data["scoring"]
        return cls(
            schema_version=schema_version,
            track_name=track["name"],
            benchmark=dict(data["benchmark"]),
            harness=dict(data["harness"]),
            backend=data.get("backend", "modal-remote"),
            gpu=track.get("gpu"),
            network_policy=track.get("network_policy", "gateway-only"),
            api_domain_allowlist=tuple(track.get("api_domain_allowlist") or ()),
            reference_only=bool(track.get("reference_only", False)),
            ranking_rule=track.get("ranking_rule", RANKING_TIME_AT_SUCCESS_BAR),
            server_config=dict(track.get("server_config") or {}),
            eval_algorithm=execution["algorithm"],
            agents_per_evaluation=(
                int(execution["agents_per_evaluation"])
                if schema_version >= 5
                else None
            ),
            concurrency=int(execution.get("concurrency", 1)),
            env_pool_size=int(execution.get("env_pool_size", 1)),
            runs_per_task=int(execution["runs_per_task"]),
            success_bar=float(scoring["success_bar"]),
            failure_costs_timeout=bool(scoring.get("failure_costs_timeout", False)),
            seed_policy=(data.get("seeds") or {}).get("policy", SEED_SCORED_V1),
            agent_runtime=(
                dict(data.get("agent_runtime") or {})
                if schema_version >= 2
                else {}
            ),
            execution_target=(
                dict(execution.get("target") or {})
                if schema_version == 3
                else {}
            ),
            execution_topology=(
                dict(execution.get("topology") or {})
                if schema_version >= 4
                else {}
            ),
        )

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def contract_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    def scoring_rules(self) -> dict[str, Any]:
        return {
            "success_bar": self.success_bar,
            "failure_costs_timeout": self.failure_costs_timeout,
        }

    def server_environment(self) -> dict[str, str]:
        environment = self.server_config.get("environment", {})
        runtime = {str(key): str(value) for key, value in environment.items()}
        runtime["CS_SERVER_CONFIG"] = json.dumps(
            _thaw(self.server_config), sort_keys=True, separators=(",", ":")
        )
        return runtime

    def resolved_agent_runtime(self) -> dict[str, Any]:
        """Runtime recipe for execution; schema-1 plans use the legacy
        current-worker fallback because they predate a stored image contract."""
        if self.schema_version < 2:
            return current_agent_runtime_contract()
        return _thaw(self.agent_runtime)

    def resolved_execution_topology(self) -> dict[str, Any]:
        """Return both placements, deriving them for legacy combined plans."""
        if self.schema_version < 4:
            return execution_topology_for_backend(self.backend).to_dict()
        return _thaw(self.execution_topology)

    def season(self) -> dict[str, str]:
        if self.schema_version >= 4:
            # Hardware belongs to the track. Placement is represented by the
            # backend and complete contract hash, not smuggled into this label.
            hardware = str(self.gpu or "CPU")
        else:
            prefix = "modal" if self.backend == "modal-remote" else "local"
            hardware = prefix + (f"-{self.gpu}" if self.gpu else "")
        algorithm = resolve_eval_algorithm(self.eval_algorithm)
        if algorithm.shared_agent_sandbox:
            agents = (
                self.agents_per_evaluation
                if self.schema_version >= 5
                else self.concurrency
            )
            hardware += f"-shared-agents{agents}"
        return {
            "benchmark": f"{self.benchmark['name']}@{self.benchmark['version']}",
            "harness": str(self.harness["cua_speedrun_version"]),
            "backend": self.backend,
            "hardware": hardware,
            "track": self.track_name,
            "algorithm": self.eval_algorithm,
            "contract": self.contract_hash[:16],
        }


def build_run_plan(
    *,
    track_name: str,
    benchmark_dir: Path,
    gpu: str | None = None,
    network_policy: str = "host-network",
    api_domain_allowlist: tuple[str, ...] | list[str] = (),
    reference_only: bool = False,
    ranking_rule: str = RANKING_TIME_AT_SUCCESS_BAR,
    eval_algorithm: str = "per-task-vllm@1",
    agents_per_evaluation: int | None = None,
    concurrency: int = 2,
    env_pool_size: int | None = None,
    runs_per_task: int = 1,
    success_bar: float = 0.9,
    failure_costs_timeout: bool = False,
    seed_policy: str = SEED_SCORED_V1,
    server_config: dict[str, Any] | None = None,
    agent_runtime: dict[str, Any] | None = None,
    execution_topology: dict[str, Any] | None = None,
    backend: str = "modal-remote",
    task_ids: list[str] | tuple[str, ...] | None = None,
) -> RunPlan:
    algorithm = resolve_eval_algorithm(eval_algorithm)
    pool = env_pool_size or concurrency * algorithm.default_env_pool_factor
    benchmark = benchmark_contract(benchmark_dir)
    if task_ids is not None:
        wanted = list(dict.fromkeys(task_ids))
        unknown = sorted(set(wanted) - set(benchmark["task_ids"]))
        if unknown:
            raise ValueError(f"unknown benchmark task ids: {unknown}")
        benchmark["task_ids"] = wanted
    requested_topology = dict(
        execution_topology
        or execution_topology_for_backend(backend).contract_dict()
    )
    topology = get_execution_topology(
        str(requested_topology["compute"]["key"]),
        str(requested_topology["environment"]["key"]),
    ).contract_dict()
    requested_identity = {
        key: value for key, value in requested_topology.items()
        if key not in ("eval_algorithms", "runtime_capabilities")
    }
    if requested_identity != topology:
        raise ValueError(
            "execution_topology does not match its registered executor"
        )
    backend = str(topology["backend"])
    return RunPlan(
        track_name=track_name,
        benchmark=benchmark,
        harness=harness_contract(algorithm.key),
        backend=backend,
        gpu=gpu,
        network_policy=network_policy,
        api_domain_allowlist=tuple(api_domain_allowlist),
        reference_only=reference_only,
        ranking_rule=ranking_rule,
        eval_algorithm=algorithm.key,
        agents_per_evaluation=(
            concurrency
            if agents_per_evaluation is None
            else agents_per_evaluation
        ),
        concurrency=concurrency,
        env_pool_size=pool,
        runs_per_task=runs_per_task,
        success_bar=success_bar,
        failure_costs_timeout=failure_costs_timeout,
        seed_policy=seed_policy,
        server_config=dict(server_config or {}),
        agent_runtime=dict(agent_runtime or current_agent_runtime_contract()),
        execution_topology=topology,
    )
