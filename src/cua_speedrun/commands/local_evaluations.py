"""Direct CLI access to one installed cua-speedrun evaluation service."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from sqlalchemy import select

from cua_speedrun.execution_placements import get_execution_topology
from cua_speedrun.service.db import BenchmarkRow, Run, Track, make_session_factory
from cua_speedrun.service.evaluations import (
    benchmark_id_for_name,
    cancel_evaluation,
    ensure_local_user,
    get_evaluation,
    list_evaluations,
    queue_evaluation,
    resume_evaluation,
)
from cua_speedrun.service.store import LocalStore
from cua_speedrun.compute_runners import resolve_runner_selection

from .paths import InstallationPaths, configure_process


@dataclass(frozen=True)
class EvaluationLaunch:
    runner: str
    log_path: Path
    external_id: str | None = None


@dataclass(frozen=True)
class LocalEvaluations:
    paths: InstallationPaths
    session_factory: Any
    store: LocalStore
    user_id: int

    @classmethod
    def open(cls, home: str | Path | None = None) -> "LocalEvaluations":
        paths = InstallationPaths.resolve(home)
        if not paths.install_record.is_file():
            raise RuntimeError(
                f"cua-speedrun is not installed at {paths.home}; run "
                f"`cua-speedrun install --home {paths.home}` first, or pass "
                "--dashboard to target another installation"
            )
        configure_process(paths)
        session_factory = make_session_factory()
        from cua_speedrun.service.catalog import sync_catalog

        sync_catalog(session_factory)
        return cls(
            paths=paths,
            session_factory=session_factory,
            store=LocalStore(paths.store),
            user_id=ensure_local_user(session_factory),
        )

    def catalog(self) -> dict[str, Any]:
        from cua_speedrun.service.templates_catalog import list_templates

        with self.session_factory() as session:
            tracks = session.scalars(select(Track).order_by(Track.name)).all()
            benchmarks = session.scalars(
                select(BenchmarkRow).where(BenchmarkRow.active).order_by(BenchmarkRow.name, BenchmarkRow.version)
            ).all()
        return {
            "tracks": [{
                "name": track.name,
                "gpu": track.gpu,
                "eval_algorithm": track.eval_algorithm,
                "agents_per_evaluation": track.agents_per_evaluation,
                "reference_only": track.reference_only,
            } for track in tracks],
            "benchmarks": [{
                "id": benchmark.id,
                "name": benchmark.name,
                "version": benchmark.version,
                "task_count": benchmark.task_count,
            } for benchmark in benchmarks],
            "templates": list_templates(),
        }

    def submit(
        self,
        *,
        submission_zip: bytes,
        name: str,
        track: str,
        benchmark: str,
        compute: str,
        environment: str,
        allocate_gpu: bool,
        parallel_evaluations: int,
        saved_environment_names: Iterable[str],
        evaluation_environment: Mapping[str, str],
        required_environment_names: Iterable[str] = (),
        gpu: str | None = None,
        eval_algorithm: str | None = None,
        runner: str | None = None,
        runner_template: str | None = None,
    ) -> dict[str, Any]:
        self._sync_modal_credentials_from_process(compute, environment)
        return queue_evaluation(
            runner=runner,
            runner_template=runner_template,
            session_factory=self.session_factory,
            store=self.store,
            user_id=self.user_id,
            submission_zip=submission_zip,
            name=name,
            track_name=track,
            benchmark_id=benchmark_id_for_name(
                self.session_factory, benchmark
            ),
            compute_placement=compute,
            environment_placement=environment,
            allocate_gpu=allocate_gpu,
            gpu=gpu,
            eval_algorithm=eval_algorithm,
            parallel_evaluations=parallel_evaluations,
            saved_environment_names=saved_environment_names,
            evaluation_environment=evaluation_environment,
            required_environment_names=required_environment_names,
        )

    def _sync_modal_credentials_from_process(
        self, compute: str, environment: str
    ) -> None:
        """Persist a local CLI's Modal pair for its detached worker.

        Provider credentials remain evaluator-only: they are stored on the
        local operator identity, never added to the environment snapshot that
        is exposed to submission code.
        """
        topology = get_execution_topology(compute, environment)
        if not topology.requires_user_credentials:
            return

        from .setup import modal_credentials

        token_id, token_secret = modal_credentials()
        if not token_id and not token_secret:
            return
        if not token_id or not token_secret:
            raise ValueError(
                "MODAL_TOKEN_ID and MODAL_TOKEN_SECRET must both be set for "
                "a remote Modal evaluation"
            )
        if not token_id.startswith("ak-") or not token_secret.startswith("as-"):
            raise ValueError(
                "invalid Modal token pair: MODAL_TOKEN_ID must start with "
                "ak- and MODAL_TOKEN_SECRET must start with as-"
            )

        from cua_speedrun.service.db import User
        from cua_speedrun.service.usersecrets import encrypt

        with self.session_factory() as session:
            user = session.get(User, self.user_id)
            if user is None:
                raise RuntimeError("local cua-speedrun user no longer exists")
            user.modal_token_id = token_id
            user.modal_token_secret_enc = encrypt(token_secret)
            session.commit()

    def start(
        self,
        run_id: int,
        runner: str | None = None,
        runner_template: str | Path | None = None,
    ) -> EvaluationLaunch:
        """Start a detached supervisor for exactly this queued evaluation."""
        with self.session_factory() as session:
            run = session.get(Run, run_id)
            selected = (
                dict(getattr(run, "runner_selection", None) or {})
                if run is not None
                else {}
            )
            run_dir = Path(run.run_dir) if run is not None and run.run_dir else None
            if run_dir is not None and not run_dir.is_absolute():
                run_dir = self.paths.home / run_dir
        # Precedence: explicit arguments, then the selection frozen at
        # submit, then the runner recorded by a previous execution.
        if runner is None and runner_template is None and selected:
            runner = selected.get("kind")
            runner_template = selected.get("template")
        plan_path = run_dir / "run_plan.json" if run_dir is not None else None
        stored = (
            json.loads(plan_path.read_text()).get("compute_runner") or {}
            if plan_path is not None and plan_path.is_file()
            else {}
        )
        if runner is None and runner_template is None and stored:
            runner = stored.get("kind")
            runner_template = stored.get("template")
        selection = resolve_runner_selection(runner, runner_template)
        log_path = self.paths.logs / f"evaluation-{run_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(selection.environment())
        with log_path.open("ab") as output:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "cua_speedrun.service.worker",
                    "--process-run",
                    str(run_id),
                ],
                cwd=self.paths.home,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        time.sleep(0.2)
        return_code = process.poll()
        if return_code not in (None, 0):
            tail = log_path.read_text(errors="replace")[-2000:].strip()
            raise RuntimeError(
                f"evaluation worker exited with status {return_code}: {tail}"
            )
        return EvaluationLaunch(
            runner=selection.kind,
            log_path=log_path,
            external_id=str(process.pid),
        )

    def status(self, run_id: int) -> dict[str, Any]:
        return get_evaluation(self.session_factory, run_id, self.user_id)

    def evaluations(
        self, *, active_only: bool = False, limit: int | None = None
    ) -> list[dict[str, Any]]:
        return list_evaluations(
            self.session_factory,
            self.user_id,
            active_only=active_only,
            limit=limit,
        )

    def cancel(self, run_id: int) -> dict[str, Any]:
        payload = cancel_evaluation(self.session_factory, run_id, self.user_id)
        deadline = time.monotonic() + 15
        current = payload
        while current["stage"] != "cancelled" and time.monotonic() < deadline:
            current = self.status(run_id)
            if current["stage"] == "cancelled":
                break
            time.sleep(0.1)

        # The supervisor must stop the evaluator before provider resources are
        # swept. Cancelling Slurm jobs first makes a still-running evaluator
        # mistake an intentional stop for infrastructure loss and replace them.
        from cua_speedrun.service.cancellation import cleanup_remote_run

        if not cleanup_remote_run(self.session_factory, run_id):
            raise RuntimeError(
                f"evaluation {run_id} requested cancellation, but one or "
                "more provider resources could not be terminated"
            )
        return current

    def resume(
        self,
        run_id: int,
        *,
        rerun_agent_failures: bool = False,
    ) -> dict[str, Any]:
        return resume_evaluation(
            self.session_factory,
            run_id,
            self.user_id,
            installation_root=self.paths.home,
            rerun_agent_failures=rerun_agent_failures,
        )


__all__ = ["LocalEvaluations"]
