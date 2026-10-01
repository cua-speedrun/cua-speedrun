"""The shared serving half of the env planes.

An env plane is transport, nothing more: it reads the launcher's CS_*
variables, asks a registered Backend to prepare the environment, and hands
the PreparedEnv to ``serve_env_plane`` here. Every preparation decision
(runner selection, guards, resets, scrubbing, instruction resolution) lives
in the Backend, shared with the local executor, so the two paths cannot
drift apart again.

Launcher contract, common to every plane:

    CS_ENV_BACKEND                  registry name for cua_speedrun.envs.get_backend
    CS_ENV_SPEC                     the task's env block as JSON (env_dir
                                    rewritten to the in-sandbox mount)
    CS_SEED                         the seed
    CS_TIMEOUT_SEC, CS_GRACE_SEC    timing
    CS_RUN_TOKEN, CS_CONTROL_TOKEN  preset tokens
    CS_GATEWAY_PORT                 port to bind (exposed as a tunnel)
    CS_TASK_LABEL                   cua-speedrun task label for the run log
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from cua_speedrun.envs.base import PreparedEnv, Verdict

WORKDIR = Path("/tmp/cs_run")


def read_env_spec() -> dict[str, Any]:
    return json.loads(os.environ["CS_ENV_SPEC"])


def serve_env_plane(
    prepared: PreparedEnv,
    *,
    workdir: Path,
    plane: str,
    instruction: str | None = None,
    checker: Callable[[], Verdict] | None = None,
    seeded: bool = False,
) -> None:
    """Serve one prepared environment on the gateway tunnel, forever.

    ``instruction`` and ``checker`` default to what the backend prepared; a
    seeded-generator plane overrides both and sets ``seeded``.
    """
    from cua_speedrun.gateway import Gateway
    from cua_speedrun.runlog import RunLogWriter

    seed = int(os.environ.get("CS_SEED", "0"))
    timeout_sec = float(os.environ.get("CS_TIMEOUT_SEC", "300"))
    grace_sec = float(os.environ.get("CS_GRACE_SEC", "1.5"))
    port = int(os.environ["CS_GATEWAY_PORT"])
    run_token = os.environ["CS_RUN_TOKEN"]
    control_token = os.environ["CS_CONTROL_TOKEN"]
    # Prefer the cua-speedrun task label (scoring/leaderboard consistency)
    # over the backend's own task id when the launcher provides it.
    task_label = (
        os.environ.get("CS_TASK_LABEL")
        or str(prepared.info.get("task_id") or "")
    )

    resolved_instruction = (
        instruction if instruction is not None else prepared.description
    )
    resolved_checker = checker if checker is not None else prepared.checker

    log = RunLogWriter(workdir / "runlog.jsonl")
    log.event(
        "header", run_id=plane, task_id=task_label, seed=seed,
        timeout_sec=timeout_sec, instruction=resolved_instruction,
        seeded=seeded,
        # What the backend actually prepared (selected runner, base image,
        # accessibility state). The local path has always logged this; the
        # remote planes did not, so a degraded desktop was invisible after
        # the fact and could only be inferred from a wrong verdict.
        env_info=prepared.info,
        prepare_time_sec=prepared.prepare_time_sec,
    )
    gateway = Gateway(
        adapter=prepared.adapter, log=log, artifacts_dir=workdir,
        timeout_sec=timeout_sec, grace_sec=grace_sec, checker=resolved_checker,
        host="0.0.0.0", port=port, run_token=run_token,
        control_token=control_token, instruction=resolved_instruction,
    )
    gateway.start()
    print(f"[{plane}] gateway serving on :{port} (ready)", flush=True)

    # Serve until the executor tears the sandbox down. Block the main thread.
    while True:
        time.sleep(1)
