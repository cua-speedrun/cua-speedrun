"""Resolve compute-runner kind and live operator template."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOCAL_RUNNER = "local"
SLURM_RUNNER = "slurm"
KNOWN_RUNNERS = frozenset({LOCAL_RUNNER, SLURM_RUNNER})


@dataclass(frozen=True)
class RunnerSelection:
    kind: str
    template_path: Path | None = None

    def environment(self) -> dict[str, str]:
        values = {"CS_COMPUTE_RUNNER": self.kind}
        if self.template_path is not None:
            values["CS_COMPUTE_RUNNER_TEMPLATE"] = str(self.template_path)
        return values


def _template_path(kind: str, requested: str) -> Path:
    supplied = Path(requested).expanduser()
    if supplied.is_file():
        return supplied.resolve()
    if supplied.is_absolute() or len(supplied.parts) > 1:
        raise FileNotFoundError(f"runner template does not exist: {supplied}")
    filename = requested if requested.endswith((".yaml", ".yml")) else f"{requested}.yaml"
    bundled = Path(__file__).resolve().parent / "templates" / kind / filename
    if bundled.is_file():
        return bundled.resolve()
    known_dir = bundled.parent
    known = ", ".join(path.stem for path in sorted(known_dir.glob("*.yaml")))
    raise FileNotFoundError(
        f"unknown {kind} runner template {requested!r}"
        + (f"; bundled templates: {known}" if known else "")
    )


def resolve_runner_selection(
    runner: str | None = None,
    runner_template: str | Path | None = None,
) -> RunnerSelection:
    kind = (runner or os.environ.get("CS_COMPUTE_RUNNER") or LOCAL_RUNNER).strip().lower()
    if kind not in KNOWN_RUNNERS:
        raise ValueError(
            f"unknown compute runner {kind!r}; known: {', '.join(sorted(KNOWN_RUNNERS))}"
        )
    requested = str(
        runner_template
        or os.environ.get("CS_COMPUTE_RUNNER_TEMPLATE", "")
    ).strip()
    if kind == LOCAL_RUNNER:
        if requested:
            raise ValueError("--runner-template is only valid with a templated runner")
        return RunnerSelection(kind=kind)
    if not requested:
        raise ValueError(f"compute runner {kind!r} requires --runner-template")
    return RunnerSelection(kind=kind, template_path=_template_path(kind, requested))


def local_runtime_overrides(
    selection: RunnerSelection, gpu: str | None
) -> dict[str, Any]:
    """Describe the compute host without probing the coordinator's GPUs."""
    if selection.kind == LOCAL_RUNNER:
        return {}
    return {
        "accelerator": {
            "kind": "scheduler-request",
            "devices": [] if gpu is None else [gpu],
        },
        "execution_scope": "scheduled-compute",
    }


__all__ = [
    "KNOWN_RUNNERS",
    "LOCAL_RUNNER",
    "RunnerSelection",
    "SLURM_RUNNER",
    "local_runtime_overrides",
    "resolve_runner_selection",
]
