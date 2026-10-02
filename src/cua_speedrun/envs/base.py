"""The two interfaces every environment backend implements.

EnvAdapter is one live environment: observe, act, finalize, close.
Backend is the factory: give it a task's env block and a seed, get back
a ready EnvAdapter and the seed-resolved task description.

Preparation (booting, checkpoint restore, applying task setup) happens
inside `prepare` and is never on the clock. The gateway owns all timing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class Observation:
    """One observation. `png` is the screenshot bytes, `meta` is extra info
    such as resolution or, where a track allows it, an accessibility tree."""

    png: bytes
    meta: dict[str, Any]


@dataclass
class Verdict:
    """A benchmark verdict with an independent 0-100 task score."""

    passed: bool
    score: float | None = None
    detail: str = ""


class EnvAdapter(ABC):
    """One live environment. Action dicts use gym-anything's schema:
    {"mouse": {...}}, {"keyboard": {...}}, {"action": "wait", "time": s}."""

    @abstractmethod
    def observe(self) -> Observation: ...

    @abstractmethod
    def step(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        """Inject a batch of actions. Returns backend info (may be empty)."""

    @abstractmethod
    def finalize(self) -> Verdict:
        """Run the task's checker against the current environment state.
        Called once, by the gateway, after the run ends plus grace."""

    def read_state(self) -> str:
        """Return a textual snapshot of on-screen state (e.g. a UI tree),
        for a host-side seeded checker to inspect. Optional; backends that
        support it override this."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support read_state; "
            "seeded tasks that inspect the screen need it"
        )

    def accessibility_tree(self) -> dict[str, Any]:
        """Return the front window's accessibility tree for an agent that asks
        for it with its observation. Optional; an agent falls back to reading
        the screenshot on backends that do not support it."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support accessibility trees"
        )

    def exec_read(self, command: str) -> str:
        """Run a shell command inside the environment and return its output,
        for a host-side seeded checker that inspects filesystem or command
        state rather than the screen. Optional. The command and its expected
        result live host-side, so the answer never touches the desktop."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support exec_read; "
            "seeded tasks that inspect shell/filesystem state need it"
        )

    @abstractmethod
    def close(self) -> None: ...


@dataclass
class PreparedEnv:
    adapter: EnvAdapter
    description: str
    prepare_time_sec: float
    info: dict[str, Any]
    # A backend whose verifier runs host-side (for example modal-native's
    # OSWorld checker) provides it here. None means the adapter's own
    # finalize() is the verdict.
    checker: Callable[[], Verdict] | None = None


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def prepare(self, env_spec: dict[str, Any], seed: int, workdir: Path) -> PreparedEnv:
        """Bring up one environment for (task, seed), untimed.

        env_spec is the task's `env` block, opaque to the core. The
        returned description is the task description with all
        seed-dependent blanks filled in.
        """
