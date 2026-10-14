"""The two interfaces every environment backend implements.

EnvAdapter is one live environment: observe, act, finalize, close.
Backend is the factory: give it a task's env block and a seed, get back
a ready EnvAdapter and the seed-resolved task description.

Preparation (booting, checkpoint restore, applying task setup) happens
inside `prepare` and is never on the clock. The gateway owns all timing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(init=False)
class Observation:
    """One observation. `image` contains encoded PNG or JPEG bytes; `meta` is info
    such as resolution or, where a track allows it, an accessibility tree."""

    image: bytes
    meta: dict[str, Any]
    # Evaluator-only diagnostic measurements; not part of the agent metadata.
    timings: dict[str, float] = field(default_factory=dict)

    def __init__(self, image: bytes | None = None, meta: dict | None = None,
                 timings: dict | None = None, *, png: bytes | None = None):
        # Existing third-party adapters construct Observation(png=...).
        if image is not None and png is not None:
            raise ValueError("provide image or png, not both")
        self.image = image if image is not None else (png if png is not None else b"")
        self.meta = meta if meta is not None else {}
        self.timings = timings if timings is not None else {}

    @property
    def png(self) -> bytes:
        """Legacy PNG access; new consumers use image to avoid transcoding."""
        from cua_speedrun.images import as_png
        return as_png(self.image)


@dataclass
class Verdict:
    """A benchmark verdict with an independent 0-100 task score."""

    passed: bool
    score: float | None = None
    detail: str = ""


class EnvAdapter(ABC):
    """One live environment. Action dicts use gym-anything's schema:
    {"mouse": {...}}, {"keyboard": {...}}, {"action": "wait", "time": s}."""

    fast_io = False

    def step_observe(self, actions: list[dict[str, Any]]) -> tuple[dict[str, Any], Observation]:
        """Act and observe once; override when step already captures a frame."""
        return self.step(actions), self.observe()

    def flush_recording(self) -> None:
        """Drain backend-owned archival writes before artifact collection."""

    @abstractmethod
    def observe(self) -> Observation: ...

    def observe_png(self) -> Observation:
        """Legacy clients; PNG-native backends need no separate implementation."""
        return self.observe()

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
