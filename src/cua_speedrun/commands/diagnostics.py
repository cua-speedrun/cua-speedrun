"""Small data model shared by doctor capability probes."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


@dataclass(frozen=True)
class Capability:
    key: str
    title: str
    checks: tuple[Check, ...]

    @property
    def ready(self) -> bool:
        return all(check.ok or not check.required for check in self.checks)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "ready": self.ready,
            "checks": [asdict(check) for check in self.checks],
        }


def module_check(name: str, import_name: str, *, required: bool = True) -> Check:
    ready = importlib.util.find_spec(import_name) is not None
    return Check(
        name,
        ready,
        f"Python module {import_name} {'available' if ready else 'missing'}",
        required,
    )


__all__ = ["Capability", "Check", "module_check"]
