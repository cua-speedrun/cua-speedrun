"""Locate the maintainer-owned data that ships with cua-speedrun.

Source checkouts keep this data at the repository root.  Built wheels place
the same files below ``cua_speedrun/_resources``.  Operator commands may copy
them into ``CUA_SPEEDRUN_HOME`` and select that copy with ``CS_RESOURCE_ROOT``.
Keeping this decision here prevents the dashboard, catalog, and benchmark
materializers from each inventing their own installation-layout rules.
"""

from __future__ import annotations

import os
from pathlib import Path


def _looks_like_resource_root(path: Path) -> bool:
    return (
        (path / "catalog" / "tracks.yaml").is_file()
        and (path / "benchmarks").is_dir()
        and (path / "agents").is_dir()
        and (path / "scripts").is_dir()
    )


def source_checkout_root() -> Path | None:
    """Return the repository root when this package runs from a checkout."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and _looks_like_resource_root(
            candidate
        ):
            return candidate
    return None


def bundled_resource_root() -> Path:
    """Return the immutable resources belonging to this package version."""
    checkout = source_checkout_root()
    if checkout is not None:
        return checkout
    packaged = Path(__file__).resolve().parent / "_resources"
    if _looks_like_resource_root(packaged):
        return packaged
    raise FileNotFoundError(
        "cua-speedrun resources are missing; reinstall the package"
    )


def resource_root() -> Path:
    """Return the active resource tree for this process."""
    configured = os.environ.get("CS_RESOURCE_ROOT", "").strip()
    if configured:
        path = Path(configured).expanduser().resolve()
        if not _looks_like_resource_root(path):
            raise FileNotFoundError(
                f"CS_RESOURCE_ROOT is not an installed resource tree: {path}"
            )
        return path
    return bundled_resource_root()


def bundled_gym_anything_root() -> Path:
    """Locate gym-anything data in either a checkout or a wheel."""
    checkout = source_checkout_root()
    if checkout is not None:
        candidate = checkout / "third_party" / "gym-anything"
    else:
        candidate = bundled_resource_root() / "gym-anything"
    if not (candidate / "benchmarks" / "cua_world" / "environments").is_dir():
        raise FileNotFoundError(
            "the bundled gym-anything environment data is missing"
        )
    return candidate


__all__ = [
    "bundled_gym_anything_root",
    "bundled_resource_root",
    "resource_root",
    "source_checkout_root",
]
