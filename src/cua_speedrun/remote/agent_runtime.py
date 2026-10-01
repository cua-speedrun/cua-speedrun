"""Versioned platform image contract for Modal agent sandboxes.

This is the platform-owned layer below a user's ``init.py``.  It must be
explicit because changing the Python base, system libraries, or platform
packages can change whether the same submission starts and how it behaves.
The complete mapping is stored in ``RunPlan`` and participates in both the
season hash and the init-snapshot cache key.
"""

from __future__ import annotations

import importlib.metadata
import os
from typing import Any, Mapping


AGENT_RUNTIME_SCHEMA_VERSION = 1
AGENT_RUNTIME_RECIPE = "modal-debian-slim-py311-ffmpeg5@1"

# Pin direct platform packages.  Debian's ffmpeg package pins its matching
# libav* family; TorchCodec needs those shared objects at import time.
BASE_IMAGE = {
    "builder": "modal.Image.debian_slim",
    "python_version": "3.11",
    "observed_python_version": "3.11.12",
}
SYSTEM_PACKAGES = ("ffmpeg=7:5.1.9-0+deb12u1",)
PYTHON_PACKAGES = ("requests==2.34.2",)


def _modal_version() -> str:
    try:
        return importlib.metadata.version("modal")
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def current_agent_runtime_contract() -> dict[str, Any]:
    """Return the exact platform image recipe used for newly planned runs."""
    return {
        "schema_version": AGENT_RUNTIME_SCHEMA_VERSION,
        "recipe": AGENT_RUNTIME_RECIPE,
        "base_image": dict(BASE_IMAGE),
        "system_packages": list(SYSTEM_PACKAGES),
        "python_packages": list(PYTHON_PACKAGES),
        "modal_client_version": _modal_version(),
        "modal_image_builder_version": os.environ.get(
            "MODAL_IMAGE_BUILDER_VERSION", "default"
        ),
    }


def validate_agent_runtime_contract(runtime: Mapping[str, Any]) -> None:
    """Fail closed if this worker cannot reproduce a stored image recipe."""
    expected = current_agent_runtime_contract()
    actual = {
        "schema_version": runtime.get("schema_version"),
        "recipe": runtime.get("recipe"),
        "base_image": dict(runtime.get("base_image") or {}),
        "system_packages": list(runtime.get("system_packages") or ()),
        "python_packages": list(runtime.get("python_packages") or ()),
        "modal_client_version": runtime.get("modal_client_version"),
        "modal_image_builder_version": runtime.get(
            "modal_image_builder_version", "default"
        ),
    }
    if actual != expected:
        raise ValueError(
            "stored agent runtime is not reproducible by this worker: "
            f"stored={actual!r}, worker={expected!r}"
        )
