"""Provider-neutral validation for variables exposed to submission code."""

from __future__ import annotations

import re
from typing import Any

ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
RESERVED_ENVIRONMENT_NAMES = {
    "HOME",
    "PATH",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
}
MAX_ENVIRONMENT_VARIABLES = 64
MAX_ENVIRONMENT_VALUE_LENGTH = 16384


def normalize_environment_name(name: Any) -> str:
    if not isinstance(name, str):
        raise ValueError("environment-variable names must be strings")
    name = name.strip()
    if (
        not ENVIRONMENT_NAME.fullmatch(name)
        or name.startswith("CS_")
        or name.startswith("MODAL_")
        or name in RESERVED_ENVIRONMENT_NAMES
    ):
        raise ValueError(
            "variable names must start with an uppercase letter, contain only "
            "uppercase letters, digits, and underscores, and cannot replace "
            "cua-speedrun or execution-runtime variables"
        )
    return name


def normalize_environment_value(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("environment-variable values must be strings")
    if not value or len(value) > MAX_ENVIRONMENT_VALUE_LENGTH:
        raise ValueError(
            "environment-variable values must contain 1 to 16384 characters"
        )
    return value
