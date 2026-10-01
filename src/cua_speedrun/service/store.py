"""Artifact storage behind a two-method interface.

Local filesystem now; an S3/R2 adapter later without touching callers.
Submission zips live here. Run directories (the ground-truth logs) are
written by the executor under the runs root and referenced by relative
path, so "artifact ref" is stable across store backends.
"""

from __future__ import annotations

import secrets
from pathlib import Path


class LocalStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, data: bytes, suffix: str = "") -> str:
        """Store a blob, return its ref."""
        ref = f"blob_{secrets.token_hex(8)}{suffix}"
        (self.root / ref).write_bytes(data)
        return ref

    def path(self, ref: str) -> Path:
        p = (self.root / ref).resolve()
        if not str(p).startswith(str(self.root.resolve())):
            raise ValueError(f"ref escapes the store: {ref!r}")
        return p
