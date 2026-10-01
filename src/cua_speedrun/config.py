"""Shared process configuration: .env loading.

Every entrypoint (CLI, service worker, enqueue, API) needs the same three
things from .env: Modal credentials and GYM_ANYTHING_ROOT for expanding
task env blocks. Existing environment variables always win.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | str = ".env") -> None:
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
