"""Reuse of init snapshots across runs, keyed by what actually built them.

init.py plus the filesystem snapshot is the expensive, untimed prologue of
every remote run: minutes for CPU submissions, tens of minutes for GPU ones,
and it used to repeat in full for every run of a byte-identical template.
The snapshot is a Modal Image that outlives the process, so a repeat run can
skip the whole prologue and spawn its task sandboxes straight from the
cached snapshot.

The cache key covers everything that shapes the snapshot: every file in the
submission directory, the harness source that gets baked into the agent
image, the versioned base-image recipe, the GPU type, extra pip packages, and
the frozen server environment. Callers represent user-supplied runtime values
with one-way hashes, so the cache index never stores plaintext. Any edit changes
the key, so a hit can only return a snapshot that byte-identical inputs produced.

A hit is validated by a probe, not trusted: the caller creates a throwaway
sandbox from the cached image before using it (Modal raises NotFoundError
for expired or deleted snapshots; filesystem snapshots carry a 30-day TTL).

Each snapshot has an atomic cache entry. Hosted controllers share the cache
directory on their persistent volume; other installations keep it beside runs.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

from cua_speedrun.specs import content_hash

_CS_SRC = Path(__file__).resolve().parents[1]  # src/cua_speedrun

CACHE_FILENAME = ".agent_snapshot_cache.json"

# The only harness files that exist inside an agent sandbox (agent.py's
# import surface is cua_speedrun.client and nothing else). The agent image
# ships exactly these, and the cache key hashes exactly these, so edits to
# the dashboard, worker, or executor never invalidate agent snapshots; only
# a change to the agent-side contract does. Keep in sync with
# modal_agent.build_agent_base_image, which imports this tuple.
AGENT_RUNTIME_FILES = ("__init__.py", "client.py")


def _tree_files(root: Path) -> list[Path]:
    return [p for p in Path(root).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
            and p.suffix != ".pyc"]


def compute_key(submission_dir: Path, gpu: str | None,
                extra_pip: list[str] | None,
                server_environment: dict[str, str] | None = None,
                agent_runtime: dict[str, Any] | None = None) -> str:
    """Hash of everything that shapes the snapshot. content_hash digests
    file names plus bytes, so the same submission extracted to a different
    temp directory (the platform worker's case) still hits the same key."""
    sub_hash = content_hash(_tree_files(submission_dir))
    harness_hash = content_hash([_CS_SRC / f for f in AGENT_RUNTIME_FILES])
    pip = ",".join(extra_pip or [])
    env = json.dumps(server_environment or {}, sort_keys=True, separators=(",", ":"))
    runtime = json.dumps(agent_runtime or {}, sort_keys=True, separators=(",", ":"))
    return (
        f"sub={sub_hash}|harness={harness_hash}|runtime={runtime}|"
        f"gpu={gpu}|pip={pip}|env={env}"
    )


def _entry_path(out_root: Path, key: str) -> Path:
    root = Path(os.environ.get("CS_AGENT_SNAPSHOT_CACHE_DIR") or Path(out_root) / ".agent_snapshots")
    return root / (hashlib.sha256(key.encode()).hexdigest() + ".json")


def lookup(out_root: Path, key: str) -> dict[str, Any] | None:
    try:
        entry = json.loads(_entry_path(out_root, key).read_text())
        if isinstance(entry, dict) and entry.get("image_id"):
            return entry
    except (OSError, json.JSONDecodeError):
        pass
    path = Path(out_root) / CACHE_FILENAME
    try:
        entries = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    entry = entries.get(key) if isinstance(entries, dict) else None
    return entry if isinstance(entry, dict) and entry.get("image_id") else None


def store(out_root: Path, key: str, image_id: str, run_id: str) -> None:
    path = _entry_path(out_root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "image_id": image_id,
        "run_id": run_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(entry))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
