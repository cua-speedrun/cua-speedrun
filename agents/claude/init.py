"""Install image support and verify Claude credentials before timed tasks."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys

PILLOW_VERSION = "11.3.0"


def ensure_pillow() -> None:
    try:
        installed = importlib.metadata.version("Pillow")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed == PILLOW_VERSION:
        return
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            f"Pillow=={PILLOW_VERSION}",
        ],
        env=env,
    )


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required")
    ensure_pillow()
    from agent import EFFORT, MODEL, anthropic_request, supports_effort

    payload = {
        "model": MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
    }
    if EFFORT and supports_effort(MODEL):
        payload["output_config"] = {"effort": EFFORT}
    try:
        response = anthropic_request(payload)
    except Exception as exc:
        raise SystemExit(f"Claude credential check failed: {exc}") from exc
    print(f"{MODEL} credential check ok: message={response.get('id', '<none>')}")


if __name__ == "__main__":
    main()
