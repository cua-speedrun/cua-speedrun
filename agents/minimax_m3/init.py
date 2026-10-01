"""init.py for the MiniMax M3 submission.

There is no local model server. The agent calls MiniMax's Anthropic-compatible
Messages endpoint directly, so init only verifies that credentials are present
and the endpoint answers before any task is revealed.
"""

from __future__ import annotations

import sys

from agent import MODEL, anthropic_request


def main() -> None:
    body = {
        "model": MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
    }
    try:
        data = anthropic_request(body)
    except Exception as exc:
        sys.exit(f"MiniMax credential check failed: {exc}")
    print(f"{MODEL} credential check ok: id={data.get('id', '<none>')}")


if __name__ == "__main__":
    main()
