"""init.py for the GLM (z.ai via OpenRouter) GUI-agent submission.

There is no local model server. The agent calls OpenRouter's Chat
Completions endpoint directly, so init only verifies that credentials are
present and the endpoint answers before any task is revealed.

NOTE: the live credential check has not been exercised yet -- no
OPENROUTER_API_KEY was available in the development environment.
"""

from __future__ import annotations

import sys

import requests

from agent import API_URL, MODEL, api_key


def main() -> None:
    try:
        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {api_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
                "max_tokens": 8,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        sys.exit(f"GLM credential check failed: {exc}")
    print(f"{MODEL} credential check ok: id={data.get('id', '<none>')}")


if __name__ == "__main__":
    main()
