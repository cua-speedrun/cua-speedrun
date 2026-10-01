"""init.py for the GLM AutoGLM-V submission.

There is no local model server. The agent calls the Zhipu open platform's
OpenAI-compatible chat-completions endpoint directly, so init only verifies
that credentials are present and the endpoint answers before any task is
revealed.
"""

from __future__ import annotations

import sys

from agent import MODEL, glm_request


def main() -> None:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
        "max_tokens": 16,
    }
    try:
        data = glm_request(payload)
    except Exception as exc:
        sys.exit(f"GLM credential check failed: {exc}")
    print(f"{MODEL} credential check ok: id={data.get('id', '<none>')}")


if __name__ == "__main__":
    main()
