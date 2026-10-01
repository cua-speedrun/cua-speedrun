"""init.py for the Gemini 3.5 Computer Use submission.

There is no local model server. The agent calls the Gemini Interactions API
directly, so init only verifies that credentials are present and the endpoint
answers before any task is revealed.
"""

from __future__ import annotations

import sys

from agent import MODEL, THINKING_LEVEL, CostLimitReached, CostTracker, gemini_request


def main() -> None:
    payload = {
        "model": MODEL,
        "input": "Reply with exactly: ready",
        "generation_config": {
            "temperature": 0,
            "max_output_tokens": 5,
            "thinking_level": THINKING_LEVEL,
        },
    }
    try:
        data = gemini_request(payload, CostTracker(), "init")
    except CostLimitReached as exc:
        sys.exit(f"Gemini credential check skipped: {exc}")
    except Exception as exc:
        sys.exit(f"Gemini credential check failed: {exc}")
    print(f"{MODEL} credential check ok: interaction={data.get('id', '<none>')}")


if __name__ == "__main__":
    main()
