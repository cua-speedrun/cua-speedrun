"""Untimed credential check for the OpenAI Responses API submission."""

from __future__ import annotations

import sys

from agent import MODEL, CostLimitReached, CostTracker, responses_request


def main() -> None:
    payload = {
        "model": MODEL,
        "input": "Reply with exactly: ready",
        "reasoning": {"effort": "none"},
        "max_output_tokens": 16,
    }
    try:
        response = responses_request(payload, CostTracker(), "init")
    except CostLimitReached as exc:
        sys.exit(f"OpenAI credential check skipped: {exc}")
    except Exception as exc:
        sys.exit(f"OpenAI credential check failed: {exc}")
    print(f"{MODEL} credential check ok: response={response.get('id', '<none>')}")


if __name__ == "__main__":
    main()
