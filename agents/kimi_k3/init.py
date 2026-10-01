"""Untimed OpenRouter credential check for the Kimi K3 submission."""

from __future__ import annotations

import sys

from agent import MODEL, build_payload, kimi_request


def main() -> None:
    payload = build_payload(
        [{"role": "user", "content": "Reply with exactly: ready"}]
    )
    # Credential validation does not execute desktop actions. Requiring a
    # computer tool here lets K3 legitimately return content=None, which is
    # not useful for this plain readiness check.
    payload.pop("tools", None)
    payload.pop("tool_choice", None)
    # Credential validation does not need the task agent's expensive maximum
    # reasoning setting; K3 always thinks, so use its lowest effort here.
    payload["max_tokens"] = 256
    payload["reasoning"] = {"effort": "low", "exclude": False}
    try:
        message = kimi_request(payload)
    except Exception as exc:
        sys.exit(f"Kimi K3 credential check failed: {exc}")
    content = message.get("content")
    preview = content[:80] if isinstance(content, str) else "ready"
    print(f"{MODEL} credential check ok: {preview}")


if __name__ == "__main__":
    main()
