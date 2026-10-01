"""init.py for the Meta super_nova_ext GUI submission.

There is no local model server. The agent calls Meta's OpenAI-compatible
relay directly, so init only verifies that credentials are present and the
endpoint answers before any task is revealed.
"""

from __future__ import annotations

import sys

from agent import MODEL, History, meta_request


def main() -> None:
    history = History()
    history.messages.append({"role": "user", "content": "Reply with exactly: ready"})
    try:
        data = meta_request(history, include_tools=False)
    except Exception as exc:
        sys.exit(f"Meta credential check failed: {exc}")
    print(f"{MODEL} credential check ok: id={data.get('id', '<none>')}")


if __name__ == "__main__":
    main()
