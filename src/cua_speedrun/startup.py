"""Structured progress for untimed evaluation preparation."""

from __future__ import annotations

from contextlib import contextmanager
import json
import threading
import time


PREFIX = "[startup] "
_OUTPUT_LOCK = threading.Lock()


def emit_phase(key: str, label: str, state: str, **fields) -> None:
    record = {"key": key, "label": label, "state": state, "at": time.time(), **fields}
    with _OUTPUT_LOCK:
        print(PREFIX + json.dumps(record), flush=True)


@contextmanager
def phase(key: str, label: str):
    started = time.monotonic()
    emit_phase(key, label, "running")
    try:
        yield
    except BaseException:
        emit_phase(key, label, "failed", elapsed_sec=time.monotonic() - started)
        raise
    else:
        emit_phase(key, label, "done", elapsed_sec=time.monotonic() - started)


class PreparationProgress:
    """Accumulate phase events while retaining a bounded diagnostic log."""

    def __init__(self):
        self.steps: dict[str, dict] = {}
        self.logs: list[str] = []

    def feed(self, line: str) -> bool:
        if line.startswith(PREFIX):
            try:
                event = json.loads(line[len(PREFIX):])
                key = event["key"]
                if not isinstance(key, str) or event["state"] not in {"running", "done", "failed"}:
                    raise ValueError("invalid preparation event")
                previous = self.steps.get(key, {})
                self.steps[key] = {**previous, **event, "started_at": previous.get("started_at", event["at"])}
                return True
            except (ValueError, KeyError, TypeError):
                pass
        self.logs.append(line)
        self.logs = self.logs[-200:]
        return False

    def payload(self) -> dict:
        return {"preparation_steps": list(self.steps.values()), "preparation": list(self.logs)}
