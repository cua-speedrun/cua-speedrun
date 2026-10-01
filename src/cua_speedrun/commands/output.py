"""Keep machine-readable stdout separate from preparation diagnostics."""

from contextlib import contextmanager, redirect_stdout
import io
import json
import os
import sys


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def interactive(args) -> bool:
    return (not getattr(args, "json", False)
            and not getattr(args, "no_input", False)
            and sys.stdin.isatty() and sys.stdout.isatty()
            and os.environ.get("TERM") != "dumb")


@contextmanager
def diagnostics(enabled: bool = True):
    """Redirect Python output and inherited subprocess stdout to stderr."""
    if not enabled:
        yield
        return
    saved = None
    try:
        sys.stdout.flush()
        try:
            fd = sys.stdout.fileno()
            saved = os.dup(fd)
            os.dup2(sys.stderr.fileno(), fd)
        except (OSError, io.UnsupportedOperation):
            if saved is not None:
                os.close(saved)
                saved = None
        with redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stderr.flush()
        if saved is not None:
            os.dup2(saved, fd)
            os.close(saved)
