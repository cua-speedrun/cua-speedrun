"""Self-contained operator commands.

The evaluation CLI predates these commands.  Registration and dispatch live
here so installation, diagnosis, and process supervision can evolve without
turning ``cua_speedrun.cli`` into an operator-runtime module.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def register_operator_commands(subparsers: argparse._SubParsersAction) -> None:
    from .dashboard_client import register_dashboard_client_commands
    from .dashboard import register_dashboard_command
    from .doctor import register_doctor_command
    from .install import register_install_command
    from .setup import register_setup_command
    from .benchmark import register_benchmark_command

    register_install_command(subparsers)
    register_doctor_command(subparsers)
    register_dashboard_command(subparsers)
    register_dashboard_client_commands(subparsers)
    register_setup_command(subparsers)
    register_benchmark_command(subparsers)


def dispatch_operator_command(args: argparse.Namespace) -> int | None:
    handler = getattr(args, "_operator_handler", None)
    if handler is None:
        return None
    try:
        return int(handler(args))
    except subprocess.CalledProcessError as exc:
        if getattr(args, "json", False):
            from .output import emit
            emit({"type": "error", "error": f"preparation command exited with status {exc.returncode}"})
            return 1
        command = exc.cmd if isinstance(exc.cmd, str) else " ".join(exc.cmd)
        print(
            f"error: command exited with status {exc.returncode}: {command}",
            file=sys.stderr,
        )
        return 1
    except (OSError, RuntimeError, ValueError) as exc:
        if getattr(args, "json", False):
            from .output import emit
            emit({"type": "error", "error": str(exc)})
            return 1
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        if getattr(args, "json", False):
            from .output import emit
            emit({"type": "interrupted"})
        else:
            print("\nCancelled.", file=sys.stderr)
        return 130


__all__ = ["dispatch_operator_command", "register_operator_commands"]
