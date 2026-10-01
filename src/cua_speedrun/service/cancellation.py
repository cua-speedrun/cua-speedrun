"""Evaluation cancellation without leaking processes or provider credentials."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

from sqlalchemy import select

from cua_speedrun.service.db import EventRow, Run, SubmissionRow, User


def _process_tree(root_pid: int, process_marker: str | None = None) -> set[int]:
    """Find descendants plus run processes deliberately re-parented to init."""
    if os.name != "posix":
        return {root_pid}
    result = subprocess.run(
        ["ps", "-e", "-o", "pid=,ppid=,args="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {root_pid}
    children: dict[int, list[int]] = {}
    marked: set[int] = set()
    for line in result.stdout.splitlines():
        try:
            pid_text, parent_text, command = line.strip().split(maxsplit=2)
            pid, parent = int(pid_text), int(parent_text)
        except (ValueError, TypeError):
            continue
        children.setdefault(parent, []).append(pid)
        if process_marker and process_marker in command:
            marked.add(pid)
    tree = {root_pid}
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        for child in children.get(parent, ()):
            if child not in tree:
                tree.add(child)
                pending.append(child)
    return tree | marked


def _alive(pids: set[int]) -> set[int]:
    remaining = set()
    for pid in pids:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            continue
        remaining.add(pid)
    return remaining


def _signal_process_tree(pids: set[int], sig: signal.Signals) -> None:
    """Signal every captured process group, including runner-created groups."""
    own_group = os.getpgrp()
    groups: set[int] = set()
    ungrouped: set[int] = set()
    for pid in pids:
        try:
            group = os.getpgid(pid)
        except (ProcessLookupError, PermissionError):
            continue
        if group == own_group:
            ungrouped.add(pid)
        else:
            groups.add(group)
    for group in groups:
        try:
            os.killpg(group, sig)
        except (ProcessLookupError, PermissionError):
            pass
    for pid in ungrouped:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def terminate_run_child(
    child: subprocess.Popen,
    process_marker: str | None = None,
) -> None:
    """Give executor cleanup a chance, then terminate its complete process tree."""
    if child.poll() is not None:
        return
    if os.name != "posix":
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        return

    tree = _process_tree(child.pid, process_marker)
    try:
        os.kill(child.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass

    tree.update(_process_tree(child.pid, process_marker))
    remaining = _alive(tree)
    if remaining:
        _signal_process_tree(remaining, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while remaining and time.monotonic() < deadline:
            time.sleep(0.1)
            remaining = _alive(remaining)
    if remaining:
        _signal_process_tree(remaining, signal.SIGKILL)
    try:
        child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def cleanup_remote_run(session_factory, run_id: int) -> bool:
    """Terminate Modal resources using the run owner's credentials.

    This function is called only by this module's short-lived CLI process.
    The queue worker therefore never imports Modal, decrypts a user's secret,
    or lets one user's cached Modal client cross into another evaluation.
    """
    from cua_speedrun.service.usersecrets import decrypt

    with session_factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            return True
        stored_plan = dict(run.execution_plan or {})
        topology = dict(
            (stored_plan.get("execution") or {}).get("topology") or {}
        )
        event_payloads = [
            payload
            for payload in session.scalars(
                select(EventRow.payload).where(EventRow.run_id == run_id)
            )
            if isinstance(payload, dict)
        ]
        scheduler_cancellations = {
            tuple(payload["cancel_command"])
            for payload in event_payloads
            if isinstance(payload.get("cancel_command"), list)
            and payload["cancel_command"]
            and all(isinstance(item, str) for item in payload["cancel_command"])
        }
        failed = False
        if scheduler_cancellations:
            for command in sorted(scheduler_cancellations):
                result = subprocess.run(
                    list(command), capture_output=True, text=True, check=False
                )
                failed = failed or result.returncode != 0
        modal_placements = any(
            (topology.get(plane) or {}).get("provider") == "modal"
            for plane in ("compute", "environment")
        )
        if not modal_placements and (run.topology_key or topology.get("key")) not in {
            "modal-remote", "modal-native",
        }:
            return not failed
        submission = session.get(SubmissionRow, run.submission_id)
        owner = session.get(User, submission.user_id) if submission else None
        token_id = owner.modal_token_id if owner else None
        token_secret = (
            decrypt(owner.modal_token_secret_enc)
            if owner and owner.modal_token_secret_enc else None
        )
        sandbox_ids = {
            payload["sandbox_id"]
            for payload in event_payloads
            if isinstance(payload.get("sandbox_id"), str)
        }

    if not sandbox_ids:
        return not failed
    if not token_id or not token_secret:
        return False
    os.environ["MODAL_TOKEN_ID"] = token_id
    os.environ["MODAL_TOKEN_SECRET"] = token_secret
    from cua_speedrun.remote.cancel import terminate_sandboxes

    modal_failures = terminate_sandboxes(sandbox_ids)
    return not failed and not modal_failures


def cleanup_remote_run_in_child(run_id: int) -> None:
    """Run provider cleanup behind the same credential boundary as execution."""
    result = subprocess.run(
        [sys.executable, "-m", "cua_speedrun.service.cancellation", str(run_id)],
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        print(
            f"run {run_id} remote cleanup exited {result.returncode}; "
            "provider sandbox timeouts remain as the fallback",
            flush=True,
        )


def main() -> None:
    from cua_speedrun.config import load_dotenv
    from cua_speedrun.service.db import make_session_factory

    load_dotenv()
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m cua_speedrun.service.cancellation RUN_ID")
    success = cleanup_remote_run(make_session_factory(), int(sys.argv[1]))
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
