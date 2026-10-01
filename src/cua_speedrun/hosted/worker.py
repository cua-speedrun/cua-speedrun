"""Run the existing evaluation service inside a single Modal sandbox."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile

from cua_speedrun.hosted import REMOTE_HOME, TERMINAL, initial_status, is_hosted
from cua_speedrun.startup import PreparationProgress


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def snapshot(home: Path, destination: Path) -> None:
    """Checkpoint artifacts and a consistent database backup, never config.env."""
    for name in ("runs", "logs"):
        source = home / name
        if source.is_dir():
            subprocess.run(["rsync", "-a", str(source), str(destination) + "/"], check=True)
    database = home / "platform.db"
    if database.is_file():
        temporary = home.parent / "state.sqlite"
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as source:
            with sqlite3.connect(temporary) as target:
                source.backup(target)
        shutil.copyfile(temporary, destination / "state.sqlite")


def main(run_id: str) -> int:
    from cua_speedrun.commands.local_evaluations import LocalEvaluations
    from cua_speedrun.hosted.inputs import unpack_inputs
    from cua_speedrun.service.cancellation import terminate_run_child, cleanup_remote_run
    from cua_speedrun.service.run_export import write_run_archive

    if not is_hosted(run_id):
        raise ValueError("invalid hosted run ID")
    work = Path("/work")
    work.mkdir(exist_ok=True)
    request_path = Path("/tmp/request.json")
    deadline = time.monotonic() + 300
    while not request_path.exists():
        if time.monotonic() > deadline:
            raise RuntimeError("hosted request was not delivered")
        time.sleep(0.2)
    request = json.loads(request_path.read_text())
    if request["run_id"] != run_id:
        raise ValueError("hosted request ID mismatch")
    saved = Path("/data") / run_id
    saved.mkdir(exist_ok=True)
    home = Path(REMOTE_HOME)
    payload = initial_status(request)
    prep_log = work / "preparation.log"
    launch_file = work / "launch.json"
    child = service = local_id = None
    last_checkpoint = 0.0
    preparation = PreparationProgress()
    prep_offset = 0
    prep_partial = b""
    os.environ["CS_AGENT_SNAPSHOT_CACHE_DIR"] = "/data/cache/agent-snapshots"

    def publish(*, durable=False):
        nonlocal last_checkpoint, prep_offset, prep_partial
        payload["run_id"] = run_id
        payload["host"] = "modal"
        payload["elapsed_sec"] = time.time() - request["created_at"]
        if prep_log.exists() and payload["stage"] == "preparing":
            with prep_log.open("rb") as file:
                file.seek(prep_offset)
                data = prep_partial + file.read()
                prep_offset = file.tell()
            lines = data.split(b"\n")
            prep_partial = lines.pop()
            for line in lines:
                preparation.feed(line.decode(errors="replace").rstrip("\r"))
            payload.update(preparation.payload())
        if durable or time.monotonic() - last_checkpoint > 15:
            snapshot(home, saved)
            if prep_log.exists():
                shutil.copyfile(prep_log, saved / "preparation.log")
            atomic_json(saved / "status.json", payload)
            # Sandbox volume writes are committed in the background and once
            # more on exit. The client waits for exit before exporting results.
            last_checkpoint = time.monotonic()
        atomic_json(work / "status.json", payload)

    try:
        publish(durable=True)
        unpack_inputs(Path("/tmp/inputs.zip").read_bytes(), work / "inputs")
        with prep_log.open("wb") as logs, launch_file.open("w") as launched:
            child = subprocess.Popen([sys.executable, "-u", "-m", "cua_speedrun.cli", *request["argv"]],
                                     stdout=launched, stderr=logs, start_new_session=True)
        while child.poll() is None:
            if (work / "cancel").exists():
                terminate_run_child(child)
                payload["stage"] = "cancelled"
                break
            publish()
            time.sleep(1)
        records = []
        for line in launch_file.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
        queued = next((r for r in records if r.get("type") == "queued"), None)
        if queued is None:
            if payload["stage"] != "cancelled":
                detail = next((r["error"] for r in records if r.get("type") == "error"), "see preparation logs")
                raise RuntimeError(f"preparation failed: {detail}")
        else:
            local_id = int(queued["run_id"])
            service = LocalEvaluations.open(REMOTE_HOME)
            while True:
                if (work / "cancel").exists() or time.time() - request["created_at"] > 23.9 * 3600:
                    service.cancel(local_id)
                payload = service.status(local_id)
                if payload["stage"] in TERMINAL:
                    break
                publish()
                time.sleep(1)
    except Exception as exc:
        payload.update(stage="failed", error=str(exc))
    finally:
        if child is not None and child.poll() is None:
            terminate_run_child(child)
        # Preparation can be interrupted after queueing but before its JSON
        # response is flushed. Recover the run from this controller's own DB.
        if service is None and (home / "install.json").exists():
            service = LocalEvaluations.open(REMOTE_HOME)
            from sqlalchemy import select
            from cua_speedrun.service.db import Run
            with service.session_factory() as session:
                local_id = session.scalar(select(Run.id).order_by(Run.id.desc()).limit(1))
        terminal_payload = dict(payload)
        payload = {**payload, "stage": "saving"}
        try:
            publish()
        except Exception as exc:
            terminal_payload.update(stage="failed", error=f"checkpoint failed: {exc}")
        if service is not None and local_id is not None:
            try:
                current = service.status(local_id)
                if current["stage"] not in TERMINAL:
                    service.cancel(local_id)
            except Exception as exc:
                terminal_payload.update(stage="failed", error=f"cancellation failed: {exc}")
            try:
                if not cleanup_remote_run(service.session_factory, local_id):
                    raise RuntimeError("some evaluation sandboxes could not be stopped")
            except Exception as exc:
                terminal_payload.update(stage="failed", error=f"cleanup failed: {exc}")
            try:
                current = service.status(local_id)
                directory = current.get("run_dir")
                if directory:
                    directory = Path(directory)
                    if not directory.is_absolute():
                        directory = home / directory
                    if directory.is_dir():
                        write_run_archive(directory, saved / "artifacts.zip")
                if terminal_payload["stage"] == "cancelled":
                    terminal_payload.update(current)
            except Exception as exc:
                terminal_payload.update(stage="failed", error=f"saving artifacts failed: {exc}")
        if not (saved / "artifacts.zip").exists():
            with zipfile.ZipFile(saved / "artifacts.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                if prep_log.exists():
                    archive.write(prep_log, f"{run_id}/preparation.log")
                archive.writestr(f"{run_id}/status.json", json.dumps(terminal_payload))
        payload = terminal_payload
        try:
            publish(durable=True)
        except Exception as exc:
            payload.update(stage="failed", error=f"checkpoint failed: {exc}")
            atomic_json(work / "status.json", payload)
            atomic_json(saved / "status.json", payload)
    return 0 if payload["stage"] in {"card_ready", "cancelled"} else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
