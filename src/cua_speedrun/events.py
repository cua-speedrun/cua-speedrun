"""Structured run events: one vocabulary for progress, three consumers.

The executor emits events instead of printing. Sinks decide what happens to
them: the console sink renders the familiar CLI lines, the jsonl sink writes
events.jsonl into the run directory (ground truth, next to the run logs),
and the platform's worker adds a database sink that feeds live status pages.
The CLI and the dashboard service therefore share one progress vocabulary, and
"live status" is a read of this stream, not a bespoke feature per stage.

Events are deliberately plain: a kind, a wall timestamp, an optional task
key ("task_id/seed") for per-task events, and a flat payload. Timing truth
stays in the gateway's run logs; events are progress reporting only.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class Event:
    kind: str
    ts: float
    task_key: str | None
    payload: dict[str, Any]


Sink = Callable[[Event], None]


@dataclass
class EventBus:
    """Fans one emit out to every sink. Thread-safe; per-task threads emit
    concurrently. A sink is dropped only after several consecutive failures,
    so a transient error (a busy database, a full pipe) cannot silently kill
    live status for the rest of a run, but a genuinely broken consumer still
    cannot take the run down."""

    sinks: list[Sink] = field(default_factory=list)
    max_consecutive_failures: int = 5

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._failures: dict[int, int] = {}

    def emit(self, kind: str, task_key: str | None = None, **payload: Any) -> None:
        event = Event(kind=kind, ts=time.time(), task_key=task_key, payload=payload)
        with self._lock:
            dead = []
            for sink in self.sinks:
                try:
                    sink(event)
                    self._failures.pop(id(sink), None)
                except Exception:
                    count = self._failures.get(id(sink), 0) + 1
                    self._failures[id(sink)] = count
                    if count >= self.max_consecutive_failures:
                        dead.append(sink)
            for sink in dead:
                self.sinks.remove(sink)
                self._failures.pop(id(sink), None)
                print(f"[events] dropping sink {sink!r} after "
                      f"{self.max_consecutive_failures} consecutive failures",
                      flush=True)


class JsonlSink:
    """Appends every event to a jsonl file, flushed per line."""

    def __init__(self, path: Path):
        self._fh = open(path, "a", encoding="utf-8")

    def __call__(self, event: Event) -> None:
        record = {"kind": event.kind, "ts": event.ts, **event.payload}
        if event.task_key:
            record["task"] = event.task_key
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class ConsoleSink:
    """Renders events as the human-readable lines the CLI has always shown."""

    def __call__(self, event: Event) -> None:
        p = event.payload
        k = event.kind
        if k == "run_started":
            print(f"run {p['run_id']}: {p['benchmark']} v{p['version']}, "
                  f"{p['n_tasks']} tasks x {p['runs_per_task']}, backend={p['backend']}",
                  flush=True)
        elif k == "init_started":
            if p.get("execution_mode") == "local":
                print("init.py once on the local execution host...", flush=True)
            elif p.get("execution_mode") == "scheduled":
                print("starting scheduled compute replicas...", flush=True)
            else:
                print("init.py once in an agent sandbox, snapshotting...", flush=True)
        elif k == "init_cache_hit":
            print(f"reusing init snapshot {p['image_id']} from run "
                  f"{p.get('from_run')} (init skipped)", flush=True)
        elif k == "init_cache_invalid":
            print(f"cached init snapshot {p['image_id']} no longer exists, "
                  "re-running init", flush=True)
        elif k == "init_cache_stored":
            print(f"init snapshot {p['image_id']} cached for reuse", flush=True)
        elif k == "init_line":
            print(f"[init] {p['line'].rstrip()}", flush=True)
        elif k == "init_sandbox_created":
            gpu = f", gpu={p['gpu']}" if p.get("gpu") else ""
            print(f"init sandbox {p.get('sandbox_id')} created in "
                  f"{p['create_sec']:.1f}s (includes image build){gpu}",
                  flush=True)
        elif k == "init_sandbox_started":
            print(f"init sandbox running after {p['queue_boot_sec']:.1f}s "
                  f"of queue+boot", flush=True)
        elif k == "init_py_done":
            print(f"init.py exited rc={p['rc']} after {p['init_py_sec']:.1f}s",
                  flush=True)
        elif k == "snapshot_started":
            print("snapshotting the agent filesystem...", flush=True)
        elif k == "snapshot_done":
            print(f"snapshot done in {p['snapshot_sec']:.1f}s "
                  f"({p.get('image_id')})", flush=True)
        elif k == "init_done":
            if p.get("execution_mode") == "scheduled":
                print(
                    f"compute replica ready in {p['duration_sec']:.1f}s",
                    flush=True,
                )
            else:
                print(f"init + snapshot done in {p['duration_sec']:.1f}s", flush=True)
        elif k == "env_ready":
            boot = p.get("env_boot_sec")
            print(f"  {event.task_key}: env ready"
                  + (f" (boot {boot:.1f}s)" if boot is not None else ""),
                  flush=True)
        elif k == "warmup_sandbox_created":
            print(f"  {event.task_key}: agent sandbox {p.get('sandbox_id')} "
                  f"created in {p['create_sec']:.1f}s", flush=True)
        elif k == "warmup_sandbox_started":
            print(f"  {event.task_key}: agent sandbox running after "
                  f"{p['queue_boot_sec']:.1f}s of queue+boot", flush=True)
        elif k == "warmup_line":
            print(f"  {event.task_key or 'run'} [warmup] {p['line'].rstrip()}",
                  flush=True)
        elif k == "warmup_done":
            print(f"  {event.task_key or 'run'}: warmup done in "
                  f"{p['warmup_sec']:.1f}s "
                  f"(agent region {p.get('agent_region')})", flush=True)
        elif k == "all_envs_ready":
            print(f"  all {p['count']} envs ready; shared agent warm in "
                  f"{p['warmup_sec']:.1f}s (region {p.get('agent_region')})",
                  flush=True)
        elif k == "warmup_fallback":
            print(f"  {event.task_key}: no capacity for the agent sandbox in "
                  f"{p['region']}, placing unpinned", flush=True)
        elif k == "env_region_fallback":
            print(f"  {event.task_key}: env sandbox cannot be placed in "
                  f"{p.get('region')}, placing unpinned", flush=True)
        elif k == "agent_network_granted":
            policy = p.get("network_policy") or "host-network"
            print(f"  {event.task_key or 'run'}: agent sandbox granted "
                  f"{policy} network ({p.get('host')})", flush=True)
        elif k == "phase_heartbeat":
            elapsed = f" {p['elapsed_sec']:.0f}s" if p.get("elapsed_sec") else ""
            print(f"  {event.task_key or 'run'}: {p['phase']}...{elapsed}",
                  flush=True)
        elif k == "task_phases":
            parts = ", ".join(
                f"{key.removesuffix('_sec')} {val:.1f}s"
                for key, val in p.items() if isinstance(val, (int, float)))
            if parts:
                print(f"  {event.task_key}: phases: {parts}", flush=True)
        elif k == "task_done":
            status = "pass" if p.get("passed") else f"FAIL ({p.get('reason')})"
            t = p.get("task_time_sec")
            print(f"  {event.task_key}: {status}"
                  + (f"  {t:.2f}s" if t is not None else ""), flush=True)
        elif k == "run_failed":
            print(f"run failed: {p.get('error')}", flush=True)
        # Other kinds (env_created, env_line, armed, agent_exit, go_delivered,
        # ...) are progress detail for the platform UI and the jsonl record;
        # the CLI stays quiet on them. env_line in particular is the env
        # sandbox's full boot log, far too noisy for a console.
