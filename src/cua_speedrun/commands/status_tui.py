"""Live terminal presentation for one evaluation status snapshot.

The view knows only the transport-neutral status payload. Fetching from a
local installation or a dashboard stays in ``dashboard_client``.
"""

from __future__ import annotations

import os
import select
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text


TERMINAL_STAGES = frozenset({
    "card_ready", "failed", "rejected", "held", "cancelled",
})
TERMINAL_TASK_STAGES = frozenset({"done", "failed", "cancelled"})


@dataclass
class StatusViewState:
    filter_mode: str = "all"
    selected: int = 0
    offset: int = 0
    details: bool = False


def _duration(value: Any, *, clock: bool = False) -> str:
    if value is None:
        return "—"
    seconds = max(0, int(float(value)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    if clock:
        return f"{minutes:02d}:{seconds:02d}"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _cost(value: Any) -> str:
    if value is None:
        return "—"
    amount = float(value)
    if amount >= 1:
        return f"${amount:,.2f}"
    if amount >= 0.01:
        return f"${amount:.4f}"
    return f"${amount:.6f}"


def _age(timestamp: Any, now: float) -> str:
    if timestamp is None:
        return "—"
    seconds = max(0, int(now - float(timestamp)))
    if seconds >= 86400:
        days, remainder = divmod(seconds, 86400)
        return f"{days}d{remainder // 3600:02d}h"
    if seconds < 60:
        return f"{seconds}s"
    return _duration(seconds)


def _task_label(task_key: str) -> str:
    if "/seed_" not in task_key:
        return task_key
    task_id, seed = task_key.rsplit("/seed_", 1)
    return f"{task_id} · seed {seed}"


def _task_state(task: dict[str, Any]) -> tuple[str, str]:
    stage = str(task.get("stage") or "queued")
    passed = task.get("passed")
    if stage == "done":
        if passed is True:
            return "✓ pass", "green"
        if passed is False:
            if task.get("reason") == "agent_error":
                return "× agent fail", "red"
            return "× verifier fail", "red"
        return "done", "white"
    if stage == "failed":
        return "! infra error", "bold red"
    if stage == "cancelled":
        return "cancelled", "dim"
    if stage == "running":
        return "● running", "bold cyan"
    if stage == "checking":
        return "● checking", "cyan"
    if stage == "retrying":
        return "↻ retrying", "cyan"
    labels = {
        "env_boot": "env boot",
        "env_ready": "env ready",
        "warming": "warming",
        "warm": "warm",
        "starting": "starting",
        "queued": "queued",
    }
    return labels.get(stage, stage.replace("_", " ")), "dim"


def _task_steps(task: dict[str, Any]) -> str:
    steps = task.get("num_steps")
    if steps is None:
        return "—"
    steps = int(steps)
    limit = task.get("step_limit")
    if not limit:
        return f"{steps} steps"
    limit = max(1, int(limit))
    filled = min(8, round(8 * min(steps, limit) / limit))
    return f"{'█' * filled}{'░' * (8 - filled)} {steps}/{limit}"


def _task_elapsed(task: dict[str, Any], now: float) -> str:
    if task.get("stage") == "done" and task.get("task_time_sec") is not None:
        return _duration(task["task_time_sec"], clock=True)
    started = (
        task.get("armed_at")
        if task.get("stage") in {"running", "checking"}
        else task.get("started_at")
    )
    ended = task.get("updated_at") if task.get("stage") in TERMINAL_TASK_STAGES else now
    if started is None or ended is None:
        return "—"
    return _duration(float(ended) - float(started), clock=True)


def _is_active(task: dict[str, Any]) -> bool:
    return task.get("stage") not in TERMINAL_TASK_STAGES | {"queued"}


def _is_failure(task: dict[str, Any]) -> bool:
    return task.get("stage") == "failed" or (
        task.get("stage") == "done" and task.get("passed") is False
    )


def _ordered_tasks(payload: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    tasks = list(payload.get("tasks") or ())
    if mode == "active":
        tasks = [task for task in tasks if _is_active(task)]
    elif mode == "failures":
        tasks = [task for task in tasks if _is_failure(task)]

    def order(task: dict[str, Any]) -> tuple[Any, ...]:
        stage = task.get("stage")
        if _is_active(task):
            group = 0
            within = float(task.get("task_index") or 0)
        elif stage == "failed":
            group = 1
            within = -float(task.get("updated_at") or 0)
        elif stage in TERMINAL_TASK_STAGES:
            group = 2
            within = -float(task.get("updated_at") or 0)
        else:
            group = 3
            within = float(task.get("task_index") or 0)
        return group, within, str(task.get("task_key") or "")

    return sorted(tasks, key=order)


def _stage_text(stage: str) -> Text:
    label = "COMPLETE" if stage == "card_ready" else stage.replace("_", " ").upper()
    if stage == "card_ready":
        style = "bold green"
    elif stage in {"failed", "rejected", "held", "cancelled"}:
        style = "bold red"
    elif stage in {"running", "initializing", "snapshot_done"}:
        style = "bold cyan"
    else:
        style = "bold white"
    return Text(label, style=style)


def _header(payload: dict[str, Any]) -> RenderableType:
    submission = payload.get("submission") or {}
    heading = Table.grid(expand=True)
    heading.add_column(ratio=1)
    heading.add_column(justify="right")
    status = _stage_text(str(payload.get("stage") or "unknown"))
    if payload.get("stop_requested"):
        status.append(" · STOP REQUESTED", style="bold red")
    heading.add_row(
        Text(f"CUA SPEEDRUN  ·  EVALUATION {payload.get('run_id', '?')}", style="bold"),
        status,
    )
    contract = " · ".join(filter(None, (
        str(submission.get("name") or "unknown agent"),
        str(payload.get("benchmark") or "unknown benchmark"),
        str(submission.get("track") or ""),
    )))
    heading.add_row(Text(contract, style="white"), Text())
    topology = str(payload.get("topology") or "unknown")
    parallel = int(payload.get("parallel_evaluations") or 1)
    parallel_label = (
        f"{topology} · {parallel} parallel "
        f"evaluation{'s' if parallel != 1 else ''}"
    )
    heading.add_row(
        Text(parallel_label, style="dim"),
        Text(f"elapsed {_duration(payload.get('elapsed_sec'), clock=True)}", style="dim"),
    )
    return Panel(heading, box=box.ROUNDED, border_style="bright_black", padding=(0, 1))


def _progress(payload: dict[str, Any], width: int) -> RenderableType:
    progress = payload.get("progress") or {}
    total = max(0, int(progress.get("total") or 0))
    finished = min(total, max(0, int(progress.get("finished") or 0)))
    percentage = (100 * finished / total) if total else 0.0
    bar = Table.grid(expand=True, padding=(0, 1))
    bar.add_column(width=7)
    bar.add_column(width=9, justify="right")
    bar.add_column(ratio=1)
    bar.add_column(width=7, justify="right")
    bar.add_row(
        Text("TASKS", style="bold"),
        Text(f"{finished} / {total}"),
        ProgressBar(
            total=max(1, total),
            completed=finished,
            width=max(10, min(48, width - 32)),
            complete_style="cyan",
            finished_style="green",
        ),
        Text(f"{percentage:5.1f}%"),
    )
    counts = Text()
    counts.append(f"{int(progress.get('passed') or 0)} pass", style="green")
    counts.append(" · ", style="dim")
    counts.append(
        f"{int(progress.get('agent_failed') or 0)} agent fail", style="red"
    )
    counts.append(" · ", style="dim")
    counts.append(
        f"{int(progress.get('verifier_failed') or 0)} verifier fail", style="red"
    )
    counts.append(" · ", style="dim")
    counts.append(
        f"{int(progress.get('infra_failed') or 0)} infrastructure errors",
        style="bold red" if progress.get("infra_failed") else "dim",
    )
    if progress.get("cancelled"):
        counts.append(" · ", style="dim")
        counts.append(f"{int(progress['cancelled'])} cancelled", style="dim")

    tasks = list(payload.get("tasks") or ())
    stages = Counter(str(task.get("stage") or "queued") for task in tasks)
    missing = max(0, total - len(tasks))
    activity = []
    running = stages["running"] + stages["checking"]
    starting = sum(stages[name] for name in ("starting", "env_boot", "warming", "warm"))
    for count, label in (
        (running, "running"),
        (stages["env_ready"], "env ready"),
        (starting, "starting"),
        (stages["retrying"], "retrying"),
        (stages["queued"] + missing, "queued"),
    ):
        if count:
            activity.append(f"{count} {label}")
    summary = Text("  ")
    summary.append_text(counts)
    activity_line = Text(
        "  " + (" · ".join(activity) or "No active tasks"), style="dim"
    )
    lines: list[RenderableType] = [bar, summary, activity_line]
    result = payload.get("result") or {}
    if result.get("success_rate") is not None:
        result_line = Text("  Result  ", style="dim")
        if result.get("mean_score") is not None:
            result_line.append(f"{float(result['mean_score']):.1%} score", style="bold")
            result_line.append(
                f" · {float(result['success_rate']):.1%} exact success",
                style="white",
            )
        else:
            result_line.append(f"{float(result['success_rate']):.1%} success", style="bold")
        if result.get("meets_success_bar") is not None:
            meets = bool(result["meets_success_bar"])
            result_line.append(
                " · success bar met" if meets else " · below success bar",
                style="green" if meets else "red",
            )
        lines.append(result_line)
    num_runs = int(result.get("num_runs") or 0)
    measured = result.get("measured_time_sec")
    if num_runs and measured is not None:
        timing = Text("  Time  ", style="dim")
        timing.append(f"{_duration(measured)} measured total", style="bold")
        timing.append(
            f" · {_duration(round(float(measured) / num_runs))}/task",
            style="white",
        )
        for label, field in (
            ("agent", "agent_time_sec"),
            ("environment", "env_time_sec"),
        ):
            value = result.get(field)
            if value is not None:
                timing.append(
                    f" · {_duration(round(float(value) / num_runs))} {label}",
                    style="dim",
                )
        lines.append(timing)
    if payload.get("cost_usd") is not None:
        cost = Text("  Cost  ", style="dim")
        cost.append(f"{_cost(payload['cost_usd'])} total", style="bold")
        lines.append(cost)
    return Panel(
        Group(*lines),
        box=box.ROUNDED,
        border_style="bright_black",
        padding=(0, 0),
    )


def _filters(state: StatusViewState) -> Text:
    line = Text("TASKS  ", style="bold")
    choices = (
        ("a", "all", "all"),
        ("r", "active", "active"),
        ("f", "failures", "failures"),
    )
    for key, label, mode in choices:
        active = state.filter_mode == mode
        line.append(f" {key}:{label} ", style="reverse bold" if active else "dim")
    return line


def _task_table(
    payload: dict[str, Any],
    state: StatusViewState,
    *,
    now: float,
    max_rows: int | None,
    width: int,
) -> tuple[RenderableType, list[dict[str, Any]]]:
    tasks = _ordered_tasks(payload, state.filter_mode)
    title = _filters(state) if max_rows is not None else Text("TASKS", style="bold")
    if not tasks:
        if payload.get("stage") not in TERMINAL_STAGES and state.filter_mode == "all":
            from .preparation_view import render_preparation
            return render_preparation(payload, details=state.details, log_lines=max_rows or 12), tasks
        return Panel(
            Text(f"No {state.filter_mode} tasks.", style="dim"),
            title=title,
            title_align="left",
            border_style="bright_black",
            box=box.ROUNDED,
        ), tasks

    state.selected = min(max(0, state.selected), len(tasks) - 1)
    if max_rows is None:
        start, visible = 0, tasks
    else:
        if state.selected < state.offset:
            state.offset = state.selected
        elif state.selected >= state.offset + max_rows:
            state.offset = state.selected - max_rows + 1
        state.offset = min(max(0, state.offset), max(0, len(tasks) - max_rows))
        start = state.offset
        visible = tasks[start:start + max_rows]

    table = Table(
        box=box.SIMPLE_HEAD,
        expand=True,
        padding=(0, 1),
        show_edge=False,
    )
    table.add_column("#", width=4, justify="right", style="dim")
    table.add_column("TASK", ratio=1, no_wrap=True, overflow="ellipsis")
    table.add_column("STATE", width=16 if width >= 80 else 12, no_wrap=True)
    table.add_column("STEPS", width=18 if width >= 90 else 10, no_wrap=True)
    show_time = width >= 72
    show_cost = width >= 110
    show_update = width >= 100 and payload.get("stage") not in TERMINAL_STAGES
    if show_time:
        table.add_column("TIME", width=8, justify="right", no_wrap=True)
    if show_cost:
        table.add_column("COST", width=10, justify="right", no_wrap=True)
    if show_update:
        table.add_column("UPDATE", width=7, justify="right", no_wrap=True)
    for absolute_index, task in enumerate(visible, start=start):
        state_label, state_style = _task_state(task)
        selected = max_rows is not None and absolute_index == state.selected
        row_style = "reverse" if selected else None
        cells: list[Any] = [
            str(task.get("task_index") or "—"),
            _task_label(str(task.get("task_key") or "unknown task")),
            Text(state_label, style=state_style),
            _task_steps(task),
        ]
        if show_time:
            cells.append(_task_elapsed(task, now))
        if show_cost:
            cells.append(_cost(task.get("cost_usd")))
        if show_update:
            cells.append(_age(task.get("updated_at"), now))
        table.add_row(*cells, style=row_style)
    if len(tasks) > len(visible):
        table.caption = f"Rows {start + 1}–{start + len(visible)} of {len(tasks)}"
        table.caption_style = "dim"
    return Panel(
        table,
        title=title,
        title_align="left",
        border_style="bright_black",
        box=box.ROUNDED,
        padding=(0, 0),
    ), tasks


def _details(task: dict[str, Any], now: float) -> RenderableType:
    label, style = _task_state(task)
    detail = Table.grid(expand=True, padding=(0, 1))
    detail.add_column(width=12, style="dim")
    detail.add_column(ratio=1)
    detail.add_row("Task", str(task.get("task_key") or "unknown"))
    detail.add_row("State", Text(label, style=style))
    detail.add_row("Progress", _task_steps(task))
    detail.add_row("Elapsed", _task_elapsed(task, now))
    if task.get("cost_usd") is not None:
        detail.add_row("Cost", _cost(task["cost_usd"]))
    if task.get("reason"):
        detail.add_row("Reason", str(task["reason"]))
    if task.get("error"):
        error = str(task["error"])
        if len(error) > 500:
            error = error[:500].rstrip() + "…"
        detail.add_row("Error", error)
    return Panel(
        detail,
        title="DETAIL",
        title_align="left",
        border_style="bright_black",
        box=box.ROUNDED,
        padding=(0, 0),
    )


def render_status(
    payload: dict[str, Any],
    *,
    state: StatusViewState | None = None,
    width: int = 100,
    height: int = 30,
    interactive: bool = False,
    result_location: str | None = None,
) -> RenderableType:
    state = state or StatusViewState()
    now = time.time()
    fixed_rows = 14 + (6 if state.details else 0)
    max_rows = max(3, height - fixed_rows) if interactive else None
    task_panel, tasks = _task_table(
        payload, state, now=now, max_rows=max_rows, width=width
    )
    parts: list[RenderableType] = [_header(payload)]
    if payload.get("tasks") or payload.get("stage") in TERMINAL_STAGES:
        parts.append(_progress(payload, width))
    parts.append(task_panel)
    if state.details and tasks:
        parts.append(_details(tasks[state.selected], now))
    if payload.get("error"):
        error = str(payload["error"])
        if interactive and len(error) > 500:
            error = error[:500].rstrip() + "…\nRun with --once or --json for the full error."
        parts.append(Panel(
            Text(error, style="red"),
            title="EXECUTION ERROR",
            title_align="left",
            border_style="red",
            box=box.ROUNDED,
        ))
    if result_location:
        parts.append(Text.assemble(("Result: ", "dim"), result_location))
    if interactive:
        parts.append(Text(
            ("↑↓/jk scroll · a all · r active · f failures · " if payload.get("tasks") else "")
            + "Enter details · q detach",
            style="dim",
            justify="center",
        ))
    return Group(*parts)


class _KeyReader:
    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.enabled = sys.stdin.isatty()
        self.saved: Any = None

    def __enter__(self) -> "_KeyReader":
        if self.enabled and os.name == "posix":
            import termios
            import tty

            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if os.name == "posix" and self.saved is not None:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def read(self, timeout: float) -> str | None:
        if not self.enabled:
            time.sleep(max(0.0, timeout))
            return None
        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if msvcrt.kbhit():
                    first = msvcrt.getwch()
                    if first in ("\x00", "\xe0"):
                        arrows = {
                            "H": "up", "P": "down",
                            "I": "page_up", "Q": "page_down",
                        }
                        return arrows.get(msvcrt.getwch())
                    return _decode_key(first)
                time.sleep(0.02)
            return None

        ready, _, _ = select.select([self.fd], [], [], max(0.0, timeout))
        if not ready:
            return None
        data = os.read(self.fd, 1)
        while select.select([self.fd], [], [], 0.005)[0]:
            data += os.read(self.fd, 16)
        return _decode_key(data.decode(errors="ignore"))


def _decode_key(value: str) -> str | None:
    arrows = {
        "\x1b[A": "up",
        "\x1b[B": "down",
        "\x1b[5~": "page_up",
        "\x1b[6~": "page_down",
        "\x1b[H": "home",
        "\x1b[F": "end",
    }
    if value in arrows:
        return arrows[value]
    if value in ("\r", "\n"):
        return "enter"
    if value in {"a", "r", "f", "q", "j", "k", "g", "G"}:
        return value
    return None


def _handle_key(
    state: StatusViewState,
    key: str,
    payload: dict[str, Any],
    page_size: int,
) -> bool:
    if key == "q":
        return False
    if key in {"a", "r", "f"}:
        state.filter_mode = {"a": "all", "r": "active", "f": "failures"}[key]
        state.selected = 0
        state.offset = 0
        state.details = False
        return True
    tasks = _ordered_tasks(payload, state.filter_mode)
    if not tasks:
        if key == "enter":
            state.details = not state.details
        return True
    if key in {"down", "j"}:
        state.selected = min(len(tasks) - 1, state.selected + 1)
    elif key in {"up", "k"}:
        state.selected = max(0, state.selected - 1)
    elif key == "page_down":
        state.selected = min(len(tasks) - 1, state.selected + page_size)
    elif key == "page_up":
        state.selected = max(0, state.selected - page_size)
    elif key in {"home", "g"}:
        state.selected = 0
    elif key in {"end", "G"}:
        state.selected = len(tasks) - 1
    elif key == "enter":
        state.details = not state.details
    return True


def print_status_snapshot(
    payload: dict[str, Any],
    *,
    result_location: str | None = None,
    console: Console | None = None,
) -> None:
    console = console or Console()
    console.print(render_status(
        payload,
        width=console.size.width,
        height=console.size.height,
        result_location=result_location,
    ))


def run_status_tui(
    fetch: Callable[[], dict[str, Any]],
    initial: dict[str, Any],
    *,
    poll_sec: float,
    result_location: Callable[[dict[str, Any]], str | None],
    console: Console | None = None,
) -> tuple[dict[str, Any], bool]:
    """Run until terminal state or detach; return ``(last_payload, detached)``."""
    console = console or Console()
    state = StatusViewState()
    payload = initial
    detached = False
    next_fetch = time.monotonic() + max(0.2, poll_sec)
    with _KeyReader() as keys, Live(
        console=console,
        screen=True,
        auto_refresh=False,
    ) as live:
        while True:
            live.update(render_status(
                payload,
                state=state,
                width=console.size.width,
                height=console.size.height,
                interactive=True,
                result_location=result_location(payload),
            ))
            live.refresh()
            if payload.get("stage") in TERMINAL_STAGES:
                break

            page_size = max(3, console.size.height - 14 - (6 if state.details else 0))
            timeout = min(0.2, max(0.0, next_fetch - time.monotonic()))
            key = keys.read(timeout)
            if key is not None and not _handle_key(
                state, key, payload, page_size
            ):
                detached = True
                break
            if time.monotonic() >= next_fetch:
                payload = fetch()
                next_fetch = time.monotonic() + max(0.2, poll_sec)

    return payload, detached


__all__ = [
    "StatusViewState",
    "print_status_snapshot",
    "render_status",
    "run_status_tui",
]
