"""Shared compact preparation view for launch and hosted status."""

from __future__ import annotations

import time

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def render_preparation(payload: dict, *, details: bool = False, log_lines: int = 8):
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=2)
    table.add_column(ratio=1)
    table.add_column(justify="right", width=9)
    steps = payload.get("preparation_steps") or []
    for step in steps:
        state = step["state"]
        symbol, color = {"done": ("✓", "green"), "failed": ("×", "red"), "running": ("◌", "cyan")}[state]
        elapsed = step.get("elapsed_sec", max(0, time.time() - step["started_at"]))
        seconds = int(elapsed)
        table.add_row(Text(symbol, style=color), Text(step["label"], style=color if state == "running" else "default"),
                      Text(f"{seconds // 60:02d}:{seconds % 60:02d}", style="dim"))
    if not steps:
        stage = payload.get("stage", "preparing")
        label = payload.get("preparation_label") or {
            "initializing": "Prepare agent", "snapshot_done": "Start agent and desktop",
            "queued": "Start evaluation", "preparing": "Prepare evaluation",
        }.get(stage, "Prepare evaluation")
        table.add_row(Text("◌", style="cyan"), Text(label), Text())
    panels = [Panel(table, title="PREPARATION", title_align="left", box=box.ROUNDED,
                    border_style="bright_black", padding=(1, 1))]
    if details:
        logs = payload.get("preparation") or ["Waiting for output…"]
        panels.append(Panel(Text.from_ansi("\n".join(logs[-max(1, log_lines):])),
                            title="DETAILS", title_align="left", border_style="bright_black"))
    return Group(*panels)
