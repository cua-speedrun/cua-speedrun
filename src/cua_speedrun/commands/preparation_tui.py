"""Preparation phases with opt-in diagnostics from a CLI subprocess."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, RichLog, Static
from cua_speedrun.startup import PreparationProgress
from .preparation_view import render_preparation


class Preparation(App):
    TITLE = "cua-speedrun"
    CSS_PATH = "wizard.tcss"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [("ctrl+c", "cancel", "Stop preparation"), ("d", "toggle_details", "Details")]

    def __init__(self, argv: list[str], title: str, *, environment: dict | None = None, exit_on_error=False):
        super().__init__()
        self.argv = argv
        self.heading = title
        self.process = None
        self.result = None
        self.finished = False
        self.started = time.monotonic()
        self.environment = dict(environment or {})
        self.cancelled = False
        self.progress = PreparationProgress()
        self.exit_on_error = exit_on_error
        self.phase_label = "Preparing installation"

    def compose(self) -> ComposeResult:
        yield Static("CUA SPEEDRUN", classes="masthead")
        yield Static("PREPARATION", classes="step")
        with Vertical(classes="body"):
            yield Static(Text(self.heading), classes="title")
            yield Static("Preparing installation", id="phase")
            yield Static("", id="elapsed")
            yield Static(render_preparation({}), id="preparation-steps")
            yield RichLog(highlight=False, markup=False, wrap=True, min_width=0, max_lines=1000, id="logs")
        with Horizontal(classes="nav"):
            yield Button("Show details", id="details")
            yield Button("Stop preparation", id="stop")
        yield Footer()

    def on_mount(self):
        self.theme = "textual-dark"
        self.query_one(RichLog).display = False
        self.query_one("#phase", Static).display = False
        self.tick()
        self.set_interval(1, self.tick)
        self.run_worker(self.prepare(), exclusive=True)

    def tick(self):
        elapsed = int(time.monotonic() - self.started)
        self.query_one("#elapsed", Static).update(f"Elapsed {elapsed // 60:02d}:{elapsed % 60:02d}")
        self.query_one("#preparation-steps", Static).update(render_preparation({
            **self.progress.payload(), "preparation_label": self.phase_label,
        }))

    async def prepare(self):
        try:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "-m", "cua_speedrun.cli", *self.argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                env={**os.environ, **self.environment},
            )
            if self.cancelled:
                await self.stop_child()
                return

            async def read(stream, *, records=False):
                while line := await stream.readline():
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if records:
                        try:
                            payload = json.loads(text)
                        except json.JSONDecodeError:
                            self.query_one(RichLog).write(Text(text))
                        else:
                            self.result = payload
                            if payload.get("type") == "error":
                                self.query_one(RichLog).write(Text(payload["error"], style="#ff9285"))
                    else:
                        clean = Text.from_ansi(text)
                        if self.progress.feed(clean.plain):
                            self.tick()
                            continue
                        self.query_one(RichLog).write(clean)
                        if clean.plain.startswith("[prepare] ") and not clean.plain.endswith((".py", ".sh")):
                            self.phase_label = clean.plain[10:]
                            self.tick()

            await asyncio.gather(read(self.process.stdout, records=True), read(self.process.stderr))
            code = await self.process.wait()
            self.finished = True
            if code == 0 and self.result and self.result.get("type") in {"ready", "queued"}:
                self.exit(self.result)
                return
            self.query_one("#phase", Static).update(Text("Preparation failed · details below", style="#ff9285"))
            self.query_one("#phase", Static).display = True
            self.query_one("#preparation-steps", Static).display = False
            self.query_one(RichLog).display = True
            self.query_one("#details", Button).label = "Hide details"
            self.query_one("#stop", Button).label = "Close"
            if self.exit_on_error:
                self.exit(self.result or {"type": "error", "error": "preparation failed"})
        except asyncio.CancelledError:
            raise
        except (OSError, ValueError) as exc:
            self.finished = True
            self.query_one("#phase", Static).update(Text("Preparation failed", style="#ff9285"))
            self.query_one("#phase", Static).display = True
            self.query_one("#preparation-steps", Static).display = False
            self.query_one(RichLog).write(Text(str(exc), style="#ff9285"))
            self.query_one(RichLog).display = True
            self.query_one("#details", Button).label = "Hide details"
            self.query_one("#stop", Button).label = "Close"
            if self.exit_on_error:
                self.exit({"type": "error", "error": str(exc)})
        finally:
            if self.process and self.process.returncode is None:
                await self.stop_child()

    async def stop_child(self):
        if not self.process or self.process.returncode is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGINT)
            await asyncio.wait_for(self.process.wait(), 5)
        except asyncio.TimeoutError:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                await asyncio.wait_for(self.process.wait(), 5)
            except asyncio.TimeoutError:
                os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass

    async def action_cancel(self):
        self.cancelled = True
        await self.stop_child()
        self.exit(self.result if self.result and self.result.get("type") == "queued" else None)

    def action_toggle_details(self):
        logs = self.query_one(RichLog)
        logs.display = not logs.display
        self.query_one("#preparation-steps", Static).display = not logs.display and not self.finished
        self.query_one("#details", Button).label = "Hide details" if logs.display else "Show details"

    async def on_button_pressed(self, event):
        if event.button.id == "stop":
            await self.action_cancel()
        elif event.button.id == "details":
            self.action_toggle_details()
