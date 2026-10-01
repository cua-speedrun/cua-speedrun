"""A small dashboard for running cua-speedrun from a Modal notebook.

    %uv pip install -q cua-speedrun
    from cua_speedrun.notebook import dashboard
    dashboard()

It uses the same hosted evaluations as `cua-speedrun benchmark --host modal`,
so every run happens in the Modal account that runs the notebook and keeps
going after the notebook stops.
"""

from __future__ import annotations

import argparse
import contextlib
from html import escape
import io
import json
import os
from pathlib import Path
import threading
import time

import yaml

UPLOAD = "Upload my own agent"


def variables_for(agent: Path | None, dataset: Path) -> tuple[list[str], list[str]]:
    """Variables the agent and task set require, and ones they can use."""
    from cua_speedrun.service.templates_catalog import agent_metadata

    metadata = agent_metadata(agent) if agent else {}
    definition = dataset / "benchmark-source.yaml"
    if not definition.is_file():
        definition = dataset / "manifest.yaml"
    data = yaml.safe_load(definition.read_text())
    required = [*metadata.get("required_environment_variables", []),
                *(data.get("evaluator_environment") or {}).get("required", [])]
    optional = [*metadata.get("optional_environment_variables", []),
                *(data.get("host_runtime") or {}).get("forward_env", [])]
    required = list(dict.fromkeys(required))
    return required, [name for name in dict.fromkeys(optional) if name not in required]


def save_upload(files: dict[str, bytes], root: Path) -> Path:
    """Write uploaded agent files into a new folder and check them."""
    from cua_speedrun.commands.benchmark import validate_agent

    extra = sorted(set(files) - {"agent.py", "init.py", "agent.json"})
    if extra or "agent.py" not in files:
        raise ValueError("upload agent.py, plus init.py and agent.json if you have them")
    folder = root / time.strftime("agent-%Y%m%d-%H%M%S")
    folder.mkdir(parents=True)
    for name, content in files.items():
        (folder / name).write_bytes(content)
    if "init.py" not in files:
        (folder / "init.py").write_text('print("ready")\n')
    validate_agent(folder)
    return folder


def parse_variables(text: str) -> dict[str, str]:
    """Read NAME=value lines."""
    values = {}
    for line in text.splitlines():
        if line.strip():
            name, separator, value = line.partition("=")
            if not separator or not name.strip():
                raise ValueError(f"write other variables as NAME=value: {line.strip()!r}")
            values[name.strip()] = value.strip()
    return values


def launch_args(agent: str, dataset: str, parallel: int, extra: list[str]) -> argparse.Namespace:
    """The same arguments as `cua-speedrun benchmark --host modal`."""
    from cua_speedrun.commands import register_operator_commands

    parser = argparse.ArgumentParser()
    register_operator_commands(parser.add_subparsers(dest="command"))
    argv = ["benchmark", "--host", "modal", "--agent", agent, "--dataset", dataset,
            "--parallel-evaluations", str(parallel)]
    for name in extra:
        argv += ["--env", name]
    return parser.parse_args(argv)


class _LineWriter(io.TextIOBase):
    """Send each printed line to a callback, from any thread."""

    def __init__(self, on_line):
        self._on_line, self._pending = on_line, ""

    def write(self, text: str) -> int:
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._on_line(line)
        return len(text)


def startup_line(line: str, steps: dict[str, dict]) -> str | None:
    """Turn a launch progress record into readable text; None for other lines."""
    from cua_speedrun.startup import PREFIX

    if not line.startswith(PREFIX):
        return None
    try:
        event = json.loads(line[len(PREFIX):])
        steps[event["key"]] = event
    except (ValueError, KeyError, TypeError):
        return None
    parts = []
    for step in steps.values():
        state = {"running": "running…", "failed": "failed"}.get(step["state"], "done")
        if step["state"] == "done" and step.get("elapsed_sec") is not None:
            state = f"done in {step['elapsed_sec']:.0f} s"
        parts.append(f"{step['label']}: {state}")
    return " · ".join(parts)


def status_html(payload: dict, width: int = 110) -> str:
    """The same status view as `cua-speedrun status`, as HTML for a notebook."""
    import io
    from rich.console import Console
    from cua_speedrun.commands.status_tui import render_status

    # In a notebook rich would otherwise display each render in the cell output.
    console = Console(record=True, width=width, file=io.StringIO(), color_system="truecolor",
                      force_jupyter=False)
    console.print(render_status(payload, width=width))
    return console.export_html(
        inline_styles=True,
        code_format=("<pre style='font-family:Menlo,Consolas,monospace;font-size:12px;"
                     "line-height:1.35;background:#111;color:#ddd;padding:12px;"
                     "border-radius:6px;overflow-x:auto'>{code}</pre>"))


def dashboard(refresh_sec: float = 5.0) -> None:
    """Show the evaluation form and the evaluations in this Modal account."""
    import ipywidgets as w
    from IPython.display import display

    from cua_speedrun.commands.paths import InstallationPaths
    from cua_speedrun.commands.wizard import catalog_entries
    from cua_speedrun.hosted.client import HostedEvaluations

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            service = HostedEvaluations.open()
    except ValueError:
        # A notebook normally has its workspace's token pair already.
        display(w.HTML(
            "<p>This notebook has no Modal token pair. Create a token at "
            "<a href='https://modal.com/settings/tokens'>modal.com/settings/tokens</a>, "
            "save it as a Modal Secret with MODAL_TOKEN_ID and MODAL_TOKEN_SECRET, "
            "attach the Secret to this notebook, and run the cell again.</p>"))
        return
    agents, datasets = catalog_entries(InstallationPaths.resolve())
    agent_paths = {a["name"]: a["path"] for a in agents}
    dataset_paths = {d["name"]: d["path"] for d in datasets}

    agent = w.Dropdown(options=[*agent_paths, UPLOAD], description="Agent")
    upload = w.FileUpload(accept=".py,.json", multiple=True, description="agent.py")
    dataset = w.Dropdown(options=[(f"{d['name']} ({d['tasks']} tasks)", d["name"]) for d in datasets],
                         description="Task set", value="osworld-50" if "osworld-50" in dataset_paths else None)
    keys = w.VBox()
    other = w.Textarea(placeholder="Other variables, one NAME=value per line (for example HF_TOKEN=...)",
                       layout=w.Layout(width="480px"))
    parallel = w.BoundedIntText(value=1, min=1, max=32, description="Parallel")
    start = w.Button(description="Start evaluation", button_style="success")
    message = w.HTML()
    runs = w.HTML("<p>Loading…</p>")
    chosen = w.Dropdown(description="Evaluation", layout=w.Layout(width="480px"))
    cancel = w.Button(description="Cancel")
    download = w.Button(description="Download results")
    log = w.HTML()
    log_lines: list[str] = []
    fields: dict[str, w.Password] = {}

    def show_keys(*_):
        upload.layout.display = "" if agent.value == UPLOAD else "none"
        required, optional = variables_for(agent_paths.get(agent.value), dataset_paths[dataset.value])
        fields.clear()
        for name in required + optional:
            hint = "already set" if os.environ.get(name) else ("required" if name in required else "optional")
            fields[name] = w.Password(description=name, placeholder=hint, layout=w.Layout(width="480px"),
                                      style={"description_width": "initial"})
        keys.children = list(fields.values())

    latest: dict[str, dict] = {}

    def show_selected(*_):
        payload = latest.get(chosen.value)
        runs.value = status_html(payload) if payload else "<p>No evaluations yet.</p>"

    def update_runs(*_):
        try:
            rows = service.evaluations(limit=10)
        except Exception as exc:
            runs.value = f"<p>Could not read evaluations: {escape(str(exc))}</p>"
            return
        latest.clear()
        latest.update({row["run_id"]: row for row in rows})
        options = [(f"{r['run_id']} · {r.get('name') or ''} · {r.get('stage') or ''}", r["run_id"])
                   for r in rows]
        if tuple(options) != tuple(chosen.options):
            current = chosen.value
            chosen.options = options
            values = [value for _, value in options]
            chosen.value = current if current in values else (values[0] if values else None)
        show_selected()

    def background(action):
        steps: dict[str, dict] = {}

        def on_line(line: str) -> None:
            progress = startup_line(line, steps)
            if progress is not None:
                message.value = f"<p>{escape(progress)}</p>"
            elif line.strip():
                log_lines.append(line)
                del log_lines[:-200]
                log.value = "<pre style='white-space:pre-wrap'>" + escape("\n".join(log_lines)) + "</pre>"

        def run():
            start.disabled = True
            writer = _LineWriter(on_line)
            try:
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    action()
            except Exception as exc:
                message.value = f"<p style='color:#b00020'>{escape(str(exc))}</p>"
            finally:
                start.disabled = False
                update_runs()
        threading.Thread(target=run, daemon=True).start()

    def on_start(_):
        def action():
            values = {name: field.value for name, field in fields.items() if field.value}
            extra = parse_variables(other.value)
            os.environ.update({**values, **extra})
            if agent.value == UPLOAD:
                files = {item["name"]: bytes(item["content"]) for item in upload.value}
                selected = str(save_upload(files, Path.home() / "cua-speedrun-agents"))
            else:
                selected = agent.value
            message.value = "<p>Starting…</p>"
            queued = service.launch(launch_args(selected, dataset.value, parallel.value, list(extra)))
            message.value = (f"<p>Started <b>{escape(queued['run_id'])}</b>. "
                             "It keeps running if you close this notebook.</p>")
            update_runs()
            chosen.value = queued["run_id"]
            tabs.selected_index = 1
        background(action)

    def on_cancel(_):
        if chosen.value:
            message.value = f"<p>Cancelling {escape(chosen.value)}…</p>"
            background(lambda: service.cancel(chosen.value))

    def on_download(_):
        if not chosen.value:
            return
        def action():
            path = Path.cwd() / f"{chosen.value}.zip"
            service.export(chosen.value, path)
            message.value = f"<p>Saved <b>{escape(path.name)}</b>. Download it from the file panel.</p>"
        background(action)

    chosen.observe(show_selected, "value")
    agent.observe(show_keys, "value")
    dataset.observe(show_keys, "value")
    start.on_click(on_start)
    cancel.on_click(on_cancel)
    download.on_click(on_download)
    show_keys()
    update_runs()

    def poll():
        while True:
            time.sleep(refresh_sec)
            update_runs()

    threading.Thread(target=poll, daemon=True).start()
    new = w.VBox([agent, upload, dataset, keys, other, parallel, start, message,
                  w.Accordion(children=[log], titles=("Log",))])
    previous = w.VBox([w.HBox([chosen, cancel, download]), runs])
    tabs = w.Tab(children=[new, previous], titles=("New evaluation", "Evaluations"))
    display(w.VBox([
        w.HTML("<h3>CUA-Speedrun</h3><p>Evaluations run in this Modal account and keep "
               "going after this notebook stops.</p>"),
        tabs,
    ]))
