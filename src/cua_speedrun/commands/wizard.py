"""Interactive configuration over the same public CLI used by automation."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import shlex
import sys

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Collapsible, Footer, Input, Label, OptionList, Select, Static
from textual.widgets.option_list import Option
import yaml

from cua_speedrun.benchmark_sources import benchmark_catalog_paths
from cua_speedrun.config import load_dotenv
from cua_speedrun.resources import resource_root
from cua_speedrun.runtime_environment import normalize_environment_name
from cua_speedrun.service.templates_catalog import agent_metadata

from .paths import InstallationPaths


def catalog_entries(paths: InstallationPaths) -> tuple[list[dict], list[dict]]:
    """Read choices without installing assets or starting an environment."""
    root = paths.resource_root if all((paths.resource_root / part).exists() for part in (
        "catalog/tracks.yaml", "agents", "benchmarks", "scripts")) else resource_root()
    agents = [{"name": p.name, "path": p, **agent_metadata(p)}
              for p in sorted((root / "agents").iterdir())
              if (p / "init.py").is_file() and (p / "agent.py").is_file()]
    datasets = []
    for path in benchmark_catalog_paths(root):
        definition = path / "benchmark-source.yaml"
        if not definition.exists():
            definition = path / "manifest.yaml"
        data = yaml.safe_load(definition.read_text())
        datasets.append({"name": str(data["name"]), "path": path,
                         "version": str(data["version"]), "tasks": len(data["tasks"]),
                         "environment": (data.get("prepare") or {}).get("default_environment", "modal")})
    return agents, datasets


def benchmark_argv(args: argparse.Namespace) -> list[str]:
    values = ["benchmark", "--json", "--background", "--no-input"]
    for key in ("home", "agent", "dataset", "gpu", "compute", "environment", "host",
                "parallel_evaluations", "name", "runner", "runner_template"):
        value = getattr(args, key, None)
        if value is not None and str(value):
            values.extend(["--" + key.replace("_", "-"), str(value)])
    for key in ("no_gpu", "no_preload"):
        if getattr(args, key, False):
            values.append("--" + key.replace("_", "-"))
    for name in args.env:
        values.extend(["--env", name])
    return values


class Wizard(App):
    TITLE = "cua-speedrun"
    CSS_PATH = "wizard.tcss"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [("ctrl+c", "quit", "Cancel")]

    def __init__(self, args, *, setup=False):
        super().__init__()
        self.args = copy.deepcopy(args)
        self.setup = setup
        self.paths = InstallationPaths.resolve(args.home)
        self.agents, self.datasets = catalog_entries(self.paths)
        self.environment = {}
        if not setup and self.args.dataset and not Path(self.args.dataset).expanduser().is_dir():
            name, _, version = self.args.dataset.partition("@")
            def normalized(value):
                return value.lower().replace("-", "").replace("_", "")
            matches = [d for d in self.datasets if normalized(d["name"]) == normalized(name)
                       and (not version or d["version"] == version)]
            if len(matches) == 1:
                self.args.dataset = matches[0]["name"]

    def on_mount(self):
        self.theme = "textual-dark"
        if self.setup:
            self.push_screen(Credentials())
        else:
            self.push_screen(Choose("agent"))


class Page(Screen):
    BINDINGS = [("escape", "back", "Back")]

    def action_back(self):
        if len(self.app.screen_stack) > 2:
            self.app.pop_screen()
        else:
            self.app.exit()

    def on_resize(self, event):
        self.set_class(event.size.width < 75, "narrow")

    def heading(self, step: str):
        yield Static("CUA SPEEDRUN", classes="masthead")
        yield Static(step, classes="step")

    def navigation(self, primary: str, *, back=True):
        with Horizontal(classes="nav"):
            if back:
                yield Button("Back", id="back")
            yield Button(primary, variant="primary", id="next")
        yield Footer()

    def show_error(self, message):
        self.query_one(".error", Static).update(Text(str(message)))


class Choose(Page):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def compose(self) -> ComposeResult:
        agent = self.kind == "agent"
        self.entries = self.app.agents if agent else self.app.datasets
        if not agent:
            selected = next((a for a in self.app.agents if a["name"] == self.app.args.agent), {})
            compatible = selected.get("compatible_benchmarks")
            if compatible:
                self.entries = [d for d in self.entries if d["name"] in compatible]
        yield from self.heading("01 / TEMPLATE" if agent else "02 / BENCHMARK")
        with Vertical(classes="body"):
            yield Static("Choose an agent template" if agent else "Choose a benchmark", classes="title")
            yield Static("↑ / ↓ to browse · Enter to select", classes="hint")
            options = []
            for item in self.entries:
                detail = (f"{item.get('gpu') or 'CPU / API'} · {item.get('description', '')}"
                          if agent else f"{item['tasks']} tasks · version {item['version']}")
                text = Text(item["name"], style="bold")
                text.append("\n" + detail, style="dim")
                options.append(Option(text, id=item["name"]))
            yield OptionList(*options, id="choices")
            yield Label("Or use your own folder", classes="custom-label")
            supplied = getattr(self.app.args, self.kind) or ""
            custom = supplied if supplied and not any(i["name"] == supplied for i in self.entries) else ""
            yield Input(custom, placeholder="/path/to/agent" if agent else "/path/to/benchmark", id="custom")
            yield Static("", classes="error")
        yield from self.navigation("Continue", back=not agent)

    def on_mount(self):
        choices = self.query_one(OptionList)
        supplied = getattr(self.app.args, self.kind)
        choices.highlighted = next((i for i, item in enumerate(self.entries) if item["name"] == supplied), 0) if self.entries else None
        choices.focus()

    def on_option_list_option_selected(self, event):
        self.choose(event.option.id)

    def on_input_submitted(self, event):
        self.advance()

    def on_button_pressed(self, event):
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "next":
            self.advance()

    def advance(self):
        custom = self.query_one("#custom", Input).value.strip()
        index = self.query_one(OptionList).highlighted
        if custom:
            self.choose(custom)
        elif index is not None:
            self.choose(self.entries[index]["name"])

    def choose(self, value):
        from .benchmark import validate_agent

        try:
            if not any(item["name"] == value for item in self.entries):
                folder = Path(value).expanduser().resolve()
                if self.kind == "agent":
                    validate_agent(folder)
                elif not (folder / "manifest.yaml").is_file():
                    raise ValueError("A custom benchmark needs manifest.yaml.")
                else:
                    data = yaml.safe_load((folder / "manifest.yaml").read_text())
                    if not isinstance(data, dict) or not all(k in data for k in ("name", "version", "tasks")):
                        raise ValueError("manifest.yaml needs name, version, and tasks.")
                    if not isinstance(data["tasks"], list) or not data["tasks"]:
                        raise ValueError("manifest.yaml must list at least one task.")
                value = str(folder)
            setattr(self.app.args, self.kind, value)
            self.app.push_screen(Choose("dataset") if self.kind == "agent" else Configuration())
        except (OSError, ValueError, yaml.YAMLError) as exc:
            self.show_error(exc)


def field(label, widget):
    with Vertical(classes="field"):
        yield Label(label)
        yield widget


class Configuration(Page):
    def compose(self) -> ComposeResult:
        args = self.app.args
        agent = next((a for a in self.app.agents if a["name"] == args.agent), None)
        if agent is None:
            agent = agent_metadata(Path(args.agent))
        dataset = next((d for d in self.app.datasets if d["name"] == args.dataset), None)
        if dataset is None:
            data = yaml.safe_load((Path(args.dataset) / "manifest.yaml").read_text())
            dataset = {"tasks": len(data["tasks"]), "environment": (data.get("prepare") or {}).get("default_environment", "modal")}
        yield from self.heading("03 / CONFIGURATION")
        with VerticalScroll(classes="body"):
            yield Static(Text(f"{args.agent}  →  {args.dataset}", style="bold"), classes="title")
            yield Static(f"{dataset['tasks']} tasks · Review the settings before starting.", classes="hint")
            with Horizontal(classes="row"):
                yield from field("Evaluation name", Input(args.name or Path(args.agent).name, id="name"))
                yield from field("Parallel evaluations", Input(str(args.parallel_evaluations), type="integer", id="parallel"))
                yield from field("Run controller on", Select([("This machine", "local"), ("Modal", "modal")],
                                                             value=args.host, allow_blank=False, id="host"))
            with Horizontal(classes="row"):
                yield from field("Agent compute", Select([("Modal", "modal"), ("Local", "local")], value=args.compute, allow_blank=False, id="compute"))
                yield from field("Desktop environment", Select([("Modal native", "modal-native"), ("Modal KVM", "modal"), ("Local", "local")], value=args.environment or dataset["environment"], allow_blank=False, id="environment"))
            with Horizontal(classes="row"):
                gpu = "CPU" if args.no_gpu else args.gpu or agent.get("gpu") or "CPU"
                yield from field("Agent hardware", Input(gpu, placeholder="CPU, L40S, A100…", id="gpu"))
                with Vertical(classes="field"):
                    yield Checkbox("Preload desktops", value=not args.no_preload, id="preload")
                    yield Checkbox("Run in background", value=args.background, id="background")
            self.required = agent.get("required_environment_variables") or []
            yield Static("", classes="hint", id="required")
            yield Button("Credentials & variables", id="credentials")
            with Collapsible(title="Advanced settings", collapsed=True):
                yield from field("Installation directory", Input(str(self.app.paths.home), id="home"))
                yield from field("Forward environment variables (names, separated by spaces)", Input(" ".join(args.env), id="env"))
                with Horizontal(classes="row"):
                    yield from field("Local runner", Select([("Installation default", ""), ("Local", "local"), ("Slurm", "slurm")], value=args.runner or "", allow_blank=False, id="runner"))
                    yield from field("Runner template", Input(args.runner_template or "", id="runner-template"))
        yield Static("", classes="error banner")
        yield from self.navigation("Start evaluation")

    def on_screen_resume(self):
        if not hasattr(self, "required"):
            return
        available = {**os.environ, **self.app.environment}
        self.query_one("#required", Static).update(Text(
            "Agent credentials: " + ", ".join(f"{name} ({'set' if available.get(name) else 'not set'})"
                                             for name in self.required) if self.required else ""))

    def read_config(self):
        args = copy.deepcopy(self.app.args)
        def value(id):
            return self.query_one("#" + id, Input).value.strip()
        args.name = value("name") or None
        args.parallel_evaluations = int(value("parallel"))
        if args.parallel_evaluations < 1:
            raise ValueError("Parallel evaluations must be at least 1.")
        args.compute = str(self.query_one("#compute", Select).value)
        args.host = str(self.query_one("#host", Select).value)
        args.environment = str(self.query_one("#environment", Select).value)
        gpu = value("gpu")
        if not gpu:
            raise ValueError("Choose CPU or a GPU type.")
        args.no_gpu = gpu.upper() == "CPU"
        args.gpu = None if args.no_gpu else gpu
        args.no_preload = not self.query_one("#preload", Checkbox).value
        args.background = self.query_one("#background", Checkbox).value
        args.home = str(InstallationPaths.resolve(value("home") or None).home)
        args.env = [normalize_environment_name(n) for n in shlex.split(value("env"))]
        args.runner = str(self.query_one("#runner", Select).value) or None
        args.runner_template = value("runner-template") or None
        args._environment_values = dict(self.app.environment)
        args.env = sorted(set(args.env) | {key for key in self.app.environment if not key.startswith("MODAL_")})
        if (args.runner or args.runner_template) and args.compute != "local":
            raise ValueError("A local runner requires local agent compute.")
        if args.host == "modal" and (args.compute != "modal" or args.environment == "local"):
            raise ValueError("A Modal controller requires Modal agent and desktop placement.")
        return args

    def on_button_pressed(self, event):
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "credentials":
            self.app.push_screen(Credentials(evaluation=True))
        elif event.button.id == "next":
            try:
                self.app.exit(self.read_config())
            except ValueError as exc:
                self.show_error(exc)


class Credentials(Page):
    def __init__(self, *, evaluation=False):
        super().__init__()
        self.values = {}
        self.evaluation = evaluation

    def compose(self) -> ComposeResult:
        from .setup import modal_credentials
        token_id, secret = modal_credentials()
        if self.evaluation:
            self.values = dict(self.app.environment)
        yield from self.heading("EVALUATION / CREDENTIALS & VARIABLES" if self.evaluation else "SETUP / CREDENTIALS")
        with VerticalScroll(classes="body"):
            yield Static("Connect your compute and models", classes="title")
            yield Static("Values apply only to this evaluation. Blank fields keep existing values." if self.evaluation else
                         "Keys are stored locally with owner-only permissions. Blank fields keep existing values.", classes="hint")
            yield Static("Modal credentials detected" if token_id and secret else "Modal credentials · optional for local runs", classes="hint")
            yield from field("Modal token ID", Input(password=True, placeholder="ak-…", id="token-id"))
            yield from field("Modal token secret", Input(password=True, placeholder="as-…", id="token-secret"))
            yield Static("API keys and environment variables", classes="title")
            yield Static("Set model options or credentials by variable name. Add HF_TOKEN here if required.", classes="hint")
            with Horizontal(classes="row"):
                yield from field("Variable name", Input(placeholder="ANTHROPIC_API_KEY", id="key-name"))
                yield from field("Value", Input(password=True, id="key-value"))
            yield Button("Add / replace", id="add")
            yield Static(Text("\n".join(f"✓ {key}" for key in self.values)), id="credential-list")
            if not self.evaluation:
                yield Static(str(self.app.paths.home / "config.env"), classes="hint")
        yield Static("", classes="error banner")
        yield from self.navigation("Apply" if self.evaluation else "Save & finish", back=self.evaluation)

    def add_credential(self):
        name = self.query_one("#key-name", Input).value.strip()
        value = self.query_one("#key-value", Input).value.strip()
        if not name and not value:
            return
        name = normalize_environment_name(name)
        if not value:
            raise ValueError("Enter a credential value.")
        if any(c in value for c in "\r\n\x00"):
            raise ValueError("Credentials must be single-line values.")
        self.values[name] = value
        self.query_one("#credential-list", Static).update(Text("\n".join(f"✓ {key}" for key in self.values)))
        self.query_one("#key-name", Input).value = ""
        self.query_one("#key-value", Input).value = ""

    def on_button_pressed(self, event):
        try:
            if event.button.id == "back":
                self.app.pop_screen()
            elif event.button.id == "add":
                self.add_credential()
                self.query_one("#key-name", Input).focus()
            elif event.button.id == "next":
                self.add_credential()
                token_id = self.query_one("#token-id", Input).value.strip()
                secret = self.query_one("#token-secret", Input).value.strip()
                if token_id or secret:
                    if not token_id.startswith("ak-") or not secret.startswith("as-"):
                        raise ValueError("Enter both Modal credentials (ak-… and as-…).")
                    self.values.update(MODAL_TOKEN_ID=token_id, MODAL_TOKEN_SECRET=secret)
                if self.evaluation:
                    self.app.environment = dict(self.values)
                    self.app.pop_screen()
                else:
                    self.app.exit(dict(self.values))
        except ValueError as exc:
            self.show_error(exc)


def run_setup_wizard(args) -> int:
    from .preparation_tui import Preparation
    from .setup import _save_credentials

    paths = InstallationPaths.resolve(args.home)
    load_dotenv(paths.config_file)
    values = Wizard(args, setup=True).run()
    if values is None:
        return 130
    result = Preparation(["setup", "--json", "--no-input", "--home", str(paths.home)], "Setting up your installation").run()
    if not result or result.get("type") != "ready":
        return 1
    _save_credentials(paths, values)
    print("Ready. Run: cua-speedrun benchmark")
    return 0


def run_benchmark_wizard(args) -> int:
    paths = InstallationPaths.resolve(args.home)
    load_dotenv(paths.config_file)
    selected = Wizard(args).run()
    if selected is None:
        return 130
    return run_benchmark_preparation(selected)


def run_benchmark_preparation(selected, *, exit_on_error=False) -> int:
    from .preparation_tui import Preparation
    from .dashboard_client import register_dashboard_client_commands, run_status

    result = Preparation(benchmark_argv(selected), f"{selected.agent} → {selected.dataset}",
                         environment=getattr(selected, "_environment_values", {}),
                         exit_on_error=exit_on_error).run()
    if not result or result.get("type") != "queued":
        if result and result.get("type") == "error":
            print(f"error: {result['error']}", file=sys.stderr)
        return 1
    run_id = result["run_id"]
    print(f"Evaluation {run_id} queued")
    if selected.background:
        home_arg = f" --home {shlex.quote(str(selected.home))}" if selected.home else ""
        print(f"Follow: cua-speedrun status {run_id}{home_arg}")
        return 0
    parser = argparse.ArgumentParser()
    register_dashboard_client_commands(parser.add_subparsers(dest="command"))
    argv = ["status", str(run_id)]
    if selected.home:
        argv += ["--home", str(selected.home)]
    return run_status(parser.parse_args(argv))
