"""Initialize a local installation without downloading model or VM assets."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys

from cua_speedrun import __version__
from cua_speedrun.runtime_environment import normalize_environment_name

from .install import _initialize_database, _write_config
from .install_assets import install_resources
from .paths import InstallationPaths, add_home_argument, configure_process


def register_setup_command(subparsers) -> None:
    parser = subparsers.add_parser("setup", help="configure this installation and credentials")
    add_home_argument(parser)
    parser.add_argument("--no-input", action="store_true", help="skip credential prompts")
    parser.add_argument("--json", action="store_true", help="emit JSON; never prompt")
    parser.set_defaults(_operator_handler=run_setup)


def initialize(paths: InstallationPaths, *, refresh: bool = False) -> None:
    installed_version = None
    if paths.install_record.is_file():
        installed_version = json.loads(paths.install_record.read_text()).get("cua_speedrun_version")
    if refresh or installed_version != __version__:
        paths.create_directories()
        _write_config(paths)
        install_resources(paths)
        configure_process(paths)
        _initialize_database()
        paths.install_record.write_text(json.dumps({
            "schema_version": 1,
            "cua_speedrun_version": __version__,
            "home": str(paths.home),
        }, indent=2) + "\n")
    else:
        configure_process(paths)


def modal_credentials() -> tuple[str, str]:
    """Use process credentials or the active Modal CLI profile as a pair."""
    token_id = os.environ.get("MODAL_TOKEN_ID", "").strip()
    token_secret = os.environ.get("MODAL_TOKEN_SECRET", "").strip()
    if not token_id and not token_secret:
        from modal.config import config

        token_id = config.get("token_id") or ""
        token_secret = config.get("token_secret") or ""
    return token_id, token_secret


def _save_credentials(paths: InstallationPaths, values: dict[str, str]) -> None:
    if not values:
        return
    lines = paths.config_file.read_text().splitlines()
    lines = [line for line in lines if line.partition("=")[0].strip() not in values]
    for name, value in values.items():
        if any(char in value for char in "\r\n\x00"):
            raise ValueError(f"{name} must be a single-line value")
        lines.append(f"{name}={value}")
    # The existing config is owner-readable only; preserve its permissions.
    paths.config_file.write_text("\n".join(lines) + "\n")


def run_setup(args: argparse.Namespace) -> int:
    from .output import diagnostics, emit, interactive

    if interactive(args):
        from .wizard import run_setup_wizard

        return run_setup_wizard(args)
    paths = InstallationPaths.resolve(args.home)
    with diagnostics(getattr(args, "json", False)):
        initialize(paths, refresh=True)
    if not getattr(args, "json", False) and not args.no_input and sys.stdin.isatty():
        values = {}
        token_id, token_secret = modal_credentials()
        if not token_id or not token_secret:
            print("For cloud runs, enter your Modal token pair; leave blank for local runs.")
            token_id = getpass.getpass("Modal token ID: ").strip()
            if token_id:
                token_secret = getpass.getpass("Modal token secret: ").strip()
                if not token_id.startswith("ak-") or not token_secret.startswith("as-"):
                    raise ValueError("Modal tokens must start with ak- and as-")
                values.update(MODAL_TOKEN_ID=token_id, MODAL_TOKEN_SECRET=token_secret)
        name = input("API key variable, e.g. ANTHROPIC_API_KEY (Enter to skip): ").strip()
        if name:
            name = normalize_environment_name(name)
            if not name.endswith("_API_KEY"):
                raise ValueError("use the API provider's variable ending in _API_KEY")
            value = getpass.getpass(f"{name}: ").strip()
            if value:
                values[name] = value
        _save_credentials(paths, values)
    if getattr(args, "json", False):
        emit({"type": "ready", "home": str(paths.home), "version": __version__})
    else:
        print("Ready. Run: cua-speedrun benchmark")
    return 0
