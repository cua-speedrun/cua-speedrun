"""Idempotent machine installation for cua-speedrun."""

from __future__ import annotations

import argparse
import json
import secrets
import stat

from cua_speedrun import __version__

from .dependencies import install_local_agent_python, install_runtime_dependencies
from .install_assets import (
    install_full_osworld,
    install_resources,
    prepare_osworld_image,
)
from .paths import InstallationPaths, add_home_argument, configure_process


def register_install_command(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "install",
        help="install machine-local data and runtime dependencies",
    )
    add_home_argument(parser)
    parser.add_argument(
        "--skip-python-deps",
        action="store_true",
        help="do not install the runtime dependency extra",
    )
    parser.add_argument(
        "--skip-full-osworld",
        action="store_true",
        help="install the representative set but do not fetch all 369 tasks",
    )
    image = parser.add_mutually_exclusive_group()
    image.add_argument(
        "--prepare-osworld-image",
        action="store_true",
        help="prepare the pinned OSWorld VM image even when auto-detection fails",
    )
    image.add_argument(
        "--skip-osworld-image",
        action="store_true",
        help="do not prepare the large machine-local OSWorld VM image",
    )
    parser.set_defaults(_operator_handler=run_install)


def _write_config(paths: InstallationPaths) -> None:
    existing = paths.config_file.read_text() if paths.config_file.is_file() else ""
    names = {
        line.partition("=")[0].strip()
        for line in existing.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }
    additions: list[str] = []
    if "CS_SECRET_KEY" not in names:
        additions.append(f"CS_SECRET_KEY={secrets.token_urlsafe(48)}")
    if "CS_DEV_LOGIN" not in names:
        additions.append("CS_DEV_LOGIN=1")
    if not existing:
        existing = (
            "# Machine-local cua-speedrun settings. Environment variables "
            "override these values.\n"
        )
    if additions:
        if existing and not existing.endswith("\n"):
            existing += "\n"
        existing += "\n".join(additions) + "\n"
        paths.config_file.write_text(existing)
    paths.config_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    print(f"[ok] Configuration: {paths.config_file}")


def _initialize_database() -> None:
    from cua_speedrun.service.catalog import sync_catalog
    from cua_speedrun.service.db import make_session_factory

    counts = sync_catalog(make_session_factory())
    print(
        "[ok] Database and catalog: "
        f"{counts['tracks_added']} tracks added, "
        f"{counts['benchmarks_added']} benchmarks added"
    )


def run_install(args: argparse.Namespace) -> int:
    paths = InstallationPaths.resolve(args.home)
    paths.create_directories()
    print(f"cua-speedrun home: {paths.home}")
    if not args.skip_python_deps:
        install_runtime_dependencies(paths)
    install_local_agent_python(paths)
    _write_config(paths)
    install_resources(paths)
    configure_process(paths)
    if not args.skip_full_osworld:
        install_full_osworld(paths)
    image_status = prepare_osworld_image(
        paths,
        force_attempt=args.prepare_osworld_image,
        skip=args.skip_osworld_image,
    )
    _initialize_database()
    paths.install_record.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cua_speedrun_version": __version__,
                "home": str(paths.home),
                "local_agent_python": str(paths.agent_python),
                "osworld_image": image_status,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("\nInstallation complete.")
    print("Next: cua-speedrun doctor")
    print("Then: cua-speedrun dashboard")
    return 0


__all__ = ["register_install_command", "run_install"]
