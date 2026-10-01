"""Dashboard/database and remote-provider capability probes."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from .diagnostics import Capability, Check, module_check
from .paths import InstallationPaths


def _config_has_secret(path: Path) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text().splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == "CS_SECRET_KEY" and value.strip():
            return True
    return False


def _database_check(url: str) -> Check:
    if not url.startswith("sqlite:///"):
        try:
            from sqlalchemy import create_engine, inspect

            engine = create_engine(url, future=True)
            names = set(inspect(engine).get_table_names())
            engine.dispose()
        except Exception as exc:
            return Check(
                "database", False, f"cannot connect to configured database: {exc}"
            )
        required = {"users", "tracks", "benchmarks", "runs"}
        missing = sorted(required - names)
        if missing:
            return Check("database", False, "missing tables: " + ", ".join(missing))
        return Check("database", True, "configured external database")
    path = Path(url.removeprefix("sqlite:///"))
    if not path.is_file():
        return Check("database", False, f"not installed: {path}")
    try:
        connection = sqlite3.connect(f"file:{path}?mode=rw", uri=True, timeout=2)
        try:
            names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            connection.close()
    except sqlite3.Error as exc:
        return Check("database", False, f"cannot open {path}: {exc}")
    required = {"users", "tracks", "benchmarks", "runs"}
    missing = sorted(required - names)
    if missing:
        return Check("database", False, "missing tables: " + ", ".join(missing))
    return Check("database", True, str(path))


def dashboard_capability(paths: InstallationPaths) -> Capability:
    resource_manifest = paths.resource_root / "catalog/tracks.yaml"
    writable = paths.home.is_dir() and os.access(paths.home, os.W_OK | os.X_OK)
    return Capability(
        "dashboard",
        "Dashboard and queue",
        (
            Check(
                "installation",
                paths.install_record.is_file(),
                str(paths.install_record)
                if paths.install_record.is_file()
                else "run cua-speedrun install",
            ),
            Check(
                "home",
                writable,
                f"{paths.home} {'is writable' if writable else 'is not writable'}",
            ),
            Check(
                "configuration",
                _config_has_secret(paths.config_file),
                str(paths.config_file)
                if paths.config_file.is_file()
                else "configuration is missing",
            ),
            Check(
                "resources",
                resource_manifest.is_file(),
                str(paths.resource_root)
                if resource_manifest.is_file()
                else "installed catalog is missing",
            ),
            _database_check(os.environ["CS_DATABASE_URL"]),
            module_check("web service", "fastapi"),
            module_check("database runtime", "sqlalchemy"),
            module_check("web server", "uvicorn"),
        ),
    )


def _saved_modal_credentials(url: str) -> bool:
    if not url.startswith("sqlite:///"):
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(url, future=True)
            with engine.connect() as connection:
                row = connection.execute(
                    text(
                        "SELECT 1 FROM users WHERE modal_token_id IS NOT NULL "
                        "AND modal_token_secret_enc IS NOT NULL LIMIT 1"
                    )
                ).first()
            engine.dispose()
            return row is not None
        except Exception:
            return False
    path = Path(url.removeprefix("sqlite:///"))
    if not path.is_file():
        return False
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
        try:
            row = connection.execute(
                "SELECT 1 FROM users WHERE modal_token_id IS NOT NULL "
                "AND modal_token_secret_enc IS NOT NULL LIMIT 1"
            ).fetchone()
        finally:
            connection.close()
        return row is not None
    except sqlite3.Error:
        return False


def modal_capability() -> Capability:
    credentials = _saved_modal_credentials(os.environ["CS_DATABASE_URL"])
    return Capability(
        "modal",
        "Remote execution through Modal",
        (
            module_check("Modal client", "modal"),
            Check(
                "user credentials",
                credentials,
                "at least one dashboard user has saved Modal credentials"
                if credentials
                else "save Modal credentials in the dashboard before a remote run",
            ),
        ),
    )


__all__ = ["dashboard_capability", "modal_capability"]
