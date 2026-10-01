"""Installed template copies must not drift from a development checkout.

`submit --template` resolves against the copy installed into
CUA_SPEEDRUN_HOME, so a template edited in the checkout but never
re-installed would otherwise ship old agent code. Template resolution fails
closed when those copies differ.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import cua_speedrun.resources as resources
from cua_speedrun.service import templates_catalog


def _make_template(root: Path, agent_body: str) -> Path:
    folder = root / "agents" / "demo"
    folder.mkdir(parents=True)
    (folder / "init.py").write_text("print('init')\n")
    (folder / "agent.py").write_text(agent_body)
    return folder


@pytest.fixture()
def installed(tmp_path: Path, monkeypatch):
    checkout = tmp_path / "checkout"
    installed_root = tmp_path / "installed"
    _make_template(checkout, "print('agent v2')\n")
    _make_template(installed_root, "print('agent v2')\n")
    monkeypatch.setattr(resources, "source_checkout_root", lambda: checkout)
    monkeypatch.setattr(
        templates_catalog, "TEMPLATES_DIR", installed_root / "agents"
    )
    return checkout, installed_root


def test_matching_copies_resolve(installed) -> None:
    _checkout, installed_root = installed
    path = templates_catalog.template_dir("demo")
    assert path == (installed_root / "agents" / "demo").resolve()


def test_stale_installed_copy_is_refused(installed) -> None:
    checkout, _installed_root = installed
    (checkout / "agents" / "demo" / "agent.py").write_text(
        "print('agent v3, fixed parser')\n"
    )
    with pytest.raises(RuntimeError, match="cua-speedrun install"):
        templates_catalog.template_dir("demo")


def test_without_a_checkout_the_installed_copy_wins(installed, monkeypatch) -> None:
    monkeypatch.setattr(resources, "source_checkout_root", lambda: None)
    assert templates_catalog.template_dir("demo") is not None


def test_checkout_resolution_skips_the_check(tmp_path: Path, monkeypatch) -> None:
    # When templates resolve straight from the checkout there is no copy to
    # drift; the guard must not compare a directory against itself.
    checkout = tmp_path / "checkout"
    _make_template(checkout, "print('agent')\n")
    monkeypatch.setattr(resources, "source_checkout_root", lambda: checkout)
    monkeypatch.setattr(
        templates_catalog, "TEMPLATES_DIR", checkout / "agents"
    )
    assert templates_catalog.template_dir("demo") is not None
