"""Scope checks; live checkpoint validation runs separately on Modal."""

import os

import pytest

from cua_speedrun.envs.cua_world_runtime import prepare


@pytest.mark.skipif(os.environ.get("MODAL_SANDBOX_ID"), reason="local-scope check")
def test_preparation_returns_before_accessing_an_environment_outside_modal():
    # No VM, runner, or simulated environment is constructed by this check.
    assert prepare(None) is None
