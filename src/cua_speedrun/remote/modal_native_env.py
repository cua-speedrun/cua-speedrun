"""Launcher for the modal-native env sandbox (runs on the executor).

The modal-native counterpart of ``create_env_sandbox`` in ``modal_env.py``. It
returns the SAME ``EnvSandbox`` handle, so everything downstream in ``run.py``
(GatewayControl, arm, exec_agent, wait_done, pull_env_artifacts) is unchanged.
The only difference from the QEMU launcher is what the sandbox boots: instead of
a debian-slim image running nested QEMU via gym-anything, it boots the benchmark's
modal-native base image (defaulting to the OSWorld snapshot built by
``scripts/build_osworld_modal_base.py``) and launches
``cua_speedrun.remote.modal_native_env_plane`` in it.

The base image is resolved from the provenance cache by ``cache_key()`` and its
provenance is validated before use, the modal-native analog of the qcow2
preflight in ``envs/gym_anything.py``. The sandbox still runs on Modal's VM
runtime, because the default sandbox runtime denies the ``unshare`` +
``pivot_root`` + systemd boot outright (verified: ``unshare`` fails with
EPERM there). What is gone relative to the old QEMU path is ``/dev/kvm``:
nothing is nested-virtualized; the desktop runs directly on the sandbox
kernel.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any, Mapping

from cua_speedrun.remote import osworld_modal_base as base

_CS_PACKAGE = Path(__file__).resolve().parents[1]  # the cua_speedrun package dir

# Modal refuses every build layer (apt/pip/run_commands) on top of a
# filesystem-snapshot image; only mount layers work. So the base snapshot
# already contains everything the env-plane process needs (Python and the
# benchmark's verifier dependencies), and this launcher only mounts the
# cua_speedrun package and the benchmark env directory into it.

# Boot budget for wait_env_healthy: covers pulling the snapshot image onto the
# worker, booting the rootfs's systemd, the GDM autologin session reaching a
# rendered gnome-shell, the task's pre_task setup, and the evaluator warm
# import.
BOOT_BUDGET_SEC = 420


def build_launch_env(
    *,
    env_spec: "Mapping[str, Any]",
    seed: int,
    timeout_sec: float,
    grace_sec: float,
    gateway_port: int,
    run_token: str,
    control_token: str,
    task_label: str | None,
    desktop_user: str,
) -> dict[str, str]:
    """The CS_* environment the modal-native env-plane reads. Pure, so the
    contract is unit-testable without Modal. ``env_spec`` is the task's env
    block, forwarded verbatim (env_dir rewritten to the in-sandbox mount) to
    the registered modal-native Backend."""
    from cua_speedrun.remote.modal_env import _env_spec_payload

    env = {
        "PYTHONPATH": "/opt/cs",
        "CS_ENV_BACKEND": "modal-native",
        "CS_ENV_SPEC": _env_spec_payload(env_spec),
        "CS_SEED": str(seed),
        "CS_TIMEOUT_SEC": str(timeout_sec),
        "CS_GRACE_SEC": str(grace_sec),
        "CS_GATEWAY_PORT": str(gateway_port),
        "CS_RUN_TOKEN": run_token,
        "CS_CONTROL_TOKEN": control_token,
        "CS_DESKTOP_USER": desktop_user,
    }
    if task_label:
        env["CS_TASK_LABEL"] = task_label
    return env


def resolve_base_image_id(
    cache: Any, *, cache_key: str | None = None, expected: dict[str, Any] | None = None,
) -> str:
    """Resolve and provenance-validate the base image id from the cache Dict.

    Raises if no image has been built for this recipe/source/delta, the same
    way a QEMU run fails preflight when the prepared qcow2 is missing."""
    record = cache.get(cache_key if cache_key is not None else base.cache_key())
    if not isinstance(record, dict):
        if cache_key is not None:
            raise RuntimeError(f"native desktop image has not been built: {cache_key}")
        raise RuntimeError(
            "no OSWorld modal-native base image is built for this contract; run "
            "scripts/build_osworld_modal_base.py first "
            f"(cache key {base.cache_key()})"
        )
    base.validate_base_provenance(
        record, expected if expected is not None else base.expected_provenance_block(),
    )
    return record["modal_snapshot_image_id"]


def create_modal_native_env_sandbox(
    *,
    env_local_dir: Path,
    env_spec: "Mapping[str, Any]",
    seed: int,
    timeout_sec: float,
    grace_sec: float,
    task_label: str | None = None,
    gateway_port: int = 8390,
    region: str | None = None,
    sandbox_timeout_sec: int = 3600,
    desktop_user: str = base.DESKTOP_USER,
    runtime_env: Mapping[str, str] | None = None,
):
    import modal

    from cua_speedrun.remote.modal_env import EnvSandbox

    run_token = secrets.token_urlsafe(16)
    control_token = secrets.token_urlsafe(16)

    from cua_speedrun.remote.modal_env import _environment_resources, _host_runtime

    runtime = _host_runtime(env_local_dir)
    if runtime.get("native_runner") == "gym-anything":
        from cua_speedrun.remote.modal_env import create_env_sandbox

        return create_env_sandbox(
            env_local_dir, env_spec, seed, timeout_sec, grace_sec,
            task_label=task_label, gateway_port=gateway_port, region=region,
            sandbox_timeout_sec=sandbox_timeout_sec, runtime_env=runtime_env,
            native_runner=True,
        )

    contract_path = Path(env_local_dir) / "native-image.json"
    contract = json.loads(contract_path.read_text()) if contract_path.is_file() else None
    if contract:
        expected = contract["expected_provenance"]
        if not (expected.get("source_image_sha256") or expected.get("source_archive_sha256")) or not expected.get("recipe_sha256"):
            raise ValueError("native image source and recipe hashes must be pinned")
        if expected.get("validated") is not True:
            raise ValueError("native image must require completed validation")
        app = modal.App.lookup(contract["app_name"], create_if_missing=True)
        cache = modal.Dict.from_name(contract["cache_name"], create_if_missing=True)
        image_id = resolve_base_image_id(
            cache, cache_key=contract["cache_key"], expected=expected,
        )
        desktop_user = contract.get("desktop_user", desktop_user)
    else:
        env_json = Path(env_local_dir) / "env.json"
        config = json.loads(env_json.read_text()) if env_json.is_file() else {}
        source = (config.get("qemu_base_image_provenance") or {}).get(
            "source_image_sha256"
        )
        if source and source != base.OSWORLD_IMAGE_CONTRACT["source_image_sha256"]:
            raise ValueError(
                "this environment needs its own native-image.json; "
                "the default desktop image does not match"
            )
        app = modal.App.lookup(base.MODAL_APP_NAME, create_if_missing=True)
        cache = modal.Dict.from_name(base.BASE_IMAGE_DICT_NAME, create_if_missing=True)
        image_id = resolve_base_image_id(cache)

    image = (
        modal.Image.from_id(image_id)
        .add_local_dir(str(_CS_PACKAGE), remote_path="/opt/cs/cua_speedrun")
        .add_local_dir(str(env_local_dir), remote_path="/envs/env")
    )

    env = build_launch_env(
        env_spec=env_spec, seed=seed, timeout_sec=timeout_sec, grace_sec=grace_sec,
        gateway_port=gateway_port, run_token=run_token, control_token=control_token,
        task_label=task_label,
        desktop_user=desktop_user,
    )

    create_kwargs: dict[str, Any] = dict(
        # Reserve CPU capacity for the llvmpipe-rendered GNOME desktop.
        app=app, image=image, cpu=4, memory=16384,
        timeout=sandbox_timeout_sec, encrypted_ports=[gateway_port], env=env,
        # The VM runtime provides the real kernel that unshare + pivot_root +
        # systemd need (the default runtime denies unshare with EPERM). No
        # /dev/kvm is requested or used; nothing is nested-virtualized.
        experimental_options={"vm_runtime": True},
    )
    runtime = _host_runtime(env_local_dir)
    available = {**os.environ, **dict(runtime_env or {})}
    forwarded = {
        name: available[name]
        for name in runtime.get("forward_env", [])
        if available.get(name)
    }
    for selector, api_key_name in list(forwarded.items()):
        if selector.endswith("_API_KEY_ENV") and available.get(api_key_name):
            forwarded[api_key_name] = available[api_key_name]
    if forwarded:
        create_kwargs["secrets"] = [modal.Secret.from_dict(forwarded)]
    if contract:
        create_kwargs["cpu"], create_kwargs["memory"] = _environment_resources(
            env_local_dir, env_spec
        )
        # These are image-owned readiness settings, not submission overrides.
        env["CS_NATIVE_SERVICES"] = json.dumps(contract.get("services", []))
        if contract.get("volumes"):
            create_kwargs["volumes"] = {
                mount: modal.Volume.from_name(name, create_if_missing=True)
                for mount, name in contract["volumes"].items()
            }
            env["GYM_ANYTHING_QEMU_CACHE"] = "/cache/qemu"
    if region:
        create_kwargs["region"] = region

    sandbox = modal.Sandbox.create(
        "bash", "-lc",
        # pipefail so the sandbox exit code is the env-plane's, not tee's; a
        # failed boot must not read as a clean exit to the executor.
        "set -o pipefail; mkdir -p /tmp/cs_run && "
        "python3 -u -m cua_speedrun.remote.modal_native_env_plane "
        "2>&1 | tee /tmp/cs_run/env_plane.log",
        **create_kwargs,
    )
    try:
        base_url = sandbox.tunnels(timeout=300)[gateway_port].url.rstrip("/")
    except BaseException:
        # Sandbox.create succeeds even when the sandbox can never start (for
        # example a region with no capacity); tunnels() is where that surfaces.
        # Terminate so a failed create does not leak an idling sandbox.
        sandbox.terminate()
        raise
    return EnvSandbox(
        sandbox=sandbox, base_url=base_url, run_token=run_token,
        control_token=control_token, gateway_port=gateway_port,
        boot_budget_sec=(
            int(contract.get("boot_budget_sec", BOOT_BUDGET_SEC))
            if contract
            else BOOT_BUDGET_SEC
        ),
    )
