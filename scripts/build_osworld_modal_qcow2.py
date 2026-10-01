"""Prepare the OSWorld qcow2 for the modal-remote (nested QEMU) path.

The modal-remote env sandbox boots the base image named by
``OSWORLD_QEMU_BASE_IMAGE`` from the shared ``gym-anything-qemu-cache``
volume. Locally that image is produced by ``scripts/prepare_osworld_qcow2.sh``;
this builder runs that exact script inside a Modal vm_runtime sandbox (KVM,
libguestfs) and installs the result onto the volume for environment sandboxes.

The pinned source archive is downloaded on first use and cached on the
``cua-osworld-port`` volume.

Usage:
    python scripts/build_osworld_modal_qcow2.py

Requires MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment or ``.env``.
Idempotent: the prepare script exits fast when the installed image's
provenance already matches.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

OUTPUT = "/cache/qemu/osworld_ubuntu.qcow2"
SOURCE_VOLUME = "cua-osworld-port"          # holds the verified Ubuntu.qcow2.zip
CACHE_VOLUME = "gym-anything-qemu-cache"    # where env sandboxes read the image


def _load_modal_env() -> None:
    env_file = _REPO / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("MODAL_") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    _load_modal_env()
    import modal

    app = modal.App.lookup("cua-osworld-port", create_if_missing=True)
    source = modal.Volume.from_name(SOURCE_VOLUME, create_if_missing=True)
    cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)
    contract = json.loads((_REPO / "benchmarks/osworld-image.json").read_text())
    upstream = contract["official_source"]
    expected = {
        "schema_version": contract["provenance_schema_version"],
        "recipe": contract["recipe"],
        "source_revision": upstream["revision"],
        "archive_sha256": upstream["archive_sha256"],
        "source_image_sha256": upstream["image_sha256"],
        "ssh_prepared": True, "tools_prepared": True,
        "nopasswd_sudo": True, "guest_auth_prepared": True,
        "ssh_user": contract["guest"]["ssh_user"],
    }
    if "--reuse" in sys.argv:
        try:
            record = json.loads(b"".join(cache.read_file("/osworld_ubuntu.qcow2.provenance.json")))
            images = list(cache.iterdir("/osworld_ubuntu.qcow2"))
        except (FileNotFoundError, json.JSONDecodeError):
            record, images = {}, []
        if images and all(record.get(key) == value for key, value in expected.items()):
            print("OSWorld desktop image is ready", flush=True)
            return 0

    # linux-image-amd64 provides the kernel the libguestfs supermin appliance
    # boots for virt-customize; the sandbox itself runs on Modal's VM runtime
    # with KVM, so the appliance is hardware-accelerated.
    image = (
        modal.Image.debian_slim()
        .apt_install(
            "libguestfs-tools", "linux-image-amd64", "qemu-utils",
            "curl", "unzip", "util-linux", "gawk",
        )
        .add_local_file(
            str(_REPO / "benchmarks" / "osworld-image.json"),
            remote_path="/repo/benchmarks/osworld-image.json",
        )
        .add_local_file(
            str(_REPO / "scripts" / "prepare_osworld_qcow2.sh"),
            remote_path="/repo/scripts/prepare_osworld_qcow2.sh",
        )
    )

    # Build and lock on sandbox-local disk; flock is unavailable on the volume.
    # Install with atomic renames and stream both stdout and stderr.
    command = (
        "( set -euo pipefail; "
        "export LIBGUESTFS_BACKEND=direct; "
        # The sandbox runs Modal's kernel (no modules in the container);
        # point libguestfs at the apt-installed kernel instead. The prepare
        # script honors LIBGUESTFS_KERNEL_VERSION for exactly this case.
        "export LIBGUESTFS_KERNEL_VERSION=$(ls /lib/modules | head -1); "
        "mkdir -p /build; "
        # The prepare script downloads and verifies the archive on first use.
        "bash /repo/scripts/prepare_osworld_qcow2.sh "
        "--output /build/osworld_ubuntu.qcow2 --download-dir /vol-src; "
        "cp /build/osworld_ubuntu.qcow2.provenance.json "
        "/cache/qemu/.provenance.tmp; "
        "cp /build/osworld_ubuntu.qcow2 /cache/qemu/.image.tmp; "
        f"mv /cache/qemu/.provenance.tmp {OUTPUT}.provenance.json; "
        f"mv /cache/qemu/.image.tmp {OUTPUT}; "
        "sync; echo PREPARED_AND_INSTALLED ) 2>&1"
    )
    sandbox = modal.Sandbox.create(
        "bash", "-lc", command,
        app=app,
        image=image,
        cpu=8,
        memory=16384,
        timeout=3 * 3600,
        volumes={"/vol-src": source, "/cache/qemu": cache},
        experimental_options={"vm_runtime": True},
    )
    print(f"builder sandbox: {sandbox.object_id}", flush=True)
    installed = False
    for line in sandbox.stdout:
        print(line, end="", flush=True)
        if "PREPARED_AND_INSTALLED" in line:
            installed = True
    code = sandbox.wait()
    # Require the completion marker before verifying the installed image.
    print(f"\nprepare script exit: {code!r} installed={installed}", flush=True)
    if not installed:
        return 1

    # Verify the installed artifacts are on the volume for env sandboxes.
    check = modal.Sandbox.create(
        "bash", "-lc",
        f"ls -la {OUTPUT} {OUTPUT}.provenance.json && "
        f"qemu-img info {OUTPUT} | head -5 && cat {OUTPUT}.provenance.json",
        app=app, image=image, cpu=2, memory=4096, timeout=600,
        volumes={"/cache/qemu": cache},
        experimental_options={"vm_runtime": True},
    )
    verified = False
    for line in check.stdout:
        print(line, end="", flush=True)
        if '"guest_auth_prepared": true' in line:
            verified = True
    verify = check.wait()
    # Require the expected provenance field in the installed image's sidecar.
    print(f"\nverify exit: {verify!r} OSWORLD_QCOW2_OK={verified}", flush=True)
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
