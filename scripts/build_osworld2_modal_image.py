#!/usr/bin/env python3
"""Build the pinned OSWorld2 desktop for Modal without KVM."""

import argparse
import base64
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_mypcbench_modal_image import execute
from osworld2_native import contract
import cua_speedrun


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    import modal
    from cua_speedrun.remote.modal_native_env import resolve_base_image_id

    recipe = contract(ROOT)
    env_dir = args.benchmark.resolve() / "environment"
    if json.loads((env_dir / "native-image.json").read_text()) != recipe:
        raise ValueError("materialized benchmark has a different native image recipe")
    app = modal.App.lookup(recipe["app_name"], create_if_missing=True)
    cache = modal.Dict.from_name(recipe["cache_name"], create_if_missing=True)
    volume = modal.Volume.from_name("gym-anything-qemu-cache", create_if_missing=True)
    snapshot = None
    if cache.get(recipe["cache_key"]):
        snapshot = resolve_base_image_id(cache, cache_key=recipe["cache_key"],
                                         expected=recipe["expected_provenance"])
        try:
            modal.Image.from_id(snapshot).build(app)
        except modal.exception.NotFoundError:
            snapshot = None
    if snapshot and args.reuse:
        print("OSWorld2 native desktop image is ready", flush=True)
        return
    if not snapshot:
        image = (modal.Image.debian_slim(python_version="3.12")
            .apt_install("libguestfs-tools", "linux-image-amd64", "qemu-utils",
                         "util-linux", "kmod", "zstd", "ffmpeg", "imagemagick")
            .pip_install_from_pyproject(str(env_dir / "osworld2/pyproject.toml"))
            .pip_install("Pillow", "PyYAML", "requests")
            .add_local_file(str(ROOT / "scripts/osworld2_native_delta.sh"), "/prepare/delta.sh"))
        builder = modal.Sandbox.create(app=app, image=image, cpu=8, memory=16384,
            timeout=6 * 3600, volumes={"/cache/qemu": volume},
            experimental_options={"vm_runtime": True})
        print("OSWorld2 native builder:", builder.object_id, flush=True)
        try:
            execute(builder, "python3 -c " + shlex.quote(
                "import hashlib,json; from pathlib import Path; "
                "p=Path('/cache/qemu/osworld2_ubuntu.qcow2'); "
                "r=json.loads(Path(str(p)+'.provenance.json').read_text()); "
                f"assert r['archive_sha256']=={recipe['expected_provenance']['source_archive_sha256']!r}; "
                "assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==r['final_image_sha256']; "
                "print('OSWORLD2_SOURCE_VERIFIED')"), timeout=600)
            execute(builder, r"""
set -euo pipefail
export LIBGUESTFS_BACKEND=direct LIBGUESTFS_BACKEND_SETTINGS=force_tcg
export SUPERMIN_KERNEL=$(ls /boot/vmlinuz-* | head -1)
export SUPERMIN_MODULES=/lib/modules/${SUPERMIN_KERNEL##*/vmlinuz-}
qemu-img create -f qcow2 -F qcow2 -b /cache/qemu/osworld2_ubuntu.qcow2 /tmp/extraction.qcow2
filesystems=$(guestfish --rw --format=qcow2 -a /tmp/extraction.qcow2 run : list-filesystems)
while read -r device filesystem; do
    case "$filesystem" in ext2|ext3|ext4)
        guestfish --rw --format=qcow2 -a /tmp/extraction.qcow2 run : e2fsck "${device%:}" correct:true
    ;; esac
done <<< "$filesystems"
guestfish --ro --format=qcow2 -a /tmp/extraction.qcow2 -i tar-out / /tmp/rootfs.tar numericowner:true xattrs:true acls:true
mkdir /osworld
tar --numeric-owner --xattrs --xattrs-include='*' --acls -xf /tmp/rootfs.tar -C /osworld
rm /tmp/extraction.qcow2 /tmp/rootfs.tar
R=/osworld bash /prepare/delta.sh
""", timeout=5 * 3600)
            snapshot = builder.snapshot_filesystem(timeout=1200, ttl=None).object_id
        finally:
            builder.terminate()

    image = (modal.Image.from_id(snapshot)
        .add_local_dir(str(Path(cua_speedrun.__file__).resolve().parent), "/opt/cs/cua_speedrun"))
    validator = modal.Sandbox.create(app=app, image=image, cpu=4, memory=16384,
        timeout=900, env={"PYTHONPATH": "/opt/cs"}, experimental_options={"vm_runtime": True})
    try:
        output = execute(validator, "python3 -c " + shlex.quote(
            "from pathlib import Path; "
            "from cua_speedrun.envs.modal_native import boot_desktop,ModalNativeAdapter; "
            "boot_desktop('user',services=[]); a=ModalNativeAdapter(); o=a.observe(); "
            "assert o.meta['resolution']==[1920,1080],o.meta; "
            "Path('/tmp/validation.png').write_bytes(o.png); print('OSWORLD2_NATIVE_VALIDATED')"), timeout=600)
        transfer = validator.exec("base64", "-w0", "/tmp/validation.png")
        encoded = transfer.stdout.read()
        transfer.wait()
        if transfer.returncode:
            raise RuntimeError("could not retrieve desktop screenshot")
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "validation.png").write_bytes(base64.b64decode(encoded, validate=True))
        (args.output / "validation.log").write_text(output)
        record = {**recipe["expected_provenance"], "modal_snapshot_image_id": snapshot}
        cache[recipe["cache_key"]] = record
        (args.output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
        print("OSWorld2 native desktop validated", snapshot, flush=True)
    finally:
        validator.terminate()


if __name__ == "__main__":
    main()
