#!/usr/bin/env python3
"""Build and validate the pinned MyPCBench desktop snapshot on Modal."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import cua_speedrun
from mypcbench_shared.native_image import contract

CS_PACKAGE = Path(cua_speedrun.__file__).resolve().parent


def execute(sandbox, command: str, timeout: int = 1800) -> str:
    proc = sandbox.exec("bash", "-lc", command + " 2>&1", timeout=timeout)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"image preparation command failed ({proc.returncode})")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True, help="materialized MyPCBench benchmark")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/mypcbench-native-validation")
    parser.add_argument("--reuse", action="store_true", help="reuse an already validated image")
    args = parser.parse_args()
    benchmark = args.benchmark.resolve()
    recipe = contract(ROOT)
    env_dir = benchmark / "environment"
    if json.loads((env_dir / "native-image.json").read_text()) != recipe:
        raise ValueError("materialized benchmark has a different native-image recipe")
    import modal
    from cua_speedrun.remote.modal_native_env import resolve_base_image_id

    app = modal.App.lookup(recipe["app_name"], create_if_missing=True)
    cache = modal.Dict.from_name(recipe["cache_name"], create_if_missing=True)
    snapshot = None
    if cache.get(recipe["cache_key"]):
        snapshot = resolve_base_image_id(
            cache, cache_key=recipe["cache_key"], expected=recipe["expected_provenance"],
        )
        try:
            modal.Image.from_id(snapshot).build(app)
        except modal.exception.NotFoundError:
            print("Cached desktop image is unavailable; rebuilding", flush=True)
            snapshot = None
    if snapshot:
        if args.reuse:
            print("MyPCBench desktop image is ready", flush=True)
            return
        print("Revalidating cached snapshot:", snapshot)
    else:
        source = json.loads((ROOT / "benchmarks/mypcbench-image.json").read_text())["official_source"]
        image = (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install("libguestfs-tools", "linux-image-amd64", "qemu-utils", "curl", "util-linux", "kmod", "zstd")
            .pip_install("Pillow==11.3.0", "PyYAML==6.0.2", "requests==2.32.4", "google-genai==1.62.0", "huggingface-hub==0.36.0", "hf-xet==1.2.0")
            .add_local_file(str(ROOT / "scripts/mypcbench_shared/native_delta.sh"), "/prepare/delta.sh")
        )
        builder = modal.Sandbox.create(app=app, image=image, cpu=8, memory=16384,
            timeout=6 * 3600, experimental_options={"vm_runtime": True})
        print("MyPCBench builder:", builder.object_id, flush=True)
        try:
            download = (
                "from huggingface_hub import hf_hub_download; import shutil; "
                f"p=hf_hub_download(repo_id={source['repository'].split('/datasets/')[-1]!r}, "
                f"repo_type='dataset', filename='michael_scott.qcow2', revision={source['revision']!r}, "
                "local_dir='/tmp/hf-download'); shutil.move(p, '/tmp/source.qcow2')"
            )
            execute(builder, "HF_XET_HIGH_PERFORMANCE=1 HF_HOME=/tmp/hf-cache python3 -c " + shlex.quote(download), timeout=3600)
            # The disk is inspected read-only. libguestfs uses software emulation
            # for its extraction appliance; the measured desktop never uses QEMU.
            execute(builder,
                "set -euo pipefail; "
                f"echo {shlex.quote(source['image_sha256'] + '  /tmp/source.qcow2')} | sha256sum -c; "
                "export LIBGUESTFS_BACKEND=direct LIBGUESTFS_BACKEND_SETTINGS=force_tcg; "
                "export SUPERMIN_KERNEL=$(ls /boot/vmlinuz-* | head -1); "
                "export SUPERMIN_MODULES=/lib/modules/${SUPERMIN_KERNEL##*/vmlinuz-}; "
                "guestfish --ro --format=qcow2 -a /tmp/source.qcow2 -i "
                "tar-out / /tmp/rootfs.tar numericowner:true xattrs:true acls:true; "
                "mkdir /osworld; tar --numeric-owner --xattrs --xattrs-include='*' --acls -xf /tmp/rootfs.tar -C /osworld; "
                "rm /tmp/source.qcow2 /tmp/rootfs.tar; "
                "R=/osworld bash /prepare/delta.sh", timeout=5 * 3600)
            execute(builder, "python3 -c 'from google import genai; from PIL import Image; import yaml, requests'", timeout=60)
            snapshot = builder.snapshot_filesystem(timeout=1200, ttl=None).object_id
        finally:
            builder.terminate()

    # Validate a fresh instance, not the mutable build sandbox. Setup has no
    # agent and no judge key; it must bring up the actual seeded apps.
    validation_image = (
        modal.Image.from_id(snapshot)
        .add_local_dir(str(CS_PACKAGE), "/opt/cs/cua_speedrun")
        .add_local_dir(str(env_dir), "/envs/env")
    )
    validator = modal.Sandbox.create(app=app, image=validation_image, cpu=8, memory=16384,
        timeout=3600, env={"PYTHONPATH": "/opt/cs"}, experimental_options={"vm_runtime": True})
    try:
        task_ids = sorted(p.name for p in (env_dir / "tasks").iterdir() if (p / "task.json").is_file())
        task_id = task_ids[0]
        code = (
            "from pathlib import Path; "
            "from cua_speedrun.envs.modal_native import boot_desktop, run_pre_task_setup, ModalNativeAdapter, scrub_privileged; "
            "pid=boot_desktop('user',services=[]); "
            f"run_pre_task_setup(pid,'/envs/env',{task_id!r}); "
            "scrub_privileged(pid); "
            "adapter=ModalNativeAdapter(); observation=adapter.observe(); "
            "assert observation.meta['resolution']==[1280,800], observation.meta; "
            "Path('/tmp/validation.png').write_bytes(observation.png); "
            "print('MYPCBENCH_NATIVE_VALIDATED')"
        )
        output = execute(validator, "python3 -c " + shlex.quote(code), timeout=2400)
        if "MYPCBENCH_NATIVE_VALIDATED" not in output:
            raise RuntimeError("desktop validation did not finish")
        args.output.mkdir(parents=True, exist_ok=True)
        transfer = validator.exec("base64", "-w0", "/tmp/validation.png")
        encoded = transfer.stdout.read()
        transfer.wait()
        if transfer.returncode:
            raise RuntimeError("could not retrieve validation screenshot")
        screenshot = base64.b64decode(encoded, validate=True)
        (args.output / "validation.png").write_bytes(screenshot)
        (args.output / "validation.log").write_text(output)
        packages = execute(validator, "python3 -m pip freeze", timeout=60)
        record = {**recipe["expected_provenance"], "modal_snapshot_image_id": snapshot,
            "packages": packages.splitlines(), "validation_task": task_id,
            "validation_screenshot_sha256": hashlib.sha256(screenshot).hexdigest(),
            "validation": "fresh native GNOME desktop, canonical app warmup, 1280x800 screenshot"}
        (args.output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
        cache[recipe["cache_key"]] = record
        print(json.dumps(record, indent=2))
    finally:
        validator.terminate()


if __name__ == "__main__":
    main()
