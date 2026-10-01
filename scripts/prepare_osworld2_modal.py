#!/usr/bin/env python3
"""Run the pinned OSWorld2 image/asset seeder in a Modal VM sandbox."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VOLUME_NAME = "gym-anything-qemu-cache"


def _load_local_env() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.startswith(("MODAL_", "HF_")):
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-only", action="store_true")
    args = parser.parse_args()
    _load_local_env()
    import modal
    from huggingface_hub import get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError(
            "Log in with `uvx --from huggingface_hub hf auth login` after "
            "accepting the gated OSWorld2 task and asset datasets."
        )

    app = modal.App.lookup("cua-speedrun-osworld2-assets", create_if_missing=True)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("coreutils", "qemu-utils")
        .pip_install("huggingface-hub>=0.35.0", "hf_xet")
        .add_local_file(
            str(ROOT / "scripts/osworld2_seed_modal.py"),
            remote_path="/seed/osworld2_seed_modal.py",
        )
    )
    sandbox = modal.Sandbox.create(
        "bash",
        "-lc",
        "python3 -u /seed/osworld2_seed_modal.py"
        + (" --assets-only" if args.assets_only else "") + " 2>&1",
        app=app,
        image=image,
        cpu=8,
        memory=16384,
        timeout=6 * 3600,
        volumes={"/cache/qemu": volume},
        secrets=[modal.Secret.from_dict({"HF_TOKEN": token})],
        experimental_options={"vm_runtime": True},
    )
    print(f"OSWorld2 asset sandbox: {sandbox.object_id}", flush=True)
    complete = False
    for line in sandbox.stdout:
        print(line, end="", flush=True)
        if "OSWORLD2_MODAL_ASSETS_OK" in line:
            complete = True
    sandbox.wait()
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
