#!/usr/bin/env python3
"""Seed the official OSWorld2 image and gated assets onto a Modal volume."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


CACHE_ROOT = Path("/cache/qemu")
IMAGE_REPO = "xlangai/v2-image"
IMAGE_REVISION = "v2026.06.24"
IMAGE_COMMIT = "8213366932c553e5fe758d0f2c8c8b81ffc3be8c"
IMAGE_FILE = "osworld-v2-ubuntu-x86.qcow2.zip"
IMAGE_ARCHIVE_SHA256 = "eb737ae70b49849e24af407de6a518439a23de05a8497096a948334ce0a909aa"
IMAGE_TARGET = CACHE_ROOT / "osworld2_ubuntu.qcow2"
ASSET_REPO = "xlangai/osworld_v2_assets_gated"
ASSET_REVISION = "v2026.06.24"
ASSET_TARGET = CACHE_ROOT / "osworld2-assets"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _image_ready() -> bool:
    provenance = _read_json(Path(f"{IMAGE_TARGET}.provenance.json"))
    return (
        IMAGE_TARGET.is_file()
        and all(
            provenance.get(key) == value
            for key, value in {
                "schema_version": 1,
                "recipe": "cua-speedrun-osworld2-qcow2@1",
                "release": "osworld-v2-2026.06.24",
                "artifact_tag": IMAGE_REVISION,
                "artifact_path": IMAGE_FILE,
                "archive_sha256": IMAGE_ARCHIVE_SHA256,
            }.items()
        )
        and int(provenance.get("virtual_size_bytes", 0)) > 0
        and _sha256(IMAGE_TARGET) == provenance.get("final_image_sha256")
    )


def _prepare_image() -> None:
    if _image_ready():
        print("OSWORLD2_IMAGE_READY", flush=True)
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the gated OSWorld2 image")

    archive = Path(
        hf_hub_download(
            repo_id=IMAGE_REPO,
            filename=IMAGE_FILE,
            repo_type="dataset",
            revision=IMAGE_COMMIT,
            cache_dir="/tmp/huggingface-image",
            token=token,
        )
    )
    actual_archive_sha = _sha256(archive)
    if actual_archive_sha != IMAGE_ARCHIVE_SHA256:
        raise RuntimeError(
            f"OSWorld2 image archive checksum mismatch: {actual_archive_sha}"
        )

    extraction = Path("/tmp/osworld2-image")
    shutil.rmtree(extraction, ignore_errors=True)
    extraction.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        members = [item for item in bundle.infolist() if not item.is_dir()]
        qcow_members = [item for item in members if item.filename.endswith(".qcow2")]
        if len(qcow_members) != 1:
            raise RuntimeError("OSWorld2 image archive must contain exactly one qcow2")
        member = qcow_members[0]
        destination = (extraction / member.filename).resolve()
        if not destination.is_relative_to(extraction.resolve()):
            raise RuntimeError("OSWorld2 image archive member escapes extraction root")
        bundle.extract(member, extraction)

    subprocess.run(["qemu-img", "check", "-q", str(destination)], check=True)
    image_info = json.loads(
        subprocess.run(
            ["qemu-img", "info", "--output=json", str(destination)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    virtual_size = int(image_info["virtual-size"])
    final_sha = _sha256(destination)
    image_temporary = CACHE_ROOT / ".osworld2_ubuntu.qcow2.tmp"
    provenance_temporary = CACHE_ROOT / ".osworld2_ubuntu.provenance.tmp"
    image_temporary.unlink(missing_ok=True)
    provenance_temporary.unlink(missing_ok=True)
    subprocess.run(
        ["cp", "--sparse=always", str(destination), str(image_temporary)],
        check=True,
    )
    provenance_temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recipe": "cua-speedrun-osworld2-qcow2@1",
                "release": "osworld-v2-2026.06.24",
                "artifact_repository": IMAGE_REPO,
                "artifact_tag": IMAGE_REVISION,
                "artifact_commit": IMAGE_COMMIT,
                "artifact_path": IMAGE_FILE,
                "archive_sha256": IMAGE_ARCHIVE_SHA256,
                "final_image_sha256": final_sha,
                "virtual_size_bytes": virtual_size,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(provenance_temporary, Path(f"{IMAGE_TARGET}.provenance.json"))
    os.replace(image_temporary, IMAGE_TARGET)
    print("OSWORLD2_IMAGE_INSTALLED", flush=True)


def _assets_ready() -> bool:
    provenance = _read_json(ASSET_TARGET / ".provenance.json")
    return ASSET_TARGET.is_dir() and all(
        provenance.get(key) == value
        for key, value in {
            "release": "osworld-v2-2026.06.24",
            "repository": ASSET_REPO,
            "repo_type": "dataset",
            "revision": ASSET_REVISION,
        }.items()
    )


def _prepare_assets() -> None:
    if _assets_ready():
        print("OSWORLD2_ASSETS_READY", flush=True)
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for gated OSWorld2 assets")

    files = HfApi(token=token).list_repo_files(
        repo_id=ASSET_REPO,
        repo_type="dataset",
        revision=ASSET_REVISION,
    )
    if not files:
        raise RuntimeError("official OSWorld2 asset snapshot is empty")

    staging = CACHE_ROOT / ".osworld2-assets-staging"
    backup = CACHE_ROOT / ".osworld2-assets-backup"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    snapshot_download(
        repo_id=ASSET_REPO,
        repo_type="dataset",
        revision=ASSET_REVISION,
        local_dir=staging,
        token=token,
        max_workers=8,
    )
    missing = [name for name in files if not (staging / name).is_file()]
    if missing:
        raise RuntimeError(
            f"downloaded OSWorld2 assets are missing {len(missing)} files"
        )
    (staging / ".provenance.json").write_text(
        json.dumps(
            {
                "release": "osworld-v2-2026.06.24",
                "repository": ASSET_REPO,
                "repo_type": "dataset",
                "revision": ASSET_REVISION,
                "file_count": len(files),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if ASSET_TARGET.exists():
        os.replace(ASSET_TARGET, backup)
    os.replace(staging, ASSET_TARGET)
    shutil.rmtree(backup, ignore_errors=True)
    print(f"OSWORLD2_ASSETS_INSTALLED files={len(files)}", flush=True)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-only", action="store_true")
    args = parser.parse_args()
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.assets_only:
        _prepare_image()
    _prepare_assets()
    print("OSWORLD2_MODAL_ASSETS_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
