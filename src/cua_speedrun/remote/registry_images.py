"""Import digest-pinned desktop images into the operator's Modal workspace."""

from __future__ import annotations

import json
from pathlib import Path
import re


def image_definition(path: Path, name: str) -> dict:
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError(f"unsupported image manifest: {path}")
    image = document["images"][name]
    reference = image.get("reference", "")
    if not re.fullmatch(r"[a-zA-Z0-9./_-]+@sha256:[0-9a-f]{64}", reference):
        raise ValueError("desktop image must be pinned by SHA-256 digest")
    if not isinstance(image.get("provenance"), dict):
        raise ValueError("desktop image needs source provenance")
    return image


def import_image(definition: dict, contract: dict) -> str:
    import modal

    reference = definition["reference"]
    provenance = definition["provenance"]
    expected = contract["expected_provenance"]
    mismatches = [key for key, value in expected.items() if provenance.get(key) != value]
    if mismatches:
        raise ValueError("published image does not match the benchmark: " + ", ".join(mismatches))
    app = modal.App.lookup(contract["app_name"], create_if_missing=True)
    cache = modal.Dict.from_name(contract["cache_name"], create_if_missing=True)
    record = cache.get(contract["cache_key"])
    if isinstance(record, dict) and record.get("registry_image") == reference:
        if all(record.get(key) == value for key, value in expected.items()):
            try:
                image = modal.Image.from_id(record["modal_snapshot_image_id"])
                image.build(app)
                print("Desktop image is ready (cached)", flush=True)
                return image.object_id
            except modal.exception.NotFoundError:
                pass
    print(f"Importing prebuilt desktop: {reference.split('@')[0]}", flush=True)
    with modal.enable_output():
        image = modal.Image.from_registry(reference).build(app)
    cache[contract["cache_key"]] = {
        **provenance, "registry_image": reference,
        "modal_snapshot_image_id": image.object_id,
    }
    print("Desktop image is ready", flush=True)
    return image.object_id
