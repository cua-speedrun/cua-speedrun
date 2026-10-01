"""Pinned MyPCBench native-image recipe shared by builder and materializer."""

import hashlib
import json
from pathlib import Path


def contract(root: Path) -> dict:
    source = json.loads((root / "benchmarks/mypcbench-image.json").read_text())["official_source"]
    inputs = [
        "benchmarks/mypcbench-image.json", "scripts/mypcbench_shared/native_delta.sh",
        "scripts/build_mypcbench_modal_image.py", "scripts/mypcbench_shared/native_image.py",
    ]
    digest = hashlib.sha256()
    for name in inputs:
        digest.update(name.encode() + b"\0" + (root / name).read_bytes())
    recipe = digest.hexdigest()
    return {
        "app_name": "cua-speedrun-native-desktops",
        "cache_name": "cua-speedrun-native-desktop-images",
        "cache_key": "mypcbench:" + recipe,
        "desktop_user": "user", "boot_budget_sec": 1800,
        # MyPCBench setup gates on all 17 seeded applications. There is no
        # OSWorld controller service in this persona image.
        "services": [],
        "expected_provenance": {
            "source_image_sha256": source["image_sha256"],
            "source_revision": source["revision"],
            "recipe_sha256": recipe, "validated": True,
        },
    }
