"""Pinned OSWorld2 native desktop recipe."""

import hashlib
from pathlib import Path

ARCHIVE_SHA256 = "eb737ae70b49849e24af407de6a518439a23de05a8497096a948334ce0a909aa"


def contract(root: Path) -> dict:
    digest = hashlib.sha256()
    for name in ("scripts/osworld2_native.py", "scripts/osworld2_native_delta.sh",
                 "scripts/build_osworld2_modal_image.py"):
        digest.update(name.encode() + b"\0" + (root / name).read_bytes())
    recipe = digest.hexdigest()
    return {
        "app_name": "cua-speedrun-native-desktops",
        "cache_name": "cua-speedrun-native-desktop-images",
        "cache_key": "osworld2:" + recipe,
        "desktop_user": "user", "boot_budget_sec": 1800, "services": [],
        "volumes": {"/cache/qemu": "gym-anything-qemu-cache"},
        "expected_provenance": {
            "source_archive_sha256": ARCHIVE_SHA256,
            "source_revision": "8213366932c553e5fe758d0f2c8c8b81ffc3be8c",
            "recipe_sha256": recipe, "validated": True,
        },
    }
