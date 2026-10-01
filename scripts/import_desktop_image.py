#!/usr/bin/env python3
"""Import a published desktop image into Modal."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from cua_speedrun.remote.registry_images import image_definition, import_image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--benchmark", type=Path)
    args = parser.parse_args()
    definition = image_definition(ROOT / "benchmarks/desktop-images.json", args.image)
    if args.benchmark:
        contract = json.loads((args.benchmark / "environment/native-image.json").read_text())
    else:
        from cua_speedrun.remote import osworld_modal_base as base

        contract = {
            "app_name": base.MODAL_APP_NAME, "cache_name": base.BASE_IMAGE_DICT_NAME,
            "cache_key": base.cache_key(), "expected_provenance": {
                **base.expected_provenance_block(), "delta_fingerprint": base.delta_fingerprint(),
            },
        }
    import_image(definition, contract)


if __name__ == "__main__":
    main()
