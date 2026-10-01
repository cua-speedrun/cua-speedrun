import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from cua_speedrun.remote.registry_images import image_definition
from cua_speedrun.remote import osworld_modal_base

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/desktop-images.json"


@pytest.mark.parametrize("name", ["osworld", "osworld2", "mypcbench", "cua-world-base"])
def test_desktop_images_are_digest_pinned(name):
    definition = image_definition(MANIFEST, name)
    assert definition["reference"].startswith("docker.io/cuaspeedrun/")
    assert "@sha256:" in definition["reference"]


def test_osworld_release_matches_source_contract():
    published = image_definition(MANIFEST, "osworld")["provenance"]
    for key, value in osworld_modal_base.expected_provenance_block().items():
        assert published[key] == value
    assert published["delta_fingerprint"] == osworld_modal_base.delta_fingerprint()


@pytest.mark.parametrize("name,script", [
    ("osworld2", "scripts/osworld2_native.py"),
    ("mypcbench", "scripts/mypcbench_shared/native_image.py"),
])
def test_desktop_release_matches_recipe(name, script):
    spec = importlib.util.spec_from_file_location(name, ROOT / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    published = image_definition(MANIFEST, name)["provenance"]
    assert published == module.contract(ROOT)["expected_provenance"]


@pytest.mark.parametrize("dataset", [
    "osworld-50", "osworld-offline", "osworld2-52", "osworld2-offline", "my-pc-bench",
])
def test_native_preparation_imports_images_without_extracting_disks(dataset):
    source = yaml.safe_load((ROOT / "benchmarks" / dataset / "benchmark-source.yaml").read_text())
    steps = source["prepare"]["modal-native"]
    assert any(step["script"] == "scripts/import_desktop_image.py" for step in steps)
    assert not any("build_" in step["script"] for step in steps)
    for step in steps:
        if step["script"] == "scripts/prepare_osworld2_modal.py":
            assert "--assets-only" in step["args"]


@pytest.mark.parametrize("dataset", ["cua-world-26", "cua-world-offline"])
def test_cua_world_base_is_published_and_fingerprinted(dataset):
    from gym_anything.runtime.runners.modal_native_image import MODAL_NATIVE_IMAGE_FINGERPRINT

    source = yaml.safe_load((ROOT / "benchmarks" / dataset / "benchmark-source.yaml").read_text())
    configured = source["native_image"]
    published = image_definition(ROOT / configured["manifest"], configured["name"])
    assert published["provenance"]["image_schema"] == MODAL_NATIVE_IMAGE_FINGERPRINT
    assert configured["manifest"] in source["materializer"]["inputs"]


def test_unpinned_image_is_rejected(tmp_path):
    manifest = tmp_path / "images.json"
    manifest.write_text(json.dumps({"schema_version": 1, "images": {
        "desktop": {"reference": "docker.io/example/desktop:latest", "provenance": {}}
    }}))
    with pytest.raises(ValueError, match="pinned"):
        image_definition(manifest, "desktop")
