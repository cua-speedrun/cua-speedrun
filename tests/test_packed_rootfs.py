import json

import pytest

from cua_speedrun.envs.packed_rootfs import _layout, _prefetch, mount_packed_rootfs


def layout_file(tmp_path):
    (tmp_path / "part-0000").write_bytes(bytes(range(256)) * 4)
    spec = {
        "schema_version": 1,
        "filesystem": "ext4",
        "mountpoint": str(tmp_path / "desktop"),
        "size": 1024,
        "capacity_bytes": 4096,
        "parts": [{"file": "part-0000", "size": 1024, "prefetch": [[0, 512]]}],
    }
    path = tmp_path / "rootfs.json"
    path.write_text(json.dumps(spec))
    return path, spec


def test_layout_and_real_file_read(tmp_path):
    path, _ = layout_file(tmp_path)
    root, parts, capacity, reads = _layout(path)
    assert root == tmp_path / "desktop"
    assert parts == [(tmp_path / "part-0000", 1024)]
    assert capacity == 4096
    assert reads == [(tmp_path / "part-0000", 0, 512)]
    _prefetch(reads[0])


@pytest.mark.parametrize("change", [
    {"mountpoint": "/"},
    {"mountpoint": "relative"},
    {"filesystem": "unknown"},
    {"schema_version": 2},
    {"size": 512},
    {"capacity_bytes": 512},
    {"parts": []},
    {"parts": [{"file": "../part-0000", "size": 1024}]},
    {"parts": [{"file": "part-0000", "size": 512}]},
    {"parts": [{"file": "part-0000", "size": 1024, "prefetch": [[512, 1024]]}]},
    {"parts": [{"file": "part-0000", "size": 1024, "prefetch": [[-1, 512]]}]},
])
def test_invalid_layout_fails_before_mounting(tmp_path, change):
    path, spec = layout_file(tmp_path)
    path.write_text(json.dumps({**spec, **change}))
    with pytest.raises(ValueError):
        mount_packed_rootfs(path)


def test_symlink_part_is_rejected(tmp_path):
    path, spec = layout_file(tmp_path)
    (tmp_path / "alias").symlink_to(tmp_path / "part-0000")
    spec["parts"][0]["file"] = "alias"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        _layout(path)


def test_read_detects_truncated_part(tmp_path):
    path, _ = layout_file(tmp_path)
    with pytest.raises(RuntimeError, match="short packed desktop read"):
        _prefetch((path.parent / "part-0000", 0, 2048))


def test_directory_image_requires_no_mount(tmp_path):
    mount_packed_rootfs(tmp_path / "absent-rootfs.json")
