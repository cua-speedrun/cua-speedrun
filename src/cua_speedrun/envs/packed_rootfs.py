"""Mount an image-owned ext4 filesystem stored in parallel-downloadable parts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess


LAYOUT = Path("/opt/cua-speedrun/desktop/rootfs.json")


def _layout(path: Path) -> tuple[Path, list[tuple[Path, int]], int, list]:
    spec = json.loads(path.read_text())
    if spec.get("schema_version") != 1 or spec.get("filesystem") != "ext4":
        raise ValueError("unsupported packed desktop filesystem")
    root = Path(spec["mountpoint"])
    if not root.is_absolute() or root == Path("/") or ".." in root.parts or root.is_symlink():
        raise ValueError("invalid packed desktop mountpoint")
    parts, reads = [], []
    names = set()
    for part in spec["parts"]:
        name, size = part["file"], part["size"]
        if not name or Path(name).name != name or name in {".", ".."} or name in names:
            raise ValueError("invalid packed desktop part name")
        names.add(name)
        source = path.parent / name
        if (type(size) is not int or size <= 0 or size % 512
                or source.is_symlink() or not source.is_file()
                or source.stat().st_size != size):
            raise ValueError(f"invalid packed desktop part: {name}")
        parts.append((source, size))
        for offset, length in part.get("prefetch", []):
            if (type(offset) is not int or type(length) is not int
                    or offset < 0 or length <= 0 or offset + length > size):
                raise ValueError(f"invalid packed desktop read range: {name}")
            reads.append((source, offset, length))
    size = sum(size for _, size in parts)
    capacity = spec["capacity_bytes"]
    if (not parts or size != spec["size"] or type(capacity) is not int
            or not size <= capacity <= 1024**4 or capacity % 512):
        raise ValueError("invalid packed desktop capacity")
    return root, parts, capacity, reads


def _prefetch(item: tuple[Path, int, int]) -> None:
    path, offset, length = item
    with path.open("rb") as stream:
        stream.seek(offset)
        while length:
            data = stream.read(min(length, 4 * 1024 * 1024))
            if not data:
                raise RuntimeError(f"short packed desktop read: {path.name}")
            length -= len(data)


def _run(*args: str, **kwargs) -> str:
    process = subprocess.run(args, capture_output=True, text=True, timeout=120, **kwargs)
    if process.returncode:
        raise RuntimeError(f"{args[0]} failed: {process.stderr[-2000:]}")
    return process.stdout.strip()


def mount_packed_rootfs(path: Path = LAYOUT) -> None:
    """Older directory-based images need no mounting; packed images use Linux loops."""
    if not path.is_file():
        return
    root, parts, capacity, reads = _layout(path)
    if os.path.ismount(root):
        return
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_prefetch, reads))

    tail = path.parent / "writable-tail"
    size = sum(size for _, size in parts)
    tail_created = False
    loops: list[str] = []
    table = []
    offset = 0
    name = "cua-rootfs-" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    mapped = False
    try:
        if capacity > size:
            with tail.open("xb") as stream:
                tail_created = True
                stream.truncate(capacity - size)
            tail.chmod(0o600)
            parts.append((tail, capacity - size))
        for source, size in parts:
            device = _run("losetup", "--find", "--show", str(source))
            loops.append(device)
            table.append(f"{offset} {size // 512} linear {device} 0")
            offset += size // 512
        _run("dmsetup", "--noudevsync", "create", name, input="\n".join(table) + "\n")
        mapped = True
        _run("dmsetup", "mknodes", name)
        device = "/dev/mapper/" + name
        _run("resize2fs", device)
        root.mkdir(parents=True, exist_ok=True)
        _run("mount", "-t", "ext4", device, str(root))
    except BaseException:
        if mapped:
            subprocess.run(["dmsetup", "--noudevsync", "remove", name],
                           capture_output=True, timeout=30)
        for device in reversed(loops):
            subprocess.run(["losetup", "--detach", device], capture_output=True, timeout=30)
        if tail_created:
            tail.unlink(missing_ok=True)
        raise
