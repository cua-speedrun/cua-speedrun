"""Install a pinned Node runtime and Codex CLI into the submission snapshot."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


NODE_VERSION = "22.23.1"
PILLOW_VERSION = "11.3.0"
CLI_PACKAGE = "@openai/codex"
CLI_VERSION = "0.153.2"
CLI_COMMAND = "codex"

NODE_SHA256 = {
    ("darwin", "arm64"): "fb526811860f81dcac7dd8b2b55eca4accfc5d61c3b7c2508f2639faee8a738d",
    ("darwin", "x64"): "efeec6641a2f15f5396d27cd0b32f5062d6689d1e9e5d89607d0b29bda890233",
    ("linux", "arm64"): "0294e8b915ab75f92c7513d2fcb830ae06e10684e6c603e99a87dbf8835389c1",
    ("linux", "x64"): "9749e988f437343b7fa832c69ded82a312e41a03116d766797ac14f6f9eee578",
}

ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = ROOT / ".cli_runtime"
NODE_DIR = RUNTIME_ROOT / f"node-v{NODE_VERSION}"
CLI_PREFIX = RUNTIME_ROOT / f"codex-{CLI_VERSION}"


def install_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("CODEX_AUTH_JSON", None)
    return env


def platform_key() -> tuple[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = {
        "aarch64": "arm64",
        "arm64": "arm64",
        "amd64": "x64",
        "x86_64": "x64",
    }.get(machine)
    key = (system, arch or machine)
    if key not in NODE_SHA256:
        raise RuntimeError(f"unsupported Node platform: {system}/{machine}")
    return key


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_node() -> Path:
    node = NODE_DIR / "bin" / "node"
    if node.is_file():
        version = subprocess.check_output([str(node), "--version"], text=True).strip()
        if version == f"v{NODE_VERSION}":
            return node

    system, arch = platform_key()
    archive_name = f"node-v{NODE_VERSION}-{system}-{arch}.tar.xz"
    url = f"https://nodejs.org/dist/v{NODE_VERSION}/{archive_name}"
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(NODE_DIR, ignore_errors=True)
    with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / archive_name
        urllib.request.urlretrieve(url, archive)
        actual = sha256(archive)
        expected = NODE_SHA256[(system, arch)]
        if actual != expected:
            raise RuntimeError(
                f"Node archive checksum mismatch: expected {expected}, got {actual}"
            )
        with tarfile.open(archive, "r:xz") as bundle:
            bundle.extractall(tmp_path)
        extracted = tmp_path / archive_name.removesuffix(".tar.xz")
        os.replace(extracted, NODE_DIR)
    return node


def ensure_pillow() -> None:
    try:
        installed = importlib.metadata.version("Pillow")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed == PILLOW_VERSION:
        return
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            f"Pillow=={PILLOW_VERSION}",
        ],
        env=install_environment(),
    )


def ensure_cli(node: Path) -> Path:
    cli = CLI_PREFIX / "bin" / CLI_COMMAND
    env = install_environment()
    env["PATH"] = f"{node.parent}{os.pathsep}{env.get('PATH', '')}"
    if cli.is_file():
        version = subprocess.check_output([str(cli), "--version"], text=True, env=env)
        if CLI_VERSION in version:
            return cli

    shutil.rmtree(CLI_PREFIX, ignore_errors=True)
    npm = node.parent / "npm"
    subprocess.check_call(
        [
            str(npm),
            "install",
            "--global",
            "--no-audit",
            "--no-fund",
            "--prefix",
            str(CLI_PREFIX),
            f"{CLI_PACKAGE}@{CLI_VERSION}",
        ],
        env=env,
    )
    version = subprocess.check_output([str(cli), "--version"], text=True, env=env)
    if CLI_VERSION not in version:
        raise RuntimeError(f"unexpected Codex CLI version: {version.strip()}")
    return cli


def main() -> None:
    node = ensure_node()
    ensure_pillow()
    cli = ensure_cli(node)
    print(f"ready: node=v{NODE_VERSION} codex={CLI_VERSION} path={cli}")


if __name__ == "__main__":
    main()
