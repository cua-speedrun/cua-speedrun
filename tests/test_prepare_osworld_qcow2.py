from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_osworld_qcow2.sh"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.skipif(
    any(shutil.which(command) is None for command in ("curl", "flock", "qemu-img", "unzip")),
    reason="OSWorld image preparation tools are not installed",
)
def test_prepare_osworld_qcow2_downloads_validates_and_reuses(tmp_path):
    source = tmp_path / "Ubuntu.qcow2"
    archive = tmp_path / "source" / "Ubuntu.qcow2.zip"
    output = tmp_path / "installed" / "osworld_ubuntu.qcow2"
    download_dir = tmp_path / "downloads"
    archive.parent.mkdir()

    subprocess.run(
        ["qemu-img", "create", "-f", "qcow2", str(source), "8M"],
        check=True,
        capture_output=True,
        text=True,
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(source, "Ubuntu.qcow2")

    env = dict(
        os.environ,
        OSWORLD_QCOW2_URL=archive.as_uri(),
        OSWORLD_QCOW2_ARCHIVE_SHA256=sha256(archive),
        OSWORLD_QCOW2_SOURCE_SHA256=sha256(source),
        OSWORLD_QCOW2_VIRTUAL_SIZE=str(8 * 1024 * 1024),
        OSWORLD_QCOW2_SKIP_SSH_PREP="1",
    )
    command = [
        "bash",
        str(SCRIPT),
        "--output",
        str(output),
        "--download-dir",
        str(download_dir),
    ]

    first = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    subprocess.run(
        ["qemu-img", "compare", "-f", "qcow2", "-F", "qcow2", str(source), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    provenance = output.with_suffix(output.suffix + ".provenance.json")
    assert provenance.is_file()
    assert "Installed verified OSWorld qcow2" in first.stdout

    second = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    assert "OSWorld qcow2 and provenance are already valid" in second.stdout


def test_prepare_script_installs_runner_tools_and_passwordless_sudo():
    # The full virt-customize path is exercised on a real host, not in unit
    # tests. Here we pin the contract that it prepares what the gym-anything
    # runner needs on top of the upstream OSWorld image.
    text = SCRIPT.read_text()
    # xdotool (+ its libxdo3 dep) is the one input tool the image lacks; pinned
    # by sha256 like the SSH debs so the build stays reproducible and offline.
    assert "download_tool_debs" in text
    assert "xdotool_3.20160805.1-4_amd64.deb" in text
    assert "libxdo3_3.20160805.1-4_amd64.deb" in text
    assert "69432493950855718836e1aadc4c490c103f35b0c5dacc0e28a0f55fc21d2613" in text
    assert "93198ce669e04f9b9d79f5e5e2d0fcd89fe411e7a7ebc948e9f4cbb563f62967" in text
    # passwordless sudo (the runner wraps hooks in `sudo -E`) and no reverse-DNS
    # stall on ssh connect.
    assert "/etc/sudoers.d/99-osworld-nopasswd" in text
    assert "NOPASSWD:ALL" in text
    assert "UseDNS no" in text


def test_apptainer_wrapper_exists_and_is_executable():
    wrapper = ROOT / "scripts" / "prepare_osworld_qcow2_apptainer.sh"
    assert wrapper.is_file()
    assert os.access(wrapper, os.X_OK)
    text = wrapper.read_text()
    # The wrapper's whole reason to exist: run the prep inside an Ubuntu
    # container so libguestfs boots on hosts without a native one.
    assert "prepare_osworld_qcow2.sh" in text
    assert "LIBGUESTFS_KERNEL_VERSION" in text


def test_native_export_checks_filesystems_on_an_overlay():
    from cua_speedrun.remote import osworld_modal_base as base

    script = ROOT / "scripts/build_osworld_modal_base.py"
    tree = ast.parse(script.read_text())
    call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and len(node.args) > 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "prepare verified desktop filesystem"
    )
    command = eval(compile(ast.Expression(call.args[2]), str(script), "eval"), {
        "base": base, "source": base.OSWORLD_IMAGE_CONTRACT,
        "_ROOTFS_TAR": "/vol/osworld_rootfs.tar.zst",
    })
    subprocess.run(["bash", "-n"], input=command, text=True, check=True)
    assert "-b /tmp/source.qcow2 /tmp/extraction.qcow2" in command
    assert 'e2fsck "${device%:}" correct:true' in command
    assert "forceall:true" not in command
    assert "--ro --format=qcow2 -a /tmp/extraction.qcow2 -i tar-out" in command
    assert "guestfish --rw --format=qcow2 -a /tmp/source.qcow2" not in command
    assert "--ignore-failed-read" not in command
    assert "cat /tmp/osworld-guestfish.log >&2" in command
    assert command.index("sha256sum -c", command.index("source.qcow2")) < command.index("qemu-img create")
