"""Contract for the OSWorld modal-native base image.

This is the modal-native analog of ``scripts/prepare_osworld_qcow2.sh`` plus
``benchmarks/osworld-image.json``. Modal removed ``/dev/kvm`` from its VM
sandboxes, so the remote env plane can no longer boot nested QEMU. Instead we
extract the pinned OSWorld rootfs and boot its own systemd directly on Modal's
kernel; the built artifact is a Modal filesystem-snapshot Image rather than a
qcow2.

The module owns three things and nothing else:

  1. The pinned upstream source identity (same revision and checksums as the
     qcow2 contract, so both images are provenance-comparable).
  2. The recipe applied on top of that source: the deterministic rootfs delta
     (kept as one blob so its bytes fingerprint the recipe) plus the env-plane
     runtime baked beside the rootfs (Modal cannot layer build steps onto a
     filesystem-snapshot image, so everything the env-plane imports must be in
     the snapshot).
  3. The provenance record and its preflight validator, mirroring the ten
     equality-checked keys of the qcow2 ``.provenance.json`` sidecar so a
     modal-native run binds its base image the same way a QEMU run does.

Building and booting the image is the operator builder's job
(``scripts/build_osworld_modal_base.py``); this module stays free of any
Modal dependency so it imports cleanly on the executor and in tests.
"""

from __future__ import annotations

import hashlib
from typing import Any

# Pinned upstream source, identical to benchmarks/osworld-image.json. The
# modal-native image is booted from this exact rootfs, so keeping these values
# in lockstep is what makes the two images provenance-comparable.
OSWORLD_IMAGE_CONTRACT: dict[str, Any] = {
    "schema_version": 1,
    "provenance_schema_version": 2,
    "source_revision": "a5d9c3eaae98eebf6e3a0beb84e7e47cf72ae133",
    "source_url": (
        "https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/"
        "a5d9c3eaae98eebf6e3a0beb84e7e47cf72ae133/Ubuntu.qcow2.zip"
    ),
    "archive_sha256": "b795b6cd4c69b252c1b4f10150a347795555032501b60fd031751ed09b896712",
    "source_image_sha256": "6bf667a852b3c307f61d9f09c42559351f45e0607e428b4997becf534cf4d313",
    "virtual_size": 53687091200,
    "ssh_user": "user",
    "ssh_password": "password",
}

# sha256 of the verified rootfs tar extracted from the pinned qcow2. The builder
# refuses to start from an intermediate that does not match this.
ROOTFS_TAR_SHA256 = "56645cd4a61c8fe09fe156946eb1045e0dd4091f28a81ae0fe3fb3ddad0b980f"

# Pinned Ubuntu snapshot mirror so the delta packages resolve deterministically,
# exactly as scripts/prepare_osworld_qcow2.sh pins snapshot.ubuntu.com.
UBUNTU_SNAPSHOT = "20260501T000000Z"

RECIPE = "cua-speedrun-osworld-modal-snapshot@1"
DESKTOP_USER = OSWORLD_IMAGE_CONTRACT["ssh_user"]
DESKTOP_UID = 1000
# Must match the xorg.conf modeline the delta writes; the session is the
# image's own GDM autologin session on the dummy Xorg driver at this size.
DESKTOP_GEOMETRY = "1920x1080"
X11_DISPLAY = ":0"

# Shared Modal resource names for the builder and the launcher. The built image
# id lives in the Dict keyed by cache_key(); the volume holds the verified
# rootfs tar the builder extracts from.
MODAL_APP_NAME = "cua-osworld-port"
MODAL_VOLUME_NAME = "cua-osworld-port"
BASE_IMAGE_DICT_NAME = "cua-osworld-modal-base-images"

# The task setup hook (osworld_setup.py) downloads seed assets with the
# Hugging Face client from INSIDE the guest, so the client must live in the
# rootfs. Same pins as scripts/prepare_osworld_qcow2.sh.
HUGGINGFACE_HUB_VERSION = "0.36.2"
HF_XET_VERSION = "1.2.0"

# Delta packages installed from the pinned snapshot mirror. Versions are
# resolved by apt against the frozen snapshot and recorded post-install.
# xserver-xorg-video-dummy lets the image's own GDM autologin session run
# without a virtual GPU (the sandbox has no /dev/dri and no VGA device, so
# none of the image's qxl/modesetting/vesa drivers can light a display).
# xdotool handles pointer input, python3-xlib handles keyboard input without
# xdotool's Unicode keymap race, and scrot captures observations;
# openssh-server mirrors the qcow2 guest contract.
DELTA_PACKAGES = [
    "xserver-xorg-video-dummy",
    "xdotool",
    "python3-xlib",
    "scrot",
    "openssh-server",
]

# --- env-plane runtime, baked into the snapshot -----------------------------
# The env-plane process (Gateway, adapter, OSWorld verifiers) runs on the
# sandbox's own Python, outside the booted rootfs. Its dependencies are
# installed at base-build time because Modal refuses every build layer
# (apt_install / pip_install / run_commands) on top of a filesystem-snapshot
# image; only mount layers such as add_local_dir work (verified against
# modal 1.5.2). torch/torchvision are installed from the CPU wheel index
# first so easyocr does not drag CUDA builds into the snapshot.
ENV_PLANE_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
ENV_PLANE_TORCH_PIP = ("torch", "torchvision")
# System dependencies used by the verifier stack: glib for opencv-headless,
# gomp for torch, and ``file`` for OSWorld's compare_image_text metric.
# They live in the sandbox's outer filesystem, where evaluators execute.
ENV_PLANE_APT = ("libglib2.0-0", "libgomp1", "file")
# Pillow (Gateway PNG validation), PyYAML and requests (direct imports of the
# evaluator modules), then pyproject's ``osworld`` extra verbatim, the same
# stack a local worker runs verifiers with; a unit test keeps them in lockstep.
ENV_PLANE_PIP = (
    "Pillow",
    "PyYAML",
    "requests",
    "beautifulsoup4",
    "borb<3",
    "chardet",
    "cssselect",
    "easyocr",
    "fastdtw",
    "formulas",
    "gdown",
    "gymnasium~=0.28.1",
    "ImageHash",
    "librosa",
    "lxml",
    "mutagen",
    "numpy",
    "odfpy",
    "opencv-python-headless",
    "openpyxl",
    "pandas",
    "paramiko",
    "pdfplumber",
    "playwright",
    "pyacoustid",
    "pydrive",
    "pymupdf",
    "pypdf",
    "pypdf2",
    "pygame",
    "python-docx",
    "python-dotenv",
    "python-pptx",
    "pytz",
    "rapidfuzz",
    "requests-toolbelt~=1.0.0",
    "scikit-image",
    "scipy",
    "tldextract",
    "xmltodict",
)
# The OSWorld evaluator source is prefetched into the verifier's cache
# directory so an env boot never depends on a GitHub download. The commit
# matches the pin in scripts/osworld_shared/osworld_verifier.py; a benchmark
# pinning a different commit still auto-fetches at boot.
OSWORLD_EVALUATOR_COMMIT = "315a7603173feadf1b8a85cbc006c93ffe1dc1a1"
OSWORLD_EVALUATOR_CACHE = "/root/.cache/cua-speedrun/osworld"

# The deterministic port delta, applied to a rootfs mounted at $R. Reproducible:
# no timestamps, no network beyond the pinned apt snapshot. Kept as one blob so
# its bytes fingerprint the recipe. Placeholders are filled by rendered_delta().
DELTA_SCRIPT = r"""
set -euo pipefail
R="${R:?rootfs root required}"
test -f "$R/etc/os-release"
grep -q 'VERSION_CODENAME=jammy' "$R/etc/os-release"

# DNS for chroot apt and for the booted system (resolved is masked below).
rm -f "$R/etc/resolv.conf"
printf 'nameserver 8.8.8.8\nnameserver 1.1.1.1\n' > "$R/etc/resolv.conf"

# Neutralize the hardware fstab (UUID root, EFI, swapfile, floppy).
printf '# neutralized for the cua-speedrun modal-native port; root is the sandbox fs\n' \
    > "$R/etc/fstab"

# Install the delta packages from the pinned Ubuntu snapshot mirror.
mount --rbind /dev "$R/dev"
mount -t proc proc "$R/proc"
mount --rbind /sys "$R/sys"
mount -t tmpfs tmpfs "$R/run"
SNAP="__UBUNTU_SNAPSHOT__"
cat > "$R/etc/apt/sources.list.d/osworld-snapshot.list" <<EOF
deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/${SNAP} jammy main universe
deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/${SNAP} jammy-updates main universe
deb [check-valid-until=no] https://snapshot.ubuntu.com/ubuntu/${SNAP} jammy-security main universe
EOF
DEBIAN_FRONTEND=noninteractive chroot "$R" apt-get \
    -o Dir::Etc::sourcelist="sources.list.d/osworld-snapshot.list" \
    -o Dir::Etc::sourceparts="-" -o APT::Get::List-Cleanup="0" update -q
DEBIAN_FRONTEND=noninteractive chroot "$R" apt-get install -y -q \
    __DELTA_PACKAGES__
# The in-guest Hugging Face client for the task setup hook's asset downloads,
# pinned exactly as the qcow2 prep pins it.
chroot "$R" python3 -m pip install --no-cache-dir \
    huggingface-hub==__HUGGINGFACE_HUB_VERSION__ hf-xet==__HF_XET_VERSION__
chroot "$R" dpkg-query -W -f='${Package}\t${Version}\n' __DELTA_PACKAGES__ \
    > "$R/var/lib/osworld-delta-packages.tsv"
printf 'huggingface-hub\t__HUGGINGFACE_HUB_VERSION__\nhf-xet\t__HF_XET_VERSION__\n' \
    >> "$R/var/lib/osworld-delta-packages.tsv"
rm -f "$R/etc/apt/sources.list.d/osworld-snapshot.list"
umount -R "$R/run" "$R/proc" 2>/dev/null || true
umount -R "$R/sys" "$R/dev" 2>/dev/null || true

# Mask units that need real hardware or fight the shared sandbox network.
# GDM is NOT masked: the image's own boot path is GDM autologin (the pristine
# custom.conf carries AutomaticLoginEnable=True, AutomaticLogin=user,
# WaylandEnable=false), and that session is what upstream OSWorld runs.
mask() { for u in "$@"; do ln -sfn /dev/null "$R/etc/systemd/system/$u"; done; }
mask acpid.path acpid.service acpid.socket
mask open-vm-tools.service 'run-vmblock\x2dfuse.mount'
mask spice-vdagent.service spice-vdagentd.service spice-vdagentd.socket
mask NetworkManager.service NetworkManager-dispatcher.service NetworkManager-wait-online.service
mask systemd-resolved.service systemd-timesyncd.service systemd-oomd.service
mask ufw.service unattended-upgrades.service ua-reboot-cmds.service ubuntu-advantage.service
mask kerneloops.service ModemManager.service wpa_supplicant.service thermald.service
mask e2scrub_reap.service grub-common.service grub-initrd-fallback.service secureboot-db.service
mask systemd-udevd.service systemd-udevd-control.socket systemd-udevd-kernel.socket
mask power-profiles-daemon.service switcheroo-control.service
mask snapd.service snapd.socket snapd.seeded.service snapd.apparmor.service \
     snapd.autoimport.service snapd.core-fixup.service snapd.recovery-chooser-trigger.service \
     snapd.aa-prompt-listener.service snapd-desktop-integration.service
for d in multi-user.target.wants snapd.mounts.target.wants local-fs.target.wants; do
    for w in "$R/etc/systemd/system/$d"/snap-*.mount "$R/etc/systemd/system/$d"/var-snap-*.mount; do
        [ -e "$w" ] || continue
        ln -sfn /dev/null "$R/etc/systemd/system/$(basename "$w")"
    done
done

# Passwordless sudo for the desktop user (mirrors the qcow2 prep).
echo '__DESKTOP_USER__ ALL=(ALL) NOPASSWD:ALL' > "$R/etc/sudoers.d/99-osworld-nopasswd"
chmod 0440 "$R/etc/sudoers.d/99-osworld-nopasswd"

# SSH password auth + UseDNS no (mirrors the qcow2 guest contract).
install -d -m 0755 "$R/etc/ssh/sshd_config.d"
cat > "$R/etc/ssh/sshd_config.d/99-osworld-password.conf" <<EOF
PasswordAuthentication yes
PubkeyAuthentication yes
UseDNS no
EOF
chroot "$R" systemctl enable ssh.service >/dev/null 2>&1 || true

# The dummy Xorg driver needs an explicit device/monitor config; the image's
# own GDM autologin then starts the stock Ubuntu session on it, exactly the
# session the official OSWorld image runs under QEMU. The modeline pins the
# desktop to __DESKTOP_GEOMETRY__ (the OSWorld highres contract).
cat > "$R/etc/X11/xorg.conf" <<'EOF'
Section "Device"
    Identifier "OSWorldDummyDevice"
    Driver "dummy"
    VideoRam 256000
EndSection
Section "Monitor"
    Identifier "OSWorldDummyMonitor"
    HorizSync 30.0-70.0
    VertRefresh 50.0-75.0
    Modeline "1920x1080" 148.50 1920 2008 2052 2200 1080 1084 1089 1125 +hsync +vsync
EndSection
Section "Screen"
    Identifier "OSWorldDummyScreen"
    Device "OSWorldDummyDevice"
    Monitor "OSWorldDummyMonitor"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Modes "1920x1080"
        Virtual 1920 1080
    EndSubSection
EndSection
EOF

echo DELTA_APPLIED_OK
"""

# The subset an env.json declares and preflight equality-checks, exactly
# mirroring qemu_base_image_provenance in the qcow2 path.
PROVENANCE_EXPECTED_KEYS = (
    "schema_version",
    "recipe",
    "source_revision",
    "archive_sha256",
    "source_image_sha256",
    "ssh_prepared",
    "tools_prepared",
    "nopasswd_sudo",
    "ssh_user",
    "guest_auth_prepared",
)


def rendered_delta() -> str:
    """The delta script with all placeholders resolved."""
    return (
        DELTA_SCRIPT
        .replace("__UBUNTU_SNAPSHOT__", UBUNTU_SNAPSHOT)
        .replace("__DELTA_PACKAGES__", " ".join(DELTA_PACKAGES))
        .replace("__DESKTOP_USER__", DESKTOP_USER)
        .replace("__DESKTOP_GEOMETRY__", DESKTOP_GEOMETRY)
        .replace("__HUGGINGFACE_HUB_VERSION__", HUGGINGFACE_HUB_VERSION)
        .replace("__HF_XET_VERSION__", HF_XET_VERSION)
    )


def delta_fingerprint() -> str:
    """Deterministic fingerprint of the recipe: delta bytes, package list,
    snapshot pin, source identity, and desktop identity. Two builds with the
    same fingerprint are the same image contract."""
    digest = hashlib.sha256()
    digest.update(RECIPE.encode())
    digest.update(b"\0")
    digest.update(rendered_delta().encode())
    digest.update(b"\0")
    for package in DELTA_PACKAGES:
        digest.update(package.encode())
        digest.update(b"\0")
    digest.update(UBUNTU_SNAPSHOT.encode())
    digest.update(ROOTFS_TAR_SHA256.encode())
    digest.update(f"{DESKTOP_USER}:{DESKTOP_UID}:{DESKTOP_GEOMETRY}:{X11_DISPLAY}".encode())
    # The env-plane runtime is part of the image contract: it is baked into
    # the snapshot because Modal cannot layer build steps on one afterward.
    digest.update(ENV_PLANE_TORCH_INDEX.encode())
    for package in ENV_PLANE_TORCH_PIP + ENV_PLANE_PIP + ENV_PLANE_APT:
        digest.update(package.encode())
        digest.update(b"\0")
    digest.update(OSWORLD_EVALUATOR_COMMIT.encode())
    return digest.hexdigest()


def cache_key() -> str:
    """Key under which the built image id is recorded, so a repeat build with
    identical inputs reuses the snapshot."""
    return (
        f"base:{RECIPE}:{OSWORLD_IMAGE_CONTRACT['source_image_sha256']}"
        f":{delta_fingerprint()}"
    )


def build_provenance(
    image_id: str,
    tools_installed: dict[str, str],
    env_plane_packages: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The .provenance.json analog. The ten PROVENANCE_EXPECTED_KEYS match the
    qcow2 sidecar contract exactly; final_image_sha256 is replaced by the Modal
    snapshot identity (modal_snapshot_image_id) plus the reproducible
    delta_fingerprint, since a filesystem snapshot has no stable whole-file hash
    the way a qcow2 does. ``env_plane_packages`` records the pip versions the
    build resolved for the env-plane runtime, the way ``tools_installed``
    records the resolved delta packages."""
    return {
        "env_plane_packages": dict(env_plane_packages or {}),
        "schema_version": OSWORLD_IMAGE_CONTRACT["provenance_schema_version"],
        "recipe": RECIPE,
        "source_url": OSWORLD_IMAGE_CONTRACT["source_url"],
        "source_revision": OSWORLD_IMAGE_CONTRACT["source_revision"],
        "archive_sha256": OSWORLD_IMAGE_CONTRACT["archive_sha256"],
        "source_image_sha256": OSWORLD_IMAGE_CONTRACT["source_image_sha256"],
        "rootfs_tar_sha256": ROOTFS_TAR_SHA256,
        "modal_snapshot_image_id": image_id,
        "delta_fingerprint": delta_fingerprint(),
        "ubuntu_snapshot": UBUNTU_SNAPSHOT,
        "ssh_prepared": True,
        "tools_prepared": True,
        "tools_installed": dict(tools_installed),
        "nopasswd_sudo": True,
        "ssh_user": DESKTOP_USER,
        "guest_auth_prepared": True,
        "reference": "https://github.com/xlang-ai/osworld_image",
    }


def expected_provenance_block() -> dict[str, Any]:
    """The equality-checked subset a benchmark would declare to bind this image,
    analogous to env.json's ``qemu_base_image_provenance``."""
    full = build_provenance(image_id="", tools_installed={})
    return {key: full[key] for key in PROVENANCE_EXPECTED_KEYS}


def validate_base_provenance(record: dict[str, Any], expected: dict[str, Any]) -> None:
    """Preflight gate, the modal-native analog of gym_anything's
    ``_validate_qemu_base_image_provenance``. Every key in ``expected`` must
    match the recorded provenance, and the record must carry a valid Modal
    snapshot image id (the snapshot's content identity, immutable by id, plays
    the role the qcow2 ``final_image_sha256`` full-file hash plays there)."""
    if not isinstance(record, dict):
        raise ValueError("modal-native base provenance must be an object")
    mismatches = {
        key: {"expected": value, "actual": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"modal-native base image was built with the wrong contract: {mismatches}"
        )
    image_id = record.get("modal_snapshot_image_id")
    if not (isinstance(image_id, str) and image_id.startswith("im-") and len(image_id) > 3):
        raise ValueError(
            f"modal-native base provenance has no valid modal_snapshot_image_id: {image_id!r}"
        )
