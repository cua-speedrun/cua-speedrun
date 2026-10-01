"""Build the OSWorld modal-native base image (a Modal filesystem snapshot).

Extract the pinned OSWorld rootfs and boot its systemd directly on Modal's
kernel. This builder:

  1. Verifies the upstream disk and exports its filesystem, repairing metadata
     on a copy-on-write overlay. A verified rootfs tar can also be reused.
  2. Applies the deterministic delta from ``osworld_modal_base``.
  3. Bakes the env-plane runtime into the sandbox's outer filesystem (CPU
     torch, the OSWorld verifier pip stack, the prefetched evaluator source)
     and gates on the evaluator package importing. Modal refuses build layers
     on snapshot images, so launch time cannot install any of this.
  4. Snapshots the filesystem to a Modal Image id.
  5. Boots a fresh sandbox from that snapshot to validate the desktop and the
     OSWorld verifier contract, then records provenance and caches the image id
     in a Modal Dict keyed by (recipe, source, delta fingerprint).

On first use, the builder downloads and verifies the pinned qcow2 and extracts
its filesystem. An existing checksum-verified rootfs archive can also be used.

Usage:
    python scripts/build_osworld_modal_base.py

Requires ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET`` in the environment or
``.env``. Idempotent: a cached image id for the same contract is reused.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from cua_speedrun.remote import osworld_modal_base as base  # noqa: E402

_MODAL_APP = base.MODAL_APP_NAME
_MODAL_VOLUME = base.MODAL_VOLUME_NAME
_CACHE_DICT = base.BASE_IMAGE_DICT_NAME
_ROOTFS_TAR = "/vol/osworld_rootfs.tar.zst"


def _load_modal_env() -> None:
    env_file = _REPO / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("MODAL_") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _run(sandbox, label, command, timeout=1800, required=False):
    print(f"\n===== {label} =====", flush=True)
    proc = sandbox.exec("bash", "-lc", command, timeout=timeout)
    out = []

    def stream_errors():
        for line in proc.stderr:
            print(line, end="", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=1) as pool:
        errors = pool.submit(stream_errors)
        for line in proc.stdout:
            out.append(line)
            print(line, end="", flush=True)
        errors.result()
    code = proc.wait()
    print(f"[rc={code}]", flush=True)
    if required and code != 0:
        raise SystemExit(f"required step failed: {label}")
    return code, "".join(out)


def main() -> int:
    _load_modal_env()
    import modal

    app = modal.App.lookup(_MODAL_APP, create_if_missing=True)
    volume = modal.Volume.from_name(_MODAL_VOLUME, create_if_missing=True)
    cache = modal.Dict.from_name(_CACHE_DICT, create_if_missing=True)
    builder_image = modal.Image.debian_slim().apt_install(
        "zstd", "util-linux", "kmod", "curl", "unzip", "qemu-utils",
        "libguestfs-tools", "linux-image-amd64",
    )

    cache_key = base.cache_key()
    print(f"recipe: {base.RECIPE}\ncache key: {cache_key}", flush=True)

    def new_sandbox(cpu, mem, timeout):
        return modal.Sandbox.create(
            app=app, image=builder_image, cpu=cpu, memory=mem, timeout=timeout,
            volumes={"/vol": volume}, experimental_options={"vm_runtime": True},
        )

    existing = cache.get(cache_key)
    snapshot_id = existing.get("modal_snapshot_image_id") if isinstance(existing, dict) else None
    tools_installed = existing.get("tools_installed", {}) if isinstance(existing, dict) else {}
    env_plane_packages = existing.get("env_plane_packages", {}) if isinstance(existing, dict) else {}
    rootfs_sha = existing.get("rootfs_tar_sha256") if isinstance(existing, dict) else None
    if snapshot_id:
        try:
            modal.Image.from_id(snapshot_id).build(app)
        except modal.exception.NotFoundError:
            print("Cached desktop image is unavailable; rebuilding", flush=True)
            snapshot_id = None
    if snapshot_id and "--reuse" in sys.argv:
        base.validate_base_provenance(existing, base.expected_provenance_block())
        print("OSWorld native desktop image is ready", flush=True)
        return 0

    if snapshot_id:
        print(f"cache HIT: reusing snapshot {snapshot_id} (will re-validate)", flush=True)
    else:
        print("cache MISS: building base image", flush=True)
        builder = new_sandbox(cpu=8, mem=16384, timeout=2 * 3600)
        print(f"builder sandbox: {builder.object_id}", flush=True)
        try:
            source = base.OSWORLD_IMAGE_CONTRACT
            _, extraction = _run(builder, "prepare verified desktop filesystem", f"""
set -euo pipefail
mkdir -p /osworld
if [ -f {_ROOTFS_TAR} ]; then
    echo '{base.ROOTFS_TAR_SHA256}  {_ROOTFS_TAR}' | sha256sum -c
    echo 'ROOTFS_SHA256={base.ROOTFS_TAR_SHA256}'
    tar --numeric-owner --xattrs --xattrs-include='*' -I 'zstd -T8 -d' -xf {_ROOTFS_TAR} -C /osworld
else
    curl --fail --location --retry 5 '{source['source_url']}' -o /tmp/Ubuntu.qcow2.zip
    echo '{source['archive_sha256']}  /tmp/Ubuntu.qcow2.zip' | sha256sum -c
    unzip -p /tmp/Ubuntu.qcow2.zip Ubuntu.qcow2 > /tmp/source.qcow2
    echo '{source['source_image_sha256']}  /tmp/source.qcow2' | sha256sum -c
    export LIBGUESTFS_BACKEND=direct LIBGUESTFS_BACKEND_SETTINGS=force_tcg
    export SUPERMIN_KERNEL=$(ls /boot/vmlinuz-* | head -1)
    export SUPERMIN_MODULES=/lib/modules/${{SUPERMIN_KERNEL##*/vmlinuz-}}
    # Check and repair filesystem metadata on an overlay, preserving the
    # checksum-verified source. Preen mode only applies safe automatic repairs.
    qemu-img create -f qcow2 -F qcow2 -b /tmp/source.qcow2 /tmp/extraction.qcow2
    filesystems=$(guestfish --rw --format=qcow2 -a /tmp/extraction.qcow2 run : list-filesystems)
    while read -r device filesystem; do
        case "$filesystem" in
            ext2|ext3|ext4)
                guestfish --rw --format=qcow2 -a /tmp/extraction.qcow2 run : e2fsck "${{device%:}}" correct:true
                ;;
        esac
    done <<< "$filesystems"
    if ! guestfish -v --ro --format=qcow2 -a /tmp/extraction.qcow2 -i tar-out / /tmp/rootfs.tar numericowner:true xattrs:true acls:true 2>/tmp/osworld-guestfish.log; then
        cat /tmp/osworld-guestfish.log >&2
        exit 1
    fi
    sha256sum /tmp/rootfs.tar | awk '{{print "ROOTFS_SHA256=" $1}}'
    tar --numeric-owner --xattrs --xattrs-include='*' --acls -xf /tmp/rootfs.tar -C /osworld
    rm /tmp/Ubuntu.qcow2.zip /tmp/extraction.qcow2 /tmp/source.qcow2 /tmp/rootfs.tar /tmp/osworld-guestfish.log
fi
test -f /osworld/etc/os-release
""", timeout=3600, required=True)
            rootfs_sha = next((line.split("=", 1)[1].strip() for line in extraction.splitlines()
                               if line.startswith("ROOTFS_SHA256=")), None)
            if not rootfs_sha or len(rootfs_sha) != 64:
                raise RuntimeError("filesystem extraction did not report its checksum")
            delta_b64 = base64.b64encode(base.rendered_delta().encode()).decode()
            _run(
                builder, "apply deterministic delta",
                f"set -e; echo {delta_b64} | base64 -d > /root/delta.sh; R=/osworld bash /root/delta.sh",
                timeout=1800, required=True,
            )
            _, pkg_out = _run(
                builder, "record resolved delta package versions",
                "cat /osworld/var/lib/osworld-delta-packages.tsv", timeout=60, required=True,
            )
            tools_installed = {}
            for line in pkg_out.splitlines():
                if "\t" in line:
                    name, version = line.split("\t", 1)
                    tools_installed[name.strip()] = version.strip()

            # The env-plane runtime must live in the snapshot: Modal refuses
            # every build layer on a filesystem-snapshot image, so nothing can
            # be pip-installed at launch time.
            import shlex

            pip_packages = " ".join(shlex.quote(p) for p in base.ENV_PLANE_PIP)
            torch_packages = " ".join(base.ENV_PLANE_TORCH_PIP)
            _run(
                builder, "install env-plane runtime (CPU torch + verifier stack)",
                "set -e; "
                "apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install "
                f"-y -q --no-install-recommends {' '.join(base.ENV_PLANE_APT)}; "
                f"pip3 install --no-cache-dir -q --index-url {base.ENV_PLANE_TORCH_INDEX} {torch_packages}; "
                f"pip3 install --no-cache-dir -q {pip_packages}; "
                "echo ENV_PLANE_DEPS_OK",
                timeout=2400, required=True,
            )
            evaluator_root = f"{base.OSWORLD_EVALUATOR_CACHE}/{base.OSWORLD_EVALUATOR_COMMIT}"
            _run(
                builder, "prefetch OSWorld evaluator source",
                "set -e; "
                f"mkdir -p {evaluator_root}; "
                "python3 -c \"import urllib.request; urllib.request.urlretrieve("
                f"'https://github.com/xlang-ai/OSWorld/archive/{base.OSWORLD_EVALUATOR_COMMIT}.tar.gz',"
                " '/tmp/osworld-src.tgz')\"; "
                f"tar -xzf /tmp/osworld-src.tgz -C {evaluator_root} --strip-components=1 "
                f"OSWorld-{base.OSWORLD_EVALUATOR_COMMIT}/desktop_env; "
                "rm /tmp/osworld-src.tgz; "
                f"test -f {evaluator_root}/desktop_env/desktop_env.py; "
                "echo OSWORLD_RUNTIME_CACHED",
                timeout=900, required=True,
            )
            # Refuse to snapshot an image whose verifiers cannot import.
            _run(
                builder, "evaluator import gate",
                f"cd {evaluator_root} && python3 -c "
                "'from desktop_env.desktop_env import DesktopEnv; "
                "print(\"OSWORLD_RUNTIME_IMPORT_OK\")'",
                timeout=900, required=True,
            )
            _, pip_out = _run(
                builder, "record resolved env-plane package versions",
                "pip3 list --format=freeze", timeout=120, required=True,
            )
            wanted = {
                p.split("<")[0].split(">")[0].split("=")[0].strip().lower().replace("_", "-")
                for p in base.ENV_PLANE_PIP + base.ENV_PLANE_TORCH_PIP
            }
            env_plane_packages = {}
            for line in pip_out.splitlines():
                if "==" in line:
                    name, version = line.split("==", 1)
                    if name.strip().lower().replace("_", "-") in wanted:
                        env_plane_packages[name.strip()] = version.strip()

            print("\ntaking filesystem snapshot of the delta-applied rootfs ...", flush=True)
            snapshot_id = builder.snapshot_filesystem(timeout=1200, ttl=None).object_id
            print(f"SNAPSHOT_IMAGE_ID={snapshot_id}", flush=True)
        finally:
            builder.terminate()

    # Validate a fresh boot from the snapshot before recording provenance.
    print(f"\nvalidating boot from snapshot {snapshot_id} ...", flush=True)
    validator = modal.Sandbox.create(
        app=app, image=modal.Image.from_id(snapshot_id), cpu=8, memory=16384,
        timeout=3600, experimental_options={"vm_runtime": True},
    )
    print(f"validator sandbox: {validator.object_id}", flush=True)
    ns = "nsenter --target $(cat /run/osworld-systemd.pid) --mount --pid --root --wd --"
    user = (
        f"sudo -u {base.DESKTOP_USER} env "
        f"DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/{base.DESKTOP_UID}/bus "
        f"HOME=/home/{base.DESKTOP_USER} DISPLAY={base.X11_DISPLAY} "
        f"XAUTHORITY=/run/user/{base.DESKTOP_UID}/gdm/Xauthority timeout 20"
    )
    validated = False
    try:
        _run(validator, "boot base image", _boot_script(), timeout=300, required=True)
        import time
        # The image's own GDM autologin session on the dummy Xorg driver;
        # ready means gnome-shell is running for the desktop user.
        deadline, active = time.time() + 300, False
        while time.time() < deadline:
            probe = validator.exec(
                "bash", "-lc",
                f"{ns} pgrep -u {base.DESKTOP_USER} -x gnome-shell 2>&1 || true",
                timeout=30,
            )
            if probe.stdout.read().strip():
                active = True
                break
            probe.wait()
            time.sleep(5)
        if not active:
            raise SystemExit("base image did not boot to the GDM autologin session")
        time.sleep(20)
        _run(
            validator,
            "canonical OSWorld boot server is ready",
            f"{ns} systemctl is-active osworld.service && "
            f"{ns} /usr/bin/python -c \"import urllib.request; "
            "assert urllib.request.urlopen('http://127.0.0.1:5000/platform', "
            "timeout=20).read()\"",
            timeout=60,
            required=True,
        )
        # Exec'd commands carry no X cookie; open local access from the
        # session once, exactly as the env-plane does at boot.
        _run(
            validator, "open local X access",
            f"{ns} {user} xhost +local:", timeout=60, required=True,
        )
        # The stock Ubuntu session extensions are what a vision agent sees:
        # the left dock, the ~/Desktop icons, and the appindicator tray.
        rc, ext_out = _run(
            validator, "ubuntu session extensions are loaded",
            f"{ns} {user} gnome-extensions list --enabled",
            timeout=60,
        )
        expected_extensions = (
            "ding@rastersoft.com", "ubuntu-dock", "ubuntu-appindicators",
        )
        if rc != 0 or any(item not in ext_out for item in expected_extensions):
            raise SystemExit(
                f"ubuntu session extensions missing (got: {ext_out.strip()!r})"
            )
        rc, _ = _run(
            validator, "gnome-terminal activates over the user bus",
            f"{ns} {user} gnome-terminal --working-directory=/home/{base.DESKTOP_USER}; "
            f"sleep 3; {ns} pgrep -u {base.DESKTOP_USER} -f gnome-terminal-server",
            timeout=90,
        )
        if rc != 0:
            raise SystemExit("gnome-terminal could not be activated on the user bus")
        # A rendered frame, not just live processes: a black or dead display
        # still runs gnome-shell.
        rc, scrot_out = _run(
            validator, "screenshot renders",
            f"{ns} {user} scrot -o /home/{base.DESKTOP_USER}/validate.png && "
            f"stat -c %s /osworld/home/{base.DESKTOP_USER}/validate.png",
            timeout=60,
        )
        size = int(scrot_out.strip().splitlines()[-1] or 0) if rc == 0 else 0
        if size < 100_000:
            raise SystemExit(f"desktop screenshot too small ({size} bytes)")
        # Report (not gate) the image's own idle/lock policy; fidelity means
        # inheriting whatever upstream ships.
        _run(
            validator, "session idle/lock policy (informational)",
            f"{ns} {user} gsettings get org.gnome.desktop.session idle-delay; "
            f"{ns} {user} gsettings get org.gnome.desktop.screensaver lock-enabled",
            timeout=60,
        )
        _run(
            validator, "autolock verifier contract",
            f"{ns} {user} gsettings set org.gnome.desktop.screensaver lock-enabled false; "
            f"echo start=$({ns} {user} gsettings get org.gnome.desktop.screensaver lock-enabled); "
            f"{ns} {user} gsettings set org.gnome.desktop.screensaver lock-enabled true; "
            f"echo final=$({ns} {user} gsettings get org.gnome.desktop.screensaver lock-enabled)",
            timeout=90,
        )
        validated = True
    finally:
        if validated and snapshot_id:
            provenance = base.build_provenance(
                snapshot_id, tools_installed, env_plane_packages
            )
            provenance["rootfs_tar_sha256"] = rootfs_sha
            cache[cache_key] = provenance
            base.validate_base_provenance(provenance, base.expected_provenance_block())
            print("\nprovenance recorded + cached:", flush=True)
            print(json.dumps(provenance, indent=2), flush=True)
        validator.terminate()

    print(f"\nBASE_IMAGE_OK={validated} image_id={snapshot_id}", flush=True)
    return 0 if validated else 1


def _boot_script() -> str:
    return r"""
cat > /osworld-boot.sh <<'EOF'
#!/bin/bash
set -e
mount --bind /osworld /osworld; cd /osworld
mount --rbind /dev dev; mount --rbind /sys sys; mount -t proc proc proc
mount -t tmpfs -o mode=0755,nosuid,nodev tmpfs run; mount -t tmpfs tmpfs tmp
ln -sfn /dev/null etc/systemd/system/acpid.path
mkdir -p old_root; pivot_root . old_root; umount -l /old_root; rmdir /old_root 2>/dev/null || true
exec env -i container=modal TERM=linux PATH=/usr/sbin:/usr/bin:/sbin:/bin /sbin/init
EOF
chmod +x /osworld-boot.sh; rm -f /run/osworld-systemd.pid
nohup unshare --fork --pid --mount --propagation private /osworld-boot.sh > /osworld-boot.log 2>&1 &
launcher=$!
for _ in $(seq 1 300); do
    child=$(cat /proc/"$launcher"/task/"$launcher"/children 2>/dev/null | awk '{print $1}' || true)
    if [ -n "$child" ] && [ -d "/proc/$child" ]; then echo "$child" > /run/osworld-systemd.pid; exit 0; fi
    kill -0 "$launcher" 2>/dev/null || { cat /osworld-boot.log; exit 1; }; sleep 0.1
done
echo "systemd never appeared"; cat /osworld-boot.log; exit 1
"""


if __name__ == "__main__":
    raise SystemExit(main())
