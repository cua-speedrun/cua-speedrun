#!/usr/bin/env bash
# Run prepare_osworld_qcow2.sh on hosts that have no native libguestfs /
# virt-customize (e.g. EL9 / RHEL-family clusters), by executing it inside an
# Ubuntu 22.04 Apptainer sandbox.
#
# Why a container: libguestfs boots a small appliance that needs a kernel plus
# matching modules. On an EL9 host the only kernel around is the el9 one, whose
# modules are not what the Ubuntu appliance wants, so virt-customize fails. The
# Ubuntu sandbox ships its own generic kernel, so the appliance boots even
# though the host kernel is el9. Everything else is the normal, reproducible,
# pinned prep: this wrapper just forwards its arguments and OSWORLD_* env vars.
#
# Requirements: apptainer, a readable/writable /dev/kvm, and OSWORLD_APPTAINER_WORK
# pointing at a writable, KVM-capable directory. Keep the output image and cache
# (OSWORLD_QEMU_BASE_IMAGE, XDG_CACHE_HOME) under that directory so they are
# visible inside the container.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
PREP="$HERE/prepare_osworld_qcow2.sh"

command -v apptainer >/dev/null 2>&1 || { printf 'apptainer is required\n' >&2; exit 1; }
[[ -w /dev/kvm ]] || { printf '/dev/kvm must be readable/writable for the libguestfs appliance\n' >&2; exit 1; }
: "${OSWORLD_APPTAINER_WORK:?set OSWORLD_APPTAINER_WORK to a writable, KVM-capable directory}"

WORK="$(python3 -c 'import os, sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$OSWORLD_APPTAINER_WORK")"
mkdir -p "$WORK"
# Fakeroot ownership emulation breaks on network filesystems (NFS/Lustre/
# CIFS): `apptainer build` chowns the staged rootfs under APPTAINER_TMPDIR
# ("ownership change not allowed"), and dpkg inside the writable sandbox
# chowns files in the sandbox rootfs itself. Clusters usually put the work
# dir on shared storage, so when $WORK is on a network filesystem default
# both onto node-local storage. The sandbox is a rebuildable cache, so a
# per-node copy only costs its one-time build. Explicit OSWORLD_APPTAINER_
# SANDBOX / APPTAINER_TMPDIR always win.
LOCAL_SCRATCH="${TMPDIR:-/var/tmp}/cua-speedrun-apptainer-$(id -u)"
DEFAULT_SBX="$WORK/ubuntu-guestfs-sandbox"
DEFAULT_APPTAINER_TMP="$WORK/apptainer-tmp"
case "$(stat -f -c %T "$WORK" 2>/dev/null)" in
    nfs*|lustre*|cifs|smb*|fuse*|gpfs|ceph*)
        DEFAULT_SBX="$LOCAL_SCRATCH/ubuntu-guestfs-sandbox"
        DEFAULT_APPTAINER_TMP="$LOCAL_SCRATCH/apptainer-tmp"
        ;;
esac
SBX="${OSWORLD_APPTAINER_SANDBOX:-$DEFAULT_SBX}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$WORK/apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$DEFAULT_APPTAINER_TMP}"
GUESTFS_HOST_DIR="$LOCAL_SCRATCH/libguestfs"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" \
    "$GUESTFS_HOST_DIR/cache" "$GUESTFS_HOST_DIR/tmp" "$GUESTFS_HOST_DIR/sock"
chmod 0700 "$GUESTFS_HOST_DIR" "$GUESTFS_HOST_DIR/cache" \
    "$GUESTFS_HOST_DIR/tmp" "$GUESTFS_HOST_DIR/sock"

# 1. One-time: build the Ubuntu sandbox with the prep script's dependencies.
if [[ ! -f "$SBX/.deps-ok" ]]; then
    printf 'Building the one-time Ubuntu guestfs sandbox: %s\n' "$SBX"
    rm -rf "$SBX"
    apptainer build --fakeroot --sandbox "$SBX" docker://ubuntu:22.04
    apptainer exec --fakeroot --writable "$SBX" bash -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y -qq \
            libguestfs-tools qemu-utils curl unzip python3 ca-certificates linux-image-generic
    '
    touch "$SBX/.deps-ok"
fi

# A writable sandbox cannot synthesize bind destinations at launch time.
# Create the absolute host paths inside it once; the bind then replaces those
# empty directories with the evaluator's real work tree.
mkdir -p "$SBX$REPO_ROOT" "$SBX$WORK" "$SBX/run/cua-speedrun-libguestfs"

# Bind the work dir and the repo into the container, deduplicating identical or
# nested paths (a parent bind already covers a child).
BINDS=(--bind /dev/kvm --bind "$GUESTFS_HOST_DIR:/run/cua-speedrun-libguestfs")
SEEN=()
bind_path() {
    local p="$1" s
    for s in "${SEEN[@]:-}"; do
        [[ -z "$s" ]] && continue
        case "$p/" in "$s/"*) return 0 ;; esac
    done
    SEEN+=("$p")
    BINDS+=(--bind "$p:$p")
}
if [[ ${#WORK} -le ${#REPO_ROOT} ]]; then
    bind_path "$WORK"; bind_path "$REPO_ROOT"
else
    bind_path "$REPO_ROOT"; bind_path "$WORK"
fi

# 2. Run the real prep inside the sandbox.
#    - unset XDG_RUNTIME_DIR: it often points at a host path absent in the
#      container, which makes libguestfs bail while parsing the environment.
#    - LIBGUESTFS_KERNEL_VERSION = the sandbox's own Ubuntu kernel, so the prep's
#      prepare_libguestfs_kernel resolves that instead of the el9 host kernel.
#    - LIBGUESTFS_BACKEND=direct: no libvirt inside the container.
exec apptainer exec --fakeroot --writable \
    "${BINDS[@]}" \
    "$SBX" \
    bash -c '
        set -euo pipefail
        unset XDG_RUNTIME_DIR
        guestfs_dir=/run/cua-speedrun-libguestfs
        export HOME=/root TMPDIR="$guestfs_dir/tmp" LIBGUESTFS_BACKEND=direct
        export LIBGUESTFS_CACHEDIR="$guestfs_dir/cache"
        export LIBGUESTFS_TMPDIR="$guestfs_dir/tmp"
        export LIBGUESTFS_SOCKDIR="$guestfs_dir/sock"
        export LIBGUESTFS_KERNEL_VERSION="$(ls -1 /lib/modules | head -1)"
        exec bash "$0" "$@"
    ' "$PREP" "$@"
