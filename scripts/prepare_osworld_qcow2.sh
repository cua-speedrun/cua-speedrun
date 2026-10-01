#!/usr/bin/env bash
# Download and prepare the upstream OSWorld Linux qcow2 used by this repo.
#
# No image is shipped by cua-speedrun. This is the Linux-QEMU preparation flow
# from xlang-ai/osworld_image, narrowed
# to what cua-speedrun needs: a pinned OSWorld base plus password SSH, xdotool,
# the Hugging Face Hub/Xet client used by upstream task assets, and passwordless
# sudo for the desktop user (the runner wraps hooks in `sudo -E`).
# It does not apply that repository's later Packer/Ansible application delta.
#
# On hosts without a native libguestfs/virt-customize (e.g. EL9 clusters), run
# this through scripts/prepare_osworld_qcow2_apptainer.sh, which executes it
# inside an Ubuntu container so the appliance kernel matches.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_CONTRACT="$HERE/../benchmarks/osworld-image.json"
mapfile -t IMAGE_DEFAULTS < <(python3 - "$IMAGE_CONTRACT" <<'PY'
import json
import sys

contract = json.load(open(sys.argv[1], encoding="utf-8"))
source = contract["official_source"]
guest = contract["guest"]
for value in (
    contract["provenance_schema_version"],
    contract["recipe"],
    source["revision"],
    source["url"],
    source["archive_sha256"],
    source["image_sha256"],
    source["virtual_size"],
    guest["ssh_user"],
    guest["ssh_password"],
):
    print(value)
PY
)

PROVENANCE_SCHEMA_VERSION="${IMAGE_DEFAULTS[0]}"
PROVENANCE_RECIPE="${IMAGE_DEFAULTS[1]}"
SOURCE_REV="${OSWORLD_QCOW2_SOURCE_REV:-${IMAGE_DEFAULTS[2]}}"
SOURCE_URL="${OSWORLD_QCOW2_URL:-${IMAGE_DEFAULTS[3]}}"
ARCHIVE_SHA256="${OSWORLD_QCOW2_ARCHIVE_SHA256:-${IMAGE_DEFAULTS[4]}}"
SOURCE_IMAGE_SHA256="${OSWORLD_QCOW2_SOURCE_SHA256:-${IMAGE_DEFAULTS[5]}}"
EXPECTED_VIRTUAL_SIZE="${OSWORLD_QCOW2_VIRTUAL_SIZE:-${IMAGE_DEFAULTS[6]}}"
DESKTOP_USER="${IMAGE_DEFAULTS[7]}"
DESKTOP_PASSWORD="${IMAGE_DEFAULTS[8]}"
SKIP_SSH_PREP="${OSWORLD_QCOW2_SKIP_SSH_PREP:-0}"
HUGGINGFACE_HUB_VERSION="0.36.2"
HF_XET_VERSION="1.2.0"
# Ubuntu's ordinary archive removes superseded security packages.  Resolve
# every guest provisioning deb through one immutable archive snapshot instead
# of relying on whichever files happen to remain on a moving mirror.
UBUNTU_SNAPSHOT="${OSWORLD_UBUNTU_SNAPSHOT:-20260501T000000Z}"
UBUNTU_SNAPSHOT_POOL="https://snapshot.ubuntu.com/ubuntu/$UBUNTU_SNAPSHOT/pool"

CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
OUTPUT="${OSWORLD_QEMU_BASE_IMAGE:-$CACHE_HOME/gym-anything/qemu/osworld_ubuntu.qcow2}"
DOWNLOAD_DIR="${OSWORLD_QCOW2_DOWNLOAD_DIR:-$CACHE_HOME/cua-speedrun/osworld-image/$SOURCE_REV}"
FORCE=0

usage() {
    cat <<'EOF'
Usage: scripts/prepare_osworld_qcow2.sh [options]

Download, checksum, enable SSH in, and atomically install the OSWorld qcow2.

Options:
  --output PATH         Installed qcow2 path. Defaults to
                        $OSWORLD_QEMU_BASE_IMAGE or
                        ~/.cache/gym-anything/qemu/osworld_ubuntu.qcow2.
  --download-dir PATH   Resumable archive and provision-asset cache.
  --force               Rebuild even if image provenance and checks pass.
  -h, --help            Show this help.

The source revision, URL, checksums, and expected virtual size can be
overridden with OSWORLD_QCOW2_SOURCE_REV, OSWORLD_QCOW2_URL,
OSWORLD_QCOW2_ARCHIVE_SHA256, OSWORLD_QCOW2_SOURCE_SHA256, and
OSWORLD_QCOW2_VIRTUAL_SIZE. Overrides are useful for an audited mirror or test.
OSWORLD_QCOW2_SKIP_SSH_PREP=1 is only for a minimal test fixture.
EOF
}

while (($#)); do
    case "$1" in
        --output)
            [[ $# -ge 2 ]] || { printf '%s\n' '--output requires a path' >&2; exit 2; }
            OUTPUT="$2"
            shift 2
            ;;
        --download-dir)
            [[ $# -ge 2 ]] || { printf '%s\n' '--download-dir requires a path' >&2; exit 2; }
            DOWNLOAD_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$SKIP_SSH_PREP" != 0 && "$SKIP_SSH_PREP" != 1 ]]; then
    printf 'OSWORLD_QCOW2_SKIP_SSH_PREP must be 0 or 1\n' >&2
    exit 2
fi

required_commands=(awk curl flock python3 qemu-img sha256sum unzip)
if [[ "$SKIP_SSH_PREP" -eq 0 ]]; then
    required_commands+=(apt-get dpkg-deb virt-customize)
fi
for command in "${required_commands[@]}"; do
    command -v "$command" >/dev/null 2>&1 || {
        printf 'Missing required command: %s\n' "$command" >&2
        exit 1
    }
done

OUTPUT="$(python3 -c 'import os, sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$OUTPUT")"
DOWNLOAD_DIR="$(python3 -c 'import os, sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$DOWNLOAD_DIR")"
mkdir -p "$(dirname "$OUTPUT")" "$DOWNLOAD_DIR"

ARCHIVE="$DOWNLOAD_DIR/Ubuntu.qcow2.zip"
PARTIAL="$ARCHIVE.part"
PROVENANCE="$OUTPUT.provenance.json"
LOCK_FILE="$OUTPUT.prepare.lock"
SSH_DEB_DIR="$DOWNLOAD_DIR/ssh-debs"
TOOL_DEB_DIR="$DOWNLOAD_DIR/tool-debs"

exec 9>"$LOCK_FILE"
printf 'Waiting for image lock: %s\n' "$LOCK_FILE"
flock 9

sha256_matches() {
    local path="$1"
    local expected="$2"
    [[ -f "$path" ]] && printf '%s  %s\n' "$expected" "$path" | sha256sum --check --status
}

validate_qcow2() {
    local path="$1"
    local info

    [[ -f "$path" ]] || return 1
    info="$(qemu-img info --output=json "$path")" || return 1
    printf '%s\n' "$info" | python3 -c '
import json
import sys

expected_size = int(sys.argv[1])
info = json.load(sys.stdin)
if info.get("format") != "qcow2":
    raise SystemExit("expected qcow2, got {!r}".format(info.get("format")))
if info.get("virtual-size") != expected_size:
    raise SystemExit(
        "expected virtual size {}, got {!r}".format(
            expected_size, info.get("virtual-size")
        )
    )
' "$EXPECTED_VIRTUAL_SIZE" || return 1
    qemu-img check -q "$path"
}

provenance_final_sha256() {
    [[ -f "$PROVENANCE" ]] || return 1
    python3 -c '
import json
import sys

(
    path,
    schema_version,
    recipe,
    revision,
    archive_sha,
    source_sha,
    skip_ssh,
    ssh_user,
    huggingface_hub_version,
    hf_xet_version,
) = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    data = json.load(stream)
expected = {
    "schema_version": int(schema_version),
    "recipe": recipe,
    "source_revision": revision,
    "archive_sha256": archive_sha,
    "source_image_sha256": source_sha,
    "ssh_prepared": skip_ssh == "0",
    "tools_prepared": skip_ssh == "0",
    "tools_installed": (
        [
            "xdotool",
            f"huggingface-hub=={huggingface_hub_version}",
            f"hf-xet=={hf_xet_version}",
        ]
        if skip_ssh == "0"
        else []
    ),
    "ssh_user": ssh_user,
    "guest_auth_prepared": skip_ssh == "0",
}
if any(data.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
print(data["final_image_sha256"])
' "$PROVENANCE" "$PROVENANCE_SCHEMA_VERSION" "$PROVENANCE_RECIPE" \
    "$SOURCE_REV" "$ARCHIVE_SHA256" "$SOURCE_IMAGE_SHA256" \
    "$SKIP_SSH_PREP" "$DESKTOP_USER" "$HUGGINGFACE_HUB_VERSION" "$HF_XET_VERSION"
}

if [[ "$FORCE" -eq 0 ]]; then
    installed_sha="$(provenance_final_sha256 2>/dev/null || true)"
    if [[ -n "$installed_sha" ]] && sha256_matches "$OUTPUT" "$installed_sha" && validate_qcow2 "$OUTPUT"; then
        printf 'OSWorld qcow2 and provenance are already valid: %s\n' "$OUTPUT"
        printf 'sha256: %s\n' "$installed_sha"
        exit 0
    fi
fi

if sha256_matches "$ARCHIVE" "$ARCHIVE_SHA256"; then
    printf 'Using verified archive: %s\n' "$ARCHIVE"
else
    if [[ -f "$ARCHIVE" ]]; then
        stale="$ARCHIVE.invalid.$(date +%s).$$"
        mv "$ARCHIVE" "$stale"
        printf 'Moved invalid archive to: %s\n' "$stale" >&2
    fi
    printf 'Downloading pinned OSWorld archive...\n'
    printf 'Source: %s\n' "$SOURCE_URL"
    curl \
        --location \
        --fail \
        --retry 5 \
        --retry-all-errors \
        --continue-at - \
        --output "$PARTIAL" \
        "$SOURCE_URL"
    if ! sha256_matches "$PARTIAL" "$ARCHIVE_SHA256"; then
        printf 'Archive checksum mismatch; preserving partial file: %s\n' "$PARTIAL" >&2
        exit 1
    fi
    mv "$PARTIAL" "$ARCHIVE"
    printf 'Archive checksum verified: %s\n' "$ARCHIVE_SHA256"
fi

mapfile -t qcow2_members < <(unzip -Z -1 "$ARCHIVE" | awk '/(^|\/)Ubuntu\.qcow2$/')
if [[ "${#qcow2_members[@]}" -ne 1 ]]; then
    printf 'Expected exactly one Ubuntu.qcow2 archive member, found %s\n' "${#qcow2_members[@]}" >&2
    exit 1
fi

TEMP_SOURCE="$OUTPUT.source.tmp.$$"
TEMP_IMAGE="$OUTPUT.tmp.$$"
TEMP_PROVENANCE="$PROVENANCE.tmp.$$"
cleanup() {
    rm -f "$TEMP_SOURCE" "$TEMP_IMAGE" "$TEMP_PROVENANCE"
}
trap cleanup EXIT

printf 'Extracting and checking %s\n' "${qcow2_members[0]}"
unzip -p "$ARCHIVE" "${qcow2_members[0]}" > "$TEMP_SOURCE"
if ! sha256_matches "$TEMP_SOURCE" "$SOURCE_IMAGE_SHA256" || ! validate_qcow2 "$TEMP_SOURCE"; then
    printf 'Extracted source image failed checksum or qcow2 validation\n' >&2
    exit 1
fi

printf 'Normalizing the source qcow2\n'
qemu-img convert -f qcow2 -O qcow2 "$TEMP_SOURCE" "$TEMP_IMAGE"

download_checked() {
    local url="$1"
    local expected="$2"
    local destination="$3"
    local partial="$destination.part"

    if sha256_matches "$destination" "$expected"; then
        return
    fi
    rm -f "$partial"
    printf 'Downloading provision asset: %s\n' "$(basename "$destination")"
    curl --location --fail --retry 3 --connect-timeout 20 --max-time 300 \
        --output "$partial" "$url"
    if ! sha256_matches "$partial" "$expected"; then
        printf 'Provision-asset checksum mismatch: %s\n' "$destination" >&2
        exit 1
    fi
    mv "$partial" "$destination"
}

download_ssh_debs() {
    mkdir -p "$SSH_DEB_DIR"
    download_checked \
        "$UBUNTU_SNAPSHOT_POOL/main/o/openssh/openssh-client_8.9p1-3ubuntu0.15_amd64.deb" \
        '36ae97d42dd34f6ae15e31758a6dbad4a4ade898acfd3e3cb9fd9332744d0cee' \
        "$SSH_DEB_DIR/openssh-client_8.9p1-3ubuntu0.15_amd64.deb"
    download_checked \
        "$UBUNTU_SNAPSHOT_POOL/main/o/openssh/openssh-sftp-server_8.9p1-3ubuntu0.15_amd64.deb" \
        '37a796b558f93bd2a0950794f3b477d602f908f239096c06fba10930391dc698' \
        "$SSH_DEB_DIR/openssh-sftp-server_8.9p1-3ubuntu0.15_amd64.deb"
    download_checked \
        "$UBUNTU_SNAPSHOT_POOL/main/o/openssh/openssh-server_8.9p1-3ubuntu0.15_amd64.deb" \
        '6b4f534348c282f0d77e27ba8d68e505b82fba60fc1296afc492f32ac51b131f' \
        "$SSH_DEB_DIR/openssh-server_8.9p1-3ubuntu0.15_amd64.deb"
    download_checked \
        "$UBUNTU_SNAPSHOT_POOL/main/n/ncurses/ncurses-term_6.3-2ubuntu0.1_all.deb" \
        'e67643b4f7af2e3f908da9140dcd3d3cdcd64dc6b2529bc63a907bf9c881ea8c' \
        "$SSH_DEB_DIR/ncurses-term_6.3-2ubuntu0.1_all.deb"
    download_checked \
        "$UBUNTU_SNAPSHOT_POOL/main/s/ssh-import-id/ssh-import-id_5.11-0ubuntu1_all.deb" \
        '245ebcce7417b587f06c38dbdc103e445334b93766aaa05594dd5ba09be142f7' \
        "$SSH_DEB_DIR/ssh-import-id_5.11-0ubuntu1_all.deb"
}

download_tool_debs() {
    # The gym-anything runner drives the desktop with xdotool (input fallback and
    # the OSWorld setup hooks) and reads it back over QEMU's own VNC, so xdotool
    # is the one tool the upstream OSWorld image lacks. wmctrl, ffmpeg, and
    # x11-utils are already present. Pinned like the SSH debs to keep the build
    # reproducible and offline.
    mkdir -p "$TOOL_DEB_DIR"
    download_checked \
        "$UBUNTU_SNAPSHOT_POOL/universe/x/xdotool/libxdo3_3.20160805.1-4_amd64.deb" \
        '69432493950855718836e1aadc4c490c103f35b0c5dacc0e28a0f55fc21d2613' \
        "$TOOL_DEB_DIR/libxdo3_3.20160805.1-4_amd64.deb"
    download_checked \
        "$UBUNTU_SNAPSHOT_POOL/universe/x/xdotool/xdotool_3.20160805.1-4_amd64.deb" \
        '93198ce669e04f9b9d79f5e5e2d0fcd89fe411e7a7ebc948e9f4cbb563f62967' \
        "$TOOL_DEB_DIR/xdotool_3.20160805.1-4_amd64.deb"
}

prepare_libguestfs_kernel() {
    local kernel_version="${LIBGUESTFS_KERNEL_VERSION:-$(uname -r)}"
    local module_dir="/lib/modules/$kernel_version"
    local kernel_file="/boot/vmlinuz-$kernel_version"

    if [[ -r "$kernel_file" ]]; then
        return
    fi
    if [[ ! -d "$module_dir" ]]; then
        printf 'Kernel module directory not found: %s\n' "$module_dir" >&2
        exit 1
    fi

    local kernel_cache="$DOWNLOAD_DIR/libguestfs-kernel/$kernel_version"
    local extracted_kernel="$kernel_cache/vmlinuz-$kernel_version"
    mkdir -p "$kernel_cache"
    if [[ ! -r "$extracted_kernel" ]]; then
        local work_dir
        work_dir="$(mktemp -d)"
        (
            cd "$work_dir"
            apt-get download "linux-image-$kernel_version"
            dpkg-deb -x "linux-image-$kernel_version"_*.deb extract
            cp "extract/boot/vmlinuz-$kernel_version" "$extracted_kernel"
        )
        rm -rf "$work_dir"
    fi
    export SUPERMIN_KERNEL="$extracted_kernel"
    export SUPERMIN_MODULES="$module_dir"
    export SUPERMIN_KERNEL_VERSION="$kernel_version"
}

if [[ "$SKIP_SSH_PREP" -eq 0 ]]; then
    prepare_libguestfs_kernel
    download_ssh_debs
    download_tool_debs
    printf 'Installing SSH, xdotool, Hugging Face Hub/Xet, and passwordless sudo in the qcow2\n'
    virt-customize \
        -a "$TEMP_IMAGE" \
        --copy-in "$SSH_DEB_DIR:/tmp" \
        --copy-in "$TOOL_DEB_DIR:/tmp" \
        --run-command 'DEBIAN_FRONTEND=noninteractive dpkg -i /tmp/ssh-debs/*.deb' \
        --run-command 'DEBIAN_FRONTEND=noninteractive dpkg -i /tmp/tool-debs/*.deb' \
        --run-command "python3 -m pip install --no-cache-dir huggingface-hub==$HUGGINGFACE_HUB_VERSION hf-xet==$HF_XET_VERSION" \
        --run-command 'install -d -m 0755 /etc/ssh/sshd_config.d' \
        --write '/etc/ssh/sshd_config.d/99-osworld-password.conf:PasswordAuthentication yes
PubkeyAuthentication yes
UseDNS no
' \
        --password "$DESKTOP_USER:password:$DESKTOP_PASSWORD" \
        --run-command 'systemctl enable ssh' \
        --write "/etc/sudoers.d/99-osworld-nopasswd:$DESKTOP_USER ALL=(ALL) NOPASSWD:ALL
" \
        --run-command 'chmod 440 /etc/sudoers.d/99-osworld-nopasswd' \
        --run-command 'visudo -cf /etc/sudoers.d/99-osworld-nopasswd'
fi

if ! validate_qcow2 "$TEMP_IMAGE"; then
    printf 'Prepared image failed qcow2 validation\n' >&2
    exit 1
fi
FINAL_SHA256="$(sha256sum "$TEMP_IMAGE" | awk '{print $1}')"

python3 - "$TEMP_PROVENANCE" "$PROVENANCE_SCHEMA_VERSION" "$PROVENANCE_RECIPE" \
    "$SOURCE_URL" "$SOURCE_REV" "$ARCHIVE_SHA256" \
    "$SOURCE_IMAGE_SHA256" "$FINAL_SHA256" "$EXPECTED_VIRTUAL_SIZE" "$SKIP_SSH_PREP" \
    "$DESKTOP_USER" "$HUGGINGFACE_HUB_VERSION" "$HF_XET_VERSION" <<'PY'
import datetime
import json
import sys

(
    path,
    schema_version,
    recipe,
    source_url,
    revision,
    archive_sha,
    source_sha,
    final_sha,
    virtual_size,
    skip_ssh,
    ssh_user,
    huggingface_hub_version,
    hf_xet_version,
) = sys.argv[1:]
data = {
    "schema_version": int(schema_version),
    "recipe": recipe,
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_url": source_url,
    "source_revision": revision,
    "archive_sha256": archive_sha,
    "source_image_sha256": source_sha,
    "final_image_sha256": final_sha,
    "virtual_size": int(virtual_size),
    "ssh_prepared": skip_ssh == "0",
    "tools_prepared": skip_ssh == "0",
    "tools_installed": (
        [
            "xdotool",
            f"huggingface-hub=={huggingface_hub_version}",
            f"hf-xet=={hf_xet_version}",
        ]
        if skip_ssh == "0"
        else []
    ),
    "nopasswd_sudo": skip_ssh == "0",
    "ssh_user": ssh_user,
    "guest_auth_prepared": skip_ssh == "0",
    "reference": "https://github.com/xlang-ai/osworld_image",
}
with open(path, "w", encoding="utf-8") as stream:
    json.dump(data, stream, indent=2)
    stream.write("\n")
PY

chmod 0644 "$TEMP_IMAGE" "$TEMP_PROVENANCE"
mv -f "$TEMP_IMAGE" "$OUTPUT"
mv -f "$TEMP_PROVENANCE" "$PROVENANCE"
rm -f "$TEMP_SOURCE"
trap - EXIT

printf 'Installed verified OSWorld qcow2: %s\n' "$OUTPUT"
printf 'sha256: %s\n' "$FINAL_SHA256"
printf 'provenance: %s\n' "$PROVENANCE"
printf 'Use it with: export OSWORLD_QEMU_BASE_IMAGE=%q\n' "$OUTPUT"
