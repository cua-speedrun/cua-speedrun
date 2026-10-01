"""Restore CUA-World desktop state after loading filesystem checkpoints.

Cache installed software and run application startup after each boot. QEMU
guests also receive the GNOME preset's /tmp retention and D-Bus support.
Savevm checkpoints keep their original lifecycle.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import tempfile
from pathlib import Path


# A d rule creates /tmp without the D rule's boot-time recursive removal.
# Keep the remaining vendor rules, especially those for X11 socket cleanup.
BOOTSTRAP = r'''
set -eu
sudo -n python3 - <<'PY'
from pathlib import Path
source = Path('/usr/lib/tmpfiles.d/tmp.conf')
if not source.exists():
    source = Path('/lib/tmpfiles.d/tmp.conf')
lines = source.read_text().splitlines()
found = False
for index, line in enumerate(lines):
    fields = line.split()
    if len(fields) >= 2 and fields[1] == '/tmp' and fields[0] in ('D', 'd'):
        lines[index] = 'd /tmp 1777 root root -'
        found = True
if not found:
    raise RuntimeError('QEMU base has an unsupported /tmp cleanup policy')
target = Path('/etc/tmpfiles.d/tmp.conf')
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text('\n'.join(lines) + '\n')
# Both Compose spellings occur in the benchmark. Prefer a real legacy binary
# when installed; otherwise expose the plugin without installing a second
# Docker engine or changing any environment's installation script.
compose = Path('/usr/local/bin/docker-compose')
if not compose.exists():
    compose.write_text("""#!/bin/sh
if [ -x /usr/bin/docker-compose ]; then
    exec /usr/bin/docker-compose "$@"
fi
exec docker compose "$@"
""")
    compose.chmod(0o755)
PY
if ! command -v dbus-launch >/dev/null; then
    sudo -n apt-get update -qq
    sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y dbus-x11
fi
echo '[cua-world-runtime] /tmp retention and desktop D-Bus configured'
'''


def prepare(env) -> None:
    """Install per-instance preparation; never patch the upstream class."""
    # This reserved marker is supplied by Modal, not inferred from a hostname
    # or from QEMU alone (which can also run on a local evaluation worker).
    if not os.environ.get('MODAL_SANDBOX_ID'):
        return
    from gym_anything.runtime.runners.qemu_apptainer import QemuApptainerRunner
    from gym_anything.runtime.runners.modal_native import ModalNativeRunner

    runner = env._runner
    if isinstance(runner, ModalNativeRunner):
        # Filesystem snapshots preserve installed software, not live windows
        # or application processes. Run startup hooks after each restore.
        env.env_spec.default_cache_level = 'pre_start'
        return
    if not isinstance(runner, QemuApptainerRunner):
        return
    if runner.is_android or runner.is_windows:
        return
    if getattr(runner, '_cs_cua_world_runtime', False):
        return

    if not env.env_spec.default_use_savevm:
        original_level = env.env_spec.default_cache_level or 'pre_start'
        env.env_spec.default_cache_level = 'pre_start'
        print(
            f'[cua-world-runtime] Modal disk-cache default: {original_level} -> '
            'pre_start; original post_start hooks run after every boot',
            flush=True,
        )

    # Never reuse a checkpoint whose boot policy has already destroyed setup
    # state. Salt only this runner's key; leave the shared base and old caches
    # untouched, including all OSWorld caches.
    recipe = Path(__file__).read_bytes()
    runner.env_hash = hashlib.sha256(
        runner.env_hash.encode() + b'\0' + recipe
    ).hexdigest()[:16]
    runner.env_checkpoint = runner.env_checkpoint.with_name(
        f'checkpoint_{runner.env_hash}.qcow2'
    )
    start = runner.start
    checkpoint = runner.create_checkpoint

    def run_checked(command):
        code = runner.exec(command)
        if code:
            raise RuntimeError(f'CUA-World QEMU runtime preparation failed: exit {code}')

    def start_with_preset(*args, **kwargs):
        result = start(*args, **kwargs)
        run_checked('bash -lc ' + shlex.quote(BOOTSTRAP))
        print('[cua-world-runtime] /tmp retention and desktop D-Bus configured', flush=True)
        return result

    def create_checkpoint(*args, **kwargs):
        if runner._use_savevm:
            result = checkpoint(*args, **kwargs)
            if not result:
                raise RuntimeError('CUA-World QEMU checkpoint creation failed')
            return result

        # A host-side sync followed by QEMU "quit" does not flush a nested
        # guest (e.g. Docker Desktop). Let the guest shut down its own services
        # and filesystems before copying its disk. Publish only complete images.
        target = runner._get_checkpoint_path()
        with runner._checkpoint_lock(blocking=True, timeout=600.0) as acquired:
            if not acquired:
                raise RuntimeError('Could not acquire CUA-World checkpoint lock')
            if not runner._running or runner._process is None:
                raise RuntimeError('Cannot checkpoint a stopped QEMU guest')
            print('[cua-world-runtime] shutting down guest before disk checkpoint', flush=True)
            # Poweroff can close SSH before its exit status arrives. QEMU's
            # successful exit below, not the disappearing SSH channel, is the
            # shutdown acknowledgement. An undelivered command times out here
            # without publishing a checkpoint.
            runner.exec('systemctl poweroff --no-block', use_pty=False)
            code = runner._process.wait(timeout=300)
            if code:
                raise RuntimeError(f'QEMU exited abnormally during shutdown: {code}')
            runner._running = False
            if not target.exists():
                with tempfile.TemporaryDirectory(
                    dir=target.parent, prefix=f'.{target.name}.'
                ) as temporary_dir:
                    temporary = Path(temporary_dir) / 'disk.qcow2'
                    converted = runner._run_qemu_img([
                        'convert', '-O', 'qcow2',
                        str(runner._instance_qcow2), str(temporary),
                    ])
                    if converted.returncode:
                        raise RuntimeError(f'Checkpoint conversion failed: {converted.stderr}')
                    temporary.replace(target)
            print(f'[cua-world-runtime] clean disk checkpoint: {target}', flush=True)
            runner._start_from_image(target, use_loadvm=False)
            return True

    runner.start = start_with_preset
    runner.create_checkpoint = create_checkpoint
    runner._cs_cua_world_runtime = True
