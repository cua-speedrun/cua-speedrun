"""Startup compatibility for Gym-Anything's native Modal desktops."""

import json
from pathlib import PurePosixPath

DESKTOP_SETUP = """
import fcntl
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

for device in json.loads(sys.argv[1]):
    try:
        descriptor = os.open(device, os.O_RDWR)
        try:
            if device == '/dev/kvm' and fcntl.ioctl(descriptor, 0xAE00, 0) != 12:
                raise OSError('unsupported KVM API')
        finally:
            os.close(descriptor)
    except OSError as error:
        raise RuntimeError('Environment requires usable ' + device + ': ' + str(error)) from error

hostname = socket.gethostname()
try:
    socket.getaddrinfo(hostname, None)
except socket.gaierror:
    with Path('/etc/hosts').open('a') as hosts:
        hosts.write('\\n127.0.1.1 ' + hostname + '\\n')
    socket.getaddrinfo(hostname, None)

# Use the capabilities declared by the installed Snap package itself.
capabilities = Path('/usr/lib/snapd/snap-confine.caps')
if capabilities.is_file():
    subprocess.run(['setcap', '-q', '-', '/usr/lib/snapd/snap-confine'],
                   input=capabilities.read_text(), text=True, check=True)
"""


def prepare(runner, *, required_devices=(), native_image=None) -> None:
    """Restore desktop prerequisites on each fresh worker."""
    if not isinstance(required_devices, (list, tuple)) or any(
        not isinstance(device, str)
        or not PurePosixPath(device).is_relative_to('/dev')
        or '..' in PurePosixPath(device).parts
        for device in required_devices
    ):
        raise ValueError('required_devices must list absolute paths below /dev')
    if native_image:
        from gym_anything.runtime.runners.modal_native_image import MODAL_NATIVE_IMAGE_FINGERPRINT

        if native_image.get('provenance', {}).get('image_schema') != MODAL_NATIVE_IMAGE_FINGERPRINT:
            raise ValueError('published desktop image does not match the native runner')
        import modal

        base_image = modal.Image.from_registry(native_image['reference'])
        create_sandbox = runner._create_sandbox

        def create_with_prebuilt(image=None):
            return create_sandbox(image=image if image is not None else base_image)

        runner._create_sandbox = create_with_prebuilt
    wait_for_desktop = runner._wait_for_desktop

    def wait_with_runtime():
        # Modal supplies /etc/hosts afresh when restoring a filesystem image.
        # TigerVNC requires the restored guest's hostname to resolve locally.
        process = runner._sandbox.exec(
            'python3', '-c', DESKTOP_SETUP, json.dumps(required_devices), timeout=30
        )
        process.wait()
        if process.returncode:
            raise RuntimeError('Native desktop runtime setup failed: ' + process.stderr.read())
        return wait_for_desktop()

    runner._wait_for_desktop = wait_with_runtime
