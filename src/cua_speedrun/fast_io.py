"""Image-aware observation wire format and screenshot delivery for the proxy.

Wire v2: four-byte big-endian metadata length, UTF-8 JSON with MIME type,
then original PNG or JPEG bytes. v1 was PNG-only.
Metadata stays in the body (accessibility trees need not fit HTTP headers).
Binary clients receive this body verbatim; shell clients receive a local path
after the proxy saves the original image bytes. Neither path uses base64.
"""

from __future__ import annotations

import json
import struct
from typing import Any

from cua_speedrun.images import image_mime_type, image_suffix

CONTENT_TYPE = "application/vnd.cua-speedrun.observation.v2"
FILE_ACCEPT = "application/vnd.cua-speedrun.screenshot-file.v1+json"

# Reference curl client options for comparison and compatibility checks.
# -q MUST be first: user curlrc files must not introduce redirects or retries.
CURL_OPTIONS = (
    "-q", "--silent", "--show-error", "--fail-with-body",
    "--noproxy", "*", "--retry", "0", "--max-time", "120",
)


def shell_command(operation: str, actions: str | None = None) -> str:
    """The native loopback client imports no Python, image, or TLS libraries."""
    import shlex
    from pathlib import Path
    args = [str(Path(__file__).with_name("fast_io_client")), operation]
    if actions is not None:
        args.append(actions)
    return shlex.join(args)


def build_shell_client(output_dir=None):
    """Compile once during untimed init; cached snapshots retain the executable."""
    import hashlib
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import tempfile
    source = Path(__file__).with_name("fast_io_client.c")
    directory = source.parent if output_dir is None else Path(output_dir)
    binary = directory / "fast_io_client"
    marker = binary.with_suffix(".sha256")
    flags = ["-O2", "-std=c11", "-Wall", "-Wextra", "-Werror"]
    digest = hashlib.sha256(source.read_bytes() + repr(flags).encode()).hexdigest()
    if os.access(binary, os.X_OK) and marker.exists() and marker.read_text() == digest:
        return binary
    compiler = shutil.which("cc")
    if compiler is None:
        if os.geteuid() != 0 or shutil.which("apt-get") is None:
            raise RuntimeError("fast I/O initialization requires a C compiler")
        subprocess.check_call(["apt-get", "update"])
        subprocess.check_call(["apt-get", "install", "-y", "--no-install-recommends", "gcc", "libc6-dev"])
        compiler = shutil.which("cc")
    with tempfile.TemporaryDirectory(prefix="fastio-build-", dir=directory) as staging:
        built = Path(staging) / binary.name
        subprocess.run([compiler, *flags, str(source), "-o", str(built)], check=True, timeout=120)
        built.replace(binary)
    marker.write_text(digest)
    return binary


def encode_observation(image: bytes, metadata: dict[str, Any]) -> bytes:
    mime_type = image_mime_type(image) if image else None
    if "mime_type" in metadata and metadata["mime_type"] != mime_type:
        raise ValueError("observation MIME type does not match image bytes")
    metadata = {**metadata, "mime_type": mime_type}
    header = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode()
    return struct.pack("!I", len(header)) + header + image


def decode_observation(data: bytes) -> dict[str, Any]:
    if len(data) < 4:
        raise ValueError("truncated observation header")
    size = struct.unpack("!I", data[:4])[0]
    if size > len(data) - 4:
        raise ValueError("truncated observation metadata")
    metadata = json.loads(data[4:4 + size])
    if not isinstance(metadata, dict):
        raise ValueError("observation metadata must be an object")
    image = data[4 + size:]
    mime_type = image_mime_type(image) if image else None
    if metadata.get("mime_type") != mime_type:
        raise ValueError("observation MIME type does not match image bytes")
    if metadata.get("ok") and not image:
        raise ValueError("successful observation has no image")
    return {**metadata, "image": image}


def save_screenshot(output, payload: dict[str, Any]):
    """Write exact received bytes under their actual format's extension."""
    from pathlib import Path
    if not payload["image"]:
        return None
    path = Path(output).with_suffix(image_suffix(payload["mime_type"]))
    path.write_bytes(payload["image"])
    return path


class ScreenshotFiles:
    """Run-owned image files written by the existing long-lived agent proxy."""

    def __init__(self, parent=None):
        import itertools
        import tempfile
        from pathlib import Path
        self._directory = tempfile.TemporaryDirectory(prefix="cs_screenshots_", dir=parent)
        self.path = Path(self._directory.name).resolve()
        self._sequence = itertools.count()

    def receive(self, wire: bytes) -> dict[str, Any]:
        payload = decode_observation(wire)
        image = payload.pop("image")
        path = None
        if image:
            path = self.path / f"screen_{next(self._sequence):05d}{image_suffix(payload['mime_type'])}"
            # Unique names: never overwrite a frame an image tool is reading.
            # The path is returned only after the file is closed.
            with path.open("xb") as output:
                output.write(image)
        return {"screenshot": str(path) if path is not None else None, **payload}

    def close(self) -> None:
        self._directory.cleanup()


def main() -> None:
    """Legacy/debug client. Production uses the native client + proxy-owned files."""
    import argparse
    import os
    import time
    from pathlib import Path

    import requests

    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("observe", "step-observe"))
    parser.add_argument("--actions", help="JSON action list (step-observe only)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    headers = {"X-Gateway-Token": os.environ["CS_COMPUTER_TOKEN"], "Accept": CONTENT_TYPE}
    body = None
    if args.operation == "step-observe":
        actions = json.loads(args.actions or "[]")
        if not isinstance(actions, list):
            parser.error("--actions must be a JSON list")
        body = {"actions": actions}
    started = time.perf_counter()
    # No retry: a lost action response must never replay the action.
    with requests.Session() as session:
        response = session.request(
            "POST" if body is not None else "GET",
            os.environ["CS_COMPUTER_URL"].rstrip("/") + "/" + args.operation,
            headers=headers, json=body, timeout=120, allow_redirects=False,
        )
        received = time.perf_counter()
        response.raise_for_status()
        if 300 <= response.status_code < 400:
            raise ValueError("proxy redirected the request; action was not replayed")
        if response.headers.get("Content-Type") != CONTENT_TYPE:
            raise ValueError("proxy did not return a binary observation")
        payload = decode_observation(response.content)
    screenshot = save_screenshot(args.output, payload)
    payload.pop("image")
    # Readiness includes decoding and saving the file the image tool will open.
    payload.update(
        screenshot=str(screenshot) if screenshot else None,
        round_trip_ms=(received - started) * 1000,
        screenshot_ready_ms=(time.perf_counter() - started) * 1000,
        steps_remaining=response.headers.get("X-Steps-Remaining"),
        episode_done=response.headers.get("X-Episode-Done"),
    )
    print(json.dumps(payload))
    if not payload.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
