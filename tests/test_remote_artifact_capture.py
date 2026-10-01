"""Artifact capture out of Modal sandboxes must be verified and loud.

These tests cover chunked, length-verified transfer of environment artifacts
and agent logs, including truncated transfers.
"""

from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path

from cua_speedrun.remote import modal_agent, modal_env


class _FakeProc:
    def __init__(self, rc: int, out: str):
        self.returncode = rc
        self.stdout = io.StringIO(out)

    def wait(self) -> None:
        return None


class _FakeSandbox:
    """Scripted exec: handlers is a list of (substring, callable) pairs;
    the first handler whose substring appears in the command answers it."""

    def __init__(self, handlers):
        self.handlers = handlers
        self.commands: list[str] = []

    def exec(self, *argv, timeout=None, env=None):
        cmd = argv[-1]
        self.commands.append(cmd)
        for fragment, handler in self.handlers:
            if fragment in cmd:
                rc, out = handler(cmd)
                return _FakeProc(rc, out)
        raise AssertionError(f"unscripted command: {cmd}")


class _FakeEnvSandbox:
    def __init__(self, sandbox):
        self.sandbox = sandbox


def _artifact_tar() -> bytes:
    import os

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, payload in (
            ("env_plane.log", b"[env_plane] booted\n"),
            # Incompressible so the gzipped archive spans several 512-byte
            # test chunks and the offset/retry arithmetic gets exercised.
            ("frame_00000.png", b"\x89PNG" + os.urandom(4096)),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def test_pull_env_artifacts_downloads_in_verified_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    tar_bytes = _artifact_tar()
    # Force several chunks so the offset arithmetic is exercised.
    monkeypatch.setattr(modal_env, "_ARTIFACT_CHUNK_BYTES", 512)
    truncated_once = {"done": False}

    def chunk_handler(cmd: str):
        offset = int(cmd.split("tail -c +")[1].split()[0]) - 1
        count = int(cmd.split("head -c ")[1].split()[0])
        data = tar_bytes[offset:offset + count]
        if offset == 512 and not truncated_once["done"]:
            # First read of the second chunk comes back short; the verified
            # loop must retry it rather than assemble a corrupt archive.
            truncated_once["done"] = True
            data = data[:100]
        return 0, base64.b64encode(data).decode()

    sandbox = _FakeSandbox([
        ("tar czf", lambda cmd: (0, "")),
        ("wc -c", lambda cmd: (0, f"{len(tar_bytes)}\n")),
        ("tail -c", chunk_handler),
        ("rm -f", lambda cmd: (0, "")),
    ])
    pulled, error = modal_env.pull_env_artifacts(
        _FakeEnvSandbox(sandbox), tmp_path
    )
    assert error is None
    assert sorted(pulled) == ["env_plane.log", "frame_00000.png"]
    assert (tmp_path / "env_plane.log").read_bytes() == b"[env_plane] booted\n"
    assert truncated_once["done"]


def test_pull_env_artifacts_reports_failures_instead_of_silence(
    tmp_path: Path,
) -> None:
    sandbox = _FakeSandbox([
        ("tar czf", lambda cmd: (1, "")),
    ])
    pulled, error = modal_env.pull_env_artifacts(
        _FakeEnvSandbox(sandbox), tmp_path
    )
    assert pulled == []
    assert error is not None and "archive" in error


def test_pull_env_artifacts_gives_up_on_a_persistently_short_chunk(
    tmp_path: Path,
) -> None:
    tar_bytes = _artifact_tar()
    sandbox = _FakeSandbox([
        ("tar czf", lambda cmd: (0, "")),
        ("wc -c", lambda cmd: (0, f"{len(tar_bytes)}\n")),
        ("tail -c", lambda cmd: (
            0, base64.b64encode(tar_bytes[:10]).decode()
        )),
    ])
    pulled, error = modal_env.pull_env_artifacts(
        _FakeEnvSandbox(sandbox), tmp_path
    )
    assert pulled == []
    assert error is not None and "bytes" in error


def test_read_remote_file_retries_a_truncated_read() -> None:
    content = "x" * 20000
    reads = {"n": 0}

    def cat_handler(cmd: str):
        reads["n"] += 1
        # First read reproduces the observed 8KB truncation; the second
        # returns the whole file.
        return 0, content[:8192] if reads["n"] == 1 else content

    sandbox = _FakeSandbox([
        ("wc -c", lambda cmd: (0, f"{len(content)}\n")),
        ("cat ", cat_handler),
    ])
    text, verified = modal_agent._read_remote_file(sandbox, "/tmp/f")
    assert verified is True
    assert text == content
    assert reads["n"] == 2


def test_read_remote_file_reports_unverified_when_truncation_persists() -> None:
    sandbox = _FakeSandbox([
        ("wc -c", lambda cmd: (0, "20000\n")),
        ("cat ", lambda cmd: (0, "y" * 8192)),
    ])
    text, verified = modal_agent._read_remote_file(sandbox, "/tmp/f")
    assert verified is False
    assert text == "y" * 8192
