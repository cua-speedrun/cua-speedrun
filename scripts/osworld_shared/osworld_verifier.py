"""Attach CUA Speedrun's live desktop to OSWorld's pinned evaluator.

This module intentionally contains no OSWorld scoring logic.  It provides only
the transport needed to let the canonical OSWorld ``DesktopEnv`` inspect an
already-running environment, then calls ``DesktopEnv.evaluate`` unchanged.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import select
import shutil
import socket
import socketserver
import sys
import tarfile
import tempfile
import threading
import urllib.request
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator


OSWORLD_COMMIT = "315a7603173feadf1b8a85cbc006c93ffe1dc1a1"
OSWORLD_ARCHIVE_URL = (
    f"https://github.com/xlang-ai/OSWorld/archive/{OSWORLD_COMMIT}.tar.gz"
)
# SHA-256 of the pinned commit's desktop_env tree.  The digest includes each
# sorted relative path and its bytes; Python bytecode caches are excluded.
OSWORLD_DESKTOP_ENV_SHA256 = (
    "e2f16e2972e0f1048139ba49af6c02d2d84e72312a1520e232eeb1571a9dc693"
)


def _cached_osworld_root() -> Path:
    cache = os.environ.get("CUA_OSWORLD_CACHE")
    base = Path(cache).expanduser() if cache else Path.home() / ".cache" / "cua-speedrun" / "osworld"
    return base / OSWORLD_COMMIT


def _normalize_osworld_root(path: Path) -> Path | None:
    path = path.expanduser().resolve()
    if (path / "desktop_env" / "desktop_env.py").is_file():
        return path
    if path.name == "desktop_env" and (path / "desktop_env.py").is_file():
        return path.parent
    return None


def _desktop_env_digest(root: Path) -> str:
    desktop_env = root / "desktop_env"
    digest = hashlib.sha256()
    for path in sorted(desktop_env.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(desktop_env)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_osworld_root(path: Path) -> Path:
    root = _normalize_osworld_root(path)
    if root is None:
        raise RuntimeError(f"OSWorld root does not contain desktop_env: {path}")
    actual = _desktop_env_digest(root)
    if actual != OSWORLD_DESKTOP_ENV_SHA256:
        raise RuntimeError(
            "OSWorld desktop_env source does not match the benchmark pin: "
            f"expected {OSWORLD_DESKTOP_ENV_SHA256}, got {actual} at {root}"
        )
    return root


def _extract_desktop_env_archive(archive_path: Path, destination: Path) -> None:
    found = False
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if len(parts) < 2 or parts[1] != "desktop_env" or ".." in parts:
                continue
            target = destination.joinpath(*parts[1:])
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                source = archive.extractfile(member)
                if source is None:
                    continue
                found = True
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    if not found:
        raise RuntimeError("pinned OSWorld archive did not contain desktop_env")


def _fetch_pinned_osworld() -> Path:
    root = _cached_osworld_root()
    if root.exists():
        return _validate_osworld_root(root)
    if os.environ.get("CUA_OSWORLD_AUTO_FETCH", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        raise RuntimeError(
            f"pinned OSWorld source is absent at {root} and auto-fetch is disabled"
        )

    root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = root.parent / f".{root.name}.{uuid.uuid4().hex}.tmp"
    archive_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz")
    archive_path = Path(archive_handle.name)
    archive_handle.close()
    try:
        urllib.request.urlretrieve(OSWORLD_ARCHIVE_URL, archive_path)
        _extract_desktop_env_archive(archive_path, temporary_root)
        _validate_osworld_root(temporary_root)
        try:
            temporary_root.rename(root)
        except FileExistsError:
            # Another evaluator may have populated the same immutable pin.
            _validate_osworld_root(root)
            shutil.rmtree(temporary_root, ignore_errors=True)
        return root
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    finally:
        archive_path.unlink(missing_ok=True)


def _select_osworld_root() -> Path:
    for variable in ("CS_OSWORLD_ROOT", "OSWORLD_ROOT"):
        configured = os.environ.get(variable)
        if configured:
            return _validate_osworld_root(Path(configured))
    return _fetch_pinned_osworld()


def _assert_import_origin(root: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name != "desktop_env" and not name.startswith("desktop_env."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        try:
            Path(origin).resolve().relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"{name} was already imported from unpinned source: {origin}"
            ) from exc


def ensure_osworld_evaluators() -> Path:
    """Validate and load the complete evaluator runtime from the OSWorld pin."""
    root = _select_osworld_root()
    _assert_import_origin(root)
    root_text = str(root)
    if not sys.path or sys.path[0] != root_text:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    module = importlib.import_module("desktop_env.desktop_env")
    _assert_import_origin(root)
    origin = Path(module.__file__).resolve()
    try:
        origin.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"DesktopEnv loaded from unpinned source: {origin}") from exc
    return root


class _ForwardServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class _ForwardHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        transport = self.server.ssh_transport
        try:
            channel = transport.open_channel(
                "direct-tcpip",
                (self.server.remote_host, self.server.remote_port),
                self.request.getpeername(),
            )
        except Exception:
            return
        if channel is None:
            return
        try:
            while True:
                readable, _, _ = select.select([self.request, channel], [], [], 1.0)
                if self.request in readable:
                    data = self.request.recv(16384)
                    if not data:
                        break
                    channel.sendall(data)
                if channel in readable:
                    data = channel.recv(16384)
                    if not data:
                        break
                    self.request.sendall(data)
        finally:
            channel.close()


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _SSHTunnel:
    def __init__(self, env_info: dict[str, Any], remote_port: int):
        import paramiko

        ssh_port = int(env_info.get("ssh_port") or 0)
        if not ssh_port:
            raise RuntimeError(f"runner did not expose ssh_port for guest port {remote_port}")
        ssh_user = str(env_info.get("ssh_user") or "user")
        ssh_password = str(env_info.get("ssh_password") or "password")

        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            hostname="127.0.0.1",
            port=ssh_port,
            username=ssh_user,
            password=ssh_password,
            timeout=30,
            banner_timeout=30,
            auth_timeout=30,
            allow_agent=False,
            look_for_keys=False,
        )
        self.local_port = _free_local_port()
        self.server = _ForwardServer(("127.0.0.1", self.local_port), _ForwardHandler)
        self.server.ssh_transport = self.client.get_transport()
        self.server.remote_host = "127.0.0.1"
        self.server.remote_port = int(remote_port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        try:
            self.server.shutdown()
            self.server.server_close()
        finally:
            self.client.close()


@contextmanager
def _guest_ports(env_info: dict[str, Any]) -> Iterator[tuple[int, int, int]]:
    if not env_info.get("ssh_port"):
        yield 5000, 9222, 8080
        return
    with ExitStack() as stack:
        tunnels = []
        for remote_port in (5000, 9222, 8080):
            tunnel = _SSHTunnel(env_info, remote_port)
            stack.callback(tunnel.close)
            tunnels.append(tunnel)
        yield tuple(tunnel.local_port for tunnel in tunnels)


def _extract_action_history(traj: dict[str, Any]) -> list[Any]:
    history: list[Any] = []

    def append(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                append(item)
        elif value is not None:
            history.append(value)

    for step in traj.get("steps") or []:
        if "action" in step:
            append(step["action"])
        elif "actions" in step:
            append(step["actions"])
    return history


def _attached_desktop_env(
    source: dict[str, Any],
    traj: dict[str, Any],
    cache_base: Path,
    ports: tuple[int, int, int],
) -> Any:
    from desktop_env.controllers.python import PythonController
    from desktop_env.controllers.setup import SetupController
    from desktop_env.desktop_env import DesktopEnv

    server_port, chromium_port, vlc_port = ports
    env = DesktopEnv.__new__(DesktopEnv)
    env.vm_ip = "127.0.0.1"
    env.server_port = server_port
    env.chromium_port = chromium_port
    env.vlc_port = vlc_port
    env.vnc_port = 8006
    env.cache_dir_base = str(cache_base)
    env.enable_proxy = False
    env.current_use_proxy = False
    env.client_password = "password"
    env.screen_width = 1920
    env.screen_height = 1080
    env.is_environment_used = False
    env.action_history = _extract_action_history(traj)
    env.controller = PythonController(env.vm_ip, env.server_port)
    env.setup_controller = SetupController(
        vm_ip=env.vm_ip,
        server_port=env.server_port,
        chromium_port=env.chromium_port,
        vlc_port=env.vlc_port,
        cache_dir=env.cache_dir_base,
        client_password=env.client_password,
        screen_width=env.screen_width,
        screen_height=env.screen_height,
    )
    DesktopEnv._set_task_info(env, source)
    env.setup_controller.reset_cache_dir(env.cache_dir)
    return env


def _evaluate(
    source: dict[str, Any], traj: dict[str, Any], env_info: dict[str, Any]
) -> float:
    ensure_osworld_evaluators()
    from desktop_env.desktop_env import DesktopEnv

    episode_dir = Path(env_info.get("episode_dir") or tempfile.mkdtemp())
    cache_base = episode_dir / "osworld_cache"
    cache_base.mkdir(parents=True, exist_ok=True)
    with _guest_ports(env_info) as ports:
        env = _attached_desktop_env(source, traj, cache_base, ports)
        return float(DesktopEnv.evaluate(env))


def check_with_source(
    source_json: Path,
    traj: dict[str, Any],
    env_info: dict[str, Any],
    task_info: dict[str, Any],
) -> dict[str, Any]:
    del task_info
    source = json.loads(Path(source_json).read_text())
    raw_score = _evaluate(source, traj, env_info)
    return {
        "passed": raw_score == 1.0,
        "score": raw_score * 100.0,
        "feedback": (
            f"OSWorld {source.get('id')} canonical reward={raw_score} "
            f"at commit {OSWORLD_COMMIT}"
        ),
    }
