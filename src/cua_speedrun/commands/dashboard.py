"""One-command dashboard and queue-worker supervisor."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

from .paths import InstallationPaths, add_home_argument, configure_process


def register_dashboard_command(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "dashboard", help="start the web dashboard and evaluation worker"
    )
    add_home_argument(parser)
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument(
        "--no-open", action="store_true", help="do not open a local web browser"
    )
    parser.add_argument(
        "--accept-topology",
        default="all",
        help="worker topology filter (default: all registered topologies)",
    )
    parser.set_defaults(_operator_handler=run_dashboard)


def _terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except OSError:
            pass


def _http_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return response.status < 500
    except (OSError, urllib.error.URLError):
        return False


def _wait_for_api(process: subprocess.Popen, url: str, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"dashboard server exited with status {code}")
        if _http_ready(url):
            return
        time.sleep(0.2)
    raise RuntimeError(f"dashboard did not become ready within {timeout:.0f}s")


def _check_installation(paths: InstallationPaths) -> None:
    missing = []
    if not paths.install_record.is_file():
        missing.append(str(paths.install_record))
    if not (paths.resource_root / "catalog/tracks.yaml").is_file():
        missing.append(str(paths.resource_root))
    if missing:
        raise RuntimeError(
            "cua-speedrun is not installed in this home; run "
            f"`cua-speedrun install --home {paths.home}` first"
        )


def _check_port_available(host: str, port: int) -> None:
    try:
        addresses = socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
        )
    except socket.gaierror as exc:
        raise RuntimeError(f"cannot resolve bind address {host!r}: {exc}") from exc
    last_error: OSError | None = None
    for family, socktype, protocol, _, address in addresses:
        probe = socket.socket(family, socktype, protocol)
        try:
            probe.bind(address)
            return
        except OSError as exc:
            last_error = exc
        finally:
            probe.close()
    raise RuntimeError(f"cannot bind {host}:{port}: {last_error}")


def run_dashboard(args: argparse.Namespace) -> int:
    from cua_speedrun.service.worker import _raise_open_file_limit

    # Both children inherit the lifted limit: the API serves one SSE stream
    # per watcher and the worker fans out to every compute replica.
    _raise_open_file_limit()
    paths = InstallationPaths.resolve(args.home)
    api: subprocess.Popen | None = None
    worker: subprocess.Popen | None = None
    try:
        _check_installation(paths)
        _check_port_available(args.host, args.port)
        configure_process(paths)
        env = os.environ.copy()
        process_options = {
            "cwd": str(paths.home),
            "env": env,
            "start_new_session": True,
        }
        api_command = [
            sys.executable,
            "-m",
            "uvicorn",
            "cua_speedrun.service.api:app",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
        worker_command = [
            sys.executable,
            "-m",
            "cua_speedrun.service.worker",
            "--accept-topology",
            args.accept_topology,
        ]
        browser_host = (
            "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
        )
        url = f"http://{browser_host}:{args.port}"
        api = subprocess.Popen(api_command, **process_options)
        _wait_for_api(api, f"{url}/api/tracks")
        worker = subprocess.Popen(worker_command, **process_options)
        print(f"\nDashboard: {url}")
        print(f"Home:      {paths.home}")
        print(f"Worker:    {args.accept_topology}")
        print("Press Ctrl-C to stop both processes.\n")
        if not args.no_open:
            webbrowser.open(url)
        while True:
            api_code = api.poll()
            worker_code = worker.poll()
            if api_code is not None:
                raise RuntimeError(f"dashboard server exited with status {api_code}")
            if worker_code is not None:
                raise RuntimeError(f"evaluation worker exited with status {worker_code}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping cua-speedrun...")
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        _terminate(worker)
        _terminate(api)


__all__ = ["register_dashboard_command", "run_dashboard"]
