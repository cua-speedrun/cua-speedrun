"""Connection and browser-login support for dashboard CLI commands."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import stat
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import requests

from cua_speedrun.config import load_dotenv

from .paths import InstallationPaths, add_home_argument, configure_process


DEFAULT_DASHBOARD = "http://127.0.0.1:8000"


def add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    add_home_argument(parser)
    parser.add_argument(
        "--dashboard",
        help=(
            "target another installation over HTTP; omitted commands use "
            "the local CUA_SPEEDRUN_HOME when it is installed"
        ),
    )


def register_dashboard_auth_commands(
    subparsers: argparse._SubParsersAction,
) -> None:
    login = subparsers.add_parser(
        "login", help="authorize this CLI with a dashboard account"
    )
    add_connection_arguments(login)
    login.add_argument(
        "--no-open", action="store_true",
        help="print the login URL and paste its token back into this terminal",
    )
    login.add_argument(
        "--timeout", type=int, default=180, help="seconds to wait for browser login"
    )
    login.set_defaults(_operator_handler=run_login)

    logout = subparsers.add_parser(
        "logout", help="remove the dashboard login saved on this machine"
    )
    add_connection_arguments(logout)
    logout.set_defaults(_operator_handler=run_logout)


def configure_client_process(paths: InstallationPaths) -> None:
    if paths.install_record.is_file():
        configure_process(paths)
    else:
        load_dotenv()


def _auth_path(paths: InstallationPaths):
    return paths.home / "cli-auth.json"


def _load_auth(paths: InstallationPaths) -> dict[str, str]:
    path = _auth_path(paths)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        key: value for key, value in data.items()
        if key in {"dashboard", "token"} and isinstance(value, str)
    }


def has_saved_dashboard(paths: InstallationPaths) -> bool:
    return bool(_load_auth(paths).get("dashboard"))


def _save_auth(paths: InstallationPaths, data: dict[str, str]) -> None:
    paths.home.mkdir(parents=True, exist_ok=True)
    path = _auth_path(paths)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
    temporary.replace(path)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _dashboard_url(args: argparse.Namespace, auth: dict[str, str]) -> str:
    value = (
        args.dashboard
        or os.environ.get("CUA_SPEEDRUN_DASHBOARD")
        or auth.get("dashboard")
        or DEFAULT_DASHBOARD
    ).strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid dashboard URL: {value!r}")
    if parsed.scheme == "http" and parsed.hostname not in {
        "127.0.0.1", "localhost", "::1",
    }:
        raise ValueError(
            "refusing to send dashboard credentials over plain HTTP; use "
            "HTTPS or an SSH tunnel to a loopback URL"
        )
    return value


class DashboardClient:
    def __init__(self, dashboard: str, token: str | None = None):
        self.dashboard = dashboard
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def request(
        self,
        method: str,
        path: str,
        *,
        allow_dev_login: bool = False,
        **kwargs: Any,
    ) -> requests.Response:
        url = self.dashboard + path
        kwargs.setdefault("timeout", 60)
        try:
            response = self.session.request(method, url, **kwargs)
            if response.status_code == 401 and allow_dev_login:
                login = self.session.get(
                    self.dashboard + "/login/dev", timeout=15
                )
                if login.ok:
                    response = self.session.request(method, url, **kwargs)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"dashboard could not be reached at {self.dashboard}: {exc}"
            ) from exc
        if not response.ok:
            try:
                detail = response.json().get("detail")
            except (ValueError, AttributeError):
                detail = None
            if response.status_code == 401:
                detail = "not signed in; run `cua-speedrun login` first"
            raise RuntimeError(
                f"dashboard request failed ({response.status_code}): "
                f"{detail or response.text.strip() or response.reason}"
            )
        return response

    def json(self, method: str, path: str, **kwargs: Any) -> Any:
        return self.request(method, path, **kwargs).json()


def dashboard_client(
    args: argparse.Namespace, *, require_auth: bool = True
) -> tuple[InstallationPaths, DashboardClient]:
    paths = InstallationPaths.resolve(args.home)
    configure_client_process(paths)
    auth = _load_auth(paths)
    dashboard = _dashboard_url(args, auth)
    token = os.environ.get("CUA_SPEEDRUN_TOKEN") or auth.get("token")
    client = DashboardClient(dashboard, token)
    if require_auth:
        client.request("GET", "/api/me", allow_dev_login=not bool(token))
    return paths, client


def run_login(args: argparse.Namespace) -> int:
    paths = InstallationPaths.resolve(args.home)
    configure_client_process(paths)
    auth = _load_auth(paths)
    dashboard = _dashboard_url(args, auth)
    state = secrets.token_urlsafe(24)
    if args.no_open:
        authorization_url = dashboard + "/cli/authorize?" + urlencode({
            "state": state,
        })
        print(f"Authorize cua-speedrun at:\n{authorization_url}\n")
        try:
            token = input("Paste authorization token: ").strip()
        except EOFError as exc:
            raise RuntimeError("no authorization token was provided") from exc
        if not token:
            raise RuntimeError("no authorization token was provided")
        DashboardClient(dashboard, token).request("GET", "/api/me")
        _save_auth(paths, {"dashboard": dashboard, "token": token})
        print(f"Logged in to {dashboard}")
        return 0

    result: dict[str, str] = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
            parsed = urlsplit(self.path)
            values = parse_qs(parsed.query)
            token = (values.get("token") or [""])[0]
            observed_state = (values.get("state") or [""])[0]
            accepted = (
                parsed.path == "/callback"
                and observed_state == state
                and bool(token)
            )
            if accepted:
                result["token"] = token
            body = (
                b"<h1>cua-speedrun CLI authorized</h1>"
                b"<p>You may close this tab.</p>"
                if accepted else
                b"<h1>Authorization failed</h1><p>Return to the terminal.</p>"
            )
            self.send_response(200 if accepted else 400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: Any) -> None:
            return

    callback_server = HTTPServer(("127.0.0.1", 0), CallbackHandler)
    callback_server.timeout = 0.5
    callback = f"http://127.0.0.1:{callback_server.server_port}/callback"
    authorization_url = dashboard + "/cli/authorize?" + urlencode({
        "callback": callback,
        "state": state,
    })
    print(f"Authorize cua-speedrun at:\n{authorization_url}\n")
    webbrowser.open(authorization_url)
    deadline = time.monotonic() + max(1, args.timeout)
    try:
        while "token" not in result and time.monotonic() < deadline:
            callback_server.handle_request()
    finally:
        callback_server.server_close()
    token = result.get("token")
    if not token:
        raise RuntimeError("dashboard login timed out")
    DashboardClient(dashboard, token).request("GET", "/api/me")
    _save_auth(paths, {"dashboard": dashboard, "token": token})
    print(f"Logged in to {dashboard}")
    return 0


def run_logout(args: argparse.Namespace) -> int:
    paths = InstallationPaths.resolve(args.home)
    path = _auth_path(paths)
    path.unlink(missing_ok=True)
    print(f"Removed dashboard login from {path}")
    return 0


__all__ = [
    "DashboardClient",
    "add_connection_arguments",
    "dashboard_client",
    "has_saved_dashboard",
    "register_dashboard_auth_commands",
]
