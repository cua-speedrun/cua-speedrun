#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pwd
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path


class SetupError(RuntimeError):
    pass


HF_CACHE = Path("/tmp/cua-speedrun-huggingface")
os.environ.setdefault("HF_XET_CACHE", str(HF_CACHE / "xet"))


def _discover_desktop_context() -> tuple[str, int, str, str, str]:
    """Resolve the real interactive desktop instead of assuming one image layout."""
    configured_user = os.environ.get("OSWORLD_DESKTOP_USER")
    account = None
    if configured_user:
        try:
            account = pwd.getpwnam(configured_user)
        except KeyError as exc:
            raise SetupError(f"configured desktop user does not exist: {configured_user}") from exc
    if account is None:
        candidates = [1000]
        if os.geteuid() != 0:
            candidates.insert(0, os.geteuid())
        for uid in candidates:
            try:
                candidate = pwd.getpwuid(uid)
            except KeyError:
                continue
            if candidate.pw_name != "root":
                account = candidate
                break
    if account is None:
        for name in ("user", "ga"):
            try:
                account = pwd.getpwnam(name)
                break
            except KeyError:
                continue
    if account is None:
        raise SetupError("could not discover a non-root desktop account")

    user = account.pw_name
    uid = account.pw_uid
    home = os.environ.get("OSWORLD_DESKTOP_HOME") or account.pw_dir
    display = os.environ.get("OSWORLD_X11_DISPLAY") or os.environ.get("DISPLAY")
    if not display:
        sockets = sorted(
            Path("/tmp/.X11-unix").glob("X*"),
            key=lambda path: int(path.name[1:]) if path.name[1:].isdigit() else 10**9,
        )
        display = f":{sockets[0].name[1:]}" if sockets else None
    if not display:
        raise SetupError("could not discover the guest X11 display")
    dbus = os.environ.get("OSWORLD_DBUS_ADDRESS") or f"unix:path=/run/user/{uid}/bus"
    return user, uid, home, display, dbus


USER, USER_ID, HOME, DISPLAY, DBUS = _discover_desktop_context()


def _map_path(text: str) -> str:
    return str(text).replace("/home/user", HOME).replace("/home/ga", HOME)


def _replace_vars(text: str) -> str:
    return (
        str(text)
        .replace("{CLIENT_PASSWORD}", "password")
        .replace("{SCREEN_WIDTH_HALF}", "960")
        .replace("{SCREEN_HEIGHT_HALF}", "540")
        .replace("{SCREEN_WIDTH}", "1920")
        .replace("{SCREEN_HEIGHT}", "1080")
    )


def _command_to_shell(command) -> str:
    if isinstance(command, list):
        return " ".join(shlex.quote(_map_path(_replace_vars(str(part)))) for part in command)
    return _map_path(_replace_vars(str(command)))


def _ga_env() -> list[str]:
    return [
        "sudo", "-u", USER, "env",
        f"HOME={HOME}",
        f"USER={USER}",
        f"LOGNAME={USER}",
        f"DISPLAY={DISPLAY}",
        f"DBUS_SESSION_BUS_ADDRESS={DBUS}",
    ]


def _run_as_ga(shell_cmd: str, *, timeout: float = 120, background: bool = False) -> int:
    shell_cmd = _map_path(_replace_vars(shell_cmd))
    if background:
        shell_cmd = (
            f"nohup bash -lc {shlex.quote(shell_cmd)} >/tmp/osworld-launch.log 2>&1 & "
            "pid=$!; sleep 1; kill -0 $pid 2>/dev/null && exit 0; "
            "wait $pid; rc=$?; cat /tmp/osworld-launch.log; exit $rc"
        )
        timeout = min(timeout, 10)
    print(f"[osworld setup] $ {shell_cmd}", flush=True)
    proc = subprocess.run(
        _ga_env() + ["bash", "-lc", shell_cmd],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.returncode:
        raise SetupError(f"desktop command exited {proc.returncode}: {shell_cmd}")
    return proc.returncode


def _run_root(shell_cmd: str, *, timeout: float = 120) -> int:
    shell_cmd = _map_path(_replace_vars(shell_cmd))
    proc = subprocess.run(
        ["bash", "-lc", shell_cmd],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if proc.stdout:
        print(proc.stdout, end="", flush=True)
    if proc.returncode:
        raise SetupError(f"root command exited {proc.returncode}: {shell_cmd}")
    return proc.returncode


def _pyautogui_to_xdotool(command) -> bool:
    if not (isinstance(command, list) and len(command) >= 3 and command[1] == "-c"):
        return False
    code = str(command[2])
    if "pyautogui" not in code:
        return False
    if "pyautogui.click" in code:
        _run_as_ga(
            'read W H <<< "$(xdotool getdisplaygeometry)"; '
            'xdotool mousemove $((W / 2)) $((H / 2)) click 1; sleep 0.5'
        )
        return True
    match = re.search(r"pyautogui\.hotkey\((.*?)\)", code)
    if match:
        keys = re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))
        if keys:
            _run_as_ga("xdotool key " + "+".join(keys))
            return True
    return False


def _validate_download(path: Path, destination: str) -> None:
    with path.open("rb") as stream:
        prefix = stream.read(256)
    if not prefix:
        raise SetupError(f"downloaded file is empty: {destination}")
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise SetupError(f"download resolved to a Git LFS pointer: {destination}")

    lowered = destination.lower().split("?", 1)[0]
    zip_suffixes = (
        ".zip", ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp",
    )
    if lowered.endswith(zip_suffixes) and not prefix.startswith(
        (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
    ):
        raise SetupError(f"downloaded file is not a ZIP archive: {destination}")
    if lowered.endswith((".gz", ".tgz")) and not prefix.startswith(b"\x1f\x8b"):
        raise SetupError(f"downloaded file is not gzip data: {destination}")
    if lowered.endswith(".pdf") and not prefix.startswith(b"%PDF-"):
        raise SetupError(f"downloaded file is not a PDF: {destination}")


def _huggingface_dataset_file(url: str) -> tuple[str, str, str] | None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.hostname not in {"huggingface.co", "www.huggingface.co"}:
        return None
    parts = parsed.path.lstrip("/").split("/")
    if len(parts) < 6 or parts[0] != "datasets" or parts[3] != "resolve":
        return None
    repo_id = "/".join(urllib.parse.unquote(part) for part in parts[1:3])
    revision = urllib.parse.unquote(parts[4])
    filename = urllib.parse.unquote("/".join(parts[5:]))
    return repo_id, revision, filename


def _fetch(url: str, destination: Path) -> None:
    huggingface_file = _huggingface_dataset_file(url)
    if huggingface_file is not None:
        try:
            import hf_xet  # noqa: F401
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SetupError(
                "Hugging Face downloads require huggingface_hub and hf_xet "
                "in the OSWorld environment image"
            ) from exc
        repo_id, revision, filename = huggingface_file
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            cache_dir=HF_CACHE / "hub",
            token=False,
        )
        shutil.copyfile(cached, destination)
        return

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "cua-speedrun-osworld-setup/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def _download(files: list[dict]) -> None:
    for item in files:
        url = str(item["url"])
        destination = Path(_map_path(item["path"]))
        if not destination.is_absolute():
            destination = Path(HOME) / destination
        created_directories = []
        parent = destination.parent
        while not parent.exists():
            created_directories.append(parent)
            parent = parent.parent
        destination.parent.mkdir(parents=True, exist_ok=True)
        for directory in created_directories:
            os.chown(directory, USER_ID, pwd.getpwnam(USER).pw_gid)
        temporary = destination.with_name(
            f".{destination.name}.download-{os.getpid()}"
        )
        print(f"[osworld setup] download {url} -> {destination}", flush=True)

        last_error: Exception | None = None
        for attempt in range(5):
            temporary.unlink(missing_ok=True)
            try:
                _fetch(url, temporary)
                _validate_download(temporary, str(destination))
                os.chown(temporary, USER_ID, pwd.getpwnam(USER).pw_gid)
                temporary.replace(destination)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                temporary.unlink(missing_ok=True)
                if attempt < 4:
                    delay = 2 ** attempt
                    print(
                        f"[osworld setup] download attempt {attempt + 1} failed: "
                        f"{exc}; retrying in {delay}s",
                        flush=True,
                    )
                    time.sleep(delay)
        if last_error is not None:
            raise SetupError(
                f"download failed after 5 attempts: {url}: {last_error}"
            ) from last_error


def _wait_for_chrome(port: int = 9222, timeout: float = 60) -> int | None:
    deadline = time.time() + timeout
    ports = [port, 1337] if port != 1337 else [1337, 9222]
    while time.time() < deadline:
        for candidate in ports:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{candidate}/json/version", timeout=2):
                    return candidate
            except Exception:
                pass
        time.sleep(1)
    return None


def _chrome_json(port: int, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as response:
        data = response.read().decode()
    return json.loads(data) if data else None


def _chrome_open_tabs(urls_to_open: list[str]) -> None:
    port = _wait_for_chrome()
    if not port:
        print("[osworld setup] chrome CDP not available; falling back to google-chrome --new-tab", flush=True)
        for url in urls_to_open:
            _run_as_ga(f"google-chrome --new-tab {shlex.quote(url)}", background=True)
        return
    for idx, url in enumerate(urls_to_open):
        quoted = urllib.parse.quote(url, safe=":/?&=#%")
        try:
            _chrome_json(port, f"/json/new?{quoted}")
        except Exception:
            _run_as_ga(f"google-chrome --new-tab {shlex.quote(url)}", background=True)
        if idx == 0:
            try:
                tabs = _chrome_json(port, "/json") or []
                for tab in tabs:
                    if tab.get("url") in {"about:blank", "chrome://newtab/"}:
                        _chrome_json(port, f"/json/close/{tab.get('id')}")
            except Exception:
                pass


def _chrome_close_tabs(urls_to_close: list[str]) -> None:
    port = _wait_for_chrome(timeout=20)
    if not port:
        return
    try:
        tabs = _chrome_json(port, "/json") or []
    except Exception:
        return
    for tab in tabs:
        url = tab.get("url") or ""
        if any(url.rstrip("/") == target.rstrip("/") for target in urls_to_close):
            try:
                _chrome_json(port, f"/json/close/{tab.get('id')}")
            except Exception:
                pass


def _chrome_history_path() -> str:
    return f"{HOME}/.config/google-chrome/Default/History"


def _update_browse_history(history: list[dict]) -> None:
    db_url = "https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/chrome/44ee5668-ecd5-4366-a6ce-c1c9b8d4e938/history_empty.sqlite?download=true"
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "history.sqlite"
        _download([{"url": db_url, "path": str(db_path)}])
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            epoch_start = datetime(1601, 1, 1)
            for history_item in history:
                visit_time = datetime.now() - timedelta(seconds=history_item["visit_time_from_now_in_seconds"])
                chrome_timestamp = int((visit_time - epoch_start).total_seconds() * 1000000)
                cursor.execute(
                    """
                    INSERT INTO urls (url, title, visit_count, typed_count, last_visit_time, hidden)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (history_item["url"], history_item["title"], 1, 0, chrome_timestamp, 0),
                )
                url_id = cursor.lastrowid
                cursor.execute(
                    """
                    INSERT INTO visits (url, visit_time, from_visit, transition, segment_id, visit_duration)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (url_id, chrome_timestamp, 0, 805306368, 0, 0),
                )
            conn.commit()
        finally:
            conn.close()

        dest = _chrome_history_path()
        _run_root(f"mkdir -p {shlex.quote(str(Path(dest).parent))}")
        _run_root(
            "pkill -x chrome || true; "
            "pkill -x google-chrome || true; "
            "pkill -x google-chrome-stable || true; "
            "pkill -x chromium || true; "
            "pkill -x chromium-browser || true; "
            "for _ in $(seq 1 50); do "
            "pgrep -x 'chrome|google-chrome|google-chrome-stable|chromium|chromium-browser' "
            ">/dev/null || exit 0; sleep 0.1; done; exit 1",
            timeout=10,
        )
        _run_root(f"cp {shlex.quote(str(db_path))} {shlex.quote(dest)} && chown {USER}:{USER} {shlex.quote(dest)}")


def _apply_task_patch(source_path: Path, source: dict) -> None:
    patch_path = source_path.with_name("setup-patch.json")
    if not patch_path.is_file():
        return
    patch = json.loads(patch_path.read_text())
    if (
        patch.get("schema_version") != 1
        or patch.get("task_name") != source_path.parent.name
        or not isinstance(patch.get("operations"), list)
    ):
        raise SetupError(f"invalid task patch: {patch_path}")
    config = source.get("config") or []
    if not isinstance(config, list):
        raise SetupError(f"task config is not a list: {source_path}")
    for operation in patch["operations"]:
        if not isinstance(operation, dict):
            raise SetupError(f"invalid task patch operation: {patch_path}")
        try:
            index = int(operation["index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SetupError(f"invalid task patch index: {patch_path}") from exc
        if not 0 <= index < len(config):
            raise SetupError(f"task patch index {index} is out of range: {patch_path}")
        if operation.get("op") == "replace" and isinstance(
            operation.get("value"), dict
        ):
            config[index] = operation["value"]
        elif operation.get("op") == "remove":
            config.pop(index)
        else:
            raise SetupError(f"invalid task patch operation: {patch_path}")
    source["config"] = config


def _run_item(item: dict) -> None:
    typ = item.get("type")
    params = item.get("parameters") or {}
    if typ in ("execute", "command"):
        command = params.get("command", "")
        if not _pyautogui_to_xdotool(command):
            _run_as_ga(_command_to_shell(command), timeout=180)
    elif typ == "launch":
        _run_as_ga(_command_to_shell(params.get("command", "")), background=True)
    elif typ == "download":
        _download(params.get("files", []))
    elif typ == "open":
        path = shlex.quote(_map_path(params.get("path", "")))
        _run_as_ga(f"xdg-open {path}", background=True)
        time.sleep(2)
    elif typ == "sleep":
        time.sleep(float(params.get("seconds", 1)))
    elif typ == "activate_window":
        name = shlex.quote(params.get("window_name", ""))
        _run_as_ga(
            "for _ in $(seq 1 40); do "
            f"wid=$(xdotool search --name {name} 2>/dev/null | tail -n1); "
            "if [ -n \"$wid\" ]; then xdotool windowactivate --sync \"$wid\"; exit 0; fi; "
            "sleep 0.5; "
            "done; exit 1",
            timeout=25,
        )
    elif typ == "close_window":
        name = shlex.quote(params.get("window_name", ""))
        _run_as_ga(f"xdotool search --name {name} windowclose", timeout=10)
    elif typ == "chrome_open_tabs":
        _chrome_open_tabs(params.get("urls_to_open", []))
    elif typ == "chrome_close_tabs":
        _chrome_close_tabs(params.get("urls_to_close", []))
    elif typ == "update_browse_history":
        _update_browse_history(params.get("history", []))
    elif typ in {"googledrive", "login"}:
        print(f"[osworld setup] skipping {typ}: credentials/settings are external to this repo", flush=True)
    else:
        raise SetupError(f"unsupported config type: {typ}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SetupError("usage: osworld_setup.py SOURCE_JSON")
    print(
        f"[osworld setup] desktop user={USER} uid={USER_ID} home={HOME} "
        f"display={DISPLAY} dbus={DBUS}",
        flush=True,
    )
    source_path = Path(sys.argv[1])
    source = json.loads(source_path.read_text())
    _apply_task_patch(source_path, source)
    # Upstream's desktop server resolves relative task paths from the user's home.
    os.chdir(HOME)
    for index, item in enumerate(source.get("config") or []):
        try:
            _run_item(item)
        except Exception as exc:
            raise SetupError(
                f"config item {index} ({item.get('type')}) failed: {exc}"
            ) from exc


if __name__ == "__main__":
    main()
