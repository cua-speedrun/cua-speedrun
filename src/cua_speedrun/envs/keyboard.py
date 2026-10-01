"""One guest keyboard implementation shared by the Linux desktop runners."""

import base64
from functools import lru_cache
import json
import os
from pathlib import Path
import time
from types import MethodType


@lru_cache(maxsize=1)
def _guest_source():
    return Path(__file__).with_name("_pyautogui_keyboard.py").read_text(encoding="utf-8")


def keyboard_script(keyboard):
    # Data never becomes executable Python or shell source. The QEMU runner
    # also base64-encodes the whole script before sending it through SSH.
    data = base64.b64encode(json.dumps(keyboard, ensure_ascii=True).encode()).decode("ascii")
    # Retain the existing runner's keymap propagation setting.
    settle_ms = os.environ.get("GYM_ANYTHING_KBD_SETTLE_MS")
    settings = f"\nMAPPING_SETTLE_SEC = {float(settle_ms) / 1000.0!r}\n" if settle_ms else "\n"
    return (_guest_source() + settings
            + f"import base64, json\nrun_keyboard(json.loads(base64.b64decode({data!r})))\n")


def patch_runner_keyboard(runner):
    if (getattr(runner, "is_windows", False) or getattr(runner, "is_android", False)
            or getattr(runner, "_fast_io", False)):
        return
    if callable(getattr(runner, "_build_keyboard_script", None)):
        runner._build_keyboard_script = keyboard_script
        runner._cs_pyautogui_keyboard_patched = True
    elif callable(getattr(runner, "_vnc_connection", None)):
        if getattr(runner, "_cs_vnc_keyboard_patched", False):
            return
        get_connection = runner._vnc_connection

        def connection_with_text_input():
            connection = get_connection()
            connection.type_text = MethodType(_type_vnc_text, connection)
            return connection

        runner._vnc_connection = connection_with_text_input
        runner._cs_vnc_keyboard_patched = True


def _type_vnc_text(connection, text: str, delay: float = 0.02):
    # RFB KeyEvent carries the resulting character's keysym, including case
    # (RFC 6143, section 7.5.4). Shift plus a lowercase keysym still types lowercase.
    controls = {"\n": "Return", "\r": "Return", "\t": "Tab"}
    for char in text:
        key = controls.get(char, char)
        connection.send_key(key, down=True)
        time.sleep(delay / 2)
        connection.send_key(key, down=False)
        time.sleep(delay / 2)
