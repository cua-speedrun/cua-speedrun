"""Modal-hosted evaluation controllers."""

import argparse
import re
import time

APP_NAME = "cua-speedrun-hosted"
VOLUME_NAME = "cua-speedrun-evaluations"
REMOTE_HOME = "/work/installation"
TERMINAL = frozenset({"card_ready", "failed", "rejected", "held", "cancelled"})


def initial_status(request):
    return {"run_id": request["run_id"], "host": "modal", "stage": "preparing",
            "submission": {"name": request["name"]},
            "benchmark": request["benchmark"], "topology": "modal-hosted",
            "progress": {"finished": 0, "total": request["task_count"], "passed": 0},
            "tasks": [],
            "elapsed_sec": time.time() - request["created_at"], "result": None}


def is_hosted(run_id) -> bool:
    return isinstance(run_id, str) and re.fullmatch(r"m-[0-9a-f]{16}", run_id) is not None


def evaluation_id(value: str):
    if is_hosted(value):
        return value
    try:
        number = int(value)
        if number > 0:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("use a numeric evaluation ID or m- followed by 16 hex digits")
