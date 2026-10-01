#!/usr/bin/env python3
"""Pre-task hook for MyPCBench desktop tasks.

Ports MyPCBench's own reset protocol (agent-harness/env.py `reset` →
`_prewarm_lazy_dbs`) to run inside the guest. Each fresh environment instance
starts from the content-addressed image snapshot, so this hook reproduces what
the canonical runner does after its hard reset and before the agent's first
observation:

  1. Warm every seeded web app: hit its bootstrap endpoint with the signed
     session cookie AND assert the per-app sqlite DB is populated. An app
     counts as warmed only when BOTH gates pass; failures retry with backoff
     and then fail the task loudly (infra error) instead of letting the agent
     run against a dead app and score a silent 0. MYPCBENCH_REQUIRE_APPS=0
     downgrades to warn-and-proceed, mirroring upstream.
  2. Apply the dinoco-airlines fare_paid backfill (guarded UPDATEs, no-op on
     rows the agent may later mutate).
  3. Run the task's optional ``pre_command``.

Persona email resolution mirrors utils/persona_registry.resolve_persona_email:
SANDBOX_LOGIN_EMAIL env var, then /opt/personas/<persona>.json identity.email,
then the legacy <persona-with-dots>@sandbox.local fallback.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

SESSION_KEY = b"mypcbench-session-2026"
STACK_DEADLINE_SEC = 600  # firstboot can take minutes to bring the app stack up
SEED_FINALIZATION_DEADLINE_SEC = 600
SEED_FINALIZATION_UNIT = "mypcbench-canon-patch-post.service"
SEED_FINALIZATION_UNIT_FILE = Path(
    "/etc/systemd/system/mypcbench-canon-patch-post.service"
)
PER_APP_RETRIES = 3

# (db_name, app, port, path) — verbatim from agent-harness/env.py _LAZY_DB_WARMUPS.
LAZY_DB_WARMUPS = [
    ("vaultbank.sqlite", "vaultbank", 3001, "/api/accounts"),
    ("batbucks.sqlite", "batbucks", 3002, "/api/portfolio"),
    ("oddsmarket.sqlite", "oddsmarket", 3003, "/api/markets"),
    ("buzzchat.sqlite", "buzzchat", 3004, "/api/conversations"),
    ("workbuzz.sqlite", "workbuzz", 3005, "/api/channels"),
    ("etaxi.sqlite", "etaxi", 3006, "/api/rides"),
    ("hangrydash.sqlite", "hangrydash", 3007, "/api/orders"),
    ("tablefind.sqlite", "tablefind", 3008, "/api/reservations"),
    ("kwik-e-mart.sqlite", "kwik-e-mart", 3009, "/api/orders"),
    ("hoolishop.sqlite", "hoolishop", 3010, "/api/orders"),
    ("dinoco-airlines.sqlite", "dinoco-airlines", 3011, "/api/flights"),
    ("cheskepdia.sqlite", "cheskepdia", 3012, "/api/bookings"),
    ("sprintboard.sqlite", "sprintboard", 3013, "/api/projects"),
    ("lockedin.sqlite", "lockedin", 3014, "/api/posts"),
    ("speedtax.sqlite", "speedtax", 3015, "/api/returns"),
    ("mail.sqlite", "mail", 3016, "/api/emails"),
    (
        "hoolicalendar.sqlite",
        "hoolicalendar",
        3017,
        "/api/events?start=2026-01-01&end=2026-12-31",
    ),
]

# Per-app post-warmup assertion — verbatim from env.py _LAZY_DB_ASSERTION_TEMPLATES.
LAZY_DB_ASSERTION_TEMPLATES = {
    "vaultbank.sqlite": "SELECT 1 FROM accounts WHERE user_email='{email}' LIMIT 1",
    "batbucks.sqlite": "SELECT 1 FROM holdings WHERE user_email='{email}' LIMIT 1",
    "oddsmarket.sqlite": "SELECT 1 FROM positions WHERE user_email='{email}' LIMIT 1",
    "buzzchat.sqlite": "SELECT 1 FROM contacts WHERE user_email='{email}' LIMIT 1",
    "workbuzz.sqlite": "SELECT 1 FROM channels LIMIT 1",
    "etaxi.sqlite": "SELECT 1 FROM rides WHERE user_email='{email}' LIMIT 1",
    "hangrydash.sqlite": "SELECT 1 FROM orders WHERE user_email='{email}' AND placed_at IS NOT NULL LIMIT 1",
    "tablefind.sqlite": "SELECT 1 FROM reservations WHERE user_email='{email}' LIMIT 1",
    "kwik-e-mart.sqlite": "SELECT 1 FROM orders WHERE user_email='{email}' LIMIT 1",
    "hoolishop.sqlite": "SELECT 1 FROM orders WHERE user_email='{email}' LIMIT 1",
    "dinoco-airlines.sqlite": "SELECT 1 FROM flights WHERE user_email='{email}' LIMIT 1",
    "cheskepdia.sqlite": "SELECT 1 FROM bookings WHERE user_email='{email}' LIMIT 1",
    "sprintboard.sqlite": "SELECT 1 FROM projects LIMIT 1",
    "lockedin.sqlite": "SELECT 1 FROM posts LIMIT 1",
    "speedtax.sqlite": "SELECT 1 FROM tax_returns WHERE user_email='{email}' LIMIT 1",
    "mail.sqlite": "SELECT 1 FROM emails LIMIT 1",
    "hoolicalendar.sqlite": "SELECT 1 FROM events WHERE user_email='{email}' LIMIT 1",
}

DINOCO_BACKFILL = (
    "UPDATE flights SET fare_paid=512 WHERE flight_number='AA1482' AND fare_paid=0; "
    "UPDATE flights SET fare_paid=387 WHERE flight_number='DL1358' AND fare_paid=0; "
    "UPDATE flights SET fare_paid=289 WHERE flight_number='AS324' AND fare_paid=0;"
)


class SetupError(RuntimeError):
    pass


def _run(command: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-lc", command], capture_output=True, text=True, timeout=timeout
    )


def resolve_persona_email(persona: str) -> str:
    import os

    env_email = os.environ.get("SANDBOX_LOGIN_EMAIL")
    if env_email:
        return env_email
    try:
        with open(f"/opt/personas/{persona}.json", encoding="utf-8") as handle:
            email = json.load(handle).get("identity", {}).get("email")
        if email:
            return str(email)
    except Exception:
        pass
    return f"{persona.replace('_', '.')}@sandbox.local"


def _session_cookie(email: str, app: str) -> str:
    payload = json.dumps({"email": email, "app": app})
    signature = hmac.new(SESSION_KEY, payload.encode(), hashlib.sha256).hexdigest()
    return base64.b64encode(f"{payload}.{signature}".encode()).decode()


def _warmup_one_app(
    email: str, app: str, port: int, path: str, timeout: int = 60
) -> bool:
    request = urllib.request.Request(
        f"http://localhost:{port}{path}",
        headers={"Cookie": f"session_{app}={_session_cookie(email, app)}"},
    )
    try:
        urllib.request.urlopen(request, timeout=timeout).read()
        return True
    except Exception:
        return False


def _db_populated(email: str, db_name: str) -> bool:
    query = LAZY_DB_ASSERTION_TEMPLATES.get(db_name)
    if not query:
        return True
    result = _run(
        f"sqlite3 /data/{db_name} {shlex.quote(query.format(email=email) + ';')} 2>&1"
    )
    return result.stdout.strip() == "1"


def _wait_for_stack() -> None:
    """First gate: the first app port answering means the stack is coming up."""
    deadline = time.monotonic() + STACK_DEADLINE_SEC
    probe_port = LAZY_DB_WARMUPS[0][2]
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{probe_port}/", timeout=5).read()
            return
        except urllib.error.HTTPError:
            return  # server answered (any HTTP status = process is up)
        except Exception:
            time.sleep(3)
    raise SetupError(
        f"app stack not answering on port {probe_port} after {STACK_DEADLINE_SEC}s"
    )


def wait_for_seed_finalization() -> None:
    """Wait for the image's post-app canonicalization to reach a fixed point.

    MyPCBench deliberately runs this service as ``Type=simple`` so it does not
    hold up the graphical target.  A desktop can therefore be visible while
    the service is still importing Maildir rows and repairing cross-app data.
    The canonical QEMU image owns the service; older images without it keep
    their pre-existing reset behavior.
    """
    if not SEED_FINALIZATION_UNIT_FILE.is_file():
        return
    command = (
        f"systemctl show {SEED_FINALIZATION_UNIT} "
        "--property=ActiveState,SubState,Result,ExecMainStatus,"
        "ExecMainStartTimestampMonotonic"
    )
    deadline = time.monotonic() + SEED_FINALIZATION_DEADLINE_SEC
    last = ""
    while time.monotonic() < deadline:
        status = _run(command, timeout=10)
        if status.returncode != 0:
            raise SetupError(
                f"could not inspect {SEED_FINALIZATION_UNIT}: {status.stderr[-1000:]}"
            )
        last = status.stdout
        values = dict(
            line.split("=", 1) for line in last.splitlines() if "=" in line
        )
        state = values.get("ActiveState")
        started = values.get("ExecMainStartTimestampMonotonic", "0") != "0"
        if state == "inactive" and started:
            if values.get("Result") == "success" and values.get(
                "ExecMainStatus"
            ) == "0":
                print("mypcbench seed finalization complete")
                return
            raise SetupError(
                f"{SEED_FINALIZATION_UNIT} did not complete successfully: {last}"
            )
        if state == "failed":
            raise SetupError(f"{SEED_FINALIZATION_UNIT} failed: {last}")
        time.sleep(1)
    raise SetupError(
        f"{SEED_FINALIZATION_UNIT} did not finish after "
        f"{SEED_FINALIZATION_DEADLINE_SEC}s: {last}"
    )


def prewarm_lazy_dbs(email: str) -> None:
    import os

    require = os.environ.get("MYPCBENCH_REQUIRE_APPS", "1") not in ("0", "false", "no")
    failed: list[str] = []
    for db_name, app, port, path in LAZY_DB_WARMUPS:
        ok = False
        for attempt in range(PER_APP_RETRIES):
            if _warmup_one_app(email, app, port, path) and _db_populated(
                email, db_name
            ):
                ok = True
                break
            time.sleep(2 + attempt * 3)
        if not ok:
            failed.append(app)
            print(
                f"warmup failed after {PER_APP_RETRIES} retries: {app}", file=sys.stderr
            )
    if failed and require:
        raise SetupError(
            f"App(s) failed warmup/health: {failed}. Aborting before the agent "
            "runs so the task errors instead of scoring a dead VM to a silent 0. "
            "Set MYPCBENCH_REQUIRE_APPS=0 to warn-and-proceed instead."
        )
    _run(
        f"sqlite3 /data/dinoco-airlines.sqlite {shlex.quote(DINOCO_BACKFILL)} 2>/dev/null; true"
    )


def wait_for_native_browser() -> None:
    """Gate the native image's delayed Firefox autostart before timing begins."""
    from pathlib import Path

    if not Path("/usr/local/bin/cua-wait-mypcbench-apps").is_file():
        return
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        windows = _run(
            "sudo -u user env DISPLAY=:0 "
            "XAUTHORITY=/run/user/1000/gdm/Xauthority "
            "xdotool search --onlyvisible --class firefox",
            timeout=5,
        )
        for window_id in windows.stdout.split():
            if not window_id.isdigit():
                continue
            title = _run(
                "sudo -u user env DISPLAY=:0 "
                "XAUTHORITY=/run/user/1000/gdm/Xauthority "
                f"xdotool getwindowname {window_id}",
                timeout=5,
            ).stdout.strip()
            if "Firefox" in title and "Problem loading page" not in title:
                _run(
                    "sudo -u user env DISPLAY=:0 "
                    "XAUTHORITY=/run/user/1000/gdm/Xauthority "
                    f"xdotool windowactivate --sync {window_id}",
                    timeout=5,
                )
                print("mypcbench browser ready:", title)
                return
        time.sleep(1)
    raise SetupError(
        "native Firefox did not reach a loaded startup page after app warmup"
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SetupError(f"usage: {sys.argv[0]} <source.json>")
    with open(sys.argv[1], encoding="utf-8") as handle:
        source = json.load(handle)

    email = resolve_persona_email(str(source.get("persona") or "michael_scott"))
    _wait_for_stack()
    wait_for_seed_finalization()
    prewarm_lazy_dbs(email)
    wait_for_native_browser()

    pre_command = str(source.get("pre_command") or "").strip()
    if pre_command:
        proc = _run(pre_command, timeout=300)
        if proc.returncode != 0:
            raise SetupError(
                f"pre_command failed ({proc.returncode}): "
                f"{shlex.quote(pre_command)}\n{proc.stderr[-2000:]}"
            )
    print("mypcbench setup ok: 17 apps warmed, persona", email)


if __name__ == "__main__":
    main()
