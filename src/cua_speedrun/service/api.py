"""The platform API: submissions, live status, cards, leaderboard.

One FastAPI app serves the CLI, the web UI, and any future client. JSON
under /api; the HTML pages mount on the same app and consume the
same routes. The API never executes user code: uploads are validated as
zip structure only and handed to the worker through the queue table.

Auth: signed-cookie browser sessions and 30-day signed CLI bearer sessions.
GitHub OAuth is used when CS_GITHUB_CLIENT_ID and CS_GITHUB_CLIENT_SECRET are
set; a dev login exists behind CS_DEV_LOGIN=1 for local development.

Run:  .venv/bin/uvicorn cua_speedrun.service.api:app --reload
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests as http_requests
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    StreamingResponse,
)
from itsdangerous import (
    BadSignature,
    SignatureExpired,
    URLSafeSerializer,
    URLSafeTimedSerializer,
)
from sqlalchemy import select
from starlette.background import BackgroundTask

import cua_speedrun
from cua_speedrun.config import load_dotenv
from cua_speedrun.eval_algorithms import list_eval_algorithms
from cua_speedrun.execution_placements import (
    list_execution_placements,
    list_execution_topologies,
)
from cua_speedrun.runtime_environment import (
    MAX_ENVIRONMENT_VARIABLES,
    normalize_environment_name,
    normalize_environment_value,
)
from cua_speedrun.service.db import (
    BenchmarkRow,
    Card,
    Entry,
    EventRow,
    Run,
    SavedEnvironmentVariable,
    SubmissionRow,
    Track,
    User,
    make_session_factory,
)
from cua_speedrun.service.evaluations import (
    EvaluationServiceError,
    TERMINAL_STAGES as TERMINAL_RUN_STAGES,
    cancel_evaluation,
    get_evaluation,
    list_evaluations,
    queue_evaluation,
    resume_evaluation,
    run_payload,
)
from cua_speedrun.service.catalog import sync_catalog
from cua_speedrun.service.store import LocalStore

load_dotenv()

SESSION_COOKIE = "cs_session"
_serializer = URLSafeSerializer(
    os.environ.get("CS_SECRET_KEY", "dev-only-not-secret"), salt="session"
)
_cli_serializer = URLSafeTimedSerializer(
    os.environ.get("CS_SECRET_KEY", "dev-only-not-secret"), salt="cli-session"
)
CLI_TOKEN_MAX_AGE_SEC = 30 * 86400

session_factory = make_session_factory()
store = LocalStore(Path(os.environ.get("CS_STORE_ROOT", "store")))

# Register the configured public benchmarks before serving pages.
# Compact sources are downloaded only when they are selected.
sync_catalog(session_factory)

app = FastAPI(title="cua-speedrun", version=cua_speedrun.__version__)


# -- sessions and auth --------------------------------------------------------

def _session_user_id(request: Request) -> int | None:
    raw = request.cookies.get(SESSION_COOKIE)
    if not raw:
        return None
    try:
        return int(_serializer.loads(raw)["user_id"])
    except (BadSignature, KeyError, ValueError):
        return None


def _cli_user_id(request: Request) -> int | None:
    authorization = request.headers.get("authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token.strip():
        return None
    try:
        payload = _cli_serializer.loads(
            token.strip(), max_age=CLI_TOKEN_MAX_AGE_SEC
        )
        return int(payload["user_id"])
    except (BadSignature, SignatureExpired, KeyError, TypeError, ValueError):
        return None


def current_user(request: Request) -> User:
    user_id = _cli_user_id(request) or _session_user_id(request)
    if user_id is None:
        raise HTTPException(401, "sign in first")
    with session_factory() as session:
        user = session.get(User, user_id)
        if user is None:
            raise HTTPException(401, "unknown user")
        return user


def _safe_next_url(value: str | None) -> str:
    value = (value or "/").strip()
    parsed = urlsplit(value)
    if not value.startswith("/") or value.startswith("//"):
        return "/"
    if parsed.scheme or parsed.netloc:
        return "/"
    return value


def _login_response(user_id: int, next_url: str = "/") -> RedirectResponse:
    resp = RedirectResponse(next_url, status_code=303)
    resp.set_cookie(
        SESSION_COOKIE,
        _serializer.dumps({"user_id": user_id}),
        httponly=True,
        samesite="lax",
        max_age=30 * 86400,
    )
    return resp


@app.get("/login/dev")
def login_dev(next: str = "/"):
    """Local development login. Enabled only with CS_DEV_LOGIN=1; the public
    deployment uses GitHub OAuth below."""
    if os.environ.get("CS_DEV_LOGIN") != "1":
        raise HTTPException(404)
    with session_factory() as session:
        user = session.execute(
            select(User).where(User.handle == "dev")
        ).scalar_one_or_none()
        if user is None:
            user = User(handle="dev")
            session.add(user)
            session.commit()
        return _login_response(user.id, _safe_next_url(next))


@app.get("/login/github")
def login_github(request: Request, invite: str = "", next: str = "/"):
    client_id = os.environ.get("CS_GITHUB_CLIENT_ID")
    if not client_id:
        raise HTTPException(503, "GitHub OAuth is not configured")
    state = secrets.token_urlsafe(16)
    redirect = str(request.url_for("github_callback"))
    resp = RedirectResponse(
        "https://github.com/login/oauth/authorize"
        f"?client_id={client_id}&redirect_uri={redirect}&state={state}"
    )
    resp.set_cookie("cs_oauth_state", state, httponly=True, max_age=600)
    resp.set_cookie(
        "cs_oauth_next",
        _safe_next_url(next),
        httponly=True,
        samesite="lax",
        max_age=600,
    )
    # Carry the invite code through the round trip so a first-time sign-in can
    # redeem it when gating is on. Existing users don't need one.
    resp.set_cookie("cs_invite", invite, httponly=True, max_age=600)
    return resp


@app.get("/auth/github/callback")
def github_callback(request: Request, code: str, state: str):
    if state != request.cookies.get("cs_oauth_state"):
        raise HTTPException(400, "bad oauth state")
    token_resp = http_requests.post(
        "https://github.com/login/oauth/access_token",
        headers={"Accept": "application/json"},
        data={
            "client_id": os.environ["CS_GITHUB_CLIENT_ID"],
            "client_secret": os.environ["CS_GITHUB_CLIENT_SECRET"],
            "code": code,
        },
        timeout=20,
    )
    token_resp.raise_for_status()
    access_token = token_resp.json()["access_token"]
    gh_user = http_requests.get(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20,
    ).json()
    with session_factory() as session:
        user = session.execute(
            select(User).where(User.github_id == str(gh_user["id"]))
        ).scalar_one_or_none()
        if user is None:
            # First-time account: enforce invite gating when it's on.
            from cua_speedrun.service.invites import claim, gating_on

            user = User(github_id=str(gh_user["id"]), handle=gh_user["login"],
                        email=gh_user.get("email"))
            session.add(user)
            session.flush()
            if gating_on():
                if not claim(session, request.cookies.get("cs_invite", ""), user.id):
                    session.rollback()
                    raise HTTPException(
                        403, "a valid invite code is required to create an "
                             "account for this season")
            session.commit()
        return _login_response(
            user.id, _safe_next_url(request.cookies.get("cs_oauth_next"))
        )


def _cli_callback_url(callback: str, state: str, token: str) -> str:
    parsed = urlsplit(callback)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/callback"
        or parsed.fragment
    ):
        raise HTTPException(
            400, "CLI callback must be an HTTP loopback /callback URL"
        )
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({"state": state, "token": token})
    return urlunsplit((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        urlencode(query),
        "",
    ))


@app.get("/cli/authorize", include_in_schema=False)
def authorize_cli(
    request: Request, state: str, callback: str | None = None
):
    """Authorize a local CLI through the user's existing browser session."""
    if not state or len(state) > 256:
        raise HTTPException(400, "invalid CLI login state")
    user_id = _session_user_id(request)
    if user_id is None:
        next_url = request.url.path + "?" + request.url.query
        login_path = (
            "/login/dev"
            if os.environ.get("CS_DEV_LOGIN") == "1"
            else "/login/github"
        )
        return RedirectResponse(
            login_path + "?" + urlencode({"next": next_url}), status_code=303
        )
    token = _cli_serializer.dumps({"user_id": user_id})
    if callback is None:
        return HTMLResponse(
            "<main style='font:16px system-ui;max-width:48rem;margin:4rem auto'>"
            "<h1>CLI authorization token</h1>"
            "<p>Paste this token into the waiting cua-speedrun command:</p>"
            f"<code style='overflow-wrap:anywhere'>{token}</code>"
            "</main>"
        )
    return RedirectResponse(
        _cli_callback_url(callback, state, token), status_code=303
    )


@app.get("/logout")
def logout():
    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


# -- catalog -------------------------------------------------------------------

@app.get("/api/tracks")
def list_tracks():
    with session_factory() as session:
        rows = session.execute(select(Track)).scalars().all()
        return [{"name": t.name, "gpu": t.gpu,
                 "eval_algorithm": t.eval_algorithm,
                 "agents_per_evaluation": t.agents_per_evaluation,
                 "reference_only": t.reference_only} for t in rows]


@app.get("/api/benchmarks")
def list_benchmarks():
    with session_factory() as session:
        rows = session.scalars(select(BenchmarkRow).where(BenchmarkRow.active)).all()
        return [{
            "id": b.id,
            "name": b.name,
            "version": b.version,
            "task_count": b.task_count,
        } for b in rows]


@app.get("/api/execution-placements")
def list_execution_placements_api():
    return [placement.to_dict() for placement in list_execution_placements()]


@app.get("/api/execution-topologies")
def list_execution_topologies_api():
    return [topology.to_dict() for topology in list_execution_topologies()]


@app.get("/api/eval-algorithms")
def list_eval_algorithms_api():
    return [algorithm.public_dict() for algorithm in list_eval_algorithms()]


@app.get("/api/templates")
def list_templates_api():
    from cua_speedrun.service.templates_catalog import list_templates

    return list_templates()


@app.get("/templates/{name}.zip", include_in_schema=False)
def download_template(name: str):
    """A starter submission the user can edit locally and re-upload."""
    from fastapi.responses import Response

    from cua_speedrun.service.templates_catalog import template_zip

    data = template_zip(name)
    if data is None:
        raise HTTPException(404, "unknown template")
    return Response(data, media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="{name}.zip"'})


# -- submissions ---------------------------------------------------------------


# -- user Modal credentials ----------------------------------------------------
# Runs execute in the SUBMITTING USER's Modal workspace, so platform access
# can never spend the operator's credits. The web tier holds a pasted token
# only inside this request; storage is encrypted (service/usersecrets.py)
# and the secret is never sent back to the browser.

@app.get("/api/me")
def get_me(user: User = Depends(current_user)):
    return {"id": user.id, "handle": user.handle}


@app.get("/api/me/modal-credentials")
def get_modal_credentials(user: User = Depends(current_user)):
    with session_factory() as session:
        u = session.get(User, user.id)
        configured = bool(u.modal_token_id and u.modal_token_secret_enc)
        return {"configured": configured,
                "token_id": u.modal_token_id if configured else None}


@app.post("/me/modal-credentials", include_in_schema=False)
def set_modal_credentials(token_id: str = Form(...),
                          token_secret: str = Form(...),
                          user: User = Depends(current_user)):
    from cua_speedrun.service.usersecrets import encrypt

    token_id = token_id.strip()
    token_secret = token_secret.strip()
    if not token_id.startswith("ak-") or not token_secret.startswith("as-"):
        raise HTTPException(
            400, "that does not look like a Modal token pair "
                 "(id starts with ak-, secret with as-)")
    with session_factory() as session:
        u = session.get(User, user.id)
        u.modal_token_id = token_id
        u.modal_token_secret_enc = encrypt(token_secret)
        session.commit()
    return RedirectResponse("/me", status_code=303)


@app.post("/me/modal-credentials/delete", include_in_schema=False)
def delete_modal_credentials(user: User = Depends(current_user)):
    with session_factory() as session:
        u = session.get(User, user.id)
        u.modal_token_id = None
        u.modal_token_secret_enc = None
        session.commit()
    return RedirectResponse("/me", status_code=303)


# -- user environment variables ----------------------------------------------

def _validate_environment_name(name: Any) -> str:
    try:
        return normalize_environment_name(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _validate_environment_value(value: Any) -> str:
    try:
        return normalize_environment_value(value)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _parse_saved_environment_names(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            400, "saved_environment_variables must be a JSON list"
        ) from exc
    if not isinstance(payload, list) or not all(
        isinstance(item, str) for item in payload
    ):
        raise HTTPException(
            400, "saved_environment_variables must be a JSON list of names"
        )
    if len(payload) > MAX_ENVIRONMENT_VARIABLES:
        raise HTTPException(400, "at most 64 environment variables may be selected")
    return sorted({_validate_environment_name(item) for item in payload})


def _parse_evaluation_environment(raw: str | None) -> list[tuple[str, str]]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            400, "evaluation_environment_variables must be a JSON list"
        ) from exc
    if not isinstance(payload, list):
        raise HTTPException(
            400, "evaluation_environment_variables must be a JSON list"
        )
    if len(payload) > MAX_ENVIRONMENT_VARIABLES:
        raise HTTPException(400, "at most 64 evaluation variables may be provided")
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict) or set(item) != {"name", "value"}:
            raise HTTPException(
                400,
                "each evaluation variable must contain exactly name and value",
            )
        name = _validate_environment_name(item["name"])
        if name in seen:
            raise HTTPException(400, f"evaluation variable {name} is duplicated")
        seen.add(name)
        result.append((name, _validate_environment_value(item["value"])))
    return result


@app.get("/api/me/environment-variables")
def get_environment_variables(user: User = Depends(current_user)):
    with session_factory() as session:
        names = session.scalars(
            select(SavedEnvironmentVariable.name)
            .where(SavedEnvironmentVariable.user_id == user.id)
            .order_by(SavedEnvironmentVariable.name)
        ).all()
    return {"names": list(names)}


@app.post("/me/environment-variables", include_in_schema=False)
def set_environment_variable(
    name: str = Form(...),
    value: str = Form(...),
    user: User = Depends(current_user),
):
    from cua_speedrun.service.usersecrets import encrypt

    name = _validate_environment_name(name)
    value = _validate_environment_value(value)
    with session_factory() as session:
        row = session.execute(
            select(SavedEnvironmentVariable).where(
                SavedEnvironmentVariable.user_id == user.id,
                SavedEnvironmentVariable.name == name,
            )
        ).scalar_one_or_none()
        if row is None:
            row = SavedEnvironmentVariable(
                user_id=user.id, name=name, value_enc=""
            )
            session.add(row)
        row.value_enc = encrypt(value)
        session.commit()
    return RedirectResponse("/me#environment-variables", status_code=303)


@app.post("/me/environment-variables/delete", include_in_schema=False)
def delete_environment_variable(
    name: str = Form(...),
    user: User = Depends(current_user),
):
    name = _validate_environment_name(name)
    with session_factory() as session:
        row = session.execute(
            select(SavedEnvironmentVariable).where(
                SavedEnvironmentVariable.user_id == user.id,
                SavedEnvironmentVariable.name == name,
            )
        ).scalar_one_or_none()
        if row is not None:
            session.delete(row)
            session.commit()
    return RedirectResponse("/me#environment-variables", status_code=303)


@app.post("/api/submissions")
async def create_submission(
    request: Request,
    name: str | None = Form(None),
    track: str | None = Form(None),
    benchmark_id: int | None = Form(None),
    compute_placement: str | None = Form(None),
    environment_placement: str | None = Form(None),
    allocate_gpu: bool = Form(True),
    gpu: str | None = Form(None),
    eval_algorithm: str | None = Form(None),
    parallel_evaluations: int = Form(1),
    file: UploadFile | None = File(None),
    template: str | None = Form(None),
    saved_environment_variables: str | None = Form(None),
    evaluation_environment_variables: str | None = Form(None),
    user: User = Depends(current_user),
):
    # Non-secret fields retain query-string fallback for older API clients.
    # Runtime values are intentionally accepted only from the request body so
    # they cannot land in browser history or access logs.
    name = (
        name if name is not None else request.query_params.get("name", "")
    ).strip()
    track = (
        track if track is not None else request.query_params.get("track", "")
    ).strip()
    compute_placement = (
        compute_placement
        if compute_placement is not None
        else request.query_params.get("compute_placement", "")
    ).strip()
    environment_placement = (
        environment_placement
        if environment_placement is not None
        else request.query_params.get("environment_placement", "")
    ).strip()
    template = (
        template
        if template is not None
        else request.query_params.get("template", "")
    ).strip()
    if benchmark_id is None:
        try:
            benchmark_id = int(request.query_params.get("benchmark_id", "0"))
        except ValueError as exc:
            raise HTTPException(400, "benchmark_id must be an integer") from exc
    if not name or len(name) > 120:
        raise HTTPException(400, "evaluation name must contain 1 to 120 characters")

    # A submission supplies user code only. Whether that code came from a
    # built-in starter or an uploaded ZIP, the separately selected track and
    # benchmark remain authoritative. A starter must never be able to move a
    # run onto a different maintainer-owned evaluation contract.
    if template:
        from cua_speedrun.service.templates_catalog import template_zip

        data = template_zip(template)
        if data is None:
            raise HTTPException(404, f"unknown template '{template}'")
    elif file is not None:
        data = await file.read()
    else:
        raise HTTPException(400, "provide either a zip file or a template name")
    selected_saved_variables = _parse_saved_environment_names(
        saved_environment_variables
    )
    evaluation_variables = _parse_evaluation_environment(
        evaluation_environment_variables
    )
    required_environment_names: list[str] = []
    if template:
        from cua_speedrun.service.templates_catalog import (
            template_required_environment_variables,
        )

        required_environment_names = template_required_environment_variables(
            template
        )
    try:
        return queue_evaluation(
            session_factory=session_factory,
            store=store,
            user_id=user.id,
            submission_zip=data,
            name=name,
            track_name=track,
            benchmark_id=benchmark_id,
            compute_placement=compute_placement,
            environment_placement=environment_placement,
            allocate_gpu=allocate_gpu,
            gpu=gpu,
            eval_algorithm=eval_algorithm,
            parallel_evaluations=parallel_evaluations,
            saved_environment_names=selected_saved_variables,
            evaluation_environment=dict(evaluation_variables),
            required_environment_names=required_environment_names,
        )
    except EvaluationServiceError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc


# -- run status ----------------------------------------------------------------


@app.get("/api/runs/{run_id}")
def get_run(run_id: int, user: User = Depends(current_user)):
    try:
        return get_evaluation(session_factory, run_id, user.id)
    except EvaluationServiceError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: int, user: User = Depends(current_user)):
    """Request cancellation of one owned queued or active evaluation."""
    try:
        return cancel_evaluation(session_factory, run_id, user.id)
    except EvaluationServiceError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc


@app.post("/api/runs/{run_id}/resume")
def resume_run(
    run_id: int,
    rerun_agent_failures: bool = False,
    user: User = Depends(current_user),
):
    """Resume selected instances on the same frozen evaluation record."""
    try:
        return resume_evaluation(
            session_factory,
            run_id,
            user.id,
            installation_root=Path(
                os.environ.get("CUA_SPEEDRUN_HOME", ".")
            ).expanduser().resolve(),
            rerun_agent_failures=rerun_agent_failures,
        )
    except EvaluationServiceError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc


@app.get("/api/runs/{run_id}/export")
def export_run(run_id: int, user: User = Depends(current_user)):
    """Download every recorded artifact for one owned, stopped evaluation."""
    from cua_speedrun.service.run_export import (
        evaluation_run_directory,
        remove_file,
        write_run_archive,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"cua-speedrun-evaluation-{run_id}-", suffix=".zip"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        run_dir = evaluation_run_directory(
            session_factory,
            run_id,
            user.id,
            installation_root=Path(
                os.environ.get("CUA_SPEEDRUN_HOME", ".")
            ).expanduser().resolve(),
        )
        write_run_archive(run_dir, temporary)
    except EvaluationServiceError as exc:
        temporary.unlink(missing_ok=True)
        raise HTTPException(exc.status_code, exc.detail) from exc
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return FileResponse(
        temporary,
        media_type="application/zip",
        filename=f"evaluation-{run_id}.zip",
        background=BackgroundTask(remove_file, temporary),
    )


@app.get("/api/runs/{run_id}/events")
def get_run_events(run_id: int, after: int = 0, user: User = Depends(current_user)):
    with session_factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(404)
        try:
            run_payload(session, run, user.id)  # ownership check
        except EvaluationServiceError as exc:
            raise HTTPException(exc.status_code, exc.detail) from exc
        rows = session.execute(
            select(EventRow).where(EventRow.run_id == run_id, EventRow.id > after)
            .order_by(EventRow.id)
        ).scalars().all()
        return [{"id": e.id, "ts": e.ts, "kind": e.kind, "task": e.task_key,
                 **e.payload} for e in rows]


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: int, request: Request, after: int = 0,
                     user: User = Depends(current_user)):
    """Server-sent events: pushes each new event row, plus a stage snapshot
    whenever the run's stage changes. Ends when the run reaches a terminal
    stage. `after` resumes past already-rendered history."""
    with session_factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(404)
        try:
            run_payload(session, run, user.id)
        except EvaluationServiceError as exc:
            raise HTTPException(exc.status_code, exc.detail) from exc

    async def generate():
        cursor = after
        last_stage = None
        while True:
            if await request.is_disconnected():
                return
            with session_factory() as session:
                run = session.get(Run, run_id)
                rows = session.execute(
                    select(EventRow)
                    .where(EventRow.run_id == run_id, EventRow.id > cursor)
                    .order_by(EventRow.id)
                ).scalars().all()
            for e in rows:
                cursor = e.id
                data = {"id": e.id, "ts": e.ts, "kind": e.kind,
                        "task": e.task_key, **e.payload}
                yield f"event: run\ndata: {json.dumps(data, default=str)}\n\n"
            if run.stage != last_stage:
                last_stage = run.stage
                yield f"event: stage\ndata: {json.dumps({'stage': run.stage})}\n\n"
            if run.stage in TERMINAL_RUN_STAGES:
                yield "event: end\ndata: {}\n\n"
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/me/runs")
def my_runs(
    active: bool = False,
    limit: int | None = None,
    user: User = Depends(current_user),
):
    try:
        return list_evaluations(
            session_factory,
            user.id,
            active_only=active,
            limit=limit,
        )
    except EvaluationServiceError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc


# -- cards and publishing ------------------------------------------------------

@app.get("/api/cards/{token}")
def get_card(token: str):
    """The private result. Anyone with the secret URL can view it, which is
    the Geekbench-style share model; publishing stays an explicit step."""
    with session_factory() as session:
        card = session.execute(
            select(Card).where(Card.token == token)
        ).scalar_one_or_none()
        if card is None:
            raise HTTPException(404)
        return {"card": card.data, "accepted_at": str(card.accepted_at or "")}


@app.post("/api/cards/{token}/publish")
def publish_card(token: str, entry_name: str, user: User = Depends(current_user)):
    with session_factory() as session:
        card = session.execute(
            select(Card).where(Card.token == token)
        ).scalar_one_or_none()
        if card is None:
            raise HTTPException(404)
        run = session.get(Run, card.run_id)
        submission = session.get(SubmissionRow, run.submission_id)
        if submission.user_id != user.id:
            raise HTTPException(403, "not your card")
        from cua_speedrun.service.publishing import (
            AlreadyPublishedError,
            SeasonFrozenError,
            publish_card_to_leaderboard,
        )

        try:
            entry = publish_card_to_leaderboard(session, card, entry_name)
        except AlreadyPublishedError as exc:
            raise HTTPException(409, str(exc)) from exc
        except SeasonFrozenError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"entry_id": entry.id, "season_key": entry.season_key}


# -- leaderboard ---------------------------------------------------------------

@app.get("/api/leaderboard")
def leaderboard(season: str | None = None):
    """Entries ranked within each season: qualifying entries first by total
    time, non-qualifying shown unranked, never compared across seasons."""
    with session_factory() as session:
        stmt = select(Entry)
        if season:
            stmt = stmt.where(Entry.season_key == season)
        entries = session.execute(stmt).scalars().all()
    by_season: dict[str, list[dict]] = {}
    for e in entries:
        by_season.setdefault(e.season_key, []).append(e.data)
    out = []
    for key, rows in by_season.items():
        rows.sort(key=lambda d: (
            bool(d.get("reference_only")),
            not d["meets_success_bar"],
            d["total_time_sec"],
        ))
        rank = 0
        ranked = []
        for d in rows:
            if d["meets_success_bar"] and not d.get("reference_only"):
                rank += 1
            ranked.append({
                **d,
                "rank": rank
                if d["meets_success_bar"] and not d.get("reference_only")
                else None,
            })
        out.append({"season_key": key, "entries": ranked})
    return out


@app.get("/api/health")
def health():
    return {"ok": True}


# HTML pages consume the same data over the same session factory.
from cua_speedrun.service.web import register_web  # noqa: E402

register_web(app, session_factory, _session_user_id)
