from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from cua_speedrun.commands import dashboard_client as dashboard_commands
from cua_speedrun.service import api
from cua_speedrun.service.db import (
    BenchmarkRow,
    Run,
    SubmissionRow,
    User,
    make_session_factory,
)
from cua_speedrun.service.run_export import write_run_archive


def _finished_evaluation(tmp_path: Path):
    session_factory = make_session_factory(f"sqlite:///{tmp_path / 'platform.db'}")
    run_dir = tmp_path / "runs" / "svc_7_abcdef"
    task_dir = run_dir / "tasks" / "osworld_demo" / "seed_42"
    task_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text('{"num_passed": 1}\n')
    (run_dir / "run_plan.json").write_text('{"schema_version": 6}\n')
    (task_dir / "frame_00000.png").write_bytes(b"not-a-real-png")
    (task_dir / "runlog.jsonl").write_text('{"event":"header"}\n')
    (task_dir / "agent.stdout").write_text("done\n")

    with session_factory() as session:
        user = User(handle="runner")
        benchmark = BenchmarkRow(name="demo", version="1", path="/unused")
        session.add_all([user, benchmark])
        session.flush()
        submission = SubmissionRow(
            user_id=user.id,
            name="agent",
            storage_ref="submission.zip",
            track="open-test",
        )
        session.add(submission)
        session.flush()
        run = Run(
            id=7,
            submission_id=submission.id,
            benchmark_id=benchmark.id,
            stage="card_ready",
            run_dir=str(run_dir),
            execution_plan={},
            task_seeds={},
        )
        session.add(run)
        session.commit()
        user_id = user.id
    return session_factory, user_id, run_dir


def test_run_archive_contains_complete_evidence_but_not_external_symlinks(
    tmp_path: Path,
) -> None:
    _session_factory, _user_id, run_dir = _finished_evaluation(tmp_path)
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("do not export\n")
    (run_dir / "outside-link.txt").symlink_to(outside)
    outside_dir = tmp_path / "outside-directory"
    outside_dir.mkdir()
    (outside_dir / "nested-secret.txt").write_text("do not export either\n")
    (run_dir / "outside-directory-link").symlink_to(outside_dir)

    output = tmp_path / "evaluation.zip"
    count = write_run_archive(run_dir, output)

    assert count == 5
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == [
            "svc_7_abcdef/result.json",
            "svc_7_abcdef/run_plan.json",
            "svc_7_abcdef/tasks/osworld_demo/seed_42/agent.stdout",
            "svc_7_abcdef/tasks/osworld_demo/seed_42/frame_00000.png",
            "svc_7_abcdef/tasks/osworld_demo/seed_42/runlog.jsonl",
        ]
        assert archive.read(
            "svc_7_abcdef/tasks/osworld_demo/seed_42/frame_00000.png"
        ) == b"not-a-real-png"
        assert "outside-link.txt" not in "\n".join(archive.namelist())
        assert "nested-secret.txt" not in "\n".join(archive.namelist())
        assert archive.getinfo(
            "svc_7_abcdef/tasks/osworld_demo/seed_42/frame_00000.png"
        ).compress_type == zipfile.ZIP_STORED
        assert archive.getinfo(
            "svc_7_abcdef/tasks/osworld_demo/seed_42/runlog.jsonl"
        ).compress_type == zipfile.ZIP_DEFLATED


def test_local_cli_export_defaults_to_current_directory(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    session_factory, user_id, _run_dir = _finished_evaluation(tmp_path)
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    service = SimpleNamespace(
        session_factory=session_factory,
        user_id=user_id,
        paths=SimpleNamespace(home=tmp_path),
        status=lambda _run_id: {"submission": {"name": "agent"}},
    )
    monkeypatch.chdir(export_dir)
    monkeypatch.setattr(dashboard_commands.getpass, "getuser", lambda: "researcher")
    monkeypatch.setattr(
        dashboard_commands, "_use_local_installation", lambda _a: True
    )
    monkeypatch.setattr(
        dashboard_commands.LocalEvaluations,
        "open",
        lambda _home: service,
    )
    args = argparse.Namespace(
        run_id=7,
        output=None,
        overwrite=False,
        home=None,
        dashboard=None,
    )

    assert dashboard_commands.run_export(args) == 0
    assert (export_dir / "researcher-agent-7.zip").is_file()
    assert "Exported evaluation 7" in capsys.readouterr().out


def test_default_export_filename_sanitizes_the_evaluation_name() -> None:
    assert dashboard_commands._export_filename(
        "  Gemini 3 Flash / Pareto:48 ☃  ",
        13,
        user="Ada Lovelace",
    ) == "Ada-Lovelace-Gemini-3-Flash-Pareto-48-13.zip"
    assert dashboard_commands._export_filename(
        "...", 13, user="..."
    ) == "user-evaluation-13.zip"


def test_remote_export_endpoint_requires_ownership_and_includes_frames(
    tmp_path: Path, monkeypatch
) -> None:
    session_factory, user_id, _run_dir = _finished_evaluation(tmp_path)
    with session_factory() as session:
        user = session.get(User, user_id)

    monkeypatch.setattr(api, "session_factory", session_factory)
    api.app.dependency_overrides[api.current_user] = lambda: user
    try:
        response = TestClient(api.app).get("/api/runs/7/export")
    finally:
        api.app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert 'filename="evaluation-7.zip"' in response.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "svc_7_abcdef/result.json" in archive.namelist()
        assert (
            "svc_7_abcdef/tasks/osworld_demo/seed_42/frame_00000.png"
            in archive.namelist()
        )

    with session_factory() as session:
        other = User(handle="other")
        session.add(other)
        session.commit()
        other_id = other.id
    api.app.dependency_overrides[api.current_user] = lambda: SimpleNamespace(id=other_id)
    try:
        forbidden = TestClient(api.app).get("/api/runs/7/export")
    finally:
        api.app.dependency_overrides.clear()
    assert forbidden.status_code == 403


def test_remote_cli_streams_export_to_requested_output(
    tmp_path: Path, monkeypatch
) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("svc_7_abcdef/result.json", "{}")

    class Response:
        def iter_content(self, chunk_size: int):
            assert chunk_size == 1024 * 1024
            yield payload.getvalue()

    class Client:
        def request(self, method: str, path: str, **kwargs):
            assert (method, path) == ("GET", "/api/runs/7/export")
            assert kwargs["stream"] is True
            return Response()

    monkeypatch.setattr(
        dashboard_commands, "_use_local_installation", lambda _a: False
    )
    monkeypatch.setattr(
        dashboard_commands,
        "dashboard_client",
        lambda _args: (None, Client()),
    )
    output = tmp_path / "shared.zip"
    args = argparse.Namespace(
        run_id=7,
        output=output,
        overwrite=False,
        home=None,
        dashboard="https://speedrun.example.org",
    )

    assert dashboard_commands.run_export(args) == 0
    assert output.read_bytes() == payload.getvalue()
