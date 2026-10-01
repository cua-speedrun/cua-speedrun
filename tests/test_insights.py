from __future__ import annotations

import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cua_speedrun.service.db import (
    BenchmarkRow,
    Card,
    Entry,
    Run,
    RunTask,
    Season,
    SubmissionRow,
    Track,
    User,
    make_session_factory,
)
from cua_speedrun.service.insights import (
    build_comparison,
    clock_profile,
    result_contract,
    result_insight,
)
from cua_speedrun.service.web import register_web


SEASON = (
    "benchmark=demo@1 | harness=2 | backend=modal-remote | hardware=modal-L4 "
    "| track=open-l4 | algorithm=per-task-vllm@1 | contract=abc123"
)


def _entry(entry_id: int, time: float, success: float = 1.0) -> dict:
    return {
        "entry_id": entry_id,
        "entry_name": f"agent-{entry_id}",
        "season_key": SEASON,
        "track_name": "open-l4",
        "total_time_sec": time,
        "measured_time_sec": time,
        "median_task_time_sec": time,
        "success_rate": success,
        "meets_success_bar": success >= 0.9,
        "num_runs": 1,
        "num_passed": int(success >= 0.9),
        "agent_time_sec": time * 0.6,
        "env_time_sec": time * 0.4,
        "submission_fingerprint": f"fp-{entry_id}",
        "reference_only": False,
    }


def test_clock_profile_is_descriptive() -> None:
    profile = clock_profile(7, 3)
    assert profile["available"] is True
    assert profile["label"] == "agent-heavy"
    assert profile["agent_percent"] == 70.0
    assert clock_profile(None, None)["available"] is False


def test_result_contract_exposes_versioned_agent_runtime() -> None:
    contract = result_contract({
        "run_plan_hash": "hash",
        "run_plan": {
            "track": {"name": "open-l40s", "gpu": "L40S"},
            "benchmark": {"name": "osworld", "version": "0.1"},
            "agent_runtime": {
                "recipe": "modal-debian-slim-py311-ffmpeg5@1",
                "base_image": {
                    "builder": "modal.Image.debian_slim",
                    "observed_python_version": "3.11.12",
                },
                "system_packages": ["ffmpeg=7:5.1.9-0+deb12u1"],
                "python_packages": ["requests==2.34.2"],
                "modal_client_version": "1.5.1",
            },
            "execution": {
                "algorithm": "per-task-vllm@1",
                "topology": {
                    "key": "modal-remote",
                    "compute": {
                        "key": "modal", "mode": "remote", "provider": "modal",
                        "requires_user_credentials": True,
                    },
                    "environment": {
                        "key": "modal", "mode": "remote", "provider": "modal",
                        "requires_user_credentials": True,
                    },
                    "environment_backend": "gym-anything-modal",
                    "eval_algorithms": ["per-task-vllm@1"],
                    "requires_user_credentials": True,
                },
            },
            "scoring": {"success_bar": 0.9},
            "seeds": {"policy": "scored-without-replacement@1"},
        },
    })
    fields = {field["label"]: field["value"] for field in contract["fields"]}
    assert fields["Agent runtime"] == "modal-debian-slim-py311-ffmpeg5@1"
    assert fields["Model / agent"] == "Remote · Modal"
    assert fields["Environment VMs"] == "Remote · Modal"
    assert fields["System packages"] == "ffmpeg=7:5.1.9-0+deb12u1"
    assert fields["Modal client"] == "1.5.1"


def test_private_result_signal_distinguishes_frontier_equivalence_and_bar() -> None:
    peers = [_entry(1, 10.0)]
    moved = result_insight(
        _entry(-1, 9.0), peers, success_bar=0.9
    )
    assert moved["status"] == "moves-frontier"
    assert moved["frontier"] is True
    assert moved["time_gap_sec"] == -1.0

    equivalent = result_insight(
        _entry(-1, 10.0), peers, success_bar=0.9
    )
    assert equivalent["status"] == "equivalent"

    below = result_insight(
        _entry(-1, 5.0, 0.8), peers, success_bar=0.9
    )
    assert below["status"] == "below-bar"
    assert round(below["success_gap_pp"], 6) == 10.0


def test_comparison_only_emits_task_deltas_for_identical_seed_sets() -> None:
    entries = [_entry(1, 12.0), _entry(2, 10.0)]
    tasks = {
        1: [
            {"task_id": "matched", "seed": "7", "passed": True,
             "task_time_sec": 12.0, "agent_time_sec": 7.0,
             "env_time_sec": 5.0, "num_steps": 4},
            {"task_id": "unpaired", "seed": "101", "passed": True,
             "task_time_sec": 4.0, "agent_time_sec": 2.0,
             "env_time_sec": 2.0, "num_steps": 2},
        ],
        2: [
            {"task_id": "matched", "seed": "7", "passed": True,
             "task_time_sec": 10.0, "agent_time_sec": 6.0,
             "env_time_sec": 4.0, "num_steps": 3},
            {"task_id": "unpaired", "seed": "202", "passed": True,
             "task_time_sec": 3.0, "agent_time_sec": 1.0,
             "env_time_sec": 2.0, "num_steps": 1},
        ],
    }
    comparison = build_comparison(
        entries, tasks, entries, success_bar=0.9
    )
    assert comparison["baseline"]["entry_id"] == 2
    matched, unpaired = comparison["task_groups"]
    assert matched["seed_comparable"] is True
    assert matched["rows"][0]["cells"][0]["delta_sec"] == 2.0
    assert matched["rows"][0]["cells"][1]["baseline"] is True
    assert unpaired["seed_comparable"] is False
    assert all(cell["delta_sec"] is None for cell in unpaired["rows"][0]["cells"])


def test_comparison_and_inspector_routes_enforce_boundaries(tmp_path) -> None:
    session_factory = make_session_factory(f"sqlite:///{tmp_path / 'inspector.db'}")
    other_season = SEASON.replace("demo@1", "other@1").replace(
        "contract=abc123", "contract=def456"
    )
    plan = {
        "schema_version": 1,
        "track": {
            "name": "open-l4", "gpu": "L4", "network_policy": "gateway-only",
            "ranking_rule": "time-at-success-bar@1", "reference_only": False,
        },
        "benchmark": {"name": "demo", "version": "1", "task_ids": ["task-a"]},
        "harness": {"cua_speedrun_version": "2"},
        "backend": "modal-remote",
        "execution": {"algorithm": "per-task-vllm@1", "concurrency": 2,
                      "env_pool_size": 2, "runs_per_task": 1},
        "scoring": {"success_bar": 0.9, "failure_costs_timeout": False,
                    "aggregation": "total-time@1"},
        "seeds": {"policy": "scored-without-replacement@1"},
    }
    now = datetime.datetime.now(datetime.timezone.utc)
    with session_factory() as session:
        user = User(handle="runner")
        track = Track(name="open-l4", gpu="L4", success_bar=0.9)
        benchmark = BenchmarkRow(name="demo", version="1", path="benchmarks/demo")
        session.add_all([user, track, benchmark])
        session.flush()
        entry_ids = []
        for index, (name, season, seconds, seed) in enumerate((
            ("fast-agent", SEASON, 10.0, 7),
            ("careful-agent", SEASON, 12.0, 7),
            ("other-season", other_season, 9.0, 9),
        ), start=1):
            submission = SubmissionRow(
                user_id=user.id, name=name, storage_ref=f"{name}.zip",
                track="open-l4", status="complete",
            )
            session.add(submission)
            session.flush()
            run = Run(
                submission_id=submission.id, benchmark_id=benchmark.id,
                season_key=season, stage="card_ready", execution_plan=plan,
                plan_hash=f"hash-{index}", task_seeds={"task-a": [seed]},
                started_at=now, finished_at=now,
            )
            session.add(run)
            session.flush()
            data = {
                **_entry(index, seconds),
                "season_key": season,
                "entry_name": name,
                "submission_fingerprint": f"fingerprint-{index}",
                "run_plan": plan,
                "run_plan_hash": f"hash-{index}",
                "rules": {"success_bar": 0.9},
            }
            card = Card(
                run_id=run.id, token=f"token-{index}", data=data, accepted_at=now
            )
            session.add(card)
            session.flush()
            published = Entry(
                season_key=season, entry_name=name, card_id=card.id, data=data
            )
            session.add(published)
            session.add(RunTask(
                run_id=run.id, task_key=f"task-a/seed_{seed}", stage="done",
                passed=True, task_time_sec=seconds,
                agent_time_sec=seconds * 0.6, env_time_sec=seconds * 0.4,
                num_steps=3,
            ))
            session.flush()
            entry_ids.append(published.id)
        session.add_all([
            Season(key=SEASON, status="open", contract_hash="a" * 64,
                   spec=plan),
            Season(key=other_season, status="open", contract_hash="b" * 64,
                   spec=plan),
        ])
        session.commit()
        user_id = user.id
        run_id = 1

    app = FastAPI()
    register_web(app, session_factory, lambda _request: user_id)
    client = TestClient(app)

    response = client.get(f"/compare?entry={entry_ids[0]}&entry={entry_ids[1]}")
    assert response.status_code == 200
    assert "<h1>Compare results</h1>" in response.text
    assert "seed matched" in response.text
    assert "fast-agent" in response.text

    boundary = client.get(f"/compare?entry={entry_ids[0]}&entry={entry_ids[2]}")
    assert boundary.status_code == 400
    assert "cross a track or season boundary" in boundary.text

    assert client.get(f"/compare?entry={entry_ids[0]}").status_code == 400
    assert "Frontier status" in client.get(f"/entries/{entry_ids[0]}").text
    assert "Clock attribution" in client.get("/cards/token-1").text
    run_page = client.get(f"/runs/{run_id}")
    assert run_page.status_code == 200
    assert "Execution contract" in run_page.text
    assert "Tasks" in run_page.text
