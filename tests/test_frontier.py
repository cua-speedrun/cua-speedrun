from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cua_speedrun.service.db import (
    BenchmarkRow,
    Card,
    Entry,
    Run,
    Season,
    SubmissionRow,
    Track,
    User,
    make_session_factory,
)
from cua_speedrun.service.frontier import (
    build_frontier_context,
    dashboard_entries,
    pareto_entry_ids,
    pareto_geometry,
    parse_season_key,
    track_label,
)
from cua_speedrun.service.web import register_web


def _entry(
    entry_id: int,
    time: float,
    success: float,
    *,
    track: str = "open-l4",
    season: str = "benchmark=demo@1 | harness=1 | backend=modal | hardware=L4",
    reference: bool = False,
    runs: int = 3,
) -> dict:
    return {
        "entry_id": entry_id,
        "entry_name": f"agent-{entry_id}",
        "season_key": season,
        "track_name": track,
        "submission_fingerprint": f"fingerprint-{entry_id}",
        "total_time_sec": time,
        "median_task_time_sec": time,
        "success_rate": success,
        "meets_success_bar": success >= 0.9,
        "num_runs": runs,
        "num_passed": round(success * runs),
        "agent_time_sec": time * 0.4,
        "env_time_sec": time * 0.6,
        "reference_only": reference,
        "rules": {"success_bar": 0.9},
    }


def test_season_parser_and_track_label_support_versioned_keys() -> None:
    key = (
        "benchmark=osworld@1 | harness=2 | backend=modal | hardware=L40S "
        "| track=open-l40s | algorithm=shared-agent-vllm@2 | contract=abc123"
    )
    assert parse_season_key(key) == {
        "benchmark": "osworld@1",
        "harness": "2",
        "backend": "modal",
        "hardware": "L40S",
        "track": "open-l40s",
        "algorithm": "shared-agent-vllm@2",
        "contract": "abc123",
    }
    assert track_label("api-h100") == "API H100"


def test_pareto_frontier_is_exact_and_excludes_reference_entries() -> None:
    entries = [
        _entry(1, 10, 0.8),
        _entry(2, 12, 0.9),
        _entry(3, 15, 0.7),
        _entry(4, 10, 0.8),  # identical objective values are equivalent
        _entry(5, 8, 1.0, reference=True),
        _entry(6, 10, 0.7),  # dominated at the same time
    ]
    assert pareto_entry_ids(entries) == {1, 2, 4}


def test_geometry_contains_success_bar_and_frontier_path() -> None:
    geometry = pareto_geometry(
        [_entry(1, 10, 0.8), _entry(2, 12, 0.9)], success_bar=0.9
    )
    assert geometry is not None
    assert geometry["success_bar"] == 0.9
    assert geometry["success_bar_y"] < geometry["mt"] + geometry["ph"]
    assert geometry["frontier_count"] == 2
    assert geometry["path"].startswith("M ")
    assert " L " in geometry["path"]


def test_geometry_staggers_labels_for_nearly_identical_points() -> None:
    geometry = pareto_geometry(
        [_entry(1, 26.40, 1.0), _entry(2, 26.41, 1.0)], success_bar=0.9
    )
    assert geometry is not None
    assert geometry["points"][0]["label_y"] != geometry["points"][1]["label_y"]


def test_dashboard_entries_use_mean_score_and_split_effort_suffix() -> None:
    entry = {
        **_entry(1, 100, 0.8, runs=4),
        "entry_name": "Gemini 3.7 Flash high (default)",
        "mean_score": 0.8962,
        "median_task_time_sec": 24.5,
        "model_type": "closed",
        "frontier": True,
        "provisional": False,
    }
    [result] = dashboard_entries([entry])
    assert result == {
        "entry_id": 1,
        "entry_name": "Gemini 3.7 Flash high (default)",
        "model": "Gemini 3.7 Flash",
        "effort": "high",
        "effort_rank": 3,
        "model_type": "closed",
        "performance": 0.8962,
        "time_per_task_sec": 24.5,
        "average_time_per_task_sec": 25.0,
        "median_time_per_task_sec": 24.5,
        "cost_usd": None,
        "turns_per_task": None,
        "release_date": None,
        "average_output_tokens": None,
        "frontier": True,
        "qualification_known": True,
        "qualifying": False,
        "reference": False,
        "provisional": False,
    }


def test_dashboard_entries_accept_explicit_model_metadata() -> None:
    entry = {
        **_entry(2, 12, 1.0),
        "model_family": "Qwen3.5-VL 9B",
        "reasoning_effort": "thinking",
        "model_type": "open",
    }
    [result] = dashboard_entries([entry])
    assert result["model"] == "Qwen3.5-VL 9B"
    assert result["effort"] == "thinking"
    assert result["effort_rank"] == 6
    assert result["model_type"] == "open"


def test_dashboard_entries_use_explicit_or_aggregate_cost_per_task() -> None:
    explicit = {
        **_entry(3, 12, 1.0, runs=50),
        "cost_per_task_usd": 0.2802,
    }
    aggregate = {
        **_entry(4, 12, 1.0, runs=50),
        "cost_usd": 14.01,
    }
    missing = _entry(5, 12, 1.0, runs=50)

    results = dashboard_entries([explicit, aggregate, missing])
    by_id = {result["entry_id"]: result for result in results}
    assert by_id[3]["cost_usd"] == 0.2802
    assert by_id[4]["cost_usd"] == 0.2802
    assert by_id[5]["cost_usd"] is None


def test_context_never_mixes_tracks_or_seasons() -> None:
    season_a = "benchmark=a@1 | harness=1 | backend=modal | hardware=cpu"
    season_b = "benchmark=b@1 | harness=1 | backend=modal | hardware=L4"
    entries = [
        _entry(1, 10, 1.0, track="alpha", season=season_a),
        _entry(2, 11, 1.0, track="alpha", season=season_a),
        _entry(3, 20, 1.0, track="beta", season=season_b),
        _entry(4, 21, 1.0, track="beta", season=season_b),
        _entry(5, 22, 1.0, track="beta", season=season_b),
    ]
    context = build_frontier_context(
        entries,
        [{"name": "alpha"}, {"name": "beta"}],
        {},
        selected_season=season_a,
    )
    assert context["track"]["name"] == "alpha"
    assert context["season_key"] == season_a
    assert {row["entry_id"] for row in context["current"]} == {1, 2}

    default_context = build_frontier_context(
        entries, [{"name": "alpha"}, {"name": "beta"}], {}
    )
    assert default_context["track"]["name"] == "beta"
    assert {row["entry_id"] for row in default_context["current"]} == {3, 4, 5}


@pytest.mark.parametrize(
    "second_time, second_rank, tied",
    [(10.3, 2, False), (10.0, 1, True)],
)
def test_ranking_ties_use_exact_times(second_time, second_rank, tied) -> None:
    season = "benchmark=a@1 | harness=1 | backend=modal | hardware=cpu"
    context = build_frontier_context(
        [
            _entry(1, 10.0, 1.0, track="alpha", season=season, runs=3),
            _entry(2, second_time, 1.0, track="alpha", season=season, runs=2),
            _entry(3, 20.0, 0.8, track="alpha", season=season, runs=3),
        ],
        [{"name": "alpha", "success_bar": 0.9}],
        {season: {"status": "open"}},
        selected_track="alpha",
    )
    first, second, below = context["current"]
    assert (first["rank"], second["rank"], below["rank"]) == (1, second_rank, None)
    assert second["tied"] is tied
    assert first["frontier"] is True
    assert second["frontier"] is tied
    assert below["frontier"] is False
    assert second["provisional"] is True


def test_fast_below_bar_tradeoff_is_pareto_but_never_ranked() -> None:
    season = "benchmark=a@1 | harness=1 | backend=modal | hardware=cpu"
    context = build_frontier_context(
        [
            _entry(1, 5.0, 0.5, track="alpha", season=season),
            _entry(2, 10.0, 1.0, track="alpha", season=season),
        ],
        [{"name": "alpha", "success_bar": 0.9}],
        {},
        selected_track="alpha",
    )
    fast, qualifying = context["current"][1], context["current"][0]
    assert fast["entry_id"] == 1
    assert fast["frontier"] is True
    assert fast["rank"] is None
    assert qualifying["frontier"] is True
    assert qualifying["rank"] == 1


def test_frontier_route_renders_track_and_comparison_boundary(tmp_path) -> None:
    database = tmp_path / "frontier.db"
    session_factory = make_session_factory(f"sqlite:///{database}")
    season_key = (
        "benchmark=demo@1 | harness=1 | backend=modal | hardware=L4 "
        "| track=open-l4 | algorithm=per-task-vllm@1 | contract=abcd1234"
    )
    with session_factory() as session:
        user = User(handle="runner")
        track = Track(name="open-l4", gpu="L4", success_bar=0.9)
        benchmark = BenchmarkRow(name="demo", version="1", path="benchmarks/demo")
        session.add_all([user, track, benchmark])
        session.flush()
        submission = SubmissionRow(
            user_id=user.id,
            name="quick-agent",
            storage_ref="artifact.zip",
            track=track.name,
            status="complete",
        )
        session.add(submission)
        session.flush()
        run = Run(
            submission_id=submission.id,
            benchmark_id=benchmark.id,
            season_key=season_key,
            stage="card_ready",
        )
        session.add(run)
        session.flush()
        card_data = {
            "entry_name": "quick-agent",
            "season_key": season_key,
            "submission_fingerprint": "abc123",
            "num_runs": 3,
            "num_passed": 3,
            "success_rate": 1.0,
            "meets_success_bar": True,
            "total_time_sec": 12.5,
            "median_task_time_sec": 4.0,
            "agent_time_sec": 5.0,
            "env_time_sec": 7.5,
            "rules": {"success_bar": 0.9},
            "run_plan": {"track": {"name": "open-l4"}},
        }
        card = Card(run_id=run.id, token="result-token", data=card_data)
        session.add(card)
        session.flush()
        session.add(Entry(
            season_key=season_key,
            entry_name="quick-agent",
            card_id=card.id,
            data=card_data,
        ))
        session.add(Season(
            key=season_key,
            status="open",
            contract_hash="a" * 64,
        ))
        session.commit()

    app = FastAPI()
    register_web(app, session_factory, lambda _request: None)
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "<h1>Results explorer</h1>" in response.text
    assert 'data-view-button="leaderboard"' in response.text
    assert 'data-view-button="curves"' in response.text
    assert 'data-view-button="frontier3d"' in response.text
    assert 'data-view-panel="frontier3d"' in response.text
    assert 'data-camera-preset="perspective"' in response.text
    assert 'data-ranking="performance"' in response.text
    assert 'data-ranking="time"' in response.text
    assert 'data-ranking="cost"' in response.text
    assert "quick-agent" in response.text
    assert "open-l4" in response.text
    for metric in (
        "average_time", "median_time", "cost", "turns", "release_date",
        "output_tokens",
    ):
        assert f'data-curve="{metric}"' in response.text
