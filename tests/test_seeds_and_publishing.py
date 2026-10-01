from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from cua_speedrun.service.db import (
    BenchmarkRow,
    Card,
    Entry,
    Run,
    SeedRow,
    SubmissionRow,
    User,
    make_session_factory,
)
from cua_speedrun.service.publishing import (
    AlreadyPublishedError,
    SeasonContractError,
    SeasonFrozenError,
    freeze_season,
    publish_card_to_leaderboard,
    register_season,
)
from cua_speedrun.service.seeds import draw_scored_task_seeds


def _database(tmp_path: Path):
    factory = make_session_factory(f"sqlite:///{tmp_path / 'platform.db'}")
    with factory() as session:
        user = User(handle="tester")
        benchmark = BenchmarkRow(name="tiny", version="1", path="/tmp/tiny")
        session.add_all([user, benchmark])
        session.flush()
        submission = SubmissionRow(
            user_id=user.id,
            name="entry",
            storage_ref="artifact.zip",
            track="open-l4",
        )
        session.add(submission)
        session.flush()
        run = Run(submission_id=submission.id, benchmark_id=benchmark.id)
        session.add(run)
        session.commit()
        return factory, benchmark.id, run.id


def test_scored_seed_map_is_unique_and_retry_stable(tmp_path: Path) -> None:
    factory, benchmark_id, run_id = _database(tmp_path)
    with factory() as session:
        first = draw_scored_task_seeds(
            session, benchmark_id, ["alpha", "beta"], run_id, runs_per_task=2
        )
        session.commit()
    with factory() as session:
        retry = draw_scored_task_seeds(
            session, benchmark_id, ["alpha", "beta"], run_id, runs_per_task=2
        )
        session.commit()
        original_run = session.get(Run, run_id)
        second_run = Run(
            submission_id=original_run.submission_id,
            benchmark_id=benchmark_id,
        )
        session.add(second_run)
        session.commit()
        second = draw_scored_task_seeds(
            session,
            benchmark_id,
            ["alpha", "beta"],
            second_run.id,
            runs_per_task=2,
        )
        session.commit()
        rows = session.execute(select(SeedRow).order_by(SeedRow.task_id, SeedRow.seed)).scalars().all()

    assert retry == first
    assert set(first["alpha"]).isdisjoint(second["alpha"])
    assert set(first["beta"]).isdisjoint(second["beta"])
    assert len(rows) == 8
    assert len({(row.task_id, row.seed) for row in rows}) == 8
    assert all(row.used_by_run is not None and row.retired_at is not None for row in rows)


def test_frozen_season_rejects_publication_and_card_publishes_once(tmp_path: Path) -> None:
    factory, _benchmark_id, run_id = _database(tmp_path)
    plan = {"schema_version": 1, "scoring": {"success_bar": 0.9}}
    with factory() as session:
        register_season(
            session, key="season-open", contract_hash="a" * 64, spec=plan
        )
        with pytest.raises(SeasonContractError):
            register_season(
                session,
                key="season-open",
                contract_hash="b" * 64,
                spec=plan,
            )
        card = Card(
            run_id=run_id,
            token="open-token",
            data={
                "season_key": "season-open",
                "run_plan_hash": "a" * 64,
                "run_plan": plan,
            },
        )
        session.add(card)
        session.commit()
        entry = publish_card_to_leaderboard(session, card, "My agent")
        assert entry.entry_name == "My agent"
        with pytest.raises(AlreadyPublishedError):
            publish_card_to_leaderboard(session, card, "Again")

    with factory() as session:
        assert session.execute(select(Entry)).scalars().one().card_id == card.id
        frozen = freeze_season(session, "season-frozen")
        assert frozen.status == "frozen"
        second_run = Run(submission_id=1, benchmark_id=1)
        session.add(second_run)
        session.flush()
        frozen_card = Card(
            run_id=second_run.id,
            token="frozen-token",
            data={"season_key": "season-frozen"},
        )
        session.add(frozen_card)
        session.commit()
        with pytest.raises(SeasonFrozenError):
            publish_card_to_leaderboard(session, frozen_card, "Too late")
