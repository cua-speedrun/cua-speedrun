"""Transactional season registration, freezing, and card publication."""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from cua_speedrun.service.db import Card, Entry, Season, utcnow


class SeasonFrozenError(ValueError):
    pass


class AlreadyPublishedError(ValueError):
    pass


class SeasonContractError(ValueError):
    pass


def register_season(
    session,
    *,
    key: str,
    contract_hash: str | None,
    spec: dict | None,
) -> Season:
    """Get or create a season and reject conflicting reuse of its key."""
    season = session.execute(
        select(Season).where(Season.key == key).with_for_update()
    ).scalar_one_or_none()
    normalized_spec = dict(spec or {})
    if season is None:
        try:
            with session.begin_nested():
                season = Season(
                    key=key,
                    contract_hash=contract_hash,
                    spec=normalized_spec,
                )
                session.add(season)
                session.flush()
        except IntegrityError:
            # Another worker may have registered the same content-addressed
            # season while this transaction was resolving it.
            season = session.execute(
                select(Season).where(Season.key == key).with_for_update()
            ).scalar_one_or_none()
            if season is None:
                raise SeasonContractError(
                    f"contract {contract_hash!r} is already registered under "
                    "a different season key"
                )

    if contract_hash and season.contract_hash not in (None, contract_hash):
        raise SeasonContractError(
            f"season {key!r} already has contract {season.contract_hash}, "
            f"not {contract_hash}"
        )
    if normalized_spec and season.spec and season.spec != normalized_spec:
        raise SeasonContractError(f"season {key!r} already has a different spec")
    if contract_hash and season.contract_hash is None:
        season.contract_hash = contract_hash
    if normalized_spec and not season.spec:
        season.spec = normalized_spec
    return season


def freeze_season(session, key: str) -> Season:
    season = session.execute(
        select(Season).where(Season.key == key).with_for_update()
    ).scalar_one_or_none()
    if season is None:
        season = Season(key=key, status="frozen")
        session.add(season)
    else:
        season.status = "frozen"
    session.commit()
    return season


def publish_card_to_leaderboard(session, card: Card, entry_name: str) -> Entry:
    """Publish once, and never append to a season after it is frozen."""
    season_key = str(card.data["season_key"])
    run_plan = card.data.get("run_plan") or {}
    contract_hash = card.data.get("run_plan_hash")
    season = register_season(
        session,
        key=season_key,
        contract_hash=contract_hash,
        spec=run_plan,
    )
    if season.status == "frozen":
        raise SeasonFrozenError(f"season {season_key!r} is frozen")
    # Besides rechecking the value, this conditional no-op UPDATE acquires a
    # write lock on SQLite (where SELECT FOR UPDATE is ignored). Publication
    # and freezing therefore have a real transaction order on both supported
    # databases.
    open_claim = session.execute(
        update(Season)
        .where(Season.id == season.id, Season.status == "open")
        .values(status="open")
    )
    if open_claim.rowcount != 1:
        raise SeasonFrozenError(f"season {season_key!r} is frozen")
    if card.accepted_at is not None:
        raise AlreadyPublishedError("card is already published")

    entry = Entry(
        season_key=season_key,
        entry_name=entry_name,
        card_id=card.id,
        data={**card.data, "entry_name": entry_name},
    )
    card.accepted_at = utcnow()
    session.add(entry)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise AlreadyPublishedError("card is already published") from exc
    return entry
