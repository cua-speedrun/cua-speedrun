"""Invite-code account gating for the first season.

Gating is on only when CS_INVITE_ONLY=1. When on, a first-time GitHub sign-in
must carry a valid, unclaimed invite code (passed through the OAuth `state`
round trip); the code is claimed atomically as the account is created.
Existing users sign in normally. Off by default so local/dev and open seasons
need no codes.

An operator mints codes with `admin mint-invite`.
"""

from __future__ import annotations

import os
import secrets

from sqlalchemy import select, update

from cua_speedrun.service.db import Invite, utcnow


def gating_on() -> bool:
    return os.environ.get("CS_INVITE_ONLY") == "1"


def mint(session, note: str | None = None) -> str:
    code = secrets.token_urlsafe(9)
    session.add(Invite(code=code, note=note))
    session.commit()
    return code


def claim(session, code: str, user_id: int) -> bool:
    """Atomically claim an unclaimed code for a user. Returns False if the
    code is unknown or already claimed (the two concurrent claimers case:
    the second sees rowcount 0)."""
    if not code:
        return False
    row = session.execute(
        select(Invite).where(Invite.code == code,
                             Invite.claimed_by.is_(None))
    ).scalar_one_or_none()
    if row is None:
        return False
    claimed = session.execute(
        update(Invite)
        .where(Invite.id == row.id, Invite.claimed_by.is_(None))
        .values(claimed_by=user_id, claimed_at=utcnow())
    )
    session.commit()
    return claimed.rowcount == 1
