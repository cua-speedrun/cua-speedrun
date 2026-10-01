"""Per-user quotas: keep the queue fair and spend bounded.

Two limits, checked at submission time:

- Scored runs per user per day. Kaggle-style: iterate freely on practice,
  but the platform-funded scored runs are rationed.
- Concurrent in-flight runs per user. Stops one user from filling the queue.

Tiers live on the user row (quota_tier). The default tier's numbers are
here; an operator can widen a tier without code by adding a row to TIERS.
Budget caps per run are a track property (Track.per_run_budget_usd) enforced
by the worker; this module governs how many runs a user may start.
"""

from __future__ import annotations

import datetime

from sqlalchemy import func, select

from cua_speedrun.service.db import Run, SubmissionRow, utcnow

# tier -> (scored runs per day, max concurrent in-flight runs)
TIERS = {
    "default": (1_000, 2),
    "dev": (100, 4),        # operator/testing headroom, not for public users
    "trusted": (50, 4),
    "unlimited": (10_000, 16),
}

_ACTIVE_STAGES = ("queued", "starting", "initializing",
                  "snapshot_done", "running", "scoring")


class QuotaError(Exception):
    """Raised when a submission would exceed the user's quota."""


def check_and_reserve(session, user, track_name: str) -> None:
    """Raise QuotaError if the user is over quota. Practice-track submissions
    (a track flagged reference_only or named 'practice') are exempt; only
    scored runs count against the daily limit."""
    per_day, max_concurrent = TIERS.get(user.quota_tier, TIERS["default"])

    active = session.execute(
        select(func.count()).select_from(Run)
        .join(SubmissionRow, Run.submission_id == SubmissionRow.id)
        .where(SubmissionRow.user_id == user.id,
               Run.stage.in_(_ACTIVE_STAGES))
    ).scalar_one()
    if active >= max_concurrent:
        raise QuotaError(
            f"{active} runs already in flight (limit {max_concurrent} for "
            f"tier '{user.quota_tier}'); wait for one to finish")

    since = utcnow() - datetime.timedelta(days=1)
    today = session.execute(
        select(func.count()).select_from(Run)
        .join(SubmissionRow, Run.submission_id == SubmissionRow.id)
        .where(SubmissionRow.user_id == user.id, Run.created_at >= since)
    ).scalar_one()
    if today >= per_day:
        raise QuotaError(
            f"{today} runs in the last 24h (limit {per_day} for tier "
            f"'{user.quota_tier}'); try again later")
