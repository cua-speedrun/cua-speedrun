"""Database schema and session plumbing for the platform.

SQLite by default (zero-setup dev), Postgres via CS_DATABASE_URL in
production; the schema is written for both. Result-bearing tables (cards,
entries) are derived views of run logs on disk and can always be recomputed
from them, from the run logs. The events table powers
live status; timing truth stays in the gateway's run logs.
"""

from __future__ import annotations

import datetime
import os
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    true,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DEFAULT_DB_URL = "sqlite:///platform.db"


def utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    github_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    handle: Mapped[str] = mapped_column(String(120), unique=True)
    email: Mapped[str | None] = mapped_column(String(255))
    quota_tier: Mapped[str] = mapped_column(String(32), default="default")
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=utcnow)
    # The user's own Modal credentials: every run of their submissions
    # executes in THEIR Modal workspace, so platform access can never spend
    # the operator's credits. The token id is a public identifier; the
    # secret is Fernet-encrypted (service/usersecrets.py) before storage.
    modal_token_id: Mapped[str | None] = mapped_column(String(64))
    modal_token_secret_enc: Mapped[str | None] = mapped_column(Text)


class SavedEnvironmentVariable(Base):
    """A reusable, user-owned runtime variable.

    Names are visible in account and submission views. Values are encrypted
    independently and are never returned by the API after they are saved.
    """

    __tablename__ = "saved_environment_variables"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "name", name="uq_saved_environment_variable_user_name"
        ),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    value_enc: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )


class Track(Base):
    """Frozen benchmark/scoring/algorithm/hardware spec. Tracks are data."""
    __tablename__ = "tracks"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    gpu: Mapped[str | None] = mapped_column(String(32))
    eval_algorithm: Mapped[str] = mapped_column(
        String(64), default="per-task-vllm@1")
    agents_per_evaluation: Mapped[int] = mapped_column(Integer, default=2)
    # Legacy catalog columns retained only to read old databases. New tracks
    # define agents_per_evaluation; concurrency and the 2C environment pool are
    # derived by the evaluation algorithm.
    max_concurrency: Mapped[int] = mapped_column(Integer, default=2)
    env_pool_size: Mapped[int | None] = mapped_column(Integer)
    runs_per_task: Mapped[int] = mapped_column(Integer, default=1)
    # Retained for old database/run-plan compatibility. New submission
    # sandboxes have ordinary outbound internet; networking no longer selects
    # an agent architecture or requires a domain list.
    network_policy: Mapped[str] = mapped_column(String(32), default="host-network")
    api_domain_allowlist: Mapped[Any] = mapped_column(JSON, default=list)
    ranking_rule: Mapped[str] = mapped_column(String(64), default="time-at-success-bar")
    success_bar: Mapped[float] = mapped_column(Float, default=0.9)
    failure_costs_timeout: Mapped[bool] = mapped_column(Boolean, default=False)
    seed_policy: Mapped[str] = mapped_column(
        String(64), default="scored-without-replacement@1")
    server_config: Mapped[Any] = mapped_column(JSON, default=dict)
    # Legacy catalog field retained for database compatibility. New plans
    # derive runtime identity from execution placement, never from the track.
    agent_runtime: Mapped[Any] = mapped_column(JSON, default=dict)
    reference_only: Mapped[bool] = mapped_column(Boolean, default=False)
    per_run_budget_usd: Mapped[float | None] = mapped_column(Float)


class BenchmarkRow(Base):
    __tablename__ = "benchmarks"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    version: Mapped[str] = mapped_column(String(32))
    path: Mapped[str] = mapped_column(Text)  # task-set location on the worker
    task_count: Mapped[int | None] = mapped_column(Integer)
    active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=true())


class Season(Base):
    """A frozen benchmark+harness+backend+track+hardware stack. Times are
    only ever compared within one season."""
    __tablename__ = "seasons"
    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(255), unique=True)
    status: Mapped[str] = mapped_column(String(16), default="open")  # open|frozen
    contract_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    spec: Mapped[Any] = mapped_column(JSON, default=dict)
    # Storage compatibility with databases created before noise ranking was removed.
    noise_floor_sec: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)


class SubmissionRow(Base):
    __tablename__ = "submissions"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String(120))
    fingerprint: Mapped[str | None] = mapped_column(String(64))
    storage_ref: Mapped[str] = mapped_column(Text)  # artifact-store ref of the zip
    track: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="submitted")
    # Existing databases require a value in this unused, non-null column.
    scrutiny: Mapped[Any] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=utcnow)


class Run(Base):
    """One benchmark run of one submission. `stage` is the live status."""
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(ForeignKey("submissions.id"))
    benchmark_id: Mapped[int] = mapped_column(ForeignKey("benchmarks.id"))
    season_key: Mapped[str | None] = mapped_column(String(255))
    stage: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    error: Mapped[str | None] = mapped_column(Text)
    run_dir: Mapped[str | None] = mapped_column(Text)  # artifact ref (ground truth)
    execution_plan: Mapped[Any] = mapped_column(JSON, default=dict)
    topology_key: Mapped[str | None] = mapped_column(String(64), index=True)
    plan_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    # Legacy scalar retained to recover runs queued before ExecutionScale.
    parallel_evaluations: Mapped[int] = mapped_column(Integer, default=1)
    # Complete operational scale, deliberately outside RunPlan because
    # isolated replica count does not change the comparison contract.
    execution_scale: Mapped[Any] = mapped_column(JSON, default=dict)
    # The compute-runner choice made at submit time ({"kind", "template"}).
    # Operational like execution_scale: outside the frozen plan, but owned
    # by the run so every claiming executor honors the same selection.
    runner_selection: Mapped[Any] = mapped_column(JSON, default=dict)
    task_seeds: Mapped[Any] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    cancel_requested_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)


class RunEnvironmentVariable(Base):
    """The encrypted runtime-variable snapshot for one evaluation.

    Account variables are copied here when the evaluation is created. A
    one-off evaluation value replaces a selected account value with the same
    name before this row is written. Workers therefore never depend on mutable
    account settings after a run enters the queue.
    """

    __tablename__ = "run_environment_variables"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "name", name="uq_run_environment_variable_run_name"
        ),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    value_enc: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(16))  # saved|evaluation
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=utcnow)


class RunTask(Base):
    """Per-task live progress and final summary, mirroring runlog rows."""
    __tablename__ = "run_tasks"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    task_key: Mapped[str] = mapped_column(String(255))  # "task_id/seed_N"
    stage: Mapped[str] = mapped_column(String(32), default="starting")
    passed: Mapped[bool | None] = mapped_column(Boolean)
    reason: Mapped[str | None] = mapped_column(String(64))
    task_time_sec: Mapped[float | None] = mapped_column(Float)
    env_time_sec: Mapped[float | None] = mapped_column(Float)
    agent_time_sec: Mapped[float | None] = mapped_column(Float)
    num_steps: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    usage: Mapped[Any | None] = mapped_column(JSON)


class EventRow(Base):
    """The live-status stream, one row per executor event."""
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    ts: Mapped[float] = mapped_column(Float)
    kind: Mapped[str] = mapped_column(String(64))
    task_key: Mapped[str | None] = mapped_column(String(255))
    payload: Mapped[Any] = mapped_column(JSON, default=dict)


class Card(Base):
    """A private result at a secret URL. Publishing is an explicit accept."""
    __tablename__ = "cards"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), unique=True)
    token: Mapped[str] = mapped_column(String(64), unique=True)
    data: Mapped[Any] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=utcnow)
    accepted_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)


class Entry(Base):
    """The published leaderboard, append-only like entries.jsonl."""
    __tablename__ = "entries"
    __table_args__ = (
        UniqueConstraint("card_id", name="uq_entries_card_id"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    season_key: Mapped[str] = mapped_column(String(255), index=True)
    entry_name: Mapped[str] = mapped_column(String(120))
    card_id: Mapped[int] = mapped_column(ForeignKey("cards.id"))
    data: Mapped[Any] = mapped_column(JSON, default=dict)
    published_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=utcnow)


class Invite(Base):
    """A single-use invite code, claimed when a new user redeems it."""
    __tablename__ = "invites"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(64), unique=True)
    note: Mapped[str | None] = mapped_column(String(255))
    claimed_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=utcnow)
    claimed_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)


class SeedRow(Base):
    """Practice versus scored seed bookkeeping."""
    __tablename__ = "seeds"
    __table_args__ = (
        UniqueConstraint("benchmark_id", "task_id", "pool", "seed",
                         name="uq_seed_instance"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    benchmark_id: Mapped[int] = mapped_column(ForeignKey("benchmarks.id"))
    task_id: Mapped[str] = mapped_column(String(120))
    seed: Mapped[int] = mapped_column(Integer)
    pool: Mapped[str] = mapped_column(String(16), default="practice")  # practice|scored
    used_by_run: Mapped[int | None] = mapped_column(ForeignKey("runs.id"))
    retired_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)


def get_engine(url: str | None = None):
    url = url or os.environ.get("CS_DATABASE_URL", DEFAULT_DB_URL)
    engine = create_engine(url, future=True)
    if url.startswith("sqlite"):
        # WAL lets the worker write while the API reads.
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()
    return engine


def _migrate_columns(engine) -> None:
    """Add columns that postdate an existing database. create_all creates
    missing tables but never alters existing ones, and this project has no
    migration framework yet; each entry here is (table, column, DDL type)."""
    from sqlalchemy import inspect, text

    added = [
        ("users", "modal_token_id", "VARCHAR(64)"),
        ("users", "modal_token_secret_enc", "TEXT"),
        ("tracks", "eval_algorithm", "VARCHAR(64) DEFAULT 'per-task-vllm@1'"),
        ("tracks", "agents_per_evaluation", "INTEGER DEFAULT 2"),
        ("tracks", "max_concurrency", "INTEGER DEFAULT 2"),
        ("tracks", "env_pool_size", "INTEGER"),
        ("tracks", "runs_per_task", "INTEGER DEFAULT 1"),
        ("tracks", "success_bar", "FLOAT DEFAULT 0.9"),
        ("tracks", "failure_costs_timeout", "BOOLEAN DEFAULT 0"),
        ("tracks", "seed_policy", "VARCHAR(64) DEFAULT 'scored-without-replacement@1'"),
        ("tracks", "server_config", "JSON"),
        ("tracks", "agent_runtime", "JSON"),
        ("seasons", "contract_hash", "VARCHAR(64)"),
        ("seasons", "spec", "JSON"),
        ("runs", "execution_plan", "JSON"),
        ("runs", "topology_key", "VARCHAR(64)"),
        ("runs", "plan_hash", "VARCHAR(64)"),
        ("runs", "parallel_evaluations", "INTEGER DEFAULT 1"),
        ("runs", "execution_scale", "JSON"),
        ("runs", "runner_selection", "JSON"),
        ("runs", "task_seeds", "JSON"),
        ("runs", "cancel_requested_at", "TIMESTAMP"),
        ("benchmarks", "task_count", "INTEGER"),
        ("benchmarks", "active", "BOOLEAN NOT NULL DEFAULT TRUE"),
        ("run_tasks", "cost_usd", "FLOAT"),
        ("run_tasks", "usage", "JSON"),
    ]
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, column, ddl_type in added:
            if table not in inspector.get_table_names():
                continue
            existing = {c["name"] for c in inspector.get_columns(table)}
            if column not in existing:
                conn.execute(text(
                    f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}"))
                if table == "tracks" and column == "agents_per_evaluation":
                    # Preserve an installation's existing track ratio instead
                    # of assuming every pre-schema-5 track used the default.
                    conn.execute(text(
                        "UPDATE tracks SET agents_per_evaluation = "
                        "max_concurrency WHERE max_concurrency IS NOT NULL"
                    ))


def make_session_factory(url: str | None = None) -> sessionmaker:
    engine = get_engine(url)
    Base.metadata.create_all(engine)
    _migrate_columns(engine)
    _ensure_schema_invariants(engine)
    return sessionmaker(engine, expire_on_commit=False, future=True)


def _ensure_schema_invariants(engine) -> None:
    """Install uniqueness guarantees create_all cannot add to old tables."""
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_seed_instance "
            "ON seeds (benchmark_id, task_id, pool, seed)"
        )
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_entries_card_id "
            "ON entries (card_id)"
        )
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_seasons_contract_hash "
            "ON seasons (contract_hash) WHERE contract_hash IS NOT NULL"
        )
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_runs_topology_key "
            "ON runs (topology_key)"
        )
