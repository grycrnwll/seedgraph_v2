"""SQLAlchemy/SQLModel engine factory for project.db (phase_5).

The project-model ORM operations (identity/merge, membership, review) run through
a SQLModel :class:`~sqlalchemy.engine.Engine`, while the schema itself is applied
by the numbered ``.sql`` migrate step (D6) — never by ``create_all``.

Why a second door alongside :mod:`seedgraph.db.connection` (which hands back raw
``sqlite3`` connections): phase_0's substrate + the ``migrate`` runner are raw
sqlite3; this engine is the ORM door over the *same* ``project.db`` file. Raw SQL
that must join an in-flight ORM transaction reaches the DBAPI connection via the
tested bridge :func:`seedgraph.db.adapter.raw_conn` — not by reopening the file.

Both functions:

* ``make_project_engine`` — build a SQLAlchemy engine on the project.db path and
  register a ``connect`` event that issues ``PRAGMA foreign_keys=ON``,
  ``PRAGMA journal_mode=WAL`` and ``PRAGMA busy_timeout=5000`` per connection
  (mirroring :func:`seedgraph.db.connection._connect`; decision 8/42).
* ``init_project_db`` — apply pending ``schema/project/*.sql`` via the existing
  ``migrate`` runner (:func:`seedgraph.db.migrations.run_migrations`) over the
  engine's raw connection, asserting ``SQLModel.metadata.create_all`` is NOT used
  to author schema (D6).
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import create_engine

from .migrations import run_migrations


def make_project_engine(path: Path) -> Engine:
    """Return a SQLModel/SQLAlchemy engine bound to the project.db at ``path``.

    The engine's per-connection ``connect`` event sets ``foreign_keys=ON``,
    ``journal_mode=WAL`` and ``busy_timeout=5000`` (decision 8/42), mirroring
    :func:`seedgraph.db.connection._connect`. Implements the ORM door of decision
    81's "one core, two thin surfaces" over the same file the raw substrate uses.
    Does NOT author schema (D6) — schema is the numbered ``.sql`` migrate step's job.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{path.as_posix()}",
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
        finally:
            cursor.close()

    return engine


def init_project_db(engine: Engine) -> None:
    """Apply pending project migrations against ``engine``'s database (the
    ``migrate`` step; D6).

    Delegates to the numbered ``schema/project/*.sql`` runner
    (:func:`seedgraph.db.migrations.run_migrations`) over the engine's raw DBAPI
    connection — NEVER to ``SQLModel.metadata`` schema authoring — so the ``.sql``
    files remain the sole schema source of truth. Idempotent: a no-op when already
    at the latest version.
    """
    raw = engine.raw_connection()
    try:
        dbapi = getattr(raw, "dbapi_connection", None) or getattr(
            raw, "driver_connection", None
        )
        if dbapi is None:  # pragma: no cover - very old SQLAlchemy fallback
            dbapi = raw.connection
        run_migrations(dbapi, "project")
    finally:
        raw.close()
