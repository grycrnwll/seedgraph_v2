"""``cache.db`` ORM engine + init (decision D6).

``init_cache_db`` applies the pending numbered ``db/schema/cache/NNNN_*.sql``
migrations IN ORDER via the shared, tested migration runner — it is **never**
``SQLModel.metadata.create_all`` (which would author schema and violate D6). It
delegates to :func:`seedgraph.db.bootstrap.ensure_cache_db`, so the cache phase
reuses the one migration path the foundation already owns.

The SQLModel ``Engine`` / ``Session`` helpers give later phases typed ORM access
to ``cache.db`` with the same per-connection pragmas the raw path sets
(``journal_mode=WAL``, ``busy_timeout=5000``, ``foreign_keys=ON`` — the FK pragma
must be re-applied per connection via a SQLAlchemy ``connect`` listener).
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool
from sqlmodel import Session, create_engine

from ..db.bootstrap import ensure_cache_db
from ..db.connection import cache_db_path


def init_cache_db(root: Path | str | None = None) -> Path:
    """Create + migrate ``cache.db`` (apply pending ``.sql`` in order, NOT
    ``create_all`` — D6); return its path. Idempotent. Thin wrapper over the
    shared :func:`ensure_cache_db` so there is a single migration code path."""
    return ensure_cache_db(root)


def cache_engine(root: Path | str | None = None) -> Engine:
    """Return a SQLModel/SQLAlchemy ``Engine`` bound to ``cache.db`` with a
    per-connection pragma listener (WAL / busy_timeout / ``foreign_keys=ON``).
    Does NOT author schema — call :func:`init_cache_db` for migrations.

    Uses ``NullPool`` so each connection is fully closed when its session is
    released (no lingering file handles that would pin a temp cache root open on
    Windows). The ``foreign_keys=ON`` pragma must be re-applied per connection via
    the ``connect`` listener (SQLite resets it for every new connection).
    """
    path = cache_db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path.as_posix()}", poolclass=NullPool)

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, _connection_record):  # pragma: no cover - thin glue
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def cache_session(root: Path | str | None = None) -> Session:
    """Open a SQLModel ``Session`` on ``cache.db`` (pragmas applied via the engine
    ``connect`` listener). Caller manages the transaction / ``commit`` and closes
    the session (releasing the connection)."""
    return Session(cache_engine(root))
