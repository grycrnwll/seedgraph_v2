"""The one tested SQLModel↔raw-SQL bridge (decision D7).

``raw_conn(session)`` returns the DBAPI ``sqlite3.Connection`` bound to a
SQLModel/SQLAlchemy session's *active transaction*, so later raw-SQL helpers
(phase_3 ``ensure_span``, phase_4) participate in the ORM transaction instead of
reaching through private session internals. A raw write through it is visible to
the same session before commit and is rolled back with the transaction.
"""

from __future__ import annotations

import sqlite3

from sqlmodel import Session


def raw_conn(session: Session) -> sqlite3.Connection:
    """Return the ``sqlite3.Connection`` bound to ``session``'s active transaction."""
    # session.connection() begins the transaction (if needed) and hands back the
    # SQLAlchemy Connection bound to it.
    sa_connection = session.connection()
    fairy = sa_connection.connection  # PoolProxiedConnection (a.k.a. ConnectionFairy)
    raw = getattr(fairy, "dbapi_connection", None)
    if raw is None:  # pragma: no cover - SQLAlchemy < 2.0 fallback
        raw = getattr(fairy, "driver_connection", None)
    if raw is None:  # pragma: no cover - very old fallback
        raw = getattr(fairy, "connection", fairy)
    return raw
