"""Versioned migration runner — the ``migrate`` step (decision D6).

Numbered ``schema/{scope}/NNNN_*.sql`` files are the single source of truth for
every ``CREATE TABLE`` / CHECK / index. The runner discovers them in order and
applies any with ``version`` greater than the recorded one, inside one
transaction each. Idempotent: a no-op when already current.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

Scope = Literal["cache", "project"]

_SCHEMA_DIR = Path(__file__).resolve().parent / "schema"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_migrations(scope: Scope) -> list[tuple[int, Path]]:
    """Ordered ``(version, path)`` for a scope, parsed from the ``NNNN_`` prefix."""
    scope_dir = _SCHEMA_DIR / scope
    migrations: list[tuple[int, Path]] = []
    for path in sorted(scope_dir.glob("*.sql")):
        version = int(path.name.split("_", 1)[0])
        migrations.append((version, path))
    migrations.sort(key=lambda item: item[0])
    return migrations


def latest_version(scope: Scope) -> int:
    """Highest available migration version on disk for a scope (0 if none)."""
    migrations = discover_migrations(scope)
    return migrations[-1][0] if migrations else 0


def current_version(conn: sqlite3.Connection) -> int:
    """Recorded schema version, or 0 if ``schema_migrations`` does not exist yet.

    Read defensively so the first migration (which itself creates
    ``schema_migrations``) is allowed to run.
    """
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    except sqlite3.OperationalError:
        return 0
    return row[0] or 0


def run_migrations(conn: sqlite3.Connection, scope: Scope) -> int:
    """Apply pending migrations for ``scope``; return the resulting version."""
    applied = current_version(conn)
    for version, path in discover_migrations(scope):
        if version <= applied:
            continue
        sql = path.read_text(encoding="utf-8")
        with conn:  # one transaction per migration
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, _now()),
            )
        applied = version
    return applied
