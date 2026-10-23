"""Frozen v2 schema (design §9): read-only copies of v2's numbered SQL files.

``project/*.sql`` and ``cache/*.sql`` are verbatim copies of
``seedgraph/db/schema/{project,cache}/``, frozen so the migration can still
recognise and its fixtures can still build v2-shaped databases after the old v2
code is deleted. Never edit these files: v2 data on disk is what they describe.

The loader mirrors v2's ``db/migrations.run_migrations``: version from the
``NNNN_`` prefix (gaps allowed), one ``executescript`` per file (0014 relies on
its autocommit ``PRAGMA foreign_keys=OFF``), then a ``schema_migrations`` row.
"""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path
from typing import Literal

Scope = Literal["project", "cache"]

SCOPES: tuple[Scope, ...] = ("project", "cache")

#: ``schema_migrations.applied_at`` written when the caller gives none. Fixed, so
#: two builds of the same fixture are identical.
DEFAULT_APPLIED_AT = "2026-06-30T00:00:00+00:00"


def schema_files(scope: Scope) -> list[tuple[int, Path]]:
    """Ordered ``(version, path)`` of the frozen SQL files for ``scope``."""
    if scope not in SCOPES:
        raise ValueError(f"unknown v2 schema scope: {scope!r}")
    root = resources.files(__name__) / scope
    files = [
        (int(entry.name.split("_", 1)[0]), Path(str(entry)))
        for entry in root.iterdir()
        if entry.name.endswith(".sql")
    ]
    return sorted(files, key=lambda item: item[0])


def latest_version(scope: Scope) -> int:
    """Highest frozen version for ``scope`` (project 17, cache 3)."""
    return schema_files(scope)[-1][0]


def apply_v2_schema(
    conn: sqlite3.Connection, scope: Scope, *, applied_at: str = DEFAULT_APPLIED_AT
) -> int:
    """Apply every frozen file for ``scope`` to an empty connection; return the version."""
    version = 0
    for version, path in schema_files(scope):
        with conn:
            conn.executescript(path.read_text(encoding="utf-8"))
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, applied_at),
            )
    return version


def create_v2_db(path: Path, scope: Scope, *, applied_at: str = DEFAULT_APPLIED_AT) -> int:
    """Create a fresh SQLite file at ``path`` with the frozen v2 schema; return the version.

    Refuses an existing file, so it can never touch a real v2 database.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to apply the v2 schema to an existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        return apply_v2_schema(conn, scope, applied_at=applied_at)
    finally:
        conn.close()
