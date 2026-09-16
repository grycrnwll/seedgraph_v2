"""SQLite connection helpers.

Both scopes get ``foreign_keys=ON`` (decision 42 — the must-fix; the cache
phase's intra-file FK graph must be enforced from the first row), plus
``journal_mode=WAL`` and ``busy_timeout=5000``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .. import paths
from ..errors import ValidationError


class ReadOnlyCorpusError(ValidationError):
    """A corpus needs explicit setup, upgrade or checkpointing before reading."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _readonly_signature(path: Path) -> tuple[int, int, int, int]:
    for suffix in ("-wal", "-journal"):
        journal = Path(str(path) + suffix)
        if journal.exists() and journal.stat().st_size:
            raise ReadOnlyCorpusError(
                "setup_required",
                "Read-only retrieval requires a checkpointed, quiescent corpus. "
                "Close writers and checkpoint/recover the database separately, then retry.",
            )
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size


class _ReadOnlyConnection(sqlite3.Connection):
    def close(self) -> None:
        try:
            if _readonly_signature(self._source_path) != self._source_signature:
                raise ReadOnlyCorpusError(
                    "corpus_changed", "The corpus changed during the read. Close writers and retry."
                )
        finally:
            super().close()


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Read a checkpointed corpus without creating or updating WAL sidecars.

    Immutable mode requires a quiescent source. Refuse an active WAL and check
    file identity again on close, so a detected concurrent change never returns
    a successful result. Callers must close the connection before delivery.
    """
    path = Path(db_path).resolve()
    if not path.is_file():
        raise ReadOnlyCorpusError(
            "setup_required", f"Database {path} is missing. Initialize the corpus separately."
        )
    signature = _readonly_signature(path)
    conn = sqlite3.connect(
        path.as_uri() + "?mode=ro&immutable=1", uri=True, factory=_ReadOnlyConnection
    )
    conn._source_path = path
    conn._source_signature = signature
    conn.row_factory = sqlite3.Row
    return conn


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def cache_db_path(root: Path | str | None = None) -> Path:
    return paths.cache_root(root) / "cache.db"


def project_db_path(slug: str, root: Path | str | None = None) -> Path:
    return paths.project_dir(slug, root) / "project.db"


def open_cache_db(root: Path | str | None = None) -> sqlite3.Connection:
    """Open ``cache/cache.db`` with pragmas applied (foreign_keys ON)."""
    return _connect(cache_db_path(root))


def open_project_db(slug: str, root: Path | str | None = None) -> sqlite3.Connection:
    """Open ``projects/{slug}/project.db`` with pragmas applied (foreign_keys ON).

    ``slug`` is validated by :func:`project_db_path` -> ``paths.project_dir``.
    """
    return _connect(project_db_path(slug, root))


def connect_project_raw(db_path: Path) -> sqlite3.Connection:
    """A raw project.db connection with ``foreign_keys=ON`` (fail-closed FKs).

    Deliberately leaner than :func:`open_project_db`: no ``row_factory``, WAL, or
    ``busy_timeout`` — the caller holds a short-lived connection over an existing
    project.db and consumes rows positionally. Takes an explicit ``db_path`` (the
    caller already resolved it, e.g. ``ProjectHandle.db_path``).
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
