"""Open, create and migrate the library store (v3 design §1, §2).

The library is ``<home>/library/library.db`` with its blobs beside it and a
separate, disposable ``provider_cache.db``. Context graphs live in the sibling
``<home>/graphs/`` folder (see :mod:`seedgraph.library.graphs`).

Every write goes through :meth:`Library.write`, a transaction that also sets
``library_meta.dirty = '1'`` - unless the library was opened in migration mode
(:func:`open_library_at` with ``migration=True``), which is how the migration
builds a staging library that the swap may later replace (§1.3).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from ..db.migrations import MIGRATE_LOCK, run_migrations
from ..errors import SeedgraphError

LIBRARY_SCOPE = "library"
PROVIDER_CACHE_SCOPE = "library/provider_cache"

ORIGIN_LOCAL = "local"
ORIGIN_MIGRATION = "migration"
#: A library `seedgraph setup` created and nothing has written to yet; a migration may
#: replace it while it holds no data (``seedgraph.migrate.files.check_replaceable``).
ORIGIN_SETUP = "setup"


class LibraryError(SeedgraphError):
    """A library operation was refused."""


def library_dir(home: Path | str | None = None) -> Path:
    """``<home>/library`` (home resolved as everywhere: --root, $SEEDGRAPH_HOME, default)."""
    return paths.resolve_home(home) / "library"


def library_db_path(lib_dir: Path) -> Path:
    return Path(lib_dir) / "library.db"


def provider_cache_path(lib_dir: Path) -> Path:
    return Path(lib_dir) / "provider_cache.db"


def graphs_root_for(lib_dir: Path) -> Path:
    """Graphs are a sibling of the library folder: ``<home>/graphs`` or
    ``staging/<run_id>/graphs`` for a staging build."""
    return Path(lib_dir).parent / "graphs"


def connect(db_path: Path, *, uri: bool = False) -> sqlite3.Connection:
    """A connection with the library pragmas: FKs on, WAL, busy timeout, Row factory.

    ``isolation_level=None``: transactions are explicit (:meth:`Library.write`).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    target = db_path.resolve().as_uri() if uri else str(db_path)
    conn = sqlite3.connect(target, uri=uri, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def open_migrated(db_path: Path, scope: str, *, uri: bool = False) -> sqlite3.Connection:
    """Open ``db_path`` and apply the numbered migrations of ``scope``.

    Holds the process-wide MIGRATE_LOCK across the open too (a brand-new file's
    WAL switch fails without a busy wait if another thread is creating it).
    """
    with MIGRATE_LOCK:
        conn = connect(db_path, uri=uri)
        try:
            run_migrations(conn, scope)  # type: ignore[arg-type]
        except BaseException:
            conn.close()
            raise
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` ... ``COMMIT`` (``ROLLBACK`` on error).

    Nested inside an open transaction it becomes a SAVEPOINT, so the outer
    transaction stays the unit of commit. Code inside must not call
    ``commit()``/``rollback()``/``BEGIN`` itself.
    """
    if conn.in_transaction:
        conn.execute("SAVEPOINT nested_write")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK TO nested_write")
            conn.execute("RELEASE nested_write")
            raise
        conn.execute("RELEASE nested_write")
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def read_meta(conn: sqlite3.Connection) -> dict[str, str | None]:
    return {r[0]: r[1] for r in conn.execute("SELECT key, value FROM library_meta ORDER BY key")}


def _set_meta(conn: sqlite3.Connection, key: str, value: str | None) -> None:
    conn.execute(
        "INSERT INTO library_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def is_dirty(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT value FROM library_meta WHERE key = 'dirty'").fetchone()
    return row is not None and row[0] == "1"


def is_held(conn: sqlite3.Connection, paper_id: str) -> bool:
    """True iff the paper has a ``paper_files`` row (the ``paper_status`` rule)."""
    row = conn.execute("SELECT status FROM paper_status WHERE paper_id = ?", (paper_id,)).fetchone()
    return row is not None and row[0] == "held"


def paper_exists(conn: sqlite3.Connection, paper_id: str) -> bool:
    return conn.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)).fetchone() is not None


@dataclass
class Library:
    """An open library. ``conn`` is for reads; write through :meth:`write`."""

    root: Path  # the library folder (holds library.db, provider_cache.db, pdfs/, markdown/)
    conn: sqlite3.Connection
    migration: bool = False

    @property
    def db_path(self) -> Path:
        return library_db_path(self.root)

    @property
    def pdfs_dir(self) -> Path:
        return self.root / "pdfs"

    @property
    def markdown_dir(self) -> Path:
        return self.root / "markdown"

    @property
    def graphs_root(self) -> Path:
        return graphs_root_for(self.root)

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """One write transaction. Outside migration mode it also sets ``dirty = '1'``
        in the same transaction, so a committed write is never left unflagged."""
        with transaction(self.conn) as conn:
            yield conn
            if not self.migration:
                _set_meta(conn, "dirty", "1")

    def mark_dirty(self) -> None:
        """Flag a non-migration write made to a store that travels with the library
        (a ``graph.db``). A no-op in migration mode."""
        if self.migration:
            return
        with transaction(self.conn) as conn:
            _set_meta(conn, "dirty", "1")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Library:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_library_at(lib_dir: Path | str, *, migration: bool = False, new_origin: str | None = None) -> Library:
    """Open (creating and migrating if needed) the library in ``lib_dir``.

    ``migration=True`` is for the migration's staging build: its writes leave
    ``dirty`` alone. A library created in migration mode gets ``origin =
    'migration'``; one created otherwise gets ``new_origin`` (``seedgraph setup``
    passes ``'setup'``), else ``origin = 'local'``. An existing library's origin is
    never changed here.
    """
    lib_dir = Path(lib_dir)
    conn = open_migrated(library_db_path(lib_dir), LIBRARY_SCOPE)
    try:
        if conn.execute("SELECT 1 FROM library_meta WHERE key = 'origin'").fetchone() is None:
            with transaction(conn):
                _set_meta(conn, "origin", ORIGIN_MIGRATION if migration else (new_origin or ORIGIN_LOCAL))
    except BaseException:
        conn.close()
        raise
    return Library(root=lib_dir, conn=conn, migration=migration)


def open_library(home: Path | str | None = None, *, new_origin: str | None = None) -> Library:
    """Open the live library at ``<home>/library`` (normal mode: writes set dirty)."""
    return open_library_at(library_dir(home), new_origin=new_origin)


def record_migration_origin(lib: Library, *, run_id: str, v2_home: str) -> None:
    """Stamp a staging library as the product of migration run ``run_id`` (clean)."""
    if not lib.migration:
        raise LibraryError("record_migration_origin needs a library opened in migration mode")
    with transaction(lib.conn) as conn:
        _set_meta(conn, "origin", ORIGIN_MIGRATION)
        _set_meta(conn, "migration_run_id", run_id)
        _set_meta(conn, "v2_home", v2_home)
        _set_meta(conn, "dirty", "0")


def open_provider_cache(lib_dir: Path | str) -> sqlite3.Connection:
    """Open (creating and migrating if needed) ``<lib_dir>/provider_cache.db``."""
    return open_migrated(provider_cache_path(Path(lib_dir)), PROVIDER_CACHE_SCOPE)
