"""The v2 snapshot (design §6.1 step 3): every v2 DB read inside one read transaction.

This module owns the only ``sqlite3.connect`` in :mod:`seedgraph.migrate` that can
reach a v2 file, :func:`connect_v2_ro`, which opens ``file:...?mode=ro`` with
``uri=True``. Nothing here (or anywhere in the migration) writes to the v2 home.

A :class:`Snapshot` holds, per v2 DB (the cache, then each project in precedence
order), every table's rows in a canonical order, the DB's ``PRAGMA data_version``,
its schema version and a logical hash, plus sorted listings of the PDF and markdown
trees, each project's ``project.yaml``, lens YAML files and saved-answer files. The
snapshot time is taken once, here.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from seedgraph.library.identity import project_rank

#: Frozen v2 schema versions the migration understands (design §6.1 preflight).
EXPECTED_CACHE_VERSION = 3
EXPECTED_PROJECT_VERSION = 17


class SnapshotError(RuntimeError):
    """The v2 home cannot be snapshotted (missing, or an unexpected schema)."""


def _is_wal_mode(path: Path) -> bool:
    """True when the file header says WAL (bytes 18/19 == 2). Read with a plain read-only open."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(20)
    except OSError:
        return False
    return len(header) == 20 and header[18] == 2 and header[19] == 2


def connect_v2_ro(db_path: Path | str) -> sqlite3.Connection:
    """Open a v2 SQLite file strictly read-only (``mode=ro`` URI, autocommit off-by-hand).

    ``isolation_level=None`` so transactions are explicit: callers run ``BEGIN`` /
    ``COMMIT`` themselves to get one consistent read transaction. A read-only
    connection never checkpoints the WAL.

    A WAL-mode file with no ``-wal`` beside it (fully checkpointed, nobody has it
    open) is opened with ``immutable=1`` as well: a plain ``mode=ro`` open would
    create ``-wal`` and ``-shm`` files in the v2 home. With a ``-wal`` present the
    normal ``mode=ro`` open is used, so the uncheckpointed pages are read.
    """
    path = Path(db_path).resolve()
    query = "?mode=ro"
    if _is_wal_mode(path) and not Path(str(path) + "-wal").exists():
        query += "&immutable=1"
    conn = sqlite3.connect(path.as_uri() + query, uri=True, isolation_level=None)
    conn.execute("PRAGMA query_only=1")
    return conn


def _jsonable(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"$bytes": bytes(value).hex()}
    return value


def canonical_row(table: str, row: dict[str, Any]) -> str:
    return json.dumps({"t": table, "r": {k: _jsonable(v) for k, v in row.items()}},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class DbSnapshot:
    """One v2 DB as read in one transaction."""

    relpath: str  # relative to the v2 home, POSIX
    tables: dict[str, list[dict[str, Any]]]
    columns: dict[str, tuple[str, ...]]
    pk: dict[str, tuple[str, ...]]
    data_version: int
    schema_version: int
    logical_hash: str

    def rows(self, table: str) -> list[dict[str, Any]]:
        return self.tables.get(table, [])


def _table_names(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]


def read_db(db_path: Path, relpath: str) -> DbSnapshot:
    """Read every table of one v2 DB inside a single read transaction."""
    conn = connect_v2_ro(db_path)
    try:
        conn.execute("BEGIN")
        try:
            data_version = conn.execute("PRAGMA data_version").fetchone()[0]
            tables: dict[str, list[dict[str, Any]]] = {}
            columns: dict[str, tuple[str, ...]] = {}
            pks: dict[str, tuple[str, ...]] = {}
            digest = hashlib.sha256()
            for name in _table_names(conn):
                info = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
                cols = tuple(r[1] for r in info)
                pk = tuple(r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5]))
                order = ", ".join(str(i + 1) for i in range(len(cols)))
                cur = conn.execute(f'SELECT * FROM "{name}" ORDER BY {order}')
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                tables[name], columns[name], pks[name] = rows, cols, pk
                digest.update(f"#{name}\n".encode())
                for row in rows:
                    digest.update(canonical_row(name, row).encode("utf-8"))
                    digest.update(b"\n")
            try:
                version = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] or 0
            except sqlite3.OperationalError:
                version = 0
        finally:
            conn.execute("COMMIT")
    finally:
        conn.close()
    return DbSnapshot(relpath, tables, columns, pks, int(data_version), int(version), digest.hexdigest())


@dataclass(frozen=True)
class FileEntry:
    relpath: str  # relative to the v2 cache root (e.g. ``pdfs/<sha>.pdf``), POSIX
    size: int
    sha256: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_tree(root: Path, base: Path) -> list[FileEntry]:
    """Sorted ``(relative path, size, sha256)`` of every file under ``root`` (no mtimes)."""
    if not root.is_dir():
        return []
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(base).as_posix())
    return [FileEntry(p.relative_to(base).as_posix(), p.stat().st_size, sha256_file(p)) for p in files]


def discover_projects(v2_home: Path) -> list[str]:
    """Project slugs with a ``project.db``, in migration precedence order."""
    root = v2_home / "projects"
    if not root.is_dir():
        return []
    slugs = [p.name for p in root.iterdir() if (p / "project.db").is_file()]
    return sorted(slugs, key=project_rank)


def _yaml_scalars(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _yaml_scalars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_yaml_scalars(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


@dataclass
class Snapshot:
    v2_home: Path
    taken_at: str
    cache: DbSnapshot
    projects: dict[str, DbSnapshot]
    pdf_files: list[FileEntry]
    markdown_files: list[FileEntry]
    project_yaml: dict[str, dict[str, Any]]
    lens_files: dict[str, dict[str, str]]          # slug -> {path relative to the project dir: text}
    answer_files: dict[str, list[str]]             # slug -> saved answer envelopes (relative to v2 home)
    run_answer_files: dict[str, list[str]]         # slug -> run-scoped answer envelopes (not carried)
    _md_cache: dict[str, str | None] = field(default_factory=dict, repr=False)
    _pdf_cache: dict[str, list[str] | None] = field(default_factory=dict, repr=False)
    pdf_text_reader: Callable[[Path], list[str] | None] | None = field(default=None, repr=False)

    @property
    def cache_root(self) -> Path:
        return self.v2_home / "cache"

    def markdown_text(self, md_hash: str) -> str | None:
        """The text of ``cache/markdown/<md_hash>.md`` (``None`` if the blob is missing)."""
        if md_hash not in self._md_cache:
            path = self.cache_root / "markdown" / f"{md_hash}.md"
            self._md_cache[md_hash] = path.read_bytes().decode("utf-8") if path.is_file() else None
        return self._md_cache[md_hash]

    def pdf_first_pages(self, sha: str) -> list[str] | None:
        """The text layer of the first pages of ``cache/pdfs/<sha>.pdf`` (memoised)."""
        if sha not in self._pdf_cache:
            path = self.cache_root / "pdfs" / f"{sha}.pdf"
            if not path.is_file():
                self._pdf_cache[sha] = None
            else:
                reader = self.pdf_text_reader
                if reader is None:
                    from seedgraph.library.titlecheck import first_pages_from_pdf as reader
                self._pdf_cache[sha] = reader(path)
        return self._pdf_cache[sha]

    def has_pdf(self, sha: str) -> bool:
        return (self.cache_root / "pdfs" / f"{sha}.pdf").is_file()

    def has_markdown(self, md_hash: str) -> bool:
        return (self.cache_root / "markdown" / f"{md_hash}.md").is_file()

    def fingerprint(self) -> str:
        """sha256 over the logical hashes and the file trees (stored in ``migration_runs``)."""
        h = hashlib.sha256()
        h.update(self.cache.logical_hash.encode())
        for slug, db in self.projects.items():
            h.update(f"{slug}:{db.logical_hash}".encode())
        for f in [*self.pdf_files, *self.markdown_files]:
            h.update(f"{f.relpath}|{f.size}|{f.sha256}".encode())
        return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sorted_rel(paths: Iterable[Path], base: Path) -> list[str]:
    return sorted(p.relative_to(base).as_posix() for p in paths)


def take_snapshot(v2_home: Path | str, *, now: str | None = None) -> Snapshot:
    """Snapshot the v2 home: cache DB, then each project DB in precedence order."""
    v2_home = Path(v2_home)
    cache_db = v2_home / "cache" / "cache.db"
    if not cache_db.is_file():
        raise SnapshotError(f"no v2 cache.db under {v2_home}")
    taken_at = now or _now()
    cache = read_db(cache_db, "cache/cache.db")
    if cache.schema_version != EXPECTED_CACHE_VERSION:
        raise SnapshotError(f"v2 cache.db is at schema {cache.schema_version}, expected {EXPECTED_CACHE_VERSION}")
    projects: dict[str, DbSnapshot] = {}
    yamls: dict[str, dict[str, Any]] = {}
    lenses: dict[str, dict[str, str]] = {}
    answers: dict[str, list[str]] = {}
    run_answers: dict[str, list[str]] = {}
    for slug in discover_projects(v2_home):
        pdir = v2_home / "projects" / slug
        db = read_db(pdir / "project.db", f"projects/{slug}/project.db")
        if db.schema_version != EXPECTED_PROJECT_VERSION:
            raise SnapshotError(f"v2 project {slug!r} is at schema {db.schema_version}, "
                                f"expected {EXPECTED_PROJECT_VERSION}")
        projects[slug] = db
        ypath = pdir / "project.yaml"
        yamls[slug] = _yaml_scalars(yaml.safe_load(ypath.read_text(encoding="utf-8"))) if ypath.is_file() else {}
        ldir = pdir / "lenses"
        lenses[slug] = ({p.relative_to(pdir).as_posix(): p.read_bytes().decode("utf-8")
                         for p in sorted(ldir.glob("*.yaml"))} if ldir.is_dir() else {})
        adir = pdir / "answers"
        answers[slug] = _sorted_rel(
            (p for p in adir.glob("*.json") if not p.name.endswith(".trace.json")), v2_home) if adir.is_dir() else []
        rdir = pdir / "runs"
        run_answers[slug] = _sorted_rel(
            (p for p in rdir.glob("*/answers/*.json") if not p.name.endswith(".trace.json")),
            v2_home) if rdir.is_dir() else []
    base = v2_home / "cache"
    return Snapshot(
        v2_home=v2_home, taken_at=taken_at, cache=cache, projects=projects,
        pdf_files=list_tree(base / "pdfs", base), markdown_files=list_tree(base / "markdown", base),
        project_yaml=yamls, lens_files=lenses, answer_files=answers, run_answer_files=run_answers,
    )
