"""Read-only access to a v2 home (``~/.seedgraph``) for the golden set.

Every database is opened with SQLite's ``mode=ro`` URI and nothing is ever
written under the v2 home. ``.db`` files are never copied (they carry ``-wal``
files), so reads go through SQLite which sees committed WAL content.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from .checks import Rendition
from .errors import GoldenError

DEFAULT_V2_HOME = Path.home() / ".seedgraph"
#: ``created_at`` is stamped just before v2 calls Marker (pending-run insert in
#: v2's ``convert_source_file``), so this is conversion wall time, not queue wait.
#: v2 sometimes ran several conversions at once, so it can include GPU contention.
V2_TIMING_LABEL = (
    "v2 conversion_runs wall time (completed_at - created_at): includes v2 per-paper "
    "model load and any contention from concurrent v2 conversions"
)


def connect_ro(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    if not path.exists():
        raise GoldenError(f"database not found: {path}")
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def _cache_db(v2_home: Path | str) -> sqlite3.Connection:
    return connect_ro(Path(v2_home) / "cache" / "cache.db")


def _project_db(v2_home: Path | str, slug: str) -> sqlite3.Connection:
    return connect_ro(Path(v2_home) / "projects" / slug / "project.db")


def _one(rows: list, what: str, prefix: str):
    if not rows:
        raise GoldenError(f"no {what} matches prefix {prefix!r}")
    if len(rows) > 1:
        raise GoldenError(f"ambiguous {what} prefix {prefix!r}: {len(rows)} matches")
    return rows[0]


def _check_hex(prefix: str) -> str:
    p = prefix.lower().removeprefix("md_")
    if not p or any(ch not in "0123456789abcdef" for ch in p):
        raise GoldenError(f"not a hex prefix: {prefix!r}")
    return p


def resolve_markdown(v2_home: Path | str, md_prefix: str) -> dict:
    """Resolve a markdown-id hex prefix to ``{markdown_id, pdf_sha256}`` (exactly one match)."""
    p = _check_hex(md_prefix)
    conn = _cache_db(v2_home)
    try:
        rows = conn.execute(
            "SELECT m.markdown_id, s.file_hash FROM markdown_documents m "
            "JOIN source_files s ON s.source_file_id = m.source_file_id "
            "WHERE m.markdown_id LIKE ?",
            (f"md_{p}%",),
        ).fetchall()
    finally:
        conn.close()
    markdown_id, pdf = _one(rows, "markdown", md_prefix)
    return {"markdown_id": markdown_id, "pdf_sha256": pdf}


def resolve_pdf(v2_home: Path | str, pdf_prefix: str) -> str:
    """Resolve a PDF sha256 prefix to the full hash (exactly one match)."""
    p = _check_hex(pdf_prefix)
    conn = _cache_db(v2_home)
    try:
        rows = conn.execute("SELECT file_hash FROM source_files WHERE file_hash LIKE ?", (f"{p}%",)).fetchall()
    finally:
        conn.close()
    return _one(rows, "source file (pdf)", pdf_prefix)[0]


def current_note(v2_home: Path | str, slug: str, markdown_id: str, work_id: Optional[str] = None) -> dict:
    """The current note on ``markdown_id`` in project ``slug``.

    Current = latest note per ``(work_id, schema_id)`` by ``(created_at, rowid)``,
    the same rule as :mod:`seedgraph.semantic.current`. Exactly one current note
    must sit on ``markdown_id`` (optionally narrowed by ``work_id``).
    """
    conn = _project_db(v2_home, slug)
    try:
        rows = conn.execute(
            """
            SELECT work_id, schema_id, note_id, created_at, markdown_id FROM (
                SELECT work_id, schema_id, note_id, created_at, markdown_id, ROW_NUMBER() OVER (
                    PARTITION BY work_id, schema_id ORDER BY created_at DESC, rowid DESC
                ) AS rn FROM structured_notes
                WHERE work_id IN (SELECT work_id FROM structured_notes WHERE markdown_id = ?)
            ) WHERE rn = 1
            """,
            (markdown_id,),
        ).fetchall()
    finally:
        conn.close()
    cands = [r for r in rows if r[4] == markdown_id and (work_id is None or r[0] == work_id)]
    if not cands:
        raise GoldenError(f"no current note on {markdown_id[:15]} in project {slug!r}")
    if len(cands) > 1:
        raise GoldenError(
            f"{len(cands)} current notes on {markdown_id[:15]} in project {slug!r}; pass work_id in the seed"
        )
    w, schema, note, created, _ = cands[0]
    return {"work_id": w, "schema_id": schema, "note_id": note, "note_created_at": created}


def note_quotes(quote_source: dict) -> list[str]:
    """Distinct ``exact_quote`` values linked to the pinned note's claims."""
    conn = _project_db(quote_source["v2_home"], quote_source["project_slug"])
    try:
        rows = conn.execute(
            "SELECT DISTINCT es.exact_quote FROM extracted_claims ec "
            "JOIN claim_spans cs ON cs.claim_id = ec.claim_id "
            "JOIN evidence_spans es ON es.span_id = cs.span_id "
            "WHERE ec.structured_note_id = ? ORDER BY es.exact_quote",
            (quote_source["note_id"],),
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def _seconds(start: Optional[str], end: Optional[str]) -> Optional[float]:
    if not start or not end:
        return None
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except ValueError:
        return None


def stored_rendition(v2_home: Path | str, markdown_id: str) -> Rendition:
    """v2's stored markdown plus its conversion wall time (see :data:`V2_TIMING_LABEL`)."""
    conn = _cache_db(v2_home)
    try:
        row = conn.execute(
            "SELECT m.storage_uri, r.converter_version, r.created_at, r.completed_at, r.run_status "
            "FROM markdown_documents m LEFT JOIN conversion_runs r "
            "ON r.conversion_run_id = m.conversion_run_id WHERE m.markdown_id = ?",
            (markdown_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise GoldenError(f"markdown {markdown_id[:15]} not in v2 cache")
    uri, version, start, end, status = row
    markdown = (Path(v2_home) / "cache" / uri).read_text(encoding="utf-8")
    seconds = _seconds(start, end) if status == "success" else None
    return Rendition(markdown=markdown, seconds=seconds, timing_label=V2_TIMING_LABEL, converter_version=version)


def pdf_path(v2_home: Path | str, pdf_sha256: str) -> Path:
    """Path of the cached PDF (``source_files.storage_uri`` relative to the cache root)."""
    conn = _cache_db(v2_home)
    try:
        row = conn.execute("SELECT storage_uri FROM source_files WHERE file_hash = ?", (pdf_sha256,)).fetchone()
    finally:
        conn.close()
    rel = row[0] if row and row[0] else f"pdfs/{pdf_sha256}.pdf"
    return Path(v2_home) / "cache" / rel
