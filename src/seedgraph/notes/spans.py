"""The one span writer on the library (design §3; port of v2 ``spans.store.ensure_span``).

Differences from v2: the markdown is read from the library blob store by
``markdown_id`` (``<library>/<markdown_documents.storage_uri>``, decoded from bytes
so ``\\r\\n`` never shifts offsets); ``evidence_spans``/``span_fts`` are keyed by
``markdown_id`` with no ``work_id`` (stage 1 §2.1); ``access_class`` is provenance
only (ADR-0012), copied from the source file, never reconciled or used as a gate.

Kept from v2: the exact-unique matcher (``anchor.locate_quote``: no match or two or
more matches -> ``None``, never a guess), dedupe on ``(markdown_id, quote_hash)``,
and the write-time invariant ``markdown[start:end] == exact_quote`` checked with
``anchor.assert_invariant`` before anything is written. Writes happen on the
caller's connection inside the caller's transaction.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .. import anchor
from ..ids import new_id
from ..segment import page_boundaries, page_for_range


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def library_root(conn: sqlite3.Connection) -> Path:
    """The library folder: the directory of the connection's main database file."""
    for row in conn.execute("PRAGMA database_list"):
        if row[1] == "main" and row[2]:
            return Path(row[2]).parent
    raise ValueError("the connection has no on-disk main database")


def read_markdown(conn: sqlite3.Connection, markdown_id: str) -> str | None:
    """The stored markdown text for ``markdown_id``, or ``None`` if unknown or missing."""
    row = conn.execute(
        "SELECT storage_uri FROM markdown_documents WHERE markdown_id = ?", (markdown_id,)
    ).fetchone()
    if row is None:
        return None
    path = library_root(conn) / row[0]
    if not path.is_file():
        return None
    return path.read_bytes().decode("utf-8")


def _section_for(conn: sqlite3.Connection, markdown_id: str, start: int) -> str | None:
    """The deepest section containing ``start`` (ties: the later section), as v2."""
    row = conn.execute(
        "SELECT section_id FROM document_sections "
        "WHERE markdown_id = ? AND start_char <= ? AND ? < end_char "
        "ORDER BY level DESC, ordinal DESC LIMIT 1",
        (markdown_id, start, start),
    ).fetchone()
    return row[0] if row else None


def ensure_span(
    conn: sqlite3.Connection,
    *,
    markdown_id: str,
    exact_quote: str | None,
    markdown_text: str | None = None,
    span_kind: str = "manual",
) -> str | None:
    """Locate-or-create the span of ``exact_quote`` in ``markdown_id``; its id or ``None``.

    ``markdown_text`` may be passed when the caller already holds the markdown (a note
    save reads it once for all its quotes); otherwise it is read from the blob store.
    """
    if not exact_quote:
        return None
    text = markdown_text if markdown_text is not None else read_markdown(conn, markdown_id)
    if text is None:
        return None
    located = anchor.locate_quote(text, exact_quote)
    if located is None:
        return None
    start, end = located
    qh = anchor.quote_hash(exact_quote)
    existing = conn.execute(
        "SELECT span_id FROM evidence_spans WHERE markdown_id = ? AND quote_hash = ? "
        "ORDER BY span_id LIMIT 1",
        (markdown_id, qh),
    ).fetchone()
    if existing is not None:
        return existing[0]

    md = conn.execute(
        "SELECT m.markdown_hash, m.source_file_id, s.file_hash, s.access_class "
        "FROM markdown_documents m JOIN source_files s ON s.source_file_id = m.source_file_id "
        "WHERE m.markdown_id = ?",
        (markdown_id,),
    ).fetchone()
    if md is None:
        return None
    span_id = new_id("span")
    # The invariant gate: refuse to write a span whose slice is not the quote.
    anchor.assert_invariant(text, SimpleNamespace(
        start_char=start, end_char=end, exact_quote=exact_quote, quote_hash=qh))
    section_id = _section_for(conn, markdown_id, start)
    page_start, page_end = page_for_range(page_boundaries(text), start, end)
    conn.execute(
        "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, source_file_id, "
        "source_file_hash, section_id, start_char, end_char, exact_quote, quote_hash, "
        "norm_version, span_kind, page_start, page_end, access_class, anchor_status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'anchored', ?)",
        (span_id, markdown_id, md[0], md[1], md[2], section_id, start, end, exact_quote, qh,
         anchor.NORM_VERSION, span_kind, page_start, page_end, md[3], _now()),
    )
    conn.execute(
        "INSERT INTO span_fts (quote_text, span_id, markdown_id, section_id) VALUES (?, ?, ?, ?)",
        (exact_quote, span_id, markdown_id, section_id),
    )
    return span_id
