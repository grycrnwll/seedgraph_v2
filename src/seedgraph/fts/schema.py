"""FTS5 availability assert + ``span_fts`` reindex helpers.

Phase 3 — Evidence Spans. The FTS5 virtual tables themselves are authored by the
numbered migration (``project/0005_evidence_spans.sql``, decision D6) — this module
does NOT ``CREATE`` them. It owns:

* ``assert_fts5_available`` — fail loud if the runtime SQLite lacks FTS5 (documented
  hard requirement; also surfaced by ``doctor``).
* ``reindex_spans`` — the concrete re-index for one markdown, run **inside** the
  span-write transaction: ``DELETE FROM span_fts WHERE markdown_id=?`` then
  ``INSERT INTO span_fts(...) SELECT exact_quote, span_id, markdown_id, work_id,
  section_id FROM evidence_spans WHERE markdown_id=?``. ``span_fts`` is a normal
  (non-external-content) FTS5 table carrying ``markdown_id UNINDEXED``, so the
  batch delete is clean (column scan, fine at MVP scale). No triggers.
* ``rebuild_all`` — full ``span_fts`` rebuild from ``evidence_spans`` (doctor/reindex).
"""

from __future__ import annotations

import sqlite3

# The exact column projection from evidence_spans -> span_fts (UNINDEXED join ids
# + the indexed quote_text). Kept here so reindex + rebuild stay byte-identical.
_INSERT_FROM_SPANS = (
    "INSERT INTO span_fts(quote_text, span_id, markdown_id, work_id, section_id) "
    "SELECT exact_quote, span_id, markdown_id, work_id, section_id FROM evidence_spans"
)


def assert_fts5_available(conn: sqlite3.Connection) -> None:
    """Raise if FTS5 is not compiled into the runtime SQLite (fail loud).

    Probes by attempting a throwaway FTS5 virtual table; a hard requirement
    documented for the whole phase and asserted by ``doctor``.
    """
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp._fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS temp._fts5_probe")
    except sqlite3.OperationalError as exc:  # pragma: no cover - depends on build
        raise RuntimeError(
            "SQLite FTS5 is not compiled into this runtime — span search requires it. "
            f"({exc})"
        ) from exc


def reindex_spans(conn: sqlite3.Connection, markdown_id: str) -> int:
    """Re-index ``span_fts`` for one ``markdown_id``; return rows indexed.

    ``DELETE FROM span_fts WHERE markdown_id=?`` then insert the document's current
    ``evidence_spans`` rows — inside the caller's span-write transaction, so FTS never
    drifts from the spans and re-index leaves no stale duplicate rows (the must-fix).
    """
    conn.execute("DELETE FROM span_fts WHERE markdown_id = ?", (markdown_id,))
    conn.execute(_INSERT_FROM_SPANS + " WHERE markdown_id = ?", (markdown_id,))
    return conn.execute(
        "SELECT COUNT(*) FROM span_fts WHERE markdown_id = ?", (markdown_id,)
    ).fetchone()[0]


def rebuild_all(conn: sqlite3.Connection) -> int:
    """Full rebuild of ``span_fts`` from every ``evidence_spans`` row; return rows indexed.

    The doctor/``reindex`` escape hatch: clears and repopulates ``span_fts`` wholesale.
    """
    conn.execute("DELETE FROM span_fts")
    conn.execute(_INSERT_FROM_SPANS)
    return conn.execute("SELECT COUNT(*) FROM span_fts").fetchone()[0]
