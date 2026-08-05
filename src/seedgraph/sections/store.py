"""Persist / read ``document_sections`` (idempotent replace-by-markdown_id).

Phase 3 — Evidence Spans. Section re-parse is a correctness hot-spot: ``section_id``
is referenced *softly* (no FK) from ``evidence_spans``, so replacing sections never
trips a foreign key (decision §4.6). Re-parse is ``DELETE FROM document_sections WHERE
markdown_id=?`` then insert via the content-deterministic ``section_id`` (effectively
``INSERT OR REPLACE`` keyed on the deterministic id) — identical bytes re-parse to
identical ids (idempotent); a ``section_parser_version`` bump produces a fresh
consistent set. Operates on the raw ``sqlite3.Connection`` obtained via
``db.adapter.raw_conn(project_session)`` (decision D7) so section writes share the
caller's transaction.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from seedgraph.sections.parser import SECTION_PARSER_VERSION, Section

# Section-shaped columns (the dataclass fields), in declaration order.
_SECTION_FIELDS = (
    "section_id",
    "markdown_id",
    "markdown_hash",
    "source_file_id",
    "source_file_hash",
    "work_id",
    "parent_section_id",
    "level",
    "ordinal",
    "heading_text",
    "heading_path",
    "section_kind",
    "start_char",
    "end_char",
    "page_start",
    "page_end",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def replace_sections(
    conn: sqlite3.Connection, markdown_id: str, sections: list[Section]
) -> int:
    """Replace all sections for ``markdown_id`` with ``sections``; return rows written.

    ``DELETE FROM document_sections WHERE markdown_id=?`` then insert each section by
    its deterministic ``section_id`` — idempotent for identical markdown (no duplicate
    rows), inside the caller's transaction. Soft ``section_id`` refs mean no FK
    violation on ``PRAGMA foreign_keys=ON``.
    """
    conn.execute("DELETE FROM document_sections WHERE markdown_id = ?", (markdown_id,))
    now = _now()
    written = 0
    for section in sections:
        conn.execute(
            "INSERT INTO document_sections "
            "(section_id, markdown_id, markdown_hash, source_file_id, source_file_hash, "
            " work_id, parent_section_id, level, ordinal, heading_text, heading_path, "
            " section_kind, start_char, end_char, page_start, page_end, "
            " section_parser_version, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                section.section_id,
                section.markdown_id,
                section.markdown_hash,
                section.source_file_id,
                section.source_file_hash,
                section.work_id,
                section.parent_section_id,
                section.level,
                section.ordinal,
                section.heading_text,
                section.heading_path,
                section.section_kind,
                section.start_char,
                section.end_char,
                section.page_start,
                section.page_end,
                SECTION_PARSER_VERSION,
                now,
            ),
        )
        written += 1
    return written


def load_sections(conn: sqlite3.Connection, markdown_id: str) -> list[Section]:
    """Load stored sections for ``markdown_id`` in ``ordinal`` order.

    Used to re-resolve ``evidence_spans.section_id`` against the current sections and
    by ``spans get`` to render the section breadcrumb.
    """
    cursor = conn.execute(
        f"SELECT {', '.join(_SECTION_FIELDS)} FROM document_sections "
        "WHERE markdown_id = ? ORDER BY ordinal",
        (markdown_id,),
    )
    return [Section(*row) for row in cursor.fetchall()]
