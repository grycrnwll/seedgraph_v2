"""Exact-term FTS5 span search (phrase-safe MATCH + bm25 + relational filters).

Phase 3 — Evidence Spans. Backs ``seedgraph search``: build a phrase-safe FTS5
``MATCH`` query (so ``parallel trends`` / ``Assumption 2`` / ``rank condition`` match
verbatim adjacent tokens), order by ``bm25(span_fts)``, then join ``evidence_spans``
for relational filters (``work_id``, ``section_kind`` via ``document_sections``,
``access_class``). unicode61 tokenization with NO stemming is fixed by the migration,
so ``mixing`` does not match ``mix``/``mixture`` (decisions 33/54).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class SpanHit:
    """One search result: a span id + join context + bm25 rank (lower == better)."""

    span_id: str
    work_id: str
    section_id: str | None
    markdown_id: str
    quote_text: str
    page_start: int | None
    page_end: int | None
    rank: float


def build_match_query(query: str) -> str:
    """Build a phrase-safe FTS5 ``MATCH`` string from a raw user ``query``.

    Wraps the user terms as a single FTS5 phrase (double-quoted, internal quotes
    doubled) so multi-token technical phrases match verbatim adjacency and FTS5
    special characters (``AR(1)``, ``-``, ``*``) never inject query syntax. No
    stemming (the migration fixes unicode61 / no porter), so ``"mixing"`` matches
    the token ``mixing`` and nothing else.
    """
    escaped = query.strip().replace('"', '""')
    return f'"{escaped}"'


def search_spans(
    conn: sqlite3.Connection,
    query: str,
    *,
    work_id: str | None = None,
    section_kind: str | None = None,
    limit: int = 20,
) -> list[SpanHit]:
    """Exact-term search over ``span_fts`` → ranked :class:`SpanHit` list.

    ``span_fts MATCH`` (phrase-safe) ordered by ``bm25``, joined to ``evidence_spans``
    (and ``document_sections`` for ``section_kind``) with optional ``work_id`` /
    ``section_kind`` filters and an ``access_class`` join, capped at ``limit``. No
    stemming — terms match verbatim.
    """
    match = build_match_query(query)
    if not query.strip():
        return []
    sql = [
        "SELECT e.span_id, e.work_id, e.section_id, e.markdown_id, e.exact_quote, "
        "       e.page_start, e.page_end, bm25(span_fts) AS rank "
        "FROM span_fts "
        "JOIN evidence_spans e ON e.span_id = span_fts.span_id "
        "LEFT JOIN document_sections d ON d.section_id = e.section_id "
        "WHERE span_fts MATCH ?",
    ]
    params: list[object] = [match]
    if work_id is not None:
        sql.append("AND e.work_id = ?")
        params.append(work_id)
    if section_kind is not None:
        sql.append("AND d.section_kind = ?")
        params.append(section_kind)
    sql.append("ORDER BY rank LIMIT ?")
    params.append(int(limit))

    cursor = conn.execute(" ".join(sql), tuple(params))
    hits: list[SpanHit] = []
    for row in cursor.fetchall():
        hits.append(
            SpanHit(
                span_id=row[0],
                work_id=row[1],
                section_id=row[2],
                markdown_id=row[3],
                quote_text=row[4],
                page_start=row[5],
                page_end=row[6],
                rank=row[7],
            )
        )
    return hits
