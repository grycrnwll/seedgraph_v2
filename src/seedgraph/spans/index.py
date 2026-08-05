"""Orchestrate paragraph-span indexing for one markdown (one transaction).

Phase 3 — Evidence Spans. ``index_document`` is the batch entry point behind
``seedgraph spans index``. In a single transaction it: reads the markdown (read-only
cache) → parses + replaces ``document_sections`` → blank-line paragraph-segments →
materializes **deterministic** paragraph spans (``span_kind='paragraph'``,
``ids.auto_span_id(...)``) via the shared ``spans.store._write_span`` writer →
resolves each span's ``section_id`` / pages / ``access_class`` → reindexes
``span_fts``.

Idempotency (the must-fix): paragraph spans use deterministic ids and re-index is
``DELETE FROM evidence_spans WHERE markdown_id=? AND span_kind='paragraph'`` then
re-insert (auto spans only; manual/claim spans untouched), with a matching
``span_fts`` delete-by-``markdown_id`` + insert. Running it twice yields the same ids,
the same row count, and the same number of FTS rows.

Empty/missing handling: if ``read_markdown`` is ``None``, the markdown is
empty/whitespace, or no markdown exists for the work, it logs + returns 0 with a flag
— never raises.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from seedgraph import cache_access
from seedgraph.fts.schema import reindex_spans
from seedgraph.ids import auto_span_id
from seedgraph.sections.parser import parse_sections
from seedgraph.sections.store import replace_sections
from seedgraph.segment import paragraphs
from seedgraph.spans.store import _write_span

_log = logging.getLogger(__name__)


def index_document(
    conn: sqlite3.Connection,
    cache_conn: sqlite3.Connection,
    cache_root: Path,
    *,
    work_id: str,
    markdown_id: str,
) -> int:
    """Index ``markdown_id`` into paragraph spans + ``span_fts``; return spans created.

    One transaction, paragraph granularity, deterministic auto-span ids through the
    shared ``_write_span`` writer; idempotent per ``markdown_id``. Returns the number of
    paragraph spans materialized (``0`` with a logged flag on empty/missing/whitespace
    markdown — never raises).
    """
    md = cache_access.read_markdown(cache_conn, cache_root, markdown_id)
    if md is None:
        _log.warning("index_document: markdown %s unresolvable; skipping", markdown_id)
        return 0
    if not md.text.strip():
        _log.warning("index_document: markdown %s is empty/whitespace; skipping", markdown_id)
        return 0

    # (1) Re-parse + replace sections (idempotent deterministic ids; soft section_id
    #     refs mean no FK violation under PRAGMA foreign_keys=ON).
    sections = parse_sections(
        md.text,
        markdown_id=markdown_id,
        markdown_hash=md.markdown_hash,
        source_file_id=md.source_file_id,
        source_file_hash=md.source_file_hash,
        work_id=work_id,
    )
    replace_sections(conn, markdown_id, sections)

    # (2) Drop only the AUTO paragraph spans for this markdown (manual/claim spans are
    #     untouched), then re-materialize them with deterministic ids.
    conn.execute(
        "DELETE FROM evidence_spans WHERE markdown_id = ? AND span_kind = 'paragraph'",
        (markdown_id,),
    )

    count = 0
    for start, end in paragraphs(md.text):
        quote = md.text[start:end]
        if not quote.strip():
            continue
        span_id = auto_span_id(md.markdown_hash, start, end, "paragraph")
        _write_span(
            conn,
            cache_conn,
            cache_root,
            markdown_id=markdown_id,
            work_id=work_id,
            start=start,
            end=end,
            exact_quote=quote,
            span_kind="paragraph",
            span_id=span_id,
        )
        count += 1

    # (3) Rebuild span_fts for this markdown from the now-current evidence_spans rows
    #     (delete-by-markdown_id + insert) so no stale/duplicate FTS rows remain.
    reindex_spans(conn, markdown_id)
    return count
