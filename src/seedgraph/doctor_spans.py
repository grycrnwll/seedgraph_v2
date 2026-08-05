"""``doctor`` sub-check for spans/sections/FTS (Phase 3 — Evidence Spans).

Wired into ``seedgraph doctor`` by the shared doctor module (see ``needs_wiring``).
Reports, without crashing:

* **FTS5 availability** — fail loud if the runtime SQLite lacks FTS5.
* **Cross-db reconcile** — LEFT-JOIN span/section ``markdown_id`` against read-only
  cache.db and classify ``ok`` / ``stale`` (a newer markdown supersedes this content) /
  ``missing`` (cache pruned), keyed on denormalized lineage so detection survives
  ``markdown_id`` GC.
* **Soft ``section_id`` resolution** — flag any span whose ``section_id`` no longer
  resolves to an existing ``document_sections`` row (dangling soft ref).
* **Invariant audit** — sample/verify ``markdown[start:end] == exact_quote`` and
  ``quote_hash`` for anchored spans.

This is deterministic and read-only against cache.db (opened ``mode=ro``).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from seedgraph import anchor, cache_access
from seedgraph.fts.schema import assert_fts5_available


@dataclass
class SpanDoctorReport:
    """Structured result of the span/section/FTS reconcile sub-check."""

    fts5_available: bool = False
    spans_total: int = 0
    spans_stale: int = 0
    spans_missing: int = 0
    spans_orphaned: int = 0
    dangling_section_refs: int = 0
    invariant_failures: int = 0
    ok: bool = False
    messages: list[str] = field(default_factory=list)


def run(
    project_conn: sqlite3.Connection,
    cache_conn: sqlite3.Connection,
    cache_root: Path | str | None,
) -> SpanDoctorReport:
    """Run the span/section/FTS reconcile sub-check; return a :class:`SpanDoctorReport`.

    Never raises on stale/missing cache rows — it classifies and reports. The milestone
    expects FTS5 present and zero dangling span/section references on a freshly indexed
    corpus.
    """
    report = SpanDoctorReport()

    # (a) FTS5 availability — fail loud requirement, classified here (not raised).
    try:
        assert_fts5_available(project_conn)
        report.fts5_available = True
    except RuntimeError as exc:  # pragma: no cover - depends on build
        report.fts5_available = False
        report.messages.append(f"FTS5 unavailable: {exc}")

    report.spans_total = project_conn.execute(
        "SELECT COUNT(*) FROM evidence_spans"
    ).fetchone()[0]
    report.spans_stale = project_conn.execute(
        "SELECT COUNT(*) FROM evidence_spans WHERE anchor_status = 'stale'"
    ).fetchone()[0]
    report.spans_orphaned = project_conn.execute(
        "SELECT COUNT(*) FROM evidence_spans WHERE anchor_status = 'orphaned'"
    ).fetchone()[0]

    # (b) Dangling soft section_id refs — a span whose section_id no longer resolves.
    report.dangling_section_refs = project_conn.execute(
        "SELECT COUNT(*) FROM evidence_spans e "
        "WHERE e.section_id IS NOT NULL AND NOT EXISTS "
        "(SELECT 1 FROM document_sections d WHERE d.section_id = e.section_id)"
    ).fetchone()[0]

    # (c) Cross-db reconcile (missing markdown) + invariant audit over anchored spans.
    rows = project_conn.execute(
        "SELECT span_id, markdown_id, markdown_hash, source_file_id, source_file_hash, "
        "       start_char, end_char, exact_quote, quote_hash, anchor_status "
        "FROM evidence_spans"
    ).fetchall()
    for row in rows:
        (
            _span_id,
            markdown_id,
            markdown_hash,
            source_file_id,
            source_file_hash,
            start_char,
            end_char,
            exact_quote,
            quote_hash,
            anchor_status,
        ) = tuple(row)
        try:
            md = cache_access.read_markdown(cache_conn, cache_root, markdown_id)
        except (sqlite3.Error, ValueError, OSError):
            md = None
        if md is None:
            # Lineage check: a newer markdown still present => stale, else missing.
            current = cache_access.current_markdown_for_source(
                cache_conn, source_file_id, source_file_hash
            )
            if current is None:
                report.spans_missing += 1
            continue
        if anchor_status == "anchored":
            actual = md.text[start_char:end_char]
            if actual != exact_quote or anchor.quote_hash(exact_quote) != quote_hash:
                report.invariant_failures += 1

    report.ok = (
        report.fts5_available
        and report.dangling_section_refs == 0
        and report.invariant_failures == 0
        and report.spans_missing == 0
    )
    if report.ok:
        report.messages.append(
            f"{report.spans_total} span(s) ok; FTS5 present; zero dangling section refs"
        )
    return report
