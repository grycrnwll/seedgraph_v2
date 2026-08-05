"""Phase 3b — per-work parsed-bibliography orchestrator (two-stage lifecycle).

:func:`build_parsed_edges` is the single per-work entry point. It runs the
two-stage model that fixes both round-1 must-fixes (decision r2-2 #4):

- **Stage A — parse/resolve**, keyed on ``(citing_work_id, markdown_hash)``. The
  expensive parse + provider resolution. **Skipped** when the work's stored
  ``reference_entries.markdown_hash`` equals the current markdown hash and not
  ``--force``. On reparse it runs the FK-safe delete-and-reinsert lifecycle.
- **Stage B — project edges into the target run**, keyed on
  ``(citing_work_id, run_id)``. Projects every current ``resolved``
  ``reference_entries`` row into a ``parsed_bibliography`` ``citation_edges`` row
  under ``run_id`` via ``edges.write_edge(..., reference_id=...)``. **Always
  runs**, even when Stage A was skipped, so a fresh provider walk minting run R2
  over unchanged markdown still gets the full parsed tier (criterion 7b).

Connection model (§6.4 — pinned): the function pins the single DBAPI connection
underlying the project ``session`` via ``db.adapter.raw_conn(session)`` and opens
**one** transaction. ``edges.write_edge`` / the lifecycle ``DELETE``s run on that
raw ``conn``; ``identity.upsert_work`` and ``review.enqueue_in_session`` run on the
``session`` bound to that **same** connection. Commit once at the end; on any
exception roll back the whole work (atomicity — criterion 7 / §6.4).

FK-safe reparse ordering (§7; ``PRAGMA foreign_keys=ON``): because
``citation_edges.reference_id → reference_entries.reference_id`` is enforced, a
reparse deletes this work's ``parsed_bibliography`` edges **run-agnostically
(all runs)** *before* deleting its ``reference_entries`` — touching only this
one source work's parsed rows, never ``provider_reference`` edges, never other
works (criterion 7c; the round-1 cross-run FK bug).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class ParsedBibResult:
    """Per-work outcome of :func:`build_parsed_edges`.

    ``reparsed`` is True iff Stage A ran (markdown changed / ``--force``).
    ``entries`` is the current ``reference_entries`` count for the work;
    ``resolved``/``ambiguous``/``suspect`` break it down by resolution status.
    ``edges_written`` is the number of ``parsed_bibliography`` edges projected
    into ``run_id`` **this invocation** (Stage B). ``skipped_reason`` is
    ``'no_markdown'`` (no ``work_source_files`` row) or ``'no_references'`` (no
    ``section_kind='references'`` range); ``None`` means the work was processed.
    """

    work_id: str
    reparsed: bool
    entries: int
    resolved: int
    ambiguous: int
    suspect: int
    edges_written: int
    skipped_reason: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parsed_fields_json(entry) -> str:
    return json.dumps(
        {
            "first_author": entry.first_author,
            "year": entry.year,
            "title": entry.title,
            "doi": entry.doi,
            "arxiv": entry.arxiv,
            "ordinal": entry.ordinal,
            "continued_author": entry.continued_author,
        }
    )


def _incoming_from(entry, record: Optional[dict]) -> dict:
    """Build an identity ``upsert_work`` mapping from the entry + chosen record."""
    from .resolve import record_to_identity

    incoming: dict = dict(record_to_identity(record)) if record else {}
    # The entry's own strong ids win when the record lacks them.
    if entry.doi and "doi" not in incoming:
        incoming["doi"] = entry.doi
    if entry.arxiv and "arxiv" not in incoming:
        incoming["arxiv"] = entry.arxiv
    if "title" not in incoming and entry.title:
        incoming["title"] = entry.title
    if "year" not in incoming and entry.year is not None:
        incoming["year"] = entry.year
    if "authors" not in incoming and entry.first_author:
        incoming["authors"] = [entry.first_author]
    return {k: v for k, v in incoming.items() if v is not None}


def ensure_metadata_only(session, work_id: str) -> None:
    """Ensure a ``metadata_only`` membership row for an auto-vivified target.

    Never downgrades an already-included work (only creates the row when absent),
    so a parsed edge to an in-corpus included work keeps it included.
    """
    from ..db.project_models import ProjectDocument

    doc = session.get(ProjectDocument, work_id)
    if doc is not None:
        return
    now = _now()
    session.add(
        ProjectDocument(
            work_id=work_id,
            inclusion_status="metadata_only",
            inclusion_reason="citation_walk",
            is_seed=0,
            access_status=None,
            created_at=now,
            updated_at=now,
        )
    )
    session.flush()


def _load_reference_ranges(conn, markdown_id: str) -> list[tuple[int, int, str]]:
    rows = conn.execute(
        "SELECT start_char, end_char, heading_text FROM document_sections "
        "WHERE markdown_id = ? AND section_kind = 'references' ORDER BY ordinal",
        (markdown_id,),
    ).fetchall()
    return [(int(r[0]), int(r[1]), r[2] or "References") for r in rows]


#: The references-heading cues the NON-ATX fallback recognizes (v1 ``REFERENCE_HEADINGS``,
#: ``extract/bib_parser.py:95-105``).
_FALLBACK_HEADINGS_ALT = "|".join(
    re.escape(h)
    for h in ("references", "bibliography", "works cited", "literature cited", "reference list")
)

#: A BOLD/italic-wrapped references heading on its OWN line: ``**References**`` /
#: ``*REFERENCES*`` / ``__Bibliography__``, optionally preceded by a marker page-anchor
#: span and/or an ``N.`` section number and followed by a colon. FULL-LINE anchored so
#: an inline "... references ..." sentence never matches.
_BOLD_REFERENCES_LINE_RE = re.compile(
    r"(?im)^[ \t]{0,3}"
    r"(?:<span[^>]*>\s*</span>[ \t]*)?"  # optional marker page-anchor span
    r"(?:\*{1,3}|_{1,3})[ \t]*"  # REQUIRED bold/italic open (the non-ATX shape)
    r"(?:\d{1,3}[.)]?[ \t]+)?"  # optional 'N.' / 'N)' section number
    rf"(?:{_FALLBACK_HEADINGS_ALT})"
    r"[ \t]*(?:\*{1,3}|_{1,3})[ \t]*:?[ \t]*$"  # REQUIRED bold/italic close + optional colon
)

#: A BARE references TITLE line (no ATX ``#``, no emphasis): exactly one of the known
#: heading texts, optional trailing colon. FULL-LINE anchored.
_BARE_REFERENCES_LINE_RE = re.compile(
    rf"(?im)^[ \t]{{0,3}}(?:{_FALLBACK_HEADINGS_ALT})[ \t]*:?[ \t]*$"
)

#: Label normalization for a synthesized fallback range (anchor span / emphasis /
#: leading number / trailing colon → a clean plain heading word).
_FALLBACK_LABEL_STRIPS = (
    re.compile(r"<span[^>]*>\s*</span>"),
    re.compile(r"[*_`]+"),
    re.compile(r"^\s*\d{1,3}[.)]?\s+"),
)


def _fallback_reference_ranges(markdown: str) -> list[tuple[int, int, str]]:
    """Synthesize references ranges from NON-ATX heading lines (citations-only recovery).

    The sections parser is ATX-only (decisions 31/52), so a bold-only ``**REFERENCES**``
    or a bare ``References`` title line never becomes a ``document_sections`` row and the
    whole bibliography would be skipped as ``no_references``. When zero references-kind
    ranges exist, this narrow full-line matcher (ported from v1 ``_HEADING_RE`` shapes,
    minus the ATX branch the parser owns) recovers those documents: each matched heading
    opens a range running to the NEXT such heading or the end of the document. The heading
    LINE itself is excluded from the range body (so it never surfaces as a spurious entry)
    and its normalized text is carried as ``section_label``. Returns ``[]`` when no
    non-ATX references heading is present.
    """
    matches = list(_BOLD_REFERENCES_LINE_RE.finditer(markdown))
    if not matches:
        matches = list(_BARE_REFERENCES_LINE_RE.finditer(markdown))
    if not matches:
        return []
    ranges: list[tuple[int, int, str]] = []
    for index, match in enumerate(matches):
        body_start = match.end()  # exclude the heading LINE from the parsed body
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        label = match.group(0).strip()
        for strip_re in _FALLBACK_LABEL_STRIPS:
            label = strip_re.sub("", label)
        label = label.strip().rstrip(":").strip() or "References"
        ranges.append((body_start, end, label))
    return ranges


def _ensure_sections(conn, cache_db, *, work_id: str, markdown_id: str) -> None:
    """Build+store document_sections for this markdown if none exist yet.

    ``cite parse`` locates the references region from phase_3's document_sections
    (``section_kind='references'``); when ``sections build`` was never run those rows
    are absent and parsing silently yields ``no_references``. Building them on demand
    (the same ``parse_sections``/``replace_sections`` the ``sections build`` CLI runs)
    makes parse self-sufficient, so ``no_references`` now means genuinely no references
    section. Idempotent: ``markdown_id`` is content-addressed, so existing rows are for
    this exact content and we skip. Writes on ``conn`` (the caller's session txn); no
    commit here.
    """
    exists = conn.execute(
        "SELECT 1 FROM document_sections WHERE markdown_id = ? LIMIT 1",
        (markdown_id,),
    ).fetchone()
    if exists is not None:
        return
    from .. import cache_access
    from ..sections.parser import parse_sections
    from ..sections.store import replace_sections

    cache_conn = cache_access.open_cache_ro(cache_db)
    try:
        md = cache_access.read_markdown(cache_conn, cache_db, markdown_id)
    finally:
        cache_conn.close()
    if md is None:
        return  # markdown bytes unresolvable — leave to the existing no-ranges path
    sections = parse_sections(
        md.text,
        markdown_id=markdown_id,
        markdown_hash=md.markdown_hash,
        source_file_id=md.source_file_id,
        source_file_hash=md.source_file_hash,
        work_id=work_id,
    )
    replace_sections(conn, markdown_id, sections)


def _read_markdown_text(cache_db, markdown_id: str) -> Optional[str]:
    from .. import cache_access

    cache_conn = cache_access.open_cache_ro(cache_db)
    try:
        md = cache_access.read_markdown(cache_conn, cache_db, markdown_id)
    finally:
        cache_conn.close()
    return md.text if md is not None else None


def build_parsed_edges(
    handle,
    cache_db: Path,
    providers,
    identity,
    *,
    work_id: str,
    run_id: str,
    force: bool = False,
) -> ParsedBibResult:
    """Run the two-stage parsed-bibliography lifecycle for one work, atomically.

    ``handle`` = a phase_5 ``ProjectHandle``; ``cache_db`` = the seedgraph HOME root
    (``cache.db`` opened read-only for markdown bytes); ``providers`` = the phase_5b
    provider chain; ``identity`` = the phase_5 identity module. ``run_id`` is the
    provider walk's run (decision r2-2 #4). ``force`` re-parses even when
    ``markdown_hash`` is unchanged.

    See the module docstring for the §6.4 connection model and §7 lifecycle. Returns
    the :class:`ParsedBibResult` summary.
    """
    from sqlmodel import Session

    from ..acquisition.bridge import resolve_work_markdown
    from ..db.adapter import raw_conn
    from ..ids import new_id
    from .bib_parser import parse_references
    from .edges import write_edge
    from .resolve import resolve_references

    with Session(handle.engine, expire_on_commit=False) as session:
        conn = raw_conn(session)

        resolved_md = resolve_work_markdown(session, work_id=work_id)
        if resolved_md is None:
            return ParsedBibResult(
                work_id=work_id, reparsed=False, entries=0, resolved=0,
                ambiguous=0, suspect=0, edges_written=0, skipped_reason="no_markdown",
            )
        markdown_id, markdown_hash = resolved_md

        # --- Stage A decision (markdown-hash idempotency guard) ----------------
        stored = conn.execute(
            "SELECT markdown_hash FROM reference_entries WHERE citing_work_id = ? LIMIT 1",
            (work_id,),
        ).fetchone()
        run_stage_a = force or stored is None or stored[0] != markdown_hash
        skipped_reason: str | None = None

        try:
            if run_stage_a:
                # (a) FK-safe edge clear: this work's parsed edges across ALL runs,
                #     BEFORE its reference_entries are deleted (must-fix #1).
                conn.execute(
                    "DELETE FROM citation_edges "
                    "WHERE source_work_id = ? AND provenance = 'parsed_bibliography'",
                    (work_id,),
                )
                # (b) Delete the work's reference_entries.
                conn.execute(
                    "DELETE FROM reference_entries WHERE citing_work_id = ?", (work_id,)
                )
                # (c) Load references-kind char ranges (phase_3 document_sections).
                #     Build them on demand when `sections build` was never run, so
                #     `no_references` reflects the doc, not a missing prerequisite.
                _ensure_sections(conn, cache_db, work_id=work_id, markdown_id=markdown_id)
                ranges = _load_reference_ranges(conn, markdown_id)
                text = _read_markdown_text(cache_db, markdown_id)
                # Non-ATX references fallback: when phase_3 yields zero references-kind
                # ranges (a bold-only/bare heading the ATX-only sections parser cannot
                # see), synthesize ranges from a narrow full-line matcher so the whole
                # bibliography is not silently skipped as `no_references`.
                if not ranges and text:
                    ranges = _fallback_reference_ranges(text)
                if not ranges:
                    skipped_reason = "no_references"
                # (d) Parse + INSERT reference_entries (stamped current anchors).
                entries = parse_references(text, ranges) if (text and ranges) else []
                ref_ids: list[str] = []
                now = _now()
                for entry in entries:
                    rid = new_id("ref")
                    conn.execute(
                        "INSERT INTO reference_entries "
                        "(reference_id, citing_work_id, raw_reference_text, parsed_fields_json, "
                        " resolved_work_id, resolution_status, resolution_source, confidence, "
                        " markdown_id, markdown_hash, section_label, created_at) "
                        "VALUES (?, ?, ?, ?, NULL, 'unresolved', NULL, NULL, ?, ?, ?, ?)",
                        (
                            rid, work_id, entry.raw, _parsed_fields_json(entry),
                            markdown_id, markdown_hash, entry.section_label, now,
                        ),
                    )
                    ref_ids.append(rid)

                # (e) Resolve + set resolution columns; upsert resolved targets;
                #     enqueue ambiguous/suspect.
                resolved = resolve_references(
                    entries, providers, identity, citing_work_id=work_id
                )
                for rid, rr in zip(ref_ids, resolved):
                    if rr.status == "resolved":
                        record = rr.candidates[0] if rr.candidates else None
                        incoming = _incoming_from(rr.entry, record)
                        work, _outcome = identity.upsert_work(session, incoming)
                        ensure_metadata_only(session, work.work_id)
                        conn.execute(
                            "UPDATE reference_entries SET resolved_work_id = ?, "
                            "resolution_status = 'resolved', resolution_source = ?, "
                            "confidence = ? WHERE reference_id = ?",
                            (work.work_id, rr.resolution_source, rr.confidence, rid),
                        )
                    elif rr.status in ("ambiguous", "suspect"):
                        conn.execute(
                            "UPDATE reference_entries SET resolution_status = ? "
                            "WHERE reference_id = ?",
                            (rr.status, rid),
                        )
                        _enqueue_resolution(
                            session, rr, reference_id=rid, citing_work_id=work_id,
                            run_id=run_id,
                        )
                    else:  # unresolved — kept raw (recall stays measurable)
                        conn.execute(
                            "UPDATE reference_entries SET resolution_status = 'unresolved' "
                            "WHERE reference_id = ?",
                            (rid,),
                        )

            # --- Stage B — project current resolved refs into run_id (ALWAYS) --
            resolved_rows = conn.execute(
                "SELECT reference_id, resolved_work_id, confidence FROM reference_entries "
                "WHERE citing_work_id = ? AND resolution_status = 'resolved' "
                "AND resolved_work_id IS NOT NULL",
                (work_id,),
            ).fetchall()
            edges_written = 0
            for rid, target, conf in resolved_rows:
                write_edge(
                    conn, source=work_id, target=target,
                    provenance="parsed_bibliography",
                    confidence=float(conf) if conf is not None else 0.0,
                    run_id=run_id, reference_id=rid,
                )
                edges_written += 1

            # --- Final per-work counts ----------------------------------------
            counts = {"resolved": 0, "ambiguous": 0, "suspect": 0, "unresolved": 0}
            for (status,) in conn.execute(
                "SELECT resolution_status FROM reference_entries WHERE citing_work_id = ?",
                (work_id,),
            ).fetchall():
                counts[status] = counts.get(status, 0) + 1
            total = sum(counts.values())

            session.commit()
        except Exception:
            session.rollback()
            raise

    return ParsedBibResult(
        work_id=work_id,
        reparsed=run_stage_a,
        entries=total,
        resolved=counts["resolved"],
        ambiguous=counts["ambiguous"],
        suspect=counts["suspect"],
        edges_written=edges_written,
        skipped_reason=skipped_reason,
    )


def _enqueue_resolution(session, rr, *, reference_id: str, citing_work_id: str, run_id: str) -> None:
    """Enqueue one ``citation_resolution`` review item (ambiguous / suspect).

    Validated by the ``CitationResolutionPayload`` variant of the ``ReviewPayload``
    union at the enqueue boundary (decision 80) and added in the SAME session/txn
    (atomic with the parsed tier — §6.4).
    """
    from ..project import review

    entry = rr.entry
    payload = {
        "kind": "citation_resolution",
        "status": rr.status,
        "citing_work_id": citing_work_id,
        "reference_id": reference_id,
        "raw": entry.raw,
        "title": entry.title,
        "year": entry.year,
        "first_author": entry.first_author,
        "doi": entry.doi,
        "arxiv": entry.arxiv,
        "candidates": list(rr.candidates),
        "rejected_record": rr.rejected_record,
        "reject_reason": rr.reject_reason,
        "run_id": run_id,
    }
    review.enqueue_in_session(
        session,
        "citation_resolution",
        target_type="reference_entry",
        target_id=reference_id,
        payload=payload,
    )
