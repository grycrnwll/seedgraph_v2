"""Save a normalised note into ``library.db`` as a new version (design §3; port of v2
``extraction.runner.save_normalized_note``).

:func:`save_note` runs on an open library connection **inside the caller's write
transaction** (``Library.write()`` / ``store.transaction``, which take SQLite's writer
lock with ``BEGIN IMMEDIATE``). It never begins, commits or rolls back. An exception
leaves the caller's transaction to roll back, so nothing is half-written.

In one go it:
1. checks staleness: the paper's primary file must still point at ``markdown_id``
   (else :class:`NoteStale`, "the paper was re-converted; create a new batch");
2. compares ``expected_current`` with the paper's current note (``note_current``)
   when given (else :class:`NoteConflict`);
3. assigns ``version_no = MAX(version_no) + 1`` over every version of
   ``(paper_id, schema_id)``, labelled ones included (the UNIQUE key covers them);
4. writes the extraction run, the note, one claim row per draft, anchors every found
   quote through :func:`seedgraph.notes.spans.ensure_span` (a found claim with no
   anchored quote is downgraded to ``ambiguous``, as v2), ``claim_spans``, and the
   ``claim_fts`` (with ``plain_words``) and ``note_fts`` rows.

It never reads or writes ``note_overlays`` (stage 3).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..errors import SeedgraphError
from ..ids import new_id
from ..vocab import StatusValue
from .format import PROMPT_VERSION, SCHEMA_ID, SCHEMA_VERSION, ClaimDraft
from .spans import ensure_span, read_markdown

_FOUND = StatusValue.found.value
_AMBIGUOUS = StatusValue.ambiguous.value
DOWNGRADE_MESSAGE = "quote could not be anchored verbatim; downgraded to ambiguous"


class NoteSaveError(SeedgraphError):
    """The note could not be saved."""


class NoteConflict(NoteSaveError):
    """The paper's current note changed after the batch was created."""


class NoteStale(NoteSaveError):
    """The paper no longer points at the markdown the note was made from."""


class _Unchecked:
    def __repr__(self) -> str:
        return "UNCHECKED"


#: Default for ``expected_current``: skip the compare-and-activate check.
UNCHECKED = _Unchecked()


@dataclass(frozen=True)
class NoteProvenance:
    """Who made the note. ``client`` is stored as the run's ``provider``."""

    client: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    access_mode: str = "agent_session"


@dataclass(frozen=True)
class SavedNote:
    note_id: str
    extraction_run_id: str
    version_no: int
    claim_count: int
    span_count: int
    quotes_found: int
    quotes_anchored: int
    downgraded: list[str] = field(default_factory=list)  # field_keys made ambiguous


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_note_id(conn: sqlite3.Connection, paper_id: str, schema_id: str = SCHEMA_ID) -> str | None:
    """The paper's current note id (``note_current``), or ``None``."""
    row = conn.execute(
        "SELECT note_id FROM note_current WHERE paper_id = ? AND schema_id IS ?",
        (paper_id, schema_id),
    ).fetchone()
    return row[0] if row else None


def next_version_no(conn: sqlite3.Connection, paper_id: str, schema_id: str = SCHEMA_ID) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(version_no), 0) + 1 FROM structured_notes "
        "WHERE paper_id = ? AND schema_id IS ?",
        (paper_id, schema_id),
    ).fetchone()
    return int(row[0])


def save_note(
    conn: sqlite3.Connection,
    *,
    paper_id: str,
    markdown_id: str,
    markdown_hash: str,
    drafts: Sequence[ClaimDraft],
    note_text: str,
    archetype: str,
    raw_note_json: str,
    provenance: NoteProvenance,
    expected_current: str | None | _Unchecked = UNCHECKED,
    extraction_mode: str = "agent_whole",
    part_count: int = 1,
    batch_id: str | None = None,
    schema_id: str = SCHEMA_ID,
) -> SavedNote:
    """Save one complete note as a new version of the paper's note. See the module doc."""
    if not conn.in_transaction:
        raise NoteSaveError("save_note must run inside the caller's write transaction")

    primary = conn.execute(
        "SELECT pf.markdown_id, m.markdown_hash FROM paper_files pf "
        "LEFT JOIN markdown_documents m ON m.markdown_id = pf.markdown_id "
        "WHERE pf.paper_id = ? AND pf.role = 'primary'",
        (paper_id,),
    ).fetchone()
    if primary is None or primary[0] != markdown_id or primary[1] != markdown_hash:
        raise NoteStale(
            f"paper {paper_id} no longer points at markdown {markdown_id}: "
            "the paper was re-converted; create a new batch"
        )
    if expected_current is not UNCHECKED:
        actual = current_note_id(conn, paper_id, schema_id)
        if actual != expected_current:
            raise NoteConflict(
                f"the current note of {paper_id} changed since the batch was created "
                f"(expected {expected_current}, found {actual})"
            )
    text = read_markdown(conn, markdown_id)
    if text is None:
        raise NoteSaveError(f"markdown {markdown_id} is missing from the library blob store")

    version_no = next_version_no(conn, paper_id, schema_id)
    run_id = new_id("extr")
    note_id = new_id("note")
    now = _now()
    access_class = conn.execute(
        "SELECT s.access_class FROM markdown_documents m "
        "JOIN source_files s ON s.source_file_id = m.source_file_id WHERE m.markdown_id = ?",
        (markdown_id,),
    ).fetchone()[0]

    conn.execute(
        "INSERT INTO extraction_runs (extraction_run_id, paper_id, markdown_id, markdown_hash, "
        "schema_id, schema_version, prompt_version, model_name, provider, access_mode, "
        "access_class, run_status, input_tokens, output_tokens, run_id, created_at, "
        "extraction_mode, chunk_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'success', ?, ?, ?, ?, ?, ?)",
        (run_id, paper_id, markdown_id, markdown_hash, schema_id, SCHEMA_VERSION, PROMPT_VERSION,
         provenance.model, provenance.client, provenance.access_mode, access_class,
         provenance.input_tokens, provenance.output_tokens, batch_id, now, extraction_mode,
         part_count),
    )
    conn.execute(
        "INSERT INTO structured_notes (note_id, extraction_run_id, paper_id, markdown_id, "
        "markdown_hash, schema_id, schema_version, prompt_version, archetype, access_class, "
        "raw_note_json, note_text, created_at, version_no) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (note_id, run_id, paper_id, markdown_id, markdown_hash, schema_id, SCHEMA_VERSION,
         PROMPT_VERSION, archetype, access_class, raw_note_json, note_text, now, version_no),
    )

    span_count = quotes_found = quotes_anchored = 0
    downgraded: list[str] = []
    for draft in drafts:
        claim_id = new_id("claim")
        status = draft.status
        explanation = draft.inferred_explanation
        span_ids: list[str] = []
        if status == _FOUND:
            quotes_found += 1
            quotes = getattr(draft, "exact_quotes", None) or (
                [draft.exact_quote] if draft.exact_quote else [])
            for quote in quotes:
                span_id = ensure_span(conn, markdown_id=markdown_id, exact_quote=quote,
                                      markdown_text=text)
                if span_id is not None and span_id not in span_ids:
                    span_ids.append(span_id)
            if span_ids:
                quotes_anchored += 1
            else:
                status = _AMBIGUOUS
                explanation = f"{explanation} | {DOWNGRADE_MESSAGE}" if explanation else DOWNGRADE_MESSAGE
                downgraded.append(draft.field_key)
        conn.execute(
            "INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, "
            "paper_id, claim_type, claim_subtype, field_key, normalized_label, claim_text, "
            "status, epistemic_type, assertion_status, inferred_explanation, confidence, "
            "access_class, created_at, source_chunk_index, source_extraction_run_id, plain_words) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (claim_id, note_id, run_id, paper_id, draft.claim_type, draft.claim_subtype,
             draft.field_key, draft.normalized_label, draft.claim_text, status,
             draft.epistemic_type, draft.assertion_status, explanation, draft.confidence,
             access_class, now, draft.source_chunk_index, None, draft.plain_words),
        )
        for rank, span_id in enumerate(span_ids):
            conn.execute(
                "INSERT INTO claim_spans (claim_id, span_id, rank, created_at) VALUES (?, ?, ?, ?)",
                (claim_id, span_id, rank, now),
            )
            span_count += 1
        conn.execute(
            "INSERT INTO claim_fts (claim_id, normalized_label, claim_text, plain_words) "
            "VALUES (?, ?, ?, ?)",
            (claim_id, draft.normalized_label, draft.claim_text, draft.plain_words),
        )
    conn.execute("INSERT INTO note_fts (note_id, note_text) VALUES (?, ?)", (note_id, note_text))

    return SavedNote(
        note_id=note_id, extraction_run_id=run_id, version_no=version_no,
        claim_count=len(drafts), span_count=span_count, quotes_found=quotes_found,
        quotes_anchored=quotes_anchored, downgraded=downgraded,
    )
