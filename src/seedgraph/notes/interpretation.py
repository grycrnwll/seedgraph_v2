"""The current interpretation of a paper's note (stage 3 design §4.3).

:func:`current` is the **one read path** for the paper page, MCP ``paper_show``, MCP and
CLI search, and the views. It reads ``note_current`` (pinned to the overlay base for a
researcher-edited paper) and either the derived ``interpretation_claims`` (overlaid
papers) or the ``current_claims`` view (all others), so both kinds of paper come back
in the same shape.

Trust labels:
- ``paper_states``: an agent claim with an anchored quote (status ``found``);
- ``agent_inferred``: an agent claim without one (ambiguous, inferred, not_found);
- ``researcher``: any added claim and any edited claim. An edited claim carries the
  agent's ``original``; a researcher assertion is never the paper's words.

Audiences: ``agent`` keeps stage 1's rules (a paper with an open ``hides_notes``
attention item has no interpretation; labelled versions are never current). Hidden
claims are excluded for every audience except ``researcher`` with ``show_hidden``.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..library.attention import HIDDEN_PAPERS_SQL
from .format import ARRAY_FIELDS, SCALAR_FIELDS, SCHEMA_ID

if TYPE_CHECKING:
    from ..library.store import Library

Audience = Literal["agent", "researcher"]
AUDIENCES: tuple[str, ...] = ("agent", "researcher")
TRUST_LABELS: dict[str, str] = {
    "paper_states": "paper states",
    "agent_inferred": "agent inferred",
    "researcher": "researcher",
}
EDITED_NOTE = "edited from the agent's claim"
ASSERTION_NOTE = "researcher assertion — not the paper's words"
#: Field order for display: the note's scalars, then its arrays, then added claims.
FIELD_ORDER: dict[str, int] = {name: i for i, name in enumerate(
    [n for n, _ in SCALAR_FIELDS] + [n for n, _ in ARRAY_FIELDS] + ["added"])}

_FIELD_RE = re.compile(r"^(.*?)(?:\[(\d+)\])?$")


@dataclass(frozen=True)
class OriginalClaim:
    """The agent's version of an edited claim."""

    claim_type: str
    normalized_label: str | None
    claim_text: str | None


@dataclass(frozen=True)
class InterpretedClaim:
    claim_id: str
    paper_id: str
    claim_type: str
    claim_subtype: str | None
    field_key: str
    normalized_label: str | None
    claim_text: str | None
    plain_words: str | None
    status: str
    trust: str                      # paper_states | agent_inferred | researcher
    edited_by_researcher: bool
    kind: str                       # agent | quote | assertion
    hidden: bool
    span_id: str | None
    original: OriginalClaim | None  # set for an edited agent claim

    @property
    def trust_label(self) -> str:
        return TRUST_LABELS[self.trust]

    @property
    def trust_note(self) -> str | None:
        """The extra line the UI and agents show: "edited from the agent's claim" or the assertion warning."""
        if self.kind == "assertion":
            return ASSERTION_NOTE
        if self.edited_by_researcher:
            return EDITED_NOTE
        return None


@dataclass(frozen=True)
class Interpretation:
    paper_id: str
    schema_id: str | None
    note_id: str                   # note_current (the overlay base for an edited paper)
    extraction_run_id: str
    overlay_id: str | None
    version_no: int | None         # the overlay head version (None without an overlay)
    claims: tuple[InterpretedClaim, ...]
    researcher_notes: str | None
    note_text: str
    newer_extraction: bool         # note_latest is newer than note_current (replay offered)


def field_sort_key(field_key: str, claim_id: str) -> tuple[int, int, str]:
    """Display order: the note's field order, the array index, then the claim id."""
    m = _FIELD_RE.match(field_key)
    root, index = (m.group(1), int(m.group(2) or 0)) if m else (field_key, 0)
    return FIELD_ORDER.get(root, len(FIELD_ORDER)), index, claim_id


def _conn(lib: Library | sqlite3.Connection) -> sqlite3.Connection:
    return lib if isinstance(lib, sqlite3.Connection) else lib.conn


def current(lib: Library | sqlite3.Connection, paper_id: str, audience: Audience = "agent", *,
            show_hidden: bool = False, schema_id: str | None = SCHEMA_ID) -> Interpretation | None:
    """The paper's current interpretation for ``audience``, or ``None`` when it has no
    current note (or, for agents, when its notes are hidden by an attention item)."""
    if audience not in AUDIENCES:
        raise ValueError(f"unknown audience {audience!r}; expected one of {', '.join(AUDIENCES)}")
    if show_hidden and audience != "researcher":
        raise ValueError("show_hidden is the researcher's own toggle")
    conn = _conn(lib)
    if audience == "agent" and conn.execute(
            f"SELECT 1 FROM ({HIDDEN_PAPERS_SQL}) WHERE paper_id = ?", (paper_id,)).fetchone():
        return None
    note = conn.execute("SELECT note_id, extraction_run_id, note_text, version_no FROM note_current "
                        "WHERE paper_id = ? AND schema_id IS ?", (paper_id, schema_id)).fetchone()
    if note is None:
        return None
    latest = conn.execute("SELECT version_no FROM note_latest WHERE paper_id = ? AND schema_id IS ?",
                          (paper_id, schema_id)).fetchone()
    ov = conn.execute(
        "SELECT o.overlay_id, o.head_version_no, v.researcher_notes FROM note_overlays o "
        "JOIN overlay_versions v ON v.overlay_id = o.overlay_id AND v.version_no = o.head_version_no "
        "WHERE o.paper_id = ? AND o.schema_id IS ? AND o.base_note_id = ?",
        (paper_id, schema_id, note["note_id"])).fetchone()

    if ov is not None:
        rows = conn.execute(
            "SELECT ic.*, ec.claim_type AS o_type, ec.normalized_label AS o_label, ec.claim_text AS o_text "
            "FROM interpretation_claims ic LEFT JOIN extracted_claims ec ON ec.claim_id = ic.claim_id "
            "WHERE ic.overlay_id = ?" + ("" if show_hidden else " AND ic.hidden = 0"),
            (ov["overlay_id"],)).fetchall()
        note_text_row = conn.execute("SELECT note_text FROM interpretation_notes WHERE paper_id = ? "
                                     "AND schema_id IS ?", (paper_id, schema_id)).fetchone()
        note_text = note_text_row[0] if note_text_row else note["note_text"]
        researcher_notes = ov["researcher_notes"]
    else:
        rows = conn.execute("SELECT *, 0 AS hidden FROM current_claims WHERE note_id = ?",
                            (note["note_id"],)).fetchall()
        note_text, researcher_notes = note["note_text"], None

    claims = []
    for r in rows:
        edited = bool(r["edited"])
        claims.append(InterpretedClaim(
            claim_id=r["claim_id"], paper_id=r["paper_id"], claim_type=r["claim_type"],
            claim_subtype=r["claim_subtype"], field_key=r["field_key"], normalized_label=r["normalized_label"],
            claim_text=r["claim_text"], plain_words=r["plain_words"], status=r["status"], trust=r["trust"],
            edited_by_researcher=edited, kind=r["kind"], hidden=bool(r["hidden"]), span_id=r["span_id"],
            original=OriginalClaim(r["o_type"], r["o_label"], r["o_text"]) if edited else None))
    claims.sort(key=lambda c: field_sort_key(c.field_key, c.claim_id))
    return Interpretation(
        paper_id=paper_id, schema_id=schema_id, note_id=note["note_id"],
        extraction_run_id=note["extraction_run_id"], overlay_id=ov["overlay_id"] if ov else None,
        version_no=ov["head_version_no"] if ov else None, claims=tuple(claims),
        researcher_notes=researcher_notes, note_text=note_text,
        newer_extraction=latest is not None and latest[0] > note["version_no"])
