"""Quote re-matching after a re-conversion (design §5.5 steps 4-5).

:func:`match_quote` is the pure ladder. For one old quote and the new markdown it tries,
in order, and stops at the first **unique** match (never a guess among duplicates):

1. ``exact`` - the quote verbatim;
2. ``folded`` - the quote folded through the shared mojibake table
   (:func:`seedgraph.library.mojibake.fold`): old quotes copied the garbage, so the raw
   string cannot match clean OCR text;
3. ``whitespace`` - the *folded* quote with whitespace runs collapsed, matched against
   the new text with its whitespace runs collapsed (old quotes from garbled text need
   both fixes: ``e¢ ciency`` plus different line breaks).

The located slice of the NEW text (never the old quote) is what a carried note anchors.

:func:`rematch` applies the ladder to every anchored quote of a paper's current note and
either carries the note to the new markdown (all re-match) or offers a re-note.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from .. import anchor
from ..notes.format import ClaimDraft
from ..notes.save import NoteProvenance, current_note_id, save_note
from ..notes.spans import read_markdown
from . import attention
from .mojibake import fold
from .store import Library, LibraryError

MatchMethod = Literal["exact", "folded", "whitespace"]

RENOTE_KIND = "renote_offered"
REMATCH_MODE = "rematch"

_WS_RE = re.compile(r"\s+")


# --- the ladder ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuoteMatch:
    start: int
    end: int
    method: MatchMethod


def _collapse(text: str) -> tuple[str, list[int]]:
    """``text`` with each whitespace run replaced by one space, and for each kept
    character its index in ``text``."""
    out: list[str] = []
    index: list[int] = []
    pos = 0
    for m in _WS_RE.finditer(text):
        for i in range(pos, m.start()):
            out.append(text[i])
            index.append(i)
        out.append(" ")
        index.append(m.start())
        pos = m.end()
    for i in range(pos, len(text)):
        out.append(text[i])
        index.append(i)
    return "".join(out), index


def match_quote(quote: str, new_text: str) -> QuoteMatch | None:
    """Locate ``quote`` in ``new_text`` by the ladder above, or ``None``."""
    if not quote:
        return None
    hit = anchor.locate_quote(new_text, quote)
    if hit is not None:
        return QuoteMatch(hit[0], hit[1], "exact")
    folded = fold(quote)
    if folded != quote:
        hit = anchor.locate_quote(new_text, folded)
        if hit is not None:
            return QuoteMatch(hit[0], hit[1], "folded")
    needle = _WS_RE.sub(" ", folded).strip()
    if not needle:
        return None
    collapsed, index = _collapse(new_text)
    hit = anchor.locate_quote(collapsed, needle)
    if hit is None:
        return None
    start = index[hit[0]]
    end = index[hit[1] - 1] + 1  # the needle is stripped: its last character is not a space
    return QuoteMatch(start, end, "whitespace")


# --- the paper's current note -------------------------------------------------------------

RematchStatus = Literal["no_note", "not_on_old_markdown", "carried", "renote_offered"]


@dataclass(frozen=True)
class QuoteRematch:
    claim_id: str
    old_quote: str
    method: MatchMethod | None  # None: no unique match
    new_quote: str | None


@dataclass(frozen=True)
class RematchResult:
    paper_id: str
    old_markdown_id: str
    new_markdown_id: str
    status: RematchStatus
    old_note_id: str | None
    previously_anchored: int
    rematched: int
    note_id: str | None  # the carried 'rematch' version
    attention_item_id: str | None  # the renote_offered item
    quotes: tuple[QuoteRematch, ...] = ()

    @property
    def match_rate(self) -> float:
        return 1.0 if self.previously_anchored == 0 else self.rematched / self.previously_anchored

    def as_dict(self) -> dict:
        return {
            "paper_id": self.paper_id, "old_markdown_id": self.old_markdown_id,
            "new_markdown_id": self.new_markdown_id, "status": self.status,
            "old_note_id": self.old_note_id, "previously_anchored": self.previously_anchored,
            "rematched": self.rematched, "match_rate": round(self.match_rate, 4),
            "note_id": self.note_id, "attention_item_id": self.attention_item_id,
            "quotes": [{"claim_id": q.claim_id, "old_quote": q.old_quote, "method": q.method,
                        "new_quote": q.new_quote} for q in self.quotes],
        }


@dataclass
class _CarriedDraft(ClaimDraft):
    """A claim carried to the new markdown: ``save_note`` anchors every ``exact_quotes``."""

    exact_quotes: tuple[str, ...] = ()


_CLAIM_COLUMNS = ("claim_id", "claim_type", "field_key", "status", "epistemic_type", "claim_subtype",
                  "normalized_label", "claim_text", "assertion_status", "inferred_explanation",
                  "confidence", "plain_words", "source_chunk_index")


def _swap_quotes(raw: str | None, swaps: dict[str, str]) -> str | None:
    """``raw_note_json`` with each string equal to an old quote replaced by its new
    slice; unchanged when it is not JSON."""
    if not raw or not swaps:
        return raw
    try:
        data = json.loads(raw)
    except ValueError:
        return raw

    def walk(node):
        if isinstance(node, str):
            return swaps.get(node, node)
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()}
        return node

    return json.dumps(walk(data), ensure_ascii=False)


def rematch(lib: Library, paper_id: str, old_markdown_id: str, new_markdown_id: str) -> RematchResult:
    """Re-match the anchored quotes of the paper's current note on ``new_markdown_id``.

    Every quote re-matches: the note is carried to the new markdown as a new version
    (``extraction_mode = 'rematch'``, parent run = the old run, same claim text, anchors
    on the new spans). Otherwise the old version stays current on its old markdown and a
    ``renote_offered`` attention item records the match rate. Nothing is re-noted.

    The paper's primary file must already point at ``new_markdown_id`` for a carry
    (``save_note``'s staleness check). A note with no anchored quotes is carried. The
    current note must be on ``old_markdown_id`` (else ``not_on_old_markdown``, nothing
    done). Must not be called inside an open library transaction.
    """
    conn = lib.conn
    note_id = current_note_id(conn, paper_id)
    base = dict(paper_id=paper_id, old_markdown_id=old_markdown_id, new_markdown_id=new_markdown_id,
                old_note_id=note_id, note_id=None, attention_item_id=None)
    if note_id is None:
        return RematchResult(status="no_note", previously_anchored=0, rematched=0, **base)
    note = conn.execute(
        "SELECT n.markdown_id, n.extraction_run_id, n.archetype, n.note_text, n.raw_note_json, "
        "r.provider, r.model_name, r.access_mode, r.chunk_count "
        "FROM structured_notes n JOIN extraction_runs r ON r.extraction_run_id = n.extraction_run_id "
        "WHERE n.note_id = ?", (note_id,)).fetchone()
    if note[0] != old_markdown_id:
        return RematchResult(status="not_on_old_markdown", previously_anchored=0, rematched=0, **base)
    new_text = read_markdown(conn, new_markdown_id)
    if new_text is None:
        raise LibraryError(f"rematch: markdown {new_markdown_id} is missing from the library")

    claims = conn.execute(
        f"SELECT {', '.join(_CLAIM_COLUMNS)} FROM extracted_claims WHERE structured_note_id = ? "
        "ORDER BY rowid", (note_id,)).fetchall()
    quotes: list[QuoteRematch] = []
    new_quotes: dict[str, list[str]] = {}
    for row in claims:
        spans = conn.execute(
            "SELECT es.exact_quote FROM claim_spans cs JOIN evidence_spans es ON es.span_id = cs.span_id "
            "WHERE cs.claim_id = ? AND es.anchor_status = 'anchored' ORDER BY cs.rank", (row[0],)).fetchall()
        for (old_quote,) in spans:
            m = match_quote(old_quote, new_text)
            new_quote = new_text[m.start:m.end] if m else None
            quotes.append(QuoteRematch(row[0], old_quote, m.method if m else None, new_quote))
            if new_quote is not None:
                new_quotes.setdefault(row[0], []).append(new_quote)
    anchored = len(quotes)
    matched = sum(1 for q in quotes if q.new_quote is not None)

    def offer() -> RematchResult:
        pct = round(100 * matched / anchored) if anchored else 100
        item = attention.raise_item(
            lib, RENOTE_KIND, {"paper_id": paper_id, "markdown_id": new_markdown_id},
            paper_id=paper_id, hides_notes=False,
            detail=(f"{matched}/{anchored} quotes re-matched ({pct}%) on the new markdown; the old note "
                    "stays current on its old markdown; a notes batch is offered for this paper"))
        return RematchResult(status="renote_offered", previously_anchored=anchored, rematched=matched,
                             quotes=tuple(quotes), **{**base, "attention_item_id": item.item_id})

    if matched < anchored:
        return offer()

    new_hash = conn.execute("SELECT markdown_hash FROM markdown_documents WHERE markdown_id = ?",
                            (new_markdown_id,)).fetchone()[0]
    drafts = [_CarriedDraft(**{k: row[k] for k in _CLAIM_COLUMNS[1:]},
                            exact_quotes=tuple(new_quotes.get(row[0], ())))
              for row in claims]
    swaps = {q.old_quote: q.new_quote for q in quotes if q.new_quote and q.new_quote != q.old_quote}
    try:
        with lib.write() as wconn:
            saved = _carry(wconn, paper_id, new_markdown_id, new_hash, drafts, note, swaps, note_id)
    except _Downgraded:
        return offer()  # a carried claim would lose its anchors: the save is rolled back
    return RematchResult(status="carried", previously_anchored=anchored, rematched=matched,
                         quotes=tuple(quotes), **{**base, "note_id": saved.note_id})


class _Downgraded(Exception):
    """A carried found claim did not anchor on the new markdown."""


def _carry(wconn, paper_id: str, new_markdown_id: str, new_hash: str, drafts, note, swaps, note_id: str):
    """Save the carried version and link its parent run (inside the caller's write)."""
    saved = save_note(
        wconn, paper_id=paper_id, markdown_id=new_markdown_id, markdown_hash=new_hash,
        drafts=drafts, note_text=note[3], archetype=note[2],
        raw_note_json=_swap_quotes(note[4], swaps) or "{}",
        provenance=NoteProvenance(client=note[5], model=note[6], access_mode=note[7] or "agent_session"),
        expected_current=note_id, extraction_mode=REMATCH_MODE, part_count=note[8] or 1)
    if saved.downgraded:
        raise _Downgraded(saved.downgraded)
    wconn.execute("UPDATE extraction_runs SET parent_extraction_run_id = ? WHERE extraction_run_id = ?",
                  (note[1], saved.extraction_run_id))
    return saved
