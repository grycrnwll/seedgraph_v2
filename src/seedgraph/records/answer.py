"""Recording an answer (design §3.3, §3.4, §3.6; slice R2).

:func:`record_answer` is the deep function behind the MCP ``record_answer`` tool (R3 wires
the tool). It takes the question, the answer, the cited evidence and the agent's own
judgement, kept apart, and in **one library transaction**:

1. resolves the stream (:func:`.sessions.resolve`) and its open record. A **helper** that
   answers gets a record of its own: its events since its previous answer move out of the
   parent's open record (the only time an event changes record), and a parent record left
   with no events is removed, so it never lingers as a path without answer. A stream with
   no open record (a follow-up answered from earlier reads) gets a record with no events;
2. runs the citation check (:func:`.path.check_citation`) against the session family's
   whole log up to this answer, not only the record's own events;
3. **freezes the evidence**: a span's quote, page and paper; a claim's type, label and text
   as the agent saw it (the current interpretation, :func:`seedgraph.notes.interpretation.current`),
   its trust label and ``(overlay_id, version_no)`` when the paper is researcher-edited; a
   paper's title, year and the page given;
4. writes question, answer, judgement, evidence, flags and counts, and sets the record
   ``answered``;
5. links the record's graphs: those named as ``graph`` in its events (``searched``), else
   the graphs holding a cited paper as a member (``cited_member``), else none.

Malformed input and ids that do not exist in the library are refused before anything is
written, so a typo is fixed rather than stored. A cited claim that exists but is not in
the paper's current interpretation (hidden, or from an older note) is not refused: it is
frozen from the stored row and noted. The check result is frozen: deleting an earlier
record later never changes it. Nothing is ever amended in place.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import Any

from .. import __version__
from ..errors import ValidationError
from ..library import graphs as graphs_mod
from ..library.store import Library
from ..work.store import fmt_ts, now_utc
from . import path as paths
from . import sessions
from .log import RECORD_VERSION, _time_id, open_record
from .sessions import CallInfo, Stream

MAX_QUESTION = 2_000
MAX_ANSWER = 20_000
MAX_JUDGEMENT = 5_000
MAX_WHY = 2_000
MAX_EVIDENCE = 100
ID_KEYS: tuple[str, ...] = ("span_id", "claim_id", "paper_id")
_ALLOWED = set(ID_KEYS) | {"page", "why"}

NEXT_ACTION_FLAGGED = ("Tell the researcher which citations were not read, or read them and record again.")
NEXT_ACTION_EVIDENCE = (
    "Pass evidence as 1 to 100 objects, each with exactly one of span_id, claim_id or paper_id "
    "(a paper may add page, a whole number), and optionally why (text).")
NEXT_ACTION_UNKNOWN = ("Check the ids against the search, paper_show or concept_show replies and record again.")
NOT_CURRENT_NOTE = "not in the paper's current interpretation"
PAGE_PATH = "/records/{record_id}"


class RecordAnswerError(ValidationError):
    """A refused ``record_answer`` call, with a plain next action. Nothing was written."""

    code = "invalid_argument"

    def __init__(self, message: str, next_action: str = NEXT_ACTION_EVIDENCE) -> None:
        super().__init__(message)
        self.next_action = next_action


class UnknownEvidence(RecordAnswerError):
    """Cited ids that do not exist in the library."""

    code = "unknown_evidence"

    def __init__(self, unknown: Sequence[str]) -> None:
        super().__init__(f"unknown evidence ids: {', '.join(unknown)}", NEXT_ACTION_UNKNOWN)
        self.unknown = tuple(unknown)


@dataclass(frozen=True)
class RecordResult:
    record_id: str
    session: str                       # the session name, or the start date/time label
    counts: dict[str, int]             # searches, opened, cited
    flags: tuple[dict[str, Any], ...]  # citation flags, then session flags
    page: str                          # /records/<record_id>
    next_action: str | None            # set when a citation is flagged
    evidence: tuple[dict[str, Any], ...]
    graphs: tuple[tuple[str, str], ...]  # (slug, how)

    def to_reply(self) -> dict[str, Any]:
        """The MCP reply body (§3.4)."""
        return {"record_id": self.record_id, "session": self.session, "counts": dict(self.counts),
                "flags": [dict(f) for f in self.flags], "page": self.page, "next_action": self.next_action}


# --- validation ---------------------------------------------------------------------------------

def _text(value: Any, what: str, limit: int, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value.strip()):
        raise RecordAnswerError(f"{what} must be non-empty text", f"Pass {what} as text of at most {limit} "
                                                                    "characters.")
    if len(value) > limit:
        raise RecordAnswerError(f"{what} is {len(value)} characters; the limit is {limit}",
                                f"Shorten {what} to at most {limit} characters.")
    return value if value.strip() else None


def _item(raw: Any, n: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RecordAnswerError(f"evidence item {n} must be an object")
    extra = sorted(set(raw) - _ALLOWED)
    if extra:
        raise RecordAnswerError(f"evidence item {n} has unknown keys: {', '.join(map(str, extra))}")
    ids = [k for k in ID_KEYS if raw.get(k) is not None]
    if len(ids) != 1 or not isinstance(raw[ids[0]], str) or not raw[ids[0]].strip():
        raise RecordAnswerError(f"evidence item {n} needs exactly one of span_id, claim_id or paper_id")
    key = ids[0]
    page = raw.get("page")
    if page is not None:
        if key != "paper_id":
            raise RecordAnswerError(f"evidence item {n}: page goes with a paper_id only")
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise RecordAnswerError(f"evidence item {n}: page must be a whole number from 1")
    why = raw.get("why")
    if why is not None and (not isinstance(why, str) or len(why) > MAX_WHY):
        raise RecordAnswerError(f"evidence item {n}: why must be text of at most {MAX_WHY} characters")
    return {"kind": key[:-3], "id": raw[key], "page": page, "why": why}


def _items(evidence: Any) -> list[dict[str, Any]]:
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)) or \
            not 1 <= len(evidence) <= MAX_EVIDENCE:
        raise RecordAnswerError(f"evidence must be a list of 1 to {MAX_EVIDENCE} items")
    return [_item(raw, n) for n, raw in enumerate(evidence)]


# --- freezing -----------------------------------------------------------------------------------

def _paper_meta(conn: sqlite3.Connection, paper_id: str | None) -> dict[str, Any]:
    row = conn.execute("SELECT title, year FROM papers WHERE paper_id = ?", (paper_id,)).fetchone() \
        if paper_id else None
    return {"title": row[0] if row else None, "year": row[1] if row else None}


def _span(conn: sqlite3.Connection, span_id: str) -> dict[str, Any] | None:
    r = conn.execute(
        "SELECT e.span_id, e.markdown_id, e.start_char, e.end_char, e.exact_quote, e.page_start, e.page_end, "
        "(SELECT MIN(pf.paper_id) FROM paper_files pf WHERE pf.markdown_id = e.markdown_id "
        " OR pf.source_file_id = e.source_file_id) AS paper_id FROM evidence_spans e WHERE e.span_id = ?",
        (span_id,)).fetchone()
    if r is None:
        return None
    return {"paper_id": r["paper_id"], "markdown_id": r["markdown_id"], "start": r["start_char"],
            "end": r["end_char"], "quote": r["exact_quote"], "page_start": r["page_start"],
            "page_end": r["page_end"]}


def _anchor(conn: sqlite3.Connection, claim_id: str, span_id: str | None) -> dict[str, Any]:
    if span_id is None:
        row = conn.execute("SELECT span_id FROM claim_spans WHERE claim_id = ? ORDER BY rank, span_id LIMIT 1",
                           (claim_id,)).fetchone()
        span_id = row[0] if row else None
    s = _span(conn, span_id) if span_id else None
    return {"span_id": span_id if s else None, "quote": s["quote"] if s else None,
            "page_start": s["page_start"] if s else None, "page_end": s["page_end"] if s else None}


def _claim(conn: sqlite3.Connection, claim_id: str) -> dict[str, Any] | None:
    from ..notes import interpretation

    base = conn.execute("SELECT paper_id, claim_type, normalized_label, claim_text, plain_words, status, "
                        "EXISTS (SELECT 1 FROM claim_spans s WHERE s.claim_id = c.claim_id) AS anchored "
                        "FROM extracted_claims c WHERE claim_id = ?", (claim_id,)).fetchone()
    added = None if base is not None else conn.execute(
        "SELECT paper_id, claim_type, normalized_label, claim_text, plain_words, trust, kind, edited, span_id "
        "FROM interpretation_claims WHERE claim_id = ? ORDER BY version_no DESC LIMIT 1", (claim_id,)).fetchone()
    if base is None and added is None:
        return None
    paper_id = (base or added)["paper_id"]
    interp = interpretation.current(conn, paper_id, "agent")
    seen = next((c for c in interp.claims if c.claim_id == claim_id), None) if interp else None
    if seen is not None:
        edited = interp.overlay_id is not None
        out = {"paper_id": paper_id, "claim_type": seen.claim_type, "label": seen.normalized_label,
               "text": seen.claim_text, "plain_words": seen.plain_words, "trust": seen.trust,
               "trust_label": seen.trust_label, "trust_note": seen.trust_note,
               "overlay_id": interp.overlay_id if edited else None, "version_no": interp.version_no if edited
               else None, "current": True, "note": None, **_anchor(conn, claim_id, seen.span_id)}
        return out
    if base is not None:
        trust = "paper_states" if base["status"] == "found" and base["anchored"] else "agent_inferred"
        row, trust_note, span_id = base, None, None
    else:
        row, trust = added, added["trust"]
        trust_note = (interpretation.ASSERTION_NOTE if added["kind"] == "assertion"
                      else interpretation.EDITED_NOTE if added["edited"] else None)
        span_id = added["span_id"]
    return {"paper_id": paper_id, "claim_type": row["claim_type"], "label": row["normalized_label"],
            "text": row["claim_text"], "plain_words": row["plain_words"], "trust": trust,
            "trust_label": interpretation.TRUST_LABELS[trust], "trust_note": trust_note, "overlay_id": None,
            "version_no": None, "current": False, "note": NOT_CURRENT_NOTE, **_anchor(conn, claim_id, span_id)}


def _freeze(conn: sqlite3.Connection, items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each item resolved and frozen; raises :class:`UnknownEvidence` naming every id that
    does not exist, in the order given."""
    out: list[dict[str, Any]] = []
    unknown: list[str] = []
    for item in items:
        kind, item_id = item["kind"], item["id"]
        if kind == "span":
            body = _span(conn, item_id)
        elif kind == "claim":
            body = _claim(conn, item_id)
        else:
            body = {"paper_id": item_id, "page": item["page"]} if conn.execute(
                "SELECT 1 FROM papers WHERE paper_id = ?", (item_id,)).fetchone() else None
        if body is None:
            unknown.append(item_id)
            continue
        out.append({"kind": kind, "id": item_id, **body, **_paper_meta(conn, body["paper_id"]),
                    "why": item["why"]})
    if unknown:
        raise UnknownEvidence(list(dict.fromkeys(unknown)))
    return out


# --- graphs -------------------------------------------------------------------------------------

def _member_graphs(lib: Library, paper_ids: set[str]) -> list[str]:
    """The graphs that hold any of ``paper_ids`` as a member, by slug."""
    out = []
    if not paper_ids:
        return out
    for info in graphs_mod.list_graphs(lib):
        with graphs_mod.open_graph(lib, info.slug) as g:
            if g.conn.execute("SELECT 1 FROM graph_papers WHERE status = 'member' AND paper_id IN "
                              "(SELECT value FROM json_each(?)) LIMIT 1",
                              (json.dumps(sorted(paper_ids)),)).fetchone():
                out.append(info.slug)
    return out


def _searched(events: Sequence[paths.Event]) -> list[str]:
    return sorted({e.args["graph"] for e in events if isinstance(e.args.get("graph"), str) and e.args["graph"]})


# --- the record ---------------------------------------------------------------------------------

def _take_helper_events(conn: sqlite3.Connection, stream: Stream, now: datetime) -> str:
    """A helper's own record: its events since its previous answer move out of the
    parent's open record; an emptied parent record is removed (its unlogged count moves
    along), so no path without answer is left behind."""
    ts = fmt_ts(now)
    record_id = _time_id("tr", now)
    parent_open = open_record(conn, stream.owner_session_id)
    span = (None, None)
    if parent_open is not None:
        span = conn.execute("SELECT MIN(at), MAX(at) FROM traversal_events WHERE record_id = ? AND session_id = ?",
                            (parent_open, stream.session_id)).fetchone()
    conn.execute("INSERT INTO traversal_records (record_id, session_id, state, opened_at, last_event_at, "
                 "answered_at, record_version, seedgraph_version) VALUES (?, ?, 'answered', ?, ?, ?, ?, ?)",
                 (record_id, stream.session_id, span[0] or ts, span[1] or ts, ts, RECORD_VERSION, __version__))
    if parent_open is None or span[0] is None:
        return record_id
    conn.execute("UPDATE traversal_events SET record_id = ? WHERE record_id = ? AND session_id = ?",
                 (record_id, parent_open, stream.session_id))
    left = conn.execute("SELECT MAX(at) FROM traversal_events WHERE record_id = ?", (parent_open,)).fetchone()[0]
    if left is None:
        unlogged = conn.execute("SELECT unlogged_calls FROM traversal_records WHERE record_id = ?",
                                (parent_open,)).fetchone()[0]
        conn.execute("UPDATE traversal_records SET unlogged_calls = unlogged_calls + ? WHERE record_id = ?",
                     (unlogged, record_id))
        conn.execute("DELETE FROM record_graphs WHERE record_id = ?", (parent_open,))
        conn.execute("DELETE FROM traversal_records WHERE record_id = ?", (parent_open,))
    else:
        conn.execute("UPDATE traversal_records SET last_event_at = ? WHERE record_id = ?", (left, parent_open))
    return record_id


def _answering_record(conn: sqlite3.Connection, stream: Stream, now: datetime) -> str:
    if stream.is_helper:
        return _take_helper_events(conn, stream, now)
    record_id = open_record(conn, stream.session_id)
    if record_id is not None:
        return record_id
    ts = fmt_ts(now)
    record_id = _time_id("tr", now)
    conn.execute("INSERT INTO traversal_records (record_id, session_id, state, opened_at, last_event_at, "
                 "answered_at, record_version, seedgraph_version) VALUES (?, ?, 'answered', ?, ?, ?, ?, ?)",
                 (record_id, stream.session_id, ts, ts, ts, RECORD_VERSION, __version__))
    return record_id


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def record_answer(lib: Library, call: CallInfo, question: str, answer: str, evidence: Sequence[Mapping[str, Any]],
                  judgement: str | None = None, *, now: datetime | None = None,
                  tz: tzinfo | None = None) -> RecordResult:
    """Record one answer of the stream ``call`` belongs to. See the module doc. Raises
    :class:`RecordAnswerError` (or :class:`.sessions.SessionNameError`) and writes nothing
    when the input is refused."""
    sessions.validate_name(call.session_name)
    question = _text(question, "question", MAX_QUESTION, required=True)
    answer = _text(answer, "answer", MAX_ANSWER, required=True)
    judgement = _text(judgement, "judgement", MAX_JUDGEMENT, required=False)
    items = _items(evidence)
    now = now or now_utc()
    ts = fmt_ts(now)

    # dirty=False: records are test data until stage 7 and must not block a re-migration
    with lib.write(dirty=False) as conn:  # a refusal below rolls everything back
        frozen = _freeze(conn, items)                     # unknown ids are refused here
        stream = sessions.resolve(lib, call, now=now, tz=tz)
        record_id = _answering_record(conn, stream, now)
        family = sessions.family(lib, stream.session_id)
        family_log = paths.load_events(conn, session_ids=family)
        own = [e for e in family_log if e.record_id == record_id]
        lookup = paths.load_lookup(conn, family_log, frozen)
        seen = paths.seen(family_log, lookup)

        flags: list[dict[str, Any]] = []
        for n, item in enumerate(frozen):
            c = paths.check_citation(seen, paths.evidence_key(item), record_id=record_id,
                                     session_id=stream.session_id)
            item.update({"check": c.status, "check_label": c.label, "flag": c.flag, "opened_by": list(c.opened_by)})
            if c.flag:
                flags.append({"kind": "citation", "evidence": n, "item": {"kind": item["kind"], "id": item["id"]},
                              "flag": c.flag})
        cited_flagged = bool(flags)
        if stream.warnings or (stream.key_kind == sessions.KEY_NAMED
                               and sessions.shared_by_connections(lib, stream.session_id)):
            flags.append({"kind": "session", "flag": sessions.WARN_TWO_CONNECTIONS})
        counts = paths.path(own, frozen, lookup=lookup).counts

        searched = _searched(own)
        links = [(s, "searched") for s in searched] or [
            (s, "cited_member") for s in _member_graphs(lib, {e["paper_id"] for e in frozen if e["paper_id"]})]
        conn.execute("UPDATE traversal_records SET state = 'answered', answered_at = ?, question = ?, answer = ?, "
                     "judgement = ?, evidence_json = ?, flags_json = ?, counts_json = ? WHERE record_id = ?",
                     (ts, question, answer, judgement, _dumps(frozen), _dumps(flags), _dumps(counts), record_id))
        conn.execute("DELETE FROM record_graphs WHERE record_id = ?", (record_id,))
        conn.executemany("INSERT INTO record_graphs (record_id, slug, how) VALUES (?, ?, ?)",
                         [(record_id, slug, how) for slug, how in links])

    return RecordResult(record_id=record_id, session=stream.name, counts=counts, flags=tuple(flags),
                        page=PAGE_PATH.format(record_id=record_id),
                        next_action=NEXT_ACTION_FLAGGED if cited_flagged else None, evidence=tuple(frozen),
                        graphs=tuple(links))
