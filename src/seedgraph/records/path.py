"""The path of a record, and its state (design §3.3, §3.5, §3.7; slice R2).

The item tags are **derived**, never stored: :func:`path` is a pure function of a record's
events and its evidence, so the record page and the export can never disagree. The same
rules (:func:`seen`) drive the citation check at answer time, over the session family's
whole log (:func:`check_citation`), so the frozen flags and the page agree too.

An item is a **paper**, a **span** or a **claim**. Its tag is the strongest thing that
happened to it:

- **found**: it appeared in a search-kind reply (a hit's paper is found with it; a claim is
  found when its anchor span was a hit);
- **opened**: an open-kind reply returned its text. A span is opened when a ``paper_read``
  slice of the same markdown covers at least half of it (in integers, so a zero-length
  span must lie inside the slice), or when ``concept_show`` returned its quote. A claim is
  opened when ``paper_show`` or ``concept_show`` returned it, or when an anchor span of it
  is opened. A paper is opened when ``paper_show`` or ``paper_read`` was called on it;
- **cited**: it is in the answer's evidence.

A span id seen only as an anchor in ``paper_show`` was not shown as text: it counts as
nothing.

**Event items** (what the MCP adapter, slice R3, writes into ``items_json``; ``args`` keep
what the agent passed, with ``graph`` when it named one):

- ``search``: one dict per hit, in rank order: ``{"kind", "id", "paper_id"?,
  "paper_ids"?, "claim_ids"?, "preview"?}``. ``kind`` ``span``, ``claim`` and ``paper``
  make the id found; ``paper_id`` / ``paper_ids`` find papers, ``claim_ids`` (a concept
  hit's example claims) find claims. Other kinds (``note``, ``concept``, ``lens``) are
  shown in the path but are not items;
- ``open_paper`` (``paper_show``): ``args["paper_id"]``; items ``{"claim_id",
  "span_ids"}`` for each claim returned, with its anchor span ids (which count as nothing);
- ``open_text`` (``paper_read``): items ``{"paper_id", "markdown_id", "start", "end",
  "section_id"?}``, the range **returned** (falls back to ``args`` when items are empty);
- ``open_concept`` (``concept_show``): items ``{"claim_id", "span_ids", "paper_id"?}``,
  the claims returned and the spans whose quotes were returned.

The library facts the rules need (span places, claim anchors) come in a :class:`Lookup`,
built by :func:`load_lookup`; :func:`path` itself reads nothing.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

PATH_IDLE = timedelta(minutes=30)
#: A gap between two events at least this long is shown in the path ("40 minutes later").
GAP_SHOWN = timedelta(minutes=10)

FOUND, OPENED, CITED = "found", "opened", "cited"
TAG_LABELS: dict[str, str] = {FOUND: "found but not opened", OPENED: "opened but not cited", CITED: "cited"}
ITEM_KINDS: tuple[str, ...] = ("paper", "span", "claim")

STATE_ANSWERED, STATE_IN_PROGRESS, STATE_NO_ANSWER = "answered", "in_progress", "path_without_answer"
STATES: tuple[str, ...] = (STATE_ANSWERED, STATE_IN_PROGRESS, STATE_NO_ANSWER)
STATE_LABELS: dict[str, str] = {STATE_ANSWERED: "answered", STATE_IN_PROGRESS: "in progress",
                                STATE_NO_ANSWER: "path without answer"}

FLAG_NOT_READ = "cited without being read"
FLAG_SEARCH_ONLY = "cited without being read (seen in search results only)"
CHECK_LABELS: dict[str, str] = {
    "opened_here": "opened in this record",
    "opened_earlier": "opened earlier in this session",
    "found_only": FLAG_SEARCH_ONLY,
    "not_returned": FLAG_NOT_READ,
}

ItemKey = tuple[str, str]  # (kind, id)


@dataclass(frozen=True)
class Event:
    """One logged call, as the rules see it. ``helper`` is the helper suffix (``a`` for
    ``<parent>/a``) when a helper stream made the call."""

    seq: int
    at: str
    tool: str
    kind: str
    session_id: str
    connection_id: str
    args: Mapping[str, Any]
    items: Sequence[Any]
    record_id: str | None = None
    helper: str | None = None
    event_id: str | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any], helper: str | None = None) -> Event:
        return cls(seq=row["seq"], at=row["at"], tool=row["tool"], kind=row["kind"], session_id=row["session_id"],
                   connection_id=row["connection_id"], args=json.loads(row["args_json"]),
                   items=json.loads(row["items_json"]), record_id=row["record_id"], helper=helper,
                   event_id=row["event_id"])


@dataclass(frozen=True)
class SpanPlace:
    markdown_id: str
    start: int
    end: int
    paper_id: str | None = None


@dataclass(frozen=True)
class Lookup:
    """Library facts for the rules: where spans lie, and each claim's anchor spans."""

    spans: Mapping[str, SpanPlace] = field(default_factory=dict)
    anchors: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    claim_papers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Seen:
    """The strongest thing that happened to an item, and the events that did it."""

    level: str                       # found | opened
    events: tuple[Event, ...]        # the events at that level, in order
    first_seq: int = 0               # the first event that returned it at all


@dataclass(frozen=True)
class Item:
    kind: str
    id: str
    paper_id: str | None
    tag: str                         # found | opened | cited
    found: bool
    opened: bool
    helpers: tuple[str, ...] = ()    # helpers that made the strongest call

    @property
    def label(self) -> str:
        return TAG_LABELS[self.tag]


@dataclass(frozen=True)
class Step:
    event: Event
    gap: str | None                  # "40 minutes later" after a long pause


@dataclass(frozen=True)
class Path:
    steps: tuple[Step, ...]
    items: tuple[Item, ...]
    counts: dict[str, int]


@dataclass(frozen=True)
class Check:
    """The citation check of one evidence item (frozen with the record)."""

    status: str                      # opened_here | opened_earlier | found_only | not_returned
    opened_by: tuple[str, ...] = ()  # helper suffixes, when only other streams opened it

    @property
    def label(self) -> str:
        return CHECK_LABELS[self.status]

    @property
    def flag(self) -> str | None:
        return CHECK_LABELS[self.status] if self.status in ("found_only", "not_returned") else None

    @property
    def notes(self) -> tuple[str, ...]:
        return tuple(f"opened by helper {h}" for h in self.opened_by)


# --- the rules ----------------------------------------------------------------------------------

def _ids(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str)]
    return []


def _dicts(items: Sequence[Any]) -> list[Mapping[str, Any]]:
    return [i for i in items if isinstance(i, Mapping)]


def _ranges(e: Event) -> list[tuple[str | None, str | None, int, int]]:
    """(paper_id, markdown_id, start, end) of each returned slice of a paper_read."""
    out = []
    for src in _dicts(e.items) or [e.args]:
        start, end = src.get("start"), src.get("end")
        if isinstance(start, int) and isinstance(end, int):
            out.append((src.get("paper_id") or e.args.get("paper_id"), src.get("markdown_id"), start, end))
    return out


def _covers(start: int, end: int, s: SpanPlace) -> bool:
    length = s.end - s.start
    if length <= 0:
        return start <= s.start <= end
    return 2 * (min(end, s.end) - max(start, s.start)) >= length


def _touched(e: Event, lookup: Lookup) -> tuple[list[ItemKey], list[ItemKey]]:
    """The items one event found and opened (directly; anchors are applied after)."""
    found: list[ItemKey] = []
    opened: list[ItemKey] = []
    if e.kind == "search":
        for hit in _dicts(e.items):
            kind, hid = hit.get("kind"), hit.get("id")
            if kind in ITEM_KINDS and isinstance(hid, str):
                found.append((kind, hid))
            found += [("paper", p) for p in _ids(hit.get("paper_id")) + _ids(hit.get("paper_ids"))]
            found += [("claim", c) for c in _ids(hit.get("claim_ids"))]
    elif e.kind == "open_paper":
        opened += [("paper", p) for p in _ids(e.args.get("paper_id"))]
        opened += [("claim", c) for i in _dicts(e.items) for c in _ids(i.get("claim_id"))]
    elif e.kind == "open_text":
        for paper_id, markdown_id, start, end in _ranges(e):
            opened += [("paper", p) for p in _ids(paper_id)]
            opened += [("span", sid) for sid, s in sorted(lookup.spans.items())
                       if s.markdown_id == markdown_id and _covers(start, end, s)]
    elif e.kind == "open_concept":
        for i in _dicts(e.items):
            opened += [("claim", c) for c in _ids(i.get("claim_id"))]
            opened += [("span", s) for s in _ids(i.get("span_ids"))]
            found += [("paper", p) for p in _ids(i.get("paper_id"))]
    return found, opened


def _anchors(events: Iterable[Event], lookup: Lookup) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {c: set(s) for c, s in lookup.anchors.items()}
    for e in events:
        if e.kind in ("open_paper", "open_concept"):
            for i in _dicts(e.items):
                for c in _ids(i.get("claim_id")):
                    out.setdefault(c, set()).update(_ids(i.get("span_ids")))
    return out


def seen(events: Iterable[Event], lookup: Lookup = Lookup()) -> dict[ItemKey, Seen]:
    """Every item the events found or opened, with its strongest level."""
    ordered = sorted(events, key=lambda e: e.seq)
    by: dict[ItemKey, dict[str, list[Event]]] = {}
    for e in ordered:
        found, opened = _touched(e, lookup)
        for key in found:
            by.setdefault(key, {}).setdefault(FOUND, []).append(e)
        for key in opened:
            by.setdefault(key, {}).setdefault(OPENED, []).append(e)
    # a claim takes the level of an anchor span (opened when one is opened, else found)
    for claim_id, spans in _anchors(ordered, lookup).items():
        for level in (FOUND, OPENED):
            via = [e for s in spans for e in by.get(("span", s), {}).get(level, [])]
            if via:
                by.setdefault(("claim", claim_id), {}).setdefault(level, []).extend(via)
    out: dict[ItemKey, Seen] = {}
    for key, levels in by.items():
        level = OPENED if OPENED in levels else FOUND
        evs = sorted({e.seq: e for e in levels[level]}.values(), key=lambda e: e.seq)
        first = min(e.seq for evs_at in levels.values() for e in evs_at)
        out[key] = Seen(level, tuple(evs), first)
    return out


def evidence_key(item: Mapping[str, Any]) -> ItemKey:
    """The item key of one frozen evidence entry (``{"kind", "id", ...}``)."""
    return item["kind"], item["id"]


def check_citation(seen_items: Mapping[ItemKey, Seen], key: ItemKey, *, record_id: str | None,
                   session_id: str) -> Check:
    """The citation check of ``key`` against what the session family's log ``seen``.
    ``record_id`` is the answering record, ``session_id`` the answering stream."""
    s = seen_items.get(key)
    if s is None:
        return Check("not_returned")
    if s.level == FOUND:
        return Check("found_only")
    here = any(e.record_id == record_id for e in s.events)
    own = [e for e in s.events if e.session_id == session_id]
    helpers = () if own else tuple(sorted({e.helper for e in s.events if e.helper}))
    return Check("opened_here" if here else "opened_earlier", helpers)


# --- the path -----------------------------------------------------------------------------------

def gap_text(gap: timedelta) -> str:
    minutes = int(gap.total_seconds() // 60)
    hours, minutes = divmod(minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes or not hours:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    return " ".join(parts) + " later"


def _paper_of(key: ItemKey, lookup: Lookup, events: Sequence[Event]) -> str | None:
    kind, item_id = key
    if kind == "paper":
        return item_id
    if kind == "span" and item_id in lookup.spans:
        return lookup.spans[item_id].paper_id
    if kind == "claim" and item_id in lookup.claim_papers:
        return lookup.claim_papers[item_id]
    for e in events:
        for hit in _dicts(e.items):
            if (hit.get("kind"), hit.get("id")) == key or hit.get("claim_id") == item_id:
                pid = hit.get("paper_id")
                if isinstance(pid, str):
                    return pid
    return None


def path(events: Iterable[Event], evidence: Sequence[Mapping[str, Any]] = (), *,
         lookup: Lookup = Lookup()) -> Path:
    """The record's path (its own events, in order, with long gaps) and its items, each
    tagged found / opened / cited. ``evidence`` is the frozen evidence (``kind``, ``id``)."""
    ordered = tuple(sorted(events, key=lambda e: e.seq))
    steps = []
    prev: datetime | None = None
    for e in ordered:
        at = datetime.fromisoformat(e.at)
        gap = at - prev if prev is not None else None
        steps.append(Step(e, gap_text(gap) if gap is not None and gap >= GAP_SHOWN else None))
        prev = at
    s = seen(ordered, lookup)
    cited = [evidence_key(item) for item in evidence]
    first = {key: v.first_seq for key, v in s.items()}
    keys = sorted(s, key=lambda k: (first[k], ITEM_KINDS.index(k[0]), k[1]))
    keys += [k for k in dict.fromkeys(cited) if k not in s]
    items = []
    for key in keys:
        v = s.get(key)
        tag = CITED if key in cited else (v.level if v else FOUND)
        items.append(Item(kind=key[0], id=key[1], paper_id=_paper_of(key, lookup, ordered), tag=tag,
                          found=v is not None, opened=v is not None and v.level == OPENED,
                          helpers=tuple(sorted({e.helper for e in v.events if e.helper})) if v else ()))
    counts = {"searches": sum(1 for e in ordered if e.kind == "search"),
              "opened": sum(1 for e in ordered if e.kind != "search"), "cited": len(evidence)}
    return Path(tuple(steps), tuple(items), counts)


# --- state --------------------------------------------------------------------------------------

def state(record: Mapping[str, Any], now: datetime, connection: Mapping[str, Any] | None) -> str:
    """``answered``, ``in_progress`` or ``path_without_answer`` (display only, §3.5): an
    open record is in progress while its last event is under :data:`PATH_IDLE` old and the
    connection of that event is live. ``connection`` is its row (``closed_at``) or None."""
    if record["state"] == "answered":
        return STATE_ANSWERED
    if record["state"] != "open":
        raise ValueError(f"unknown record state {record['state']!r}")
    if connection is not None and connection["closed_at"] is not None:
        return STATE_NO_ANSWER
    idle = now - datetime.fromisoformat(record["last_event_at"])
    return STATE_NO_ANSWER if idle >= PATH_IDLE else STATE_IN_PROGRESS


# --- loading (reads the library) ----------------------------------------------------------------

def load_events(conn: sqlite3.Connection, *, record_id: str | None = None,
                session_ids: Sequence[str] | None = None) -> list[Event]:
    """The events of one record, or of the given streams (a session family's log), in
    ``seq`` order, each with its helper suffix."""
    if (record_id is None) == (session_ids is None):
        raise ValueError("give record_id or session_ids")
    where, arg = (("e.record_id = ?", record_id) if record_id is not None
                  else ("e.session_id IN (SELECT value FROM json_each(?))", json.dumps(list(session_ids or ()))))
    rows = conn.execute(
        "SELECT e.*, s.name AS s_name, s.parent_session_id AS s_parent FROM traversal_events e "
        f"JOIN record_sessions s ON s.session_id = e.session_id WHERE {where} ORDER BY e.seq", (arg,)).fetchall()
    return [Event.from_row(r, helper=r["s_name"].partition("/")[2] if r["s_parent"] else None) for r in rows]


def _mentioned(events: Iterable[Event], evidence: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    spans: set[str] = set()
    claims: set[str] = set()
    for e in events:
        for i in _dicts(e.items):
            if i.get("kind") == "span":
                spans.update(_ids(i.get("id")))
            elif i.get("kind") == "claim":
                claims.update(_ids(i.get("id")))
            claims.update(_ids(i.get("claim_ids")) + _ids(i.get("claim_id")))
            spans.update(_ids(i.get("span_ids")))
    for item in evidence:
        if item.get("kind") == "span":
            spans.add(item["id"])
        elif item.get("kind") == "claim":
            claims.add(item["id"])
            spans.update(_ids(item.get("span_id")))
    return spans, claims


def load_lookup(conn: sqlite3.Connection, events: Iterable[Event],
                evidence: Sequence[Mapping[str, Any]] = ()) -> Lookup:
    """The span places and claim anchors of every span and claim the events or evidence
    mention (a span is opened by a ``paper_read`` only if it is one of these)."""
    events = list(events)
    span_ids, claim_ids = _mentioned(events, evidence)
    anchors: dict[str, tuple[str, ...]] = {}
    papers: dict[str, str] = {}
    if claim_ids:
        cj = json.dumps(sorted(claim_ids))
        rows = conn.execute(
            "SELECT claim_id, span_id FROM claim_spans WHERE claim_id IN (SELECT value FROM json_each(?)) "
            "UNION SELECT claim_id, span_id FROM interpretation_claims "
            "WHERE span_id IS NOT NULL AND claim_id IN (SELECT value FROM json_each(?))", (cj, cj)).fetchall()
        acc: dict[str, set[str]] = {}
        for claim_id, span_id in rows:
            acc.setdefault(claim_id, set()).add(span_id)
        anchors = {c: tuple(sorted(s)) for c, s in acc.items()}
        span_ids |= {s for v in anchors.values() for s in v}
        papers = dict(conn.execute(
            "SELECT claim_id, paper_id FROM extracted_claims WHERE claim_id IN (SELECT value FROM json_each(?)) "
            "UNION SELECT claim_id, paper_id FROM interpretation_claims "
            "WHERE claim_id IN (SELECT value FROM json_each(?))", (cj, cj)).fetchall())
    places: dict[str, SpanPlace] = {}
    if span_ids:
        for r in conn.execute(
                "SELECT e.span_id, e.markdown_id, e.start_char, e.end_char, "
                "(SELECT MIN(pf.paper_id) FROM paper_files pf WHERE pf.markdown_id = e.markdown_id "
                " OR pf.source_file_id = e.source_file_id) FROM evidence_spans e "
                "WHERE e.span_id IN (SELECT value FROM json_each(?))", (json.dumps(sorted(span_ids)),)):
            places[r[0]] = SpanPlace(r[1], r[2], r[3], r[4])
    return Lookup(spans=places, anchors=anchors, claim_papers=papers)
