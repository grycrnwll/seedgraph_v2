"""Managing records: the list, one record, star, annotate, delete (design §3.5-3.7, §6.1-6.3;
slice R2).

Records are kept indefinitely; the researcher stars, annotates, deletes and exports them.
:func:`list_records` is the one list query (the Records screen, a graph's Records tab and
the Home feed all use it); :func:`get_record` is the one loader of a record with its path
(the record page and :func:`.export.export_markdown` both use it, so they never disagree).

- **State** is display only (:func:`.path.state`): answered, in progress, or path without
  answer once idle :data:`.path.PATH_IDLE` or its connection closed.
- **Graphs**: a record's ``record_graphs`` links. The graph filter lists records that
  searched the graph **and** library-wide records that cite a paper of it (marked,
  decision Q3). :data:`LIBRARY_WIDE` lists records that searched no graph.
- **Flags** are the frozen citation flags; **notes** are "session told apart by
  connection only", "session name used from two connections" (frozen at answer time,
  derived while open), "n calls could not be logged" and "graph deleted: <slug>".
- **Delete** removes the record's events and graph links first (no cascading foreign keys),
  then the record. Frozen checks of later records never change.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..errors import SeedgraphError
from ..library import graphs as graphs_mod
from ..library.store import Library
from ..work.store import fmt_ts, now_utc
from . import path as paths
from . import sessions

#: The graph filter value for records that searched no graph (not a valid slug).
LIBRARY_WIDE = ":library-wide"
MAX_ANNOTATION = 10_000
CITED_MEMBER_MARK = "library-wide search; cites papers in this graph"
NOTE_CONNECTION_ONLY = "session told apart by connection only"


class RecordError(SeedgraphError):
    """A refused records action."""


class RecordNotFound(RecordError):
    def __init__(self, record_id: str) -> None:
        super().__init__(f"no record {record_id!r}")
        self.record_id = record_id


@dataclass(frozen=True)
class Filters:
    graph: str | None = None                      # a slug, or LIBRARY_WIDE
    state: str | tuple[str, ...] | None = None    # answered | in_progress | path_without_answer
    starred: bool | None = None
    flagged: bool | None = None
    client: str | None = None
    text: str | None = None                       # matched in the question, any case


@dataclass(frozen=True)
class GraphLink:
    slug: str
    how: str                                      # searched | cited_member
    deleted: bool = False


@dataclass(frozen=True)
class RecordRow:
    record_id: str
    state: str                                    # answered | in_progress | path_without_answer
    opened_at: str
    last_event_at: str
    answered_at: str | None
    question: str | None
    graphs: tuple[GraphLink, ...]
    session_id: str
    session_name: str                             # the name, or the start date/time label
    client: str | None
    key_kind: str                                 # named | connection
    counts: dict[str, int]                        # searches, opened, cited
    flags: tuple[str, ...]                        # frozen citation flags, in evidence order
    notes: tuple[str, ...]
    starred: bool
    annotation: str | None
    unlogged_calls: int

    @property
    def when(self) -> str:
        return self.answered_at or self.last_event_at

    @property
    def state_label(self) -> str:
        return paths.STATE_LABELS[self.state]

    @property
    def title(self) -> str:
        """The question, or "path without answer" / "in progress"."""
        return self.question if self.state == paths.STATE_ANSWERED and self.question else self.state_label

    @property
    def spans_label(self) -> str | None:
        """"spans n graphs: A, B" for a record in two or more graphs."""
        if len(self.graphs) < 2:
            return None
        return f"spans {len(self.graphs)} graphs: " + ", ".join(g.slug for g in self.graphs)

    def graph_mark(self, slug: str) -> str | None:
        """The mark shown in ``slug``'s list: set for a library-wide record citing its papers."""
        link = next((g for g in self.graphs if g.slug == slug), None)
        return CITED_MEMBER_MARK if link is not None and link.how == "cited_member" else None


@dataclass(frozen=True)
class RecordPage:
    rows: tuple[RecordRow, ...]
    total: int
    offset: int
    limit: int


@dataclass(frozen=True)
class RecordDetail:
    row: RecordRow
    answer: str | None
    judgement: str | None
    evidence: tuple[dict[str, Any], ...]          # frozen, with each item's check
    flag_entries: tuple[dict[str, Any], ...]      # flags_json as stored
    events: tuple[paths.Event, ...]
    path: paths.Path
    papers: dict[str, tuple[str | None, int | None]]  # paper_id -> (title, year), for display
    annotated_at: str | None

    @property
    def notes(self) -> tuple[str, ...]:
        return self.row.notes


# --- loading ------------------------------------------------------------------------------------

_ROW_SQL = (
    "SELECT r.*, s.name AS s_name, s.client AS s_client, s.key_kind AS s_kind, "
    "(SELECT c.closed_at FROM traversal_events e JOIN record_connections c ON c.connection_id = e.connection_id "
    " WHERE e.record_id = r.record_id ORDER BY e.seq DESC LIMIT 1) AS closed_at, "
    "(SELECT COUNT(*) FROM traversal_events e WHERE e.record_id = r.record_id AND e.kind = 'search') AS n_search, "
    "(SELECT COUNT(*) FROM traversal_events e WHERE e.record_id = r.record_id AND e.kind != 'search') AS n_open "
    "FROM traversal_records r JOIN record_sessions s ON s.session_id = r.session_id")


def _links(conn: sqlite3.Connection, record_ids: Sequence[str], lib: Library) -> dict[str, tuple[GraphLink, ...]]:
    out: dict[str, list[GraphLink]] = {}
    exists: dict[str, bool] = {}
    for rid, slug, how in conn.execute(
            "SELECT record_id, slug, how FROM record_graphs WHERE record_id IN (SELECT value FROM json_each(?)) "
            "ORDER BY record_id, slug", (json.dumps(list(record_ids)),)):
        if slug not in exists:
            try:
                exists[slug] = graphs_mod.graph_db_path(lib, slug).is_file()
            except Exception:  # an unreadable slug is shown as deleted, never raised
                exists[slug] = False
        out.setdefault(rid, []).append(GraphLink(slug, how, not exists[slug]))
    return {rid: tuple(v) for rid, v in out.items()}


def _row(lib: Library, r: sqlite3.Row, links: tuple[GraphLink, ...], now: datetime) -> RecordRow:
    st = paths.state(r, now, {"closed_at": r["closed_at"]})
    stored = json.loads(r["flags_json"] or "[]")
    flags = tuple(f["flag"] for f in stored if f.get("kind") == "citation")
    if r["state"] == "answered":
        counts = json.loads(r["counts_json"] or "{}")
        two = any(f.get("kind") == "session" for f in stored)
    else:
        counts = {"searches": r["n_search"], "opened": r["n_open"], "cited": 0}
        two = r["s_kind"] == sessions.KEY_NAMED and sessions.shared_by_connections(lib, r["session_id"])
    notes = []
    if r["s_kind"] == sessions.KEY_CONNECTION:
        notes.append(NOTE_CONNECTION_ONLY)
    if two:
        notes.append(sessions.WARN_TWO_CONNECTIONS)
    if r["unlogged_calls"]:
        n = r["unlogged_calls"]
        notes.append(f"{n} call{'s' if n != 1 else ''} could not be logged")
    notes += [f"graph deleted: {g.slug}" for g in links if g.deleted]
    return RecordRow(
        record_id=r["record_id"], state=st, opened_at=r["opened_at"], last_event_at=r["last_event_at"],
        answered_at=r["answered_at"], question=r["question"], graphs=links, session_id=r["session_id"],
        session_name=r["s_name"], client=r["s_client"], key_kind=r["s_kind"], counts=counts, flags=flags,
        notes=tuple(notes), starred=bool(r["starred"]), annotation=r["annotation"],
        unlogged_calls=r["unlogged_calls"])


def _states(value: str | tuple[str, ...] | None) -> set[str] | None:
    if value is None:
        return None
    wanted = {value} if isinstance(value, str) else set(value)
    bad = wanted - set(paths.STATES)
    if bad:
        raise ValueError(f"unknown record state {sorted(bad)!r}; expected one of {', '.join(paths.STATES)}")
    return wanted


def list_records(lib: Library, filters: Filters | None = None, *, limit: int = 50, offset: int = 0,
                 now: datetime | None = None) -> RecordPage:
    """Records newest first (by answer time, else last event), filtered and paged."""
    f = filters or Filters()
    states = _states(f.state)
    now = now or now_utc()
    where, args = [], []
    if f.graph == LIBRARY_WIDE:
        where.append("NOT EXISTS (SELECT 1 FROM record_graphs g WHERE g.record_id = r.record_id "
                     "AND g.how = 'searched')")
    elif f.graph is not None:
        where.append("EXISTS (SELECT 1 FROM record_graphs g WHERE g.record_id = r.record_id AND g.slug = ?)")
        args.append(f.graph)
    if f.starred is not None:
        where.append("r.starred = ?")
        args.append(int(f.starred))
    if f.flagged is not None:
        where.append(("" if f.flagged else "NOT ") + "EXISTS (SELECT 1 FROM json_each(r.flags_json) j "
                     "WHERE json_extract(j.value, '$.kind') = 'citation')")
    if f.client is not None:
        where.append("s.client = ?")
        args.append(f.client)
    if f.text:
        where.append("r.question LIKE ? ESCAPE '\\'")
        args.append("%" + f.text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
    sql = _ROW_SQL + (" WHERE " + " AND ".join(where) if where else "") + \
        " ORDER BY COALESCE(r.answered_at, r.last_event_at) DESC, r.record_id DESC"
    rows = lib.conn.execute(sql, args).fetchall()
    links = _links(lib.conn, [r["record_id"] for r in rows], lib)
    out = [_row(lib, r, links.get(r["record_id"], ()), now) for r in rows]
    if states is not None:
        out = [r for r in out if r.state in states]
    return RecordPage(rows=tuple(out[offset:offset + limit]), total=len(out), offset=offset, limit=limit)


def _fetch(lib: Library, record_id: str) -> sqlite3.Row:
    r = lib.conn.execute(_ROW_SQL + " WHERE r.record_id = ?", (record_id,)).fetchone()
    if r is None:
        raise RecordNotFound(record_id)
    return r


def get_record(lib: Library, record_id: str, *, now: datetime | None = None) -> RecordDetail:
    """One record with its derived path (its own events, tagged against its evidence)."""
    r = _fetch(lib, record_id)
    row = _row(lib, r, _links(lib.conn, [record_id], lib).get(record_id, ()), now or now_utc())
    events = tuple(paths.load_events(lib.conn, record_id=record_id))
    evidence = tuple(json.loads(r["evidence_json"] or "[]"))
    lookup = paths.load_lookup(lib.conn, events, evidence)
    p = paths.path(events, evidence, lookup=lookup)
    pids = {i.paper_id for i in p.items if i.paper_id} | {e["paper_id"] for e in evidence if e.get("paper_id")}
    papers = {pid: (t, y) for pid, t, y in lib.conn.execute(
        "SELECT paper_id, title, year FROM papers WHERE paper_id IN (SELECT value FROM json_each(?))",
        (json.dumps(sorted(pids)),))}
    return RecordDetail(row=row, answer=r["answer"], judgement=r["judgement"], evidence=evidence,
                        flag_entries=tuple(json.loads(r["flags_json"] or "[]")), events=events, path=p,
                        papers=papers, annotated_at=r["annotated_at"])


# --- actions ------------------------------------------------------------------------------------

def _require(conn: sqlite3.Connection, record_id: str) -> None:
    if conn.execute("SELECT 1 FROM traversal_records WHERE record_id = ?", (record_id,)).fetchone() is None:
        raise RecordNotFound(record_id)


def star(lib: Library, record_id: str, starred: bool = True) -> None:
    with lib.write() as conn:
        _require(conn, record_id)
        conn.execute("UPDATE traversal_records SET starred = ? WHERE record_id = ?", (int(starred), record_id))


def annotate(lib: Library, record_id: str, text: str | None, *, now: datetime | None = None) -> None:
    """Set the researcher's one free-text annotation; empty or blank text clears it."""
    if text is not None and len(text) > MAX_ANNOTATION:
        raise RecordError(f"the annotation is {len(text)} characters; the limit is {MAX_ANNOTATION}")
    value = text if text and text.strip() else None
    with lib.write() as conn:
        _require(conn, record_id)
        conn.execute("UPDATE traversal_records SET annotation = ?, annotated_at = ? WHERE record_id = ?",
                     (value, fmt_ts(now or now_utc()) if value else None, record_id))


def delete(lib: Library, record_id: str) -> None:
    """Delete the record with its events and graph links."""
    with lib.write() as conn:
        _require(conn, record_id)
        conn.execute("DELETE FROM traversal_events WHERE record_id = ?", (record_id,))
        conn.execute("DELETE FROM record_graphs WHERE record_id = ?", (record_id,))
        conn.execute("DELETE FROM traversal_records WHERE record_id = ?", (record_id,))
