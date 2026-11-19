"""Automated probes of the stage 5 acceptance (design §10.2 to §10.4).

They run through the in-memory MCP client against a **copy** of a home's library: the real
home is never written. The same probes run in two places: against the researcher's real home
(opt-in, ``SEEDGRAPH_STAGE5_HOME``) and against an invented library (always, the harness's own
self-check). Probe results carry ids and counts, never paper text.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from seedgraph.library import store as libstore
from seedgraph.mcp.server import build_server, close_connection

FLAG_NOT_READ = "cited without being read"
FLAG_SEARCH_ONLY = "cited without being read (seen in search results only)"
ANSWER = {"question": "probe question", "answer": "probe answer"}


# --- the copy -----------------------------------------------------------------------------------

def copy_library(src_home: Path, dst_home: Path) -> Path:
    """A consistent copy of ``<src>/library/library.db`` (SQLite backup, so a WAL is honoured).
    Markdown files are copied later, only for the papers a probe uses."""
    src_db = libstore.library_db_path(libstore.library_dir(src_home))
    dst_dir = libstore.library_dir(dst_home)
    dst_dir.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"{src_db.resolve().as_uri()}?mode=ro", uri=True)
    dest = sqlite3.connect(libstore.library_db_path(dst_dir))
    try:
        source.backup(dest)
    finally:
        dest.close()
        source.close()
    return dst_home


def copy_markdown(src_home: Path, dst_home: Path, storage_uris: list[str]) -> None:
    for uri in storage_uris:
        src = libstore.library_dir(src_home) / uri
        if src.is_file():
            dst = libstore.library_dir(dst_home) / uri
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)


# --- calling a server ---------------------------------------------------------------------------

class Probe:
    """One server (one MCP connection) over a home."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.server = build_server(home=home)

    @property
    def connection_id(self) -> str:
        return self.server.seedgraph_context.worker_id

    def raw(self, tool: str, args: dict[str, Any], meta: dict[str, Any] | None = None):
        async def _driver():
            async with create_connected_server_and_client_session(self.server._mcp_server) as client:
                return await client.call_tool(tool, args, meta=meta)
        return anyio.run(_driver)

    def ok(self, tool: str, args: dict[str, Any], meta: dict[str, Any] | None = None) -> dict:
        res = self.raw(tool, args, meta)
        assert res.isError is False, f"{tool} refused: {res.content[0].text[:200]}"
        return json.loads(res.content[0].text)

    def err(self, tool: str, args: dict[str, Any]) -> dict:
        res = self.raw(tool, args)
        assert res.isError is True, f"{tool} was expected to refuse"
        text = res.content[0].text
        return json.loads(text[text.index("{"):])

    def close(self) -> None:
        close_connection(self.server.seedgraph_context)


def open_lib(home: Path):
    return libstore.open_library_at(libstore.library_dir(home))


# --- finding material in a library --------------------------------------------------------------

@dataclass(frozen=True)
class Target:
    span_id: str
    paper_id: str
    start: int
    end: int
    query: str


def _query(text: str) -> str | None:
    words = re.findall(r"[A-Za-z0-9]+", text)  # consecutive words: the search matches one phrase
    return " ".join(words[:6]) if len(words) >= 4 else None


def spans(home: Path, limit_papers: int = 30) -> dict[str, list[Target]]:
    """Papers with at least two spans of usable length, each with a phrase query that matches
    it. Keyed by paper id, in id order."""
    lib = open_lib(home)
    try:
        rows = lib.conn.execute(
            "SELECT e.span_id, pf.paper_id, e.start_char, e.end_char, e.exact_quote FROM evidence_spans e "
            "JOIN paper_files pf ON pf.markdown_id = e.markdown_id "
            "WHERE length(e.exact_quote) BETWEEN 30 AND 400 AND e.end_char > e.start_char "
            "ORDER BY pf.paper_id, e.start_char, e.span_id").fetchall()
    finally:
        lib.close()
    out: dict[str, list[Target]] = {}
    for sid, pid, a, b, quote in rows:
        q = _query(quote)
        if q:
            out.setdefault(pid, []).append(Target(sid, pid, a, b, q))
    return dict(list(sorted((p, t) for p, t in out.items() if len(t) >= 2))[:limit_papers])


def storage_uris(home: Path, paper_ids: list[str]) -> list[str]:
    lib = open_lib(home)
    try:
        return [r[0] for r in lib.conn.execute(
            "SELECT m.storage_uri FROM paper_files pf JOIN markdown_documents m ON m.markdown_id = pf.markdown_id "
            "WHERE pf.paper_id IN (SELECT value FROM json_each(?))", (json.dumps(paper_ids),))]
    finally:
        lib.close()


def search_ids(reply: dict) -> set[str]:
    return {h["span_id"] for h in reply["hits"] if h.get("span_id")}


def records_of(home: Path, session: str) -> list[sqlite3.Row]:
    lib = open_lib(home)
    try:
        return lib.conn.execute(
            "SELECT r.* FROM traversal_records r JOIN record_sessions s ON s.session_id = r.session_id "
            "WHERE s.name = ? ORDER BY r.opened_at", (session,)).fetchall()
    finally:
        lib.close()


def flags(reply: dict) -> list[str]:
    return [f["flag"] for f in reply["flags"]]


# --- §10.2: the planted unread citation ---------------------------------------------------------

def planted_citation(src_home: Path, work_home: Path, tag: str = "plant-1432") -> dict[str, Any]:
    """Run the four planted-citation scenarios and the unknown-id refusal on a copy of
    ``src_home`` placed at ``work_home``. Returns the flags each scenario produced."""
    copy_library(src_home, work_home)
    by_paper = spans(work_home)
    if len(by_paper) < 2:
        pytest.skip("the library has fewer than two papers with two usable spans each")
    (pa, ta), (pb, tb) = list(by_paper.items())[:2]
    copy_markdown(src_home, work_home, storage_uris(work_home, [pa, pb]))
    probe = Probe(work_home)
    result: dict[str, Any] = {}

    # 1. search, read one passage, cite a span of another paper that no reply returned
    s1 = f"{tag}-a"
    hit = search_ids(probe.ok("search", {"query": ta[0].query, "session": s1}))
    read = ta[1] if ta[1].span_id not in hit else ta[0]
    probe.ok("paper_read", {"paper_id": pa, "start": read.start, "end": read.end, "session": s1})
    unreturned = next((t for t in tb if t.span_id not in hit), None)
    if unreturned is None:
        pytest.skip("every span of the second paper was a search hit")
    r1 = probe.ok("record_answer", {**ANSWER, "evidence": [{"span_id": unreturned.span_id}], "session": s1})
    result["never_returned"] = flags(r1)

    # 2. cite a span that was only in search hits
    s2 = f"{tag}-b"
    hit2 = search_ids(probe.ok("search", {"query": ta[0].query, "session": s2}))
    if ta[0].span_id not in hit2:
        pytest.skip("the probe query did not return its own span")
    r2 = probe.ok("record_answer", {**ANSWER, "evidence": [{"span_id": ta[0].span_id}], "session": s2})
    result["search_only"] = flags(r2)

    # 3. cite a span seen only as a paper_show anchor
    s3 = f"{tag}-c"
    anchor = _anchor_only(probe, work_home, s3)
    if anchor is None:
        pytest.skip("no paper with a noted claim anchor in the library")
    r3 = probe.ok("record_answer", {**ANSWER, "evidence": [{"span_id": anchor}], "session": s3})
    result["anchor_only"] = flags(r3)

    # 4. cite the passage that was read
    s4 = f"{tag}-d"
    probe.ok("search", {"query": ta[0].query, "session": s4})
    probe.ok("paper_read", {"paper_id": pa, "start": ta[0].start, "end": ta[0].end, "session": s4})
    r4 = probe.ok("record_answer", {**ANSWER, "evidence": [{"span_id": ta[0].span_id}], "session": s4})
    result["read"] = flags(r4)
    result["read_next_action"] = r4["next_action"]

    # 5. an id that does not exist is refused, nothing stored
    before = len(records_of(work_home, f"{tag}-e"))
    result["unknown_code"] = probe.err("record_answer", {**ANSWER, "evidence": [{"span_id": "sp_no_such_span"}],
                                                          "session": f"{tag}-e"})["code"]
    result["unknown_stored"] = len(records_of(work_home, f"{tag}-e")) - before
    return result


def _anchor_only(probe: Probe, home: Path, session: str) -> str | None:
    lib = open_lib(home)
    try:
        papers = [r[0] for r in lib.conn.execute(
            "SELECT DISTINCT ec.paper_id FROM extracted_claims ec JOIN claim_spans cs ON cs.claim_id = ec.claim_id "
            "ORDER BY ec.paper_id LIMIT 20")]
    finally:
        lib.close()
    for pid in papers:
        note = (probe.ok("paper_show", {"paper_id": pid, "session": session}).get("current_note") or {})
        for claim in note.get("claims", []):
            for a in claim.get("anchors", []):
                if a.get("span_id"):
                    return a["span_id"]
    return None


# --- §10.3: path without answer -----------------------------------------------------------------

def path_without_answer(src_home: Path, work_home: Path) -> dict[str, Any]:
    from datetime import datetime, timedelta, timezone

    from seedgraph.records import manage
    from seedgraph.records import path as P
    from seedgraph.web.views import home as home_view
    from seedgraph.web.views import records as records_view

    copy_library(src_home, work_home)
    by_paper = spans(work_home)
    if not by_paper:
        pytest.skip("no paper with usable spans")
    query = next(iter(by_paper.values()))[0].query
    idle, closed = Probe(work_home), Probe(work_home)
    idle.ok("search", {"query": query, "session": "idle-1432"})
    closed.ok("search", {"query": query, "session": "closed-1432"})
    closed.close()
    now = datetime.now(timezone.utc)
    lib = open_lib(work_home)
    try:
        def states(at):
            return {r.session_name: r.state for r in manage.list_records(lib, limit=10_000, now=at).rows
                    if r.session_name in ("idle-1432", "closed-1432")}
        out: dict[str, Any] = {"now": states(now), "later": states(now + P.PATH_IDLE + timedelta(minutes=1))}
        later = now + P.PATH_IDLE + timedelta(minutes=1)
        out["feed_later"] = [i.text for i in records_view.feed(lib, now=later)]
        out["feed_now"] = [i.text for i in records_view.feed(lib, now=now)]
        out["home_later"] = [e.text for e in home_view.activity(lib, now=later).entries if e.what == "record"]
        out["list_filter_later"] = sorted(r.session_name for r in manage.list_records(
            lib, manage.Filters(state=P.STATE_NO_ANSWER), limit=10_000, now=later).rows)
        return out
    finally:
        lib.close()


# --- §10.4: the identity probe ------------------------------------------------------------------

def identity_probe(src_home: Path, work_home: Path) -> dict[str, Any]:
    from seedgraph.records import manage

    copy_library(src_home, work_home)
    by_paper = spans(work_home)
    if not by_paper:
        pytest.skip("no paper with usable spans")
    papers = list(by_paper)
    t0 = by_paper[papers[0]][0]
    copy_markdown(src_home, work_home, storage_uris(work_home, papers[:1]))
    out: dict[str, Any] = {}

    # a parent and two helpers, named, on one connection; and an independent named stream
    p = Probe(work_home)
    name = "idprobe-1432"
    p.ok("search", {"query": t0.query, "session": name}, meta={"probe_marker": "never-stored"})
    p.ok("search", {"query": t0.query, "session": f"{name}/a"})
    p.ok("paper_read", {"paper_id": t0.paper_id, "start": t0.start, "end": t0.end, "session": f"{name}/b"})
    p.ok("search", {"query": t0.query, "session": "idother-1432"})
    lib = open_lib(work_home)
    try:
        rows = {r.session_name: r for r in manage.list_records(lib, limit=10_000).rows
                if r.session_name in (name, "idother-1432")}
        out["named_records"] = len(rows)
        parent = manage.get_record(lib, rows[name].record_id)
        out["parent_events"] = sorted((e.helper or "", e.tool) for e in parent.events)
        other = manage.get_record(lib, rows["idother-1432"].record_id)
        out["other_events"] = len(other.events)
        out["cross_owned"] = rows[name].record_id == rows["idother-1432"].record_id
    finally:
        lib.close()

    # the same without names: one connection stream, marked
    u = Probe(work_home)
    u.ok("search", {"query": t0.query})
    u.ok("paper_read", {"paper_id": t0.paper_id, "start": t0.start, "end": t0.end})
    lib = open_lib(work_home)
    try:
        mine = [r for r in manage.list_records(lib, limit=10_000).rows if r.key_kind == "connection"
                and manage.get_record(lib, r.record_id).events[0].connection_id == u.connection_id]
        out["unnamed_records"] = len(mine)
        out["unnamed_notes"] = list(mine[0].notes) if mine else []
        out["unnamed_events"] = len(manage.get_record(lib, mine[0].record_id).events) if mine else 0
    finally:
        lib.close()

    # a restart mid-question with a name keeps one record; without a name it splits
    r1 = Probe(work_home)
    r1.ok("search", {"query": t0.query, "session": "restart-1432"})
    r1.ok("search", {"query": t0.query})
    r1.close()
    r2 = Probe(work_home)
    r2.ok("paper_read", {"paper_id": t0.paper_id, "start": t0.start, "end": t0.end, "session": "restart-1432"})
    r2.ok("paper_read", {"paper_id": t0.paper_id, "start": t0.start, "end": t0.end})
    lib = open_lib(work_home)
    try:
        named = [r for r in manage.list_records(lib, limit=10_000).rows if r.session_name == "restart-1432"]
        out["restart_named_records"] = len(named)
        out["restart_named_events"] = len(manage.get_record(lib, named[0].record_id).events) if named else 0
        out["restart_unnamed_streams"] = lib.conn.execute(
            "SELECT COUNT(DISTINCT e.session_id) FROM traversal_events e WHERE e.connection_id IN (?, ?) "
            "AND e.session_id IN (SELECT session_id FROM record_sessions WHERE key_kind = 'connection')",
            (r1.connection_id, r2.connection_id)).fetchone()[0]
        row = lib.conn.execute("SELECT client, meta_keys_json FROM record_connections WHERE connection_id = ?",
                               (p.connection_id,)).fetchone()
        out["meta"] = {"client": row[0], "keys": json.loads(row[1] or "[]"), "values_stored": "never-stored" in row[1]}
    finally:
        lib.close()
    return out
