"""Library and context-graph read tools for the v3 MCP server (design §7.2).

Everything here is read-only and uses the agent audience: a paper with an open attention
item that hides notes (for example a possible wrong PDF) shows no note, no claims and no
lens results, only ``notes_hidden: true`` and the item. Paper text (``paper_read``) is
never hidden; it is the paper itself.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .server_v3 import READ, tool_error

if TYPE_CHECKING:
    import sqlite3

    from mcp.server.fastmcp import FastMCP

    from ..library.store import Library
    from .server_v3 import V3Context

PAPER_STATUSES = ("held", "cited_only", "member", "frontier")
NEEDS = ("note", "metadata")
SEARCH_KINDS = ("spans", "claims", "notes")
MAX_LIST = 200
MAX_SEARCH = 100
DEFAULT_READ_CHARS = 12_000
MAX_READ_CHARS = 48_000


# --- shared helpers ---------------------------------------------------------------------------

def _hidden(conn: sqlite3.Connection) -> frozenset[str]:
    from ..library.attention import hidden_paper_ids

    return hidden_paper_ids(conn)


def _paper_row(conn: sqlite3.Connection, paper_id: str) -> sqlite3.Row | tuple:
    row = conn.execute(
        "SELECT p.paper_id, p.title, p.year, p.authors_json, p.venue, p.title_source, p.abstract, s.status "
        "FROM papers p JOIN paper_status s ON s.paper_id = p.paper_id WHERE p.paper_id = ?",
        (paper_id,)).fetchone()
    if row is None:
        tool_error("paper_not_found", f"no paper {paper_id!r} in the library", paper_id=paper_id)
    return row


def _authors(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(a) for a in data] if isinstance(data, list) else []


def _brief(conn: sqlite3.Connection, paper_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT title, year FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    return {"paper_id": paper_id, "title": row[0] if row else None, "year": row[1] if row else None}


def _current_note(conn: sqlite3.Connection, paper_id: str) -> str | None:
    from ..notes import current_note_id

    return current_note_id(conn, paper_id)


def _members(lib: Library, graph: str) -> list[Any]:
    from ..library import graphs

    with graphs.open_graph(lib, graph) as g:
        return graphs.membership(g)


def _bounded(value: int, name: str, lower: int, upper: int) -> int:
    if not lower <= value <= upper:
        tool_error("invalid_argument", f"{name} must be between {lower} and {upper}, not {value}")
    return value


def _markdown(lib: Library, paper_id: str) -> tuple[str, str]:
    from ..notes import read_markdown
    from ..work.packages import primary_markdown

    md = primary_markdown(lib.conn, paper_id)
    text = read_markdown(lib.conn, md[0]) if md else None
    if md is None or text is None:
        tool_error("no_text", f"paper {paper_id!r} has no converted text", paper_id=paper_id)
    return md[0], text


def _sections(conn: sqlite3.Connection, markdown_id: str) -> list[dict[str, Any]]:
    return [{"section_id": r[0], "heading": r[1], "section_kind": r[2], "start_char": r[3], "end_char": r[4],
             "page_start": r[5], "page_end": r[6]}
            for r in conn.execute(
                "SELECT section_id, heading_text, section_kind, start_char, end_char, page_start, page_end "
                "FROM document_sections WHERE markdown_id = ? ORDER BY ordinal", (markdown_id,))]


# --- graph tools ------------------------------------------------------------------------------

def _graph_list(lib: Library) -> dict[str, Any]:
    from ..library import graphs

    out = []
    for info in graphs.list_graphs(lib):
        rows = _members(lib, info.slug)
        out.append({"graph": info.slug, "display_name": info.display_name, "created_at": info.created_at,
                    "members": sum(1 for r in rows if r.status == "member"),
                    "frontier": sum(1 for r in rows if r.status == "frontier"),
                    "seeds": sum(1 for r in rows if r.is_seed)})
    return {"graphs": out}


def _graph_show(lib: Library, graph: str) -> dict[str, Any]:
    from ..library import graphs

    with graphs.open_graph(lib, graph) as g:
        info = graphs.graph_info(g)
        rows = graphs.membership(g)
        lenses = graphs.chosen_lenses(g)
    conn = lib.conn
    needing = [_brief(conn, r.paper_id) for r in rows
               if r.status == "member" and _current_note(conn, r.paper_id) is None]
    return {
        "graph": info.slug, "display_name": info.display_name, "created_at": info.created_at,
        "counts": {"member": sum(1 for r in rows if r.status == "member"),
                   "frontier": sum(1 for r in rows if r.status == "frontier"),
                   "noted_members": sum(1 for r in rows if r.status == "member") - len(needing)},
        "seeds": [_brief(conn, r.paper_id) for r in rows if r.is_seed],
        "lenses": lenses,
        "papers_needing_notes": needing,
    }


# --- paper tools ------------------------------------------------------------------------------

def _paper_list(lib: Library, graph: str | None, status: str | None, needs: str | None, query: str | None,
                offset: int, limit: int) -> dict[str, Any]:
    from ..work.tasks_metadata import needs_metadata

    if status is not None and status not in PAPER_STATUSES:
        tool_error("invalid_argument", f"status must be one of {PAPER_STATUSES}, not {status!r}")
    if status in ("member", "frontier") and graph is None:
        tool_error("invalid_argument", f"status {status!r} needs a graph")
    if needs is not None and needs not in NEEDS:
        tool_error("invalid_argument", f"needs must be one of {NEEDS}, not {needs!r}")
    _bounded(limit, "limit", 1, MAX_LIST)
    _bounded(offset, "offset", 0, 10**9)
    conn = lib.conn
    scope = {r.paper_id: r for r in _members(lib, graph)} if graph is not None else None
    hidden = _hidden(conn)
    rows = conn.execute(
        "SELECT p.paper_id, p.title, p.year, p.authors_json, s.status FROM papers p "
        "JOIN paper_status s ON s.paper_id = p.paper_id ORDER BY p.paper_id").fetchall()
    needle = query.strip().lower() if query else None
    out: list[dict[str, Any]] = []
    for pid, title, year, authors, held in rows:
        membership = scope.get(pid) if scope is not None else None
        if scope is not None and membership is None:
            continue
        if status in ("held", "cited_only") and held != status:
            continue
        if status in ("member", "frontier") and membership.status != status:
            continue
        if needle and needle not in (title or "").lower():
            continue
        has_note = _current_note(conn, pid) is not None
        if needs == "note" and (held != "held" or has_note):
            continue
        if needs == "metadata" and not needs_metadata(conn, pid):
            continue
        row: dict[str, Any] = {"paper_id": pid, "title": title, "year": year, "authors": _authors(authors),
                               "status": held, "has_current_note": has_note, "notes_hidden": pid in hidden}
        if membership is not None:
            row["graph_status"] = membership.status
            row["is_seed"] = membership.is_seed
        out.append(row)
    page = out[offset:offset + limit]
    nxt = offset + limit if offset + limit < len(out) else None
    return {"total": len(out), "offset": offset, "limit": limit, "next_offset": nxt, "papers": page}


def _claims(conn: sqlite3.Connection, note_id: str) -> list[dict[str, Any]]:
    out = []
    for cid, field_key, ctype, csub, label, text, status, plain in conn.execute(
            "SELECT claim_id, field_key, claim_type, claim_subtype, normalized_label, claim_text, status, "
            "plain_words FROM extracted_claims WHERE structured_note_id = ? ORDER BY field_key, claim_id",
            (note_id,)):
        spans = conn.execute(
            "SELECT e.span_id, e.page_start, e.page_end FROM claim_spans c JOIN evidence_spans e "
            "ON e.span_id = c.span_id WHERE c.claim_id = ? ORDER BY c.rank, c.span_id", (cid,)).fetchall()
        out.append({"claim_id": cid, "field_key": field_key, "claim_type": ctype, "claim_subtype": csub,
                    "label": label, "claim_text": text, "status": status, "plain_words": plain,
                    "anchors": [{"span_id": s[0], "page_start": s[1], "page_end": s[2]} for s in spans]})
    return out


def _paper_show(lib: Library, paper_id: str) -> dict[str, Any]:
    from ..library import attention

    conn = lib.conn
    pid, title, year, authors, venue, title_source, abstract, held = _paper_row(conn, paper_id)
    files = [{"source_file_id": r[0], "markdown_id": r[1], "pdf_check_status": r[2]} for r in conn.execute(
        "SELECT source_file_id, markdown_id, pdf_check_status FROM paper_files WHERE paper_id = ? "
        "ORDER BY source_file_id", (pid,))]
    ids = [{"id_type": r[0], "id_value": r[1]} for r in conn.execute(
        "SELECT id_type, id_value FROM identifiers WHERE paper_id = ? ORDER BY id_type, id_value", (pid,))]
    items = attention.list_items(lib, status="open", paper_id=pid)
    hidden = pid in _hidden(conn)
    note_id = _current_note(conn, pid)
    note: dict[str, Any] | None = None
    if note_id is not None and not hidden:
        version_no, schema_version, created_at = conn.execute(
            "SELECT version_no, schema_version, created_at FROM structured_notes WHERE note_id = ?",
            (note_id,)).fetchone()
        claims = _claims(conn, note_id)
        note = {"note_id": note_id, "version_no": version_no, "schema_version": schema_version,
                "created_at": created_at,
                "claims_found": sum(1 for c in claims if c["status"] == "found"), "claims": claims}
    from ..work.packages import primary_markdown

    current = primary_markdown(conn, pid)
    return {
        "paper_id": pid, "title": title, "title_source": title_source, "authors": _authors(authors),
        "year": year, "venue": venue, "abstract": abstract, "status": held, "identifiers": ids, "files": files,
        "sections": _sections(conn, current[0]) if current else [],
        "has_text": current is not None,
        "current_note": note, "notes_hidden": hidden,
        "attention": [{"item_id": i.item_id, "kind": i.kind, "detail": i.detail, "hides_notes": i.hides_notes}
                      for i in items],
    }


def _paper_read(lib: Library, paper_id: str, start: int | None, end: int | None, section: str | None,
                max_chars: int) -> dict[str, Any]:
    _paper_row(lib.conn, paper_id)
    _bounded(max_chars, "max_chars", 1, MAX_READ_CHARS)
    markdown_id, text = _markdown(lib, paper_id)
    total = len(text)
    chosen: dict[str, Any] | None = None
    if section is not None:
        if start is not None or end is not None:
            tool_error("invalid_argument", "give start/end or a section, not both")
        secs = _sections(lib.conn, markdown_id)
        wanted = section.strip().lower()
        matches = [s for s in secs if s["section_id"] == section] or \
                  [s for s in secs if (s["heading"] or "").strip().lower() == wanted]
        if not matches:
            tool_error("invalid_argument", f"no section {section!r} in this paper",
                       sections=[{"section_id": s["section_id"], "heading": s["heading"]} for s in secs])
        chosen = matches[0]
        lo, hi = chosen["start_char"], chosen["end_char"]
    else:
        lo = 0 if start is None else start
        hi = total if end is None else end
    if lo < 0 or hi < lo:
        tool_error("invalid_argument", f"need 0 <= start <= end, not start={lo}, end={hi}")
    lo, limit_hi = min(lo, total), min(hi, total)
    stop = min(limit_hi, lo + max_chars)
    return {"paper_id": paper_id, "markdown_id": markdown_id, "total_chars": total, "start": lo, "end": stop,
            "truncated": stop < limit_hi, "next_start": stop if stop < limit_hi else None,
            "section": None if chosen is None else {k: chosen[k] for k in ("section_id", "heading", "section_kind")},
            "text": text[lo:stop]}


# --- search and lenses ------------------------------------------------------------------------

def _hit(h: Any) -> dict[str, Any]:
    out = {"kind": h.kind, "rank": h.rank, "paper_ids": list(h.paper_ids), "text": h.text}
    for key in ("span_id", "claim_id", "note_id", "section_id", "start_char", "end_char", "page_start",
                "page_end"):
        value = getattr(h, key)
        if value is not None:
            out[key] = value
    return out


def _search(lib: Library, query: str, kind: str, graph: str | None, paper_id: str | None,
            section_kind: str | None, include_frontier: bool, limit: int) -> dict[str, Any]:
    from ..library import search as libsearch

    if kind not in SEARCH_KINDS:
        tool_error("invalid_argument", f"kind must be one of {SEARCH_KINDS}, not {kind!r}")
    _bounded(limit, "limit", 1, MAX_SEARCH)
    hits = libsearch.search(lib, query, kind=kind, graph=graph, include_frontier=include_frontier,  # type: ignore[arg-type]
                            paper_id=paper_id, section_kind=section_kind, audience="agent", limit=limit)
    return {"query": query, "kind": kind, "graph": graph, "hits": [_hit(h) for h in hits]}


def _lens_results(lib: Library, lens_id: str, graph: str | None, status: str) -> dict[str, Any]:
    from ..work import tasks_lens

    conn = lib.conn
    lens = tasks_lens.load_lens(conn, lens_id)
    scope: list[str] | None = None
    chosen = None
    if graph is not None:
        from ..library import graphs

        with graphs.open_graph(lib, graph) as g:
            scope = [r.paper_id for r in graphs.membership(g, status="member")]
            chosen = lens_id in graphs.chosen_lenses(g)
    hidden = _hidden(conn)
    rows = tasks_lens.lens_results(lib, lens_id, status=status, paper_ids=scope)
    out: dict[str, Any] = {"lens_id": lens_id, "graph": graph, "status": status,
                           "results": [r for r in rows if r["paper_id"] not in hidden]}
    if scope is not None:
        out["lens_chosen_for_graph"] = chosen
        out["papers_without_current_results"] = [
            pid for pid in scope if pid not in hidden and not tasks_lens.lens_is_current(conn, lens, pid)]
    return out


# --- registration -----------------------------------------------------------------------------

def register_library_tools(server: FastMCP, ctx: V3Context) -> None:
    """Register the read tools on ``server``; every body opens the library per call."""

    @server.tool(annotations=READ)
    def graph_list() -> dict[str, Any]:
        """Every context graph in the library with its member, frontier and seed counts."""
        with ctx.library() as lib:
            return _graph_list(lib)

    @server.tool(annotations=READ)
    def graph_show(graph: str) -> dict[str, Any]:
        """One context graph: membership counts, seeds, chosen lenses and the member papers
        that have no current note. `graph` is the graph slug from graph_list."""
        with ctx.library() as lib:
            return _graph_show(lib, graph)

    @server.tool(annotations=READ)
    def paper_list(graph: str | None = None, status: str | None = None, needs: str | None = None,
                   query: str | None = None, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """Library papers, ordered by paper id. `graph` limits to one context graph's papers;
        `status` is held, cited_only, member or frontier (the last two need `graph`);
        `needs` is note (held, no current note) or metadata; `query` matches title text."""
        with ctx.library() as lib:
            return _paper_list(lib, graph, status, needs, query, offset, limit)

    @server.tool(annotations=READ)
    def paper_show(paper_id: str) -> dict[str, Any]:
        """One paper: metadata, files, section outline, open attention items, and its current
        note's claims with anchor pages (hidden while an attention item hides its notes)."""
        with ctx.library() as lib:
            return _paper_show(lib, paper_id)

    @server.tool(annotations=READ)
    def search(query: str, kind: str = "spans", graph: str | None = None, paper_id: str | None = None,
               section_kind: str | None = None, include_frontier: bool = False,
               limit: int = 20) -> dict[str, Any]:
        """Full-text search of the library: `kind` spans (paper text), claims or notes.
        The query is matched as one phrase. `graph` limits to a context graph's members
        (`include_frontier` adds its frontier); `section_kind` applies to spans only."""
        with ctx.library() as lib:
            return _search(lib, query, kind, graph, paper_id, section_kind, include_frontier, limit)

    @server.tool(annotations=READ)
    def paper_read(paper_id: str, start: int | None = None, end: int | None = None,
                   section: str | None = None, max_chars: int = DEFAULT_READ_CHARS) -> dict[str, Any]:
        """A bounded slice of a paper's current markdown: character offsets `start`..`end`
        (half-open), or one `section` (section id or exact heading from paper_show). At most
        `max_chars` (up to 48000) per call; continue from `next_start`."""
        with ctx.library() as lib:
            return _paper_read(lib, paper_id, start, end, section, max_chars)

    @server.tool(annotations=READ)
    def lens_results(lens_id: str, graph: str | None = None, status: str = "found") -> dict[str, Any]:
        """Current results of one lens (each paper's latest run matching the lens definition
        and the paper's current text). `graph` limits to its members and lists the members
        without current results; `status` is found, not_found, ambiguous or all."""
        with ctx.library() as lib:
            return _lens_results(lib, lens_id, graph, status)
