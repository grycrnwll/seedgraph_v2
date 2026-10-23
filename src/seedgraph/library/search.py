"""Library search over spans, claims and notes (v3 design §8.1).

One interface, :func:`search`, for the three FTS5 tables (``span_fts``,
``claim_fts``, ``note_fts``), plus :func:`match_set`, the unordered, unlimited
key set that parity (§8.2) compares against v2.

**Matching is v2's.** The ``MATCH`` string is built by v2's
:func:`seedgraph.fts.search.build_match_query` (the ``search --project`` /
``search_spans`` path): the whole query becomes one FTS5 phrase with internal
quotes doubled, so ``parallel trends`` matches the adjacent tokens verbatim and
FTS5 syntax in the query (``AR(1)``, ``-``, ``*``, ``OR``) is inert. The
library tables use v2's tokenizer (unicode61, remove_diacritics 2, no stemming).
An empty query returns ``[]``, as in v2. Results are ranked by ``bm25``.

**Graph scope.** ``graph=<slug>`` restricts hits to the graph's members (and its
frontier with ``include_frontier=True``). A span is in scope when its markdown is
the current ``paper_files.markdown_id`` of an in-scope paper; a span hit lists
every in-scope paper on that markdown (``paper_ids``), once per span.

**Audience** (§2.1 "Hidden from agents").
- ``agent``: notes and claims come only from the current versions (``note_current``
  and, for lens claims, the latest run per ``(paper_id, lens_id)``; claims with
  neither a note nor a lens always count, as in v2's ``CURRENT_CLAIMS_CTE``). Notes
  and claims of papers with an open ``hides_notes`` attention item are dropped, and
  labelled versions are never current. Spans are never hidden.
- ``researcher``: every version, each hit labelled with ``is_current``,
  ``version_label`` and ``hidden`` (what the agent audience would not see).

**Order** is ``rank`` then explicit ids, never rowid: spans by ``rank``, first
in-scope ``paper_id``, ``start_char``, ``span_id``; claims and notes by ``rank``,
``paper_id``, then the claim or note id.

**Pages** are the stored ``page_start``/``page_end`` that :mod:`seedgraph.segment`
wrote when the span was created. A claim takes the pages of its top anchor
(``claim_spans`` by ``rank``, then ``span_id``). Notes have no page.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Literal

from ..fts.search import build_match_query
from . import graphs
from .attention import HIDDEN_PAPERS_SQL
from .store import Library, LibraryError

Kind = Literal["spans", "claims", "notes"]
Audience = Literal["agent", "researcher"]
KINDS: tuple[str, ...] = ("spans", "claims", "notes")
AUDIENCES: tuple[str, ...] = ("agent", "researcher")


class SearchError(LibraryError):
    """A search request was refused (bad kind, audience or graph)."""


@dataclass(frozen=True)
class Hit:
    """One search result. Unused id fields are None for a given ``kind``."""

    kind: str
    rank: float
    paper_ids: tuple[str, ...]
    text: str
    span_id: str | None = None
    markdown_id: str | None = None
    markdown_hash: str | None = None
    section_id: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    claim_id: str | None = None
    note_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    is_current: bool = True
    version_label: str | None = None
    hidden: bool = False

    @property
    def paper_id(self) -> str | None:
        """The first (smallest) paper id, or None for a span on no paper's current markdown."""
        return self.paper_ids[0] if self.paper_ids else None

    @property
    def key(self) -> tuple[str, int, int] | str:
        """The parity key (§8.2): ``(markdown_hash, start_char, end_char)`` for a span,
        else the claim or note id."""
        if self.kind == "spans":
            assert self.markdown_hash is not None
            assert self.start_char is not None and self.end_char is not None
            return (self.markdown_hash, self.start_char, self.end_char)
        if self.kind == "claims":
            assert self.claim_id is not None
            return self.claim_id
        assert self.note_id is not None
        return self.note_id


def match_query(query: str) -> str:
    """The FTS5 ``MATCH`` string for ``query``: v2's phrase-safe builder, unchanged."""
    return build_match_query(query)


def _scope_ids(lib: Library, graph: str, include_frontier: bool) -> list[str]:
    statuses = ("member", "frontier") if include_frontier else ("member",)
    try:
        g = graphs.open_graph(lib, graph)
    except graphs.GraphError as exc:
        raise SearchError(str(exc)) from exc
    with g:
        return [p.paper_id for p in graphs.membership(g) if p.status in statuses]


def search(
    lib: Library,
    query: str,
    *,
    kind: Kind,
    graph: str | None = None,
    include_frontier: bool = False,
    paper_id: str | None = None,
    section_kind: str | None = None,
    audience: Audience = "agent",
    limit: int | None = 20,
) -> list[Hit]:
    """Search ``kind`` for ``query``; see the module docstring for every rule.

    ``section_kind`` applies to spans only. ``limit=None`` returns every match.
    """
    if kind not in KINDS:
        raise SearchError(f"invalid search kind {kind!r}; expected one of {KINDS}")
    if audience not in AUDIENCES:
        raise SearchError(f"invalid audience {audience!r}; expected one of {AUDIENCES}")
    if not query.strip():
        return []
    scope = None if graph is None else _scope_ids(lib, graph, include_frontier)
    match = match_query(query)
    conn = lib.conn
    if kind == "spans":
        return _spans(conn, match, scope, paper_id, section_kind, limit)
    if kind == "claims":
        return _claims(conn, match, scope, paper_id, audience, limit)
    return _notes(conn, match, scope, paper_id, audience, limit)


def match_set(
    lib: Library,
    query: str,
    *,
    kind: Kind,
    graph: str | None = None,
    include_frontier: bool = False,
    paper_id: str | None = None,
    section_kind: str | None = None,
    audience: Audience = "agent",
) -> frozenset:
    """Every match's :attr:`Hit.key`, unlimited and unordered (for parity, §8.2)."""
    hits = search(lib, query, kind=kind, graph=graph, include_frontier=include_frontier,
                  paper_id=paper_id, section_kind=section_kind, audience=audience, limit=None)
    return frozenset(h.key for h in hits)


# --- internals ---------------------------------------------------------------------------------

def _limit_clause(limit: int | None, params: list[object]) -> str:
    if limit is None:
        return ""
    params.append(int(limit))
    return " LIMIT ?"


def _scope_cte(scope: list[str] | None, params: list[object]) -> str:
    """A ``scope(paper_id)`` CTE over the graph's in-scope papers (json_each, no temp table)."""
    if scope is None:
        return ""
    params.append(json.dumps(sorted(scope)))
    return "scope(paper_id) AS (SELECT value FROM json_each(?)), "


def _spans(
    conn: sqlite3.Connection,
    match: str,
    scope: list[str] | None,
    paper_id: str | None,
    section_kind: str | None,
    limit: int | None,
) -> list[Hit]:
    params: list[object] = []
    cte = _scope_cte(scope, params)
    in_scope = " AND pf.paper_id IN (SELECT paper_id FROM scope)" if scope is not None else ""
    sql = [
        f"WITH {cte}hits AS (",
        "SELECT e.span_id, e.markdown_id, e.markdown_hash, e.section_id, e.start_char, "
        "       e.end_char, e.exact_quote, e.page_start, e.page_end, bm25(span_fts) AS rank, "
        "       (SELECT MIN(pf.paper_id) FROM paper_files pf "
        f"        WHERE pf.markdown_id = e.markdown_id{in_scope}) AS first_paper "
        "FROM span_fts JOIN evidence_spans e ON e.span_id = span_fts.span_id "
        "LEFT JOIN document_sections d ON d.section_id = e.section_id "
        "WHERE span_fts MATCH ?",
    ]
    params.append(match)
    if scope is not None:
        sql.append(
            "AND EXISTS (SELECT 1 FROM paper_files pf WHERE pf.markdown_id = e.markdown_id"
            f"{in_scope})"
        )
    if paper_id is not None:
        sql.append(
            "AND EXISTS (SELECT 1 FROM paper_files pf WHERE pf.markdown_id = e.markdown_id "
            f"AND pf.paper_id = ?{in_scope})"
        )
        params.append(paper_id)
    if section_kind is not None:
        sql.append("AND d.section_kind = ?")
        params.append(section_kind)
    sql.append(") SELECT * FROM hits ORDER BY rank, first_paper IS NULL, first_paper, "
               "start_char, span_id")
    sql.append(_limit_clause(limit, params))
    rows = conn.execute(" ".join(sql), params).fetchall()

    papers = _papers_on_markdowns(conn, {r["markdown_id"] for r in rows}, scope)
    return [
        Hit(
            kind="spans", rank=float(r["rank"]), paper_ids=papers.get(r["markdown_id"], ()),
            text=r["exact_quote"], span_id=r["span_id"], markdown_id=r["markdown_id"],
            markdown_hash=r["markdown_hash"], section_id=r["section_id"],
            start_char=r["start_char"], end_char=r["end_char"],
            page_start=r["page_start"], page_end=r["page_end"],
        )
        for r in rows
    ]


def _papers_on_markdowns(
    conn: sqlite3.Connection, markdown_ids: set[str], scope: list[str] | None
) -> dict[str, tuple[str, ...]]:
    """markdown_id -> sorted in-scope paper ids whose current markdown it is."""
    if not markdown_ids:
        return {}
    allowed = None if scope is None else set(scope)
    out: dict[str, list[str]] = {}
    rows = conn.execute(
        "SELECT markdown_id, paper_id FROM paper_files "
        "WHERE markdown_id IN (SELECT value FROM json_each(?)) ORDER BY markdown_id, paper_id",
        (json.dumps(sorted(markdown_ids)),),
    )
    for md, pid in rows:
        if allowed is None or pid in allowed:
            out.setdefault(md, []).append(pid)
    return {md: tuple(pids) for md, pids in out.items()}


#: v2's ``current_lens_runs`` without the rowid tie-break: the latest run per
#: (paper_id, lens_id) by created_at, then extraction_run_id.
_CURRENT_LENS_RUNS = (
    "current_lens_runs AS (SELECT extraction_run_id FROM ("
    " SELECT extraction_run_id, ROW_NUMBER() OVER ("
    "  PARTITION BY paper_id, lens_id ORDER BY created_at DESC, extraction_run_id DESC"
    " ) AS rn FROM extraction_runs WHERE lens_id IS NOT NULL) WHERE rn = 1), "
    f"hidden_papers(paper_id) AS ({HIDDEN_PAPERS_SQL}), "
    "current_notes(note_id) AS (SELECT note_id FROM note_current)"
)

#: v2's ``current_claims`` predicate over library tables (claims ``c``, runs ``er``).
_CLAIM_IS_CURRENT = (
    "(CASE WHEN c.structured_note_id IS NOT NULL "
    "      THEN c.structured_note_id IN (SELECT note_id FROM current_notes) "
    "      WHEN er.lens_id IS NOT NULL "
    "      THEN c.extraction_run_id IN (SELECT extraction_run_id FROM current_lens_runs) "
    "      ELSE 1 END)"
)


def _claims(
    conn: sqlite3.Connection,
    match: str,
    scope: list[str] | None,
    paper_id: str | None,
    audience: str,
    limit: int | None,
) -> list[Hit]:
    params: list[object] = []
    cte = _scope_cte(scope, params)
    sql = [
        f"WITH {cte}{_CURRENT_LENS_RUNS}, hits AS (",
        "SELECT c.claim_id, c.paper_id, c.structured_note_id, c.normalized_label, c.claim_text, "
        "       bm25(claim_fts) AS rank, n.version_label, "
        f"      {_CLAIM_IS_CURRENT} AS is_current, "
        "       c.paper_id IN (SELECT paper_id FROM hidden_papers) AS paper_hidden, "
        "       (SELECT cs.span_id FROM claim_spans cs WHERE cs.claim_id = c.claim_id "
        "        ORDER BY cs.rank, cs.span_id LIMIT 1) AS span_id "
        "FROM claim_fts JOIN extracted_claims c ON c.claim_id = claim_fts.claim_id "
        "LEFT JOIN structured_notes n ON n.note_id = c.structured_note_id "
        "LEFT JOIN extraction_runs er ON er.extraction_run_id = c.extraction_run_id "
        "WHERE claim_fts MATCH ?",
    ]
    params.append(match)
    if scope is not None:
        sql.append("AND c.paper_id IN (SELECT paper_id FROM scope)")
    if paper_id is not None:
        sql.append("AND c.paper_id = ?")
        params.append(paper_id)
    sql.append(") SELECT h.*, e.markdown_id, e.page_start, e.page_end FROM hits h "
               "LEFT JOIN evidence_spans e ON e.span_id = h.span_id")
    if audience == "agent":
        sql.append("WHERE h.is_current AND NOT h.paper_hidden AND h.version_label IS NULL")
    sql.append("ORDER BY h.rank, h.paper_id, h.claim_id")
    sql.append(_limit_clause(limit, params))
    return [
        Hit(
            kind="claims", rank=float(r["rank"]), paper_ids=(r["paper_id"],),
            text=r["claim_text"] or r["normalized_label"] or "", claim_id=r["claim_id"],
            note_id=r["structured_note_id"], span_id=r["span_id"], markdown_id=r["markdown_id"],
            page_start=r["page_start"], page_end=r["page_end"],
            is_current=bool(r["is_current"]), version_label=r["version_label"],
            hidden=bool(r["paper_hidden"]) or r["version_label"] is not None,
        )
        for r in conn.execute(" ".join(sql), params)
    ]


def _notes(
    conn: sqlite3.Connection,
    match: str,
    scope: list[str] | None,
    paper_id: str | None,
    audience: str,
    limit: int | None,
) -> list[Hit]:
    params: list[object] = []
    cte = _scope_cte(scope, params)
    sql = [
        f"WITH {cte}{_CURRENT_LENS_RUNS}, hits AS (",
        "SELECT n.note_id, n.paper_id, n.markdown_id, n.note_text, n.version_label, "
        "       bm25(note_fts) AS rank, "
        "       n.note_id IN (SELECT note_id FROM current_notes) AS is_current, "
        "       n.paper_id IN (SELECT paper_id FROM hidden_papers) AS paper_hidden "
        "FROM note_fts JOIN structured_notes n ON n.note_id = note_fts.note_id "
        "WHERE note_fts MATCH ?",
    ]
    params.append(match)
    if scope is not None:
        sql.append("AND n.paper_id IN (SELECT paper_id FROM scope)")
    if paper_id is not None:
        sql.append("AND n.paper_id = ?")
        params.append(paper_id)
    sql.append(") SELECT * FROM hits")
    if audience == "agent":
        sql.append("WHERE is_current AND NOT paper_hidden AND version_label IS NULL")
    sql.append("ORDER BY rank, paper_id, note_id")
    sql.append(_limit_clause(limit, params))
    return [
        Hit(
            kind="notes", rank=float(r["rank"]), paper_ids=(r["paper_id"],), text=r["note_text"],
            note_id=r["note_id"], markdown_id=r["markdown_id"],
            is_current=bool(r["is_current"]), version_label=r["version_label"],
            hidden=bool(r["paper_hidden"]) or r["version_label"] is not None,
        )
        for r in conn.execute(" ".join(sql), params)
    ]
