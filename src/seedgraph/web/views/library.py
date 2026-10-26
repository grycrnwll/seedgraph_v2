"""The Library page (design §2.2): list views with counts, the "Needs a PDF" view, the
title filter plus note search, and the "Write notes" estimate. Pure reads.

Views: ``held`` (default), ``cited-only``, ``needs-notes`` (held, no current note or a
stale one), ``held-back`` (open text / PDF-identity items, with ``accept_text`` where it
applies), ``needs-pdf`` (cited-only papers with a ``https://doi.org/`` link;
:func:`seedgraph.library.ingest.needs_pdf`) and ``all``. Each can be filtered by graph
and narrowed by a title search.

Rules chosen where the design is silent:
- The graph filter keeps papers in the graph with any status but ``excluded``.
- Note state: ``none`` (no note), ``current``, ``stale`` (the note was made on another
  markdown than the paper's current one), ``edited`` (a researcher overlay), ``newer
  extraction`` (edited, and a later extraction exists).
- Quotes anchored/total count the current claims with ``status = 'found'``; anchored ones
  have a span.
- 100 rows per page.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ...library import graphs as graphs_mod
from ...library.store import Library
from ...notes.format import SCHEMA_ID

VIEWS: tuple[tuple[str, str], ...] = (
    ("held", "Held"), ("cited-only", "Cited only"), ("needs-notes", "Needs notes"),
    ("held-back", "Held back"), ("needs-pdf", "Needs a PDF"), ("all", "All"),
)
VIEW_KEYS = frozenset(k for k, _ in VIEWS)
HELD_BACK_KINDS: tuple[str, ...] = ("text_garbled", "text_missing", "pdf_title_miss", "shared_pdf")
ACCEPT_TEXT_KINDS = frozenset({"text_garbled", "text_missing"})
PAGE_SIZE = 100


@dataclass(frozen=True)
class HeldBack:
    item_id: str
    kind: str
    reason: str
    can_accept_text: bool


@dataclass(frozen=True)
class PaperRow:
    paper_id: str
    title: str | None
    year: int | None
    authors: tuple[str, ...]
    held: bool
    graphs: tuple[str, ...]
    note_state: str          # none | current | stale | edited | newer extraction
    quotes_anchored: int
    quotes_total: int
    badges: tuple[str, ...] = ()
    held_back: tuple[HeldBack, ...] = ()


@dataclass(frozen=True)
class NeedsPdfRow:
    paper_id: str
    title: str | None
    year: int | None
    authors: tuple[str, ...]
    doi: str | None
    doi_url: str | None
    graphs: tuple[str, ...]


@dataclass(frozen=True)
class SearchHit:
    paper_id: str | None
    title: str | None
    text: str
    trust_label: str | None
    page: int | None


@dataclass(frozen=True)
class GraphOption:
    slug: str
    display_name: str


@dataclass(frozen=True)
class LibraryPage:
    view: str
    graph: str | None
    q: str
    frontier: bool
    page: int
    pages: int
    total: int
    rows: tuple[PaperRow, ...]
    needs_pdf: tuple[NeedsPdfRow, ...]
    counts: dict[str, int]
    graphs: tuple[GraphOption, ...]
    hits: tuple[SearchHit, ...] = ()
    views: tuple[tuple[str, str], ...] = VIEWS

    def query(self, **changes: object) -> str:
        """The page's query string with ``changes`` applied (for links)."""
        from urllib.parse import urlencode

        params: dict[str, object] = {"view": self.view, "graph": self.graph or "", "q": self.q}
        params.update(changes)
        return urlencode({k: v for k, v in params.items() if v not in ("", None, False)})


class UnknownGraph(LookupError):
    """The ``graph`` filter names no graph."""


# --- shared paper facts ----------------------------------------------------------------------


def _marks(n: int) -> str:
    return ",".join("?" * n)


def graph_index(lib: Library) -> dict[str, tuple[str, ...]]:
    """paper_id -> slugs of the graphs it is in (any status but ``excluded``)."""
    out: dict[str, list[str]] = {}
    for info in graphs_mod.list_graphs(lib):
        with graphs_mod.open_graph(lib, info.slug) as g:
            for gp in graphs_mod.membership(g):
                if gp.status != "excluded":
                    out.setdefault(gp.paper_id, []).append(info.slug)
    return {k: tuple(v) for k, v in out.items()}


def graph_options(lib: Library) -> tuple[GraphOption, ...]:
    return tuple(GraphOption(i.slug, i.display_name) for i in graphs_mod.list_graphs(lib))


def note_states(lib: Library, paper_ids: Sequence[str]) -> dict[str, str]:
    """paper_id -> note state (module doc) for held or cited-only papers."""
    ids = list(dict.fromkeys(paper_ids))
    out = {pid: "none" for pid in ids}
    conn = lib.conn
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        latest = {r[0]: (r[1], r[2]) for r in conn.execute(
            f"SELECT paper_id, note_id, markdown_hash FROM note_latest WHERE schema_id IS ? "
            f"AND paper_id IN ({_marks(len(chunk))})", [SCHEMA_ID, *chunk])}
        current_md = {r[0]: r[1] for r in conn.execute(
            f"SELECT pf.paper_id, m.markdown_hash FROM paper_files pf JOIN markdown_documents m "
            f"ON m.markdown_id = pf.markdown_id WHERE pf.role = 'primary' "
            f"AND pf.paper_id IN ({_marks(len(chunk))})", chunk)}
        overlays = {r[0]: r[1] for r in conn.execute(
            f"SELECT paper_id, base_note_id FROM note_overlays WHERE schema_id IS ? "
            f"AND paper_id IN ({_marks(len(chunk))})", [SCHEMA_ID, *chunk])}
        for pid in chunk:
            if pid not in latest:
                continue
            note_id, md_hash = latest[pid]
            if pid in overlays:
                out[pid] = "edited" if overlays[pid] == note_id else "newer extraction"
            else:
                out[pid] = "current" if current_md.get(pid) == md_hash else "stale"
    return out


def needs_notes(state: str) -> bool:
    return state in ("none", "stale")


def _quotes(lib: Library, ids: Sequence[str]) -> dict[str, tuple[int, int]]:
    if not ids:
        return {}
    return {r[0]: (int(r[1] or 0), int(r[2])) for r in lib.conn.execute(
        f"SELECT paper_id, SUM(span_id IS NOT NULL), COUNT(*) FROM current_claims "
        f"WHERE status = 'found' AND schema_id IS ? AND paper_id IN ({_marks(len(ids))}) GROUP BY paper_id",
        [SCHEMA_ID, *ids])}


def _held_back(lib: Library, ids: Sequence[str]) -> dict[str, list[HeldBack]]:
    if not ids:
        return {}
    out: dict[str, list[HeldBack]] = {}
    for item_id, kind, pid, detail in lib.conn.execute(
            f"SELECT item_id, kind, paper_id, detail FROM attention_items WHERE status = 'open' "
            f"AND kind IN ({_marks(len(HELD_BACK_KINDS))}) AND paper_id IN ({_marks(len(ids))}) "
            f"ORDER BY kind, item_id", [*HELD_BACK_KINDS, *ids]):
        out.setdefault(pid, []).append(HeldBack(item_id, kind, detail or kind.replace("_", " "),
                                                kind in ACCEPT_TEXT_KINDS))
    return out


def _badges(lib: Library, ids: Sequence[str]) -> dict[str, list[str]]:
    if not ids:
        return {}
    out: dict[str, list[str]] = {}
    for pid, n, hides in lib.conn.execute(
            f"SELECT paper_id, COUNT(*), MAX(hides_notes) FROM attention_items WHERE status = 'open' "
            f"AND paper_id IN ({_marks(len(ids))}) GROUP BY paper_id", list(ids)):
        out.setdefault(pid, []).append(f"{n} open item{'s' if n != 1 else ''}")
        if hides:
            out[pid].append("notes hidden from agents")
    for (pid,) in lib.conn.execute(
            f"SELECT paper_id FROM concept_refresh WHERE paper_id IN ({_marks(len(ids))})", list(ids)):
        out.setdefault(pid, []).append("concepts out of date")
    return out


def _authors(raw: str | None) -> tuple[str, ...]:
    try:
        return tuple(str(a) for a in json.loads(raw)) if raw else ()
    except (TypeError, ValueError):
        return ()


def paper_rows(lib: Library, paper_ids: Sequence[str], *, index: dict[str, tuple[str, ...]] | None = None
               ) -> list[PaperRow]:
    """Display rows for ``paper_ids`` (in that order; unknown ids are dropped)."""
    ids = list(dict.fromkeys(paper_ids))
    if not ids:
        return []
    index = graph_index(lib) if index is None else index
    base = {r[0]: r for r in lib.conn.execute(
        f"SELECT p.paper_id, p.title, p.year, p.authors_json, s.status FROM papers p "
        f"JOIN paper_status s ON s.paper_id = p.paper_id WHERE p.paper_id IN ({_marks(len(ids))})", ids)}
    states = note_states(lib, ids)
    quotes = _quotes(lib, ids)
    held_back = _held_back(lib, ids)
    badges = _badges(lib, ids)
    rows = []
    for pid in ids:
        if pid not in base:
            continue
        _, title, year, authors_json, status = base[pid]
        anchored, total = quotes.get(pid, (0, 0))
        rows.append(PaperRow(pid, title, year, _authors(authors_json), status == "held", index.get(pid, ()),
                             states[pid], anchored, total, tuple(badges.get(pid, ())),
                             tuple(held_back.get(pid, ()))))
    return rows


# --- the list --------------------------------------------------------------------------------


def _ordered_ids(lib: Library, q: str) -> list[tuple[str, str]]:
    """(paper_id, status) by title, narrowed by a case-insensitive title substring."""
    sql = ("SELECT p.paper_id, s.status FROM papers p JOIN paper_status s ON s.paper_id = p.paper_id")
    args: list[str] = []
    if q:
        sql += " WHERE p.title LIKE ? ESCAPE '\\'"
        args.append("%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
    sql += " ORDER BY p.title IS NULL, p.title COLLATE NOCASE, p.paper_id"
    return [(r[0], r[1]) for r in lib.conn.execute(sql, args)]


def _held_back_ids(lib: Library) -> set[str]:
    return {r[0] for r in lib.conn.execute(
        f"SELECT DISTINCT paper_id FROM attention_items WHERE status = 'open' AND paper_id IS NOT NULL "
        f"AND kind IN ({_marks(len(HELD_BACK_KINDS))})", HELD_BACK_KINDS)}


def view_ids(lib: Library, view: str, *, graph: str | None = None, q: str = "",
             index: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """The paper ids a list view shows, in display order (not ``needs-pdf``)."""
    index = graph_index(lib) if index is None else index
    pairs = _ordered_ids(lib, q.strip())
    if graph:
        pairs = [(p, s) for p, s in pairs if graph in index.get(p, ())]
    if view == "cited-only":
        return [p for p, s in pairs if s == "cited_only"]
    if view == "all":
        return [p for p, _ in pairs]
    held = [p for p, s in pairs if s == "held"]
    if view == "needs-notes":
        states = note_states(lib, held)
        return [p for p in held if needs_notes(states[p])]
    if view == "held-back":
        flagged = _held_back_ids(lib)
        return [p for p in held if p in flagged]
    return held


def _needs_pdf(lib: Library, graph: str | None, frontier: bool, q: str) -> list[NeedsPdfRow]:
    from ...library import ingest

    rows = ingest.needs_pdf(lib, graph=graph, frontier=frontier)
    needle = q.strip().lower()
    return [NeedsPdfRow(r.paper_id, r.title, r.year, tuple(r.authors), r.doi, r.doi_url, tuple(r.graphs))
            for r in rows if not needle or needle in (r.title or "").lower()]


def _hits(lib: Library, q: str, graph: str | None) -> tuple[SearchHit, ...]:
    from ...library import search

    if not q.strip():
        return ()
    hits = search.search(lib, q, kind="claims", graph=graph, audience="researcher", limit=20)
    hits = [h for h in hits if h.is_current]
    titles = {r[0]: r[1] for r in lib.conn.execute(
        f"SELECT paper_id, title FROM papers WHERE paper_id IN ({_marks(len(hits))})",
        [h.paper_id for h in hits])} if hits else {}
    return tuple(SearchHit(h.paper_id, titles.get(h.paper_id), h.text, h.trust_label, h.page_start)
                 for h in hits)


def page(lib: Library, *, view: str = "held", graph: str | None = None, q: str = "", page_no: int = 1,
         frontier: bool = False) -> LibraryPage:
    """The Library page. Raises :class:`UnknownGraph` for an unknown ``graph``."""
    view = view if view in VIEW_KEYS else "held"
    graph = graph or None
    options = graph_options(lib)
    if graph is not None and graph not in {o.slug for o in options}:
        raise UnknownGraph(graph)
    index = graph_index(lib)
    counts = {key: len(view_ids(lib, key, graph=graph, q=q, index=index))
              for key, _ in VIEWS if key != "needs-pdf"}
    needs = tuple(_needs_pdf(lib, graph, frontier, q))
    counts["needs-pdf"] = len(needs)
    rows: tuple[PaperRow, ...] = ()
    total = len(needs)
    page_no = max(1, page_no)
    if view != "needs-pdf":
        ids = view_ids(lib, view, graph=graph, q=q, index=index)
        total = len(ids)
        pages = max(1, -(-total // PAGE_SIZE))
        page_no = min(page_no, pages)
        rows = tuple(paper_rows(lib, ids[(page_no - 1) * PAGE_SIZE: page_no * PAGE_SIZE], index=index))
    pages = max(1, -(-total // PAGE_SIZE))
    return LibraryPage(view=view, graph=graph, q=q, frontier=frontier, page=min(page_no, pages), pages=pages,
                       total=total, rows=rows, needs_pdf=needs if view == "needs-pdf" else (),
                       counts=counts, graphs=options, hits=_hits(lib, q, graph))


def metadata_needed_count(lib: Library) -> int:
    (n,) = lib.conn.execute(
        "SELECT COUNT(*) FROM attention_items WHERE status = 'open' AND kind = 'metadata_needed'").fetchone()
    return int(n)


# --- the job strip ---------------------------------------------------------------------------


JOB_STEPS = {"ingest": "adding PDFs", "folder": "adding a folder", "doi": "looking up DOIs",
             "convert": "converting", "metadata": "metadata"}


@dataclass(frozen=True)
class JobRow:
    job_id: str
    kind: str
    label: str
    state: str          # queued | running | done | failed
    step: str           # e.g. "converting", "converted", "metadata done"
    lines: tuple[str, ...]
    finished_at: str | None


@dataclass(frozen=True)
class JobsStrip:
    active: bool
    rows: tuple[JobRow, ...]


def jobs_strip(jobs: Sequence, *, kinds: Iterable[str], limit: int = 20) -> JobsStrip:
    """The add-papers job strip from :class:`~seedgraph.web.jobs.Job` snapshots (newest
    first; at most ``limit`` rows, every active job included)."""
    wanted = frozenset(kinds)
    picked = [j for j in jobs if j.kind in wanted]
    rows = []
    for j in reversed(picked):
        step = JOB_STEPS.get(j.kind, j.kind)
        if j.state == "done":
            step = "converted" if j.kind == "convert" else f"{step}: done"
        elif j.state == "failed":
            step = f"{step}: failed"
        elif j.state == "queued":
            step = f"{step}: waiting"
        rows.append(JobRow(j.job_id, j.kind, j.label, j.state, step,
                           tuple(line for line in (j.message or "").splitlines() if line.strip()), j.finished_at))
    active = [r for r in rows if r.state in ("queued", "running")]
    rest = [r for r in rows if r.state not in ("queued", "running")]
    return JobsStrip(active=bool(active), rows=tuple(active + rest[: max(0, limit - len(active))]))


# --- "Write notes" ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EstimateRow:
    paper_id: str
    title: str | None
    held: bool
    state: str
    reason: str | None
    text_tokens: int
    agent_tokens_low: int
    agent_tokens_high: int


@dataclass(frozen=True)
class WriteNotesView:
    paper_ids: tuple[str, ...]            # the queued ones (what the batch is created with)
    queued: tuple[EstimateRow, ...]
    skipped: tuple[EstimateRow, ...]
    text_tokens: int
    agent_tokens_low: int
    agent_tokens_high: int
    replace: bool
    unknown: tuple[str, ...] = field(default_factory=tuple)


def write_notes(lib: Library, paper_ids: Iterable[str], *, replace: bool = False) -> WriteNotesView:
    """The confirmation panel: :func:`seedgraph.work.estimate.estimate` for ``notes``."""
    from ...work import estimate as estimate_mod

    ids = list(dict.fromkeys(str(p) for p in paper_ids if str(p).strip()))
    known = {r[0]: (r[1], r[2] == "held") for r in lib.conn.execute(
        f"SELECT p.paper_id, p.title, s.status FROM papers p JOIN paper_status s ON s.paper_id = p.paper_id "
        f"WHERE p.paper_id IN ({_marks(len(ids))})", ids)} if ids else {}
    unknown = tuple(p for p in ids if p not in known)
    ids = [p for p in ids if p in known]
    est = estimate_mod.estimate(lib, "notes", ids, replace=replace) if ids else None
    rows = [] if est is None else [
        EstimateRow(p.paper_id, known[p.paper_id][0], known[p.paper_id][1], p.state, p.reason, p.text_tokens,
                    p.agent_tokens_low, p.agent_tokens_high) for p in est.papers]
    queued = tuple(r for r in rows if r.state == "queued")
    return WriteNotesView(
        paper_ids=tuple(r.paper_id for r in queued), queued=queued,
        skipped=tuple(r for r in rows if r.state != "queued"),
        text_tokens=est.text_tokens if est else 0, agent_tokens_low=est.agent_tokens_low if est else 0,
        agent_tokens_high=est.agent_tokens_high if est else 0, replace=replace, unknown=unknown)
