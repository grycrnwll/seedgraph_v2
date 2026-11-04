"""Context graph pages (UI design §2.3; stage 4 design §6, §7, §8): the list, and the graph
page's tabs - Overview (counts, papers, literatures, bridges), Concepts (the graph slice and
outside-graph suggestions), Graph (the 2D SVG, :mod:`.graph_svg`), Grow (the guided growth
wizard) and Lenses. Pure reads: nothing here writes or refreshes (a GET never writes);
when the derived concept tables are behind, the Concepts tab shows the C7 banner.

Rules chosen where the design is silent:
- The researcher's view counts every supporting claim (``agent_only=False``); the agent
  audience rule applies to the MCP tools only.
- **Outside-graph suggestions** come only from *confirmed* senses (a sense check or a
  split), of the concept itself or of a concept it is closely related to (a link). An
  unchecked or ambiguous label suggests nothing: the same words elsewhere may mean
  something else. Papers already in the graph at any status (excluded included) are never
  suggested. A held paper would join as a member, a cited-only one as frontier.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ...concepts import bridges as br
from ...concepts.families import CONCEPT_FAMILIES
from ...library import graphs as graphs_mod
from ...library import literatures as lits_mod
from ...library.store import Library, is_held
from . import concepts as concepts_view
from . import library as library_view

STATUS_ORDER = {"member": 0, "frontier": 1, "excluded": 2}
TABS = (("", "Overview"), ("concepts", "Concepts"), ("graph", "Graph"), ("grow", "Grow"), ("lenses", "Lenses"),
        ("records", "Records"))  # the Records tab (stage 5 §6.3) is served by routes/records.py
TOP_PER_FAMILY = 15
SUGGESTIONS_SHOWN = 10


@dataclass(frozen=True)
class GraphRow:
    slug: str
    display_name: str
    members: int
    seeds: int
    frontier: int
    excluded: int
    needs_notes: int
    literatures: int = 0
    bridges: int = 0


@dataclass(frozen=True)
class MemberRow:
    paper_id: str
    title: str | None
    year: int | None
    held: bool
    status: str
    is_seed: bool
    note_state: str
    reason: str | None
    literature: str | None = None


@dataclass(frozen=True)
class GraphDetail:
    slug: str
    display_name: str
    created_at: str | None
    rows: tuple[MemberRow, ...]
    lenses: tuple[str, ...]
    members: int
    frontier: int
    seeds: int


def exists(lib: Library, slug: str) -> bool:
    return slug in {i.slug for i in graphs_mod.list_graphs(lib)}


def list_page(lib: Library) -> list[GraphRow]:
    out = []
    for info in graphs_mod.list_graphs(lib):
        with graphs_mod.open_graph(lib, info.slug) as g:
            rows = graphs_mod.membership(g)
            n_lits = len(lits_mod.literatures(g))
        member_ids = [r.paper_id for r in rows if r.status == "member"]
        states = library_view.note_states(lib, member_ids)
        n_bridges = len(br.bridges(lib.conn, br.load_slice(lib, info.slug, agent_only=False))[0]) if n_lits else 0
        out.append(GraphRow(
            slug=info.slug, display_name=info.display_name, members=len(member_ids),
            seeds=sum(1 for r in rows if r.is_seed), frontier=sum(1 for r in rows if r.status == "frontier"),
            excluded=sum(1 for r in rows if r.status == "excluded"),
            needs_notes=sum(1 for p in member_ids if library_view.needs_notes(states[p])),
            literatures=n_lits, bridges=n_bridges))
    return out


def detail(lib: Library, slug: str) -> GraphDetail | None:
    """The graph's header and papers table, or ``None`` for an unknown slug."""
    if not exists(lib, slug):
        return None
    with graphs_mod.open_graph(lib, slug) as g:
        info = graphs_mod.graph_info(g)
        rows = graphs_mod.membership(g)
        lenses = tuple(graphs_mod.chosen_lenses(g))
        placed = lits_mod.paper_literatures(g)
    facts = {r.paper_id: r for r in library_view.paper_rows(lib, [r.paper_id for r in rows], index={})}
    members = []
    for r in rows:
        f = facts.get(r.paper_id)
        lit = placed.get(r.paper_id)
        members.append(MemberRow(r.paper_id, f.title if f else None, f.year if f else None,
                                 bool(f and f.held), r.status, r.is_seed, f.note_state if f else "none", r.reason,
                                 lit.name if lit else None))
    members.sort(key=lambda m: (STATUS_ORDER.get(m.status, 9), not m.is_seed, (m.title or "").lower(), m.paper_id))
    return GraphDetail(slug=info.slug, display_name=info.display_name, created_at=info.created_at,
                       rows=tuple(members), lenses=lenses,
                       members=sum(1 for m in members if m.status == "member"),
                       frontier=sum(1 for m in members if m.status == "frontier"),
                       seeds=sum(1 for m in members if m.is_seed))


# --- Overview: literatures and bridges --------------------------------------------------------------

@dataclass(frozen=True)
class LiteratureCard:
    literature_id: str
    ordinal: int
    name: str
    name_source: str | None  # "researcher" | "agent inferred" | None (unnamed)
    description: str | None
    size: int
    top_concepts: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Overview:
    detail: GraphDetail
    grouped: bool
    stale: bool
    resolution: float | None
    chosen_by: str | None
    rule_met: bool
    ari_mean: float | None
    unstable: bool
    can_fewer: bool
    can_more: bool
    literatures: tuple[LiteratureCard, ...]
    unplaced: int
    bridges: tuple[dict[str, Any], ...]
    unchecked: tuple[dict[str, Any], ...]


def concept_href(handle: str) -> str:
    """``family/label[~sense]`` -> the C7 concept page URL."""
    from ...concepts.model import parse_handle

    family, key, sense = parse_handle(handle)
    return concepts_view.concept_url(family, key, sense)


def overview(lib: Library, slug: str) -> Overview | None:
    d = detail(lib, slug)
    if d is None:
        return None
    with graphs_mod.open_graph(lib, slug) as g:
        run = lits_mod.latest_run(g)
        stale = lits_mod.is_stale(g)
        lits = lits_mod.literatures(g)
        unplaced = len(lits_mod.unplaced(g)) if run else 0
    cards: list[LiteratureCard] = []
    bridges: list[dict[str, Any]] = []
    unchecked: list[dict[str, Any]] = []
    if run is not None:
        s = br.load_slice(lib, slug, agent_only=False)
        bridges, unchecked = br.bridges(lib.conn, s)
        for lit in lits:
            source = {"researcher": "researcher", "agent": "agent inferred"}.get(lit.name_source or "")
            cards.append(LiteratureCard(lit.literature_id, lit.ordinal, lit.name, source,
                                        None if lit.researcher_name else lit.agent_description, lit.size,
                                        tuple(br.top_concepts(s, lit))))
    ladder = lits_mod.LADDER
    return Overview(
        detail=d, grouped=run is not None, stale=bool(run is not None and stale),
        resolution=run.resolution if run else None, chosen_by=run.chosen_by if run else None,
        rule_met=bool(run and run.rule_met), ari_mean=run.ari_mean if run else None,
        unstable=bool(run and run.unstable),
        can_fewer=bool(run and any(r < run.resolution for r in ladder)),
        can_more=bool(run and any(r > run.resolution for r in ladder)),
        literatures=tuple(cards), unplaced=unplaced, bridges=tuple(bridges), unchecked=tuple(unchecked))


# --- Concepts: the slice and outside-graph suggestions ------------------------------------------------

@dataclass(frozen=True)
class OutsidePaper:
    paper_id: str
    title: str | None
    year: int | None
    held: bool
    graphs: tuple[str, ...]
    via: str  # the concept name the paper shares (this concept, or a linked one)

    @property
    def joins_as(self) -> str:
        return "member" if self.held else "frontier"


@dataclass(frozen=True)
class ConceptRow:
    concept: br.SliceConcept
    href: str
    suggestions: tuple[OutsidePaper, ...] = ()


@dataclass(frozen=True)
class ConceptsTab:
    detail: GraphDetail
    staleness: Any
    families: tuple[tuple[str, tuple[ConceptRow, ...]], ...]
    total: int
    suggestion_papers: int


def outside_suggestions(lib: Library, slug: str, s: br.Slice | None = None) -> dict[str, tuple[OutsidePaper, ...]]:
    """``{concept_id: papers}``: library papers outside the graph sharing a confirmed sense of
    the concept or of a closely related concept (module doc). Ranked held first, then title."""
    s = s or br.load_slice(lib, slug, agent_only=False)
    with graphs_mod.open_graph(lib, slug) as g:
        in_graph = {r.paper_id for r in graphs_mod.membership(g)}
    library = br.Support(lib.conn, agent_only=False)
    confirmed_papers: dict[str, set[str]] = {}
    for cid, concept in s.concepts.items():
        keys = br.counted_senses(s, concept, confirmed_only=True)
        papers = set().union(*(library.papers(library.by_sense.get(k, [])) for k in keys)) if keys else set()
        if papers - in_graph:
            confirmed_papers[cid] = papers - in_graph
    linked: dict[str, set[str]] = defaultdict(set)
    for a, b in br.linked_pairs(lib.conn, s.concepts):
        linked[a].add(b)
        linked[b].add(a)
    shared: dict[str, dict[str, str]] = {}
    for cid in s.support.by_concept:
        if cid not in s.concepts:
            continue
        found: dict[str, str] = {}
        for pid in sorted(confirmed_papers.get(cid, ())):
            found[pid] = s.concepts[cid].name
        for other in sorted(linked.get(cid, ())):
            for pid in sorted(confirmed_papers.get(other, ())):
                found.setdefault(pid, s.concepts[other].name)
        if found:
            shared[cid] = found
    ids = sorted({pid for f in shared.values() for pid in f})
    facts = {r.paper_id: r for r in library_view.paper_rows(lib, ids)}
    out: dict[str, tuple[OutsidePaper, ...]] = {}
    for cid, found in shared.items():
        rows = [OutsidePaper(pid, facts[pid].title if pid in facts else None,
                             facts[pid].year if pid in facts else None, is_held(lib.conn, pid),
                             facts[pid].graphs if pid in facts else (), via) for pid, via in found.items()]
        rows.sort(key=lambda r: (not r.held, (r.title or "").lower(), r.paper_id))
        out[cid] = tuple(rows)
    return out


def concepts_tab(lib: Library, slug: str, *, per_family: int = TOP_PER_FAMILY) -> ConceptsTab | None:
    d = detail(lib, slug)
    if d is None:
        return None
    s = br.load_slice(lib, slug, agent_only=False)
    rows = br.slice_concepts(s)
    suggestions = outside_suggestions(lib, slug, s)
    by_family: dict[str, list[ConceptRow]] = defaultdict(list)
    for c in rows:
        by_family[c.family].append(ConceptRow(c, concept_href(c.handle), suggestions.get(c.concept_id, ())))
    families = tuple((f, tuple(by_family[f][:per_family])) for f in CONCEPT_FAMILIES if by_family.get(f))
    return ConceptsTab(d, concepts_view.staleness(lib), families, len(rows),
                       len({p.paper_id for ps in suggestions.values() for p in ps}))


@dataclass(frozen=True)
class SuggestionConfirm:
    detail: GraphDetail
    paper: OutsidePaper
    concept_name: str


def suggestion_confirm(lib: Library, slug: str, paper_id: str) -> SuggestionConfirm | None:
    """The confirmation step before adding an outside-graph suggestion, or ``None`` when the
    paper is not a current suggestion."""
    d = detail(lib, slug)
    if d is None:
        return None
    s = br.load_slice(lib, slug, agent_only=False)
    for cid, papers in outside_suggestions(lib, slug, s).items():
        for paper in papers:
            if paper.paper_id == paper_id:
                return SuggestionConfirm(d, paper, s.concepts[cid].name)
    return None


# --- Grow: the guided growth wizard -------------------------------------------------------------------

GROWTH_JOB = "growth"
CHOICES = ("member", "frontier", "skip")


@dataclass(frozen=True)
class SeedOption:
    paper_id: str
    title: str | None
    year: int | None
    checked: bool


@dataclass(frozen=True)
class ReviewRow:
    paper_id: str
    title: str | None
    year: int | None
    held: bool
    oa: str  # yes | no | unknown | held
    cited_by_members: int
    reasons: tuple[str, ...]
    doi_url: str | None
    source: str  # "walk" | "concepts"

    @property
    def choices(self) -> tuple[str, ...]:
        return ("member", "frontier") if self.held else ("frontier",)


@dataclass(frozen=True)
class GrowTab:
    detail: GraphDetail
    seeds: tuple[SeedOption, ...]
    depth: int
    cap: int
    max_depth: int
    estimate: dict | None
    found: tuple[ReviewRow, ...]
    suggestions: tuple[ReviewRow, ...]
    needs_pdf: tuple[Any, ...]
    jobs: tuple[Any, ...]


def growth_jobs(jobs: Any, slug: str) -> tuple[Any, ...]:
    """This graph's growth jobs (walk, approval), newest first."""
    return tuple(reversed([j for j in jobs if j.kind == GROWTH_JOB and j.label.startswith(f"{slug}: ")]))


def _review_rows(lib: Library, slug: str) -> tuple[tuple[ReviewRow, ...], tuple[ReviewRow, ...]]:
    from ...library import growth

    found = tuple(ReviewRow(f.paper_id, f.title, f.year, f.held, "held" if f.held else f.oa, f.cited_by_members,
                            f.reasons, f.doi_url, "walk") for f in growth.review(lib, slug))
    seen = {r.paper_id for r in found}
    outside: dict[str, ReviewRow] = {}
    for papers in outside_suggestions(lib, slug).values():
        for p in papers:
            if p.paper_id in seen or p.paper_id in outside or not p.held:
                continue
            outside[p.paper_id] = ReviewRow(p.paper_id, p.title, p.year, True, "held", 0,
                                            (f"shares {p.via} (a checked sense)",), None, "concepts")
    return found, tuple(sorted(outside.values(), key=lambda r: ((r.title or "").lower(), r.paper_id)))


def grow_tab(lib: Library, slug: str, *, seeds: list[str] | None = None, depth: int = 1, cap: int = 50,
             estimate: bool = False, jobs: Any = ()) -> GrowTab | None:
    from ...library import growth, ingest

    d = detail(lib, slug)
    if d is None:
        return None
    chosen = set(seeds) if seeds else set(growth.default_seeds(lib, slug))
    options = tuple(SeedOption(m.paper_id, m.title, m.year, m.paper_id in chosen)
                    for m in d.rows if m.status == "member")
    found, suggestions = _review_rows(lib, slug)
    est = growth.estimate(sum(1 for o in options if o.checked), depth, cap) if estimate else None
    return GrowTab(d, options, depth, cap, growth.MAX_DEPTH, est, found, suggestions,
                   tuple(ingest.needs_pdf(lib, graph=slug)), growth_jobs(jobs, slug))


def check_choices(lib: Library, slug: str, choices: dict[str, str]) -> list[str]:
    """Why an approval would be refused (empty when it may go ahead): only rows on the review
    list may be approved, and a cited-only paper joins the frontier, never as a member."""
    found, suggestions = _review_rows(lib, slug)
    rows = {r.paper_id: r for r in (*found, *suggestions)}
    errors = []
    for pid, choice in choices.items():
        row = rows.get(pid)
        if row is None:
            errors.append(f"{pid} is not on the review list")
        elif choice not in row.choices:
            errors.append(f"{pid} is cited-only: it can join the frontier, not as a {choice}")
    return errors


# --- Lenses -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LensChoice:
    lens_id: str
    name: str
    status: str
    chosen: bool
    in_library: bool  # False: chosen for this graph, but no definition is stored in the library
    missing: int  # members with no successful run of this lens

    @property
    def command(self) -> str:
        return f"seedgraph work create lens:{self.lens_id}"


@dataclass(frozen=True)
class LensesTab:
    detail: GraphDetail
    lenses: tuple[LensChoice, ...]


def lenses_tab(lib: Library, slug: str) -> LensesTab | None:
    d = detail(lib, slug)
    if d is None:
        return None
    defs: dict[str, tuple[str, str]] = {}
    for lens_id, name, status in lib.conn.execute(
            "SELECT lens_id, name, status FROM lens_definitions ORDER BY lens_id, created_at, definition_hash"):
        defs[lens_id] = (name, status)  # the latest version wins
    members = [m.paper_id for m in d.rows if m.status == "member"]
    out = []
    for lens_id in sorted(set(defs) | set(d.lenses)):
        done = {r[0] for r in lib.conn.execute(
            "SELECT DISTINCT paper_id FROM extraction_runs WHERE lens_id = ? AND run_status = 'success'", (lens_id,))}
        name, status = defs.get(lens_id, (lens_id, "unknown"))
        out.append(LensChoice(lens_id, name, status, lens_id in d.lenses, lens_id in defs,
                              sum(1 for p in members if p not in done)))
    return LensesTab(d, tuple(out))
