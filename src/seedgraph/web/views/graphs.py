"""Context graphs (design §2.3, minimal until stage 4): the list with counts and the
per-graph membership table. Pure reads."""

from __future__ import annotations

from dataclasses import dataclass

from ...library import graphs as graphs_mod
from ...library.store import Library
from . import library as library_view

STATUS_ORDER = {"member": 0, "frontier": 1, "excluded": 2}


@dataclass(frozen=True)
class GraphRow:
    slug: str
    display_name: str
    members: int
    seeds: int
    frontier: int
    excluded: int
    needs_notes: int


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


def list_page(lib: Library) -> list[GraphRow]:
    out = []
    for info in graphs_mod.list_graphs(lib):
        with graphs_mod.open_graph(lib, info.slug) as g:
            rows = graphs_mod.membership(g)
        member_ids = [r.paper_id for r in rows if r.status == "member"]
        states = library_view.note_states(lib, member_ids)
        out.append(GraphRow(
            slug=info.slug, display_name=info.display_name, members=len(member_ids),
            seeds=sum(1 for r in rows if r.is_seed), frontier=sum(1 for r in rows if r.status == "frontier"),
            excluded=sum(1 for r in rows if r.status == "excluded"),
            needs_notes=sum(1 for p in member_ids if library_view.needs_notes(states[p]))))
    return out


def detail(lib: Library, slug: str) -> GraphDetail | None:
    """The graph page, or ``None`` for an unknown slug."""
    if slug not in {i.slug for i in graphs_mod.list_graphs(lib)}:
        return None
    with graphs_mod.open_graph(lib, slug) as g:
        info = graphs_mod.graph_info(g)
        rows = graphs_mod.membership(g)
        lenses = tuple(graphs_mod.chosen_lenses(g))
    facts = {r.paper_id: r for r in library_view.paper_rows(lib, [r.paper_id for r in rows], index={})}
    members = []
    for r in rows:
        f = facts.get(r.paper_id)
        members.append(MemberRow(r.paper_id, f.title if f else None, f.year if f else None,
                                 bool(f and f.held), r.status, r.is_seed, f.note_state if f else "none", r.reason))
    members.sort(key=lambda m: (STATUS_ORDER.get(m.status, 9), not m.is_seed, (m.title or "").lower(), m.paper_id))
    return GraphDetail(slug=info.slug, display_name=info.display_name, created_at=info.created_at,
                       rows=tuple(members), lenses=lenses,
                       members=sum(1 for m in members if m.status == "member"),
                       frontier=sum(1 for m in members if m.status == "frontier"),
                       seeds=sum(1 for m in members if m.is_seed))
