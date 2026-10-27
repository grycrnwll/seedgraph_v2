"""The 2D graph on a graph page (stage 4 design §8.3): a server-rendered SVG, deterministic
and testable, with presentation attributes and CSS classes only (no ``style`` attribute,
no script: the CSP holds unchanged). Nodes are ``<a>`` links with a ``<title>`` tooltip.

- **Concept view** (default): the top ``TOP_N`` supported concepts by slice weight x papers.
  Node size grows with papers. An edge joins two concepts co-supported in at least
  ``MIN_SHARED`` member papers; closely related links are drawn dashed. Nodes are coloured
  by the literature where they are most supported (counted senses only; grey when none),
  and bridges get a ring.
- **Citation view**: member papers coloured by literature, unplaced ones grey, in-graph
  citations as edges. Above ``MAX_SVG_NODES`` only placed papers are drawn, with a count
  of the rest.

Layout. ``networkx.spring_layout`` needs numpy, which this project does not carry, so the
concept view uses a small pure-Python Fruchterman-Reingold (seeded ``random.Random(0)``,
fixed iterations, nodes in sorted id order, coordinates rounded to 0.1). The citation view
places each literature as a cluster (clusters on a circle, papers on a sunflower spiral in
paper-id order): linear time, and the literatures stay visibly apart.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

from ...concepts import bridges as br
from ...library import graphs as graphs_mod
from ...library import literatures as lits_mod
from ...library.store import Library
from . import library as library_view

TOP_N = 40
MIN_SHARED = 2
MAX_SVG_NODES = 800
WIDTH, HEIGHT, MARGIN = 900, 640, 40
ITERATIONS = 80
PALETTE = ("#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#9c755f",
           "#ff9da7", "#86bcb6")
GREY = "#b0b0b0"
View = Literal["concept", "citation"]


@dataclass(frozen=True)
class SvgNode:
    node_id: str
    x: float
    y: float
    r: float
    fill: str
    css: str  # "sg-lit-<ordinal>" or "sg-lit-none"
    href: str
    title: str
    label: str
    ring: bool = False


@dataclass(frozen=True)
class SvgEdge:
    a: str
    b: str
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    dashed: bool = False


@dataclass(frozen=True)
class SvgGraph:
    view: View
    width: int
    height: int
    nodes: tuple[SvgNode, ...]
    edges: tuple[SvgEdge, ...]
    legend: tuple[tuple[str, str, str], ...]  # (css class, fill, name)
    omitted: int = 0
    note: str = ""


def _lit_style(ordinal: int | None) -> tuple[str, str]:
    if ordinal is None:
        return "sg-lit-none", GREY
    return f"sg-lit-{ordinal}", PALETTE[(ordinal - 1) % len(PALETTE)]


# --- layouts ----------------------------------------------------------------------------------------

def spring_layout(nodes: Sequence[str], edges: Iterable[tuple[str, str, float]], *, seed: int = 0,
                  iterations: int = ITERATIONS) -> dict[str, tuple[float, float]]:
    """Fruchterman-Reingold in the unit square, deterministic for a node set and edge set."""
    order = sorted(set(nodes))
    n = len(order)
    if n == 0:
        return {}
    if n == 1:
        return {order[0]: (0.5, 0.5)}
    rng = random.Random(seed)
    pos = {v: [rng.random(), rng.random()] for v in order}
    weights: dict[tuple[str, str], float] = {}
    for a, b, w in edges:
        if a != b and a in pos and b in pos:
            key = (a, b) if a < b else (b, a)
            weights[key] = weights.get(key, 0.0) + float(w)
    adj = sorted(weights.items())
    k = math.sqrt(1.0 / n)
    t = 0.1
    cool = t / (iterations + 1)
    for _ in range(iterations):
        disp = {v: [0.0, 0.0] for v in order}
        for i, a in enumerate(order):
            ax, ay = pos[a]
            for b in order[i + 1:]:
                dx, dy = ax - pos[b][0], ay - pos[b][1]
                d = max(math.hypot(dx, dy), 1e-4)
                f = k * k / d
                disp[a][0] += dx / d * f
                disp[a][1] += dy / d * f
                disp[b][0] -= dx / d * f
                disp[b][1] -= dy / d * f
        for (a, b), w in adj:
            dx, dy = pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]
            d = max(math.hypot(dx, dy), 1e-4)
            f = d * d / k * min(w, 4.0)
            disp[a][0] -= dx / d * f
            disp[a][1] -= dy / d * f
            disp[b][0] += dx / d * f
            disp[b][1] += dy / d * f
        for v in order:
            dx, dy = disp[v]
            d = max(math.hypot(dx, dy), 1e-9)
            step = min(d, t)
            pos[v][0] += dx / d * step
            pos[v][1] += dy / d * step
        t -= cool
    return {v: (pos[v][0], pos[v][1]) for v in order}


def cluster_layout(groups: Sequence[Sequence[str]]) -> dict[str, tuple[float, float]]:
    """Clusters on a circle (largest first, clockwise from the top), each a sunflower spiral
    of its papers in the given order."""
    groups = [list(g) for g in groups if g]
    out: dict[str, tuple[float, float]] = {}
    if not groups:
        return out
    golden = math.pi * (3 - math.sqrt(5))
    total = sum(len(g) for g in groups)
    spread = 0.0 if len(groups) == 1 else 1.0
    for gi, members in enumerate(groups):
        angle = 2 * math.pi * gi / len(groups) - math.pi / 2
        cx, cy = spread * math.cos(angle), spread * math.sin(angle)
        radius = 0.45 * math.sqrt(len(members) / max(total, 1)) * (1 if len(groups) > 1 else 2)
        for i, v in enumerate(members):
            rr = radius * math.sqrt((i + 0.5) / len(members))
            out[v] = (cx + rr * math.cos(i * golden), cy + rr * math.sin(i * golden))
    return out


def _fit(pos: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """Scale positions into the drawing area, rounded to 0.1 (stable bytes)."""
    if not pos:
        return {}
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    w = (max(xs) - min(xs)) or 1.0
    h = (max(ys) - min(ys)) or 1.0
    sx, sy = (WIDTH - 2 * MARGIN) / w, (HEIGHT - 2 * MARGIN) / h
    return {v: (round(MARGIN + (x - min(xs)) * sx, 1) if len(set(xs)) > 1 else WIDTH / 2,
                round(MARGIN + (y - min(ys)) * sy, 1) if len(set(ys)) > 1 else HEIGHT / 2)
            for v, (x, y) in pos.items()}


def _edges(pos: dict[str, tuple[float, float]], pairs: Iterable[tuple[str, str, float, bool]]) -> tuple[SvgEdge, ...]:
    out = []
    for a, b, w, dashed in sorted(pairs, key=lambda t: (t[0], t[1], t[3])):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        out.append(SvgEdge(a, b, x1, y1, x2, y2, round(min(1.0 + 0.5 * (w - 1), 4.0), 1), dashed))
    return tuple(out)


# --- concept view -----------------------------------------------------------------------------------

def concept_graph(lib: Library, slug: str, *, top_n: int = TOP_N) -> SvgGraph:
    from .graphs import concept_href

    s = br.load_slice(lib, slug, agent_only=False)
    ranked = sorted(br.slice_concepts(s), key=lambda c: (-c.score, -c.papers, c.name, c.concept_id))[:top_n]
    papers = {c.concept_id: set(c.paper_ids) for c in ranked}
    pairs: dict[tuple[str, str], tuple[float, bool]] = {}
    for a, b in combinations(sorted(papers), 2):
        shared = len(papers[a] & papers[b])
        if shared >= MIN_SHARED:
            pairs[(a, b)] = (float(shared), False)
    for a, b in br.linked_pairs(lib.conn, papers):
        pairs.setdefault((a, b), (1.0, True))
        pairs[(a, b)] = (pairs[(a, b)][0], True)
    raw = spring_layout(list(papers), [(a, b, w) for (a, b), (w, _) in pairs.items()])
    pos = _fit(raw)
    ordinal = {lit.literature_id: lit.ordinal for lit in s.literatures}
    names = {lit.ordinal: lit.name for lit in s.literatures}
    nodes = []
    used: set[int | None] = set()
    for c in sorted(ranked, key=lambda c: c.concept_id):
        o = ordinal.get(c.main_literature) if c.main_literature else None
        used.add(o)
        css, fill = _lit_style(o)
        x, y = pos[c.concept_id]
        nodes.append(SvgNode(c.concept_id, x, y, round(4 + 3 * math.sqrt(c.papers), 1), fill, css,
                             concept_href(c.handle),
                             f"{c.name} ({c.family}): {c.papers} paper{'s' if c.papers != 1 else ''}"
                             + (", a bridge" if c.is_bridge else ""), c.name, c.is_bridge))
    legend = tuple((*_lit_style(o), names[o]) for o in sorted(o for o in used if o is not None))
    if None in used:
        legend += (("sg-lit-none", GREY, "no single literature"),)
    note = "" if ranked else "No concepts in this graph yet."
    return SvgGraph("concept", WIDTH, HEIGHT, tuple(nodes),
                    _edges(pos, [(a, b, w, d) for (a, b), (w, d) in pairs.items()]), legend, 0, note)


# --- citation view ----------------------------------------------------------------------------------

def citation_graph(lib: Library, slug: str, *, max_nodes: int | None = None) -> SvgGraph:
    max_nodes = MAX_SVG_NODES if max_nodes is None else max_nodes
    with graphs_mod.open_graph(lib, slug) as g:
        members = [r.paper_id for r in graphs_mod.membership(g, status="member")]
        lits = lits_mod.literatures(g)
        cites = graphs_mod.graph_citations(g)
    placed = {p: lit for lit in lits for p in lit.members}
    unplaced = sorted(p for p in members if p not in placed)
    omitted = 0
    drawn_unplaced = unplaced
    if len(members) > max_nodes:
        omitted, drawn_unplaced = len(unplaced), []
    groups = [[p for p in lit.members if p in set(members)] for lit in sorted(lits, key=lambda x: x.ordinal)]
    groups.append(drawn_unplaced)
    pos = _fit(cluster_layout(groups))
    facts = {r.paper_id: r for r in library_view.paper_rows(lib, list(pos), index={})}
    nodes = []
    for pid in sorted(pos):
        lit = placed.get(pid)
        css, fill = _lit_style(lit.ordinal if lit else None)
        f = facts.get(pid)
        title = (f.title if f and f.title else pid) + (f" ({f.year})" if f and f.year else "")
        title += f" - {lit.name}" if lit else " - unplaced"
        x, y = pos[pid]
        nodes.append(SvgNode(pid, x, y, 5.0, fill, css, f"/papers/{pid}", title, ""))
    seen: dict[tuple[str, str], float] = defaultdict(float)
    for c in cites:
        a, b = sorted((c.source_paper_id, c.target_paper_id))
        if a != b and a in pos and b in pos:
            seen[(a, b)] = 1.0
    legend = tuple((*_lit_style(lit.ordinal), lit.name) for lit in sorted(lits, key=lambda x: x.ordinal))
    if unplaced:
        legend += (("sg-lit-none", GREY, "unplaced"),)
    note = "" if members else "No members yet."
    return SvgGraph("citation", WIDTH, HEIGHT, tuple(nodes), _edges(pos, [(a, b, w, False) for (a, b), w in seen.items()]),
                    legend, omitted, note)


def graph(lib: Library, slug: str, view: str) -> SvgGraph:
    return citation_graph(lib, slug) if view == "citation" else concept_graph(lib, slug)
