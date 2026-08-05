"""Citation-graph traversal + on-demand co-citation gap-finding (plan §5 / §6.5).

Wraps the phase-7 NetworkX project citation graph (``semantic.graph_build`` over
``citation_edges``) for lineage/influence lookups (08 §5 step 7), and computes
``gap_finding`` recommendations on demand over **existing rows only** — no provider
fan-out, no inbound-citation expansion, no vivification (decision 74 / D3). The
graph is passed in by the caller; NetworkX is not imported at module scope.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .types import Recommendation


def cites_of(graph: Any, work_id: str) -> list[str]:
    """Outbound: work_ids that ``work_id`` cites (08 §5 step 7 / ``citation_search``).

    ``graph`` is the phase-7 NetworkX ``DiGraph``; returns the out-neighbour work_ids
    reached over a citation edge (``is_citation``), in deterministic (sorted) order.
    Unknown ``work_id`` → ``[]``.
    """
    if graph is None or work_id not in graph:
        return []
    out = {
        v
        for _u, v, d in graph.out_edges(work_id, data=True)
        if d.get("is_citation")
    }
    return sorted(out)


def cited_by(graph: Any, work_id: str) -> list[str]:
    """Inbound: work_ids that cite ``work_id`` (08 §5 step 7 / ``citation_search``).

    In-neighbour work_ids over the phase-7 graph (deterministic sorted order).
    Unknown ``work_id`` → ``[]``.
    """
    if graph is None or work_id not in graph:
        return []
    out = {
        u
        for u, _v, d in graph.in_edges(work_id, data=True)
        if d.get("is_citation")
    }
    return sorted(out)


def neighborhood(graph: Any, seeds: "list[str]", depth: int = 1) -> dict:
    """Build B chunk 8 — depth-bounded UNDIRECTED citation neighborhood.

    Union ego-BFS over the citation projection treated undirected (both in- and
    out-citations), bounded to ``depth`` hops, unioned across every seed. Nodes
    come back in deterministic DISCOVERY order (seeds first, then per-level,
    per-seed ``cited_by + cites_of`` order) — at ``depth=1`` this reproduces the
    pre-Build-B one-hop union exactly. Unknown seeds contribute nothing (never
    invented) but are reported under ``unknown_seeds`` so a typo'd id surfaces
    loudly instead of silently shrinking the result.

    Returns a self-contained subgraph dict::

        {nodes, edges, seeds, unknown_seeds, depth, coverage_ratio, engulfed}

    where ``edges`` is the induced citation-edge list (both endpoints reached),
    ``coverage_ratio = |reached| / |citation-projection nodes|`` and
    ``engulfed = coverage_ratio >= 0.8`` — the guard informs (warn, never
    truncate); the human decides.
    """
    from ..graph.analyze import citation_projection

    seed_list = list(dict.fromkeys(seeds or []))
    empty = {
        "nodes": [],
        "edges": [],
        "seeds": [],
        "unknown_seeds": seed_list,
        "depth": depth,
        "coverage_ratio": 0.0,
        "engulfed": False,
    }
    if graph is None:
        return empty
    proj = citation_projection(graph)
    present = [s for s in seed_list if s in proj]
    unknown = [s for s in seed_list if s not in proj]

    reached: list[str] = list(present)
    seen: set[str] = set(present)
    frontier: list[str] = list(present)
    for _ in range(max(0, depth)):
        nxt: list[str] = []
        for u in frontier:
            for v in cited_by(graph, u) + cites_of(graph, u):
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
                    reached.append(v)
        if not nxt:
            break
        frontier = nxt

    edges = sorted((u, v) for u, v in proj.edges() if u in seen and v in seen)
    total = proj.number_of_nodes()
    coverage = (len(seen) / total) if total else 0.0
    return {
        "nodes": reached,
        "edges": [list(e) for e in edges],
        "seeds": present,
        "unknown_seeds": unknown,
        "depth": depth,
        "coverage_ratio": coverage,
        "engulfed": coverage >= 0.8,
    }


def bridges_in(graph: Any) -> dict:
    """Build B chunk 8 — query-time bridge RE-DERIVATION (flagged vs derived).

    Re-derives cross-community bridges from the ``community`` node labels on any
    annotated (sub)graph — pass ``graph.subgraph(nbhd["nodes"])`` to scope it to
    a :func:`neighborhood`. Each derived bridge edge carries
    ``flagged_is_bridge`` (was it ALSO stamped by
    ``annotate_citation_communities`` at build time?) so a bridge claim is
    auditable: derived-but-not-flagged means the flag is stale or the scope
    differs. Nodes without a community label (un-annotated, or Concept nodes)
    never form bridges.
    """
    if graph is None:
        return {"bridge_edges": [], "bridge_nodes": [], "flagged_bridge_nodes": []}
    comm = {n: d.get("community") for n, d in graph.nodes(data=True)}
    bridge_edges: list[dict] = []
    bridge_nodes: set[str] = set()
    for u, v, d in graph.edges(data=True):
        if not d.get("is_citation", "is_citation" not in d):
            continue
        cu, cv = comm.get(u), comm.get(v)
        if cu is None or cv is None:
            continue
        if cu != cv:
            bridge_edges.append(
                {
                    "source_work_id": u,
                    "target_work_id": v,
                    "source_community": cu,
                    "target_community": cv,
                    "flagged_is_bridge": bool(d.get("is_bridge", False)),
                }
            )
            bridge_nodes.add(u)
            bridge_nodes.add(v)
    bridge_edges.sort(key=lambda e: (e["source_work_id"], e["target_work_id"]))
    flagged = sorted(
        n for n, d in graph.nodes(data=True)
        if d.get("is_bridge") and d.get("node_type", "Work") == "Work"
    )
    return {
        "bridge_edges": bridge_edges,
        "bridge_nodes": sorted(bridge_nodes),
        "flagged_bridge_nodes": flagged,
    }


def read_next_candidates(
    conn: sqlite3.Connection,
    graph: Any,
    top: int,
) -> list[Recommendation]:
    """Build B chunk 4 — read-next recommendations from the citation-community
    analysis (decision 38's deterministic floor surfaced at answer time).

    Annotates ``graph`` via :func:`graph.analyze.annotate_citation_communities`
    (unread map derived from one SQL pass) and emits the ranked
    ``analysis["read_next"]`` works as :class:`Recommendation`s — never
    citations. Excluded works are filtered (defensively; the unread map already
    marks them read). The ``action_hint`` distinguishes a work with no PDF at
    all ("upload") from one whose PDF is already bridged via
    ``work_source_files`` but unextracted ("run extraction"). Returns up to
    ``top`` rows; ``[]`` on a missing graph.
    """
    if graph is None:
        return []
    from ..graph.analyze import annotate_citation_communities
    from ..semantic.graph_build import unread_work_map

    annotate_citation_communities(graph, unread=unread_work_map(conn))
    ranked = graph.graph.get("analysis", {}).get("read_next", [])

    out: list[Recommendation] = []
    for rank, work_id in enumerate(ranked, start=1):
        row = conn.execute(
            "SELECT w.canonical_title, w.year, d.inclusion_status, "
            "EXISTS(SELECT 1 FROM work_source_files f WHERE f.work_id = w.work_id) "
            "FROM works w LEFT JOIN project_documents d ON d.work_id = w.work_id "
            "WHERE w.work_id = ?",
            (work_id,),
        ).fetchone()
        title, year, status, bridged = row if row is not None else (None, None, None, 0)
        if status == "excluded":
            continue
        out.append(
            Recommendation(
                work_id=work_id,
                title=title,
                year=year,
                reason=(
                    f"high-centrality unread: ranks {rank} by citation-graph "
                    "centrality"
                ),
                status="metadata_only" if status == "metadata_only" else "unavailable",
                # ponytail: bridge-row existence == "PDF present"; one hint
                # covers both post-upload steps (convert-pending vs
                # extract-pending are not distinguished).
                action_hint=(
                    "run extraction on the uploaded PDF to extract evidence "
                    "and cite this work"
                    if bridged
                    else "upload the PDF to extract evidence and cite this work"
                ),
            )
        )
        if len(out) >= top:
            break
    return out


def cocitation_candidates(
    conn: sqlite3.Connection,
    project_slug: str,
    top: int,
) -> list[Recommendation]:
    """08 §11 — on-demand co-citation gap-finding over existing rows only.

    Counts, per target work, how many distinct project works cite it
    (``citation_edges`` in-degree), and surfaces the most-cited works that are
    ``metadata_only`` / span-less as :class:`Recommendation`s (never citations). No
    provider calls, no vivification (decision 74/D3). Returns up to ``top`` rows.
    """
    try:
        rows = conn.execute(
            "SELECT e.target_work_id, COUNT(DISTINCT e.source_work_id) AS cocite, "
            "       w.canonical_title, w.year, "
            "       (SELECT COUNT(*) FROM evidence_spans s WHERE s.work_id = e.target_work_id) AS span_count, "
            "       d.inclusion_status "
            "FROM citation_edges e "
            "JOIN works w ON w.work_id = e.target_work_id "
            "LEFT JOIN project_documents d ON d.work_id = e.target_work_id "
            "GROUP BY e.target_work_id "
            "ORDER BY cocite DESC, e.target_work_id"
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    out: list[Recommendation] = []
    for (target, cocite, title, year, span_count, status) in rows:
        if span_count and span_count > 0:
            continue  # has spans -> citable, not a gap recommendation
        if status == "excluded":
            continue
        out.append(
            Recommendation(
                work_id=target,
                title=title,
                year=year,
                reason=f"co-cited by {cocite} project work(s); no extracted full text",
                status="metadata_only" if status == "metadata_only" else "unavailable",
                action_hint="upload the PDF to extract evidence and cite this work",
            )
        )
        if len(out) >= top:
            break
    return out
