"""Pure server-side transform for the AnswerTrace subgraph pane (trace_plans 00 §6.2).

The trace view renders the citation-graph neighborhood captured on an
:class:`~seedgraph.answer.trace.AnswerTrace` (T7/C4 membership: node ids + edge
pairs) as an explorable **works-only, single-plane** subgraph. This module is the
one pure function :func:`build_trace_subgraph_payload`, following the
``web/graph3d.py:build_3d_payload`` discipline: it performs **no** DB / networkx /
filesystem access — the route's read-model (``web/ui.py:build_trace_context``) owns
the enrichment queries and passes the already-materialized dicts in.

Deliberately NOT the two-plane concept machinery of ``graph3d.py`` (works only, no
concept plane, no z-lock) — only its pattern is reused (C7). Each node carries a
``role ∈ {seed, neighbor, recommendation}`` so a click can answer "why is this work
present?": a retrieval seed, a citation neighbor, or a graph-derived recommendation.
"""

from __future__ import annotations

from typing import Any, Iterable

__all__ = ["build_trace_subgraph_payload"]


def build_trace_subgraph_payload(
    neighborhood: dict,
    *,
    recommendation_work_ids: Iterable[str] = (),
    work_meta: dict[str, dict] | None = None,
) -> dict:
    """Transform a trace neighborhood dict into the subgraph render payload.

    Pure: no DB / networkx / filesystem access. Inputs:

    * ``neighborhood`` — the :class:`~seedgraph.answer.trace.NeighborhoodTrace`
      ``model_dump`` (or the ``traverse.neighborhood`` dict): ``{"seeds", "nodes",
      "edges", "depth", "coverage_ratio", "engulfed", "unknown_seeds"}``. ``nodes``
      is the authoritative membership (work ids); ``edges`` is a list of
      ``[source, target]`` work-id pairs.
    * ``recommendation_work_ids`` — the envelope's recommendation work ids; a node
      in the neighborhood that is also a recommendation is flagged ``recommendation``.
    * ``work_meta`` — ``{work_id: {"title", "year"}}`` from the route's enrichment
      query; a work absent from the map keeps ``title=None`` (the view renders
      "unknown" rather than crashing).

    Roles are assigned by precedence **seed → recommendation → neighbor**. Returns
    ``{"nodes", "links", "counts", "seeds", "depth", "coverage_ratio", "engulfed"}``
    where::

        node = {id, name, title, year, role}   # role: 'seed'|'neighbor'|'recommendation'
        link = {source, target}                # both endpoints in the node set; no self-loops
    """
    work_meta = work_meta or {}
    seeds = list(neighborhood.get("seeds", []) or [])
    seed_set = set(seeds)
    rec_set = {w for w in recommendation_work_ids if w}

    # nodes: the neighborhood membership is authoritative; dedup, preserve order.
    node_ids: list[str] = []
    seen: set[str] = set()
    for nid in neighborhood.get("nodes", []) or []:
        if nid not in seen:
            seen.add(nid)
            node_ids.append(nid)

    nodes: list[dict[str, Any]] = []
    counts = {"seed": 0, "neighbor": 0, "recommendation": 0}
    for nid in node_ids:
        if nid in seed_set:
            role = "seed"
        elif nid in rec_set:
            role = "recommendation"
        else:
            role = "neighbor"
        counts[role] += 1
        meta = work_meta.get(nid) or {}
        title = meta.get("title")
        nodes.append(
            {
                "id": nid,
                "name": title or nid,
                "title": title,
                "year": meta.get("year"),
                "role": role,
            }
        )

    # links: keep only edges whose endpoints are both on the canvas; drop self-loops
    # and exact-duplicate pairs (the membership set is small and deterministic).
    links: list[dict[str, str]] = []
    link_seen: set[tuple[str, str]] = set()
    dangling = 0
    for edge in neighborhood.get("edges", []) or []:
        if not edge or len(edge) < 2:
            continue
        s, t = str(edge[0]), str(edge[1])
        if s == t:
            continue
        if s not in seen or t not in seen:
            dangling += 1
            continue
        key = (s, t)
        if key in link_seen:
            continue
        link_seen.add(key)
        links.append({"source": s, "target": t})

    return {
        "nodes": nodes,
        "links": links,
        "counts": {
            **counts,
            "nodes": len(nodes),
            "links": len(links),
            "dangling_dropped": dangling,
        },
        "seeds": seeds,
        "depth": neighborhood.get("depth"),
        "coverage_ratio": neighborhood.get("coverage_ratio"),
        "engulfed": bool(neighborhood.get("engulfed", False)),
    }
