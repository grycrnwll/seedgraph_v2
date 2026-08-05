"""Pure server-side transform for the 3D two-plane concept/citation graph (Track 3).

Ported from the v1 prototype ``web/graph3d.py`` (camera/plane constants, the
deterministic scatter, the link-strength + concept-label-threshold logic), then
retargeted onto **v2 data**: the input is ``nx.node_link_data(build_graph(conn,
run_id), edges="links")`` rather than the v1 ``graphify`` json. Classification is
by the v2 ``node_type`` attribute (``Work`` / ``Concept`` / ``ConceptAlias``), NOT
the v1 ``concept::`` id-prefix heuristic.

This module is a single pure function — :func:`build_3d_payload`. It performs **no**
DB / networkx / filesystem access (no imports of any of those): the route owns the
one ``build_graph`` call and the supplemental enrichment queries and passes the
already-materialized dicts in. The output is local-session-only internal UI data
(plan review #8): it is computed in-memory, never persisted to ``runs/``, never
exported, and distinct from the public ``graph.json`` route.

Two-plane model:
  * ``Work``  nodes live on the paper   plane (``z = PAPER_Z``).
  * ``Concept`` nodes live on the concept plane (``z = CONCEPT_Z``).
  * ``ConceptAlias`` nodes are **dropped from the canvas** (counted in
    ``counts.aliases_hidden``); any ``has_alias`` edge touching one then drops as
    dangling.

Links are classified purely by **endpoint type** (the relational ``edge_type`` is
carried through as an attribute but does NOT drive the plane classification):
``paper↔paper`` → ``citation``, ``concept↔concept`` → ``concept``, mixed → ``strut``.
Self-loops and dangling links are dropped. ``z`` is locked per plane (the JS layout
holds it); the transform only scatters ``x``/``y`` deterministically via a per-call
``random.Random(seed)``.
"""

from __future__ import annotations

import random
import re
from typing import Any

# planes / scatter (z is locked in JS; the transform only sets the initial frame).
# Ported verbatim from the v1 prototype web/graph3d.py.
PAPER_Z = 0
CONCEPT_Z = 400
SCATTER = 300

# concept-label degree threshold target window (~30-50 labeled hubs).
_TARGET_LO, _TARGET_HI = 30, 50

__all__ = ["build_3d_payload", "PAPER_Z", "CONCEPT_Z", "SCATTER"]


def clean_concept_name(s: Any) -> str:
    """Strip a leading ``concept::`` (the v2 deterministic concept-id prefix) plus
    one optional field-type prefix (``method::`` / ``dataset::`` / …), returning a
    human phrase. Used only as a fallback when a Concept node carries no
    ``canonical_label``."""
    s = str(s)
    while s.lower().startswith("concept::"):
        s = s[len("concept::"):]
    s = re.sub(r"^[a-z_]+::", "", s)
    return s.strip()


def _node_id(n: dict) -> str:
    return str(n.get("id"))


def build_3d_payload(
    graph_nodelink_dict: dict,
    provenance_counts: dict | None = None,
    document_links: dict | None = None,
    *,
    seed: int = 42,
) -> dict:
    """Transform a NetworkX node-link dict into the 3D render payload.

    Pure: no DB / networkx / filesystem access. Inputs:

    * ``graph_nodelink_dict`` — ``nx.node_link_data(build_graph(conn, run_id),
      edges="links")``: ``{"nodes": [...], "links": [...], ...}``. Work nodes carry
      ``node_type='Work'`` (+ ``title`` / ``year``); Concept nodes carry
      ``node_type='Concept'`` (+ ``canonical_label`` / ``concept_type`` /
      ``paper_frequency`` / ``status`` / ``epistemic_type`` / ``access_class``);
      ConceptAlias nodes carry ``node_type='ConceptAlias'`` and are dropped. Each
      link carries ``edge_type`` / ``epistemic_type`` / ``access_class`` (and, when
      the route enriched it, ``confidence``).
    * ``provenance_counts`` — ``{concept_id: {"papers", "claims", "spans"}}`` from
      :func:`semantic.query.concept_provenance_counts`; feeds each Concept's
      ``prov`` field.
    * ``document_links`` — ``{work_id: {"has_pdf", "has_markdown", "access_class",
      "inclusion_status", "seed", ...}}`` composed by the route from
      :func:`acquisition.bridge.resolve_work_source`; feeds each Work's
      ``pdf`` / ``md`` / ``access_class`` / ``inclusion_status`` / ``seed``.

    Returns ``{"nodes", "links", "threshold", "counts", "warnings"}`` where::

        Work node    = {id, name, type:'paper', seed, deg, citation_deg, year,
                        pdf, md, inclusion_status, access_class,
                        community, is_god_node, is_bridge, x, y, z}
        Concept node = {id, name, type:'concept', seed, deg, concept_type,
                        paper_frequency, weight, status, epistemic_type,
                        access_class, prov:{papers,claims,spans}, x, y, z}
        link         = {source, target, type:'citation'|'concept'|'strut',
                        edge_type, epistemic_type, access_class, confidence,
                        is_bridge, shared_count, k}
    """
    # ponytail: per-call local RNG (never module-global random) so concurrent
    # graph3d.json requests can't interleave and corrupt each other's scatter.
    rng = random.Random(seed)

    provenance_counts = provenance_counts or {}
    document_links = document_links or {}

    raw_nodes = graph_nodelink_dict.get("nodes", []) or []
    raw_links = graph_nodelink_dict.get("links", []) or []

    # endpoint resolver: handle string ids, {.id} objects, or int indices
    id_by_index = [_node_id(n) for n in raw_nodes]

    def resolve_endpoint(ep: Any) -> str:
        if isinstance(ep, dict):
            return str(ep.get("id"))
        if isinstance(ep, bool):  # guard: bool is an int subclass
            return str(ep)
        if isinstance(ep, int):
            if 0 <= ep < len(id_by_index):
                return id_by_index[ep]
            return str(ep)
        return str(ep)

    # --- build node records, classify by node_type --------------------------
    type_of: dict[str, str] = {}      # retained ids only -> 'paper' | 'concept'
    nodes_out: dict[str, dict] = {}
    aliases_hidden = 0

    for n in raw_nodes:
        nid = _node_id(n)
        node_type = n.get("node_type")

        if node_type == "ConceptAlias":
            aliases_hidden += 1
            continue
        if node_type not in ("Work", "Concept"):
            # Unknown/None node_type: keep it off the canvas (any edge to it then
            # drops as dangling). Bare citation endpoints always carry node_type
            # 'Work', so this only fires on genuinely malformed input.
            continue

        if node_type == "Concept":
            type_of[nid] = "concept"
            prov = provenance_counts.get(nid) or {}
            nodes_out[nid] = {
                "id": nid,
                "name": n.get("canonical_label") or clean_concept_name(nid),
                "type": "concept",
                "seed": False,
                "deg": 0,
                "concept_type": n.get("concept_type"),
                "paper_frequency": n.get("paper_frequency"),
                # IDF discriminativeness weight (Build B chunk 5) — NOT importance.
                "weight": n.get("weight"),
                "status": n.get("status"),
                "epistemic_type": n.get("epistemic_type"),
                "access_class": n.get("access_class"),
                "prov": {
                    "papers": prov.get("papers", 0),
                    "claims": prov.get("claims", 0),
                    "spans": prov.get("spans", 0),
                },
            }
        else:  # Work
            type_of[nid] = "paper"
            dl = document_links.get(nid) or {}
            nodes_out[nid] = {
                "id": nid,
                # never-blank: the graph_build `label` (Build B chunk 9) wins,
                # then title/name, then the raw id as the last resort.
                "name": n.get("label") or n.get("title") or n.get("name") or nid,
                "type": "paper",
                "seed": bool(dl.get("seed", n.get("is_seed", False))),
                "deg": 0,
                "citation_deg": 0,
                "year": n.get("year"),
                "pdf": bool(dl.get("has_pdf", False)),
                "md": bool(dl.get("has_markdown", False)),
                "inclusion_status": dl.get("inclusion_status"),
                "access_class": dl.get("access_class"),
                # Build B chunk 3: citation-community analysis passthrough.
                "community": n.get("community"),
                "is_god_node": bool(n.get("is_god_node", False)),
                "is_bridge": bool(n.get("is_bridge", False)),
            }

    # --- classify links by ENDPOINT TYPE; drop self-loops + dangling --------
    links_out: list[dict] = []
    self_loops_dropped = 0
    dangling_dropped = 0
    citation_links = concept_links = strut_links = 0

    for l in raw_links:
        s = resolve_endpoint(l.get("source"))
        t = resolve_endpoint(l.get("target"))
        if s == t:
            self_loops_dropped += 1
            continue
        if s not in type_of or t not in type_of:
            dangling_dropped += 1
            continue
        st, tt = type_of[s], type_of[t]
        if st == "paper" and tt == "paper":
            ltype = "citation"
            citation_links += 1
        elif st == "concept" and tt == "concept":
            ltype = "concept"
            concept_links += 1
        else:
            ltype = "strut"
            strut_links += 1

        nodes_out[s]["deg"] += 1
        nodes_out[t]["deg"] += 1
        if ltype == "citation":
            nodes_out[s]["citation_deg"] += 1
            nodes_out[t]["citation_deg"] += 1

        links_out.append(
            {
                "source": s,
                "target": t,
                "type": ltype,
                "edge_type": l.get("edge_type"),
                "epistemic_type": l.get("epistemic_type"),
                "access_class": l.get("access_class"),
                "confidence": l.get("confidence"),
                # Build B chunk 3/7: bridge flag + co-occurrence strength.
                "is_bridge": bool(l.get("is_bridge", False)),
                "shared_count": l.get("shared_count"),
            }
        )

    # per-link force strength, degree-normalized (computed once degrees are final).
    # 1/min(deg) is d3's anti-collapse rule for in-plane edges; struts stay weak.
    for l in links_out:
        if l["type"] == "strut":
            l["k"] = 0.03
        else:
            ds = nodes_out[l["source"]]["deg"]
            dt = nodes_out[l["target"]]["deg"]
            l["k"] = round(1.0 / max(1, min(ds, dt)), 4)

    # deterministic initial scatter (x,y free; z locked per plane). Iterate in
    # insertion order so a fixed seed yields an identical frame across calls.
    final_nodes: list[dict] = []
    for nd in nodes_out.values():
        nd["x"] = rng.uniform(-SCATTER, SCATTER)
        nd["y"] = rng.uniform(-SCATTER, SCATTER)
        nd["z"] = PAPER_Z if nd["type"] == "paper" else CONCEPT_Z
        final_nodes.append(nd)

    threshold = _concept_label_threshold(final_nodes)

    n_works = sum(1 for nd in final_nodes if nd["type"] == "paper")
    n_concepts = sum(1 for nd in final_nodes if nd["type"] == "concept")
    counts = {
        "works": n_works,
        "concepts": n_concepts,
        "aliases_hidden": aliases_hidden,
        "citation_links": citation_links,
        "concept_links": concept_links,
        "strut_links": strut_links,
        "self_loops_dropped": self_loops_dropped,
        "dangling_dropped": dangling_dropped,
    }

    warnings: list[str] = []
    if not final_nodes:
        warnings.append("graph is empty for this run")
    elif n_concepts == 0:
        warnings.append("no concepts in this run — only the citation plane is shown")
    if dangling_dropped:
        warnings.append(f"{dangling_dropped} link(s) dropped (endpoint not on canvas)")

    return {
        "nodes": final_nodes,
        "links": links_out,
        "threshold": threshold,
        "counts": counts,
        "warnings": warnings,
    }


def _concept_label_threshold(final_nodes: list[dict]) -> int:
    """Choose the concept-degree cutoff so only ~30-50 concept hubs get labels.
    Ported from the v1 prototype."""
    concept_degs = sorted(
        (n["deg"] for n in final_nodes if n["type"] == "concept"), reverse=True
    )
    thr = 1
    for cand in range(max(concept_degs, default=1), 0, -1):
        n_at = sum(1 for d in concept_degs if d >= cand)
        if n_at >= _TARGET_LO:
            thr = cand
            if n_at <= _TARGET_HI:
                break
            # too many at this degree; bump one to shrink toward target.
            thr = cand + 1
            break
    return thr
