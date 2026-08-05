"""Pure NetworkX graph analysis: communities, centrality, god-nodes, bridges,
read-next (Build B chunk 1; ported nearly verbatim from v1 ``graph/analyze.py``).

Pure, in-memory, **no I/O, no network, no SQL**: callers pass the computed
NetworkX view (phase-2 ``graph.build.build_citation_graph`` or phase-7
``semantic.graph_build.build_graph``) plus an injected ``unread`` map (derived
from SQL by the caller — this module never opens a database).

The one deliberate exception is :func:`analyze_summary` (decision M10): the deep
service seam that the ``graph analyze`` CLI and the MCP ``graph_analyze`` tool
both render. It DOES open (and close) a project connection to build the run's
graph before delegating to the pure primitives above — one source of truth so
the two adapters cannot drift.

**Clustering scope (the v1 contract, restated verbatim so a future implementer
cannot naively cluster the mixed graph):** clustering runs on the CITATION
subgraph only (paper nodes + ``cites``), so the ``community`` field stays a pure
citation-community label. The FULL canonical graph (papers + concepts + struts)
is then annotated from the citation-only map, so concept / strut nodes correctly
get ``community = None``. Dense concept struts would otherwise drag papers into
fake communities — the prototype scar. Use :func:`annotate_citation_communities`
on any mixed graph; :func:`analyze_graph` itself assumes a citation-only input.

Everything degrades gracefully on tiny/empty graphs: a 1-node, 0-edge graph (or
the empty graph) must not crash — centralities collapse to ``0.0``, the single
node is its own community, nothing is a bridge.

Determinism: Louvain is seeded (``LOUVAIN_SEED``) AND run over a node set sorted
by node id (we relabel to a deterministic integer order first), so the partition
is stable across runs for the same graph — tests can assert on it.

All outputs are ``epistemic_type='deterministic'`` (decision 38's no-LLM floor).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import networkx as nx

if TYPE_CHECKING:
    import sqlite3

    from ..project.service import ProjectHandle

__all__ = [
    "LOUVAIN_SEED",
    "DEFAULT_TOP_K_GODS",
    "DEGREE_WEIGHT",
    "analyze_graph",
    "annotate_graph",
    "annotate_citation_communities",
    "analyze_summary",
    "citation_projection",
    "louvain_communities",
    "centrality_scores",
    "god_nodes",
    "bridges",
    "read_next",
]

#: Fixed seed so the Louvain partition is deterministic across runs (tests
#: depend on a stable community assignment).
LOUVAIN_SEED = 20240101

#: Default number of top-centrality nodes flagged as "god-nodes".
DEFAULT_TOP_K_GODS = 5

#: Blend weight for degree vs betweenness in the god-node / read-next ranking.
#: god_score = DEGREE_WEIGHT*degree + (1-DEGREE_WEIGHT)*betweenness.
DEGREE_WEIGHT = 0.5


# ---------------------------------------------------------------------------
# Communities (Louvain)
# ---------------------------------------------------------------------------
def louvain_communities(graph: nx.Graph, *, seed: int = LOUVAIN_SEED) -> dict[str, int]:
    """Return ``{node: community_id}`` via Louvain (deterministic).

    Louvain operates on an UNDIRECTED graph; a DiGraph is collapsed to its
    undirected form first (citation direction does not change membership). The
    partition is made deterministic by (a) sorting nodes and relabeling to a
    fixed integer order before partitioning and (b) seeding the algorithm.

    Empty graph → ``{}``. Single node / no edges → every node is its own
    community (id ``0, 1, …`` in sorted order). Never raises on tiny graphs.
    """

    nodes = sorted(graph.nodes())
    if not nodes:
        return {}

    undirected = nx.Graph()
    # Add nodes in deterministic order so isolated nodes get stable ids.
    undirected.add_nodes_from(nodes)
    for u, v in graph.edges():
        if u == v:
            continue  # self-loops are irrelevant to community structure
        undirected.add_edge(u, v)

    if undirected.number_of_edges() == 0:
        # No structure: each node is a singleton community in sorted order.
        return {uid: i for i, uid in enumerate(nodes)}

    # Relabel to a deterministic integer order so the partition does not depend
    # on dict/hash iteration order of arbitrary uid strings.
    order = {uid: i for i, uid in enumerate(nodes)}
    inv = {i: uid for uid, i in order.items()}
    relabeled = nx.relabel_nodes(undirected, order, copy=True)

    partition = _best_partition(relabeled, seed=seed)

    # Normalize community ids to a compact 0..k-1 range in first-seen (sorted)
    # order so the labels are stable and small.
    remap: dict[int, int] = {}
    out: dict[str, int] = {}
    for i in sorted(relabeled.nodes()):
        comm = partition[i]
        if comm not in remap:
            remap[comm] = len(remap)
        out[inv[i]] = remap[comm]
    return out


def _best_partition(graph: nx.Graph, *, seed: int) -> dict[int, int]:
    """Seeded Louvain partition ``{node: community}`` via NetworkX's built-in
    ``louvain_communities`` (v2 keeps zero extra dependencies — no
    python-louvain; the v1 fallback path was already NetworkX)."""

    comms = nx.community.louvain_communities(graph, seed=seed)
    part: dict[int, int] = {}
    for cid, members in enumerate(comms):
        for n in members:
            part[n] = cid
    return part


# ---------------------------------------------------------------------------
# Centrality (degree + betweenness) → god-nodes
# ---------------------------------------------------------------------------
def centrality_scores(graph: nx.Graph) -> tuple[dict[str, float], dict[str, float]]:
    """Return ``(degree_centrality, betweenness_centrality)`` per node.

    Computed on the undirected projection (a paper's structural importance does
    not depend on citation direction). Empty graph → two empty dicts; a single
    node → both scores ``0.0`` (no crash).
    """

    if graph.number_of_nodes() == 0:
        return {}, {}

    undirected = graph.to_undirected() if graph.is_directed() else graph

    if graph.number_of_nodes() == 1:
        only = next(iter(graph.nodes()))
        return {only: 0.0}, {only: 0.0}

    degree = nx.degree_centrality(undirected)
    # betweenness is deterministic when normalized + no sampling (k=None).
    # ponytail: exact betweenness is O(V*E); sample (k=) only if a real corpus
    # ever makes this the wall — it runs on the citation projection only.
    betweenness = nx.betweenness_centrality(undirected, normalized=True)
    return degree, betweenness


def _god_score(degree: float, betweenness: float) -> float:
    """Blend degree and betweenness into a single god/read-next score."""

    return DEGREE_WEIGHT * degree + (1.0 - DEGREE_WEIGHT) * betweenness


def god_nodes(
    degree: dict[str, float],
    betweenness: dict[str, float],
    *,
    top_k: int = DEFAULT_TOP_K_GODS,
) -> set[str]:
    """Return the top-``k`` "god-node" ids by the centrality blend.

    Ties are broken by node id so the set is deterministic. Returns at most
    ``k`` nodes (fewer if the graph is smaller). ``top_k <= 0`` or an empty
    graph → empty set.
    """

    if top_k <= 0 or not degree:
        return set()

    scored = [
        (_god_score(degree.get(u, 0.0), betweenness.get(u, 0.0)), u)
        for u in degree
    ]
    # Sort by score desc, then uid asc (deterministic tie-break).
    scored.sort(key=lambda t: (-t[0], t[1]))
    return {u for _score, u in scored[:top_k]}


# ---------------------------------------------------------------------------
# Bridges (community-boundary crossings, high betweenness)
# ---------------------------------------------------------------------------
def bridges(
    graph: nx.Graph,
    community: dict[str, int],
    betweenness: dict[str, float],
) -> tuple[set[str], set[tuple[str, str]]]:
    """Detect community-bridging nodes and edges.

    A **bridge edge** connects two nodes in DIFFERENT communities. A **bridge
    node** is an endpoint of at least one bridge edge that also has *positive*
    betweenness (it actually carries cross-community shortest paths). When all
    betweenness is zero (e.g. a single bridging edge whose endpoints have no
    other paths), every endpoint of a cross-community edge is still flagged so a
    known structural bridge is never missed.

    Returns ``(bridge_node_uids, bridge_edge_pairs)``. Edge pairs are ordered
    ``(u, v)`` with ``u < v`` for determinism. Tiny/empty graphs → two empty
    sets.
    """

    bridge_edges: set[tuple[str, str]] = set()
    endpoint_nodes: set[str] = set()
    for u, v in graph.edges():
        if u == v:
            continue
        if community.get(u) != community.get(v):
            pair = (u, v) if u <= v else (v, u)
            bridge_edges.add(pair)
            endpoint_nodes.add(u)
            endpoint_nodes.add(v)

    if not endpoint_nodes:
        return set(), set()

    # Prefer endpoints that actually carry cross-community shortest paths
    # (positive betweenness). If betweenness is degenerate (all zero) keep every
    # endpoint so a structurally obvious bridge is still surfaced.
    positive = {u for u in endpoint_nodes if betweenness.get(u, 0.0) > 0.0}
    bridge_nodes = positive or endpoint_nodes
    return bridge_nodes, bridge_edges


# ---------------------------------------------------------------------------
# "What to read next": high-centrality unread nodes
# ---------------------------------------------------------------------------
def read_next(
    degree: dict[str, float],
    betweenness: dict[str, float],
    unread: dict[str, bool],
) -> list[str]:
    """Return unread node ids ranked by centrality (best to read first).

    Ranks the nodes flagged ``unread`` by the same degree/betweenness blend used
    for god-nodes, descending, tie-broken by node id. A high-centrality unread
    node is the highest-leverage thing to read next.

    v2 change vs v1: ``unread`` is an INJECTED ``{node: bool}`` map — callers
    derive it from SQL (no filesystem ``pdf_dir`` leg; this module stays pure).
    """

    candidates = [u for u, is_unread in unread.items() if is_unread]
    candidates.sort(
        key=lambda u: (-_god_score(degree.get(u, 0.0), betweenness.get(u, 0.0)), u)
    )
    return candidates


# ---------------------------------------------------------------------------
# Top-level: analyze / annotate a (citation-only) graph.
# ---------------------------------------------------------------------------
def analyze_graph(
    graph: nx.Graph,
    *,
    unread: Optional[dict[str, bool]] = None,
    top_k_gods: int = DEFAULT_TOP_K_GODS,
    louvain_seed: int = LOUVAIN_SEED,
) -> dict:
    """Compute the full analysis and return a JSON-serializable summary dict.

    Assumes ``graph`` is already citation-only (see module docstring); mixed
    graphs go through :func:`annotate_citation_communities`. Does NOT mutate
    ``graph`` (use :func:`annotate_graph` for in-place node attributes). Pure +
    offline. Degrades gracefully on tiny/empty graphs.

    The returned dict has::

        {
          "community": {uid: int},
          "degree": {uid: float},
          "betweenness": {uid: float},
          "is_god_node": {uid: bool},
          "is_bridge": {uid: bool},
          "bridge_edges": [[u, v], ...],
          "unread": {uid: bool},
          "read_next": [uid, ...],            # ranked
          "communities": {int: [uid, ...]},   # members per community
          "god_nodes": [uid, ...],            # the top-k set, ranked
          "stats": {...},
        }
    """

    nodes = list(graph.nodes())
    unread_map = {u: bool((unread or {}).get(u, False)) for u in nodes}

    community = louvain_communities(graph, seed=louvain_seed)
    degree, betweenness = centrality_scores(graph)
    gods = god_nodes(degree, betweenness, top_k=top_k_gods)
    bridge_node_set, bridge_edge_set = bridges(graph, community, betweenness)
    read_order = read_next(degree, betweenness, unread_map)

    is_god = {u: (u in gods) for u in nodes}
    is_bridge = {u: (u in bridge_node_set) for u in nodes}

    communities: dict[int, list[str]] = {}
    for uid in sorted(nodes):
        communities.setdefault(community.get(uid, 0), []).append(uid)

    # god_nodes ranked (deterministic): by blended score desc, uid asc.
    ranked_gods = sorted(
        gods, key=lambda u: (-_god_score(degree.get(u, 0.0), betweenness.get(u, 0.0)), u)
    )

    return {
        "community": community,
        "degree": degree,
        "betweenness": betweenness,
        "is_god_node": is_god,
        "is_bridge": is_bridge,
        "bridge_edges": [list(e) for e in sorted(bridge_edge_set)],
        "unread": unread_map,
        "read_next": read_order,
        "communities": communities,
        "god_nodes": ranked_gods,
        "stats": {
            "node_count": len(nodes),
            "edge_count": graph.number_of_edges(),
            "community_count": len(communities),
            "god_node_count": len(gods),
            "bridge_node_count": len(bridge_node_set),
            "bridge_edge_count": len(bridge_edge_set),
            "unread_count": sum(1 for v in unread_map.values() if v),
        },
    }


def _analysis_block(summary: dict) -> dict:
    """The graph-level ``analysis`` block stashed on ``graph.graph`` (rides into
    ``graph.json``'s node-link ``graph`` key). ``epistemic_type`` is explicit
    per decision 38 (Design #5)."""
    is_bridge = summary["is_bridge"]
    return {
        "community_count": summary["stats"]["community_count"],
        "communities": summary["communities"],
        "god_nodes": summary["god_nodes"],
        "bridge_nodes": sorted(u for u, flag in is_bridge.items() if flag),
        "bridge_edges": summary["bridge_edges"],
        "read_next": summary["read_next"],
        "stats": summary["stats"],
        "epistemic_type": "deterministic",
    }


def annotate_graph(
    graph: nx.Graph,
    *,
    unread: Optional[dict[str, bool]] = None,
    top_k_gods: int = DEFAULT_TOP_K_GODS,
    louvain_seed: int = LOUVAIN_SEED,
) -> dict:
    """Run :func:`analyze_graph` and write the per-node fields ONTO the graph.

    For citation-only graphs (mixed graphs go through
    :func:`annotate_citation_communities`). Adds these node attributes
    (additive — never removes existing fields, keeping the graph.json schema
    backward compatible):

    * ``community`` (int), ``degree`` (float), ``betweenness`` (float),
    * ``is_god_node`` (bool), ``is_bridge`` (bool), ``unread`` (bool).

    Also flags bridge edges with ``is_bridge=True`` on the matching edge data
    and stashes the run-level summary under ``graph.graph["analysis"]``. Returns
    the same summary dict as :func:`analyze_graph`. Mutates ``graph`` in place
    and is safe to call on tiny/empty graphs.
    """

    summary = analyze_graph(
        graph, unread=unread, top_k_gods=top_k_gods, louvain_seed=louvain_seed
    )

    community = summary["community"]
    degree = summary["degree"]
    betweenness = summary["betweenness"]
    is_god = summary["is_god_node"]
    is_bridge = summary["is_bridge"]
    unread_map = summary["unread"]

    for uid in graph.nodes():
        data = graph.nodes[uid]
        data["community"] = community.get(uid, 0)
        data["degree"] = degree.get(uid, 0.0)
        data["betweenness"] = betweenness.get(uid, 0.0)
        data["is_god_node"] = bool(is_god.get(uid, False))
        data["is_bridge"] = bool(is_bridge.get(uid, False))
        data["unread"] = bool(unread_map.get(uid, False))

    bridge_edge_pairs = {tuple(e) for e in summary["bridge_edges"]}
    for u, v in graph.edges():
        pair = (u, v) if u <= v else (v, u)
        graph.edges[u, v]["is_bridge"] = pair in bridge_edge_pairs

    graph.graph["analysis"] = _analysis_block(summary)
    return summary


# ---------------------------------------------------------------------------
# Deep service seam: the shared `graph analyze` / MCP `graph_analyze` summary.
# The ONE function in this module that touches SQL (decision M10). It opens the
# project connection, assembles the run graph + citation communities + bridges,
# and returns a plain, JSON-serializable dict the adapters render. Lifted out of
# ``cli.py`` verbatim so a second adapter cannot re-implement (and drift from) it.
# ---------------------------------------------------------------------------
def analyze_summary(
    h: "ProjectHandle",
    *,
    run_id: Optional[str] = None,
    conn: "sqlite3.Connection | None" = None,
) -> dict:
    """Assemble the deterministic ``graph analyze`` summary as a plain dict.

    Composes the exact same functions the CLI body used —
    :func:`semantic.graph_build.build_graph`,
    :func:`annotate_citation_communities` (with the SQL-derived
    :func:`semantic.graph_build.unread_work_map`), and
    :func:`answer.traverse.bridges_in` — resolving a never-blank display label
    for every god / read-next work via :func:`display.derive_label`.

    ``run_id`` defaults to :func:`run.latest_run_id`, and finally to the literal
    ``"adhoc"`` when the project has no citation run yet — mirroring the CLI
    byte-for-byte. (There is deliberately NO error on a missing run: the empty
    citation projection degrades gracefully; the sibling ``graph export`` command
    is the one that hard-errors on no run, not ``analyze``.)

    Opens and closes a project SQLite connection internally (the module's one
    I/O exception; see the module docstring) — UNLESS the caller passes ``conn``,
    in which case that connection is used and left open (the M10 x M7 rider,
    design 00 §3.1): the MCP ``graph_analyze`` tool passes its M7-managed
    connection (WAL + ``busy_timeout=5000``) so a concurrent ``serve`` cannot make
    it fail on lock contention, while the CLI keeps opening its own via
    ``connect_project_raw`` — additive, byte-parity preserved. Returns a
    JSON-serializable dict::

        {
          "run_id": str,
          "node_count": int, "edge_count": int,        # citation projection
          "community_count": int,
          "communities": [{"community_id": int, "size": int}, ...],  # by id asc
          "god_nodes": [{"work_id": str, "title": str}, ...],        # ranked
          "bridge_edge_count": int, "bridge_node_count": int,
          "bridges": [                                    # by (source, target)
            {"source_work_id": str, "target_work_id": str,
             "source_community": int, "target_community": int,
             "flagged_is_bridge": bool}, ...
          ],
          "unread_count": int,
          "read_next": [{"work_id": str, "title": str}, ...],        # ranked
        }
    """
    from ..answer.traverse import bridges_in
    from ..db.connection import connect_project_raw
    from ..display import derive_label
    from ..run import latest_run_id
    from ..semantic.graph_build import build_graph, unread_work_map

    rid = run_id or latest_run_id(h.slug, root=h.root) or "adhoc"
    owns_conn = conn is None
    if owns_conn:
        conn = connect_project_raw(h.db_path)
    try:
        graph = build_graph(conn, run_id=rid)
        annotate_citation_communities(graph, unread=unread_work_map(conn))
    finally:
        if owns_conn:
            conn.close()

    def _label(work_id: str) -> str:
        data = graph.nodes.get(work_id, {})
        return data.get("label") or derive_label(
            title=data.get("title"), year=data.get("year"), work_id=work_id
        )

    block = graph.graph["analysis"]
    stats = block["stats"]
    return {
        "run_id": rid,
        "node_count": stats["node_count"],
        "edge_count": stats["edge_count"],
        "community_count": block["community_count"],
        "communities": [
            {"community_id": cid, "size": len(block["communities"][cid])}
            for cid in sorted(block["communities"])
        ],
        "god_nodes": [
            {"work_id": wid, "title": _label(wid)} for wid in block["god_nodes"]
        ],
        "bridge_edge_count": stats["bridge_edge_count"],
        "bridge_node_count": stats["bridge_node_count"],
        "bridges": [
            {
                "source_work_id": e["source_work_id"],
                "target_work_id": e["target_work_id"],
                "source_community": e["source_community"],
                "target_community": e["target_community"],
                "flagged_is_bridge": e["flagged_is_bridge"],
            }
            for e in bridges_in(graph)["bridge_edges"]
        ],
        "unread_count": stats["unread_count"],
        "read_next": [
            {"work_id": wid, "title": _label(wid)} for wid in block["read_next"]
        ],
    }


# ---------------------------------------------------------------------------
# Citation-only projection + mixed-graph annotation (the v2 seam)
# ---------------------------------------------------------------------------
def citation_projection(graph: nx.Graph) -> "nx.DiGraph":
    """Project the citation subgraph out of a (possibly mixed) computed view.

    Keeps nodes whose ``node_type`` is absent-or-``'Work'`` and edges flagged
    ``is_citation=True`` — or with the key absent, for the phase-2 citation
    graph whose nodes/edges carry neither attribute. Concept / ConceptAlias
    nodes and discusses/co-occurrence/strut edges are excluded, enforcing the
    citation-only clustering scope (module docstring).
    """
    proj = nx.DiGraph()
    for node, data in graph.nodes(data=True):
        node_type = data.get("node_type")
        if node_type is None or node_type == "Work":
            proj.add_node(node, **data)
    for u, v, data in graph.edges(data=True):
        if u not in proj or v not in proj:
            continue
        if data.get("is_citation", "is_citation" not in data):
            proj.add_edge(u, v, **data)
    return proj


def annotate_citation_communities(
    graph: nx.Graph,
    *,
    unread: Optional[dict[str, bool]] = None,
    top_k_gods: int = DEFAULT_TOP_K_GODS,
    louvain_seed: int = LOUVAIN_SEED,
) -> dict:
    """Analyze the CITATION PROJECTION only; write results back onto the FULL graph.

    The v1 scope contract (module docstring): analysis runs over
    :func:`citation_projection`, then

    * Work nodes get ``community`` / ``degree`` / ``betweenness`` /
      ``is_god_node`` / ``is_bridge`` / ``unread``;
    * Concept / ConceptAlias (any non-Work) nodes get ``community=None`` and NO
      centrality attributes — they were never clustered;
    * citation edges get ``is_bridge``; non-citation edges are untouched;
    * ``graph.graph["analysis"]`` carries the deterministic summary block.

    Returns the projection's :func:`analyze_graph` summary. Mutates ``graph``.
    """
    proj = citation_projection(graph)
    proj_unread = {
        u: bool((unread or {}).get(u, False)) for u in proj.nodes()
    }
    summary = analyze_graph(
        proj, unread=proj_unread, top_k_gods=top_k_gods, louvain_seed=louvain_seed
    )

    community = summary["community"]
    degree = summary["degree"]
    betweenness = summary["betweenness"]
    is_god = summary["is_god_node"]
    is_bridge = summary["is_bridge"]
    unread_map = summary["unread"]

    for uid in graph.nodes():
        data = graph.nodes[uid]
        if uid in proj:
            data["community"] = community.get(uid, 0)
            data["degree"] = degree.get(uid, 0.0)
            data["betweenness"] = betweenness.get(uid, 0.0)
            data["is_god_node"] = bool(is_god.get(uid, False))
            data["is_bridge"] = bool(is_bridge.get(uid, False))
            data["unread"] = bool(unread_map.get(uid, False))
        else:
            # non-Work node: never clustered — community is explicitly None.
            data["community"] = None

    bridge_edge_pairs = {tuple(e) for e in summary["bridge_edges"]}
    for u, v, data in graph.edges(data=True):
        if data.get("is_citation", "is_citation" not in data):
            pair = (u, v) if u <= v else (v, u)
            data["is_bridge"] = pair in bridge_edge_pairs

    graph.graph["analysis"] = _analysis_block(summary)
    return summary
