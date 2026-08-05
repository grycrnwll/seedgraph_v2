"""Build B chunk 1/2 — pure graph analysis (`graph/analyze.py`) + export wiring.

Pins the ported v1 determinism discipline via STRUCTURAL facts (community counts
on unambiguous fixtures, bridge identity) — never raw community ids on ambiguous
graphs — so a networkx version bump cannot break the pin. Fully offline.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import networkx as nx
import pytest

from seedgraph.graph import analyze


def _two_triangles() -> nx.DiGraph:
    """Two triangles joined by ONE edge: unambiguously 2 communities, the joining
    edge is the only bridge."""
    g = nx.DiGraph()
    for a, b in [("a1", "a2"), ("a2", "a3"), ("a3", "a1"),
                 ("b1", "b2"), ("b2", "b3"), ("b3", "b1"),
                 ("a1", "b1")]:
        g.add_edge(a, b)
    return g


# --------------------------------------------------------------------------
# determinism + structural facts
# --------------------------------------------------------------------------

def test_partition_deterministic_and_structural():
    g = _two_triangles()
    p1 = analyze.louvain_communities(g)
    p2 = analyze.louvain_communities(g)
    assert p1 == p2  # same graph twice => identical partition
    # structural facts, not raw ids: 2 communities; each triangle is one.
    assert len(set(p1.values())) == 2
    assert p1["a1"] == p1["a2"] == p1["a3"]
    assert p1["b1"] == p1["b2"] == p1["b3"]
    assert p1["a1"] != p1["b1"]


def test_bridge_is_the_joining_edge():
    g = _two_triangles()
    community = analyze.louvain_communities(g)
    _degree, betweenness = analyze.centrality_scores(g)
    bridge_nodes, bridge_edges = analyze.bridges(g, community, betweenness)
    assert bridge_edges == {("a1", "b1")}
    assert bridge_nodes <= {"a1", "b1"} and bridge_nodes


def test_bridges_degenerate_all_zero_betweenness_keeps_endpoints():
    # two singletons joined by one edge: betweenness is all-zero; the structural
    # bridge must still surface (v1 degenerate fallback).
    g = nx.DiGraph()
    g.add_edge("x", "y")
    community = {"x": 0, "y": 1}
    bridge_nodes, bridge_edges = analyze.bridges(g, community, {"x": 0.0, "y": 0.0})
    assert bridge_edges == {("x", "y")}
    assert bridge_nodes == {"x", "y"}


# --------------------------------------------------------------------------
# tiny/empty degradation
# --------------------------------------------------------------------------

def test_empty_graph_degrades():
    g = nx.DiGraph()
    summary = analyze.analyze_graph(g)
    assert summary["community"] == {}
    assert summary["stats"]["node_count"] == 0
    assert summary["read_next"] == []


def test_single_node_degrades():
    g = nx.DiGraph()
    g.add_node("only")
    summary = analyze.analyze_graph(g)
    assert summary["degree"] == {"only": 0.0}
    assert summary["betweenness"] == {"only": 0.0}
    assert summary["community"] == {"only": 0}
    assert summary["is_bridge"] == {"only": False}


def test_no_edges_each_node_singleton_community_sorted():
    g = nx.DiGraph()
    g.add_nodes_from(["c", "a", "b"])
    assert analyze.louvain_communities(g) == {"a": 0, "b": 1, "c": 2}


# --------------------------------------------------------------------------
# god nodes
# --------------------------------------------------------------------------

def test_god_node_tie_break_by_uid():
    # identical centralities => the uid-ascending prefix wins deterministically.
    degree = {u: 0.5 for u in ("n4", "n2", "n1", "n3")}
    betweenness = {u: 0.0 for u in degree}
    assert analyze.god_nodes(degree, betweenness, top_k=2) == {"n1", "n2"}
    assert analyze.god_nodes(degree, betweenness, top_k=0) == set()
    assert analyze.god_nodes({}, {}, top_k=3) == set()


def test_god_nodes_top_k_by_blend():
    g = _two_triangles()
    degree, betweenness = analyze.centrality_scores(g)
    gods = analyze.god_nodes(degree, betweenness, top_k=2)
    # a1 and b1 carry the joining edge: highest degree + all the betweenness.
    assert gods == {"a1", "b1"}


# --------------------------------------------------------------------------
# citation-only scope (the concept-strut fixture)
# --------------------------------------------------------------------------

def _mixed_graph() -> nx.DiGraph:
    """Two paper triangles + a dense concept strut web that would merge them if
    the mixed graph were clustered naively."""
    g = _two_triangles()
    for n in list(g.nodes()):
        g.nodes[n]["node_type"] = "Work"
    for u, v in g.edges():
        g.edges[u, v]["is_citation"] = True
    g.add_node("concept::x", node_type="Concept")
    g.add_node("alias::concept::x::y", node_type="ConceptAlias")
    # dense struts: the concept discusses EVERY paper in both triangles.
    for n in ("a1", "a2", "a3", "b1", "b2", "b3"):
        g.add_edge(n, "concept::x", edge_type="discusses", is_citation=False)
    g.add_edge("concept::x", "alias::concept::x::y", edge_type="has_alias",
               is_citation=False)
    return g


def test_citation_projection_excludes_concepts_and_struts():
    g = _mixed_graph()
    proj = analyze.citation_projection(g)
    assert set(proj.nodes()) == {"a1", "a2", "a3", "b1", "b2", "b3"}
    assert all(d.get("is_citation") for _u, _v, d in proj.edges(data=True))
    assert proj.number_of_edges() == 7


def test_citation_projection_keeps_attributeless_phase2_graph():
    # phase-2 graphs carry neither node_type nor is_citation: everything is kept.
    g = _two_triangles()
    proj = analyze.citation_projection(g)
    assert set(proj.nodes()) == set(g.nodes())
    assert proj.number_of_edges() == g.number_of_edges()


def test_mixed_graph_partition_equals_struts_removed_partition():
    mixed = _mixed_graph()
    analyze.annotate_citation_communities(mixed)
    pure = analyze.louvain_communities(analyze.citation_projection(mixed))
    for n in ("a1", "a2", "a3", "b1", "b2", "b3"):
        assert mixed.nodes[n]["community"] == pure[n]
    # the strut web did NOT merge the two paper communities.
    assert mixed.nodes["a1"]["community"] != mixed.nodes["b1"]["community"]
    # Concept/alias nodes: community=None, no centrality attrs.
    for cn in ("concept::x", "alias::concept::x::y"):
        assert mixed.nodes[cn]["community"] is None
        assert "degree" not in mixed.nodes[cn]
        assert "betweenness" not in mixed.nodes[cn]
        assert "is_god_node" not in mixed.nodes[cn]
    # bridge flag on the joining CITATION edge; strut edges untouched.
    assert mixed.edges["a1", "b1"]["is_bridge"] is True
    assert "is_bridge" not in mixed.edges["a1", "concept::x"]
    # graph-level analysis block is present and deterministic-typed.
    block = mixed.graph["analysis"]
    assert block["epistemic_type"] == "deterministic"
    assert block["community_count"] == 2
    assert block["bridge_edges"] == [["a1", "b1"]]


# --------------------------------------------------------------------------
# read_next with an injected unread map
# --------------------------------------------------------------------------

def test_read_next_ranks_unread_by_centrality():
    g = _two_triangles()
    degree, betweenness = analyze.centrality_scores(g)
    unread = {n: True for n in g.nodes()}
    order = analyze.read_next(degree, betweenness, unread)
    assert set(order) == set(g.nodes())
    # the two hub nodes lead (highest blend), uid tie-break makes it a1 then b1.
    assert order[:2] == ["a1", "b1"]

    # only unread nodes are ranked; read nodes never appear.
    some = analyze.read_next(degree, betweenness, {"a2": True, "b3": False})
    assert some == ["a2"]


def test_annotate_citation_communities_threads_unread():
    g = _mixed_graph()
    summary = analyze.annotate_citation_communities(
        g, unread={"a1": True, "concept::x": True}
    )
    assert g.nodes["a1"]["unread"] is True
    assert g.nodes["a2"]["unread"] is False
    # a concept can never be "unread": it is not in the projection.
    assert "unread" not in g.nodes["concept::x"]
    assert summary["read_next"] == ["a1"]
    assert g.graph["analysis"]["read_next"] == ["a1"]


# --------------------------------------------------------------------------
# annotate_graph (in-place annotation of a citation-only graph)
# --------------------------------------------------------------------------

def test_annotate_graph_writes_attrs_and_stash_deterministically():
    g = _two_triangles()
    summary = analyze.annotate_graph(g, unread={"b2": True})
    for n in g.nodes():
        data = g.nodes[n]
        assert isinstance(data["community"], int)
        assert isinstance(data["degree"], float)
        assert isinstance(data["betweenness"], float)
        assert isinstance(data["is_god_node"], bool)
        assert isinstance(data["is_bridge"], bool)
    assert g.nodes["b2"]["unread"] is True
    assert g.nodes["a1"]["unread"] is False
    # edge flags: only the joining edge bridges.
    assert g.edges["a1", "b1"]["is_bridge"] is True
    assert g.edges["a1", "a2"]["is_bridge"] is False
    # stash mirrors the summary and is deterministic across a second run.
    block = g.graph["analysis"]
    assert block["epistemic_type"] == "deterministic"
    assert block["read_next"] == ["b2"] == summary["read_next"]
    assert block["bridge_edges"] == [["a1", "b1"]]
    again = analyze.annotate_graph(_two_triangles(), unread={"b2": True})
    assert again == summary


# --------------------------------------------------------------------------
# chunk 2 — export wiring (annotation + unread SQL + manifest block)
# --------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_export_project(tmp_path) -> sqlite3.Connection:
    import seedgraph.db.migrations as migrations

    tmp_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(tmp_path / "project.db"))
    conn.execute("PRAGMA foreign_keys=ON")
    migrations.run_migrations(conn, "project")

    def add_work(work_id, status="included", spans=0):
        conn.execute(
            "INSERT INTO works (work_id, canonical_title, created_at) VALUES (?,?,?)",
            (work_id, f"Paper {work_id}", _now()),
        )
        conn.execute(
            "INSERT INTO project_documents (work_id, inclusion_status, is_seed, "
            "created_at, updated_at) VALUES (?,?,0,?,?)",
            (work_id, status, _now(), _now()),
        )
        for i in range(spans):
            conn.execute(
                "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, "
                "source_file_id, source_file_hash, work_id, start_char, end_char, "
                "exact_quote, quote_hash, access_class, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"sp_{work_id}_{i}", "md_" + work_id, "h", "sf", "sfh", work_id,
                 0, 5, "quote", f"qh_{work_id}_{i}", "open_access", _now()),
            )

    add_work("work_read", spans=1)
    add_work("work_unread", spans=0)
    add_work("work_meta", status="metadata_only")
    add_work("work_excl", status="excluded")
    conn.execute(
        "INSERT INTO concepts (concept_id, normalized_label, canonical_label, "
        "concept_type, paper_frequency, status, epistemic_type, access_class, "
        "created_at, updated_at) VALUES ('concept::x','x','X','method',1,'auto',"
        "'deterministic','open_access',?,?)",
        (_now(), _now()),
    )
    for s, t in [("work_read", "work_unread"), ("work_unread", "work_meta")]:
        conn.execute(
            "INSERT INTO citation_edges (source_work_id, target_work_id, edge_type, "
            "provenance, confidence, run_id, created_at) VALUES (?,?,?,?,?,?,?)",
            (s, t, "cites", "provider_reference", 1.0, "r1", _now()),
        )
    conn.commit()
    return conn


def test_unread_work_map_from_sql(tmp_path):
    from seedgraph.semantic.graph_build import unread_work_map

    conn = _seed_export_project(tmp_path)
    unread = unread_work_map(conn)
    assert unread["work_read"] is False       # has a span
    assert unread["work_unread"] is True      # zero spans
    assert unread["work_meta"] is True        # metadata_only
    assert unread["work_excl"] is False       # excluded works are NOT unread
    conn.close()


def test_export_graph_carries_analysis(tmp_path):
    from seedgraph.semantic import export

    conn = _seed_export_project(tmp_path)
    written = export.export_graph(conn, slug="x", run_id="r1", fmt="json",
                                  root=tmp_path / "h1")
    doc = json.loads([p for p in written if p.name == "graph.json"][0].read_text())
    nodes = {n["id"]: n for n in doc["nodes"]}
    # integer community + float centrality on Work nodes.
    for wid in ("work_read", "work_unread", "work_meta"):
        assert isinstance(nodes[wid]["community"], int)
        assert isinstance(nodes[wid]["degree"], float)
        assert isinstance(nodes[wid]["betweenness"], float)
        assert isinstance(nodes[wid]["is_god_node"], bool)
        assert isinstance(nodes[wid]["is_bridge"], bool)
    # Concept nodes: community None (never clustered), no centrality attrs.
    assert nodes["concept::x"]["community"] is None
    assert "degree" not in nodes["concept::x"]
    # top-level analysis block.
    block = doc["graph"]["analysis"]
    assert block["epistemic_type"] == "deterministic"
    assert set(block) >= {"communities", "god_nodes", "bridge_edges", "read_next"}
    assert "work_unread" in block["read_next"]
    assert "work_excl" not in block["read_next"]

    # byte-identical double export (determinism through the export path).
    again = export.export_graph(conn, slug="x", run_id="r1", fmt="json",
                                root=tmp_path / "h2")
    a = [p for p in written if p.name == "graph.json"][0].read_bytes()
    b = [p for p in again if p.name == "graph.json"][0].read_bytes()
    assert a == b

    # manifest gains the idempotent section-keyed analysis block.
    manifest = json.loads(
        ((tmp_path / "h1") / "projects" / "x" / "runs" / "r1" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    ana = manifest["sections"]["analysis"]
    assert set(ana) >= {"community_count", "god_node_count", "bridge_edge_count",
                        "unread_count"}
    assert ana["unread_count"] == 2  # work_unread + work_meta
    conn.close()
