"""Build B chunks 4 + 8 — graph analysis surfaced in the answer path.

Chunk 4: ``traverse.read_next_candidates`` turns the deterministic
citation-community analysis (``analysis["read_next"]``) into
:class:`Recommendation`s — never citations — and ``harness.answer`` merges them
into ``gap_finding`` envelopes AFTER the co-citation recommendations
(``_merge_recommendations``: additive, first-wins dedup).

Chunk 8: ``traverse.neighborhood`` (depth-bounded UNDIRECTED citation
neighborhood + engulf guard; depth-1 pins the pre-Build-B per-seed
``cited_by + cites_of`` union, nodes AND order) and ``traverse.bridges_in``
(query-time bridge re-derivation with ``flagged_is_bridge`` corroboration).

Fully offline over hand-built NetworkX graphs + the phase-8 fixture ``project.db``.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import networkx as nx
import pytest
from _phase8_helpers import (
    build_fixture_project,
    canned_envelope_json,
    make_answer_config,
)

from seedgraph.answer import answer, compose, traverse
from seedgraph.graph.analyze import annotate_citation_communities
from seedgraph.llm.backend import FakeLLMBackend
from seedgraph.semantic.graph_build import build_graph


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture(autouse=True)
def _clear_backend():
    compose._BACKEND_OVERRIDE = None
    yield
    compose._BACKEND_OVERRIDE = None


@pytest.fixture
def project():
    """Phase-8 fixture corpus + ``work_d``: included, span-less, never a
    citation-edge TARGET — invisible to co-citation gap-finding, so only the
    read-next (high-centrality unread) path can surface it."""
    handle = build_fixture_project("ansgraph")
    conn = sqlite3.connect(str(handle.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        "INSERT INTO works (work_id, canonical_title, year, created_at) "
        "VALUES ('work_d','Paper D: Unread Island',2022,?)",
        (_now(),),
    )
    conn.execute(
        "INSERT INTO project_documents (work_id, inclusion_status, is_seed, "
        "created_at, updated_at) VALUES ('work_d','included',0,?,?)",
        (_now(), _now()),
    )
    conn.commit()
    conn.close()
    return handle


@pytest.fixture
def conn(project):
    c = sqlite3.connect(str(project.db_path))
    c.execute("PRAGMA foreign_keys=ON")
    yield c
    c.close()


# --- traverse.read_next_candidates (unit) -----------------------------------

def test_read_next_candidates_emits_ranked_unread_recommendations(conn):
    """Unread works come back centrality-ranked with the chunk-4 reason string;
    read works (spans) and excluded works never appear."""
    graph = build_graph(conn, run_id="ask")
    recs = traverse.read_next_candidates(conn, graph, 10)
    assert [r.work_id for r in recs] == ["work_c", "work_d"]
    # work_c (cited by work_a) outranks the isolated work_d.
    assert recs[0].reason == (
        "high-centrality unread: ranks 1 by citation-graph centrality"
    )
    assert recs[1].reason == (
        "high-centrality unread: ranks 2 by citation-graph centrality"
    )
    assert recs[0].status == "metadata_only"
    assert recs[1].status == "unavailable"
    # extracted (work_a/work_b) and excluded (work_x) works are never unread.
    assert {"work_a", "work_b", "work_x"}.isdisjoint({r.work_id for r in recs})


def test_read_next_candidates_respects_top_cap(conn):
    graph = build_graph(conn, run_id="ask")
    recs = traverse.read_next_candidates(conn, graph, 1)
    assert [r.work_id for r in recs] == ["work_c"]


def test_read_next_candidates_missing_graph_returns_empty(conn):
    assert traverse.read_next_candidates(conn, None, 5) == []


def test_read_next_candidates_distinguishes_upload_vs_extract_hint(conn):
    """R4 pin: an included span-less work whose PDF is already bridged
    (``work_source_files`` row) is told to RUN EXTRACTION, not to re-upload;
    a work with no bridge row keeps the upload hint."""
    conn.execute(
        "INSERT INTO work_source_files (work_id, source_file_id, file_hash, "
        "acquisition_method, created_at, updated_at) VALUES "
        "('work_d','sf_d','d','manual_upload',?,?)",
        (_now(), _now()),
    )
    conn.commit()
    graph = build_graph(conn, run_id="ask")
    recs = {r.work_id: r for r in traverse.read_next_candidates(conn, graph, 10)}
    assert recs["work_d"].action_hint == (
        "run extraction on the uploaded PDF to extract evidence and cite this work"
    )
    # unchanged path: no bridge row -> the original upload hint.
    assert recs["work_c"].action_hint == (
        "upload the PDF to extract evidence and cite this work"
    )


# --- gap_finding envelope merge (harness.answer) ----------------------------

def test_gap_finding_envelope_merges_cocitation_then_read_next(project):
    """The gap_finding envelope carries BOTH recommendation sources, deduped
    first-wins: work_c (in both) keeps its co-citation reason and leads;
    work_d (read-next only) follows with the chunk-4 reason. Recommendations
    only — never citations, never an LLM call."""
    backend = FakeLLMBackend(
        response=canned_envelope_json(answer_text="never", cited_markers=[1])
    )
    compose._BACKEND_OVERRIDE = backend

    env, _ = answer(
        "What research gaps remain in the corpus?", project,
        config=make_answer_config(),
    )
    assert backend.calls == []  # gap_finding never dispatches the LLM
    assert "gap_finding" in env.warnings

    ids = [r.work_id for r in env.recommendations]
    # dedup: work_c is surfaced by both paths but appears exactly once ...
    assert ids.count("work_c") == 1
    # ... keeping the FIRST (co-citation) reason, not the read-next one.
    rec_c = next(r for r in env.recommendations if r.work_id == "work_c")
    assert rec_c.reason.startswith("co-cited by")
    # read-next-only work_d is merged in AFTER the existing co-citation recs.
    rec_d = next(r for r in env.recommendations if r.work_id == "work_d")
    assert rec_d.reason.startswith("high-centrality unread: ranks")
    assert ids.index("work_c") < ids.index("work_d")
    # recommendations, never citations (decision 38's deterministic floor).
    assert env.citations == []
    assert env.cited_work_ids == []
    assert env.insufficient_evidence is True


def test_gap_finding_no_unread_works_yields_no_read_next_recs():
    """A project whose every work is read (spans) or excluded produces zero
    read-next recommendations — and does not crash the gap_finding branch."""
    handle = build_fixture_project("allread")
    conn = sqlite3.connect(str(handle.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        # the fixture's only unread work becomes excluded -> nothing unread.
        conn.execute(
            "UPDATE project_documents SET inclusion_status='excluded' "
            "WHERE work_id='work_c'"
        )
        conn.commit()
        graph = build_graph(conn, run_id="ask")
        assert traverse.read_next_candidates(conn, graph, 10) == []
    finally:
        conn.close()

    env, _ = answer(
        "What research gaps remain in the corpus?", handle,
        config=make_answer_config(),
    )
    assert env.recommendations == []
    assert env.insufficient_evidence is True


# =============================================================================
# Build B chunk 8 — depth-bounded neighborhood + engulf guard
# =============================================================================

def _work_graph(citation_edges, extra_work_nodes=()) -> nx.DiGraph:
    """A phase-7-shaped citation graph: Work nodes, ``is_citation=True`` edges."""
    g = nx.DiGraph()
    for n in extra_work_nodes:
        g.add_node(n, node_type="Work")
    for u, v in citation_edges:
        for n in (u, v):
            if n not in g:
                g.add_node(n, node_type="Work")
        g.add_edge(
            u, v, edge_type="cites", epistemic_type="deterministic",
            access_class="metadata_only", is_citation=True,
        )
    return g


# --- (1) depth-1 regression pin ----------------------------------------------

def test_neighborhood_depth1_matches_pre_build_b_one_hop_union():
    """THE pinned regression: at depth=1, ``neighborhood`` reproduces the
    pre-Build-B per-seed ``cited_by + cites_of`` union — nodes AND order
    (seeds first in given order, then first-wins per-seed discovery)."""
    g = _work_graph(
        [("s2", "n1"), ("s2", "n3"), ("n2", "s2"), ("s1", "n3"), ("n0", "s1")],
        extra_work_nodes=("z1", "z2"),  # unreached; keep coverage < 0.8
    )
    # a Concept strut must never leak into the citation neighborhood.
    g.add_node("concept_q", node_type="Concept")
    g.add_edge("concept_q", "s2", edge_type="mentions_concept", is_citation=False)

    seeds = ["s2", "s1"]  # non-sorted seed order is preserved
    nbhd = traverse.neighborhood(g, seeds, depth=1)

    # the pre-Build-B one-hop union, computed with the pre-B primitives:
    expected = list(seeds)
    seen = set(expected)
    for s in seeds:
        for v in traverse.cited_by(g, s) + traverse.cites_of(g, s):
            if v not in seen:
                seen.add(v)
                expected.append(v)
    assert nbhd["nodes"] == expected  # nodes AND order
    assert nbhd["nodes"] == ["s2", "s1", "n2", "n1", "n3", "n0"]
    # the harness's neighbour view (nodes minus seeds) == the pre-B rec order.
    assert [n for n in nbhd["nodes"] if n not in seeds] == ["n2", "n1", "n3", "n0"]
    assert nbhd["seeds"] == seeds and nbhd["unknown_seeds"] == []
    assert "concept_q" not in nbhd["nodes"]
    # induced (both endpoints reached), deterministic sorted edge list.
    assert nbhd["edges"] == [
        ["n0", "s1"], ["n2", "s2"], ["s1", "n3"], ["s2", "n1"], ["s2", "n3"],
    ]
    assert nbhd["depth"] == 1
    assert nbhd["coverage_ratio"] == pytest.approx(6 / 8)
    assert nbhd["engulfed"] is False


def test_neighborhood_depth1_pins_project_graph_one_hop(conn):
    """Same pin over the real phase-7 project graph (work_a cites work_c)."""
    graph = build_graph(conn, run_id="ask")
    nbhd = traverse.neighborhood(graph, ["work_a", "work_b"], depth=1)
    assert nbhd["nodes"] == ["work_a", "work_b", "work_c"]
    assert nbhd["edges"] == [["work_a", "work_c"]]
    # works: a, b, c, x, d -> 3 of 5 reached; the guard stays quiet.
    assert nbhd["coverage_ratio"] == pytest.approx(3 / 5)
    assert nbhd["engulfed"] is False


# --- (2) depth-2 joins two rings depth-1 leaves separate ---------------------

def _two_rings_via_middle() -> nx.DiGraph:
    """Two directed 3-rings reachable only through the middle seed ``m``:
    depth 1 from ``m`` touches one entry node per ring; depth 2 pulls in BOTH
    complete rings (and their internal edges). ``z1``/``z2`` stay unreached so
    the engulf guard never fires here."""
    return _work_graph(
        [("a1", "a2"), ("a2", "a3"), ("a3", "a1"),
         ("b1", "b2"), ("b2", "b3"), ("b3", "b1"),
         ("m", "a1"), ("m", "b1")],
        extra_work_nodes=("z1", "z2"),
    )


def test_neighborhood_depth2_joins_rings_depth1_leaves_separate():
    g = _two_rings_via_middle()
    ring_interiors = {"a2", "a3", "b2", "b3"}

    d1 = traverse.neighborhood(g, ["m"], depth=1)
    assert d1["nodes"] == ["m", "a1", "b1"]
    assert ring_interiors.isdisjoint(d1["nodes"])  # rings stay separate
    assert d1["edges"] == [["m", "a1"], ["m", "b1"]]  # no ring edge induced

    d2 = traverse.neighborhood(g, ["m"], depth=2)
    # BFS discovery order: seeds, hop-1 (a1, b1), then per-frontier-node
    # cited_by + cites_of at hop 2.
    assert d2["nodes"] == ["m", "a1", "b1", "a3", "a2", "b3", "b2"]
    assert ring_interiors <= set(d2["nodes"])  # both rings joined in
    assert d2["edges"] == [
        ["a1", "a2"], ["a2", "a3"], ["a3", "a1"],
        ["b1", "b2"], ["b2", "b3"], ["b3", "b1"],
        ["m", "a1"], ["m", "b1"],
    ]
    assert d2["coverage_ratio"] == pytest.approx(7 / 9)
    assert d2["engulfed"] is False
    assert {"z1", "z2"}.isdisjoint(d2["nodes"])


# --- (3) hub engulf guard (unit + envelope warning) --------------------------

def _hub_graph() -> nx.DiGraph:
    """One hub citing 20 leaves — the engulf fixture."""
    return _work_graph([("hub", f"l{i:02d}") for i in range(1, 21)])


def test_neighborhood_hub_engulfs_at_depth2():
    g = _hub_graph()
    d1 = traverse.neighborhood(g, ["l01"], depth=1)
    assert d1["nodes"] == ["l01", "hub"]
    assert d1["coverage_ratio"] == pytest.approx(2 / 21)
    assert d1["engulfed"] is False

    d2 = traverse.neighborhood(g, ["l01"], depth=2)
    assert d2["nodes"] == ["l01", "hub"] + [f"l{i:02d}" for i in range(2, 21)]
    assert d2["coverage_ratio"] == pytest.approx(1.0)
    assert d2["engulfed"] is True  # >= 0.8: warn, never truncate


@pytest.fixture
def hub_project():
    """Phase-8 fixture + a hub citing 20 leaves, reachable from the retrieved
    seed ``work_a`` at hop 2: work_a -> work_hub -> work_leaf01..20. Depth 2
    then covers 24/25 works (>= 0.8) while depth 1 covers 4/25."""
    handle = build_fixture_project("anshub")
    conn = sqlite3.connect(str(handle.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        "INSERT INTO works (work_id, canonical_title, year, created_at) "
        "VALUES ('work_hub','Hub Survey',2019,?)",
        (_now(),),
    )
    conn.execute(
        "INSERT INTO citation_edges (source_work_id, target_work_id, edge_type, "
        "provenance, confidence, run_id, created_at) VALUES "
        "('work_a','work_hub','cites','provider_reference',1.0,'cite_hub',?)",
        (_now(),),
    )
    for i in range(1, 21):
        leaf = f"work_leaf{i:02d}"
        conn.execute(
            "INSERT INTO works (work_id, canonical_title, year, created_at) "
            "VALUES (?,?,2020,?)",
            (leaf, f"Leaf paper {i:02d}", _now()),
        )
        conn.execute(
            "INSERT INTO citation_edges (source_work_id, target_work_id, edge_type, "
            "provenance, confidence, run_id, created_at) VALUES "
            "('work_hub',?,'cites','provider_reference',1.0,'cite_hub',?)",
            (leaf, _now()),
        )
    conn.commit()
    conn.close()
    return handle


def test_citation_search_envelope_engulf_warning_at_depth2(hub_project):
    """``answer(..., graph_depth=2)`` on the hub fixture sets the engulf warning
    on the envelope; the default depth 1 stays quiet and reproduces the pre-B
    one-hop recommendation surface (no leaf works)."""
    q = "Which works are cited by Paper A?"

    env2, _ = answer(q, hub_project, no_llm=True, graph_depth=2,
                  config=make_answer_config())
    assert any(
        "covers >=80% of the citation graph" in w for w in env2.warnings
    ), env2.warnings
    rec_ids2 = {r.work_id for r in env2.recommendations}
    # warn, never truncate: the hop-2 leaves ARE surfaced.
    assert {"work_hub", "work_leaf01"} <= rec_ids2

    env1, _ = answer(q, hub_project, no_llm=True, config=make_answer_config())
    assert not any("covers >=80%" in w for w in env1.warnings), env1.warnings
    rec_ids1 = {r.work_id for r in env1.recommendations}
    # today's one-hop union: span-less neighbours of the seeds only.
    assert {"work_hub", "work_c"} <= rec_ids1
    assert not any(r.startswith("work_leaf") for r in rec_ids1)


# --- (4) unknown seeds are reported, never invented --------------------------

def test_neighborhood_unknown_seeds_reported_never_invented():
    g = _work_graph([("w1", "w2")])
    g.add_node("concept_x", node_type="Concept")
    g.add_edge("concept_x", "w1", edge_type="mentions_concept", is_citation=False)

    # duplicate seed deduped; a typo'd work and a Concept id (present in the
    # FULL graph but outside the citation projection) both land in
    # unknown_seeds and contribute nothing.
    nbhd = traverse.neighborhood(g, ["w1", "work_ghost", "concept_x", "w1"], depth=1)
    assert nbhd["seeds"] == ["w1"]
    assert nbhd["unknown_seeds"] == ["work_ghost", "concept_x"]
    assert nbhd["nodes"] == ["w1", "w2"]  # never invented into the subgraph
    assert all("work_ghost" not in e and "concept_x" not in e for e in nbhd["edges"])

    # all seeds unknown: empty result, guard quiet, nothing fabricated.
    none = traverse.neighborhood(g, ["nope"], depth=3)
    assert none["seeds"] == [] and none["unknown_seeds"] == ["nope"]
    assert none["nodes"] == [] and none["edges"] == []
    assert none["coverage_ratio"] == 0.0 and none["engulfed"] is False

    # missing graph: same shape, seeds all reported unknown.
    missing = traverse.neighborhood(None, ["a", "b"], depth=2)
    assert missing["nodes"] == [] and missing["unknown_seeds"] == ["a", "b"]
    assert missing["engulfed"] is False


# --- (5) bridges_in: flagged vs derived --------------------------------------

def test_bridges_in_distinguishes_flagged_vs_derived():
    """``bridges_in`` re-derives bridges from community labels and corroborates
    each against the build-time flag: the annotated joining edge reports
    ``flagged_is_bridge=True``; a cross-community edge added AFTER annotation
    (a stale flag scenario) is still derived but reports ``False``."""
    g = _work_graph(
        [("a1", "a2"), ("a2", "a3"), ("a3", "a1"),
         ("b1", "b2"), ("b2", "b3"), ("b3", "b1"),
         ("a1", "b1")]
    )
    # concept struts never form bridges (community=None, non-citation edge).
    g.add_node("concept_x", node_type="Concept")
    g.add_edge("concept_x", "b1", edge_type="mentions_concept", is_citation=False)

    # un-annotated graph: no community labels -> nothing derives as a bridge.
    assert traverse.bridges_in(g)["bridge_edges"] == []

    annotate_citation_communities(g)
    out = traverse.bridges_in(g)
    assert [
        (e["source_work_id"], e["target_work_id"], e["flagged_is_bridge"])
        for e in out["bridge_edges"]
    ] == [("a1", "b1", True)]
    assert out["bridge_edges"][0]["source_community"] != (
        out["bridge_edges"][0]["target_community"]
    )
    assert set(out["flagged_bridge_nodes"]) == {"a1", "b1"}

    # a NEW cross-community citation edge, added after annotation: derived as a
    # bridge but NOT flagged — the auditable flagged-vs-derived distinction.
    g.add_edge("a2", "b2", edge_type="cites", is_citation=True)
    out2 = traverse.bridges_in(g)
    flags = {
        (e["source_work_id"], e["target_work_id"]): e["flagged_is_bridge"]
        for e in out2["bridge_edges"]
    }
    assert flags == {("a1", "b1"): True, ("a2", "b2"): False}
    assert out2["bridge_nodes"] == ["a1", "a2", "b1", "b2"]  # derived endpoints
    assert set(out2["flagged_bridge_nodes"]) == {"a1", "b1"}  # build-time flags
    assert all(
        "concept_x" not in (e["source_work_id"], e["target_work_id"])
        for e in out2["bridge_edges"]
    )
