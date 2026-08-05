"""Track 3 Stage A — unit tests for the pure 3D transform ``web/graph3d.py``.

The transform is pure (no DB / networkx / filesystem): every input is a hand-written
dict mimicking ``nx.node_link_data(build_graph(conn, run_id), edges="links")``. These
tests pin classification by ``node_type``, the ConceptAlias-hidden rule, scatter
determinism for a fixed seed, and that every per-type node field is populated.
"""

from __future__ import annotations

from collections import Counter

from seedgraph.web.graph3d import CONCEPT_Z, PAPER_Z, build_3d_payload


def _nodelink() -> dict:
    """A hand graph in node-link (edges='links') shape: 2 works + 2 concepts + 1
    ConceptAlias; 1 citation, 1 concept edge, 2 struts, 1 has_alias (drops with the
    alias), 1 self-loop, 1 dangling."""
    return {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [
            {"id": "work_a", "node_type": "Work", "title": "Paper A", "year": 2020},
            {"id": "work_b", "node_type": "Work", "title": "Paper B", "year": 2021},
            {
                "id": "concept::ols",
                "node_type": "Concept",
                "canonical_label": "OLS",
                "concept_type": "method",
                "paper_frequency": 2,
                "status": "auto",
                "epistemic_type": "deterministic",
                "access_class": "open_access",
            },
            {
                # no canonical_label -> name falls back to clean_concept_name
                "id": "concept::selection bias",
                "node_type": "Concept",
                "concept_type": "concept",
                "paper_frequency": 1,
                "status": "provisional",
                "epistemic_type": "llm_extracted",
                "access_class": "user_supplied_private",
            },
            {
                "id": "alias::concept::ols::least squares",
                "node_type": "ConceptAlias",
                "alias_label": "least squares",
                "fold_reason": "exact_key",
            },
        ],
        "links": [
            {
                "source": "work_a",
                "target": "work_b",
                "edge_type": "cites",
                "epistemic_type": "deterministic",
                "access_class": "metadata_only",
                "is_citation": True,
            },
            {
                "source": "concept::ols",
                "target": "concept::selection bias",
                "edge_type": "related_to",
                "epistemic_type": "llm_inferred",
                "access_class": "open_access",
                "confidence": 0.8,
            },
            {
                "source": "work_a",
                "target": "concept::ols",
                "edge_type": "discusses",
                "epistemic_type": "deterministic",
                "access_class": "open_access",
            },
            {
                "source": "work_b",
                "target": "concept::selection bias",
                "edge_type": "discusses",
                "epistemic_type": "deterministic",
                "access_class": "open_access",
            },
            {
                "source": "concept::ols",
                "target": "alias::concept::ols::least squares",
                "edge_type": "has_alias",
                "epistemic_type": "deterministic",
                "access_class": "open_access",
            },
            {  # self-loop -> dropped
                "source": "work_a",
                "target": "work_a",
                "edge_type": "cites",
                "epistemic_type": "deterministic",
                "access_class": "metadata_only",
            },
            {  # dangling -> dropped
                "source": "work_a",
                "target": "ghost",
                "edge_type": "cites",
                "epistemic_type": "deterministic",
                "access_class": "metadata_only",
            },
        ],
    }


def _document_links() -> dict:
    return {
        "work_a": {
            "has_pdf": True,
            "has_markdown": False,
            "access_class": "open_access",
            "inclusion_status": "included",
            "seed": True,
        },
        "work_b": {
            "has_pdf": False,
            "has_markdown": True,
            "access_class": "user_supplied_private",
            "inclusion_status": "metadata_only",
        },
    }


def _provenance_counts() -> dict:
    return {
        "concept::ols": {"papers": 2, "claims": 3, "spans": 1},
        "concept::selection bias": {"papers": 1, "claims": 1, "spans": 0},
    }


def _build() -> dict:
    return build_3d_payload(_nodelink(), _provenance_counts(), _document_links())


def test_classification_by_endpoint_type():
    out = _build()
    types = Counter(l["type"] for l in out["links"])
    # has_alias drops with the alias (dangling); self-loop + ghost drop.
    assert types == {"citation": 1, "concept": 1, "strut": 2}, types
    assert len(out["links"]) == 4
    # carried-through link attributes survive.
    concept_link = next(l for l in out["links"] if l["type"] == "concept")
    assert concept_link["edge_type"] == "related_to"
    assert concept_link["epistemic_type"] == "llm_inferred"
    assert concept_link["confidence"] == 0.8
    assert "k" in concept_link


def test_alias_dropped_from_canvas_and_counted():
    out = _build()
    ids = {n["id"] for n in out["nodes"]}
    assert "alias::concept::ols::least squares" not in ids
    assert len(out["nodes"]) == 4  # 2 works + 2 concepts
    assert out["counts"]["aliases_hidden"] == 1
    assert out["counts"]["self_loops_dropped"] == 1
    # has_alias (to the hidden alias) + ghost both drop as dangling.
    assert out["counts"]["dangling_dropped"] == 2
    assert out["counts"]["citation_links"] == 1
    assert out["counts"]["concept_links"] == 1
    assert out["counts"]["strut_links"] == 2
    assert out["counts"]["works"] == 2
    assert out["counts"]["concepts"] == 2
    assert isinstance(out["warnings"], list)


def test_work_node_fields_populated():
    out = _build()
    by_id = {n["id"]: n for n in out["nodes"]}
    a = by_id["work_a"]
    assert a["type"] == "paper"
    assert a["z"] == PAPER_Z
    assert a["name"] == "Paper A"
    assert a["year"] == 2020
    assert a["pdf"] is True and a["md"] is False
    assert a["seed"] is True
    assert a["inclusion_status"] == "included"
    assert a["access_class"] == "open_access"
    assert a["citation_deg"] == 1  # one citation edge work_a<->work_b
    assert a["deg"] == 2  # citation + strut to concept::ols
    b = by_id["work_b"]
    assert b["pdf"] is False and b["md"] is True
    assert b["seed"] is False
    assert b["citation_deg"] == 1


def test_concept_node_fields_and_prov():
    out = _build()
    by_id = {n["id"]: n for n in out["nodes"]}
    ols = by_id["concept::ols"]
    assert ols["type"] == "concept"
    assert ols["z"] == CONCEPT_Z
    assert ols["name"] == "OLS"
    assert ols["concept_type"] == "method"
    assert ols["paper_frequency"] == 2
    assert ols["status"] == "auto"
    assert ols["epistemic_type"] == "deterministic"
    assert ols["access_class"] == "open_access"
    assert ols["prov"] == {"papers": 2, "claims": 3, "spans": 1}
    # canonical_label-less concept falls back to the cleaned id.
    bias = by_id["concept::selection bias"]
    assert bias["name"] == "selection bias"
    assert bias["prov"] == {"papers": 1, "claims": 1, "spans": 0}


def test_scatter_determinism_for_fixed_seed():
    a = build_3d_payload(_nodelink(), _provenance_counts(), _document_links(), seed=42)
    b = build_3d_payload(_nodelink(), _provenance_counts(), _document_links(), seed=42)
    pa = {n["id"]: (n["x"], n["y"]) for n in a["nodes"]}
    pb = {n["id"]: (n["x"], n["y"]) for n in b["nodes"]}
    assert pa == pb
    # a different seed gives a different frame (sanity: not accidentally constant).
    c = build_3d_payload(_nodelink(), _provenance_counts(), _document_links(), seed=7)
    pc = {n["id"]: (n["x"], n["y"]) for n in c["nodes"]}
    assert pa != pc


def test_analysis_fields_pass_through_on_papers_and_links():
    """Build B chunk 3: community / is_god_node / is_bridge ride through the pure
    transform on paper records; links carry is_bridge + shared_count; concepts
    carry the IDF weight (chunk 5) and community is never invented for them."""
    nl = _nodelink()
    by_id = {n["id"]: n for n in nl["nodes"]}
    by_id["work_a"].update({"community": 0, "is_god_node": True, "is_bridge": True})
    by_id["work_b"].update({"community": 1, "is_god_node": False, "is_bridge": True})
    by_id["concept::ols"]["weight"] = 0.6931
    nl["links"][0]["is_bridge"] = True  # the work_a -> work_b citation
    nl["links"][1]["shared_count"] = 3  # the concept co-occurrence edge

    out = build_3d_payload(nl, _provenance_counts(), _document_links())
    nodes = {n["id"]: n for n in out["nodes"]}
    assert nodes["work_a"]["community"] == 0
    assert nodes["work_a"]["is_god_node"] is True
    assert nodes["work_a"]["is_bridge"] is True
    assert nodes["work_b"]["community"] == 1
    assert nodes["work_b"]["is_god_node"] is False
    assert nodes["concept::ols"]["weight"] == 0.6931
    # concepts get no community field invented by the transform.
    assert "community" not in nodes["concept::ols"]
    citation = next(l for l in out["links"] if l["type"] == "citation")
    assert citation["is_bridge"] is True
    concept_link = next(l for l in out["links"] if l["type"] == "concept")
    assert concept_link["is_bridge"] is False
    assert concept_link["shared_count"] == 3


def test_analysis_fields_default_when_absent():
    """Un-annotated input (no analysis attrs) still yields well-formed records."""
    out = _build()
    nodes = {n["id"]: n for n in out["nodes"]}
    assert nodes["work_a"]["community"] is None
    assert nodes["work_a"]["is_god_node"] is False
    assert nodes["work_a"]["is_bridge"] is False
    assert nodes["concept::ols"]["weight"] is None
    assert all(l["is_bridge"] is False for l in out["links"])


def test_empty_and_concept_less_graphs_warn():
    empty = build_3d_payload({"nodes": [], "links": []})
    assert empty["nodes"] == [] and empty["links"] == []
    assert any("empty" in w for w in empty["warnings"])
    citation_only = build_3d_payload(
        {
            "nodes": [
                {"id": "w1", "node_type": "Work", "title": "W1"},
                {"id": "w2", "node_type": "Work", "title": "W2"},
            ],
            "links": [{"source": "w1", "target": "w2", "edge_type": "cites"}],
        }
    )
    assert citation_only["counts"]["concepts"] == 0
    assert any("no concepts" in w for w in citation_only["warnings"])
