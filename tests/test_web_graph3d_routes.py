"""Track 3 Stage B — graph3d routes (view dispatch, graph3d.json, provenance,
source serving), all behind ``require_local_session``.

Offline / keyless: a real migrated project.db + a real cache.db (built with a
plain, FK-off connection so the markdown fixture needn't materialize a
conversion_runs row) under a throwaway ``SEEDGRAPH_HOME``; the FastAPI surface is
exercised via Starlette's ``TestClient`` with the ``require_local_session``
dependency override (cross-cutting #1 seam). The auth tests deliberately do NOT
override, proving the gate is wired (non-loopback host + no cookie → 403).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import seedgraph.web as web_pkg
from seedgraph.api.app import app
from seedgraph.db.connection import cache_db_path
import seedgraph.db.migrations as migrations
from seedgraph.cache import store as cache_store
from seedgraph.project import service
from seedgraph.run import ensure_run
from seedgraph.web import serve

SLUG = "g3d"
SECRET_DEF = "selection-on-observables identifies the ATE under this design"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# fixture builders (project.db rows + cache.db rows + blobs)
# --------------------------------------------------------------------------

def _pconn(handle) -> sqlite3.Connection:
    conn = sqlite3.connect(str(handle.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _add_work(conn, work_id, title, year=2020, is_seed=0, doi=None):
    conn.execute(
        "INSERT INTO works (work_id, canonical_title, year, doi, created_at) VALUES (?,?,?,?,?)",
        (work_id, title, year, doi, _now()),
    )
    conn.execute(
        "INSERT INTO project_documents (work_id, inclusion_status, is_seed, created_at, "
        "updated_at) VALUES (?,?,?,?,?)",
        (work_id, "included", is_seed, _now(), _now()),
    )


def _add_run(conn, run_id, work_id, model="llama3", provider="ollama", access_mode="local"):
    conn.execute(
        "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, "
        "markdown_hash, schema_version, prompt_version, model_name, provider, access_mode, "
        "access_class, run_status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, work_id, "md_" + work_id, "h", "v1", "p1", model, provider, access_mode,
         "open_access", "success", _now()),
    )


def _add_claim(conn, claim_id, run_id, work_id, claim_type="method", subtype=None,
               text="some claim", access_class="open_access"):
    conn.execute(
        "INSERT INTO extracted_claims (claim_id, extraction_run_id, work_id, claim_type, "
        "claim_subtype, field_key, normalized_label, claim_text, status, epistemic_type, "
        "access_class, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (claim_id, run_id, work_id, claim_type, subtype, "f", "lbl", text, "found",
         "llm_extracted", access_class, _now()),
    )


def _add_section(conn, section_id, work_id, heading_path, heading_text):
    conn.execute(
        "INSERT INTO document_sections (section_id, markdown_id, markdown_hash, "
        "source_file_id, source_file_hash, work_id, level, ordinal, heading_text, "
        "heading_path, section_kind, start_char, end_char, section_parser_version, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (section_id, "md_" + work_id, "h", "sf", "sfh", work_id, 2, 0, heading_text,
         heading_path, "body", 0, 500, "secv1", _now()),
    )


def _add_span(conn, span_id, work_id, quote, section_id=None):
    conn.execute(
        "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, source_file_id, "
        "source_file_hash, work_id, section_id, start_char, end_char, exact_quote, "
        "quote_hash, page_start, page_end, access_class, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (span_id, "md_" + work_id, "h", "sf", "sfh", work_id, section_id, 10, 29, quote,
         "qh_" + span_id, 4, 4, "open_access", _now()),
    )


def _link_claim_span(conn, claim_id, span_id):
    conn.execute(
        "INSERT INTO claim_spans (claim_id, span_id, rank, created_at) VALUES (?,?,?,?)",
        (claim_id, span_id, 0, _now()),
    )


def _add_concept(conn, concept_id, label, ctype="method", access="open_access", definition=None):
    conn.execute(
        "INSERT INTO concepts (concept_id, normalized_label, canonical_label, concept_type, "
        "definition, paper_frequency, status, epistemic_type, access_class, created_at, "
        "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (concept_id, label, label.upper(), ctype, definition, 2, "auto", "deterministic",
         access, _now(), _now()),
    )


def _add_alias(conn, concept_id, alias_label, fold_reason="acronym"):
    conn.execute(
        "INSERT INTO concept_aliases (concept_id, alias_label, fold_reason, epistemic_type, "
        "created_at) VALUES (?,?,?,?,?)",
        (concept_id, alias_label, fold_reason, "deterministic", _now()),
    )


def _link_claim_concept(conn, claim_id, concept_id, work_id):
    conn.execute(
        "INSERT INTO claim_concepts (claim_id, concept_id, work_id, epistemic_type, "
        "created_at) VALUES (?,?,?,?,?)",
        (claim_id, concept_id, work_id, "deterministic", _now()),
    )


def _add_pge(conn, edge_id, s_type, s_id, t_type, t_id, edge_type, epistemic, access,
             confidence=None):
    conn.execute(
        "INSERT INTO project_graph_edges (edge_id, source_node_type, source_node_id, "
        "target_node_type, target_node_id, edge_type, epistemic_type, confidence, "
        "access_class, run_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (edge_id, s_type, s_id, t_type, t_id, edge_type, epistemic, confidence, access,
         "r1", _now()),
    )


def _add_citation(conn, source_work_id, target_work_id, confidence=1.0):
    conn.execute(
        "INSERT INTO citation_edges (source_work_id, target_work_id, edge_type, provenance, "
        "confidence, run_id, created_at) VALUES (?,?,?,?,?,?,?)",
        (source_work_id, target_work_id, "cites", "provider_reference", confidence, "r1",
         _now()),
    )


def _add_bridge(conn, work_id, source_file_id, file_hash, markdown_id=None, markdown_hash=None):
    conn.execute(
        "INSERT INTO work_source_files (work_id, source_file_id, file_hash, markdown_id, "
        "markdown_hash, acquisition_method, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
        (work_id, source_file_id, file_hash, markdown_id, markdown_hash,
         "already_cached_local", _now(), _now()),
    )


def _seed_cache(*, pdf_uri, md_uri):
    """Build cache.db (source_files + markdown_documents) with a FK-off connection."""
    path = cache_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))  # plain: FK enforcement OFF for the fixture
    try:
        migrations.run_migrations(conn, "cache")
        # work_a: a real pdf blob + a real markdown blob.
        conn.execute(
            "INSERT INTO source_files (source_file_id, file_hash, file_type, access_class, "
            "storage_uri, acquisition_method, created_at) VALUES (?,?,?,?,?,?,?)",
            ("sf_a", "a", "pdf", "open_access", pdf_uri, "already_cached_local", _now()),
        )
        conn.execute(
            "INSERT INTO markdown_documents (markdown_id, conversion_run_id, source_file_id, "
            "markdown_hash, storage_uri, conversion_status, byte_size, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("md_a", "conv_a", "sf_a", "ma", md_uri, "success", 7, _now()),
        )
        # work_b: a source_files row whose blob is MISSING on disk (404-on-serve case).
        conn.execute(
            "INSERT INTO source_files (source_file_id, file_hash, file_type, access_class, "
            "storage_uri, acquisition_method, created_at) VALUES (?,?,?,?,?,?,?)",
            ("sf_b", "b", "pdf", "open_access", "pdfs/missing.pdf", "already_cached_local",
             _now()),
        )
        conn.commit()
    finally:
        conn.close()


def _setup() -> str:
    """Build the full project + cache fixture; return the run_id."""
    handle = service.create_project(SLUG)
    conn = _pconn(handle)
    try:
        _add_work(conn, "work_a", "Paper A", 2020, is_seed=1, doi="10.1/a")
        _add_work(conn, "work_b", "Paper B", 2021)
        _add_run(conn, "extr_a", "work_a")
        _add_run(conn, "extr_b", "work_b")

        # concept::ols (open) + concept::bias (PRIVATE, with a full-text-derived definition)
        _add_concept(conn, "concept::ols", "ols", ctype="method", access="open_access")
        _add_concept(conn, "concept::bias", "selection bias", ctype="concept",
                     access="user_supplied_private", definition=SECRET_DEF)
        _add_alias(conn, "concept::ols", "least squares")

        # claim_1 (work_a) with a span, linked to BOTH concepts (shared work for structural)
        _add_claim(conn, "claim_1", "extr_a", "work_a", text="We estimate by OLS.")
        _add_section(conn, "sec_1", "work_a", "2 Identification > 2.1 Estimator", "2.1 Estimator")
        _add_span(conn, "span_1", "work_a", "We estimate by OLS.", section_id="sec_1")
        _link_claim_span(conn, "claim_1", "span_1")
        _link_claim_concept(conn, "claim_1", "concept::ols", "work_a")
        _link_claim_concept(conn, "claim_1", "concept::bias", "work_a")

        # claim_2 (work_b), span-less, links concept::ols (second paper for ols)
        _add_claim(conn, "claim_2", "extr_b", "work_b", text="OLS is unbiased.")
        _link_claim_concept(conn, "claim_2", "concept::ols", "work_b")

        # edges: strut (work_a discusses ols), PRIVATE llm_inferred concept-concept, citation
        _add_pge(conn, "pge_1", "Work", "work_a", "Concept", "concept::ols", "discusses",
                 "deterministic", "open_access")
        _add_pge(conn, "pge_2", "Concept", "concept::ols", "Concept", "concept::bias",
                 "related_to", "llm_inferred", "user_supplied_private", confidence=0.8)
        _add_citation(conn, "work_a", "work_b", confidence=1.0)

        # bridge: work_a -> pdf + markdown (servable); work_b -> pdf only (blob missing)
        _add_bridge(conn, "work_a", "sf_a", "a", markdown_id="md_a", markdown_hash="ma")
        _add_bridge(conn, "work_b", "sf_b", "b")
        conn.commit()
    finally:
        conn.close()

    # real blobs under the cache root for work_a.
    cache_root = cache_store.cache_root()
    (cache_root / "pdfs").mkdir(parents=True, exist_ok=True)
    (cache_root / "markdown").mkdir(parents=True, exist_ok=True)
    (cache_root / "pdfs" / "a.pdf").write_bytes(b"%PDF-1.4 fixture\n")
    (cache_root / "markdown" / "a.md").write_text("# hello", encoding="utf-8")
    _seed_cache(pdf_uri="pdfs/a.pdf", md_uri="markdown/a.md")

    return ensure_run(SLUG)


def _client() -> TestClient:
    app.dependency_overrides[serve.require_local_session] = lambda: None
    return TestClient(app)


def _clear_override() -> None:
    app.dependency_overrides.pop(serve.require_local_session, None)


# --------------------------------------------------------------------------
# view=auto dispatch: 3d iff concepts exist, else the table (2d alias) view
# --------------------------------------------------------------------------

def test_view_auto_picks_3d_when_concepts_exist():
    run_id = _setup()
    try:
        client = _client()
        resp = client.get(f"/ui/projects/{SLUG}/runs/{run_id}/graph?view=auto")
        assert resp.status_code == 200
        # 3d template: the two-plane WebGL canvas that fetches graph3d.json client-side.
        assert "3D concept view" in resp.text
        assert "graph3d.json" in resp.text
        assert "graph (table view)" not in resp.text
    finally:
        _clear_override()


def test_view_auto_falls_back_to_table_without_concepts():
    service.create_project("g3d_nc")
    conn = _pconn(service.open_project("g3d_nc"))
    try:
        _add_work(conn, "w1", "W1")
        _add_work(conn, "w2", "W2")
        _add_citation(conn, "w1", "w2")
        conn.commit()
    finally:
        conn.close()
    run_id = ensure_run("g3d_nc")
    try:
        client = _client()
        resp = client.get(f"/ui/projects/g3d_nc/runs/{run_id}/graph?view=auto")
        assert resp.status_code == 200
        assert "graph (table view)" in resp.text  # server-rendered table, not the 3d canvas
        assert "3D concept view" not in resp.text
    finally:
        _clear_override()


def test_view_2d_aliases_table_and_explicit_3d_table():
    run_id = _setup()
    try:
        client = _client()
        two_d = client.get(f"/ui/projects/{SLUG}/runs/{run_id}/graph?view=2d")
        assert two_d.status_code == 200
        assert "graph (table view)" in two_d.text  # 2d aliases the table view
        three_d = client.get(f"/ui/projects/{SLUG}/runs/{run_id}/graph?view=3d")
        assert "3D concept view" in three_d.text
        table = client.get(f"/ui/projects/{SLUG}/runs/{run_id}/graph?view=table")
        assert "graph (table view)" in table.text
    finally:
        _clear_override()


# --------------------------------------------------------------------------
# graph3d.json shape + private definition / llm_inferred PRESENT (gate-exempt
# CONTENT behind an auth-gated ROUTE — NOT the public graph.json filter)
# --------------------------------------------------------------------------

def test_graph3d_json_shape_and_private_content_present():
    run_id = _setup()
    try:
        client = _client()
        resp = client.get(f"/api/projects/{SLUG}/runs/{run_id}/graph3d.json")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) >= {"nodes", "links", "threshold", "counts", "warnings"}

        by_id = {n["id"]: n for n in body["nodes"]}
        # works + concepts present; the private concept's full-text-derived definition
        # is present for local display (the public graph.json would withhold it).
        assert "work_a" in by_id and "concept::ols" in by_id and "concept::bias" in by_id
        priv = by_id["concept::bias"]
        assert priv["access_class"] == "user_supplied_private"
        assert priv["definition"] == SECRET_DEF

        # the private llm_inferred concept-concept edge survives into the local view.
        assert any(
            l.get("type") == "concept" and l.get("epistemic_type") == "llm_inferred"
            for l in body["links"]
        )
        # Work document affordances composed from the bridge + cache.
        assert by_id["work_a"]["pdf"] is True
        assert by_id["work_a"]["md"] is True
        assert by_id["work_a"]["seed"] is True
    finally:
        _clear_override()


def test_graph3d_json_carries_community_analysis():
    """Build B chunk 3: the route annotates the graph, so paper records carry an
    integer community + god/bridge flags for a seeded two-community fixture."""
    service.create_project("g3d_comm")
    conn = _pconn(service.open_project("g3d_comm"))
    try:
        for wid in ("a1", "a2", "a3", "b1", "b2", "b3"):
            _add_work(conn, wid, f"Paper {wid}")
        # two citation triangles joined by ONE edge -> 2 communities, 1 bridge.
        for s, t in [("a1", "a2"), ("a2", "a3"), ("a3", "a1"),
                     ("b1", "b2"), ("b2", "b3"), ("b3", "b1"), ("a1", "b1")]:
            _add_citation(conn, s, t)
        conn.commit()
    finally:
        conn.close()
    run_id = ensure_run("g3d_comm")
    try:
        client = _client()
        resp = client.get(f"/api/projects/g3d_comm/runs/{run_id}/graph3d.json")
        assert resp.status_code == 200
        nodes = {n["id"]: n for n in resp.json()["nodes"]}
        assert isinstance(nodes["a1"]["community"], int)
        assert nodes["a1"]["community"] == nodes["a2"]["community"]
        assert nodes["a1"]["community"] != nodes["b1"]["community"]
        # the joining edge's endpoints are flagged bridges; god nodes exist.
        assert nodes["a1"]["is_bridge"] is True and nodes["b1"]["is_bridge"] is True
        assert any(n.get("is_god_node") for n in nodes.values())
        links = resp.json()["links"]
        bridge_links = [l for l in links if l.get("is_bridge")]
        assert {(l["source"], l["target"]) for l in bridge_links} == {("a1", "b1")}
    finally:
        _clear_override()


# --------------------------------------------------------------------------
# BOUNDARY: graph3d.json route writes NO file under runs/ (computed in-memory)
# --------------------------------------------------------------------------

def test_graph3d_json_persists_nothing_under_runs():
    run_id = _setup()
    runs_dir = service.open_project(SLUG).db_path.parent / "runs"

    def snapshot():
        return {p.relative_to(runs_dir).as_posix(): p.stat().st_size
                for p in runs_dir.rglob("*") if p.is_file()}

    before = snapshot()
    try:
        client = _client()
        assert client.get(f"/api/projects/{SLUG}/runs/{run_id}/graph3d.json").status_code == 200
    finally:
        _clear_override()
    after = snapshot()
    assert before == after  # no graph3d.json (or anything) written to the run dir
    assert not (runs_dir / run_id / "graph3d.json").exists()
    assert not (runs_dir / run_id / "graph.json").exists()  # export path untouched


# --------------------------------------------------------------------------
# node / edge provenance drawers
# --------------------------------------------------------------------------

def test_concept_node_provenance_grouped_with_aliases_and_run_meta():
    run_id = _setup()
    try:
        client = _client()
        resp = client.get(
            f"/api/projects/{SLUG}/runs/{run_id}/nodes/concept::ols/provenance"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["node_type"] == "Concept"
        assert body["concept"]["concept_id"] == "concept::ols"
        # grouped by work: work_a (span-bearing) + work_b (span-less).
        works = {w["work_id"]: w for w in body["works"]}
        assert set(works) == {"work_a", "work_b"}
        claim = works["work_a"]["claims"][0]
        assert claim["claim_id"] == "claim_1"
        assert claim["spans"][0]["exact_quote"] == "We estimate by OLS."
        assert claim["spans"][0]["heading_path"] == "2 Identification > 2.1 Estimator"
        assert works["work_b"]["claims"][0]["spans"] == []  # LEFT JOIN, no span
        # aliases + extraction run model/provider.
        assert any(a["alias_label"] == "least squares" for a in body["aliases"])
        assert body["extraction_runs"]["extr_a"]["model_name"] == "llama3"
        assert body["extraction_runs"]["extr_a"]["provider"] == "ollama"
    finally:
        _clear_override()


def test_work_node_provenance_identifiers_and_document_link():
    run_id = _setup()
    try:
        client = _client()
        resp = client.get(f"/api/projects/{SLUG}/runs/{run_id}/nodes/work_a/provenance")
        assert resp.status_code == 200
        body = resp.json()
        assert body["node_type"] == "Work"
        assert body["work"]["identifiers"]["doi"] == "10.1/a"
        assert body["work"]["is_seed"] is True
        assert body["document"]["has_pdf"] is True
        assert body["document"]["pdf_url"] == f"/api/projects/{SLUG}/works/work_a/pdf"
    finally:
        _clear_override()


def test_unknown_node_provenance_is_404():
    run_id = _setup()
    try:
        client = _client()
        resp = client.get(f"/api/projects/{SLUG}/runs/{run_id}/nodes/ghost/provenance")
        assert resp.status_code == 404
    finally:
        _clear_override()


def test_strut_edge_provenance_returns_span_claims():
    run_id = _setup()
    try:
        client = _client()
        resp = client.get(
            f"/api/projects/{SLUG}/runs/{run_id}/edges/provenance",
            params={"source": "work_a", "target": "concept::ols", "edge_type": "discusses"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["evidence_mode"] == "spans"
        assert body["concept_id"] == "concept::ols" and body["work_id"] == "work_a"
        assert body["claims"][0]["claim_id"] == "claim_1"
        assert body["claims"][0]["spans"][0]["span_id"] == "span_1"
    finally:
        _clear_override()


def test_concept_concept_edge_provenance_is_structural_only():
    run_id = _setup()
    try:
        client = _client()
        resp = client.get(
            f"/api/projects/{SLUG}/runs/{run_id}/edges/provenance",
            params={"source": "concept::ols", "target": "concept::bias",
                    "edge_type": "related_to"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["evidence_mode"] == "structural"
        assert body["has_span_evidence"] is False
        assert "work_a" in {w["work_id"] for w in body["shared_works"]}
    finally:
        _clear_override()


# --------------------------------------------------------------------------
# AUTH: every private route refuses a non-loopback / cookieless client (403)
# --------------------------------------------------------------------------

def _private_paths(run_id: str) -> list[str]:
    return [
        f"/ui/projects/{SLUG}/runs/{run_id}/graph",
        f"/api/projects/{SLUG}/runs/{run_id}/graph3d.json",
        f"/api/projects/{SLUG}/runs/{run_id}/nodes/work_a/provenance",
        f"/api/projects/{SLUG}/runs/{run_id}/edges/provenance?source=work_a&target=work_b",
        f"/api/projects/{SLUG}/works/work_a/pdf",
        f"/api/projects/{SLUG}/works/work_a/markdown",
    ]


def test_every_private_route_403_without_session():
    run_id = _setup()
    # TestClient host is "testclient" (non-loopback) and carries no sg_session cookie.
    client = TestClient(app)
    for path in _private_paths(run_id):
        assert client.get(path).status_code == 403, path


def test_private_routes_reachable_with_override():
    run_id = _setup()
    try:
        client = _client()
        for path in _private_paths(run_id):
            assert client.get(path).status_code != 403, path
    finally:
        _clear_override()


# --------------------------------------------------------------------------
# PATH-SAFETY: pdf/markdown serving + traversal / membership 404s (no path leak)
# --------------------------------------------------------------------------

def test_pdf_and_markdown_served_for_member_with_blob():
    run_id = _setup()
    try:
        client = _client()
        pdf = client.get(f"/api/projects/{SLUG}/works/work_a/pdf")
        assert pdf.status_code == 200
        assert pdf.headers["content-type"].startswith("application/pdf")
        assert pdf.content == b"%PDF-1.4 fixture\n"
        md = client.get(f"/api/projects/{SLUG}/works/work_a/markdown")
        assert md.status_code == 200
        assert md.content == b"# hello"
    finally:
        _clear_override()


def test_pdf_404_for_non_member_and_missing_blob():
    run_id = _setup()
    home = str(cache_store.cache_root())
    try:
        client = _client()
        # not a work_source_files member -> 404, no path in the body.
        non_member = client.get(f"/api/projects/{SLUG}/works/work_ghost/pdf")
        assert non_member.status_code == 404
        assert home not in non_member.text
        # member whose cached blob is missing on disk -> 404 (existence-checked).
        missing = client.get(f"/api/projects/{SLUG}/works/work_b/pdf")
        assert missing.status_code == 404
        assert home not in missing.text
        # member with no markdown backfilled -> markdown 404.
        assert client.get(f"/api/projects/{SLUG}/works/work_b/markdown").status_code == 404
    finally:
        _clear_override()


# --------------------------------------------------------------------------
# FRONTEND (Stage C): vendored bundle load order + presence (offline, no browser)
# --------------------------------------------------------------------------

#: The exact UMD load order the two-plane scene depends on. ``three`` MUST be the
#: global first (so our meshes share THE SAME THREE instance the graph renders
#: with); ``d3-octree`` MUST precede ``d3-force-3d`` — the standalone force-3d UMD
#: does not bundle the octree, and forceCollide()/many-body call ``octree()``
#: internally (else "e.octree is not a function" on the first tick).
_VENDOR_ORDER = [
    "three.min.js",
    "three-spritetext.min.js",
    "3d-force-graph.min.js",
    "d3-octree.min.js",
    "d3-force-3d.min.js",
]


def _web_dir() -> Path:
    return Path(web_pkg.__file__).resolve().parent


def test_vendored_libs_present_as_package_data():
    """All 5 UMD bundles ship inside the package (no CDN at view time)."""
    vendor = _web_dir() / "static" / "vendor"
    for name in _VENDOR_ORDER:
        blob = vendor / name
        assert blob.is_file(), f"missing vendored lib: {name}"
        # the real minified bundles are kilobytes+; guards against an empty stub.
        assert blob.stat().st_size > 1000, f"vendored lib looks empty: {name}"


def test_graph3d_template_script_load_order():
    """three -> three-spritetext -> 3d-force-graph -> d3-octree -> d3-force-3d.

    Asserted directly against the template bytes (the Jinja render emits the
    <script> tags verbatim), so this is fully offline / browser-free.
    """
    tpl = (_web_dir() / "templates" / "graph3d.html").read_text(encoding="utf-8")
    # each src appears exactly once; .index raises if a tag is missing.
    positions = [tpl.index(f"/static/vendor/{name}") for name in _VENDOR_ORDER]
    assert positions == sorted(positions), (
        f"vendored <script> tags out of order: {list(zip(_VENDOR_ORDER, positions))}"
    )
    # the load-bearing constraint, called out explicitly.
    assert tpl.index("/static/vendor/d3-octree.min.js") < tpl.index(
        "/static/vendor/d3-force-3d.min.js"
    )


def test_graph3d_template_wires_the_gated_endpoints():
    """The ported template fetches the v2 run-scoped, session-gated surfaces."""
    tpl = (_web_dir() / "templates" / "graph3d.html").read_text(encoding="utf-8")
    assert "/graph3d.json" in tpl
    assert "/nodes/" in tpl and "/provenance" in tpl  # node drawer
    assert "/edges/provenance" in tpl  # edge drawer
    assert "/pdf" in tpl and "/markdown" in tpl  # source open
    assert "evidence_mode" in tpl and "structural" in tpl  # structural banner branch
    assert "#graph { position:absolute; inset:0; }" in tpl  # full-bleed scene


@pytest.mark.parametrize(
    "path",
    [
        # encoded slash / dot-dot in slug, work_id, run_id -> bare 404, never a path.
        f"/api/projects/a%2Fb/works/work_a/pdf",
        f"/api/projects/{SLUG}/works/a%2Fb/pdf",
        f"/api/projects/{SLUG}/works/%2e%2e%2f%2e%2e/markdown",
        f"/api/projects/{SLUG}/runs/a%2Fb/graph3d.json",
        f"/api/projects/{SLUG}/runs/%2e%2e/nodes/work_a/provenance",
    ],
)
def test_path_traversal_attempts_are_path_free_404(path):
    _setup()
    home = str(cache_store.cache_root())
    try:
        client = _client()
        resp = client.get(path)
        assert resp.status_code == 404
        assert home not in resp.text  # the body never leaks a filesystem path
    finally:
        _clear_override()
