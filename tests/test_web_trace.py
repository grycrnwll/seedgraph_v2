"""Chunk 3 — the web AnswerTrace view + explorable subgraph pane.

Offline / keyless: the FastAPI surface is exercised via Starlette's ``TestClient`` under
the ``require_local_session`` override (the same seam every gated ``/ui`` screen uses),
over the phase_8 fixture corpus + the ``FakeLLMBackend``. The 3D pane is NOT
JS-executed (house precedent: template/route tests only) — the subgraph is
data-attribute-pinned, so the tests assert the payload JSON embedded in the page. The
pure payload function is unit-tested directly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from _phase8_helpers import build_fixture_project, canned_envelope_json, make_answer_config

from seedgraph.answer import answer, compose, save_answer, save_trace
from seedgraph.answer.trace import AnswerTrace, NeighborhoodTrace, TraceCandidate
from seedgraph.answer.types import AnswerCategory, AnswerEnvelope, QuerySpec, QueryType
from seedgraph.api.app import app
from seedgraph.llm.backend import FakeLLMBackend
from seedgraph.web import serve
from seedgraph.web.trace_view import build_trace_subgraph_payload

GROUNDED_Q = 'According to Paper A, what holds "across groups"?'
CITE_Q = "Which works are cited by Paper A?"


@pytest.fixture(autouse=True)
def _clear_backend():
    compose._BACKEND_OVERRIDE = None
    yield
    compose._BACKEND_OVERRIDE = None


@pytest.fixture
def client():
    app.dependency_overrides[serve.require_local_session] = lambda: None
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(serve.require_local_session, None)


def _persist(handle, env, tr, *, run_id=None):
    save_answer(env, slug=handle.slug, root=handle.root, run_id=run_id)
    save_trace(tr, slug=handle.slug, root=handle.root, run_id=run_id)


# --------------------------------------------------------------------------
# route 200 — all schema sections render per path
# --------------------------------------------------------------------------

def test_retrieval_only_trace_renders_all_sections_except_subgraph(client):
    """A grounded no_llm ask has shown candidates but NO neighborhood and NO
    shown_evidence — the page renders header + classification + candidate table, the
    subgraph pane is absent, and the shown rows fall back to text_preview (not a crash
    on the missing shown_evidence — advisor trap 1)."""
    h = build_fixture_project("trace_view_ro")
    env, tr = answer(GROUNDED_Q, h, no_llm=True, config=make_answer_config())
    assert tr.outcome == "retrieval_only" and tr.shown_evidence is None
    _persist(h, env, tr)

    resp = client.get(f"/ui/projects/{h.slug}/answers/{env.answer_id}/trace")
    assert resp.status_code == 200, resp.text[:300]
    text = resp.text
    assert "According to Paper A" in text          # the question (header)
    assert "Query classification" in text
    assert "Ranked candidates" in text
    assert "shown" in text                          # disposition string
    assert "project_membership" in text             # per-row boost breakdown
    assert "Citation neighborhood" not in text      # no neighborhood on this path


def test_citation_search_trace_renders_subgraph_pane(client):
    """A citation_search ask captures a neighborhood → the subgraph pane, the embedded
    payload JSON, and the vendored force-graph stack all render (works-only, T7)."""
    h = build_fixture_project("trace_view_cite")
    env, tr = answer(CITE_Q, h, no_llm=True, config=make_answer_config())
    assert tr.neighborhood is not None and "work_c" in tr.neighborhood.nodes
    _persist(h, env, tr)

    resp = client.get(f"/ui/projects/{h.slug}/answers/{env.answer_id}/trace")
    assert resp.status_code == 200, resp.text[:300]
    text = resp.text
    assert "Citation neighborhood" in text
    assert 'id="trace-subgraph-data"' in text       # embedded, test-pinned payload
    assert "work_c" in text                          # neighbor/recommendation node id
    assert "/static/vendor/3d-force-graph.min.js" in text
    assert "/static/vendor/three.min.js" in text


def test_llm_trace_renders_shown_evidence_and_exact_badge(client):
    """The full LLM path is exact: the shown rows expand to their [E{n}] fragment from
    shown_evidence and the page flags exact dispositions."""
    h = build_fixture_project("trace_view_llm")
    compose._BACKEND_OVERRIDE = FakeLLMBackend(
        response=canned_envelope_json(answer_text="It holds across groups [E1].", cited_markers=[1])
    )
    env, tr = answer(GROUNDED_Q, h, config=make_answer_config())
    assert tr.outcome == "llm" and tr.shown_evidence
    _persist(h, env, tr)

    resp = client.get(f"/ui/projects/{h.slug}/answers/{env.answer_id}/trace")
    assert resp.status_code == 200, resp.text[:300]
    text = resp.text
    assert "exact dispositions" in text
    assert "[E1]" in text                            # shown-evidence marker in the details summary
    assert "shown" in text


def test_run_scoped_trace_route(client):
    """The run-scoped variant resolves the run-nested artifact locations."""
    h = build_fixture_project("trace_view_run")
    env, tr = answer(CITE_Q, h, no_llm=True, config=make_answer_config())
    _persist(h, env, tr, run_id="run_t")

    resp = client.get(
        f"/ui/projects/{h.slug}/runs/run_t/answers/{env.answer_id}/trace"
    )
    assert resp.status_code == 200, resp.text[:300]
    assert "Citation neighborhood" in resp.text
    # the same answer is NOT at the ad-hoc location → that URL 404s.
    assert client.get(f"/ui/projects/{h.slug}/answers/{env.answer_id}/trace").status_code == 404


# --------------------------------------------------------------------------
# 404 page — pre-feature answers / unknown ids
# --------------------------------------------------------------------------

def test_unknown_answer_id_is_404_page(client):
    h = build_fixture_project("trace_view_404")
    resp = client.get(f"/ui/projects/{h.slug}/answers/ans_doesnotexist/trace")
    assert resp.status_code == 404
    assert "No trace is available" in resp.text     # friendly page, not a bare error


def test_envelope_without_trace_is_404_page(client):
    """A pre-feature answer (envelope persisted, no sibling trace) 404s gracefully."""
    h = build_fixture_project("trace_view_notrace")
    env, tr = answer(GROUNDED_Q, h, no_llm=True, config=make_answer_config())
    save_answer(env, slug=h.slug, root=h.root)      # envelope only — no save_trace
    resp = client.get(f"/ui/projects/{h.slug}/answers/{env.answer_id}/trace")
    assert resp.status_code == 404
    assert "No trace is available" in resp.text


# --------------------------------------------------------------------------
# answer_id traversal guard (choke point + both routes)
# --------------------------------------------------------------------------

def test_trace_answer_paths_rejects_traversal_answer_id():
    """The shared choke point rejects a path-separator / ``..`` answer_id BEFORE building
    any path — both web routes and the CLI export inherit the guard through this one call
    (routing quirks can't collapse an embedded ``..``, so the guard is proven directly)."""
    from seedgraph.web.ui import InvalidTraceAnswerId, trace_answer_paths

    h = build_fixture_project("trace_view_guard")
    for bad in ["a/b", "a\\b", "a..b", "../secret", "..", "x/../y"]:
        with pytest.raises(InvalidTraceAnswerId):
            trace_answer_paths(h, bad)
        with pytest.raises(InvalidTraceAnswerId):
            trace_answer_paths(h, bad, run_id="run_1")
    # a clean id still resolves to the sibling envelope / trace paths.
    env_p, tr_p = trace_answer_paths(h, "ans_" + "0" * 32)
    assert env_p.name.endswith(".json") and tr_p.name.endswith(".trace.json")


def test_traversal_answer_id_is_guard_404_not_friendly_page(client):
    """A traversal-shaped answer_id that reaches the handler (``..`` embedded in ONE path
    segment, so httpx/Starlette keep it) gets the plain guard 404 — the same shape the
    run_id guard gives — NOT the friendly "no trace" page (which would imply a valid but
    missing id) and NOT a file read."""
    h = build_fixture_project("trace_view_traversal")

    resp = client.get(f"/ui/projects/{h.slug}/answers/a..b/trace")
    assert resp.status_code == 404
    assert "invalid answer id" in resp.text
    assert "No trace is available" not in resp.text          # not the friendly missing-file page

    resp2 = client.get(f"/ui/projects/{h.slug}/runs/run_1/answers/a..b/trace")
    assert resp2.status_code == 404
    assert "invalid answer id" in resp2.text
    assert "No trace is available" not in resp2.text


# --------------------------------------------------------------------------
# enrichment tolerates a work missing from the DB (renders "unknown", no crash)
# --------------------------------------------------------------------------

def test_enrichment_tolerates_unknown_work(client):
    """A candidate + neighborhood node pointing at a work absent from ``works`` renders
    "unknown" rather than crashing the view."""
    h = build_fixture_project("trace_view_ghost")
    aid = "ans_" + "0" * 32
    env = AnswerEnvelope(
        answer_id=aid, question="ghost?", query_type=QueryType.factual,
        answer_category=AnswerCategory.source_grounded, answer_text="",
    )
    tr = AnswerTrace(
        answer_id=aid, question="ghost?", created_at="2026-01-01T00:00:00+00:00",
        seedgraph_version="test",
        spec=QuerySpec(question="ghost?", protocol_hint=QueryType.factual).model_dump(mode="json"),
        path="standard", outcome="retrieval_only", retrieved_count=1,
        candidates=[
            TraceCandidate(
                item_id="s_ghost", kind="span", work_id="work_ghost",
                disposition="shown", rank_position=1, rank_score=1.0, bm25_score=-1.0,
                access_class="open_access", text_preview="ghost preview",
            )
        ],
        neighborhood=NeighborhoodTrace(
            seeds=["work_ghost"], unknown_seeds=[], depth=1, nodes=["work_ghost"],
            edges=[], coverage_ratio=1.0, engulfed=False,
        ),
    )
    _persist(h, env, tr)

    resp = client.get(f"/ui/projects/{h.slug}/answers/{aid}/trace")
    assert resp.status_code == 200, resp.text[:300]
    assert "unknown" in resp.text                    # the missing-work fallback
    assert "work_ghost" in resp.text                 # the id is still shown for traceability


# --------------------------------------------------------------------------
# the answer render links to its trace
# --------------------------------------------------------------------------

def test_ask_render_links_to_trace(client):
    h = build_fixture_project("trace_view_link")
    resp = client.get(f"/ui/projects/{h.slug}/ask", params={"q": "across groups"})
    assert resp.status_code == 200
    assert "View answer trace" in resp.text
    assert "/answers/ans_" in resp.text and "/trace" in resp.text


# --------------------------------------------------------------------------
# pure subgraph payload function (roles, dedup, works-only, missing meta)
# --------------------------------------------------------------------------

def test_subgraph_payload_roles_dedup_and_membership():
    neigh = {
        "seeds": ["w1"], "unknown_seeds": [], "depth": 1,
        "nodes": ["w1", "w2", "w3", "w2"],                       # w2 duplicated
        "edges": [["w1", "w2"], ["w2", "w3"], ["w1", "w1"],       # self-loop
                  ["w1", "w2"], ["w2", "ghost"]],                 # dup + dangling
        "coverage_ratio": 0.5, "engulfed": False,
    }
    out = build_trace_subgraph_payload(
        neigh,
        recommendation_work_ids=["w3"],
        work_meta={"w1": {"title": "One", "year": 2020}, "w2": {"title": "Two", "year": 2021}},
    )
    # nodes deduped, insertion order preserved.
    assert [n["id"] for n in out["nodes"]] == ["w1", "w2", "w3"]
    # role precedence: seed > recommendation > neighbor.
    assert {n["id"]: n["role"] for n in out["nodes"]} == {
        "w1": "seed", "w2": "neighbor", "w3": "recommendation"
    }
    # enriched meta on found works; missing meta -> title None, name falls back to id.
    by_id = {n["id"]: n for n in out["nodes"]}
    assert by_id["w1"]["title"] == "One" and by_id["w1"]["name"] == "One"
    assert by_id["w3"]["title"] is None and by_id["w3"]["name"] == "w3"
    # links: self-loop dropped, duplicate collapsed, dangling endpoint dropped.
    assert out["links"] == [
        {"source": "w1", "target": "w2"}, {"source": "w2", "target": "w3"}
    ]
    assert out["counts"] == {
        "seed": 1, "neighbor": 1, "recommendation": 1,
        "nodes": 3, "links": 2, "dangling_dropped": 1,
    }


def test_subgraph_payload_empty_neighborhood():
    out = build_trace_subgraph_payload({"seeds": [], "nodes": [], "edges": []})
    assert out["nodes"] == [] and out["links"] == []
    assert out["counts"]["nodes"] == 0
