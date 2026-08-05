"""Phase 8 — answer harness: pre-classify / retrieve / rank / recommendations /
save-path / milestone acceptance tests (plan §11).

Fully offline against a fake LLM backend and a tiny fixture ``project.db`` (~3 works,
spans, claims, citation_edges, plus 1 metadata_only work).
"""

from __future__ import annotations

import itertools
import re
import sqlite3

import pytest
from _phase8_helpers import (
    build_fixture_project,
    canned_envelope_json,
    empty_fts_project,
    make_answer_config,
    make_spec,
)

from seedgraph.answer import answer, save_answer, AnswerEnvelope  # noqa: F401
from seedgraph.answer import compose, harness, rank, retrieve, traverse  # noqa: F401
from seedgraph.semantic.graph_build import build_graph
from seedgraph.answer.types import (  # noqa: F401
    AnswerCategory,
    AnswerMode,
    QuerySpec,
    QueryType,
    RankedCandidate,
    RetrievedItem,
)
from seedgraph.llm.backend import FakeLLMBackend

ANSWER_ID_RE = re.compile(r"^ans_[0-9a-f]{32}$")


@pytest.fixture
def project():
    return build_fixture_project("ph8_main")


@pytest.fixture
def conn(project):
    c = sqlite3.connect(str(project.db_path))
    c.execute("PRAGMA foreign_keys=ON")
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _clear_backend():
    compose._BACKEND_OVERRIDE = None
    yield
    compose._BACKEND_OVERRIDE = None


# --- pre-classifier (08 §4 / §6.1 steps 1-2) -------------------------------

def test_preclassify_maps_each_example_question_to_protocol():
    """Each 08 §4 example question maps to its expected ``QueryType`` protocol hint."""
    cases = {
        "Which estimator does Paper A use?": QueryType.factual,
        'Compare "Paper A" and "Paper B" on identification.': QueryType.comparative,
        "Synthesize the findings on identification.": QueryType.synthesis,
        "What research gaps remain in the corpus?": QueryType.gap_finding,
        "Which works are cited by Paper A?": QueryType.citation_search,
        "What is the rank condition?": QueryType.concept_explanation,
        "What assumptions does Paper A rely on?": QueryType.assumption_search,
        "Show me the evidence for parallel trends.": QueryType.evidence_request,
    }
    for question, expected in cases.items():
        assert harness._preclassify(question).protocol_hint == expected, question


def test_preclassify_comparative_two_named_works_extracts_work_tokens():
    """A comparative question naming two works extracts the two work_tokens that drive
    per-work partitioning (§6.4) — the token-extraction precondition."""
    spec = harness._preclassify('Compare "Paper A" and "Paper B" on assumptions.')
    assert spec.protocol_hint == QueryType.comparative
    assert spec.work_tokens == ["Paper A", "Paper B"]


def test_comparative_retrieval_partitions_candidates_by_work(project, conn):
    """§6.4 behavior — a comparative question naming two works runs the factual
    protocol once per named work and keeps candidates partitioned by ``work_id``
    (contiguous per-work blocks, no leakage of unrelated works). This is the real
    partition-behavior check the §11 criterion requires (not just token extraction)."""
    spec = harness._preclassify(
        'Compare "Paper A" and "Paper B" on parallel trends and the rank condition.'
    )
    assert spec.protocol_hint == QueryType.comparative
    items = harness._comparative_retrieve(conn, "ph8_main", spec, 40)
    work_ids = [it.work_id for it in items]
    # both named works contribute evidence ...
    assert "work_a" in work_ids and "work_b" in work_ids
    # ... candidates are kept in contiguous per-work partitions (work_a block, then
    # work_b block) — NOT a single OR'd retrieval interleaved by score ...
    assert [k for k, _ in itertools.groupby(work_ids)] == ["work_a", "work_b"]
    # ... and every candidate is scoped to one of the two named works.
    assert set(work_ids) <= {"work_a", "work_b"}
    # each partition's items are all that work's (post-filtered purity).
    for wid, group in itertools.groupby(items, key=lambda it: it.work_id):
        assert all(it.work_id == wid for it in group)


def test_comparative_single_resolved_work_degrades_to_single_retrieval(conn):
    """When fewer than two named works resolve, ``_comparative_retrieve`` degrades to a
    single un-partitioned retrieval identical to ``retrieve.retrieve`` (a single-work
    comparative is pre-classified as factual; this guards the resolution-degrade
    path, §6.4)."""
    spec = harness._preclassify('Compare "Paper A" and "Nonexistent Paper Z" on assumptions.')
    degraded = harness._comparative_retrieve(conn, "ph8_main", spec, 40)
    plain = retrieve.retrieve(conn, "ph8_main", spec, 40)
    # only work_a resolves -> identical to the standard single retrieval (no partition).
    assert [it.item_id for it in degraded] == [it.item_id for it in plain]


# --- citation-graph traversal (08 §5 step 7; §6.1 / §6.5) ------------------

def test_traverse_cites_of_and_cited_by_over_project_graph(conn):
    """``traverse.cites_of`` / ``cited_by`` walk the phase-7 citation graph: work_a
    cites work_c (08 §5 step 7). Unknown works -> ``[]`` (no raise)."""
    graph = build_graph(conn, run_id="ask")
    assert traverse.cites_of(graph, "work_a") == ["work_c"]
    assert traverse.cited_by(graph, "work_c") == ["work_a"]
    assert traverse.cites_of(graph, "work_c") == []
    assert traverse.cited_by(graph, "work_a") == []
    assert traverse.cites_of(graph, "work_missing") == []


def test_cocitation_candidates_surfaces_metadata_only_gap(conn):
    """``traverse.cocitation_candidates`` surfaces the co-cited, span-less
    ``metadata_only`` work as a gap :class:`Recommendation` over existing rows only
    (08 §11)."""
    recs = traverse.cocitation_candidates(conn, "ph8_main", 5)
    assert any(r.work_id == "work_c" and r.status == "metadata_only" for r in recs)
    # work_a / work_b have spans -> never surfaced as a gap recommendation.
    assert all(r.work_id not in ("work_a", "work_b") for r in recs)


def test_gap_finding_protocol_uses_cocitation_traversal_no_llm(project):
    """A gap_finding question runs step-7 co-citation traversal (NOT compose's
    metadata sweep), returns recommendations sourced from ``traverse``, and makes NO
    LLM call (§6.1 — gap_finding is recommendations only)."""
    backend = FakeLLMBackend(
        response=canned_envelope_json(answer_text="should never dispatch", cited_markers=[1])
    )
    compose._BACKEND_OVERRIDE = backend
    env, _ = answer("What research gaps remain in the corpus?", project,
                 config=make_answer_config())
    assert backend.calls == []  # gap_finding never calls the LLM
    assert "gap_finding" in env.warnings
    assert any(r.work_id == "work_c" and r.status == "metadata_only"
               for r in env.recommendations)
    assert env.answer_text == ""
    assert env.cited_work_ids == []


def test_citation_search_protocol_surfaces_graph_neighbors_as_recommendations(project):
    """A citation_search question runs step-7 graph traversal (``cited_by``/
    ``cites_of``) and surfaces the cited, span-less neighbour work as a graph-derived
    recommendation (distinct reason proves ``traverse`` is the source, §6.1)."""
    compose._BACKEND_OVERRIDE = FakeLLMBackend(
        response=canned_envelope_json(answer_text="Paper A cites foundational work [E1].",
                                      cited_markers=[1])
    )
    env, _ = answer("Which works are cited by Paper A?", project, config=make_answer_config())
    graph_rec = next((r for r in env.recommendations if r.work_id == "work_c"), None)
    assert graph_rec is not None
    assert "citation graph" in graph_rec.reason  # sourced from traverse, not the sweep
    assert "work_c" not in env.cited_work_ids     # a recommendation, never a citation


# --- CLI `ask` + FastAPI read view (decision 81; §6 / §11 milestone) --------

def test_cli_ask_no_llm_smoke(project):
    """``seedgraph ask --no-llm`` runs end to end (exit 0) and prints a real, traceable
    citation — the decision-81 complete surface guarded by CI (§11 milestone)."""
    from typer.testing import CliRunner

    from seedgraph.cli import app as cli_app

    result = CliRunner().invoke(
        cli_app,
        ["ask", 'According to Paper A, what holds "across groups"?',
         "--project", "ph8_main", "--no-llm"],
    )
    assert result.exit_code == 0, result.output
    assert "work_a" in result.output           # a real citation surfaced
    assert "retrieval-only" in result.output    # honest no-LLM degrade banner


def test_cli_ask_no_evidence_prints_insufficient_banner(project):
    """``seedgraph ask`` on a no-evidence question prints the insufficient-evidence
    banner and no fabricated prose (exit 0)."""
    from typer.testing import CliRunner

    from seedgraph.cli import app as cli_app

    result = CliRunner().invoke(
        cli_app, ["ask", "What is the capital of France?", "--project", "ph8_main", "--no-llm"]
    )
    assert result.exit_code == 0, result.output
    assert "insufficient evidence" in result.output


def test_web_answer_route_returns_envelope(project):
    """The FastAPI ``GET /projects/{slug}/answer`` read view returns the envelope as
    JSON via an in-process ``answer()`` call (decision 81 thin read view, §6 / §11)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from seedgraph.web.routes import router
    from seedgraph.web.serve import require_local_session

    api = FastAPI()
    api.include_router(router)
    # /answer is a private-evidence GET behind require_local_session (cross-cutting
    # #1); exercise it via the dependency-override seam (not TestClient loopback).
    api.dependency_overrides[require_local_session] = lambda: None
    client = TestClient(api)
    resp = client.get(
        "/projects/ph8_main/answer",
        params={"q": 'According to Paper A, what holds "across groups"?', "no_llm": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "answer_id" in body and body["answer_id"].startswith("ans_")
    assert "work_a" in body["cited_work_ids"]
    assert body["mode"] == "retrieval_only"


def test_web_answer_route_rejects_bad_mode(project):
    """The read view validates ``mode`` (400 on an unknown value)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from seedgraph.web.routes import router
    from seedgraph.web.serve import require_local_session

    api = FastAPI()
    api.include_router(router)
    # Override the local-session gate so the request reaches the mode validation
    # (the 403 gate would otherwise fire before the 400 mode check).
    api.dependency_overrides[require_local_session] = lambda: None
    client = TestClient(api)
    resp = client.get("/projects/ph8_main/answer", params={"q": "x", "mode": "bogus"})
    assert resp.status_code == 400


def test_preclassify_single_work_comparative_degrades_to_factual():
    """A comparative question where only one work resolves degrades to factual (§6.4)."""
    spec = harness._preclassify('Compare "Paper A" against the field.')
    assert spec.protocol_hint == QueryType.factual


def test_preclassify_extracts_quoted_phrases():
    """Quoted substrings are extracted into ``QuerySpec.phrases`` for FTS5 phrases."""
    spec = harness._preclassify('Where does Paper A discuss "Assumption 2"?')
    assert "Assumption 2" in spec.phrases


# --- retrieval (08 §6 / §2; plan §4 empty/missing safety) ------------------

@pytest.mark.parametrize(
    "phrase,expected_span",
    [("Assumption 2", "s_a"), ("parallel trends", "s_a"), ("rank condition", "s_b")],
)
def test_retrieve_fts_exact_phrase_returns_right_span(conn, phrase, expected_span):
    """``span_fts`` exact-phrase queries return the correct span (exact-string req)."""
    spec = make_spec(phrase, phrases=[phrase])
    items = retrieve.retrieve(conn, "ph8_main", spec, 40)
    span_ids = {it.span_id for it in items if it.kind == "span"}
    assert expected_span in span_ids


def test_retrieve_relational_filters_scope_correctly(conn):
    """work / section / claim_type filters narrow the retrieved set to matching rows."""
    # work filter: 'identification' matches spans in both works -> scope to work_b.
    spec_b = make_spec("identification", phrases=["identification"], work_id="work_b")
    items_b = retrieve.retrieve(conn, "ph8_main", spec_b, 40)
    assert items_b and all(it.work_id == "work_b" for it in items_b)

    # claim_type filter narrows claims().
    spec_ct = make_spec("rank condition", phrases=["rank condition"],
                        claim_type="regularity_condition")
    claims = retrieve.claims(conn, spec_ct, 40)
    assert claims and all(it.claim_id == "c_b" for it in claims)
    spec_ct_none = make_spec("rank condition", phrases=["rank condition"],
                             claim_type="identification_assumption")
    assert retrieve.claims(conn, spec_ct_none, 40) == []

    # section filter narrows spans(): s_a is in a 'body' section, not 'references'.
    spec_sec = make_spec("parallel trends", phrases=["parallel trends"], section="body")
    assert any(it.span_id == "s_a" for it in retrieve.spans(conn, spec_sec, 40))
    spec_ref = make_spec("parallel trends", phrases=["parallel trends"], section="references")
    assert retrieve.spans(conn, spec_ref, 40) == []


def test_retrieve_access_class_respected(conn):
    """The ``access_class`` filter is applied at the SQL boundary (30/60/76)."""
    spec_open = make_spec("identification", phrases=["identification"],
                          access_class="open_access")
    open_items = retrieve.retrieve(conn, "ph8_main", spec_open, 40)
    assert open_items
    assert all(it.access_class == "open_access" for it in open_items)
    assert "s_priv" not in {it.item_id for it in open_items}

    spec_priv = make_spec("hidden data", phrases=["hidden data"],
                          access_class="user_supplied_private")
    priv = retrieve.spans(conn, spec_priv, 40)
    assert {it.span_id for it in priv} == {"s_priv"}


def test_retrieve_empty_fts_returns_empty_without_raising():
    """An ingested-but-unextracted project (empty FTS) yields ``[]`` (plan §4)."""
    h = empty_fts_project("ph8_emptyfts")
    c = sqlite3.connect(str(h.db_path))
    try:
        spec = make_spec("parallel trends", phrases=["parallel trends"])
        assert retrieve.retrieve(c, "ph8_emptyfts", spec, 40) == []
    finally:
        c.close()


def test_retrieve_missing_fts_table_returns_empty_without_raising(conn):
    """A missing FTS table (``no such table``) maps to ``[]`` identically (plan §4)."""
    for table in ("span_fts", "claim_fts", "note_fts"):
        conn.execute(f"DROP TABLE {table}")
    conn.commit()
    spec = make_spec("parallel trends", phrases=["parallel trends"])
    assert retrieve.retrieve(conn, "ph8_main", spec, 40) == []


def test_retrieve_scoped_to_single_project_db(conn):
    """Every retrieval query is scoped to the one project's ``project.db`` (09 §10)."""
    other = build_fixture_project("ph8_other")
    # add a uniquely-named span to the OTHER project.
    oc = sqlite3.connect(str(other.db_path))
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    oc.execute(
        "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, source_file_id, "
        "source_file_hash, work_id, start_char, end_char, exact_quote, quote_hash, "
        "access_class, created_at) VALUES ('s_unique','md_work_a','h','sf','sfh','work_a',"
        "0,30,'zzzunique marker phrase','qh_unique','open_access',?)",
        (now,),
    )
    oc.execute(
        "INSERT INTO span_fts(quote_text, span_id, markdown_id, work_id, section_id) "
        "VALUES ('zzzunique marker phrase','s_unique','md_work_a','work_a',NULL)"
    )
    oc.commit()
    oc.close()
    # Querying the FIRST project for the OTHER project's unique phrase returns nothing.
    spec = make_spec("zzzunique marker phrase", phrases=["zzzunique marker phrase"])
    assert retrieve.retrieve(conn, "ph8_main", spec, 40) == []


# --- deterministic rank (08 §5 step 8; decisions 57/82) --------------------

def _item(item_id, work_id, text, *, claim_id=None, kind="span", bm25=-2.0):
    return RetrievedItem(
        item_id=item_id, kind=kind, work_id=work_id,
        span_id=item_id if kind == "span" else None, claim_id=claim_id,
        text=text, bm25_score=bm25, access_class="open_access",
    )


def test_rank_deterministic_ordering_is_stable_and_reproducible():
    """``deterministic_rank`` produces the same ordering across runs (no randomness)."""
    items = [
        _item("s1", "work_a", "parallel trends hold here", claim_id="c1"),
        _item("s2", "work_b", "the rank condition", bm25=-1.0),
        _item("s3", "work_c", "unrelated text", bm25=-0.5),
    ]
    spec = make_spec("q", phrases=["parallel trends"])
    order1 = [c.item.item_id for c in rank.deterministic_rank(items, spec)]
    order2 = [c.item.item_id for c in rank.deterministic_rank(list(items), spec)]
    assert order1 == order2


def test_rank_boosts_applied_as_specified():
    """Each boost shifts rank as specified and is recorded in ``RankedCandidate.boosts``."""
    items = [
        _item("s1", "work_a", "parallel trends hold here", claim_id="c1"),
        _item("s2", "work_b", "completely unrelated content"),
    ]
    spec = make_spec("q", phrases=["parallel trends"])
    ranked = rank.deterministic_rank(items, spec)
    top = ranked[0]
    assert top.item.item_id == "s1"  # exact phrase + claim link win
    assert "exact_phrase" in top.boosts
    assert "span_claim_link" in top.boosts
    assert "project_membership" in top.boosts
    assert "exact_phrase" not in ranked[1].boosts


def test_rank_fusion_is_passthrough():
    """``rank_fusion`` returns its input order unchanged (the deferred-vector seam)."""
    items = [_item("s1", "work_a", "x"), _item("s2", "work_b", "y")]
    ranked = rank.deterministic_rank(items, make_spec("q"))
    fused = rank.rank_fusion(ranked)
    assert fused is ranked
    assert [c.item.item_id for c in fused] == [c.item.item_id for c in ranked]


# --- recommendations (08 §12) ----------------------------------------------

def test_metadata_only_work_surfaced_as_recommendation_not_citation(project):
    """A factual query surfaces the metadata_only fixture work as a ``Recommendation``
    (status=metadata_only) and never as a ``Citation`` (08 §12)."""
    compose._BACKEND_OVERRIDE = FakeLLMBackend(
        response=canned_envelope_json(answer_text="It holds across groups [E1].",
                                      cited_markers=[1])
    )
    env, _ = answer('According to Paper A, what holds "across groups"?', project,
                 config=make_answer_config())
    assert any(r.work_id == "work_c" and r.status == "metadata_only"
               for r in env.recommendations)
    assert "work_c" not in env.cited_work_ids


# --- --save serialization (plan §4) ----------------------------------------

def test_save_adhoc_writes_projects_answers_path(project):
    """An ad-hoc save writes ``projects/{slug}/answers/{answer_id}.json``."""
    env, _ = answer("anything", project, no_llm=True, config=make_answer_config())
    path = save_answer(env, slug="ph8_main", root=project.root)
    assert path.exists()
    assert path.parent == project.root / "projects" / "ph8_main" / "answers"
    assert path.name == f"{env.answer_id}.json"


def test_save_within_run_writes_runs_answers_path(project):
    """A save with ``run_id`` writes under ``runs/{run_id}/answers/`` (plan §4)."""
    env, _ = answer("anything", project, no_llm=True, config=make_answer_config())
    path = save_answer(env, slug="ph8_main", root=project.root, run_id="run_abc")
    assert path.exists()
    assert path.parent == project.root / "projects" / "ph8_main" / "runs" / "run_abc" / "answers"


def test_answer_id_matches_prefixed_uuid_format(project):
    """``answer_id`` matches ``^ans_[0-9a-f]{32}$`` (decisions 29/44/65)."""
    env, _ = answer("anything", project, no_llm=True, config=make_answer_config())
    assert ANSWER_ID_RE.match(env.answer_id)


# --- milestone (doc 10 §12 + doc 09 §12), fully offline --------------------

def test_milestone_ask_returns_prose_with_resolving_citations(project, conn):
    """End-to-end ``answer()`` returns prose whose every citation resolves to a real
    project ``work_id`` + inspectable ``span_id`` (doc 10 §12 milestone)."""
    compose._BACKEND_OVERRIDE = FakeLLMBackend(
        response=canned_envelope_json(answer_text="It holds across groups [E1].",
                                      cited_markers=[1])
    )
    env, _ = answer('According to Paper A, what holds "across groups"?', project,
                 config=make_answer_config())
    assert env.answer_text
    assert env.cited_span_ids and env.cited_work_ids
    for work_id in env.cited_work_ids:
        assert conn.execute("SELECT 1 FROM works WHERE work_id=?", (work_id,)).fetchone()
    for span_id in env.cited_span_ids:
        row = conn.execute(
            "SELECT exact_quote FROM evidence_spans WHERE span_id=?", (span_id,)
        ).fetchone()
        assert row is not None  # the span is inspectable (verbatim quote)
    # the cited quote IS the stored verbatim span.
    assert env.citations[0].quote == conn.execute(
        "SELECT exact_quote FROM evidence_spans WHERE span_id=?",
        (env.cited_span_ids[0],),
    ).fetchone()[0]


def test_milestone_no_evidence_question_returns_not_found_banner(project):
    """A no-evidence question returns the not-found banner rather than fabricated
    prose (doc 09 §12)."""
    compose._BACKEND_OVERRIDE = FakeLLMBackend(
        response=canned_envelope_json(answer_text="x", cited_markers=[1])
    )
    env, _ = answer("What is the capital of France?", project, config=make_answer_config())
    assert env.insufficient_evidence is True
    assert env.answer_category == AnswerCategory.unresolved


def test_milestone_empty_fts_project_returns_insufficient_evidence_cleanly():
    """An ingested-but-unextracted (empty-FTS) project returns ``insufficient_evidence``
    cleanly with ``empty_corpus`` and NO LLM call (plan §4)."""
    h = empty_fts_project("ph8_empty_milestone")
    backend = FakeLLMBackend(response=canned_envelope_json(answer_text="x", cited_markers=[1]))
    compose._BACKEND_OVERRIDE = backend
    env, _ = answer("parallel trends?", h, config=make_answer_config())
    assert env.insufficient_evidence is True
    assert "empty_corpus" in env.warnings
    assert backend.calls == []  # no dispatch on a clean short-circuit
