"""AnswerTrace — contract + assembly + persistence + exact instrumentation
(trace_plans 00/01, chunks 0-1).

Fully offline against the phase_8 fixture corpus and the fake LLM backend. Covers:
envelope byte-parity across every path (C9, criterion 1) against committed snapshots;
the trace lands beside the envelope with a matching ``answer_id`` on every persisting
surface (harness/CLI/web); path+outcome coverage; QuerySpec round-trip; neighborhood
only for citation_search; determinism modulo ``answer_id`` / ``created_at`` / paths; and
the chunk-1 exact markers (``dispositions_exact=True``, ``shown_evidence`` + ``prompt_*``
populated on the LLM path, ``cut_budget`` split from ``cut_rank`` by the budget break).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _phase8_helpers import (
    build_fixture_project,
    canned_envelope_json,
    empty_fts_project,
    make_answer_config,
)
from _trace_cases import normalize_envelope, parity_cases

from seedgraph import __version__
from seedgraph.answer import AnswerTrace, answer, compose, rank, save_answer, save_trace
from seedgraph.answer.trace import TEXT_PREVIEW_CHARS
from seedgraph.answer.types import QuerySpec, QueryType
from seedgraph.llm.backend import FakeLLMBackend

_FIX = Path(__file__).parent / "fixtures" / "trace_parity"
_PATHS = ["retrieval_only", "llm", "abstain_empty", "gap_finding", "comparative", "citation_search"]

GROUNDED_Q = 'According to Paper A, what holds "across groups"?'


@pytest.fixture(autouse=True)
def _clear_backend():
    compose._BACKEND_OVERRIDE = None
    yield
    compose._BACKEND_OVERRIDE = None


@pytest.fixture
def project():
    return build_fixture_project("trace_main")


def _grounded_backend():
    compose._BACKEND_OVERRIDE = FakeLLMBackend(
        response=canned_envelope_json(answer_text="It holds across groups [E1].", cited_markers=[1])
    )


def _answers_dir(root, slug: str) -> Path:
    return Path(root) / "projects" / slug / "answers"


# --- C9 envelope byte-parity (criterion 1) ---------------------------------

@pytest.fixture
def parity_envelopes(project):
    """Every path's envelope, recomputed post-change through the tuple return."""
    return dict(parity_cases(lambda q, **kw: answer(q, project, **kw)[0]))


@pytest.mark.parametrize("name", _PATHS)
def test_envelope_parity(parity_envelopes, name):
    """The instrument does not perturb the measurement (C9): the post-trace envelope is
    field-identical (modulo the random ``answer_id``) to the pre-trace snapshot."""
    expected = json.loads((_FIX / f"{name}.json").read_text(encoding="utf-8"))
    got = normalize_envelope(parity_envelopes[name].model_dump(mode="json"))
    assert got == expected, f"envelope drift on the {name} path"


# --- path / outcome coverage (C6) ------------------------------------------

def test_trace_gap_finding_path(project):
    """gap_finding returns before retrieval: no candidates, no neighborhood."""
    env, tr = answer("What research gaps remain in the corpus?", project, config=make_answer_config())
    assert (tr.path, tr.outcome) == ("gap_finding", "abstain_gap_finding")
    assert tr.candidates == []
    assert tr.neighborhood is None
    assert tr.retrieved_count == 0
    assert tr.answer_id == env.answer_id
    AnswerTrace.model_validate(tr.model_dump())  # validates whole


def test_trace_abstain_empty_path(project):
    """A no-evidence question short-circuits: standard path, abstain_empty outcome."""
    env, tr = answer("What is the capital of France?", project, config=make_answer_config())
    assert (tr.path, tr.outcome) == ("standard", "abstain_empty")
    assert tr.answer_id == env.answer_id
    assert env.insufficient_evidence is True


def test_trace_retrieval_only_path(project):
    """A grounded no_llm ask is standard/retrieval_only with all candidates shown."""
    env, tr = answer(GROUNDED_Q, project, no_llm=True, config=make_answer_config())
    assert (tr.path, tr.outcome) == ("standard", "retrieval_only")
    assert tr.candidates and all(c.disposition == "shown" for c in tr.candidates)


def test_trace_llm_path(project):
    """The full LLM path is standard/llm; the shown candidate ties to the allowed set."""
    _grounded_backend()
    env, tr = answer(GROUNDED_Q, project, config=make_answer_config())
    assert (tr.path, tr.outcome) == ("standard", "llm")
    shown = [c for c in tr.candidates if c.disposition == "shown"]
    assert shown, "the LLM path must show at least one candidate"
    assert all(c.work_id in env.retrieved_item_ids for c in shown)


def test_trace_comparative_path(project):
    """A two-work comparative ask records path=comparative."""
    _, tr = answer('Compare "Paper A" and "Paper B" on identification.', project,
                   no_llm=True, config=make_answer_config())
    assert tr.path == "comparative"


# --- neighborhood: citation_search only (C4/T7) ----------------------------

def test_neighborhood_present_only_for_citation_search(project):
    """The neighborhood is captured for a citation_search ask and absent otherwise."""
    _, tr_cite = answer("Which works are cited by Paper A?", project, no_llm=True,
                        config=make_answer_config())
    assert tr_cite.neighborhood is not None
    # work_a's cited, span-less neighbour work_c enters the depth-1 neighborhood.
    assert "work_c" in tr_cite.neighborhood.nodes
    assert tr_cite.neighborhood.depth == 1

    _, tr_fact = answer(GROUNDED_Q, project, no_llm=True, config=make_answer_config())
    assert tr_fact.neighborhood is None

    _, tr_gap = answer("What research gaps remain in the corpus?", project,
                       config=make_answer_config())
    assert tr_gap.neighborhood is None


# --- QuerySpec round-trip (C3) ---------------------------------------------

def test_spec_round_trips_whole(project):
    """``trace.spec`` reconstructs the exact deterministic QuerySpec (C3) — this is
    where the protocol_hint vs LLM-declared query_type divergence becomes visible."""
    q = 'Compare "Paper A" and "Paper B" on identification.'
    _, tr = answer(q, project, no_llm=True, config=make_answer_config())
    spec = QuerySpec(**tr.spec)
    assert spec.protocol_hint == QueryType.comparative
    assert spec.work_tokens == ["Paper A", "Paper B"]
    assert spec.model_dump(mode="json") == tr.spec  # whole round-trip


# --- determinism modulo answer_id / created_at / paths ---------------------

def test_trace_determinism_two_asks(project):
    """Two no_llm asks of the same question differ only in ``answer_id`` /
    ``created_at`` (and the file paths callers choose)."""
    _, t1 = answer(GROUNDED_Q, project, no_llm=True, config=make_answer_config())
    _, t2 = answer(GROUNDED_Q, project, no_llm=True, config=make_answer_config())
    d1, d2 = t1.model_dump(mode="json"), t2.model_dump(mode="json")
    for d in (d1, d2):
        d.pop("answer_id")
        d.pop("created_at")
    assert d1 == d2


# --- exact markers + self-describing scoring (C5) ---------------------------

def test_exact_markers_and_self_description(project):
    """Chunk 1: the LLM-path trace is exact (dispositions + shown-list + prompt hash) and
    still self-describing (version + boost constants)."""
    _grounded_backend()
    _, tr = answer(GROUNDED_Q, project, config=make_answer_config())
    assert tr.trace_version == 1
    assert tr.dispositions_exact is True
    assert tr.shown_evidence is not None
    assert tr.prompt_version == "answer_v1"
    assert tr.prompt_sha256 is not None and len(tr.prompt_sha256) == 64
    assert tr.seedgraph_version == __version__
    assert tr.boost_constants == rank.boost_constants()
    # every ranked candidate carries its verbatim boost breakdown + a bounded preview.
    for c in tr.candidates:
        assert "project_membership" in c.boosts
        assert len(c.text_preview) <= 161  # ~160 chars + the ellipsis


# --- chunk 1: exact dispositions + shown evidence (criterion 3) -------------

def test_llm_shown_evidence_joins_candidates_1to1(project):
    """The shown candidates join 1:1 with ``shown_evidence`` (same item_ids), and the
    [E{n}] markers are the contiguous 1..k the envelope citations reference (T3)."""
    _grounded_backend()
    _, tr = answer(GROUNDED_Q, project, config=make_answer_config())
    shown_ids = {c.item_id for c in tr.candidates if c.disposition == "shown"}
    assert shown_ids and shown_ids == {se.item_id for se in tr.shown_evidence}
    assert [se.n for se in tr.shown_evidence] == list(range(1, len(tr.shown_evidence) + 1))


def test_budget_cut_disposition_is_exact(project):
    """A token cap that fits only the top candidate tags the first ``shown`` and every
    other ranked candidate ``cut_budget`` (never ``cut_rank``) — read from
    ``build_prompt``'s break index, and ``shown_evidence`` joins the shown set 1:1."""
    _grounded_backend()
    cfg = make_answer_config()
    cfg.answer.max_evidence_tokens = 1  # only the first candidate fits (always-shown edge)
    # A question that ranks >= 2 candidates so a budget cut is non-vacuous.
    _, tr = answer("parallel trends and rank condition", project, config=cfg)
    assert len(tr.candidates) >= 2
    shown = [c for c in tr.candidates if c.disposition == "shown"]
    cut_budget = [c for c in tr.candidates if c.disposition == "cut_budget"]
    assert len(shown) == 1 and shown[0].rank_position == 1
    assert len(cut_budget) == len(tr.candidates) - 1
    assert all(c.disposition != "cut_rank" for c in tr.candidates)
    # the disposition split matches the report the shown-list was built from.
    assert {c.item_id for c in shown} == {se.item_id for se in tr.shown_evidence}
    assert tr.dispositions_exact is True


def test_degrade_site_reasons_match_envelope(project):
    """Each degrade site records ``degrade_reasons`` mirroring the warning it appends to
    the envelope (00 §3): a budget block, and a dispatch failure (a prompt WAS built, so
    the failed-dispatch trace still carries shown_evidence + the prompt hash)."""
    # Pricing gate: local_ollama_default (llama3) pricing unverified + a USD limit.
    env_b, tr_b = answer(GROUNDED_Q, project,
                         config=make_answer_config(usd_limit=5.0))
    assert tr_b.outcome == "retrieval_only"
    assert tr_b.degrade_reasons == ["budget_exceeded"]
    assert "budget_exceeded" in env_b.warnings
    assert tr_b.shown_evidence is None  # degraded before a prompt was built

    # Failed dispatch: two unrepairable JSON replies -> compose_failed, but a prompt was
    # built + dispatched, so the trace records what was shown + the hash.
    compose._BACKEND_OVERRIDE = FakeLLMBackend(responses=["not json", "still not json"])
    env_d, tr_d = answer(GROUNDED_Q, project, config=make_answer_config())
    assert tr_d.outcome == "retrieval_only"
    assert tr_d.degrade_reasons == ["compose_failed"]
    assert "compose_failed" in env_d.warnings
    assert tr_d.shown_evidence is not None
    assert tr_d.prompt_version == "answer_v1"
    assert tr_d.prompt_sha256 is not None


# --- rank truncation recorded as cut_rank (design C2 / 00 §1) ----------------
#
# NOTE: the standard path can never overflow the ``[:max_candidates]`` slice —
# ``retrieve.retrieve`` already caps its merge at ``limit == max_candidates``
# (retrieve.py:345), and ``deterministic_rank`` is one-per-item, so ``ranked_full`` has
# ``<= max_candidates`` rows there. Over-fetching for the trace would change which rows
# survive the slice and break envelope parity (C9). So ``cut_rank`` is exercised two ways:
# a DIRECT unit test of the disposition table (real >160-char truncation, decoupled from
# retrieval), and end-to-end via the COMPARATIVE partitions — the only branch whose
# ``per = max(1, limit // num_works)`` floor returns more rows than the slice keeps.

def test_trace_candidates_tags_rank_truncated_rows_cut_rank():
    """The one disposition table (C2): every candidate past the ``max_candidates`` slice is
    ``cut_rank`` with its scores/boosts intact and text truncated to
    :data:`TEXT_PREVIEW_CHARS`; a survivor off the prompt path (``cut_index is None``) is
    ``shown``. Direct unit test — decoupled from retrieval, with a genuinely long text the
    fixture corpus can't provide."""
    from seedgraph.answer.harness import _trace_candidates
    from seedgraph.answer.types import RankedCandidate, RetrievedItem

    long_text = "x" * (TEXT_PREVIEW_CHARS + 50)
    ranked = [
        RankedCandidate(
            item=RetrievedItem(
                item_id=f"s{i}", kind="span", work_id=f"w{i}", span_id=f"s{i}",
                text=long_text, bm25_score=-float(i + 1), access_class="open_access",
            ),
            rank_score=1.0 - i * 0.1,
            boosts={"project_membership": 0.05, "exact_phrase": 0.3},
        )
        for i in range(4)
    ]
    out = _trace_candidates(ranked, cut_index=None, max_candidates=2)
    assert [c.disposition for c in out] == ["shown", "shown", "cut_rank", "cut_rank"]
    by_pos = {c.rank_position: c for c in out}
    for pos in (3, 4):  # rank-truncated rows keep full scores/boosts + a truncated preview.
        c = by_pos[pos]
        assert c.boosts == {"project_membership": 0.05, "exact_phrase": 0.3}
        assert c.rank_score > 0 and c.bm25_score < 0
        assert c.text_preview.endswith("…") and len(c.text_preview) == TEXT_PREVIEW_CHARS + 1


def test_trace_candidates_three_dispositions_coexist():
    """With a prompt built (``cut_index`` set) the survivors split ``shown``/``cut_budget``
    by the break index while the overflow is still ``cut_rank`` — all three mechanisms in
    one table, positionally, never inferred from AllowedSet membership."""
    from seedgraph.answer.harness import _trace_candidates
    from seedgraph.answer.types import RankedCandidate, RetrievedItem

    ranked = [
        RankedCandidate(
            item=RetrievedItem(item_id=f"s{i}", kind="span", work_id="w",
                               text="short", bm25_score=-1.0, access_class="open_access"),
            rank_score=1.0, boosts={},
        )
        for i in range(5)
    ]
    out = _trace_candidates(ranked, cut_index=1, max_candidates=3)
    assert [c.disposition for c in out] == [
        "shown", "cut_budget", "cut_budget", "cut_rank", "cut_rank"
    ]


def test_comparative_overfill_shows_dropped_named_work_candidate(project):
    """Two named works with a small ``max_candidates`` overflow the slice (one partition
    per work): a dropped candidate from a named work now surfaces as ``cut_rank`` — the
    comparative-overfill case from the review."""
    _, tr = answer('Compare "Paper A" and "Paper B" on identification.', project,
                   no_llm=True, max_candidates=1, config=make_answer_config())
    assert tr.path == "comparative"
    assert len(tr.candidates) >= 2  # one partition per named work overflows max_candidates=1
    shown = [c for c in tr.candidates if c.disposition == "shown"]
    cut_rank = [c for c in tr.candidates if c.disposition == "cut_rank"]
    assert len(shown) == 1 and shown[0].rank_position == 1
    assert cut_rank and all(c.rank_position > 1 for c in cut_rank)
    # the dropped candidate belongs to one of the compared works, scores/boosts intact.
    assert all(c.work_id in {"work_a", "work_b"} for c in cut_rank)
    assert all("project_membership" in c.boosts for c in cut_rank)


def test_abstain_path_cut_rank_only_for_rank_overflow(project):
    """On the abstain path ``cut_rank`` now means ONLY rank truncation, never "nothing
    shown": a comparative abstain whose partitions overflow the slice records the survivor
    ``shown`` and only the sliced-off partition ``cut_rank`` (design C2; review Spec
    finding). "Nothing reached a model" is recorded by ``shown_evidence=None`` +
    ``outcome="abstain_empty"``, not by the disposition."""
    cfg = make_answer_config()
    cfg.answer.absent_floor = 100.0  # force the empty/weak short-circuit WITH candidates present
    env, tr = answer('Compare "Paper A" and "Paper B" on identification.', project,
                     no_llm=True, max_candidates=1, config=cfg)
    assert (tr.path, tr.outcome) == ("comparative", "abstain_empty")
    assert env.insufficient_evidence is True
    assert tr.shown_evidence is None
    assert len(tr.candidates) >= 2  # the two named-work partitions overflow max_candidates=1
    shown = [c for c in tr.candidates if c.disposition == "shown"]
    cut_rank = [c for c in tr.candidates if c.disposition == "cut_rank"]
    assert len(shown) == 1 and shown[0].rank_position == 1  # survivor shown, not blanket cut_rank
    assert cut_rank and all(c.rank_position > 1 for c in cut_rank)
    assert all(c.disposition != "cut_budget" for c in tr.candidates)  # no prompt was built


def test_abstain_survivors_shown_without_overflow(project):
    """When abstain's ranked list fits under ``max_candidates`` (no overflow), EVERY
    candidate is ``shown`` — the clearest contrast with the old blanket-``cut_rank``."""
    cfg = make_answer_config()
    cfg.answer.absent_floor = 100.0  # force abstain with candidates present, default max_candidates
    env, tr = answer(GROUNDED_Q, project, no_llm=True, config=cfg)
    assert (tr.path, tr.outcome) == ("standard", "abstain_empty")
    assert tr.candidates, "retrieval produced candidates the abstain path must still record"
    assert all(c.disposition == "shown" for c in tr.candidates)


# --- persistence: trace lands beside envelope, matching answer_id (00 §5) ----

def test_save_trace_adhoc_beside_envelope(project):
    """``save_trace`` writes ``answers/{answer_id}.trace.json`` beside the envelope."""
    env, tr = answer("anything", project, no_llm=True, config=make_answer_config())
    apath = save_answer(env, slug="trace_main", root=project.root)
    tpath = save_trace(tr, slug="trace_main", root=project.root)
    assert tpath.exists()
    assert tpath.parent == apath.parent
    assert tpath.name == f"{env.answer_id}.trace.json"
    assert tr.answer_id == env.answer_id
    # the file round-trips back through the model.
    assert AnswerTrace.model_validate_json(tpath.read_text(encoding="utf-8")).answer_id == env.answer_id


def test_save_trace_within_run_nested(project):
    """A run-scoped save nests under ``runs/{run_id}/answers/`` exactly like the envelope."""
    _, tr = answer("anything", project, no_llm=True, config=make_answer_config())
    tpath = save_trace(tr, slug="trace_main", root=project.root, run_id="run_z")
    assert tpath.parent == project.root / "projects" / "trace_main" / "runs" / "run_z" / "answers"
    assert tpath.name == f"{tr.answer_id}.trace.json"


def test_cli_ask_persists_envelope_and_trace():
    """``seedgraph ask`` always persists the pair and echoes both paths (00 §5)."""
    from typer.testing import CliRunner

    from seedgraph.cli import app as cli_app

    project = build_fixture_project("trace_cli")
    result = CliRunner().invoke(
        cli_app, ["ask", GROUNDED_Q, "--project", "trace_cli", "--no-llm"]
    )
    assert result.exit_code == 0, result.output
    assert result.output.count("saved ") >= 2  # envelope + trace paths echoed
    answers = _answers_dir(project.root, "trace_cli")
    traces = list(answers.glob("ans_*.trace.json"))
    envs = [p for p in answers.glob("ans_*.json") if not p.name.endswith(".trace.json")]
    assert len(envs) == 1 and len(traces) == 1
    assert traces[0].name[: -len(".trace.json")] == envs[0].stem  # matching answer_id


def test_web_get_ask_persists_envelope_and_trace():
    """The web GET-with-q render persists envelope + trace before rendering (00 §5)."""
    from fastapi.testclient import TestClient

    from seedgraph.api.app import app
    from seedgraph.web import serve

    project = build_fixture_project("trace_web")
    app.dependency_overrides[serve.require_local_session] = lambda: None
    try:
        client = TestClient(app)
        resp = client.get("/ui/projects/trace_web/ask", params={"q": "across groups"})
        assert resp.status_code == 200, resp.text
    finally:
        app.dependency_overrides.pop(serve.require_local_session, None)
    answers = _answers_dir(project.root, "trace_web")
    traces = list(answers.glob("ans_*.trace.json"))
    envs = [p for p in answers.glob("ans_*.json") if not p.name.endswith(".trace.json")]
    assert len(envs) == 1 and len(traces) == 1
    assert traces[0].name[: -len(".trace.json")] == envs[0].stem


# --- empty-corpus abstain (a distinct sub-path of abstain_empty) ------------

def test_empty_corpus_abstain_trace():
    """An ingested-but-unextracted project yields an abstain_empty trace with no
    candidates and the empty_corpus warning on the envelope."""
    h = empty_fts_project("trace_empty")
    env, tr = answer("parallel trends?", h, config=make_answer_config())
    assert (tr.path, tr.outcome) == ("standard", "abstain_empty")
    assert tr.candidates == []
    assert tr.retrieved_count == 0
    assert "empty_corpus" in env.warnings
    assert env.insufficient_evidence is True
