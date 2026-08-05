"""Phase 8 — compose acceptance tests (plan §11 `compose` group; 08 §5 steps
6/9/10, decisions 82/53/15/D9).

Fully offline: the LLM seam is the phase_4 ``FakeLLMBackend`` injected through
``compose._BACKEND_OVERRIDE`` so number→id mapping, token budgeting, and usage
logging are exercised without a key.
"""

from __future__ import annotations

import sqlite3

import pytest
from _phase8_helpers import (
    build_fixture_project,
    canned_envelope_json,
    make_answer_config,
    make_spec,
)

from seedgraph.answer import compose
from seedgraph.answer.compose import (  # noqa: F401
    build_prompt,
    build_recommendations,
    expand_context,
    generate,
    materialize_citations,
)
from seedgraph.answer.types import AnswerMode, QueryType, RankedCandidate, RetrievedItem
from seedgraph.llm.backend import FakeLLMBackend


@pytest.fixture
def project():
    return build_fixture_project("ph8_compose")


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


def _candidate(span_id, work_id, text, *, claim_id=None, epistemic="llm_extracted",
               assertion="stated", access="open_access", score=0.9) -> RankedCandidate:
    item = RetrievedItem(
        item_id=span_id, kind="span", work_id=work_id, span_id=span_id, claim_id=claim_id,
        text=text, bm25_score=-3.0, access_class=access, epistemic_type=epistemic,
        assertion_status=assertion,
    )
    return RankedCandidate(item=item, rank_score=score, boosts={})


# --- number→id mapping (plan §6.3) -----------------------------------------

def test_number_to_id_mapping_resolves_markers_to_correct_span_and_work(conn):
    """``materialize_citations`` maps each ``[E{n}]`` marker through ``evidence_index``
    to the correct span/work ids (plan §6.3)."""
    cands = [
        _candidate("s_a", "work_a", "We invoke Assumption 2: parallel trends hold across groups."),
        _candidate("s_b", "work_b", "The rank condition ensures point identification."),
    ]
    _prompt, evidence_index, _allowed, _shown = build_prompt(cands, make_spec("q"))
    assert evidence_index[1] == ("work_a", ["s_a"])
    assert evidence_index[2] == ("work_b", ["s_b"])

    citations, warnings = materialize_citations([2], evidence_index, conn)
    assert warnings == []
    assert len(citations) == 1
    assert citations[0].work_id == "work_b"
    assert citations[0].span_ids == ["s_b"]


def test_out_of_range_or_invented_marker_is_dropped_and_warned(conn):
    """A cited marker not a key in ``evidence_index`` (out-of-range / invented) is
    dropped at mapping time and recorded as a warning (plan §6.3)."""
    cands = [_candidate("s_a", "work_a", "Assumption 2 holds.")]
    _p, evidence_index, _a, _s = build_prompt(cands, make_spec("q"))
    citations, warnings = materialize_citations([1, 99], evidence_index, conn)
    assert [c.work_id for c in citations] == ["work_a"]
    assert any("99" in w for w in warnings)
    assert all(c.work_id != "work_99" for c in citations)


def test_citation_quote_equals_stored_exact_quote(conn):
    """``Citation.quote`` equals the verbatim ``evidence_spans.exact_quote`` (53)."""
    stored = conn.execute(
        "SELECT exact_quote FROM evidence_spans WHERE span_id='s_a'"
    ).fetchone()[0]
    cands = [_candidate("s_a", "work_a", stored)]
    _p, evidence_index, _a, _s = build_prompt(cands, make_spec("q"))
    citations, _w = materialize_citations([1], evidence_index, conn)
    assert citations[0].quote == stored


def test_epistemic_type_and_assertion_status_propagate_onto_citation(conn):
    """Both D2 provenance fields propagate from ``extracted_claims`` onto the
    ``Citation`` (origin + stated/inferred)."""
    cands = [_candidate("s_a", "work_a", "Assumption 2 holds.", claim_id="c_a")]
    _p, evidence_index, _a, _s = build_prompt(cands, make_spec("q"))
    citations, _w = materialize_citations([1], evidence_index, conn)
    assert citations[0].epistemic_type == "llm_extracted"
    assert citations[0].assertion_status == "stated"


def test_context_expansion_falls_back_and_warns_on_cache_open_failure(conn):
    """``expand_context`` falls back to the stored ``exact_quote`` + ``stale_context``
    when the cache cannot be opened/read at all (``cache_root=None`` → ``open_cache_ro``
    raises → the generic ``except`` fallback) — never a hard error (24/42/53)."""
    text, warnings = expand_context(conn, None, span_id="s_a")
    assert "stale_context" in warnings
    stored = conn.execute(
        "SELECT exact_quote FROM evidence_spans WHERE span_id='s_a'"
    ).fetchone()[0]
    assert text == stored  # verbatim authority preserved


def test_context_expansion_falls_back_and_warns_on_hash_mismatch(conn, monkeypatch):
    """``expand_context`` reaches the explicit *staleness* branch directly: the cache
    opens and returns markdown whose ``markdown_hash`` no longer matches the span's
    anchor, so it falls back to the stored ``exact_quote`` + ``stale_context`` and the
    stale markdown body is NEVER surfaced (08 §5 step 6; 24/42/53). Distinct from the
    generic open-failure path above — this exercises compose's hash-mismatch guard
    (``md.markdown_hash != md_hash``), not the ``except Exception`` fallback."""
    from seedgraph import cache_access
    from seedgraph.cache_access import MarkdownRow

    class _DummyConn:
        def close(self):  # opened then closed in the finally block
            pass

    # Cache opens fine and returns markdown — but its hash differs from the span's
    # stored anchor ('h'), so the staleness guard must fire.
    monkeypatch.setattr(cache_access, "open_cache_ro", lambda root: _DummyConn())
    monkeypatch.setattr(
        cache_access,
        "read_markdown",
        lambda _conn, _root, _md_id: MarkdownRow(
            text="STALE full markdown body that must NOT be returned",
            markdown_hash="stale_hash_distinct_from_the_span_anchor",  # != span 'h'
            storage_uri="x",
            source_file_id="sf",
            source_file_hash="sfh",
            access_class="open_access",
        ),
    )
    text, warnings = expand_context(conn, "/any/cache/root", span_id="s_a")
    assert "stale_context" in warnings
    stored = conn.execute(
        "SELECT exact_quote FROM evidence_spans WHERE span_id='s_a'"
    ).fetchone()[0]
    assert text == stored  # verbatim authority preserved
    assert "STALE full markdown body" not in text  # stale cache never surfaced


# --- token budgeting (plan §8, D9) -----------------------------------------

def test_token_budget_truncation_drops_over_budget_candidates():
    """``build_prompt`` drops candidates beyond the effective cap and those items are
    absent from the allowed set — the model cannot cite what it never saw (plan §8)."""
    big = "x " * 400  # ~ a couple hundred tokens
    cands = [
        _candidate("s_a", "work_a", big),
        _candidate("s_b", "work_b", big),
        _candidate("s_c", "work_c", big),
    ]
    _p, evidence_index, allowed, _s = build_prompt(
        cands, make_spec("q"), max_evidence_tokens=120, context_window_tokens=None
    )
    # Only the first candidate fits the tiny budget.
    assert list(evidence_index) == [1]
    assert "s_a" in allowed.span_ids
    assert "s_b" not in allowed.span_ids
    assert "s_c" not in allowed.span_ids


def test_context_window_tightens_the_effective_cap():
    """A small ``context_window_tokens`` overrides ``max_evidence_tokens`` via
    ``min(...)`` (D9 — a small-window model gets a tighter budget)."""
    big = "x " * 400
    cands = [_candidate("s_a", "work_a", big), _candidate("s_b", "work_b", big)]
    _p, idx, allowed, _s = build_prompt(
        cands, make_spec("q"), max_evidence_tokens=100000,
        context_window_tokens=300, prompt_overhead_tokens=100, reserved_output_tokens=100,
    )
    assert list(idx) == [1]  # effective cap = 300-100-100 = 100 tok -> only one block
    assert "s_b" not in allowed.span_ids


def test_per_quote_fragment_truncation_keeps_fragments_bounded():
    """Each evidence quote is truncated to the max fragment length so only bounded
    fragments — never full documents — reach the model (plan §7/§8)."""
    long_quote = "A" * 5000
    cands = [_candidate("s_a", "work_a", long_quote)]
    prompt, _idx, _allowed, _s = build_prompt(cands, make_spec("q"), max_fragment_chars=100)
    assert "A" * 5000 not in prompt
    assert "A" * 100 in prompt  # truncated fragment present
    assert "…" in prompt        # truncation marker


# --- shown-evidence report + ComposeTrace (00 §4.1, chunk 1) ---------------

def test_build_prompt_shown_report_records_blocks_and_break_index():
    """``build_prompt`` reports each rendered [E{n}] block (marker, item identity, token
    estimate, truncation) plus ``cut_index`` = the token-budget break index (00 §4.1)."""
    big = "x " * 400
    cands = [
        _candidate("s_a", "work_a", big),
        _candidate("s_b", "work_b", big),
        _candidate("s_c", "work_c", big),
    ]
    _p, idx, _a, report = build_prompt(
        cands, make_spec("q"), max_evidence_tokens=120, context_window_tokens=None
    )
    # only the first candidate fits the tiny cap -> break index 1, one shown block.
    assert report.cut_index == 1
    assert [se.item_id for se in report.evidence] == ["s_a"]
    assert list(idx) == [se.n for se in report.evidence] == [1]
    block = report.evidence[0]
    assert block.n == 1 and block.item_id == "s_a" and block.work_id == "work_a"
    assert block.span_ids == ["s_a"]
    assert block.token_estimate > 0 and block.truncated is False  # 800 chars < default 1200


def test_build_prompt_first_candidate_always_shown_even_over_budget():
    """The first candidate is always rendered (the ``if blocks and`` guard), so a single
    over-cap candidate still yields exactly one shown block (00 §4.1 edge)."""
    _p, idx, allowed, report = build_prompt(
        [_candidate("s_a", "work_a", "x " * 5000)], make_spec("q"),
        max_evidence_tokens=1, context_window_tokens=None,
    )
    assert report.cut_index == 1
    assert [se.item_id for se in report.evidence] == ["s_a"]
    assert list(idx) == [1]
    assert "s_a" in allowed.span_ids


def test_build_prompt_shown_report_flags_fragment_truncation():
    """``truncated`` is set on a shown block whose quote was capped at
    ``max_fragment_chars`` (T3), and clear otherwise."""
    _p, _i, _a, report = build_prompt(
        [_candidate("s_a", "work_a", "A" * 5000)], make_spec("q"), max_fragment_chars=100
    )
    assert report.evidence[0].truncated is True
    assert report.evidence[0].quote.endswith("…") and len(report.evidence[0].quote) == 101

    _p, _i, _a, r2 = build_prompt([_candidate("s_b", "work_b", "short")], make_spec("q"))
    assert r2.evidence[0].truncated is False


def test_build_prompt_shown_report_joins_evidence_index_1to1():
    """When everything fits, every candidate is a shown block whose n/item join 1:1 with
    ``evidence_index`` (the marker map ``materialize_citations`` resolves through)."""
    cands = [
        _candidate("s_a", "work_a", "parallel trends"),
        _candidate("s_b", "work_b", "rank condition"),
        _candidate("s_c", "work_c", "identification"),
    ]
    _p, idx, _a, report = build_prompt(cands, make_spec("q"))
    assert report.cut_index == 3
    assert [se.n for se in report.evidence] == [1, 2, 3]
    assert list(idx) == [1, 2, 3]
    assert [se.item_id for se in report.evidence] == ["s_a", "s_b", "s_c"]


def test_generate_returns_compose_trace_on_llm_success(conn):
    """``generate`` returns a ComposeTrace with the shown-list, the prompt version + a
    64-hex sha256 (hash only, T3), and empty degrade_reasons on the LLM path."""
    backend = FakeLLMBackend(response=canned_envelope_json(answer_text="A [E1].", cited_markers=[1]))
    cfg = make_answer_config()
    cands = [_candidate("s_a", "work_a", "Assumption 2 holds.")]
    _env, _allowed, ct = generate(cands, make_spec("q"), cfg, project_conn=conn, backend=backend)
    assert ct is not None
    assert ct.prompt_version == "answer_v1"
    assert len(ct.prompt_sha256) == 64
    assert ct.degrade_reasons == []
    assert [se.item_id for se in ct.shown_evidence] == ["s_a"]
    assert ct.cut_index == len(ct.shown_evidence) == 1


def test_generate_compose_trace_degrade_reasons_per_site(conn, monkeypatch):
    """Each degrade site sets ``degrade_reasons`` mirroring the warning it appends, and
    leaves shown_evidence ``None`` before a prompt is built — non-``None`` on the
    failed-dispatch degrade, which did build + dispatch one (00 §3)."""
    good = canned_envelope_json(answer_text="A [E1].", cited_markers=[1])
    cands = [_candidate("s_a", "work_a", "Assumption 2 holds.")]

    # no_llm: retrieval_only, no prompt, no degrade *reason* (a deliberate mode).
    _e, _a, ct_nollm = generate(cands, make_spec("q"), make_answer_config(),
                                project_conn=conn, backend=FakeLLMBackend(response=good), no_llm=True)
    assert ct_nollm.degrade_reasons == [] and ct_nollm.shown_evidence is None

    # pricing block: llama3 unverified pricing + a USD limit arms the fail-closed gate.
    _e, _a, ct_budget = generate(cands, make_spec("q"),
                                 make_answer_config(preferred="local_ollama_default", usd_limit=5.0),
                                 project_conn=conn, backend=FakeLLMBackend(response=good))
    assert ct_budget.degrade_reasons == ["budget_exceeded"] and ct_budget.shown_evidence is None

    # private-local-only: external preferred, no local profile, external forbidden.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    from seedgraph.config.models import default_profiles
    profs = {"anthropic_api_default": default_profiles()["anthropic_api_default"],
             "no_llm": default_profiles()["no_llm"]}
    cfg_priv = make_answer_config(preferred="anthropic_api_default", private_external=False, profiles=profs)
    priv = [_candidate("s_priv", "work_a", "private", access="user_supplied_private")]
    _e, _a, ct_priv = generate(priv, make_spec("q"), cfg_priv, project_conn=conn,
                               backend=FakeLLMBackend(response=good))
    assert ct_priv.degrade_reasons == ["private_content_local_only"] and ct_priv.shown_evidence is None

    # failed dispatch: two unrepairable JSON replies -> a prompt WAS built + dispatched.
    _e, _a, ct_fail = generate(cands, make_spec("q"), make_answer_config(),
                               project_conn=conn, backend=FakeLLMBackend(responses=["bad", "bad"]))
    assert ct_fail.degrade_reasons == ["compose_failed"]
    assert [se.item_id for se in ct_fail.shown_evidence] == ["s_a"]
    assert ct_fail.prompt_version == "answer_v1" and len(ct_fail.prompt_sha256) == 64


# --- strict JSON + repair (plan §10 step 6d) -------------------------------

def test_strict_json_parse_with_one_repair_retry(conn):
    """A malformed first response triggers exactly one repair retry; success on retry
    parses cleanly."""
    good = canned_envelope_json(answer_text="Assumption 2 is parallel trends [E1].",
                                cited_markers=[1])
    backend = FakeLLMBackend(responses=["this is not json", good])
    cfg = make_answer_config()
    cands = [_candidate("s_a", "work_a", "Assumption 2 holds.")]
    env, _allowed, _ct = generate(
        cands, make_spec("q"), cfg, project_conn=conn, backend=backend
    )
    assert len(backend.calls) == 2  # one bad + one repaired
    assert env.answer_text == "Assumption 2 is parallel trends [E1]."
    assert env.cited_work_ids == ["work_a"]


def test_second_json_parse_failure_degrades_to_retrieval_only(conn):
    """A second JSON-parse failure degrades to ``retrieval_only`` with a
    ``compose_failed`` warning rather than fabricating prose."""
    backend = FakeLLMBackend(responses=["bad one", "bad two"])
    cfg = make_answer_config()
    cands = [_candidate("s_a", "work_a", "Assumption 2 holds.")]
    env, _allowed, _ct = generate(cands, make_spec("q"), cfg, project_conn=conn, backend=backend)
    assert len(backend.calls) == 2
    assert env.mode == AnswerMode.RETRIEVAL_ONLY
    assert env.answer_text == ""
    assert "compose_failed" in env.warnings


def test_retrieval_only_path_returns_ranked_cited_list_empty_answer_text(conn):
    """The ``retrieval_only`` branch (no_llm) returns the ranked, cited evidence list
    with ``answer_text=""`` and makes no LLM call (decisions 58/38)."""
    backend = FakeLLMBackend(response=canned_envelope_json(answer_text="x", cited_markers=[1]))
    cfg = make_answer_config()
    cands = [_candidate("s_a", "work_a", "Assumption 2 holds.")]
    env, _allowed, _ct = generate(
        cands, make_spec("q"), cfg, project_conn=conn, backend=backend, no_llm=True
    )
    assert backend.calls == []  # NO dispatch in no-LLM mode
    assert env.answer_text == ""
    assert env.mode == AnswerMode.RETRIEVAL_ONLY
    assert [c.work_id for c in env.citations] == ["work_a"]


# --- usage logging (plan §7/§8, doc 13 §11/§15) ----------------------------

def test_usage_event_written_with_correct_provenance(conn):
    """One ``llm_usage_events`` row per call with ``task_type=answer_generation``,
    ``external_full_text=false``, the max-restrictive ``source_access_class``, and
    ``prompt_version=answer_v1``."""
    backend = FakeLLMBackend(response=canned_envelope_json(answer_text="A [E1].",
                                                           cited_markers=[1]))
    cfg = make_answer_config()
    cands = [_candidate("s_a", "work_a", "Assumption 2 holds.", access="open_access")]
    env, _allowed, _ct = generate(cands, make_spec("q"), cfg, project_conn=conn, backend=backend)
    row = conn.execute(
        "SELECT task_type, external_full_text, source_access_class FROM llm_usage_events"
    ).fetchall()
    assert len(row) == 1
    assert row[0][0] == "answer_generation"
    assert row[0][1] == 0  # external_full_text=false (bounded fragments only)
    assert row[0][2] == "open_access"
    assert env.llm_provenance["prompt_version"] == "answer_v1"
    assert env.llm_provenance["source_text_left_machine"] is False


def test_usage_source_access_class_is_most_restrictive(conn):
    """``source_access_class`` is the max-restrictive over feeding spans (a private
    fragment makes the whole call private in the audit)."""
    backend = FakeLLMBackend(response=canned_envelope_json(answer_text="A [E1].",
                                                           cited_markers=[1]))
    # private fragment allowed external here (policy True) so the call proceeds.
    cfg = make_answer_config(private_external=True)
    cands = [
        _candidate("s_a", "work_a", "open fragment", access="open_access"),
        _candidate("s_priv", "work_a", "private fragment", access="user_supplied_private"),
    ]
    generate(cands, make_spec("q"), cfg, project_conn=conn, backend=backend)
    row = conn.execute("SELECT source_access_class FROM llm_usage_events").fetchone()
    assert row[0] == "user_supplied_private"


# --- content-access gate (plan §7) -----------------------------------------

def test_private_fragment_routes_local_or_degrades_not_external(conn, monkeypatch):
    """A private feeding fragment under a policy forbidding external answer-generation
    routes to a LOCAL profile (never an external API)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    backend = FakeLLMBackend(response=canned_envelope_json(answer_text="A [E1].",
                                                           cited_markers=[1]))
    # external preferred + a local profile present in the config -> route local.
    cfg = make_answer_config(preferred="anthropic_api_default", private_external=False)
    cands = [_candidate("s_priv", "work_a", "private fragment", access="user_supplied_private")]
    env, _allowed, _ct = generate(cands, make_spec("q"), cfg, project_conn=conn, backend=backend)
    # routed to the local fallback (offline fake), NOT the external profile.
    assert env.llm_provenance["access_mode"] == "local"
    assert env.llm_provenance["source_text_left_machine"] is False


def test_private_fragment_degrades_when_no_local_profile(conn, monkeypatch):
    """With no local profile available, a forbidden private fragment degrades to
    ``retrieval_only`` with ``private_content_local_only`` — NEVER sent external."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    from seedgraph.config.models import default_profiles

    backend = FakeLLMBackend(response=canned_envelope_json(answer_text="A [E1].",
                                                           cited_markers=[1]))
    profs = {
        "anthropic_api_default": default_profiles()["anthropic_api_default"],
        "no_llm": default_profiles()["no_llm"],
    }
    cfg = make_answer_config(preferred="anthropic_api_default", private_external=False,
                             profiles=profs)
    cands = [_candidate("s_priv", "work_a", "private fragment", access="user_supplied_private")]
    env, _allowed, _ct = generate(cands, make_spec("q"), cfg, project_conn=conn, backend=backend)
    assert backend.calls == []  # NOTHING dispatched to any backend
    assert env.mode == AnswerMode.RETRIEVAL_ONLY
    assert "private_content_local_only" in env.warnings
    assert conn.execute("SELECT COUNT(*) FROM llm_usage_events").fetchone()[0] == 0


# --- budget gate (plan §8) -------------------------------------------------

def test_budget_block_degrades_to_retrieval_only(conn):
    """An over-budget / unverified-pricing block routes through the phase-0 budget
    enforcer and degrades to ``retrieval_only`` with a ``budget_exceeded`` warning
    (fail-closed-on-unverified-pricing)."""
    backend = FakeLLMBackend(response=canned_envelope_json(answer_text="A [E1].",
                                                           cited_markers=[1]))
    # llama3 pricing_status != "verified"; a USD limit arms the fail-closed gate.
    cfg = make_answer_config(preferred="local_ollama_default", usd_limit=5.0)
    cands = [_candidate("s_a", "work_a", "Assumption 2 holds.")]
    env, _allowed, _ct = generate(cands, make_spec("q"), cfg, project_conn=conn, backend=backend)
    assert backend.calls == []  # blocked before dispatch
    assert env.mode == AnswerMode.RETRIEVAL_ONLY
    assert "budget_exceeded" in env.warnings


# --- recommendations (08 §12) ----------------------------------------------

def test_metadata_only_work_is_a_recommendation_not_a_citation(conn):
    """The metadata_only fixture work surfaces as a ``Recommendation`` and never as a
    ``Citation`` (it has no spans) — 08 §12."""
    cands = [_candidate("s_a", "work_a", "Assumption 2 holds.")]
    recs = build_recommendations(conn, make_spec("q"), cands)
    assert any(r.work_id == "work_c" and r.status == "metadata_only" for r in recs)
    assert all(r.action_hint for r in recs)
