"""Phase 8 — guard acceptance tests (plan §11 `guard` group; 08 §8, decisions
82/62).

These exercise the code-enforced faithfulness backstop — nothing is trusted to the
prompt. The guard runs on plain in-process data (no LLM, no DB), so each invariant
is checked deterministically.
"""

from __future__ import annotations

import sqlite3

from _phase8_helpers import build_fixture_project, make_allowed, make_citation, make_envelope

from seedgraph.answer import guard  # noqa: F401
from seedgraph.answer.guard import NOT_FOUND_MESSAGE, enforce, reconcile, short_circuit_if_empty
from seedgraph.answer.types import (
    AnswerCategory,
    AnswerMode,
    RankedCandidate,
    RetrievedItem,
)


def _candidate(item_id="s1", work_id="work_a", score=0.5) -> RankedCandidate:
    item = RetrievedItem(
        item_id=item_id, kind="span", work_id=work_id, span_id=item_id,
        text="evidence", bm25_score=-2.0, access_class="open_access",
    )
    return RankedCandidate(item=item, rank_score=score, boosts={})


def test_no_invented_citations_membership_drop():
    """A fabricated ``work_id``/``span_id`` in a citation is dropped by the membership
    check, a warning is added, and the fabricated citation is absent from the
    returned envelope (plan §10 step 7a)."""
    allowed = make_allowed({"work_a"}, {"s_a"})
    env = make_envelope(
        answer_text="A holds [E1] and X holds [E2].",
        citations=[make_citation("work_a", ["s_a"]), make_citation("work_x", ["s_x"])],
    )
    out = enforce(env, allowed, AnswerMode.PROJECT_ONLY)
    assert [c.work_id for c in out.citations] == ["work_a"]
    assert out.cited_work_ids == ["work_a"]
    assert out.cited_span_ids == ["s_a"]
    assert any(w.startswith("dropped_citation_work_not_retrieved") for w in out.warnings)


def test_post_drop_abstention_when_all_citations_drop():
    """Model cites ONLY fabricated/unretrieved ids while emitting prose ⇒ after the
    membership drop the citation set is empty ⇒ guard forces abstention, replaces the
    prose, and warns ``unsupported_prose_suppressed`` (must-fix #1, plan §10 7b)."""
    allowed = make_allowed({"work_a"}, {"s_a"})
    env = make_envelope(
        answer_text="The corpus proves it conclusively [E1].",
        citations=[make_citation("work_fake", ["s_fake"])],
    )
    out = enforce(env, allowed, AnswerMode.PROJECT_ONLY)
    assert out.insufficient_evidence is True
    assert out.answer_category == AnswerCategory.unresolved
    assert out.answer_text == NOT_FOUND_MESSAGE
    assert out.citations == []
    assert "unsupported_prose_suppressed" in out.warnings


def test_post_drop_abstention_distinct_from_partial_drop():
    """When a fabricated id is dropped but a valid citation remains, the answer stands
    (no forced abstention) — distinct from the all-drop case."""
    allowed = make_allowed({"work_a"}, {"s_a"})
    env = make_envelope(
        answer_text="A holds [E1]; spurious [E2].",
        citations=[make_citation("work_a", ["s_a"]), make_citation("work_fake", ["s_fake"])],
    )
    out = enforce(env, allowed, AnswerMode.PROJECT_ONLY)
    assert out.insufficient_evidence is False
    assert out.answer_text == "A holds [E1]; spurious [E2]."
    assert [c.work_id for c in out.citations] == ["work_a"]
    assert "unsupported_prose_suppressed" not in out.warnings


def test_empty_retrieval_short_circuits_without_llm_call():
    """Empty retrieval ⇒ ``short_circuit_if_empty`` signals abstention with an
    ``empty_corpus`` warning (the harness then makes NO LLM call; plan §10 7c)."""
    short, warnings = short_circuit_if_empty([])
    assert short is True
    assert "empty_corpus" in warnings


def test_below_weak_evidence_floor_proceeds_with_warning():
    """A non-empty retrieval whose top score is below ``weak_evidence_floor`` (but
    above ``absent_floor``) proceeds but carries a ``weak_evidence`` warning."""
    short, warnings = short_circuit_if_empty(
        [_candidate(score=0.05)], weak_evidence_floor=0.15, absent_floor=0.0
    )
    assert short is False
    assert "weak_evidence" in warnings

    # all-below absent_floor (score 0) short-circuits as absent.
    short2, warnings2 = short_circuit_if_empty(
        [_candidate(score=0.0)], weak_evidence_floor=0.15, absent_floor=0.0
    )
    assert short2 is True
    assert "empty_corpus" in warnings2


def test_project_only_outside_corpus_coerced_to_unresolved():
    """In ``project_only`` mode an LLM-declared ``outside_corpus`` category is coerced
    to ``unresolved`` / abstain (leakage guard, 09 §9 / plan §10 7d)."""
    allowed = make_allowed({"work_a"}, {"s_a"})
    env = make_envelope(
        answer_text="From general knowledge, the answer is 42.",
        citations=[make_citation("work_a", ["s_a"])],
        category="outside_corpus",
    )
    out = enforce(env, allowed, AnswerMode.PROJECT_ONLY)
    assert out.answer_category == AnswerCategory.unresolved
    assert out.insufficient_evidence is True
    assert out.answer_text == ""
    assert "outside_corpus_suppressed" in out.warnings


def test_allow_outside_permits_labeled_outside_corpus_answer():
    """In ``allow_outside`` mode an ``outside_corpus`` answer is permitted and labeled
    (not coerced) — the complement of the leakage guard."""
    allowed = make_allowed({"work_a"}, {"s_a"})
    env = make_envelope(
        answer_text="This rests on knowledge outside the project corpus.",
        citations=[],
        category="outside_corpus",
    )
    out = enforce(env, allowed, AnswerMode.ALLOW_OUTSIDE)
    assert out.answer_category == AnswerCategory.outside_corpus
    assert out.insufficient_evidence is False
    assert out.answer_text.startswith("This rests on knowledge")


def test_category_reconciliation_source_grounded():
    """≥1 supporting span ⇒ guard reconciles ``source_grounded`` (plan §10 7e)."""
    allowed = make_allowed({"work_a"}, {"s_a"})
    env = make_envelope(
        answer_text="A holds [E1].",
        citations=[make_citation("work_a", ["s_a"])],
        category="unresolved",
    )
    out = enforce(env, allowed, AnswerMode.PROJECT_ONLY)
    assert out.answer_category == AnswerCategory.source_grounded


def test_category_reconciliation_corpus_synthesis():
    """Supporting spans across ≥2 works with no single *stating* span ⇒
    ``corpus_synthesis`` (plan §10 7e)."""
    allowed = make_allowed({"work_a", "work_b"}, {"s_a", "s_b"})
    env = make_envelope(
        answer_text="Across A and B [E1][E2].",
        citations=[
            make_citation("work_a", ["s_a"], assertion="inferred"),
            make_citation("work_b", ["s_b"], assertion="inferred"),
        ],
        category="source_grounded",
    )
    out = enforce(env, allowed, AnswerMode.PROJECT_ONLY)
    assert out.answer_category == AnswerCategory.corpus_synthesis


def test_category_reconciliation_project_graph_inference():
    """Support only via work-level (graph) citations with no spans ⇒
    ``project_graph_inference`` (plan §10 7e)."""
    allowed = make_allowed({"work_a"}, set())
    env = make_envelope(
        answer_text="A is connected to B in the project graph [E1].",
        citations=[make_citation("work_a", [], quote=None)],
        category="source_grounded",
    )
    out = enforce(env, allowed, AnswerMode.PROJECT_ONLY)
    assert out.answer_category == AnswerCategory.project_graph_inference
    assert out.insufficient_evidence is False  # graph citation counts as support


def test_enforce_recomputes_cited_ids_from_surviving_citations():
    """``enforce`` recomputes ``cited_work_ids``/``cited_span_ids`` from the citations
    that survived the membership drop, keeping the envelope self-consistent."""
    allowed = make_allowed({"work_a", "work_b"}, {"s_a", "s_b"})
    env = make_envelope(
        answer_text="A and B [E1][E2]; fake [E3].",
        citations=[
            make_citation("work_a", ["s_a"]),
            make_citation("work_b", ["s_b"]),
            make_citation("work_gone", ["s_gone"]),
        ],
    )
    out = enforce(env, allowed, AnswerMode.PROJECT_ONLY)
    assert set(out.cited_work_ids) == {"work_a", "work_b"}
    assert set(out.cited_span_ids) == {"s_a", "s_b"}
    assert all(w in allowed.work_ids for w in out.cited_work_ids)
    assert all(s in allowed.span_ids for s in out.cited_span_ids)


def test_reconcile_direct_source_grounded():
    """``reconcile`` alone sets ``source_grounded`` for a single span-backed work."""
    allowed = make_allowed({"work_a"}, {"s_a"})
    env = make_envelope(citations=[make_citation("work_a", ["s_a"])], category="unresolved")
    out = reconcile(env, allowed)
    assert out.answer_category == AnswerCategory.source_grounded


def test_excluded_work_citation_dropped_by_guard_inclusion_check():
    """Belt-and-suspenders (defect 1, plan §11): even if retrieval somehow surfaced a
    citation for an EXCLUDED work (e.g. one excluded *after* extraction that retains
    spans), the guard drops it by resolving ``project_documents.inclusion_status`` —
    code-enforcing 'every cited work is included/metadata_only', never prompt-trusted
    (decision 82). The included work survives, so the assertion is NON-VACUOUS.

    This FAILS against the pre-fix guard, which had no inclusion-status check (the conn
    parameter did not even exist)."""
    project = build_fixture_project("ph8_guard_excl")
    conn = sqlite3.connect(str(project.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        # the allowed set lets BOTH the included work and the excluded work through the
        # membership check — isolating the inclusion-status layer.
        allowed = make_allowed({"work_a", "work_x"}, {"s_a", "s_x"})
        env = make_envelope(
            answer_text="A holds [E1]; the excluded work says [E2].",
            citations=[make_citation("work_a", ["s_a"]), make_citation("work_x", ["s_x"])],
        )
        out = enforce(env, allowed, AnswerMode.PROJECT_ONLY, conn=conn)

        assert "work_x" not in out.cited_work_ids
        assert [c.work_id for c in out.citations] == ["work_a"]
        assert out.cited_work_ids == ["work_a"]
        assert out.cited_span_ids == ["s_a"]
        assert any(w.startswith("excluded_work_citation_dropped") for w in out.warnings)
        # the surviving cited work resolves to an included/metadata_only membership row.
        for wid in out.cited_work_ids:
            status = conn.execute(
                "SELECT inclusion_status FROM project_documents WHERE work_id=?", (wid,)
            ).fetchone()
            assert status is not None and status[0] in ("included", "metadata_only")
    finally:
        conn.close()
