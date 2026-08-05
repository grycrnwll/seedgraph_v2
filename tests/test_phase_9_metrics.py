"""Phase 9 — unit metric tests (plan §11 "Unit", maps to test_eval_metrics.py).

Deterministic, key-free: the metrics module has no upstream-phase dependency at
import time and no LLM/I-O. Every value below is hand-computed.
"""

from __future__ import annotations

from seedgraph.eval import metrics


def test_recall_precision_at_k_work_and_span_level():
    """recall_at_k / precision_at_k match hand-computed values on a fixed
    retrieved/relevant fixture, asserted at BOTH work-level (relevant_work_ids)
    and span-level (relevant_span_ids) granularity (plan §11; doc 09 §7)."""
    # work level: top-3 of [w1,w2,w3,w4] is {w1,w2,w3}; relevant {w1,w3,w5}.
    work_retrieved = ["w1", "w2", "w3", "w4"]
    work_relevant = {"w1", "w3", "w5"}
    assert metrics.recall_at_k(work_retrieved, work_relevant, 3) == 2 / 3
    assert metrics.precision_at_k(work_retrieved, work_relevant, 3) == 2 / 3

    # span level: top-2 of [s1,s2] is {s1,s2}; relevant {s2}.
    span_retrieved = ["s1", "s2"]
    span_relevant = {"s2"}
    assert metrics.recall_at_k(span_retrieved, span_relevant, 2) == 1.0
    assert metrics.precision_at_k(span_retrieved, span_relevant, 2) == 0.5

    # empty-relevant guard (no division by zero).
    assert metrics.recall_at_k(work_retrieved, set(), 3) == 0.0
    assert metrics.precision_at_k([], work_relevant, 3) == 0.0


def test_reciprocal_rank_and_corpus_coverage():
    """reciprocal_rank returns 1/rank of the first relevant id (0.0 if none), and
    corpus_coverage returns |⋃retrieved ∩ corpus| / |corpus| on a hand-computed
    fixture (doc 09 §7)."""
    assert metrics.reciprocal_rank(["a", "b", "c"], {"b", "c"}) == 0.5
    assert metrics.reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert metrics.reciprocal_rank(["a"], {"z"}) == 0.0

    assert metrics.corpus_coverage([["a", "b"], ["c"]], {"a", "c", "d"}) == 2 / 3
    assert metrics.corpus_coverage([["a"]], set()) == 0.0


def test_span_indexing_quality_exact_match_fraction():
    """span_indexing_quality returns the fraction of spans whose md[start:end]==quote
    (quote_hash) on a fixture mixing verbatim-anchored and drifted spans (doc 09 §3/§5)."""
    md = "hello world foo"
    spans = [
        {"markdown": md, "start": 0, "end": 5, "quote": "hello"},   # verbatim
        {"markdown": md, "start": 6, "end": 11, "quote": "WORLD"},  # drifted (case)
        {"markdown": None, "start": 0, "end": 3, "quote": "abc"},   # unavailable -> fail-closed
    ]
    assert metrics.span_indexing_quality(spans) == 1 / 3
    assert metrics.span_indexing_quality([]) == 0.0


def test_unsupported_synthesis_rate_over_envelopes():
    """unsupported_synthesis_rate returns the fraction of recorded envelopes that
    synthesize/infer without adequate cited support — emitted distinctly from
    answer faithfulness (doc 09 §8)."""
    envelopes = [
        {"insufficient_evidence": False, "answer_text": "asserted", "cited_span_ids": []},     # unsupported
        {"insufficient_evidence": False, "answer_text": "grounded", "cited_span_ids": ["s1"]}, # supported
        {"insufficient_evidence": True, "answer_text": "", "cited_span_ids": []},              # abstained
    ]
    assert metrics.unsupported_synthesis_rate(envelopes) == 1 / 3
    assert metrics.unsupported_synthesis_rate([]) == 0.0


def test_user_correction_rate_reject_plus_edit_over_resolved():
    """user_correction_rate returns (#reject + #edit)/#resolved over a hand-built
    list of validation decisions (decision 64; doc 09 §2)."""
    assert metrics.user_correction_rate(["accept", "reject", "edit", "accept"]) == 0.5
    assert metrics.user_correction_rate([]) == 0.0
    assert metrics.user_correction_rate(["accept", "accept"]) == 0.0


def test_concept_identity_noise_over_and_under_merge_rates():
    """concept_identity_noise returns the expected {over_merge_rate, under_merge_rate}
    on a hand-graded concept-merge sample — no perfect-identity assumption (decision D10)."""
    verdicts = ["correct", "over_merge", "under_merge", "correct", "over_merge"]
    noise = metrics.concept_identity_noise(verdicts)
    assert noise == {"over_merge_rate": 0.4, "under_merge_rate": 0.2}
    assert metrics.concept_identity_noise([]) == {"over_merge_rate": 0.0, "under_merge_rate": 0.0}


def test_note_coverage_excludes_skipped_oversize_from_denominator():
    """note_coverage scopes the denominator to window-fitting papers and reports the
    skipped_oversize count separately — oversize papers are never a coverage failure
    nor silently dropped (decision D4)."""
    # 4 window-fitting papers, 3 with a note; 2 oversize-skipped (NOT in denominator).
    result = metrics.note_coverage([1, 1, 0, 1], skipped_oversize=2)
    assert result["window_fitting"] == 4.0
    assert result["covered"] == 3.0
    assert result["coverage"] == 0.75
    assert result["oversize_skipped"] == 2.0


def test_unresolved_target_rate_matches_phase2_diagnostics():
    """unresolved_target_rate equals hand-counted phase_2 referenced-but-absent
    targets / total references — a citation-coverage gap, link-only (decision D3)."""
    assert metrics.unresolved_target_rate(citation_edges=8, unresolved_targets=2) == 0.2
    assert metrics.unresolved_target_rate(citation_edges=0, unresolved_targets=0) == 0.0
