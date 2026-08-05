"""Deterministic, pure-stdlib evaluation metrics (phase_9 §6).

Every function here is total, side-effect-free, and KEY-FREE: retrieval
recall/precision/ranking/coverage (doc 09 §7 — computed work-level AND span-level
by the caller, which invokes these twice), span-indexing exact-match rate
(§3/§5), unsupported-synthesis rate (§8), user-correction rate (decision 64),
concept-identity over/under-merge noise (decision D10), oversize-aware
note-coverage (decision D4), and the phase_2 unresolved-target coverage gap
(decision D3). No LLM, no I/O — this is the MVP path and the basis of the CI gate.

Upstream object types (``AnswerEnvelope``, ``Span``, ``Note``) are owned by other
phases and referenced ONLY in annotations (PEP 563 — never evaluated at runtime).
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # imported only for type-checkers; never executed at runtime
    from seedgraph.answer.types import AnswerEnvelope  # phase_8 owns (plan §5/§9)

    # Span / Note are produced upstream (phase_3 evidence_spans / phase_4 notes);
    # their real types replace these provisional aliases once those phases land.
    Span = object  # noqa: F811 — provisional: phase_3 owns the real Span type
    Note = object  # noqa: F811 — provisional: phase_4 owns the real Note type


# --- value access helper (attr OR mapping key) -----------------------------

def _attr(obj: Any, *names: str) -> Any:
    """Read the first present attribute/key in ``names`` from a mapping or object."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return None


# --- retrieval metrics (doc 09 §7) -----------------------------------------

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of ``relevant`` ids appearing in the top-``k`` of ``retrieved``.

    ``|relevant ∩ set(retrieved[:k])| / |relevant|`` (0.0 when ``relevant`` is
    empty). Called at BOTH granularities by ``report.py`` — once over
    ``relevant_work_ids`` (work-level) and once over ``relevant_span_ids``
    (span-level). Pure stdlib (doc 09 §7).
    """
    if not relevant:
        return 0.0
    top = set(retrieved[: max(k, 0)])
    return len(relevant & top) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-``k`` retrieved ids that are in ``relevant``.

    ``hits / min(k, len(retrieved))`` (0.0 when the top-``k`` is empty). Work-level
    and span-level, as for :func:`recall_at_k`.
    """
    denom = min(max(k, 0), len(retrieved))
    if denom <= 0:
        return 0.0
    top = retrieved[:denom]
    hits = sum(1 for item in top if item in relevant)
    return hits / denom


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Ranking-quality proxy: ``1 / rank`` of the FIRST relevant id in
    ``retrieved`` (1-indexed), or 0.0 if none of ``relevant`` is retrieved
    (doc 09 §7). The per-query mean is the MRR reported by ``report.py``.
    """
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def corpus_coverage(retrieved_per_q: list[list[str]], corpus_ids: set[str]) -> float:
    """Fraction of ``corpus_ids`` surfaced across the union of all queries'
    retrieved lists — ``|(⋃ retrieved_per_q) ∩ corpus_ids| / |corpus_ids|``
    (0.0 when ``corpus_ids`` empty). Coverage breadth of the gold set (doc 09 §7).
    """
    if not corpus_ids:
        return 0.0
    seen: set[str] = set()
    for retrieved in retrieved_per_q:
        seen.update(retrieved)
    return len(seen & corpus_ids) / len(corpus_ids)


# --- span / answer quality rates -------------------------------------------

def span_indexing_quality(spans: list["Span"]) -> float:
    """Fraction of evidence ``spans`` whose stored slice re-verifies — i.e.
    ``md[start:end] == quote`` (quote_hash match) when the markdown is re-sliced
    (doc 09 §3/§5). The deterministic span_indexing exact-match rate over all
    spans; 1.0 == every span is verbatim-anchored. 0.0 for an empty list.

    Each ``span`` carries its own re-sliceable evidence as attributes OR mapping
    keys: ``markdown`` (the source text), ``start``/``start_char``, ``end``/
    ``end_char``, and ``quote``/``exact_quote``. A span whose markdown is
    unavailable (None) counts as NOT verified (fail-closed).
    """
    if not spans:
        return 0.0
    ok = 0
    for span in spans:
        markdown = _attr(span, "markdown", "markdown_text")
        start = _attr(span, "start", "start_char")
        end = _attr(span, "end", "end_char")
        quote = _attr(span, "quote", "exact_quote")
        if markdown is None or start is None or end is None or quote is None:
            continue
        if markdown[start:end] == quote:
            ok += 1
    return ok / len(spans)


def unsupported_synthesis_rate(envelopes: list["AnswerEnvelope"]) -> float:
    """Fraction of answer ``envelopes`` that synthesize/infer a conclusion WITHOUT
    adequate cited support (doc 09 §8) — emitted DISTINCTLY from answer
    faithfulness. An envelope counts as unsupported synthesis when it makes a
    NON-abstaining assertion (``insufficient_evidence`` is False and ``answer_text``
    is non-empty) yet cites ZERO evidence spans. No LLM. 0.0 for an empty list.
    """
    if not envelopes:
        return 0.0
    unsupported = 0
    for env in envelopes:
        insufficient = bool(_attr(env, "insufficient_evidence"))
        text = (_attr(env, "answer_text") or "").strip()
        cited = _attr(env, "cited_span_ids") or []
        if (not insufficient) and text and not cited:
            unsupported += 1
    return unsupported / len(envelopes)


def user_correction_rate(audit_decisions: list[str]) -> float:
    """``(#reject + #edit) / #resolved`` over the validation ``decision`` values
    of resolved audit rows (decision 64; doc 09 §2). Measures how often a human
    overrode the system. 0.0 when there are no resolved decisions.
    """
    decisions = [d for d in audit_decisions if d]
    if not decisions:
        return 0.0
    corrections = sum(1 for d in decisions if d in ("reject", "edit"))
    return corrections / len(decisions)


def concept_identity_noise(merge_verdicts: list[str]) -> dict[str, float]:
    """Estimate ``{over_merge_rate, under_merge_rate}`` from a human-graded
    concept-merge sample (decision D10). Does NOT assume perfect identity: a
    merge graded ``over_merge`` (wrongly joined distinct concepts) counts toward
    over-merge; a ``under_merge`` (missed a true merge) toward under-merge. The
    rates feed ``report.py`` and are compared against the stated
    ``report.NOISE_TOLERANCE`` (findings, not a hard gate). Empty → zeros.
    """
    if not merge_verdicts:
        return {"over_merge_rate": 0.0, "under_merge_rate": 0.0}
    total = len(merge_verdicts)
    over = sum(1 for v in merge_verdicts if v == "over_merge")
    under = sum(1 for v in merge_verdicts if v == "under_merge")
    return {"over_merge_rate": over / total, "under_merge_rate": under / total}


def note_coverage(notes: list["Note"], skipped_oversize: int) -> dict[str, float]:
    """Note-coverage scoped to WINDOW-FITTING papers (decision D4): the
    denominator is the count of window-fitting papers (``len(notes)``) and
    EXCLUDES phase_4 ``run_status='skipped_oversize'`` papers, which are reported
    separately via ``oversize_skipped``. Each element of ``notes`` is one
    window-fitting paper; a truthy element is a paper that HAS a default note.
    Returns ``{coverage, covered, window_fitting, oversize_skipped}``. Oversize
    papers are never a coverage failure (phase_4b retires the skip), never
    silently dropped.
    """
    window_fitting = len(notes)
    covered = sum(1 for note in notes if note)
    coverage = covered / window_fitting if window_fitting else 0.0
    return {
        "coverage": coverage,
        "covered": float(covered),
        "window_fitting": float(window_fitting),
        "oversize_skipped": float(skipped_oversize),
    }


def unresolved_target_rate(citation_edges: int, unresolved_targets: int) -> float:
    """phase_2 referenced-but-absent targets as a share of all references
    (decision D3): ``unresolved_targets / (citation_edges + unresolved_targets)``
    (0.0 when there are no references). A citation-coverage GAP metric — phase_2
    is link-only and never vivifies a target; eval only reports the diagnostic.
    """
    total = citation_edges + unresolved_targets
    if total <= 0:
        return 0.0
    return unresolved_targets / total
