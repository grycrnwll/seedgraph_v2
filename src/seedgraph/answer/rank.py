"""Deterministic reranking (plan §5 / §6.5, 08 §5 step 8; decisions 57/82).

Single-signal deterministic rank: a bounded normalization of ``bm25()`` plus
relational boosts (exact-phrase hit, exact ``normalized_label`` / claim-type match,
span-has-claim link, structured-note hit, project membership). **No learned
reranker, no cross-encoder, no LLM-as-reranker** — ``reranking`` is mapped to
deterministic/no_llm and makes zero LLM calls (decision 57).

``rank_fusion()`` is the single extension seam (plan §9): today a one-line
passthrough over the single ranked list. It is deliberately **not** an abstract
``Reranker`` Protocol and carries no RRF machinery — the deferred vector signal
plugs in here, and only then is fusion generalized (decisions 51/57/67).
"""

from __future__ import annotations

from .types import QuerySpec, RankedCandidate, RetrievedItem

# Boost weights (plan §10 step 4). Fixed constants -> reproducible ordering.
_BOOST_EXACT_PHRASE = 0.30
_BOOST_LABEL_OR_CLAIM_TYPE = 0.10
_BOOST_SPAN_CLAIM_LINK = 0.15
_BOOST_STRUCTURED_NOTE = 0.10
_BOOST_PROJECT_MEMBERSHIP = 0.05
# Build B chunk 6 (decision 71): concept sharpness = 0.10 * w/(w+1) for items
# carrying a concept-lookup IDF weight — ubiquitous (w≈0) earns ~nothing, sharp
# approaches the full boost. Bounded, small: weight is DISCRIMINATIVENESS, not
# importance; this is a relative reorder inside the boost table, never a veto.
_BOOST_CONCEPT_SHARPNESS = 0.10


def boost_constants() -> dict[str, float]:
    """The six deterministic boost weights, keyed by the same names
    :func:`deterministic_rank` writes into ``RankedCandidate.boosts`` (C5).

    The AnswerTrace embeds this dict so a trace stays interpretable after the weights
    change — the trace reads the constants from this one source rather than copying
    literals. ``concept_sharpness`` records the *ceiling* weight; each candidate's
    actual boost is the bounded ``0.10 * w/(w+1)`` fraction stored in its ``boosts``.
    """
    return {
        "exact_phrase": _BOOST_EXACT_PHRASE,
        "claim_type_match": _BOOST_LABEL_OR_CLAIM_TYPE,
        "span_claim_link": _BOOST_SPAN_CLAIM_LINK,
        "structured_note": _BOOST_STRUCTURED_NOTE,
        "concept_sharpness": _BOOST_CONCEPT_SHARPNESS,
        "project_membership": _BOOST_PROJECT_MEMBERSHIP,
    }


def _base_score(bm25_score: float) -> float:
    """Map a raw FTS5 ``bm25`` (lower == better; ``<= 0``) onto a bounded base in
    ``[0, 1)``. A concept-hit's ``0.0`` rank maps to ``0`` base (it earns its weight
    from boosts, not text overlap)."""
    strength = max(0.0, -float(bm25_score))
    return strength / (strength + 1.0)


def deterministic_rank(items: list[RetrievedItem], spec: QuerySpec) -> list[RankedCandidate]:
    """Rank ``items`` by a bounded ``bm25`` base + deterministic boosts (08 §5 step 8).

    Boosts (plan §10 step 4): exact-phrase hit, exact ``normalized_label`` /
    claim-type match, span-has-claim link, structured-note hit, project membership.
    Ordering is stable and reproducible for a fixed input (no randomness, no
    wall-clock; ties break on ``item_id``). Each :class:`RankedCandidate` records
    its per-boost breakdown.
    """
    phrases = [p.strip().lower() for p in spec.phrases if p.strip()]
    ranked: list[RankedCandidate] = []
    for item in items:
        boosts: dict[str, float] = {}
        text_lower = item.text.lower()
        if phrases and any(phrase in text_lower for phrase in phrases):
            boosts["exact_phrase"] = _BOOST_EXACT_PHRASE
        if spec.claim_type is not None and item.kind == "claim":
            boosts["claim_type_match"] = _BOOST_LABEL_OR_CLAIM_TYPE
        if item.kind == "span" and item.claim_id is not None:
            boosts["span_claim_link"] = _BOOST_SPAN_CLAIM_LINK
        if item.kind == "note":
            boosts["structured_note"] = _BOOST_STRUCTURED_NOTE
        if item.concept_weight is not None:
            # concept-lookup items only: replaces the flat treatment where every
            # concept hit scored identically (bm25=0 + fixed boosts), so a
            # ubiquitous concept's claims can no longer flood the candidate
            # pool on equal footing with a sharp concept's.
            w = max(0.0, float(item.concept_weight))
            boosts["concept_sharpness"] = _BOOST_CONCEPT_SHARPNESS * w / (w + 1.0)
        # Every retrieved row is a member of this project's corpus (decision 12).
        boosts["project_membership"] = _BOOST_PROJECT_MEMBERSHIP

        score = _base_score(item.bm25_score) + sum(boosts.values())
        ranked.append(RankedCandidate(item=item, rank_score=score, boosts=boosts))

    # Best-first; deterministic tie-break on item_id so the order is reproducible.
    ranked.sort(key=lambda c: (-c.rank_score, c.item.item_id))
    return ranked


def rank_fusion(ranked: list[RankedCandidate]) -> list[RankedCandidate]:
    """The deferred-vector seam (plan §9, decisions 51/57/67).

    Today: a one-line passthrough returning ``ranked`` unchanged (output order ==
    input order). No RRF code path exists until a second ranking signal actually
    lands; this is the single place that signal will be fused.
    """
    return ranked
