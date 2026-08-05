"""Answer-harness data contracts (phase_8 §4 / §6.2).

This module is the *contract surface* the eval phase, the FastAPI read view, and
any agent surface consume (decision 62/82): the :class:`AnswerEnvelope` carries
everything needed to check faithfulness, leakage, and abstention deterministically
*as data* — no behaviour is trusted.

Scaffold note — closed enums home:
    Plan §10 step 1 places ``QueryType`` / ``AnswerCategory`` / ``AnswerMode`` in
    the shared ``vocab.py`` (decision 14). ``vocab.py`` is a protected cross-phase
    file in this scaffolding pass, so the three enums are defined here as the
    phase-private placeholder. The single wiring stage MUST relocate them to
    ``vocab.py`` and re-point the imports in this package (see ``needs_wiring``).
    ``AnswerMode`` member casing follows the §6 signature literally
    (``AnswerMode.PROJECT_ONLY``); ``QueryType`` / ``AnswerCategory`` follow the
    ``vocab.py`` house style (member == lowercase value).

Decisions implemented by the shapes here:
* 82 — single self-declaring LLM call: ``query_type`` + ``answer_category`` are
  LLM-declared then validated/reconciled by the guard.
* 62 — structured envelope so faithfulness/leakage are deterministic.
* 53/20/83 — verbatim span quote is authoritative (``Citation.quote`` is the
  stored ``evidence_spans.exact_quote``).
* D2 — two orthogonal provenance fields (``epistemic_type`` origin +
  ``assertion_status`` stated/inferred) ride from ``extracted_claims`` onto each
  ``Citation``.
* 29/44/65 — ``answer_id`` is a type-prefixed opaque id (``"ans_" + uuid4hex``).
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

# Closed enums are the single source of truth in vocab.py (plan §10 step 1,
# decision 14) — imported + re-exported here so the answer package and its
# consumers (eval / FastAPI / agents) keep one import surface. NOT re-defined.
from ..vocab import AnswerCategory, AnswerMode, QueryType

__all__ = [
    "QueryType",
    "AnswerCategory",
    "AnswerMode",
    "QuerySpec",
    "RetrievedItem",
    "RankedCandidate",
    "Citation",
    "Recommendation",
    "AnswerEnvelope",
    "AllowedSet",
]


# --- Query parse / retrieval intermediates ---------------------------------

class QuerySpec(BaseModel):
    """Parsed question + merged retrieval filters (plan §5 — ``Query`` and
    ``Filters`` merged into one ``QuerySpec``). Produced by
    ``harness._preclassify`` (08 §5 steps 1-2)."""

    model_config = ConfigDict(extra="forbid")

    question: str
    protocol_hint: QueryType
    # exact phrases to issue as FTS5 phrase queries (e.g. ``"Assumption 2"``).
    phrases: list[str] = Field(default_factory=list)
    # work-name / title tokens the question named (drives comparative partitioning).
    work_tokens: list[str] = Field(default_factory=list)
    # candidate concept/method/assumption tokens (08 §5 step 2).
    concept_tokens: list[str] = Field(default_factory=list)
    # relational filters (08 §6): None == unfiltered.
    work_id: str | None = None
    section: str | None = None
    claim_type: str | None = None
    lens: str | None = None
    access_class: str | None = None
    mode: AnswerMode = AnswerMode.PROJECT_ONLY


class RetrievedItem(BaseModel):
    """One unified retrieved row across spans/claims/notes/concepts
    (``retrieve.py``). Carries the raw ``bm25()`` score and both source-claim
    provenance fields (decision D2) so rank/guard never re-query."""

    model_config = ConfigDict(extra="forbid")

    item_id: str                         # span_id / claim_id / note_id / concept_id
    kind: str                            # "span" | "claim" | "note" | "concept"
    work_id: str
    span_id: str | None = None
    claim_id: str | None = None
    note_id: str | None = None
    concept_id: str | None = None
    section: str | None = None
    text: str                            # snippet / exact_quote (verbatim, decision 53)
    bm25_score: float                    # raw FTS5 bm25() (lower == better)
    access_class: str                    # denormalized (decisions 30/60/76)
    epistemic_type: str | None = None    # D2 origin (6-tier), from extracted_claims
    assertion_status: str | None = None  # D2 stated|inferred, from extracted_claims
    # Build B chunk 6 — concept-lookup provenance for the deterministic
    # ubiquity down-ranking (decision 71): the matched concept's IDF weight +
    # raw paper_frequency. None on non-concept-derived items (spans/notes/FTS
    # claims), which therefore rank bit-identically to before.
    concept_weight: float | None = None
    concept_paper_frequency: int | None = None


class RankedCandidate(BaseModel):
    """A ``RetrievedItem`` plus its deterministic rank score and boost breakdown
    (``rank.py``). ``rank_fusion`` is a passthrough over a list of these until a
    second signal lands (decisions 51/57/67)."""

    model_config = ConfigDict(extra="forbid")

    item: RetrievedItem
    rank_score: float                    # normalized; higher == better
    boosts: dict[str, float] = Field(default_factory=dict)


# --- Envelope contract (the artifact later phases consume) ------------------

class Citation(BaseModel):
    """A traceable, in-corpus citation (08 §8 / §6.2). Only emitted for works
    that survived the membership guard; ``quote`` is the verbatim
    ``evidence_spans.exact_quote`` (decision 53)."""

    model_config = ConfigDict(extra="forbid")

    work_id: str
    title: str | None = None
    year: int | None = None
    span_ids: list[str] = Field(default_factory=list)
    quote: str | None = None
    section: str | None = None
    epistemic_type: str                  # D2 origin, copied from extracted_claims
    assertion_status: str | None = None  # D2 stated|inferred; None when N/A


class Recommendation(BaseModel):
    """A work present in the citation graph but unavailable (08 §12): metadata
    only / no spans. Surfaced as a recommendation, **never** as a citation."""

    model_config = ConfigDict(extra="forbid")

    work_id: str
    title: str | None = None
    year: int | None = None
    reason: str                          # why surfaced (co-citation count, overlap)
    status: str                          # "metadata_only" | "unavailable"
    action_hint: str                     # e.g. "upload PDF to access full text"


class AnswerEnvelope(BaseModel):
    """The structured answer artifact (08 §6.2, decision 62/82). Faithfulness,
    leakage, and abstention are all checkable from these fields alone."""

    model_config = ConfigDict(extra="forbid")

    answer_id: str                            # "ans_" + uuid4hex
    question: str
    query_type: QueryType                     # LLM-declared, validated vs vocab
    answer_category: AnswerCategory           # guard-reconciled
    answer_text: str                          # "" in retrieval_only / after abstention
    citations: list[Citation] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    cited_work_ids: list[str] = Field(default_factory=list)
    cited_span_ids: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False       # the abstain / not-found flag
    retrieved_item_ids: list[str] = Field(default_factory=list)  # ALLOWED citation set
    warnings: list[str] = Field(default_factory=list)
    mode: AnswerMode = AnswerMode.PROJECT_ONLY
    llm_provenance: dict | None = None        # task/profile/model/access_mode/... (doc 13 §15)


# --- Guard input -----------------------------------------------------------

@dataclass(frozen=True)
class AllowedSet:
    """The code-enforced citation allowlist the guard checks membership against
    (08 §6.3). Built from the candidates **actually shown** to the model (after
    ``max_candidates`` + token-budget truncation, §8) — anything dropped for
    budget is absent here, so the model cannot cite what it never saw.

    ``retrieved_item_ids`` preserves shown order; ``work_ids`` / ``span_ids`` are
    the fast membership sets the guard intersects against.
    """

    work_ids: frozenset[str]
    span_ids: frozenset[str]
    retrieved_item_ids: tuple[str, ...]
