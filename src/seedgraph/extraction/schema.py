"""``default_research_note_v1`` — the field-agnostic default-note Pydantic schema.

Implements plan §4.0 (doc 05 §5 + §6 coverage). Generic research-note fields only
— no field-specific economics terms (econ specifics are phase_6 lens work, doc 05
§1). Each claim-bearing field/element carries the per-claim provenance envelope:

* ``status``           — found | not_found | ambiguous | not_applicable
                         (vocab.StatusValue once vocab.py is extended — needs_wiring).
* ``epistemic_type``   — 6-tier origin (vocab.Provenance / D2). ``llm_extracted``
                         or ``llm_inferred`` for default-note claims.
* ``assertion_status`` — stated | inferred | None; INDEPENDENT of epistemic_type (D2).
* ``inferred_explanation`` — required when ``assertion_status == 'inferred'``.
* ``confidence``       — model-reported [0, 1] confidence.
* ``exact_quote``      — verbatim span text (required for ``found`` substantive
                         claims; the phase_3 ``ensure_span`` anchor).

These string-typed status/origin fields are deliberately loose at the Pydantic
layer (validated against the closed vocab in ``normalize``) so this module imports
cleanly before the vocab.py extension (StatusValue/AssertionStatus/CLAIM_TYPES)
lands in the wiring stage.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Stamped onto every run/note/claim and used as the staleness invalidation key
# together with the cached ``markdown_hash`` (plan §7). Bumping SCHEMA_VERSION or
# PROMPT_VERSION deliberately invalidates the current note for every work.
SCHEMA_ID: str = "default_research_note_v1"
# 1.1.0 (Build C chunk 4, design D4): archetype became a canonical-ordered
# multi-label over {theoretical, empirical, simulation} with simulation
# detection; the bump re-rolls cached notes and is batched into the same
# release as the chunk-3 PROMPT_VERSION bump so notes re-roll once.
SCHEMA_VERSION: str = "1.1.0"
# 1.1.0 (Build C chunk 3, design D4): ONE bump covering BOTH prompt-hardening
# items — the standard-community-canonical-name rule for `normalized_label` and
# the references-tail exclusion from LLM input — so cached notes re-roll once.
PROMPT_VERSION: str = "1.1.0"


class ClaimEnvelope(BaseModel):
    """Per-claim provenance envelope shared by every schema field/element."""

    model_config = ConfigDict(extra="ignore")

    claim_text: str | None = None
    normalized_label: str | None = None
    status: str = "found"
    epistemic_type: str | None = None
    assertion_status: str | None = None
    inferred_explanation: str | None = None
    confidence: float | None = None
    exact_quote: str | None = None


class MethodOrModel(ClaimEnvelope):
    """A method or model element. ``method_type`` becomes ``claim_subtype``; the
    element's nature picks ``claim_type`` ``method`` vs ``model`` (plan §4.0)."""

    method_type: str | None = None


class DataSource(ClaimEnvelope):
    """A data source element → claim_type ``data`` (doc 05 §6 coverage gap)."""


class Assumption(ClaimEnvelope):
    """An assumption element. ``assumption_type`` folds into ``claim_type`` per
    decision 47 (identification→identification_assumption, regularity→
    regularity_condition, else→general_assumption) — NEVER ``claim_subtype``."""

    assumption_type: str | None = None


class Result(ClaimEnvelope):
    """A main-result element. ``result_type`` becomes ``claim_subtype``."""

    result_type: str | None = None


class Limitation(ClaimEnvelope):
    """A limitation element. ``limitation_type`` becomes ``claim_subtype``."""

    limitation_type: str | None = None


class DefaultNoteV1(BaseModel):
    """Validated default research note (schema id ``default_research_note_v1``).

    Scalars are optional single envelopes; arrays default to empty lists. An
    absent/None scalar or empty array still yields an explicit ``not_found`` /
    ``not_applicable`` claim row downstream in ``normalize`` (absence is recorded,
    never dropped — plan §1).
    """

    model_config = ConfigDict(extra="ignore")

    research_question: ClaimEnvelope | None = None
    main_contribution: ClaimEnvelope | None = None
    method_or_model: list[MethodOrModel] = Field(default_factory=list)
    data_sources: list[DataSource] = Field(default_factory=list)
    setting: ClaimEnvelope | None = None
    estimand_or_target_object: ClaimEnvelope | None = None
    assumptions: list[Assumption] = Field(default_factory=list)
    main_results: list[Result] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    robustness_checks: list[ClaimEnvelope] = Field(default_factory=list)
    open_questions: list[ClaimEnvelope] = Field(default_factory=list)
