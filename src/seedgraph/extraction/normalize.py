"""Flatten a validated ``DefaultNoteV1`` into claim drafts (plan §5/§10 step 7).

One :class:`ClaimDraft` per schema field (scalars) / per array element (arrays),
INCLUDING explicit ``not_found`` / ``not_applicable`` rows for absent fields —
absence is recorded, never dropped (plan §1). Implements decision 47 (assumption
type folds into ``claim_type``, never ``claim_subtype``) and D2 (independent
``epistemic_type`` + ``assertion_status``: ``llm_inferred`` when
``assertion_status == 'inferred'``, else ``llm_extracted``; an ``inferred`` claim
without an explanation is rejected). Also derives the flattened ``note_text`` (the
``note_fts`` source) and the ``archetype`` (decision 9, from populated fields).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..vocab import (
    AssertionStatus,
    EpistemicType,
    StatusValue,
    normalize_claim_type,
)
from .schema import ClaimEnvelope, DefaultNoteV1


@dataclass
class ClaimDraft:
    """A single pending ``extracted_claims`` row, pre-persistence.

    ``claim_type`` is the doc 05 §6 vocab value (assumptions already folded per
    decision 47). ``claim_subtype`` is set ONLY for method/result/limitation
    elements. ``epistemic_type`` is stamped here (D2); ``access_class`` is stamped
    later by the runner (most-restrictive, plan §7). ``exact_quote`` (when present)
    is what the runner anchors via phase_3 ``ensure_span``.
    """

    claim_type: str
    field_key: str
    status: str
    epistemic_type: str
    claim_subtype: str | None = None
    normalized_label: str | None = None
    claim_text: str | None = None
    assertion_status: str | None = None
    inferred_explanation: str | None = None
    confidence: float | None = None
    exact_quote: str | None = None


_INFERRED = AssertionStatus.inferred.value


def map_assumption_claim_type(assumption_type: str | None) -> str:
    """Fold an ``assumption_type`` onto its ``claim_type`` (decision 47).

    ``identification`` → ``identification_assumption``; ``regularity`` →
    ``regularity_condition``; anything else (incl. None) → ``general_assumption``.
    Keeps FTS faceting and the phase_7 concept feed correct.
    """
    value = (assumption_type or "").strip().lower()
    if value == "identification":
        return "identification_assumption"
    if value == "regularity":
        return "regularity_condition"
    return "general_assumption"


def _normalize_status(value: str | None) -> str:
    try:
        return StatusValue(value).value
    except (ValueError, TypeError):
        return StatusValue.found.value


def _epistemic_for(assertion_status: str | None) -> str:
    if assertion_status == _INFERRED:
        return EpistemicType.llm_inferred.value
    return EpistemicType.llm_extracted.value


def _draft_from_envelope(
    env: ClaimEnvelope,
    *,
    claim_type: str,
    field_key: str,
    claim_subtype: str | None = None,
) -> ClaimDraft:
    """Build one claim draft from a populated schema envelope (D2 stamping)."""
    status = _normalize_status(env.status)
    assertion_status = env.assertion_status
    if assertion_status is not None and assertion_status not in (
        AssertionStatus.stated.value,
        AssertionStatus.inferred.value,
    ):
        assertion_status = None
    if assertion_status == _INFERRED and not (env.inferred_explanation or "").strip():
        raise ValueError(
            f"claim {field_key!r} has assertion_status='inferred' but no "
            f"inferred_explanation (D2 — an inferred claim must explain the inference)"
        )
    return ClaimDraft(
        claim_type=claim_type,
        field_key=field_key,
        status=status,
        epistemic_type=_epistemic_for(assertion_status),
        claim_subtype=claim_subtype,
        normalized_label=env.normalized_label,
        claim_text=env.claim_text,
        assertion_status=assertion_status,
        inferred_explanation=env.inferred_explanation,
        confidence=env.confidence,
        exact_quote=env.exact_quote,
    )


def _absent_draft(claim_type: str, field_key: str) -> ClaimDraft:
    """An explicit ``not_found`` row for an absent scalar / empty array (plan §1)."""
    return ClaimDraft(
        claim_type=claim_type,
        field_key=field_key,
        status=StatusValue.not_found.value,
        epistemic_type=EpistemicType.llm_extracted.value,
    )


def _method_claim_type(method_type: str | None) -> str:
    """A ``method_or_model`` element is ``model`` iff its type names a model."""
    if method_type and "model" in method_type.strip().lower():
        return "model"
    return "method"


def normalize_note(note: DefaultNoteV1) -> tuple[list[ClaimDraft], str, str]:
    """Flatten ``note`` → ``(claims, note_text, archetype)``.

    Emits exactly one draft per scalar field and one per array element (and a
    ``not_found``/``not_applicable`` draft for absent scalars / empty arrays).
    Folds ``assumption_type`` into ``claim_type`` (decision 47); sets
    ``claim_subtype`` only for method/result/limitation; stamps ``epistemic_type``
    (``llm_inferred`` iff ``assertion_status='inferred'``, else ``llm_extracted``);
    raises on an ``inferred`` claim missing ``inferred_explanation``. Returns the
    flattened ``note_text`` and derived ``archetype`` alongside the drafts.
    """
    drafts: list[ClaimDraft] = []

    # --- scalars (one row each; absent -> not_found) -----------------------
    scalar_fields: list[tuple[str, ClaimEnvelope | None, str]] = [
        ("research_question", note.research_question, "research_question"),
        ("main_contribution", note.main_contribution, "main_contribution"),
        ("setting", note.setting, "setting"),
        ("estimand_or_target_object", note.estimand_or_target_object, "estimand"),
    ]
    for field_name, env, claim_type in scalar_fields:
        if env is None:
            drafts.append(_absent_draft(claim_type, field_name))
        else:
            drafts.append(
                _draft_from_envelope(env, claim_type=claim_type, field_key=field_name)
            )

    # --- arrays (one row per element; empty -> a single not_found) ----------
    for i, el in enumerate(note.method_or_model):
        drafts.append(
            _draft_from_envelope(
                el,
                claim_type=_method_claim_type(el.method_type),
                field_key=f"method_or_model[{i}]",
                claim_subtype=el.method_type,
            )
        )
    if not note.method_or_model:
        drafts.append(_absent_draft("method", "method_or_model"))

    for i, el in enumerate(note.data_sources):
        drafts.append(
            _draft_from_envelope(el, claim_type="data", field_key=f"data_sources[{i}]")
        )
    if not note.data_sources:
        drafts.append(_absent_draft("data", "data_sources"))

    for i, el in enumerate(note.assumptions):
        drafts.append(
            _draft_from_envelope(
                el,
                claim_type=map_assumption_claim_type(el.assumption_type),
                field_key=f"assumptions[{i}]",
                claim_subtype=None,  # decision 47: assumptions never use claim_subtype
            )
        )
    if not note.assumptions:
        drafts.append(_absent_draft("general_assumption", "assumptions"))

    for i, el in enumerate(note.main_results):
        drafts.append(
            _draft_from_envelope(
                el,
                claim_type="result",
                field_key=f"main_results[{i}]",
                claim_subtype=el.result_type,
            )
        )
    if not note.main_results:
        drafts.append(_absent_draft("result", "main_results"))

    for i, el in enumerate(note.limitations):
        drafts.append(
            _draft_from_envelope(
                el,
                claim_type="limitation",
                field_key=f"limitations[{i}]",
                claim_subtype=el.limitation_type,
            )
        )
    if not note.limitations:
        drafts.append(_absent_draft("limitation", "limitations"))

    for i, el in enumerate(note.robustness_checks):
        drafts.append(
            _draft_from_envelope(
                el, claim_type="robustness_check", field_key=f"robustness_checks[{i}]"
            )
        )
    if not note.robustness_checks:
        drafts.append(_absent_draft("robustness_check", "robustness_checks"))

    for i, el in enumerate(note.open_questions):
        drafts.append(
            _draft_from_envelope(
                el, claim_type="open_question", field_key=f"open_questions[{i}]"
            )
        )
    if not note.open_questions:
        drafts.append(_absent_draft("open_question", "open_questions"))

    # Defensive: keep every claim_type within the closed vocab (passthrough known).
    for d in drafts:
        d.claim_type = normalize_claim_type(d.claim_type)

    note_text = _build_note_text(drafts)
    archetype = _derive_archetype(drafts)
    return drafts, note_text, archetype


def _build_note_text(drafts: list[ClaimDraft]) -> str:
    """Flattened summary used as the ``note_fts`` source (plan §4.2)."""
    parts: list[str] = []
    for d in drafts:
        if d.status == StatusValue.found.value:
            label = d.normalized_label or d.claim_type
            text = d.claim_text or d.exact_quote or ""
            parts.append(f"{label}: {text}".strip())
    return "\n".join(p for p in parts if p)


_FOUND = StatusValue.found.value

#: Decision-47 assumption family — any found member marks the note theoretical.
_ASSUMPTION_CLAIM_TYPES = frozenset(
    {"identification_assumption", "regularity_condition", "general_assumption"}
)

#: ``method_or_model`` elements land as these claim_types (:func:`_method_claim_type`).
_METHOD_CLAIM_TYPES = frozenset({"method", "model"})

#: Canonical emission order for the archetype multi-label (prototype
#: reader.py ``ARCHETYPE_ORDER``): theoretical, then empirical, then simulation.
_ARCHETYPE_ORDER: tuple[str, ...] = ("theoretical", "empirical", "simulation")

#: Simulation evidence in a method/model claim (prototype ``infer_archetype``).
_SIMULATION_RE = re.compile(
    r"\b(simulation|simulated|monte[\s-]?carlo|synthetic data)\b", re.IGNORECASE
)


def _derive_archetype(drafts: list[ClaimDraft]) -> str:
    """Derive the multi-label note archetype from the populated fields (decision 9).

    Canonical-ordered multi-label over ``{theoretical, empirical, simulation}``,
    joined with ``"+"`` in the fixed :data:`_ARCHETYPE_ORDER` (prototype
    reader.py ``infer_archetype``): a found ``data`` claim ⇒ ``empirical``; a
    found assumption-family claim ⇒ ``theoretical`` (labels combine, so
    theory+empirical is representable); the simulation regex firing on a found
    ``method_or_model`` claim's ``claim_text`` or ``claim_subtype`` ⇒
    ``simulation``; when no label fires, default ``empirical`` (prototype §5.2:
    results-but-no-proof-no-named-dataset reads as empirical). Zero found
    claims ⇒ ``"empty"`` — a deliberate deviation from v1's unconditional
    default label: honest for a claim-less note.
    """
    found = [d for d in drafts if d.status == _FOUND]
    if not found:
        return "empty"

    labels: set[str] = set()
    found_types = {d.claim_type for d in found}
    if "data" in found_types:
        labels.add("empirical")
    if found_types & _ASSUMPTION_CLAIM_TYPES:
        labels.add("theoretical")
    for d in found:
        if d.claim_type in _METHOD_CLAIM_TYPES and any(
            _SIMULATION_RE.search(t) for t in (d.claim_text, d.claim_subtype) if t
        ):
            labels.add("simulation")
            break

    if not labels:
        return "empirical"
    return "+".join(label for label in _ARCHETYPE_ORDER if label in labels)
