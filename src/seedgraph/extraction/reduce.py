"""Phase 4b — deterministic reduce / merge of per-chunk claim drafts (plan §4.4).

The *reduce* half of phase_4b's map-reduce. Given the per-chunk lists of
``ClaimDraft`` produced by the map step (each reusing the phase-4
``default_research_note_v1`` schema/validator/normalizer verbatim), merge them
into ONE note's worth of claims plus a flattened ``note_text`` and derived
``archetype`` — exactly the triple phase-4 ``normalize_note`` returns, so the
written artifact is schema-identical to a phase-4 note (decision D4: a single
merged ``structured_notes`` row per oversize paper).

The merge is **pure and deterministic** — there is deliberately NO second LLM
"summarize the summaries" pass (plan §2 out-of-scope; avoids a second
hallucination surface). The contract (plan §4.4):

* **Array fields** (method_or_model, data_sources, assumptions, main_results,
  limitations, robustness_checks, open_questions): union all found/inferred
  element claims across chunks; dedup by ``(claim_type, normalized_label_casefold)``
  — on a duplicate keep the highest-confidence draft and **union its spans** (so
  evidence found in multiple chunks survives — plan §12 lost-evidence mitigation);
  divergent ``claim_text`` for the same key at comparable confidence stays as
  separate rows unless ``normalized_label`` matches exactly.
* **Scalar fields** (research_question, main_contribution, setting,
  estimand_or_target_object): select the single highest-confidence ``found`` draft
  across chunks; ties broken by earliest ``chunk_index`` (deterministic). Losing
  drafts are dropped (recoverable via map-run provenance), not emitted as dups.
* **not_found / not_applicable**: emitted only when NO chunk produced a
  found/inferred/ambiguous draft for that field; a field found in any chunk is
  found for the paper.
* **epistemic_type / assertion_status** (decision D2) are carried from the
  winning draft; an inferred winner keeps ``epistemic_type='llm_inferred'`` +
  ``assertion_status='inferred'`` + ``inferred_explanation``.
* Every surviving claim records ``source_chunk_index`` + ``source_extraction_run_id``
  of its winning draft (the new phase_4b provenance columns, migration §4.1).

Decisions implemented: D4 (one merged note), D2 (two orthogonal provenance fields
carried through the merge).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..vocab import StatusValue
from .normalize import (
    ClaimDraft,
    _build_note_text,
    _derive_archetype,
)

# Scalar field families (one merged claim each); array families are merged
# element-wise. Mirrors phase-4 ``normalize.normalize_note`` field grouping.
_SCALAR_FIELDS: tuple[str, ...] = (
    "research_question",
    "main_contribution",
    "setting",
    "estimand_or_target_object",
)
_ARRAY_FIELDS: tuple[str, ...] = (
    "method_or_model",
    "data_sources",
    "assumptions",
    "main_results",
    "limitations",
    "robustness_checks",
    "open_questions",
)

_FOUND = StatusValue.found.value
_AMBIGUOUS = StatusValue.ambiguous.value
_NOT_FOUND = StatusValue.not_found.value
# A draft "found the field" (so a not_found is NOT emitted) when its status is one
# of these — i.e. anything other than not_found / not_applicable.
_SUBSTANTIVE = (_FOUND, _AMBIGUOUS)


@dataclass
class MergedDraft(ClaimDraft):
    """A merged claim — a phase-4 :class:`ClaimDraft` plus phase-4b provenance.

    Adds the union of contributing chunk quotes (``exact_quotes`` — so evidence
    from multiple chunks survives the reduce; plan §4.4 span union) and the
    winning draft's chunk origin (``source_chunk_index`` / ``source_extraction_run_id``,
    the new 0008 provenance columns). The inherited singular ``exact_quote`` is the
    winner's quote (kept so phase-4 ``_build_note_text`` works unchanged); the
    runner anchors EVERY quote in ``exact_quotes`` to retain all spans.
    """

    exact_quotes: list[str] = field(default_factory=list)
    source_chunk_index: int | None = None
    source_extraction_run_id: str | None = None


def _conf(draft: ClaimDraft) -> float:
    """Confidence sort key — a missing confidence sorts below any real value."""
    return draft.confidence if draft.confidence is not None else -1.0


def _quote_list(draft: ClaimDraft) -> list[str]:
    q = (draft.exact_quote or "").strip()
    return [draft.exact_quote] if q else []


def _to_merged(
    draft: ClaimDraft,
    *,
    chunk_index: int,
    source_run_ids: list[str] | None,
    field_key: str,
    exact_quotes: list[str],
) -> MergedDraft:
    run_id = (
        source_run_ids[chunk_index]
        if source_run_ids is not None and 0 <= chunk_index < len(source_run_ids)
        else None
    )
    return MergedDraft(
        claim_type=draft.claim_type,
        field_key=field_key,
        status=draft.status,
        epistemic_type=draft.epistemic_type,
        claim_subtype=draft.claim_subtype,
        normalized_label=draft.normalized_label,
        claim_text=draft.claim_text,
        assertion_status=draft.assertion_status,
        inferred_explanation=draft.inferred_explanation,
        confidence=draft.confidence,
        exact_quote=exact_quotes[0] if exact_quotes else None,
        exact_quotes=list(exact_quotes),
        source_chunk_index=chunk_index,
        source_extraction_run_id=run_id,
    )


def _not_found_draft(claim_type: str, field_key: str) -> MergedDraft:
    """A single explicit ``not_found`` merged claim for a field absent in all chunks."""
    from ..vocab import EpistemicType

    return MergedDraft(
        claim_type=claim_type,
        field_key=field_key,
        status=_NOT_FOUND,
        epistemic_type=EpistemicType.llm_extracted.value,
        exact_quotes=[],
        source_chunk_index=None,
    )


def merge_chunk_drafts(
    per_chunk_drafts: list[list["ClaimDraft"]],
    *,
    source_run_ids: list[str] | None = None,
) -> tuple[list["ClaimDraft"], str, str]:
    """Merge per-chunk claim drafts into one note's claims, note_text, and archetype.

    Implements the deterministic merge contract of plan §4.4 (array union/dedup +
    span union, scalar best-confidence selection with ``chunk_index`` tie-break,
    not_found only when absent in every chunk, D2 provenance carried from the
    winner, ``source_chunk_index``/``source_extraction_run_id`` stamped). Returns
    the same ``(claims, note_text, archetype)`` triple shape as the phase-4
    ``normalize_note`` so downstream writing/anchoring is identical.

    Args:
        per_chunk_drafts: one inner list of ``ClaimDraft`` per successfully
            extracted map chunk, in ``chunk_index`` order. Failed/empty chunks
            contribute an empty list (their absence must not manufacture a
            ``not_found`` if another chunk found the field).

    Returns:
        ``(merged_claims, note_text, archetype)`` — ``merged_claims`` is the
        deduped/aggregated claim list (one ``not_found`` per field absent in all
        chunks); ``note_text`` is the flattened summary feeding ``note_fts``;
        ``archetype`` is derived from the populated fields (phase-4 rule reused).

    Pure: no I/O, no LLM call, no DB access. Implements D4, D2.

    ``source_run_ids`` (optional, phase-4b additive) is the per-chunk ``chunked_map``
    run id list aligned to ``per_chunk_drafts``; when given, each surviving claim is
    stamped with ``source_extraction_run_id`` of its winning draft's chunk.
    """
    # Index drafts by (chunk_index, draft) so every selection is deterministic and
    # the winner's chunk origin is recoverable.
    tagged: list[tuple[int, ClaimDraft]] = []
    for chunk_index, drafts in enumerate(per_chunk_drafts):
        for draft in drafts:
            tagged.append((chunk_index, draft))

    merged: list[MergedDraft] = []

    # --- scalar fields: single highest-confidence found draft (chunk_index tie) --
    for field_name in _SCALAR_FIELDS:
        candidates = [(ci, d) for ci, d in tagged if d.field_key == field_name]
        found = [(ci, d) for ci, d in candidates if d.status == _FOUND]
        ambiguous = [(ci, d) for ci, d in candidates if d.status == _AMBIGUOUS]
        if found:
            # Highest confidence; ties broken by earliest chunk_index (deterministic).
            ci, winner = min(found, key=lambda t: (-_conf(t[1]), t[0]))
            merged.append(
                _to_merged(
                    winner,
                    chunk_index=ci,
                    source_run_ids=source_run_ids,
                    field_key=field_name,
                    exact_quotes=_quote_list(winner),
                )
            )
        elif ambiguous:
            ci, winner = min(ambiguous, key=lambda t: (-_conf(t[1]), t[0]))
            merged.append(
                _to_merged(
                    winner,
                    chunk_index=ci,
                    source_run_ids=source_run_ids,
                    field_key=field_name,
                    exact_quotes=[],  # ambiguous never anchors (phase-4 rule)
                )
            )
        else:
            claim_type = candidates[0][1].claim_type if candidates else _SCALAR_TYPE[field_name]
            merged.append(_not_found_draft(claim_type, field_name))

    # --- array fields: union elements, dedup by (claim_type, normalized_label) ---
    for family in _ARRAY_FIELDS:
        # Element drafts are keyed ``family[i]``; an empty array in a chunk emits a
        # single not_found marker keyed ``family`` (no index) — exclude those from
        # the element set but use them to decide the all-absent case.
        elements = [
            (ci, d)
            for ci, d in tagged
            if d.field_key.startswith(f"{family}[") and d.status in _SUBSTANTIVE
        ]
        if not elements:
            claim_type = _ARRAY_TYPE[family]
            merged.append(_not_found_draft(claim_type, family))
            continue

        # Dedup by (claim_type, normalized_label.casefold()); fall back to claim_text
        # so distinct unlabeled elements are NOT collapsed (plan §4.4). Preserve
        # first-appearance order (chunk order then element order).
        groups: dict[tuple[str, str], list[tuple[int, ClaimDraft]]] = {}
        order: list[tuple[str, str]] = []
        for ci, d in elements:
            label = (d.normalized_label or d.claim_text or "").strip().casefold()
            key = (d.claim_type, label)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((ci, d))

        for k, group in ((k, groups[k]) for k in order):
            # Winner: highest confidence, earliest chunk on a tie.
            ci, winner = min(group, key=lambda t: (-_conf(t[1]), t[0]))
            # Union spans across ALL contributing chunks (dedup, ordered by chunk
            # then occurrence) — a merged claim NEVER loses a contributing span.
            quotes: list[str] = []
            for gci, gd in sorted(group, key=lambda t: t[0]):
                for q in _quote_list(gd):
                    if q not in quotes:
                        quotes.append(q)
            field_key = f"{family}[{len(merged_in_family(merged, family))}]"
            merged.append(
                _to_merged(
                    winner,
                    chunk_index=ci,
                    source_run_ids=source_run_ids,
                    field_key=field_key,
                    exact_quotes=quotes,
                )
            )

    note_text = _build_note_text(merged)
    archetype = _derive_archetype(merged)
    return merged, note_text, archetype


def merged_in_family(merged: list["MergedDraft"], family: str) -> list["MergedDraft"]:
    """Merged claims already emitted for an array ``family`` (for sequential indexing)."""
    return [m for m in merged if m.field_key.startswith(f"{family}[")]


# claim_type for an absent scalar / empty array (phase-4 normalize parity).
_SCALAR_TYPE: dict[str, str] = {
    "research_question": "research_question",
    "main_contribution": "main_contribution",
    "setting": "setting",
    "estimand_or_target_object": "estimand",
}
_ARRAY_TYPE: dict[str, str] = {
    "method_or_model": "method",
    "data_sources": "data",
    "assumptions": "general_assumption",
    "main_results": "result",
    "limitations": "limitation",
    "robustness_checks": "robustness_check",
    "open_questions": "open_question",
}
