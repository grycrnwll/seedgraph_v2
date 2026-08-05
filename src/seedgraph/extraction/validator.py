"""JSON extraction + Pydantic validation for the default note (plan §5/§10 step 6).

``validate_note`` bracket-extracts the first balanced JSON object from raw model
text (tolerating prose preamble / code fences / trailing tokens), then validates
it against :class:`~seedgraph.extraction.schema.DefaultNoteV1`, returning
``(note | None, errors)``. It does NOT call the model: the single repair
re-prompt is orchestrated by the runner (which holds the LLM client) and feeds the
second response back through ``validate_note`` (plan §8). A second failure is
terminal and the runner records ``run_status='extraction_failed'`` with no note.
"""

from __future__ import annotations

import json
import types
from typing import Union, get_args, get_origin

from pydantic import ValidationError

# Thin re-export: the brace-depth extractor moved to the shared parser home
# (Build C chunk 1, D1); existing importers of the old location stay valid.
from ..llm.parse import extract_json_object
from .schema import (
    Assumption,
    ClaimEnvelope,
    DataSource,
    DefaultNoteV1,
    Limitation,
    MethodOrModel,
    Result,
)

__all__ = ["validate_note", "extract_json_object"]


def _is_str_annotation(ann: object) -> bool:
    """True for ``str`` and optional-str unions (``str | None``)."""
    if ann is str:
        return True
    if get_origin(ann) in (types.UnionType, Union):
        args = set(get_args(ann))
        return str in args and args <= {str, type(None)}
    return False


#: Every string-typed field across the note's claim envelopes, derived from the
#: Pydantic models themselves so any future string field gets the tolerant
#: coercion pass automatically (and non-string fields like ``confidence`` never do).
_ENVELOPE_STRING_FIELDS: frozenset[str] = frozenset(
    name
    for model in (ClaimEnvelope, MethodOrModel, DataSource, Assumption, Result, Limitation)
    for name, f in model.model_fields.items()
    if _is_str_annotation(f.annotation)
)


def _coerce_str_value(value: object) -> object:
    """Coerce an UNAMBIGUOUS malformed string value; return anything else unchanged.

    Live catch (first semantic run on test_proj, gpt-5.4-mini): the model
    stochastically emits a NON-STRING ``method_type`` in the note JSON — e.g.
    ``["synthetic control", "difference-in-differences"]`` — causing
    ``schema_validation_failed``; one paper (Synthetic DiD) failed 6 consecutive
    attempts because the repair re-prompt reproduced the same shape. Coercions:

    * non-empty list of strings  -> the single element, or ``"; "``-joined
    * single-key dict wrapping a string -> the wrapped string
    * ``None`` stays ``None`` (falls through untouched)

    Everything else (multi-key dicts, lists containing non-strings, empty lists,
    numbers, ...) is returned as-is so Pydantic still rejects genuinely unusable
    shapes.
    """
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return value[0] if len(value) == 1 else "; ".join(value)
    if isinstance(value, dict) and len(value) == 1:
        (inner,) = value.values()
        if isinstance(inner, str):
            return inner
    return value


def _coerce_note_strings(data: dict) -> dict:
    """Apply :func:`_coerce_str_value` to every envelope string field in ``data``.

    Walks the note's only two shapes — scalar envelope dicts and arrays of
    envelope dicts. Non-dict elements and unknown keys pass through untouched
    (``extra="ignore"`` / normal Pydantic validation handles them).
    """

    def fix_env(env: object) -> object:
        if not isinstance(env, dict):
            return env
        return {
            k: _coerce_str_value(v) if k in _ENVELOPE_STRING_FIELDS else v
            for k, v in env.items()
        }

    return {
        key: [fix_env(el) for el in value] if isinstance(value, list) else fix_env(value)
        for key, value in data.items()
    }


def validate_note(raw_text: str) -> tuple[DefaultNoteV1 | None, list[str]]:
    """Extract + validate one ``DefaultNoteV1`` from raw model text.

    Returns ``(note, [])`` on success or ``(None, errors)`` when no JSON object
    can be extracted or Pydantic validation fails. Pure (no network) so the
    runner can call it on both the first and the repaired response.
    """
    candidate = extract_json_object(raw_text)
    if candidate is None:
        return None, ["no JSON object found in model output"]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, [f"JSON decode error: {exc}"]
    if not isinstance(data, dict):
        return None, [f"top-level JSON is {type(data).__name__}, expected object"]
    # Tolerant pre-validation pass for unambiguous non-string malformations
    # (live catch: gpt-5.4-mini's stochastic list-valued method_type).
    data = _coerce_note_strings(data)
    try:
        note = DefaultNoteV1.model_validate(data)
    except ValidationError as exc:
        return None, [str(err) for err in exc.errors()] or [str(exc)]
    return note, []
