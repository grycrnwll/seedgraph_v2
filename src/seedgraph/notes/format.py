"""The note format ``default_research_note_v1`` at schema 2.0.0 (design §3, §4).

Three pure steps, in the order the protocol runs them:

- :func:`validate_note` - bracket extraction (raw text) or a dict, the tolerant
  string coercion, then Pydantic. Pydantic stays the validator.
- :func:`check_note` - the notes task's own checks: required plain-words statements
  (§4.3), an inferred claim's explanation, and the fixed vocabularies (§4.1; an
  unknown value is a *warning*, never an error).
- :func:`normalize_parts` - one claim draft per scalar and per array element, explicit
  ``not_found`` rows, decision 47 assumption folding, vocabulary mapping, derived
  ``epistemic_type``, ``note_text`` and archetype. It takes the parts of a paper in
  order (one element for a whole paper): scalars come from part 1, arrays are
  concatenated with continuous ``field_key`` indices and ``source_chunk_index`` set
  to the 0-based part index (``None`` for a single part).

Changes from v2's 1.1.0 envelope: ``epistemic_type`` is gone from the agent-facing
models (ignored if sent, ``extra="ignore"``; the column is derived from
``assertion_status``); ``plain_words`` on ``Assumption``, ``MethodOrModel`` and the
estimand scalar.
"""

from __future__ import annotations

import json
import re
import types
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..llm.parse import extract_json_object
from ..vocab import AssertionStatus, EpistemicType, StatusValue, normalize_claim_type
from . import vocab

SCHEMA_ID: str = "default_research_note_v1"
#: 2.0.0 (v3 stage 2): fixed vocabularies, plain words, no agent-facing epistemic_type.
SCHEMA_VERSION: str = "2.0.0"
#: 2.0.0 (v3 stage 2): instructions.md field table, example and quote rules.
PROMPT_VERSION: str = "2.0.0"

_FOUND = StatusValue.found.value
_INFERRED = AssertionStatus.inferred.value


# --- Pydantic envelopes (agent-facing) ----------------------------------------------------

class ClaimEnvelope(BaseModel):
    """Per-claim envelope shared by every field and element."""

    model_config = ConfigDict(extra="ignore")

    claim_text: str | None = None
    normalized_label: str | None = None
    status: str = "found"
    assertion_status: str | None = None
    inferred_explanation: str | None = None
    confidence: float | None = None
    exact_quote: str | None = None


class EstimandEnvelope(ClaimEnvelope):
    """The ``estimand_or_target_object`` scalar."""

    plain_words: str | None = None


class MethodOrModel(ClaimEnvelope):
    method_type: str | None = None
    plain_words: str | None = None


class DataSource(ClaimEnvelope):
    pass


class Assumption(ClaimEnvelope):
    assumption_type: str | None = None
    plain_words: str | None = None


class Result(ClaimEnvelope):
    result_type: str | None = None


class Limitation(ClaimEnvelope):
    limitation_type: str | None = None


class NoteV2(BaseModel):
    """A validated note. Scalars may be null, arrays may be empty."""

    model_config = ConfigDict(extra="ignore")

    research_question: ClaimEnvelope | None = None
    main_contribution: ClaimEnvelope | None = None
    method_or_model: list[MethodOrModel] = Field(default_factory=list)
    data_sources: list[DataSource] = Field(default_factory=list)
    setting: ClaimEnvelope | None = None
    estimand_or_target_object: EstimandEnvelope | None = None
    assumptions: list[Assumption] = Field(default_factory=list)
    main_results: list[Result] = Field(default_factory=list)
    limitations: list[Limitation] = Field(default_factory=list)
    robustness_checks: list[ClaimEnvelope] = Field(default_factory=list)
    open_questions: list[ClaimEnvelope] = Field(default_factory=list)


#: Scalar fields in emission order -> claim_type. Part 1 owns these (design §1.3).
SCALAR_FIELDS: tuple[tuple[str, str], ...] = (
    ("research_question", "research_question"),
    ("main_contribution", "main_contribution"),
    ("setting", "setting"),
    ("estimand_or_target_object", "estimand"),
)
#: Array fields in emission order -> claim_type of their not_found row.
ARRAY_FIELDS: tuple[tuple[str, str], ...] = (
    ("method_or_model", "method"),
    ("data_sources", "data"),
    ("assumptions", "general_assumption"),
    ("main_results", "result"),
    ("limitations", "limitation"),
    ("robustness_checks", "robustness_check"),
    ("open_questions", "open_question"),
)
#: Array field -> its vocabulary-typed attribute.
_TYPED_ARRAYS: dict[str, str] = {
    "method_or_model": "method_type",
    "assumptions": "assumption_type",
    "main_results": "result_type",
    "limitations": "limitation_type",
}
_PLAIN_WORDS_ARRAYS = ("method_or_model", "assumptions")
_PLAIN_WORDS_SCALAR = "estimand_or_target_object"


# --- validation ---------------------------------------------------------------------------

def _is_str_annotation(ann: object) -> bool:
    if ann is str:
        return True
    if get_origin(ann) in (types.UnionType, Union):
        args = set(get_args(ann))
        return str in args and args <= {str, type(None)}
    return False


_STRING_FIELDS: frozenset[str] = frozenset(
    name
    for model in (ClaimEnvelope, EstimandEnvelope, MethodOrModel, DataSource, Assumption, Result,
                  Limitation)
    for name, f in model.model_fields.items()
    if _is_str_annotation(f.annotation)
)


def _coerce_str_value(value: object) -> object:
    """A list of strings -> one string (``"; "``-joined); a one-key dict wrapping a string
    -> that string; anything else unchanged (v2's tolerant pass, kept)."""
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return value[0] if len(value) == 1 else "; ".join(value)
    if isinstance(value, dict) and len(value) == 1:
        (inner,) = value.values()
        if isinstance(inner, str):
            return inner
    return value


def _coerce(data: dict) -> dict:
    def fix(env: object) -> object:
        if not isinstance(env, dict):
            return env
        return {k: _coerce_str_value(v) if k in _STRING_FIELDS else v for k, v in env.items()}

    return {k: [fix(e) for e in v] if isinstance(v, list) else fix(v) for k, v in data.items()}


def validate_note(raw: str | dict) -> tuple[NoteV2 | None, list[str]]:
    """``(note, [])`` or ``(None, errors)``. ``raw`` is model text or an already-parsed dict."""
    if isinstance(raw, dict):
        data: object = raw
    else:
        candidate = extract_json_object(raw)
        if candidate is None:
            return None, ["no JSON object found"]
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return None, [f"JSON decode error: {exc}"]
    if not isinstance(data, dict):
        return None, [f"top-level JSON is {type(data).__name__}, expected object"]
    try:
        return NoteV2.model_validate(_coerce(data)), []
    except ValidationError as exc:
        return None, [
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        ] or [str(exc)]


# --- task checks --------------------------------------------------------------------------

@dataclass(frozen=True)
class Issue:
    path: str  # e.g. "assumptions[2].plain_words"
    message: str


@dataclass(frozen=True)
class NoteCheck:
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


_PLAIN_WORDS_MESSAGE = (
    "plain_words is required for a found {what}: one or two sentences that state it "
    "without the paper's notation"
)


def _envelopes(note: NoteV2):
    """Yield ``(path, array_field | None, envelope)`` in emission order."""
    for name, _ in SCALAR_FIELDS:
        env = getattr(note, name)
        if env is not None:
            yield name, None, env
    for name, _ in ARRAY_FIELDS:
        for i, env in enumerate(getattr(note, name)):
            yield f"{name}[{i}]", name, env


def check_note(note: NoteV2) -> NoteCheck:
    """The notes task's own checks. Errors make the submission ``invalid``; warnings
    are feedback only."""
    errors: list[Issue] = []
    warnings: list[Issue] = []
    for path, array, env in _envelopes(note):
        found = _status(env.status) == _FOUND  # same reading as normalisation
        needs_plain = (array in _PLAIN_WORDS_ARRAYS) or (path == _PLAIN_WORDS_SCALAR)
        if found and needs_plain and not (getattr(env, "plain_words", None) or "").strip():
            what = {"method_or_model": "method or model", "assumptions": "assumption"}.get(
                array or "", "estimand or target object")
            errors.append(Issue(f"{path}.plain_words", _PLAIN_WORDS_MESSAGE.format(what=what)))
        if env.assertion_status == _INFERRED and not (env.inferred_explanation or "").strip():
            errors.append(Issue(
                f"{path}.inferred_explanation",
                "assertion_status is 'inferred', so inferred_explanation must say how it was inferred",
            ))
        if array in _TYPED_ARRAYS:
            attr = _TYPED_ARRAYS[array]
            match = vocab.map_label(attr, getattr(env, attr))
            if not match.known:
                allowed = ", ".join(vocab.VOCABULARIES[attr])
                warnings.append(Issue(
                    f"{path}.{attr}",
                    f"{match.raw!r} is not a known {attr}; saved as {match.value!r}. "
                    f"Allowed values: {allowed}",
                ))
    return NoteCheck(errors=errors, warnings=warnings)


# --- normalisation ------------------------------------------------------------------------

@dataclass
class ClaimDraft:
    """One pending ``extracted_claims`` row."""

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
    plain_words: str | None = None
    source_chunk_index: int | None = None


@dataclass(frozen=True)
class NormalizedNote:
    drafts: list[ClaimDraft]
    note_text: str
    archetype: str


def _status(value: str | None) -> str:
    try:
        return StatusValue(value).value
    except (ValueError, TypeError):
        return _FOUND


def _draft(env: ClaimEnvelope, *, claim_type: str, field_key: str, subtype: str | None,
           chunk: int | None) -> ClaimDraft:
    assertion = env.assertion_status
    if assertion not in (None, AssertionStatus.stated.value, _INFERRED):
        assertion = None
    epistemic = (EpistemicType.llm_inferred if assertion == _INFERRED
                 else EpistemicType.llm_extracted).value
    plain = getattr(env, "plain_words", None)
    return ClaimDraft(
        claim_type=normalize_claim_type(claim_type), field_key=field_key,
        status=_status(env.status), epistemic_type=epistemic, claim_subtype=subtype,
        normalized_label=env.normalized_label, claim_text=env.claim_text,
        assertion_status=assertion, inferred_explanation=env.inferred_explanation,
        confidence=env.confidence, exact_quote=env.exact_quote,
        plain_words=plain.strip() if plain and plain.strip() else None,
        source_chunk_index=chunk,
    )


def _absent(claim_type: str, field_key: str) -> ClaimDraft:
    return ClaimDraft(claim_type=claim_type, field_key=field_key,
                      status=StatusValue.not_found.value,
                      epistemic_type=EpistemicType.llm_extracted.value)


_ASSUMPTION_CLAIM_TYPE = {
    "identification": "identification_assumption",
    "regularity": "regularity_condition",
    "general": "general_assumption",
}


def _array_types(name: str, env: ClaimEnvelope) -> tuple[str, str | None]:
    """``(claim_type, claim_subtype)`` for one array element."""
    if name == "assumptions":  # decision 47: folds into claim_type, never claim_subtype
        value = vocab.map_label("assumption_type", env.assumption_type).value or "general"
        return _ASSUMPTION_CLAIM_TYPE[value], None
    if name == "method_or_model":
        sub = vocab.map_label("method_type", env.method_type).value
        return ("model" if sub == "model" else "method"), sub
    if name == "main_results":
        return "result", vocab.map_label("result_type", env.result_type).value
    if name == "limitations":
        return "limitation", vocab.map_label("limitation_type", env.limitation_type).value
    return dict(ARRAY_FIELDS)[name], None


def normalize_parts(parts: Sequence[NoteV2]) -> NormalizedNote:
    """Flatten a paper's parts (in part order; one element for a whole paper)."""
    if not parts:
        raise ValueError("normalize_parts needs at least one part")
    multi = len(parts) > 1
    drafts: list[ClaimDraft] = []
    first = parts[0]
    for name, claim_type in SCALAR_FIELDS:
        env = getattr(first, name)
        if env is None:
            drafts.append(_absent(claim_type, name))
        else:
            drafts.append(_draft(env, claim_type=claim_type, field_key=name, subtype=None,
                                 chunk=0 if multi else None))
    for name, absent_type in ARRAY_FIELDS:
        i = 0
        for part_index, part in enumerate(parts):
            for env in getattr(part, name):
                claim_type, subtype = _array_types(name, env)
                drafts.append(_draft(env, claim_type=claim_type, field_key=f"{name}[{i}]",
                                     subtype=subtype, chunk=part_index if multi else None))
                i += 1
        if i == 0:
            drafts.append(_absent(absent_type, name))
    return NormalizedNote(drafts=drafts, note_text=_note_text(drafts),
                          archetype=_archetype(drafts))


def normalize_note(note: NoteV2) -> NormalizedNote:
    """A whole (single-part) paper: ``normalize_parts([note])``."""
    return normalize_parts([note])


def _note_text(drafts: list[ClaimDraft]) -> str:
    """The ``note_fts`` source, as v2."""
    lines = []
    for d in drafts:
        if d.status == _FOUND:
            line = f"{d.normalized_label or d.claim_type}: {d.claim_text or d.exact_quote or ''}".strip()
            if line:
                lines.append(line)
    return "\n".join(lines)


_ASSUMPTION_TYPES = frozenset(_ASSUMPTION_CLAIM_TYPE.values())
_SIMULATION_RE = re.compile(r"\b(simulation|simulated|monte[\s-]?carlo|synthetic data)\b",
                            re.IGNORECASE)


def _archetype(drafts: list[ClaimDraft]) -> str:
    """v2's decision 9 multi-label over theoretical / empirical / simulation."""
    found = [d for d in drafts if d.status == _FOUND]
    if not found:
        return "empty"
    labels: set[str] = set()
    found_types = {d.claim_type for d in found}
    if "data" in found_types:
        labels.add("empirical")
    if found_types & _ASSUMPTION_TYPES:
        labels.add("theoretical")
    if any(d.claim_type in ("method", "model")
           and any(_SIMULATION_RE.search(t) for t in (d.claim_text, d.claim_subtype) if t)
           for d in found):
        labels.add("simulation")
    if not labels:
        return "empirical"
    return "+".join(x for x in ("theoretical", "empirical", "simulation") if x in labels)
