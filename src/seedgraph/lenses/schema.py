"""Lens definition model + YAML load + record validation (plan §5/§6.2).

`LensDefinition` is the parsed, validated representation of a lens YAML
(docs/06 §4). It carries the field-specific instruction surface (object_type,
positive/negative anchors), a typed `output_schema` (a lightweight stdlib
field-spec map — NOT a dynamically generated Pydantic class, see §12 risk note),
and an `evidence_policy`.

Decisions implemented here:
- D2 — `validate_record` surfaces the record's author-assertion as
  `assertion_status` ({stated, inferred}); the origin `epistemic_type` is set by
  the runner, not here.
- decision 49/14 — the typed instance is preserved verbatim for `fields_json`;
  `object_type`→`claim_type` namespacing is the runner's job (uses the vocab
  helper), not the schema's.
- §6.2 evidence_policy gates — `require_evidence_span` (a span-less `found`
  record is demoted to `ambiguous`) and `inferred_requires_explanation`
  (`assertion_status='inferred'` with empty notes → `extraction_failed`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from ..ids import sha256_hex

# Closed set of field-spec scalar types the stdlib checker understands. A field
# whose ``type`` is outside this set is a malformed output_schema (from_yaml raises).
_KNOWN_FIELD_TYPES: frozenset[str] = frozenset(
    {"string", "string_or_null", "number", "integer", "boolean", "array", "object"}
)


# --- field-spec & policy sub-models ----------------------------------------

class FieldSpec(BaseModel):
    """One field of a lens `output_schema` (plan §12: stdlib field-spec, not a
    generated Pydantic class). Scalar shorthand (``field: string``) is normalized
    by :meth:`LensDefinition.from_yaml` into this shape.

    ``field_type`` accepts ``string|number|integer|boolean|array|object|
    string_or_null``; ``allowed_values`` enumerates a closed set (enum check);
    ``item_type`` types array elements.
    """

    field_type: str = Field(default="string", alias="type")
    allowed_values: list[Any] | None = None
    required: bool = False
    item_type: str | None = None

    model_config = {"populate_by_name": True, "extra": "ignore"}


class EvidencePolicy(BaseModel):
    """Post-validation gates (docs/06 §4 `evidence_policy`, plan §6.2/§8)."""

    require_evidence_span: bool = True
    allow_inferred: bool = True
    inferred_requires_explanation: bool = True
    record_not_found: bool = True


class ValidatedRecord(BaseModel):
    """Result of :meth:`LensDefinition.validate_record`.

    ``fields`` is the verbatim validated instance destined for
    ``lens_outputs.fields_json``. ``status`` reflects any evidence_policy
    demotion (found→ambiguous / →extraction_failed). ``assertion_status`` is the
    D2 author-assertion lifted from the record; ``issues`` lists the violations
    that triggered a demotion (empty when accepted as ``found``).
    """

    status: str
    fields: dict[str, Any]
    assertion_status: str | None = None
    evidence_span_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None
    notes: str | None = None
    issues: list[str] = Field(default_factory=list)


# --- the lens definition ----------------------------------------------------

class LensDefinition(BaseModel):
    """Parsed + validated lens YAML (docs/06 §4). Source of truth for
    `definition_hash` (the version key driving staleness/idempotency, §7)."""

    lens_id: str
    name: str
    scope: Literal["project"] = "project"
    description: str | None = None
    object_type: str
    positive_anchors: list[str]
    negative_anchors: list[str] = Field(default_factory=list)
    output_schema: dict[str, FieldSpec]
    evidence_policy: EvidencePolicy = Field(default_factory=EvidencePolicy)

    model_config = {"extra": "ignore"}

    # -- load ---------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path) -> "LensDefinition":
        """Load + validate a lens YAML file into a `LensDefinition`.

        Normalizes the heterogeneous docs/06 §4 `output_schema` forms (scalar
        shorthand ``field: string``, ``string_or_null``, and full
        ``{type, allowed_values}`` specs) into `FieldSpec` instances, then
        validates the whole document. Raises ``ValueError`` on malformed schema /
        bad enum (test_lens_schema).
        """
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"lens YAML {path} did not parse to a mapping")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "LensDefinition":
        """Build (+validate) a `LensDefinition` from an already-parsed mapping.

        The ``output_schema`` scalar shorthand is normalized to FieldSpec dicts;
        unknown field-spec types raise ``ValueError`` (malformed output_schema).
        """
        data = dict(data)
        raw_schema = data.get("output_schema")
        if not isinstance(raw_schema, dict) or not raw_schema:
            raise ValueError("lens output_schema must be a non-empty mapping")
        norm_schema: dict[str, dict] = {}
        for key, val in raw_schema.items():
            if isinstance(val, str):
                norm_schema[key] = {"type": val}
            elif isinstance(val, dict):
                norm_schema[key] = val
            else:
                raise ValueError(
                    f"output_schema.{key}: must be a scalar type name or a mapping, "
                    f"got {type(val).__name__}"
                )
        data["output_schema"] = norm_schema

        try:
            lens = cls.model_validate(data)
        except Exception as exc:  # noqa: BLE001 — surface as a clean ValueError
            raise ValueError(f"invalid lens definition: {exc}") from exc

        # Malformed field-spec type guard (closed set; plan §12 stdlib checker).
        for key, spec in lens.output_schema.items():
            if spec.field_type not in _KNOWN_FIELD_TYPES:
                raise ValueError(
                    f"output_schema.{key}: unknown field type {spec.field_type!r} "
                    f"(allowed: {sorted(_KNOWN_FIELD_TYPES)})"
                )
            if spec.allowed_values is not None and not isinstance(spec.allowed_values, list):
                raise ValueError(f"output_schema.{key}: allowed_values must be a list")
        return lens

    # -- hashing ------------------------------------------------------------

    def _canonical(self) -> dict:
        """Order-insensitive canonical view used for `definition_hash`.

        Anchor lists and each field's ``allowed_values`` are SORTED (they are
        semantically sets), and dict keys are emitted sorted by ``json.dumps`` —
        so reordering anchors / schema keys does NOT change the hash, while any
        meaningful edit does (plan §7 staleness key). The cosmetic ``description``
        is intentionally excluded so a docstring reflow never invalidates runs.
        """
        return {
            "lens_id": self.lens_id,
            "name": self.name,
            "scope": self.scope,
            "object_type": self.object_type,
            "positive_anchors": sorted(self.positive_anchors),
            "negative_anchors": sorted(self.negative_anchors),
            "output_schema": {
                key: {
                    "type": spec.field_type,
                    "allowed_values": (
                        sorted(spec.allowed_values, key=str)
                        if spec.allowed_values is not None
                        else None
                    ),
                    "required": spec.required,
                    "item_type": spec.item_type,
                }
                for key, spec in self.output_schema.items()
            },
            "evidence_policy": self.evidence_policy.model_dump(),
        }

    def definition_hash(self) -> str:
        """sha256 hex of the canonical, order-insensitive resolved YAML.

        This is the lens *version key*: equal definitions hash equal regardless
        of key/anchor ordering; any meaningful edit changes the hash, marking
        prior runs stale (decisions 18/22/77). Implements the §7 staleness key.
        """
        payload = json.dumps(self._canonical(), sort_keys=True, ensure_ascii=False)
        return sha256_hex(payload.encode("utf-8"))

    def definition_snapshot(self) -> str:
        """Frozen, deterministic JSON snapshot of the resolved definition.

        Stored into ``lenses.definition_yaml`` once on promotion to ``active``
        (decision 50 frozen reproducibility). Deterministic given the definition,
        so the snapshot is reproducible across rebuilds.
        """
        return json.dumps(self._canonical(), sort_keys=True, ensure_ascii=False, indent=2)

    # -- per-record validation + evidence_policy gates ----------------------

    def validate_record(self, obj: dict) -> ValidatedRecord:
        """Validate one extracted record against `output_schema` + enforce
        `evidence_policy`.

        Type/enum/required checks against `output_schema`; then the two policy
        gates (plan §6.2/§8): `require_evidence_span` demotes a span-less
        ``found`` record to ``ambiguous``; `inferred_requires_explanation` fails
        an ``assertion_status='inferred'`` record with empty notes to
        ``extraction_failed``. Lifts the record's stated/inferred value into the
        D2 `assertion_status` field of the returned `ValidatedRecord`.

        A type/enum/required violation is a hard validation failure
        (``extraction_failed``) — never silently dropped.
        """
        declared = (obj.get("status") or "found")
        assertion_status = obj.get("stated_or_inferred") or obj.get("assertion_status")
        notes = obj.get("notes")
        confidence = obj.get("confidence")
        raw_spans = obj.get("evidence_span_ids") or []
        evidence_span_ids = [s for s in raw_spans if s]

        issues: list[str] = []
        for key, spec in self.output_schema.items():
            present = key in obj and obj[key] is not None
            if not present:
                if spec.required:
                    issues.append(f"missing required field {key!r}")
                continue
            value = obj[key]
            ok, msg = _check_type(value, spec)
            if not ok:
                issues.append(f"field {key!r}: {msg}")
                continue
            if spec.allowed_values is not None:
                values = value if isinstance(value, list) else [value]
                for v in values:
                    if v not in spec.allowed_values:
                        issues.append(
                            f"field {key!r}: {v!r} not in allowed_values "
                            f"{spec.allowed_values}"
                        )

        def _result(status: str) -> ValidatedRecord:
            return ValidatedRecord(
                status=status,
                fields=obj,
                assertion_status=assertion_status,
                evidence_span_ids=evidence_span_ids,
                confidence=confidence,
                notes=notes,
                issues=issues,
            )

        # A type/enum/required failure is terminal (extraction_failed, no drop).
        if issues:
            return _result("extraction_failed")

        # not_found / not_applicable records are round-tripped verbatim; the found
        # evidence gates do not apply (decision 49 — preserve the not-found shape).
        if declared in ("not_found", "not_applicable"):
            return _result(declared)

        # evidence_policy gate 1: inferred_requires_explanation.
        if (
            self.evidence_policy.inferred_requires_explanation
            and assertion_status == "inferred"
            and not (isinstance(notes, str) and notes.strip())
        ):
            issues.append(
                "assertion_status='inferred' requires a non-empty 'notes' explanation "
                "(inferred_requires_explanation)"
            )
            return _result("extraction_failed")

        # evidence_policy gate 2: require_evidence_span (span-less found -> ambiguous).
        if self.evidence_policy.require_evidence_span and not evidence_span_ids:
            issues.append(
                "found record has no anchored evidence_span (require_evidence_span)"
            )
            return _result("ambiguous")

        return _result("found")


def _check_type(value: Any, spec: FieldSpec) -> tuple[bool, str]:
    """Lightweight stdlib type check for one field value (plan §12)."""
    t = spec.field_type
    if t == "string":
        return (isinstance(value, str), "expected string")
    if t == "string_or_null":
        return (value is None or isinstance(value, str), "expected string or null")
    if t == "number":
        return (isinstance(value, (int, float)) and not isinstance(value, bool), "expected number")
    if t == "integer":
        return (isinstance(value, int) and not isinstance(value, bool), "expected integer")
    if t == "boolean":
        return (isinstance(value, bool), "expected boolean")
    if t == "array":
        return (isinstance(value, list), "expected array")
    if t == "object":
        return (isinstance(value, dict), "expected object")
    return (True, "")
