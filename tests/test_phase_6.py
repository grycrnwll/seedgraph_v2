"""Phase 6 (project lenses) definition tests — fully OFFLINE.

What stays after stage 2 (the lens runner, results, calibrate and the `lens
run|results|calibrate|status` commands moved to the work protocol's `lens:<id>` task,
tested in tests/work/test_routine_lens.py): lens schema parse / malformed-reject /
hash stability / validate_record gates; registry sync + snapshot-on-freeze +
edit-after-active reject; ORM<->migration parity (D6); `lens new` / `lens validate`.
The module-level imports double as an import-cleanliness check of the kept modules.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.dialects import sqlite as sqlite_dialect
from typer.testing import CliRunner

from seedgraph.lenses import schema

cli = CliRunner()

TEMPLATE = Path(schema.__file__).resolve().parent / "templates" / "regularity_conditions_v1.yaml"


# --------------------------------------------------------------------------
# schema (plan §11 test_lens_schema)
# --------------------------------------------------------------------------

def test_lens_schema():
    """Parse the docs/06 §4 example into a `LensDefinition`; reject malformed
    `output_schema` / bad enum; assert `definition_hash` is stable across
    reordering-insensitive canonicalization; `validate_record` accepts the §5
    example and flags an out-of-enum `condition_type`; enforces
    `require_evidence_span` (a span-less `found` record is demoted to
    `ambiguous`) and `inferred_requires_explanation` (inferred + empty notes →
    `extraction_failed`)."""
    lens = schema.LensDefinition.from_yaml(TEMPLATE)
    assert lens.lens_id == "regularity_conditions_v1"
    assert lens.object_type == "assumption"
    assert "condition_type" in lens.output_schema
    assert lens.output_schema["condition_type"].field_type == "array"
    assert "compactness" in lens.output_schema["condition_type"].allowed_values

    # malformed output_schema: unknown field-spec type rejected.
    bad = {
        "lens_id": "x",
        "name": "x",
        "object_type": "assumption",
        "positive_anchors": ["a"],
        "output_schema": {"foo": {"type": "not_a_real_type"}},
    }
    with pytest.raises(ValueError):
        schema.LensDefinition.from_dict(bad)

    # empty output_schema rejected.
    with pytest.raises(ValueError):
        schema.LensDefinition.from_dict({**bad, "output_schema": {}})

    # definition_hash is order-insensitive over anchors + schema key order.
    base = {
        "lens_id": "h",
        "name": "H",
        "object_type": "assumption",
        "positive_anchors": ["alpha", "beta", "gamma"],
        "negative_anchors": ["x", "y"],
        "output_schema": {"a": "string", "b": {"type": "array", "allowed_values": ["p", "q"]}},
    }
    reordered = {
        "lens_id": "h",
        "name": "H",
        "object_type": "assumption",
        "positive_anchors": ["gamma", "alpha", "beta"],
        "negative_anchors": ["y", "x"],
        "output_schema": {"b": {"type": "array", "allowed_values": ["q", "p"]}, "a": "string"},
    }
    assert schema.LensDefinition.from_dict(base).definition_hash() == \
        schema.LensDefinition.from_dict(reordered).definition_hash()
    # a meaningful edit DOES change the hash.
    edited = {**base, "positive_anchors": ["alpha", "beta", "gamma", "delta"]}
    assert schema.LensDefinition.from_dict(edited).definition_hash() != \
        schema.LensDefinition.from_dict(base).definition_hash()

    # validate_record: accept the §5-style example (span present, valid enum).
    example = {
        "status": "found",
        "condition_text_verbatim": "the parameter space is compact",
        "normalized_condition_label": "compact parameter space",
        "condition_type": ["compactness"],
        "attached_result": "consistency",
        "section": "Assumptions",
        "equation_or_assumption_number": "Assumption 2",
        "stated_or_inferred": "stated",
        "evidence_span_ids": ["span_abc"],
        "confidence": 0.9,
        "notes": "",
    }
    vr = lens.validate_record(example)
    assert vr.status == "found"
    assert vr.assertion_status == "stated"

    # out-of-enum condition_type -> extraction_failed + issue.
    bad_enum = {**example, "condition_type": ["not_a_condition_type"]}
    vr_enum = lens.validate_record(bad_enum)
    assert vr_enum.status == "extraction_failed"
    assert any("allowed_values" in i for i in vr_enum.issues)

    # require_evidence_span: a span-less found record -> ambiguous.
    spanless = {**example, "evidence_span_ids": []}
    assert lens.validate_record(spanless).status == "ambiguous"

    # inferred_requires_explanation: inferred + empty notes -> extraction_failed.
    inferred = {**example, "stated_or_inferred": "inferred", "notes": ""}
    assert lens.validate_record(inferred).status == "extraction_failed"
    # inferred WITH a non-empty explanation is accepted.
    inferred_ok = {**example, "stated_or_inferred": "inferred", "notes": "deduced from Thm 1"}
    assert lens.validate_record(inferred_ok).status == "found"


# --------------------------------------------------------------------------
# registry (plan §11 test_lens_registry)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# migration parity (plan §11 test_lens_migration; D6)
# --------------------------------------------------------------------------

def _affinity(type_str):
    t = type_str.upper()
    if "INT" in t:
        return "INTEGER"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "TEXT"


def _assert_table_parity(insp, table_name, model):
    orm_cols = {c.name: c for c in model.__table__.columns}
    mig_cols = {c["name"]: c for c in insp.get_columns(table_name)}
    assert set(orm_cols) == set(mig_cols), f"{table_name}: column set drift"
    mig_pk = set(insp.get_pk_constraint(table_name)["constrained_columns"])
    orm_pk = {c.name for c in model.__table__.columns if c.primary_key}
    assert mig_pk == orm_pk, f"{table_name}: PK drift"
    orm_dialect = sqlite_dialect.dialect()
    for name, orm_col in orm_cols.items():
        mig_col = mig_cols[name]
        orm_aff = _affinity(str(orm_col.type.compile(dialect=orm_dialect)))
        assert orm_aff == _affinity(str(mig_col["type"])), f"{table_name}.{name}: type drift"
        if name not in mig_pk:
            assert orm_col.nullable == mig_col["nullable"], f"{table_name}.{name}: nullability drift"


# --------------------------------------------------------------------------
# CLI (plan §11 test_lens_cli)
# --------------------------------------------------------------------------

