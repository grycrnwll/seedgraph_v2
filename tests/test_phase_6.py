"""Phase 6 (project lenses) definition tests — fully OFFLINE.

What stays after stage 2 (the lens runner, results, calibrate and the `lens
run|results|calibrate|status` commands moved to the work protocol's `lens:<id>` task,
tested in tests/work/test_routine_lens.py): lens schema parse / malformed-reject /
hash stability / validate_record gates; registry sync + snapshot-on-freeze +
edit-after-active reject; ORM<->migration parity (D6); `lens new` / `lens validate`.
The module-level imports double as an import-cleanliness check of the kept modules.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects import sqlite as sqlite_dialect
from sqlmodel import Session
from typer.testing import CliRunner

from seedgraph.cli import app
from seedgraph.db.migrations import discover_migrations
from seedgraph.db.models_project import ExtractionRun, Lens, LensOutput
from seedgraph.lenses import prompt, registry, schema  # noqa: F401 — import-cleanliness
from seedgraph.project import service

cli = CliRunner()

TEMPLATE = Path(schema.__file__).resolve().parent / "templates" / "regularity_conditions_v1.yaml"


def _install_lens(h, lens_id="regularity_conditions_v1", content=None):
    """Write the lens YAML into the project + sync; return (project_dir, lens)."""
    project_dir = h.root / "projects" / h.slug
    lens_dir = project_dir / "lenses"
    lens_dir.mkdir(parents=True, exist_ok=True)
    text = content if content is not None else TEMPLATE.read_text(encoding="utf-8")
    if lens_id != "regularity_conditions_v1" and content is None:
        text = text.replace("lens_id: regularity_conditions_v1", f"lens_id: {lens_id}")
    (lens_dir / f"{lens_id}.yaml").write_text(text, encoding="utf-8")
    with Session(h.engine) as session:
        registry.sync_lens(session, project_dir, lens_id)
    return project_dir, registry.load_lens(project_dir, lens_id)


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

def test_lens_registry():
    """`sync_lens` creates a `lenses` row with the correct `definition_hash` and
    NULL `definition_yaml` while `draft`; editing the YAML in `draft`/
    `calibrating` updates the hash (prior runs go stale, not deleted); promotion
    to `active` snapshots `definition_yaml` once; editing after `active` is
    rejected (requires a new `_v2` lens_id)."""
    from seedgraph.errors import ValidationError

    h = service.create_project("reg_registry")
    project_dir, lens = _install_lens(h)
    with Session(h.engine) as session:
        row = registry.get_lens_row(session, "regularity_conditions_v1")
        assert row is not None
        assert row.status == "draft"
        assert row.definition_yaml is None
        assert row.definition_hash == lens.definition_hash()

        # edit YAML while draft -> hash refreshes in place, definition_yaml stays NULL.
        text = (project_dir / "lenses" / "regularity_conditions_v1.yaml").read_text()
        text2 = text.replace("- continuity", "- continuity\n  - boundedness")
        (project_dir / "lenses" / "regularity_conditions_v1.yaml").write_text(text2)
        row2 = registry.sync_lens(session, project_dir, "regularity_conditions_v1")
        assert row2.definition_hash != row.definition_hash
        assert row2.definition_yaml is None

        # promote to active -> definition_yaml snapshotted once.
        lens2 = registry.load_lens(project_dir, "regularity_conditions_v1")
        promoted = registry.promote_to_active(session, lens2)
        assert promoted.status == "active"
        assert promoted.definition_yaml is not None

        # editing after active is rejected (immutability-on-use).
        text3 = text2.replace("- boundedness", "- boundedness\n  - coercivity")
        (project_dir / "lenses" / "regularity_conditions_v1.yaml").write_text(text3)
        with pytest.raises(ValidationError):
            registry.sync_lens(session, project_dir, "regularity_conditions_v1")


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


def test_lens_migration(tmp_path):
    """Applying `0009_lenses.sql` creates `lenses` + `lens_outputs` and adds
    `extraction_runs.lens_definition_hash`; the SQLModel metadata mirrors the
    migrated schema (D6; `create_all()` is not used to author schema)."""
    db = tmp_path / "parity6.db"
    conn = sqlite3.connect(str(db))
    for version, path in discover_migrations("project"):
        if version > 9:
            continue
        conn.executescript(path.read_text(encoding="utf-8"))
    conn.close()

    insp = inspect(create_engine(f"sqlite:///{db.as_posix()}"))
    names = insp.get_table_names()
    assert "lenses" in names and "lens_outputs" in names

    _assert_table_parity(insp, "lenses", Lens)
    _assert_table_parity(insp, "lens_outputs", LensOutput)

    # extraction_runs gained the (nullable) lens_definition_hash column at 0009.
    run_cols = {c["name"]: c for c in insp.get_columns("extraction_runs")}
    assert "lens_definition_hash" in run_cols
    assert run_cols["lens_definition_hash"]["nullable"] is True
    # schema_id is nullable (lens-origin runs leave it NULL; decision 22 / §4.3).
    assert run_cols["schema_id"]["nullable"] is True
    # D6 mirror: the ExtractionRun ORM now mirrors the post-0009 migrated
    # extraction_runs (incl. lens_definition_hash + the nullable schema_id), so
    # assert full ORM == migrated parity for the co-owned table this phase ALTERs.
    _assert_table_parity(insp, "extraction_runs", ExtractionRun)

    # lens_outputs FK targets + access_class default present.
    referred = {fk["referred_table"] for fk in insp.get_foreign_keys("lens_outputs")}
    assert {"extraction_runs", "lenses", "works", "extracted_claims"} <= referred
    lo_cols = {c["name"]: c for c in insp.get_columns("lens_outputs")}
    assert lo_cols["claim_id"]["nullable"] is True
    assert lo_cols["fields_json"]["nullable"] is False


# --------------------------------------------------------------------------
# CLI (plan §11 test_lens_cli)
# --------------------------------------------------------------------------

def test_lens_cli():
    """`lens validate` exits non-zero on bad YAML and 0 on the template; `lens new
    --from-template` scaffolds the project-local YAML."""
    h = service.create_project("reg_cli")

    # lens new --from-template scaffolds the YAML + registers it.
    res_new = cli.invoke(
        app, ["lens", "new", "regularity_conditions_v1",
               "--from-template", "regularity_conditions_v1", "--project", "reg_cli"]
    )
    assert res_new.exit_code == 0, res_new.output
    assert (h.root / "projects" / "reg_cli" / "lenses" / "regularity_conditions_v1.yaml").exists()

    # lens validate on a BAD yaml -> non-zero exit.
    bad_dir = h.root / "projects" / "reg_cli" / "lenses"
    (bad_dir / "broken_v1.yaml").write_text(
        "lens_id: broken_v1\nname: B\nobject_type: assumption\n"
        "positive_anchors: [a]\noutput_schema:\n  foo: {type: not_a_type}\n"
    )
    res_bad = cli.invoke(app, ["lens", "validate", "broken_v1", "--project", "reg_cli"])
    assert res_bad.exit_code != 0

    # lens validate on the good template -> exit 0.
    res_ok = cli.invoke(app, ["lens", "validate", "regularity_conditions_v1", "--project", "reg_cli"])
    assert res_ok.exit_code == 0

