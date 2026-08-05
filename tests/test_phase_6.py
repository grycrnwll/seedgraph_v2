"""Phase 6 (project lenses) acceptance tests — fully OFFLINE.

Every plan §11 acceptance check is implemented with real assertions and runs with
NO network, NO real LLM, NO key: a synthetic markdown is pushed through the real
cache (ingest + FakeMarkerBackend convert), bridged work->markdown, then a lens is
authored as project-local YAML, synced, and run through the phase-6 surface with a
deterministic FakeLLMBackend that returns canned strict-JSON records. The
module-level imports double as a whole-phase import-cleanliness check.

Contracts exercised (plan §11): lens schema parse / malformed-reject / hash
stability / validate_record gates; registry sync + snapshot-on-freeze +
edit-after-active reject; runner found/not_found/invalid-JSON/access-class/anchor/
no-markdown; idempotency no-op + --force + un-skip on YAML/markdown change;
deterministic no-LLM fallback honesty; ORM<->migration parity (D6); results +
coverage + staleness; CLI validate exit-code + results --json; the doc-10 §10
milestone end-to-end.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects import sqlite as sqlite_dialect
from sqlmodel import Session
from typer.testing import CliRunner

from seedgraph import anchor, cache_access
from seedgraph.acquisition.bridge import resolve_work_markdown, write_bridge
from seedgraph.cache.convert import convert_source_file
from seedgraph.cache.ingest import ingest_file
from seedgraph.cache.marker_backend import FakeMarkerBackend
from seedgraph.cli import app
from seedgraph.db.migrations import discover_migrations
from seedgraph.db.models_project import ExtractionRun, Lens, LensOutput
from seedgraph.lenses import (  # noqa: F401 — whole-package import-cleanliness
    calibrate,
    fallback,
    prompt,
    registry,
    results,
    runner,
    schema,
)
from seedgraph.lenses.runner import LensRouter, run_lens
from seedgraph.llm.backend import FakeLLMBackend, LLMCompletion
from seedgraph.project import service
from seedgraph.vocab import AccessClass, AcquisitionMethod, is_shareable, lens_claim_type

cli = CliRunner()

TEMPLATE = Path(schema.__file__).resolve().parent / "templates" / "regularity_conditions_v1.yaml"

# A paper that carries regularity conditions; each found quote is a UNIQUE verbatim
# substring so the phase_3 anchor path locates it exactly.
MD_REG = (
    "# Asymptotic Theory for a GMM Estimator\n\n"
    "## Assumptions\n\n"
    "The parameter space is compact.\n\n"
    "The moment matrix has full rank.\n\n"
    "## Identification\n\n"
    "The estimator is consistent under standard regularity conditions.\n"
)

# A paper with NO regularity conditions / positive anchors.
MD_PLAIN = (
    "# A Survey of Minimum Wage Studies\n\n"
    "## Introduction\n\n"
    "We summarize the empirical literature on labor market effects.\n"
)


# --------------------------------------------------------------------------
# Offline fixtures / helpers
# --------------------------------------------------------------------------

def _add_work(h, tag, markdown, *, access_class=AccessClass.open_access, doi=None):
    """Add one work + bridged converted markdown; return (work_id, md)."""
    method = (
        AcquisitionMethod.open_access_fetch
        if access_class == AccessClass.open_access
        else AcquisitionMethod.upload
    )
    w = service.add_work(h, ids={"doi": doi or f"10.1/{tag}"}, title="W " + tag)
    p = Path(os.environ["SEEDGRAPH_HOME"]) / f"{tag}.pdf"
    p.write_bytes(b"%PDF-1.4 " + tag.encode() + b" body content here")
    src = ingest_file(p, access_class=access_class, acquisition_method=method, root=None)
    md = convert_source_file(
        src.source_file_id, backend=FakeMarkerBackend(markdown=markdown), root=None
    )
    with Session(h.engine) as s:
        write_bridge(
            s,
            work_id=w.work_id,
            source_file_id=src.source_file_id,
            file_hash=src.file_hash,
            markdown_id=md.markdown_id,
            markdown_hash=md.markdown_hash,
            acquisition_method=method.value,
        )
        s.commit()
    return w.work_id, md


def _rebridge(h, work_id, markdown, nonce):
    """Re-point a work's bridge at a GENUINELY new markdown (distinct source bytes,
    so convert produces a new markdown_id/hash rather than returning the cache)."""
    method = AcquisitionMethod.open_access_fetch
    p = Path(os.environ["SEEDGRAPH_HOME"]) / f"{nonce}.pdf"
    p.write_bytes(b"%PDF-1.4 " + nonce.encode() + b" CHANGED distinct body bytes")
    src = ingest_file(p, access_class=AccessClass.open_access, acquisition_method=method, root=None)
    md = convert_source_file(
        src.source_file_id, backend=FakeMarkerBackend(markdown=markdown), root=None
    )
    with Session(h.engine) as s:
        write_bridge(
            s,
            work_id=work_id,
            source_file_id=src.source_file_id,
            file_hash=src.file_hash,
            markdown_id=md.markdown_id,
            markdown_hash=md.markdown_hash,
            acquisition_method=method.value,
        )
        s.commit()
    return md


def _add_metadata_only_work(h, tag):
    """Add an included work with NO source/bridge (metadata-only)."""
    w = service.add_work(h, ids={"doi": f"10.1/{tag}"}, title="W " + tag)
    return w.work_id


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


def _router(backend):
    return LensRouter(config=GlobalConfigFactory(), backend=backend)


def GlobalConfigFactory():
    from seedgraph.config.models import GlobalConfig

    return GlobalConfig()  # default profiles+routes now include project_lens_extraction


def _conn(h):
    c = sqlite3.connect(str(h.db_path))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    return c


def _found_resp():
    return {
        "records": [
            {
                "status": "found",
                "condition_text_verbatim": "The parameter space is compact.",
                "normalized_condition_label": "compact parameter space",
                "condition_type": ["compactness"],
                "attached_result": "consistency of the estimator",
                "section": "Assumptions",
                "equation_or_assumption_number": "Assumption 1",
                "stated_or_inferred": "stated",
                "evidence_span_ids": [],
                "confidence": 0.95,
                "notes": "",
            }
        ],
        "not_found": {"searched_sections": ["Assumptions"], "notes": "", "confidence": 0.0},
    }


def _not_found_resp():
    return {
        "records": [],
        "not_found": {
            "searched_sections": ["Introduction"],
            "notes": "no regularity conditions present",
            "confidence": 0.2,
        },
    }


@dataclass
class _ContentAwareBackend:
    """A deterministic offline backend that returns found/not_found by markdown
    content (so a multi-work CLI run gets the right record per work regardless of
    order). Records every call so router-call assertions hold."""

    calls: list = field(default_factory=list)

    def complete(self, system_prompt, user_prompt, *, model=None, temperature=0.0, max_tokens=4096):
        self.calls.append({"user": user_prompt, "model": model})
        body = _found_resp() if "The parameter space is compact." in user_prompt else _not_found_resp()
        text = json.dumps(body)
        return LLMCompletion(text=text, input_tokens=12, output_tokens=12)


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
# runner (plan §11 test_lens_runner)
# --------------------------------------------------------------------------

def test_lens_runner():
    """With a mock backend: work→markdown resolves via `work_source_files`; a found
    record yields `lens_outputs(found)` + `extracted_claims(claim_type=
    'general_assumption')` with `condition_type[]` in `fields_json` +
    `claim_spans(rank=0)` + a `claim_fts` row; a not-found work yields
    `lens_outputs(not_found, claim_id NULL)` with `searched_sections[]` in
    `fields_json`; invalid JSON → `extraction_failed` + a lens_record review item;
    `access_class` is stamped fail-closed; the verbatim quote anchors to an
    `evidence_span` whose `quote_hash` matches; a no-markdown work is skipped and
    recorded in `skipped_no_markdown` (NOT `extraction_failed`)."""
    h = service.create_project("reg_runner")
    # wReg is INGESTED PRIVATE so we also assert private-by-default stamping.
    w_reg, md_reg = _add_work(h, "reg", MD_REG, access_class=AccessClass.user_supplied_private)
    w_plain, _ = _add_work(h, "plain", MD_PLAIN)
    w_nomd = _add_metadata_only_work(h, "nomd")
    _, lens = _install_lens(h)

    fake = FakeLLMBackend(responses=[_found_resp(), _not_found_resp()])
    with Session(h.engine) as session:
        result = run_lens(session, None, lens, [w_reg, w_plain, w_nomd], _router(fake))

    assert result.found == 1
    assert result.not_found == 1
    assert result.skipped_no_markdown == [w_nomd]
    assert result.extraction_failed == 0
    assert result.router_calls == 2  # only the two markdown-bearing works

    c = _conn(h)
    try:
        # found -> lens_outputs(found) with claim_id + condition_type in fields_json.
        lo = c.execute(
            "SELECT * FROM lens_outputs WHERE work_id=? AND status='found'", (w_reg,)
        ).fetchone()
        assert lo is not None
        assert lo["claim_id"] is not None
        fields = json.loads(lo["fields_json"])
        assert fields["condition_type"] == ["compactness"]
        # access_class fail-closed: private source -> private stamp, withheld by export.
        assert lo["access_class"] == AccessClass.user_supplied_private.value
        assert is_shareable(lo["access_class"]) is False

        # extracted_claims(claim_type='general_assumption') + provenance + access.
        claim = c.execute(
            "SELECT * FROM extracted_claims WHERE claim_id=?", (lo["claim_id"],)
        ).fetchone()
        assert claim["claim_type"] == "general_assumption"
        assert claim["epistemic_type"] == "llm_extracted"
        assert claim["assertion_status"] == "stated"
        assert claim["claim_text"] == "The parameter space is compact."
        assert claim["access_class"] == AccessClass.user_supplied_private.value
        assert claim["structured_note_id"] is None  # lens claims mint no note header

        # claim_spans(rank=0) + the span's quote_hash matches the verbatim quote.
        cs = c.execute(
            "SELECT span_id, rank FROM claim_spans WHERE claim_id=?", (lo["claim_id"],)
        ).fetchall()
        assert len(cs) == 1 and cs[0]["rank"] == 0
        span = c.execute(
            "SELECT quote_hash, exact_quote FROM evidence_spans WHERE span_id=?", (cs[0]["span_id"],)
        ).fetchone()
        assert span["quote_hash"] == anchor.quote_hash("The parameter space is compact.")

        # claim_fts row present for the lens claim.
        fts = c.execute(
            "SELECT claim_id FROM claim_fts WHERE claim_id=?", (lo["claim_id"],)
        ).fetchone()
        assert fts is not None

        # not_found work -> lens_outputs(not_found, claim_id NULL) + searched_sections.
        nf = c.execute(
            "SELECT * FROM lens_outputs WHERE work_id=? AND status='not_found'", (w_plain,)
        ).fetchone()
        assert nf is not None and nf["claim_id"] is None
        nf_fields = json.loads(nf["fields_json"])
        assert nf_fields["searched_sections"] == ["Introduction"]

        # the extraction_run carries the lens stamp + NULL schema_id (sibling schema,
        # mutually exclusive w/ schema_id; decision 22 / §4.3).
        run = c.execute(
            "SELECT lens_id, lens_definition_hash, schema_id FROM extraction_runs WHERE work_id=?",
            (w_reg,),
        ).fetchone()
        assert run["lens_id"] == "regularity_conditions_v1"
        assert run["lens_definition_hash"] == lens.definition_hash()
        assert run["schema_id"] is None

        # no extraction_run minted for the no-markdown work (skipped, not failed).
        assert c.execute(
            "SELECT COUNT(*) FROM extraction_runs WHERE work_id=?", (w_nomd,)
        ).fetchone()[0] == 0
    finally:
        c.close()

    # invalid JSON overall -> extraction_failed + a lens_record review item.
    w_bad, _ = _add_work(h, "bad", MD_REG)
    bad_fake = FakeLLMBackend(response="this is not json at all")
    with Session(h.engine) as session:
        bad_result = run_lens(session, None, lens, [w_bad], _router(bad_fake))
    assert bad_result.extraction_failed == 1
    c = _conn(h)
    try:
        lo_bad = c.execute(
            "SELECT * FROM lens_outputs WHERE work_id=? AND status='extraction_failed'", (w_bad,)
        ).fetchone()
        assert lo_bad is not None and lo_bad["claim_id"] is None
        rq = c.execute(
            "SELECT * FROM review_queue WHERE item_type='lens_record' AND status='open'"
        ).fetchall()
        assert any(json.loads(r["payload"])["work_id"] == w_bad for r in rq)
    finally:
        c.close()

    # access_class fail-closed default when cache.db lacks the source row.
    cache_conn = cache_access.open_cache_ro(None)
    try:
        from seedgraph.extraction.runner import resolve_source_access_class

        assert resolve_source_access_class(cache_conn, "md_does_not_exist") == \
            AccessClass.user_supplied_private.value
    finally:
        cache_conn.close()


# --------------------------------------------------------------------------
# idempotency (plan §11 test_lens_runner_idempotency)
# --------------------------------------------------------------------------

def _run_count(h, lens_id):
    c = _conn(h)
    try:
        return c.execute(
            "SELECT COUNT(*) FROM extraction_runs WHERE lens_id=?", (lens_id,)
        ).fetchone()[0]
    finally:
        c.close()


def test_lens_runner_idempotency():
    """A second `lens run --all` with unchanged definition + markdown mints ZERO
    new `extraction_run` rows and makes ZERO backend calls; `--force` reprocesses;
    changing the YAML (definition_hash) or the `markdown_hash` un-skips exactly
    the affected works."""
    h = service.create_project("reg_idem")
    w_reg, _ = _add_work(h, "reg", MD_REG)
    project_dir, lens = _install_lens(h)

    # First run processes the work.
    with Session(h.engine) as session:
        r1 = run_lens(session, None, lens, [w_reg], _router(FakeLLMBackend(response=_found_resp())))
    assert r1.found == 1 and r1.router_calls == 1
    assert _run_count(h, "regularity_conditions_v1") == 1

    # Second unchanged run is a TRUE no-op: zero new runs AND zero backend calls.
    fake2 = FakeLLMBackend(response=_found_resp())
    with Session(h.engine) as session:
        r2 = run_lens(session, None, lens, [w_reg], _router(fake2))
    assert r2.skipped_current == [w_reg]
    assert r2.router_calls == 0
    assert fake2.calls == []
    assert _run_count(h, "regularity_conditions_v1") == 1

    # --force reprocesses (append-only; prior run preserved).
    with Session(h.engine) as session:
        r3 = run_lens(session, None, lens, [w_reg], _router(FakeLLMBackend(response=_found_resp())), force=True)
    assert r3.router_calls == 1
    assert _run_count(h, "regularity_conditions_v1") == 2

    # Changing the YAML (definition_hash) un-skips the work (revise loop; lens is
    # still draft/calibrating so sync_lens is allowed).
    text = (project_dir / "lenses" / "regularity_conditions_v1.yaml").read_text()
    (project_dir / "lenses" / "regularity_conditions_v1.yaml").write_text(
        text.replace("- continuity", "- continuity\n  - boundedness")
    )
    with Session(h.engine) as session:
        registry.sync_lens(session, project_dir, "regularity_conditions_v1")
        lens_v2 = registry.load_lens(project_dir, "regularity_conditions_v1")
        r4 = run_lens(session, None, lens_v2, [w_reg], _router(FakeLLMBackend(response=_found_resp())))
    assert r4.skipped_current == []  # un-skipped by the hash change
    assert r4.router_calls == 1

    # Changing the markdown_hash un-skips the work (same definition).
    _rebridge(h, w_reg, MD_REG + "\n\nAn additional remark.\n", "regchg")
    with Session(h.engine) as session:
        r5 = run_lens(session, None, lens_v2, [w_reg], _router(FakeLLMBackend(response=_found_resp())))
    assert r5.skipped_current == []
    assert r5.router_calls == 1


# --------------------------------------------------------------------------
# fallback (plan §11 test_lens_fallback)
# --------------------------------------------------------------------------

def test_lens_fallback():
    """With no permitted LLM profile, the anchor/FTS match yields
    `epistemic_type='deterministic'` saved-search spans with
    `assertion_status=NULL` and no typed-claim fabrication; not-found is recorded
    honestly; the reduced-capability gap is noted (degraded)."""
    h = service.create_project("reg_fallback")
    w_reg, _ = _add_work(h, "reg", MD_REG)
    w_plain, _ = _add_work(h, "plain", MD_PLAIN)
    _, lens = _install_lens(h)

    with Session(h.engine) as session:
        result = fallback.deterministic_lens_pass(session, lens, [w_reg, w_plain], cache_db=None)

    assert result.degraded is True
    assert result.capability_note and "deterministic" in result.capability_note
    assert result.found == 1
    assert result.not_found == 1

    c = _conn(h)
    try:
        lo = c.execute(
            "SELECT * FROM lens_outputs WHERE work_id=? AND status='found'", (w_reg,)
        ).fetchone()
        assert lo is not None
        # No typed claim fabricated.
        assert lo["claim_id"] is None
        assert lo["confidence"] is None
        fields = json.loads(lo["fields_json"])
        assert fields["epistemic_type"] == "deterministic"
        assert fields["assertion_status"] is None  # NEVER fabricated
        assert fields["evidence_span_ids"]  # an actual saved-search span

        # zero typed claims across the whole deterministic pass.
        assert c.execute("SELECT COUNT(*) FROM extracted_claims").fetchone()[0] == 0

        # honest not_found for the anchor-less work.
        nf = c.execute(
            "SELECT * FROM lens_outputs WHERE work_id=? AND status='not_found'", (w_plain,)
        ).fetchone()
        assert nf is not None
        assert json.loads(nf["fields_json"])["epistemic_type"] == "deterministic"
    finally:
        c.close()


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
# results / coverage / staleness (plan §11 test_lens_results)
# --------------------------------------------------------------------------

def test_lens_results():
    """`lens_results` returns all found conditions across multiple works with span
    back-references; `coverage` separates found / not_found / skipped-no-markdown;
    `stale_outputs` flips a run after a YAML `definition_hash` change AND after a
    `markdown_hash` change."""
    h = service.create_project("reg_results")
    w1, _ = _add_work(h, "reg1", MD_REG)
    w2, _ = _add_work(h, "reg2", MD_REG)
    w_plain, _ = _add_work(h, "plain", MD_PLAIN)
    w_nomd = _add_metadata_only_work(h, "nomd")
    project_dir, lens = _install_lens(h)

    with Session(h.engine) as session:
        run_lens(
            session, None, lens, [w1, w2, w_plain, w_nomd],
            _router(FakeLLMBackend(responses=[_found_resp(), _found_resp(), _not_found_resp()])),
        )

        found = results.lens_results(session, "regularity_conditions_v1", status="found")
        assert {v.work_id for v in found} == {w1, w2}
        for v in found:
            assert v.span_ids  # evidence-span back-reference present
            assert v.normalized_label == "compact parameter space"
            assert v.section == "Assumptions"
            assert v.fields.get("condition_type") == ["compactness"]

        cov = results.coverage(session, "regularity_conditions_v1")
        assert cov.found == 2
        assert cov.not_found == 1
        assert cov.skipped_no_markdown == 1  # the metadata-only work
        assert cov.works_total == 4

        # No stale runs yet.
        assert results.stale_outputs(session, lens) == []

        # YAML definition_hash change flips ALL runs stale.
        text = (project_dir / "lenses" / "regularity_conditions_v1.yaml").read_text()
        (project_dir / "lenses" / "regularity_conditions_v1.yaml").write_text(
            text.replace("- continuity", "- continuity\n  - boundedness")
        )
        lens_v2 = registry.load_lens(project_dir, "regularity_conditions_v1")
        stale_after_yaml = results.stale_outputs(session, lens_v2)
        assert len(stale_after_yaml) == 3  # all three processed runs are stale

    # markdown_hash change flips the affected work's run stale (original lens).
    _rebridge(h, w1, MD_REG + "\n\nA further remark.\n", "reg1chg")  # re-bridge w1's markdown
    with Session(h.engine) as session:
        stale_after_md = results.stale_outputs(session, lens)
        # w1's run is now stale (markdown_hash mismatch); w2/plain runs are not.
        assert len(stale_after_md) >= 1


# --------------------------------------------------------------------------
# CLI (plan §11 test_lens_cli)
# --------------------------------------------------------------------------

def test_lens_cli(monkeypatch):
    """`lens validate` exits non-zero on bad YAML; `lens results --json` emits the
    documented shape; `lens new --from-template` scaffolds the project-local YAML."""
    h = service.create_project("reg_cli")
    w_reg, _ = _add_work(h, "reg", MD_REG)

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

    # Run the lens via the CLI (injected backend) then assert results --json shape.
    monkeypatch.setattr("seedgraph.lenses.runner._BACKEND_OVERRIDE", _ContentAwareBackend())
    res_run = cli.invoke(app, ["lens", "run", "regularity_conditions_v1", "--all", "--project", "reg_cli"])
    assert res_run.exit_code == 0, res_run.output

    res_json = cli.invoke(
        app, ["lens", "results", "regularity_conditions_v1", "--status", "found", "--json", "--project", "reg_cli"]
    )
    assert res_json.exit_code == 0
    payload = json.loads(res_json.output)
    assert isinstance(payload, list) and len(payload) == 1
    row = payload[0]
    assert row["work_id"] == w_reg
    assert row["status"] == "found"
    assert row["condition_type"] == ["compactness"]
    assert row["span_ids"]
    assert row["access_class"] == AccessClass.open_access.value


# --------------------------------------------------------------------------
# milestone end-to-end (plan §11 test_lens_milestone_end_to_end)
# --------------------------------------------------------------------------

def test_lens_milestone_end_to_end(monkeypatch):
    """Milestone (doc-10 §10): `lens new --from-template` → `lens calibrate
    --sample 3` → revise (edit YAML) → `lens run --all` → `lens results` returns
    project-wide conditions with verbatim quotes + condition_type + section +
    evidence span; not-found works appear; metadata-only works are skipped (not
    failed); a repeat `lens run --all` is a no-op (zero new runs); every output
    carries an access_class withheld by the default export allowlist."""
    h = service.create_project("reg_milestone")
    # Private sources so the denormalized access_class is private-by-default and the
    # default-deny export allowlist withholds every lens output (milestone clause).
    priv = AccessClass.user_supplied_private
    w1, _ = _add_work(h, "reg1", MD_REG, access_class=priv)
    w2, _ = _add_work(h, "reg2", MD_REG, access_class=priv)
    w_plain, _ = _add_work(h, "plain", MD_PLAIN, access_class=priv)
    w_meta = _add_metadata_only_work(h, "meta")

    backend = _ContentAwareBackend()
    monkeypatch.setattr("seedgraph.lenses.runner._BACKEND_OVERRIDE", backend)

    slug = "reg_milestone"
    assert cli.invoke(app, ["lens", "new", "regularity_conditions_v1",
                            "--from-template", "regularity_conditions_v1", "--project", slug]).exit_code == 0
    assert cli.invoke(app, ["lens", "calibrate", "regularity_conditions_v1",
                            "--sample", "3", "--project", slug]).exit_code == 0

    # revise loop: edit the YAML (the lens is still calibrating, edits allowed) so
    # the subsequent full run actually reprocesses (sample runs go stale).
    yaml_path = h.root / "projects" / slug / "lenses" / "regularity_conditions_v1.yaml"
    yaml_path.write_text(yaml_path.read_text().replace("- continuity", "- continuity\n  - boundedness"))

    assert cli.invoke(app, ["lens", "run", "regularity_conditions_v1", "--all", "--project", slug]).exit_code == 0

    # lens results returns the project-wide conditions with full evidence.
    res = cli.invoke(app, ["lens", "results", "regularity_conditions_v1", "--status", "found", "--json", "--project", slug])
    assert res.exit_code == 0
    found = json.loads(res.output)
    assert {r["work_id"] for r in found} == {w1, w2}
    for r in found:
        assert r["claim_text"] == "The parameter space is compact."
        assert r["condition_type"] == ["compactness"]
        assert r["section"] == "Assumptions"
        assert r["span_ids"]

    # not-found works appear as explicit not_found records.
    res_nf = cli.invoke(app, ["lens", "results", "regularity_conditions_v1", "--status", "not_found", "--json", "--project", slug])
    nf = json.loads(res_nf.output)
    assert w_plain in {r["work_id"] for r in nf}

    # the lens is now active; metadata-only work skipped (not failed); export-withheld.
    h2 = service.open_project(slug)
    c = _conn(h2)
    try:
        assert c.execute("SELECT status FROM lenses WHERE lens_id='regularity_conditions_v1'").fetchone()[0] == "active"
        assert c.execute("SELECT COUNT(*) FROM extraction_runs WHERE work_id=?", (w_meta,)).fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM lens_outputs WHERE status='extraction_failed'").fetchone()[0] == 0
        for (ac,) in c.execute("SELECT access_class FROM lens_outputs").fetchall():
            assert is_shareable(ac) is False  # private-by-default; default-deny export
        runs_before = c.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0]
    finally:
        c.close()

    # a repeat `lens run --all` is a no-op: ZERO new extraction_runs.
    calls_before = len(backend.calls)
    assert cli.invoke(app, ["lens", "run", "regularity_conditions_v1", "--all", "--project", slug]).exit_code == 0
    c = _conn(h2)
    try:
        runs_after = c.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0]
    finally:
        c.close()
    assert runs_after == runs_before  # zero new runs
    assert len(backend.calls) == calls_before  # zero backend calls
