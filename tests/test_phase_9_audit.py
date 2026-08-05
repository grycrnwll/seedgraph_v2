"""Phase 9 — audit/sampling unit tests (plan §11 "Unit", maps to test_eval_audit.py).

Fully offline: the audit module takes a raw sqlite3 connection over a migrated
project.db. Includes the D6 ORM-vs-migration parity test for ``audit_records``.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects import sqlite as sqlite_dialect

from seedgraph.db import migrations
from seedgraph.db.models_project import AuditRecord
from seedgraph.eval import audit


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _migrated_conn(tmp_path) -> sqlite3.Connection:
    db = tmp_path / "project.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA foreign_keys=ON")
    migrations.run_migrations(conn, "project")
    return conn


def _seed_corpus(conn: sqlite3.Connection, *, n_claims: int = 25, n_spans: int = 22) -> None:
    """Two works + one run + N claims + M spans + concept/alias + 2 citation edges."""
    conn.execute("INSERT INTO works (work_id, created_at) VALUES ('w', ?)", (_now(),))
    conn.execute("INSERT INTO works (work_id, created_at) VALUES ('w2', ?)", (_now(),))
    conn.execute(
        "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, markdown_hash, "
        "schema_version, prompt_version, access_class, run_status, created_at) "
        "VALUES ('r','w','md','h','v1','p1','open_access','success',?)",
        (_now(),),
    )
    for i in range(n_claims):
        conn.execute(
            "INSERT INTO extracted_claims (claim_id, extraction_run_id, work_id, claim_type, "
            "field_key, status, epistemic_type, access_class, created_at) "
            "VALUES (?, 'r','w','finding','f','found','llm_extracted','open_access',?)",
            (f"claim_{i:03d}", _now()),
        )
    for i in range(n_spans):
        conn.execute(
            "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, source_file_id, "
            "source_file_hash, work_id, start_char, end_char, exact_quote, quote_hash, "
            "access_class, created_at) VALUES (?, 'md','h','sf','sfh','w',0,3,'abc','qh',"
            "'open_access',?)",
            (f"span_{i:03d}", _now()),
        )
    conn.execute(
        "INSERT INTO concepts (concept_id, normalized_label, canonical_label, concept_type, "
        "status, epistemic_type, access_class, created_at, updated_at) "
        "VALUES ('concept::x','x','X','assumption','auto','deterministic','open_access',?,?)",
        (_now(), _now()),
    )
    conn.execute(
        "INSERT INTO concept_aliases (concept_id, alias_label, fold_reason, epistemic_type, "
        "created_at) VALUES ('concept::x','y','acronym','deterministic',?)",
        (_now(),),
    )
    for src, tgt in (("w", "w2"), ("w2", "w")):
        conn.execute(
            "INSERT INTO citation_edges (source_work_id, target_work_id, edge_type, provenance, "
            "confidence, run_id, created_at) VALUES (?,?,'cites','provider_reference',1.0,'run1',?)",
            (src, tgt, _now()),
        )
    conn.commit()


def _subjects(conn: sqlite3.Connection, batch_id: str) -> list:
    return sorted(
        conn.execute(
            "SELECT subject_type, subject_id FROM audit_records WHERE sample_batch_id=?",
            (batch_id,),
        ).fetchall()
    )


def test_draw_sample_deterministic_for_fixed_seed(tmp_path):
    """draw_sample(conn, run_id, seed) produces the IDENTICAL draw (same audit
    subjects) when re-run with the same seed over the same corpus (plan §11)."""
    assert audit.SAMPLE_COUNTS  # vocab/constants present
    conn = _migrated_conn(tmp_path)
    _seed_corpus(conn)
    b1 = audit.draw_sample(conn, "run1", 42)
    b2 = audit.draw_sample(conn, "run1", 42)
    assert _subjects(conn, b1.sample_batch_id) == _subjects(conn, b2.sample_batch_id)
    # a different seed yields a different draw of the 20-of-25 claim subsample.
    b3 = audit.draw_sample(conn, "run1", 99)
    assert _subjects(conn, b1.sample_batch_id) != _subjects(conn, b3.sample_batch_id)
    conn.close()


def test_draw_sample_respects_section_11_counts_and_underdraws(tmp_path):
    """draw_sample honors the §11 per-subject counts and UNDER-DRAWS gracefully when
    a subject kind has fewer items than its target (plan §11)."""
    conn = _migrated_conn(tmp_path)
    _seed_corpus(conn, n_claims=25, n_spans=22)
    batch = audit.draw_sample(conn, "run1", 7)
    assert batch.counts["claim"] == 20      # capped at §11 target (25 available)
    assert batch.counts["span"] == 20       # capped at §11 target (22 available)
    assert batch.counts["work"] == 2        # under-draw (only 2 works)
    assert batch.counts["concept_merge"] == 1
    assert batch.counts["citation_edge"] == 2
    assert batch.counts["answer"] == 0      # no answers table/rows -> graceful 0
    assert batch.counts["content_access_boundary"] == 0
    conn.close()


def test_open_audit_stamps_id_status_and_access_class(tmp_path):
    """open_audit inserts one row with an 'audit_'-prefixed id, status='open', the
    propagated fail-closed access_class, and the given audit_type/subject_type/
    subject_id (decision 65; §4/§7)."""
    conn = _migrated_conn(tmp_path)
    aid = audit.open_audit(conn, "span_relevance", "span", "span_x", "run1", "open_access")
    assert aid.startswith("audit_")
    row = conn.execute(
        "SELECT audit_type, subject_type, subject_id, status, access_class FROM audit_records "
        "WHERE audit_id=?",
        (aid,),
    ).fetchone()
    assert row == ("span_relevance", "span", "span_x", "open", "open_access")
    with pytest.raises(ValueError):
        audit.open_audit(conn, "not_a_type", "span", "s", None, "open_access")
    conn.close()


def test_record_verdict_idempotent_once_resolved(tmp_path):
    """record_verdict marks an open row resolved with its grade; a second call on a
    row already status='resolved' is a no-op (idempotent), not a re-grade (plan §11)."""
    conn = _migrated_conn(tmp_path)
    aid = audit.open_audit(conn, "claim_faithfulness", "claim", "c1", "run1", "open_access")
    audit.record_verdict(conn, aid, "accept", verdict="supported", severity="low")
    audit.record_verdict(conn, aid, "reject", verdict="unsupported")  # must NOT re-grade
    row = conn.execute(
        "SELECT decision, verdict, status FROM audit_records WHERE audit_id=?", (aid,)
    ).fetchone()
    assert row == ("accept", "supported", "resolved")
    conn.close()


def test_record_verdict_edit_persists_edit_payload(tmp_path):
    """record_verdict(decision='edit', edit_payload=...) persists the JSON
    edit_payload alongside the resolved grade (plan §11; §4)."""
    conn = _migrated_conn(tmp_path)
    aid = audit.open_audit(conn, "span_relevance", "span", "s1", "run1", "open_access")
    audit.record_verdict(conn, aid, "edit", edit_payload='{"start_char": 5}')
    row = conn.execute(
        "SELECT decision, edit_payload, status FROM audit_records WHERE audit_id=?", (aid,)
    ).fetchone()
    assert row == ("edit", '{"start_char": 5}', "resolved")
    conn.close()


def test_sample_metadata_resolution_opens_identifier_audits(tmp_path):
    """sample_metadata_resolution opens metadata_resolution audits over phase_5b
    identifier subjects (review_queue duplicate_candidate items) (r2 gap-12)."""
    conn = _migrated_conn(tmp_path)
    for i in range(3):
        conn.execute(
            "INSERT INTO review_queue (item_id, item_type, target_type, target_id, status, "
            "created_at) VALUES (?, 'duplicate_candidate','identifier',?, 'open', ?)",
            (f"rq_{i}", f"work_{i}", _now()),
        )
    # a non-duplicate item must NOT be sampled.
    conn.execute(
        "INSERT INTO review_queue (item_id, item_type, status, created_at) "
        "VALUES ('rq_other','citation_resolution','open',?)",
        (_now(),),
    )
    conn.commit()
    opened = audit.sample_metadata_resolution(conn, "run1", 0)
    assert len(opened) == 3
    rows = conn.execute(
        "SELECT DISTINCT audit_type, subject_type FROM audit_records"
    ).fetchall()
    assert rows == [("metadata_resolution", "identifier")]
    conn.close()


def test_sample_reference_extraction_opens_reference_entry_audits(tmp_path):
    """sample_reference_extraction opens reference_extraction audits over phase_3b
    reference_entry subjects with resolution_status in {ambiguous, suspect} (r2 gap-12)."""
    conn = _migrated_conn(tmp_path)
    conn.execute("INSERT INTO works (work_id, created_at) VALUES ('w', ?)", (_now(),))
    statuses = [("ref_a", "ambiguous"), ("ref_b", "suspect"), ("ref_c", "resolved")]
    for rid, status in statuses:
        conn.execute(
            "INSERT INTO reference_entries (reference_id, citing_work_id, raw_reference_text, "
            "resolution_status, created_at) VALUES (?, 'w', 'raw', ?, ?)",
            (rid, status, _now()),
        )
    conn.commit()
    opened = audit.sample_reference_extraction(conn, "run1", 0)
    assert len(opened) == 2  # the resolved entry is NOT sampled
    subjects = sorted(
        conn.execute(
            "SELECT subject_id FROM audit_records WHERE audit_type='reference_extraction'"
        ).fetchall()
    )
    assert subjects == [("ref_a",), ("ref_b",)]
    types = conn.execute("SELECT DISTINCT subject_type FROM audit_records").fetchall()
    assert types == [("reference_entry",)]
    conn.close()


# --------------------------------------------------------------------------
# D6 — ORM/migration parity for audit_records (extends the §10 step-2 parity test)
# --------------------------------------------------------------------------


def _affinity(type_text: str) -> str:
    t = type_text.upper()
    if "INT" in t:
        return "INTEGER"
    if any(k in t for k in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if any(k in t for k in ("REAL", "FLOA", "DOUB")):
        return "REAL"
    return "NUMERIC"


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
        mig_aff = _affinity(str(mig_col["type"]))
        assert orm_aff == mig_aff, f"{table_name}.{name}: type {orm_aff} != {mig_aff}"
        if name not in mig_pk:
            assert orm_col.nullable == mig_col["nullable"], f"{table_name}.{name}: nullability drift"

    mig_ix = {tuple(ix["column_names"]) for ix in insp.get_indexes(table_name) if not ix.get("unique")}
    orm_ix = {tuple(c.name for c in ix.columns) for ix in model.__table__.indexes}
    assert mig_ix == orm_ix, f"{table_name}: index column-sets drift ({mig_ix} != {orm_ix})"

    mig_uq = {frozenset(u["column_names"]) for u in insp.get_unique_constraints(table_name)}
    orm_uq = {
        frozenset(c.name for c in u.columns)
        for u in model.__table__.constraints
        if u.__class__.__name__ == "UniqueConstraint"
    }
    assert mig_uq == orm_uq, f"{table_name}: unique-constraint drift ({mig_uq} != {orm_uq})"


def test_audit_records_orm_migration_parity(tmp_path):
    """The AuditRecord ORM class mirrors the migrated 0012 audit_records schema
    exactly — columns / PK / types / nullability / indexes (no create_all; D6)."""
    db = tmp_path / "parity.db"
    conn = sqlite3.connect(str(db))
    migrations.run_migrations(conn, "project")
    conn.close()

    insp = inspect(create_engine(f"sqlite:///{db.as_posix()}"))
    assert "audit_records" in insp.get_table_names()
    _assert_table_parity(insp, "audit_records", AuditRecord)
