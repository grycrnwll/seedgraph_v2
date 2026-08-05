"""Track 3 Stage A — provenance query + work-source resolver tests.

Real sqlite fixture (the actual numbered project migrations): pins
``semantic.query.concept_provenance`` (joined claim+span rows incl. claim_id +
offsets + heading_path + page; LEFT JOIN keeps span-less claims),
``concept_provenance_counts`` (papers/claims/spans rollup), and
``acquisition.bridge.resolve_work_source`` (bridge fields + None when absent).
Fully offline; no cache.db read.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlmodel import Session

import seedgraph.db.migrations as migrations
from seedgraph.acquisition.bridge import (
    WorkSourceResolution,
    resolve_work_source,
)
from seedgraph.semantic.query import concept_provenance, concept_provenance_counts


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _migrated(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "project.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA foreign_keys=ON")
    migrations.run_migrations(conn, "project")
    return conn, db


def _add_work(conn, work_id, title, year=2020):
    conn.execute(
        "INSERT INTO works (work_id, canonical_title, year, created_at) VALUES (?,?,?,?)",
        (work_id, title, year, _now()),
    )
    conn.execute(
        "INSERT INTO project_documents (work_id, inclusion_status, is_seed, created_at, "
        "updated_at) VALUES (?,?,?,?,?)",
        (work_id, "included", 0, _now(), _now()),
    )


def _add_run(conn, run_id, work_id):
    conn.execute(
        "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, "
        "markdown_hash, schema_version, prompt_version, access_class, run_status, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (run_id, work_id, "md_" + work_id, "h", "v1", "p1", "open_access", "success", _now()),
    )


def _add_claim(conn, claim_id, run_id, work_id, claim_type="method", subtype=None,
               text="some claim text", access_class="open_access"):
    conn.execute(
        "INSERT INTO extracted_claims (claim_id, extraction_run_id, work_id, claim_type, "
        "claim_subtype, field_key, normalized_label, claim_text, status, epistemic_type, "
        "access_class, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (claim_id, run_id, work_id, claim_type, subtype, "f", "lbl", text, "found",
         "llm_extracted", access_class, _now()),
    )


def _add_section(conn, section_id, work_id, heading_path, heading_text):
    conn.execute(
        "INSERT INTO document_sections (section_id, markdown_id, markdown_hash, "
        "source_file_id, source_file_hash, work_id, level, ordinal, heading_text, "
        "heading_path, section_kind, start_char, end_char, section_parser_version, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (section_id, "md_" + work_id, "h", "sf", "sfh", work_id, 2, 0, heading_text,
         heading_path, "body", 0, 500, "secv1", _now()),
    )


def _add_span(conn, span_id, work_id, quote, section_id=None, start=10, end=30,
              page_start=3, page_end=3, access_class="open_access"):
    conn.execute(
        "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, source_file_id, "
        "source_file_hash, work_id, section_id, start_char, end_char, exact_quote, "
        "quote_hash, page_start, page_end, access_class, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (span_id, "md_" + work_id, "h", "sf", "sfh", work_id, section_id, start, end,
         quote, "qh_" + span_id, page_start, page_end, access_class, _now()),
    )


def _link_claim_span(conn, claim_id, span_id, rank=0):
    conn.execute(
        "INSERT INTO claim_spans (claim_id, span_id, rank, created_at) VALUES (?,?,?,?)",
        (claim_id, span_id, rank, _now()),
    )


def _add_concept(conn, concept_id, label):
    conn.execute(
        "INSERT INTO concepts (concept_id, normalized_label, canonical_label, concept_type, "
        "paper_frequency, status, epistemic_type, access_class, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (concept_id, label, label.upper(), "method", 2, "auto", "deterministic",
         "open_access", _now(), _now()),
    )


def _link_claim_concept(conn, claim_id, concept_id, work_id):
    conn.execute(
        "INSERT INTO claim_concepts (claim_id, concept_id, work_id, epistemic_type, "
        "created_at) VALUES (?,?,?,?,?)",
        (claim_id, concept_id, work_id, "deterministic", _now()),
    )


def _seed(conn):
    """concept::ols links a span-bearing claim (work_a) AND a span-less claim (work_b)."""
    _add_work(conn, "work_a", "Paper A", 2020)
    _add_work(conn, "work_b", "Paper B", 2021)
    _add_run(conn, "extr_a", "work_a")
    _add_run(conn, "extr_b", "work_b")
    _add_concept(conn, "concept::ols", "ols")

    # work_a: claim_1 WITH a span anchored in a section.
    _add_claim(conn, "claim_1", "extr_a", "work_a", claim_type="method",
               subtype="estimator", text="We estimate by OLS.")
    _add_section(conn, "sec_1", "work_a", "2 Identification > 2.1 Estimator", "2.1 Estimator")
    _add_span(conn, "span_1", "work_a", "We estimate by OLS.", section_id="sec_1",
              start=10, end=29, page_start=4, page_end=4)
    _link_claim_span(conn, "claim_1", "span_1")
    _link_claim_concept(conn, "claim_1", "concept::ols", "work_a")

    # work_b: claim_2 with NO span (LEFT JOIN must still return it).
    _add_claim(conn, "claim_2", "extr_b", "work_b", claim_type="method",
               text="OLS is unbiased.")
    _link_claim_concept(conn, "claim_2", "concept::ols", "work_b")
    conn.commit()


def test_concept_provenance_joined_rows_and_left_join(tmp_path):
    conn, _ = _migrated(tmp_path)
    _seed(conn)

    rows = concept_provenance(conn, "concept::ols")
    # span-bearing claim_1 (work_a) + span-less claim_2 (work_b) => 2 rows,
    # ordered by work_id then claim_id.
    assert [r["claim_id"] for r in rows] == ["claim_1", "claim_2"]
    assert [r["work_id"] for r in rows] == ["work_a", "work_b"]

    r1 = rows[0]
    assert r1["title"] == "Paper A"
    assert r1["year"] == 2020
    assert r1["claim_type"] == "method"
    assert r1["claim_subtype"] == "estimator"
    assert r1["claim_text"] == "We estimate by OLS."
    assert r1["extraction_run_id"] == "extr_a"
    assert r1["access_class"] == "open_access"
    # span side of the JOIN populated:
    assert r1["span_id"] == "span_1"
    assert r1["exact_quote"] == "We estimate by OLS."
    assert r1["start_char"] == 10 and r1["end_char"] == 29
    assert r1["page_start"] == 4 and r1["page_end"] == 4
    assert r1["span_access_class"] == "open_access"
    # section side of the JOIN populated:
    assert r1["heading_path"] == "2 Identification > 2.1 Estimator"
    assert r1["heading_text"] == "2.1 Estimator"
    assert r1["rank"] == 0

    # LEFT JOIN keeps the span-less claim, with NULL span/section columns.
    r2 = rows[1]
    assert r2["claim_id"] == "claim_2"
    assert r2["span_id"] is None
    assert r2["exact_quote"] is None
    assert r2["heading_path"] is None
    assert r2["start_char"] is None

    # work_id narrowing returns only that work's rows.
    only_a = concept_provenance(conn, "concept::ols", work_id="work_a")
    assert [r["claim_id"] for r in only_a] == ["claim_1"]
    conn.close()


def test_concept_provenance_counts(tmp_path):
    conn, _ = _migrated(tmp_path)
    _seed(conn)

    counts = concept_provenance_counts(conn, ["concept::ols", "concept::absent"])
    assert counts["concept::ols"] == {"papers": 2, "claims": 2, "spans": 1}
    # an id with no rows is present with all-zero counts.
    assert counts["concept::absent"] == {"papers": 0, "claims": 0, "spans": 0}
    # empty input -> empty dict (no SQL).
    assert concept_provenance_counts(conn, []) == {}
    conn.close()


def test_resolve_work_source_returns_fields_and_none(tmp_path):
    conn, db = _migrated(tmp_path)
    _add_work(conn, "work_a", "Paper A")
    conn.execute(
        "INSERT INTO work_source_files (work_id, source_file_id, file_hash, markdown_id, "
        "markdown_hash, acquisition_method, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("work_a", "sf_abc", "abc", "md_def", "def", "already_cached_local", _now(), _now()),
    )
    conn.commit()
    conn.close()

    engine = create_engine(f"sqlite:///{db}")
    with Session(engine) as s:
        res = resolve_work_source(s, work_id="work_a")
        assert isinstance(res, WorkSourceResolution)
        assert res.work_id == "work_a"
        assert res.source_file_id == "sf_abc"
        assert res.file_hash == "abc"
        assert res.markdown_id == "md_def"
        assert res.markdown_hash == "def"
        # no bridge row -> None (never raises).
        assert resolve_work_source(s, work_id="work_missing") is None
    engine.dispose()


def test_resolve_work_source_pdf_only_pre_conversion(tmp_path):
    """A pdf-only work (markdown not yet backfilled) still resolves; markdown is None."""
    conn, db = _migrated(tmp_path)
    _add_work(conn, "work_a", "Paper A")
    conn.execute(
        "INSERT INTO work_source_files (work_id, source_file_id, file_hash, "
        "acquisition_method, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        ("work_a", "sf_abc", "abc", "open_access_fetch", _now(), _now()),
    )
    conn.commit()
    conn.close()

    engine = create_engine(f"sqlite:///{db}")
    with Session(engine) as s:
        res = resolve_work_source(s, work_id="work_a")
        assert res is not None
        assert res.source_file_id == "sf_abc"
        assert res.markdown_id is None and res.markdown_hash is None
    engine.dispose()
