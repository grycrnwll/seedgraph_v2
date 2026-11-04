"""Library store: schema, foreign keys, views, FTS and the dirty flag (design §1, §2)."""

from __future__ import annotations

import sqlite3

import pytest

from seedgraph.db.bootstrap import ensure_cache_db
from seedgraph.library import store

STAGE1_TABLES = {
    "schema_migrations", "library_meta", "migration_runs", "v2_id_map",
    "papers", "identifiers",
    "source_files", "conversion_runs", "markdown_documents", "cache_events", "paper_files",
    "document_sections", "evidence_spans",
    "extraction_runs", "structured_notes", "extracted_claims", "claim_spans",
    "reference_entries", "citation_links", "citation_review_decisions",
    "duplicate_decisions", "lens_definitions", "lens_outputs", "concept_decisions",
    "attention_items", "conversion_queue",
}
STAGE1_FTS = {"span_fts", "claim_fts", "note_fts"}
STAGE1_VIEWS = {"paper_status", "paper_converter", "note_current"}
STAGE3_VIEWS = {"note_latest", "current_claims"}  # 0009 note overlays
STAGE4_TABLES = {"concepts", "concept_members", "concept_links", "claim_concepts"}  # 0010 concepts (stage 4)
STAGE5_TABLES = {"record_connections", "record_sessions", "traversal_records", "traversal_events",
                 "record_graphs", "spot_check_samples", "spot_check_items"}  # 0011 records, 0012 spot-check
STAGE5_VIEWS = {"spot_check_accuracy"}  # 0012 spot-check
NOW = "2026-01-01T00:00:00+00:00"


def _objects(conn, kind):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = ?", (kind,))}


# --- helpers that build invented rows ---------------------------------------------------

def add_paper(conn, paper_id, title="An invented paper about widgets"):
    conn.execute(
        "INSERT INTO papers (paper_id, title, created_at) VALUES (?, ?, ?)",
        (paper_id, title, NOW),
    )


def add_source_file(conn, n=1):
    h = f"{n:064x}"
    conn.execute(
        "INSERT INTO source_files (source_file_id, file_hash, storage_uri, created_at) "
        "VALUES (?, ?, ?, ?)",
        (f"sf_{h}", h, f"pdfs/{h}.pdf", NOW),
    )
    return f"sf_{h}"


def add_markdown(conn, sf_id, n=1, converter_version="9.9.9"):
    run_id = f"conv_{n}"
    mh = f"{n + 100:064x}"
    conn.execute(
        "INSERT INTO conversion_runs (conversion_run_id, source_file_id, source_file_hash, "
        "converter_name, converter_package, converter_version, python_version, config_json, "
        "conversion_fingerprint, conversion_epistemic_type, run_status, markdown_hash, created_at) "
        "VALUES (?, ?, ?, 'marker', 'marker-pdf', ?, '3.13', '{}', 'fp', "
        "'local_deterministic_conversion', 'success', ?, ?)",
        (run_id, sf_id, sf_id[3:], converter_version, mh, NOW),
    )
    conn.execute(
        "INSERT INTO markdown_documents (markdown_id, conversion_run_id, source_file_id, "
        "markdown_hash, storage_uri, conversion_status, byte_size, created_at) "
        "VALUES (?, ?, ?, ?, ?, 'success', 10, ?)",
        (f"md_{mh}", run_id, sf_id, mh, f"markdown/{mh}.md", NOW),
    )
    return f"md_{mh}"


def attach_file(conn, paper_id, sf_id, md_id=None):
    conn.execute(
        "INSERT INTO paper_files (paper_id, source_file_id, markdown_id, attached_at) "
        "VALUES (?, ?, ?, ?)",
        (paper_id, sf_id, md_id, NOW),
    )


def add_note(conn, paper_id, n, version_no, label=None, schema_id="default_research_note_v1"):
    run = f"extr_{n}"
    conn.execute(
        "INSERT INTO extraction_runs (extraction_run_id, paper_id, markdown_id, markdown_hash, "
        "schema_id, schema_version, prompt_version, run_status, created_at) "
        "VALUES (?, ?, 'md_x', 'x', ?, '1', '1', 'success', ?)",
        (run, paper_id, schema_id, NOW),
    )
    conn.execute(
        "INSERT INTO structured_notes (note_id, extraction_run_id, paper_id, markdown_id, "
        "markdown_hash, schema_id, schema_version, prompt_version, raw_note_json, note_text, "
        "version_no, version_label, created_at) "
        "VALUES (?, ?, ?, 'md_x', 'x', ?, '1', '1', '{}', 'invented note', ?, ?, ?)",
        (f"note_{n}", run, paper_id, schema_id, version_no, label, NOW),
    )
    return f"note_{n}"


# --- schema --------------------------------------------------------------------------------

def test_open_library_creates_full_stage1_schema(tmp_path):
    lib = store.open_library(tmp_path)
    try:
        assert lib.db_path == tmp_path / "library" / "library.db"
        assert lib.db_path.is_file()
        conn = lib.conn
        tables = _objects(conn, "table")
        assert STAGE1_TABLES <= tables
        assert STAGE1_FTS <= tables
        assert STAGE1_VIEWS | STAGE3_VIEWS | STAGE5_VIEWS == _objects(conn, "view")
        assert STAGE4_TABLES <= tables
        assert STAGE5_TABLES <= tables
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        lib.close()


def test_open_is_idempotent(tmp_path):
    store.open_library(tmp_path).close()
    lib = store.open_library(tmp_path)
    try:
        versions = [r[0] for r in lib.conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version")]
        assert versions == sorted(set(versions)) and versions
    finally:
        lib.close()


def test_open_library_defaults_to_resolved_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SEEDGRAPH_HOME", str(tmp_path / "h"))
    lib = store.open_library()
    try:
        assert lib.db_path == tmp_path / "h" / "library" / "library.db"
        assert lib.graphs_root == tmp_path / "h" / "graphs"
    finally:
        lib.close()


@pytest.mark.parametrize("table", ["source_files", "conversion_runs", "markdown_documents",
                                   "cache_events"])
def test_stage0_cache_tables_are_verbatim(tmp_path, table):
    """Same columns, order, types, defaults and nullability as stage 0's cache.db."""
    cache_db = ensure_cache_db(tmp_path / "v2like")
    oracle = sqlite3.connect(cache_db)
    lib = store.open_library(tmp_path / "home")
    try:
        expected = oracle.execute(f"PRAGMA table_info({table})").fetchall()
        actual = [tuple(r) for r in lib.conn.execute(f"PRAGMA table_info({table})")]
        assert actual == expected
    finally:
        oracle.close()
        lib.close()


def test_provider_cache_is_a_separate_db_with_stage0_ddl(tmp_path):
    cache_db = ensure_cache_db(tmp_path / "v2like")
    oracle = sqlite3.connect(cache_db)
    lib = store.open_library(tmp_path / "home")
    pc = store.open_provider_cache(lib.root)
    try:
        assert lib.root / "provider_cache.db" == store.provider_cache_path(lib.root)
        assert "provider_cache" not in _objects(lib.conn, "table")
        assert ([tuple(r) for r in pc.execute("PRAGMA table_info(provider_cache)")]
                == oracle.execute("PRAGMA table_info(provider_cache)").fetchall())
    finally:
        pc.close()
        oracle.close()
        lib.close()


# --- foreign keys ------------------------------------------------------------------------

def test_foreign_keys_are_enforced(tmp_path):
    lib = store.open_library(tmp_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            with lib.write() as conn:
                attach_file(conn, "lp_missing", "sf_missing")
        with pytest.raises(sqlite3.IntegrityError):
            with lib.write() as conn:
                conn.execute(
                    "INSERT INTO identifiers (paper_id, id_type, id_value) "
                    "VALUES ('lp_missing', 'doi', '10.1/x')")
    finally:
        lib.close()


def test_every_fk_child_accepts_a_valid_row(tmp_path):
    """Catches FKs that point at a non-unique parent ('foreign key mismatch')."""
    lib = store.open_library(tmp_path)
    try:
        with lib.write() as conn:
            add_paper(conn, "lp_a")
            add_paper(conn, "lp_b")
            sf = add_source_file(conn)
            md = add_markdown(conn, sf)
            attach_file(conn, "lp_a", sf, md)
            conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value, source) "
                         "VALUES ('lp_a', 'doi', '10.1/x', 'user')")
            conn.execute(
                "INSERT INTO cache_events (object_type, object_id, event_type, created_at) "
                "VALUES ('source_file', ?, 'source_file_ingested', ?)", (sf, NOW))
            conn.execute(
                "INSERT INTO document_sections (section_id, markdown_id, markdown_hash, "
                "source_file_id, source_file_hash, level, ordinal, start_char, end_char, "
                "section_parser_version, created_at) VALUES ('sec_1', ?, ?, ?, ?, 0, 0, 0, 5, "
                "'1', ?)", (md, md[3:], sf, sf[3:], NOW))
            conn.execute(
                "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, source_file_id, "
                "source_file_hash, section_id, start_char, end_char, exact_quote, quote_hash, "
                "created_at) VALUES ('span_1', ?, ?, ?, ?, 'sec_1', 0, 5, 'hello', 'qh', ?)",
                (md, md[3:], sf, sf[3:], NOW))
            note = add_note(conn, "lp_a", 1, 1)
            conn.execute(
                "INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, "
                "paper_id, claim_type, field_key, status, epistemic_type, created_at) "
                "VALUES ('claim_1', ?, 'extr_1', 'lp_a', 'method', 'method', 'found', "
                "'llm_extracted', ?)", (note, NOW))
            conn.execute("INSERT INTO claim_spans (claim_id, span_id, created_at) "
                         "VALUES ('claim_1', 'span_1', ?)", (NOW,))
            conn.execute(
                "INSERT INTO reference_entries (reference_id, citing_paper_id, raw_reference_text, "
                "resolved_paper_id, resolution_status, ordinal, created_at) "
                "VALUES ('ref_1', 'lp_a', 'Invented, A. (2001) Widgets.', 'lp_b', 'resolved', 0, ?)",
                (NOW,))
            conn.execute(
                "INSERT INTO citation_links (source_paper_id, target_paper_id, provenance, "
                "confidence, reference_id, created_at) "
                "VALUES ('lp_a', 'lp_b', 'parsed_bibliography', 0.9, 'ref_1', ?)", (NOW,))
            conn.execute(
                "INSERT INTO citation_review_decisions (reference_id, decision, reason, decided_at) "
                "VALUES ('ref_1', 'reject', 'invented', ?)", (NOW,))
            conn.execute(
                "INSERT INTO lens_definitions (lens_id, name, object_type, status, "
                "definition_hash, definition_yaml, created_at) "
                "VALUES ('widget_lens_v1', 'Widgets', 'assumption', 'active', 'dh', 'x: 1', ?)",
                (NOW,))
            conn.execute(
                "INSERT INTO lens_outputs (lens_output_id, extraction_run_id, lens_id, paper_id, "
                "status, claim_id, fields_json, created_at) "
                "VALUES ('lensout_1', 'extr_1', 'widget_lens_v1', 'lp_a', 'found', 'claim_1', "
                "'{}', ?)", (NOW,))
            conn.execute(
                "INSERT INTO attention_items (item_id, kind, paper_id, subject_json, hides_notes, "
                "status, created_at) VALUES ('att_1', 'pdf_title_miss', 'lp_a', '{}', 1, 'open', ?)",
                (NOW,))
            conn.execute("INSERT INTO conversion_queue (source_file_id, reason, queued_at) "
                         "VALUES (?, 'not_converted', ?)", (sf, NOW))
            conn.execute(
                "INSERT INTO duplicate_decisions (paper_a, paper_b, verdict, decided_by, decided_at) "
                "VALUES ('lp_a', 'lp_b', 'different', 'researcher', ?)", (NOW,))
            conn.execute(
                "INSERT INTO concept_decisions (decision_id, kind, label_a, label_b, decided_by, "
                "state, decided_at) VALUES ('cd_1', 'merge', 'alpha', 'beta', 'researcher', "
                "'pending_apply', ?)", (NOW,))
            conn.execute("INSERT INTO v2_id_map (v2_project, v2_table, v2_id, v3_table, v3_id) "
                         "VALUES ('proj', 'works', 'work_1', 'papers', 'lp_a')")
        assert lib.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        lib.close()


# --- views --------------------------------------------------------------------------------

def test_paper_status_is_derived_from_paper_files(tmp_path):
    lib = store.open_library(tmp_path)
    try:
        with lib.write() as conn:
            add_paper(conn, "lp_held")
            add_paper(conn, "lp_cited")
            attach_file(conn, "lp_held", add_source_file(conn))
        rows = dict(lib.conn.execute("SELECT paper_id, status FROM paper_status"))
        assert rows == {"lp_held": "held", "lp_cited": "cited_only"}
        assert store.is_held(lib.conn, "lp_held") is True
        assert store.is_held(lib.conn, "lp_cited") is False
    finally:
        lib.close()


def test_paper_converter_reports_the_run_version_of_the_current_markdown(tmp_path):
    lib = store.open_library(tmp_path)
    try:
        with lib.write() as conn:
            add_paper(conn, "lp_a")
            add_paper(conn, "lp_b")
            sf = add_source_file(conn)
            md = add_markdown(conn, sf, converter_version="1.2.3")
            attach_file(conn, "lp_a", sf, md)
            attach_file(conn, "lp_b", add_source_file(conn, 2))  # not converted yet
        rows = {r["paper_id"]: tuple(r) for r in lib.conn.execute(
            "SELECT paper_id, source_file_id, markdown_id, converter_version FROM paper_converter")}
        assert rows["lp_a"] == ("lp_a", sf, md, "1.2.3")
        assert rows["lp_b"][2:] == (None, None)
    finally:
        lib.close()


def test_note_current_takes_highest_unlabelled_version(tmp_path):
    lib = store.open_library(tmp_path)
    try:
        with lib.write() as conn:
            add_paper(conn, "lp_a")
            add_paper(conn, "lp_b")
            add_note(conn, "lp_a", 1, 1)
            add_note(conn, "lp_a", 2, 2)
            add_note(conn, "lp_a", 3, 3, label="wrong_pdf")
            add_note(conn, "lp_b", 4, 1, label="other_markdown")
        current = {r["paper_id"]: (r["note_id"], r["version_no"]) for r in lib.conn.execute(
            "SELECT paper_id, note_id, version_no FROM note_current")}
        assert current == {"lp_a": ("note_2", 2)}
    finally:
        lib.close()


def test_note_versions_are_unique_per_paper_and_schema(tmp_path):
    lib = store.open_library(tmp_path)
    try:
        with lib.write() as conn:
            add_paper(conn, "lp_a")
            add_note(conn, "lp_a", 1, 1)
        with pytest.raises(sqlite3.IntegrityError):
            with lib.write() as conn:
                add_note(conn, "lp_a", 2, 1)
        with pytest.raises(sqlite3.IntegrityError):
            with lib.write() as conn:
                add_note(conn, "lp_a", 3, 2, label="not_a_label")
    finally:
        lib.close()


# --- FTS ------------------------------------------------------------------------------------

def test_fts_tables_match_invented_text(tmp_path):
    lib = store.open_library(tmp_path)
    try:
        with lib.write() as conn:
            conn.execute("INSERT INTO span_fts (quote_text, span_id, markdown_id, section_id) "
                         "VALUES ('The zorblat estimator is consistent under mixing.', "
                         "'span_1', 'md_1', 'sec_1')")
            conn.execute("INSERT INTO claim_fts (claim_id, normalized_label, claim_text) "
                         "VALUES ('claim_1', 'quuxian ergodicity', 'Assumes Quuxian ergodicity.')")
            conn.execute("INSERT INTO note_fts (note_id, note_text) "
                         "VALUES ('note_1', 'A note on Fréchet widgets.')")
        c = lib.conn
        assert [r[0] for r in c.execute("SELECT span_id FROM span_fts WHERE span_fts MATCH 'zorblat'"
                         ).fetchall()] == ["span_1"]
        assert [r[0] for r in c.execute("SELECT span_id FROM span_fts WHERE span_fts MATCH 'zo*'"
                         ).fetchall()] == ["span_1"]
        assert [r[0] for r in c.execute("SELECT claim_id FROM claim_fts WHERE claim_fts MATCH '\"quuxian ergodicity\"'"
                         ).fetchall()] == ["claim_1"]
        assert [r[0] for r in c.execute("SELECT note_id FROM note_fts WHERE note_fts MATCH 'frechet'"
                         ).fetchall()] == ["note_1"]
        cols = [r[1] for r in c.execute("PRAGMA table_info(span_fts)")]
        assert cols == ["quote_text", "span_id", "markdown_id", "section_id"]
    finally:
        lib.close()


# --- dirty flag and origin ------------------------------------------------------------------

def test_fresh_library_is_clean_with_local_origin(tmp_path):
    lib = store.open_library(tmp_path)
    try:
        meta = store.read_meta(lib.conn)
        assert meta["dirty"] == "0"
        assert meta["origin"] == "local"
        assert store.is_dirty(lib.conn) is False
    finally:
        lib.close()


def test_a_normal_write_sets_dirty(tmp_path):
    lib = store.open_library(tmp_path)
    try:
        with lib.write() as conn:
            add_paper(conn, "lp_a")
        assert store.is_dirty(lib.conn) is True
    finally:
        lib.close()


def test_a_failed_write_leaves_dirty_unchanged_and_rolls_back(tmp_path):
    lib = store.open_library(tmp_path)
    try:
        with pytest.raises(RuntimeError):
            with lib.write() as conn:
                add_paper(conn, "lp_a")
                raise RuntimeError("boom")
        assert store.is_dirty(lib.conn) is False
        assert lib.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
    finally:
        lib.close()


def test_migration_mode_writes_do_not_set_dirty(tmp_path):
    staging_lib = tmp_path / "migration" / "staging" / "run1" / "library"
    lib = store.open_library_at(staging_lib, migration=True)
    try:
        assert lib.migration is True
        assert lib.graphs_root == staging_lib.parent / "graphs"
        with lib.write() as conn:
            add_paper(conn, "lp_a")
        store.record_migration_origin(lib, run_id="run1", v2_home="C:/invented/v2")
        meta = store.read_meta(lib.conn)
        assert meta["dirty"] == "0"
        assert meta["origin"] == "migration"
        assert meta["migration_run_id"] == "run1"
        assert meta["v2_home"] == "C:/invented/v2"
    finally:
        lib.close()
    # Re-opened normally after the swap, the first write marks it dirty.
    lib = store.open_library_at(staging_lib)
    try:
        assert store.is_dirty(lib.conn) is False
        with lib.write() as conn:
            add_paper(conn, "lp_b")
        meta = store.read_meta(lib.conn)
        assert meta["dirty"] == "1" and meta["origin"] == "migration"
    finally:
        lib.close()


def test_record_migration_origin_refused_outside_migration_mode(tmp_path):
    lib = store.open_library(tmp_path)
    try:
        with pytest.raises(store.LibraryError):
            store.record_migration_origin(lib, run_id="r", v2_home="x")
    finally:
        lib.close()


def test_nested_write_rolls_back_only_the_inner_block(tmp_path):
    lib = store.open_library(tmp_path)
    try:
        with lib.write() as conn:
            add_paper(conn, "lp_outer")
            with pytest.raises(RuntimeError):
                with lib.write() as inner:
                    add_paper(inner, "lp_inner")
                    raise RuntimeError("inner fails")
        ids = [r[0] for r in lib.conn.execute("SELECT paper_id FROM papers ORDER BY paper_id")]
        assert ids == ["lp_outer"]
        assert store.is_dirty(lib.conn)
    finally:
        lib.close()
