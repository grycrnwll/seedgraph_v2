"""0008_work_protocol.sql: protocol tables, plain_words, claim_fts recreated and repopulated (§1.7)."""

from __future__ import annotations

import sqlite3

from seedgraph.db.migrations import discover_migrations, run_migrations

PROTOCOL_TABLES = {"batches", "batch_items", "work_packages", "package_claims", "package_submissions",
                   "batch_events", "text_quality"}


def _apply_up_to(conn, last):
    for version, path in discover_migrations("library"):
        if version > last:
            break
        with conn:
            conn.executescript(path.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (version, applied_at) VALUES (?, 'x')", (version,))


def test_upgrade_from_stage1_keeps_claims_searchable_with_plain_words(tmp_path):
    conn = sqlite3.connect(tmp_path / "library.db")
    conn.execute("PRAGMA foreign_keys=ON")
    _apply_up_to(conn, 7)
    with conn:
        conn.execute("INSERT INTO papers (paper_id, title, created_at) VALUES ('lp_a', 'Invented', 't')")
        conn.execute(
            "INSERT INTO extraction_runs (extraction_run_id, paper_id, markdown_id, markdown_hash, "
            "schema_version, prompt_version, run_status, created_at) "
            "VALUES ('run_1', 'lp_a', 'md_1', 'h1', '1', '1', 'ok', 't')")
        conn.execute(
            "INSERT INTO extracted_claims (claim_id, extraction_run_id, paper_id, claim_type, field_key, "
            "normalized_label, claim_text, status, epistemic_type, created_at) "
            "VALUES ('c_1', 'run_1', 'lp_a', 'method', 'methods', 'widget sorting', 'sort the widgets', "
            "'found', 'llm_extracted', 't')")
        conn.execute("INSERT INTO claim_fts (claim_id, normalized_label, claim_text) "
                     "VALUES ('c_1', 'widget sorting', 'sort the widgets')")

    assert run_migrations(conn, "library") >= 8
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert PROTOCOL_TABLES <= tables
    assert "plain_words" in {r[1] for r in conn.execute("PRAGMA table_info(extracted_claims)")}
    assert conn.execute("SELECT claim_id FROM claim_fts WHERE claim_fts MATCH 'widgets'").fetchall() == [("c_1",)]

    with conn:
        conn.execute("UPDATE extracted_claims SET plain_words = 'units are sorted by colour' WHERE claim_id = 'c_1'")
        conn.execute("UPDATE claim_fts SET plain_words = 'units are sorted by colour' WHERE claim_id = 'c_1'")
    hits = conn.execute("SELECT claim_id FROM claim_fts WHERE claim_fts MATCH 'plain_words: colour'").fetchall()
    assert hits == [("c_1",)]
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()
