import sqlite3

import pytest

from seedgraph.db.bootstrap import ensure_cache_db, ensure_project_db
from seedgraph.db.connection import open_cache_db, open_project_db
from seedgraph.db.migrations import (
    current_version,
    discover_migrations,
    latest_version,
    run_migrations,
)


def test_fresh_migrate_then_idempotent():
    ensure_cache_db()
    conn = open_cache_db()
    try:
        # Version-count-agnostic: cache accrues migrations across phases (>=2 now).
        latest = latest_version("cache")
        assert current_version(conn) == latest
        # re-running is a no-op: run_migrations returns the resulting (latest) version
        assert run_migrations(conn, "cache") == latest
    finally:
        conn.close()


def test_foreign_keys_on_in_both_scopes():
    ensure_cache_db()
    ensure_project_db("proj")
    cache = open_cache_db()
    project = open_project_db("proj")
    try:
        assert cache.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert project.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        cache.close()
        project.close()


def test_tables_present():
    ensure_cache_db()
    ensure_project_db("proj")
    cache = open_cache_db()
    project = open_project_db("proj")
    try:
        cache_tables = {
            r[0] for r in cache.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"schema_migrations", "source_files"} <= cache_tables
        project_tables = {
            r[0] for r in project.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {
            "schema_migrations",
            "review_queue",
            "llm_usage_events",
            "llm_key_refs",
        } <= project_tables
    finally:
        cache.close()
        project.close()


def test_access_class_check_rejects_bad_value():
    ensure_cache_db()
    conn = open_cache_db()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO source_files (source_file_id, file_hash, access_class, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("sf_x", "hashx", "not_a_class", "2026-01-01T00:00:00Z"),
            )
            conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Build A ch8 — migration 0014: reference_entries table-recreate widening the
# resolution_source CHECK to admit 'manual_override' (script-internal FK toggle).
# ---------------------------------------------------------------------------

_REF_IDX = {
    "idx_reference_entries_citing",
    "idx_reference_entries_resolved",
    "idx_reference_entries_status",
}


def _ref_indexes(conn) -> set:
    return {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='reference_entries'"
        ).fetchall()
    }


def test_0014_recreate_applies_on_fresh_db():
    ensure_project_db("m14fresh")
    conn = open_project_db("m14fresh")
    try:
        assert current_version(conn) >= 14
        # the widened CHECK admits 'manual_override' ...
        conn.execute(
            "INSERT INTO works (work_id, created_at) VALUES ('work_a', 't')")
        conn.execute(
            "INSERT INTO reference_entries (reference_id, citing_work_id, "
            "raw_reference_text, resolution_status, resolution_source, created_at) "
            "VALUES ('ref_ok', 'work_a', 'raw', 'resolved', 'manual_override', 't')")
        # ... and still rejects out-of-vocab values
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO reference_entries (reference_id, citing_work_id, "
                "raw_reference_text, resolution_source, created_at) "
                "VALUES ('ref_bad', 'work_a', 'raw', 'bogus_source', 't')")
        conn.commit()
        assert _REF_IDX <= _ref_indexes(conn)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_0014_recreate_applies_on_populated_db(tmp_path):
    """The case a naive FKs-ON `DROP TABLE` fails: a populated DB whose
    citation_edges.reference_id is SET. The script's internal
    PRAGMA foreign_keys=OFF/ON makes the recreate apply cleanly; data survives;
    indexes recreated; foreign_key_check empty."""
    db = tmp_path / "populated.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        # bring the DB to the pre-0014 version, mimicking the runner's ledger.
        for version, path in discover_migrations("project"):
            if version >= 14:
                continue
            conn.executescript(path.read_text(encoding="utf-8"))
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, "t"),
            )
            conn.commit()
        assert current_version(conn) == 13
        # populate: works + a reference row + an edge REFERENCING the ref row.
        conn.execute("INSERT INTO works (work_id, created_at) VALUES ('work_1', 't')")
        conn.execute("INSERT INTO works (work_id, created_at) VALUES ('work_2', 't')")
        conn.execute(
            "INSERT INTO reference_entries (reference_id, citing_work_id, "
            "raw_reference_text, resolved_work_id, resolution_status, "
            "resolution_source, confidence, created_at) "
            "VALUES ('ref_1', 'work_1', 'raw text', 'work_2', 'resolved', 'doi', 1.0, 't')")
        conn.execute(
            "INSERT INTO citation_edges (source_work_id, target_work_id, edge_type, "
            "provenance, confidence, reference_id, run_id, created_at) "
            "VALUES ('work_1', 'work_2', 'cites', 'parsed_bibliography', 0.9, 'ref_1', 'R', 't')")
        conn.commit()

        applied = run_migrations(conn, "project")
        assert applied == latest_version("project") >= 14

        # data survived the recreate verbatim; the FK edge still resolves.
        assert conn.execute(
            "SELECT citing_work_id, resolved_work_id, resolution_source "
            "FROM reference_entries WHERE reference_id='ref_1'"
        ).fetchone() == ("work_1", "work_2", "doi")
        assert conn.execute(
            "SELECT reference_id FROM citation_edges").fetchone() == ("ref_1",)
        # widened CHECK live; FK enforcement back ON; indexes present.
        conn.execute(
            "INSERT INTO reference_entries (reference_id, citing_work_id, "
            "raw_reference_text, resolution_status, resolution_source, created_at) "
            "VALUES ('ref_2', 'work_1', 'raw', 'resolved', 'manual_override', 't')")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO reference_entries (reference_id, citing_work_id, "
                "raw_reference_text, created_at) VALUES ('ref_3', 'work_missing', 'raw', 't')")
        conn.commit()
        assert _REF_IDX <= _ref_indexes(conn)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Build B ch5 — migration 0015: concepts.weight + project_graph_edges.shared_count
# ---------------------------------------------------------------------------


def test_0015_analysis_ranking_columns_present():
    ensure_project_db("m15fresh")
    conn = open_project_db("m15fresh")
    try:
        assert current_version(conn) >= 15
        concept_cols = {
            r[1]: r for r in conn.execute("PRAGMA table_info(concepts)").fetchall()
        }
        # weight REAL NOT NULL DEFAULT 0.0
        assert "weight" in concept_cols
        assert concept_cols["weight"][2].upper() == "REAL"
        assert concept_cols["weight"][3] == 1  # NOT NULL
        pge_cols = {
            r[1]: r
            for r in conn.execute("PRAGMA table_info(project_graph_edges)").fetchall()
        }
        # shared_count INTEGER, nullable (NULL on non-co-occurrence edge types).
        assert "shared_count" in pge_cols
        assert pge_cols["shared_count"][2].upper() == "INTEGER"
        assert pge_cols["shared_count"][3] == 0
        # the DEFAULT back-fills: an INSERT omitting weight lands at 0.0.
        conn.execute(
            "INSERT INTO concepts (concept_id, normalized_label, canonical_label, "
            "concept_type, paper_frequency, status, epistemic_type, access_class, "
            "created_at, updated_at) VALUES ('concept::x','x','X','method',1,"
            "'auto','deterministic','open_access','t','t')"
        )
        assert conn.execute(
            "SELECT weight FROM concepts WHERE concept_id='concept::x'"
        ).fetchone()[0] == 0.0
        # fresh-migrate-then-idempotent still holds with 0015 in the ledger.
        assert run_migrations(conn, "project") == latest_version("project") >= 15
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Build D ch10 — migration 0017: works.abstract + works.oa_status (last of the
# program's four project migrations; merges after C's 0016).
# ---------------------------------------------------------------------------


def test_0017_works_abstract_oa_status_columns():
    ensure_project_db("m17fresh")
    conn = open_project_db("m17fresh")
    try:
        assert current_version(conn) >= 17
        cols = {r[1]: r for r in conn.execute("PRAGMA table_info(works)").fetchall()}
        for name in ("abstract", "oa_status"):
            assert name in cols
            assert cols[name][2].upper() == "TEXT"
            assert cols[name][3] == 0  # nullable — no backfill needed on populated dbs
        # Nullable: an INSERT omitting both succeeds (pre-ch10 cache entries lack
        # the fields — harmless). oa_status is FREE TEXT (no CHECK/enum): any OA
        # color string is accepted, diamond included.
        conn.execute("INSERT INTO works (work_id, created_at) VALUES ('work_plain', 't')")
        conn.execute(
            "INSERT INTO works (work_id, abstract, oa_status, created_at) "
            "VALUES ('work_oa', 'An abstract.', 'diamond', 't')")
        conn.commit()
        assert tuple(conn.execute(
            "SELECT abstract, oa_status FROM works WHERE work_id='work_plain'"
        ).fetchone()) == (None, None)
        assert tuple(conn.execute(
            "SELECT abstract, oa_status FROM works WHERE work_id='work_oa'"
        ).fetchone()) == ("An abstract.", "diamond")
        # fresh-migrate-then-idempotent still holds with 0017 in the ledger.
        assert run_migrations(conn, "project") == latest_version("project") >= 17
    finally:
        conn.close()


def test_0014_rerunnable_after_interruption(tmp_path):
    """Non-atomicity is handled by re-runnability: a leftover
    reference_entries_new from a mid-script crash does not break a retry."""
    db = tmp_path / "interrupted.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        for version, path in discover_migrations("project"):
            if version >= 14:
                continue
            conn.executescript(path.read_text(encoding="utf-8"))
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, "t"),
            )
            conn.commit()
        # simulate an interrupted earlier attempt: the temp table exists.
        conn.execute("CREATE TABLE reference_entries_new (x INTEGER)")
        conn.commit()
        assert run_migrations(conn, "project") >= 14
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "reference_entries_new" not in tables
        assert "reference_entries" in tables
    finally:
        conn.close()
