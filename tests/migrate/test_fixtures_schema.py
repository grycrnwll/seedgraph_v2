"""The frozen v2 schema (design §9): a verbatim copy of v2's numbered SQL that the
migration and its fixtures can still apply after the old v2 code is deleted."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from seedgraph.migrate import v2_schema

_LIVE_SCHEMA = Path(__file__).resolve().parents[2] / "src" / "seedgraph" / "db" / "schema"


def _schema_rows(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
    ).fetchall()


@pytest.mark.parametrize("scope", ["project", "cache"])
def test_frozen_copy_matches_live_v2_schema_byte_for_byte(scope):
    live_dir = _LIVE_SCHEMA / scope
    if not live_dir.is_dir():
        pytest.skip("v2 schema deleted; the frozen copy is now the only source")
    live = {p.name: p.read_bytes() for p in sorted(live_dir.glob("*.sql"))}
    frozen = {p.name: p.read_bytes() for _, p in v2_schema.schema_files(scope)}
    assert sorted(frozen) == sorted(live)
    for name, data in live.items():
        assert frozen[name] == data, f"{scope}/{name} drifted from the live v2 schema"


def test_latest_versions_match_the_v2_homes_in_use():
    # The real v2 project.db files are at 17 and cache.db at 3 (gaps 0006/0011 allowed).
    assert v2_schema.latest_version("project") == 17
    assert v2_schema.latest_version("cache") == 3
    assert [v for v, _ in v2_schema.schema_files("project")][:6] == [1, 2, 3, 4, 5, 7]


@pytest.mark.parametrize("scope", ["project", "cache"])
def test_create_v2_db_applies_every_file_in_order(tmp_path, scope):
    db = tmp_path / f"{scope}.db"
    version = v2_schema.create_v2_db(db, scope, applied_at="2026-01-01T00:00:00+00:00")
    assert version == v2_schema.latest_version(scope)
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT version, applied_at FROM schema_migrations ORDER BY version").fetchall()
        assert [v for v, _ in rows] == [v for v, _ in v2_schema.schema_files(scope)]
        assert {a for _, a in rows} == {"2026-01-01T00:00:00+00:00"}
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()


def test_create_v2_db_refuses_an_existing_file(tmp_path):
    db = tmp_path / "project.db"
    db.write_bytes(b"")
    with pytest.raises(FileExistsError):
        v2_schema.create_v2_db(db, "project")


@pytest.mark.parametrize("scope", ["project", "cache"])
def test_frozen_schema_equals_what_v2_runner_builds(tmp_path, scope):
    migrations = pytest.importorskip("seedgraph.db.migrations")
    frozen_db = tmp_path / "frozen.db"
    v2_schema.create_v2_db(frozen_db, scope)
    live = sqlite3.connect(tmp_path / "live.db")
    frozen = sqlite3.connect(frozen_db)
    try:
        migrations.run_migrations(live, scope)
        assert _schema_rows(frozen) == _schema_rows(live)
    finally:
        live.close()
        frozen.close()
