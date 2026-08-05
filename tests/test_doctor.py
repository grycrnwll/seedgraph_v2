import sqlite3

from seedgraph.db.bootstrap import ensure_cache_db
from seedgraph.db.connection import cache_db_path
from seedgraph.doctor import collect_checks, has_failure


def test_doctor_fresh_passes():
    results = collect_checks(root=None, slug=None)
    assert not has_failure(results), [r for r in results if not r.ok and r.severity == "error"]
    assert cache_db_path().exists()


def test_doctor_tampered_schema_version_fails():
    ensure_cache_db()
    conn = sqlite3.connect(str(cache_db_path()))
    # Scope the tamper to the single max-version row — cache has >=2 migration rows
    # now, so an unscoped UPDATE would collide on the schema_migrations.version UNIQUE.
    conn.execute(
        "UPDATE schema_migrations SET version = 999 "
        "WHERE version = (SELECT MAX(version) FROM schema_migrations)"
    )
    conn.commit()
    conn.close()

    results = collect_checks()
    assert has_failure(results)
    version_check = next(r for r in results if r.name == "schema_version_cache")
    assert not version_check.ok
    assert "migrate" in version_check.detail.lower()


def test_doctor_bad_project_slug_fails():
    results = collect_checks(slug="../evil")
    assert has_failure(results)
    slug_check = next(r for r in results if r.name == "project_slug")
    assert not slug_check.ok
