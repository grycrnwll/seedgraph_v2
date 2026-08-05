import sqlite3

import pytest

from seedgraph.db.bootstrap import ensure_cache_db
from seedgraph.db.connection import open_cache_db


def test_duplicate_file_hash_rejected():
    """The 'same PDF recognized by hash' success-criterion guard (doc 10 Phase 2)."""
    ensure_cache_db()
    conn = open_cache_db()
    try:
        conn.execute(
            "INSERT INTO source_files (source_file_id, file_hash, access_class, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("sf_aaa", "dup_hash", "open_access", "2026-01-01T00:00:00Z"),
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO source_files (source_file_id, file_hash, access_class, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("sf_bbb", "dup_hash", "open_access", "2026-01-01T00:00:00Z"),
            )
            conn.commit()
    finally:
        conn.close()
