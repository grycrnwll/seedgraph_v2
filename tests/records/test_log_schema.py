"""The records and spot-check schema (stage 5 design §3.7, §5.5; slice R1): the tables a
fresh library gets, their checks, and the ``spot_check_accuracy`` view's latest-mark
rule. Invented rows only."""

from __future__ import annotations

import sqlite3

import pytest

from seedgraph.library import store

NOW = "2026-11-04T14:32:05.000000+00:00"
RECORD_TABLES = {"record_sessions", "record_connections", "traversal_records", "traversal_events",
                 "record_graphs"}
SPOT_CHECK_TABLES = {"spot_check_samples", "spot_check_items"}


@pytest.fixture
def lib(tmp_path):
    lib = store.open_library_at(tmp_path / "library")
    yield lib
    lib.close()


def _names(conn, kind):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = ?", (kind,))}


def test_a_fresh_library_has_the_record_and_spot_check_tables(lib):
    assert RECORD_TABLES | SPOT_CHECK_TABLES <= _names(lib.conn, "table")
    assert "spot_check_accuracy" in _names(lib.conn, "view")
    versions = [r[0] for r in lib.conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
    assert versions[-2:] == [11, 12]
    # nothing seeded: a fresh library stays empty (and so replaceable by a migration)
    for t in RECORD_TABLES | SPOT_CHECK_TABLES:
        assert lib.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0


def _session(conn, sid="ss_a", kind="named", name="ate-review", connection_id=None):
    conn.execute("INSERT INTO record_sessions (session_id, key_kind, client, name, connection_id, first_at, "
                 "last_at) VALUES (?, ?, 'claude', ?, ?, ?, ?)", (sid, kind, name, connection_id, NOW, NOW))


@pytest.mark.parametrize("sql", [
    "INSERT INTO record_sessions (session_id, key_kind, name, first_at, last_at) "
    "VALUES ('ss_x', 'subagent', 'ate', '{0}', '{0}')",
    # a connection session names its connection; a named one does not
    "INSERT INTO record_sessions (session_id, key_kind, name, first_at, last_at) "
    "VALUES ('ss_x', 'connection', '2026-11-04 14:32', '{0}', '{0}')",
    "INSERT INTO traversal_records (record_id, session_id, state, opened_at, last_event_at, seedgraph_version) "
    "VALUES ('tr_x', 'ss_a', 'abandoned', '{0}', '{0}', '0')",
    "INSERT INTO traversal_records (record_id, session_id, state, opened_at, last_event_at, seedgraph_version, "
    "starred) VALUES ('tr_x', 'ss_a', 'open', '{0}', '{0}', '0', 2)",
    "INSERT INTO record_graphs (record_id, slug, how) VALUES ('tr_a', 'diff_in_diff', 'mentioned')",
])
def test_checks_refuse_bad_rows(lib, sql):
    with lib.write() as conn:
        _session(conn)
        conn.execute("INSERT INTO traversal_records (record_id, session_id, state, opened_at, last_event_at, "
                     "answered_at, seedgraph_version) VALUES ('tr_a', 'ss_a', 'answered', ?, ?, ?, '0')",
                     (NOW, NOW, NOW))
    with pytest.raises(sqlite3.IntegrityError):
        with lib.write() as conn:
            conn.execute(sql.format(NOW))


def test_answered_and_answered_at_go_together(lib):
    with lib.write() as conn:
        _session(conn)
    with pytest.raises(sqlite3.IntegrityError):
        with lib.write() as conn:
            conn.execute("INSERT INTO traversal_records (record_id, session_id, state, opened_at, last_event_at, "
                         "seedgraph_version) VALUES ('tr_b', 'ss_a', 'answered', ?, ?, '0')", (NOW, NOW))


def test_an_event_kind_outside_the_four_is_refused(lib):
    with lib.write() as conn:
        _session(conn)
        conn.execute("INSERT INTO record_connections (connection_id, client, started_at) "
                     "VALUES ('mcp-a', 'claude', ?)", (NOW,))
        conn.execute("INSERT INTO traversal_records (record_id, session_id, state, opened_at, last_event_at, "
                     "seedgraph_version) VALUES ('tr_a', 'ss_a', 'open', ?, ?, '0')", (NOW, NOW))
        for n, kind in enumerate(("search", "open_paper", "open_text", "open_concept"), start=1):
            conn.execute("INSERT INTO traversal_events (event_id, record_id, session_id, seq, at, tool, kind, "
                         "connection_id, args_json, items_json) VALUES (?, 'tr_a', 'ss_a', ?, ?, 't', ?, "
                         "'mcp-a', '{}', '[]')", (f"te_{n}", n, NOW, kind))
    with pytest.raises(sqlite3.IntegrityError):
        with lib.write() as conn:
            conn.execute("INSERT INTO traversal_events (event_id, record_id, session_id, seq, at, tool, kind, "
                         "connection_id, args_json, items_json) VALUES ('te_9', 'tr_a', 'ss_a', 9, ?, "
                         "'graph_list', 'orientation', 'mcp-a', '{}', '[]')", (NOW,))


def test_two_connection_sessions_cannot_share_a_time_name(lib):
    with lib.write() as conn:
        for cid in ("mcp-a", "mcp-b"):
            conn.execute("INSERT INTO record_connections (connection_id, client, started_at) VALUES (?, 'c', ?)",
                         (cid, NOW))
        _session(conn, "ss_a", "connection", "2026-11-04 14:32", "mcp-a")
    with pytest.raises(sqlite3.IntegrityError):
        with lib.write() as conn:
            _session(conn, "ss_b", "connection", "2026-11-04 14:32", "mcp-b")


# --- spot-check -----------------------------------------------------------------------------------

def _item(conn, sample, claim, mark, marked_at, *, ordinal=1, ctype="result", model="model not recorded"):
    conn.execute("INSERT INTO spot_check_items (sample_id, claim_id, ordinal, paper_id, extraction_run_id, "
                 "claim_type, model_key, claim_fingerprint, mark, marked_at) "
                 "VALUES (?, ?, ?, 'p_ate', 'run_1', ?, ?, '{}', ?, ?)",
                 (sample, claim, ordinal, ctype, model, mark, marked_at))


def test_spot_check_accuracy_counts_the_latest_mark_per_claim(lib):
    with lib.write() as conn:
        for sid in ("sc_1", "sc_2"):
            conn.execute("INSERT INTO spot_check_samples (sample_id, seed, size, created_at) VALUES (?, 7, 20, ?)",
                         (sid, NOW))
        _item(conn, "sc_1", "c_1", "right", "2026-11-04T10:00:00.000000+00:00", ordinal=1)
        _item(conn, "sc_1", "c_2", "wrong", "2026-11-04T10:01:00.000000+00:00", ordinal=2)
        _item(conn, "sc_1", "c_3", None, None, ordinal=3)                      # not marked yet
        _item(conn, "sc_1", "c_4", "partly", "2026-11-04T10:02:00.000000+00:00", ordinal=4,
              ctype="identification_assumption", model="model-x")
        # the same claim again in a later sample: only its latest mark counts
        _item(conn, "sc_2", "c_2", "right", "2026-11-04T11:00:00.000000+00:00", ordinal=1)
    rows = {(r["claim_type"], r["model_key"]): tuple(r)[2:] for r in lib.conn.execute(
        "SELECT claim_type, model_key, n_right, n_partly, n_wrong, n_marked FROM spot_check_accuracy")}
    assert rows == {("result", "model not recorded"): (2, 0, 0, 2),
                    ("identification_assumption", "model-x"): (0, 1, 0, 1)}


@pytest.mark.parametrize("mark", ["half", "RIGHT", ""])
def test_a_mark_outside_the_three_is_refused(lib, mark):
    with lib.write() as conn:
        conn.execute("INSERT INTO spot_check_samples (sample_id, seed, size, created_at) VALUES ('sc_1', 1, 20, ?)",
                     (NOW,))
    with pytest.raises(sqlite3.IntegrityError):
        with lib.write() as conn:
            _item(conn, "sc_1", "c_1", mark, NOW)


def test_an_item_ordinal_is_unique_in_its_sample(lib):
    with lib.write() as conn:
        conn.execute("INSERT INTO spot_check_samples (sample_id, seed, size, created_at) VALUES ('sc_1', 1, 20, ?)",
                     (NOW,))
        _item(conn, "sc_1", "c_1", None, None, ordinal=1)
    with pytest.raises(sqlite3.IntegrityError):
        with lib.write() as conn:
            _item(conn, "sc_1", "c_2", None, None, ordinal=1)
