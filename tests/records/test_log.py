"""The session log (stage 5 design §3.2, §3.5; slice R1): the first event of a stream
opens a record, helpers log into their parent's open record, a failed write is counted
and never raised, and a closed connection is marked. Invented ids and topics only."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import seedgraph
from seedgraph.library import store
from seedgraph.records import log, sessions
from seedgraph.records.sessions import CallInfo, SessionNameError

T0 = datetime(2026, 11, 4, 14, 32, 5, tzinfo=timezone.utc)
UTC = timezone.utc


@pytest.fixture
def lib(tmp_path):
    lib = store.open_library_at(tmp_path / "library")
    yield lib
    lib.close()


def _log(lib, conn_id="mcp-aaaa0001", name=None, *, tool="search", kind="search", args=None, items=None,
         now=T0, client="claude", meta_keys=()):
    log.log_event(lib, CallInfo(conn_id, client, name, tuple(meta_keys)), tool, kind,
                  args if args is not None else {"query": "parallel trends", "kind": "spans"},
                  items if items is not None else [], now=now, tz=UTC)


def _records(lib):
    return [dict(r) for r in lib.conn.execute(
        "SELECT record_id, session_id, state, opened_at, last_event_at, unlogged_calls, record_version, "
        "seedgraph_version, question, answer, evidence_json, starred FROM traversal_records "
        "ORDER BY opened_at, record_id")]


def _events(lib):
    return [dict(r) for r in lib.conn.execute(
        "SELECT event_id, record_id, session_id, seq, at, tool, kind, connection_id, args_json, items_json "
        "FROM traversal_events ORDER BY seq")]


def _answer(lib, record_id):
    # R2's record_answer is not built yet; answering is one state change here.
    with lib.write() as conn:
        conn.execute("UPDATE traversal_records SET state = 'answered', answered_at = ? WHERE record_id = ?",
                     (T0.isoformat(), record_id))


# --- the first event opens a record ------------------------------------------------------------

def test_the_first_event_of_a_stream_opens_a_record(lib):
    hits = [{"kind": "span", "id": "span_0001", "paper_id": "p_ate", "preview": "the ATE is identified"}]
    _log(lib, items=hits)
    [rec] = _records(lib)
    assert rec["record_id"].startswith("tr_") and rec["state"] == "open"
    assert rec["session_id"] == sessions.connection_session_id("mcp-aaaa0001")
    assert rec["opened_at"] == rec["last_event_at"] == "2026-11-04T14:32:05.000000+00:00"
    assert rec["unlogged_calls"] == 0 and rec["record_version"] == 1 and rec["starred"] == 0
    assert rec["seedgraph_version"] == seedgraph.__version__
    assert rec["question"] is None and rec["answer"] is None and rec["evidence_json"] is None
    [ev] = _events(lib)
    assert ev["record_id"] == rec["record_id"] and ev["session_id"] == rec["session_id"]
    assert ev["event_id"].startswith("te_") and ev["seq"] == 1
    assert (ev["tool"], ev["kind"], ev["connection_id"]) == ("search", "search", "mcp-aaaa0001")
    assert json.loads(ev["args_json"]) == {"query": "parallel trends", "kind": "spans"}
    assert json.loads(ev["items_json"]) == hits


def test_later_events_join_the_open_record_in_order(lib):
    _log(lib)
    _log(lib, tool="paper_show", kind="open_paper", args={"paper_id": "p_ate"},
         items=[{"claim_id": "c_1", "span_ids": ["span_0001"]}], now=T0 + timedelta(minutes=1))
    _log(lib, tool="paper_read", kind="open_text", args={"paper_id": "p_ate", "markdown_id": "md_x",
                                                         "start": 0, "end": 900}, now=T0 + timedelta(minutes=2))
    [rec] = _records(lib)
    evs = _events(lib)
    assert [e["seq"] for e in evs] == [1, 2, 3]
    assert [e["kind"] for e in evs] == ["search", "open_paper", "open_text"]
    assert {e["record_id"] for e in evs} == {rec["record_id"]}
    assert rec["last_event_at"] == "2026-11-04T14:34:05.000000+00:00"


def test_item_order_is_kept_as_given(lib):
    hits = [{"kind": "span", "id": f"span_{n:04d}", "paper_id": "p_did", "preview": "x"} for n in (9, 2, 5)]
    _log(lib, items=hits)
    assert [h["id"] for h in json.loads(_events(lib)[0]["items_json"])] == ["span_0009", "span_0002", "span_0005"]


def test_an_answered_record_is_closed_to_new_events(lib):
    _log(lib)
    first = _records(lib)[0]["record_id"]
    _answer(lib, first)
    _log(lib, now=T0 + timedelta(minutes=3))
    recs = _records(lib)
    assert [r["state"] for r in recs] == ["answered", "open"]
    assert _events(lib)[1]["record_id"] == recs[1]["record_id"] != first


def test_a_stream_has_at_most_one_open_record(lib):
    _log(lib)
    sid = sessions.connection_session_id("mcp-aaaa0001")
    with pytest.raises(sqlite3.IntegrityError):
        with lib.write() as conn:
            conn.execute("INSERT INTO traversal_records (record_id, session_id, state, opened_at, last_event_at, "
                         "seedgraph_version) VALUES ('tr_dup', ?, 'open', ?, ?, '0')",
                         (sid, T0.isoformat(), T0.isoformat()))
    assert log.open_record(lib.conn, sid) == _records(lib)[0]["record_id"]
    assert log.open_record(lib.conn, "ss_none") is None


def test_two_streams_keep_their_own_records(lib):
    _log(lib, "mcp-aaaa0001", "parallel-trends-1432")
    _log(lib, "mcp-aaaa0001", "ate-review")
    _log(lib, "mcp-bbbb0002")
    recs = _records(lib)
    assert len(recs) == 3 and len({r["session_id"] for r in recs}) == 3


def test_seq_is_one_order_across_streams(lib):
    _log(lib, "mcp-aaaa0001")
    _log(lib, "mcp-bbbb0002")
    _log(lib, "mcp-aaaa0001")
    assert [e["seq"] for e in _events(lib)] == [1, 2, 3]
    assert [e["connection_id"] for e in _events(lib)] == ["mcp-aaaa0001", "mcp-bbbb0002", "mcp-aaaa0001"]


# --- restarts -------------------------------------------------------------------------------------

def test_restart_keeps_the_open_record_of_a_named_stream(lib):
    _log(lib, "mcp-aaaa0001", "parallel-trends-1432")
    log.close_connection(lib, "mcp-aaaa0001", now=T0 + timedelta(minutes=1))
    _log(lib, "mcp-bbbb0002", "parallel-trends-1432", now=T0 + timedelta(minutes=2))
    [rec] = _records(lib)
    assert [e["connection_id"] for e in _events(lib)] == ["mcp-aaaa0001", "mcp-bbbb0002"]
    assert {e["record_id"] for e in _events(lib)} == {rec["record_id"]}


def test_restart_splits_a_connection_stream_into_a_new_record(lib):
    _log(lib, "mcp-aaaa0001")
    log.close_connection(lib, "mcp-aaaa0001", now=T0 + timedelta(minutes=1))
    _log(lib, "mcp-bbbb0002", now=T0 + timedelta(minutes=2))
    recs = _records(lib)
    assert [r["state"] for r in recs] == ["open", "open"]  # the first is now a path without answer
    assert recs[0]["session_id"] != recs[1]["session_id"]


# --- helpers --------------------------------------------------------------------------------------

def test_a_helper_logs_into_its_parents_open_record(lib):
    _log(lib, name="parallel-trends-1432")
    _log(lib, name="parallel-trends-1432/a", now=T0 + timedelta(seconds=10))
    [rec] = _records(lib)
    parent = sessions.named_session_id("claude", "parallel-trends-1432")
    helper = sessions.named_session_id("claude", "parallel-trends-1432/a")
    assert rec["session_id"] == parent
    assert [(e["record_id"], e["session_id"]) for e in _events(lib)] == [
        (rec["record_id"], parent), (rec["record_id"], helper)]


def test_a_helper_with_no_open_parent_record_opens_one_for_the_parent(lib):
    _log(lib, name="parallel-trends-1432/b")
    [rec] = _records(lib)
    assert rec["session_id"] == sessions.named_session_id("claude", "parallel-trends-1432")
    assert log.open_record(lib.conn, sessions.named_session_id("claude", "parallel-trends-1432/b")) is None
    _log(lib, name="parallel-trends-1432", now=T0 + timedelta(seconds=5))
    assert len(_records(lib)) == 1


# --- refusals happen before the write --------------------------------------------------------------

def test_an_invalid_session_name_is_raised_not_counted(lib):
    with pytest.raises(SessionNameError):
        _log(lib, "mcp-eeee0005", name="Parallel Trends")
    assert log.unlogged_calls("mcp-eeee0005") == 0
    assert _events(lib) == []


def test_an_unknown_event_kind_is_a_programming_error(lib):
    with pytest.raises(ValueError):
        _log(lib, kind="graph_list")
    assert _events(lib) == []


# --- the unlogged-call count -------------------------------------------------------------------------

def test_a_failed_write_is_counted_and_written_with_the_next_event(lib, monkeypatch):
    conn_id = "mcp-ffff0006"
    real = log._write_event

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(log, "_write_event", locked)
    _log(lib, conn_id)  # never raises
    _log(lib, conn_id, now=T0 + timedelta(seconds=1))
    assert log.unlogged_calls(conn_id) == 2
    assert _events(lib) == []
    monkeypatch.setattr(log, "_write_event", real)
    _log(lib, conn_id, now=T0 + timedelta(seconds=2))
    [rec] = _records(lib)
    assert rec["unlogged_calls"] == 2
    assert log.unlogged_calls(conn_id) == 0
    _log(lib, conn_id, now=T0 + timedelta(seconds=3))
    assert _records(lib)[0]["unlogged_calls"] == 2


def test_a_real_lock_is_counted_not_raised(lib, tmp_path):
    conn_id = "mcp-gggg0007"
    other = store.connect(lib.db_path)
    lib.conn.execute("PRAGMA busy_timeout=0")
    other.execute("BEGIN IMMEDIATE")
    try:
        _log(lib, conn_id)
    finally:
        other.execute("ROLLBACK")
        other.close()
    assert log.unlogged_calls(conn_id) == 1
    _log(lib, conn_id, now=T0 + timedelta(seconds=1))
    assert _records(lib)[0]["unlogged_calls"] == 1
    assert len(_events(lib)) == 1


def test_counts_are_kept_per_connection(lib, monkeypatch):
    monkeypatch.setattr(log, "_write_event", lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("x")))
    _log(lib, "mcp-hhhh0008")
    assert log.unlogged_calls("mcp-hhhh0008") == 1
    assert log.unlogged_calls("mcp-iiii0009") == 0


# --- connections -------------------------------------------------------------------------------------

def test_meta_keys_are_kept_as_a_sorted_union_on_the_connection(lib):
    _log(lib, meta_keys=("progressToken",))
    _log(lib, meta_keys=("claudecode/toolUseId", "progressToken"), now=T0 + timedelta(seconds=1))
    row = lib.conn.execute("SELECT meta_keys_json FROM record_connections WHERE connection_id = 'mcp-aaaa0001'"
                           ).fetchone()
    assert json.loads(row[0]) == ["claudecode/toolUseId", "progressToken"]


def test_close_connection_marks_it_once(lib):
    _log(lib)
    log.close_connection(lib, "mcp-aaaa0001", now=T0 + timedelta(minutes=1))
    log.close_connection(lib, "mcp-aaaa0001", now=T0 + timedelta(minutes=9))
    row = lib.conn.execute("SELECT closed_at FROM record_connections WHERE connection_id = 'mcp-aaaa0001'"
                           ).fetchone()
    assert row[0] == "2026-11-04T14:33:05.000000+00:00"


def test_close_connection_of_a_connection_that_logged_nothing_is_a_no_op(lib):
    log.close_connection(lib, "mcp-zzzz0000", now=T0)
    assert lib.conn.execute("SELECT COUNT(*) FROM record_connections").fetchone()[0] == 0


def test_close_connection_never_raises(lib):
    lib.conn.close()
    log.close_connection(lib, "mcp-aaaa0001", now=T0)  # the server is shutting down: swallow it
    lib.conn = store.connect(lib.db_path)


def test_logging_writes_no_graph_links(lib):
    _log(lib, args={"query": "serial correlation", "graph": "diff_in_diff"})
    assert lib.conn.execute("SELECT COUNT(*) FROM record_graphs").fetchone()[0] == 0
