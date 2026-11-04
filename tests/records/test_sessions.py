"""Session identity (stage 5 design §3.1; slice R1): the key order, name validation,
helpers, the start-date/time name of a connection session, and the two-connection
warning. Invented names and topics only."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

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


def _resolve(lib, conn_id, name=None, *, client="claude", now=T0):
    return sessions.resolve(lib, CallInfo(conn_id, client, name), now=now, tz=UTC)


def _search(lib, conn_id, name=None, *, client="claude", now=T0):
    log.log_event(lib, CallInfo(conn_id, client, name), "search", "search",
                  {"query": "parallel trends"}, [], now=now, tz=UTC)


# --- key order ----------------------------------------------------------------------------

def test_a_named_call_is_keyed_by_client_and_name_not_connection(lib):
    a = _resolve(lib, "mcp-aaaa0001", "parallel-trends-1432")
    b = _resolve(lib, "mcp-bbbb0002", "parallel-trends-1432")
    assert a.key_kind == "named" and a.name == "parallel-trends-1432"
    assert a.session_id == b.session_id == sessions.named_session_id("claude", "parallel-trends-1432")
    assert a.session_id.startswith("ss_") and a.connection_id is None


def test_the_same_name_from_another_client_is_another_stream(lib):
    a = _resolve(lib, "mcp-aaaa0001", "ate-review", client="claude")
    b = _resolve(lib, "mcp-bbbb0002", "ate-review", client="codex")
    assert a.session_id != b.session_id


def test_an_unnamed_call_falls_back_to_the_connection(lib):
    a = _resolve(lib, "mcp-aaaa0001")
    again = _resolve(lib, "mcp-aaaa0001", now=T0 + timedelta(minutes=5))
    other = _resolve(lib, "mcp-bbbb0002")
    assert a.key_kind == "connection" and a.connection_id == "mcp-aaaa0001"
    assert a.session_id == again.session_id == sessions.connection_session_id("mcp-aaaa0001")
    assert other.session_id != a.session_id
    assert a.parent_session_id is None and a.owner_session_id == a.session_id


def test_session_ids_are_deterministic_hashes_of_the_key():
    assert sessions.named_session_id("claude", "ate-review") == sessions.named_session_id("claude", "ate-review")
    assert sessions.named_session_id(None, "ate-review") != sessions.named_session_id("claude", "ate-review")
    # a name can never collide with a connection key of the same text
    assert sessions.named_session_id(None, "mcp-aaaa0001") != sessions.connection_session_id("mcp-aaaa0001")


def test_resolve_records_the_stream_and_its_connection(lib):
    s = _resolve(lib, "mcp-aaaa0001", "parallel-trends-1432")
    row = lib.conn.execute("SELECT key_kind, client, name, parent_session_id, first_at, last_at "
                           "FROM record_sessions WHERE session_id = ?", (s.session_id,)).fetchone()
    assert tuple(row) == ("named", "claude", "parallel-trends-1432", None,
                          "2026-11-04T14:32:05.000000+00:00", "2026-11-04T14:32:05.000000+00:00")
    conn_row = lib.conn.execute("SELECT client, started_at, closed_at FROM record_connections "
                                "WHERE connection_id = 'mcp-aaaa0001'").fetchone()
    assert tuple(conn_row) == ("claude", "2026-11-04T14:32:05.000000+00:00", None)


# --- name validation ------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "ate", "parallel-trends-1432", "a" * 48, "0-serial-correlation", "parallel-trends-1432/a",
    "parallel-trends-1432/b2", "ate-review/" + "x" * 48,
])
def test_valid_names_are_accepted(name):
    assert sessions.validate_name(name) == name


@pytest.mark.parametrize("name", [
    "at", "a" * 49, "-ate", "Parallel-Trends", "parallel trends", "ate_review", "ate-review\n",
    "parallel-trends-1432/", "parallel-trends-1432/-a", "a/b/c", "ate-review/A", "/a", "ate.review",
])
def test_invalid_names_are_refused_with_a_plain_next_action(name):
    with pytest.raises(SessionNameError) as info:
        sessions.validate_name(name)
    err = info.value
    assert err.code == "invalid_session"
    assert "parallel-trends-1432" in err.next_action and "/a" in err.next_action
    assert repr(name) in str(err)


@pytest.mark.parametrize("name", [None, ""])
def test_no_name_means_the_connection(name):
    assert sessions.validate_name(name) is None


def test_resolve_refuses_an_invalid_name_and_writes_nothing(lib):
    with pytest.raises(SessionNameError):
        _resolve(lib, "mcp-aaaa0001", "Parallel Trends")
    assert lib.conn.execute("SELECT COUNT(*) FROM record_sessions").fetchone()[0] == 0
    assert lib.conn.execute("SELECT COUNT(*) FROM record_connections").fetchone()[0] == 0


# --- helpers --------------------------------------------------------------------------------

def test_a_helper_points_to_its_parent_which_is_created_if_missing(lib):
    h = _resolve(lib, "mcp-aaaa0001", "parallel-trends-1432/a")
    parent_id = sessions.named_session_id("claude", "parallel-trends-1432")
    assert h.key_kind == "named" and h.name == "parallel-trends-1432/a"
    assert h.parent_session_id == parent_id and h.owner_session_id == parent_id and h.is_helper
    row = lib.conn.execute("SELECT key_kind, name, parent_session_id FROM record_sessions WHERE session_id = ?",
                           (parent_id,)).fetchone()
    assert tuple(row) == ("named", "parallel-trends-1432", None)
    parent = _resolve(lib, "mcp-aaaa0001", "parallel-trends-1432")
    assert parent.session_id == parent_id and not parent.is_helper


def test_family_is_the_parent_then_its_helpers_from_either_end(lib):
    _resolve(lib, "mcp-aaaa0001", "parallel-trends-1432/b")
    _resolve(lib, "mcp-aaaa0001", "parallel-trends-1432/a")
    _resolve(lib, "mcp-aaaa0001", "ate-review/a")  # another family
    parent = sessions.named_session_id("claude", "parallel-trends-1432")
    a = sessions.named_session_id("claude", "parallel-trends-1432/a")
    b = sessions.named_session_id("claude", "parallel-trends-1432/b")
    expected = [parent, *sorted([a, b])]
    assert sessions.family(lib, parent) == expected
    assert sessions.family(lib, a) == expected
    lone = _resolve(lib, "mcp-cccc0003")
    assert sessions.family(lib, lone.session_id) == [lone.session_id]
    assert sessions.family(lib, "ss_unknown") == []


# --- restarts and two connections ---------------------------------------------------------------

def test_restart_keeps_a_named_stream_and_does_not_warn(lib):
    _search(lib, "mcp-aaaa0001", "parallel-trends-1432")
    log.close_connection(lib, "mcp-aaaa0001", now=T0 + timedelta(minutes=1))
    s = _resolve(lib, "mcp-bbbb0002", "parallel-trends-1432", now=T0 + timedelta(minutes=2))
    assert s.session_id == sessions.named_session_id("claude", "parallel-trends-1432")
    assert s.warnings == ()


def test_restart_splits_a_connection_stream(lib):
    _search(lib, "mcp-aaaa0001")
    log.close_connection(lib, "mcp-aaaa0001", now=T0 + timedelta(minutes=1))
    s = _resolve(lib, "mcp-bbbb0002", now=T0 + timedelta(minutes=2))
    assert s.session_id != sessions.connection_session_id("mcp-aaaa0001")


def test_the_same_name_from_two_live_connections_warns(lib):
    _search(lib, "mcp-aaaa0001", "parallel-trends-1432")
    s = _resolve(lib, "mcp-bbbb0002", "parallel-trends-1432", now=T0 + timedelta(minutes=2))
    assert s.warnings == (sessions.WARN_TWO_CONNECTIONS,)
    assert sessions.WARN_TWO_CONNECTIONS == "session name used from two connections"
    assert sessions.shared_by_connections(lib, s.session_id) is False  # B has logged nothing yet
    _search(lib, "mcp-bbbb0002", "parallel-trends-1432", now=T0 + timedelta(minutes=3))
    assert sessions.shared_by_connections(lib, s.session_id) is True
    # the first window, still live, sees it too
    again = _resolve(lib, "mcp-aaaa0001", "parallel-trends-1432", now=T0 + timedelta(minutes=4))
    assert again.warnings == (sessions.WARN_TWO_CONNECTIONS,)


def test_a_connection_closed_after_the_other_started_still_counts_as_two(lib):
    _search(lib, "mcp-aaaa0001", "ate-review")
    _search(lib, "mcp-bbbb0002", "ate-review", now=T0 + timedelta(minutes=1))
    log.close_connection(lib, "mcp-aaaa0001", now=T0 + timedelta(minutes=2))
    assert sessions.shared_by_connections(lib, sessions.named_session_id("claude", "ate-review")) is True


def test_one_connection_never_warns(lib):
    _search(lib, "mcp-aaaa0001", "ate-review")
    s = _resolve(lib, "mcp-aaaa0001", "ate-review", now=T0 + timedelta(minutes=5))
    assert s.warnings == ()


# --- the start-date/time name of a connection session --------------------------------------------

def test_a_connection_session_is_named_by_its_start_date_and_time(lib):
    s = _resolve(lib, "mcp-aaaa0001")
    assert s.name == "2026-11-04 14:32"
    later = _resolve(lib, "mcp-aaaa0001", now=T0 + timedelta(hours=1))
    assert later.name == "2026-11-04 14:32"  # a label fixed at the start


def test_same_minute_connection_sessions_get_a_suffix_and_stay_apart(lib):
    a = _resolve(lib, "mcp-aaaa0001")
    b = _resolve(lib, "mcp-bbbb0002", now=T0 + timedelta(seconds=30))
    c = _resolve(lib, "mcp-cccc0003", now=T0 + timedelta(seconds=40), client="codex")
    d = _resolve(lib, "mcp-dddd0004", now=T0 + timedelta(minutes=1))
    assert [a.name, b.name, c.name, d.name] == [
        "2026-11-04 14:32", "2026-11-04 14:32-2", "2026-11-04 14:32-3", "2026-11-04 14:33"]
    assert len({a.session_id, b.session_id, c.session_id}) == 3


def test_a_named_session_does_not_take_a_time_name(lib):
    _resolve(lib, "mcp-aaaa0001", "ate-review")
    s = _resolve(lib, "mcp-bbbb0002")
    assert s.name == "2026-11-04 14:32"


def test_the_time_name_is_local_time_by_default(lib):
    s = sessions.resolve(lib, CallInfo("mcp-aaaa0001", "claude", None), now=T0)
    assert s.name == T0.astimezone().strftime("%Y-%m-%d %H:%M")


def test_the_time_name_follows_the_given_zone(lib):
    est = timezone(timedelta(hours=-5))
    s = sessions.resolve(lib, CallInfo("mcp-aaaa0001", "claude", None), now=T0, tz=est)
    assert s.name == "2026-11-04 09:32"
