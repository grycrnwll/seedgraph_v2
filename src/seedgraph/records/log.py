"""The session log (design §3.2, §3.5; slice R1).

Each logged read call becomes one ``traversal_events`` row, written after the tool body
succeeded, in the call's own library handle, as one short write (the stage 1 busy
timeout). The first event of a stream with no open record **opens** one; a helper's
events go into its parent's open record (opening one if there is none). There is no
"start record" call, and a long pause never splits ownership.

A failed write never fails the read: it is counted in memory per connection and written
onto the next successful event's record (``unlogged_calls``, "n calls could not be
logged"). Only database errors count; an invalid session name or event kind is raised,
so the adapter refuses the call before it runs.

Events hold ids, ranges and short previews only, never paper text.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, tzinfo
from typing import Any

from .. import __version__
from ..library.store import Library
from ..work.store import fmt_ts, now_utc
from . import sessions
from .sessions import CallInfo

KINDS: tuple[str, ...] = ("search", "open_paper", "open_text", "open_concept")
RECORD_VERSION = 1

# connection_id -> calls that could not be logged since the last successful event
_UNLOGGED: dict[str, int] = {}
_UNLOGGED_LOCK = threading.Lock()


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _time_id(prefix: str, now: datetime) -> str:
    """``prefix`` + 14 hex digits of UTC microseconds + 6 random hex: time-ordered."""
    micros = int(now.timestamp() * 1_000_000)
    return f"{prefix}_{micros:014x}{secrets.token_hex(3)}"


def log_event(lib: Library, call: CallInfo, tool: str, kind: str, args: Mapping[str, Any],
              items: Sequence[Any], *, now: datetime | None = None, tz: tzinfo | None = None) -> None:
    """Log one call of ``tool`` (an event of ``kind``) with what the agent passed (``args``)
    and what it was shown (``items``, in reply order). Never raises on a write failure."""
    if kind not in KINDS:
        raise ValueError(f"unknown event kind {kind!r} (expected one of {', '.join(KINDS)})")
    sessions.validate_name(call.session_name)
    args_json, items_json = _dumps(dict(args)), _dumps(list(items))
    now = now or now_utc()
    with _UNLOGGED_LOCK:
        pending = _UNLOGGED.pop(call.connection_id, 0)
    try:
        _write_event(lib, call, tool, kind, args_json, items_json, now=now, tz=tz, pending=pending)
    except sqlite3.Error:
        with _UNLOGGED_LOCK:
            _UNLOGGED[call.connection_id] = _UNLOGGED.get(call.connection_id, 0) + pending + 1
    except BaseException:
        with _UNLOGGED_LOCK:  # not a write failure: keep the earlier count and let it surface
            _UNLOGGED[call.connection_id] = _UNLOGGED.get(call.connection_id, 0) + pending
        raise


def _write_event(lib: Library, call: CallInfo, tool: str, kind: str, args_json: str, items_json: str, *,
                 now: datetime, tz: tzinfo | None, pending: int) -> None:
    ts = fmt_ts(now)
    with lib.write() as conn:
        stream = sessions.resolve(lib, call, now=now, tz=tz)
        record_id = open_record(conn, stream.owner_session_id) or _open_record(conn, stream.owner_session_id,
                                                                               now)
        seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM traversal_events").fetchone()[0]
        conn.execute("INSERT INTO traversal_events (event_id, record_id, session_id, seq, at, tool, kind, "
                     "connection_id, args_json, items_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (_time_id("te", now), record_id, stream.session_id, seq, ts, tool, kind,
                      call.connection_id, args_json, items_json))
        conn.execute("UPDATE traversal_records SET last_event_at = MAX(last_event_at, ?), "
                     "unlogged_calls = unlogged_calls + ? WHERE record_id = ?", (ts, pending, record_id))


def open_record(conn: sqlite3.Connection, session_id: str) -> str | None:
    """The open record of stream ``session_id`` (at most one), or ``None``."""
    row = conn.execute("SELECT record_id FROM traversal_records WHERE session_id = ? AND state = 'open'",
                       (session_id,)).fetchone()
    return row[0] if row else None


def _open_record(conn: sqlite3.Connection, session_id: str, now: datetime) -> str:
    ts = fmt_ts(now)
    record_id = _time_id("tr", now)
    conn.execute("INSERT INTO traversal_records (record_id, session_id, state, opened_at, last_event_at, "
                 "record_version, seedgraph_version) VALUES (?, ?, 'open', ?, ?, ?, ?)",
                 (record_id, session_id, ts, ts, RECORD_VERSION, __version__))
    return record_id


def unlogged_calls(connection_id: str) -> int:
    """Calls on this connection that could not be logged and are not yet written down."""
    with _UNLOGGED_LOCK:
        return _UNLOGGED.get(connection_id, 0)


def close_connection(lib: Library, connection_id: str, *, now: datetime | None = None) -> None:
    """Mark the connection closed (the stdio server calls this when stdin ends). The first
    close counts; a connection that logged nothing has no row and is left alone. Never
    raises: the server is shutting down."""
    try:
        if lib.conn.execute("SELECT 1 FROM record_connections WHERE connection_id = ? AND closed_at IS NULL",
                            (connection_id,)).fetchone() is None:
            return
        with lib.write() as conn:
            conn.execute("UPDATE record_connections SET closed_at = ? WHERE connection_id = ? "
                         "AND closed_at IS NULL", (fmt_ts(now or now_utc()), connection_id))
    except sqlite3.Error:
        pass
