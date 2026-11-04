"""Session identity: which stream a logged call belongs to (design §3.1; slice R1).

A **session** is one stream of logged calls. Its key, in this order:

1. **Named**: the call passes ``session``, a short name the agent chose. The key is
   ``(client, name)``, scoped to the library, so the stream survives reconnects and server
   restarts. ``<parent>/<suffix>`` is a **helper stream** of ``<parent>`` (one level); it
   owns no record of its own until it answers, and logs into its parent's open record.
2. **Connection** (no name): the key is the connection, the server process's
   ``worker_id``. A restart is a new connection and so a new stream. The session is named
   by its start date and time, local (``2026-11-04 14:32``, ``-2`` on a same-minute
   collision); that name is a label only, and two windows started in the same minute stay
   apart by connection.

A named stream reached from a second connection while the first is still live carries
:data:`WARN_TWO_CONNECTIONS`; a clean restart (the first closed before the second started)
does not.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, tzinfo

from ..errors import ValidationError
from ..library.store import Library
from ..work.store import fmt_ts, now_utc

KEY_NAMED = "named"
KEY_CONNECTION = "connection"
WARN_TWO_CONNECTIONS = "session name used from two connections"

# A helper suffix may be one character (`<yours>/a`, design §3.1 and §8.1); the parent part
# is the 3-48 character form.
_NAME_RE = re.compile(r"(?P<parent>[a-z0-9][a-z0-9-]{2,47})(?:/(?P<suffix>[a-z0-9][a-z0-9-]{0,47}))?")
NAME_NEXT_ACTION = (
    "Pass session as 3 to 48 lowercase letters, digits or hyphens, starting with a letter or "
    "digit, for example parallel-trends-1432. A helper adds one suffix: parallel-trends-1432/a. "
    "Or leave session out.")


class SessionNameError(ValidationError):
    """An invalid session name; refused before the call runs, with a plain next action."""

    code = "invalid_session"

    def __init__(self, name: str) -> None:
        super().__init__(f"session name {name!r} is not valid")
        self.name = name
        self.next_action = NAME_NEXT_ACTION


@dataclass(frozen=True)
class CallInfo:
    """Who made a logged call, built once per call by the MCP adapter (§4)."""

    connection_id: str                  # the server process's worker_id
    client: str | None = None           # 'claude', 'codex' or the client's own name
    session_name: str | None = None     # the `session` argument as passed
    meta_keys: tuple[str, ...] = ()     # the keys (never the values) of the request's MCP _meta


@dataclass(frozen=True)
class Stream:
    """A resolved stream. ``owner_session_id`` is the stream whose open record the call
    joins: the parent for a helper, else the stream itself."""

    session_id: str
    key_kind: str
    client: str | None
    name: str                           # the session name, or the start date/time label
    parent_session_id: str | None
    connection_id: str | None           # connection streams only
    warnings: tuple[str, ...] = ()

    @property
    def is_helper(self) -> bool:
        return self.parent_session_id is not None

    @property
    def owner_session_id(self) -> str:
        return self.parent_session_id or self.session_id


# --- keys -----------------------------------------------------------------------------------

def _hash_id(*parts: str) -> str:
    return "ss_" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def named_session_id(client: str | None, name: str) -> str:
    """``'ss_' + hash('named', client, name)``."""
    return _hash_id(KEY_NAMED, client or "", name)


def connection_session_id(connection_id: str) -> str:
    """``'ss_' + hash('connection', connection_id)``."""
    return _hash_id(KEY_CONNECTION, connection_id)


def validate_name(name: str | None) -> str | None:
    """The name unchanged, or ``None`` for no name (``None`` or ``''``). Anything else that
    is not ``[a-z0-9][a-z0-9-]{2,47}`` with an optional ``/suffix`` raises
    :class:`SessionNameError`. Nothing is normalized: a refused name is fixed by the agent."""
    if name is None or name == "":
        return None
    if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
        raise SessionNameError(str(name))
    return name


# --- resolve --------------------------------------------------------------------------------

def resolve(lib: Library, call: CallInfo, *, now: datetime | None = None, tz: tzinfo | None = None) -> Stream:
    """The stream of ``call``, creating its rows (and a helper's parent) on first use, and
    noting the connection. Safe inside an open ``lib.write()`` (it nests as a savepoint).
    ``tz`` is the zone of a connection session's start-time name (default: local)."""
    name = validate_name(call.session_name)
    now = now or now_utc()
    ts = fmt_ts(now)
    with lib.write() as conn:
        _touch_connection(conn, call, ts)
        if name is None:
            stream = _connection_stream(conn, call, now, ts, tz)
        else:
            stream = _named_stream(conn, call.client, name, ts)
            if _overlap(conn, stream.session_id, current=call.connection_id):
                stream = replace(stream, warnings=(WARN_TWO_CONNECTIONS,))
    return stream


def _touch_connection(conn: sqlite3.Connection, call: CallInfo, ts: str) -> None:
    row = conn.execute("SELECT client, meta_keys_json FROM record_connections WHERE connection_id = ?",
                       (call.connection_id,)).fetchone()
    if row is None:
        conn.execute("INSERT INTO record_connections (connection_id, client, started_at, meta_keys_json) "
                     "VALUES (?, ?, ?, ?)",
                     (call.connection_id, call.client, ts, json.dumps(sorted(set(call.meta_keys)))))
        return
    keys = sorted(set(json.loads(row["meta_keys_json"])) | set(call.meta_keys))
    if keys != json.loads(row["meta_keys_json"]) or (row["client"] is None and call.client):
        conn.execute("UPDATE record_connections SET meta_keys_json = ?, client = COALESCE(client, ?) "
                     "WHERE connection_id = ?", (json.dumps(keys), call.client, call.connection_id))


def _upsert_named(conn: sqlite3.Connection, client: str | None, name: str, parent: str | None, ts: str,
                  *, touch: bool) -> str:
    sid = named_session_id(client, name)
    conn.execute("INSERT INTO record_sessions (session_id, key_kind, client, name, parent_session_id, first_at, "
                 "last_at) VALUES (?, 'named', ?, ?, ?, ?, ?) ON CONFLICT(session_id) DO "
                 + ("UPDATE SET last_at = MAX(last_at, excluded.last_at)" if touch else "NOTHING"),
                 (sid, client, name, parent, ts, ts))
    return sid


def _named_stream(conn: sqlite3.Connection, client: str | None, name: str, ts: str) -> Stream:
    parent_name, _, suffix = name.partition("/")
    parent = _upsert_named(conn, client, parent_name, None, ts, touch=False) if suffix else None
    sid = _upsert_named(conn, client, name, parent, ts, touch=True)
    return Stream(sid, KEY_NAMED, client, name, parent, None)


def _connection_stream(conn: sqlite3.Connection, call: CallInfo, now: datetime, ts: str,
                       tz: tzinfo | None) -> Stream:
    sid = connection_session_id(call.connection_id)
    row = conn.execute("SELECT client, name FROM record_sessions WHERE session_id = ?", (sid,)).fetchone()
    if row is not None:
        conn.execute("UPDATE record_sessions SET last_at = MAX(last_at, ?) WHERE session_id = ?", (ts, sid))
        return Stream(sid, KEY_CONNECTION, row["client"], row["name"], None, call.connection_id)
    label = _free_label(conn, now.astimezone(tz).strftime("%Y-%m-%d %H:%M"))
    conn.execute("INSERT INTO record_sessions (session_id, key_kind, client, name, connection_id, first_at, "
                 "last_at) VALUES (?, 'connection', ?, ?, ?, ?, ?)",
                 (sid, call.client, label, call.connection_id, ts, ts))
    return Stream(sid, KEY_CONNECTION, call.client, label, None, call.connection_id)


def _free_label(conn: sqlite3.Connection, base: str) -> str:
    """``base``, or ``base-2``, ``base-3``... when connection sessions already hold it."""
    taken = {r[0] for r in conn.execute(
        "SELECT name FROM record_sessions WHERE key_kind = 'connection' AND (name = ? OR name LIKE ?)",
        (base, base + "-%"))}
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


# --- the two-connection warning ----------------------------------------------------------------

def _overlap(conn: sqlite3.Connection, session_id: str, *, current: str | None = None) -> bool:
    """True when two connections that logged into ``session_id`` (plus ``current``) were
    live at once: an earlier one not closed, or closed after a later one started. A clean
    restart closes the first before the second starts. A crashed server never sets
    ``closed_at``, so a restart after a crash also counts as two."""
    rows = conn.execute(
        "SELECT connection_id, started_at, closed_at FROM record_connections "
        "WHERE connection_id IN (SELECT connection_id FROM traversal_events WHERE session_id = ?) "
        "OR connection_id = ? ORDER BY started_at, connection_id", (session_id, current or "")).fetchall()
    latest_close: str | None = ""  # None = still live (later than any timestamp)
    for i, r in enumerate(rows):
        if i and (latest_close is None or latest_close > r["started_at"]):
            return True
        if r["closed_at"] is None:
            latest_close = None
        elif latest_close is not None:
            latest_close = max(latest_close, r["closed_at"])
    return False


def shared_by_connections(lib: Library, session_id: str) -> bool:
    """Whether the stream's own events came from two connections live at once (shown on a
    record as :data:`WARN_TWO_CONNECTIONS`)."""
    return _overlap(lib.conn, session_id)


# --- family ---------------------------------------------------------------------------------

def family(lib: Library, session_id: str) -> list[str]:
    """The stream's family: the parent first, then its helpers by id. Given a helper, the
    same list as for its parent. ``[]`` for an unknown session."""
    row = lib.conn.execute("SELECT parent_session_id FROM record_sessions WHERE session_id = ?",
                           (session_id,)).fetchone()
    if row is None:
        return []
    root = row[0] or session_id
    helpers = [r[0] for r in lib.conn.execute(
        "SELECT session_id FROM record_sessions WHERE parent_session_id = ? ORDER BY session_id", (root,))]
    return [root, *helpers]
