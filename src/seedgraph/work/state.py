"""Batch state and the pause model (design §1.6).

A batch shows exactly one state. Two pause states are stored on ``batches``
(``pause_kind`` = ``'stopped'`` / ``'usage_limit'``); the rest are derived from package
claims at read time, so they can never be stale:

==================  =====================================================================
``finished``        no package is open or claimed (wins over everything: nothing to pause)
``stopped``         stored, set by :func:`stop`
``usage_limit``     stored, set by :func:`report_limit`
``working``         at least one live claim
``claim_expired``   an open package whose latest claim expired, unacknowledged
``waiting``         open packages, no live claims, none of the above
==================  =====================================================================

The design orders the four pause states stopped > usage limit > claim expired >
waiting. ``working`` sits between the stored and derived pauses (the derived ones are
defined as "no live claims"), and ``finished`` above all of them.

:func:`continue_batch` is the only resume: it clears a stored pause and acknowledges
expired-claim flags. Every public function here expires lapsed claims first (lazy
expiry, no background process); a read opens a write only when something lapsed, so a
plain status read never dirties the library.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from . import store
from .store import WorkError, fmt_ts

if TYPE_CHECKING:
    from ..library.store import Library
    from .claims import WorkerRef

STATES: tuple[str, ...] = ("finished", "stopped", "usage_limit", "working", "claim_expired", "waiting")
PAUSED_STATES: frozenset[str] = frozenset({"stopped", "usage_limit"})

StopMode = Literal["finish", "now"]

_NEXT_ACTION = {
    "stopped": "The batch was stopped on purpose. Stop working and report your counts; "
               "the researcher can resume it with continue.",
    "usage_limit": "The batch is paused for a usage limit. Stop working and report your counts; "
                   "resume with continue once the limit has reset.",
    "finished": "Nothing is left to do in this batch. Report your counts.",
    "working": "Claim the next package.",
    "claim_expired": "Claim the next package; a package whose worker stopped responding is back in the pool.",
    "waiting": "Claim the next package.",
}


@dataclass(frozen=True)
class BatchState:
    batch_id: str
    state: str
    label: str
    pause_kind: str | None
    stop_mode: str | None
    resets_at: str | None
    pause_detail: str | None
    live_claims: int
    workers: tuple[str, ...]
    expired_packets: tuple[str, ...]
    packages: dict[str, int] = field(default_factory=dict)
    items: dict[str, int] = field(default_factory=dict)

    @property
    def paused(self) -> bool:
        return self.state in PAUSED_STATES

    @property
    def next_action(self) -> str:
        return _NEXT_ACTION[self.state]


def pause_reply(state: BatchState) -> dict[str, Any]:
    """The reply to a claim refused because of a pause (design §1.6)."""
    return {"status": "paused", "state": state.state, "label": state.label,
            "resets_at": state.resets_at, "next_action": state.next_action}


def _counts(conn: sqlite3.Connection, sql: str, batch_id: str) -> dict[str, int]:
    return {r[0]: r[1] for r in conn.execute(sql, (batch_id,))}


def derive_state(conn: sqlite3.Connection, batch_id: str) -> BatchState:
    """The batch's one state from its stored pause and its claims (pure read; the
    caller has already expired lapsed claims)."""
    batch = store.require_batch(conn, batch_id)
    packages = _counts(conn, "SELECT state, COUNT(*) FROM work_packages WHERE batch_id = ? "
                             "GROUP BY state ORDER BY state", batch_id)
    items = _counts(conn, "SELECT state, COUNT(*) FROM batch_items WHERE batch_id = ? "
                          "GROUP BY state ORDER BY state", batch_id)
    workers = tuple(r[0] for r in conn.execute(
        "SELECT DISTINCT c.worker_id FROM package_claims c JOIN work_packages p ON p.packet_id = c.packet_id "
        "WHERE p.batch_id = ? AND c.released_at IS NULL ORDER BY c.worker_id", (batch_id,)))
    (live,) = conn.execute(
        "SELECT COUNT(*) FROM package_claims c JOIN work_packages p ON p.packet_id = c.packet_id "
        "WHERE p.batch_id = ? AND c.released_at IS NULL", (batch_id,)).fetchone()
    expired = tuple(r[0] for r in conn.execute(
        "SELECT p.packet_id FROM work_packages p JOIN package_claims c ON c.packet_id = p.packet_id "
        "WHERE p.batch_id = ? AND p.state = 'open' AND c.release_reason = 'expired' AND c.acknowledged = 0 "
        "AND c.claim_no = (SELECT MAX(o.claim_no) FROM package_claims o WHERE o.packet_id = p.packet_id) "
        "ORDER BY p.ordinal, p.part, p.attempt, p.packet_id", (batch_id,)))

    open_or_claimed = packages.get("open", 0) + packages.get("claimed", 0)
    if open_or_claimed == 0:
        state = "finished"
        label = (f"Finished: {items.get('done', 0)} done / {items.get('held_back', 0)} held back / "
                 f"{items.get('failed', 0)} failed")
    elif batch.pause_kind == "stopped":
        state = "stopped"
        label = "Stopped" + (f" ({live} claim{'s' if live != 1 else ''} finishing)"
                             if batch.stop_mode == "finish" and live else "")
    elif batch.pause_kind == "usage_limit":
        state = "usage_limit"
        label = f"Paused: usage limit, resets {batch.resets_at}" if batch.resets_at \
            else "Paused: usage limit, resets: unknown"
    elif live:
        state = "working"
        label = f"Working: {len(workers)} agent{'s' if len(workers) != 1 else ''}"
    elif expired:
        state = "claim_expired"
        label = "Paused: a worker stopped responding; its package is back in the pool"
    else:
        state = "waiting"
        label = "Waiting for an agent"
    return BatchState(batch_id=batch_id, state=state, label=label, pause_kind=batch.pause_kind,
                      stop_mode=batch.stop_mode, resets_at=batch.resets_at, pause_detail=batch.pause_detail,
                      live_claims=live, workers=workers, expired_packets=expired,
                      packages=packages, items=items)


def expire_if_needed(lib: Library, *, now: datetime | None = None, batch_id: str | None = None) -> int:
    """Expire lapsed claims, opening a write only when one has lapsed. Returns the count."""
    if not store.has_lapsed(lib.conn, now=now, batch_id=batch_id):
        return 0
    with lib.write() as conn:
        return len(store.expire_lapsed(conn, now=now, batch_id=batch_id))


def batch_state(lib: Library, batch_id: str, *, now: datetime | None = None) -> BatchState:
    """Read-only status (after lazy expiry)."""
    store.require_batch(lib.conn, batch_id)
    expire_if_needed(lib, now=now, batch_id=batch_id)
    return derive_state(lib.conn, batch_id)


def stop(lib: Library, batch_id: str, *, mode: StopMode, now: datetime | None = None) -> BatchState:
    """Stop on purpose. ``finish`` lets live claims finish (they may still renew and be
    saved); ``now`` releases them (``release_reason = 'stopped'``) so their submits are
    refused. Either way no new package is claimed until :func:`continue_batch`. A stop
    replaces a usage-limit pause (stopped wins)."""
    if mode not in ("finish", "now"):
        raise WorkError(f"stop mode must be 'finish' or 'now', not {mode!r}")
    with lib.write() as conn:
        store.require_batch(conn, batch_id)
        store.expire_lapsed(conn, now=now, batch_id=batch_id)
        conn.execute("UPDATE batches SET pause_kind = 'stopped', stop_mode = ?, resets_at = NULL, "
                     "pause_detail = NULL, paused_at = ? WHERE batch_id = ?",
                     (mode, fmt_ts(now or store.now_utc()), batch_id))
        released = store.release_live_claims(conn, batch_id, "stopped", now=now) if mode == "now" else []
        store.append_event(conn, batch_id, "stopped",
                           detail={"mode": mode, "released": [c.claim_id for c in released]}, now=now)
        return derive_state(conn, batch_id)


def _normalize_resets_at(resets_at: str | None) -> str | None:
    if resets_at is None or not str(resets_at).strip():
        return None
    try:
        dt = datetime.fromisoformat(str(resets_at).strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise WorkError(f"resets_at must be an ISO-8601 time, not {resets_at!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # a naive time is read as UTC
    return fmt_ts(dt)


def report_limit(lib: Library, batch_id: str, worker: WorkerRef, *, resets_at: str | None = None,
                 message: str | None = None, now: datetime | None = None) -> BatchState:
    """A worker hit a usage limit: pause the batch (unless it is already stopped, which
    takes precedence) and release that worker's live claims at once."""
    resets = _normalize_resets_at(resets_at)
    with lib.write() as conn:
        batch = store.require_batch(conn, batch_id)
        store.expire_lapsed(conn, now=now, batch_id=batch_id)
        if batch.pause_kind != "stopped":
            conn.execute("UPDATE batches SET pause_kind = 'usage_limit', stop_mode = NULL, resets_at = ?, "
                         "pause_detail = ?, paused_at = ? WHERE batch_id = ?",
                         (resets, message, fmt_ts(now or store.now_utc()), batch_id))
        released = store.release_live_claims(conn, batch_id, "limit", now=now, worker_id=worker.worker_id)
        store.append_event(conn, batch_id, "usage_limit", worker_id=worker.worker_id,
                           detail={"resets_at": resets, "message": message,
                                   "released": [c.claim_id for c in released]}, now=now)
        return derive_state(conn, batch_id)


def continue_batch(lib: Library, batch_id: str, *, now: datetime | None = None) -> BatchState:
    """The only resume: clear a stored pause and acknowledge expired-claim flags."""
    with lib.write() as conn:
        batch = store.require_batch(conn, batch_id)
        store.expire_lapsed(conn, now=now, batch_id=batch_id)
        conn.execute("UPDATE batches SET pause_kind = NULL, stop_mode = NULL, resets_at = NULL, "
                     "pause_detail = NULL, paused_at = NULL WHERE batch_id = ?", (batch_id,))
        acked = conn.execute(
            "UPDATE package_claims SET acknowledged = 1 WHERE release_reason = 'expired' AND acknowledged = 0 "
            "AND packet_id IN (SELECT packet_id FROM work_packages WHERE batch_id = ?)", (batch_id,)).rowcount
        store.append_event(conn, batch_id, "continued",
                           detail={"cleared": batch.pause_kind, "acknowledged": acked}, now=now)
        return derive_state(conn, batch_id)
