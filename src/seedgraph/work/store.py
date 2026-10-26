"""Rows of the work protocol tables (design §1.7): batches, batch items, work packages,
package claims, submissions and the batch event feed.

Every function here takes an open ``sqlite3.Connection``. Writers expect to run inside
the caller's transaction (``Library.write()``); they never commit, roll back or BEGIN.
Readers work on any connection.

Conventions:
- Timestamps are UTC ISO-8601 with microseconds and ``+00:00`` (:func:`fmt_ts`), so they
  compare correctly as strings in SQL.
- Ordering is always explicit: items by ``ordinal``, packages by ``(ordinal, part)``,
  claims by ``claim_no``, events by ``seq``. ``claim_no``, ``submission_no`` and ``seq``
  are assigned ``MAX + 1`` inside the writing transaction.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..library.store import LibraryError

DEFAULT_CLAIM_TTL_S = 1800

ITEM_STATES: tuple[str, ...] = (
    "queued", "held_back", "preserved", "failed", "in_progress", "done", "stale", "conflict")
PACKAGE_STATES: tuple[str, ...] = ("open", "claimed", "saved", "void")
RELEASE_REASONS: tuple[str, ...] = ("submitted", "expired", "released", "stopped", "limit")
WORKER_KINDS: tuple[str, ...] = ("interactive", "background", "ollama")
OUTCOMES: tuple[str, ...] = ("accepted", "needs_revision", "invalid")


class WorkError(LibraryError):
    """A work protocol operation was refused."""


# --- time -------------------------------------------------------------------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def fmt_ts(dt: datetime) -> str:
    """The one stored timestamp format: UTC, microseconds, ``+00:00``."""
    if dt.tzinfo is None:
        raise WorkError("timestamps must be timezone-aware")
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _ts(now: datetime | None) -> str:
    return fmt_ts(now or now_utc())


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# --- row types ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Batch:
    batch_id: str
    task_type: str
    task_version: str
    selection_query: str | None
    selection_json: str
    graph_slug: str | None
    options_json: str
    instructions_sha256: str | None
    labels_sha256: str | None
    pause_kind: str | None
    stop_mode: str | None
    resets_at: str | None
    pause_detail: str | None
    paused_at: str | None
    created_at: str
    created_by_client: str | None
    origin: str

    @property
    def options(self) -> dict[str, Any]:
        return json.loads(self.options_json)

    @property
    def selection(self) -> list[str]:
        return json.loads(self.selection_json)

    @property
    def claim_ttl_s(self) -> int:
        return int(self.options.get("claim_ttl_s") or DEFAULT_CLAIM_TTL_S)


@dataclass(frozen=True)
class BatchItem:
    batch_id: str
    ordinal: int
    subject_kind: str
    subject_id: str
    state: str
    markdown_id: str | None
    markdown_hash: str | None
    expected_current_note_id: str | None
    part_count: int | None
    result_ref: str | None
    reason: str | None
    attention_item_id: str | None
    updated_at: str | None
    origin: str


@dataclass(frozen=True)
class WorkPackage:
    packet_id: str
    batch_id: str
    ordinal: int
    part: int
    part_count: int
    start_char: int | None
    end_char: int | None
    est_tokens: int | None
    text_path: str | None
    state: str
    attempt: int
    created_at: str
    origin: str


@dataclass(frozen=True)
class PackageClaim:
    claim_id: str
    packet_id: str
    claim_no: int
    worker_id: str
    worker_kind: str
    client: str | None
    model: str | None
    claimed_at: str
    expires_at: str
    released_at: str | None
    release_reason: str | None
    acknowledged: bool
    origin: str

    @property
    def live(self) -> bool:
        return self.released_at is None


@dataclass(frozen=True)
class Submission:
    submission_id: str
    packet_id: str
    claim_id: str
    submission_no: int
    received_at: str
    payload_json: str
    payload_sha256: str
    outcome: str
    feedback_json: str | None
    usage_json: str | None
    model: str | None
    origin: str


@dataclass(frozen=True)
class BatchEvent:
    batch_id: str
    seq: int
    at: str
    kind: str
    packet_id: str | None
    worker_id: str | None
    detail_json: str | None
    origin: str

    @property
    def detail(self) -> Any:
        return json.loads(self.detail_json) if self.detail_json is not None else None


def _row(cls, row: sqlite3.Row | None):
    if row is None:
        return None
    values = dict(row)
    if cls is PackageClaim:
        values["acknowledged"] = bool(values["acknowledged"])
    return cls(**values)


def _cols(cls) -> str:
    return ", ".join(cls.__dataclass_fields__)


# --- ids --------------------------------------------------------------------------------

def new_batch_id(now: datetime | None = None) -> str:
    """``'b_'`` + 14 hex digits of UTC microseconds + 6 random hex: time-ordered."""
    micros = int((now or now_utc()).timestamp() * 1_000_000)
    return f"b_{micros:014x}{secrets.token_hex(3)}"


def new_claim_id() -> str:
    return "pc_" + secrets.token_hex(12)


def packet_id_for(*, batch_id: str, subject_kind: str, subject_id: str, markdown_hash: str | None,
                  part: int, part_count: int, start_char: int | None, end_char: int | None,
                  task_version: str, attempt: int) -> str:
    """``'pk_'`` + sha256 of the package binding (design §1.7)."""
    binding = _dumps([batch_id, subject_kind, subject_id, markdown_hash, part, part_count,
                      start_char, end_char, task_version, attempt])
    return "pk_" + hashlib.sha256(binding.encode()).hexdigest()


# --- batches ----------------------------------------------------------------------------

def insert_batch(conn: sqlite3.Connection, *, task_type: str, task_version: str,
                 selection: Sequence[str], selection_query: str | None = None,
                 graph_slug: str | None = None, options: Mapping[str, Any] | None = None,
                 instructions_sha256: str | None = None, labels_sha256: str | None = None,
                 created_by_client: str | None = None, origin: str = "library",
                 now: datetime | None = None, batch_id: str | None = None) -> str:
    """Insert a batch row (no pause) and its ``batch_created`` event; return its id."""
    batch_id = batch_id or new_batch_id(now)
    created_at = _ts(now)
    conn.execute(
        "INSERT INTO batches (batch_id, task_type, task_version, selection_query, selection_json, "
        "graph_slug, options_json, instructions_sha256, labels_sha256, created_at, created_by_client, origin) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (batch_id, task_type, task_version, selection_query, _dumps(list(selection)), graph_slug,
         _dumps(dict(options or {})), instructions_sha256, labels_sha256, created_at,
         created_by_client, origin),
    )
    append_event(conn, batch_id, "batch_created", detail={"task_type": task_type}, now=now)
    return batch_id


def set_batch_files(conn: sqlite3.Connection, batch_id: str, *, instructions_sha256: str | None,
                    labels_sha256: str | None) -> None:
    conn.execute("UPDATE batches SET instructions_sha256 = ?, labels_sha256 = ? WHERE batch_id = ?",
                 (instructions_sha256, labels_sha256, batch_id))


def get_batch(conn: sqlite3.Connection, batch_id: str) -> Batch | None:
    return _row(Batch, conn.execute(f"SELECT {_cols(Batch)} FROM batches WHERE batch_id = ?",
                                    (batch_id,)).fetchone())


def require_batch(conn: sqlite3.Connection, batch_id: str) -> Batch:
    batch = get_batch(conn, batch_id)
    if batch is None:
        raise WorkError(f"no batch {batch_id!r}")
    return batch


def list_batches(conn: sqlite3.Connection) -> list[Batch]:
    """Every batch, oldest first (``created_at``, then ``batch_id``)."""
    rows = conn.execute(f"SELECT {_cols(Batch)} FROM batches ORDER BY created_at, batch_id")
    return [_row(Batch, r) for r in rows]


# --- items ------------------------------------------------------------------------------

def insert_item(conn: sqlite3.Connection, batch_id: str, ordinal: int, subject_id: str, *,
                subject_kind: str = "paper", state: str = "queued", markdown_id: str | None = None,
                markdown_hash: str | None = None, expected_current_note_id: str | None = None,
                part_count: int | None = None, reason: str | None = None,
                attention_item_id: str | None = None, origin: str = "library",
                now: datetime | None = None) -> None:
    if state not in ITEM_STATES:
        raise WorkError(f"unknown item state {state!r}")
    conn.execute(
        "INSERT INTO batch_items (batch_id, ordinal, subject_kind, subject_id, state, markdown_id, "
        "markdown_hash, expected_current_note_id, part_count, reason, attention_item_id, updated_at, origin) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (batch_id, ordinal, subject_kind, subject_id, state, markdown_id, markdown_hash,
         expected_current_note_id, part_count, reason, attention_item_id, _ts(now), origin),
    )


_ITEM_UPDATABLE = frozenset({"state", "part_count", "result_ref", "reason", "attention_item_id",
                             "markdown_id", "markdown_hash", "expected_current_note_id"})


def update_item(conn: sqlite3.Connection, batch_id: str, ordinal: int, *, now: datetime | None = None,
                **fields: Any) -> None:
    """Set the given columns of one item (and ``updated_at``)."""
    unknown = set(fields) - _ITEM_UPDATABLE
    if unknown:
        raise WorkError(f"batch_items has no updatable column(s) {sorted(unknown)}")
    if "state" in fields and fields["state"] not in ITEM_STATES:
        raise WorkError(f"unknown item state {fields['state']!r}")
    names = sorted(fields)
    sets = ", ".join(f"{n} = ?" for n in names)
    conn.execute(f"UPDATE batch_items SET {sets}, updated_at = ? WHERE batch_id = ? AND ordinal = ?",
                 (*[fields[n] for n in names], _ts(now), batch_id, ordinal))


def get_item(conn: sqlite3.Connection, batch_id: str, ordinal: int) -> BatchItem | None:
    return _row(BatchItem, conn.execute(
        f"SELECT {_cols(BatchItem)} FROM batch_items WHERE batch_id = ? AND ordinal = ?",
        (batch_id, ordinal)).fetchone())


def find_item(conn: sqlite3.Connection, batch_id: str, subject_id: str,
              subject_kind: str = "paper") -> BatchItem | None:
    return _row(BatchItem, conn.execute(
        f"SELECT {_cols(BatchItem)} FROM batch_items WHERE batch_id = ? AND subject_kind = ? "
        "AND subject_id = ?", (batch_id, subject_kind, subject_id)).fetchone())


def list_items(conn: sqlite3.Connection, batch_id: str) -> list[BatchItem]:
    rows = conn.execute(f"SELECT {_cols(BatchItem)} FROM batch_items WHERE batch_id = ? ORDER BY ordinal",
                        (batch_id,))
    return [_row(BatchItem, r) for r in rows]


# --- packages ---------------------------------------------------------------------------

def insert_package(conn: sqlite3.Connection, *, batch_id: str, ordinal: int, part: int, part_count: int,
                   task_version: str, markdown_hash: str | None = None, start_char: int | None = None,
                   end_char: int | None = None, est_tokens: int | None = None,
                   text_path: str | None = None, attempt: int = 1, origin: str = "library",
                   now: datetime | None = None) -> str:
    """Insert an ``open`` package for item ``(batch_id, ordinal)``; return its ``packet_id``."""
    item = get_item(conn, batch_id, ordinal)
    if item is None:
        raise WorkError(f"batch {batch_id!r} has no item {ordinal}")
    packet_id = packet_id_for(batch_id=batch_id, subject_kind=item.subject_kind, subject_id=item.subject_id,
                              markdown_hash=markdown_hash, part=part, part_count=part_count,
                              start_char=start_char, end_char=end_char, task_version=task_version,
                              attempt=attempt)
    conn.execute(
        "INSERT INTO work_packages (packet_id, batch_id, ordinal, part, part_count, start_char, end_char, "
        "est_tokens, text_path, state, attempt, created_at, origin) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)",
        (packet_id, batch_id, ordinal, part, part_count, start_char, end_char, est_tokens, text_path,
         attempt, _ts(now), origin),
    )
    return packet_id


def set_package_text_path(conn: sqlite3.Connection, packet_id: str, text_path: str) -> None:
    conn.execute("UPDATE work_packages SET text_path = ? WHERE packet_id = ?", (text_path, packet_id))


def set_package_state(conn: sqlite3.Connection, packet_id: str, state: str) -> None:
    if state not in PACKAGE_STATES:
        raise WorkError(f"unknown package state {state!r}")
    conn.execute("UPDATE work_packages SET state = ? WHERE packet_id = ?", (state, packet_id))


def get_package(conn: sqlite3.Connection, packet_id: str) -> WorkPackage | None:
    return _row(WorkPackage, conn.execute(
        f"SELECT {_cols(WorkPackage)} FROM work_packages WHERE packet_id = ?", (packet_id,)).fetchone())


def list_packages(conn: sqlite3.Connection, batch_id: str, *, ordinal: int | None = None) -> list[WorkPackage]:
    sql = f"SELECT {_cols(WorkPackage)} FROM work_packages WHERE batch_id = ?"
    args: list[Any] = [batch_id]
    if ordinal is not None:
        sql += " AND ordinal = ?"
        args.append(ordinal)
    rows = conn.execute(sql + " ORDER BY ordinal, part, attempt, packet_id", args)
    return [_row(WorkPackage, r) for r in rows]


# --- claims -----------------------------------------------------------------------------

def get_claim(conn: sqlite3.Connection, claim_id: str) -> PackageClaim | None:
    return _row(PackageClaim, conn.execute(
        f"SELECT {_cols(PackageClaim)} FROM package_claims WHERE claim_id = ?", (claim_id,)).fetchone())


def list_claims(conn: sqlite3.Connection, *, packet_id: str | None = None,
                batch_id: str | None = None) -> list[PackageClaim]:
    """Claims of one package (by ``claim_no``) or of a whole batch (by package, then ``claim_no``)."""
    cols = ", ".join(f"c.{c}" for c in PackageClaim.__dataclass_fields__)
    if packet_id is not None:
        rows = conn.execute(f"SELECT {cols} FROM package_claims c WHERE c.packet_id = ? ORDER BY c.claim_no",
                            (packet_id,))
    elif batch_id is not None:
        rows = conn.execute(
            f"SELECT {cols} FROM package_claims c JOIN work_packages p ON p.packet_id = c.packet_id "
            "WHERE p.batch_id = ? ORDER BY p.ordinal, p.part, p.attempt, p.packet_id, c.claim_no", (batch_id,))
    else:
        raise WorkError("list_claims needs packet_id or batch_id")
    return [_row(PackageClaim, r) for r in rows]


def latest_claim(conn: sqlite3.Connection, packet_id: str) -> PackageClaim | None:
    return _row(PackageClaim, conn.execute(
        f"SELECT {_cols(PackageClaim)} FROM package_claims WHERE packet_id = ? ORDER BY claim_no DESC LIMIT 1",
        (packet_id,)).fetchone())


# --- submissions ------------------------------------------------------------------------

def record_submission(conn: sqlite3.Connection, *, packet_id: str, claim_id: str, payload_json: str,
                      outcome: str, feedback: Any = None, usage: Any = None, model: str | None = None,
                      origin: str = "library", now: datetime | None = None) -> str:
    """Store one submission (the note once, as ``payload_json``); return its id."""
    if outcome not in OUTCOMES:
        raise WorkError(f"unknown submission outcome {outcome!r}")
    (n,) = conn.execute("SELECT COALESCE(MAX(submission_no), 0) + 1 FROM package_submissions "
                        "WHERE packet_id = ?", (packet_id,)).fetchone()
    submission_id = f"ps_{packet_id[3:27]}_{n:04d}"
    conn.execute(
        "INSERT INTO package_submissions (submission_id, packet_id, claim_id, submission_no, received_at, "
        "payload_json, payload_sha256, outcome, feedback_json, usage_json, model, origin) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (submission_id, packet_id, claim_id, n, _ts(now), payload_json,
         hashlib.sha256(payload_json.encode()).hexdigest(), outcome,
         None if feedback is None else _dumps(feedback), None if usage is None else _dumps(usage),
         model, origin),
    )
    return submission_id


def list_submissions(conn: sqlite3.Connection, packet_id: str) -> list[Submission]:
    rows = conn.execute(f"SELECT {_cols(Submission)} FROM package_submissions WHERE packet_id = ? "
                        "ORDER BY submission_no", (packet_id,))
    return [_row(Submission, r) for r in rows]


def latest_accepted_submission(conn: sqlite3.Connection, packet_id: str) -> Submission | None:
    return _row(Submission, conn.execute(
        f"SELECT {_cols(Submission)} FROM package_submissions WHERE packet_id = ? AND outcome = 'accepted' "
        "ORDER BY submission_no DESC LIMIT 1", (packet_id,)).fetchone())


# --- events -----------------------------------------------------------------------------

def append_event(conn: sqlite3.Connection, batch_id: str, kind: str, *, packet_id: str | None = None,
                 worker_id: str | None = None, detail: Any = None, now: datetime | None = None) -> int:
    """Append one event to the batch feed; return its ``seq`` (1..n per batch)."""
    (seq,) = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM batch_events WHERE batch_id = ?",
                          (batch_id,)).fetchone()
    conn.execute(
        "INSERT INTO batch_events (batch_id, seq, at, kind, packet_id, worker_id, detail_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (batch_id, seq, _ts(now), kind, packet_id, worker_id, None if detail is None else _dumps(detail)),
    )
    return seq


def list_events(conn: sqlite3.Connection, batch_id: str, *, after_seq: int = 0,
                limit: int | None = None) -> list[BatchEvent]:
    sql = f"SELECT {_cols(BatchEvent)} FROM batch_events WHERE batch_id = ? AND seq > ? ORDER BY seq"
    args: list[Any] = [batch_id, after_seq]
    if limit is not None:
        sql += " LIMIT ?"
        args.append(limit)
    return [_row(BatchEvent, r) for r in conn.execute(sql, args)]


# --- claim lifecycle at row level (used by claims and state) ---------------------------

def _live_claims(conn: sqlite3.Connection, where: str, args: Sequence[Any]) -> list[tuple[str, PackageClaim]]:
    cols = ", ".join(f"c.{c}" for c in PackageClaim.__dataclass_fields__)
    sql = (f"SELECT p.batch_id AS _batch_id, {cols} FROM package_claims c "
           "JOIN work_packages p ON p.packet_id = c.packet_id "
           f"WHERE c.released_at IS NULL {where} "
           "ORDER BY p.batch_id, p.ordinal, p.part, p.attempt, p.packet_id, c.claim_no")
    out = []
    for r in conn.execute(sql, args):
        values = dict(r)
        batch_id = values.pop("_batch_id")
        values["acknowledged"] = bool(values["acknowledged"])
        out.append((batch_id, PackageClaim(**values)))
    return out


def _lapsed_where(batch_id: str | None) -> tuple[str, list[Any]]:
    where = "AND c.expires_at <= ?"
    return (where + " AND p.batch_id = ?", [batch_id]) if batch_id is not None else (where, [])


def has_lapsed(conn: sqlite3.Connection, *, now: datetime | None = None, batch_id: str | None = None) -> bool:
    """True if a live claim's ``expires_at`` has passed (read-only)."""
    where, extra = _lapsed_where(batch_id)
    sql = ("SELECT 1 FROM package_claims c JOIN work_packages p ON p.packet_id = c.packet_id "
           f"WHERE c.released_at IS NULL {where} LIMIT 1")
    return conn.execute(sql, [_ts(now), *extra]).fetchone() is not None


def expire_lapsed(conn: sqlite3.Connection, *, now: datetime | None = None,
                  batch_id: str | None = None) -> list[PackageClaim]:
    """Lazy expiry: end every lapsed live claim with ``release_reason = 'expired'``
    (``released_at`` = its ``expires_at``) and return its package to the pool."""
    where, extra = _lapsed_where(batch_id)
    lapsed = _live_claims(conn, where, [_ts(now), *extra])
    for owner_batch, claim in lapsed:
        conn.execute("UPDATE package_claims SET released_at = expires_at, release_reason = 'expired' "
                     "WHERE claim_id = ?", (claim.claim_id,))
        conn.execute("UPDATE work_packages SET state = 'open' WHERE packet_id = ? AND state = 'claimed'",
                     (claim.packet_id,))
        append_event(conn, owner_batch, "claim_expired", packet_id=claim.packet_id, worker_id=claim.worker_id,
                     detail={"claim_id": claim.claim_id, "expires_at": claim.expires_at}, now=now)
    for owner_batch in sorted({b for b, _ in lapsed}):
        requeue_idle_items(conn, owner_batch, now=now)
    return [c for _, c in lapsed]


def requeue_idle_items(conn: sqlite3.Connection, batch_id: str, *, now: datetime | None = None) -> int:
    """Put every ``in_progress`` item of the batch with no live claim and no saved
    package back to ``queued`` (its packages are back in the pool). An item with a saved
    part stays ``in_progress``: it has partial work. Sweeps the whole batch, so a stale
    row left by an older version heals on the next write. Returns the count."""
    return conn.execute(
        "UPDATE batch_items SET state = 'queued', updated_at = ? "
        "WHERE batch_id = ? AND state = 'in_progress' "
        "AND NOT EXISTS (SELECT 1 FROM work_packages p WHERE p.batch_id = batch_items.batch_id "
        "                AND p.ordinal = batch_items.ordinal AND p.state IN ('claimed', 'saved')) "
        "AND NOT EXISTS (SELECT 1 FROM package_claims c JOIN work_packages p ON p.packet_id = c.packet_id "
        "                WHERE p.batch_id = batch_items.batch_id AND p.ordinal = batch_items.ordinal "
        "                AND c.released_at IS NULL)",
        (_ts(now), batch_id)).rowcount


def mark_item_in_progress(conn: sqlite3.Connection, batch_id: str, ordinal: int, *,
                          now: datetime | None = None) -> None:
    """A claim (or a revived one) is working on this item: ``queued`` -> ``in_progress``."""
    conn.execute("UPDATE batch_items SET state = 'in_progress', updated_at = ? "
                 "WHERE batch_id = ? AND ordinal = ? AND state = 'queued'", (_ts(now), batch_id, ordinal))


def release_live_claims(conn: sqlite3.Connection, batch_id: str, reason: str, *, now: datetime | None = None,
                        worker_id: str | None = None) -> list[PackageClaim]:
    """End the batch's live claims (only ``worker_id``'s, if given) with ``reason`` and
    return their packages to the pool."""
    if reason not in ("released", "stopped", "limit"):
        raise WorkError(f"release reason {reason!r} is not a bulk release")
    where, args = "AND p.batch_id = ?", [batch_id]
    if worker_id is not None:
        where += " AND c.worker_id = ?"
        args.append(worker_id)
    live = [c for _, c in _live_claims(conn, where, args)]
    for claim in live:
        conn.execute("UPDATE package_claims SET released_at = ?, release_reason = ? WHERE claim_id = ?",
                     (_ts(now), reason, claim.claim_id))
        conn.execute("UPDATE work_packages SET state = 'open' WHERE packet_id = ? AND state = 'claimed'",
                     (claim.packet_id,))
        append_event(conn, batch_id, "claim_released", packet_id=claim.packet_id, worker_id=claim.worker_id,
                     detail={"claim_id": claim.claim_id, "reason": reason}, now=now)
    if live:
        requeue_idle_items(conn, batch_id, now=now)
    return live
