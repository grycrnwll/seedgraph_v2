"""Package claims (design §1.4): claim, renew (or revive), complete, release.

- :func:`claim_package` runs in ONE write transaction (``BEGIN IMMEDIATE``, or a
  SAVEPOINT inside the caller's): expire lapsed claims, refuse if the batch has a
  stored pause, pick the open package with the lowest ``(ordinal, part)`` (preferring
  further parts of a paper this worker holds or finished, or the given paper), insert
  the claim. SQLite's writer lock makes two simultaneous claimers get different
  packages; the partial unique index ``ux_package_claims_live`` backs it up.
- Refusals that are normal replies (paused, no package, finished) are returned as
  values, so the expiry done in the same transaction is kept. Claim-id failures raise
  :class:`ClaimRefused` only after the write has committed.
- A late submit revives an expired claim when (a) it expired (``release_reason =
  'expired'``), (b) it is the package's latest claim and (c) the package is open,
  i.e. nobody re-claimed it - and the batch is not stopped with ``now``. Claims ended by
  a stop-now, a usage limit, an explicit release or a completed save are never revived.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from . import store
from .state import BatchState, derive_state, pause_reply
from .store import PackageClaim, WorkError, WorkPackage, fmt_ts

if TYPE_CHECKING:
    from ..library.store import Library

ClaimStatus = Literal["claimed", "paused", "no_package", "finished"]

_REFUSAL_MESSAGES = {
    "claim_unknown": "this claim_id is not a claim on this package",
    "claim_reclaimed": "claim expired and the package was re-claimed by another worker",
    "claim_released": "this claim was released; claim a package again",
    "claim_stopped": "the batch was stopped now and this claim was released; its submits are refused",
    "claim_limit": "this claim was released when a usage limit was reported; claim a package again",
    "package_saved": "this package is already saved; a resubmit needs a new batch",
    "batch_stopped": "the batch was stopped now; an expired claim cannot be revived",
}


class ClaimRefused(WorkError):
    """A submit/renew/release with a ``claim_id`` that is not (or no longer) the package's
    live claim. ``code`` is one of the keys of ``_REFUSAL_MESSAGES``."""

    def __init__(self, code: str, packet_id: str, claim_id: str):
        self.code = code
        self.packet_id = packet_id
        self.claim_id = claim_id
        super().__init__(_REFUSAL_MESSAGES[code])


@dataclass(frozen=True)
class WorkerRef:
    """Who is working (for reporting). Correctness rests on ``claim_id``, never on this."""

    worker_kind: str  # 'interactive' | 'background' | 'ollama'
    worker_id: str
    client: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if self.worker_kind not in store.WORKER_KINDS:
            raise WorkError(f"unknown worker kind {self.worker_kind!r}")
        if not self.worker_id:
            raise WorkError("worker_id must not be empty")


@dataclass(frozen=True)
class ClaimResult:
    status: ClaimStatus
    state: BatchState
    package: WorkPackage | None = None
    claim: PackageClaim | None = None

    def pause(self) -> dict[str, Any] | None:
        """The paused reply (design §1.6) when ``status == 'paused'``."""
        return pause_reply(self.state) if self.status == "paused" else None


def _expiry(now: datetime | None, ttl_s: int) -> str:
    return fmt_ts((now or store.now_utc()) + timedelta(seconds=ttl_s))


_PICK_SQL = (
    "SELECT p.packet_id FROM work_packages p "
    "JOIN batch_items i ON i.batch_id = p.batch_id AND i.ordinal = p.ordinal "
    "WHERE p.batch_id = :batch AND p.state = 'open' AND i.state IN ('queued', 'in_progress') {paper} "
    "ORDER BY EXISTS (SELECT 1 FROM package_claims c JOIN work_packages q ON q.packet_id = c.packet_id "
    "                 WHERE q.batch_id = p.batch_id AND q.ordinal = p.ordinal AND c.worker_id = :worker "
    "                   AND (c.released_at IS NULL OR c.release_reason = 'submitted')) DESC, "
    "         p.ordinal, p.part, p.attempt, p.packet_id "
    "LIMIT 1"
)


def claim_package(lib: Library, batch_id: str, worker: WorkerRef, *, paper_id: str | None = None,
                  now: datetime | None = None) -> ClaimResult:
    """Claim the next package of ``batch_id`` for ``worker`` (design §1.4 steps 1-4)."""
    store.require_batch(lib.conn, batch_id)
    with lib.write() as conn:
        batch = store.require_batch(conn, batch_id)
        store.expire_lapsed(conn, now=now, batch_id=batch_id)
        if batch.pause_kind is not None:
            state = derive_state(conn, batch_id)  # a finished batch says finished, even if paused
            return ClaimResult(status="finished" if state.state == "finished" else "paused", state=state)
        sql = _PICK_SQL.format(paper="AND i.subject_kind = 'paper' AND i.subject_id = :paper" if paper_id else "")
        row = conn.execute(sql, {"batch": batch_id, "worker": worker.worker_id, "paper": paper_id}).fetchone()
        if row is None:
            state = derive_state(conn, batch_id)
            return ClaimResult(status="finished" if state.state == "finished" else "no_package", state=state)
        packet_id = row[0]
        (claim_no,) = conn.execute("SELECT COALESCE(MAX(claim_no), 0) + 1 FROM package_claims WHERE packet_id = ?",
                                   (packet_id,)).fetchone()
        claim_id = store.new_claim_id()
        conn.execute(
            "INSERT INTO package_claims (claim_id, packet_id, claim_no, worker_id, worker_kind, client, model, "
            "claimed_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (claim_id, packet_id, claim_no, worker.worker_id, worker.worker_kind, worker.client, worker.model,
             fmt_ts(now or store.now_utc()), _expiry(now, batch.claim_ttl_s)),
        )
        store.set_package_state(conn, packet_id, "claimed")
        package = store.get_package(conn, packet_id)
        assert package is not None
        store.mark_item_in_progress(conn, batch_id, package.ordinal, now=now)
        store.append_event(conn, batch_id, "claimed", packet_id=packet_id, worker_id=worker.worker_id,
                           detail={"claim_id": claim_id, "claim_no": claim_no, "worker_kind": worker.worker_kind,
                                   "client": worker.client}, now=now)
        return ClaimResult(status="claimed", state=derive_state(conn, batch_id), package=package,
                           claim=store.get_claim(conn, claim_id))


def _check(conn, packet_id: str, claim_id: str) -> tuple[PackageClaim | None, WorkPackage | None, str | None]:
    """(claim, package, refusal code or None if the claim is live). Revivable expired
    claims come back with code ``'revive'``."""
    claim = store.get_claim(conn, claim_id)
    package = store.get_package(conn, packet_id)
    if claim is None or package is None or claim.packet_id != packet_id:
        return claim, package, "claim_unknown"
    if claim.live:
        return claim, package, None
    reason = claim.release_reason
    if reason == "submitted":
        return claim, package, "package_saved"
    if reason == "expired":
        latest = store.latest_claim(conn, packet_id)
        if latest is None or latest.claim_id != claim_id or package.state != "open":
            return claim, package, "package_saved" if package.state == "saved" else "claim_reclaimed"
        batch = store.require_batch(conn, package.batch_id)
        if batch.pause_kind == "stopped" and batch.stop_mode == "now":
            return claim, package, "batch_stopped"
        return claim, package, "revive"
    if package.state == "saved":
        return claim, package, "package_saved"
    return claim, package, {"stopped": "claim_stopped", "limit": "claim_limit"}.get(reason or "", "claim_released")


def renew_claim(lib: Library, packet_id: str, claim_id: str, *, now: datetime | None = None) -> PackageClaim:
    """Check that ``claim_id`` is the package's live claim and push its expiry to
    ``now + claim_ttl_s``; an expired claim nobody re-claimed is revived. Every submit,
    including a ``needs_revision`` one, calls this first. Raises :class:`ClaimRefused`."""
    refusal: str | None = None
    with lib.write() as conn:
        package = store.get_package(conn, packet_id)
        if package is not None:
            store.expire_lapsed(conn, now=now, batch_id=package.batch_id)
        claim, package, code = _check(conn, packet_id, claim_id)
        if code in (None, "revive"):
            assert package is not None and claim is not None
            batch = store.require_batch(conn, package.batch_id)
            conn.execute("UPDATE package_claims SET expires_at = ?, released_at = NULL, release_reason = NULL "
                         "WHERE claim_id = ?", (_expiry(now, batch.claim_ttl_s), claim_id))
            if code == "revive":
                store.set_package_state(conn, packet_id, "claimed")
                store.mark_item_in_progress(conn, package.batch_id, package.ordinal, now=now)
                store.append_event(conn, package.batch_id, "claim_revived", packet_id=packet_id,
                                   worker_id=claim.worker_id, detail={"claim_id": claim_id}, now=now)
            renewed = store.get_claim(conn, claim_id)
        else:
            refusal = code
    if refusal is not None:
        raise ClaimRefused(refusal, packet_id, claim_id)
    assert renewed is not None
    return renewed


def _end_live(lib: Library, packet_id: str, claim_id: str, *, reason: str, package_state: str, event: str,
              detail: dict[str, Any], now: datetime | None) -> None:
    refusal: str | None = None
    with lib.write() as conn:
        package = store.get_package(conn, packet_id)
        if package is not None:
            store.expire_lapsed(conn, now=now, batch_id=package.batch_id)
        claim, package, code = _check(conn, packet_id, claim_id)
        if code in (None, "revive"):  # a revivable expired claim may still finish or hand back
            assert claim is not None and package is not None
            conn.execute("UPDATE package_claims SET released_at = ?, release_reason = ? WHERE claim_id = ?",
                         (fmt_ts(now or store.now_utc()), reason, claim_id))
            store.set_package_state(conn, packet_id, package_state)
            if package_state == "open":
                store.requeue_idle_items(conn, package.batch_id, now=now)
            store.append_event(conn, package.batch_id, event, packet_id=packet_id, worker_id=claim.worker_id,
                               detail={"claim_id": claim_id, **detail}, now=now)
        else:
            refusal = code
    if refusal is not None:
        raise ClaimRefused(refusal, packet_id, claim_id)


def complete_claim(lib: Library, packet_id: str, claim_id: str, *, now: datetime | None = None) -> None:
    """The package's submission was saved: end the live claim (``'submitted'``) and mark
    the package ``saved``. Call inside the save transaction (it nests as a SAVEPOINT)."""
    _end_live(lib, packet_id, claim_id, reason="submitted", package_state="saved", event="package_saved",
              detail={}, now=now)


def release_claim(lib: Library, packet_id: str, claim_id: str, *, package_state: str = "open",
                  note: str | None = None, now: datetime | None = None) -> None:
    """Hand a package back early (``release_reason = 'released'``). ``package_state='open'``
    returns it to the pool (give_back); ``'void'`` retires it (e.g. paper_problem, with
    the item state set by the caller). ``note`` is recorded in the event."""
    if package_state not in ("open", "void"):
        raise WorkError("a released package goes back 'open' or becomes 'void'")
    _end_live(lib, packet_id, claim_id, reason="released", package_state=package_state, event="released",
              detail={"package_state": package_state, "note": note}, now=now)
