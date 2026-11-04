"""Spot-check marks (design §5.3, §5.5; slice R4).

A mark is **right**, **partly** (partly right) or **wrong**, with optional reasons and a
note. It is stored on the sample item, keyed by the immutable ``extracted_claims`` id, so
it always judges the agent's original claim. A changed mark replaces the old one in place:
the last mark counts. The sample is finished when its last item is marked.

**Fixed** is derived, never stored: the paper's overlay changed the claim's ops after the
mark. ``ops_json`` holds the *complete* op set at each version, so a version after the
mark that merely carries an earlier edit forward (a notes-only save, say) is not a fix:
the claim's ops at the last version at or before the mark are compared with each later
version's, ignoring the target (a replay re-keys it to the new claim id). An op is the
claim's when its target names the claim id, carries the stored fingerprint, or is the
same claim by the stage 3 identity tiers (:func:`seedgraph.notes.identity.tier`).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from ..library.store import Library
from ..notes.identity import ClaimFingerprint, tier
from ..work.store import fmt_ts, now_utc
from .sample import Item, SpotCheckError
from .sample import sample as _sample

MARKS: tuple[str, ...] = ("right", "partly", "wrong")
MARK_LABELS: dict[str, str] = {"right": "right", "partly": "partly right", "wrong": "wrong"}
#: Marks after which "fix it now" is offered.
FIX_MARKS: tuple[str, ...] = ("partly", "wrong")
REASONS: tuple[str, ...] = ("wrong_type", "misreads", "quote_unsupported", "not_in_paper")
REASON_LABELS: dict[str, str] = {
    "wrong_type": "wrong type",
    "misreads": "text misreads the paper",
    "quote_unsupported": "quote does not support it",
    "not_in_paper": "not in the paper",
}


def _check_reasons(value: str, reasons: Iterable[str]) -> list[str]:
    given = list(reasons or ())
    bad = sorted({r for r in given if r not in REASONS})
    if bad:
        raise SpotCheckError(f"unknown reason {', '.join(map(repr, bad))}; use {', '.join(REASONS)}")
    if given and value == "right":
        raise SpotCheckError("reasons are for a claim marked partly right or wrong")
    return [r for r in REASONS if r in given]


def mark(lib: Library, sample_id: str, claim_id: str, value: str, *, reasons: Iterable[str] = (),
         note: str | None = None, now: str | None = None) -> Item:
    """Mark one item of a sample (replacing any earlier mark); returns the item."""
    if value not in MARKS:
        raise SpotCheckError(f"{value!r} is not a mark; use right, partly (partly right) or wrong")
    clean = _check_reasons(value, reasons)
    text = " ".join((note or "").split()) or None
    now = now or fmt_ts(now_utc())
    with lib.write() as conn:
        cur = conn.execute(
            "UPDATE spot_check_items SET mark = ?, reasons_json = ?, note = ?, marked_at = ? "
            "WHERE sample_id = ? AND claim_id = ?", (value, json.dumps(clean), text, now, sample_id, claim_id))
        if cur.rowcount == 0:
            raise SpotCheckError("this claim is not in that spot-check sample")
        if conn.execute("SELECT 1 FROM spot_check_items WHERE sample_id = ? AND mark IS NULL LIMIT 1",
                        (sample_id,)).fetchone() is None:
            conn.execute("UPDATE spot_check_samples SET finished_at = ? WHERE sample_id = ? AND finished_at IS NULL",
                         (now, sample_id))
    s = _sample(lib, sample_id)
    assert s is not None
    item = s.item(claim_id)
    assert item is not None
    return item


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _same_claim(target: Mapping[str, Any], claim_id: str, fp: Mapping[str, Any], fp_obj: ClaimFingerprint) -> bool:
    if target.get("claim_id") == claim_id:
        return True
    other = target.get("fingerprint")
    if not isinstance(other, Mapping):
        return False
    if dict(other) == dict(fp):
        return True
    return tier(fp_obj, ClaimFingerprint.from_dict("op", other)) is not None


def _claim_ops(ops: list[dict[str, Any]], claim_id: str, fp: Mapping[str, Any],
               fp_obj: ClaimFingerprint) -> list[str]:
    return sorted(json.dumps([o["op"], o.get("set")], sort_keys=True) for o in ops
                  if o.get("op") in ("edit", "hide") and _same_claim(o.get("target") or {}, claim_id, fp, fp_obj))


def is_fixed(conn: sqlite3.Connection, paper_id: str, claim_id: str, fp: Mapping[str, Any],
             marked_at: str | None) -> bool:
    """Whether an overlay version after ``marked_at`` changed this claim's edit or hide."""
    if marked_at is None:
        return False
    after = _ts(marked_at)
    fp_obj = ClaimFingerprint.from_dict(claim_id, fp)
    rows = conn.execute(
        "SELECT v.overlay_id, v.ops_json, v.created_at FROM overlay_versions v "
        "JOIN note_overlays o ON o.overlay_id = v.overlay_id WHERE o.paper_id = ? "
        "ORDER BY v.overlay_id, v.version_no", (paper_id,)).fetchall()
    before: dict[str, list[str]] = {}
    for overlay_id, ops_json, created_at in rows:
        ops = _claim_ops(json.loads(ops_json), claim_id, fp, fp_obj)
        if _ts(created_at) <= after:
            before[overlay_id] = ops
        elif ops != before.get(overlay_id, []):
            return True
    return False


def fixed(lib: Library | sqlite3.Connection, sample_id: str, claim_id: str) -> bool:
    """Whether the researcher changed this item's claim (edit or hide) after marking it."""
    conn = lib if isinstance(lib, sqlite3.Connection) else lib.conn
    r = conn.execute("SELECT paper_id, claim_fingerprint, marked_at FROM spot_check_items "
                     "WHERE sample_id = ? AND claim_id = ?", (sample_id, claim_id)).fetchone()
    if r is None:
        raise SpotCheckError("this claim is not in that spot-check sample")
    return is_fixed(conn, r[0], claim_id, json.loads(r[1]), r[2])
