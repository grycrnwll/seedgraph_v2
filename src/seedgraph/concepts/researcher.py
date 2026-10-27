"""Researcher concept decisions (stage 4 design §2.4, §7, §11): the writers behind the
concept pages. Researcher decisions always override agent ones.

Writers (each one library transaction that ends with a full deterministic refresh, so the
derived tables a page reads next are current):

- :func:`merge`, :func:`link` (the researcher's own note and a quote from each side),
  :func:`never_connect`, :func:`unrelated`: researcher rows (``decided_by = 'researcher'``).
  A direct ``never_merge`` / ``never_connect`` on the same pair is overridden (the newest
  researcher decision wins); a veto reached only **through other members** of a group
  is refused with :class:`ResearcherError`, as is a merge across families. Agent rows on
  the pair are set ``undone`` (``veto_reason = 'overridden by <id> from <state>'``), so
  queued ones leave the queue; :func:`undo` restores them.
- :func:`split_decision`: split an applied merge or remove an applied link (any maker):
  the row becomes ``undone`` and a researcher ``never_connect`` is recorded with
  ``origin = 'split:<id>'``. :func:`undo` of that record reverses both.
- :func:`split_label` / :func:`undo_label_split`: one label into senses ``s1..sn`` with
  glosses and claim assignments (keyed by claim fingerprint); undo collapses the label
  back to ``s0`` and keeps the assignments for history. Agent decisions naming the
  replaced senses become ``needs_rejudge``.
- :func:`add_never_merge` / :func:`remove_never_merge` and :func:`edit_ambiguous` (the
  settings lists; ambiguous patterns may be library-wide or per graph).
- :func:`settle`: the short queue (:func:`queue`) - accept, reject, merge, link instead.
- :func:`create_review_sample` / :func:`mark`: the frozen, stratified acceptance sample of
  applied agent merges and links (§11.2), its marks in ``concept_reviews`` and
  :func:`review_summary`.

Rules chosen where the design is silent:

- ``concept_reviews.mark`` (schema 0010) has no "should have been a merge". A link marked
  so is stored as ``mark = 'wrong'`` with the note prefix :data:`SHOULD_BE_MERGE_PREFIX`,
  and :func:`review_summary` decodes it.
- The review population is agent rows, ``applied``, ``kind in (merge, link)``, with a
  ``batch_id`` (rule and migrated rows excluded). The sample's decision ids, seed and
  size are frozen in ``library_meta`` under :data:`SAMPLE_META_PREFIX` when it is created.
- Accepting a queued agent proposal rewrites that row as the researcher's (same subjects,
  same id), keeping its batch, note and quotes; it then counts as a researcher decision.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from . import guardrails as g
from . import refresh as rf
from .decide import GLOSS_MAX, NOTE_MAX, Guard
from .families import CONCEPT_FAMILIES
from .model import (
    DEFAULT_SENSE,
    SenseKey,
    claim_fingerprint,
    decision_id,
    label_key,
    ordered_pair,
)

if TYPE_CHECKING:
    from ..library.store import Library

SHOULD_BE_MERGE_PREFIX = "[should_be_merge]"
SAMPLE_META_PREFIX = "concept_review_sample:"
SAMPLE_SIZE = 50
AMBIGUOUS_MIN = 10
SPLIT_ORIGIN = "split:"
SPLIT_S0_REF = "replaced_by_split"
LIVE = ("applied", "pending_apply", "queued")

MARKS = {
    "merge": ("right", "wrong", "should_be_link", "should_be_split"),
    "link": ("right", "wrong", "should_be_merge"),
}
QUEUE_ACTIONS = ("accept", "reject", "merge", "link", "unrelated")


class ResearcherError(ValueError):
    """A researcher action that cannot be carried out (the message says why)."""


@dataclass(frozen=True)
class Outcome:
    decision_id: str | None
    message: str
    restored: tuple[str, ...] = ()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _refresh(conn: sqlite3.Connection, now: str) -> None:
    rf.apply_plan(conn, rf.plan(conn, now=now))


def _norm(key: SenseKey) -> SenseKey:
    return SenseKey(key.family, label_key(key.label_key), key.sense_id or DEFAULT_SENSE)


def _row(conn: sqlite3.Connection, did: str) -> dict[str, Any] | None:
    cols = rf.DECISION_COLS + ("quote_a_json", "quote_b_json", "confidence", "batch_id", "packet_id", "model")
    r = conn.execute(f"SELECT {', '.join(cols)} FROM concept_decisions WHERE decision_id = ?", (did,)).fetchone()
    return dict(zip(cols, r)) if r else None


def _require_member(guard: Guard, key: SenseKey) -> None:
    if key not in guard.members:
        raise ResearcherError(f"no concept for {key.family}/{key.label_key} ({key.sense_id}); refresh concepts first")


def _sides(row: Mapping[str, Any]) -> tuple[SenseKey, SenseKey] | None:
    """The two sense keys of a decision row (``None`` when it names no family)."""
    fam = row["family"]
    if not fam:
        return None
    fb = row["family_b"] or fam
    return (SenseKey(fam, row["label_a"], row["sense_a"] or DEFAULT_SENSE),
            SenseKey(fb, row["label_b"], row["sense_b"] or DEFAULT_SENSE))


def _ordered(kind: str, a: SenseKey, b: SenseKey) -> tuple[str, dict[str, Any], bool]:
    sa = None if a.sense_id == DEFAULT_SENSE else a.sense_id
    sb = None if b.sense_id == DEFAULT_SENSE else b.sense_id
    la, sa_, fa, lb, sb_, fb = ordered_pair(a.label_key, sa, b.label_key, sb, a.family, b.family)
    swapped = (la, sa_, fa) != (a.label_key, sa, a.family)
    family_b = fb if fb != fa else None
    did = decision_id(kind, la, lb, family=fa, family_b=family_b, sense_a=sa_, sense_b=sb_)
    cols = {"family": fa, "family_b": family_b, "label_a": la, "sense_a": sa_, "label_b": lb, "sense_b": sb_}
    return did, cols, swapped


def _insert(conn: sqlite3.Connection, kind: str, a: SenseKey, b: SenseKey, *, now: str, name: str | None = None,
            note: str | None = None, qa: str | None = None, qb: str | None = None, origin: str = "library",
            carry: Mapping[str, Any] | None = None) -> str:
    did, cols, swapped = _ordered(kind, a, b)
    if swapped:
        qa, qb = qb, qa
    carry = carry or {}
    conn.execute(
        "INSERT OR REPLACE INTO concept_decisions (decision_id, kind, family, family_b, label_a, sense_a, label_b, "
        "sense_b, name, decided_by, state, note, quote_a_json, quote_b_json, confidence, veto_reason, batch_id, "
        "packet_id, model, origin, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'researcher', 'applied', ?, ?, ?, "
        "NULL, NULL, ?, ?, ?, ?, ?)",
        (did, kind, cols["family"], cols["family_b"], cols["label_a"], cols["sense_a"], cols["label_b"],
         cols["sense_b"], name, note, qa, qb, carry.get("batch_id"), carry.get("packet_id"), carry.get("model"),
         origin, now))
    return did


def _override(conn: sqlite3.Connection, guard: Guard, a: SenseKey, b: SenseKey, did: str,
              kinds: Iterable[str]) -> list[str]:
    """Set the live rows of ``kinds`` on the pair (other than ``did``) undone, recording
    what to restore on undo. Researcher rows are overridden only when they are vetoes."""
    kinds = set(kinds)
    out = []
    for d in guard.pair_rows(a, b):
        if d["decision_id"] == did or d["kind"] not in kinds:
            continue
        if d["decided_by"] == "researcher" and d["kind"] not in ("never_merge", "never_connect"):
            continue
        conn.execute("UPDATE concept_decisions SET state = 'undone', veto_reason = ? WHERE decision_id = ?",
                     (_overridden(did, d["state"], d["veto_reason"]), d["decision_id"]))
        out.append(d["decision_id"])
    return out


def _overridden(did: str, state: str, reason: str | None) -> str:
    return f"overridden by {did} from {state}|{reason or ''}"


def _vetoes_without(guard: Guard, skip: set[str]) -> g.VetoIndex:
    return g.VetoIndex(
        g.Veto(d["kind"], d["label_a"], d["label_b"], d["family"], d["family_b"] or d["family"], d["sense_a"],
               d["sense_b"], d["decision_id"])
        for d in guard.decisions if d["kind"] in ("never_merge", "never_connect")
        and d["state"] in ("applied", "pending_apply") and d["decision_id"] not in skip)


def _direct_vetoes(guard: Guard, a: SenseKey, b: SenseKey, kinds: tuple[str, ...]) -> set[str]:
    return {d["decision_id"] for d in guard.pair_rows(a, b) if d["kind"] in kinds}


# --- merge, link, never connect, unrelated ----------------------------------------------------------

def _merge(conn: sqlite3.Connection, a: SenseKey, b: SenseKey, *, name: str | None, now: str,
           carry: Mapping[str, Any] | None = None, note: str | None = None) -> Outcome:
    if a.family != b.family:
        raise ResearcherError(f"a merge stays within one family ({a.family} vs {b.family}); link them instead")
    if a == b:
        raise ResearcherError("choose two different concepts to merge")
    guard = Guard.load(conn)
    _require_member(guard, a)
    _require_member(guard, b)
    ga, gb = guard.group(a), guard.group(b)
    direct = _direct_vetoes(guard, a, b, ("never_merge", "never_connect"))
    if ga != gb:
        why = g.merge_veto(ga, gb, _vetoes_without(guard, direct), stoplist=False)
        if why:
            raise ResearcherError(f"merge refused: the two groups would hold a {why.replace('_', '-')} pair "
                                  "through other members; split that merge first")
    if name is not None and not g.closure_ok(name, [k.label_key for k in (*ga, *gb)]):
        raise ResearcherError("the name must be one of the member labels")
    did, _, _ = _ordered("merge", a, b)
    _override(conn, guard, a, b, did, ("never_merge", "never_connect", "merge", "link", "unrelated"))
    _insert(conn, "merge", a, b, now=now, name=label_key(name) if name else None, note=note, carry=carry)
    return Outcome(did, "merged")


def merge(lib: Library, a: SenseKey, b: SenseKey, *, name: str | None = None, now: str | None = None) -> Outcome:
    """A researcher merge of two senses' concepts (see the module doc)."""
    now = now or _now()
    with lib.write() as conn:
        out = _merge(conn, _norm(a), _norm(b), name=name, now=now)
        _refresh(conn, now)
    return out


def _quote_for(conn: sqlite3.Connection, key: SenseKey, claim_id: str | None) -> str:
    params: list[Any] = [key.family, key.label_key, key.sense_id]
    extra = ""
    if claim_id:
        extra = " AND k.claim_id = ?"
        params.append(claim_id)
    r = conn.execute(
        "SELECT cc.claim_id, cc.paper_id, cc.span_id, cc.claim_type, cc.field_key, cc.normalized_label, "
        "cc.claim_text, e.quote_hash, e.markdown_hash, e.start_char, e.end_char FROM claim_concepts k "
        "JOIN current_claims cc ON cc.claim_id = k.claim_id LEFT JOIN evidence_spans e ON e.span_id = cc.span_id "
        "WHERE k.family = ? AND k.label_key = ? AND k.sense_id = ? AND k.supports = 1 AND cc.span_id IS NOT NULL"
        f"{extra} ORDER BY cc.paper_id, cc.claim_id LIMIT 1", params).fetchone()
    if r is None:
        what = f"claim {claim_id} is not a quoted, supporting claim of" if claim_id else "no quoted claim supports"
        raise ResearcherError(f"{what} {key.family}/{key.label_key}")
    cols = ("claim_id", "paper_id", "span_id", "claim_type", "field_key", "normalized_label", "claim_text",
            "quote_hash", "markdown_hash", "start_char", "end_char")
    row = dict(zip(cols, r))
    fp = claim_fingerprint(row, row if row["quote_hash"] is not None else None)
    return json.dumps({"paper_id": row["paper_id"], "span_id": row["span_id"], "claim_fingerprint": fp},
                      ensure_ascii=False, sort_keys=True)


def _clean_note(note: str | None, *, required: bool) -> str | None:
    text = " ".join((note or "").split())
    if required and not text:
        raise ResearcherError("a link needs your note: the difference, or the history for drift")
    if len(text) > NOTE_MAX:
        raise ResearcherError(f"the note is longer than {NOTE_MAX} characters")
    return text or None


def _link(conn: sqlite3.Connection, a: SenseKey, b: SenseKey, *, note: str | None, claim_a: str | None,
          claim_b: str | None, now: str, carry: Mapping[str, Any] | None = None,
          quotes: tuple[str | None, str | None] | None = None) -> Outcome:
    text = _clean_note(note, required=True)
    guard = Guard.load(conn)
    _require_member(guard, a)
    _require_member(guard, b)
    if guard.group(a) == guard.group(b):
        raise ResearcherError("these senses are already one concept; split the merge first")
    direct = _direct_vetoes(guard, a, b, ("never_connect",))
    if g.link_veto(guard.group(a), guard.group(b), _vetoes_without(guard, direct)):
        raise ResearcherError("link refused: a never-connect pair lies between the two concepts")
    if quotes and quotes[0] and quotes[1] and not (claim_a or claim_b):
        qa, qb = quotes
    else:
        qa, qb = _quote_for(conn, a, claim_a), _quote_for(conn, b, claim_b)
    did, _, _ = _ordered("link", a, b)
    _override(conn, guard, a, b, did, ("never_connect", "merge", "link", "unrelated"))
    _insert(conn, "link", a, b, now=now, note=text, qa=qa, qb=qb, carry=carry)
    return Outcome(did, "linked")


def link(lib: Library, a: SenseKey, b: SenseKey, *, note: str, claim_a: str | None = None,
         claim_b: str | None = None, now: str | None = None) -> Outcome:
    """A researcher closely related link with the researcher's own note and a quote from
    each side (the chosen claims, else the first quoted supporting claim of each sense)."""
    now = now or _now()
    with lib.write() as conn:
        out = _link(conn, _norm(a), _norm(b), note=note, claim_a=claim_a, claim_b=claim_b, now=now)
        _refresh(conn, now)
    return out


def _veto_pair(conn: sqlite3.Connection, kind: str, a: SenseKey, b: SenseKey, *, note: str | None,
               now: str, origin: str = "library") -> Outcome:
    if a == b:
        raise ResearcherError("choose two different concepts")
    guard = Guard.load(conn)
    did, _, _ = _ordered(kind, a, b)
    undo_kinds = {"never_connect": ("merge", "link", "unrelated"), "unrelated": ("merge", "link", "unrelated")}
    _override(conn, guard, a, b, did, undo_kinds[kind])
    _insert(conn, kind, a, b, now=now, note=_clean_note(note, required=False), origin=origin)
    return Outcome(did, kind.replace("_", " "))


def never_connect(lib: Library, a: SenseKey, b: SenseKey, *, note: str | None = None,
                  now: str | None = None) -> Outcome:
    """Neither merge nor link these two senses again (agent rows on the pair are undone)."""
    now = now or _now()
    with lib.write() as conn:
        out = _veto_pair(conn, "never_connect", _norm(a), _norm(b), note=note, now=now)
        _refresh(conn, now)
    return out


def unrelated(lib: Library, a: SenseKey, b: SenseKey, *, now: str | None = None) -> Outcome:
    """The researcher's "no connection" (never judged again)."""
    now = now or _now()
    with lib.write() as conn:
        out = _veto_pair(conn, "unrelated", _norm(a), _norm(b), note=None, now=now)
        _refresh(conn, now)
    return out


# --- split / undo of decisions ----------------------------------------------------------------------

def split_decision(lib: Library, did: str, *, now: str | None = None) -> Outcome:
    """Split an applied merge, or remove an applied link (any maker): the row becomes
    ``undone`` and a researcher ``never_connect`` records it (``origin = 'split:<id>'``)."""
    now = now or _now()
    with lib.write() as conn:
        row = _row(conn, did)
        if row is None or row["kind"] not in ("merge", "link") or row["state"] != "applied":
            raise ResearcherError("only an applied merge or link can be split")
        conn.execute("UPDATE concept_decisions SET state = 'undone', veto_reason = 'split by researcher' "
                     "WHERE decision_id = ?", (did,))
        nc = decision_id("never_connect", row["label_a"], row["label_b"], family=row["family"],
                         family_b=row["family_b"], sense_a=row["sense_a"], sense_b=row["sense_b"])
        conn.execute(
            "INSERT OR REPLACE INTO concept_decisions (decision_id, kind, family, family_b, label_a, sense_a, "
            "label_b, sense_b, decided_by, state, note, origin, decided_at) VALUES (?, 'never_connect', ?, ?, ?, ?, "
            "?, ?, 'researcher', 'applied', NULL, ?, ?)",
            (nc, row["family"], row["family_b"], row["label_a"], row["sense_a"], row["label_b"], row["sense_b"],
             SPLIT_ORIGIN + did, now))
        _refresh(conn, now)
    return Outcome(nc, "split (recorded as never connect)" if row["kind"] == "merge" else
                   "link removed (recorded as never connect)")


def _restore_overridden(conn: sqlite3.Connection, did: str) -> list[str]:
    out = []
    prefix = f"overridden by {did} from "
    for rid, reason in conn.execute("SELECT decision_id, veto_reason FROM concept_decisions WHERE state = 'undone' "
                                    "AND veto_reason LIKE ? ORDER BY decision_id", (prefix + "%",)).fetchall():
        state, _, prior_reason = reason[len(prefix):].partition("|")
        conn.execute("UPDATE concept_decisions SET state = ?, veto_reason = ? WHERE decision_id = ?",
                     (state, prior_reason or None, rid))
        out.append(rid)
    return out


def undo(lib: Library, did: str, *, now: str | None = None) -> Outcome:
    """Undo a split record (the merge or link comes back) or a researcher decision (the
    rows it overrode come back). Agent and rule decisions are split, not undone."""
    now = now or _now()
    with lib.write() as conn:
        row = _row(conn, did)
        if row is None or row["state"] not in ("applied", "pending_apply", "queued"):
            raise ResearcherError("nothing to undo: no such live decision")
        restored: list[str] = []
        if row["kind"] == "never_connect" and (row["origin"] or "").startswith(SPLIT_ORIGIN):
            original = row["origin"][len(SPLIT_ORIGIN):]
            conn.execute("UPDATE concept_decisions SET state = 'undone', veto_reason = 'undone by researcher' "
                         "WHERE decision_id = ?", (did,))
            conn.execute("UPDATE concept_decisions SET state = 'applied', veto_reason = NULL WHERE decision_id = ? "
                         "AND state = 'undone'", (original,))
            restored.append(original)
            msg = "split undone"
        elif row["decided_by"] == "researcher":
            conn.execute("UPDATE concept_decisions SET state = 'undone', veto_reason = 'undone by researcher' "
                         "WHERE decision_id = ?", (did,))
            restored = _restore_overridden(conn, did)
            msg = f"{row['kind'].replace('_', ' ')} undone"
        else:
            raise ResearcherError("agent and rule decisions are split or removed, not undone")
        _refresh(conn, now)
    return Outcome(did, msg, tuple(restored))


# --- senses -----------------------------------------------------------------------------------------

def _active_split(conn: sqlite3.Connection, family: str, key: str) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT sense_id FROM concept_senses WHERE family = ? AND label_key = ? AND sense_id <> ? "
        "AND state IN ('default', 'confirmed') ORDER BY sense_id", (family, key, DEFAULT_SENSE))]


def _claim_rows(conn: sqlite3.Connection, family: str, key: str) -> dict[str, dict[str, Any]]:
    cols = ("claim_id", "paper_id", "claim_type", "field_key", "normalized_label", "claim_text", "quote_hash",
            "markdown_hash", "start_char", "end_char")
    out = {}
    for r in conn.execute(
            "SELECT cc.claim_id, cc.paper_id, cc.claim_type, cc.field_key, cc.normalized_label, cc.claim_text, "
            "e.quote_hash, e.markdown_hash, e.start_char, e.end_char FROM claim_concepts k "
            "JOIN current_claims cc ON cc.claim_id = k.claim_id LEFT JOIN evidence_spans e ON e.span_id = cc.span_id "
            "WHERE k.family = ? AND k.label_key = ? ORDER BY cc.claim_id", (family, key)):
        out[r[0]] = dict(zip(cols, r))
    return out


def split_label(lib: Library, family: str, label: str, senses: Sequence[tuple[str, Sequence[str]]], *,
                now: str | None = None) -> Outcome:
    """Split one label into senses ``s1..sn``: ``senses`` is ``[(gloss, [claim_id, ...]), ...]``
    (at least two, each glossed; a claim in at most one sense; unassigned claims show as
    "needs sense"). Agent decisions on the label's default sense become ``needs_rejudge``."""
    now = now or _now()
    key = label_key(label)
    if len(senses) < 2:
        raise ResearcherError("a split needs at least two senses")
    seen: Counter = Counter()
    for i, (gloss, claims) in enumerate(senses, start=1):
        text = " ".join((gloss or "").split())
        if not text:
            raise ResearcherError(f"sense {i} needs a gloss")
        if len(text) > GLOSS_MAX:
            raise ResearcherError(f"sense {i}: the gloss is longer than {GLOSS_MAX} characters")
        seen.update(claims)
    twice = sorted(c for c, n in seen.items() if n > 1)
    if twice:
        raise ResearcherError(f"a claim can belong to one sense only: {', '.join(twice)}")
    with lib.write() as conn:
        if _active_split(conn, family, key):
            raise ResearcherError("this label is already split; undo that split first")
        rows = _claim_rows(conn, family, key)
        if not rows:
            raise ResearcherError(f"no claims carry the label {family}/{key}")
        unknown = sorted(c for c in seen if c not in rows)
        if unknown:
            raise ResearcherError(f"not claims of this label: {', '.join(unknown)}")
        conn.execute("DELETE FROM sense_assignments WHERE family = ? AND label_key = ?", (family, key))
        conn.execute("UPDATE concept_senses SET state = 'undone', decision_ref = ? WHERE family = ? AND label_key = ? "
                     "AND sense_id = ? AND state = 'confirmed'", (SPLIT_S0_REF, family, key, DEFAULT_SENSE))
        conn.execute("UPDATE concept_senses SET state = 'undone' WHERE family = ? AND label_key = ? AND sense_id = ? "
                     "AND state IN ('default', 'queued')", (family, key, DEFAULT_SENSE))
        for i, (gloss, claims) in enumerate(senses, start=1):
            sid = f"s{i}"
            conn.execute(
                "INSERT OR REPLACE INTO concept_senses (family, label_key, sense_id, gloss, decided_by, state, "
                "decision_ref, origin, decided_at) VALUES (?, ?, ?, ?, 'researcher', 'confirmed', NULL, 'library', ?)",
                (family, key, sid, " ".join(gloss.split()), now))
            for cid in claims:
                r = rows[cid]
                fp = claim_fingerprint(r, r if r["quote_hash"] is not None else None)
                conn.execute(
                    "INSERT OR REPLACE INTO sense_assignments (family, label_key, paper_id, claim_fingerprint, "
                    "sense_id, decided_by, decided_at) VALUES (?, ?, ?, ?, ?, 'researcher', ?)",
                    (family, key, r["paper_id"], fp, sid, now))
        conn.execute("UPDATE concept_senses SET state = 'undone' WHERE family = ? AND label_key = ? AND sense_id > ? "
                     "AND sense_id <> ? AND decided_by <> 'researcher'", (family, key, f"s{len(senses)}",
                                                                        DEFAULT_SENSE))
        conn.execute(
            "UPDATE concept_decisions SET state = 'needs_rejudge' WHERE decided_by = 'agent' AND state IN "
            "('applied', 'queued') AND kind IN ('merge', 'link', 'unrelated') AND (family IS NULL OR family = ? OR "
            "family_b = ?) AND ((label_a = ? AND sense_a IS NULL) OR (label_b = ? AND sense_b IS NULL))",
            (family, family, key, key))
        _refresh(conn, now)
    return Outcome(None, f"split into {len(senses)} senses")


def undo_label_split(lib: Library, family: str, label: str, *, now: str | None = None) -> Outcome:
    """Collapse a split label back to ``s0`` (assignments kept for history); agent
    decisions naming ``s1..sn`` become ``needs_rejudge``."""
    now = now or _now()
    key = label_key(label)
    with lib.write() as conn:
        sids = _active_split(conn, family, key)
        if not sids:
            raise ResearcherError("this label is not split")
        marks = ", ".join("?" * len(sids))
        conn.execute(f"UPDATE concept_senses SET state = 'undone' WHERE family = ? AND label_key = ? "
                     f"AND sense_id IN ({marks})", (family, key, *sids))
        conn.execute("UPDATE concept_senses SET state = 'confirmed', decision_ref = NULL WHERE family = ? AND "
                     "label_key = ? AND sense_id = ? AND decision_ref = ?", (family, key, DEFAULT_SENSE, SPLIT_S0_REF))
        conn.execute(
            f"UPDATE concept_decisions SET state = 'needs_rejudge' WHERE decided_by = 'agent' AND state IN "
            f"('applied', 'queued') AND (family IS NULL OR family = ? OR family_b = ?) AND "
            f"((label_a = ? AND sense_a IN ({marks})) OR (label_b = ? AND sense_b IN ({marks})))",
            (family, family, key, *sids, key, *sids))
        _refresh(conn, now)
    return Outcome(None, "split undone: one sense again")


# --- the settings lists -----------------------------------------------------------------------------

def add_never_merge(lib: Library, label_a: str, label_b: str, *, family: str | None = None,
                    note: str | None = None, now: str | None = None) -> Outcome:
    """A researcher never-merge row (library-wide; ``family`` narrows it to one family)."""
    now = now or _now()
    la, lb = sorted((label_key(label_a), label_key(label_b)))
    if not la or not lb or la == lb:
        raise ResearcherError("give two different labels")
    if family is not None and family not in CONCEPT_FAMILIES:
        raise ResearcherError(f"unknown family {family!r}")
    did = decision_id("never_merge", la, lb, family=family)
    with lib.write() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO concept_decisions (decision_id, kind, family, label_a, label_b, decided_by, state, "
            "note, origin, decided_at) VALUES (?, 'never_merge', ?, ?, ?, 'researcher', 'applied', ?, 'library', ?)",
            (did, family, la, lb, _clean_note(note, required=False), now))
        _refresh(conn, now)
    return Outcome(did, "added to the never-merge list")


def remove_never_merge(lib: Library, did: str, *, now: str | None = None) -> Outcome:
    now = now or _now()
    with lib.write() as conn:
        row = _row(conn, did)
        if row is None or row["kind"] != "never_merge" or row["state"] not in ("applied", "pending_apply"):
            raise ResearcherError("no such never-merge row")
        conn.execute("UPDATE concept_decisions SET state = 'undone', veto_reason = 'removed by researcher' "
                     "WHERE decision_id = ?", (did,))
        _refresh(conn, now)
    return Outcome(did, "removed from the never-merge list")


AMBIGUOUS_ACTIONS = ("add", "single_sense", "remove")


def edit_ambiguous(lib: Library, action: str, pattern: str, *, match: str = "label", graph: str | None = None,
                   note: str | None = None, now: str | None = None) -> Outcome:
    """The ambiguous-label list: ``add`` (library-wide, or to one graph), ``single_sense``
    (a library pattern is single-sense within ``graph``), ``remove`` (a library row or a
    graph override). Library changes refresh the concepts (they change sense checks)."""
    now = now or _now()
    try:
        if action == "add":
            key = g.add_ambiguous(lib, pattern, match=match, graph=graph or None, note=note, now=now)
            msg = f"{key!r} is ambiguous" + (f" in {graph}" if graph else "")
        elif action == "single_sense":
            key = g.mark_single_sense(lib, pattern, graph or "", note=note, now=now)
            msg = f"{key!r} is single-sense in {graph}"
        elif action == "remove":
            if not g.remove_ambiguous(lib, pattern, graph=graph or None):
                raise ResearcherError("no such ambiguous-label row")
            msg = "removed from the ambiguous list"
        else:
            raise ResearcherError(f"unknown action {action!r}")
    except ValueError as exc:
        raise ResearcherError(str(exc)) from None
    if not graph:
        rf.refresh(lib, now=now)
    return Outcome(None, msg)


# --- the queue --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class QueueItem:
    item_id: str                    # a decision id, or 'sense:<family>/<label_key>'
    kind: str                       # 'pair' | 'sense'
    proposal: str                   # merge | link | unsure | unrelated | split | same_sense
    reason: str | None
    a: SenseKey
    b: SenseKey | None
    note: str | None
    name: str | None
    confidence: str | None
    batch_id: str | None
    decided_by: str
    reach: int
    glosses: tuple[str, ...] = ()

    @property
    def actions(self) -> tuple[str, ...]:
        if self.kind == "sense":
            return ("accept", "reject")
        if self.proposal in ("merge", "link"):
            return ("accept", "reject", "link") if self.proposal == "merge" else ("accept", "reject")
        return ("merge", "link", "unrelated")


def _papers(conn: sqlite3.Connection) -> dict[SenseKey, int]:
    """Concept paper counts by member sense."""
    return {SenseKey(r[0], r[1], r[2]): r[3] for r in conn.execute(
        "SELECT m.family, m.label_key, m.sense_id, c.paper_count FROM concept_members m "
        "JOIN concepts c ON c.concept_id = m.concept_id")}


def _family_for(conn: sqlite3.Connection, row: Mapping[str, Any]) -> str | None:
    fams = [r[0] for r in conn.execute(
        "SELECT family FROM concept_members WHERE label_key = ? INTERSECT SELECT family FROM concept_members "
        "WHERE label_key = ? ORDER BY 1", (row["label_a"], row["label_b"]))]
    return fams[0] if fams else None


def queue(conn: sqlite3.Connection) -> list[QueueItem]:
    """The researcher's short queue: queued decisions and queued senses, by reach."""
    papers = _papers(conn)
    items: list[QueueItem] = []
    cols = rf.DECISION_COLS + ("confidence", "batch_id")
    for row in rf._rows(conn, "concept_decisions", cols, "decision_id"):
        if row["state"] != "queued":
            continue
        if row["kind"] not in ("merge", "link", "unrelated"):
            continue
        if not row["family"]:
            fam = _family_for(conn, row)
            if fam is None:
                continue
            row = {**row, "family": fam}
        sides = _sides(row)
        assert sides is not None
        a, b = sides
        proposal = "unsure" if row["veto_reason"] == "unsure" else row["kind"]
        items.append(QueueItem(row["decision_id"], "pair", proposal, row["veto_reason"], a, b, row["note"],
                               row["name"], row["confidence"], row["batch_id"], row["decided_by"],
                               min(papers.get(a, 0), papers.get(b, 0))))
    by_label: dict[tuple[str, str], list[tuple[str, str | None]]] = defaultdict(list)
    for fam, key, sid, gloss in conn.execute(
            "SELECT family, label_key, sense_id, gloss FROM concept_senses WHERE state = 'queued' "
            "ORDER BY family, label_key, sense_id"):
        by_label[(fam, key)].append((sid, gloss))
    for (fam, key), senses in sorted(by_label.items()):
        split = [s for s in senses if s[0] != DEFAULT_SENSE]
        a = SenseKey(fam, key, DEFAULT_SENSE)
        reach = max((n for k, n in papers.items() if k.family == fam and k.label_key == key), default=0)
        items.append(QueueItem(f"sense:{fam}/{key}", "sense", "split" if split else "same_sense",
                               "sense check", a, None, None, None, None, None, "agent", reach,
                               tuple(gl or "" for _, gl in split)))
    items.sort(key=lambda it: (-it.reach, it.item_id))
    return items


def _settle_sense(conn: sqlite3.Connection, family: str, key: str, action: str, now: str) -> Outcome:
    rows = [r[0] for r in conn.execute("SELECT sense_id FROM concept_senses WHERE family = ? AND label_key = ? "
                                       "AND state = 'queued' ORDER BY sense_id", (family, key))]
    if not rows:
        raise ResearcherError("no queued sense proposal for this label")
    if action == "reject":
        conn.execute("UPDATE concept_senses SET state = 'undone' WHERE family = ? AND label_key = ? "
                     "AND state = 'queued'", (family, key))
        return Outcome(None, "sense proposal rejected")
    if action != "accept":
        raise ResearcherError(f"{action!r} is not an action for a sense proposal")
    if any(s != DEFAULT_SENSE for s in rows):
        if _active_split(conn, family, key):
            raise ResearcherError("this label is already split; undo that split first")
        conn.execute("UPDATE concept_senses SET state = 'undone' WHERE family = ? AND label_key = ? AND sense_id = ? "
                     "AND state = 'queued'", (family, key, DEFAULT_SENSE))
        conn.execute("UPDATE concept_senses SET state = 'confirmed', decided_by = 'researcher', decided_at = ? "
                     "WHERE family = ? AND label_key = ? AND state = 'queued'", (now, family, key))
        conn.execute(
            "UPDATE concept_decisions SET state = 'needs_rejudge' WHERE decided_by = 'agent' AND state IN "
            "('applied', 'queued') AND kind IN ('merge', 'link', 'unrelated') AND (family IS NULL OR family = ? OR "
            "family_b = ?) AND ((label_a = ? AND sense_a IS NULL) OR (label_b = ? AND sense_b IS NULL))",
            (family, family, key, key))
        return Outcome(None, "split accepted")
    conn.execute("UPDATE concept_senses SET state = 'confirmed', decided_by = 'researcher', decided_at = ? "
                 "WHERE family = ? AND label_key = ? AND state = 'queued'", (now, family, key))
    return Outcome(None, "single sense confirmed")


def settle(lib: Library, item_id: str, action: str, *, note: str | None = None, name: str | None = None,
           now: str | None = None) -> Outcome:
    """Settle one queue item: ``accept`` (the proposal becomes the researcher's),
    ``reject`` (recorded as unrelated; for senses: the proposal is dropped), ``merge``,
    ``link`` (link instead, with the researcher's note) or ``unrelated``."""
    now = now or _now()
    if action not in QUEUE_ACTIONS:
        raise ResearcherError(f"unknown queue action {action!r}")
    with lib.write() as conn:
        if item_id.startswith("sense:"):
            fam, _, key = item_id[len("sense:"):].partition("/")
            out = _settle_sense(conn, fam, key, action, now)
            _refresh(conn, now)
            return out
        row = _row(conn, item_id)
        if row is None or row["state"] != "queued":
            raise ResearcherError("this item is no longer in the queue")
        if not row["family"]:
            fam = _family_for(conn, row)
            if fam is None:
                raise ResearcherError("neither label has a concept now")
            row = {**row, "family": fam}
        a, b = _sides(row)  # type: ignore[misc]
        unsure = row["veto_reason"] == "unsure"
        carry = {k: row[k] for k in ("batch_id", "packet_id", "model")}
        if action == "accept":
            if unsure or row["kind"] not in ("merge", "link"):
                raise ResearcherError("the agent was unsure: choose merge, link or unrelated")
            action = row["kind"]
            if action == "merge":
                out = _merge(conn, a, b, name=row["name"], now=now, carry=carry, note=row["note"])
            else:
                out = _link(conn, a, b, note=note or row["note"], claim_a=None, claim_b=None, now=now,
                            carry=carry, quotes=(row["quote_a_json"], row["quote_b_json"]))
        elif action == "merge":
            out = _merge(conn, a, b, name=name, now=now)
        elif action == "link":
            out = _link(conn, a, b, note=note, claim_a=None, claim_b=None, now=now)
        else:  # reject / unrelated
            out = _veto_pair(conn, "unrelated", a, b, note=None, now=now)
        if out.decision_id != item_id:
            conn.execute("UPDATE concept_decisions SET state = 'undone', veto_reason = ? WHERE decision_id = ? "
                         "AND state = 'queued'", (_overridden(out.decision_id, "queued", row["veto_reason"]), item_id))
        _refresh(conn, now)
    return out


# --- the review sample (§11.2) ----------------------------------------------------------------------

@dataclass(frozen=True)
class ReviewSample:
    sample_id: str
    seed: int
    size: int
    created_at: str
    decision_ids: tuple[str, ...]


def _population(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cols = rf.DECISION_COLS + ("batch_id",)
    return [d for d in rf._rows(conn, "concept_decisions", cols, "decision_id")
            if d["decided_by"] == "agent" and d["state"] == "applied" and d["kind"] in ("merge", "link")
            and d["batch_id"]]


def _draw(population: list[dict[str, Any]], n: int, seed: int, ambiguous: g.AmbiguousList) -> list[str]:
    rng = random.Random(seed)
    pop = sorted(population, key=lambda d: d["decision_id"])
    if len(pop) <= n:
        return [d["decision_id"] for d in pop]
    amb = [d for d in pop if ambiguous.is_ambiguous(d["label_a"]) or ambiguous.is_ambiguous(d["label_b"])]
    chosen = rng.sample([d["decision_id"] for d in amb], min(len(amb), AMBIGUOUS_MIN, n))
    rest = [d for d in pop if d["decision_id"] not in set(chosen)]
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for d in rest:
        strata[(d["kind"], d["family"] or "")].append(d["decision_id"])
    quota = n - len(chosen)
    total = len(rest)
    shares = {k: quota * len(v) / total for k, v in sorted(strata.items())}
    alloc = {k: int(s) for k, s in shares.items()}
    for k in sorted(shares, key=lambda k: (-(shares[k] - alloc[k]), k))[:quota - sum(alloc.values())]:
        alloc[k] += 1
    for k in sorted(strata):
        chosen += rng.sample(strata[k], min(alloc[k], len(strata[k])))
    return sorted(chosen)


def create_review_sample(lib: Library, *, n: int = SAMPLE_SIZE, seed: int = 0, now: str | None = None) -> ReviewSample:
    """Draw and freeze a stratified sample of applied agent merges and links: verdict x
    family strata, ambiguous-label items over-sampled to at least :data:`AMBIGUOUS_MIN`,
    a seeded draw. The same population and seed give the same sample."""
    now = now or _now()
    if n < 1:
        raise ResearcherError("the sample needs at least one item")
    with lib.write() as conn:
        ids = _draw(_population(conn), n, seed, g.effective_ambiguous(conn))
        if not ids:
            raise ResearcherError("no applied agent merges or links to sample yet")
        sid = "rs_" + hashlib.sha256(json.dumps([seed, n, ids]).encode()).hexdigest()[:12]
        body = {"seed": seed, "size": n, "created_at": now, "decision_ids": ids}
        conn.execute("INSERT OR IGNORE INTO library_meta (key, value) VALUES (?, ?)",
                     (SAMPLE_META_PREFIX + sid, json.dumps(body, sort_keys=True)))
    return review_sample(lib.conn, sid)  # type: ignore[return-value]


def review_sample(conn: sqlite3.Connection, sample_id: str) -> ReviewSample | None:
    r = conn.execute("SELECT value FROM library_meta WHERE key = ?", (SAMPLE_META_PREFIX + sample_id,)).fetchone()
    if r is None:
        return None
    body = json.loads(r[0])
    return ReviewSample(sample_id, body["seed"], body["size"], body["created_at"], tuple(body["decision_ids"]))


def review_samples(conn: sqlite3.Connection) -> list[ReviewSample]:
    keys = [r[0][len(SAMPLE_META_PREFIX):] for r in conn.execute(
        "SELECT key FROM library_meta WHERE key LIKE ? ORDER BY key", (SAMPLE_META_PREFIX + "%",))]
    out = [s for s in (review_sample(conn, k) for k in keys) if s is not None]
    return sorted(out, key=lambda s: (s.created_at, s.sample_id), reverse=True)


def mark(lib: Library, sample_id: str, did: str, value: str, *, note: str | None = None,
         now: str | None = None) -> str:
    """Store one review mark. Merges: right / wrong / should_be_link / should_be_split.
    Links: right / wrong / should_be_merge (stored as ``wrong`` + the note prefix)."""
    now = now or _now()
    sample = review_sample(lib.conn, sample_id)
    if sample is None or did not in sample.decision_ids:
        raise ResearcherError("this decision is not in that review sample")
    kind = lib.conn.execute("SELECT kind FROM concept_decisions WHERE decision_id = ?", (did,)).fetchone()[0]
    if value not in MARKS.get(kind, ()):
        raise ResearcherError(f"{value!r} is not a mark for a {kind}; use one of {', '.join(MARKS[kind])}")
    text = " ".join((note or "").split()) or None
    stored = value
    if value == "should_be_merge":
        stored, text = "wrong", f"{SHOULD_BE_MERGE_PREFIX} {text}" if text else SHOULD_BE_MERGE_PREFIX
    with lib.write() as conn:
        conn.execute("INSERT OR REPLACE INTO concept_reviews (sample_id, decision_id, mark, note, marked_at) "
                     "VALUES (?, ?, ?, ?, ?)", (sample_id, did, stored, text, now))
    return value


def decode_mark(stored: str, note: str | None) -> tuple[str, str | None]:
    """``(mark, note)`` as the researcher gave them (undoing the should_be_merge encoding)."""
    if stored == "wrong" and note and note.startswith(SHOULD_BE_MERGE_PREFIX):
        rest = note[len(SHOULD_BE_MERGE_PREFIX):].strip()
        return "should_be_merge", rest or None
    return stored, note


@dataclass
class ReviewSummary:
    sample_id: str
    size: int
    marked: int
    by_mark: dict[str, int] = field(default_factory=dict)
    by_verdict: dict[str, dict[str, int]] = field(default_factory=dict)
    by_family: dict[str, dict[str, int]] = field(default_factory=dict)
    merges_sampled: int = 0
    wrong_merges: tuple[str, ...] = ()

    @property
    def gate_limit(self) -> float:
        """Allowed wrong merges: 3, or 10% when fewer than 30 merges are sampled (§11.2)."""
        return 3 if self.merges_sampled >= 30 else self.merges_sampled * 0.10

    @property
    def gate_passed(self) -> bool | None:
        if self.marked < self.size:
            return None
        return len(self.wrong_merges) <= self.gate_limit


def marks(conn: sqlite3.Connection, sample_id: str) -> dict[str, tuple[str, str | None]]:
    return {r[0]: decode_mark(r[1], r[2]) for r in conn.execute(
        "SELECT decision_id, mark, note FROM concept_reviews WHERE sample_id = ? ORDER BY decision_id",
        (sample_id,))}


def review_summary(conn: sqlite3.Connection, sample_id: str) -> ReviewSummary | None:
    sample = review_sample(conn, sample_id)
    if sample is None:
        return None
    got = marks(conn, sample_id)
    meta = {r[0]: (r[1], r[2]) for r in conn.execute(
        f"SELECT decision_id, kind, family FROM concept_decisions WHERE decision_id IN "
        f"({', '.join('?' * len(sample.decision_ids))})", sample.decision_ids)}
    s = ReviewSummary(sample_id, len(sample.decision_ids), len(got))
    s.merges_sampled = sum(1 for d in sample.decision_ids if meta.get(d, ("", ""))[0] == "merge")
    by_mark: Counter = Counter()
    by_verdict: dict[str, Counter] = defaultdict(Counter)
    by_family: dict[str, Counter] = defaultdict(Counter)
    wrong = []
    for did, (value, _) in got.items():
        kind, fam = meta.get(did, ("?", None))
        by_mark[value] += 1
        by_verdict[kind][value] += 1
        by_family[fam or "any"][value] += 1
        if kind == "merge" and value == "wrong":
            wrong.append(did)
    s.by_mark = dict(sorted(by_mark.items()))
    s.by_verdict = {k: dict(sorted(v.items())) for k, v in sorted(by_verdict.items())}
    s.by_family = {k: dict(sorted(v.items())) for k, v in sorted(by_family.items())}
    s.wrong_merges = tuple(sorted(wrong))
    return s


__all__ = [
    "MARKS", "Outcome", "QueueItem", "ResearcherError", "ReviewSample", "ReviewSummary", "add_never_merge",
    "create_review_sample", "decode_mark", "edit_ambiguous", "link", "mark", "marks", "merge", "never_connect",
    "queue", "remove_never_merge", "review_sample", "review_samples", "review_summary", "settle", "split_decision",
    "split_label", "undo", "undo_label_split", "unrelated",
]
