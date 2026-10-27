"""Cleanup of a stored concept batch (stage 4 design §11; slice C11).

The first real bridge batch was saved under the old, label-level disagreement rule, so a
few kinds of row ended up stored that the current save path refuses. :func:`findings` is
a read-only audit of one batch's agent rows over the concepts a refresh would build now:

- **conflicts**: two concepts with more than one live verdict of different kinds (through
  different member labels). ``both_applied`` when two different verdicts are applied.
- **inside**: a live ``link`` / ``unrelated`` row whose two sides are now one concept.
- **contradicted**: a live ``unrelated`` row inside one concept, with the merges on the
  shortest merge path that joined its two sides.
- **dead**: a live row naming a sense its label no longer has (the label was split after
  the package was frozen).

:func:`run` applies the resolutions in one library transaction and refreshes the concepts;
with ``apply=False`` it does the same work and rolls back, so the dry-run plan is exactly
what an apply writes. Order (each step sees the grouping the previous one left):

1. contradicted merges -> ``queued`` (``disagreement``); repeated until the unrelated
   pair is in two concepts or only researcher / rule merges join it (reported as blocked);
2. dead-sense rows -> ``needs_rejudge`` (``cleanup:label_split``);
3. rows on the given falsely vetoed label pairs -> ``needs_rejudge``
   (``cleanup:stoplist_refusal``);
4. link rows inside one concept -> ``undone`` (``cleanup:inside_concept``);
5. every live row of an unsettled conflict -> ``queued`` (``disagreement``): the
   researcher decides, no winner is picked.

A row matched by an earlier step is not touched again. Running the cleanup a second time
changes nothing. Each change is returned with the row's old state and reason, and
:func:`record` stores the list in ``library_meta`` (``concept_cleanup:<batch>:<time>``).
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from . import refresh as rf
from .model import DEFAULT_SENSE, SenseKey

if TYPE_CHECKING:
    from ..library.store import Library

LIVE = ("applied", "pending_apply", "queued")
DISAGREEMENT = "disagreement"
META_PREFIX = "concept_cleanup:"
REASONS = {
    "contradicted_merge": DISAGREEMENT,
    "dead_sense": "cleanup:label_split",
    "false_veto": "cleanup:stoplist_refusal",
    "inside_concept": "cleanup:inside_concept",
    "disagreement": DISAGREEMENT,
}
NEW_STATE = {
    "contradicted_merge": "queued",
    "dead_sense": "needs_rejudge",
    "false_veto": "needs_rejudge",
    "inside_concept": "undone",
    "disagreement": "queued",
}
RULES = tuple(REASONS)
_COLS = rf.DECISION_COLS + ("confidence", "batch_id")


class CleanupMismatch(RuntimeError):
    """The planned counts differ from the expected ones; nothing was written."""


class _Rollback(Exception):
    pass


# --- the read-only audit ------------------------------------------------------------------------

@dataclass(frozen=True)
class Conflict:
    concept_a: str
    concept_b: str
    name_a: str
    name_b: str
    rows: tuple[dict[str, Any], ...]

    @property
    def decision_ids(self) -> tuple[str, ...]:
        return tuple(r["decision_id"] for r in self.rows)

    @property
    def both_applied(self) -> bool:
        return len({r["kind"] for r in self.rows if r["state"] == "applied"}) > 1

    @property
    def settled(self) -> bool:
        """Every live row already waits for the researcher as a disagreement."""
        return all(r["state"] == "queued" and r["veto_reason"] == DISAGREEMENT for r in self.rows)


@dataclass(frozen=True)
class Contradiction:
    unrelated_id: str
    concept: str
    name: str
    merge_ids: tuple[str, ...]          # agent merges on the path (queueable)
    blocked_by: tuple[str, ...]         # researcher / rule merges on the path


@dataclass
class Findings:
    conflicts: list[Conflict] = field(default_factory=list)
    inside: list[dict[str, Any]] = field(default_factory=list)
    contradicted: list[Contradiction] = field(default_factory=list)
    dead: list[dict[str, Any]] = field(default_factory=list)
    missing: list[dict[str, Any]] = field(default_factory=list)   # a side whose label has no concept
    #: inside one concept, but already the researcher's: joined only by researcher / rule
    #: merges (the researcher accepted the merge), or queued as a disagreement
    overridden: list[dict[str, Any]] = field(default_factory=list)

    @property
    def unsettled(self) -> list[Conflict]:
        return [c for c in self.conflicts if not c.settled]

    @property
    def applied_conflicts(self) -> list[Conflict]:
        return [c for c in self.conflicts if c.both_applied]

    def counts(self) -> dict[str, int]:
        return {"conflicts": len(self.conflicts), "conflicts_both_applied": len(self.applied_conflicts),
                "conflicts_unsettled": len(self.unsettled), "inside": len(self.inside),
                "inside_links": sum(1 for r in self.inside if r["kind"] == "link"),
                "inside_unrelated": sum(1 for r in self.inside if r["kind"] == "unrelated"),
                "contradicted": len(self.contradicted), "dead": len(self.dead), "missing": len(self.missing),
                "overridden": len(self.overridden)}


def _sides(row: Mapping[str, Any]) -> tuple[SenseKey, SenseKey] | None:
    fam = row["family"]
    if not fam:
        return None
    return (SenseKey(fam, row["label_a"], row["sense_a"] or DEFAULT_SENSE),
            SenseKey(row["family_b"] or fam, row["label_b"], row["sense_b"] or DEFAULT_SENSE))


@dataclass(frozen=True)
class _Grouping:
    concept_of: dict[SenseKey, str]
    names: dict[str, str]
    labels: set[tuple[str, str]]

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> _Grouping:
        p = rf.plan(conn)
        concept_of = {SenseKey(f, k, s): cid for cid, f, k, s in p.members}
        return cls(concept_of, {r[0]: r[2] for r in p.concepts}, {(k.family, k.label_key) for k in concept_of})


def _rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return rf._rows(conn, "concept_decisions", _COLS, "decision_id")


def _merge_path(grouping: _Grouping, merges: Sequence[Mapping[str, Any]], a: SenseKey, b: SenseKey,
                cid: str) -> list[Mapping[str, Any]]:
    """The merge rows on the shortest path from ``a`` to ``b`` inside concept ``cid``."""
    adj: dict[SenseKey, list[tuple[str, SenseKey, Mapping[str, Any]]]] = defaultdict(list)
    fams = sorted({a.family, b.family})
    for m in merges:
        for fam in ([m["family"]] if m["family"] else fams):
            x = SenseKey(fam, m["label_a"], m["sense_a"] or DEFAULT_SENSE)
            y = SenseKey(m["family_b"] or fam, m["label_b"], m["sense_b"] or DEFAULT_SENSE)
            if grouping.concept_of.get(x) == cid and grouping.concept_of.get(y) == cid:
                adj[x].append((m["decision_id"], y, m))
                adj[y].append((m["decision_id"], x, m))
    prev: dict[SenseKey, tuple[SenseKey, Mapping[str, Any]] | None] = {a: None}
    todo = deque([a])
    while todo:
        x = todo.popleft()
        for _, y, m in sorted(adj[x], key=lambda t: t[0]):
            if y not in prev:
                prev[y] = (x, m)
                todo.append(y)
    path: list[Mapping[str, Any]] = []
    x = b
    while prev.get(x) is not None:
        px, m = prev[x]  # type: ignore[misc]
        path.append(m)
        x = px
    return path


def _findings(conn: sqlite3.Connection, batch_id: str, grouping: _Grouping) -> Findings:
    rows = _rows(conn)
    batch = [r for r in rows if r["batch_id"] == batch_id and r["decided_by"] == "agent" and r["state"] in LIVE
             and r["kind"] in ("merge", "link", "unrelated")]
    merges = [r for r in rows if r["kind"] == "merge" and r["state"] in ("applied", "pending_apply")]
    f = Findings()
    pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in batch:
        sides = _sides(r)
        if sides is None:
            continue
        a, b = sides
        ca, cb = grouping.concept_of.get(a), grouping.concept_of.get(b)
        if ca is None or cb is None:
            gone = [k for k, c in ((a, ca), (b, cb)) if c is None]
            if all((k.family, k.label_key) in grouping.labels for k in gone):
                f.dead.append(r)
            else:
                f.missing.append(r)
            continue
        if ca == cb:
            if r["kind"] == "merge":
                continue
            if r["state"] == "queued" and r["veto_reason"] == DISAGREEMENT:
                f.overridden.append(r)      # waits for the researcher
                continue
            agent_path: tuple[str, ...] = ()
            other_path: tuple[str, ...] = ()
            if r["kind"] == "unrelated":
                path = _merge_path(grouping, merges, a, b, ca)
                agent_path = tuple(sorted(m["decision_id"] for m in path if m["decided_by"] == "agent"))
                other_path = tuple(sorted(m["decision_id"] for m in path if m["decided_by"] != "agent"))
                if not agent_path and other_path:
                    f.overridden.append(r)  # the researcher (or a rule) joined the two sides
                    continue
            f.inside.append(r)
            if r["kind"] == "unrelated":
                f.contradicted.append(Contradiction(r["decision_id"], ca, grouping.names.get(ca, ""),
                                                    agent_path, other_path))
            continue
        pairs[tuple(sorted((ca, cb)))].append(r)  # type: ignore[index]
    for (ca, cb), group in pairs.items():
        if len({r["kind"] for r in group}) > 1:
            f.conflicts.append(Conflict(ca, cb, grouping.names.get(ca, ""), grouping.names.get(cb, ""),
                                        tuple(sorted(group, key=lambda r: r["decision_id"]))))
    f.conflicts.sort(key=lambda c: (not c.both_applied, sorted((c.name_a.lower(), c.name_b.lower())),
                                    c.concept_a))
    f.contradicted.sort(key=lambda c: c.unrelated_id)
    return f


def findings(conn: sqlite3.Connection, batch_id: str) -> Findings:
    """The read-only audit of one batch over the concepts a refresh would build now."""
    return _findings(conn, batch_id, _Grouping.load(conn))


# --- the cleanup run ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Change:
    decision_id: str
    rule: str
    old_state: str
    old_reason: str | None
    new_state: str
    new_reason: str
    kind: str
    family: str | None
    family_b: str | None
    label_a: str
    sense_a: str | None
    label_b: str
    sense_b: str | None
    confidence: str | None
    context: str = ""                   # concept names, for the report (home only: real labels)


@dataclass
class CleanupResult:
    batch_id: str
    applied: bool
    at: str
    changes: list[Change] = field(default_factory=list)
    before: dict[str, int] = field(default_factory=dict)       # findings counts before
    after: dict[str, int] = field(default_factory=dict)        # findings counts after (in the transaction)
    blocked: list[Contradiction] = field(default_factory=list)  # joined by researcher / rule merges only
    conflicts: list[Conflict] = field(default_factory=list)     # the conflicts as found (step 5)
    refresh: dict[str, Any] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        out = {r: 0 for r in RULES}
        for ch in self.changes:
            out[ch.rule] += 1
        return out

    def as_dict(self) -> dict[str, Any]:
        return {"batch_id": self.batch_id, "applied": self.applied, "at": self.at, "counts": self.counts(),
                "before": self.before, "after": self.after,
                "blocked": [asdict(c) for c in self.blocked],
                "changes": [asdict(c) for c in self.changes]}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _change(conn: sqlite3.Connection, row: Mapping[str, Any], rule: str, context: str,
            out: list[Change], done: set[str]) -> None:
    did = row["decision_id"]
    if did in done:
        return
    new_state, new_reason = NEW_STATE[rule], REASONS[rule]
    if row["state"] == new_state and row["veto_reason"] == new_reason:
        return
    done.add(did)
    conn.execute("UPDATE concept_decisions SET state = ?, veto_reason = ? WHERE decision_id = ?",
                 (new_state, new_reason, did))
    out.append(Change(did, rule, row["state"], row["veto_reason"], new_state, new_reason, row["kind"],
                      row["family"], row["family_b"], row["label_a"], row["sense_a"], row["label_b"],
                      row["sense_b"], row["confidence"], context))


def _pair_keys(pairs: Iterable[Sequence[str]]) -> set[tuple[str, str]]:
    from .model import label_key
    return {tuple(sorted((label_key(x), label_key(y)))) for x, y in pairs}  # type: ignore[misc]


def _steps(conn: sqlite3.Connection, res: CleanupResult, false_vetoes: set[tuple[str, str]]) -> None:
    done: set[str] = set()
    out = res.changes
    by_id = {r["decision_id"]: r for r in _rows(conn)}
    grouping = _Grouping.load(conn)
    f = _findings(conn, res.batch_id, grouping)
    res.before = f.counts()
    # 1. contradicted merges, until each unrelated pair is apart (or only non-agent merges join it)
    seen_blocked: dict[str, Contradiction] = {}
    for _ in range(50):
        todo = [c for c in f.contradicted if c.merge_ids]
        for c in f.contradicted:
            if not c.merge_ids:
                seen_blocked[c.unrelated_id] = c
        if not todo:
            break
        for c in todo:
            for mid in c.merge_ids:
                _change(conn, by_id[mid], "contradicted_merge", c.name, out, done)
        by_id = {r["decision_id"]: r for r in _rows(conn)}
        grouping = _Grouping.load(conn)
        f = _findings(conn, res.batch_id, grouping)
    res.blocked = sorted(seen_blocked.values(), key=lambda c: c.unrelated_id)
    # 2. rows on a sense that no longer exists
    for r in f.dead:
        _change(conn, r, "dead_sense", "", out, done)
    # 3. rows stored for falsely vetoed label pairs
    if false_vetoes:
        for r in _rows(conn):
            if r["batch_id"] == res.batch_id and r["decided_by"] == "agent" and r["state"] in LIVE \
                    and r["kind"] in ("merge", "link") \
                    and tuple(sorted((r["label_a"], r["label_b"]))) in false_vetoes:
                _change(conn, r, "false_veto", "", out, done)
    # 4. link rows inside one concept (no derived effect)
    for r in f.inside:
        if r["kind"] == "link":
            sides = _sides(r)
            name = grouping.names.get(grouping.concept_of[sides[0]], "") if sides else ""
            _change(conn, r, "inside_concept", name, out, done)
    # 5. conflicts among the rows still live (no grouping change since step 1)
    f = _findings(conn, res.batch_id, grouping)
    res.conflicts = f.conflicts
    for c in f.unsettled:
        for r in c.rows:
            _change(conn, r, "disagreement", f"{c.name_a} | {c.name_b}", out, done)
    res.after = _findings(conn, res.batch_id, _Grouping.load(conn)).counts()


def run(lib: Library, batch_id: str, *, apply: bool = False, false_vetoes: Iterable[Sequence[str]] = (),
        expect: Mapping[str, int] | None = None, now: str | None = None) -> CleanupResult:
    """Plan (``apply=False``: rolled back) or apply the cleanup of one batch, then refresh.
    ``expect`` maps rule -> count; any difference raises :class:`CleanupMismatch` before
    the commit, so nothing is written."""
    res = CleanupResult(batch_id, False, now or _now())
    pairs = _pair_keys(false_vetoes)
    try:
        with lib.write() as conn:
            _steps(conn, res, pairs)
            report = rf.plan(conn, now=res.at)
            rf.apply_plan(conn, report)
            res.refresh = report.report.summary()
            if expect is not None:
                counts = res.counts()
                diff = {k: (counts.get(k, 0), v) for k, v in expect.items() if counts.get(k, 0) != v}
                if diff:
                    raise CleanupMismatch("planned vs expected: " + ", ".join(
                        f"{k} {got} != {want}" for k, (got, want) in sorted(diff.items())))
            if not apply:
                raise _Rollback
    except _Rollback:
        return res
    res.applied = True
    return res


def record(lib: Library, res: CleanupResult) -> str:
    """Store an applied result in ``library_meta``; returns the key."""
    if not res.applied:
        raise ValueError("only an applied cleanup is recorded")
    key = f"{META_PREFIX}{res.batch_id}:{res.at}"
    with lib.write() as conn:
        conn.execute("INSERT OR REPLACE INTO library_meta (key, value) VALUES (?, ?)",
                     (key, json.dumps(res.as_dict(), sort_keys=True, ensure_ascii=False)))
    return key


def records(conn: sqlite3.Connection, batch_id: str | None = None) -> list[dict[str, Any]]:
    """The stored cleanup records (oldest first), each with its ``key``."""
    prefix = META_PREFIX + (f"{batch_id}:" if batch_id else "")
    out = []
    for key, value in conn.execute("SELECT key, value FROM library_meta WHERE key LIKE ? ORDER BY key",
                                   (prefix + "%",)):
        out.append({"key": key, **json.loads(value)})
    return out


__all__ = ["Change", "CleanupMismatch", "CleanupResult", "Conflict", "Contradiction", "Findings", "findings",
           "record", "records", "run"]
