"""Deterministic concept refresh (stage 4 design §1.4-§1.5, §2.4 migrated decisions).

:func:`plan` is a pure read: from ``current_claims``, the decisions, senses, sense
assignments and the library's ambiguous list it computes every derived row
(``claim_concepts``, ``concepts``, ``concept_members``, ``concept_links``) plus the few
stored-state updates a refresh makes:

- the seed never-merge rows and ambiguous labels, once;
- new **rule merges** (generic suffix fold and acronym fold, within a family, closure
  required, never on an ambiguous or split label), stored as ``decided_by = 'rule'``
  rows so Recent merges can show and split them;
- migrated ``pending_apply`` rows (stage 1 §6.10) settled: ``applied``, ``vetoed`` (with
  the reason, listed in the report; agent rows only) or left pending while a label is
  absent from the corpus; of two researcher rows on one pair that disagree, the newest
  wins and the other becomes ``undone``;
- sense assignments re-keyed onto the paper's current claims through
  ``notes.identity.match`` (so a lightly reworded re-note keeps its senses);
- ``concept_build_state`` and a cleared ``concept_refresh``.

:func:`refresh` applies a plan in one library transaction. Running it again is a no-op
on the derived tables (byte-stable). :func:`plan` also works on a pre-stage-4 library
opened read-only (missing tables read as empty, the seed lists stand in), which gives a
no-write planning pass.

Merges join senses with union-find in a fixed order (researcher, then rule, then agent;
by ``decided_at`` then id inside each), and every join is checked against the vetoes of
:mod:`.guardrails` over the whole resulting group, so a derived concept never holds a
never-merge pair, spans families, or (for rule and agent merges) has a stoplist
conflict.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..library.attention import hidden_paper_ids
from ..notes import identity
from . import guardrails as g
from .families import family_of, is_concept_family
from .model import DEFAULT_SENSE, SenseKey, claim_fingerprint, concept_id, decision_id, label_key, supports

if TYPE_CHECKING:
    from ..library.store import Library

#: Claims with these statuses carry no concept (the "nothing found" rows).
SKIP_STATUSES: frozenset[str] = frozenset({"not_found", "not_applicable"})
ACTIVE_SENSE_STATES: tuple[str, ...] = ("default", "confirmed")
RULE_ORIGIN = "rule"
MAKER_ORDER = {"researcher": 0, "rule": 1, "agent": 2}

DECISION_COLS = ("decision_id", "kind", "family", "family_b", "label_a", "sense_a", "label_b", "sense_b",
                 "name", "decided_by", "state", "note", "veto_reason", "origin", "decided_at")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- state ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Claim:
    claim_id: str
    paper_id: str
    family: str
    label_key: str
    surface: str
    supports: bool
    agent_visible: bool
    has_quote: bool
    fingerprint: str


@dataclass
class State:
    claims: list[Claim]
    paper_sha: dict[str, str]
    decisions: list[dict[str, Any]]
    senses: list[dict[str, Any]]
    assignments: list[dict[str, Any]]
    ambiguous: g.AmbiguousList
    build_state: dict[str, str]
    marked: set[str]
    seeded: bool


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (name,)).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}


def _rows(conn: sqlite3.Connection, table: str, cols: Iterable[str], order: str) -> list[dict[str, Any]]:
    """Rows of ``table`` with the wanted columns; a missing table or column reads as empty/NULL."""
    cols = tuple(cols)
    if not _table_exists(conn, table):
        return []
    have = _columns(conn, table)
    sel = ", ".join(c if c in have else f"NULL AS {c}" for c in cols)
    return [dict(zip(cols, r)) for r in conn.execute(f"SELECT {sel} FROM {table} ORDER BY {order}")]


def load_state(conn: sqlite3.Connection) -> State:
    hidden = hidden_paper_ids(conn)
    claims: list[Claim] = []
    per_paper: dict[str, list] = defaultdict(list)
    sql = (
        "SELECT cc.claim_id, cc.paper_id, cc.claim_type, cc.field_key, cc.normalized_label, cc.claim_text, "
        "cc.status, cc.trust, cc.kind, cc.span_id, n.version_label, e.quote_hash, e.markdown_hash, "
        "e.start_char, e.end_char FROM current_claims cc "
        "LEFT JOIN structured_notes n ON n.note_id = cc.note_id "
        "LEFT JOIN evidence_spans e ON e.span_id = cc.span_id ORDER BY cc.paper_id, cc.claim_id")
    cols = ("claim_id", "paper_id", "claim_type", "field_key", "normalized_label", "claim_text", "status", "trust",
            "kind", "span_id", "version_label", "quote_hash", "markdown_hash", "start_char", "end_char")
    for r in conn.execute(sql):
        row = dict(zip(cols, r))
        visible = row["paper_id"] not in hidden and row["version_label"] is None
        per_paper[row["paper_id"]].append(
            [row[c] for c in cols[:10]] + [int(visible)])
        family = family_of(row["claim_type"])
        key = label_key(row["normalized_label"])
        if not is_concept_family(family) or not key or row["status"] in SKIP_STATUSES:
            continue
        span = row if row["quote_hash"] is not None else None
        claims.append(Claim(
            claim_id=row["claim_id"], paper_id=row["paper_id"], family=family, label_key=key,  # type: ignore[arg-type]
            surface=" ".join((row["normalized_label"] or "").split()), supports=supports(row),
            agent_visible=visible, has_quote=row["span_id"] is not None,
            fingerprint=claim_fingerprint(row, span)))
    paper_sha = {p: hashlib.sha256(json.dumps(v, ensure_ascii=False).encode()).hexdigest()
                 for p, v in per_paper.items()}
    decisions = _rows(conn, "concept_decisions", DECISION_COLS, "decision_id")
    senses = _rows(conn, "concept_senses", ("family", "label_key", "sense_id", "gloss", "state"),
                   "family, label_key, sense_id")
    assignments = _rows(conn, "sense_assignments", ("family", "label_key", "paper_id", "claim_fingerprint",
                                                    "sense_id", "decided_by", "decided_at"),
                        "family, label_key, paper_id, claim_fingerprint")
    build = {r["paper_id"]: r["claims_sha256"] for r in
             _rows(conn, "concept_build_state", ("paper_id", "claims_sha256"), "paper_id")}
    marked = {r["paper_id"] for r in _rows(conn, "concept_refresh", ("paper_id",), "paper_id")}
    return State(claims, paper_sha, decisions, senses, assignments, g.effective_ambiguous(conn),
                 build, marked, g.seeds_written(conn))


# --- out of date ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class OutOfDate:
    changed: tuple[str, ...]   # papers whose current claims differ from the last refresh
    marked: tuple[str, ...]    # concept_refresh rows (overlay saves, merges, reassignments)

    @property
    def papers(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.changed) | set(self.marked)))

    @property
    def stale(self) -> bool:
        return bool(self.changed or self.marked)


def out_of_date(conn: sqlite3.Connection) -> OutOfDate:
    """Papers whose concepts are out of date (Home's "concepts out of date (n papers)")."""
    s = load_state(conn)
    papers = set(s.paper_sha) | set(s.build_state)
    changed = tuple(sorted(p for p in papers if s.paper_sha.get(p) != s.build_state.get(p)))
    return OutOfDate(changed, tuple(sorted(s.marked)))


# --- plan -----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionOutcome:
    decision_id: str
    kind: str
    decided_by: str
    label_a: str
    label_b: str
    family: str | None
    outcome: str                 # applied | vetoed | pending | undone | queued | blocked
    reason: str | None = None


@dataclass
class RefreshReport:
    claims: int = 0
    supporting_claims: int = 0
    needs_sense_claims: int = 0
    labels: int = 0
    senses: int = 0
    concepts: int = 0
    supported_concepts: int = 0
    unsupported_concepts: int = 0
    needs_sense_check_concepts: int = 0
    links: int = 0
    by_family: dict[str, int] = field(default_factory=dict)
    rule_merges_added: int = 0
    rule_merges_skipped: list[DecisionOutcome] = field(default_factory=list)
    migrated_applied: list[DecisionOutcome] = field(default_factory=list)
    migrated_vetoed: list[DecisionOutcome] = field(default_factory=list)
    migrated_pending: list[DecisionOutcome] = field(default_factory=list)
    superseded: list[DecisionOutcome] = field(default_factory=list)
    blocked: list[DecisionOutcome] = field(default_factory=list)
    reassigned: int = 0
    seeded: bool = False

    def summary(self) -> dict[str, Any]:
        """Counts only (no labels): safe to print for the real library."""
        return {
            "claims": self.claims, "supporting_claims": self.supporting_claims,
            "needs_sense_claims": self.needs_sense_claims, "labels": self.labels, "senses": self.senses,
            "concepts": self.concepts, "supported_concepts": self.supported_concepts,
            "unsupported_concepts": self.unsupported_concepts,
            "needs_sense_check_concepts": self.needs_sense_check_concepts, "links": self.links,
            "by_family": dict(sorted(self.by_family.items())), "rule_merges_added": self.rule_merges_added,
            "rule_merges_skipped": len(self.rule_merges_skipped),
            "migrated_applied": dict(Counter(f"{o.kind}:{o.decided_by}" for o in self.migrated_applied)),
            "migrated_vetoed": dict(Counter(f"{o.kind}:{o.decided_by}:{o.reason}" for o in self.migrated_vetoed)),
            "migrated_pending": dict(Counter(f"{o.kind}:{o.decided_by}" for o in self.migrated_pending)),
            "superseded": len(self.superseded), "blocked": len(self.blocked), "reassigned": self.reassigned,
            "seeded": self.seeded,
        }


@dataclass
class RefreshPlan:
    now: str
    claim_concepts: list[tuple] = field(default_factory=list)
    concepts: list[tuple] = field(default_factory=list)
    members: list[tuple] = field(default_factory=list)
    links: list[tuple] = field(default_factory=list)
    new_rule_decisions: list[dict[str, Any]] = field(default_factory=list)
    state_updates: list[tuple[str, str, str | None]] = field(default_factory=list)  # (id, state, reason)
    assignment_rewrites: list[tuple[dict[str, Any], str]] = field(default_factory=list)  # (old row, new fp)
    build_state: list[tuple[str, str]] = field(default_factory=list)
    seed: bool = False
    report: RefreshReport = field(default_factory=RefreshReport)


class _UnionFind:
    def __init__(self, keys: Iterable[SenseKey]) -> None:
        self.parent = {k: k for k in keys}
        self.members = {k: {k} for k in self.parent}

    def find(self, k: SenseKey) -> SenseKey:
        while self.parent[k] != k:
            self.parent[k] = self.parent[self.parent[k]]
            k = self.parent[k]
        return k

    def union(self, a: SenseKey, b: SenseKey) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        lo, hi = sorted((ra, rb))
        self.parent[hi] = lo
        self.members[lo] |= self.members.pop(hi)


def _seed_rows(state: State, now: str) -> list[dict[str, Any]]:
    """The seed never-merge rows a first refresh would write (for the in-memory plan)."""
    if state.seeded:
        return []
    have = {(d["label_a"], d["label_b"]) for d in state.decisions
            if d["family"] is None and d["sense_a"] is None and d["sense_b"] is None
            and d["decided_by"] == "researcher"}
    out = []
    for a, b in g.SEED_NEVER_MERGE:
        a, b = sorted((a, b))
        if (a, b) not in have:
            out.append(dict(decision_id=decision_id("never_merge", a, b), kind="never_merge", family=None,
                            family_b=None, label_a=a, sense_a=None, label_b=b, sense_b=None, name=None,
                            decided_by="researcher", state="applied", note=None, veto_reason=None,
                            origin=g.SEED_ORIGIN, decided_at=now))
    return out


def _supersede(decisions: list[dict[str, Any]]) -> dict[str, str]:
    """Researcher merge vs never_merge on one pair: the newest wins; the loser -> undone."""
    out: dict[str, str] = {}
    by_pair: dict[tuple, list[dict]] = defaultdict(list)
    for d in decisions:
        if d["decided_by"] == "researcher" and d["kind"] in ("merge", "never_merge") \
                and d["state"] in ("applied", "pending_apply"):
            by_pair[(d["family"], d["label_a"], d["sense_a"], d["label_b"], d["sense_b"])].append(d)
    for rows in by_pair.values():
        if len({d["kind"] for d in rows}) < 2:
            continue
        win = max(rows, key=lambda d: (d["decided_at"], d["decision_id"]))
        for d in rows:
            if d["kind"] != win["kind"]:
                out[d["decision_id"]] = win["decision_id"]
    return out


def _rule_candidates(labels: dict[str, set[str]], split: set[tuple[str, str]],
                     ambiguous: g.AmbiguousList) -> list[tuple[str, str, str, str]]:
    """``(family, label_a, label_b, note)``: suffix folds and acronym folds (closure holds by
    construction: both labels are used in the family)."""
    out = []
    for family, keys in sorted(labels.items()):
        ok = {k for k in keys if (family, k) not in split and not ambiguous.is_ambiguous(k)}
        for k in sorted(ok):
            toks = k.split()
            if len(toks) >= 2 and toks[-1] in g.SUFFIX_WORDS:
                base = " ".join(toks[:-1])
                if base in ok:
                    out.append((family, base, k, "rule: generic suffix fold"))
        singles = {k for k in ok if " " not in k and len(k) >= 2}
        for k in sorted(ok):
            toks = k.split()
            if len(toks) >= 2:
                initials = "".join(t[0] for t in toks)
                if initials in singles and initials != k:
                    lo, hi = sorted((initials, k))  # the schema orders a pair label_a < label_b
                    out.append((family, lo, hi, "rule: acronym fold"))
    return out


def plan(conn: sqlite3.Connection, *, now: str | None = None) -> RefreshPlan:
    """Compute a refresh without writing anything (see the module doc)."""
    now = now or _now()
    state = load_state(conn)
    p = RefreshPlan(now=now, seed=not state.seeded)
    rep = p.report
    rep.seeded = not state.seeded
    decisions = state.decisions + _seed_rows(state, now)

    # labels and senses
    labels: dict[str, set[str]] = defaultdict(set)
    for c in state.claims:
        labels[c.family].add(c.label_key)
    label_families: dict[str, set[str]] = defaultdict(set)
    for fam, keys in labels.items():
        for k in keys:
            label_families[k].add(fam)
    active = [s for s in state.senses if s["state"] in ACTIVE_SENSE_STATES]
    split_senses: dict[tuple[str, str], list[dict]] = defaultdict(list)
    confirmed: set[tuple[str, str]] = set()
    for s in active:
        if s["sense_id"] != DEFAULT_SENSE:
            split_senses[(s["family"], s["label_key"])].append(s)
        elif s["state"] == "confirmed":
            confirmed.add((s["family"], s["label_key"]))
    gloss = {(s["family"], s["label_key"], s["sense_id"]): s["gloss"] for s in active}
    universe: set[SenseKey] = set()
    for fam, keys in labels.items():
        for k in keys:
            if (fam, k) in split_senses:
                universe |= {SenseKey(fam, k, s["sense_id"]) for s in split_senses[(fam, k)]}
            else:
                universe.add(SenseKey(fam, k, DEFAULT_SENSE))

    # sense of every claim (split labels: re-match stored assignments onto current claims)
    claim_sense: dict[str, str | None] = {}
    by_label_paper: dict[tuple[str, str, str], list[Claim]] = defaultdict(list)
    for c in state.claims:
        if (c.family, c.label_key) in split_senses:
            by_label_paper[(c.family, c.label_key, c.paper_id)].append(c)
        else:
            claim_sense[c.claim_id] = DEFAULT_SENSE
    assigned: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for a in state.assignments:
        if (a["family"], a["label_key"]) in split_senses:
            assigned[(a["family"], a["label_key"], a["paper_id"])].append(a)
    for key in sorted(set(by_label_paper) | set(assigned)):
        claims = by_label_paper.get(key, [])
        olds = assigned.get(key, [])
        valid = {s["sense_id"] for s in split_senses[(key[0], key[1])]}
        old_fps = [identity.ClaimFingerprint.from_dict(a["claim_fingerprint"], json.loads(a["claim_fingerprint"]))
                   for a in olds]
        new_fps = [identity.ClaimFingerprint.from_dict(c.claim_id, json.loads(c.fingerprint)) for c in claims]
        matching = identity.match(old_fps, new_fps)
        old_by_fp = {a["claim_fingerprint"]: a for a in olds}
        fp_of = {c.claim_id: c.fingerprint for c in claims}
        for c in claims:
            claim_sense[c.claim_id] = None
        for pair in matching.pairs:
            a = old_by_fp[pair.old]
            if a["sense_id"] in valid:
                claim_sense[pair.new] = a["sense_id"]
            if fp_of[pair.new] != pair.old:
                p.assignment_rewrites.append((a, fp_of[pair.new]))
    rep.reassigned = len(p.assignment_rewrites)

    # vetoes
    superseded = _supersede(decisions)
    veto_rows = [d for d in decisions if d["kind"] in ("never_merge", "never_connect")
                 and d["state"] in ("applied", "pending_apply") and d["decision_id"] not in superseded]
    vetoes = g.VetoIndex(g.Veto(d["kind"], d["label_a"], d["label_b"], d["family"], d["family_b"] or d["family"],
                                d["sense_a"], d["sense_b"], d["decision_id"]) for d in veto_rows)

    # merges in a fixed order
    existing_ids = {d["decision_id"] for d in decisions}
    rule_rows = []
    for fam, a, b, note in _rule_candidates(labels, set(split_senses), state.ambiguous):
        did = decision_id("merge", a, b, family=fam)
        if did in existing_ids:
            continue
        existing_ids.add(did)
        rule_rows.append(dict(decision_id=did, kind="merge", family=fam, family_b=None, label_a=a, sense_a=None,
                              label_b=b, sense_b=None, name=None, decided_by="rule", state="applied", note=note,
                              veto_reason=None, origin=RULE_ORIGIN, decided_at=now, _new=True))
    merges = [d for d in decisions if d["kind"] == "merge" and d["state"] in ("applied", "pending_apply")
              and d["decision_id"] not in superseded] + rule_rows
    merges.sort(key=lambda d: (MAKER_ORDER[d["decided_by"]], d["decided_at"] if d["decided_by"] != "rule" else "",
                               d["decision_id"]))
    uf = _UnionFind(sorted(universe))
    applied_names: list[tuple[dict, frozenset]] = []

    def outcome(d: dict, result: str, reason: str | None = None) -> DecisionOutcome:
        return DecisionOutcome(d["decision_id"], d["kind"], d["decided_by"], d["label_a"], d["label_b"],
                               d["family"], result, reason)

    for d in merges:
        migrated = d["state"] == "pending_apply"
        fams = [d["family"]] if d["family"] else sorted(label_families[d["label_a"]] & label_families[d["label_b"]])
        reason = None
        if not d["family"] and not fams and d["decided_by"] == "agent"                 and label_families[d["label_a"]] and label_families[d["label_b"]]:
            reason = "type_fence"
        if reason is None and migrated and d["decided_by"] == "agent" and \
                (state.ambiguous.is_ambiguous(d["label_a"]) or state.ambiguous.is_ambiguous(d["label_b"])):
            reason = "ambiguous_label"
        joined = []
        if reason is None:
            for fam in fams:
                ka = SenseKey(fam, d["label_a"], d["sense_a"] or DEFAULT_SENSE)
                kb = SenseKey(fam, d["label_b"], d["sense_b"] or DEFAULT_SENSE)
                if ka not in universe or kb not in universe:
                    continue
                ra, rb = uf.find(ka), uf.find(kb)
                if ra != rb:
                    why = g.merge_veto(uf.members[ra], uf.members[rb], vetoes,
                                       stoplist=d["decided_by"] != "researcher")
                    if why is not None:
                        reason = reason or why
                        continue
                    uf.union(ka, kb)
                joined.append((ka, kb))
        if joined:
            applied_names.append((d, frozenset(k for pair in joined for k in pair)))
            if d.get("_new"):
                p.new_rule_decisions.append({k: v for k, v in d.items() if k != "_new"})
                rep.rule_merges_added += 1
            elif migrated:
                p.state_updates.append((d["decision_id"], "applied", None))
                rep.migrated_applied.append(outcome(d, "applied", reason))
            elif reason:
                rep.blocked.append(outcome(d, "blocked", reason))
            continue
        if d.get("_new"):
            if reason:
                rep.rule_merges_skipped.append(outcome(d, "blocked", reason))
            continue
        if reason is None:  # inert: a label is absent from the corpus (or its sense is)
            if migrated:
                rep.migrated_pending.append(outcome(d, "pending"))
            continue
        if migrated and d["decided_by"] == "agent":
            p.state_updates.append((d["decision_id"], "vetoed", reason))
            rep.migrated_vetoed.append(outcome(d, "vetoed", reason))
        elif migrated:  # researcher rows are never vetoed: held for the researcher's queue
            p.state_updates.append((d["decision_id"], "queued", reason))
            rep.blocked.append(outcome(d, "queued", reason))
        else:
            rep.blocked.append(outcome(d, "blocked", reason))
    for d in decisions:
        if d["decision_id"] in superseded:
            p.state_updates.append((d["decision_id"], "undone", f"superseded by {superseded[d['decision_id']]}"))
            rep.superseded.append(outcome(d, "undone", superseded[d["decision_id"]]))
        elif d["kind"] in ("never_merge", "never_connect") and d["state"] == "pending_apply":
            p.state_updates.append((d["decision_id"], "applied", None))
            rep.migrated_applied.append(outcome(d, "applied"))

    # concepts
    sense_claims: dict[SenseKey, list[Claim]] = defaultdict(list)
    for c in state.claims:
        s = claim_sense.get(c.claim_id)
        if s is not None:
            sense_claims[SenseKey(c.family, c.label_key, s)].append(c)
    n_papers = len({c.paper_id for c in state.claims if c.supports})
    surfaces: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for c in state.claims:
        surfaces[(c.family, c.label_key)][c.surface] += 1
    cid_of: dict[SenseKey, str] = {}
    groups = sorted(uf.members.items())
    named: dict[SenseKey, list[dict]] = defaultdict(list)
    for d, keys in applied_names:
        if d["name"]:
            named[uf.find(next(iter(keys)))].append(d)
    concept_rows = []
    for root, members in groups:
        smallest = min(members)
        cid = concept_id(root.family, smallest)
        for m in members:
            cid_of[m] = cid
        claims = [c for m in members for c in sense_claims.get(m, [])]
        sup = [c for c in claims if c.supports]
        papers = {c.paper_id for c in sup}
        member_labels = {m.label_key for m in members}
        name_key = None
        for d in sorted(named.get(root, ()), key=lambda d: (MAKER_ORDER[d["decided_by"]], _neg(d["decided_at"]),
                                                            d["decision_id"])):
            if g.closure_ok(d["name"], member_labels):
                name_key = label_key(d["name"])
                break
        if name_key is None:
            label_claims = Counter(c.label_key for c in claims)
            per_label = Counter(k for k, _ in {(c.label_key, c.paper_id) for c in sup})
            name_key = min(member_labels, key=lambda k: (-per_label[k], -label_claims[k], len(k), k))
        surf = surfaces.get((root.family, name_key))
        name = min(surf.items(), key=lambda kv: (-kv[1], kv[0]))[0] if surf else name_key
        glosses = [gloss[(m.family, m.label_key, m.sense_id)] for m in sorted(members)
                   if gloss.get((m.family, m.label_key, m.sense_id))]
        needs_check = any(m.sense_id == DEFAULT_SENSE and (m.family, m.label_key) not in confirmed
                          and state.ambiguous.is_ambiguous(m.label_key) for m in members)
        weight = round(math.log(n_papers / len(papers)), 6) if papers else None
        concept_rows.append((cid, root.family, name, glosses[0] if glosses else None, len(papers), len(claims),
                             sum(1 for c in sup if c.has_quote), int(bool(sup)), weight, int(needs_check)))
        rep.by_family[root.family] = rep.by_family.get(root.family, 0) + int(bool(sup))
    p.concepts = sorted(concept_rows)
    p.members = sorted((cid_of[m], m.family, m.label_key, m.sense_id) for m in universe)
    p.claim_concepts = sorted(
        (c.claim_id, c.paper_id, c.family, c.label_key, claim_sense.get(c.claim_id),
         cid_of.get(SenseKey(c.family, c.label_key, claim_sense[c.claim_id])) if claim_sense.get(c.claim_id)
         else None, int(c.supports), int(c.agent_visible)) for c in state.claims)

    # links
    link_rows = set()
    for d in decisions:
        if d["kind"] != "link" or d["state"] != "applied":
            continue
        if d["family"]:
            pairs = [(d["family"], d["family_b"] or d["family"])]
        else:
            pairs = [(f, f) for f in sorted(label_families[d["label_a"]] & label_families[d["label_b"]])]
        for fa, fb in pairs:
            ka = SenseKey(fa, d["label_a"], d["sense_a"] or DEFAULT_SENSE)
            kb = SenseKey(fb, d["label_b"], d["sense_b"] or DEFAULT_SENSE)
            if ka not in cid_of or kb not in cid_of or cid_of[ka] == cid_of[kb]:
                continue
            ra, rb = uf.find(ka), uf.find(kb)
            if g.link_veto(uf.members[ra], uf.members[rb], vetoes):
                rep.blocked.append(outcome(d, "blocked", "never_connect"))
                continue
            a, b = sorted((cid_of[ka], cid_of[kb]))
            link_rows.add((a, b, d["decision_id"], d["note"], d["decided_by"]))
    p.links = sorted(link_rows)

    p.build_state = sorted(state.paper_sha.items())
    rep.claims = len(state.claims)
    rep.supporting_claims = sum(1 for c in state.claims if c.supports)
    rep.needs_sense_claims = sum(1 for c in state.claims if claim_sense.get(c.claim_id) is None)
    rep.labels = sum(len(k) for k in labels.values())
    rep.senses = len(universe)
    rep.concepts = len(p.concepts)
    rep.supported_concepts = sum(1 for r in p.concepts if r[7])
    rep.unsupported_concepts = rep.concepts - rep.supported_concepts
    rep.needs_sense_check_concepts = sum(1 for r in p.concepts if r[9] and r[7])
    rep.links = len(p.links)
    return p


def _neg(text: str) -> tuple[int, ...]:
    """Sort key that orders ISO timestamps newest first."""
    return tuple(-ord(ch) for ch in text or "")


# --- apply ----------------------------------------------------------------------------------------

def apply_plan(conn: sqlite3.Connection, p: RefreshPlan) -> None:
    """Write a plan (caller's transaction)."""
    if p.seed:
        g.ensure_seeds(conn, p.now)
    for d in p.new_rule_decisions:
        cols = list(DECISION_COLS)
        conn.execute(f"INSERT INTO concept_decisions ({', '.join(cols)}) VALUES "
                     f"({', '.join('?' * len(cols))})", tuple(d[c] for c in cols))
    for did, st, reason in p.state_updates:
        conn.execute("UPDATE concept_decisions SET state = ?, veto_reason = ? WHERE decision_id = ?",
                     (st, reason, did))
    for old, new_fp in p.assignment_rewrites:
        conn.execute("DELETE FROM sense_assignments WHERE family = ? AND label_key = ? AND paper_id = ? "
                     "AND claim_fingerprint = ?", (old["family"], old["label_key"], old["paper_id"],
                                                   old["claim_fingerprint"]))
        conn.execute("INSERT OR REPLACE INTO sense_assignments (family, label_key, paper_id, claim_fingerprint, "
                     "sense_id, decided_by, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (old["family"], old["label_key"], old["paper_id"], new_fp, old["sense_id"],
                      old["decided_by"], old["decided_at"]))
    for table in ("claim_concepts", "concepts", "concept_members", "concept_links", "concept_build_state"):
        conn.execute(f"DELETE FROM {table}")
    conn.executemany("INSERT INTO claim_concepts (claim_id, paper_id, family, label_key, sense_id, concept_id, "
                     "supports, agent_visible) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", p.claim_concepts)
    conn.executemany("INSERT INTO concepts (concept_id, family, name, gloss, paper_count, claim_count, quote_count, "
                     "supported, weight, needs_sense_check) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", p.concepts)
    conn.executemany("INSERT INTO concept_members (concept_id, family, label_key, sense_id) VALUES (?, ?, ?, ?)",
                     p.members)
    conn.executemany("INSERT INTO concept_links (concept_a, concept_b, decision_id, note, decided_by) "
                     "VALUES (?, ?, ?, ?, ?)", p.links)
    conn.executemany("INSERT INTO concept_build_state (paper_id, claims_sha256, built_at) VALUES (?, ?, ?)",
                     [(pid, sha, p.now) for pid, sha in p.build_state])
    conn.execute("DELETE FROM concept_refresh")


def refresh(lib: Library, *, now: str | None = None) -> RefreshReport:
    """Recompute every derived concept table (one library transaction)."""
    with lib.write() as conn:
        p = plan(conn, now=now)
        apply_plan(conn, p)
    return p.report


def refresh_if_stale(lib: Library, *, now: str | None = None) -> RefreshReport | None:
    """What every concept view and task calls first."""
    if not out_of_date(lib.conn).stale and g.seeds_written(lib.conn):
        return None
    return refresh(lib, now=now)
