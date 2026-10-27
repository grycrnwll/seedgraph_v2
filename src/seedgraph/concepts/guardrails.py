"""Concept guardrails (stage 4 design §2.2-§2.3): hard vetoes on merges and links, the
seed never-merge list, and the ambiguous-label list (library-wide plus per-graph
overrides).

Vetoes, checked on every merge (agent proposals at save; refresh re-checks while it
builds the groups, so no derived concept ever violates them):

1. **type fence**: a merge whose senses are in different families is refused;
2. **never-merge list**: no merge across a ``never_merge`` pair and no merge or link
   across a ``never_connect`` pair, checked **transitively** over the whole resulting
   group (a merge that would put a vetoed pair into one concept through other members
   is refused);
3. **ambiguous labels** (:class:`AmbiguousList`): no rule merges; agent work must compare
   example claims (the concept tasks enforce that);
4. **closure**: a concept name must be one of its member labels (:func:`closure_ok`);
5. **stoplist**: v2's discriminating tokens (versions, sizes, numerals, static/dynamic),
   compared over the whole resulting group (:func:`discriminating_conflict`).

v2's token-overlap precondition is **not** a veto here (it would block every merge across
notation); lexical overlap only generates candidates.

Moved from ``semantic/canon.py`` (``DISCRIMINATING_TOKENS``, ``discriminating_conflict``);
``canon.py`` stays until slice C10 deletes it.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..semantic.merge import normalize
from .model import SenseKey, decision_id, label_key

if TYPE_CHECKING:
    from ..library.store import Library

#: Frozen discriminating-token stoplist (v2 ``canon.DISCRIMINATING_TOKENS``). Compared
#: after ``merge.normalize``, which strips dots, so the bare numerals are listed.
DISCRIMINATING_TOKENS: frozenset[str] = frozenset({
    "v1", "v2", "v3", "v4", "base", "large", "small", "xl", "xxl", "mini", "tiny", "encoder", "decoder",
    "cased", "uncased", "static", "dynamic", "1", "2", "3", "4", "0", "11", "i", "ii", "iii",
})

#: Generic words: the rule-merge suffixes, and the words a ``head`` pattern ignores.
SUFFIX_WORDS: tuple[str, ...] = ("assumption", "condition", "restriction", "hypothesis")
GENERIC_WORDS: frozenset[str] = frozenset({
    *SUFFIX_WORDS, "assumptions", "conditions", "restrictions", "hypotheses", "limitation", "limitations",
    "analysis", "test", "tests", "testing", "bias", "problem", "problems", "issue", "issues", "concern",
    "concerns", "check", "checks", "property", "properties",
})

#: Seed never-merge rows (``decided_by = 'researcher'``, ``origin = 'seed'``): textbook
#: pairs that may be linked, never merged. Expanded forms and spelling variants are seeded
#: too, so a rule fold onto the acronym cannot sidestep the pair.
SEED_NEVER_MERGE: tuple[tuple[str, str], ...] = (
    ("ate", "att"),
    ("average treatment effect", "average treatment effect on the treated"),
    ("ddd", "did"),
    ("difference in differences", "difference in difference in differences"),
    ("difference in differences", "triple differences"),
    ("heterogeneity", "heteroskedasticity"),
    ("heterogeneity", "heteroscedasticity"),
)

#: Seed ambiguous labels (generic econometrics vocabulary; editable).
SEED_AMBIGUOUS: tuple[tuple[str, str], ...] = (
    ("identification", "head"), ("efficiency", "head"), ("consistency", "head"), ("structural", "head"),
    ("robustness", "head"), ("heterogeneity", "head"), ("endogeneity", "head"), ("exogeneity", "head"),
    ("independence", "head"), ("selection", "head"), ("stationarity", "head"),
    ("identification limitation", "label"),
)
SEED_ORIGIN = "seed"
#: ``library_meta`` key set once the seeds are written (so a researcher delete sticks).
SEED_META_KEY = "concept_seeds"
SEED_VERSION = "1"
LIBRARY_SCOPE = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- stoplist and closure -------------------------------------------------------------------------

def discriminating_conflict(labels: Iterable[str]) -> bool:
    """True when a stoplist token is present in some of ``labels`` but not all."""
    token_sets = [set(normalize(m).split()) for m in set(labels)]
    if len(token_sets) < 2:
        return False
    differing = set().union(*token_sets) - set(token_sets[0]).intersection(*token_sets[1:])
    return any(tok in DISCRIMINATING_TOKENS for tok in differing)


#: How a refusal names a stoplist token (the rest are "variant" words).
VERSION_TOKENS: frozenset[str] = frozenset({"v1", "v2", "v3", "v4", "1", "2", "3", "4", "0", "11", "i", "ii", "iii"})
SIZE_TOKENS: frozenset[str] = frozenset({"base", "large", "small", "xl", "xxl", "mini", "tiny"})


def stoplist_triggers(labels_a: Iterable[str], labels_b: Iterable[str]) -> tuple[list[str], list[str]]:
    """The stoplist tokens behind a :func:`discriminating_conflict` of the two groups'
    labels together, as found on each side (sorted; a side can have none)."""
    sides = [[set(normalize(m).split()) for m in set(side)] for side in (labels_a, labels_b)]
    every = sides[0] + sides[1]
    if len(every) < 2:
        return [], []
    differing = (set().union(*every) - set(every[0]).intersection(*every[1:])) & DISCRIMINATING_TOKENS
    found_a, found_b = (sorted(differing & set().union(*side)) for side in sides)
    return found_a, found_b


def stoplist_kind(tokens: Iterable[str]) -> str:
    """``version``, ``size`` or ``variant`` (or ``version or size`` for a mix)."""
    kinds = {"version" if t in VERSION_TOKENS else "size" if t in SIZE_TOKENS else "variant" for t in tokens}
    return kinds.pop() if len(kinds) == 1 else "version or size"


def closure_ok(name: str, member_labels: Iterable[str]) -> bool:
    """A concept name must be one of its member labels (compared as label keys)."""
    return label_key(name) in {label_key(m) for m in member_labels}


# --- ambiguous labels -----------------------------------------------------------------------------

def content_tokens(key: str) -> tuple[str, ...]:
    return tuple(t for t in key.split() if t not in GENERIC_WORDS)


def pattern_matches(key: str, pattern: str, match: str) -> bool:
    """``label``: an exact label key. ``head``: the label's content tokens (generic words
    removed) are exactly the pattern's ("exogeneity assumption" matches the head
    ``exogeneity``; "strict exogeneity" does not)."""
    if match == "label":
        return key == pattern
    return bool(content_tokens(key)) and content_tokens(key) == content_tokens(pattern)


@dataclass(frozen=True)
class AmbiguousPattern:
    pattern: str
    match: str
    scope: str = LIBRARY_SCOPE
    note: str | None = None
    decided_by: str = "researcher"
    origin: str = SEED_ORIGIN


@dataclass(frozen=True)
class AmbiguousList:
    """The effective list for one scope: library patterns, plus a graph's added patterns,
    minus the library patterns that graph marked single-sense."""

    patterns: tuple[AmbiguousPattern, ...]
    graph: str | None = None

    def __post_init__(self) -> None:
        labels: dict[str, AmbiguousPattern] = {}
        heads: dict[tuple[str, ...], AmbiguousPattern] = {}
        for p in self.patterns:
            if p.match == "label":
                labels.setdefault(p.pattern, p)
            elif content_tokens(p.pattern):
                heads.setdefault(content_tokens(p.pattern), p)
        object.__setattr__(self, "_labels", labels)
        object.__setattr__(self, "_heads", heads)

    def matches(self, key: str) -> AmbiguousPattern | None:
        """The matching pattern (a ``label`` pattern before a ``head`` one), else ``None``."""
        hit = self._labels.get(key)  # type: ignore[attr-defined]
        if hit is not None:
            return hit
        tokens = content_tokens(key)
        return self._heads.get(tokens) if tokens else None  # type: ignore[attr-defined]

    def is_ambiguous(self, key: str) -> bool:
        return self.matches(key) is not None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone() \
        is not None


def seeds_written(conn: sqlite3.Connection) -> bool:
    if not _table_exists(conn, "library_meta"):
        return False
    row = conn.execute("SELECT value FROM library_meta WHERE key = ?", (SEED_META_KEY,)).fetchone()
    return row is not None


def _seed_patterns() -> list[AmbiguousPattern]:
    return [AmbiguousPattern(p, m) for p, m in SEED_AMBIGUOUS]


def effective_ambiguous(conn: sqlite3.Connection, graph: str | None = None) -> AmbiguousList:
    """The ambiguous list for a scope. ``graph=None``: the library list (library-wide work:
    rule merges, the library refresh, library candidates). ``graph=slug``: library +
    that graph's overrides (graph-slice views and suggestions). Before the seeds are
    written (or on a pre-stage-4 library) the seed list stands in."""
    rows: list[AmbiguousPattern] = []
    if _table_exists(conn, "ambiguous_labels"):
        rows = [AmbiguousPattern(r[0], r[1], r[2], r[3], r[4], r[5]) for r in conn.execute(
            "SELECT pattern, match, scope, note, decided_by, origin FROM ambiguous_labels "
            "WHERE scope = ? AND effect = 'ambiguous' ORDER BY pattern", (LIBRARY_SCOPE,))]
    if not seeds_written(conn):
        have = {p.pattern for p in rows}
        rows += [p for p in _seed_patterns() if p.pattern not in have]
    if graph is not None and _table_exists(conn, "ambiguous_labels"):
        overrides = conn.execute(
            "SELECT pattern, match, effect, note, decided_by, origin FROM ambiguous_labels WHERE scope = ? "
            "ORDER BY pattern", (graph,)).fetchall()
        single = {r[0] for r in overrides if r[2] == "single_sense"}
        rows = [p for p in rows if p.pattern not in single]
        have = {p.pattern for p in rows}
        rows += [AmbiguousPattern(r[0], r[1], graph, r[3], r[4], r[5]) for r in overrides
                 if r[2] == "ambiguous" and r[0] not in have]
    return AmbiguousList(tuple(sorted(rows, key=lambda p: (p.pattern, p.scope))), graph)


def _norm_pattern(pattern: str) -> str:
    key = label_key(pattern)
    if not key:
        raise ValueError("an ambiguous-label pattern needs at least one letter or digit")
    return key


def add_ambiguous(lib: Library, pattern: str, *, match: str = "label", graph: str | None = None,
                  note: str | None = None, decided_by: str = "researcher", now: str | None = None) -> str:
    """Add a pattern to the library list (``graph=None``) or to one graph's overrides.
    Re-adding replaces the row (so a graph can turn a single-sense mark back into an
    addition). Returns the stored pattern (a label key)."""
    if match not in ("label", "head"):
        raise ValueError("match must be 'label' or 'head'")
    key = _norm_pattern(pattern)
    with lib.write() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO ambiguous_labels (pattern, scope, match, effect, note, decided_by, origin, "
            "created_at) VALUES (?, ?, ?, 'ambiguous', ?, ?, ?, ?)",
            (key, graph or LIBRARY_SCOPE, match, note, decided_by, decided_by, now or _now()))
    return key


def mark_single_sense(lib: Library, pattern: str, graph: str, *, note: str | None = None,
                      decided_by: str = "researcher", now: str | None = None) -> str:
    """Within ``graph``, treat a library-ambiguous pattern as single-sense."""
    if not graph:
        raise ValueError("single-sense marks are per graph; remove the library pattern instead")
    key = _norm_pattern(pattern)
    with lib.write() as conn:
        match = conn.execute("SELECT match FROM ambiguous_labels WHERE pattern = ? AND scope = ?",
                             (key, LIBRARY_SCOPE)).fetchone()
        conn.execute(
            "INSERT OR REPLACE INTO ambiguous_labels (pattern, scope, match, effect, note, decided_by, origin, "
            "created_at) VALUES (?, ?, ?, 'single_sense', ?, ?, ?, ?)",
            (key, graph, match[0] if match else "label", note, decided_by, decided_by, now or _now()))
    return key


def remove_ambiguous(lib: Library, pattern: str, *, graph: str | None = None) -> bool:
    """Delete one row of the library list or of a graph's overrides."""
    key = _norm_pattern(pattern)
    with lib.write() as conn:
        cur = conn.execute("DELETE FROM ambiguous_labels WHERE pattern = ? AND scope = ?",
                           (key, graph or LIBRARY_SCOPE))
    return cur.rowcount > 0


# --- seeds ----------------------------------------------------------------------------------------

def ensure_seeds(conn: sqlite3.Connection, now: str) -> bool:
    """Write the seed never-merge rows and ambiguous labels once (caller's transaction).
    A pair the researcher already decided keeps that decision. Returns whether anything was seeded."""
    if seeds_written(conn):
        return False
    for a, b in SEED_NEVER_MERGE:
        a, b = sorted((a, b))
        if conn.execute("SELECT 1 FROM concept_decisions WHERE label_a = ? AND label_b = ? AND family IS NULL "
                        "AND sense_a IS NULL AND sense_b IS NULL AND decided_by = 'researcher'", (a, b)).fetchone():
            continue
        conn.execute(
            "INSERT INTO concept_decisions (decision_id, kind, label_a, label_b, decided_by, state, note, origin, "
            "decided_at) VALUES (?, 'never_merge', ?, ?, 'researcher', 'applied', ?, ?, ?)",
            (decision_id("never_merge", a, b), a, b, "seed: may be linked, never merged", SEED_ORIGIN, now))
    for p, m in SEED_AMBIGUOUS:
        conn.execute(
            "INSERT OR IGNORE INTO ambiguous_labels (pattern, scope, match, effect, note, decided_by, origin, "
            "created_at) VALUES (?, ?, ?, 'ambiguous', NULL, 'researcher', ?, ?)",
            (p, LIBRARY_SCOPE, m, SEED_ORIGIN, now))
    conn.execute("INSERT OR REPLACE INTO library_meta (key, value) VALUES (?, ?)", (SEED_META_KEY, SEED_VERSION))
    return True


# --- vetoes ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Veto:
    """One ``never_merge`` / ``never_connect`` row. ``None`` family = every family;
    ``None`` sense = every sense of the label (conservative: a later split keeps it)."""

    kind: str
    label_a: str
    label_b: str
    family_a: str | None = None
    family_b: str | None = None
    sense_a: str | None = None
    sense_b: str | None = None
    decision_id: str = ""


def _side_matches(key: SenseKey, label: str, family: str | None, sense: str | None) -> bool:
    return key.label_key == label and (family is None or key.family == family) and \
        (sense is None or key.sense_id == sense)


class VetoIndex:
    """Veto rows indexed by label, for the transitive group checks."""

    def __init__(self, vetoes: Iterable[Veto]) -> None:
        self._by_label: dict[str, list[tuple[Veto, bool]]] = defaultdict(list)
        for v in vetoes:
            self._by_label[v.label_a].append((v, True))
            if v.label_b != v.label_a:
                self._by_label[v.label_b].append((v, False))

    def between(self, group_a: Iterable[SenseKey], group_b: Iterable[SenseKey],
                kinds: tuple[str, ...] = ("never_merge", "never_connect")) -> Veto | None:
        """The first veto (of ``kinds``) with one side in each group, else ``None``."""
        group_b = list(group_b)
        b_labels = {k.label_key for k in group_b}
        for x in sorted(group_a):
            for v, x_is_a in self._by_label.get(x.label_key, ()):
                if v.kind not in kinds:
                    continue
                mine = (v.label_a, v.family_a, v.sense_a) if x_is_a else (v.label_b, v.family_b, v.sense_b)
                other = (v.label_b, v.family_b, v.sense_b) if x_is_a else (v.label_a, v.family_a, v.sense_a)
                if other[0] not in b_labels or not _side_matches(x, *mine):
                    continue
                if any(_side_matches(y, *other) for y in group_b):
                    return v
        return None


def merge_veto(group_a: Iterable[SenseKey], group_b: Iterable[SenseKey], vetoes: VetoIndex, *,
               stoplist: bool = True) -> str | None:
    """Why joining two groups (each one concept's member senses) is refused, or ``None``.
    Reasons: ``type_fence``, ``never_merge``, ``never_connect``, ``stoplist``."""
    group_a, group_b = list(group_a), list(group_b)
    if {k.family for k in group_a} | {k.family for k in group_b} != {group_a[0].family}:
        return "type_fence"
    v = vetoes.between(group_a, group_b)
    if v is not None:
        return v.kind
    if stoplist and discriminating_conflict([k.label_key for k in (*group_a, *group_b)]):
        return "stoplist"
    return None


def link_veto(group_a: Iterable[SenseKey], group_b: Iterable[SenseKey], vetoes: VetoIndex) -> str | None:
    """A link is refused only across a ``never_connect`` pair."""
    v = vetoes.between(group_a, group_b, kinds=("never_connect",))
    return v.kind if v is not None else None
