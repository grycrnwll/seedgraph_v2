"""Graph slice, literatures and bridges (stage 4 design §6.1, §6.3), computed on demand and
never stored. Shared by the web graph pages (researcher view) and the MCP concept tools
(agent audience).

**Support.** Counts use claims that *support* their sense (``claim_concepts.supports = 1``).
``agent_only=True`` (the MCP audience) also requires ``agent_visible = 1``: claims on a
paper with an open ``hides_notes`` item, or of a labelled note version, support nothing.
The researcher's pages pass ``agent_only=False``.

**Slice** (§6.1). The senses and concepts supported in a graph's member papers. Concepts
are summed from senses; the slice weight is ``log(N_members / slice_papers)``.

**Bridges** (§6.3). A concept with support in at least two literatures of the graph,
counting only senses that are confirmed or sit in one literature; or a closely related
link whose two concepts are in different literatures. A sense whose papers span
literatures without a sense check is listed apart as "same words, sense not checked"; an
ambiguous label's unchecked sense likewise (reason "ambiguous label"). Neither counts. A
graph slice uses the library ambiguous list plus that graph's overrides.

Nothing here writes or refreshes: callers refresh the derived tables first (MCP) or show
the "concepts out of date" banner (web GETs never write).
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..library import graphs
from ..library import literatures as lits_mod
from ..library.store import Library
from .model import SenseKey, handle, label_key

UNCHECKED = "same words, sense not checked"
AMBIGUOUS = "ambiguous label"
TOP_CONCEPTS = 5


def by(decided_by: str | None) -> str | None:
    """Who wrote a piece of text, as shown: researcher, rule or agent inferred."""
    if decided_by is None:
        return None
    return "researcher" if decided_by == "researcher" else ("rule" if decided_by == "rule" else "agent inferred")


def _visible_sql(agent_only: bool) -> str:
    return "supports = 1 AND agent_visible = 1" if agent_only else "supports = 1"


# --- support --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Claim:
    claim_id: str
    paper_id: str
    key: SenseKey
    concept_id: str | None


class Support:
    """The supporting claims (optionally inside a set of papers and families), by sense and
    by concept."""

    def __init__(self, conn: sqlite3.Connection, papers: set[str] | None = None,
                 families: Iterable[str] | None = None, *, agent_only: bool = True) -> None:
        fams = set(families) if families else None
        self.claims: dict[str, Claim] = {}
        self.by_sense: dict[SenseKey, list[Claim]] = defaultdict(list)
        self.by_concept: dict[str, list[Claim]] = defaultdict(list)
        for r in conn.execute(
                "SELECT claim_id, paper_id, family, label_key, sense_id, concept_id FROM claim_concepts "
                f"WHERE {_visible_sql(agent_only)} AND sense_id IS NOT NULL ORDER BY paper_id, claim_id"):
            if (papers is not None and r[1] not in papers) or (fams is not None and r[2] not in fams):
                continue
            c = Claim(r[0], r[1], SenseKey(r[2], r[3], r[4]), r[5])
            self.claims[c.claim_id] = c
            self.by_sense[c.key].append(c)
            if c.concept_id:
                self.by_concept[c.concept_id].append(c)

    def papers(self, claims: Iterable[Claim]) -> set[str]:
        return {c.paper_id for c in claims}

    def concept_papers(self) -> dict[str, set[str]]:
        return {cid: self.papers(cs) for cid, cs in self.by_concept.items()}


@dataclass(frozen=True)
class SenseMeta:
    gloss: str | None
    gloss_by: str | None
    state: str | None


def sense_meta(conn: sqlite3.Connection) -> dict[SenseKey, SenseMeta]:
    return {SenseKey(r[0], r[1], r[2]): SenseMeta(r[3], by(r[4]) if r[3] else None, r[5]) for r in conn.execute(
        "SELECT family, label_key, sense_id, gloss, decided_by, state FROM concept_senses")}


@dataclass(frozen=True)
class Concept:
    concept_id: str
    family: str
    name: str
    needs_sense_check: bool
    members: tuple[SenseKey, ...]

    @property
    def handle(self) -> str:
        named = [m for m in self.members if m.label_key == label_key(self.name)]
        return handle(min(named or self.members))


def concepts(conn: sqlite3.Connection, *, agent_only: bool = True) -> dict[str, Concept]:
    """The concepts as the audience sees them: only member senses with supporting claims
    (agent-visible ones for ``agent_only``) anywhere in the library, so a label known only
    from hidden notes never shows; a concept named by such a label is named by its best
    visible member."""
    papers: dict[SenseKey, int] = {SenseKey(r[0], r[1], r[2]): r[3] for r in conn.execute(
        "SELECT family, label_key, sense_id, COUNT(DISTINCT paper_id) FROM claim_concepts WHERE "
        f"{_visible_sql(agent_only)} AND sense_id IS NOT NULL GROUP BY family, label_key, sense_id")}
    members: dict[str, list[SenseKey]] = defaultdict(list)
    for r in conn.execute("SELECT concept_id, family, label_key, sense_id FROM concept_members"):
        key = SenseKey(r[1], r[2], r[3])
        if key in papers:
            members[r[0]].append(key)
    out = {}
    for r in conn.execute("SELECT concept_id, family, name, needs_sense_check FROM concepts"):
        visible = tuple(sorted(members.get(r[0], ())))
        if not visible:
            continue
        name = r[2]
        if label_key(name) not in {m.label_key for m in visible}:
            name = min(visible, key=lambda m: (-papers[m], m)).label_key
        out[r[0]] = Concept(r[0], r[1], name, bool(r[3]), visible)
    return out


class Checks:
    """Which senses count across literatures: a confirmed sense (a sense check or a split)
    counts; an unchecked default sense counts only inside one literature; an ambiguous
    label's unchecked sense counts for nothing across literatures. ``graph`` adds that
    graph's ambiguous overrides to the library list."""

    def __init__(self, conn: sqlite3.Connection, meta: dict[SenseKey, SenseMeta], graph: str | None = None) -> None:
        from . import guardrails

        self.meta = meta
        self.ambiguous = guardrails.effective_ambiguous(conn, graph)

    def checked(self, key: SenseKey) -> bool:
        m = self.meta.get(key)
        return m is not None and m.state == "confirmed"

    def is_ambiguous(self, key: SenseKey) -> bool:
        return self.ambiguous.is_ambiguous(key.label_key)

    def status(self, key: SenseKey, lit_count: int) -> str | None:
        """``None`` when the sense counts, else why it does not (only matters across literatures)."""
        if self.checked(key):
            return None
        if self.is_ambiguous(key):
            return AMBIGUOUS
        return UNCHECKED if lit_count >= 2 else None


# --- the slice ------------------------------------------------------------------------------------

@dataclass
class Slice:
    graph: str
    run: Any
    lit_of: dict[str, Any]
    literatures: list[Any]
    members: set[str]
    support: Support
    concepts: dict[str, Concept]
    checks: Checks
    meta: dict[SenseKey, SenseMeta]


def literature_map(lib: Library, graph: str) -> tuple[Any, dict[str, Any], set[str]]:
    """``(run, {paper_id: Literature}, members)`` of the graph's latest grouping."""
    with graphs.open_graph(lib, graph) as g:
        run = lits_mod.latest_run(g)
        lits = lits_mod.literatures(g) if run else []
        members = {p.paper_id for p in graphs.membership(g, status="member")}
    return run, {pid: lit for lit in lits for pid in lit.members}, members


def load_slice(lib: Library, graph: str, *, agent_only: bool = True) -> Slice:
    """The graph's slice as the derived concept tables stand (no refresh)."""
    conn = lib.conn
    run, lit_of, members = literature_map(lib, graph)
    meta = sense_meta(conn)
    lits = sorted({lit.literature_id: lit for lit in lit_of.values()}.values(), key=lambda x: x.ordinal)
    return Slice(graph, run, lit_of, lits, members, Support(conn, members, agent_only=agent_only),
                 concepts(conn, agent_only=agent_only), Checks(conn, meta, graph), meta)


def counted_literatures(s: Slice, concept: Concept) -> tuple[set[str], list[tuple[SenseKey, set[str], str]]]:
    """The literatures a concept counts in, and its senses that do not count (with why)."""
    counted: set[str] = set()
    excluded = []
    for key in concept.members:
        lits = {s.lit_of[c.paper_id].literature_id for c in s.support.by_sense.get(key, []) if c.paper_id in s.lit_of}
        if not lits:
            continue
        why = s.checks.status(key, len(lits))
        if why is None:
            counted |= lits
        elif len(lits) >= 2:
            excluded.append((key, lits, why))
        else:
            counted |= lits
    return counted, excluded


def literature_names(s: Slice, ids: Iterable[str]) -> list[str]:
    by_id = {lit.literature_id: lit for lit in s.literatures}
    return [by_id[i].name for i in sorted(ids, key=lambda i: by_id[i].ordinal)]


def bridges(conn: sqlite3.Connection, s: Slice) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(bridges, unchecked)``: bridges ranked by literatures spanned, then papers, then name."""
    found: list[tuple[tuple, dict[str, Any]]] = []
    unchecked: list[dict[str, Any]] = []
    counted_of: dict[str, set[str]] = {}
    for cid in s.support.by_concept:
        concept = s.concepts.get(cid)
        if concept is None:
            continue
        counted, excluded = counted_literatures(s, concept)
        counted_of[cid] = counted
        for key, lits, why in excluded:
            unchecked.append({"handle": handle(key), "name": concept.name, "family": key.family, "reason": why,
                              "literatures": literature_names(s, lits),
                              "papers": len(s.support.papers(s.support.by_sense[key]))})
        if len(counted) >= 2:
            papers = len(s.support.papers(s.support.by_concept[cid]))
            found.append(((-len(counted), -papers, concept.name), {
                "kind": "concept", "handle": concept.handle, "name": concept.name, "family": concept.family,
                "literature_count": len(counted), "literatures": literature_names(s, counted), "papers": papers}))
    for r in conn.execute("SELECT concept_a, concept_b, note, decided_by FROM concept_links ORDER BY decision_id"):
        la, lb = counted_of.get(r[0], set()), counted_of.get(r[1], set())
        if not la or not lb or len(la | lb) < 2 or (la == lb and len(la) == 1):
            continue
        ca, cb = s.concepts[r[0]], s.concepts[r[1]]
        found.append(((-len(la | lb), 0, ca.name), {
            "kind": "link", "a": {"handle": ca.handle, "name": ca.name, "literatures": literature_names(s, la)},
            "b": {"handle": cb.handle, "name": cb.name, "literatures": literature_names(s, lb)},
            "note": r[2], "note_by": by(r[3]) if r[2] else None, "literature_count": len(la | lb)}))
    found.sort(key=lambda t: t[0])
    unchecked.sort(key=lambda u: (-len(u["literatures"]), u["handle"]))
    return [b for _, b in found], unchecked


def top_concepts(s: Slice, lit: Any, limit: int = TOP_CONCEPTS) -> list[dict[str, Any]]:
    """A literature's most distinctive concepts (slice tf-idf)."""
    members = set(lit.members)
    n = max(1, len({p for p in s.lit_of}))
    scored = []
    for cid, ps in s.support.concept_papers().items():
        tf = len(ps & members)
        if tf and cid in s.concepts:
            scored.append((-(tf * math.log(1 + n / len(ps))), s.concepts[cid].name, cid, tf))
    return [{"handle": s.concepts[cid].handle, "name": name, "papers": tf}
            for _, name, cid, tf in sorted(scored)[:limit]]


# --- the slice concepts (web graph pages) ---------------------------------------------------------

@dataclass(frozen=True)
class SliceConcept:
    concept_id: str
    handle: str
    family: str
    name: str
    papers: int
    claims: int
    weight: float  # log(N_members / slice papers)
    literature_count: int  # literatures it counts in (unchecked senses across literatures count nowhere)
    main_literature: str | None  # the literature where it is most supported (counted senses), or None
    is_bridge: bool
    glosses: tuple[tuple[str, str | None, str | None], ...]  # (sense handle, gloss, gloss_by)
    paper_ids: tuple[str, ...]

    @property
    def score(self) -> float:
        return self.weight * self.papers


def slice_concepts(s: Slice) -> list[SliceConcept]:
    """Every concept supported inside the graph's members, by family then papers desc."""
    n = max(1, len(s.members))
    ordinal = {lit.literature_id: lit.ordinal for lit in s.literatures}
    out = []
    for cid, claims in s.support.by_concept.items():
        concept = s.concepts.get(cid)
        if concept is None:
            continue
        ps = s.support.papers(claims)
        counted, _ = counted_literatures(s, concept)
        main = None
        if counted:
            tally: dict[str, int] = defaultdict(int)
            for pid in ps:
                lit = s.lit_of.get(pid)
                if lit is not None and lit.literature_id in counted:
                    tally[lit.literature_id] += 1
            main = min(tally, key=lambda lid: (-tally[lid], ordinal.get(lid, 0), lid)) if tally else None
        glosses = tuple((handle(k), s.meta[k].gloss if k in s.meta else None,
                         s.meta[k].gloss_by if k in s.meta else None)
                        for k in concept.members if s.support.by_sense.get(k))
        out.append(SliceConcept(cid, concept.handle, concept.family, concept.name, len(ps), len(claims),
                                math.log(n / len(ps)), len(counted), main, len(counted) >= 2, glosses,
                                tuple(sorted(ps))))
    out.sort(key=lambda c: (c.family, -c.papers, -c.claims, c.name, c.concept_id))
    return out


def linked_pairs(conn: sqlite3.Connection, concept_ids: Iterable[str]) -> list[tuple[str, str]]:
    """Closely related links with both ends in ``concept_ids`` (sorted pairs)."""
    ids = set(concept_ids)
    return sorted({tuple(sorted((r[0], r[1]))) for r in conn.execute("SELECT concept_a, concept_b FROM concept_links")
                   if r[0] in ids and r[1] in ids})  # type: ignore[misc]


def counted_senses(s: Slice, concept: Concept, *, confirmed_only: bool) -> list[SenseKey]:
    """A concept's senses that may connect it beyond one literature: confirmed senses
    (``confirmed_only``), else also unchecked senses of labels that are not ambiguous."""
    out = []
    for key in concept.members:
        if s.checks.checked(key) or (not confirmed_only and not s.checks.is_ambiguous(key)):
            out.append(key)
    return out
