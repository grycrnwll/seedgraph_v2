"""Concept read tools for the v3 MCP server (stage 4 design §9): thin adapters over
:mod:`seedgraph.concepts` and :mod:`seedgraph.library.literatures`.

- ``concept_search(query, family?, graph?, k=20)`` - sense-level hits (handle, name, gloss,
  counts, the best example claim with its quote) and the ``search_mode`` marker.
- ``concept_show(handle, graph?, claims_offset, claims_limit)`` - a concept's members and
  senses, supporting claims with refs and quotes, merges, closely related links with their
  notes and quotes, literatures (with ``graph``) and often-appears-with. A split label
  answers with a sense chooser.
- ``literatures(graph)`` - the graph's literatures (name, who named it, size, top
  concepts), bridges across them, and the "same words, sense not checked" list.
- ``concept_hubs(graph?, family?, limit=30)`` - the most common supported concepts per
  family (and, with a graph, the top bridges): agent orientation. Also the resource
  ``seedgraph://graphs/{slug}/hubs``.

**Agent audience.** Every count, claim, example and literature spread uses only claims that
*support* their sense and are *agent visible* (``claim_concepts.supports = 1 AND
agent_visible = 1``): claims on a paper with an open ``hides_notes`` item, or of a labelled
note version, support nothing here. A concept with no such claim is not shown.

**Labels.** Text written by an agent (glosses, link notes, literature names) carries
"agent inferred"; researcher text carries "researcher". Researcher decisions always win.

**Bridges** (§6.3, computed, never stored): a concept with support in at least two
literatures, counting only senses that are confirmed or sit in one literature; or a closely
related link whose two concepts are in different literatures. A sense whose papers span
literatures without a sense check (or an ambiguous label's unchecked default sense) is
listed apart as "same words, sense not checked" and counts for nothing.

Every view refreshes the derived concept tables first when they are stale.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .server import READ, tool_error

if TYPE_CHECKING:
    import sqlite3

    from mcp.server.fastmcp import FastMCP

    from ..concepts.model import SenseKey
    from ..library.store import Library
    from .server import V3Context

MAX_K = 100
MAX_CLAIMS = 100
MAX_HUBS = 200
TOP_CONCEPTS = 5
OFTEN_WITH = 10
HUB_BRIDGES = 10
UNCHECKED = "same words, sense not checked"
NO_RUN_ACTION = ("This graph has no literature grouping yet. Ask the researcher to group its literatures on "
                 "the graph page; do not infer literatures yourself.")
CONCEPT_NOT_FOUND_ACTION = "Call concept_search for concept handles (family/label or family/label~sense)."
TRUST_LABELS = {"paper_states": "paper states", "researcher": "researcher", "agent_inferred": "agent inferred"}
AGENT_RULES = (
    "Concept identity is label + sense: the same words in two literatures can mean different things.",
    f"Never treat a concept marked \"{UNCHECKED}\" or an ambiguous label as evidence of a cross-literature "
    "connection.",
    "A merge joins true synonyms in one sense; a closely related link keeps both concepts and its note records "
    "the difference or the history.",
    "Researcher decisions always win over agent decisions.",
    "Say \"agent inferred\" when you use an agent's gloss, link note or literature name.",
    "Cite claims by paper and quote; read the passage (paper_read) before relying on it.",
)


def _by(decided_by: str | None) -> str | None:
    if decided_by is None:
        return None
    return "researcher" if decided_by == "researcher" else ("rule" if decided_by == "rule" else "agent inferred")


def _bounded(value: int, name: str, lower: int, upper: int) -> int:
    if not lower <= value <= upper:
        tool_error("invalid_argument", f"{name} must be between {lower} and {upper}, not {value}")
    return value


def _family(family: str | None) -> str | None:
    from ..concepts.families import CONCEPT_FAMILIES

    if family is not None and family not in CONCEPT_FAMILIES:
        tool_error("invalid_argument", f"unknown family {family!r}; use one of {', '.join(CONCEPT_FAMILIES)}")
    return family


def _prepare(lib: Library) -> None:
    from ..concepts import refresh as rf

    rf.refresh_if_stale(lib)


def _members_of_graph(lib: Library, graph: str) -> set[str]:
    from ..library import graphs

    with graphs.open_graph(lib, graph) as g:
        return {p.paper_id for p in graphs.membership(g, status="member")}


# --- the agent-visible support ------------------------------------------------------------------

@dataclass(frozen=True)
class _Claim:
    claim_id: str
    paper_id: str
    key: SenseKey
    concept_id: str | None


class _Support:
    """The supporting, agent-visible claims (optionally inside a set of papers), by sense and
    by concept."""

    def __init__(self, conn: sqlite3.Connection, papers: set[str] | None = None,
                 families: Iterable[str] | None = None) -> None:
        from ..concepts.model import SenseKey

        fams = set(families) if families else None
        self.claims: dict[str, _Claim] = {}
        self.by_sense: dict[SenseKey, list[_Claim]] = defaultdict(list)
        self.by_concept: dict[str, list[_Claim]] = defaultdict(list)
        for r in conn.execute(
                "SELECT claim_id, paper_id, family, label_key, sense_id, concept_id FROM claim_concepts "
                "WHERE supports = 1 AND agent_visible = 1 AND sense_id IS NOT NULL ORDER BY paper_id, claim_id"):
            if (papers is not None and r[1] not in papers) or (fams is not None and r[2] not in fams):
                continue
            c = _Claim(r[0], r[1], SenseKey(r[2], r[3], r[4]), r[5])
            self.claims[c.claim_id] = c
            self.by_sense[c.key].append(c)
            if c.concept_id:
                self.by_concept[c.concept_id].append(c)

    def papers(self, claims: Iterable[_Claim]) -> set[str]:
        return {c.paper_id for c in claims}

    def concept_papers(self) -> dict[str, set[str]]:
        return {cid: self.papers(cs) for cid, cs in self.by_concept.items()}


# --- shared lookups -----------------------------------------------------------------------------

def _paper(conn: sqlite3.Connection, paper_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT title, year FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    return {"paper_id": paper_id, "title": row[0] if row else None, "year": row[1] if row else None}


def _claim_rows(conn: sqlite3.Connection, claim_ids: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i in range(0, len(claim_ids), 500):
        chunk = claim_ids[i:i + 500]
        for r in conn.execute(
                "SELECT cc.claim_id, cc.paper_id, cc.claim_type, cc.normalized_label, cc.claim_text, cc.plain_words, "
                "cc.trust, cc.kind, cc.span_id, e.exact_quote FROM current_claims cc "
                f"LEFT JOIN evidence_spans e ON e.span_id = cc.span_id WHERE cc.claim_id IN "
                f"({', '.join('?' * len(chunk))})", chunk):
            out[r[0]] = r
    return out


def _claim_view(conn: sqlite3.Connection, row: Any, key: SenseKey | None = None) -> dict[str, Any]:
    from ..concepts.model import handle

    out = {"claim_id": row[0], "paper": _paper(conn, row[1]), "claim_type": row[2], "label": row[3],
           "claim_text": row[4], "plain_words": row[5], "trust": TRUST_LABELS.get(row[6], row[6]),
           "claim_kind": row[7],
           "quote": {"span_id": row[8], "text": row[9]} if row[8] else None}
    if row[7] == "assertion":
        out["trust_note"] = "researcher assertion - not the paper's words"
    if key is not None:
        out["sense"] = handle(key)
    return out


@dataclass(frozen=True)
class _SenseMeta:
    gloss: str | None
    gloss_by: str | None
    state: str | None


def _sense_meta(conn: sqlite3.Connection) -> dict[SenseKey, _SenseMeta]:
    from ..concepts.model import SenseKey

    return {SenseKey(r[0], r[1], r[2]): _SenseMeta(r[3], _by(r[4]) if r[3] else None, r[5]) for r in conn.execute(
        "SELECT family, label_key, sense_id, gloss, decided_by, state FROM concept_senses")}


@dataclass(frozen=True)
class _Concept:
    concept_id: str
    family: str
    name: str
    needs_sense_check: bool
    members: tuple[SenseKey, ...]

    @property
    def handle(self) -> str:
        from ..concepts.model import handle, label_key

        named = [m for m in self.members if m.label_key == label_key(self.name)]
        return handle(min(named or self.members))


def _concepts(conn: sqlite3.Connection) -> dict[str, _Concept]:
    """The concepts as the agent audience sees them: only member senses with supporting,
    agent-visible claims (anywhere in the library), so a label known only from hidden notes
    never shows; a concept named by such a label is named by its best visible member."""
    from ..concepts.model import SenseKey, label_key

    papers: dict[SenseKey, int] = {SenseKey(r[0], r[1], r[2]): r[3] for r in conn.execute(
        "SELECT family, label_key, sense_id, COUNT(DISTINCT paper_id) FROM claim_concepts WHERE supports = 1 "
        "AND agent_visible = 1 AND sense_id IS NOT NULL GROUP BY family, label_key, sense_id")}
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
        out[r[0]] = _Concept(r[0], r[1], name, bool(r[3]), visible)
    return out


class _Checks:
    """Which senses count across literatures: a confirmed sense (a sense check or a split)
    counts; an unchecked default sense counts only inside one literature; an ambiguous
    label's unchecked default sense counts for nothing across literatures."""

    def __init__(self, conn: sqlite3.Connection, meta: dict[SenseKey, _SenseMeta]) -> None:
        from ..concepts import guardrails as g

        self.meta = meta
        self.ambiguous = g.effective_ambiguous(conn)

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
            return "ambiguous label"
        return UNCHECKED if lit_count >= 2 else None


# --- concept_search -----------------------------------------------------------------------------

def _search(lib: Library, query: str, family: str | None, graph: str | None, k: int) -> dict[str, Any]:
    from ..concepts import search as cs
    from ..concepts.model import handle

    if not (query or "").strip():
        tool_error("invalid_argument", "query must not be empty")
    family = _family(family)
    _bounded(k, "k", 1, MAX_K)
    _prepare(lib)
    conn = lib.conn
    papers = _members_of_graph(lib, graph) if graph is not None else None
    support = _Support(conn, papers, [family] if family else None)
    adopted = cs.adoption(conn).adopted
    mode, fallback = ("meaning" if adopted else "keyword"), None
    ranked: list[tuple[SenseKey, float, str | None]] = []
    if adopted:
        hits = cs.find(lib, query, family=family, graph=graph, k=max(k * 3, 50))
        if hits:
            mode, fallback = hits[0].mode, hits[0].fallback
        elif not cs.library_sense_vectors(conn, cs.adoption(conn).model):
            mode, fallback = "keyword", "no_vectors"  # what find() would have said
        for h in hits:
            visible = support.by_sense.get(h.key)
            if not visible:
                continue
            best = h.best_claim_id if h.best_claim_id in support.claims else visible[0].claim_id
            ranked.append((h.key, h.score, best))
    else:
        scores = cs.keyword_claim_scores(conn, cs.text_terms(query), families=[family] if family else None,
                                         papers=sorted(support.papers(support.claims.values())))
        best: dict[SenseKey, tuple[float, str]] = {}
        for cid, score in scores.items():
            c = support.claims.get(cid)
            if c is None:
                continue
            prev = best.get(c.key)
            if prev is None or score > prev[0] or (score == prev[0] and cid < prev[1]):
                best[c.key] = (score, cid)
        ranked = sorted(((s, sc, cid) for s, (sc, cid) in best.items()), key=lambda t: (-t[1], tuple(t[0])))
    ranked = ranked[:k]
    meta = _sense_meta(conn)
    checks = _Checks(conn, meta)
    concepts = _concepts(conn)
    rows = _claim_rows(conn, [cid for _, _, cid in ranked if cid])
    out = []
    for key, score, cid in ranked:
        claims = support.by_sense[key]
        cid_concept = claims[0].concept_id
        concept = concepts.get(cid_concept) if cid_concept else None
        m = meta.get(key)
        out.append({
            "handle": handle(key), "family": key.family, "label_key": key.label_key, "sense_id": key.sense_id,
            "concept_id": cid_concept, "concept_name": concept.name if concept else None,
            "concept_handle": concept.handle if concept else None,
            "gloss": m.gloss if m else None, "gloss_by": m.gloss_by if m else None,
            "papers": len(support.papers(claims)), "claims": len(claims),
            "sense_checked": checks.checked(key), "ambiguous_label": checks.is_ambiguous(key),
            "score": round(score, 6),
            "example": _claim_view(conn, rows[cid]) if cid in rows else None,
        })
    return {"query": query, "family": family, "graph": graph, "search_mode": mode, "embeddings_adopted": adopted,
            "fallback": fallback, "hits": out,
            "note": "Hits are senses (label + sense). Counts use only claims you may see that are backed by a "
                    "quote or by the researcher."}


# --- concept_show -------------------------------------------------------------------------------

def _literature_map(lib: Library, graph: str) -> tuple[Any, dict[str, Any], set[str]]:
    """``(run, {paper_id: Literature}, members)`` of the graph's latest grouping."""
    from ..library import graphs, literatures

    with graphs.open_graph(lib, graph) as g:
        run = literatures.latest_run(g)
        lits = literatures.literatures(g) if run else []
        members = {p.paper_id for p in graphs.membership(g, status="member")}
    return run, {pid: lit for lit in lits for pid in lit.members}, members


def _lit_brief(lit: Any) -> dict[str, Any]:
    return {"literature_id": lit.literature_id, "name": lit.name,
            "name_source": "researcher" if lit.name_source == "researcher" else (
                "agent inferred" if lit.name_source == "agent" else None)}


def _not_found(text: str, why: str) -> None:
    tool_error("concept_not_found", f"no concept {text!r}: {why}", next_action=CONCEPT_NOT_FOUND_ACTION,
               handle=text)


def _decisions(conn: sqlite3.Connection, concept: _Concept, kinds: tuple[str, ...]) -> list[dict[str, Any]]:
    labels = sorted({m.label_key for m in concept.members})
    marks = ", ".join("?" * len(labels))
    kmarks = ", ".join("?" * len(kinds))
    out = []
    for r in conn.execute(
            f"SELECT decision_id, kind, family, label_a, sense_a, label_b, sense_b, name, decided_by, state, note, "
            f"origin, decided_at FROM concept_decisions WHERE kind IN ({kmarks}) AND state = 'applied' "
            f"AND (label_a IN ({marks}) OR label_b IN ({marks})) AND (family IS NULL OR family = ?) "
            "ORDER BY decided_at DESC, decision_id", (*kinds, *labels, *labels, concept.family)):
        if r[3] not in labels or r[5] not in labels:  # both sides must be visible members
            continue
        out.append({"decision_id": r[0], "kind": r[1], "label_a": r[3], "sense_a": r[4] or "s0", "label_b": r[5],
                    "sense_b": r[6] or "s0", "name": r[7], "decided_by": r[8], "origin": r[11],
                    "note": r[10], "note_by": _by(r[8]) if r[10] else None, "decided_at": r[12]})
    return out


def _quote_ref(conn: sqlite3.Connection, raw: str | None, hidden: frozenset[str]) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        q = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(q, dict) or q.get("paper_id") in hidden:
        return None
    text = None
    if q.get("span_id"):
        row = conn.execute("SELECT exact_quote FROM evidence_spans WHERE span_id = ?", (q["span_id"],)).fetchone()
        text = row[0] if row else None
    return {"paper": _paper(conn, q.get("paper_id")), "span_id": q.get("span_id"), "text": text}


def _links(conn: sqlite3.Connection, concept: _Concept, concepts: dict[str, _Concept],
           support: _Support) -> list[dict[str, Any]]:
    from ..library.attention import hidden_paper_ids

    hidden = hidden_paper_ids(conn)
    out = []
    for r in conn.execute(
            "SELECT l.concept_a, l.concept_b, l.decision_id, l.note, l.decided_by, d.quote_a_json, d.quote_b_json "
            "FROM concept_links l LEFT JOIN concept_decisions d ON d.decision_id = l.decision_id "
            "WHERE l.concept_a = ? OR l.concept_b = ? ORDER BY l.decision_id",
            (concept.concept_id, concept.concept_id)):
        other_id = r[1] if r[0] == concept.concept_id else r[0]
        other = concepts.get(other_id)
        if other is None or not support.by_concept.get(other_id):
            continue
        out.append({"other": {"concept_id": other_id, "name": other.name, "handle": other.handle,
                              "family": other.family},
                    "decision_id": r[2], "note": r[3], "note_by": _by(r[4]) if r[3] else None, "decided_by": r[4],
                    "quotes": [q for q in (_quote_ref(conn, r[5], hidden), _quote_ref(conn, r[6], hidden)) if q]})
    return out


def _often_with(concept: _Concept, concepts: dict[str, _Concept], support: _Support) -> list[dict[str, Any]]:
    by_concept = support.concept_papers()
    mine = by_concept.get(concept.concept_id, set())
    n = len(support.papers(support.claims.values())) or 1
    scored = []
    for cid, ps in by_concept.items():
        co = len(ps & mine)
        if cid == concept.concept_id or not co or cid not in concepts:
            continue
        lift = co * n / (len(ps) * len(mine))
        scored.append((-lift, -co, concepts[cid].name, cid, co, lift))
    return [{"handle": concepts[cid].handle, "name": name, "shared_papers": co, "lift": round(lift, 3)}
            for _, _, name, cid, co, lift in sorted(scored)[:OFTEN_WITH]]


def _show(lib: Library, text: str, graph: str | None, claims_offset: int, claims_limit: int) -> dict[str, Any]:
    from ..concepts import model

    _bounded(claims_offset, "claims_offset", 0, 10**9)
    _bounded(claims_limit, "claims_limit", 0, MAX_CLAIMS)
    try:
        model.parse_handle(text)
    except ValueError as exc:
        tool_error("invalid_argument", str(exc), next_action=CONCEPT_NOT_FOUND_ACTION)
    _prepare(lib)
    conn = lib.conn
    support = _Support(conn)
    meta = _sense_meta(conn)
    res = model.resolve(conn, text)
    if res.kind == "missing":
        _not_found(text, "no such label or sense")
    if res.kind == "chooser":
        senses = []
        for s in res.senses:
            key = model.SenseKey(*model.parse_handle(s.handle)[:2], s.sense_id)
            claims = support.by_sense.get(key, [])
            if not claims:
                continue
            m = meta.get(key)
            senses.append({"handle": s.handle, "sense_id": s.sense_id, "gloss": s.gloss,
                           "gloss_by": m.gloss_by if m else None, "papers": len(support.papers(claims)),
                           "claims": len(claims)})
        if not senses:
            _not_found(text, "no supporting claims you may see")
        return {"kind": "chooser", "handle": res.handle, "senses": senses,
                "next_action": "This label has several senses; call concept_show with one sense's handle."}
    concepts = _concepts(conn)
    concept = concepts.get(res.concept_id)
    claims = support.by_concept.get(concept.concept_id, []) if concept else []
    if concept is None or not claims:
        _not_found(text, "no supporting claims you may see")
    checks = _Checks(conn, meta)
    lit_of: dict[str, Any] = {}
    if graph is not None:
        _, lit_of, members_in_graph = _literature_map(lib, graph)
    member_views = []
    statuses = []
    for key in concept.members:
        sc = support.by_sense.get(key, [])
        m = meta.get(key)
        lits = {lit_of[c.paper_id].literature_id for c in sc if c.paper_id in lit_of}
        status = checks.status(key, len(lits)) if sc else None
        if status:
            statuses.append(status)
        member_views.append({"handle": model.handle(key), "label_key": key.label_key, "sense_id": key.sense_id,
                             "gloss": m.gloss if m else None, "gloss_by": m.gloss_by if m else None,
                             "sense_checked": checks.checked(key), "ambiguous_label": checks.is_ambiguous(key),
                             "papers": len(support.papers(sc)), "claims": len(sc)})
    rows = _claim_rows(conn, [c.claim_id for c in claims])
    page = claims[claims_offset:claims_offset + claims_limit]
    quotes = sum(1 for c in claims if c.claim_id in rows and rows[c.claim_id][8])
    first_gloss = next((mv for mv in member_views if mv["gloss"]), None)
    body: dict[str, Any] = {
        "kind": "concept",
        "concept": {"concept_id": concept.concept_id, "handle": concept.handle, "name": concept.name,
                    "family": concept.family,
                    "gloss": first_gloss["gloss"] if first_gloss else None,
                    "gloss_by": first_gloss["gloss_by"] if first_gloss else None,
                    "papers": len(support.papers(claims)), "claims": len(claims), "quotes": quotes,
                    "needs_sense_check": concept.needs_sense_check,
                    "sense_status": "ambiguous label" if "ambiguous label" in statuses else (
                        UNCHECKED if UNCHECKED in statuses else "ok")},
        "members": member_views,
        "claims": [_claim_view(conn, rows[c.claim_id], c.key) for c in page if c.claim_id in rows],
        "claims_offset": claims_offset, "claims_total": len(claims),
        "merges": _decisions(conn, concept, ("merge",)),
        "links": _links(conn, concept, concepts, support),
        "often_appears_with": _often_with(concept, concepts, support),
        "agent_rules": list(AGENT_RULES),
    }
    if graph is not None:
        counts: dict[str, int] = defaultdict(int)
        lit_by_id = {}
        for pid in support.papers(claims):
            if pid in lit_of:
                counts[lit_of[pid].literature_id] += 1
                lit_by_id[lit_of[pid].literature_id] = lit_of[pid]
        body["graph"] = graph
        body["graph_papers"] = len(support.papers(claims) & members_in_graph)
        body["literatures"] = [{**_lit_brief(lit_by_id[lid]), "papers": n} for lid, n in sorted(counts.items())]
    return body


# --- literatures and bridges --------------------------------------------------------------------

@dataclass
class _Slice:
    run: Any
    lit_of: dict[str, Any]
    literatures: list[Any]
    support: _Support
    concepts: dict[str, _Concept]
    checks: _Checks
    meta: dict[SenseKey, _SenseMeta]


def _slice(lib: Library, graph: str) -> _Slice:
    _prepare(lib)
    conn = lib.conn
    run, lit_of, members = _literature_map(lib, graph)
    meta = _sense_meta(conn)
    lits = sorted({lit.literature_id: lit for lit in lit_of.values()}.values(), key=lambda x: x.ordinal)
    return _Slice(run, lit_of, lits, _Support(conn, members), _concepts(conn), _Checks(conn, meta), meta)


def _counted_lits(s: _Slice, concept: _Concept) -> tuple[set[str], list[tuple[SenseKey, set[str], str]]]:
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


def _lit_names(s: _Slice, ids: Iterable[str]) -> list[str]:
    by_id = {lit.literature_id: lit for lit in s.literatures}
    return [by_id[i].name for i in sorted(ids, key=lambda i: by_id[i].ordinal)]


def _bridges(conn: sqlite3.Connection, s: _Slice) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from ..concepts.model import handle

    bridges: list[tuple[tuple, dict[str, Any]]] = []
    unchecked: list[dict[str, Any]] = []
    counted_of: dict[str, set[str]] = {}
    for cid in s.support.by_concept:
        concept = s.concepts.get(cid)
        if concept is None:
            continue
        counted, excluded = _counted_lits(s, concept)
        counted_of[cid] = counted
        for key, lits, why in excluded:
            unchecked.append({"handle": handle(key), "name": concept.name, "family": key.family, "reason": why,
                              "literatures": _lit_names(s, lits),
                              "papers": len(s.support.papers(s.support.by_sense[key]))})
        if len(counted) >= 2:
            papers = len(s.support.papers(s.support.by_concept[cid]))
            bridges.append(((-len(counted), -papers, concept.name), {
                "kind": "concept", "handle": concept.handle, "name": concept.name, "family": concept.family,
                "literature_count": len(counted), "literatures": _lit_names(s, counted), "papers": papers}))
    for r in conn.execute("SELECT concept_a, concept_b, note, decided_by FROM concept_links ORDER BY decision_id"):
        la, lb = counted_of.get(r[0], set()), counted_of.get(r[1], set())
        if not la or not lb or len(la | lb) < 2 or (la == lb and len(la) == 1):
            continue
        ca, cb = s.concepts[r[0]], s.concepts[r[1]]
        bridges.append(((-len(la | lb), 0, ca.name), {
            "kind": "link", "a": {"handle": ca.handle, "name": ca.name, "literatures": _lit_names(s, la)},
            "b": {"handle": cb.handle, "name": cb.name, "literatures": _lit_names(s, lb)},
            "note": r[2], "note_by": _by(r[3]) if r[2] else None, "literature_count": len(la | lb)}))
    bridges.sort(key=lambda t: t[0])
    unchecked.sort(key=lambda u: (-len(u["literatures"]), u["handle"]))
    return [b for _, b in bridges], unchecked


def _top_concepts(s: _Slice, lit: Any) -> list[dict[str, Any]]:
    members = set(lit.members)
    n = max(1, len({p for p in s.lit_of}))
    scored = []
    for cid, ps in s.support.concept_papers().items():
        tf = len(ps & members)
        if tf and cid in s.concepts:
            scored.append((-(tf * math.log(1 + n / len(ps))), s.concepts[cid].name, cid, tf))
    return [{"handle": s.concepts[cid].handle, "name": name, "papers": tf}
            for _, name, cid, tf in sorted(scored)[:TOP_CONCEPTS]]


def _literatures(lib: Library, graph: str) -> dict[str, Any]:
    s = _slice(lib, graph)
    if s.run is None:
        return {"graph": graph, "literatures": [], "bridges": [], "unchecked": [], "next_action": NO_RUN_ACTION,
                "rules": list(AGENT_RULES)}
    bridges, unchecked = _bridges(lib.conn, s)
    lits = [{**_lit_brief(lit), "ordinal": lit.ordinal, "size": lit.size,
             "description": lit.agent_description if not lit.researcher_name else None,
             "description_by": "agent inferred" if lit.agent_description and not lit.researcher_name else None,
             "top_concepts": _top_concepts(s, lit)} for lit in s.literatures]
    return {"graph": graph, "literatures": lits, "bridges": bridges, "unchecked": unchecked,
            "unchecked_note": f"\"{UNCHECKED}\": the same words appear in several literatures, but no one has "
                              "checked they mean the same thing. They are not bridges.",
            "stability": {"ari_mean": s.run.ari_mean, "unstable": s.run.unstable, "resolution": s.run.resolution},
            "rules": list(AGENT_RULES)}


# --- hubs ---------------------------------------------------------------------------------------

def _hubs(lib: Library, graph: str | None, family: str | None, limit: int) -> dict[str, Any]:
    family = _family(family)
    _bounded(limit, "limit", 1, MAX_HUBS)
    s: _Slice | None = None
    if graph is not None:
        s = _slice(lib, graph)
        support, concepts, meta = s.support, s.concepts, s.meta
    else:
        _prepare(lib)
        support, concepts, meta = _Support(lib.conn), _concepts(lib.conn), _sense_meta(lib.conn)
    per_family: dict[str, list[tuple[tuple, dict[str, Any]]]] = defaultdict(list)
    for cid, claims in support.by_concept.items():
        concept = concepts.get(cid)
        if concept is None or (family and concept.family != family):
            continue
        papers = support.papers(claims)
        gloss = next((meta[k] for k in concept.members if k in meta and meta[k].gloss), None)
        entry: dict[str, Any] = {"handle": concept.handle, "name": concept.name, "papers": len(papers),
                                 "claims": len(claims), "gloss": gloss.gloss if gloss else None,
                                 "gloss_by": gloss.gloss_by if gloss else None,
                                 "needs_sense_check": concept.needs_sense_check}
        if s is not None and s.run is not None:
            counted, _ = _counted_lits(s, concept)
            entry["literature_count"] = len(counted)
        per_family[concept.family].append(((-len(papers), -len(claims), concept.name), entry))
    from ..concepts.families import CONCEPT_FAMILIES

    body: dict[str, Any] = {"graph": graph, "families": {
        f: [e for _, e in sorted(per_family[f], key=lambda t: t[0])[:limit]]
        for f in CONCEPT_FAMILIES if per_family.get(f)}}
    if s is not None:
        body["bridges"] = _bridges(lib.conn, s)[0][:HUB_BRIDGES] if s.run is not None else []
        if s.run is None:
            body["next_action"] = NO_RUN_ACTION
    body["rules"] = list(AGENT_RULES)
    return body


# --- registration -------------------------------------------------------------------------------

def register_concept_tools(server: FastMCP, ctx: V3Context) -> None:
    """Register the concept tools and the hubs resource on ``server``."""

    @server.tool(annotations=READ)
    def concept_search(query: str, family: str | None = None, graph: str | None = None,
                       k: int = 20) -> dict[str, Any]:
        """Search concepts by words. Hits are senses (label + sense): handle, name, gloss
        (labelled agent inferred or researcher), paper and claim counts, and the best
        example claim with its quote. `search_mode` says how the search ran (keyword or
        meaning). Filter by `family` (assumption, estimand, method, ...) or a context
        `graph` (members only). Search concepts before spans when the question is about
        an idea."""
        with ctx.library() as lib:
            return _search(lib, query, family, graph, k)

    @server.tool(annotations=READ)
    def concept_show(handle: str, graph: str | None = None, claims_offset: int = 0,
                     claims_limit: int = 20) -> dict[str, Any]:
        """One concept by handle (`family/label` or `family/label~sense`): its member
        senses, supporting claims with quotes (paged), merges, closely related links with
        their notes and quotes, often-appears-with, and with `graph` its literatures. A
        label with several senses answers with a chooser. A concept marked "same words,
        sense not checked" is not evidence of a cross-literature connection."""
        with ctx.library() as lib:
            return _show(lib, handle, graph, claims_offset, claims_limit)

    @server.tool(annotations=READ)
    def literatures(graph: str) -> dict[str, Any]:
        """The literatures of a context graph (groups of papers that cite each other):
        name (agent inferred or researcher), size and top concepts; bridges (concepts or
        closely related links spanning literatures, counting only checked or
        single-literature senses); and the "same words, sense not checked" list, which
        are not bridges."""
        with ctx.library() as lib:
            return _literatures(lib, graph)

    @server.tool(annotations=READ)
    def concept_hubs(graph: str | None = None, family: str | None = None, limit: int = 30) -> dict[str, Any]:
        """Orientation: the most common supported concepts per family (in a context
        `graph`, or the whole library), each with paper count, gloss and literature
        spread, plus the graph's top bridges. Start here, with `literatures`, before
        answering a question about ideas across a graph."""
        with ctx.library() as lib:
            return _hubs(lib, graph, family, limit)

    @server.resource("seedgraph://graphs/{slug}/hubs", mime_type="application/json")
    def graph_hubs(slug: str) -> str:
        """The hub list of one context graph (as concept_hubs with graph=slug)."""
        with ctx.library() as lib:
            return json.dumps(_hubs(lib, slug, None, 30), ensure_ascii=False)
