"""Concept pages (stage 4 design §7, §2.4, §11.2): search, the concept page, Recent merges,
the short queue, the review sample, the settings lists and the concept-batch panel.

Pure reads over the derived concept tables as they stand: nothing here refreshes (a GET
never writes). When the derived tables are behind the claims, every page shows the
"concepts out of date" banner with a Refresh button (a POST).

Grounding (§1.3): a concept with no live support is not shown (its page answers 404, and
search counts supporting claims only). Agent-written text (glosses, link notes) is
labelled "agent inferred" with the concept's paper / claim / quote counts; researcher
text is labelled "researcher".
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode

from ...concepts import refresh as rf
from ...concepts import researcher as rs
from ...concepts import search as cs
from ...concepts.families import CONCEPT_FAMILIES
from ...concepts.guardrails import LIBRARY_SCOPE, effective_ambiguous
from ...concepts.model import DEFAULT_SENSE, SenseKey, handle, label_key
from ...library import graphs
from ...library.store import Library

CLAIMS_PAGE = 50
EXAMPLES = 3
OFTEN_WITH = 10
BATCH_TASKS = ("bridge", "merge")
MAX_SENSES = 4


def concept_url(family: str, key: str, sense: str | None = None) -> str:
    url = f"/concepts/{quote(family)}/{quote(key)}"
    if sense and sense != DEFAULT_SENSE:
        url += "?" + urlencode({"sense": sense})
    return url


def key_url(key: SenseKey) -> str:
    return concept_url(key.family, key.label_key, key.sense_id)


def maker_label(decided_by: str | None, batch_id: str | None = None, origin: str | None = None) -> str:
    if decided_by == "researcher":
        if batch_id:
            return "researcher (accepted an agent proposal)"
        return "researcher (seed list)" if origin == "seed" else "researcher"
    if decided_by == "rule":
        return "rule"
    if origin and origin.startswith("migrat"):
        return "agent inferred (migrated from v2)"
    return "agent inferred"


# --- staleness ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Staleness:
    papers: int
    built: bool

    @property
    def stale(self) -> bool:
        return self.papers > 0 or not self.built


def staleness(lib: Library) -> Staleness:
    ood = rf.out_of_date(lib.conn)
    built = lib.conn.execute("SELECT 1 FROM concept_build_state LIMIT 1").fetchone() is not None
    return Staleness(len(ood.papers), built or not ood.papers)


def graph_slugs(lib: Library) -> list[graphs.GraphInfo]:
    return graphs.list_graphs(lib)


# --- shared reads -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Quote:
    paper_id: str
    paper_title: str
    year: int | None
    text: str | None
    claim_id: str | None

    @property
    def url(self) -> str:
        return f"/papers/{self.paper_id}" + (f"#claim-{self.claim_id}" if self.claim_id else "")


def _paper(conn: sqlite3.Connection, pid: str) -> tuple[str, int | None]:
    r = conn.execute("SELECT title, year FROM papers WHERE paper_id = ?", (pid,)).fetchone()
    return (r[0] or pid, r[1]) if r else (pid, None)


def _quote(conn: sqlite3.Connection, raw: str | None) -> Quote | None:
    if not raw:
        return None
    try:
        ref = json.loads(raw)
    except ValueError:
        return None
    pid, span = ref.get("paper_id"), ref.get("span_id")
    if not pid:
        return None
    text = None
    claim_id = None
    if span:
        r = conn.execute("SELECT exact_quote FROM evidence_spans WHERE span_id = ?", (span,)).fetchone()
        text = r[0] if r else None
        c = conn.execute("SELECT claim_id FROM current_claims WHERE paper_id = ? AND span_id = ? LIMIT 1",
                         (pid, span)).fetchone()
        claim_id = c[0] if c else None
    title, year = _paper(conn, pid)
    return Quote(pid, title, year, text, claim_id)


@dataclass(frozen=True)
class ConceptRef:
    concept_id: str
    name: str
    family: str
    handle: str
    url: str
    paper_count: int
    supported: bool
    gloss: str | None = None


def _concept_ref(conn: sqlite3.Connection, cid: str) -> ConceptRef | None:
    r = conn.execute("SELECT c.name, c.family, c.paper_count, c.supported, m.label_key, m.sense_id, c.gloss "
                     "FROM concepts c "
                     "JOIN concept_members m ON m.concept_id = c.concept_id WHERE c.concept_id = ? "
                     "ORDER BY m.label_key, m.sense_id LIMIT 1", (cid,)).fetchone()
    if r is None:
        return None
    key = SenseKey(r[1], r[4], r[5])
    named = conn.execute("SELECT label_key, sense_id FROM concept_members WHERE concept_id = ? AND label_key = ? "
                         "ORDER BY sense_id LIMIT 1", (cid, label_key(r[0]))).fetchone()
    if named:
        key = SenseKey(r[1], named[0], named[1])
    return ConceptRef(cid, r[0], r[1], handle(key), key_url(key), r[2], bool(r[3]), r[6])


def _key_concept(conn: sqlite3.Connection, key: SenseKey) -> str | None:
    r = conn.execute("SELECT concept_id FROM concept_members WHERE family = ? AND label_key = ? AND sense_id = ?",
                     tuple(key)).fetchone()
    return r[0] if r else None


@dataclass(frozen=True)
class Example:
    claim_id: str
    paper_id: str
    paper_title: str
    year: int | None
    claim_type: str
    label: str
    text: str
    quote: str | None
    supports: bool
    trust: str

    @property
    def url(self) -> str:
        return f"/papers/{self.paper_id}#claim-{self.claim_id}"

    @property
    def trust_label(self) -> str:
        if self.supports:
            return "researcher" if self.trust == "researcher" else "paper states (anchored quote)"
        return "agent inferred, no quote"


_EXAMPLE_SQL = (
    "SELECT k.claim_id, k.paper_id, p.title, p.year, cc.claim_type, cc.normalized_label, cc.claim_text, "
    "e.exact_quote, k.supports, cc.trust FROM claim_concepts k JOIN current_claims cc ON cc.claim_id = k.claim_id "
    "LEFT JOIN papers p ON p.paper_id = k.paper_id LEFT JOIN evidence_spans e ON e.span_id = cc.span_id ")


def _example(r: Iterable[Any]) -> Example:
    r = tuple(r)
    return Example(r[0], r[1], r[2] or r[1], r[3], r[4] or "", r[5] or "", r[6] or "", r[7], bool(r[8]), r[9] or "")


def examples_for(conn: sqlite3.Connection, key: SenseKey, limit: int = EXAMPLES) -> list[Example]:
    """Up to ``limit`` supporting claims of one sense, one per paper first."""
    rows = [_example(r) for r in conn.execute(
        _EXAMPLE_SQL + "WHERE k.family = ? AND k.label_key = ? AND k.sense_id = ? AND k.supports = 1 "
        "ORDER BY k.paper_id, k.claim_id", tuple(key))]
    seen: set[str] = set()
    first = []
    for ex in rows:
        if ex.paper_id not in seen:
            seen.add(ex.paper_id)
            first.append(ex)
    rest = [ex for ex in rows if ex not in first]
    return (first + rest)[:limit]


# --- search -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchHit:
    hit: cs.SenseHit
    url: str
    best_quote: str | None
    best_url: str | None
    paper_title: str | None


@dataclass(frozen=True)
class SearchView:
    q: str
    family: str | None
    graph: str | None
    mode: str
    fallback: str | None
    hits: tuple[SearchHit, ...]
    families: tuple[str, ...]
    graphs: tuple[graphs.GraphInfo, ...]
    staleness: Staleness
    error: str | None = None


def search_page(lib: Library, q: str, *, family: str | None = None, graph: str | None = None) -> SearchView:
    conn = lib.conn
    gl = tuple(graph_slugs(lib))
    family = family if family in CONCEPT_FAMILIES else None
    graph = graph if graph and any(x.slug == graph for x in gl) else None
    adopted = cs.adoption(conn).adopted
    mode = "meaning" if adopted else "keyword"
    hits: list[SearchHit] = []
    fallback = None
    error = None
    if q.strip():
        try:
            found = cs.find(lib, q, family=family, graph=graph, k=30, mode="auto")
        except Exception as exc:  # noqa: BLE001 - a broken search index never breaks the page
            found, error = [], f"search failed: {exc}"
        for h in found:
            if not h.paper_count:
                continue
            mode, fallback = h.mode, h.fallback
            quote_text = None
            if h.best_claim_id:
                r = conn.execute("SELECT e.exact_quote FROM current_claims cc LEFT JOIN evidence_spans e "
                                 "ON e.span_id = cc.span_id WHERE cc.claim_id = ?", (h.best_claim_id,)).fetchone()
                quote_text = r[0] if r else None
            title = _paper(conn, h.best_paper_id)[0] if h.best_paper_id else None
            best_url = f"/papers/{h.best_paper_id}#claim-{h.best_claim_id}" if h.best_paper_id else None
            hits.append(SearchHit(h, key_url(h.key), quote_text, best_url, title))
    return SearchView(q, family, graph, mode, fallback, tuple(hits), CONCEPT_FAMILIES, gl, staleness(lib), error)


# --- the concept page ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Member:
    key: SenseKey
    url: str
    gloss: str | None
    gloss_by: str | None
    origin: str
    claims: int


@dataclass(frozen=True)
class Connection:
    decision_id: str
    kind: str
    other: ConceptRef | None
    other_label: str
    note: str | None
    maker: str
    decided_by: str
    quote_a: Quote | None
    quote_b: Quote | None
    can_split: bool
    can_undo: bool


@dataclass(frozen=True)
class PaperGroup:
    paper_id: str
    title: str
    year: int | None
    claims: tuple[Example, ...]


@dataclass(frozen=True)
class OftenWith:
    concept: ConceptRef
    shared_papers: int
    lift: float


@dataclass(frozen=True)
class GraphLiteratures:
    slug: str
    display_name: str
    literatures: tuple[tuple[str, str | None, int], ...]   # (name, name source, papers)
    outside: int                                             # concept papers not placed in this graph


@dataclass(frozen=True)
class SenseOption:
    sense_id: str
    gloss: str | None
    url: str
    current: bool


@dataclass(frozen=True)
class ConceptView:
    concept_id: str
    key: SenseKey
    handle: str
    name: str
    family: str
    gloss: str | None
    gloss_by: str | None
    paper_count: int
    claim_count: int
    quote_count: int
    needs_sense_check: bool
    ambiguous: bool
    members: tuple[Member, ...]
    senses: tuple[SenseOption, ...]
    papers: tuple[PaperGroup, ...]
    claims_total: int
    offset: int
    limit: int
    unsorted: tuple[Example, ...]
    merges: tuple[Connection, ...]
    links: tuple[Connection, ...]
    often_with: tuple[OftenWith, ...]
    literatures: tuple[GraphLiteratures, ...]
    label_claims: tuple[Example, ...]            # the split form: every claim of this label
    label_split: bool
    staleness: Staleness
    agent_text: bool = False

    @property
    def next_offset(self) -> int | None:
        return self.offset + self.limit if self.offset + self.limit < self.claims_total else None

    @property
    def prev_offset(self) -> int | None:
        return max(self.offset - self.limit, 0) if self.offset > 0 else None


@dataclass(frozen=True)
class Chooser:
    family: str
    label_key: str
    senses: tuple[SenseOption, ...]
    staleness: Staleness


def _sense_rows(conn: sqlite3.Connection, family: str, key: str) -> list[tuple[str, str | None, str | None]]:
    rows = conn.execute(
        "SELECT m.sense_id, s.gloss, s.decided_by FROM concept_members m LEFT JOIN concept_senses s ON "
        "s.family = m.family AND s.label_key = m.label_key AND s.sense_id = m.sense_id "
        "WHERE m.family = ? AND m.label_key = ? ORDER BY m.sense_id", (family, key)).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def _supported(conn: sqlite3.Connection, cid: str) -> bool:
    r = conn.execute("SELECT supported FROM concepts WHERE concept_id = ?", (cid,)).fetchone()
    return bool(r and r[0])


def concept_page(lib: Library, family: str, label: str, sense: str | None = None, *, offset: int = 0,
                 limit: int = CLAIMS_PAGE) -> ConceptView | Chooser | None:
    """``None`` when there is no such concept or it has no live support (grounding)."""
    conn = lib.conn
    if family not in CONCEPT_FAMILIES:
        return None
    key_text = label_key(label)
    senses = _sense_rows(conn, family, key_text)
    if not senses:
        return None
    if sense is None:
        if any(s[0] == DEFAULT_SENSE for s in senses):
            sense = DEFAULT_SENSE
        else:
            options = tuple(SenseOption(s[0], s[1], concept_url(family, key_text, s[0]), False) for s in senses
                            if _supported(conn, _key_concept(conn, SenseKey(family, key_text, s[0])) or ""))
            if not options:
                return None
            return Chooser(family, key_text, options, staleness(lib))
    key = SenseKey(family, key_text, sense)
    cid = _key_concept(conn, key)
    if cid is None:
        return None
    row = conn.execute("SELECT name, gloss, paper_count, claim_count, quote_count, supported, needs_sense_check "
                       "FROM concepts WHERE concept_id = ?", (cid,)).fetchone()
    if row is None or not row[5]:
        return None
    name, gloss, n_papers, n_claims, n_quotes, _, needs_check = row

    member_keys = [SenseKey(r[0], r[1], r[2]) for r in conn.execute(
        "SELECT family, label_key, sense_id FROM concept_members WHERE concept_id = ? "
        "ORDER BY label_key, sense_id", (cid,))]
    labels = {m.label_key for m in member_keys}
    decisions = _decisions_touching(conn, labels)
    gloss_meta = {(r[0], r[1], r[2]): (r[3], r[4]) for r in conn.execute(
        "SELECT family, label_key, sense_id, gloss, decided_by FROM concept_senses "
        "WHERE state IN ('default', 'confirmed')")}
    claim_counts = Counter((r[0], r[1], r[2]) for r in conn.execute(
        "SELECT family, label_key, sense_id FROM claim_concepts WHERE concept_id = ?", (cid,)))
    members = []
    for mk in member_keys:
        origin = "own claims"
        for d in decisions:
            if d["kind"] == "merge" and d["state"] == "applied" and mk.label_key in (d["label_a"], d["label_b"]) \
                    and ({d["label_a"], d["label_b"]} <= labels):
                origin = f"merged by {maker_label(d['decided_by'], d['batch_id'], d['origin'])}"
                break
        gm = gloss_meta.get(tuple(mk), (None, None))
        members.append(Member(mk, key_url(mk), gm[0], gm[1], origin, claim_counts.get(tuple(mk), 0)))
    gloss_by = next((m.gloss_by for m in members if m.gloss and m.gloss == gloss), None)

    sense_opts = tuple(SenseOption(s[0], s[1], concept_url(family, key_text, s[0]), s[0] == sense)
                       for s in senses)

    # supporting claims, grouped by paper, paged
    all_claims = [_example(r) for r in conn.execute(
        _EXAMPLE_SQL + "WHERE k.concept_id = ? ORDER BY k.supports DESC, k.paper_id, k.claim_id", (cid,))]
    page = all_claims[offset:offset + limit]
    groups: dict[str, list[Example]] = defaultdict(list)
    for ex in page:
        groups[ex.paper_id].append(ex)
    papers = tuple(PaperGroup(pid, exs[0].paper_title, exs[0].year, tuple(exs)) for pid, exs in groups.items())
    params = sorted(labels)
    unsorted = tuple(_example(r) for r in conn.execute(
        _EXAMPLE_SQL + f"WHERE k.family = ? AND k.label_key IN ({', '.join('?' * len(params))}) "
        "AND k.sense_id IS NULL ORDER BY k.paper_id, k.claim_id", (family, *params)))

    merges, links = _connections(conn, cid, member_keys, decisions)
    label_claims = tuple(_example(r) for r in conn.execute(
        _EXAMPLE_SQL + "WHERE k.family = ? AND k.label_key = ? ORDER BY k.paper_id, k.claim_id", (family, key_text)))
    supporting_papers = sorted({ex.paper_id for ex in all_claims if ex.supports})
    agent_text = gloss_by == "agent" or any(c.decided_by == "agent" and c.note for c in links)
    return ConceptView(
        concept_id=cid, key=key, handle=handle(key), name=name, family=family, gloss=gloss, gloss_by=gloss_by,
        paper_count=n_papers, claim_count=n_claims, quote_count=n_quotes, needs_sense_check=bool(needs_check),
        ambiguous=effective_ambiguous(conn).is_ambiguous(key_text), members=tuple(members), senses=sense_opts,
        papers=papers, claims_total=len(all_claims), offset=offset, limit=limit, unsorted=unsorted,
        merges=merges, links=links, often_with=often_with(conn, cid), literatures=literatures_for(lib,
                                                                                                    supporting_papers),
        label_claims=label_claims, label_split=any(s[0] != DEFAULT_SENSE for s in senses),
        staleness=staleness(lib), agent_text=agent_text)


_DCOLS = ("decision_id", "kind", "family", "family_b", "label_a", "sense_a", "label_b", "sense_b", "name",
          "decided_by", "state", "note", "quote_a_json", "quote_b_json", "batch_id", "origin", "decided_at")


def _decisions_touching(conn: sqlite3.Connection, labels: set[str]) -> list[dict[str, Any]]:
    if not labels:
        return []
    ls = sorted(labels)
    marks = ", ".join("?" * len(ls))
    return [dict(zip(_DCOLS, r)) for r in conn.execute(
        f"SELECT {', '.join(_DCOLS)} FROM concept_decisions WHERE (label_a IN ({marks}) OR label_b IN ({marks})) "
        "AND state = 'applied' ORDER BY decided_at DESC, decision_id", (*ls, *ls))]


def _connections(conn: sqlite3.Connection, cid: str, member_keys: list[SenseKey],
                 decisions: list[dict[str, Any]]) -> tuple[tuple[Connection, ...], tuple[Connection, ...]]:
    labels = {m.label_key for m in member_keys}
    merges = []
    for d in decisions:
        if d["kind"] != "merge" or not ({d["label_a"], d["label_b"]} <= labels):
            continue
        merges.append(Connection(d["decision_id"], "merge", None, f"{d['label_a']} + {d['label_b']}", d["note"],
                                 maker_label(d["decided_by"], d["batch_id"], d["origin"]), d["decided_by"],
                                 _quote(conn, d["quote_a_json"]), _quote(conn, d["quote_b_json"]),
                                 True, d["decided_by"] == "researcher"))
    links = []
    for a, b, did, note, by in conn.execute("SELECT concept_a, concept_b, decision_id, note, decided_by FROM "
                                            "concept_links WHERE concept_a = ? OR concept_b = ? ORDER BY decision_id",
                                            (cid, cid)).fetchall():
        other_id = b if a == cid else a
        other = _concept_ref(conn, other_id)
        if other is not None and not other.supported:
            continue
        d = conn.execute("SELECT quote_a_json, quote_b_json, batch_id, origin, label_a, label_b FROM "
                         "concept_decisions WHERE decision_id = ?", (did,)).fetchone()
        qa, qb = (_quote(conn, d[0]), _quote(conn, d[1])) if d else (None, None)
        links.append(Connection(did, "link", other, other.name if other else other_id, note,
                                maker_label(by, d[2] if d else None, d[3] if d else None), by, qa, qb, True,
                                by == "researcher"))
    return tuple(merges), tuple(links)


def often_with(conn: sqlite3.Connection, cid: str, limit: int = OFTEN_WITH) -> tuple[OftenWith, ...]:
    """Concepts co-supported in the same papers, ranked by lift (computed now, never stored)."""
    mine = {r[0] for r in conn.execute("SELECT DISTINCT paper_id FROM claim_concepts WHERE concept_id = ? "
                                       "AND supports = 1", (cid,))}
    if not mine:
        return ()
    n_total = conn.execute("SELECT COUNT(DISTINCT paper_id) FROM claim_concepts WHERE supports = 1").fetchone()[0]
    marks = ", ".join("?" * len(mine))
    co = conn.execute(
        f"SELECT k.concept_id, COUNT(DISTINCT k.paper_id) FROM claim_concepts k JOIN concepts c ON "
        f"c.concept_id = k.concept_id WHERE k.supports = 1 AND c.supported = 1 AND k.concept_id <> ? AND "
        f"k.paper_id IN ({marks}) GROUP BY k.concept_id", (cid, *sorted(mine))).fetchall()
    scored = []
    for other_id, shared in co:
        ref = _concept_ref(conn, other_id)
        if ref is None or not ref.paper_count:
            continue
        lift = (shared / len(mine)) / (ref.paper_count / n_total) if n_total else 0.0
        scored.append(OftenWith(ref, shared, round(lift, 2)))
    scored.sort(key=lambda o: (-o.lift, -o.shared_papers, o.concept.name, o.concept.concept_id))
    return tuple(scored[:limit])


def literatures_for(lib: Library, papers: list[str]) -> tuple[GraphLiteratures, ...]:
    """Per graph: the literatures the concept's supporting papers fall in."""
    out = []
    if not papers:
        return ()
    from ...library import literatures as lits

    for info in graph_slugs(lib):
        with graphs.open_graph(lib, info.slug) as gr:
            members = {p.paper_id for p in graphs.membership(gr, status="member")}
            mine = [p for p in papers if p in members]
            if not mine:
                continue
            placed = lits.paper_literatures(gr)
        counts: Counter = Counter()
        source: dict[str, str | None] = {}
        for p in mine:
            lit = placed.get(p)
            if lit is None:
                continue
            counts[lit.name] += 1
            source[lit.name] = lit.name_source
        rows = tuple((n, source[n], c) for n, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        out.append(GraphLiteratures(info.slug, info.display_name, rows, len(mine) - sum(counts.values())))
    return tuple(out)


# --- recent merges ------------------------------------------------------------------------------

@dataclass(frozen=True)
class RecentRow:
    decision_id: str
    kind: str                 # merge | link | split_record | label_split
    a: SenseKey | None
    b: SenseKey | None
    a_url: str | None
    b_url: str | None
    text: str
    note: str | None
    maker: str
    decided_by: str
    batch_id: str | None
    decided_at: str
    can_split: bool
    can_undo: bool


@dataclass(frozen=True)
class RecentView:
    rows: tuple[RecentRow, ...]
    batch: str | None
    batches: tuple[str, ...]
    staleness: Staleness


def _keys(d: Mapping[str, Any]) -> tuple[SenseKey | None, SenseKey | None]:
    fam = d["family"]
    if not fam:
        return None, None
    return (SenseKey(fam, d["label_a"], d["sense_a"] or DEFAULT_SENSE),
            SenseKey(d["family_b"] or fam, d["label_b"], d["sense_b"] or DEFAULT_SENSE))


def recent_page(lib: Library, *, batch: str | None = None, limit: int = 200) -> RecentView:
    """Merges, links and splits, newest first (applied ones, and the split records)."""
    conn = lib.conn
    rows = [dict(zip(_DCOLS, r)) for r in conn.execute(
        f"SELECT {', '.join(_DCOLS)} FROM concept_decisions WHERE state = 'applied' AND "
        "(kind IN ('merge', 'link') OR (kind = 'never_connect' AND origin LIKE 'split:%')) "
        "ORDER BY decided_at DESC, decision_id")]
    batches = tuple(sorted({d["batch_id"] for d in rows if d["batch_id"]}))
    if batch:
        rows = [d for d in rows if d["batch_id"] == batch]
    out: list[RecentRow] = []
    for d in rows[:limit]:
        a, b = _keys(d)
        if d["kind"] == "never_connect":
            kind, text = "split_record", f"split / removed: {d['label_a']} | {d['label_b']} (never connect)"
        else:
            kind = d["kind"]
            text = f"{d['label_a']} {'+' if kind == 'merge' else '~'} {d['label_b']}"
        out.append(RecentRow(d["decision_id"], kind, a, b, key_url(a) if a else None, key_url(b) if b else None,
                             text, d["note"], maker_label(d["decided_by"], d["batch_id"], d["origin"]),
                             d["decided_by"], d["batch_id"], d["decided_at"], kind in ("merge", "link"),
                             kind == "split_record" or d["decided_by"] == "researcher"))
    if not batch:
        for fam, key, by, at in conn.execute(
                "SELECT family, label_key, MIN(decided_by), MAX(decided_at) FROM concept_senses WHERE sense_id <> 's0' "
                "AND state = 'confirmed' GROUP BY family, label_key ORDER BY 4 DESC"):
            k = SenseKey(fam, key, DEFAULT_SENSE)
            out.append(RecentRow(f"sense:{fam}/{key}", "label_split", k, None, concept_url(fam, key), None,
                                 f"{key} split into senses", None, maker_label(by), by, None, at, False, True))
        out.sort(key=lambda r: (r.decided_at, r.decision_id), reverse=True)
    return RecentView(tuple(out), batch, batches, staleness(lib))


# --- the queue ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class QueueSide:
    key: SenseKey
    url: str
    name: str | None
    papers: int
    examples: tuple[Example, ...]


@dataclass(frozen=True)
class QueueRow:
    item: rs.QueueItem
    a: QueueSide
    b: QueueSide | None
    maker: str


@dataclass(frozen=True)
class QueueView:
    rows: tuple[QueueRow, ...]
    staleness: Staleness


def _side(conn: sqlite3.Connection, key: SenseKey) -> QueueSide:
    cid = _key_concept(conn, key)
    name, papers = None, 0
    if cid:
        r = conn.execute("SELECT name, paper_count FROM concepts WHERE concept_id = ?", (cid,)).fetchone()
        if r:
            name, papers = r
    exs = examples_for(conn, key) if cid else []
    if not exs and key.sense_id == DEFAULT_SENSE:  # a queued split: claims carry no sense yet
        exs = [_example(r) for r in conn.execute(
            _EXAMPLE_SQL + "WHERE k.family = ? AND k.label_key = ? AND k.supports = 1 ORDER BY k.paper_id, "
            "k.claim_id LIMIT ?", (key.family, key.label_key, EXAMPLES))]
    return QueueSide(key, key_url(key), name, papers, tuple(exs))


def queue_page(lib: Library) -> QueueView:
    conn = lib.conn
    rows = []
    for it in rs.queue(conn):
        rows.append(QueueRow(it, _side(conn, it.a), _side(conn, it.b) if it.b else None,
                             maker_label(it.decided_by, it.batch_id)))
    return QueueView(tuple(rows), staleness(lib))


# --- review sample ------------------------------------------------------------------------------

@dataclass(frozen=True)
class ReviewItem:
    decision_id: str
    kind: str
    family: str | None
    a: QueueSide | None
    b: QueueSide | None
    label_a: str
    label_b: str
    note: str | None
    quote_a: Quote | None
    quote_b: Quote | None
    marks: tuple[str, ...]
    mark: str | None
    mark_note: str | None
    state: str


@dataclass(frozen=True)
class ReviewView:
    sample: rs.ReviewSample
    items: tuple[ReviewItem, ...]
    summary: rs.ReviewSummary


@dataclass(frozen=True)
class ReviewIndex:
    samples: tuple[tuple[rs.ReviewSample, rs.ReviewSummary], ...]
    population: int
    default_size: int = rs.SAMPLE_SIZE


def review_index(lib: Library) -> ReviewIndex:
    conn = lib.conn
    samples = tuple((s, rs.review_summary(conn, s.sample_id)) for s in rs.review_samples(conn))
    population = conn.execute("SELECT COUNT(*) FROM concept_decisions WHERE decided_by = 'agent' AND state = "
                              "'applied' AND kind IN ('merge', 'link') AND batch_id IS NOT NULL").fetchone()[0]
    return ReviewIndex(samples, population)  # type: ignore[arg-type]


def review_page(lib: Library, sample_id: str) -> ReviewView | None:
    conn = lib.conn
    sample = rs.review_sample(conn, sample_id)
    if sample is None:
        return None
    got = rs.marks(conn, sample_id)
    items = []
    for did in sample.decision_ids:
        r = conn.execute(f"SELECT {', '.join(_DCOLS)} FROM concept_decisions WHERE decision_id = ?",
                         (did,)).fetchone()
        if r is None:
            continue
        d = dict(zip(_DCOLS, r))
        a, b = _keys(d)
        mark, mnote = got.get(did, (None, None))
        items.append(ReviewItem(did, d["kind"], d["family"], _side(conn, a) if a else None,
                                _side(conn, b) if b else None, d["label_a"], d["label_b"], d["note"],
                                _quote(conn, d["quote_a_json"]), _quote(conn, d["quote_b_json"]),
                                rs.MARKS.get(d["kind"], ()), mark, mnote, d["state"]))
    return ReviewView(sample, tuple(items), rs.review_summary(conn, sample_id))  # type: ignore[arg-type]


# --- settings lists -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AmbiguousRow:
    pattern: str
    match: str
    scope: str
    effect: str
    note: str | None
    origin: str


@dataclass(frozen=True)
class NeverMergeRow:
    decision_id: str
    label_a: str
    label_b: str
    family: str | None
    note: str | None
    origin: str


@dataclass(frozen=True)
class ListsView:
    library_ambiguous: tuple[AmbiguousRow, ...]
    graph_ambiguous: dict[str, tuple[AmbiguousRow, ...]]
    never_merge: tuple[NeverMergeRow, ...]
    graphs: tuple[graphs.GraphInfo, ...]
    families: tuple[str, ...]
    staleness: Staleness


def lists_page(lib: Library) -> ListsView:
    conn = lib.conn
    amb = [AmbiguousRow(*r) for r in conn.execute(
        "SELECT pattern, match, scope, effect, note, origin FROM ambiguous_labels ORDER BY scope, pattern")]
    if not any(r.scope == LIBRARY_SCOPE for r in amb):  # seeds not written yet: show the effective list
        amb = [AmbiguousRow(p.pattern, p.match, LIBRARY_SCOPE, "ambiguous", p.note, p.origin)
               for p in effective_ambiguous(conn).patterns] + [r for r in amb if r.scope != LIBRARY_SCOPE]
    per_graph: dict[str, list[AmbiguousRow]] = defaultdict(list)
    for r in amb:
        if r.scope != LIBRARY_SCOPE:
            per_graph[r.scope].append(r)
    nm = tuple(NeverMergeRow(*r) for r in conn.execute(
        "SELECT decision_id, label_a, label_b, family, note, origin FROM concept_decisions WHERE kind = 'never_merge' "
        "AND state IN ('applied', 'pending_apply') ORDER BY label_a, label_b, decision_id"))
    return ListsView(tuple(r for r in amb if r.scope == LIBRARY_SCOPE), {k: tuple(v) for k, v in per_graph.items()},
                     nm, tuple(graph_slugs(lib)), CONCEPT_FAMILIES, staleness(lib))


# --- concept batches ----------------------------------------------------------------------------

@dataclass(frozen=True)
class BatchPanel:
    task: str
    families: tuple[str, ...]
    k: int | None
    budget_tokens: int | None
    graph: str | None
    estimate: Any
    error: str | None
    command: str
    options: dict[str, Any]
    all_families: tuple[str, ...]
    graphs: tuple[graphs.GraphInfo, ...]
    search_mode: str


def batch_options(task: str, families: Iterable[str], k: int | None, budget: int | None) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    fams = [f for f in families if f in CONCEPT_FAMILIES]
    if fams:
        opts["families"] = ",".join(fams)
    if task == "bridge" and k:
        opts["k"] = k
    if budget:
        opts["budget_tokens"] = budget
    return opts


def batch_command(task: str, options: Mapping[str, Any], graph: str | None) -> str:
    parts = ["seedgraph", "work", "create", f"concepts:{task}"]
    if graph:
        parts += ["--graph", graph]
    for k, v in options.items():
        parts += ["--option", f"{k}={v}"]
    return " ".join(parts)


def batch_panel(lib: Library, task: str, families: Iterable[str], k: int | None, budget: int | None,
                graph: str | None) -> BatchPanel:
    from ...errors import SeedgraphError
    from ...work import estimate as west

    task = task if task in BATCH_TASKS else "bridge"
    gl = tuple(graph_slugs(lib))
    graph = graph if graph and any(x.slug == graph for x in gl) else None
    opts = batch_options(task, families, k, budget)
    est, error = None, None
    try:
        est = west.estimate(lib, f"concepts:{task}", options=opts, graph=graph)
    except (SeedgraphError, ValueError) as exc:
        error = str(exc)
    fams = tuple(f for f in (opts.get("families") or "").split(",") if f)
    return BatchPanel(task, fams, k, budget, graph, est, error, batch_command(task, opts, graph), opts,
                      CONCEPT_FAMILIES, gl, "meaning" if cs.adoption(lib.conn).adopted else "keyword")


def as_int(value: Any) -> int | None:
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


__all__ = ["batch_panel", "concept_page", "concept_url", "lists_page", "queue_page", "recent_page", "review_index",
           "review_page", "search_page", "staleness"]
