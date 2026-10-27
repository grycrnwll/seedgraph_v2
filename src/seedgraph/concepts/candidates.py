"""Candidate generation for the concept tasks (stage 4 design §3.2, §4.1, §4.3). No AI cost.

:func:`load` reads a :class:`ConceptState` from :func:`seedgraph.concepts.refresh.plan`
(a pure read: it works on a read-only connection and on a library whose derived tables
are stale or missing), plus the stored decisions, sense rows, the ambiguous list (the
library list, or library + graph overrides when ``graph`` is given), the citation graph
and the text of every supporting claim.

With ``graph`` the work is graph-scoped: only the graph's member papers support senses
(the slice), and the ambiguous list is the library list plus that graph's overrides.
Library-wide work uses the library list alone.

- :func:`merge_candidates` (``concepts:merge``): **lexical groups** within a family -
  labels joined by token Jaccard >= :data:`TAU_LEX` on content tokens (generic words
  removed) or by acronym expansion (v2 ``canon.candidate_clusters``, indexed by token so
  it scales). Groups are cut to :data:`MAX_GROUP` labels keeping the strongest edges; the
  surviving edges inside a group are the items.
- :func:`bridge_candidates` (``concepts:bridge``): **sense checks** (ambiguous labels with
  support in >= 2 papers, and senses whose papers sit at citation distance >= 3 or span
  literatures, never checked before) and **neighbour pairs**: each sense's ``k`` nearest
  senses above :data:`SIM_FLOOR`, in its family plus the adjacent pairs
  ``assumption<->method`` and ``estimand<->method`` (cross-family pairs belong to the
  earlier family's pass; across families only links are allowed). Ordered: sense checks
  first, then pairs by descending citation distance (``inf``, 3, 2, 1, 0), similarity
  as the tie-break.

**Neighbour seam.** The default finder, :func:`keyword_neighbours`, is a TF-IDF cosine
over each sense's label + claim texts + plain words (the keyword fallback of design
§5.4). Meaning-based neighbours (C4's ``concepts.search`` with embeddings) plug in as
``finder=`` with the same signature.

**Dropped before packaging** (both tasks): pairs inside one concept already, pairs whose
supporting papers are one and the same single paper, pairs with any stored decision
(any kind, either maker; ``undone`` and ``needs_rejudge`` rows do not count), pairs whose
two **concepts** already have a stored decision between any of their members, pairs
across a ``never_connect`` veto, and unsupported senses. Only agent-visible support
counts (design §1.3, agent audience). A concept pair yields one candidate per batch
(the first in priority order), so two member labels of one concept never put the same
question to two packages.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from ..semantic.merge import is_acronym_expansion, normalize
from . import guardrails as g
from . import refresh as rf
from .families import BY_NAME, CONCEPT_FAMILIES
from .model import DEFAULT_SENSE, SenseKey

TAU_LEX = 0.5
MAX_GROUP = 12
K_DEFAULT = 5
SIM_FLOOR = 0.12
MAX_DISTANCE = 3
#: Terms in more than this share of sense documents carry no signal for neighbours.
MAX_DF_SHARE = 0.08
MIN_DF_CAP = 50
#: Families whose senses may pair across the fence (links only); the pair belongs to the
#: first family's pass (bridge order).
ADJACENT: Mapping[str, tuple[str, ...]] = {"assumption": ("method",), "estimand": ("method",)}
FIRST_PASS: tuple[str, ...] = ("assumption", "estimand", "method")
SECOND_PASS: tuple[str, ...] = ("result", "data")
SENSE_CHECK_MODES = ("all", "ambiguous", "none")
#: Decision states that do not count as "decided" for candidate search.
_NOT_DECIDED = ("undone", "needs_rejudge")

_STOP = frozenset("""a an the of in on at to for from by with and or not no is are was were be been being as that
this these those it its their there which who whom whose than then into over under between within without
we our they them he she his her i you your can may might must should would could will shall do does did done
has have had also such each any all some more most less least very only both either neither other same so if
when where while about across after before during per via via using use used based invented""".split())


# --- state ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimText:
    claim_id: str
    paper_id: str
    family: str
    label_key: str
    surface: str
    claim_type: str
    text: str
    plain_words: str | None
    span_id: str | None
    quote: str | None
    fingerprint: str
    title: str | None
    year: int | None


@dataclass(frozen=True)
class Sense:
    key: SenseKey
    concept_id: str
    papers: frozenset[str]
    claims: tuple[str, ...]          # supporting, agent-visible claim ids (paper, claim order)
    ambiguous: bool
    checked: bool                    # a stored sense row (confirmed, split or queued) exists


class DecidedIndex:
    """Stored decisions by label pair. A NULL family or sense matches any (conservative:
    a pair judged once is never judged again)."""

    def __init__(self, rows: Iterable[Mapping]) -> None:
        self._by_pair: dict[tuple[str, str], list[Mapping]] = defaultdict(list)
        for d in rows:
            if d.get("state") in _NOT_DECIDED:
                continue
            self._by_pair[tuple(sorted((d["label_a"], d["label_b"])))].append(d)

    def rows(self, a: SenseKey, b: SenseKey) -> list[Mapping]:
        out = []
        for d in self._by_pair.get(tuple(sorted((a.label_key, b.label_key))), ()):
            fam_a, fam_b = d.get("family"), d.get("family_b") or d.get("family")
            for x, y in ((a, b), (b, a)):
                if x.label_key != d["label_a"] or y.label_key != d["label_b"]:
                    continue
                if fam_a is not None and x.family != fam_a:
                    continue
                if fam_b is not None and y.family != fam_b:
                    continue
                sa, sb = d.get("sense_a"), d.get("sense_b")
                if (sa is None or sa == x.sense_id) and (sb is None or sb == y.sense_id):
                    out.append(d)
                    break
        return out

    def decided(self, a: SenseKey, b: SenseKey) -> bool:
        return bool(self.rows(a, b))


class CitationGraph:
    """The library citation graph, undirected, over ``citation_links`` (cited-only papers
    are nodes too). Distances are capped at :data:`MAX_DISTANCE`; ``None`` means farther."""

    def __init__(self, edges: Iterable[tuple[str, str]]) -> None:
        self.adj: dict[str, set[str]] = defaultdict(set)
        for s, t in edges:
            if s != t:
                self.adj[s].add(t)
                self.adj[t].add(s)
        self._balls: dict[str, dict[str, int]] = {}

    def ball(self, paper: str) -> dict[str, int]:
        hit = self._balls.get(paper)
        if hit is None:
            hit = {paper: 0}
            queue = deque([paper])
            while queue:
                x = queue.popleft()
                d = hit[x]
                if d == MAX_DISTANCE:
                    continue
                for y in self.adj.get(x, ()):
                    if y not in hit:
                        hit[y] = d + 1
                        queue.append(y)
            self._balls[paper] = hit
        return hit

    def distance(self, a: Iterable[str], b: Iterable[str]) -> int | None:
        """Minimum distance over the two paper sets (BFS from the smaller side)."""
        a, b = set(a), set(b)
        if a & b:
            return 0
        if len(a) > len(b):
            a, b = b, a
        best = None
        for p in sorted(a):
            ball = self.ball(p)
            for q in b:
                d = ball.get(q)
                if d is not None and (best is None or d < best):
                    best = d
                    if best == 1:
                        return 1
        return best

    def spread(self, papers: Iterable[str]) -> bool:
        """True when some two of ``papers`` sit at distance >= 3 (or farther)."""
        ps = sorted(set(papers))
        for i, p in enumerate(ps):
            ball = self.ball(p)
            for q in ps[i + 1:]:
                d = ball.get(q)
                if d is None or d >= 3:
                    return True
        return False


@dataclass
class ConceptState:
    senses: dict[SenseKey, Sense]
    groups: dict[str, frozenset[SenseKey]]           # concept_id -> member senses
    concept_papers: dict[str, int]                    # concept_id -> paper count
    claims: dict[str, ClaimText]
    decided: DecidedIndex
    vetoes: g.VetoIndex
    ambiguous: g.AmbiguousList
    citations: CitationGraph
    graph: str | None = None
    literature_of: dict[str, str] = field(default_factory=dict)   # paper -> literature name (graph only)

    def group_of(self, key: SenseKey) -> frozenset[SenseKey]:
        return self.groups.get(self.senses[key].concept_id, frozenset({key}))

    def concept_of(self, key: SenseKey) -> str:
        s = self.senses.get(key)
        return s.concept_id if s else f"sense:{key}"

    def concepts_decided(self, a: SenseKey, b: SenseKey) -> bool:
        """True when a stored decision already joins the two senses' concepts through any
        of their members (one link or unrelated between two concepts settles every member
        pair; a merge inside the batch's frozen state made them one concept)."""
        cache = self.__dict__.get("_decided_concepts")
        if cache is None:
            by_label: dict[str, list[SenseKey]] = defaultdict(list)
            for k in self.senses:
                by_label[k.label_key].append(k)
            cache = set()
            for rows in self.decided._by_pair.values():  # noqa: SLF001
                for d in rows:
                    if d["kind"] not in ("merge", "link", "unrelated"):
                        continue  # a never-merge still allows a link between other members
                    for x in by_label.get(d["label_a"], ()):
                        for y in by_label.get(d["label_b"], ()):
                            if self.decided.rows(x, y):
                                cache.add(frozenset((self.concept_of(x), self.concept_of(y))))
            self.__dict__["_decided_concepts"] = cache
        return frozenset((self.concept_of(a), self.concept_of(b))) in cache

    def never_connect(self, a: SenseKey, b: SenseKey) -> bool:
        return self.vetoes.between(self.group_of(a), self.group_of(b), kinds=("never_connect",)) is not None

    def surface(self, key: SenseKey) -> str:
        s = self.senses.get(key)
        if s:
            counts = Counter(self.claims[c].surface for c in s.claims if c in self.claims)
            if counts:
                return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        return key.label_key


def _decision_rows(conn: sqlite3.Connection, plan: rf.RefreshPlan) -> list[dict]:
    rows = rf._rows(conn, "concept_decisions", rf.DECISION_COLS, "decision_id")  # noqa: SLF001
    if not g.seeds_written(conn):
        have = {(d["label_a"], d["label_b"]) for d in rows if d["family"] is None and d["sense_a"] is None
                and d["sense_b"] is None and d["decided_by"] == "researcher"}
        for a, b in g.SEED_NEVER_MERGE:
            a, b = sorted((a, b))
            if (a, b) not in have:
                rows.append(dict(decision_id="seed", kind="never_merge", family=None, family_b=None, label_a=a,
                                 sense_a=None, label_b=b, sense_b=None, name=None, decided_by="researcher",
                                 state="applied", note=None, veto_reason=None, origin=g.SEED_ORIGIN,
                                 decided_at=plan.now))
    return rows + [dict(d) for d in plan.new_rule_decisions]


def _vetoes(rows: Iterable[Mapping]) -> g.VetoIndex:
    return g.VetoIndex(
        g.Veto(d["kind"], d["label_a"], d["label_b"], d["family"], d.get("family_b") or d["family"],
               d["sense_a"], d["sense_b"], d["decision_id"])
        for d in rows if d["kind"] in ("never_merge", "never_connect") and d["state"] in ("applied", "pending_apply"))


def _claim_texts(conn: sqlite3.Connection, wanted: set[str]) -> dict[str, ClaimText]:
    from .families import family_of
    from .model import claim_fingerprint, label_key

    out: dict[str, ClaimText] = {}
    sql = ("SELECT cc.claim_id, cc.paper_id, cc.claim_type, cc.field_key, cc.normalized_label, cc.claim_text, "
           "cc.plain_words, cc.span_id, e.exact_quote, e.quote_hash, e.markdown_hash, e.start_char, e.end_char, "
           "p.title, p.year FROM current_claims cc LEFT JOIN evidence_spans e ON e.span_id = cc.span_id "
           "LEFT JOIN papers p ON p.paper_id = cc.paper_id ORDER BY cc.paper_id, cc.claim_id")
    for r in conn.execute(sql):
        if r[0] not in wanted:
            continue
        row = {"claim_type": r[2], "field_key": r[3], "normalized_label": r[4], "claim_text": r[5],
               "quote_hash": r[9], "markdown_hash": r[10], "start_char": r[11], "end_char": r[12]}
        span = row if r[9] is not None else None
        out[r[0]] = ClaimText(
            claim_id=r[0], paper_id=r[1], family=family_of(r[2]) or "", label_key=label_key(r[4]),
            surface=" ".join((r[4] or "").split()), claim_type=r[2] or "", text=" ".join((r[5] or "").split()),
            plain_words=" ".join(r[6].split()) if r[6] else None, span_id=r[7],
            quote=" ".join(r[8].split()) if r[8] else None, fingerprint=claim_fingerprint(row, span),
            title=r[13], year=r[14])
    return out


def _literatures(conn: sqlite3.Connection, graph: str | None, root) -> dict[str, str]:
    if graph is None or root is None:
        return {}
    from ..library import graphs, literatures
    from ..library.store import Library

    lib = Library(root=root, conn=conn)
    try:
        with graphs.open_graph(lib, graph) as gr:
            return {p: lit.name for p, lit in literatures.paper_literatures(gr).items()}
    except Exception:  # noqa: BLE001 - a graph with no grouping yet: no literature names
        return {}


def _graph_members(conn: sqlite3.Connection, graph: str | None, root) -> set[str] | None:
    """The graph's member papers (``None`` = library-wide work)."""
    if graph is None:
        return None
    from ..library import graphs
    from ..library.store import Library

    if root is None:
        raise ValueError("graph-scoped concept work needs the library folder")
    with graphs.open_graph(Library(root=root, conn=conn), graph) as gr:
        return {p.paper_id for p in graphs.membership(gr, status="member")}


def load(conn: sqlite3.Connection, *, graph: str | None = None, families: Sequence[str] | None = None,
         library_root=None) -> ConceptState:
    """The candidate-search state (pure read; see the module doc). ``families`` limits the
    claim texts loaded (every concept family by default)."""
    plan = rf.plan(conn)
    fams = set(families or CONCEPT_FAMILIES)
    sense_rows = rf._rows(conn, "concept_senses", ("family", "label_key", "sense_id", "state"),  # noqa: SLF001
                          "family, label_key, sense_id")
    checked = {(s["family"], s["label_key"]) for s in sense_rows if s["state"] in ("default", "confirmed", "queued")}
    ambiguous = g.effective_ambiguous(conn, graph)
    members = _graph_members(conn, graph, library_root)
    by_sense: dict[SenseKey, list[tuple[str, str]]] = defaultdict(list)
    for claim_id, paper_id, family, key, sense_id, cid, supports, visible in plan.claim_concepts:
        if sense_id is None or not supports or not visible:
            continue
        if members is not None and paper_id not in members:
            continue  # graph-scoped work: the graph's slice only
        by_sense[SenseKey(family, key, sense_id)].append((paper_id, claim_id))
    groups: dict[str, set[SenseKey]] = defaultdict(set)
    cid_of: dict[SenseKey, str] = {}
    for cid, family, key, sense_id in plan.members:
        k = SenseKey(family, key, sense_id)
        groups[cid].add(k)
        cid_of[k] = cid
    senses: dict[SenseKey, Sense] = {}
    wanted: set[str] = set()
    for k, rows in by_sense.items():
        rows.sort()
        senses[k] = Sense(k, cid_of[k], frozenset(p for p, _ in rows), tuple(c for _, c in rows),
                          ambiguous.is_ambiguous(k.label_key), (k.family, k.label_key) in checked)
        if k.family in fams:
            wanted.update(c for _, c in rows)
    decisions = _decision_rows(conn, plan)
    edges = [(r[0], r[1]) for r in conn.execute("SELECT source_paper_id, target_paper_id FROM citation_links")]
    return ConceptState(
        senses=senses, groups={c: frozenset(v) for c, v in groups.items()},
        concept_papers={r[0]: r[4] for r in plan.concepts}, claims=_claim_texts(conn, wanted),
        decided=DecidedIndex(decisions), vetoes=_vetoes(decisions), ambiguous=ambiguous,
        citations=CitationGraph(edges), graph=graph, literature_of=_literatures(conn, graph, library_root))


# --- candidates -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """One item to judge: a ``pair`` (``a``, ``b``) or a ``sense_check`` (``a`` only).
    ``group`` numbers the lexical group (merge task); ``distance`` is the citation distance
    (``None`` = farther than 3); ``similarity`` the neighbour score (or lexical weight)."""

    kind: str
    a: SenseKey
    b: SenseKey | None = None
    group: int | None = None
    distance: int | None = None
    similarity: float = 0.0
    reason: str | None = None        # sense checks: 'ambiguous' | 'spread'
    family: str = ""                 # the pass family (a cross-family pair belongs to the earlier one)


def distance_rank(d: int | None) -> int:
    """Sort rank: farther first (``None`` = farther than 3 ranks highest)."""
    return MAX_DISTANCE + 1 if d is None else d


def _droppable(state: ConceptState, a: SenseKey, b: SenseKey) -> bool:
    sa, sb = state.senses.get(a), state.senses.get(b)
    if sa is None or sb is None:
        return True
    if sa.concept_id == sb.concept_id:
        return True
    if len(sa.papers) == 1 and sa.papers == sb.papers:
        return True
    if state.decided.decided(a, b) or state.concepts_decided(a, b):
        return True
    return state.never_connect(a, b)


def _once_per_concept_pair(state: ConceptState, cands: Iterable, seen: set[frozenset[str]]) -> list:
    """Keep the first candidate of each concept pair (``cands`` are in priority order;
    ``seen`` carries across families, so a concept pair appears once per batch)."""
    out = []
    for c in cands:
        x, y = (c[1], c[2]) if isinstance(c, tuple) else (c.a, c.b)
        pair = frozenset((state.concept_of(x), state.concept_of(y)))
        if pair in seen:
            continue
        seen.add(pair)
        out.append(c)
    return out


def _lex_edges(keys: Sequence[SenseKey]) -> list[tuple[float, SenseKey, SenseKey]]:
    toks = {k: frozenset(g.content_tokens(k.label_key)) for k in keys}
    index: dict[str, list[SenseKey]] = defaultdict(list)
    for k in keys:
        for t in toks[k]:
            index[t].append(k)
    seen: dict[tuple[SenseKey, SenseKey], float] = {}
    for k in keys:
        if not toks[k]:
            continue
        cands: set[SenseKey] = set()
        for t in toks[k]:
            cands.update(index[t])
        for o in cands:
            if o <= k:
                continue
            inter = len(toks[k] & toks[o])
            jac = inter / len(toks[k] | toks[o])
            if jac >= TAU_LEX:
                seen[(k, o)] = max(seen.get((k, o), 0.0), round(jac, 6))
    singles: dict[str, list[SenseKey]] = defaultdict(list)
    for k in keys:
        if " " not in k.label_key and len(k.label_key) >= 2:
            singles[k.label_key].append(k)
    for k in keys:
        words = k.label_key.split()
        if len(words) < 2:
            continue
        initials = "".join(w[0] for w in words)
        for o in singles.get(initials, ()):
            if o != k and is_acronym_expansion(o.label_key, k.label_key):
                lo, hi = sorted((o, k))
                seen[(lo, hi)] = 1.0
    return sorted(((w, x, y) for (x, y), w in seen.items()), key=lambda e: (-e[0], e[1], e[2]))


def merge_candidates(state: ConceptState, *, families: Sequence[str] | None = None) -> list[Candidate]:
    """Lexical groups (see the module doc), family by family in table order."""
    out: list[Candidate] = []
    group_no = 0
    seen: set[frozenset[str]] = set()
    for family in [f for f in CONCEPT_FAMILIES if families is None or f in families]:
        keys = sorted(k for k in state.senses if k.family == family)
        edges = _once_per_concept_pair(
            state, [e for e in _lex_edges(keys) if not _droppable(state, e[1], e[2])], seen)
        parent: dict[SenseKey, SenseKey] = {}
        size: dict[SenseKey, int] = {}

        def find(x: SenseKey, parent: dict = parent, size: dict = size) -> SenseKey:
            parent.setdefault(x, x)
            size.setdefault(x, 1)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for _, x, y in edges:  # strongest first; a union that would exceed MAX_GROUP is skipped
            rx, ry = find(x), find(y)
            if rx != ry and size[rx] + size[ry] <= MAX_GROUP:
                lo, hi = sorted((rx, ry))
                parent[hi] = lo
                size[lo] += size.pop(hi)
        by_root: dict[SenseKey, list[tuple[float, SenseKey, SenseKey]]] = defaultdict(list)
        for w, x, y in edges:
            if find(x) == find(y):
                by_root[find(x)].append((w, x, y))
        for root in sorted(by_root):
            group_no += 1
            for w, x, y in sorted(by_root[root], key=lambda e: (e[1], e[2])):
                out.append(Candidate("pair", x, y, group=group_no, similarity=w, family=family))
    return out


# --- neighbours -----------------------------------------------------------------------------------

Finder = Callable[[ConceptState, Sequence[SenseKey], Sequence[SenseKey], int], dict[SenseKey, list[tuple[SenseKey, float]]]]
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _doc(state: ConceptState, key: SenseKey) -> Counter:
    s = state.senses[key]
    words: list[str] = []
    words += _TOKEN_RE.findall(normalize(key.label_key))
    for cid in s.claims:
        c = state.claims.get(cid)
        if c is None:
            continue
        words += _TOKEN_RE.findall(normalize(c.text))
        if c.plain_words:
            words += _TOKEN_RE.findall(normalize(c.plain_words))
    return Counter(w for w in words if len(w) > 1 and w not in _STOP and not w.isdigit())


def keyword_neighbours(state: ConceptState, queries: Sequence[SenseKey], pool: Sequence[SenseKey],
                       k: int) -> dict[SenseKey, list[tuple[SenseKey, float]]]:
    """TF-IDF cosine neighbours of each query sense among ``pool`` (top ``k`` above
    :data:`SIM_FLOOR`; ties by key). Pure Python with an inverted index; very common terms
    (df share above :data:`MAX_DF_SHARE`) are skipped."""
    universe = sorted(set(queries) | set(pool))
    docs = {key: _doc(state, key) for key in universe}
    n = len(universe) or 1
    df: Counter = Counter()
    for d in docs.values():
        df.update(d.keys())
    max_df = max(MIN_DF_CAP, int(MAX_DF_SHARE * n))
    vecs: dict[SenseKey, dict[str, float]] = {}
    for key, d in docs.items():
        v = {t: (1 + math.log(c)) * math.log(1 + n / df[t]) for t, c in d.items() if df[t] <= max_df}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs[key] = {t: x / norm for t, x in v.items()}
    pool_set = set(pool)
    postings: dict[str, list[tuple[SenseKey, float]]] = defaultdict(list)
    for key in sorted(pool_set):
        for t, x in vecs[key].items():
            postings[t].append((key, x))
    out: dict[SenseKey, list[tuple[SenseKey, float]]] = {}
    for q in queries:
        scores: dict[SenseKey, float] = defaultdict(float)
        for t, x in vecs[q].items():
            for o, y in postings.get(t, ()):
                if o != q:
                    scores[o] += x * y
        top = sorted(((o, round(s, 6)) for o, s in scores.items() if s >= SIM_FLOOR), key=lambda e: (-e[1], e[0]))
        out[q] = top[:k]
    return out


def bridge_candidates(state: ConceptState, *, families: Sequence[str] = FIRST_PASS, k: int = K_DEFAULT,
                      sense_checks: str = "all", finder: Finder | None = None) -> list[Candidate]:
    """Sense checks then neighbour pairs, family by family in bridge order (module doc)."""
    if sense_checks not in SENSE_CHECK_MODES:
        raise ValueError(f"sense_checks must be one of {SENSE_CHECK_MODES}, not {sense_checks!r}")
    finder = finder or keyword_neighbours
    order = sorted(families, key=lambda f: (BY_NAME[f].bridge_order or 99, f))
    out: list[Candidate] = []
    seen_pairs: set[frozenset[str]] = set()
    for family in order:
        keys = sorted(k_ for k_ in state.senses if k_.family == family)
        checks: list[Candidate] = []
        if sense_checks != "none":
            for key in keys:
                s = state.senses[key]
                if s.checked or key.sense_id != DEFAULT_SENSE or len(s.papers) < 2:
                    continue
                if s.ambiguous:
                    checks.append(Candidate("sense_check", key, reason="ambiguous", family=family))
                elif sense_checks == "all" and _spread(state, s):
                    checks.append(Candidate("sense_check", key, reason="spread", family=family))
        checks.sort(key=lambda c: (c.reason != "ambiguous", -len(state.senses[c.a].papers), c.a))
        out += checks
        adjacent = list(ADJACENT.get(family, ()))
        pool = keys + sorted(k_ for k_ in state.senses if k_.family in adjacent)
        found = finder(state, keys, pool, k)
        seen: dict[tuple[SenseKey, SenseKey], float] = {}
        for q in keys:
            for o, sim in found.get(q, ()):
                lo, hi = sorted((q, o))
                seen[(lo, hi)] = max(seen.get((lo, hi), 0.0), sim)
        pairs = []
        for (x, y), sim in seen.items():
            if _droppable(state, x, y):
                continue
            d = state.citations.distance(state.senses[x].papers, state.senses[y].papers)
            pairs.append(Candidate("pair", x, y, distance=d, similarity=sim, family=family))
        pairs.sort(key=lambda c: (-distance_rank(c.distance), -c.similarity, c.a, c.b))
        out += _once_per_concept_pair(state, pairs, seen_pairs)
    return out


def _spread(state: ConceptState, s: Sense) -> bool:
    if state.literature_of:
        lits = {state.literature_of[p] for p in s.papers if p in state.literature_of}
        if len(lits) >= 2:
            return True
    return state.citations.spread(s.papers)
