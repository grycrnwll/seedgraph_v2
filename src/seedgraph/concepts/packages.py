"""Concept packages (stage 4 design §3.3, §4.4-§4.5): items with example claims, the package
text the agent reads, the frozen manifest ``check`` and ``save`` read back, and the
budget cut.

One function, :func:`build`, turns ordered candidates into packages. The task's ``select``
/ ``plan`` and the dry run (:mod:`seedgraph.concepts.estimate`) both call it, so the dry
run's pair, package and token counts equal the created batch's exactly.

**Items.** Each side of a pair shows its family, label and paper count, and
:data:`E_DEFAULT` example claim (:data:`E_RICH` for ambiguous labels and senses with at
least :data:`REACH_MIN` papers); a sense check lists up to :data:`E_CHECK` claims. Examples
come from different papers (and literatures, in a graph) where possible, and from papers
the other side does not use. Every example carries a short claim ref (``c17``), the claim
text, its plain words when present, the anchored quote (at most :data:`QUOTE_CHARS`
characters, centred) and the paper's short title, year and literature.

**Packages** hold :data:`PAIRS_PER_PACKAGE` pairs, :data:`CHECKS_PER_PACKAGE` sense
checks or :data:`GROUPS_PER_PACKAGE` lexical groups (a group is never split), one family
and one item kind each. The subject id names them (``assumption-pairs-00051-00100``) and
the subject label reads "assumption pairs 51-100".

**Manifest.** Fingerprints and span ids stay out of the text the agent reads. Each
package's manifest (items, sides' sense keys, ref -> ``{paper_id, claim_id, span_id,
claim_fingerprint}``) is frozen in its own content-addressed file, named from the batch's
``labels.json`` through a small index (:func:`seedgraph.work.tasks_concepts.write_manifests`),
so checking or saving one package reads only that package's manifest.

**Budget.** With ``budget_tokens`` the items are taken in priority order while the running
**high** agent-token estimate (:func:`agent_tokens`) stays within the budget; the rest wait
for the next pass. The cut is at item granularity (a lexical group counts as one unit).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..llm.tokens import estimate_tokens
from .candidates import Candidate, ConceptState, distance_rank
from .decide import REACH_MIN
from .model import DEFAULT_SENSE, SenseKey

E_DEFAULT = 1
E_RICH = 2
E_CHECK = 4
QUOTE_CHARS = 240
TITLE_CHARS = 60
PAIRS_PER_PACKAGE = 50
CHECKS_PER_PACKAGE = 25
GROUPS_PER_PACKAGE = 12
#: Agent tokens per package-text token (the stage 2 measured overhead range).
RHO_LOW = 1.5
RHO_HIGH = 2.7
#: Answer tokens per item (a compact JSON answer line).
ANSWER_TOKENS_LOW = 30
ANSWER_TOKENS_HIGH = 70

KIND_WORDS = {"pair": "pairs", "sense_check": "sense checks", "group": "groups", "literature": "literatures"}
KIND_SLUGS = {"pair": "pairs", "sense_check": "checks", "group": "groups", "literature": "literatures"}


# --- value types ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Example:
    ref: str
    claim_id: str
    paper_id: str
    span_id: str | None
    fingerprint: str
    text: str
    plain_words: str | None
    quote: str | None
    title: str | None
    year: int | None
    literature: str | None

    def manifest(self) -> dict[str, Any]:
        return {"paper_id": self.paper_id, "claim_id": self.claim_id, "span_id": self.span_id,
                "claim_fingerprint": self.fingerprint}


@dataclass(frozen=True)
class Side:
    key: SenseKey
    label: str
    papers: int
    ambiguous: bool
    examples: tuple[Example, ...]


@dataclass(frozen=True)
class Item:
    """``kind``: ``pair`` / ``sense_check`` / ``literature``. ``unit`` groups items that
    must stay in one package (a lexical group)."""

    kind: str
    family: str
    a: Side | None = None
    b: Side | None = None
    group: int | None = None
    distance: int | None = None
    reason: str | None = None
    literature: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Package:
    subject_id: str
    label: str
    task: str
    family: str
    kind: str                       # pair | sense_check | group | literature
    items: tuple[Item, ...]         # with refs numbered for this package
    text: str
    est_tokens: int

    @property
    def agent_tokens_low(self) -> int:
        return agent_tokens(self.est_tokens, len(self.items), high=False)

    @property
    def agent_tokens_high(self) -> int:
        return agent_tokens(self.est_tokens, len(self.items), high=True)

    def manifest(self) -> dict[str, Any]:
        return {"subject_id": self.subject_id, "task": self.task, "family": self.family, "kind": self.kind,
                "items": [_item_manifest(n, it) for n, it in enumerate(self.items, start=1)]}


def agent_tokens(text_tokens: int, n_items: int, *, high: bool, instructions_tokens: int = 0) -> int:
    rho = RHO_HIGH if high else RHO_LOW
    answer = ANSWER_TOKENS_HIGH if high else ANSWER_TOKENS_LOW
    return round(rho * (text_tokens + instructions_tokens) + answer * n_items)


def _side_manifest(s: Side | None) -> dict[str, Any] | None:
    if s is None:
        return None
    return {"family": s.key.family, "label_key": s.key.label_key, "sense_id": s.key.sense_id, "label": s.label,
            "papers": s.papers, "ambiguous": s.ambiguous, "refs": {e.ref: e.manifest() for e in s.examples}}


def _item_manifest(n: int, it: Item) -> dict[str, Any]:
    return {"item": n, "kind": it.kind, "family": it.family, "group": it.group, "distance": it.distance,
            "reason": it.reason, "a": _side_manifest(it.a), "b": _side_manifest(it.b),
            "literature": dict(it.literature) if it.literature else None}


# --- examples -------------------------------------------------------------------------------------

def _quote(text: str | None) -> str | None:
    if not text:
        return None
    if len(text) <= QUOTE_CHARS:
        return text
    start = (len(text) - QUOTE_CHARS) // 2
    return "..." + text[start:start + QUOTE_CHARS - 6].strip() + "..."


def _n_examples(state: ConceptState, key: SenseKey, kind: str) -> int:
    if kind == "sense_check":
        return E_CHECK
    s = state.senses[key]
    return E_RICH if s.ambiguous or len(s.papers) >= REACH_MIN else E_DEFAULT


def _pick(state: ConceptState, key: SenseKey, n: int, avoid: Iterable[str] = ()) -> list[str]:
    """Up to ``n`` claim ids of the sense: quoted claims first, one per paper, papers the
    other side uses last, new literatures first (deterministic)."""
    s = state.senses[key]
    avoid = set(avoid)
    claims = [state.claims[c] for c in s.claims if c in state.claims]
    claims.sort(key=lambda c: (c.paper_id in avoid, c.quote is None, c.paper_id, c.claim_id))
    out: list[str] = []
    papers: set[str] = set()
    lits: set[str] = set()
    pool = list(claims)
    while pool and len(out) < n:
        fresh = [c for c in pool if c.paper_id not in papers]
        if not fresh:
            break
        new_lit = [c for c in fresh if state.literature_of.get(c.paper_id) not in lits] if state.literature_of else fresh
        c = (new_lit or fresh)[0]
        out.append(c.claim_id)
        papers.add(c.paper_id)
        if state.literature_of.get(c.paper_id):
            lits.add(state.literature_of[c.paper_id])
        pool.remove(c)
    return out


def _side(state: ConceptState, key: SenseKey, claim_ids: Sequence[str]) -> Side:
    s = state.senses[key]
    examples = []
    for cid in claim_ids:
        c = state.claims[cid]
        examples.append(Example("", c.claim_id, c.paper_id, c.span_id, c.fingerprint, c.text, c.plain_words,
                                _quote(c.quote), c.title, c.year, state.literature_of.get(c.paper_id)))
    return Side(key, state.surface(key), len(s.papers), s.ambiguous, tuple(examples))


def make_item(state: ConceptState, cand: Candidate) -> Item:
    if cand.kind == "sense_check":
        a = _side(state, cand.a, _pick(state, cand.a, E_CHECK))
        return Item("sense_check", cand.family or cand.a.family, a, None, reason=cand.reason)
    ca = _pick(state, cand.a, _n_examples(state, cand.a, "pair"))
    assert cand.b is not None
    papers_a = {state.claims[c].paper_id for c in ca}
    cb = _pick(state, cand.b, _n_examples(state, cand.b, "pair"), avoid=papers_a)
    return Item("pair", cand.family or cand.a.family, _side(state, cand.a, ca), _side(state, cand.b, cb),
                group=cand.group, distance=cand.distance)


def _number(items: Sequence[Item]) -> tuple[Item, ...]:
    """Assign refs ``c1..cn`` in reading order."""
    n = 0
    out = []
    for it in items:
        sides = []
        for s in (it.a, it.b):
            if s is None:
                sides.append(None)
                continue
            exs = []
            for e in s.examples:
                n += 1
                exs.append(Example(f"c{n}", *[getattr(e, f) for f in Example.__dataclass_fields__ if f != "ref"]))
            sides.append(Side(s.key, s.label, s.papers, s.ambiguous, tuple(exs)))
        out.append(Item(it.kind, it.family, sides[0], sides[1], it.group, it.distance, it.reason, it.literature))
    return tuple(out)


# --- text -----------------------------------------------------------------------------------------

def _paper_line(e: Example) -> str:
    title = (e.title or e.paper_id).strip()
    if len(title) > TITLE_CHARS:
        title = title[:TITLE_CHARS - 3].rstrip() + "..."
    bits = [title + (f" ({e.year})" if e.year else "")]
    if e.literature:
        bits.append(f"literature: {e.literature}")
    return "; ".join(bits)


def _example_lines(s: Side) -> list[str]:
    lines = []
    for e in s.examples:
        parts = [f"- {e.ref}: {e.text}"]
        if e.plain_words:
            parts.append(f"  plain words: {e.plain_words}")
        if e.quote:
            parts.append(f'  quote: "{e.quote}"')
        parts.append(f"  paper: {_paper_line(e)}")
        lines += parts
    return lines


def _side_header(tag: str, s: Side) -> str:
    sense = "" if s.key.sense_id == DEFAULT_SENSE else f" (sense {s.key.sense_id})"
    amb = " [ambiguous label: compare the claims, never the label alone]" if s.ambiguous else ""
    return f"{tag}: {s.key.family} / \"{s.label}\"{sense}, {s.papers} paper{'s' if s.papers != 1 else ''}{amb}"


def _distance_text(d: int | None) -> str:
    return "more than 3" if d is None else str(d)


def item_text(n: int, it: Item) -> str:
    if it.kind == "literature":
        lit = it.literature or {}
        lines = [f"## {n}. literature {lit.get('ordinal')} ({lit.get('size')} papers)", "Most cited papers:"]
        lines += [f"- {p}" for p in lit.get("papers", ())]
        lines.append("Distinctive concepts: " + ("; ".join(lit.get("concepts", ())) or "none yet"))
        return "\n".join(lines) + "\n"
    if it.kind == "sense_check":
        assert it.a is not None
        why = "ambiguous label" if it.reason == "ambiguous" else "papers far apart in the citation graph"
        lines = [f"## {n}. sense check ({why})", _side_header("Label", it.a)] + _example_lines(it.a)
        return "\n".join(lines) + "\n"
    assert it.a is not None and it.b is not None
    head = f"## {n}. pair"
    if it.distance is not None or it.group is None:
        head += f" (citation distance {_distance_text(it.distance)})"
    lines = [head, _side_header("A", it.a)] + _example_lines(it.a) + [_side_header("B", it.b)] + _example_lines(it.b)
    return "\n".join(lines) + "\n"


def render(task: str, label: str, kind: str, items: Sequence[Item]) -> str:
    lines = [f"# Concept package: {label}", "",
             f"Task {task}. Answer every numbered item once, citing claim refs (c1, c2, ...) only; never write "
             "quotes. The rules and the answer schema are in instructions.md.", ""]
    body = []
    last_group = None
    for n, it in enumerate(items, start=1):
        if kind == "group" and it.group != last_group:
            labels = sorted({s.label for x in items if x.group == it.group for s in (x.a, x.b) if s})
            body.append(f"### Lexical group {it.group}: " + "; ".join(f'"{lab}"' for lab in labels) + "\n")
            last_group = it.group
        body.append(item_text(n, it))
    return "\n".join(lines) + "\n".join(body)


# --- packaging ------------------------------------------------------------------------------------

@dataclass
class BuildResult:
    packages: list[Package]
    items_total: int                 # candidates before the budget cut (units: items)
    items_packaged: int
    cut_at: int | None = None        # the first item (1-based, priority order) left out by the budget
    stats: dict[str, Any] = field(default_factory=dict)


def _unit_key(task_kind: str, it: Item) -> tuple:
    return (it.family, it.kind, it.group) if task_kind == "group" else (it.family, it.kind, id(it))


def _capacity(kind: str) -> int:
    return {"pair": PAIRS_PER_PACKAGE, "sense_check": CHECKS_PER_PACKAGE, "group": GROUPS_PER_PACKAGE,
            "literature": 10_000}[kind]


def _package_kind(task_kind: str, it: Item) -> str:
    return "group" if task_kind == "group" else it.kind


def build(task: str, items: Sequence[Item], *, task_kind: str = "pair", budget_tokens: int | None = None,
          subject_prefix: str = "", instructions_tokens: int = 0) -> BuildResult:
    """Package ``items`` (priority order). ``task_kind='group'`` keeps lexical groups whole
    and counts :data:`GROUPS_PER_PACKAGE` groups per package."""
    # units: consecutive items that must stay together
    units: list[list[Item]] = []
    for it in items:
        if units and task_kind == "group" and _unit_key(task_kind, units[-1][0]) == _unit_key(task_kind, it):
            units[-1].append(it)
        else:
            units.append([it])
    # budget: running high estimate over the units, per-package overhead counted once
    kept: list[list[Item]] = []
    cut_at = None
    if budget_tokens is not None:
        spent = 0
        open_key = None
        open_count = 0
        overhead = estimate_tokens(render(task, "x" * 30, "pair", []))
        for idx, unit in enumerate(units):
            pk = (unit[0].family, _package_kind(task_kind, unit[0]))
            new_pkg = open_key != pk or open_count >= _capacity(pk[1])
            cost = sum(agent_tokens(estimate_tokens(item_text(99, it)), 1, high=True) for it in unit)
            if new_pkg:
                cost += agent_tokens(overhead, 0, high=True, instructions_tokens=instructions_tokens)
            if spent + cost > budget_tokens:
                cut_at = sum(len(u) for u in units[:idx]) + 1
                break
            spent += cost
            kept.append(unit)
            if new_pkg:
                open_key, open_count = pk, 0
            open_count += 1
    else:
        kept = list(units)
    packages = _render_units(task, kept, task_kind, subject_prefix)
    if budget_tokens is not None:  # the running estimate is approximate: trim until the real total fits
        while kept and total_high(packages, instructions_tokens) > budget_tokens:
            kept.pop()
            cut_at = sum(len(u) for u in kept) + 1
            packages = _render_units(task, kept, task_kind, subject_prefix)
    packaged = sum(len(p.items) for p in packages)
    return BuildResult(packages, len(items), packaged, cut_at)


def total_high(packages: Sequence[Package], instructions_tokens: int) -> int:
    """The high agent-token estimate of a batch (instructions read once per package)."""
    return sum(p.agent_tokens_high for p in packages) + round(RHO_HIGH * instructions_tokens) * len(packages)


def total_low(packages: Sequence[Package], instructions_tokens: int) -> int:
    """The low agent-token estimate of a batch (instructions read once)."""
    return sum(p.agent_tokens_low for p in packages) + (round(RHO_LOW * instructions_tokens) if packages else 0)


def _render_units(task: str, units: Sequence[Sequence[Item]], task_kind: str, subject_prefix: str) -> list[Package]:
    packages: list[Package] = []
    counters: dict[tuple[str, str], int] = {}
    current: list[Sequence[Item]] = []

    def flush() -> None:
        if not current:
            return
        first = current[0][0]
        pkind = _package_kind(task_kind, first)
        flat = [it for unit in current for it in unit]
        key = (first.family, pkind)
        start = counters.get(key, 0) + 1
        count = len(current) if pkind == "group" else len(flat)
        if pkind == "group":
            start_no, end_no = current[0][0].group, current[-1][0].group
        else:
            start_no, end_no = start, start + count - 1
        counters[key] = counters.get(key, 0) + count
        label = f"{first.family} {KIND_WORDS[pkind]} {start_no}-{end_no}"
        sid = f"{subject_prefix}{first.family}-{KIND_SLUGS[pkind]}-{start_no:05d}-{end_no:05d}"
        numbered = _number(flat)
        text = render(task, label, pkind, numbered)
        packages.append(Package(sid, label, task, first.family, pkind, numbered, text, estimate_tokens(text)))
        current.clear()

    for unit in units:
        key = (unit[0].family, _package_kind(task_kind, unit[0]))
        if current:
            ck = (current[0][0].family, _package_kind(task_kind, current[0][0]))
            full = (len(current) if key[1] == "group" else sum(len(u) for u in current)) >= _capacity(key[1])
            if ck != key or full:
                flush()
        current.append(unit)
    flush()
    return packages


def from_candidates(state: ConceptState, candidates: Sequence[Candidate], task: str, *, task_kind: str = "pair",
                    budget_tokens: int | None = None, instructions_tokens: int = 0) -> BuildResult:
    items = [make_item(state, c) for c in candidates]
    res = build(task, items, task_kind=task_kind, budget_tokens=budget_tokens,
                instructions_tokens=instructions_tokens)
    res.stats = stats(candidates[:res.items_packaged] if res.cut_at else candidates, res)
    return res


def stats(candidates: Sequence[Candidate], res: BuildResult) -> dict[str, Any]:
    """Counts per family (pairs, sense checks, groups, packages) and the share of pairs at
    each citation distance."""
    fam: dict[str, dict[str, int]] = {}
    for c in candidates:
        f = fam.setdefault(c.family or c.a.family, {"pairs": 0, "sense_checks": 0, "groups": 0, "packages": 0})
        f["pairs" if c.kind == "pair" else "sense_checks"] += 1
    groups: dict[str, set[int]] = {}
    for c in candidates:
        if c.group is not None:
            groups.setdefault(c.family or c.a.family, set()).add(c.group)
    for f, gs in groups.items():
        fam[f]["groups"] = len(gs)
    for p in res.packages:
        fam.setdefault(p.family, {"pairs": 0, "sense_checks": 0, "groups": 0, "packages": 0})["packages"] += 1
    dist: dict[str, int] = {}
    pairs = [c for c in candidates if c.kind == "pair" and c.group is None]
    for c in pairs:
        k = "inf" if c.distance is None else str(c.distance)
        dist[k] = dist.get(k, 0) + 1
    order = sorted(dist, key=lambda k: -distance_rank(None if k == "inf" else int(k)))
    share = {k: round(dist[k] / len(pairs), 4) for k in order} if pairs else {}
    return {"families": fam, "distance_share": share}


# --- literatures (concepts:name_literatures) ------------------------------------------------------

TOP_PAPERS = 10
TOP_CONCEPTS = 8
NAMED_FAMILIES = ("assumption", "estimand", "method", "result", "data")


def literature_build(conn, graph: str | None, *, include_named: bool = False, library_root=None) -> BuildResult:
    """One package naming the literatures of ``graph``'s latest grouping: each shows its
    :data:`TOP_PAPERS` most cited papers (in-graph citations) and its :data:`TOP_CONCEPTS`
    most distinctive supported, sense-confirmed concepts (slice tf-idf). Literatures with a
    researcher name are skipped (a researcher rename wins); so are agent-named ones unless
    ``include_named``. The manifest freezes the run id: literature ids change per regroup."""
    import math
    from collections import Counter

    from ..library import graphs, literatures
    from ..library.store import Library
    from . import refresh as rf

    if not graph:
        raise ValueError("concepts:name_literatures needs a context graph (graph=<slug>)")
    lib = Library(root=library_root, conn=conn)
    with graphs.open_graph(lib, graph) as gr:
        run = literatures.latest_run(gr)
        lits = literatures.literatures(gr) if run else []
        cites = Counter(c.target_paper_id for c in graphs.graph_citations(gr))
        members = {p.paper_id for p in graphs.membership(gr, status="member")}
    plan = rf.plan(conn)
    names = {r[0]: r[2] for r in plan.concepts if r[7] and not r[9] and r[1] in NAMED_FAMILIES}
    concept_papers: dict[str, set[str]] = {}
    for _cid, paper, _fam, _key, _sense, cid, supports, visible in plan.claim_concepts:
        if cid in names and supports and visible and paper in members:
            concept_papers.setdefault(cid, set()).add(paper)
    n_members = max(1, len(members))
    titles = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT paper_id, title, year FROM papers")}
    items: list[Item] = []
    for lit in lits:
        if lit.researcher_name or (lit.agent_name and not include_named):
            continue
        mem = set(lit.members)
        top = sorted(mem, key=lambda p: (-cites.get(p, 0), p))[:TOP_PAPERS]
        papers = []
        for p in top:
            t, y = titles.get(p, (None, None))
            papers.append(f"{(t or p)[:TITLE_CHARS]}" + (f" ({y})" if y else ""))
        scored = []
        for cid, ps in concept_papers.items():
            tf = len(ps & mem)
            if tf:
                scored.append((-(tf * math.log(1 + n_members / len(ps))), names[cid], cid))
        concepts = [name for _, name, _ in sorted(scored)[:TOP_CONCEPTS]]
        items.append(Item("literature", "literatures", literature={
            "literature_id": lit.literature_id, "run_id": lit.run_id, "ordinal": lit.ordinal, "size": lit.size,
            "papers": papers, "concepts": concepts, "graph": graph}))
    if not items or run is None:
        return BuildResult([], 0, 0)
    label = f"literatures of {graph} (run {run.run_id})"
    numbered = tuple(items)
    text = render("concepts:name_literatures", label, "literature", numbered)
    sid = f"literatures-{graph}-run{run.run_id}"
    return BuildResult([Package(sid, label, "concepts:name_literatures", "literatures", "literature", numbered,
                                text, estimate_tokens(text))], len(items), len(items))
