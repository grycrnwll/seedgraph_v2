"""Literatures: citation-based grouping of a context graph's member papers (v3 design §6.2).

Input is the in-graph citation view (members only, undirected, deduplicated). Method is
seeded Louvain (networkx) over a resolution ladder; the lowest rung whose literatures meet
the size rule wins, else the rung with the best (gamma = 1) modularity.

- **Unplaced**: members with no in-graph citation, plus communities smaller than
  ``MIN_LIT``. Shown as a list, never a literature.
- **Stability**: the chosen rung is re-run with other seeds; the mean adjusted Rand index
  (and NMI) against seed 0 is reported, over every citation-connected paper using the raw
  Louvain labels (small communities are not lumped together, which would add fake
  agreement).
- **Fingerprint**: sha256 over the sorted member ids and the canonical edge list. It
  changes when membership or in-graph citations change; a stored run whose fingerprint
  differs is stale.
- **Regroup** stores a run, matching new literatures to the previous run's by member
  Jaccard >= 0.5 (greedy, one-to-one), which carries agent and researcher names forward.

Determinism: members are sorted, edges are canonicalised ``(min, max)``, de-duplicated,
sorted, and the graph is built in that order over integer labels before seeding Louvain,
so shuffled or duplicated input gives the same partition. (Ported from v2's
``graph/analyze.py`` recipe, which only sorted nodes.)
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import networkx as nx

from .graphs import Graph, GraphError, _write, graph_citations, membership

ALGORITHM = "louvain"
LADDER: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
MIN_LIT = 5
MAX_SHARE = 0.30
MIN_LITERATURES = 3
BASE_SEED = 0
STABILITY_SEEDS: tuple[int, ...] = tuple(range(1, 10))
UNSTABLE_BELOW = 0.6
JACCARD_CARRY = 0.5

ChosenBy = Literal["rule", "researcher"]


# --- pure core ---------------------------------------------------------------


def _canonical(members: Iterable[str], edges: Iterable[tuple[str, str]]) -> tuple[list[str], list[tuple[str, str]]]:
    nodes = sorted(set(members))
    present = set(nodes)
    canon = {
        (min(u, v), max(u, v))
        for u, v in edges
        if u != v and u in present and v in present
    }
    return nodes, sorted(canon)


def citation_fingerprint(members: Iterable[str], edges: Iterable[tuple[str, str]]) -> str:
    """sha256 over sorted members and canonical undirected edges (membership AND citations)."""
    nodes, canon = _canonical(members, edges)
    h = hashlib.sha256()
    for n in nodes:
        h.update(b"m\t" + n.encode("utf-8") + b"\n")
    for u, v in canon:
        h.update(b"e\t" + u.encode("utf-8") + b"\t" + v.encode("utf-8") + b"\n")
    return h.hexdigest()


def _comb2(n: int) -> float:
    return n * (n - 1) / 2.0


def adjusted_rand_index(a: Mapping[Hashable, Hashable], b: Mapping[Hashable, Hashable]) -> float:
    """Hubert-Arabie ARI over the nodes both labellings share."""
    keys = sorted(set(a) & set(b), key=str)
    n = len(keys)
    if n < 2:
        return 1.0
    cells = Counter((a[k], b[k]) for k in keys)
    rows = Counter(a[k] for k in keys)
    cols = Counter(b[k] for k in keys)
    index = sum(_comb2(c) for c in cells.values())
    sr = sum(_comb2(c) for c in rows.values())
    sc = sum(_comb2(c) for c in cols.values())
    expected = sr * sc / _comb2(n)
    max_index = (sr + sc) / 2.0
    if max_index == expected:  # both trivial (all one cluster or all singletons)
        return 1.0
    return (index - expected) / (max_index - expected)


def normalized_mutual_info(a: Mapping[Hashable, Hashable], b: Mapping[Hashable, Hashable]) -> float:
    """NMI with arithmetic-mean normalisation over the shared nodes."""
    keys = sorted(set(a) & set(b), key=str)
    n = len(keys)
    if n == 0:
        return 1.0
    cells = Counter((a[k], b[k]) for k in keys)
    rows = Counter(a[k] for k in keys)
    cols = Counter(b[k] for k in keys)

    def entropy(c: Counter) -> float:
        return -sum((v / n) * math.log(v / n) for v in c.values())

    ha, hb = entropy(rows), entropy(cols)
    if ha == 0.0 and hb == 0.0:
        return 1.0
    mi = sum(
        (v / n) * math.log((v * n) / (rows[x] * cols[y])) for (x, y), v in cells.items()
    )
    return mi / ((ha + hb) / 2.0)


@dataclass(frozen=True)
class Rung:
    """One ladder rung: raw community sizes (connected papers only) and gamma=1 modularity."""

    resolution: float
    sizes: tuple[int, ...]
    modularity: float


def meets_size_rule(sizes: Sequence[int]) -> bool:
    """Largest literature <= 30% of placed papers and >= 3 literatures of >= MIN_LIT."""
    big = [s for s in sizes if s >= MIN_LIT]
    placed = sum(big)
    if len(big) < MIN_LITERATURES or placed == 0:
        return False
    return max(big) <= MAX_SHARE * placed


def choose_rung(rungs: Sequence[Rung]) -> tuple[Rung, bool]:
    """The lowest-resolution rung meeting the size rule, else the best-modularity rung."""
    if not rungs:
        raise ValueError("no rungs to choose from")
    ordered = sorted(rungs, key=lambda r: r.resolution)
    for r in ordered:
        if meets_size_rule(r.sizes):
            return r, True
    return max(ordered, key=lambda r: (r.modularity, -r.resolution)), False


@dataclass(frozen=True)
class LiteraturePlan:
    """The result of planning literatures; nothing is written."""

    resolution: float
    seed: int
    modularity: float
    rule_met: bool
    groups: tuple[tuple[str, ...], ...]  # by size desc, then smallest member id
    unplaced: tuple[str, ...]
    rungs: tuple[Rung, ...]
    ari_mean: float | None
    nmi_mean: float | None
    stability_seeds: tuple[int, ...]
    fingerprint: str

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(len(g) for g in self.groups)

    @property
    def unstable(self) -> bool:
        return self.ari_mean is not None and self.ari_mean < UNSTABLE_BELOW


class _Prepared:
    """Canonical, integer-labelled citation graph of the connected members."""

    def __init__(self, members: Iterable[str], edges: Iterable[tuple[str, str]]):
        self.nodes, self.edges = _canonical(members, edges)
        connected = sorted({p for e in self.edges for p in e})
        self.isolated = tuple(sorted(set(self.nodes) - set(connected)))
        self.ids = connected
        index = {p: i for i, p in enumerate(connected)}
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(len(connected)))
        self.graph.add_edges_from((index[u], index[v]) for u, v in self.edges)

    def partition(self, resolution: float, seed: int) -> list[frozenset[int]]:
        if self.graph.number_of_edges() == 0:
            return []
        comms = nx.community.louvain_communities(self.graph, resolution=resolution, seed=seed)
        return [frozenset(c) for c in comms]

    def labels(self, comms: list[frozenset[int]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for cid, c in enumerate(comms):
            for i in c:
                out[self.ids[i]] = cid
        return out

    def modularity(self, comms: list[frozenset[int]]) -> float:
        if not comms:
            return 0.0
        return float(nx.community.modularity(self.graph, comms, resolution=1.0))


def plan_literatures(
    members: Iterable[str],
    edges: Iterable[tuple[str, str]],
    *,
    resolution: float | None = None,
    seed: int = BASE_SEED,
    stability_seeds: Iterable[int] = STABILITY_SEEDS,
) -> LiteraturePlan:
    """Plan literatures from a member list and (directed or not) citation pairs.

    ``resolution=None`` walks :data:`LADDER` and applies the size rule; an explicit
    resolution (a researcher's "fewer / more" step) is used as given.
    """
    prep = _Prepared(members, edges)
    fp = citation_fingerprint(prep.nodes, prep.edges)
    stability = tuple(stability_seeds)

    def rung(gamma: float) -> tuple[Rung, list[frozenset[int]]]:
        comms = prep.partition(gamma, seed)
        sizes = tuple(sorted((len(c) for c in comms), reverse=True))
        return Rung(gamma, sizes, prep.modularity(comms)), comms

    computed = {g: rung(g) for g in LADDER}
    if resolution is None:
        chosen, rule_met = choose_rung([r for r, _ in computed.values()])
    else:
        if resolution not in computed:
            computed[resolution] = rung(resolution)
        chosen = computed[resolution][0]
        rule_met = meets_size_rule(chosen.sizes)
    comms = computed[chosen.resolution][1]

    groups = sorted(
        (tuple(sorted(prep.ids[i] for i in c)) for c in comms if len(c) >= MIN_LIT),
        key=lambda g: (-len(g), g[0]),
    )
    small = [prep.ids[i] for c in comms if len(c) < MIN_LIT for i in c]
    unplaced = tuple(sorted(set(prep.isolated) | set(small)))

    ari = nmi = None
    if stability and comms:
        base = prep.labels(comms)
        aris, nmis = [], []
        for s in stability:
            other = prep.labels(prep.partition(chosen.resolution, s))
            aris.append(adjusted_rand_index(base, other))
            nmis.append(normalized_mutual_info(base, other))
        ari, nmi = sum(aris) / len(aris), sum(nmis) / len(nmis)

    return LiteraturePlan(
        resolution=chosen.resolution,
        seed=seed,
        modularity=chosen.modularity,
        rule_met=rule_met,
        groups=tuple(groups),
        unplaced=unplaced,
        rungs=tuple(sorted((r for r, _ in computed.values()), key=lambda r: r.resolution)),
        ari_mean=ari,
        nmi_mean=nmi,
        stability_seeds=stability,
        fingerprint=fp,
    )


# --- per graph: plan, regroup, names -----------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _graph_input(g: Graph) -> tuple[list[str], list[tuple[str, str]]]:
    members = [p.paper_id for p in membership(g, status="member")]
    edges = [(c.source_paper_id, c.target_paper_id) for c in graph_citations(g)]
    return members, edges


def current_fingerprint(g: Graph) -> str:
    """Fingerprint of the graph's members and in-graph citations as they are now."""
    return citation_fingerprint(*_graph_input(g))


def plan_for_graph(
    g: Graph,
    *,
    resolution: float | None = None,
    stability_seeds: Iterable[int] = STABILITY_SEEDS,
) -> LiteraturePlan:
    """Plan the graph's literatures from its current citation view. Writes nothing."""
    members, edges = _graph_input(g)
    return plan_literatures(members, edges, resolution=resolution, stability_seeds=stability_seeds)


@dataclass(frozen=True)
class LiteratureRun:
    run_id: int
    algorithm: str
    resolution: float
    chosen_by: str
    rule_met: bool
    seed: int
    modularity: float | None
    ari_mean: float | None
    nmi_mean: float | None
    stability_seeds: tuple[int, ...]
    citation_sha256: str
    built_at: str

    @property
    def unstable(self) -> bool:
        return self.ari_mean is not None and self.ari_mean < UNSTABLE_BELOW


@dataclass(frozen=True)
class Literature:
    literature_id: str
    run_id: int
    ordinal: int
    size: int
    members: tuple[str, ...]
    agent_name: str | None
    agent_description: str | None
    researcher_name: str | None
    named_at: str | None
    carried_from: str | None

    @property
    def name_source(self) -> Literal["researcher", "agent"] | None:
        if self.researcher_name:
            return "researcher"
        if self.agent_name:
            return "agent"
        return None

    @property
    def name(self) -> str:
        return self.researcher_name or self.agent_name or f"unnamed ({self.size} papers)"


_RUN_COLS = (
    "run_id, algorithm, resolution, chosen_by, rule_met, seed, modularity, ari_mean, "
    "nmi_mean, stability_seeds, citation_sha256, built_at"
)


def _run(row) -> LiteratureRun:
    seeds = tuple(int(s) for s in row[9].split(",") if s) if row[9] else ()
    return LiteratureRun(
        row[0], row[1], row[2], row[3], bool(row[4]), row[5], row[6], row[7], row[8],
        seeds, row[10], row[11],
    )


def latest_run(g: Graph) -> LiteratureRun | None:
    row = g.conn.execute(
        f"SELECT {_RUN_COLS} FROM literature_runs ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    return _run(row) if row else None


def is_stale(g: Graph) -> bool:
    """True when there is no grouping yet, or members / in-graph citations changed since."""
    run = latest_run(g)
    return run is None or run.citation_sha256 != current_fingerprint(g)


def literatures(g: Graph, run_id: int | None = None) -> list[Literature]:
    """The literatures of ``run_id`` (default: the latest run), by ordinal."""
    if run_id is None:
        run = latest_run(g)
        if run is None:
            return []
        run_id = run.run_id
    rows = g.conn.execute(
        "SELECT literature_id, run_id, ordinal, size, agent_name, agent_description, "
        "researcher_name, named_at, carried_from FROM literatures WHERE run_id = ? "
        "ORDER BY ordinal",
        (run_id,),
    ).fetchall()
    out = []
    for r in rows:
        members = tuple(
            m[0]
            for m in g.conn.execute(
                "SELECT paper_id FROM literature_members WHERE literature_id = ? ORDER BY paper_id",
                (r[0],),
            )
        )
        out.append(Literature(r[0], r[1], r[2], r[3], members, r[4], r[5], r[6], r[7], r[8]))
    return out


def unplaced(g: Graph) -> list[str]:
    """Current members in no literature of the latest run (sorted)."""
    placed = {p for lit in literatures(g) for p in lit.members}
    return [p.paper_id for p in membership(g, status="member") if p.paper_id not in placed]


def paper_literatures(g: Graph) -> dict[str, Literature]:
    """``{paper_id: literature}`` for the placed papers of the latest run."""
    return {p: lit for lit in literatures(g) for p in lit.members}


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _match(new: Sequence[tuple[str, ...]], old: Sequence[Literature]) -> dict[int, Literature]:
    """Greedy one-to-one match of new groups to old literatures with Jaccard >= 0.5.
    Highest Jaccard first; ties by new index, then old ordinal (deterministic)."""
    pairs = []
    for i, grp in enumerate(new):
        gs = frozenset(grp)
        for lit in old:
            j = _jaccard(gs, frozenset(lit.members))
            if j >= JACCARD_CARRY:
                pairs.append((-j, i, lit.ordinal, lit))
    pairs.sort(key=lambda t: t[:3])
    out: dict[int, Literature] = {}
    used: set[str] = set()
    for _, i, _, lit in pairs:
        if i in out or lit.literature_id in used:
            continue
        out[i] = lit
        used.add(lit.literature_id)
    return out


def regroup(
    g: Graph,
    *,
    resolution: float | None = None,
    reset_to_rule: bool = False,
    stability_seeds: Iterable[int] = STABILITY_SEEDS,
    built_at: str | None = None,
) -> LiteratureRun:
    """Recompute and store the graph's literatures as a new run.

    With no ``resolution``, a researcher's earlier rung choice is kept (unless
    ``reset_to_rule``); otherwise the ladder and size rule choose. An explicit
    ``resolution`` is recorded as the researcher's choice. Names carry forward by
    member Jaccard >= 0.5.
    """
    prev = latest_run(g)
    chosen_by: ChosenBy = "rule"
    if resolution is not None:
        chosen_by = "researcher"
    elif prev is not None and prev.chosen_by == "researcher" and not reset_to_rule:
        resolution, chosen_by = prev.resolution, "researcher"
    plan = plan_for_graph(g, resolution=resolution, stability_seeds=stability_seeds)
    matched = _match(plan.groups, literatures(g))
    with _write(g) as conn:
        cur = conn.execute(
            "INSERT INTO literature_runs (algorithm, resolution, chosen_by, rule_met, seed, "
            "modularity, ari_mean, nmi_mean, stability_seeds, citation_sha256, built_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ALGORITHM, plan.resolution, chosen_by, int(plan.rule_met), plan.seed,
                plan.modularity, plan.ari_mean, plan.nmi_mean,
                ",".join(str(s) for s in plan.stability_seeds), plan.fingerprint,
                built_at or _now(),
            ),
        )
        run_id = int(cur.lastrowid)
        for i, grp in enumerate(plan.groups):
            lit_id = f"lit_{run_id}_{i + 1}"
            src = matched.get(i)
            conn.execute(
                "INSERT INTO literatures (literature_id, run_id, ordinal, size, agent_name, "
                "agent_description, researcher_name, named_at, carried_from) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    lit_id, run_id, i + 1, len(grp),
                    src.agent_name if src else None,
                    src.agent_description if src else None,
                    src.researcher_name if src else None,
                    src.named_at if src else None,
                    src.literature_id if src else None,
                ),
            )
            conn.executemany(
                "INSERT INTO literature_members (literature_id, paper_id) VALUES (?, ?)",
                [(lit_id, p) for p in grp],
            )
    run = latest_run(g)
    assert run is not None
    return run


def step_resolution(
    g: Graph,
    direction: Literal["fewer", "more"],
    *,
    stability_seeds: Iterable[int] = STABILITY_SEEDS,
    built_at: str | None = None,
) -> LiteratureRun:
    """The researcher's "fewer / more literatures" control: regroup one ladder rung
    down / up from the current run, and keep that choice for later regroups."""
    if direction not in ("fewer", "more"):
        raise GraphError(f"direction must be 'fewer' or 'more', not {direction!r}")
    prev = latest_run(g)
    if prev is None:
        raise GraphError("no literatures yet; regroup first")
    if direction == "fewer":
        options = [r for r in LADDER if r < prev.resolution]
        target = max(options) if options else None
    else:
        options = [r for r in LADDER if r > prev.resolution]
        target = min(options) if options else None
    if target is None:
        raise GraphError(f"already at the {'lowest' if direction == 'fewer' else 'highest'} rung")
    return regroup(g, resolution=target, stability_seeds=stability_seeds, built_at=built_at)


def _literature_exists(g: Graph, literature_id: str) -> None:
    if g.conn.execute(
        "SELECT 1 FROM literatures WHERE literature_id = ?", (literature_id,)
    ).fetchone() is None:
        raise GraphError(f"no literature {literature_id!r} in graph {g.slug!r}")


def rename_literature(
    g: Graph, literature_id: str, name: str | None, *, named_at: str | None = None
) -> None:
    """The researcher's name for a literature; ``None`` clears it (back to the agent's
    name or "unnamed"). It overrides the agent name and survives regrouping."""
    _literature_exists(g, literature_id)
    if name is not None:
        name = name.strip()
        if not name:
            raise GraphError("literature name must not be empty")
    with _write(g) as conn:
        conn.execute(
            "UPDATE literatures SET researcher_name = ?, named_at = ? WHERE literature_id = ?",
            (name, named_at or _now(), literature_id),
        )


def set_agent_name(
    g: Graph,
    literature_id: str,
    name: str,
    description: str | None = None,
    *,
    named_at: str | None = None,
) -> None:
    """Record the naming agent's (agent inferred) name and one-line description. Hook
    for the ``concepts:name_literatures`` task; it never overrides a researcher name."""
    _literature_exists(g, literature_id)
    if not name or not name.strip():
        raise GraphError("literature name must not be empty")
    with _write(g) as conn:
        conn.execute(
            "UPDATE literatures SET agent_name = ?, agent_description = ?, named_at = ? "
            "WHERE literature_id = ?",
            (name.strip(), description, named_at or _now(), literature_id),
        )
