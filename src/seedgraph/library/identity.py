"""Library paper identity (design §4). Pure: no DB, no clock, no randomness.

Two jobs:

* **Key and id derivation** (§4.1). ``paper_id = 'lp_' + sha256(key)[:20]`` where the key
  is the first of the winning strong id (doi → openalex → arxiv), the primary PDF hash,
  ``title:<normalize_title>|<year>``, or ``v2:<slug>:<work_id>``. Collisions append
  ``:2``, ``:3`` … to the key.
* **Merge grouping** (§4.2–4.3). :func:`plan_identity` takes plain :class:`WorkRecord`
  values (one per v2 work, from any project) plus the researcher's "not duplicates"
  pairs, and returns an :class:`IdentityPlan`: the papers with their members, ids and
  winning fields, one record per automatic merge, the blocked and skipped unions, the
  shared PDFs, the id collisions and the "check these" items.

Everything iterates in sorted order (project precedence, then work id; key values
sorted), so the same input in any order gives an equal plan.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Collection, Iterable, Literal, NamedTuple, Sequence

from seedgraph.library.titlecheck import content_tokens, normalize_tokens, variants
from seedgraph.project.identity import normalize_title, strong_ids

#: Id types that join and conflict, in union-find order (§4.2). s2/ssrn are carried as
#: data only.
MERGE_ID_TYPES: tuple[str, ...] = ("doi", "openalex", "arxiv")
#: Every id type carried into ``identifiers``.
CARRIED_ID_TYPES: tuple[str, ...] = ("doi", "openalex", "arxiv", "s2", "ssrn")
#: v2 project precedence for field winners and iteration (§4.2). ``dbg_*`` projects
#: come next, then any other project, each group alphabetically.
DEFAULT_PROJECT_PRECEDENCE: tuple[str, ...] = ("second_test", "land_value", "test_proj", "oct2026test")
#: Nearest-title candidates listed on a "no match" check item.
NEAREST_CANDIDATES = 3

KeyType = Literal["doi", "openalex", "arxiv", "pdf", "title_year"]
CheckReason = Literal["no_match", "ambiguous_title", "ambiguous_pdf", "blocked_conflict", "never_merge_skipped"]


class WorkRef(NamedTuple):
    """A v2 work: ``(project slug, work_id)``."""

    project: str
    work_id: str


@dataclass(frozen=True)
class WorkRecord:
    """One v2 work as the identity planner sees it.

    ``ids`` holds raw ``(id_type, value)`` pairs, any number per type (from the works id
    columns and the ``identifiers`` table); they are normalised here. ``pdf_hashes`` are
    the sha256 of the PDFs the work holds (empty for a cited-only work).
    """

    project: str
    work_id: str
    title: str | None = None
    year: int | None = None
    authors: str | None = None
    venue: str | None = None
    abstract: str | None = None
    ids: tuple[tuple[str, str], ...] = ()
    pdf_hashes: tuple[str, ...] = ()

    @property
    def ref(self) -> WorkRef:
        return WorkRef(self.project, self.work_id)


@dataclass(frozen=True)
class MintedId:
    paper_id: str
    key: str  # the key actually hashed, suffix included
    collided: bool


@dataclass(frozen=True)
class PlannedPaper:
    """One library paper. ``members`` are in precedence order; ``ids`` are normalised,
    sorted ``(id_type, value)`` pairs; the field values are the winners (§4.2)."""

    paper_id: str
    key: str
    base_key: str
    members: tuple[WorkRef, ...]
    ids: tuple[tuple[str, str], ...]
    pdf_hashes: tuple[str, ...]
    title: str | None
    authors: str | None
    year: int | None
    venue: str | None
    abstract: str | None

    @property
    def held(self) -> bool:
        return bool(self.pdf_hashes)


@dataclass(frozen=True)
class RuleMerge:
    """An automatic merge: ``work_b`` joined ``work_a``'s group through this key.
    Becomes a ``duplicate_decisions`` row (``decided_by='rule'``, ``reason`` = key)."""

    work_a: WorkRef
    work_b: WorkRef
    key_type: KeyType
    key_value: str


@dataclass(frozen=True)
class BlockedUnion:
    """A union refused because the groups hold different values of an id type.
    ``key_type == 'pdf'`` means different papers sharing one PDF (§7.3)."""

    work_a: WorkRef
    work_b: WorkRef
    key_type: KeyType
    key_value: str
    conflict_types: tuple[str, ...]


@dataclass(frozen=True)
class SkippedMerge:
    """A rule merge skipped because the researcher said "not duplicates" (``decision``)."""

    work_a: WorkRef
    work_b: WorkRef
    key_type: KeyType
    key_value: str
    decision: tuple[WorkRef, WorkRef]


@dataclass(frozen=True)
class SharedPdf:
    """One PDF held by two or more papers after grouping."""

    pdf_hash: str
    paper_ids: tuple[str, ...]


@dataclass(frozen=True)
class Collision:
    """A derived id already taken; the paper got ``key`` = ``base_key:<n>``."""

    base_key: str
    key: str
    paper_id: str


@dataclass(frozen=True)
class CheckCandidate:
    paper_id: str
    title: str | None
    year: int | None
    similarity: float
    shared_keys: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CheckItem:
    """A "check these" item (§4.3, attention ``kind='identity_check'``).

    ``paper_id``/``works`` name the subject; ``candidates`` the other papers with the
    evidence (shared keys, title similarity from the shared matcher, years)."""

    reason: CheckReason
    paper_id: str
    works: tuple[WorkRef, ...]
    title: str | None
    year: int | None
    key_type: str | None
    key_value: str | None
    candidates: tuple[CheckCandidate, ...]


@dataclass(frozen=True)
class IdentityPlan:
    papers: tuple[PlannedPaper, ...]
    rule_merges: tuple[RuleMerge, ...]
    blocked: tuple[BlockedUnion, ...]
    skipped_never_merge: tuple[SkippedMerge, ...]
    shared_pdfs: tuple[SharedPdf, ...]
    collisions: tuple[Collision, ...]
    check_items: tuple[CheckItem, ...]

    def paper_for(self, ref: WorkRef) -> PlannedPaper:
        for paper in self.papers:
            if ref in paper.members:
                return paper
        raise KeyError(ref)


# --- ordering ---------------------------------------------------------------------------


def project_rank(slug: str, precedence: Sequence[str] = DEFAULT_PROJECT_PRECEDENCE) -> tuple[int, str]:
    """Sort key for a project: listed projects in order, then ``dbg_*``, then the rest."""
    if slug in precedence:
        return (list(precedence).index(slug), slug)
    if slug.startswith("dbg_"):
        return (len(precedence), slug)
    return (len(precedence) + 1, slug)


def _ref_key(ref: WorkRef, precedence: Sequence[str]) -> tuple[tuple[int, str], str]:
    return (project_rank(ref.project, precedence), ref.work_id)


# --- key and id derivation --------------------------------------------------------------


def normalize_ids(raw: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """Normalise raw ``(id_type, value)`` pairs (``normalize_id`` + the arXiv-DOI fold of
    ``strong_ids``), one pair at a time so alias values of one type survive. Sorted,
    unique; unknown types are dropped."""
    out: set[tuple[str, str]] = set()
    for id_type, value in raw:
        if id_type not in CARRIED_ID_TYPES or value is None:
            continue
        out.update(strong_ids({id_type: value}))
    return tuple(sorted(out))


def derive_key(
    ids: Iterable[tuple[str, str]] = (),
    pdf_hashes: Iterable[str] = (),
    title: str | None = None,
    year: int | None = None,
    v2_ref: WorkRef | None = None,
) -> str:
    """The identity key of §4.1 (before any collision suffix).

    The winning strong id is the smallest value of the first present type in
    doi → openalex → arxiv. With several PDFs the smallest hash is the primary one.
    """
    norm = normalize_ids(ids)
    for id_type in MERGE_ID_TYPES:
        values = sorted(v for t, v in norm if t == id_type)
        if values:
            return f"{id_type}:{values[0]}"
    hashes = sorted(h for h in pdf_hashes if h)
    if hashes:
        return f"pdf:{hashes[0]}"
    nt = normalize_title(title) if title else ""
    if nt:
        return f"title:{nt}|{'' if year is None else year}"
    if v2_ref is None:
        raise ValueError("no id, PDF or usable title, and no v2 work to key on")
    return f"v2:{v2_ref.project}:{v2_ref.work_id}"


def paper_id_for_key(key: str) -> str:
    return "lp_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def mint_paper_id(key: str, taken: Collection[str]) -> MintedId:
    """Mint the id for ``key``; if it is in ``taken`` (ids already used by a different
    paper) append ``:2``, ``:3`` … to the key until free. The caller raises the
    attention item when ``collided``."""
    pid = paper_id_for_key(key)
    if pid not in taken:
        return MintedId(pid, key, False)
    n = 2
    while True:
        candidate = f"{key}:{n}"
        pid = paper_id_for_key(candidate)
        if pid not in taken:
            return MintedId(pid, candidate, True)
        n += 1


# --- merge grouping ---------------------------------------------------------------------


@dataclass
class _Group:
    members: list[WorkRef]
    ids: dict[str, set[str]] = field(default_factory=dict)

    def has_ids(self) -> bool:
        return any(self.ids.get(t) for t in MERGE_ID_TYPES)


class _Planner:
    def __init__(self, works: Sequence[WorkRecord], never_merge, precedence: Sequence[str]):
        self.prec = precedence
        self.works = sorted(works, key=lambda r: _ref_key(r.ref, precedence))
        self.by_ref = {r.ref: r for r in self.works}
        if len(self.by_ref) != len(self.works):
            raise ValueError("duplicate WorkRef in input")
        self.norm_ids = {r.ref: normalize_ids(r.ids) for r in self.works}
        self.parent: dict[WorkRef, WorkRef] = {r.ref: r.ref for r in self.works}
        self.groups: dict[WorkRef, _Group] = {}
        for r in self.works:
            g = _Group([r.ref])
            for t, v in self.norm_ids[r.ref]:
                g.ids.setdefault(t, set()).add(v)
            self.groups[r.ref] = g
        self.never: set[frozenset[WorkRef]] = set()
        for a, b in never_merge:
            a, b = WorkRef(*a), WorkRef(*b)
            if a != b:
                self.never.add(frozenset((a, b)))
        self.rule_merges: list[RuleMerge] = []
        self.blocked: list[BlockedUnion] = []
        self.skipped: list[SkippedMerge] = []
        self.pdf_ambiguous: dict[WorkRef, tuple[str, list[WorkRef]]] = {}

    # union-find ----------------------------------------------------------------------
    def find(self, ref: WorkRef) -> WorkRef:
        while self.parent[ref] != ref:
            self.parent[ref] = self.parent[self.parent[ref]]
            ref = self.parent[ref]
        return ref

    def _conflicts(self, ga: _Group, gb: _Group) -> tuple[str, ...]:
        # Two value sets of one type conflict unless one contains the other (a work's own
        # alias DOIs are not a conflict; two different DOIs are).
        out = []
        for t in MERGE_ID_TYPES:
            va, vb = ga.ids.get(t, set()), gb.ids.get(t, set())
            if va and vb and not (va <= vb or vb <= va):
                out.append(t)
        return tuple(out)

    def _never_pair(self, ga: _Group, gb: _Group) -> tuple[WorkRef, WorkRef] | None:
        if not self.never:
            return None
        for a in ga.members:
            for b in gb.members:
                if frozenset((a, b)) in self.never:
                    pair = sorted((a, b), key=lambda r: _ref_key(r, self.prec))
                    return (pair[0], pair[1])
        return None

    def try_union(self, a: WorkRef, b: WorkRef, key_type: KeyType, key_value: str) -> bool:
        """Join b's group into a's. Returns False (and records why) when blocked."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        ga, gb = self.groups[ra], self.groups[rb]
        conflict = self._conflicts(ga, gb)
        if conflict:
            self.blocked.append(BlockedUnion(a, b, key_type, key_value, conflict))
            return False
        decision = self._never_pair(ga, gb)
        if decision is not None:
            self.skipped.append(SkippedMerge(a, b, key_type, key_value, decision))
            return False
        # Root = the precedence-first member, so roots are stable across input orders.
        if _ref_key(rb, self.prec) < _ref_key(ra, self.prec):
            ra, rb, ga, gb = rb, ra, gb, ga
        self.parent[rb] = ra
        ga.members = sorted(ga.members + gb.members, key=lambda r: _ref_key(r, self.prec))
        for t, vs in gb.ids.items():
            ga.ids.setdefault(t, set()).update(vs)
        del self.groups[rb]
        self.rule_merges.append(RuleMerge(a, b, key_type, key_value))
        return True

    def _join_all(self, refs: list[WorkRef], key_type: KeyType, key_value: str) -> None:
        # Each work tries the earlier works' groups in order and joins the first that
        # accepts it; every refusal is recorded.
        for i, ref in enumerate(refs):
            tried: set[WorkRef] = set()
            for earlier in refs[:i]:
                root = self.find(earlier)
                if root == self.find(ref):
                    break  # already together
                if root in tried:
                    continue
                tried.add(root)
                if self.try_union(earlier, ref, key_type, key_value):
                    break

    # passes --------------------------------------------------------------------------
    def id_passes(self) -> None:
        for id_type in MERGE_ID_TYPES:
            index: dict[str, list[WorkRef]] = {}
            for r in self.works:
                for t, v in self.norm_ids[r.ref]:
                    if t == id_type:
                        index.setdefault(v, []).append(r.ref)
            for value in sorted(index):
                refs = index[value]
                if len(refs) > 1:
                    self._join_all(refs, id_type, value)  # type: ignore[arg-type]

    def pdf_pass(self) -> None:
        index: dict[str, list[WorkRef]] = {}
        for r in self.works:
            for h in sorted(set(r.pdf_hashes)):
                index.setdefault(h, []).append(r.ref)
        for h in sorted(index):
            refs = index[h]
            if len(refs) < 2:
                continue
            identified = [r for r in refs if self.groups[self.find(r)].has_ids()]
            unidentified = [r for r in refs if r not in identified]
            self._join_all(identified, "pdf", h)
            roots: list[WorkRef] = []
            for r in identified:
                root = self.find(r)
                if root not in roots:
                    roots.append(root)
            for r in unidentified:
                if self.groups[self.find(r)].has_ids():
                    continue  # already joined through another of its PDFs
                if len(roots) == 1:
                    anchor = next(x for x in identified if self.find(x) == roots[0])
                    self.try_union(anchor, r, "pdf", h)
                elif len(roots) > 1:
                    self.pdf_ambiguous.setdefault(r, (h, list(roots)))

    def title_pass(self) -> dict[WorkRef, tuple[str, list[WorkRef]]]:
        index: dict[tuple[str, int], list[WorkRef]] = {}
        for r in self.works:
            if not self.norm_ids[r.ref] or r.year is None or not r.title:
                continue
            nt = normalize_title(r.title)
            if nt:
                index.setdefault((nt, r.year), []).append(r.ref)
        ambiguous: dict[WorkRef, tuple[str, list[WorkRef]]] = {}
        for r in self.works:
            if self.groups[self.find(r.ref)].has_ids() or r.year is None or not r.title:
                continue
            nt = normalize_title(r.title)
            if not nt:
                continue
            hits = index.get((nt, r.year), [])
            roots: list[WorkRef] = []
            for h in hits:
                root = self.find(h)
                if root not in roots:
                    roots.append(root)
            if len(roots) == 1:
                self.try_union(hits[0], r.ref, "title_year", f"{nt}|{r.year}")
            elif len(roots) > 1:
                ambiguous[r.ref] = (f"{nt}|{r.year}", roots)
        return ambiguous


def _first(values: Iterable):
    for v in values:
        if v is not None and (not isinstance(v, str) or v.strip()):
            return v
    return None


def plan_identity(
    works: Sequence[WorkRecord],
    never_merge: Iterable[tuple[WorkRef, WorkRef]] = (),
    *,
    precedence: Sequence[str] = DEFAULT_PROJECT_PRECEDENCE,
    taken: Collection[str] = (),
) -> IdentityPlan:
    """Group v2 works into library papers (§4.2) and derive their ids (§4.1).

    Union-find over DOI → OpenAlex → arXiv → identical PDF. A union that would put two
    different values of one id type in a group is blocked (PDF unions too). No-id works
    join only an identified, non-conflicting group through a shared PDF, or a unique
    exact ``normalize_title`` + year match (year required). ``never_merge`` pairs are
    never merged by rule; the skip is reported. ``taken`` are paper ids already in use.
    """
    pl = _Planner(works, never_merge, precedence)
    pl.id_passes()
    pl.pdf_pass()
    title_ambiguous = pl.title_pass()

    # Papers, in order of their first member.
    roots = sorted(pl.groups, key=lambda r: _ref_key(r, precedence))
    used = set(taken)
    papers: list[PlannedPaper] = []
    collisions: list[Collision] = []
    paper_of: dict[WorkRef, PlannedPaper] = {}
    for root in roots:
        members = tuple(pl.groups[root].members)
        recs = [pl.by_ref[m] for m in members]
        ids = tuple(sorted({p for m in members for p in pl.norm_ids[m]}))
        hashes = tuple(sorted({h for rec in recs for h in rec.pdf_hashes if h}))
        title = _first(rec.title for rec in recs)
        year = _first(rec.year for rec in recs)
        base = derive_key(ids, hashes, title, year, members[0])
        minted = mint_paper_id(base, used)
        used.add(minted.paper_id)
        if minted.collided:
            collisions.append(Collision(base, minted.key, minted.paper_id))
        paper = PlannedPaper(
            paper_id=minted.paper_id, key=minted.key, base_key=base, members=members, ids=ids,
            pdf_hashes=hashes, title=title, authors=_first(r.authors for r in recs), year=year,
            venue=_first(r.venue for r in recs), abstract=_first(r.abstract for r in recs),
        )
        papers.append(paper)
        for m in members:
            paper_of[m] = paper

    shared: list[SharedPdf] = []
    by_hash: dict[str, list[str]] = {}
    for p in papers:
        for h in p.pdf_hashes:
            by_hash.setdefault(h, []).append(p.paper_id)
    for h in sorted(by_hash):
        if len(by_hash[h]) > 1:
            shared.append(SharedPdf(h, tuple(by_hash[h])))

    checks = _check_items(pl, papers, paper_of, title_ambiguous)
    return IdentityPlan(
        papers=tuple(papers), rule_merges=tuple(pl.rule_merges), blocked=tuple(pl.blocked),
        skipped_never_merge=tuple(pl.skipped), shared_pdfs=tuple(shared),
        collisions=tuple(collisions), check_items=tuple(checks),
    )


# --- "check these" ----------------------------------------------------------------------


def _token_sets(title: str | None) -> list[frozenset[str]]:
    if not title:
        return []
    return [frozenset(content_tokens(normalize_tokens(v))) for v in variants(title)]


def _similarity(a: list[frozenset[str]], b: list[frozenset[str]]) -> float:
    best = 0.0
    for ta in a:
        for tb in b:
            if ta and tb:
                best = max(best, len(ta & tb) / len(ta | tb))
    return round(best, 4)


def _check_items(pl: _Planner, papers, paper_of, title_ambiguous) -> list[CheckItem]:
    tokens = {p.paper_id: _token_sets(p.title) for p in papers}

    def shared_keys(a: PlannedPaper, b: PlannedPaper) -> tuple[tuple[str, str], ...]:
        keys = set(a.ids) & set(b.ids)
        keys |= {("pdf", h) for h in set(a.pdf_hashes) & set(b.pdf_hashes)}
        return tuple(sorted(keys))

    def cand(subject: PlannedPaper, other: PlannedPaper) -> CheckCandidate:
        return CheckCandidate(other.paper_id, other.title, other.year,
                              _similarity(tokens[subject.paper_id], tokens[other.paper_id]),
                              shared_keys(subject, other))

    def item(reason, subject: PlannedPaper, key_type, key_value, others) -> CheckItem:
        seen: list[PlannedPaper] = []
        for o in others:
            if o.paper_id != subject.paper_id and o not in seen:
                seen.append(o)
        return CheckItem(reason, subject.paper_id, subject.members, subject.title, subject.year,
                         key_type, key_value, tuple(cand(subject, o) for o in seen))

    out: list[CheckItem] = []
    for b in pl.blocked:
        if b.key_type != "pdf":
            out.append(item("blocked_conflict", paper_of[b.work_b], b.key_type, b.key_value, [paper_of[b.work_a]]))
    for s in pl.skipped:
        out.append(item("never_merge_skipped", paper_of[s.work_b], s.key_type, s.key_value, [paper_of[s.work_a]]))

    for rec in pl.works:
        paper = paper_of[rec.ref]
        if paper.ids or len(paper.members) > 1 or not rec.pdf_hashes:
            continue  # only unmerged no-id held works
        if rec.ref in pl.pdf_ambiguous:
            h, roots = pl.pdf_ambiguous[rec.ref]
            out.append(item("ambiguous_pdf", paper, "pdf", h, [paper_of[r] for r in roots]))
        elif rec.ref in title_ambiguous:
            key, roots = title_ambiguous[rec.ref]
            out.append(item("ambiguous_title", paper, "title_year", key, [paper_of[r] for r in roots]))
        else:
            mine = tokens[paper.paper_id]
            scored = []
            for pos, other in enumerate(papers):
                if other.paper_id == paper.paper_id:
                    continue
                sim = _similarity(mine, tokens[other.paper_id])
                if sim > 0:
                    scored.append((-sim, pos, other))
            scored.sort(key=lambda x: (x[0], x[1]))
            out.append(item("no_match", paper, None, None, [o for _, _, o in scored[:NEAREST_CANDIDATES]]))
    return out
