"""Review screens (design §2.6): attention items grouped by queue with their resolve
actions, duplicate / identity checks, and the migrated citation decisions (read only).
Pure reads.

Every action is a POST that calls :func:`seedgraph.library.attention.resolve_item` with a
fixed ``resolution`` from the item's queue (:func:`allowed_resolutions`); any kind no
queue names falls back to the generic row (**Mark resolved**).

Identity items: **Same paper** is a link to the merge preview (:func:`merge_preview`, one
per candidate), whose Confirm calls :func:`seedgraph.library.merge.merge_papers` and then
resolves the item ``same``; a bare ``same`` resolution is not accepted. **Different papers**
records ``duplicate_decisions`` ``different`` for the item's paper and each candidate
(:func:`record_different`), then resolves the item.

Rules chosen where the design is silent:
- The preview's default survivor is :func:`seedgraph.library.merge.suggest_survivor` (held,
  then strong ids, then notes); the researcher can swap it and pick the primary PDF.
- An ``identity_collision`` item names no candidate, so it offers no merge.
- Titles and metadata items get **Mark resolved** plus links (paper page for editing,
  the Library's "Run Ollama metadata").
- The graph filter keeps items whose paper is in the graph (any status but ``excluded``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ...library import merge
from ...library.merge import MergePlan
from ...library.store import Library


@dataclass(frozen=True)
class Action:
    resolution: str
    label: str


@dataclass(frozen=True)
class Queue:
    key: str
    title: str
    kinds: tuple[str, ...]
    actions: tuple[Action, ...]
    hint: str = ""


MARK_RESOLVED = Action("resolved_by_researcher", "Mark resolved")
QUEUES: tuple[Queue, ...] = (
    Queue("text", "Held back: text", ("text_garbled", "text_missing"), (Action("accept_text", "Note anyway"),),
          "Note anyway lets agents note the paper despite its text. To re-convert with OCR or replace the "
          "PDF, open the paper."),
    Queue("pdf", "PDF may not be this paper", ("pdf_title_miss", "shared_pdf"),
          (Action("confirmed", "It is this paper"),),
          "To detach the PDF or attach the right one, open the paper."),
    Queue("identity", "Identity", ("identity_check", "identity_collision"),
          (Action("different", "Different papers"),),
          "Same paper opens a merge preview: which paper survives, what moves, and anything that blocks it. "
          "Different papers is recorded, so the pair is not raised again."),
    Queue("metadata", "Titles and metadata",
          ("malformed_title", "metadata_needed", "metadata_generic_title", "metadata_doi_mismatch"),
          (MARK_RESOLVED,),
          "Edit metadata on the paper page, or run Ollama metadata from the Library."),
    Queue("renote", "Re-note offers", ("renote_offered",), (Action("dismissed", "Dismiss"),),
          "Write notes creates a one-paper batch."),
    Queue("other", "Other items", (), (MARK_RESOLVED,)),
)
_BY_KIND = {k: q for q in QUEUES for k in q.kinds}
GENERIC = QUEUES[-1]


def queue_for(kind: str) -> Queue:
    return _BY_KIND.get(kind, GENERIC)


def allowed_resolutions(kind: str) -> frozenset[str]:
    return frozenset(a.resolution for a in queue_for(kind).actions)


@dataclass(frozen=True)
class PaperRef:
    paper_id: str
    title: str | None
    held: bool


@dataclass(frozen=True)
class ItemRow:
    item_id: str
    kind: str
    status: str
    resolution: str | None
    paper: PaperRef | None
    detail: str | None
    subject_json: str
    hides_notes: bool
    created_at: str
    origin: str
    candidates: tuple[PaperRef, ...]
    queue: Queue

    @property
    def actions(self) -> tuple[Action, ...]:
        return self.queue.actions

    @property
    def merge_candidates(self) -> tuple[PaperRef, ...]:
        """The papers "Same paper" can merge this item's paper with (identity checks only)."""
        if self.kind != "identity_check" or self.paper is None:
            return ()
        return tuple(c for c in self.candidates if c.paper_id != self.paper.paper_id)


@dataclass(frozen=True)
class Group:
    queue: Queue
    items: tuple[ItemRow, ...]


@dataclass(frozen=True)
class KindCount:
    kind: str
    count: int


@dataclass(frozen=True)
class AttentionPage:
    kind: str | None
    graph: str | None
    groups: tuple[Group, ...]
    kinds: tuple[KindCount, ...]
    graphs: tuple[tuple[str, str], ...]
    total: int


def open_count(lib: Library) -> int:
    (n,) = lib.conn.execute("SELECT COUNT(*) FROM attention_items WHERE status = 'open'").fetchone()
    return int(n)


def _papers(lib: Library, ids: set[str]) -> dict[str, PaperRef]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    return {r[0]: PaperRef(r[0], r[1], r[2] == "held") for r in lib.conn.execute(
        f"SELECT p.paper_id, p.title, s.status FROM papers p JOIN paper_status s ON s.paper_id = p.paper_id "
        f"WHERE p.paper_id IN ({marks})", sorted(ids))}


def _candidates(subject_json: str, detail: str | None = None) -> tuple[str, ...]:
    """Papers an item names besides its own: the subject's ``candidates`` and paper keys
    (ingest), and the evidence's ``candidates`` (the migration's identity checks keep them in
    ``detail`` as ``{"paper_id": ...}`` objects)."""
    found: list[str] = []
    for text, keys in ((subject_json, ("paper_a", "paper_b", "other_paper_id")), (detail, ())):
        try:
            obj = json.loads(text) if text else None
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        for c in obj.get("candidates") or ():
            if isinstance(c, str):
                found.append(c)
            elif isinstance(c, dict) and isinstance(c.get("paper_id"), str):
                found.append(c["paper_id"])
        for key in keys:
            if isinstance(obj.get(key), str):
                found.append(obj[key])
    return tuple(dict.fromkeys(found))


def _rows(lib: Library, raw: list) -> list[ItemRow]:
    """``raw``: attention item rows ``(item_id, kind, paper_id, subject_json, detail,
    hides_notes, status, resolution, created_at, origin)``."""
    ids = {r[2] for r in raw if r[2]}
    for r in raw:
        ids |= set(_candidates(r[3], r[4]))
    papers = _papers(lib, ids)
    out = []
    for item_id, kind, pid, subject_json, detail, hides, status, resolution, created_at, origin in raw:
        out.append(ItemRow(
            item_id=item_id, kind=kind, status=status, resolution=resolution,
            paper=papers.get(pid) or (PaperRef(pid, None, False) if pid else None), detail=detail,
            subject_json=subject_json, hides_notes=bool(hides), created_at=created_at, origin=origin,
            candidates=tuple(papers.get(c) or PaperRef(c, None, False)
                             for c in _candidates(subject_json, detail)),
            queue=queue_for(kind)))
    return out


_COLS = ("item_id, kind, paper_id, subject_json, detail, hides_notes, status, resolution, created_at, origin")


def attention_page(lib: Library, *, kind: str | None = None, graph: str | None = None) -> AttentionPage:
    """Open items, grouped in :data:`QUEUES` order."""
    from . import library as library_view

    options = tuple((g.slug, g.display_name) for g in library_view.graph_options(lib))
    graph = graph if graph in {s for s, _ in options} else None
    raw = lib.conn.execute(
        f"SELECT {_COLS} FROM attention_items WHERE status = 'open' "
        f"ORDER BY kind, COALESCE(paper_id, ''), item_id").fetchall()
    if graph:
        index = library_view.graph_index(lib)
        raw = [r for r in raw if r[2] and graph in index.get(r[2], ())]
    counts: dict[str, int] = {}
    for r in raw:
        counts[r[1]] = counts.get(r[1], 0) + 1
    if kind:
        raw = [r for r in raw if r[1] == kind]
    rows = _rows(lib, raw)
    groups = tuple(Group(q, tuple(r for r in rows if r.queue.key == q.key)) for q in QUEUES)
    return AttentionPage(kind=kind or None, graph=graph, groups=tuple(g for g in groups if g.items),
                         kinds=tuple(KindCount(k, n) for k, n in sorted(counts.items())), graphs=options,
                         total=len(rows))


def item_row(lib: Library, item_id: str) -> ItemRow | None:
    raw = lib.conn.execute(f"SELECT {_COLS} FROM attention_items WHERE item_id = ?", (item_id,)).fetchall()
    rows = _rows(lib, raw)
    return rows[0] if rows else None


# --- duplicates and citations ----------------------------------------------------------------


@dataclass(frozen=True)
class DuplicateDecision:
    paper_a: PaperRef
    paper_b: PaperRef
    verdict: str
    decided_by: str
    reason: str | None
    decided_at: str
    origin: str


@dataclass(frozen=True)
class DuplicatesPage:
    checks: tuple[ItemRow, ...]
    decisions: tuple[DuplicateDecision, ...]


def duplicates_page(lib: Library) -> DuplicatesPage:
    identity = next(q for q in QUEUES if q.key == "identity")
    marks = ",".join("?" * len(identity.kinds))
    raw = lib.conn.execute(
        f"SELECT {_COLS} FROM attention_items WHERE status = 'open' AND kind IN ({marks}) "
        f"ORDER BY created_at, item_id", identity.kinds).fetchall()
    decided = lib.conn.execute(
        "SELECT paper_a, paper_b, verdict, decided_by, reason, decided_at, origin FROM duplicate_decisions "
        "ORDER BY decided_at DESC, paper_a, paper_b").fetchall()
    papers = _papers(lib, {r[0] for r in decided} | {r[1] for r in decided})
    ref = lambda pid: papers.get(pid) or PaperRef(pid, None, False)  # noqa: E731
    return DuplicatesPage(
        checks=tuple(_rows(lib, raw)),
        decisions=tuple(DuplicateDecision(ref(a), ref(b), v, by, reason, at, origin)
                        for a, b, v, by, reason, at, origin in decided))


@dataclass(frozen=True)
class CitationDecision:
    reference_id: str
    citing: PaperRef | None
    reference_text: str | None
    decision: str
    rejected_record: str | None
    reason: str | None
    decided_at: str
    origin: str


def citations_page(lib: Library) -> tuple[CitationDecision, ...]:
    """The migrated citation review decisions (read only; the candidate queue arrives with
    its producer in stage 4)."""
    rows = lib.conn.execute(
        "SELECT d.reference_id, r.citing_paper_id, r.raw_reference_text, d.decision, d.rejected_record_json, "
        "d.reason, d.decided_at, d.origin FROM citation_review_decisions d "
        "LEFT JOIN reference_entries r ON r.reference_id = d.reference_id "
        "ORDER BY d.decided_at DESC, d.reference_id").fetchall()
    papers = _papers(lib, {r[1] for r in rows if r[1]})
    return tuple(CitationDecision(rid, papers.get(cp) if cp else None, text, decision, rec, reason, at, origin)
                 for rid, cp, text, decision, rec, reason, at, origin in rows)


# --- merge preview and "different papers" ---------------------------------------------------------


@dataclass(frozen=True)
class FileChoice:
    source_file_id: str
    owner: str
    chosen: bool
    url: str


@dataclass(frozen=True)
class MergePreview:
    item: ItemRow
    with_id: str
    plan: MergePlan
    keep: PaperRef
    other: PaperRef
    primary_arg: str  # the primary PDF the researcher picked ('' = the plan's default)
    swap_url: str
    files: tuple[FileChoice, ...]

    @property
    def url(self) -> str:
        return _preview_url(self.item.item_id, self.with_id, self.keep.paper_id, self.primary_arg)


def _preview_url(item_id: str, with_id: str, keep: str | None = None, primary: str | None = None) -> str:
    from urllib.parse import urlencode

    q = {"item": item_id, "with": with_id}
    if keep:
        q["keep"] = keep
    if primary:
        q["primary"] = primary
    return "/review/merge?" + urlencode(q)


def merge_pair(lib: Library, item_id: str, with_id: str) -> tuple[ItemRow, str, str] | None:
    """``(row, a, b)``: the item and the two live papers it would merge, or ``None`` when the
    item is unknown or does not name ``with_id`` as a candidate. Ids merged away earlier are
    followed to their survivor."""
    row = item_row(lib, item_id)
    if row is None or with_id not in {c.paper_id for c in row.merge_candidates}:
        return None
    assert row.paper is not None
    a, b = merge.resolve_merged(lib.conn, row.paper.paper_id), merge.resolve_merged(lib.conn, with_id)
    return row, a, b


def merge_preview(lib: Library, item_id: str, with_id: str, *, keep: str | None = None,
                  primary: str | None = None) -> MergePreview | None:
    """The merge plan for an identity item and one of its candidates (pure). ``None``: no such
    pair. Raises :class:`seedgraph.library.merge.MergeError` for a pair that cannot be planned
    (already one paper, unknown primary PDF)."""
    found = merge_pair(lib, item_id, with_id)
    if found is None:
        return None
    row, a, b = found
    if keep in (a, b):
        keep_id, other_id = keep, (b if keep == a else a)
    else:
        keep_id, other_id = merge.suggest_survivor(lib.conn, a, b)
    plan = merge.plan_merge(lib, keep_id, other_id, primary_file=primary or None)
    refs = _papers(lib, {keep_id, other_id})
    files = []
    for pid in (keep_id, other_id):
        for (sf,) in lib.conn.execute("SELECT source_file_id FROM paper_files WHERE paper_id = ? "
                                      "ORDER BY source_file_id", (pid,)):
            if any(f.source_file_id == sf for f in files):
                continue
            files.append(FileChoice(sf, pid, sf == plan.primary_file, _preview_url(item_id, with_id, keep_id, sf)))
    return MergePreview(item=row, with_id=with_id, plan=plan, keep=refs[keep_id], other=refs[other_id],
                        primary_arg=primary or "", swap_url=_preview_url(item_id, with_id, other_id),
                        files=tuple(files) if len(files) > 1 else ())


def record_different(lib: Library, row: ItemRow) -> None:
    """Record "different papers" for an identity item's paper and each of its candidates."""
    if row.paper is None:
        return
    for c in row.merge_candidates:
        a, b = merge.resolve_merged(lib.conn, row.paper.paper_id), merge.resolve_merged(lib.conn, c.paper_id)
        if a != b:
            merge.record_different(lib, a, b, reason=f"review item {row.item_id}")
