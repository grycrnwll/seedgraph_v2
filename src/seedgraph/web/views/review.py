"""Review screens (design §2.6): attention items grouped by queue with their resolve
actions, duplicate / identity checks, and the migrated citation decisions (read only).
Pure reads.

Every action is a POST that calls :func:`seedgraph.library.attention.resolve_item` with a
fixed ``resolution`` from the item's queue (:func:`allowed_resolutions`); any kind no
queue names falls back to the generic row (**Mark resolved**).

Rules chosen where the design is silent:
- "Same paper" records the researcher's verdict as the item's resolution (``same``); no
  stage 3 deep function merges papers or writes ``duplicate_decisions`` yet, so the page
  says the merge is not automatic.
- Titles and metadata items get **Mark resolved** plus links (paper page for editing,
  the Library's "Run Ollama metadata").
- The graph filter keeps items whose paper is in the graph (any status but ``excluded``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

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
          (Action("same", "Same paper"), Action("different", "Different papers")),
          "Your verdict is recorded on the item; merging two papers is not automatic yet."),
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


def _candidates(subject_json: str) -> tuple[str, ...]:
    try:
        subject = json.loads(subject_json)
    except ValueError:
        return ()
    if not isinstance(subject, dict):
        return ()
    found = [str(c) for c in subject.get("candidates") or () if isinstance(c, str)]
    for key in ("paper_a", "paper_b", "other_paper_id"):
        if isinstance(subject.get(key), str):
            found.append(subject[key])
    return tuple(dict.fromkeys(found))


def _rows(lib: Library, raw: list) -> list[ItemRow]:
    """``raw``: attention item rows ``(item_id, kind, paper_id, subject_json, detail,
    hides_notes, status, resolution, created_at, origin)``."""
    ids = {r[2] for r in raw if r[2]}
    for r in raw:
        ids |= set(_candidates(r[3]))
    papers = _papers(lib, ids)
    out = []
    for item_id, kind, pid, subject_json, detail, hides, status, resolution, created_at, origin in raw:
        out.append(ItemRow(
            item_id=item_id, kind=kind, status=status, resolution=resolution,
            paper=papers.get(pid) or (PaperRef(pid, None, False) if pid else None), detail=detail,
            subject_json=subject_json, hides_notes=bool(hides), created_at=created_at, origin=origin,
            candidates=tuple(papers.get(c) or PaperRef(c, None, False) for c in _candidates(subject_json)),
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
