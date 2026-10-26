"""Batches (design §2.5): the list and the detail page. Wraps
:func:`seedgraph.work.routine.batch_view` / ``list_batches``, whose state label and next
action come from :mod:`seedgraph.work.state`, so pause states read exactly as there.

Note: ``batch_view`` expires lapsed claims first (stage 2's lazy expiry), so reading a
batch whose claim has lapsed records that expiry; nothing else here writes.

Per-paper state words: ``queued``, ``claimed by <worker> since <t>`` (a live claim),
``in progress`` (some parts saved, none claimed), ``done``, ``held back``, ``failed``,
``preserved``, ``stale``, ``conflict``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ...library.store import Library
from ...work import routine
from ...work.commands import WorkCommands


@dataclass(frozen=True)
class BatchRow:
    batch_id: str
    task: str
    selection: str
    graph: str | None
    state: str
    label: str
    done: int
    held_back: int
    failed: int
    total: int
    created_at: str


@dataclass(frozen=True)
class ItemRow:
    ordinal: int
    paper_id: str
    title: str | None
    held: bool
    state: str
    state_text: str
    parts: str | None              # "k/n" for a split paper
    quotes_anchored: int | None    # done papers only
    quotes_total: int | None
    reason: str | None
    attention_item_id: str | None
    claims: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BatchDetail:
    batch_id: str
    task: str
    selection: str
    graph: str | None
    state: str
    label: str
    next_action: str
    stop_mode: str | None
    paused: bool
    finished: bool
    pause_detail: str | None
    commands: WorkCommands
    accepted_unanchored: int
    counts: dict[str, int]
    items: tuple[ItemRow, ...]
    item_total: int
    work_dir: str
    created_at: str


def _selection(selection_query: str | None, total: int) -> str:
    return selection_query or f"{total} selected paper{'s' if total != 1 else ''}"


def _created(lib: Library) -> dict[str, str]:
    return {r[0]: r[1] for r in lib.conn.execute("SELECT batch_id, created_at FROM batches")}


def list_page(lib: Library, *, now: datetime | None = None) -> list[BatchRow]:
    """Every batch, newest first."""
    created = _created(lib)
    rows = []
    for v in routine.list_batches(lib, now=now):
        items = v.state.items
        rows.append(BatchRow(
            batch_id=v.batch_id, task=v.task_type, selection=_selection(v.selection_query, v.item_total),
            graph=v.graph, state=v.state.state, label=v.state.label, done=items.get("done", 0),
            held_back=items.get("held_back", 0), failed=items.get("failed", 0), total=v.item_total,
            created_at=created.get(v.batch_id, "")))
    rows.sort(key=lambda r: (r.created_at, r.batch_id), reverse=True)
    return rows


def exists(lib: Library, batch_id: str) -> bool:
    return lib.conn.execute("SELECT 1 FROM batches WHERE batch_id = ?", (batch_id,)).fetchone() is not None


def _quotes(lib: Library, note_ids: list[str]) -> dict[str, tuple[int, int]]:
    if not note_ids:
        return {}
    marks = ",".join("?" * len(note_ids))
    return {r[0]: (int(r[1] or 0), int(r[2])) for r in lib.conn.execute(
        f"SELECT c.structured_note_id, SUM(EXISTS (SELECT 1 FROM claim_spans s WHERE s.claim_id = c.claim_id)), "
        f"COUNT(*) FROM extracted_claims c WHERE c.status = 'found' AND c.structured_note_id IN ({marks}) "
        f"GROUP BY c.structured_note_id", note_ids)}


def _state_text(state: str, claims: list[dict[str, Any]]) -> str:
    live = [c for c in claims if c.get("ended") == "live"]
    if live:
        c = live[-1]
        return f"claimed by {c.get('worker')} since {c.get('claimed_at')}"
    return {"held_back": "held back", "in_progress": "in progress"}.get(state, state)


def detail(lib: Library, batch_id: str, *, now: datetime | None = None) -> BatchDetail | None:
    """The batch page, or ``None`` for an unknown batch."""
    if not exists(lib, batch_id):
        return None
    v = routine.batch_view(lib, batch_id, now=now)
    paper_ids = [i.subject_id for i in v.items if i.subject_kind == "paper"]
    papers: dict[str, tuple[str | None, bool]] = {}
    if paper_ids:
        marks = ",".join("?" * len(paper_ids))
        papers = {r[0]: (r[1], r[2] == "held") for r in lib.conn.execute(
            f"SELECT p.paper_id, p.title, s.status FROM papers p JOIN paper_status s ON s.paper_id = p.paper_id "
            f"WHERE p.paper_id IN ({marks})", paper_ids)}
    quotes = _quotes(lib, [i.result_ref for i in v.items if i.state == "done" and i.result_ref])
    items = []
    for i in v.items:
        title, held = papers.get(i.subject_id, (None, False))
        q = quotes.get(i.result_ref or "") if i.state == "done" else None
        items.append(ItemRow(
            ordinal=i.ordinal, paper_id=i.subject_id, title=title, held=held, state=i.state,
            state_text=_state_text(i.state, i.claims),
            parts=f"{i.parts_done}/{i.part_count}" if (i.part_count or 0) > 1 else None,
            quotes_anchored=q[0] if q else None, quotes_total=q[1] if q else None, reason=i.reason,
            attention_item_id=i.attention_item_id, claims=tuple(i.claims)))
    st = v.state
    return BatchDetail(
        batch_id=v.batch_id, task=v.task_type, selection=_selection(v.selection_query, v.item_total),
        graph=v.graph, state=st.state, label=st.label, next_action=st.next_action, stop_mode=st.stop_mode,
        paused=st.state in ("stopped", "usage_limit", "claim_expired"), finished=st.state == "finished",
        pause_detail=st.pause_detail, commands=v.commands, accepted_unanchored=v.accepted_unanchored,
        counts=dict(st.items), items=tuple(items), item_total=v.item_total, work_dir=v.work_dir,
        created_at=_created(lib).get(v.batch_id, ""))
