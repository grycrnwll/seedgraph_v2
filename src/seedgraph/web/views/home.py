"""Home (design §2.1): a "what's next" checklist per graph and the activity feed.
Pure reads (batch states go through ``work.routine``, whose lazy claim expiry is the one
write a read can cause).

Checklist lines are derived on every read and appear only while true:
``needs_notes``, ``held_back``, ``attention``, ``batch:<id>`` (a paused batch holding
one of the graph's members: its ``work.state`` label), ``newer_extraction`` and
``concepts``. The feed merges the latest ``batch_events`` (each with its batch's current
``work.state`` label), finished add-paper jobs and overlay saves, newest first.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlencode

from ...library import graphs as graphs_mod
from ...library.store import Library
from ...work import routine
from . import library as library_view

PAUSED = frozenset({"stopped", "usage_limit", "claim_expired"})
EVENT_WORDS = {"batch_created": "batch created", "items_decided": "papers decided", "claimed": "claimed",
               "submitted": "submitted", "claim_released": "released", "claim_expired": "a worker stopped responding",
               "claim_revived": "claim revived", "paper_failed": "paper failed", "stopped": "stopped",
               "usage_limit": "paused: usage limit", "continued": "continued"}


@dataclass(frozen=True)
class Line:
    key: str
    text: str
    href: str
    links: tuple[tuple[str, str], ...] = ()   # extra (text, href), e.g. the papers to review


@dataclass(frozen=True)
class GraphChecklist:
    slug: str
    display_name: str
    lines: tuple[Line, ...]


@dataclass(frozen=True)
class Entry:
    at: str
    what: str
    text: str
    href: str | None
    batch_id: str | None = None
    label: str | None = None    # the batch's current work.state label


@dataclass(frozen=True)
class Activity:
    entries: tuple[Entry, ...]
    working: bool               # a batch is working: the feed polls


@dataclass(frozen=True)
class HomePage:
    graphs: tuple[GraphChecklist, ...]
    activity: Activity


def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def _batch_members(lib: Library) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for batch_id, subject in lib.conn.execute(
            "SELECT batch_id, subject_id FROM batch_items WHERE subject_kind = 'paper'"):
        out.setdefault(batch_id, set()).add(subject)
    return out


def _checklist(lib: Library, slug: str, display_name: str, batches: Sequence, members_of: dict[str, set[str]],
               ) -> GraphChecklist:
    with graphs_mod.open_graph(lib, slug) as g:
        rows = graphs_mod.membership(g)
    members = [r.paper_id for r in rows if r.status == "member"]
    in_graph = {r.paper_id for r in rows if r.status != "excluded"}
    held = set(library_view.view_ids(lib, "held", index={p: (slug,) for p in members}, graph=slug))
    states = library_view.note_states(lib, [p for p in members if p in held])
    q = lambda **kw: urlencode(kw)  # noqa: E731
    lines: list[Line] = []

    need = [p for p, s in states.items() if library_view.needs_notes(s)]
    if need:
        lines.append(Line("needs_notes", f"{_plural(len(need), 'member paper')} need{'s' if len(need) == 1 else ''} notes",
                          f"/library/write-notes?{q(view='needs-notes', graph=slug)}"))
    held_back = set(library_view.view_ids(lib, "held-back", graph=slug, index={p: (slug,) for p in in_graph}))
    if held_back:
        lines.append(Line("held_back", f"{_plural(len(held_back), 'paper')} held back",
                          f"/library?{q(view='held-back', graph=slug)}"))
    if in_graph:
        marks = ",".join("?" * len(in_graph))
        (n_items,) = lib.conn.execute(
            f"SELECT COUNT(*) FROM attention_items WHERE status = 'open' AND paper_id IN ({marks})",
            sorted(in_graph)).fetchone()
        if n_items:
            lines.append(Line("attention", f"{_plural(n_items, 'attention item')} on this graph's papers",
                              f"/review/attention?{q(graph=slug)}"))
    for b in batches:
        if b.state.state in PAUSED and (b.graph == slug or members_of.get(b.batch_id, set()) & set(members)):
            lines.append(Line(f"batch:{b.batch_id}", f"batch {b.batch_id}: {b.state.label}",
                              f"/batches/{b.batch_id}"))
    newer = [p for p, s in states.items() if s == "newer extraction"]
    if newer:
        lines.append(Line("newer_extraction", f"{_plural(len(newer), 'paper has', 'papers have')} a newer extraction "
                          f"to review", f"/papers/{newer[0]}", tuple((p, f"/papers/{p}") for p in newer)))
    if members:
        marks = ",".join("?" * len(members))
        (changed,) = lib.conn.execute(
            f"SELECT COUNT(*) FROM concept_refresh WHERE paper_id IN ({marks})", members).fetchone()
        if changed:
            lines.append(Line("concepts", f"concepts out of date ({_plural(changed, 'paper')} changed)",
                              f"/graphs/{slug}"))
    return GraphChecklist(slug, display_name, tuple(lines))


def activity(lib: Library, *, jobs: Sequence = (), now: datetime | None = None, limit: int = 40) -> Activity:
    batches = routine.list_batches(lib, now=now)
    labels = {b.batch_id: b.state.label for b in batches}
    entries: list[Entry] = []
    for batch_id, at, kind, worker, detail in lib.conn.execute(
            "SELECT batch_id, at, kind, worker_id, detail_json FROM batch_events ORDER BY at DESC, seq DESC LIMIT ?",
            (limit,)):
        what = EVENT_WORDS.get(kind, kind.replace("_", " "))
        text = what + (f" by {worker}" if worker else "")
        if kind == "items_decided" and detail:
            try:
                counts = json.loads(detail)
                text = what + ": " + ", ".join(f"{n} {k.replace('_', ' ')}" for k, n in counts.items() if n)
            except (TypeError, ValueError, AttributeError):
                pass
        entries.append(Entry(at, what, text, f"/batches/{batch_id}", batch_id, labels.get(batch_id)))
    for pid, title, version, action, summary, at in lib.conn.execute(
            "SELECT o.paper_id, p.title, v.version_no, v.action, v.summary, v.created_at FROM overlay_versions v "
            "JOIN note_overlays o ON o.overlay_id = v.overlay_id JOIN papers p ON p.paper_id = o.paper_id "
            "ORDER BY v.created_at DESC LIMIT ?", (limit,)):
        entries.append(Entry(at, "note edited",
                             f"note edited ({action}{': ' + summary if summary else ''}) on {title or pid}, "
                             f"version {version}", f"/papers/{pid}"))
    for j in jobs:
        if j.state in ("done", "failed") and j.finished_at:
            entries.append(Entry(j.finished_at, f"job {j.state}", f"{j.label}: {j.state}", "/library#sg-add-papers"))
    entries.sort(key=lambda e: e.at, reverse=True)
    return Activity(tuple(entries[:limit]), working=any(b.state.state == "working" for b in batches))


def page(lib: Library, *, jobs: Sequence = (), now: datetime | None = None) -> HomePage:
    batches = routine.list_batches(lib, now=now)
    members_of = _batch_members(lib)
    graphs = tuple(_checklist(lib, info.slug, info.display_name, batches, members_of)
                   for info in graphs_mod.list_graphs(lib))
    return HomePage(graphs=graphs, activity=activity(lib, jobs=jobs, now=now))
