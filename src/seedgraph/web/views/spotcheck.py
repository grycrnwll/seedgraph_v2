"""Spot-check screens (stage 5 design §5.3-5.4, §6.5; slice R6): the accuracy tables, the
start form, earlier samples and one sample's items.

Pure reads over :mod:`seedgraph.spotcheck` (``accuracy.accuracy``, ``sample.samples``,
``sample.sample``, ``sample.population`` for the filter choices); nothing here writes.

Rules chosen where the design is silent:
- Every item shows the agent's **original** claim (the stored ``extracted_claims`` row),
  never the overlay's edit; ``claim_text`` and ``quote`` can be missing.
- "Fix it now" shows when the mark is partly right or wrong (``marks.FIX_MARKS``). It opens
  the paper page with that claim's edit form open (``?edit=<claim>&from=spot-check:<sample>``;
  the paper page turns on "show hidden" for a hidden claim). An item from an earlier
  extraction opens the paper page at the current note, with no ``edit``.
- "fixed" is the derived :attr:`Item.fixed`; the accuracy figure is never changed by a fix.
- The start form's size must be a whole number between 1 and :data:`MAX_SIZE`; filter values
  are checked by ``create_sample``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote, urlencode

from ...library import graphs as graphs_mod
from ...library.store import Library
from ...spotcheck import DEFAULT_SIZE, MARK_LABELS, MARKS, REASON_LABELS, REASONS, Item, Sample
from ...spotcheck import accuracy as accuracy_mod
from ...spotcheck import marks as marks_mod
from ...spotcheck import sample as sample_mod
from . import paper as paper_view

MAX_SIZE = 100
FROM_PREFIX = "spot-check:"
SAMPLE_ID = re.compile(r"sc_[0-9a-f]+")
FIX_MARKS = marks_mod.FIX_MARKS
EARLIER_EXTRACTION = "this claim is from an earlier extraction"
HIDDEN_NOTE = "you hid this claim"
STANDING_NOTE = ("Each claim is marked once. Fixing a claim does not change the figures: they count the agent's "
                 "original claim.")
MARK_CHOICES = tuple((m, MARK_LABELS[m]) for m in MARKS)
REASON_CHOICES = tuple((r, REASON_LABELS[r]) for r in REASONS)


def local_time(ts: str | None) -> str:
    if not ts:
        return ""
    return datetime.fromisoformat(ts).astimezone().strftime("%Y-%m-%d %H:%M")


def from_param(sample_id: str) -> str:
    return FROM_PREFIX + sample_id


def back_sample(value: str | None) -> str | None:
    """The sample id named by a ``from=spot-check:<id>`` parameter, else ``None``."""
    if not value or not value.startswith(FROM_PREFIX):
        return None
    sid = value[len(FROM_PREFIX):]
    return sid if SAMPLE_ID.fullmatch(sid) else None


def paper_href(paper_id: str, claim_id: str | None = None, *, sample_id: str | None = None,
               edit: bool = False) -> str:
    """The paper page; with ``edit`` its edit form for ``claim_id`` is open and scrolled to."""
    query: dict[str, str] = {}
    if edit and claim_id:
        query["edit"] = claim_id
    if sample_id:
        query["from"] = from_param(sample_id)
    href = f"/papers/{quote(paper_id)}" + (f"?{urlencode(query, safe=':')}" if query else "")
    return href + (f"#claim-{claim_id}" if claim_id else "")


# --- accuracy and the start form ---------------------------------------------------------------

@dataclass(frozen=True)
class AccuracyLine:
    label: str
    summary: str
    n_marked: int
    too_few: bool


@dataclass(frozen=True)
class AccuracyView:
    overall: str
    overall_too_few: bool
    n_marked: int
    by_type: tuple[AccuracyLine, ...]
    by_model: tuple[AccuracyLine, ...]
    cells: tuple[AccuracyLine, ...]


def _lines(rows, label) -> tuple[AccuracyLine, ...]:
    return tuple(AccuracyLine(label(r), r.summary, r.n_marked, r.too_few) for r in rows)


def accuracy(lib: Library) -> AccuracyView:
    t = accuracy_mod.accuracy(lib)
    return AccuracyView(
        overall=t.overall.summary, overall_too_few=t.overall.too_few, n_marked=t.overall.n_marked,
        by_type=_lines(t.by_type, lambda r: paper_view.claim_type_label(r.claim_type or "")),
        by_model=_lines(t.by_model, lambda r: r.model_key or ""),
        cells=_lines(t.cells, lambda r: f"{paper_view.claim_type_label(r.claim_type or '')} · {r.model_key}"))


def _filters_text(filters) -> str:
    return ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in sorted(filters.items()))


@dataclass(frozen=True)
class SampleRow:
    sample_id: str
    when: str
    size: int
    drawn: int
    n_marked: int
    finished: bool
    filters: str


def _row(s: Sample) -> SampleRow:
    return SampleRow(s.sample_id, local_time(s.created_at), s.size, len(s.items), s.n_marked,
                     s.finished_at is not None, _filters_text(s.filters))


@dataclass(frozen=True)
class IndexPage:
    accuracy: AccuracyView
    samples: tuple[SampleRow, ...]
    default_size: int
    max_size: int
    graph_choices: tuple[tuple[str, str], ...]
    type_choices: tuple[tuple[str, str], ...]
    model_choices: tuple[str, ...]
    note: str = STANDING_NOTE


def index(lib: Library) -> IndexPage:
    pop = sample_mod.population(lib, include_marked=True)
    graph_choices = tuple(sorted(((g.slug, g.display_name) for g in graphs_mod.list_graphs(lib)),
                                 key=lambda kv: kv[1].lower()))
    types = sorted({c.claim_type for c in pop})
    return IndexPage(
        accuracy=accuracy(lib), samples=tuple(_row(s) for s in sample_mod.samples(lib)),
        default_size=DEFAULT_SIZE, max_size=MAX_SIZE, graph_choices=graph_choices,
        type_choices=tuple((t, paper_view.claim_type_label(t)) for t in types),
        model_choices=tuple(sorted({c.model_key for c in pop})))


# --- one sample --------------------------------------------------------------------------------

@dataclass(frozen=True)
class ItemView:
    item: Item
    sample_id: str
    type_label: str
    mark_label: str | None
    reason_labels: tuple[str, ...]
    held: bool
    paper_href: str
    show_href: str | None          # the claim on the paper page (its quote is highlighted there)
    fix_href: str | None           # "fix it now"
    when: str
    pages: str | None
    message: str | None = None
    message_level: str = "info"


def _pages(item: Item) -> str | None:
    if item.page_start is None:
        return None
    if item.page_end is None or item.page_end == item.page_start:
        return f"p. {item.page_start}"
    return f"pp. {item.page_start}–{item.page_end}"


def item_view(lib: Library, sid: str, item: Item, *, message: str | None = None,
              message_level: str = "info") -> ItemView:
    held = lib.conn.execute("SELECT 1 FROM paper_files WHERE paper_id = ?", (item.paper_id,)).fetchone() is not None
    fix = None
    if item.mark in FIX_MARKS:
        fix = paper_href(item.paper_id, None if item.earlier_extraction else item.claim_id, sample_id=sid,
                         edit=not item.earlier_extraction)
    show = None if item.earlier_extraction else paper_href(item.paper_id, item.claim_id)
    return ItemView(
        item=item, sample_id=sid, type_label=paper_view.claim_type_label(item.claim_type),
        mark_label=MARK_LABELS.get(item.mark) if item.mark else None,
        reason_labels=tuple(REASON_LABELS[r] for r in item.reasons if r in REASON_LABELS), held=held,
        paper_href=paper_href(item.paper_id, sample_id=sid), show_href=show, fix_href=fix,
        when=local_time(item.extracted_at), pages=_pages(item), message=message, message_level=message_level)


def running_line(s: Sample) -> str:
    c = s.counts
    return (f"marked {s.n_marked} of {len(s.items)}: right {c['right']}, partly right {c['partly']}, "
            f"wrong {c['wrong']}")


@dataclass(frozen=True)
class SamplePage:
    sample_id: str
    when: str
    size: int
    filters: str
    finished: bool
    running: str
    items: tuple[ItemView, ...]


def sample_page(lib: Library, sample_id: str) -> SamplePage | None:
    s = sample_mod.sample(lib, sample_id)
    if s is None:
        return None
    return SamplePage(
        sample_id=s.sample_id, when=local_time(s.created_at), size=s.size, filters=_filters_text(s.filters),
        finished=s.finished_at is not None, running=running_line(s),
        items=tuple(item_view(lib, s.sample_id, i) for i in s.items))


@dataclass(frozen=True)
class ItemSwap:
    """What an htmx mark returns: the refreshed item and the sample's running line."""

    item: ItemView
    running: str


def item_swap(lib: Library, sample_id: str, claim_id: str, *, message: str | None = None,
              message_level: str = "info") -> ItemSwap | None:
    s = sample_mod.sample(lib, sample_id)
    item = s.item(claim_id) if s else None
    if s is None or item is None:
        return None
    return ItemSwap(item_view(lib, sample_id, item, message=message, message_level=message_level),
                    running_line(s))
