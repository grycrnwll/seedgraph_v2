"""Records screens (stage 5 design §6.1-6.4, §13 Q3; slice R5): the Records list with its
filters, a graph's Records tab, the record page, and the Home feed's record entries.

Pure reads over :mod:`seedgraph.records.manage` (``list_records``, ``get_record``): nothing
here writes. Graph display names come from :func:`seedgraph.library.graphs.list_graphs`
(a record's links carry the slug only); a deleted graph is shown by its slug, marked.

Rules chosen where the design is silent:
- A record is shown as **Library-wide** when it searched no graph; the graphs it reaches
  only through a cited paper are listed beside it, marked "library-wide search; cites
  papers in <graph>" (decision Q3). On a graph's tab the mark is
  :meth:`~seedgraph.records.manage.RecordRow.graph_mark`.
- A filter value that is not one of the choices (an unknown state, a bad offset) is
  ignored, so a hand-edited URL never fails the page.
- **Edited since this answer** (claim evidence only): the paper's overlay head now differs
  from the ``(overlay_id, version_no)`` frozen with the claim; a claim frozen with no
  overlay counts as edited once the paper has one.
- **Quote links** reuse the paper page's claim anchor (``/papers/<id>#claim-<claim>``, as
  the concept pages do); the passage is highlighted from the claim's quote there. A cited
  span links to the current claim anchored on it, else to the paper. A cited span is the
  paper's own words, so it is labelled "paper states".
- Times are shown in local time (the session names are local too).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

from ...library import graphs as graphs_mod
from ...library.store import Library
from ...notes import interpretation, overlay
from ...records import manage
from ...records import path as paths

PAGE_SIZE = 50
SHORT_QUESTION = 80
SCOPE_LINE = ("Only answers an agent recorded from your library appear here, with paths that ended without an "
              "answer. Questions answered without searching the library do not.")
EDITED_SINCE = "edited since this answer"
SPAN_TRUST = "paper states"
FEED_STATES = (paths.STATE_ANSWERED, paths.STATE_NO_ANSWER)
STATE_CHOICES = tuple((s, paths.STATE_LABELS[s]) for s in paths.STATES)
LIBRARY_WIDE_LABEL = "Library-wide"


def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def local_time(ts: str | None) -> str:
    if not ts:
        return ""
    return datetime.fromisoformat(ts).astimezone().strftime("%Y-%m-%d %H:%M")


def shorten(text: str, limit: int = SHORT_QUESTION) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def graph_names(lib: Library) -> dict[str, str]:
    return {g.slug: g.display_name for g in graphs_mod.list_graphs(lib)}


# --- the list -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphTag:
    slug: str
    name: str
    deleted: bool
    cited_only: bool               # reached through a cited paper only (a library-wide search)

    @property
    def href(self) -> str | None:
        return None if self.deleted else f"/graphs/{self.slug}"


@dataclass(frozen=True)
class RowView:
    row: manage.RecordRow
    graphs: tuple[GraphTag, ...]
    mark: str | None               # the graph tab's "cites papers in this graph" mark

    @property
    def href(self) -> str:
        return f"/records/{self.row.record_id}"

    @property
    def when(self) -> str:
        return local_time(self.row.when)

    @property
    def searched(self) -> tuple[GraphTag, ...]:
        return tuple(g for g in self.graphs if not g.cited_only)

    @property
    def cited_only(self) -> tuple[GraphTag, ...]:
        return tuple(g for g in self.graphs if g.cited_only)

    @property
    def library_wide(self) -> bool:
        return not self.searched

    @property
    def counts_text(self) -> str:
        c = self.row.counts
        return (f"{_plural(c.get('searches', 0), 'search', 'searches')}, {c.get('opened', 0)} opened, "
                f"{c.get('cited', 0)} cited")


@dataclass(frozen=True)
class FilterForm:
    graph: str | None
    state: str | None
    starred: bool
    flagged: bool | None
    client: str | None
    text: str
    offset: int

    def query(self, **override: Any) -> dict[str, str]:
        values = {"graph": self.graph, "state": self.state, "starred": "1" if self.starred else None,
                  "flagged": None if self.flagged is None else ("1" if self.flagged else "0"),
                  "client": self.client, "q": self.text or None}
        values.update(override)
        return {k: str(v) for k, v in values.items() if v not in (None, "", 0)}

    @property
    def filters(self) -> manage.Filters:
        return manage.Filters(graph=self.graph, state=self.state, starred=True if self.starred else None,
                              flagged=self.flagged, client=self.client, text=self.text or None)


@dataclass(frozen=True)
class RecordsList:
    rows: tuple[RowView, ...]
    total: int
    form: FilterForm
    base: str                      # /records, or /graphs/<slug>/records
    graph_choices: tuple[tuple[str, str], ...]
    state_choices: tuple[tuple[str, str], ...]
    client_choices: tuple[str, ...]
    prev_href: str | None
    next_href: str | None
    here: str                      # this page's URL, for the star buttons' return

    @property
    def scope_line(self) -> str:
        return SCOPE_LINE

    @property
    def first(self) -> int:
        return self.form.offset + 1 if self.rows else 0

    @property
    def last(self) -> int:
        return self.form.offset + len(self.rows)


def _as_offset(value: Any) -> int:
    try:
        n = int(str(value))
    except (TypeError, ValueError):
        return 0
    return max(n, 0)


def parse_form(params: Mapping[str, Any], *, names: Mapping[str, str], clients: Sequence[str],
               graph: str | None = None) -> FilterForm:
    """The filters from query parameters; values that are not choices are ignored."""
    g = graph
    if g is None:
        raw = str(params.get("graph") or "")
        g = raw if raw == manage.LIBRARY_WIDE or raw in names else None
    state = str(params.get("state") or "")
    flagged = {"1": True, "0": False}.get(str(params.get("flagged") or ""))
    client = str(params.get("client") or "")
    return FilterForm(graph=g, state=state if state in paths.STATES else None,
                      starred=str(params.get("starred") or "") == "1", flagged=flagged,
                      client=client if client in clients else None, text=str(params.get("q") or "").strip(),
                      offset=_as_offset(params.get("offset")))


def _tags(row: manage.RecordRow, names: Mapping[str, str]) -> tuple[GraphTag, ...]:
    return tuple(GraphTag(g.slug, names.get(g.slug, g.slug), g.deleted or g.slug not in names,
                          g.how == "cited_member") for g in row.graphs)


def list_page(lib: Library, params: Mapping[str, Any], *, graph: str | None = None,
              now: datetime | None = None) -> RecordsList:
    """The Records list (``graph`` None) or a graph's Records tab, filtered and paged."""
    names = graph_names(lib)
    everything = manage.list_records(lib, limit=10**9, now=now).rows
    clients = tuple(sorted({r.client for r in everything if r.client}))
    form = parse_form(params, names=names, clients=clients, graph=graph)
    page = manage.list_records(lib, form.filters, limit=PAGE_SIZE, offset=form.offset, now=now)
    rows = tuple(RowView(r, _tags(r, names), r.graph_mark(graph) if graph else None) for r in page.rows)
    base = f"/graphs/{graph}/records" if graph else "/records"
    q = form.query(graph=None) if graph else form.query()

    def href(offset: int) -> str:
        qs = urlencode({**q, **({"offset": str(offset)} if offset else {})})
        return base + (f"?{qs}" if qs else "")

    prev_href = href(max(form.offset - PAGE_SIZE, 0)) if form.offset > 0 else None
    next_href = href(form.offset + PAGE_SIZE) if form.offset + PAGE_SIZE < page.total else None
    choices = ((manage.LIBRARY_WIDE, LIBRARY_WIDE_LABEL), *sorted(names.items(), key=lambda kv: kv[1].lower()))
    return RecordsList(rows=rows, total=page.total, form=form, base=base, graph_choices=choices,
                       state_choices=STATE_CHOICES, client_choices=clients, prev_href=prev_href,
                       next_href=next_href, here=href(form.offset))


# --- the record page ----------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceView:
    n: int
    kind: str                      # span | claim | paper
    item_id: str
    paper_id: str | None
    paper_title: str
    year: int | None
    page: str | None               # "p. 3" / "pp. 3-4"
    quote: str | None
    claim_type: str | None
    label: str | None
    text: str | None
    trust_label: str | None
    trust_note: str | None
    note: str | None
    why: str | None
    check: str | None
    flag: str | None
    opened_by: tuple[str, ...]
    href: str | None
    edited_since: bool


@dataclass(frozen=True)
class HitView:
    kind: str
    item_id: str
    paper: str | None
    tag: str | None


@dataclass(frozen=True)
class StepView:
    n: int
    at: str
    gap: str | None
    tool: str
    kind: str
    helper: str | None
    query: str | None
    scope: tuple[str, ...]         # "kind spans", "graph DiD reading", ...
    hits: tuple[HitView, ...]
    paper: str | None
    paper_href: str | None
    detail: str | None             # "characters 10-90, section s_1", "concept x: 3 claims"


@dataclass(frozen=True)
class ItemView:
    kind: str
    item_id: str
    paper: str | None
    paper_href: str | None
    label: str
    helpers: tuple[str, ...]


@dataclass(frozen=True)
class RecordPage:
    detail: manage.RecordDetail
    row: RowView
    evidence: tuple[EvidenceView, ...]
    steps: tuple[StepView, ...]
    items: tuple[ItemView, ...]

    @property
    def answered(self) -> bool:
        return self.detail.row.state == paths.STATE_ANSWERED

    @property
    def heading(self) -> str:
        return self.detail.row.title if self.answered else self.detail.row.state_label.capitalize()

    @property
    def opened(self) -> str:
        return local_time(self.detail.row.opened_at)

    @property
    def last_call(self) -> str:
        return local_time(self.detail.row.last_event_at)

    @property
    def answered_at(self) -> str:
        return local_time(self.detail.row.answered_at)

    @property
    def annotated_at(self) -> str:
        return local_time(self.detail.annotated_at)


def _page_text(ev: Mapping[str, Any]) -> str | None:
    if ev.get("page"):
        return f"p. {ev['page']}"
    start, end = ev.get("page_start"), ev.get("page_end")
    if start and end and end != start:
        return f"pp. {start}-{end}"
    return f"p. {start}" if start else None


class _Papers:
    """Per-request lookups: titles, overlay heads, the current claim anchored on a span."""

    def __init__(self, lib: Library, titles: Mapping[str, tuple[str | None, int | None]]) -> None:
        self.lib, self.titles = lib, dict(titles)
        self._heads: dict[str, tuple[str | None, int | None]] = {}
        self._anchored: dict[str, dict[str, str]] = {}

    def name(self, paper_id: str | None) -> str | None:
        if not paper_id:
            return None
        title, year = self.titles.get(paper_id, (None, None))
        name = title or f"paper {paper_id}"
        return f"{name} ({year})" if year else name

    def head(self, paper_id: str) -> tuple[str | None, int | None]:
        if paper_id not in self._heads:
            h = overlay.head(self.lib.conn, paper_id)
            self._heads[paper_id] = (h.overlay_id, h.version_no) if h else (None, None)
        return self._heads[paper_id]

    def claim_on_span(self, paper_id: str, span_id: str) -> str | None:
        if paper_id not in self._anchored:
            interp = interpretation.current(self.lib, paper_id, "researcher")
            self._anchored[paper_id] = {c.span_id: c.claim_id for c in (interp.claims if interp else ())
                                        if c.span_id}
        return self._anchored[paper_id].get(span_id)


def _evidence(n: int, ev: Mapping[str, Any], papers: _Papers) -> EvidenceView:
    pid, kind = ev.get("paper_id"), ev["kind"]
    href = f"/papers/{pid}" if pid else None
    edited = False
    trust = ev.get("trust_label")
    if kind == "claim" and pid:
        href = f"/papers/{pid}#claim-{ev['id']}"
        current = papers.head(pid)
        edited = current[0] is not None and (ev.get("overlay_id"), ev.get("version_no")) != current
    elif kind == "span" and pid:
        claim = papers.claim_on_span(pid, ev["id"])
        href = f"/papers/{pid}" + (f"#claim-{claim}" if claim else "")
        trust = trust or SPAN_TRUST
    return EvidenceView(
        n=n, kind=kind, item_id=ev["id"], paper_id=pid, paper_title=ev.get("title") or f"paper {pid}",
        year=ev.get("year"), page=_page_text(ev), quote=ev.get("quote"), claim_type=ev.get("claim_type"),
        label=ev.get("label"), text=ev.get("text"), trust_label=trust, trust_note=ev.get("trust_note"),
        note=ev.get("note"), why=ev.get("why"), check=ev.get("check_label"), flag=ev.get("flag"),
        opened_by=tuple(ev.get("opened_by") or ()), href=href, edited_since=edited)


def _step(n: int, step: paths.Step, tags: Mapping[paths.ItemKey, str], papers: _Papers,
          names: Mapping[str, str]) -> StepView:
    e = step.event
    a = e.args
    scope: list[str] = []
    hits: list[HitView] = []
    paper_id = a.get("paper_id") if isinstance(a.get("paper_id"), str) else None
    detail = None
    query = None
    if e.kind == "search":
        query = a.get("query") if isinstance(a.get("query"), str) else None
        if a.get("kind"):
            scope.append(f"kind {a['kind']}")
        if a.get("graph"):
            scope.append(f"graph {names.get(str(a['graph']), a['graph'])}")
        scope += [f"{k.replace('_id', '')} {a[k]}" for k in ("family", "paper_id", "lens_id", "section_kind")
                  if a.get(k)]
        for h in e.items:
            if not isinstance(h, Mapping):
                continue
            key = (str(h.get("kind")), str(h.get("id")))
            tag = tags.get(key)
            hits.append(HitView(key[0], key[1], papers.name(h.get("paper_id")) if isinstance(h.get("paper_id"), str)
                                else None, paths.TAG_LABELS[tag] if tag else None))
        paper_id = None
    elif e.kind == "open_text":
        r = next((i for i in e.items if isinstance(i, Mapping)), a)
        paper_id = r.get("paper_id") or paper_id
        detail = f"characters {r.get('start')}-{r.get('end')}"
        if r.get("section_id"):
            detail += f", section {r['section_id']}"
    elif e.kind == "open_concept":
        n_claims = sum(1 for i in e.items if isinstance(i, Mapping))
        detail = f"concept {a.get('handle')}: {_plural(n_claims, 'claim')}"
        if a.get("graph"):
            scope.append(f"graph {names.get(str(a['graph']), a['graph'])}")
    elif e.kind == "open_paper":
        n_claims = sum(1 for i in e.items if isinstance(i, Mapping))
        detail = _plural(n_claims, "claim")
    return StepView(n=n, at=local_time(e.at), gap=step.gap, tool=e.tool, kind=e.kind, helper=e.helper,
                    query=query, scope=tuple(scope), hits=tuple(hits), paper=papers.name(paper_id),
                    paper_href=f"/papers/{paper_id}" if paper_id else None, detail=detail)


def record_page(lib: Library, record_id: str, *, now: datetime | None = None) -> RecordPage | None:
    """The record page, or ``None`` for an unknown record."""
    try:
        d = manage.get_record(lib, record_id, now=now)
    except manage.RecordNotFound:
        return None
    names = graph_names(lib)
    papers = _Papers(lib, d.papers)
    tags = {(i.kind, i.id): i.tag for i in d.path.items}
    evidence = tuple(_evidence(n, ev, papers) for n, ev in enumerate(d.evidence, start=1))
    steps = tuple(_step(n, s, tags, papers, names) for n, s in enumerate(d.path.steps, start=1))
    items = tuple(ItemView(i.kind, i.id, papers.name(i.paper_id), f"/papers/{i.paper_id}" if i.paper_id else None,
                           i.label, i.helpers) for i in d.path.items)
    return RecordPage(detail=d, row=RowView(d.row, _tags(d.row, names), None), evidence=evidence, steps=steps,
                      items=items)


# --- the Home feed ------------------------------------------------------------------------------

@dataclass(frozen=True)
class FeedItem:
    at: str
    text: str
    href: str


def feed(lib: Library, *, now: datetime | None = None, limit: int = 40) -> tuple[FeedItem, ...]:
    """The latest answered records and paths without answer (in progress is left out)."""
    out = []
    for r in manage.list_records(lib, manage.Filters(state=FEED_STATES), limit=limit, now=now).rows:
        c = r.counts
        if r.state == paths.STATE_ANSWERED:
            extra = [f"{c.get('cited', 0)} cited"] + ([_plural(len(r.flags), "flag")] if r.flags else [])
            text = f"record: {shorten(r.question or '')} ({', '.join(extra)})"
        else:
            text = f"path without answer ({_plural(c.get('searches', 0), 'search', 'searches')})"
        out.append(FeedItem(r.when, text, f"/records/{r.record_id}"))
    return tuple(out)
