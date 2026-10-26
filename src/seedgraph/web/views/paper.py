"""The paper page's view models (docs/V3_UI_DESIGN.md §2.4, §3, §4, §5).

Pure reads: every function takes a :class:`Library` and returns frozen dataclasses for the
templates (``paper/*.html``). Writes go through the deep modules from the routes.

- :func:`header` - metadata with provenance, badges, graphs, the Write-notes estimate;
- :func:`notes` - the current interpretation (researcher audience) with trust labels,
  quotes and their page locators, grouped by claim type, plus ``expected_version``;
- :func:`history` - overlay versions, newest first;
- :func:`text_view` - the markdown of one page with code-point offsets (selection fallback);
- :func:`text_selection` - a text-view selection checked as an exact unique anchor;
- :func:`replay` - the side-by-side replay view of a newer extraction;
- :func:`pdf_file` - the paper's PDF blob, looked up by ``paper_id`` only;
- :func:`edit_set` - the fields of an edit form that differ from the displayed claim.

Rules chosen where the design is silent:
- "note state" is one of ``none`` / ``current`` / ``stale`` (the markdown changed since the
  note) / ``edited`` (an overlay exists); "newer extraction" is a separate badge.
- The Write-notes estimate uses ``replace`` when a current note exists (a one-paper batch
  that redoes the note), as the design's header action says.
- A text-view page with no page markers shows the whole markdown.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ... import anchor
from ...library import attention, graphs, pages
from ...library.store import Library, is_held, paper_exists
from ...notes import interpretation, overlay
from ...notes import replay as replay_mod
from ...notes.overlay import NOTE_CLAIM_TYPES

LAYOUTS: tuple[str, ...] = ("split", "pdf", "notes")
PAPER_ID_RE = re.compile(r"^lp_[0-9A-Za-z]{1,64}$")

TITLE_SOURCES: dict[str, str] = {
    "v2": "v2",
    "openalex": "OpenAlex (verified by DOI)",
    "crossref": "Crossref (verified by DOI)",
    "semantic_scholar": "Semantic Scholar (verified by DOI)",
    "ollama": "Ollama (unverified)",
    "user": "you",
    "provider_cache": "provider cache",
}
PDF_CHECK_LABELS: dict[str, str] = {
    "pass": "PDF matches the paper",
    "weak": "PDF probably matches the paper",
    "miss": "this PDF may not be this paper",
    "unknown": "PDF check: could not check",
}
TEXT_QUALITY_LABELS: dict[str, str] = {
    "pass": "text OK",
    "garbled": "text garbled",
    "no_text": "no text layer",
}
CLAIM_TYPE_LABELS: dict[str, str] = {
    "research_question": "Research question",
    "main_contribution": "Main contribution",
    "setting": "Setting",
    "estimand": "Estimand",
    "method": "Methods",
    "model": "Models",
    "data": "Data",
    "identification_assumption": "Identification assumptions",
    "regularity_condition": "Regularity conditions",
    "general_assumption": "Assumptions",
    "result": "Results",
    "limitation": "Limitations",
    "robustness_check": "Robustness checks",
    "open_question": "Open questions",
}
OP_STATUS_LABELS: dict[str, str] = {
    "kept": "your edit is kept",
    "still_hidden": "still hidden",
    "added_kept": "your added claim is kept",
    "added_unanchored": "your quote no longer anchors: kept as an assertion",
    "unmatched": "your edit no longer matches a claim",
    "ambiguous": "your edit matches several new claims equally well",
    "hide_gone": "hidden claim is not in the new extraction",
}
NO_PAGES_NOTICE = "no page information: re-convert to get pages"


def claim_type_label(claim_type: str) -> str:
    return CLAIM_TYPE_LABELS.get(claim_type, claim_type.replace("_", " "))


def claim_type_choices() -> tuple[str, ...]:
    return tuple(sorted(NOTE_CLAIM_TYPES))


def valid_paper_id(paper_id: str) -> bool:
    return bool(PAPER_ID_RE.match(paper_id or ""))


def exists(lib: Library, paper_id: str) -> bool:
    return valid_paper_id(paper_id) and paper_exists(lib.conn, paper_id)


# --- header ------------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphRow:
    slug: str
    display_name: str
    status: str | None = None
    is_seed: bool = False


@dataclass(frozen=True)
class AttentionRow:
    item_id: str
    kind: str
    detail: str | None
    hides_notes: bool


@dataclass(frozen=True)
class NotesEstimate:
    queued: bool
    reason: str | None
    text_tokens: int
    agent_tokens_low: int
    agent_tokens_high: int
    replace: bool


@dataclass(frozen=True)
class Header:
    paper_id: str
    title: str | None
    title_source: str | None
    title_source_label: str
    authors: tuple[str, ...]
    year: int | None
    venue: str | None
    identifiers: tuple[tuple[str, str], ...]
    held: bool
    pdf_check: str | None
    pdf_check_label: str | None
    text_quality: str | None
    text_quality_label: str | None
    converter_version: str | None
    converted: bool
    has_page_info: bool
    note_state: str            # none | current | stale | edited
    edited: bool
    newer_extraction: bool
    attention: tuple[AttentionRow, ...]
    graphs: tuple[GraphRow, ...]
    other_graphs: tuple[GraphRow, ...]
    estimate: NotesEstimate | None

    @property
    def display_title(self) -> str:
        return self.title or "(untitled paper)"


def _paper_row(lib: Library, paper_id: str):
    return lib.conn.execute(
        "SELECT paper_id, title, title_source, authors_json, year, venue FROM papers WHERE paper_id = ?",
        (paper_id,)).fetchone()


def _authors(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    try:
        value = json.loads(raw)
    except ValueError:
        return (raw,)
    return tuple(str(a) for a in value) if isinstance(value, list) else (str(value),)


def _graph_rows(lib: Library, paper_id: str) -> tuple[tuple[GraphRow, ...], tuple[GraphRow, ...]]:
    mine: list[GraphRow] = []
    others: list[GraphRow] = []
    for info in graphs.list_graphs(lib):
        with graphs.open_graph(lib, info.slug) as g:
            row = next((m for m in graphs.membership(g) if m.paper_id == paper_id), None)
        if row is None:
            others.append(GraphRow(info.slug, info.display_name))
        else:
            mine.append(GraphRow(info.slug, info.display_name, row.status, row.is_seed))
    return tuple(mine), tuple(others)


def _estimate(lib: Library, paper_id: str, replace: bool) -> NotesEstimate | None:
    from ...work.estimate import estimate  # imported lazily: work pulls in the task registry

    try:
        est = estimate(lib, "notes", [paper_id], replace=replace)
    except Exception:  # noqa: BLE001 - the header still renders without an estimate
        return None
    [p] = est.papers
    return NotesEstimate(p.state == "queued", p.reason, p.text_tokens, p.agent_tokens_low,
                         p.agent_tokens_high, replace)


def header(lib: Library, paper_id: str) -> Header | None:
    """The header of the paper page, or ``None`` for an unknown paper."""
    if not valid_paper_id(paper_id):
        return None
    row = _paper_row(lib, paper_id)
    if row is None:
        return None
    conn = lib.conn
    held = is_held(conn, paper_id)
    pf = conn.execute("SELECT source_file_id, markdown_id, pdf_check_status FROM paper_files "
                      "WHERE paper_id = ? AND role = 'primary'", (paper_id,)).fetchone()
    md_id = pf["markdown_id"] if pf else None
    tq = conv = None
    if md_id:
        r = conn.execute("SELECT status FROM text_quality WHERE markdown_id = ?", (md_id,)).fetchone()
        tq = r[0] if r else None
        r = conn.execute("SELECT c.converter_version FROM markdown_documents m JOIN conversion_runs c "
                         "ON c.conversion_run_id = m.conversion_run_id WHERE m.markdown_id = ?",
                         (md_id,)).fetchone()
        conv = r[0] if r else None
    has_pages = bool(md_id) and pages.PageAssigner(lib).has_markers(md_id)
    interp = interpretation.current(lib, paper_id, "researcher")
    edited = overlay.has_edits(conn, paper_id)
    if interp is None:
        note_state = "none"
    elif edited:
        note_state = "edited"
    else:
        from ...work.tasks_notes import note_freshness

        note_state = "current" if note_freshness(conn, paper_id) == "fresh" else "stale"
    items = tuple(AttentionRow(i.item_id, i.kind, i.detail, i.hides_notes)
                  for i in attention.list_items(lib, status="open", paper_id=paper_id))
    mine, others = _graph_rows(lib, paper_id)
    source = row["title_source"]
    return Header(
        paper_id=paper_id, title=row["title"], title_source=source,
        title_source_label=TITLE_SOURCES.get(source or "", source or "unknown"),
        authors=_authors(row["authors_json"]), year=row["year"], venue=row["venue"],
        identifiers=tuple((r[0], r[1]) for r in conn.execute(
            "SELECT id_type, id_value FROM identifiers WHERE paper_id = ? ORDER BY id_type, id_value",
            (paper_id,))),
        held=held, pdf_check=pf["pdf_check_status"] if pf else None,
        pdf_check_label=PDF_CHECK_LABELS.get(pf["pdf_check_status"]) if pf else None,
        text_quality=tq, text_quality_label=TEXT_QUALITY_LABELS.get(tq or "") if tq else None,
        converter_version=conv, converted=bool(md_id), has_page_info=has_pages,
        note_state=note_state, edited=edited,
        newer_extraction=bool(interp and interp.newer_extraction), attention=items,
        graphs=mine, other_graphs=others,
        estimate=_estimate(lib, paper_id, replace=interp is not None) if held else None)


# --- notes pane --------------------------------------------------------------------------


@dataclass(frozen=True)
class Locator:
    page: int | None
    page_end: int | None
    how: str
    needle: str
    warning: str | None


@dataclass(frozen=True)
class ClaimRow:
    claim_id: str
    claim_type: str
    claim_type_label: str
    normalized_label: str | None
    claim_text: str | None
    plain_words: str | None
    status: str
    trust: str
    trust_label: str
    trust_note: str | None
    edited: bool
    kind: str                     # agent | quote | assertion
    hidden: bool
    quote: str | None
    locator: Locator | None
    original: interpretation.OriginalClaim | None

    @property
    def added(self) -> bool:
        return self.kind in ("quote", "assertion")


@dataclass(frozen=True)
class ClaimGroup:
    claim_type: str
    label: str
    claims: tuple[ClaimRow, ...]


@dataclass(frozen=True)
class NotesPane:
    paper_id: str
    has_note: bool
    expected_version: int
    groups: tuple[ClaimGroup, ...]
    researcher_notes: str | None
    newer_extraction: bool
    edited: bool
    show_hidden: bool
    has_page_info: bool
    claim_types: tuple[str, ...]
    message: str | None = None
    message_level: str = "info"
    history: History | None = None

    @property
    def claims(self) -> tuple[ClaimRow, ...]:
        return tuple(c for g in self.groups for c in g.claims)


def _quotes(lib: Library, span_ids: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for sid in span_ids:
        r = lib.conn.execute("SELECT exact_quote FROM evidence_spans WHERE span_id = ?", (sid,)).fetchone()
        if r is not None:
            out[sid] = r[0]
    return out


def notes(lib: Library, paper_id: str, *, show_hidden: bool = False, message: str | None = None,
          message_level: str = "info") -> NotesPane:
    """The notes pane: the researcher's view of the current interpretation."""
    interp = interpretation.current(lib, paper_id, "researcher", show_hidden=show_hidden)
    md_id, _ = pages.current_markdown(lib, paper_id)
    has_pages = bool(md_id) and pages.PageAssigner(lib).has_markers(md_id)
    expected = overlay.head_version_no(lib.conn, paper_id)
    if interp is None:
        return NotesPane(paper_id, False, expected, (), None, False, overlay.has_edits(lib.conn, paper_id),
                         show_hidden, has_pages, claim_type_choices(), message, message_level,
                         history(lib, paper_id))
    locs = {loc.claim_id: loc for loc in pages.quote_locators(lib, paper_id)}
    quotes = _quotes(lib, [c.span_id for c in interp.claims if c.span_id])
    groups: dict[str, list[ClaimRow]] = {}
    for c in interp.claims:
        if c.claim_type.startswith("lens:"):
            continue
        loc = locs.get(c.claim_id)
        groups.setdefault(c.claim_type, []).append(ClaimRow(
            claim_id=c.claim_id, claim_type=c.claim_type, claim_type_label=claim_type_label(c.claim_type),
            normalized_label=c.normalized_label, claim_text=c.claim_text, plain_words=c.plain_words,
            status=c.status, trust=c.trust, trust_label=c.trust_label, trust_note=c.trust_note,
            edited=c.edited_by_researcher, kind=c.kind, hidden=c.hidden,
            quote=quotes.get(c.span_id) if c.span_id else None,
            locator=Locator(loc.page, loc.page_end, loc.how, loc.needle, loc.warning) if loc else None,
            original=c.original))
    return NotesPane(
        paper_id=paper_id, has_note=True, expected_version=expected,
        groups=tuple(ClaimGroup(t, claim_type_label(t), tuple(rows)) for t, rows in groups.items()),
        researcher_notes=interp.researcher_notes, newer_extraction=interp.newer_extraction,
        edited=interp.overlay_id is not None, show_hidden=show_hidden, has_page_info=has_pages,
        claim_types=claim_type_choices(), message=message, message_level=message_level,
        history=history(lib, paper_id))


def edit_set(lib: Library, paper_id: str, claim_id: str, form: Mapping[str, Any]) -> dict[str, str]:
    """The editable fields of ``form`` that differ from the claim as displayed now.

    An unchanged submit therefore writes nothing (it would otherwise flip the claim to
    "researcher")."""
    interp = interpretation.current(lib, paper_id, "researcher", show_hidden=True)
    claim = next((c for c in (interp.claims if interp else ()) if c.claim_id == claim_id), None)
    if claim is None:
        return {k: str(form[k]) for k in overlay.EDITABLE_FIELDS if form.get(k) is not None}
    out: dict[str, str] = {}
    for key in overlay.EDITABLE_FIELDS:
        value = form.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value != (getattr(claim, key) or ""):
            out[key] = value
    return out


# --- history -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryRow:
    version_no: int
    action: str
    summary: str | None
    created_at: str
    reverted_to: int | None
    is_head: bool


@dataclass(frozen=True)
class History:
    paper_id: str
    expected_version: int
    rows: tuple[HistoryRow, ...]


def history(lib: Library, paper_id: str) -> History:
    versions = overlay.history(lib.conn, paper_id)
    head_no = versions[0].version_no if versions else 0
    return History(paper_id, head_no, tuple(
        HistoryRow(v.version_no, v.action, v.summary, v.created_at, v.reverted_to, v.version_no == head_no)
        for v in versions))


# --- text view (selection fallback) ------------------------------------------------------


@dataclass(frozen=True)
class TextChunk:
    offset: int   # code-point offset of ``text`` in the markdown
    text: str


@dataclass(frozen=True)
class TextView:
    paper_id: str
    markdown_id: str | None
    page: int | None
    page_count: int
    has_page_info: bool
    chunks: tuple[TextChunk, ...]

    @property
    def pages(self) -> range:
        return range(1, self.page_count + 1)


_PARA = re.compile(r"\n{2,}")


def _chunks(md: str, start: int, end: int) -> tuple[TextChunk, ...]:
    out: list[TextChunk] = []
    pos = start
    for m in _PARA.finditer(md, start, end):
        if m.start() > pos:
            out.append(TextChunk(pos, md[pos:m.start()]))
        pos = m.end()
    if end > pos:
        out.append(TextChunk(pos, md[pos:end]))
    return tuple(out)


def text_view(lib: Library, paper_id: str, page: int | None = None) -> TextView:
    md_id, _ = pages.current_markdown(lib, paper_id)
    assigner = pages.PageAssigner(lib)
    md = assigner.markdown_text(md_id)
    if md is None:
        return TextView(paper_id, md_id, None, 0, False, ())
    bounds = assigner.bounds(md_id)
    if not bounds:
        return TextView(paper_id, md_id, None, 0, False, _chunks(md, 0, len(md)))
    numbers = [p for _, p in bounds]
    count = max(numbers)
    page = page if page in numbers else numbers[0]
    i = numbers.index(page)
    start = bounds[i][0]
    end = bounds[i + 1][0] if i + 1 < len(bounds) else len(md)
    # end at the next delimiter line, not after it
    nxt = md.rfind("\n", start, end) if i + 1 < len(bounds) else -1
    if nxt > start and re.fullmatch(r"\{\d+\}-{3,}", md[nxt + 1:end].strip()):
        end = nxt
    return TextView(paper_id, md_id, page, count, True, _chunks(md, start, end))


def text_selection(lib: Library, paper_id: str, start: int, end: int) -> pages.AnchorResult:
    """A selection made in the text view (code-point offsets into the current markdown),
    widened to whole words and required to anchor exactly and uniquely (design §5.4)."""
    md_id, _ = pages.current_markdown(lib, paper_id)
    md = pages.PageAssigner(lib).markdown_text(md_id)
    if md is None:
        return pages.AnchorResult("no_markdown", message="the paper has no converted text")
    if not (0 <= start < end <= len(md)):
        return pages.AnchorResult("not_found", markdown_id=md_id, message="the selection is outside the text")
    while start < end and md[start].isspace():
        start += 1
    while end > start and md[end - 1].isspace():
        end -= 1
    if start >= end:
        return pages.AnchorResult("empty", markdown_id=md_id, message="the selection is empty")
    start, end = pages._widen_to_words(md, start, end)  # noqa: SLF001 - the shared word rule
    quote = md[start:end]
    if anchor.locate_quote(md, quote) != (start, end):
        return pages.AnchorResult("ambiguous", markdown_id=md_id, matches=2,
                                  message="the passage also appears elsewhere in the paper; "
                                          "select a longer passage")
    return pages.AnchorResult("ok", quote=quote, start_char=start, end_char=end, markdown_id=md_id, matches=1)


# --- replay view -------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimSide:
    claim_id: str | None
    claim_type: str | None
    normalized_label: str | None
    claim_text: str | None


@dataclass(frozen=True)
class ReplayOpRow:
    key: str
    op: str                     # edit | hide | add
    status: str
    status_label: str
    yours: ClaimSide            # the researcher's version (or the hidden claim)
    agent_old: ClaimSide | None
    agent_new: ClaimSide | None
    candidates: tuple[ClaimSide, ...]
    needs_decision: bool
    keep_as: str | None


@dataclass(frozen=True)
class ReplayView:
    paper_id: str
    head_version_no: int
    new_note_id: str
    ops: tuple[ReplayOpRow, ...]
    new_claims: tuple[ClaimSide, ...]
    changed: tuple[tuple[ClaimSide, ClaimSide], ...]
    gone: tuple[ClaimSide, ...]
    kept: int
    still_hidden: int
    unmatched: int
    hides_gone: int
    error: str | None = None


def _side(lib: Library, claim_id: str | None) -> ClaimSide | None:
    if claim_id is None:
        return None
    r = lib.conn.execute("SELECT claim_type, normalized_label, claim_text FROM extracted_claims WHERE claim_id = ?",
                         (claim_id,)).fetchone()
    return ClaimSide(claim_id, r[0], r[1], r[2]) if r else ClaimSide(claim_id, None, None, None)


def replay_plan(lib: Library, paper_id: str) -> replay_mod.ReplayPlan:
    return replay_mod.plan_replay(lib.conn, paper_id)


def replay(lib: Library, paper_id: str) -> ReplayView | None:
    """The replay view, or ``None`` when there is no newer extraction to compare."""
    try:
        plan = replay_plan(lib, paper_id)
    except replay_mod.ReplayError:
        return None
    rows = []
    for p in plan.ops:
        kind = p.op["op"]
        old = _side(lib, p.old_claim_id)
        if kind == "edit":
            s = p.op["set"]
            yours = ClaimSide(p.old_claim_id, s.get("claim_type", old.claim_type if old else None),
                              s.get("normalized_label", old.normalized_label if old else None),
                              s.get("claim_text", old.claim_text if old else None))
        elif kind == "add":
            yours = ClaimSide(p.op["add_id"], p.op["claim_type"], p.op["normalized_label"], p.op["claim_text"])
        else:
            yours = old or ClaimSide(None, None, None, None)
        new = (ClaimSide(p.new_claim_id, p.new_agent.claim_type, p.new_agent.normalized_label,
                         p.new_agent.claim_text) if p.new_agent else _side(lib, p.new_claim_id))
        rows.append(ReplayOpRow(
            key=p.key, op=kind, status=p.status, status_label=OP_STATUS_LABELS.get(p.status, p.status),
            yours=yours, agent_old=old, agent_new=new,
            candidates=tuple(s for s in (_side(lib, c) for c in p.candidates) if s),
            needs_decision=p in plan.needs_decision, keep_as=p.keep_as))
    return ReplayView(
        paper_id=paper_id, head_version_no=plan.head_version_no, new_note_id=plan.new_note_id,
        ops=tuple(rows),
        new_claims=tuple(ClaimSide(c.claim_id, c.claim_type, c.normalized_label, c.claim_text)
                         for c in plan.new_claims),
        changed=tuple((ClaimSide(w.old_claim_id, None, w.old_label, w.old_text),
                       ClaimSide(w.claim_id, None, w.new_label, w.new_text)) for w in plan.changed),
        gone=tuple(s for s in (_side(lib, g) for g in plan.gone) if s),
        kept=plan.kept, still_hidden=plan.still_hidden, unmatched=plan.unmatched, hides_gone=plan.hides_gone)


# --- the PDF blob ------------------------------------------------------------------------


def pdf_file(lib: Library, paper_id: str) -> Path | None:
    """The paper's primary PDF inside the library, or ``None`` (unknown, cited-only, missing)."""
    if not valid_paper_id(paper_id):
        return None
    _, sf_id = pages.current_markdown(lib, paper_id)
    if sf_id is None:
        return None
    path = pages.pdf_path(lib, sf_id)
    if path is None:
        return None
    root = lib.root.resolve()
    resolved = path.resolve()
    if root not in resolved.parents:
        return None
    return resolved


@dataclass(frozen=True)
class PaperPage:
    header: Header
    notes: NotesPane
    layout: str
    pdf_url: str | None


def page(lib: Library, paper_id: str, *, layout: str = "split", show_hidden: bool = False) -> PaperPage | None:
    h = header(lib, paper_id)
    if h is None:
        return None
    return PaperPage(header=h, notes=notes(lib, paper_id, show_hidden=show_hidden),
                     layout=layout if layout in LAYOUTS else "split",
                     pdf_url=f"/papers/{paper_id}/pdf" if pdf_file(lib, paper_id) else None)
