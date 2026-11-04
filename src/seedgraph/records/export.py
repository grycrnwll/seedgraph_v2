"""Export a record to markdown (design §3.8; slice R2).

:func:`export_markdown` lays out one record from :func:`.manage.get_record`, the loader the
record page uses, so the page and the export never disagree:

- a header list: record id, state, session (name or start date/time) and client, dates,
  graphs (or Library-wide), and the record notes;
- the question as the title (or "Path without answer" / "In progress"); the answer; the
  **Agent's judgement** as its own section;
- the cited evidence: each frozen quote or claim text, with paper title, year and page,
  its trust label, its check and any flag;
- the path: each search with its query and its hits (tagged), each paper or section
  opened, in order, with long gaps; then every item with its tag;
- the researcher's annotation.

Only ids, titles and the frozen cited text go in: no PDF, no paper text and no search
previews.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone, tzinfo
from typing import Any

from ..library.store import Library
from . import manage
from . import path as paths


def _when(ts: str | None, tz: tzinfo | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    if not ts:
        return "-"
    dt = datetime.fromisoformat(ts)
    if tz is None or tz == timezone.utc:
        return dt.astimezone(timezone.utc).strftime(fmt) + (" UTC" if fmt != "%H:%M" else "")
    return dt.astimezone(tz).strftime(fmt)


def _paper(papers: Mapping[str, tuple[str | None, int | None]], paper_id: str | None) -> str:
    if not paper_id:
        return "unknown paper"
    title, year = papers.get(paper_id, (None, None))
    name = title or f"paper {paper_id}"
    return f"{name} ({year})" if year else name


def _page(ev: Mapping[str, Any]) -> str:
    if ev.get("page"):
        return f", page {ev['page']}"
    start, end = ev.get("page_start"), ev.get("page_end")
    if start and end and end != start:
        return f", pages {start}-{end}"
    return f", page {start}" if start else ""


def _quote(text: str) -> list[str]:
    return ["   > " + line if line else "   >" for line in text.splitlines()]


def _evidence(n: int, ev: Mapping[str, Any], papers: Mapping[str, tuple[str | None, int | None]]) -> list[str]:
    title = ev.get("title") or f"paper {ev.get('paper_id')}"
    head = f"{title} ({ev['year']})" if ev.get("year") else title
    out = [f"{n}. {head}{_page(ev)}" + (f", {ev['trust_label']}" if ev.get("trust_label") else "")]
    if ev["kind"] == "claim":
        label = ", ".join(x for x in (ev.get("claim_type"), ev.get("label")) if x)
        out.append(f"   Claim ({label}): {ev.get('text') or ''}".rstrip())
        if ev.get("trust_note"):
            out.append(f"   {ev['trust_note']}")
        if ev.get("note"):
            out.append(f"   Note: {ev['note']}")
    if ev.get("quote"):
        out += _quote(ev["quote"])
    if ev.get("why"):
        out.append(f"   Why: {ev['why']}")
    check = ev.get("check_label")
    if check and not ev.get("flag"):
        helpers = ev.get("opened_by") or []
        out.append(f"   Check: {check}" + "".join(f" (opened by helper {h})" for h in helpers))
    if ev.get("flag"):
        out.append(f"   **Flag: {ev['flag']}**")
    return out


def _item_name(kind: str, item_id: str) -> str:
    return f"{kind} `{item_id}`"


def _step(n: int, step: paths.Step, tags: Mapping[paths.ItemKey, str],
          papers: Mapping[str, tuple[str | None, int | None]], tz: tzinfo | None) -> list[str]:
    e = step.event
    out = [f"   _{step.gap}_"] if step.gap else []
    who = f" (helper {e.helper})" if e.helper else ""
    at = _when(e.at, tz, "%H:%M")
    if e.kind == "search":
        query = e.args.get("query")
        scope = [str(e.args[k]) for k in ("kind",) if e.args.get(k)]
        scope += [f"{k} {e.args[k]}" for k in ("graph", "family", "paper_id", "lens_id") if e.args.get(k)]
        hits = [h for h in e.items if isinstance(h, Mapping)]
        what = f'"{query}"' if query else ""
        where = f" ({'; '.join(scope)})" if scope else ""
        out.append(f"{n}. {at} {e.tool} {what}{where}: {len(hits)} hit{'s' if len(hits) != 1 else ''}{who}"
                   .replace("  ", " "))
        for h in hits:
            key = (h.get("kind"), h.get("id"))
            name = _item_name(str(h.get("kind")), str(h.get("id")))
            paper = f", {_paper(papers, h['paper_id'])}" if isinstance(h.get("paper_id"), str) else ""
            tag = tags.get(key)  # type: ignore[arg-type]
            out.append(f"   - {name}{paper}" + (f": {paths.TAG_LABELS[tag]}" if tag else ""))
        return out
    paper_id = e.args.get("paper_id")
    detail = ""
    if e.kind == "open_text":
        ranges = [i for i in e.items if isinstance(i, Mapping)] or [e.args]
        r = ranges[0]
        paper_id = r.get("paper_id") or paper_id
        detail = f", characters {r.get('start')}-{r.get('end')}"
        if r.get("section_id"):
            detail += f", section `{r['section_id']}`"
    elif e.kind == "open_concept":
        detail = f" `{e.args.get('handle')}`" + (f" (graph {e.args['graph']})" if e.args.get("graph") else "")
        n_claims = sum(1 for i in e.items if isinstance(i, Mapping))
        out.append(f"{n}. {at} {e.tool}{detail}: {n_claims} claim{'s' if n_claims != 1 else ''}{who}")
        return out
    out.append(f"{n}. {at} {e.tool} {_paper(papers, paper_id)}{detail}{who}")
    return out


def export_markdown(lib: Library, record_id: str, *, now: datetime | None = None,
                    tz: tzinfo | None = None) -> str:
    """The record as one markdown document. ``tz`` is the zone of the shown times
    (default UTC). Raises :class:`.manage.RecordNotFound`."""
    d = manage.get_record(lib, record_id, now=now)
    row = d.row
    answered = row.state == paths.STATE_ANSWERED
    title = row.question if answered and row.question else row.state_label.capitalize()
    lines = [f"# {title}", ""]
    lines.append(f"- Record: `{row.record_id}`")
    lines.append(f"- State: {row.state_label}")
    lines.append(f"- Session: {row.session_name}" + (f" ({row.client})" if row.client else ""))
    lines.append(f"- Opened: {_when(row.opened_at, tz)}")
    if answered:
        lines.append(f"- Answered: {_when(row.answered_at, tz)}")
    else:
        lines.append(f"- Last call: {_when(row.last_event_at, tz)}")
    if row.graphs:
        lines.append("- Graphs: " + ", ".join(
            f"{g.slug} ({manage.CITED_MEMBER_MARK if g.how == 'cited_member' else g.how}"
            + (", graph deleted" if g.deleted else "") + ")" for g in row.graphs))
    else:
        lines.append("- Graphs: Library-wide")
    if row.spans_label:
        lines.append(f"- {row.spans_label[0].upper()}{row.spans_label[1:]}")
    c = row.counts
    n_search = c.get("searches", 0)
    lines.append(f"- Counts: {n_search} search{'es' if n_search != 1 else ''}, {c.get('opened', 0)} opened, "
                 f"{c.get('cited', 0)} cited")
    if row.notes:
        lines.append("- Notes: " + "; ".join(row.notes))
    lines.append("")

    if answered:
        lines += ["## Answer", "", d.answer or "", ""]
        lines += ["## Agent's judgement", "", d.judgement or "(none given)", ""]
        lines += ["## Cited evidence", ""]
        for n, ev in enumerate(d.evidence, start=1):
            lines += _evidence(n, ev, d.papers)
            lines.append("")

    tags = {(i.kind, i.id): i.tag for i in d.path.items}
    lines += ["## Path", ""]
    if not d.path.steps:
        lines.append("No searches or reads in this record (answered from earlier reads in the session).")
    for n, step in enumerate(d.path.steps, start=1):
        lines += _step(n, step, tags, d.papers, tz)
    lines.append("")

    lines += ["## Items", ""]
    if d.path.items:
        lines += ["| item | paper | tag |", "|---|---|---|"]
        for i in d.path.items:
            paper = _paper(d.papers, i.paper_id) if i.paper_id else ""
            lines.append(f"| {_item_name(i.kind, i.id)} | {paper} | {i.label} |")
    else:
        lines.append("None.")
    lines.append("")

    if row.annotation:
        lines += ["## Annotation", "", row.annotation, ""]
    return "\n".join(lines).rstrip() + "\n"
