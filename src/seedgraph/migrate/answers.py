"""v2 answers export (design §6.11, ticket 13 decision 8).

Every saved answer of a carried v2 project (``projects/<slug>/answers/<id>.json``
plus its ``<id>.trace.json``) is rendered once to
``v2_answers/<slug>/<answer_id>.md``: question, answer, cited evidence (with the
library paper each cited v2 work became), a trace summary and the original JSON in
fenced blocks. ``v2_answers/index.md`` lists them by answer time (the trace's
``created_at``; answers without a trace sort last), then ``answer_id``.

Run-scoped answers (``projects/<slug>/runs/*/answers``) are not exported, and answers
are never turned into traversal records. ``dbg_*`` projects are not carried.

The export runs as the migration's ``stage_hook`` (:func:`stage_hook`), so the
folder is built in staging, covered by the canonical dump and swapped in with the
library. Output is deterministic: inputs are read in sorted order, nothing depends on
the wall clock, and files are written as UTF-8 with LF line ends.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from . import StageContext

FOLDER = "v2_answers"
INDEX = "index.md"

PaperOf = Callable[[str, str], "str | None"]


@dataclass(frozen=True)
class ExportedAnswer:
    slug: str
    answer_id: str
    path: str                 # relative to the folder holding ``v2_answers/`` (POSIX)
    source: str               # the v2 envelope, relative to the v2 home
    created_at: str | None    # from the trace; None when there is no trace
    question: str | None


@dataclass
class AnswersExport:
    exported: list[ExportedAnswer] = field(default_factory=list)
    skipped_projects: list[str] = field(default_factory=list)   # not carried (their answers are not exported)
    problems: list[str] = field(default_factory=list)           # unreadable envelopes (exported as raw text)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def _load(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes().decode("utf-8")
    try:
        return json.loads(raw), raw
    except ValueError:
        return None, raw


def _fence(raw: str) -> str:
    ticks = "```"
    while ticks in raw:
        ticks += "`"
    return f"{ticks}json\n{raw.rstrip()}\n{ticks}"


def _cell(value: Any) -> str:
    return "" if value is None else str(value).replace("|", "\\|").replace("\n", " ")


def render_answer(slug: str, answer_id: str, envelope: Any, env_raw: str, trace: Any, trace_raw: str | None,
                  paper_of: PaperOf | None = None) -> str:
    """The markdown page for one answer (pure)."""
    env = envelope if isinstance(envelope, dict) else {}
    tr = trace if isinstance(trace, dict) else {}
    question = env.get("question") or tr.get("question")
    lines = [f"# {question or answer_id}", "",
             f"- Answer id: `{answer_id}`"
             + (f" (the answer file itself says `{env['answer_id']}`)"
                if env.get("answer_id") not in (None, answer_id) else ""),
             f"- v2 project: `{slug}`",
             f"- Asked at: {tr.get('created_at') or 'unknown (no trace)'}",
             f"- Mode: {env.get('mode') or 'unknown'}; category: {env.get('answer_category') or 'unknown'}"]
    if env.get("insufficient_evidence"):
        lines.append("- v2 marked this answer as having insufficient evidence")
    lines += ["", "## Answer", "", (env.get("answer_text") or "_(no answer text)_").rstrip(), ""]
    cites = env.get("citations") or []
    lines += ["## Cited evidence", ""]
    if cites:
        lines += ["| # | Title | Year | v2 work | Library paper | Spans |", "|---|---|---|---|---|---|"]
        for n, c in enumerate(cites, start=1):
            c = c if isinstance(c, dict) else {}
            wid = c.get("work_id")
            pid = paper_of(slug, wid) if (paper_of and wid) else None
            spans = ", ".join(f"`{s}`" for s in c.get("span_ids") or [])
            lines.append(f"| {n} | {_cell(c.get('title'))} | {_cell(c.get('year'))} | `{_cell(wid)}` | "
                         f"{'`' + pid + '`' if pid else 'not carried'} | {spans} |")
    else:
        lines.append("_No citations._")
    lines += ["", "## Trace summary", ""]
    if tr:
        for label, key in (("Path", "path"), ("Outcome", "outcome"), ("Retrieved items", "retrieved_count"),
                           ("Degrade reasons", "degrade_reasons"), ("Prompt version", "prompt_version"),
                           ("seedgraph version", "seedgraph_version")):
            value = tr.get(key)
            if isinstance(value, list):
                value = ", ".join(map(str, value)) or "none"
            lines.append(f"- {label}: {value if value is not None else 'none'}")
    else:
        lines.append("_No trace file was saved for this answer._")
    lines += ["", "## Original answer JSON", "", _fence(env_raw), ""]
    if trace_raw is not None:
        lines += ["## Original trace JSON", "", _fence(trace_raw), ""]
    return "\n".join(lines)


def _sort_key(e: ExportedAnswer) -> tuple[int, str, str, str]:
    return (e.created_at is None, e.created_at or "", e.answer_id, e.slug)


def render_index(exported: Sequence[ExportedAnswer]) -> str:
    lines = ["# v2 answers", "",
             "Saved answers from seedgraph v2, exported once by the migration. They are a read-only record; "
             "v3 does not use them for search or traversal.", ""]
    if not exported:
        lines.append("_No saved answers were found in the carried v2 projects._")
    else:
        lines += [f"{len(exported)} answers, oldest first.", "",
                  "| Asked at | Project | Question | File |", "|---|---|---|---|"]
        for e in sorted(exported, key=_sort_key):
            rel = e.path.split("/", 1)[1] if "/" in e.path else e.path
            lines.append(f"| {e.created_at or 'unknown'} | {e.slug} | {_cell(e.question)} | [{e.answer_id}]({rel}) |")
    return "\n".join(lines) + "\n"


def export_answers(v2_home: Path | str, answer_files: Mapping[str, Sequence[str]], dest: Path | str, *,
                   projects: Mapping[str, str], paper_of: PaperOf | None = None) -> AnswersExport:
    """Render every saved answer of a carried project into ``dest`` (the ``v2_answers`` folder).

    ``answer_files`` is :attr:`Snapshot.answer_files` (slug -> envelope paths relative
    to the v2 home, ``.trace.json`` excluded). ``projects`` is the plan's project
    outcome map (``plan.report['projects']``): slugs marked ``not_carried`` (or
    absent) are skipped. ``paper_of(slug, work_id)`` names the library paper a cited
    work became. ``index.md`` is always written.
    """
    v2_home, dest = Path(v2_home), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = AnswersExport()
    for slug in sorted(answer_files):
        if projects.get(slug, "not_carried") == "not_carried":
            if answer_files[slug]:
                out.skipped_projects.append(slug)
            continue
        for rel in sorted(answer_files[slug]):
            path = v2_home / rel
            answer_id = path.name[: -len(".json")]
            envelope, env_raw = _load(path)
            if envelope is None:
                out.problems.append(f"{rel}: not valid JSON; exported as raw text")
            tpath = path.with_name(answer_id + ".trace.json")
            trace, trace_raw = _load(tpath) if tpath.is_file() else (None, None)
            if isinstance(envelope, dict) and envelope.get("answer_id") not in (None, answer_id):
                out.problems.append(f"{rel}: the envelope says answer_id {envelope['answer_id']!r}; "
                                    "the page is named after the file")
            page = render_answer(slug, answer_id, envelope, env_raw, trace, trace_raw, paper_of)
            _write(dest / slug / f"{answer_id}.md", page)
            env = envelope if isinstance(envelope, dict) else {}
            tr = trace if isinstance(trace, dict) else {}
            out.exported.append(ExportedAnswer(
                slug=slug, answer_id=answer_id, path=f"{dest.name}/{slug}/{answer_id}.md", source=rel,
                created_at=tr.get("created_at"), question=env.get("question") or tr.get("question")))
    _write(dest / INDEX, render_index(out.exported))
    return out


def plan_paper_of(plan: Any) -> PaperOf:
    """``paper_of`` over a :class:`~seedgraph.migrate.plan.MigrationPlan`'s ledger."""
    def paper_of(slug: str, work_id: str) -> str | None:
        d = plan.ledger.get(slug, "works", work_id)
        return d.v3_id if d is not None and d.v3_table == "papers" else None
    return paper_of


def stage_hook(ctx: StageContext) -> AnswersExport:
    """The migration ``stage_hook``: export into ``<staging>/v2_answers``."""
    return export_answers(ctx.v2_home, ctx.snapshot.answer_files, ctx.staging_dir / FOLDER,
                          projects=ctx.plan.report.get("projects", {}), paper_of=plan_paper_of(ctx.plan))
