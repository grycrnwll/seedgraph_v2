"""`seedgraph-v3 records ...`: the stage 5 acceptance harness (design §10; slice R8).

Thin adapters over :mod:`seedgraph.records.acceptance`:

    records acceptance-report [--sample ID] [--out DIR]   write the checklists and the report
    records check-transcript TRANSCRIPT [--record ID]... [--server NAME] [--no-save]

Reports go to ``<home>/acceptance/stage5/`` (``questions.yaml``, ``records_check.md``,
``walkthrough_check.md``, ``crosscheck.json``, ``report.md`` / ``report.json``,
``spotcheck.md`` / ``spotcheck.json``). The library is opened read-only. The two checklist
files are the researcher's: they are written once and never rewritten (new records are only
appended to ``records_check.md``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from . import _common

records_app = typer.Typer(name="records", help="Traversal records: the stage 5 acceptance harness.",
                          no_args_is_help=True)

ROOT = _common.ROOT_OPTION
JSON = _common.JSON_OPTION


@records_app.callback()
def _utf8() -> None:
    _common.force_utf8_stdout()


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


@records_app.command("acceptance-report")
def acceptance_report_cmd(
    sample: Optional[str] = typer.Option(None, "--sample", help="The spot-check sample (default: the newest)."),
    out: Optional[Path] = typer.Option(None, "--out", help="Report folder (default <home>/acceptance/stage5)."),
    root: Optional[Path] = ROOT,
) -> None:
    """Write the per-record checklist, the walk-through checklist and the stage 5 report (§10)."""
    from ..records import acceptance as acc
    from ..records import manage
    from .concepts_work import readonly_library

    home = _common.home(root)
    out_dir = out or acc.stage_dir(home)
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    qfile = out_dir / acc.QUESTIONS_FILE
    if not qfile.is_file():
        qfile.write_text(acc.QUESTIONS_TEMPLATE, encoding="utf-8")
        notes.append(f"wrote {qfile}: write the five questions there, then set `record:` after each answer")
    try:
        questions = acc.load_questions(qfile)
    except acc.AcceptanceError as exc:
        _common.fail(str(exc))
    checklist_path, walk_path = out_dir / acc.CHECKLIST_FILE, out_dir / acc.WALKTHROUGH_FILE
    prev_path = out_dir / acc.SPOTCHECK_JSON
    previous = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.is_file() else None
    cross_path = out_dir / acc.CROSSCHECK_FILE
    cross = json.loads(cross_path.read_text(encoding="utf-8")) if cross_path.is_file() else {}
    with _common.cli_errors(), readonly_library(root) as lib:
        details = []
        for q in questions:
            if not q.record:
                continue
            try:
                details.append((manage.get_record(lib, q.record), q.id))
            except manage.RecordNotFound:
                notes.append(f"{q.id}: no record {q.record} in the library")
        text, added = acc.sync_checklist(_read(checklist_path), details)
        walk = acc.sync_walkthrough(_read(walk_path))
        data = acc.gather(lib, home, sample_id=sample, checklist=text, walkthrough=walk, cross_checks=cross,
                          questions=questions, previous_spotcheck=previous)
    if added or not checklist_path.is_file():
        checklist_path.write_text(text, encoding="utf-8")
    if not walk_path.is_file():
        walk_path.write_text(walk, encoding="utf-8")
    sc = data["spotcheck"]
    (out_dir / acc.SPOTCHECK_MD).write_text(acc.render_spotcheck(sc), encoding="utf-8")
    if sc is not None:
        prev_path.write_text(json.dumps(sc, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    (out_dir / acc.REPORT_FILE).write_text(acc.render(data), encoding="utf-8")
    (out_dir / acc.REPORT_JSON).write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False),
                                           encoding="utf-8")
    for n in notes:
        typer.echo(n)
    if added:
        typer.echo(f"checklist: added {', '.join(added)}")
    typer.echo(f"report: {out_dir / acc.REPORT_FILE}")
    typer.echo(f"checklist: {checklist_path}")
    typer.echo(f"walk-through: {walk_path}")
    s = data["questions_summary"]
    typer.echo(f"questions: {'PASS' if data['questions_pass'] else 'not yet'} "
               f"({s['linked']} of {s['asked']} linked, {s['checklists_complete']} checklists complete, "
               f"{s['cross_checks_clean']} cross-checks clean)")
    if sc is not None:
        typer.echo(f"spot-check: marked {sc['marked']} of {sc['items']}, fixed {len(sc['fixed'])}, "
                   f"figure unchanged after a fix: {sc['unchanged_after_fix']}")


@records_app.command("check-transcript")
def check_transcript_cmd(
    transcript: Path = typer.Argument(..., help="A Claude Code session transcript (.jsonl). Subagent transcripts "
                                      "in the folder beside it are read too."),
    record: Optional[List[str]] = typer.Option(None, "--record", help="A record id to check (repeatable). Default: "
                                               "every event inside the transcript's time span."),
    server: str = typer.Option("seedgraph_v3", "--server", help="The MCP server name in Claude Code's config."),
    save: bool = typer.Option(True, "--save/--no-save", help="Keep the result in <home>/acceptance/stage5/"
                              "crosscheck.json (the report reads it)."),
    root: Optional[Path] = ROOT,
    as_json: bool = JSON,
) -> None:
    """Compare a Claude Code transcript's seedgraph calls with the session log (§10.1, Claude Code only)."""
    from ..records import acceptance as acc
    from .concepts_work import readonly_library

    home = _common.home(root)
    try:
        tr = acc.read_transcript(transcript, server=server)
    except acc.AcceptanceError as exc:
        _common.fail(str(exc))
    with _common.cli_errors(), readonly_library(root) as lib:
        try:
            checks = acc.check_transcript(lib.conn, tr, record_ids=list(record or ()))
        except acc.AcceptanceError as exc:
            _common.fail(str(exc))
    if save:
        acc.save_cross_checks(home, checks)
    if as_json:
        _common.echo_json({k: acc.cross_check_json(v) for k, v in checks.items()})
    else:
        typer.echo(f"transcript files: {', '.join(tr.files)}")
        for name, cc in checks.items():
            typer.echo(acc.render_cross_check(name, cc))
    if not all(cc.ok for cc in checks.values()):
        raise typer.Exit(_common.EXIT_FAILED)
