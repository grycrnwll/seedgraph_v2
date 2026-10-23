"""`seedgraph migrate run|verify|parity|accept` (design §6, §8, §11 gate 2).

Thin adapters: ``run`` over :func:`seedgraph.migrate.report.run_and_report` (the
migration plus its report), ``verify`` over :func:`seedgraph.migrate.dump.first_difference`,
``parity`` over :mod:`seedgraph.migrate.parity` and ``accept`` over
:func:`seedgraph.migrate.accept.accept`. Exit codes: a preflight refusal is 3; a report
that cannot account for every item, a parity or acceptance input error, an unexplained
parity difference, differing dumps or a failed acceptance is 1.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional

import typer

from . import _common

migrate_app = typer.Typer(
    name="migrate",
    help="Migrate a v2 home into the v3 library and its context graphs. The v2 home is only read.",
    no_args_is_help=True,
)

_DEFAULT_V2_HOME = Path.home() / ".seedgraph"
V2_HOME_OPTION = typer.Option(_DEFAULT_V2_HOME, "--v2-home", help="The v2 home to read (never written).")
GRAPH_NAME_OPTION = typer.Option(
    [], "--graph-name", help="Display name for a context graph, as SLUG=Name (repeatable).")


def _graph_names(pairs: List[str]) -> dict[str, str]:
    names: dict[str, str] = {}
    for pair in pairs:
        slug, sep, name = pair.partition("=")
        if not sep or not slug.strip() or not name.strip():
            raise typer.BadParameter(f"expected SLUG=Display name, got {pair!r}", param_hint="--graph-name")
        names[slug.strip()] = name.strip()
    return names


@contextmanager
def _s7_errors() -> Iterator[None]:
    """ParityError / AcceptanceError -> exit 1 (refusals stay exit 3)."""
    from ..migrate.accept import AcceptanceError
    from ..migrate.parity import ParityError

    with _common.cli_errors():
        try:
            yield
        except (ParityError, AcceptanceError) as exc:
            _common.fail(str(exc), _common.EXIT_FAILED)


# --- run ----------------------------------------------------------------------------------------

def _summary(result: Any, report_path: Optional[Path], problems: tuple[str, ...]) -> dict[str, Any]:
    return {
        "ok": result.ok and not problems,
        "run_id": result.run_id,
        "v2_home": str(result.v2_home),
        "v3_home": str(result.v3_home),
        "swapped": result.swapped,
        "library_dir": str(result.library_dir),
        "replaced_dir": str(result.replaced_dir) if result.replaced_dir else None,
        "run_dir": str(result.run_dir),
        "dump_sha256": result.dump_sha256,
        "guard": result.guard.status,
        "counts": result.counts,
        "graph_counts": result.graph_counts,
        "attention_count": result.attention_count,
        "report": str(report_path) if report_path else None,
        "report_problems": list(problems),
        "warnings": list(result.warnings),
        "errors": list(result.errors),
    }


@migrate_app.command("run")
def run(
    v2_home: Path = V2_HOME_OPTION,
    replace_dirty: bool = typer.Option(
        False, "--replace-dirty",
        help="Replace a v3 library that has changes since the last migration (it is moved to "
             "migration/replaced/, never deleted)."),
    graph_name: List[str] = GRAPH_NAME_OPTION,
    root: Optional[Path] = _common.ROOT_OPTION,
    json_out: bool = _common.JSON_OPTION,
) -> None:
    """Build the v3 library and context graphs from the v2 home, swap them in, write the report."""
    from ..migrate import report as rep

    names = _graph_names(graph_name)
    failure: Optional[str] = None
    with _common.cli_errors():
        try:
            outcome = rep.run_and_report(v2_home, _common.home(root), graph_names=names or None,
                                         replace_dirty=replace_dirty)
            result, files = outcome.result, outcome.files
        except rep.ReportError as exc:
            if exc.result is None:
                _common.fail(str(exc), _common.EXIT_FAILED)
            result, files = exc.result, exc.files
            failure = f"the report does not account for every item: {exc}"
    report_path = files.report_md if files is not None else None
    problems = tuple(files.problems) if files is not None else ()
    summary = _summary(result, report_path, problems)
    if json_out:
        _common.echo_json(summary)
    else:
        typer.echo(f"run {result.run_id}: {'ok' if summary['ok'] else 'FAILED'}; "
                   f"{'swapped into ' + str(result.library_dir) if result.swapped else 'not swapped'}")
        typer.echo(f"dump sha256 {result.dump_sha256}; v2 guard {result.guard.status}")
        for table, n in sorted(result.counts.items()):
            typer.echo(f"  {table}: {n}")
        for slug, counts in sorted(result.graph_counts.items()):
            typer.echo(f"  graph {slug}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        typer.echo(f"attention items: {result.attention_count}")
        typer.echo(f"run folder: {result.run_dir}")
        if files is not None:
            typer.echo(f"report: {files.report_md}")
            typer.echo(f"attention list: {files.attention_md}")
        for w in result.warnings:
            typer.echo(f"warning: {w}")
        for e in result.errors:
            typer.echo(f"error: {e}", err=True)
    if failure is not None:
        _common.fail(failure, _common.EXIT_FAILED)
    if not summary["ok"]:
        raise typer.Exit(_common.EXIT_FAILED)


# --- verify -------------------------------------------------------------------------------------

def _runs_with_dumps(home: Path) -> list[Path]:
    """Run folders holding a canonical dump, oldest first (by when the dump was written)."""
    runs = home / "migration" / "runs"
    if not runs.is_dir():
        return []
    found = [d for d in runs.iterdir() if (d / "dump.jsonl.gz").is_file()]
    return sorted(found, key=lambda d: ((d / "dump.jsonl.gz").stat().st_mtime_ns, d.name))


@migrate_app.command("verify")
def verify(
    run_id: Optional[str] = typer.Option(None, "--run", help="Run id (default: the latest run)."),
    against: Optional[str] = typer.Option(
        None, "--against", help="Run id to compare with (default: the run before --run)."),
    root: Optional[Path] = _common.ROOT_OPTION,
    json_out: bool = _common.JSON_OPTION,
) -> None:
    """Compare two migration runs' canonical dumps: "identical" or the first differing row."""
    from ..migrate.dump import first_difference

    home = _common.home(root)
    runs = _runs_with_dumps(home)
    by_id = {d.name: d for d in runs}
    for rid in (run_id, against):
        if rid is not None and rid not in by_id:
            _common.fail(f"no migration run {rid!r} with a dump under {home / 'migration' / 'runs'}")
    a = by_id[run_id] if run_id else (runs[-1] if runs else None)
    if against:
        b: Optional[Path] = by_id[against]
    else:
        earlier = runs[: runs.index(a)] if a is not None else []
        b = earlier[-1] if earlier else None
    if a is None or b is None:
        _common.fail("verify needs two migration runs with dumps (run `seedgraph migrate run` twice, "
                     "or name them with --run/--against)")
    diff = first_difference(b / "dump.jsonl.gz", a / "dump.jsonl.gz")
    if json_out:
        _common.echo_json({"run": a.name, "against": b.name, "identical": diff is None,
                           "first_difference": diff})
    elif diff is None:
        typer.echo(f"{a.name} vs {b.name}: identical")
    else:
        typer.echo(f"{a.name} vs {b.name}: differ at {diff}")
    if diff is not None:
        raise typer.Exit(_common.EXIT_FAILED)


# --- parity -------------------------------------------------------------------------------------

@migrate_app.command("parity")
def parity(
    seed_queries: bool = typer.Option(
        False, "--seed-queries", help="Write the default query set to migration/parity/queries.json "
                                      "(refuses to overwrite yours)."),
    capture: bool = typer.Option(
        False, "--capture", help="Capture v2's results for the query set (reads the v2 home only)."),
    check: bool = typer.Option(
        False, "--check", help="Compare v3 with the capture; exit 1 on any unexplained difference. "
                               "The default when no other option is given."),
    run_id: Optional[str] = typer.Option(None, "--run", help="Run id (default: the live library's run)."),
    v2_home: Path = V2_HOME_OPTION,
    root: Optional[Path] = _common.ROOT_OPTION,
    json_out: bool = _common.JSON_OPTION,
) -> None:
    """Compare v2 and v3 answers to the same queries; classify every difference."""
    from ..migrate import parity as par

    home = _common.home(root)
    if not (seed_queries or capture or check):
        check = True
    res = None
    with _s7_errors():
        if seed_queries:
            typer.echo(f"queries: {par.seed_queries(v2_home, home)}")
        if capture:
            typer.echo(f"v2 capture: {par.capture(v2_home, home)}")
        if check:
            res = par.check(home, run_id=run_id)
    if res is None:
        return
    if json_out:
        _common.echo_json({"run_id": res.run_id, "ok": res.ok, "differences": len(res.differences),
                           "unexplained": len(res.unexplained), "by_class": res.by_class(),
                           "report": str(res.path) if res.path else None, "notes": res.notes})
    else:
        typer.echo(f"{len(res.differences)} differences, {len(res.unexplained)} unexplained")
        for cls, n in sorted(res.by_class().items()):
            typer.echo(f"  {cls}: {n}")
        for note in res.notes:
            typer.echo(f"note: {note}")
        if res.path:
            typer.echo(f"report: {res.path}")
    if not res.ok:
        raise typer.Exit(_common.EXIT_FAILED)


# --- accept -------------------------------------------------------------------------------------

@migrate_app.command("accept")
def accept(
    v2_home: Path = V2_HOME_OPTION,
    replace_dirty: bool = typer.Option(
        False, "--replace-dirty", help="Allow replacing a v3 library with changes since the last migration."),
    graph_name: List[str] = GRAPH_NAME_OPTION,
    root: Optional[Path] = _common.ROOT_OPTION,
) -> None:
    """The real-data acceptance run: preflight, run, re-run, v2 guard, parity, expected counts."""
    from ..migrate import accept as acc

    names = _graph_names(graph_name)
    with _s7_errors():
        res = acc.accept(v2_home, _common.home(root), graph_names=names or None,
                         replace_dirty=replace_dirty, echo=typer.echo)
    if res.path:
        typer.echo(f"acceptance: {res.path}")
    if not res.ok:
        raise typer.Exit(_common.EXIT_FAILED)
