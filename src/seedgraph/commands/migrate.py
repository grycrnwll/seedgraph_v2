"""`seedgraph migrate run|verify|parity|accept` (design §6, §8, §11 gate 2).

``run`` is a thin adapter over :func:`seedgraph.migrate.run_migration`. ``verify``,
``parity`` and ``accept`` adapt S7's ``seedgraph.migrate.{report,parity,accept}``; until
those are wired here they exit 4 ("not available yet").
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, List, Optional

import typer

from . import _common

migrate_app = typer.Typer(
    name="migrate",
    help="Migrate a v2 home into the v3 library and its context graphs. The v2 home is only read.",
    no_args_is_help=True,
)

_DEFAULT_V2_HOME = Path.home() / ".seedgraph"
V2_HOME_OPTION = typer.Option(_DEFAULT_V2_HOME, "--v2-home", help="The v2 home to read (never written).")


def _graph_names(pairs: List[str]) -> dict[str, str]:
    names: dict[str, str] = {}
    for pair in pairs:
        slug, sep, name = pair.partition("=")
        if not sep or not slug.strip() or not name.strip():
            raise typer.BadParameter(f"expected SLUG=Display name, got {pair!r}", param_hint="--graph-name")
        names[slug.strip()] = name.strip()
    return names


# --- S7 seams -----------------------------------------------------------------------------------
# Wire these when S7's modules land:
#   _stage_hook()      -> S7's answers export, passed to run_migration(stage_hook=...)
#   _write_report(res) -> S7's report writer (runs/<run_id>/report.{json,md}); returns its path


def _stage_hook() -> Any:
    return None


def _write_report(result: Any) -> Optional[Path]:
    return None


def _s7_module(name: str) -> Any:
    """Import ``seedgraph.migrate.<name>`` or refuse with exit 4."""
    try:
        return importlib.import_module(f"seedgraph.migrate.{name}")
    except ImportError as exc:
        raise _common.NotAvailable(
            f"`migrate {name}` is not available yet (seedgraph.migrate.{name} is not built)") from exc


# --- run ----------------------------------------------------------------------------------------

def _summary(result: Any, report_path: Optional[Path]) -> dict[str, Any]:
    return {
        "ok": result.ok,
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
    graph_name: List[str] = typer.Option(
        [], "--graph-name", help="Display name for a context graph, as SLUG=Name (repeatable)."),
    root: Optional[Path] = _common.ROOT_OPTION,
    json_out: bool = _common.JSON_OPTION,
) -> None:
    """Build the v3 library and context graphs from the v2 home and swap them into the v3 home."""
    from ..migrate import run_migration

    names = _graph_names(graph_name)
    with _common.cli_errors():
        result = run_migration(v2_home, _common.home(root), graph_names=names or None,
                               replace_dirty=replace_dirty, stage_hook=_stage_hook())
        report_path = _write_report(result)
    summary = _summary(result, report_path)
    if json_out:
        _common.echo_json(summary)
    else:
        typer.echo(f"run {result.run_id}: {'ok' if result.ok else 'FAILED'}; "
                   f"{'swapped into ' + str(result.library_dir) if result.swapped else 'not swapped'}")
        typer.echo(f"dump sha256 {result.dump_sha256}; v2 guard {result.guard.status}")
        for table, n in sorted(result.counts.items()):
            typer.echo(f"  {table}: {n}")
        for slug, counts in sorted(result.graph_counts.items()):
            typer.echo(f"  graph {slug}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        typer.echo(f"attention items: {result.attention_count}")
        typer.echo(f"run folder: {result.run_dir}" + (f"; report: {report_path}" if report_path else ""))
        for w in result.warnings:
            typer.echo(f"warning: {w}")
        for e in result.errors:
            typer.echo(f"error: {e}", err=True)
    if not result.ok:
        raise typer.Exit(_common.EXIT_FAILED)


# --- S7 verbs (thin adapters; exit 4 until wired) ------------------------------------------------

@migrate_app.command("verify")
def verify(
    run_id: Optional[str] = typer.Option(None, "--run", help="Run id (default: the latest run)."),
    v2_home: Path = V2_HOME_OPTION,
    root: Optional[Path] = _common.ROOT_OPTION,
    json_out: bool = _common.JSON_OPTION,
) -> None:
    """Check a migration run's report accounts for every v2 item."""
    with _common.cli_errors():
        _s7_module("report")
        raise _common.NotAvailable("`migrate verify` is not available yet (not wired to the CLI)")


@migrate_app.command("parity")
def parity(
    check: bool = typer.Option(False, "--check", help="Exit non-zero on any unexplained difference."),
    v2_home: Path = V2_HOME_OPTION,
    root: Optional[Path] = _common.ROOT_OPTION,
    json_out: bool = _common.JSON_OPTION,
) -> None:
    """Compare v2 and v3 answers to the same queries; classify every difference."""
    with _common.cli_errors():
        _s7_module("parity")
        raise _common.NotAvailable("`migrate parity` is not available yet (not wired to the CLI)")


@migrate_app.command("accept")
def accept(
    v2_home: Path = V2_HOME_OPTION,
    root: Optional[Path] = _common.ROOT_OPTION,
    json_out: bool = _common.JSON_OPTION,
) -> None:
    """The real-data acceptance run: preflight, run, re-run, v2 guard, parity, expected counts."""
    with _common.cli_errors():
        _s7_module("accept")
        raise _common.NotAvailable("`migrate accept` is not available yet (not wired to the CLI)")
