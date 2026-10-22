"""``seedgraph-v3 golden`` - golden-set manifest and converter-upgrade check."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from seedgraph.paths import resolve_home

from . import manifest, report, sources, v2
from .errors import GoldenError
from .run import run_check

golden_app = typer.Typer(
    name="golden",
    help="Golden-set manifest and converter-upgrade check (stage 0).",
    no_args_is_help=True,
)

_root_option = typer.Option(None, "--root", help="Override the v3 home directory.")


def _die(msg: str) -> None:
    typer.echo(f"error: {msg}", err=True)
    raise typer.Exit(2)


@golden_app.command("init")
def golden_init(
    seed: Path = typer.Argument(..., help="Seed JSON (machine-local; see seedgraph.golden.manifest)."),
    root: Optional[Path] = _root_option,
    v2_home: Optional[Path] = typer.Option(None, "--v2-home", help="Default v2 home (read-only)."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing manifest."),
) -> None:
    """Resolve the seed against v2's cache (read-only) and write <home>/golden_set.json."""
    home = resolve_home(root)
    try:
        if manifest.manifest_path(home).exists() and not force:
            _die(f"{manifest.manifest_path(home)} exists; pass --force to overwrite")
        entries = manifest.build_manifest(manifest.load_seed(seed), default_v2_home=v2_home or v2.DEFAULT_V2_HOME)
        path = manifest.write_manifest(entries, home)
    except GoldenError as exc:
        _die(str(exc))
    typer.echo(f"wrote {len(entries)} golden entries to {path}")


@golden_app.command("check")
def golden_check(
    old: str = typer.Option("v2-stored", "--old", help="OLD source: v2-stored | convert | <dir>."),
    new: str = typer.Option(..., "--new", help="NEW source: v2-stored | convert | <dir>."),
    ids: Optional[str] = typer.Option(None, "--ids", help="Comma-separated golden IDs (default: all)."),
    root: Optional[Path] = _root_option,
) -> None:
    """Compare OLD vs NEW markdown per golden PDF; exit 1 if any check FAILs."""
    home = resolve_home(root)
    try:
        entries = manifest.load_manifest(home)
        if ids:
            wanted = [i.strip() for i in ids.split(",") if i.strip()]
            known = {e["id"] for e in entries}
            unknown = [i for i in wanted if i not in known]
            if unknown:
                _die(f"unknown golden id(s): {', '.join(unknown)}")
            entries = [e for e in entries if e["id"] in wanted]
        manifest.guard_v3_home(home, [e["quote_source"]["v2_home"] for e in entries])
        old_src, new_src = sources.make_source(old), sources.make_source(new)
        results = run_check(entries, old_src, new_src)
    except GoldenError as exc:
        _die(str(exc))
    typer.echo(report.render_table(results))
    out = report.write_report(results, home, meta={"old": old_src.label, "new": new_src.label,
                                                   "ids": [e["id"] for e in entries]})
    typer.echo(f"\nreport: {out}")
    failed = [r.golden_id for r in results if r.status == "fail"]
    if failed:
        typer.echo(f"FAIL: {', '.join(failed)}")
        raise typer.Exit(1)
    typer.echo("all golden checks passed (warnings, if any, are listed in the report)")
