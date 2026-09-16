"""CLI for explicit active-session extraction; no model dispatch lives here."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ..errors import SeedgraphError
from ..project.service import open_project
from .agent import batch_status, next_packet, prepare_batch, submit_packet

agent_extract_app = typer.Typer(help="Prepare, resume and import active-session paper extraction.")


def _call(fn, project, root, *args, **kwargs):
    try:
        result = fn(open_project(project, root=root), *args, **kwargs)
    except (SeedgraphError, ValueError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, ensure_ascii=False))


@agent_extract_app.command("prepare")
def prepare(
    project: str,
    work: list[str] = typer.Option(..., "--work", help="Repeat for an ordered selected batch."),
    input_budget_tokens: int = typer.Option(12000, "--input-budget-tokens"),
    replace: bool = typer.Option(False, "--replace"),
    confirm_external: bool = typer.Option(False, "--confirm-external"),
    client: Optional[str] = typer.Option(None, "--client"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Freeze selected papers and coverage; returns a batch ID, without calling an LLM."""
    _call(prepare_batch, project, root, work, input_budget_tokens=input_budget_tokens,
          replace=replace, confirm_external=confirm_external, client=client)


@agent_extract_app.command("next")
def next_(
    project: str, batch: str,
    retry_failed: bool = typer.Option(False, "--retry-failed"),
    resume: bool = typer.Option(False, "--resume"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Return the same unfinished packet until submission; resume pauses explicitly."""
    _call(next_packet, project, root, batch, retry_failed=retry_failed, resume=resume)


@agent_extract_app.command("submit")
def submit(
    project: str, batch: str,
    packet: str = typer.Option(..., "--packet"),
    result: Optional[Path] = typer.Option(None, "--result"),
    failure: Optional[str] = typer.Option(None, "--failure"),
    model: Optional[str] = typer.Option(None, "--model"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Submit result JSON or a reported failure; never writes model output directly to DB."""
    if (result is None) == (failure is None):
        raise typer.BadParameter("provide exactly one of --result or --failure")
    try:
        payload = result.read_text(encoding="utf-8") if result else None
    except OSError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    _call(submit_packet, project, root, batch, packet, result=payload, failure=failure, model=model)


@agent_extract_app.command("status")
def status(
    project: str, batch: str,
    offset: int = typer.Option(0, "--offset"),
    limit: int = typer.Option(50, "--limit"),
    root: Optional[Path] = typer.Option(None, "--root"),
) -> None:
    """Read bounded checkpoint progress."""
    _call(batch_status, project, root, batch, offset=offset, limit=limit)
