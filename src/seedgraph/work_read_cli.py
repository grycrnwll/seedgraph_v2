"""Thin CLI for the bounded source reader."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .errors import SeedgraphError
from .project.service import open_project
from .work_read import read_work


def register_work_read(app: typer.Typer) -> None:
    @app.command("work-read")
    def work_read(
        project: str, work_id: str,
        section_id: str | None = typer.Option(None, "--section"),
        subtree: bool = typer.Option(False, "--subtree"),
        span_id: str | None = typer.Option(None, "--span"),
        cursor: str | None = typer.Option(None, "--cursor"),
        max_chars: int = typer.Option(12000, "--max-chars"),
        section_offset: int = typer.Option(0, "--section-offset"),
        section_limit: int = typer.Option(50, "--section-limit"),
        external: bool = typer.Option(False, "--external"),
        confirm_external: bool = typer.Option(False, "--confirm-external"),
        redact_private: bool = typer.Option(False, "--redact-private"),
        json_output: bool = typer.Option(False, "--json"),
        root: Path | None = typer.Option(None, "--root"),
    ) -> None:
        """Read bounded source text; --span reads that anchor's full source version."""
        try:
            handle = open_project(project, root=root, read_only=True)
            result = read_work(handle, work_id=work_id, section_id=section_id, subtree=subtree,
                               span_id=span_id, cursor=cursor, max_chars=max_chars,
                               section_offset=section_offset, section_limit=section_limit,
                               external=external, confirm_external=confirm_external,
                               redact_private=redact_private)
        except SeedgraphError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        # Even plain output keeps continuation and provenance inspectable.
        typer.echo(json.dumps(result, ensure_ascii=False, indent=None if json_output else 2))
