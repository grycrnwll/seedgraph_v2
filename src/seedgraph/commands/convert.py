"""`seedgraph convert [PDF...] [--queued] [--paper ID]`: a thin adapter over
:func:`seedgraph.library.files.convert_files` (the stage 0 warm converter pool).
It refuses while ``seedgraph serve`` holds the converter lock (UI design §6)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from ..library import files
from . import _common


def convert(
    pdfs: List[Path] = typer.Argument(
        None, exists=True, dir_okay=False, help="PDF files to add to the library (unattached) and convert."),
    queued: bool = typer.Option(False, "--queued", help="Convert every file waiting in the conversion queue."),
    paper: Optional[str] = typer.Option(None, "--paper", help="Convert this library paper's PDF(s)."),
    force: bool = typer.Option(False, "--force", help="Convert again even if a conversion exists."),
    root: Optional[Path] = _common.ROOT_OPTION,
    json_out: bool = _common.JSON_OPTION,
) -> None:
    """Convert PDFs to markdown in the library. A failed file does not stop the batch (exit 1).
    Refused while `seedgraph serve` runs: it owns the converters (add PDFs in the UI)."""
    from ..web.jobs import check_converter_free  # the lock API; no web framework import

    with _common.cli_errors():
        check_converter_free(_common.home(root))
    with _common.cli_errors(), _common.existing_library(root) as lib:
        outcomes = files.convert_files(lib, pdfs=tuple(pdfs or ()), queued=queued, paper_id=paper,
                                       force=force)
    if json_out:
        _common.echo_json(outcomes)
    elif not outcomes:
        typer.echo("nothing to convert")
    else:
        _common.echo_table(
            ["source_file_id", "status", "markdown_id", "papers", "error"],
            [[o.source_file_id, o.status, o.markdown_id, ",".join(o.paper_ids), _common.one_line(o.error)]
             for o in outcomes])
    if any(o.status == "failed" for o in outcomes):
        raise typer.Exit(_common.EXIT_FAILED)
