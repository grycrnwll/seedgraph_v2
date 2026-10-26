"""Typer CLI entry point (``seedgraph-v3`` / ``python -m seedgraph``).

The commands live in :mod:`seedgraph.commands` and are mounted by one
``commands.register(app)`` call (docs/V3_UI_DESIGN.md §1.3). Stage 3 deleted the v2
groups (``corpus``, ``cite``, ``sections``, ``spans``, ``lens``, ``concepts``,
``graph``, ``cache``, ``project``, ``review``, ``doctor``, ``work-read``)."""

from __future__ import annotations

import sqlite3
import sys

import typer

from . import __version__, commands

# Windows consoles default to cp1252; CLI output carries typographic dashes and other
# non-cp1252 glyphs, which made `typer.echo` die with UnicodeEncodeError mid-listing.
# Force UTF-8 at import so every entry path (console script + `python -m`) is safe;
# errors="replace" is the floor if a stream can't be reconfigured.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # captured/replaced streams (pytest) — leave as-is
        pass

app = typer.Typer(
    name="seedgraph",
    help="Seedgraph v3 — a local-first paper library, context graphs and agent work batches.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print package + SQLite versions."""
    typer.echo(f"seedgraph {__version__}")
    typer.echo(f"sqlite {sqlite3.sqlite_version}")


commands.register(app)


if __name__ == "__main__":
    app()
