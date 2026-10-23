"""Shared bits of the core commands: ``--root``, ``--json``, error -> exit code, rendering.

Exit codes (every core command):

- ``0`` done.
- ``1`` the command ran but reports a failure (a migration that is not ok, a failed
  conversion).
- ``2`` usage error (Click's own code: bad option, missing argument).
- ``3`` refused: the request was understood but not carried out (no library, unknown
  paper or graph, a migration preflight refusal, a stale delete plan).
- ``4`` not available yet: the command exists but the code behind it arrives later.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import typer

from .. import paths
from ..errors import SeedgraphError
from ..library.store import Library, library_db_path, library_dir, open_library

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3
EXIT_NOT_AVAILABLE = 4

ROOT_OPTION = typer.Option(
    None, "--root", help="The v3 home (default: $SEEDGRAPH_HOME, else ~/.seedgraph-v3).")
JSON_OPTION = typer.Option(False, "--json", help="Print JSON instead of text.")


class NotAvailable(Exception):
    """The command exists, but what it calls is not built yet."""


def home(root: Optional[Path]) -> Path:
    return paths.resolve_home(root)


def fail(message: str, code: int = EXIT_REFUSED) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)


@contextmanager
def cli_errors() -> Iterator[None]:
    """Map the deep modules' refusals onto exit codes, with a one-line message."""
    from ..migrate.files import MigrationRefused  # local: keep the CLI import light

    try:
        yield
    except NotAvailable as exc:
        fail(str(exc), EXIT_NOT_AVAILABLE)
    except (SeedgraphError, MigrationRefused) as exc:
        fail(str(exc), EXIT_REFUSED)


@contextmanager
def existing_library(root: Optional[Path]) -> Iterator[Library]:
    """Open the library of the home, refusing (rather than creating) when there is none."""
    lib_dir = library_dir(root)
    if not library_db_path(lib_dir).is_file():
        fail(f"no library at {lib_dir}; run `seedgraph setup` or `seedgraph migrate run` first")
    lib = open_library(root)
    try:
        yield lib
    finally:
        lib.close()


# --- rendering ----------------------------------------------------------------------------------

def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)
                if f.repr}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def echo_json(obj: Any) -> None:
    typer.echo(json.dumps(to_jsonable(obj), indent=2, sort_keys=True, ensure_ascii=False))


def echo_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """A tab-separated table (header line, then one line per row)."""
    typer.echo("\t".join(headers))
    for row in rows:
        typer.echo("\t".join("" if v is None else str(v) for v in row))


def one_line(text: str | None, width: int = 80) -> str:
    flat = (text or "").replace("\r", " ").replace("\n", " ")
    return flat if len(flat) <= width else flat[: width - 3] + "..."
