"""Shared bits of the core commands: ``--root``, ``--json``, error -> exit code, rendering.

Exit codes (every core command):

- ``0`` done.
- ``1`` the command ran but reports a failure (a migration that is not ok, a failed
  conversion).
- ``2`` usage error (Click's own code: bad option, missing argument).
- ``3`` refused: the request was understood but not carried out (no library, unknown
  paper or graph, a migration preflight refusal, a stale delete plan).
- ``4`` not available yet: the command exists but the code behind it arrives later.
- ``5`` claim refused (``seedgraph work``): the ``claim_id`` is not, or no longer, the
  package's live claim; the JSON body's ``code`` says why (``claim_reclaimed`` ...).

The ``work`` commands print every refusal as JSON on stdout,
``{"error": {"code", "message", "detail", "next_action"}}``, with the exit code above.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
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
EXIT_CLAIM_REFUSED = 5

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


# --- JSON refusals (the `work` commands) ----------------------------------------------------------

WORK_NEXT_ACTIONS = {
    "library_not_found": "Point --root or SEEDGRAPH_HOME at a v3 home that holds a library "
                         "(run `seedgraph-v3 setup` or `seedgraph-v3 migrate accept` first).",
    "work_refused": "Read the message; run `seedgraph-v3 work status BATCH` for the batch's state.",
    "note_save_failed": "Run `seedgraph-v3 work status BATCH`; create a new batch if the paper changed.",
    "invalid_argument": "Fix the argument named in the message and run the command again.",
    "library_busy": "Another process is writing to the library; wait a few seconds and run it again.",
    "claim_unknown": "Use the claim_id that `work claim` printed for this packet_id.",
    "claim_reclaimed": "Another worker holds this package now; run `work claim` for a new one.",
    "claim_released": "Run `work claim` for a new package.",
    "claim_stopped": "The batch was stopped; stop working and report your counts.",
    "claim_limit": "Run `work claim` again after the usage limit resets.",
    "package_saved": "This package is saved; run `work claim` for the next one.",
    "batch_stopped": "The batch was stopped; stop working and report your counts.",
}


def fail_json(code: str, message: str, exit_code: int = EXIT_REFUSED, **detail: Any) -> None:
    """Print ``{"error": {code, message, detail, next_action}}`` on stdout and exit."""
    echo_json({"error": {"code": code, "message": message, "detail": detail,
                         "next_action": WORK_NEXT_ACTIONS.get(code, "Read the message and try again.")}})
    raise typer.Exit(exit_code)


@contextmanager
def work_errors() -> Iterator[None]:
    """Map the protocol's refusals onto a JSON body and an exit code."""
    from ..notes import NoteSaveError  # local: keep the CLI import light
    from ..work.claims import ClaimRefused
    from ..work.store import WorkError

    try:
        yield
    except typer.Exit:
        raise
    except ClaimRefused as exc:
        fail_json(exc.code, str(exc), EXIT_CLAIM_REFUSED, packet_id=exc.packet_id, claim_id=exc.claim_id)
    except NoteSaveError as exc:
        fail_json("note_save_failed", str(exc))
    except WorkError as exc:
        fail_json("work_refused", str(exc))
    except (SeedgraphError, ValueError) as exc:
        fail_json("invalid_argument", str(exc))
    except sqlite3.OperationalError as exc:
        fail_json("library_busy", str(exc), EXIT_FAILED)


@contextmanager
def work_library(root: Optional[Path]) -> Iterator[Library]:
    """Like :func:`existing_library`, but the refusal is the JSON body."""
    lib_dir = library_dir(root)
    if not library_db_path(lib_dir).is_file():
        fail_json("library_not_found", f"no library at {lib_dir}", home=str(lib_dir.parent))
    lib = open_library(root)
    try:
        yield lib
    finally:
        lib.close()


def force_utf8_stdout() -> None:
    """UTF-8 on stdout/stderr whatever the console code page (so no PYTHONIOENCODING)."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # replaced streams (CliRunner, pytest)
            pass


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
