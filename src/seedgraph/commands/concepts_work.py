"""`seedgraph concepts estimate | check`: the concept tasks' dry run and the guardrail audit
(stage 4 design §4.5, §11.4; slice C5). Both open the library **read-only** (a ``mode=ro``
connection: no migrations, no refresh, no writes), so they are safe on any home.

    concepts estimate merge|bridge|name_literatures [--family F]... [--k N] [--budget-tokens N]
                      [--sense-checks all|ambiguous|none] [--graph G] [--json]
    concepts check [--json]          exit 1 when a guardrail is violated (counts only)

Mounted on C4's ``concepts_app`` by :func:`register` (one line in ``commands/__init__.py``).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer

from . import _common

ROOT = _common.ROOT_OPTION
JSON = _common.JSON_OPTION


class Task(str, Enum):
    merge = "merge"
    bridge = "bridge"
    name_literatures = "name_literatures"


class SenseChecks(str, Enum):
    all = "all"
    ambiguous = "ambiguous"
    none = "none"


@contextmanager
def readonly_library(root: Optional[Path]) -> Iterator:
    """A :class:`~seedgraph.library.store.Library` on a ``mode=ro`` connection."""
    from ..library.store import Library, library_db_path, library_dir

    lib_dir = library_dir(root)
    db = library_db_path(lib_dir)
    if not db.is_file():
        _common.fail(f"no library at {lib_dir}")
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield Library(root=lib_dir, conn=conn)
    finally:
        conn.close()


def estimate_cmd(
    task: Task = typer.Argument(..., help="merge (tidy labels), bridge (find bridges) or name_literatures."),
    family: Optional[List[str]] = typer.Option(None, "--family", help="A family to include (repeatable, or "
                                               "comma-separated). Default: every family (merge) or "
                                               "assumption, estimand, method (bridge)."),
    k: Optional[int] = typer.Option(None, "--k", min=1, help="Neighbours per sense (bridge; default 5)."),
    budget_tokens: Optional[int] = typer.Option(None, "--budget-tokens", min=1,
                                                help="Stop packaging when the high agent-token estimate "
                                                     "would pass this."),
    sense_checks: Optional[SenseChecks] = typer.Option(None, "--sense-checks", help="Which sense checks (bridge)."),
    graph: Optional[str] = typer.Option(None, "--graph", help="Graph scope (name_literatures needs one)."),
    root: Optional[Path] = ROOT,
    as_json: bool = JSON,
) -> None:
    """The dry run of a concept task: pairs, sense checks, packages and the token range. Writes nothing."""
    from ..concepts import estimate as cest

    options: dict = {}
    if family:
        options["families"] = ",".join(family)
    if k is not None:
        options["k"] = k
    if budget_tokens is not None:
        options["budget_tokens"] = budget_tokens
    if sense_checks is not None:
        options["sense_checks"] = sense_checks.value
    if graph:
        options["graph"] = graph
    with _common.cli_errors(), readonly_library(root) as lib:
        try:
            est = cest.estimate(lib.conn, task.value, options, library_root=lib.root)
        except (cest.OptionError, ValueError) as exc:
            _common.fail(str(exc))
    body = est.as_dict()
    if as_json:
        _common.echo_json(body)
        return
    for key in ("task", "families", "candidate_source", "pairs", "sense_checks", "groups", "literatures", "packages",
                "text_tokens", "agent_tokens_low", "agent_tokens_high"):
        typer.echo(f"{key}: {body[key]}")
    if "cut" in body:
        typer.echo(f"budget: {body['cut']}")
    typer.echo(f"citation distance share: {body['distance_share']}")


def check_cmd(root: Optional[Path] = ROOT, as_json: bool = JSON) -> None:
    """The concept guardrail checks (counts of violations only). Exit 1 when any fails."""
    from ..concepts import decide

    with _common.cli_errors(), readonly_library(root) as lib:
        res = decide.audit(lib.conn)
    if as_json:
        _common.echo_json(res.as_dict())
    else:
        for name, n in res.checks.items():
            typer.echo(f"{'ok  ' if not n else 'FAIL'} {name}: {n}")
    if not res.ok:
        raise typer.Exit(_common.EXIT_FAILED)


def register(app: typer.Typer) -> None:
    """Add ``estimate`` and ``check`` to the ``concepts`` command group."""
    app.command("estimate")(estimate_cmd)
    app.command("check")(check_cmd)
