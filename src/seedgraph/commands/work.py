"""`seedgraph work ...`: the CLI mirror of the work protocol (design §10.3).

One command per call, inputs as arguments or file paths, JSON on stdout (UTF-8 forced
inside the program). Each command is a thin adapter over one call of
:mod:`seedgraph.work.routine` (or the Ollama worker loop); refusals are the JSON body
``{"error": {...}}`` with the exit codes of :mod:`._common` (5 = claim refused).
Pauses, ``invalid`` and ``needs_revision`` are normal replies (exit 0).

    work create TASK --paper ID... | --query "..." | --graph G [--replace] [--ttl S] [--split-threshold N]
    work claim BATCH --worker NAME [--client C] [--paper ID]
    work submit PACKET --claim CLAIM --note-file PATH [--accept-unanchored] [--model M] [--usage-file PATH]
    work release PACKET --claim CLAIM --reason give_back|paper_problem [--message T]
    work status BATCH [--offset N] [--limit N]  |  work list [--active]
    work stop BATCH --mode finish|now  |  work continue BATCH
    work limit BATCH [--resets-at T] [--message T] [--worker NAME]
    work ollama BATCH --model M [--max N]  |  work command BATCH
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

import typer

from . import _common

work_app = typer.Typer(
    name="work",
    help="Work a batch from the shell: the protocol's MCP tools, one command per call, JSON on stdout.",
    no_args_is_help=True,
)

ROOT = _common.ROOT_OPTION
DEFAULT_CLI_WORKER = "cli"


class Reason(str, Enum):
    give_back = "give_back"
    paper_problem = "paper_problem"


class Mode(str, Enum):
    finish = "finish"
    now = "now"


@work_app.callback()
def _utf8() -> None:
    _common.force_utf8_stdout()


def _worker(name: str | None, client: str | None = None) -> Any:
    from ..work.claims import WorkerRef

    return WorkerRef("interactive", (name or "").strip() or DEFAULT_CLI_WORKER, client=client)


def _out(obj: Any) -> None:
    _common.echo_json(obj.as_dict() if hasattr(obj, "as_dict") else obj)


@work_app.command("create")
def create(
    task: str = typer.Argument(..., help="notes, metadata or lens:<lens_id>."),
    paper: Optional[List[str]] = typer.Option(None, "--paper", help="A library paper id (repeatable)."),
    query: Optional[str] = typer.Option(None, "--query",
                                        help='A selection such as "held in graph G without current note".'),
    graph: Optional[str] = typer.Option(None, "--graph", help="The context graph (its held papers)."),
    replace: bool = typer.Option(False, "--replace", help="Re-do papers that already have a current result."),
    ttl: Optional[int] = typer.Option(None, "--ttl", min=1, help="Claim time-to-live in seconds."),
    split_threshold: Optional[int] = typer.Option(None, "--split-threshold", min=1,
                                                  help="Split papers above this many tokens."),
    root: Optional[Path] = ROOT,
) -> None:
    """Create a batch and print it (with the copyable work commands)."""
    from ..work import routine

    options: dict[str, Any] = {"replace": replace}
    if ttl is not None:
        options["claim_ttl_s"] = ttl
    if split_threshold is not None:
        options["split_threshold_tokens"] = split_threshold
    with _common.work_errors(), _common.work_library(root) as lib:
        _out(routine.create_batch(lib, task, paper_ids=paper or None, query=query, graph=graph,
                                  options=options, client="cli"))


@work_app.command("claim")
def claim(
    batch: str = typer.Argument(..., help="Batch id."),
    worker: str = typer.Option(..., "--worker", help="Your worker name (for reports)."),
    client: Optional[str] = typer.Option(None, "--client", help="claude, codex, ..."),
    paper: Optional[str] = typer.Option(None, "--paper", help="Claim this paper's next package."),
    root: Optional[Path] = ROOT,
) -> None:
    """Claim the next package; prints packet_id, claim_id and the files to read and write."""
    from ..work import routine

    with _common.work_errors(), _common.work_library(root) as lib:
        _out(routine.claim_next(lib, batch, _worker(worker, client), paper_id=paper))


@work_app.command("submit")
def submit(
    packet: str = typer.Argument(..., help="Packet id."),
    claim_id: str = typer.Option(..., "--claim", help="The claim_id `work claim` printed."),
    note_file: Path = typer.Option(..., "--note-file", help="The result .json file."),
    accept_unanchored: bool = typer.Option(False, "--accept-unanchored",
                                           help="Save unanchored quotes as ambiguous."),
    model: Optional[str] = typer.Option(None, "--model", help="The model that wrote the note."),
    usage_file: Optional[Path] = typer.Option(None, "--usage-file", help="A JSON object of token usage."),
    root: Optional[Path] = ROOT,
) -> None:
    """Submit a package's result: accepted, needs_revision or invalid."""
    from ..work import routine

    with _common.work_errors():
        usage = None
        if usage_file is not None:
            try:
                usage = json.loads(usage_file.read_bytes().decode("utf-8-sig"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"--usage-file: {exc}") from exc
            if not isinstance(usage, dict):
                raise ValueError("--usage-file must hold a JSON object")
        with _common.work_library(root) as lib:
            _out(routine.submit(lib, packet, claim_id, note_path=note_file, accept_unanchored=accept_unanchored,
                                model=model, usage=usage))


@work_app.command("release")
def release(
    packet: str = typer.Argument(..., help="Packet id."),
    claim_id: str = typer.Option(..., "--claim", help="The claim_id `work claim` printed."),
    reason: Reason = typer.Option(..., "--reason", help="give_back or paper_problem."),
    message: Optional[str] = typer.Option(None, "--message", help="Why (for paper_problem)."),
    root: Optional[Path] = ROOT,
) -> None:
    """Hand a package back (give_back) or fail its paper (paper_problem)."""
    from ..work import routine

    with _common.work_errors(), _common.work_library(root) as lib:
        _out(routine.release(lib, packet, claim_id, reason.value, message=message))


@work_app.command("status")
def status(
    batch: str = typer.Argument(..., help="Batch id."),
    offset: int = typer.Option(0, "--offset", min=0, help="First item to list."),
    limit: Optional[int] = typer.Option(None, "--limit", min=0, help="Items to list (default all)."),
    root: Optional[Path] = ROOT,
) -> None:
    """A batch's state, counts, next action and its papers."""
    from ..work import routine

    with _common.work_errors(), _common.work_library(root) as lib:
        _out(routine.batch_view(lib, batch, offset=offset, limit=limit))


@work_app.command("list")
def list_(
    active: bool = typer.Option(False, "--active", help="Drop finished batches."),
    root: Optional[Path] = ROOT,
) -> None:
    """Every batch, oldest first, without its papers."""
    from ..work import routine

    with _common.work_errors(), _common.work_library(root) as lib:
        _out({"batches": [v.as_dict() for v in routine.list_batches(lib, active_only=active)]})


@work_app.command("stop")
def stop(
    batch: str = typer.Argument(..., help="Batch id."),
    mode: Mode = typer.Option(..., "--mode", help="finish (live claims may finish) or now."),
    root: Optional[Path] = ROOT,
) -> None:
    """Stop a batch."""
    from ..work import routine

    with _common.work_errors(), _common.work_library(root) as lib:
        _out(routine.stop(lib, batch, mode=mode.value))


@work_app.command("continue")
def continue_(batch: str = typer.Argument(..., help="Batch id."), root: Optional[Path] = ROOT) -> None:
    """Continue a stopped or paused batch."""
    from ..work import routine

    with _common.work_errors(), _common.work_library(root) as lib:
        _out(routine.continue_batch(lib, batch))


@work_app.command("limit")
def limit(
    batch: str = typer.Argument(..., help="Batch id."),
    resets_at: Optional[str] = typer.Option(None, "--resets-at", help="When the limit resets (UTC ISO)."),
    message: Optional[str] = typer.Option(None, "--message", help="What the client said."),
    worker: Optional[str] = typer.Option(None, "--worker", help="Your worker name."),
    root: Optional[Path] = ROOT,
) -> None:
    """Report a client usage limit: the batch pauses and your claims go back to the pool."""
    from ..work import routine

    with _common.work_errors(), _common.work_library(root) as lib:
        _out(routine.report_limit(lib, batch, _worker(worker), resets_at=resets_at, message=message))


@work_app.command("ollama")
def ollama(
    batch: str = typer.Argument(..., help="Batch id (a metadata batch)."),
    model: Optional[str] = typer.Option(None, "--model",
                                        help="The Ollama model (default: $SEEDGRAPH_OLLAMA_MODEL)."),
    max_packages: Optional[int] = typer.Option(None, "--max", min=1, help="Stop after this many packages."),
    root: Optional[Path] = ROOT,
) -> None:
    """Work a batch with a local Ollama model until it stops; prints the run report.
    The server is $SEEDGRAPH_OLLAMA_URL, else $OLLAMA_HOST, else http://localhost:11434."""
    from ..library import metadata_verify
    from ..llm.ollama import default_model
    from ..work import ollama_worker

    with _common.work_errors():
        model = model or default_model()
        if not model:
            raise ValueError("give --model or set SEEDGRAPH_OLLAMA_MODEL")
        with _common.work_library(root) as lib:
            # The provider chain verifies each title Ollama finds against its DOI (design §7).
            chain = metadata_verify.default_chain(lib)
            try:
                _out(ollama_worker.run_ollama_worker(lib, batch, model=model, max_packages=max_packages,
                                                     chain=chain))
            finally:
                chain.cache_engine.close()


@work_app.command("command")
def command(batch: str = typer.Argument(..., help="Batch id."), root: Optional[Path] = ROOT) -> None:
    """The copyable Claude Code command and Codex prompt for a batch."""
    from ..work import store
    from ..work.commands import work_commands

    with _common.work_errors(), _common.work_library(root) as lib:
        store.require_batch(lib.conn, batch)
        _out(work_commands(batch))
