"""Work protocol tools for the v3 MCP server (design §7.2): thin adapters over
:mod:`seedgraph.work.routine`. Replies are the routine's own ``as_dict()``.

Worker identity: this server process mints one worker id (``mcp-<hex>``); ``work_claim``
and ``batch_report_limit`` take an optional ``worker`` name so subagents sharing one
server process report separately. Correctness rests on ``claim_id``, never on the name.
The client (``claude``, ``codex``) comes from the MCP handshake.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import Context

from .server_v3 import MUTATION, READ, client_name, tool_error

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from .server_v3 import V3Context

MAX_ITEMS = 500


def register_work_tools(server: FastMCP, ctx: V3Context) -> None:
    """Register the protocol tools on ``server``."""

    def _worker(name: str | None, mcp_ctx: Context | None) -> Any:
        from ..work.claims import WorkerRef

        return WorkerRef("interactive", (name or "").strip() or ctx.worker_id, client=client_name(mcp_ctx))

    @server.tool(annotations=MUTATION)
    def batch_create(task: str, paper_ids: list[str] | None = None, query: str | None = None,
                     graph: str | None = None, options: dict[str, Any] | None = None,
                     mcp_ctx: Context | None = None) -> dict[str, Any]:
        """Create a work batch. `task` is notes, metadata or lens:<lens_id>. Choose papers by
        `paper_ids`, or a `query` such as "held in graph <slug> without current note", or a
        `graph` (its held papers). `options`: claim_ttl_s, replace, split_threshold_tokens,
        part_target_tokens. Held-back papers are listed with their reason. Only create a
        batch when the researcher asks for one."""
        from ..work import routine

        with ctx.library() as lib:
            view = routine.create_batch(lib, task, paper_ids=paper_ids, query=query, graph=graph,
                                        options=options, client=client_name(mcp_ctx))
            return view.as_dict()

    @server.tool(annotations=READ)
    def batch_status(batch_id: str, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """A batch's state (working, waiting, paused, stopped, finished), counts, next action
        and a page of its papers (`offset`, `limit`)."""
        from ..work import routine

        if offset < 0 or not 0 <= limit <= MAX_ITEMS:
            tool_error("invalid_argument", f"need offset >= 0 and 0 <= limit <= {MAX_ITEMS}")
        with ctx.library() as lib:
            return routine.batch_view(lib, batch_id, offset=offset, limit=limit).as_dict()

    @server.tool(annotations=READ)
    def batch_list(active_only: bool = False) -> dict[str, Any]:
        """Every batch, oldest first, without its papers; `active_only` drops finished ones."""
        from ..work import routine

        with ctx.library() as lib:
            return {"batches": [v.as_dict() for v in routine.list_batches(lib, active_only=active_only)]}

    @server.tool(annotations=MUTATION)
    def work_claim(batch_id: str, paper_id: str | None = None, worker: str | None = None,
                   mcp_ctx: Context | None = None) -> dict[str, Any]:
        """Claim the next package of a batch. The reply's `package` names the text file,
        instructions.md, labels.json and result_path, plus packet_id and claim_id for
        work_submit. A paused, finished or empty batch is a normal reply with next_action.
        `worker` is an optional name for reports when several agents share this server."""
        from ..work import routine

        with ctx.library() as lib:
            return routine.claim_next(lib, batch_id, _worker(worker, mcp_ctx), paper_id=paper_id).as_dict()

    @server.tool(annotations=MUTATION)
    def work_submit(packet_id: str, claim_id: str, note: dict[str, Any] | str | None = None,
                    note_path: str | None = None, accept_unanchored: bool = False, model: str | None = None,
                    usage: dict[str, Any] | None = None) -> dict[str, Any]:
        """Submit the result for a claimed package: `note_path` (the package's result_path, a
        .json file) or the `note` inline. Replies accepted, needs_revision (copy each failed
        quote's markdown_slice and resubmit) or invalid (fix the errors and resubmit).
        `accept_unanchored` saves unanchored quotes as ambiguous."""
        from ..work import routine

        with ctx.library() as lib:
            return routine.submit(lib, packet_id, claim_id, note=note, note_path=note_path,
                                  accept_unanchored=accept_unanchored, model=model, usage=usage).as_dict()

    @server.tool(annotations=MUTATION)
    def work_release(packet_id: str, claim_id: str, reason: str, message: str | None = None) -> dict[str, Any]:
        """Hand a package back: `reason` give_back (back to the pool) or paper_problem (the
        paper cannot be worked; say why in `message`)."""
        from ..work import routine

        with ctx.library() as lib:
            return routine.release(lib, packet_id, claim_id, reason, message=message).as_dict()

    @server.tool(annotations=MUTATION)
    def batch_stop(batch_id: str, mode: str = "finish") -> dict[str, Any]:
        """Stop a batch: `mode` finish (live claims may finish) or now (live claims are
        released and their submits refused)."""
        from ..work import routine

        if mode not in ("finish", "now"):
            tool_error("invalid_argument", f"mode must be finish or now, not {mode!r}")
        with ctx.library() as lib:
            return routine.stop(lib, batch_id, mode=mode).as_dict()

    @server.tool(annotations=MUTATION)
    def batch_continue(batch_id: str) -> dict[str, Any]:
        """Continue a stopped or paused batch (the only resume)."""
        from ..work import routine

        with ctx.library() as lib:
            return routine.continue_batch(lib, batch_id).as_dict()

    @server.tool(annotations=MUTATION)
    def batch_report_limit(batch_id: str, resets_at: str | None = None, message: str | None = None,
                           worker: str | None = None, mcp_ctx: Context | None = None) -> dict[str, Any]:
        """Report that your client hit a usage limit: the batch pauses and your claims go back
        to the pool. `resets_at` is the reset time (UTC ISO) if the client said it."""
        from ..work import routine

        with ctx.library() as lib:
            return routine.report_limit(lib, batch_id, _worker(worker, mcp_ctx), resets_at=resets_at,
                                        message=message).as_dict()
