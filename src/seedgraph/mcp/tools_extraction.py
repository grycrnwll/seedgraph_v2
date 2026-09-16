"""Thin, explicitly mutating active-session extraction tools."""

from __future__ import annotations

from typing import Any

from ..errors import SeedgraphError
from ..extraction.agent import batch_status, next_packet, prepare_batch, submit_packet


def register_extraction_tools(mcp: Any, ctx: Any) -> None:
    def call(fn, project, *args, **kwargs):
        try:
            return fn(ctx.get_handle(project), *args, **kwargs)
        except (SeedgraphError, ValueError, OSError) as exc:
            from .server import _tool_error

            _tool_error("extraction_failed", str(exc))

    @mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
    def extraction_prepare(
        project: str, work_ids: list[str], input_budget_tokens: int = 12000,
        replace: bool = False, confirm_external: bool = False, client: str | None = None,
    ) -> dict:
        """Explicit mutation: prepare a selected extraction batch for this active session."""
        return call(prepare_batch, project, work_ids, input_budget_tokens=input_budget_tokens,
                    replace=replace, confirm_external=confirm_external, client=client)

    @mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
    def extraction_next(project: str, batch_id: str, retry_failed: bool = False,
                        resume: bool = False) -> dict:
        """Return the pending packet; may checkpoint/reconcile activation. No LLM call."""
        return call(next_packet, project, batch_id, retry_failed=retry_failed, resume=resume,
                    redact_private=ctx.redact_private)

    @mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
    def extraction_submit(
        project: str, batch_id: str, packet_id: str, result: dict | str | None = None,
        failure: str | None = None, model: str | None = None,
    ) -> dict:
        """Explicit mutation: validate submitted extraction or record a reported failure."""
        return call(submit_packet, project, batch_id, packet_id, result=result,
                    failure=failure, model=model)

    @mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False})
    def extraction_status(project: str, batch_id: str, offset: int = 0, limit: int = 50) -> dict:
        """Read bounded progress and pause information for a prepared batch."""
        return call(batch_status, project, batch_id, offset=offset, limit=limit)
