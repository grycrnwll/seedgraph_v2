"""The record-answer tool for the v3 MCP server (stage 5 design §3.4, §7): a thin adapter over
:func:`seedgraph.records.answer.record_answer`.

``record_answer`` is a mutation that writes only records: it answers the session's open
record (everything searched and read since the previous recorded answer in the same
session), checks each citation against the session family's log, and freezes the cited
evidence. The reply is ``{record_id, session, counts, flags, page, next_action}``;
``next_action`` is set when a citation is flagged. Refusals keep their own code and next
action: ``invalid_session``, ``invalid_argument`` (malformed input) and
``unknown_evidence`` (ids not in the library, listed in ``detail.unknown``). A refused
call writes nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import Context

from .server import MUTATION, call_info, tool_error

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from .server import V3Context


def register_record_tools(server: FastMCP, ctx: V3Context) -> None:
    """Register ``record_answer`` on ``server``."""

    @server.tool(annotations=MUTATION)
    def record_answer(question: str, answer: str, evidence: list[dict[str, Any]], judgement: str | None = None,
                      session: str | None = None, mcp_ctx: Context | None = None) -> dict[str, Any]:
        """Record one answer you gave the researcher from the library: call it once per
        answer, at the end. `question` is the researcher's question as you understood it,
        `answer` what the papers say, `judgement` your own view (kept apart). `evidence` is
        1 to 100 items, each {"span_id": ...}, {"claim_id": ...} or {"paper_id": ...,
        "page": n}, with an optional "why". Pass the same `session` as on your searches.
        Each citation is checked against what you opened in this session: a span you saw
        only in search results is flagged; read it with paper_read first. The reply gives
        record_id, counts, flags, the record's page and, when something is flagged, a
        next_action."""
        from ..records import answer as answer_mod

        call = call_info(ctx, mcp_ctx, session)
        with ctx.library() as lib:
            try:
                result = answer_mod.record_answer(lib, call, question, answer, evidence, judgement)
            except answer_mod.UnknownEvidence as exc:
                tool_error(exc.code, str(exc), next_action=exc.next_action, unknown=list(exc.unknown))
            except answer_mod.RecordAnswerError as exc:
                tool_error(exc.code, str(exc), next_action=exc.next_action)
            return result.to_reply()
