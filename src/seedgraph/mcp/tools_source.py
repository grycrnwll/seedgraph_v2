"""SDK-free registration of the bounded source reader."""

from __future__ import annotations

from typing import Any

from ..errors import SeedgraphError
from ..work_read import read_work


def register_source_tools(mcp: Any, ctx: Any) -> None:
    @mcp.tool()
    def work_read(
        project: str, work_id: str, section_id: str | None = None, subtree: bool = False,
        span_id: str | None = None, cursor: str | None = None, max_chars: int = 12000,
        section_offset: int = 0, section_limit: int = 50, confirm_external: bool = False,
    ) -> dict:
        """Read bounded work/section text or an anchored span's full original source.

        External content policy applies. Repeat the selection with continuation
        as cursor; offsets are Unicode codepoints and completeness is selection-wide.
        """
        handle = ctx.get_handle(project, read_only=True)
        try:
            return read_work(handle, work_id=work_id, section_id=section_id, subtree=subtree,
                             span_id=span_id, cursor=cursor, max_chars=max_chars,
                             section_offset=section_offset, section_limit=section_limit,
                             redact_private=ctx.redact_private, external=True,
                             confirm_external=confirm_external)
        except SeedgraphError as exc:
            from .server import _tool_error

            _tool_error(getattr(exc, "code", "work_read_failed"), str(exc))
