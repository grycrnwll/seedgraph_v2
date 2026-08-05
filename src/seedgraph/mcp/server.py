"""FastMCP server construction + tool registration (design 00, decisions M2/M4/M9).

Every ``mcp`` SDK import lives here (and the deferred ``_tool_error`` import in
``context.py``); the base package and ``seedgraph.cli`` never import the SDK. The
``seedgraph mcp serve`` CLI command lazy-imports this module inside its body, so a
no-extra install imports and runs the full existing suite.

Import-shadowing note (M4 / design 00 §3.2): this package is ``seedgraph.mcp``
while the SDK is top-level ``mcp``. Python-3 absolute imports make the two lines
below resolve to the *installed SDK*, not this package — a chunk-0 test pins it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, NoReturn

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from .. import __version__, paths
from .context import ServerContext
from .resources import register_prompts, register_resources
from .tools_read import (
    register_ask_tool,
    register_concept_graph_tools,
    register_read_tools,
)


def _tool_error(code: str, message: str, **detail: Any) -> NoReturn:
    """Raise the SDK tool-error type with a JSON ``{code, message, detail}`` body (M9).

    Tool bodies must be able to branch on ``code`` (e.g. ``project_not_found``,
    ``budget_confirmation_required``), so the structured body must reach the client
    verbatim — never a bare traceback string. Raising :class:`ToolError` (the
    *tool*-level error, not the protocol-level ``McpError``) is what FastMCP maps
    to an MCP tool error result. Later chunks reuse this helper.
    """
    raise ToolError(json.dumps({"code": code, "message": message, "detail": detail}))


def build_server(*, root: Path | None = None, redact_private: bool = False) -> FastMCP:
    """Construct the ``seedgraph`` FastMCP server and register the v1 tool surface.

    Home is resolved once here at startup (M6) and stored on the shared
    :class:`ServerContext`, which carries the per-slug handle cache and the
    redaction flag. Chunk 1 registers the read/query tools (which close over
    ``ctx`` lexically and pass their returns through the §5.1 redaction filter);
    ``version`` stays the context-free boot canary.
    """
    ctx = ServerContext(root=paths.resolve_home(root), redact_private=redact_private)
    mcp = FastMCP("seedgraph")

    # Advertise OUR version in the `initialize` handshake. FastMCP's constructor
    # takes no `version` and does not forward one to the lowlevel server, which
    # then falls back to `pkg_version("mcp")` (lowlevel/server.py: `server_version=
    # self.version if self.version else pkg_version("mcp")`) — so the server would
    # otherwise tell every host it is version 1.28.1, the SDK's version, not
    # seedgraph's. `_mcp_server` is the same accessor the in-memory test transport
    # already uses; FastMCP 1.28.1 exposes no public alternative.
    mcp._mcp_server.version = __version__

    @mcp.tool()
    def version() -> dict[str, str]:
        """Package + SQLite versions — a zero-arg boot canary (mirrors CLI ``version``)."""
        return {"seedgraph": __version__, "sqlite": sqlite3.sqlite_version}

    # Chunk-1 read/query tools + chunk-2 concept/search/graph tools + the chunk-3
    # `ask` tool (which owns the §5.3 budget handshake), closing over `ctx`
    # lexically (handle cache + redaction flag). Registered here, not via any
    # stashed attribute.
    register_read_tools(mcp, ctx)
    register_concept_graph_tools(mcp, ctx)
    register_ask_tool(mcp, ctx)

    # Chunk 4: the three §4.2 resources (the `graph.json` export — hard
    # `allow_private=False`, no parameter path to True — plus the two static docs
    # reads) and the single §4.3 `semantic_query` prompt.
    register_resources(mcp, ctx)
    register_prompts(mcp, ctx)

    # `ctx` is also stashed so a caller can introspect server state; the tools
    # themselves close over the same object, they do not read it back off here.
    mcp.seedgraph_context = ctx  # type: ignore[attr-defined]
    return mcp
