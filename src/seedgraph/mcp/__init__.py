"""The Seedgraph MCP server (v3 design §7): ``seedgraph mcp serve``.

:mod:`.server` builds the stdio FastMCP server over the library, its context graphs
(:mod:`.tools_library`) and the work protocol (:mod:`.tools_work`). The v2 server
(project.db handles, ``ask``, ``extraction_*``, the content-access gate) was replaced
in place in stage 2.

This ``__init__`` imports nothing eagerly, so ``import seedgraph.mcp`` does not load
the SDK. Import the submodules directly: ``from seedgraph.mcp.server import build_server``.
"""
