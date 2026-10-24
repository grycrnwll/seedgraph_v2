"""The v3 MCP server (design §7): a stdio FastMCP server named ``seedgraph`` over the
library, its context graphs and the agent work protocol.

- Library and graph reads (:mod:`.tools_library`): ``graph_list``, ``graph_show``,
  ``paper_list``, ``paper_show``, ``search``, ``paper_read``, ``lens_results``. Search
  and every note read use the **agent** audience: notes hidden by an open attention
  item stay hidden, and there is no parameter that widens it.
- The protocol (:mod:`.tools_work`): ``batch_create``, ``batch_status``,
  ``batch_list``, ``work_claim``, ``work_submit``, ``work_release``, ``batch_stop``,
  ``batch_continue``, ``batch_report_limit``; thin adapters over
  :mod:`seedgraph.work.routine`.

Every result is a JSON object. Errors are tool errors whose text is the JSON
``{code, message, detail, next_action}`` (FastMCP prefixes it with
``Error executing tool <name>: ``); a pause is a normal reply, never an error.

The server never creates a library: each call opens ``<home>/library`` and refuses with
``library_not_found`` if it is missing. Building the server touches no disk. Imports of
the library and protocol modules are deferred to the tool bodies, so a cold start stays
fast. ``seedgraph mcp serve`` (:mod:`seedgraph.commands.mcp`) runs it; so does
``python -m seedgraph.mcp.server``.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from .. import __version__, paths

if TYPE_CHECKING:
    from ..library.store import Library

SERVER_NAME = "seedgraph"
READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
MUTATION = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

NEXT_ACTIONS = {
    "library_not_found": "Point SEEDGRAPH_HOME at a v3 home that holds a library (run `seedgraph-v3 setup` or "
                         "`seedgraph-v3 migrate accept` first).",
    "graph_not_found": "Call graph_list for the graph slugs in this library.",
    "paper_not_found": "Call paper_list or search for paper ids.",
    "no_text": "This paper has no converted text; pick another paper.",
    "invalid_argument": "Fix the argument named in the message and call again.",
    "search_refused": "Fix the search arguments named in the message and call again.",
    "work_refused": "Read the message; call batch_status for the batch's state.",
    "note_save_failed": "Call batch_status; create a new batch if the paper changed.",
    "claim_unknown": "Use the claim_id that work_claim returned for this packet_id.",
    "claim_reclaimed": "Another worker holds this package now; call work_claim for a new one.",
    "claim_released": "Call work_claim for a new package.",
    "claim_stopped": "The batch was stopped; stop working and report your counts.",
    "claim_limit": "Call work_claim again after the usage limit resets.",
    "package_saved": "This package is saved; call work_claim for the next one.",
    "batch_stopped": "The batch was stopped; stop working and report your counts.",
    "library_busy": "Another process is writing to the library; wait a few seconds and call again.",
    "internal_error": "This is a Seedgraph bug; report the message to the researcher and stop.",
}


def tool_error(code: str, message: str, *, next_action: str | None = None, **detail: Any) -> NoReturn:
    """Raise a tool error with the JSON ``{code, message, detail, next_action}`` body."""
    body = {"code": code, "message": message, "detail": detail,
            "next_action": next_action or NEXT_ACTIONS.get(code, "Read the message and try again.")}
    raise ToolError(json.dumps(body, ensure_ascii=False))


@contextmanager
def refusals() -> Iterator[None]:
    """Map library and protocol refusals to structured tool errors."""
    from ..errors import SeedgraphError, ValidationError
    from ..library.graphs import GraphError
    from ..library.search import SearchError
    from ..notes import NoteSaveError
    from ..work.claims import ClaimRefused
    from ..work.store import WorkError

    try:
        yield
    except ToolError:
        raise
    except ClaimRefused as exc:
        tool_error(exc.code, str(exc), packet_id=exc.packet_id, claim_id=exc.claim_id)
    except WorkError as exc:
        tool_error("work_refused", str(exc))
    except SearchError as exc:
        code = "graph_not_found" if "no graph" in str(exc) else "search_refused"
        tool_error(code, str(exc))
    except GraphError as exc:
        tool_error("graph_not_found", str(exc))
    except NoteSaveError as exc:
        tool_error("note_save_failed", str(exc))
    except ValidationError as exc:
        tool_error("invalid_argument", str(exc))
    except SeedgraphError as exc:
        tool_error("invalid_argument", str(exc))
    except sqlite3.OperationalError as exc:
        # e.g. "database is locked" after the 5 s busy timeout, with another writer active
        tool_error("library_busy", str(exc))
    except Exception as exc:  # noqa: BLE001 - every failure reaches the client structured
        tool_error("internal_error", f"{type(exc).__name__}: {exc}")


@dataclass
class V3Context:
    """The home resolved once at startup and the worker id minted for this process."""

    home: Path
    worker_id: str = field(default_factory=lambda: f"mcp-{uuid.uuid4().hex[:8]}")

    @contextmanager
    def library(self) -> Iterator[Library]:
        """A fresh library handle per call (sqlite connections are thread-affine). Never
        creates one: a missing ``library.db`` is the ``library_not_found`` refusal."""
        from ..library import store as libstore

        lib_dir = libstore.library_dir(self.home)
        if not libstore.library_db_path(lib_dir).is_file():
            tool_error("library_not_found", f"no library at {lib_dir}", home=str(self.home))
        with refusals():
            lib = libstore.open_library_at(lib_dir)
            try:
                yield lib
            finally:
                lib.close()


def client_name(mcp_ctx: Any) -> str | None:
    """The connecting client (``claude``, ``codex`` or its own name), from the MCP
    ``initialize`` handshake; ``None`` when unknown."""
    try:
        name = mcp_ctx.session.client_params.clientInfo.name
    except Exception:  # noqa: BLE001 - no session or no handshake data
        return None
    if not name:
        return None
    low = str(name).lower()
    for known in ("claude", "codex"):
        if known in low:
            return known
    return str(name)


def build_server(*, home: Path | str | None = None) -> FastMCP:
    """Construct the v3 ``seedgraph`` server. ``home`` defaults to ``$SEEDGRAPH_HOME``
    (resolved once, here); nothing is read or written until a tool is called."""
    from .tools_library import register_library_tools
    from .tools_work import register_work_tools

    ctx = V3Context(home=paths.resolve_home(home))
    server = FastMCP(SERVER_NAME)
    # FastMCP does not forward a version to the low-level server, which would otherwise
    # advertise the SDK's version in `initialize` (same fix as the v2 server).
    server._mcp_server.version = __version__

    @server.tool(annotations=READ)
    def version() -> dict[str, Any]:
        """Seedgraph and SQLite versions and the home this server reads (touches no data)."""
        return {"seedgraph": __version__, "sqlite": sqlite3.sqlite_version, "home": str(ctx.home),
                "server": "v3"}

    register_library_tools(server, ctx)
    register_work_tools(server, ctx)
    server.seedgraph_context = ctx  # type: ignore[attr-defined]
    return server


def main(home: Path | str | None = None) -> None:
    """Run the stdio server (``seedgraph mcp serve``). The SDK's stdio transport reads and
    writes UTF-8; stderr is reconfigured too so log lines with non-ASCII text never fail
    on a Windows console."""
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
    build_server(home=home).run("stdio")


if __name__ == "__main__":
    main()
