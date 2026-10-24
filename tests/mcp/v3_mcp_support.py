"""Helpers and fixtures for the v3 MCP server tests (invented text only).

No ``__init__.py`` and no ``conftest.py`` here on purpose: a ``tests/mcp`` package would
import as ``mcp`` and shadow the SDK, and a second rootless ``conftest`` module would
shadow ``tests/conftest.py`` in ``sys.modules``. Test files import the fixtures below by
name and carry unique ``test_v3_server_*`` basenames.

Every test talks to the server through the SDK's in-memory client transport; nothing is
spawned (the stdio smoke is behind the opt-in ``mcp_client`` marker).
"""

from __future__ import annotations

import json

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from work.routine_support import good_note, make_paper

from seedgraph.library import graphs
from seedgraph.library import store as libstore
from seedgraph.work import routine
from seedgraph.work.claims import WorkerRef

GRAPH = "invented_gadgets"


def call(server, name: str, args: dict | None = None):
    """Call one tool over the in-memory transport; return the CallToolResult."""

    async def _driver():
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            return await client.call_tool(name, args or {})

    return anyio.run(_driver)


def list_tools(server):
    async def _driver():
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            return (await client.list_tools()).tools

    return anyio.run(_driver)


def ok(result) -> dict:
    """The JSON object a successful tool call returned."""
    assert result.isError is False, result.content[0].text
    return json.loads(result.content[0].text)


def err(result) -> dict:
    """The ``{code, message, detail, next_action}`` body of a tool error. FastMCP prefixes
    the JSON with ``Error executing tool <name>: ``."""
    assert result.isError is True, result.content[0].text
    text = result.content[0].text
    return json.loads(text[text.index("{"):])


class Seeded:
    """A library with three held papers in graph ``invented_gadgets``:
    p1 (seed, noted), p2 (member, no note), p3 (frontier, no note) and p4 outside any graph."""

    def __init__(self, home):
        lib = libstore.open_library_at(home / "library")
        try:
            self.p1 = make_paper(lib, 1, title="Invented gadget subsidies and μ")
            self.p2 = make_paper(lib, 2)
            self.p3 = make_paper(lib, 3)
            self.p4 = make_paper(lib, 4)
            with graphs.create_graph(lib, GRAPH, display_name="Invented gadgets") as g:
                graphs.set_membership(g, self.p1.paper_id, "member", is_seed=True)
                graphs.set_membership(g, self.p2.paper_id, "member")
                graphs.set_membership(g, self.p3.paper_id, "frontier")
            view = routine.create_batch(lib, "notes", paper_ids=[self.p1.paper_id], client="claude")
            msg = routine.claim_next(lib, view.batch_id, WorkerRef("interactive", "seed")).message
            res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note=good_note())
            assert res.paper_status == "done"
            self.note_id = res.note_id
            self.noted_batch = view.batch_id
        finally:
            lib.close()


@pytest.fixture
def home(isolated_home):
    return isolated_home


@pytest.fixture
def server(home):
    from seedgraph.mcp.server import build_server

    return build_server(home=home)


@pytest.fixture
def seeded(home):
    return Seeded(home)
