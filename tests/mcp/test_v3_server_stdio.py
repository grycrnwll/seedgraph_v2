"""Opt-in cold-start smoke over real stdio (design §7.1, §11 P7): spawns
``python -m seedgraph mcp serve`` against a temp home, times the cold start, and
round-trips non-ASCII text (μ in a title, the Windows stdio encoding risk).

Deselected by default (``mcp_client`` marker; the suite does not spawn processes).
Run with ``uv run pytest -m mcp_client tests/mcp``. Until P8 re-points ``seedgraph mcp
serve``, the module entry point is the launch command.
"""

from __future__ import annotations

import json
import os
import sys
import time

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from v3_mcp_support import home, seeded  # noqa: F401 (fixtures)

pytestmark = pytest.mark.mcp_client

COLD_START_TARGET_S = 3.0


def test_cold_start_and_utf8_round_trip(seeded, home):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env["SEEDGRAPH_HOME"] = str(home)
    params = StdioServerParameters(command=sys.executable, args=["-m", "seedgraph", "mcp", "serve"], env=env)

    async def _drive():
        started = time.perf_counter()
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            version = await session.call_tool("version", {})
            ready = time.perf_counter() - started
            shown = await session.call_tool("paper_show", {"paper_id": seeded.p1.paper_id})
            return ready, json.loads(version.content[0].text), json.loads(shown.content[0].text)

    ready, version, shown = anyio.run(_drive)
    print(f"v3 MCP cold start to first reply: {ready:.2f} s")
    assert version["server"] == "v3" and version["home"] == str(home)
    assert shown["title"] == "Invented gadget subsidies and μ"
    assert ready < COLD_START_TARGET_S, f"cold start took {ready:.2f} s"
