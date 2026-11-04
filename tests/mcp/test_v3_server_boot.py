"""v3 MCP server boot (design §7.1, §7.2): the version canary, the tool surface and its
annotations, the "graph, never project" rule, and the refusal to create a library."""

from __future__ import annotations

from v3_mcp_support import call, err, list_tools, ok, home, server  # noqa: F401 (fixtures)

from seedgraph import __version__

READ_TOOLS = {"version", "graph_list", "graph_show", "paper_list", "paper_show", "search", "paper_read",
              "lens_results", "batch_status", "batch_list", "concept_search", "concept_show", "literatures",
              "concept_hubs"}
MUTATION_TOOLS = {"batch_create", "work_claim", "work_submit", "work_release", "batch_stop", "batch_continue",
                  "batch_report_limit", "record_answer"}
GRAPH_SCOPED = {"graph_show", "paper_list", "search", "lens_results", "batch_create", "concept_search",
                "concept_show", "literatures", "concept_hubs"}


def test_build_touches_no_disk(home):
    from seedgraph.mcp.server import build_server

    server = build_server(home=home)
    assert list(home.iterdir()) == []
    assert ok(call(server, "version"))["seedgraph"] == __version__
    assert list(home.iterdir()) == []


def test_server_name_and_version(server):
    assert server.name == "seedgraph"
    assert server._mcp_server.version == __version__


def test_missing_library_is_refused_not_created(server, home):
    body = err(call(server, "graph_list"))
    assert body["code"] == "library_not_found"
    assert body["next_action"]
    assert not (home / "library").exists()


def test_tool_surface_and_annotations(server):
    tools = {t.name: t for t in list_tools(server)}
    assert set(tools) == READ_TOOLS | MUTATION_TOOLS
    for name in READ_TOOLS:
        assert tools[name].annotations.readOnlyHint is True, name
    for name in MUTATION_TOOLS:
        assert tools[name].annotations.readOnlyHint is False, name


def test_graph_never_project(server):
    for tool in list_tools(server):
        props = tool.inputSchema.get("properties", {})
        assert "project" not in (tool.description or "").lower(), tool.name
        assert "mcp_ctx" not in props, tool.name
        for pname, prop in props.items():
            assert "project" not in pname.lower(), (tool.name, pname)
            assert "project" not in (prop.get("description") or "").lower(), (tool.name, pname)
        if tool.name in GRAPH_SCOPED:
            assert "graph" in props, tool.name


def test_search_has_no_audience_parameter(server):
    (search,) = [t for t in list_tools(server) if t.name == "search"]
    assert "audience" not in search.inputSchema.get("properties", {})
