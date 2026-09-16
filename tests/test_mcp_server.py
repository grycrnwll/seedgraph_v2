"""Chunk-0 MCP server tests (plan 01) — fully offline, no subprocess network.

Skipped cleanly when the optional ``[mcp]`` extra is absent (``importorskip``), so
the base install stays green; CI installs the extra so these run there. All boot
tests use the SDK's in-memory client<->server transport (no spawned process).
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

import pytest

# Skips the whole module cleanly on a no-extra install. NOTE: this line itself puts
# `mcp` into sys.modules, which is exactly why the no-eager-import check below must
# run in a *subprocess* rather than asserting on this process's sys.modules.
pytest.importorskip("mcp", reason="the optional [mcp] extra is not installed")

import anyio  # noqa: E402  (an mcp dependency; safe to import once importorskip passed)
from mcp.shared.memory import (  # noqa: E402
    create_connected_server_and_client_session,
)
from typer.testing import CliRunner  # noqa: E402

from seedgraph import __version__ as SEEDGRAPH_VERSION  # noqa: E402
from seedgraph.cli import app  # noqa: E402


@pytest.fixture
def run_mcp():
    """Drive an in-memory client<->server session: ``run_mcp(server, coro_fn)``.

    Builds the anyio event loop and the in-memory transport in one place so SDK
    churn (design risk 2) lands here. ``coro_fn`` is an async callable taking the
    connected ``ClientSession`` and returning whatever the test needs. Later
    chunks reuse this fixture (they build a server pointed at a fixture ``root``).
    The helper takes ``server._mcp_server`` — the low-level ``Server`` the
    in-memory transport wires up, not the FastMCP wrapper.
    """

    def _run(server, coro_fn):
        async def _driver():
            async with create_connected_server_and_client_session(
                server._mcp_server
            ) as client:
                return await coro_fn(client)

        return anyio.run(_driver)

    return _run


def _payload(result) -> dict:
    """Extract the version tool's dict payload from a CallToolResult."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict) and "seedgraph" in structured:
        return structured
    return json.loads(result.content[0].text)


def test_import_shadowing_pin():
    """`from mcp.server.fastmcp import FastMCP` inside seedgraph.mcp resolves to the SDK."""
    import seedgraph.mcp.server as srv
    from mcp.server.fastmcp import FastMCP

    # The name bound inside the seedgraph.mcp.server module is the SDK's FastMCP,
    # not a shadowed seedgraph.mcp.* symbol.
    assert srv.FastMCP is FastMCP


def test_version_tool_boots(run_mcp):
    """In-memory client lists tools and calls the `version` boot canary."""
    from seedgraph.mcp.server import build_server

    server = build_server()

    async def _call(client):
        listed = await client.list_tools()
        assert "version" in {t.name for t in listed.tools}
        return await client.call_tool("version", {})

    result = run_mcp(server, _call)
    assert result.isError is False
    payload = _payload(result)
    assert payload["seedgraph"] == SEEDGRAPH_VERSION
    assert "sqlite" in payload and payload["sqlite"]


def test_server_advertises_seedgraph_version_not_sdk_version():
    """The `initialize` handshake must report seedgraph's version, not the SDK's.

    FastMCP's constructor takes no `version` and does not forward one to the
    lowlevel ``Server``, which then falls back to ``pkg_version("mcp")``
    (``server_version=self.version if self.version else pkg_version("mcp")``).
    Without an explicit set, every host is told seedgraph is version 1.28.1 — the
    MCP SDK's version. This pins the field the SDK actually reads when it builds
    ``serverInfo``; the real wire value is covered by the opt-in stdio test.
    """
    from importlib.metadata import version as pkg_version

    from seedgraph.mcp.server import build_server

    advertised = build_server()._mcp_server.version

    assert advertised == SEEDGRAPH_VERSION
    # Names the regression: the fallback would have reported the SDK's version.
    assert advertised != pkg_version("mcp")


def test_missing_extra_serve_exits_with_hint(monkeypatch):
    """`mcp serve` with the extra absent exits 1 and prints the install hint.

    A ``None`` entry in ``sys.modules`` makes ``import seedgraph.mcp.server`` raise
    ImportError — the import lives in the command body, so patching a module
    attribute would not work.
    """
    monkeypatch.setitem(sys.modules, "seedgraph.mcp.server", None)
    result = CliRunner().invoke(app, ["mcp", "serve"])
    assert result.exit_code == 1
    assert ".[mcp]" in result.output


def test_base_import_does_not_pull_sdk():
    """A fresh `import seedgraph` / `seedgraph.cli` must not import the SDK.

    Runs in a subprocess: this test module already imported `mcp` (via
    importorskip), so an in-process ``"mcp" not in sys.modules`` check would fail
    spuriously. A clean interpreter is the only honest proof.
    """
    code = "import sys, seedgraph, seedgraph.cli; assert 'mcp' not in sys.modules"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


# ===========================================================================
# Chunk 1 — read/query tools (plan 01 chunk 1). Fully offline, in-memory client.
# ===========================================================================


def _tool_payload(result) -> dict:
    """Parse a chunk-1 tool's dict payload off the text channel.

    Chunk-1 tools return a bare ``dict`` (no Pydantic output schema, by design),
    so FastMCP serializes it as JSON on ``content[0].text``.
    """
    return json.loads(result.content[0].text)


def _tool_error_body(result) -> dict:
    """Parse the ``{code, message, detail}`` body off a structured tool error.

    FastMCP PREFIXES the JSON with ``Error executing tool <name>: `` — strip to
    the first brace before parsing (the SDK fact pinned in chunk 0).
    """
    text = result.content[0].text
    return json.loads(text[text.index("{"):])


def _seed_project(slug: str = "proj"):
    """Create a project with two works and one run's worth of events.

    Reuses the service-layer fixture builders (the same seams the CLI/UI use), so
    the parity assertions compare the tool against the identical call path.
    """
    from seedgraph import run as run_mod
    from seedgraph.project import service

    h = service.create_project(slug)
    service.add_work(
        h, ids={"doi": "10.1000/aaa", "openalex": "W_A"}, title="Paper A", is_seed=True
    )
    service.add_work(
        h,
        ids={"doi": "10.1000/bbb", "openalex": "W_B"},
        title="Paper B",
        inclusion_status="metadata_only",
    )
    run_mod.append_event(slug, "run_0001", phase="sections", event="started")
    run_mod.append_event(slug, "run_0001", phase="sections", event="progress")
    return h


def _rt(value):
    """JSON round-trip a service return so it compares equal to a tool's payload
    (tuples -> lists, and any non-JSON-native scalars normalize identically)."""
    return json.loads(json.dumps(value))


def test_read_tools_registered(run_mcp):
    """All 9 chunk-1 tools plus `version` are advertised by the server."""
    from seedgraph.mcp.server import build_server

    server = build_server()

    async def _call(client):
        listed = await client.list_tools()
        return {t.name for t in listed.tools}

    names = run_mcp(server, _call)
    expected = {
        "project_list",
        "project_dashboard",
        "list_documents",
        "corpus_status",
        "review_list",
        "run_list",
        "run_events",
        "lens_results",
        "lens_staleness",
    }
    assert expected <= names
    assert "version" in names


def test_tool_schema_preserved(run_mcp):
    """The redaction wrapper preserves each tool's JSON schema (params, not `**kwargs`).

    The discriminating check: a param-bearing tool must still advertise its own
    properties — if `functools.wraps`/`__signature__` failed, the wrapper would
    flatten the schema to nothing.
    """
    from seedgraph.mcp.server import build_server

    server = build_server()

    async def _call(client):
        listed = await client.list_tools()
        return {t.name: t.inputSchema for t in listed.tools}

    schemas = run_mcp(server, _call)
    events_props = set(schemas["run_events"].get("properties", {}))
    assert {"project", "run_id", "after"} <= events_props
    docs_props = set(schemas["list_documents"].get("properties", {}))
    assert {"project", "statuses"} <= docs_props


def test_read_tool_parity(run_mcp, isolated_home):
    """Each read tool's payload equals the direct service call, JSON-round-tripped."""
    from seedgraph import run as run_mod
    from seedgraph.acquisition import service as acq_service
    from seedgraph.lenses import registry as lens_registry
    from seedgraph.lenses import runner as lens_runner
    from seedgraph.mcp.server import build_server
    from seedgraph.project import review as project_review
    from seedgraph.project import service
    from sqlmodel import Session

    h = _seed_project("proj")
    server = build_server(root=isolated_home)

    def _call(name, args):
        async def _c(client):
            return await client.call_tool(name, args)

        result = run_mcp(server, _c)
        assert result.isError is False, name
        return _tool_payload(result)

    # project_list
    assert _call("project_list", {}) == _rt(
        {"projects": service.list_projects(root=isolated_home)}
    )
    # project_dashboard
    assert _call("project_dashboard", {"project": "proj"}) == _rt(
        service.project_dashboard(h)
    )
    # list_documents
    assert _call("list_documents", {"project": "proj"}) == _rt(
        {"documents": service.list_documents(h)}
    )
    # corpus_status
    assert _call("corpus_status", {"project": "proj"}) == _rt(
        {"rows": acq_service.corpus_rows(h)}
    )
    # review_list
    assert _call("review_list", {"project": "proj"}) == _rt(
        {"items": project_review.review_rows(h, project_review.list_open(h))}
    )
    # run_list
    assert _call("run_list", {"project": "proj"}) == _rt(
        {"runs": run_mod.list_runs("proj", root=isolated_home)}
    )
    # lens_results (no lens seeded -> empty, still a real parity check)
    with Session(h.engine) as session:
        direct_lens = lens_runner.lens_results(session, "lens_x")
    assert _call("lens_results", {"project": "proj", "lens_id": "lens_x"}) == _rt(
        {"results": direct_lens}
    )
    # lens_staleness (unregistered -> zeroed dict)
    project_dir = h.root / "projects" / h.slug
    with Session(h.engine) as session:
        direct_stale = lens_registry.lens_staleness(session, project_dir, "lens_x")
    assert _call("lens_staleness", {"project": "proj", "lens_id": "lens_x"}) == _rt(
        direct_stale
    )


def test_run_events_polling(run_mcp, isolated_home):
    """`run_events` polls monotonically and reports the terminal status."""
    from seedgraph import run as run_mod
    from seedgraph.mcp.server import build_server
    from seedgraph.project import service

    # Create the project first (get_handle needs the dir), then log its run's
    # events — appending first would mkdir the project dir and make create_project
    # hard-error on an existing directory.
    service.create_project("proj2")
    run_mod.append_event("proj2", "r1", phase="p", event="started")
    run_mod.append_event("proj2", "r1", phase="p", event="progress")

    server = build_server(root=isolated_home)

    def _call(args):
        async def _c(client):
            return await client.call_tool("run_events", args)

        return _tool_payload(run_mcp(server, _c))

    first = _call({"project": "proj2", "run_id": "r1"})
    seqs = [e["seq"] for e in first["events"]]
    assert seqs == sorted(seqs)  # monotonic
    assert seqs[0] == 0
    assert first["status"] == "running"
    assert first["next_after"] == seqs[-1]

    # Poll again after the last seq -> no new events, status still running.
    nxt = _call({"project": "proj2", "run_id": "r1", "after": first["next_after"]})
    assert nxt["events"] == []
    assert nxt["next_after"] == first["next_after"]

    # A terminal event flips the derived status.
    run_mod.append_event("proj2", "r1", phase="p", event="finished")
    done = _call({"project": "proj2", "run_id": "r1"})
    assert done["status"] == "finished"

    # A run_id with no events yet: empty log -> status None, next_after unchanged.
    empty = _call({"project": "proj2", "run_id": "no_such_run"})
    assert empty["events"] == []
    assert empty["status"] is None
    assert empty["next_after"] == -1


def test_redact_response_unit():
    """The §5.1 walk: private text -> marker, shareable text untouched, off = no-op.

    Points at BOTH the chunk-1 carrier field (`claim_text`) and the chunk-3
    `ask`-envelope fields (`text`/`quote`) so "chunk 3 plugs in for free" is an
    assertion, not a hope. The redaction verdict is checked against the export
    single source (`vocab.is_shareable`) on both access classes.
    """
    from seedgraph.mcp.tools_read import redact_response
    from seedgraph.vocab import is_shareable

    lens_payload = {
        "results": [
            {"claim_text": "SECRET", "access_class": "user_supplied_private"},
            {"claim_text": "PUBLIC", "access_class": "open_access"},
        ]
    }
    envelope = {
        "spans": [{"text": "priv span", "access_class": "user_supplied_private"}],
        "citations": [{"quote": "open quote", "access_class": "open_access"}],
    }

    # Flag off: identical object, no mutation.
    assert redact_response(lens_payload, redact_private=False) is lens_payload

    on = redact_response(lens_payload, redact_private=True)
    assert on["results"][0]["claim_text"] == {
        "redacted": True,
        "access_class": "user_supplied_private",
    }
    assert on["results"][1]["claim_text"] == "PUBLIC"
    # Original untouched (walk builds fresh containers).
    assert lens_payload["results"][0]["claim_text"] == "SECRET"

    # Verdict matches the export single source on both classes.
    for orig in lens_payload["results"]:
        ac = orig["access_class"]
        row = next(r for r in on["results"] if r["access_class"] == ac)
        is_marker = isinstance(row["claim_text"], dict) and row["claim_text"].get(
            "redacted"
        )
        assert bool(is_marker) == (not is_shareable(ac))

    env_on = redact_response(envelope, redact_private=True)
    assert env_on["spans"][0]["text"] == {
        "redacted": True,
        "access_class": "user_supplied_private",
    }
    assert env_on["citations"][0]["quote"] == "open quote"


def test_lens_results_redaction_wired(run_mcp, isolated_home, monkeypatch):
    """The wrapper actively redacts a real tool's output end-to-end.

    Monkeypatch the `lenses.runner.lens_results` seam (patched as a module
    attribute — the tool calls it via `lens_runner.lens_results`, so the patch
    bites) so the tool round-trips synthetic rows through the in-memory client and
    the registration wrapper, without the heavy non-stale-run seeding chain.
    """
    from seedgraph.mcp.server import build_server
    from seedgraph.project import service

    service.create_project("proj3")

    def fake_lens_results(session, lens_id, status="found"):
        return [
            {
                "lens_output_id": "lo1",
                "work_id": "W_A",
                "status": "found",
                "claim_text": "PRIVATE CLAIM TEXT",
                "normalized_label": "label-a",
                "access_class": "user_supplied_private",
            },
            {
                "lens_output_id": "lo2",
                "work_id": "W_B",
                "status": "found",
                "claim_text": "OPEN CLAIM TEXT",
                "normalized_label": "label-b",
                "access_class": "open_access",
            },
        ]

    monkeypatch.setattr(
        "seedgraph.lenses.runner.lens_results", fake_lens_results
    )

    def _rows(redact: bool):
        server = build_server(root=isolated_home, redact_private=redact)

        async def _c(client):
            return await client.call_tool(
                "lens_results", {"project": "proj3", "lens_id": "L"}
            )

        return _tool_payload(run_mcp(server, _c))["results"]

    off = _rows(False)
    assert off[0]["claim_text"] == "PRIVATE CLAIM TEXT"
    assert off[1]["claim_text"] == "OPEN CLAIM TEXT"

    on = _rows(True)
    assert on[0]["claim_text"] == {
        "redacted": True,
        "access_class": "user_supplied_private",
    }
    # normalized_label is synthesis, not verbatim text -> not blanked.
    assert on[0]["normalized_label"] == "label-a"
    # open_access row untouched.
    assert on[1]["claim_text"] == "OPEN CLAIM TEXT"


def test_redaction_wrapper_noops_on_metadata(run_mcp, isolated_home):
    """The wrapper is applied to every tool but no-ops on metadata rows.

    Most chunk-1 tools carry no `access_class`/text pair, so redact-on and
    redact-off payloads are byte-identical — proving the wrapper is wired and
    harmless where there is nothing to redact. (Chunk 3's `ask` exercises the
    active path fully.)
    """
    from seedgraph.mcp.server import build_server

    _seed_project("proj4")

    def _payload(redact: bool, name: str, args: dict):
        server = build_server(root=isolated_home, redact_private=redact)

        async def _c(client):
            return await client.call_tool(name, args)

        return _tool_payload(run_mcp(server, _c))

    for name, args in (
        ("list_documents", {"project": "proj4"}),
        ("corpus_status", {"project": "proj4"}),
        ("project_dashboard", {"project": "proj4"}),
    ):
        assert _payload(False, name, args) == _payload(True, name, args), name


def test_project_not_found(run_mcp, isolated_home):
    """A bad slug surfaces the structured `project_not_found` error (M9)."""
    from seedgraph.mcp.server import build_server

    server = build_server(root=isolated_home)

    async def _c(client):
        return await client.call_tool("project_dashboard", {"project": "no_such_proj"})

    result = run_mcp(server, _c)
    assert result.isError is True
    body = _tool_error_body(result)
    assert body["code"] == "project_not_found"


# ===========================================================================
# Chunk 2 — concept / search / graph tools + the M10 x M7 `conn=` rider.
# Fully offline, in-memory client; real seeded rows (not monkeypatches) so the
# tool-body access_class stamp runs its real SELECT.
# ===========================================================================


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _seed_semantic(slug: str):
    """A project with one concept (``concept::ols``) linked to a PUBLIC and a
    PRIVATE claim, each anchored to a span, plus ``span_fts`` rows so
    ``search_spans`` finds them. Two access classes give redaction something to
    withhold (private) and something to leave (open_access). Raw INSERTs mirror the
    proven ``tests/test_web_graph3d_provenance.py`` / ``tests/test_phase_3.py``
    seeding (the concept->claim->span chain + the FTS index)."""
    from seedgraph.db.connection import connect_project_raw
    from seedgraph.project import service

    h = service.create_project(slug)
    conn = connect_project_raw(h.db_path)
    try:

        def add_work(wid, title, access):
            conn.execute(
                "INSERT INTO works (work_id, canonical_title, year, created_at) "
                "VALUES (?,?,?,?)",
                (wid, title, 2020, _now()),
            )
            conn.execute(
                "INSERT INTO project_documents (work_id, inclusion_status, is_seed, "
                "created_at, updated_at) VALUES (?,?,?,?,?)",
                (wid, "included", 0, _now(), _now()),
            )
            conn.execute(
                "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, "
                "markdown_hash, schema_version, prompt_version, access_class, "
                "run_status, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                ("extr_" + wid, wid, "md_" + wid, "h", "v1", "p1", access, "success",
                 _now()),
            )

        def add_claim(cid, wid, text, access):
            conn.execute(
                "INSERT INTO extracted_claims (claim_id, extraction_run_id, work_id, "
                "claim_type, claim_subtype, field_key, normalized_label, claim_text, "
                "status, epistemic_type, access_class, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (cid, "extr_" + wid, wid, "method", None, "f", "lbl", text, "found",
                 "llm_extracted", access, _now()),
            )
            conn.execute(
                "INSERT INTO claim_concepts (claim_id, concept_id, work_id, "
                "epistemic_type, created_at) VALUES (?,?,?,?,?)",
                (cid, "concept::ols", wid, "deterministic", _now()),
            )

        def add_span(sid, cid, wid, quote, access):
            conn.execute(
                "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, "
                "source_file_id, source_file_hash, work_id, section_id, start_char, "
                "end_char, exact_quote, quote_hash, page_start, page_end, "
                "access_class, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, "md_" + wid, "h", "sf", "sfh", wid, None, 0, 20, quote,
                 "qh_" + sid, 1, 1, access, _now()),
            )
            conn.execute(
                "INSERT INTO span_fts (quote_text, span_id, markdown_id, work_id, "
                "section_id) VALUES (?,?,?,?,?)",
                (quote, sid, "md_" + wid, wid, None),
            )
            conn.execute(
                "INSERT INTO claim_spans (claim_id, span_id, rank, created_at) "
                "VALUES (?,?,?,?)",
                (cid, sid, 0, _now()),
            )

        conn.execute(
            "INSERT INTO concepts (concept_id, normalized_label, canonical_label, "
            "concept_type, paper_frequency, weight, status, epistemic_type, "
            "access_class, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("concept::ols", "ols", "OLS", "method", 2, 0.5, "auto", "deterministic",
             "open_access", _now(), _now()),
        )
        add_work("work_pub", "Public Paper", "open_access")
        add_work("work_priv", "Private Paper", "user_supplied_private")
        add_claim("claim_pub", "work_pub", "PUBLIC CLAIM parallel trends", "open_access")
        add_claim(
            "claim_priv", "work_priv", "PRIVATE CLAIM parallel trends",
            "user_supplied_private",
        )
        add_span("span_pub", "claim_pub", "work_pub", "PUBLIC SPAN parallel trends",
                 "open_access")
        add_span("span_priv", "claim_priv", "work_priv", "PRIVATE SPAN parallel trends",
                 "user_supplied_private")
        conn.commit()
    finally:
        conn.close()
    return h


def _seed_edge_run(slug: str):
    """A project with one provider_reference edge A->B under run 'R', with the run
    dir materialized (via ``build_and_export``) so ``run.latest_run_id`` resolves
    to 'R' — mirrors ``tests/test_run_view.py`` run-seeding."""
    from seedgraph.citation.edges import write_edge
    from seedgraph.db.connection import connect_project_raw
    from seedgraph.graph.run_view import build_and_export
    from seedgraph.project import service

    h = service.create_project(slug)
    a = service.add_work(h, ids={"doi": "10.5/a", "openalex": "W_GA"}, title="Paper A")
    b = service.add_work(h, ids={"doi": "10.5/b", "openalex": "W_GB"}, title="Paper B")
    conn = connect_project_raw(h.db_path)
    try:
        write_edge(
            conn, source=a.work_id, target=b.work_id,
            provenance="provider_reference", confidence=1.0, run_id="R",
        )
        conn.commit()
    finally:
        conn.close()
    build_and_export(h.slug, h.root, h.root, "R", open_world=False)
    return h


def _direct_conn(slug, isolated_home):
    from seedgraph.db.connection import open_project_db

    return open_project_db(slug, root=isolated_home)


def test_concept_graph_tools_registered(run_mcp, isolated_home):
    """All 5 chunk-2 tools are advertised by the server."""
    from seedgraph.mcp.server import build_server

    server = build_server(root=isolated_home)

    async def _call(client):
        listed = await client.list_tools()
        return {t.name for t in listed.tools}

    names = run_mcp(server, _call)
    assert {
        "concepts_overview",
        "concepts_list",
        "concept_show",
        "search_spans",
        "graph_analyze",
    } <= names


def test_concepts_and_search_parity(run_mcp, isolated_home):
    """`concepts_overview` / `concepts_list` / `search_spans` payloads equal the
    direct service call, JSON-round-tripped (default server: flag off, no stamp)."""
    from dataclasses import asdict

    from seedgraph.fts import search as fts_search
    from seedgraph.mcp.server import build_server
    from seedgraph.semantic import query as semantic_query

    _seed_semantic("c2parity")
    server = build_server(root=isolated_home)

    def _call(name, args):
        async def _c(client):
            return await client.call_tool(name, args)

        result = run_mcp(server, _c)
        assert result.isError is False, name
        return _tool_payload(result)

    conn = _direct_conn("c2parity", isolated_home)
    try:
        assert _call("concepts_overview", {"project": "c2parity"}) == _rt(
            semantic_query.concept_overview(conn)
        )
        assert _call("concepts_list", {"project": "c2parity"}) == _rt(
            {"concepts": semantic_query.list_concepts(conn)}
        )
        # Typed filter is a structural pass-through of the service arg.
        assert _call(
            "concepts_list", {"project": "c2parity", "concept_type": "method"}
        ) == _rt(
            {"concepts": semantic_query.list_concepts(conn, concept_type="method")}
        )
        hits = fts_search.search_spans(conn, "parallel trends")
        assert _call(
            "search_spans", {"project": "c2parity", "query": "parallel trends"}
        ) == _rt({"hits": [asdict(hit) for hit in hits]})
        # work_id filter behaves as the service call.
        only_pub = fts_search.search_spans(
            conn, "parallel trends", work_id="work_pub"
        )
        assert _call(
            "search_spans",
            {"project": "c2parity", "query": "parallel trends", "work_id": "work_pub"},
        ) == _rt({"hits": [asdict(hit) for hit in only_pub]})
    finally:
        conn.close()


def test_concept_show_found_and_not_found(run_mcp, isolated_home):
    """`concept_show` found -> equals `concept_detail`; unknown id -> typed error."""
    from seedgraph.mcp.server import build_server
    from seedgraph.semantic import query as semantic_query

    _seed_semantic("c2show")
    server = build_server(root=isolated_home)

    async def _found(client):
        return await client.call_tool(
            "concept_show", {"project": "c2show", "concept_id": "concept::ols"}
        )

    result = run_mcp(server, _found)
    assert result.isError is False
    conn = _direct_conn("c2show", isolated_home)
    try:
        assert _tool_payload(result) == _rt(
            semantic_query.concept_detail(conn, "concept::ols")
        )
    finally:
        conn.close()

    async def _missing(client):
        return await client.call_tool(
            "concept_show",
            {"project": "c2show", "concept_id": "concept::does-not-exist"},
        )

    err = run_mcp(server, _missing)
    assert err.isError is True
    assert _tool_error_body(err)["code"] == "concept_not_found"


def test_graph_analyze_tool_and_conn_rider(run_mcp, isolated_home):
    """`graph_analyze` returns the summary dict AND the M7-conn path equals the
    own-conn path (the additive `conn=` rider changes nothing)."""
    from seedgraph.graph.analyze import analyze_summary
    from seedgraph.mcp.server import build_server

    h = _seed_edge_run("c2ga")
    server = build_server(root=isolated_home)

    async def _c(client):
        return await client.call_tool("graph_analyze", {"project": "c2ga"})

    result = run_mcp(server, _c)
    assert result.isError is False
    payload = _tool_payload(result)
    # M7-conn (tool) == own-conn (direct analyze_summary) -> the rider is byte-safe.
    assert payload == _rt(analyze_summary(h))
    # Shape (the MCP graph_analyze contract) + the fixture's single edge.
    assert set(payload) == {
        "run_id", "node_count", "edge_count", "community_count", "communities",
        "god_nodes", "bridge_edge_count", "bridge_node_count", "bridges",
        "unread_count", "read_next",
    }
    assert payload["run_id"] == "R"
    assert payload["node_count"] == 2
    assert payload["edge_count"] == 1


def test_graph_analyze_no_citation_run(run_mcp, isolated_home):
    """No citation run at all -> the MCP-only strict `no_citation_run` error (M9)."""
    from seedgraph.mcp.server import build_server
    from seedgraph.project import service

    service.create_project("c2ga_empty")
    server = build_server(root=isolated_home)

    async def _c(client):
        return await client.call_tool("graph_analyze", {"project": "c2ga_empty"})

    result = run_mcp(server, _c)
    assert result.isError is True
    assert _tool_error_body(result)["code"] == "no_citation_run"


def test_concept_show_redaction(run_mcp, isolated_home):
    """`concept_show`: flag off shows private claim/span text; flag on shows the
    `{redacted, access_class}` marker; open_access rows untouched.

    Proves the chunk-2 tool-body fix: `concept_detail`'s claim rows carry no
    access_class of their own (stamped in the body) and its span text field is
    `exact_quote` (registered), so both are covered only via this chunk's changes.
    """
    from seedgraph.mcp.server import build_server

    _seed_semantic("c2redshow")

    def _show(redact: bool) -> dict:
        server = build_server(root=isolated_home, redact_private=redact)

        async def _c(client):
            return await client.call_tool(
                "concept_show", {"project": "c2redshow", "concept_id": "concept::ols"}
            )

        return _tool_payload(run_mcp(server, _c))

    off = _show(False)
    claims_off = {c["claim_id"]: c for c in off["claims"]}
    spans_off = {s["span_id"]: s for s in off["spans"]}
    assert claims_off["claim_priv"]["claim_text"] == "PRIVATE CLAIM parallel trends"
    assert spans_off["span_priv"]["exact_quote"] == "PRIVATE SPAN parallel trends"

    on = _show(True)
    claims_on = {c["claim_id"]: c for c in on["claims"]}
    spans_on = {s["span_id"]: s for s in on["spans"]}
    # Private claim_text + span exact_quote -> withheld marker.
    assert claims_on["claim_priv"]["claim_text"] == {
        "redacted": True,
        "access_class": "user_supplied_private",
    }
    assert spans_on["span_priv"]["exact_quote"] == {
        "redacted": True,
        "access_class": "user_supplied_private",
    }
    # open_access claim + span untouched.
    assert claims_on["claim_pub"]["claim_text"] == "PUBLIC CLAIM parallel trends"
    assert spans_on["span_pub"]["exact_quote"] == "PUBLIC SPAN parallel trends"


def test_search_spans_redaction(run_mcp, isolated_home):
    """`search_spans`: flag off shows private `quote_text`; flag on shows the
    marker; open_access hit untouched (the SpanHit-lacks-access_class gap fix)."""
    from seedgraph.mcp.server import build_server

    _seed_semantic("c2redsearch")

    def _hits(redact: bool) -> dict:
        server = build_server(root=isolated_home, redact_private=redact)

        async def _c(client):
            return await client.call_tool(
                "search_spans", {"project": "c2redsearch", "query": "parallel trends"}
            )

        return {
            hit["span_id"]: hit
            for hit in _tool_payload(run_mcp(server, _c))["hits"]
        }

    off = _hits(False)
    assert off["span_priv"]["quote_text"] == "PRIVATE SPAN parallel trends"
    assert off["span_pub"]["quote_text"] == "PUBLIC SPAN parallel trends"

    on = _hits(True)
    assert on["span_priv"]["quote_text"] == {
        "redacted": True,
        "access_class": "user_supplied_private",
    }
    assert on["span_pub"]["quote_text"] == "PUBLIC SPAN parallel trends"


# ===========================================================================
# Chunk 3 — `ask` + the §5.3 budget handshake.
#
# FULLY OFFLINE AND KEYLESS. The LLM seam is the repo's existing mock:
# ``answer.compose._BACKEND_OVERRIDE`` fed a ``llm.backend.FakeLLMBackend``
# (the same seam ``tests/test_phase_8*.py`` uses), whose ``.calls`` list is the
# cost-safety assertion — a real provider is never constructed, let alone called.
# ===========================================================================


@pytest.fixture
def fake_backend():
    """Install a FakeLLMBackend on the compose override seam; yield it for assertions.

    The teardown clear mirrors ``tests/test_phase_8.py::_clear_backend`` so a leaked
    override can never make a later test dispatch.
    """
    from seedgraph.answer import compose
    from seedgraph.llm.backend import FakeLLMBackend

    backend = FakeLLMBackend(
        response=json.dumps(
            {
                "query_type": "factual",
                "answer_category": "source_grounded",
                "answer_text": "Parallel trends is assumed [E1].",
                "cited_markers": [1],
                "insufficient_evidence": False,
            }
        )
    )
    compose._BACKEND_OVERRIDE = backend
    yield backend
    compose._BACKEND_OVERRIDE = None


def _ask(run_mcp, server, args: dict):
    async def _c(client):
        return await client.call_tool("ask", args)

    return run_mcp(server, _c)


def test_ask_tool_registered_with_free_defaults(run_mcp, isolated_home):
    """`ask` is advertised and its schema pins the FREE default (`no_llm=true`).

    The default is the cost contract: a client that omits `no_llm` must land on the
    deterministic floor, so the advertised default is asserted, not assumed.
    """
    from seedgraph.mcp.server import build_server

    server = build_server(root=isolated_home)

    async def _call(client):
        listed = await client.list_tools()
        return {t.name: t.inputSchema for t in listed.tools}

    schemas = run_mcp(server, _call)
    assert "ask" in schemas
    props = schemas["ask"]["properties"]
    assert {"project", "question", "no_llm", "mode", "graph_depth",
            "max_candidates", "confirm_spend"} <= set(props)
    assert props["no_llm"]["default"] is True
    assert props["confirm_spend"]["default"] is False
    assert props["mode"]["default"] == "project_only"
    # `question` has no default -> it is required; `no_llm` must NOT be required.
    assert set(schemas["ask"].get("required", [])) == {"project", "question"}


def test_ask_default_path_is_free_and_makes_zero_backend_calls(
    run_mcp, isolated_home, fake_backend
):
    """THE cost-safety test: `ask` with DEFAULTS returns a retrieval-only envelope
    and the mock LLM backend is invoked ZERO times.

    The backend is installed and armed with a valid response, so a dispatch would
    succeed and produce prose — `answer_text == ""` plus `calls == []` is therefore
    proof the free path never reached a model, not proof the mock was broken.
    """
    from seedgraph.mcp.server import build_server

    _seed_semantic("c3free")
    server = build_server(root=isolated_home)

    result = _ask(run_mcp, server, {"project": "c3free", "question": "parallel trends?"})
    assert result.isError is False
    env = _tool_payload(result)

    assert fake_backend.calls == []  # <- zero paid dispatches on the default path
    assert env["mode"] == "retrieval_only"
    assert env["answer_text"] == ""
    assert env["llm_provenance"] is None
    assert env["answer_id"].startswith("ans_")
    # Smoke: citations resolve to fixture work_ids (guards the serialization path).
    assert set(env["cited_work_ids"]) <= {"work_pub", "work_priv"}
    assert env["cited_work_ids"]


def test_ask_rejects_invalid_mode(run_mcp, isolated_home, fake_backend):
    """An unknown `mode` is a structured error, and still dispatches nothing."""
    from seedgraph.mcp.server import build_server

    _seed_semantic("c3mode")
    server = build_server(root=isolated_home)

    result = _ask(
        run_mcp, server,
        {"project": "c3mode", "question": "q?", "mode": "retrieval_only"},
    )
    assert result.isError is True
    assert _tool_error_body(result)["code"] == "invalid_mode"
    assert fake_backend.calls == []


def _arm_confirmation(slug, isolated_home, *, threshold=0.0):
    """Point `answer_generation` at the always-available LOCAL profile and arm the
    project's confirmation policy at `threshold` USD.

    `require_confirmation_above_usd = 0.0` fires for ANY estimate (`run >= threshold`),
    so the handshake is exercised with a $0 local estimate — no key, no network, no
    spend. Written through the real `write_project_overrides` seam so the tool's own
    `load_project_config` read is the thing under test.
    """
    from seedgraph.config.loader import write_project_overrides

    write_project_overrides(
        slug,
        {
            "llm": {
                "routes": {
                    "answer_generation": {
                        "preferred_profile": "local_ollama_default",
                        "fallback_profile": "no_llm",
                    }
                }
            },
            "budget": {"require_confirmation_above_usd": threshold},
        },
        root=isolated_home,
    )


def test_ask_budget_handshake_fails_closed_then_proceeds(
    run_mcp, isolated_home, fake_backend
):
    """§5.3: armed policy + `no_llm=false` + `confirm_spend=false` fails closed with
    the estimate fields; the SAME call with `confirm_spend=true` proceeds.

    Fail-closed must also mean *no dispatch* — asserted on the mock between the two
    calls, so the error cannot be a post-hoc report of money already spent.
    """
    from seedgraph.mcp.server import build_server

    _seed_semantic("c3armed")
    _arm_confirmation("c3armed", isolated_home)
    server = build_server(root=isolated_home)

    refused = _ask(
        run_mcp, server,
        {"project": "c3armed", "question": "parallel trends?", "no_llm": False},
    )
    assert refused.isError is True
    body = _tool_error_body(refused)
    assert body["code"] == "budget_confirmation_required"
    detail = body["detail"]
    assert {"estimate_usd", "monthly_spend_usd", "monthly_limit_usd", "profile"} <= set(
        detail
    )
    assert detail["profile"] == "local_ollama_default"
    assert isinstance(detail["estimate_usd"], (int, float))
    assert detail["monthly_spend_usd"] == 0.0
    assert fake_backend.calls == []  # fail-closed happened BEFORE any dispatch

    proceeded = _ask(
        run_mcp, server,
        {
            "project": "c3armed",
            "question": "parallel trends?",
            "no_llm": False,
            "confirm_spend": True,
        },
    )
    assert proceeded.isError is False
    env = _tool_payload(proceeded)
    assert len(fake_backend.calls) == 1  # exactly one confirmed dispatch
    assert env["answer_text"] == "Parallel trends is assumed [E1]."
    assert env["mode"] == "project_only"
    assert env["llm_provenance"]["profile"] == "local_ollama_default"
    assert set(env["cited_work_ids"]) <= {"work_pub", "work_priv"}


def test_ask_unarmed_policy_never_asks_for_confirmation(
    run_mcp, isolated_home, fake_backend
):
    """Unarmed policy: `no_llm=false` succeeds with NO error, and `confirm_spend=true`
    on a call that needed no confirmation is a harmless no-op (not an error)."""
    from seedgraph.config.loader import write_project_overrides
    from seedgraph.mcp.server import build_server

    _seed_semantic("c3unarmed")
    # Route to the local profile but leave `require_confirmation_above_usd` unset.
    write_project_overrides(
        "c3unarmed",
        {
            "llm": {
                "routes": {
                    "answer_generation": {
                        "preferred_profile": "local_ollama_default",
                        "fallback_profile": "no_llm",
                    }
                }
            }
        },
        root=isolated_home,
    )
    server = build_server(root=isolated_home)

    plain = _ask(
        run_mcp, server,
        {"project": "c3unarmed", "question": "parallel trends?", "no_llm": False},
    )
    assert plain.isError is False
    assert _tool_payload(plain)["answer_text"] == "Parallel trends is assumed [E1]."

    confirmed = _ask(
        run_mcp, server,
        {
            "project": "c3unarmed",
            "question": "parallel trends?",
            "no_llm": False,
            "confirm_spend": True,
        },
    )
    assert confirmed.isError is False  # no-op, never an error
    assert _tool_payload(confirmed)["answer_text"] == "Parallel trends is assumed [E1]."
    assert len(fake_backend.calls) == 2


def test_ask_monthly_soft_limit_also_requires_confirmation(
    run_mcp, isolated_home, fake_backend
):
    """The OTHER arm of §5.3: crossing the monthly soft limit requires confirmation
    even when `require_confirmation_above_usd` is unset.

    Prior spend is seeded through the real `llm.usage.log_usage` seam (the same
    table `llm.cost.monthly_spend` sums), so the DB-backed month-to-date figure —
    not a monkeypatch — is what trips the gate.
    """
    from seedgraph.config.loader import write_project_overrides
    from seedgraph.db.connection import connect_project_raw
    from seedgraph.llm.usage import UsageEvent, log_usage
    from seedgraph.mcp.server import build_server

    h = _seed_semantic("c3soft")
    conn = connect_project_raw(h.db_path)
    try:
        log_usage(
            conn,
            UsageEvent(task_type="answer_generation", estimated_cost=0.50,
                       status="success"),
        )
    finally:
        conn.close()

    write_project_overrides(
        "c3soft",
        {
            "llm": {"routes": {"answer_generation": {
                "preferred_profile": "local_ollama_default",
                "fallback_profile": "no_llm"}}},
            # Soft limit only — confirmation THRESHOLD deliberately left unset, so a
            # pass here can only come from the soft-limit arm.
            "budget": {"monthly_soft_limit_usd": 0.10},
        },
        root=isolated_home,
    )
    server = build_server(root=isolated_home)

    refused = _ask(
        run_mcp, server,
        {"project": "c3soft", "question": "parallel trends?", "no_llm": False},
    )
    assert refused.isError is True
    detail = _tool_error_body(refused)["detail"]
    assert _tool_error_body(refused)["code"] == "budget_confirmation_required"
    assert detail["over_monthly_soft_limit"] is True
    assert detail["requires_confirmation"] is False  # the isolating assertion
    assert detail["monthly_spend_usd"] == 0.50
    assert detail["monthly_limit_usd"] == 0.10
    assert fake_backend.calls == []

    confirmed = _ask(
        run_mcp, server,
        {"project": "c3soft", "question": "parallel trends?", "no_llm": False,
         "confirm_spend": True},
    )
    assert confirmed.isError is False
    assert len(fake_backend.calls) == 1


def test_ask_gate_refusal_preserves_router_message(
    run_mcp, isolated_home, fake_backend, monkeypatch
):
    """A private corpus routed to an external, source-text-sending profile surfaces
    `gate_refused` carrying the ROUTER's original message verbatim (38/58).

    The gate itself lives in `llm.routing.resolve_route` and fires inside the callee
    (§5.2) — this asserts the MCP layer neither re-words it nor swallows it into a
    silent degrade. The key is a dummy set only so the external profile is
    *selectable*; the router refuses before any transport exists.
    """
    from seedgraph.config.loader import write_project_overrides
    from seedgraph.llm.routing import _CONTENT_GATE_MESSAGE
    from seedgraph.mcp.server import build_server

    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-never-used")
    _seed_semantic("c3gate")
    write_project_overrides(
        "c3gate",
        {
            "llm": {
                "routes": {
                    "answer_generation": {
                        "preferred_profile": "anthropic_api_default",
                        "fallback_profile": None,
                        # The route now claims it sends SOURCE TEXT, so the
                        # content-access gate applies to the retrieved spans'
                        # access_class (private, from the fixture).
                        "requires_source_text": True,
                    }
                }
            },
            # Belt: policy forbids private full text leaving the machine (the default).
            "content_policy": {"external_llm_for_private_full_text": False},
        },
        root=isolated_home,
    )
    server = build_server(root=isolated_home)

    result = _ask(
        run_mcp, server,
        {"project": "c3gate", "question": "parallel trends?", "no_llm": False},
    )
    assert result.isError is True
    body = _tool_error_body(result)
    assert body["code"] == "gate_refused"
    # The router's OWN words, verbatim — not a paraphrase authored by the MCP layer.
    expected = _CONTENT_GATE_MESSAGE.format(
        task_type="answer_generation",
        profile_id="anthropic_api_default",
        access_class="user_supplied_private",
    )
    assert body["message"] == expected
    assert fake_backend.calls == []  # a refused gate never dispatches


def test_ask_no_paid_path_without_both_conditions(
    run_mcp, isolated_home, fake_backend, monkeypatch
):
    """No LLM dispatch is reachable without BOTH `no_llm=false` AND a passed gate.

    Sweeps the two failure axes against a fully paid-capable config: leaving
    `no_llm` at its default never dispatches even when the route is live, and a
    refused gate never dispatches even with `no_llm=false` and `confirm_spend=true`.
    """
    from seedgraph.config.loader import write_project_overrides
    from seedgraph.mcp.server import build_server

    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-never-used")
    _seed_semantic("c3both")
    server = build_server(root=isolated_home)

    # Axis 1: a LIVE local route, but `no_llm` left at its default -> no dispatch.
    write_project_overrides(
        "c3both",
        {"llm": {"routes": {"answer_generation": {
            "preferred_profile": "local_ollama_default", "fallback_profile": "no_llm"}}}},
        root=isolated_home,
    )
    for args in (
        {"project": "c3both", "question": "parallel trends?"},
        {"project": "c3both", "question": "parallel trends?", "confirm_spend": True},
    ):
        result = _ask(run_mcp, server, args)
        assert result.isError is False
        assert _tool_payload(result)["mode"] == "retrieval_only"
    assert fake_backend.calls == []

    # Axis 2: `no_llm=false` + `confirm_spend=true`, but the gate refuses -> no dispatch.
    write_project_overrides(
        "c3both",
        {"llm": {"routes": {"answer_generation": {
            "preferred_profile": "anthropic_api_default",
            "fallback_profile": None,
            "requires_source_text": True}}}},
        root=isolated_home,
    )
    refused = _ask(
        run_mcp, server,
        {
            "project": "c3both",
            "question": "parallel trends?",
            "no_llm": False,
            "confirm_spend": True,
        },
    )
    assert refused.isError is True
    assert _tool_error_body(refused)["code"] == "gate_refused"
    assert fake_backend.calls == []


def test_ask_envelope_citation_redaction(run_mcp, isolated_home):
    """The envelope's verbatim `Citation.quote` is withheld for non-shareable works.

    Redaction-coverage finding for chunk 3: `answer.types.Citation` carries `quote`
    but NO `access_class` (and is `extra="forbid"`), so the §5.1 walk could not see
    it — private span text would have leaked with `--redact-private` on. The tool
    body stamps each citation's access_class from the span its quote was built from;
    the verdict stays `vocab.is_shareable`.
    """
    from seedgraph.mcp.server import build_server
    from seedgraph.vocab import is_shareable

    _seed_semantic("c3redact")

    def _citations(redact: bool) -> dict:
        server = build_server(root=isolated_home, redact_private=redact)
        result = _ask(
            run_mcp, server,
            {"project": "c3redact", "question": "parallel trends?"},
        )
        assert result.isError is False
        return {c["work_id"]: c for c in _tool_payload(result)["citations"]}

    off = _citations(False)
    assert off["work_priv"]["quote"] == "PRIVATE SPAN parallel trends"
    assert off["work_pub"]["quote"] == "PUBLIC SPAN parallel trends"
    assert "access_class" not in off["work_priv"]  # the gap: nothing to key on

    on = _citations(True)
    assert on["work_priv"]["quote"] == {
        "redacted": True,
        "access_class": "user_supplied_private",
    }
    assert on["work_pub"]["quote"] == "PUBLIC SPAN parallel trends"
    # Verdict matches the export single source on both classes.
    for row in on.values():
        marker = isinstance(row["quote"], dict) and row["quote"].get("redacted")
        assert bool(marker) == (not is_shareable(row["access_class"]))
    # Non-text fields survive redaction (the model still sees WHAT was withheld).
    assert on["work_priv"]["work_id"] == "work_priv"
    assert on["work_priv"]["span_ids"] == ["span_priv"]


# ===========================================================================
# Chunk 2 (trace plan) — `ask` persists envelope + trace (T6 reversal of the
# earlier "an answer is a query, not a run" v1 stance). Response shape and
# redaction are untouched; persistence is a side effect on local disk.
# ===========================================================================


def test_ask_persists_envelope_and_trace(run_mcp, isolated_home):
    """A default (`no_llm`) `ask` writes BOTH `{answer_id}.json` and
    `{answer_id}.trace.json` under the project's `answers/`, keyed by the
    response's `answer_id` — no `run_id` nesting for an ad-hoc ask (00 §5)."""
    from seedgraph.mcp.server import build_server
    from seedgraph.project import layout

    _seed_semantic("c2persist")
    server = build_server(root=isolated_home)

    result = _ask(
        run_mcp, server, {"project": "c2persist", "question": "parallel trends?"}
    )
    assert result.isError is False
    env = _tool_payload(result)
    answer_id = env["answer_id"]

    answers_dir = layout.project_dir("c2persist", isolated_home) / "answers"
    envelope_path = answers_dir / f"{answer_id}.json"
    trace_path = answers_dir / f"{answer_id}.trace.json"
    assert envelope_path.is_file()
    assert trace_path.is_file()

    saved_env = json.loads(envelope_path.read_text(encoding="utf-8"))
    assert saved_env["answer_id"] == answer_id
    saved_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert saved_trace["answer_id"] == answer_id


def test_ask_response_adds_persistence_status(run_mcp, isolated_home):
    """Delivery adds persistence alongside the existing envelope fields."""
    from seedgraph.answer.types import AnswerEnvelope
    from seedgraph.mcp.server import build_server

    _seed_semantic("c2shape")
    server = build_server(root=isolated_home)

    result = _ask(
        run_mcp, server, {"project": "c2shape", "question": "parallel trends?"}
    )
    assert result.isError is False
    payload = _tool_payload(result)
    assert set(payload) == set(AnswerEnvelope.model_fields) | {"persistence"}
    assert payload["persistence"]["status"] == "saved"


def test_ask_no_save_is_readonly_even_with_redaction(run_mcp, isolated_home):
    from _phase8_helpers import build_fixture_project
    from seedgraph.mcp.server import build_server

    h = build_fixture_project("readonly_mcp")
    h.engine.dispose()
    conn = sqlite3.connect(h.db_path)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    before = {p.relative_to(isolated_home): p.read_bytes()
              for p in isolated_home.rglob("*") if p.is_file()}
    server = build_server(root=isolated_home, redact_private=True)
    result = _ask(run_mcp, server, {
        "project": h.slug, "question": "secret identification trick", "no_save": True,
    })
    assert not result.isError, result.content
    payload = _tool_payload(result)
    assert payload["persistence"]["status"] == "skipped"
    assert payload["citations"]
    assert all("secret identification trick" not in str(c.get("quote"))
               for c in payload["citations"])
    assert before == {p.relative_to(isolated_home): p.read_bytes()
                      for p in isolated_home.rglob("*") if p.is_file()}


def test_ask_delivers_evidence_after_save_failure(run_mcp, isolated_home):
    from _phase8_helpers import build_fixture_project
    from seedgraph.mcp.server import build_server

    h = build_fixture_project("failed_save")
    (h.db_path.parent / "answers").write_text("blocked", encoding="utf-8")
    result = _ask(run_mcp, build_server(root=isolated_home), {
        "project": h.slug, "question": "across groups",
    })
    assert not result.isError, result.content
    payload = _tool_payload(result)
    assert payload["citations"]
    assert payload["persistence"]["status"] == "not_saved"


def test_ask_redaction_read_failure_returns_typed_error(run_mcp, isolated_home, monkeypatch):
    from _phase8_helpers import build_fixture_project
    from seedgraph.mcp.server import build_server

    h = build_fixture_project("redaction_failure")
    server = build_server(root=isolated_home, redact_private=True)

    def unavailable(_handle):
        raise sqlite3.DatabaseError("redaction source unavailable")

    monkeypatch.setattr(server.seedgraph_context, "project_conn", unavailable)
    result = _ask(run_mcp, server, {"project": h.slug, "question": "across groups"})
    assert result.isError
    assert _tool_error_body(result)["code"] == "retrieval_failed"


def test_ask_persists_unredacted_while_response_redacts(run_mcp, isolated_home):
    """`--redact-private` governs only the RESPONSE; the persisted envelope AND
    trace files are written UNREDACTED on disk — same posture as the envelope
    file the CLI writes (00 §8, C8)."""
    from seedgraph.mcp.server import build_server
    from seedgraph.project import layout

    _seed_semantic("c2redactpersist")
    server = build_server(root=isolated_home, redact_private=True)

    result = _ask(
        run_mcp, server,
        {"project": "c2redactpersist", "question": "parallel trends?"},
    )
    assert result.isError is False
    env = _tool_payload(result)
    answer_id = env["answer_id"]

    # Response: still redacted.
    citations = {c["work_id"]: c for c in env["citations"]}
    assert citations["work_priv"]["quote"] == {
        "redacted": True,
        "access_class": "user_supplied_private",
    }

    # Disk: unredacted. The saved envelope's citation carries the verbatim quote...
    answers_dir = layout.project_dir("c2redactpersist", isolated_home) / "answers"
    saved_env = json.loads(
        (answers_dir / f"{answer_id}.json").read_text(encoding="utf-8")
    )
    saved_citations = {c["work_id"]: c for c in saved_env["citations"]}
    assert saved_citations["work_priv"]["quote"] == "PRIVATE SPAN parallel trends"

    # ...and the trace's candidate table (which is never redacted, C8: no MCP
    # tool serves traces in v1) still carries the private text verbatim too.
    saved_trace = json.loads(
        (answers_dir / f"{answer_id}.trace.json").read_text(encoding="utf-8")
    )
    previews = [c["text_preview"] for c in saved_trace["candidates"]]
    assert any("PRIVATE" in p for p in previews)


# ===========================================================================
# Chunk 4 — the three §4.2 resources + the §4.3 `semantic_query` prompt.
#
# SDK shape (verified against the installed SDK, and NOT what earlier chunks
# would lead you to assume): a URI containing `{slug}` registers as a resource
# TEMPLATE, so `graph.json` is advertised by `list_resource_templates()` as a
# `uriTemplate` and is absent from `list_resources()`; the two parameterless
# docs resources are advertised by `list_resources()`. All three are read via
# `read_resource(<concrete AnyUrl>)`. Prompts: `list_prompts()` /
# `get_prompt(name, {arg: str})`, and a `str` return is wrapped into one message.
# ===========================================================================

from pathlib import Path  # noqa: E402

from pydantic import AnyUrl  # noqa: E402  (an mcp dependency)

_GRAPH_URI_TEMPLATE = "seedgraph://projects/{slug}/graph.json"
_QUICKSTART_URI = "seedgraph://docs/quickstart"
_HANDBOOK_URI = "seedgraph://docs/user-handbook"

# Derived from THIS file's location (tests/ sits at the repo root), deliberately
# independent of the resource's own resolver — otherwise the content-equality
# assertion below would be circular.
_REPO_DOCS = Path(__file__).resolve().parents[1] / "docs"


def _read_resource(run_mcp, server, uri: str) -> str:
    async def _c(client):
        return await client.read_resource(AnyUrl(uri))

    result = run_mcp(server, _c)
    return result.contents[0].text


def test_resources_and_prompt_registered(run_mcp, isolated_home):
    """All three §4.2 URIs are advertised (across BOTH listings) plus the prompt."""
    from seedgraph.mcp.server import build_server

    server = build_server(root=isolated_home)

    async def _c(client):
        return (
            await client.list_resources(),
            await client.list_resource_templates(),
            await client.list_prompts(),
        )

    listed, templates, prompts = run_mcp(server, _c)

    # The two static docs resources land in list_resources...
    assert {str(r.uri) for r in listed.resources} == {_QUICKSTART_URI, _HANDBOOK_URI}
    # ...and the parameterized graph export lands in list_resource_templates.
    assert _GRAPH_URI_TEMPLATE in {t.uriTemplate for t in templates.resourceTemplates}

    by_name = {p.name: p for p in prompts.prompts}
    assert "semantic_query" in by_name
    args = {
        a.name: bool(a.required) for a in (by_name["semantic_query"].arguments or [])
    }
    assert args == {"term": True, "project": False}


def test_docs_resources_return_file_content(run_mcp, isolated_home):
    """Both docs resources return the ACTUAL repo file text, byte for byte."""
    from seedgraph.mcp.server import build_server

    server = build_server(root=isolated_home)

    for uri, filename in (
        (_QUICKSTART_URI, "QUICKSTART.md"),
        (_HANDBOOK_URI, "USER_HANDBOOK.md"),
    ):
        expected = (_REPO_DOCS / filename).read_text(encoding="utf-8")
        assert expected.strip(), filename  # a blank fixture would prove nothing
        assert _read_resource(run_mcp, server, uri) == expected, filename


def _seed_export_project(slug: str):
    """A project with BOTH a shareable and a NON-shareable concept, a private
    `discusses` edge, and a materialized run — so the export guard has something
    real to withhold.

    Builds on `_seed_semantic` (open_access `concept::ols` + a public/private
    work-claim-span chain), then adds:
      * `concept::secret_method` — `user_supplied_private`, WITH a definition
        (definition is full-text-derived, decision 21);
      * a `discusses` edge work_priv -> that concept, `user_supplied_private`;
      * a `discusses` edge work_pub -> `concept::ols`, `open_access` (the
        positive control: a shareable overlay edge that MUST survive);
      * a deterministic citation edge under run 'R' + `build_and_export`, so
        `run.latest_run_id` resolves to 'R' (mirrors `_seed_edge_run`).
    """
    from seedgraph.citation.edges import write_edge
    from seedgraph.db.connection import connect_project_raw
    from seedgraph.graph.run_view import build_and_export

    h = _seed_semantic(slug)
    conn = connect_project_raw(h.db_path)
    try:
        conn.execute(
            "INSERT INTO concepts (concept_id, normalized_label, canonical_label, "
            "concept_type, definition, paper_frequency, weight, status, "
            "epistemic_type, access_class, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ("concept::secret_method", "secret method", "SECRET METHOD", "method",
             "A CONFIDENTIAL DEFINITION FROM PRIVATE FULL TEXT", 1, 0.9, "auto",
             "llm_extracted", "user_supplied_private", _now(), _now()),
        )
        for eid, work, concept, access in (
            ("pge_priv", "work_priv", "concept::secret_method", "user_supplied_private"),
            ("pge_pub", "work_pub", "concept::ols", "open_access"),
        ):
            conn.execute(
                "INSERT INTO project_graph_edges (edge_id, source_node_type, "
                "source_node_id, target_node_type, target_node_id, edge_type, "
                "epistemic_type, access_class, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (eid, "Work", work, "Concept", concept, "discusses",
                 "deterministic", access, _now()),
            )
        write_edge(
            conn, source="work_pub", target="work_priv",
            provenance="provider_reference", confidence=1.0, run_id="R",
        )
        conn.commit()
    finally:
        conn.close()
    build_and_export(h.slug, h.root, h.root, "R", open_world=False)
    return h


def _dicts_with_access_class(obj):
    """Every dict anywhere in ``obj`` that carries an ``access_class`` key."""
    found = []
    if isinstance(obj, dict):
        if "access_class" in obj:
            found.append(obj)
        for value in obj.values():
            found.extend(_dicts_with_access_class(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_dicts_with_access_class(item))
    return found


def test_graph_json_resource_export_guard_closed(run_mcp, isolated_home):
    """THE export-guard test: the `graph.json` resource leaks ZERO non-shareable
    content, judged by the guard's OWN vocabulary (`vocab.is_shareable`).

    Three axes, because "every access_class in the payload is shareable" is
    vacuous on its own — a guard that dropped EVERYTHING would also pass it:

    1. NEGATIVE CONTROL (the fixture really is leakable): exporting the same run
       with `allow_private=True` writes `exports/graph.private.json` containing
       the private concept and its definition. So the private content is
       export-REACHABLE, not merely sitting in a table the exporter never reads.
    2. GUARD CLOSED: in the resource payload, every dict carrying an
       `access_class` satisfies `is_shareable`, and the private concept id /
       definition text are absent entirely.
    3. POSITIVE CONTROL: the shareable concept and the shareable overlay edge ARE
       present — so an empty or broken export cannot pass vacuously.

    Work nodes carry no `access_class` and are correctly not gated: they are D8
    bibliographic metadata, never full-text-derived content.
    """
    from seedgraph.db.connection import open_project_db
    from seedgraph.mcp.server import build_server
    from seedgraph.semantic.export import export_graph
    from seedgraph.vocab import is_shareable

    _seed_export_project("c4guard")
    private_id = "concept::secret_method"
    private_definition = "A CONFIDENTIAL DEFINITION FROM PRIVATE FULL TEXT"

    # --- 1. the fixture genuinely contains export-reachable private content ---
    conn = open_project_db("c4guard", root=isolated_home)
    try:
        written = export_graph(
            conn, slug="c4guard", run_id="R", fmt="json",
            allow_private=True, root=isolated_home,
        )
    finally:
        conn.close()
    private_path = next(p for p in written if p.name == "graph.private.json")
    private_raw = private_path.read_text(encoding="utf-8")
    private_export = json.loads(private_raw)
    private_ids = {n["id"] for n in private_export["nodes"]}
    assert private_id in private_ids, "fixture has no export-reachable private concept"
    # Its LABEL leaks under --allow-private; that is the content the public
    # resource must withhold.
    assert "SECRET METHOD" in private_raw
    # Its DEFINITION does not leak even here: `_filtered_graph` nulls `definition`
    # for any non-shareable concept regardless of `allow_private` (decision 21 —
    # definition is full-text-derived). Pinned so the negative control states what
    # is actually reachable rather than overclaiming.
    assert private_definition not in private_raw
    # And a non-shareable dict really does exist in that unguarded view.
    assert any(
        not is_shareable(d["access_class"])
        for d in _dicts_with_access_class(private_export)
    )

    # --- 2/3. the resource payload, through the hard-false guard ---------------
    server = build_server(root=isolated_home)
    raw = _read_resource(run_mcp, server, "seedgraph://projects/c4guard/graph.json")
    payload = json.loads(raw)

    carriers = _dicts_with_access_class(payload)
    assert carriers, "no access_class-bearing rows at all — the walk proves nothing"
    offenders = [d for d in carriers if not is_shareable(d.get("access_class"))]
    assert offenders == [], f"non-shareable content in the public export: {offenders}"

    node_ids = {n["id"] for n in payload["nodes"]}
    assert private_id not in node_ids
    assert private_definition not in raw
    assert "SECRET METHOD" not in raw
    # Positive control: the shareable concept + shareable overlay edge survived.
    assert "concept::ols" in node_ids
    assert any(
        link["source"] == "work_pub" and link["target"] == "concept::ols"
        for link in payload["links"]
    )
    # And the private overlay edge did not.
    assert not any(link["target"] == private_id for link in payload["links"])


def test_graph_json_resource_has_no_allow_private_surface(isolated_home):
    """The resource exposes NO parameter that could set `allow_private=True`.

    Structural, not textual: the registered template's function signature is
    exactly `(slug)` — so there is no argument a client could pass to relax the
    default-deny export guard.
    """
    import inspect

    from seedgraph.mcp.server import build_server

    server = build_server(root=isolated_home)
    template = server._resource_manager._templates[_GRAPH_URI_TEMPLATE]
    assert list(inspect.signature(template.fn).parameters) == ["slug"]


def test_semantic_query_prompt_renders_with_and_without_project(run_mcp, isolated_home):
    """The §4.3 prompt renders both ways and carries the honesty rules verbatim.

    The pinned substrings are load-bearing POLICY, not prose: each is a rule the
    reconciled §4.3 requires the answer to carry. They are asserted as string
    presence so that weakening one into a paraphrase fails this test — a future
    edit to the prompt has to be deliberate.
    """
    from seedgraph.mcp.server import build_server

    server = build_server(root=isolated_home)

    def _render(args: dict) -> str:
        async def _c(client):
            return await client.get_prompt("semantic_query", args)

        result = run_mcp(server, _c)
        assert len(result.messages) == 1
        return result.messages[0].content.text

    without = _render({"term": "parallel trends"})
    with_project = _render({"term": "parallel trends", "project": "c4proj"})

    # The term is carried through both renderings; the project scopes only one.
    assert "parallel trends" in without and "parallel trends" in with_project
    assert "c4proj" in with_project
    assert "c4proj" not in without
    assert "project_list" in without  # no project -> resolve one, never guess a slug

    for text in (without, with_project):
        # --- workflow: overview-first, typed slice, whole-list is a FALLBACK ---
        assert "`concepts_overview` FIRST" in text
        assert "SMALL-CORPUS FALLBACK" in text
        assert "no_llm=true" in text
        assert "graph_analyze" in text
        # --- the six load-bearing honesty pins ---
        # recurrence vs weight
        assert "distinct-paper RECURRENCE (`paper_frequency`), not IDF" in text
        assert "IDF-style DISCRIMINATIVENESS, not importance" in text
        # judgment, not a computed score
        assert "JUDGMENT by meaning, NOT a computed" in text
        assert "similarity score" in text
        # semantic reach != citation reach
        assert "Semantic reach (a shared concept) is NOT citation reach" in text
        # ground quotes in spans, never labels
        assert "Ground quotes in EVIDENCE SPANS, never in concept labels" in text
        # a negative answer is valid
        assert "A NEGATIVE ANSWER IS A VALID ANSWER" in text
        # rank-and-page, never a pruned view
        assert "Rank-and-page, NEVER a pruned view" in text
        # --- the remaining §4.3 rules (thin evidence + provenance horizon) ---
        assert "`metadata_only` = THIN EVIDENCE" in text
        assert "the seeds span X-Y" in text
        assert "blind spot" in text


# ===========================================================================
# Chunk 5 — REAL-TRANSPORT smoke (opt-in, marked `mcp_client`).
#
# Every test above this line drives the SDK's in-memory client<->server
# transport: no process is spawned, so plan 01 risk 6 ("Windows stdio quirks —
# CRLF/encoding on spawned stdio pipes") was never actually exercised. This test
# closes that gap on the path a real host (Claude Code / Desktop / Inspector)
# uses: spawn `python -m seedgraph mcp serve` and speak MCP over its stdin and
# stdout.
#
# It is DESELECTED BY DEFAULT (`addopts` carries `and not mcp_client`) because it
# spawns a subprocess, which the default suite's offline/keyless/no-subprocess
# posture excludes. Run it with `pytest -m mcp_client`.
#
# As-built findings from the chunk-5 walk on Windows 11 / CPython 3.13 / SDK
# 1.28.1 — real stdio worked with NO code change:
#   * Frames are terminated `\r\n`, not bare `\n` (the child's stdout is a
#     Windows text-mode pipe). Exactly ONE CR per frame, in the terminator; no
#     CR is ever embedded in a message. Both official SDKs cope (the TS client
#     strips a trailing `\r`; `json.loads` ignores trailing whitespace), so this
#     is benign and is deliberately NOT "fixed" — forcing a binary/LF stdout
#     would be an unforced change to a working transport.
#   * UTF-8 survives both directions: the em-dashes in the tool descriptions
#     arrive intact, and non-ASCII tool ARGUMENTS round-trip.
# ===========================================================================

_STDIO_TIMEOUT_S = 60.0

# The complete tool surface includes bounded reading and explicit extraction.
# Asserted by EQUALITY here (the in-memory tests above assert subsets per chunk)
# so that a tool silently added to — or dropped from — the surface fails the one
# test that speaks the transport a real host uses.
_EXPECTED_V1_TOOLS = {
    "version",
    # chunk 1 — project / corpus reads
    "project_list",
    "project_dashboard",
    "list_documents",
    "corpus_status",
    "review_list",
    "run_list",
    "run_events",
    "lens_results",
    "lens_staleness",
    # chunk 2 — concept / search / graph
    "concepts_overview",
    "concepts_list",
    "concept_show",
    "search_spans",
    "graph_analyze",
    # chunk 3
    "ask",
    "work_read",
    "extraction_prepare",
    "extraction_next",
    "extraction_submit",
    "extraction_status",
}


@pytest.mark.mcp_client
def test_real_stdio_transport_end_to_end(isolated_home):
    """Spawn the server as a subprocess and drive it over real stdio pipes.

    Uses ``sys.executable`` — the interpreter running the suite, which the
    module-level ``importorskip('mcp')`` already proved has the extra — so the
    test stays correct if the repo or venv moves. Home is pinned with the
    ``--root`` CLI flag rather than an env var: the SDK's default child
    environment is a filtered allow-list that would drop ``SEEDGRAPH_HOME``.

    ``anyio.fail_after`` bounds the whole exchange, so a framing/buffering
    regression fails as a timeout assertion instead of hanging the suite.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "seedgraph", "mcp", "serve", "--root", str(isolated_home)],
    )

    async def _drive():
        with anyio.fail_after(_STDIO_TIMEOUT_S):
            # errlog -> DEVNULL: the server's INFO request log is noise here, and
            # an unread stderr pipe is a deadlock risk on a chatty run.
            with open(os.devnull, "w") as devnull:
                async with stdio_client(params, errlog=devnull) as (read, write):
                    async with ClientSession(read, write) as client:
                        init = await client.initialize()
                        tools = await client.list_tools()
                        called = await client.call_tool("version", {})
                        resources = await client.list_resources()
                        templates = await client.list_resource_templates()
                        prompts = await client.list_prompts()
                        return init, tools, called, resources, templates, prompts

    init, tools, called, resources, templates, prompts = anyio.run(_drive)

    # --- the handshake actually completed over the pipe ---
    assert init.serverInfo.name == "seedgraph"

    # --- the full v1 tool surface is advertised over real stdio ---
    assert {t.name for t in tools.tools} == _EXPECTED_V1_TOOLS

    # --- a real round-trip call returns a correctly framed, parseable result ---
    assert called.isError is False
    assert called.structuredContent == {
        "seedgraph": SEEDGRAPH_VERSION,
        "sqlite": sqlite3.sqlite_version,
    }

    # --- §4.2 shape: two plain resources; `graph.json` is a TEMPLATE ({slug}) ---
    assert {str(r.uri) for r in resources.resources} == {
        "seedgraph://docs/quickstart",
        "seedgraph://docs/user-handbook",
    }
    assert [t.uriTemplate for t in templates.resourceTemplates] == [
        _GRAPH_URI_TEMPLATE
    ]

    # --- §4.3: the single prompt ---
    assert [p.name for p in prompts.prompts] == ["semantic_query"]


@pytest.mark.mcp_client
def test_real_stdio_preserves_utf8_and_structured_errors(isolated_home):
    """Non-ASCII survives the Windows pipe, and the M9 error body arrives intact.

    Two things a consumer depends on, neither observable in-memory:

    1. **Encoding.** Tool descriptions contain U+2014 em-dashes; a mis-encoded
       pipe would mojibake or fail to decode them.
    2. **Error shape (M9).** FastMCP RE-WRAPS a ``ToolError``, so the structured
       ``{code, message, detail}`` body arrives PREFIXED with
       ``Error executing tool <name>: ``. A client branching on ``code`` must
       therefore strip to the first ``{`` — pinned here so the documented client
       contract stays true.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "seedgraph", "mcp", "serve", "--root", str(isolated_home)],
    )

    async def _drive():
        with anyio.fail_after(_STDIO_TIMEOUT_S):
            with open(os.devnull, "w") as devnull:
                async with stdio_client(params, errlog=devnull) as (read, write):
                    async with ClientSession(read, write) as client:
                        await client.initialize()
                        tools = await client.list_tools()
                        # A non-ASCII ARGUMENT exercises the read direction.
                        err = await client.call_tool(
                            "project_dashboard", {"project": "—no-such-café"}
                        )
                        return tools, err

    tools, err = anyio.run(_drive)

    # 1. write direction: the em-dash in a docstring survived the pipe verbatim.
    version_tool = next(t for t in tools.tools if t.name == "version")
    assert "—" in version_tool.description

    # 2. the M9 body is present, prefixed, and recoverable by strip-to-first-`{`.
    assert err.isError is True
    text = err.content[0].text
    assert text.startswith("Error executing tool project_dashboard: ")
    body = json.loads(text[text.index("{") :])
    assert body["code"] == "project_not_found"
