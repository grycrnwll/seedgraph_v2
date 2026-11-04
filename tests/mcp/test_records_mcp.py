"""Stage 5 slice R3 (design §3.2, §3.4, §7): the session log in the MCP read tools, the
``record_answer`` tool, the reminder line, the connection close hook and the ``dirty`` rule,
over the in-memory MCP client. Invented text only."""

from __future__ import annotations

import json
import sqlite3

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from v3_mcp_support import GRAPH, call, err, list_tools, ok, home, seeded, server  # noqa: F401 (fixtures)
from work.routine_support import ASSUMPTION_QUOTE

from seedgraph.library import store as libstore
from seedgraph.mcp.server import RECORD_REMINDER

LOGGED = {"search", "paper_list", "paper_show", "paper_read", "lens_results", "concept_search", "concept_show",
          "literatures", "concept_hubs"}
SEARCH_KIND = {"search", "paper_list", "lens_results", "concept_search", "literatures", "concept_hubs"}
NAME = "parallel-trends-1432"


def _lib(home):
    return libstore.open_library_at(home / "library")


def events(home) -> list[dict]:
    lib = _lib(home)
    try:
        rows = lib.conn.execute("SELECT e.*, s.name AS session_name, s.key_kind FROM traversal_events e "
                                "JOIN record_sessions s ON s.session_id = e.session_id ORDER BY seq").fetchall()
        return [{**dict(r), "args": json.loads(r["args_json"]), "items": json.loads(r["items_json"])} for r in rows]
    finally:
        lib.close()


def query(home, sql: str, *args):
    lib = _lib(home)
    try:
        return lib.conn.execute(sql, args).fetchall()
    finally:
        lib.close()


def set_clean(home) -> None:
    lib = _lib(home)
    try:
        lib.conn.execute("UPDATE library_meta SET value = '0' WHERE key = 'dirty'")
        lib.conn.commit()
        assert not libstore.is_dirty(lib.conn)
    finally:
        lib.close()


def is_dirty(home) -> bool:
    lib = _lib(home)
    try:
        return libstore.is_dirty(lib.conn)
    finally:
        lib.close()


def call_with_meta(server, name: str, args: dict, meta: dict):
    async def _driver():
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            return await client.call_tool(name, args, meta=meta)

    return anyio.run(_driver)


def span_hit(server, session=NAME) -> dict:
    body = ok(call(server, "search", {"query": "anticipate", "session": session}))
    return next(h for h in body["hits"] if h["kind"] == "spans")


# --- the tool surface ---------------------------------------------------------------------------

def test_record_answer_is_a_mutation_and_logged_tools_take_session(server):
    tools = {t.name: t for t in list_tools(server)}
    assert tools["record_answer"].annotations.readOnlyHint is False
    props = tools["record_answer"].inputSchema["properties"]
    assert {"question", "answer", "evidence", "judgement", "session"} <= set(props)
    assert set(tools["record_answer"].inputSchema["required"]) == {"question", "answer", "evidence"}
    for name, tool in tools.items():
        props = tool.inputSchema.get("properties", {})
        assert "mcp_ctx" not in props, name
        assert ("session" in props) == (name in LOGGED | {"record_answer"}), name
        if name in LOGGED:
            assert tool.annotations.readOnlyHint is True, name


# --- one event per logged call ------------------------------------------------------------------

def test_search_logs_one_event_with_hits_in_rank_order(server, seeded, home):
    body = ok(call(server, "search", {"query": "anticipate", "session": NAME}))
    assert body["session"] == NAME and body["record_reminder"] == RECORD_REMINDER
    (e,) = events(home)
    assert (e["tool"], e["kind"], e["session_name"], e["key_kind"]) == ("search", "search", NAME, "named")
    assert e["args"] == {"query": "anticipate", "kind": "spans", "limit": 20}
    assert [(i["kind"], i["id"]) for i in e["items"]] == [("span", h["span_id"]) for h in body["hits"]]
    assert e["items"][0]["paper_ids"] == body["hits"][0]["paper_ids"]
    assert e["items"][0]["preview"] == body["hits"][0]["text"][:160]
    assert e["connection_id"] == server.seedgraph_context.worker_id


def test_search_names_graph_only_when_the_agent_did_and_maps_claim_hits(server, seeded, home):
    ok(call(server, "search", {"query": "anticipation", "kind": "claims", "graph": GRAPH}))
    ok(call(server, "search", {"query": "gadget", "kind": "notes"}))
    claims, notes = events(home)
    assert claims["args"]["graph"] == GRAPH and "graph" not in notes["args"]
    assert claims["items"] and {i["kind"] for i in claims["items"]} == {"claim"}
    assert all(i["id"] and i["paper_ids"] == [seeded.p1.paper_id] for i in claims["items"])
    assert notes["items"] and {i["kind"] for i in notes["items"]} == {"note"}
    assert notes["items"][0]["id"] == seeded.note_id


def test_paper_show_logs_returned_claims_with_their_anchors(server, seeded, home):
    body = ok(call(server, "paper_show", {"paper_id": seeded.p1.paper_id, "session": NAME}))
    assert body["session"] == NAME and "record_reminder" not in body
    (e,) = events(home)
    assert (e["tool"], e["kind"], e["args"]) == ("paper_show", "open_paper", {"paper_id": seeded.p1.paper_id})
    claims = body["current_note"]["claims"]
    assert e["items"] == [{"claim_id": c["claim_id"], "span_ids": [a["span_id"] for a in c["anchors"]]}
                          for c in claims]


def test_paper_read_logs_the_range_actually_returned(server, seeded, home):
    body = ok(call(server, "paper_read", {"paper_id": seeded.p1.paper_id, "start": 5, "end": 10**6,
                                          "max_chars": 40}))
    assert body["end"] == 45 and "record_reminder" not in body
    (e,) = events(home)
    assert e["kind"] == "open_text"
    assert e["items"] == [{"paper_id": seeded.p1.paper_id, "markdown_id": body["markdown_id"], "start": 5,
                           "end": 45, "section_id": None}]
    assert e["args"] == {"paper_id": seeded.p1.paper_id, "start": 5, "end": 10**6, "max_chars": 40}


def test_paper_list_logs_only_with_a_query(server, seeded, home):
    plain = ok(call(server, "paper_list", {"graph": GRAPH}))
    assert "session" not in plain and "record_reminder" not in plain
    assert events(home) == []
    body = ok(call(server, "paper_list", {"query": "gadget subsidies", "session": NAME}))
    assert body["record_reminder"] == RECORD_REMINDER and body["session"] == NAME
    (e,) = events(home)
    assert (e["tool"], e["kind"]) == ("paper_list", "search")
    assert e["items"] == [{"kind": "paper", "id": seeded.p1.paper_id}]


def test_unlogged_tools_and_errors_write_no_event(server, seeded, home):
    ok(call(server, "version"))
    ok(call(server, "graph_list"))
    ok(call(server, "graph_show", {"graph": GRAPH}))
    ok(call(server, "batch_list"))
    ok(call(server, "batch_status", {"batch_id": seeded.noted_batch}))
    assert err(call(server, "paper_show", {"paper_id": "lp_nope", "session": NAME}))["code"] == "paper_not_found"
    assert err(call(server, "search", {"query": "x", "kind": "nope"}))["code"] == "invalid_argument"
    assert events(home) == []
    assert query(home, "SELECT COUNT(*) FROM traversal_records")[0][0] == 0


def test_an_invalid_session_is_refused_before_the_call_runs(server, seeded, home):
    for tool, args in (("search", {"query": "anticipate"}), ("paper_show", {"paper_id": seeded.p1.paper_id}),
                       ("record_answer", {"question": "q", "answer": "a",
                                          "evidence": [{"paper_id": seeded.p1.paper_id}]})):
        body = err(call(server, tool, {**args, "session": "Bad Name!"}))
        assert body["code"] == "invalid_session", tool
        assert "parallel-trends-1432" in body["next_action"]
    assert events(home) == []
    assert query(home, "SELECT COUNT(*) FROM traversal_records")[0][0] == 0


def test_an_unnamed_session_is_named_by_its_start_time_and_echoed(server, seeded, home):
    first = ok(call(server, "search", {"query": "anticipate"}))
    second = ok(call(server, "paper_show", {"paper_id": seeded.p1.paper_id}))
    a, b = events(home)
    assert a["key_kind"] == "connection" and a["session_id"] == b["session_id"]
    assert first["session"] == second["session"] == a["session_name"]
    assert len(a["session_name"]) >= 16 and a["session_name"][4] == "-" and a["session_name"][13] == ":"


def test_helpers_log_into_the_parents_open_record(server, seeded, home):
    ok(call(server, "search", {"query": "anticipate", "session": NAME}))
    body = ok(call(server, "paper_show", {"paper_id": seeded.p1.paper_id, "session": f"{NAME}/a"}))
    assert body["session"] == f"{NAME}/a"
    a, b = events(home)
    assert a["record_id"] == b["record_id"] and a["session_id"] != b["session_id"]


def test_meta_keys_are_kept_never_the_values(server, seeded, home):
    ok(call_with_meta(server, "search", {"query": "anticipate"}, {"probe_marker": "secret-value", "x": 1}))
    (row,) = query(home, "SELECT meta_keys_json FROM record_connections")
    assert set(json.loads(row[0])) >= {"probe_marker", "x"}
    assert "secret-value" not in row[0]


def test_a_failed_log_write_still_answers_and_is_counted(server, seeded, home, monkeypatch):
    from seedgraph.records import log

    real = log._write_event

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(log, "_write_event", locked)
    body = ok(call(server, "search", {"query": "anticipate", "session": NAME}))
    assert body["hits"] and body["session"] == NAME
    unnamed = ok(call(server, "paper_show", {"paper_id": seeded.p1.paper_id}))
    assert unnamed["paper_id"] == seeded.p1.paper_id and unnamed["session"] is None
    assert events(home) == []
    monkeypatch.setattr(log, "_write_event", real)
    ok(call(server, "search", {"query": "anticipate", "session": NAME}))
    (e,) = events(home)
    (unlogged,) = query(home, "SELECT unlogged_calls FROM traversal_records WHERE record_id = ?", e["record_id"])[0]
    assert unlogged == 2


def test_a_fault_while_building_items_still_answers(server, seeded, home, monkeypatch):
    from seedgraph.mcp import tools_library

    def broken(reply):
        raise KeyError("hits")

    monkeypatch.setattr(tools_library, "_search_items", broken)
    assert ok(call(server, "search", {"query": "anticipate"}))["hits"]
    assert events(home) == []


# --- record_answer ------------------------------------------------------------------------------

def test_record_answer_round_trip(server, seeded, home):
    hit = span_hit(server)
    ok(call(server, "paper_read", {"paper_id": seeded.p1.paper_id, "start": hit["start_char"],
                                   "end": hit["end_char"], "session": NAME}))
    body = ok(call(server, "record_answer", {
        "question": "Do workshops anticipate the programme?", "answer": "The paper says they do not.",
        "evidence": [{"span_id": hit["span_id"], "why": "the assumption"}], "judgement": "Plausible.",
        "session": NAME}))
    assert set(body) == {"record_id", "session", "counts", "flags", "page", "next_action"}
    assert body["session"] == NAME and body["flags"] == [] and body["next_action"] is None
    assert body["page"] == f"/records/{body['record_id']}"
    assert body["counts"]["searches"] == 1 and body["counts"]["cited"] == 1
    (row,) = query(home, "SELECT state, question, answer, judgement, evidence_json FROM traversal_records "
                         "WHERE record_id = ?", body["record_id"])
    assert row["state"] == "answered" and row["judgement"] == "Plausible."
    (frozen,) = json.loads(row["evidence_json"])
    assert frozen["quote"] == ASSUMPTION_QUOTE and frozen["why"] == "the assumption"
    assert {e["record_id"] for e in events(home)} == {body["record_id"]}
    # the next call opens the next record; record_answer itself logs no event
    ok(call(server, "search", {"query": "anticipate", "session": NAME}))
    assert len(events(home)) == 3 and events(home)[-1]["record_id"] != body["record_id"]


def test_a_search_hit_cited_unread_is_flagged_and_reading_it_clears_the_flag(server, seeded, home):
    """R2's suggested end-to-end check: real search, cite a hit -> flagged; real
    paper_read over the span, cite again -> no flag."""
    hit = span_hit(server)
    first = ok(call(server, "record_answer", {"question": "q", "answer": "a", "evidence": [{"span_id": hit["span_id"]}],
                                              "session": NAME}))
    (flag,) = first["flags"]
    assert flag["flag"] == "cited without being read (seen in search results only)"
    assert flag["item"] == {"kind": "span", "id": hit["span_id"]}
    assert first["next_action"] and "read them" in first["next_action"]
    ok(call(server, "paper_read", {"paper_id": hit["paper_ids"][0], "start": hit["start_char"],
                                   "end": hit["end_char"], "session": NAME}))
    second = ok(call(server, "record_answer", {"question": "q", "answer": "a",
                                               "evidence": [{"span_id": hit["span_id"]}], "session": NAME}))
    assert second["flags"] == [] and second["next_action"] is None
    assert second["record_id"] != first["record_id"]


def test_a_paper_show_anchor_alone_is_not_read(server, seeded, home):
    body = ok(call(server, "paper_show", {"paper_id": seeded.p1.paper_id, "session": NAME}))
    anchor = next(a["span_id"] for c in body["current_note"]["claims"] for a in c["anchors"])
    reply = ok(call(server, "record_answer", {"question": "q", "answer": "a", "evidence": [{"span_id": anchor}],
                                              "session": NAME}))
    assert [f["flag"] for f in reply["flags"]] == ["cited without being read"]


def test_record_answer_refusals_keep_their_code_and_next_action(server, seeded, home):
    body = err(call(server, "record_answer", {"question": "q", "answer": "a",
                                              "evidence": [{"span_id": "sp_nope"}, {"claim_id": "cl_nope"}]}))
    assert body["code"] == "unknown_evidence"
    assert body["detail"]["unknown"] == ["sp_nope", "cl_nope"]
    assert "record again" in body["next_action"]
    body = err(call(server, "record_answer", {"question": "q", "answer": "a",
                                              "evidence": [{"span_id": "a", "claim_id": "b"}]}))
    assert body["code"] == "invalid_argument" and "exactly one of span_id" in body["next_action"]
    body = err(call(server, "record_answer", {"question": "", "answer": "a",
                                              "evidence": [{"paper_id": seeded.p1.paper_id}]}))
    assert body["code"] == "invalid_argument" and "question" in body["next_action"]
    assert query(home, "SELECT COUNT(*) FROM traversal_records")[0][0] == 0


# --- dirty, and the connection close ------------------------------------------------------------

def test_logging_and_recording_leave_dirty_unchanged(server, seeded, home):
    set_clean(home)
    hit = span_hit(server)
    ok(call(server, "paper_read", {"paper_id": seeded.p1.paper_id, "start": 0, "end": 50, "session": NAME}))
    ok(call(server, "paper_show", {"paper_id": seeded.p1.paper_id}))
    ok(call(server, "record_answer", {"question": "q", "answer": "a", "evidence": [{"span_id": hit["span_id"]}],
                                      "session": NAME}))
    from seedgraph.mcp.server import close_connection

    close_connection(server.seedgraph_context)
    assert len(events(home)) == 3
    assert not is_dirty(home)


def test_an_ordinary_write_still_sets_dirty(home, seeded):
    set_clean(home)
    lib = _lib(home)
    try:
        with lib.write(dirty=False) as conn:
            with lib.write() as inner:  # a dirty write nested in a clean one still flags
                inner.execute("UPDATE papers SET title = title")
            conn.execute("UPDATE papers SET title = title")
        assert libstore.is_dirty(lib.conn)
    finally:
        lib.close()


def test_close_connection_marks_the_connection_closed(server, seeded, home):
    from seedgraph.mcp.server import close_connection

    ok(call(server, "search", {"query": "anticipate"}))
    close_connection(server.seedgraph_context)
    (row,) = query(home, "SELECT connection_id, closed_at FROM record_connections")
    assert row[0] == server.seedgraph_context.worker_id and row[1]


def test_close_connection_never_creates_a_library(server, home):
    from seedgraph.mcp.server import close_connection

    close_connection(server.seedgraph_context)
    assert not (home / "library").exists()


def test_main_marks_the_connection_closed_when_stdin_ends(seeded, home, monkeypatch):
    from mcp.server.fastmcp import FastMCP

    from seedgraph.mcp import server as server_mod

    seen = {}

    def fake_run(self, transport="stdio"):
        seen["transport"] = transport
        seen["worker"] = self.seedgraph_context.worker_id
        ok(call(self, "search", {"query": "anticipate"}))

    monkeypatch.setattr(FastMCP, "run", fake_run)
    server_mod.main(home=home)
    assert seen["transport"] == "stdio"
    (row,) = query(home, "SELECT connection_id, closed_at FROM record_connections")
    assert row[0] == seen["worker"] and row[1]


@pytest.mark.parametrize("tool", sorted(SEARCH_KIND - {"concept_search", "literatures", "concept_hubs"}))
def test_reminder_is_on_search_kind_replies(server, seeded, home, tool, monkeypatch):
    args = {"search": {"query": "anticipate"}, "paper_list": {"query": "gadget"},
            "lens_results": {"lens_id": "invented_lens_v1"}}[tool]
    if tool == "lens_results":
        from seedgraph.mcp import tools_library

        monkeypatch.setattr(tools_library, "_lens_results",
                            lambda lib, lens_id, graph, status: {"lens_id": lens_id, "graph": graph, "status": status,
                                                                 "results": [{"paper_id": seeded.p1.paper_id},
                                                                             {"paper_id": seeded.p1.paper_id}]})
    body = ok(call(server, tool, {**args, "session": NAME}))
    assert body["record_reminder"] == RECORD_REMINDER
    (e,) = events(home)
    assert e["kind"] == "search"
    if tool == "lens_results":
        assert e["items"] == [{"kind": "paper", "id": seeded.p1.paper_id}]
        assert e["args"] == {"lens_id": "invented_lens_v1", "status": "found"}
