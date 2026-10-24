"""v3 MCP read tools over an invented library (design §7.2): one in-memory call per tool,
`graph` parameters, and the agent audience (hidden notes stay hidden)."""

from __future__ import annotations

import pytest
from v3_mcp_support import GRAPH, call, err, ok, home, seeded, server  # noqa: F401 (fixtures)
from work.routine_support import ASSUMPTION_QUOTE, INTRO_QUOTE, METHOD_QUOTE

from seedgraph.lenses.schema import LensDefinition
from seedgraph.library import attention
from seedgraph.library import store as libstore
from seedgraph.work import routine, tasks_lens
from seedgraph.work.claims import WorkerRef

LENS = {
    "lens_id": "invented_conditions_v1",
    "name": "Invented conditions",
    "object_type": "assumption",
    "positive_anchors": ["anticipate"],
    "negative_anchors": ["related literature"],
    "output_schema": {
        "condition_text_verbatim": "string",
        "normalized_condition_label": "string",
        "condition_type": {"type": "array", "allowed_values": ["identification", "other"]},
        "section": "string",
        "stated_or_inferred": {"allowed_values": ["stated", "inferred"]},
        "evidence_span_ids": {"type": "array"},
        "confidence": "number",
        "notes": "string",
    },
}


def _hide(home, paper_id):
    lib = libstore.open_library_at(home / "library")
    try:
        attention.raise_item(lib, "pdf_title_miss", {"paper_id": paper_id}, paper_id=paper_id,
                             detail="the PDF title does not match", hides_notes=True)
    finally:
        lib.close()


def _lens_run(home, seeded):
    """Register the invented lens and save a found record for p1 and p2."""
    lib = libstore.open_library_at(home / "library")
    try:
        tasks_lens.register_lens(lib, LensDefinition.from_dict(LENS), status="active")
        view = routine.create_batch(lib, f"lens:{LENS['lens_id']}",
                                    paper_ids=[seeded.p1.paper_id, seeded.p2.paper_id])
        rec = {"condition_text_verbatim": ASSUMPTION_QUOTE, "normalized_condition_label": "no anticipation",
               "condition_type": ["identification"], "section": "Method", "stated_or_inferred": "stated",
               "evidence_span_ids": [], "confidence": 0.7}
        for _ in range(2):
            msg = routine.claim_next(lib, view.batch_id, WorkerRef("interactive", "lens")).message
            assert routine.submit(lib, msg["packet_id"], msg["claim_id"],
                                  note={"records": [rec]}).paper_status == "done"
    finally:
        lib.close()


# --- graphs ------------------------------------------------------------------------------------

def test_graph_list(server, seeded):
    (g,) = ok(call(server, "graph_list"))["graphs"]
    assert g == {"graph": GRAPH, "display_name": "Invented gadgets", "created_at": g["created_at"],
                 "members": 2, "frontier": 1, "seeds": 1}


def test_graph_show(server, seeded):
    body = ok(call(server, "graph_show", {"graph": GRAPH}))
    assert body["counts"] == {"member": 2, "frontier": 1, "noted_members": 1}
    assert [s["paper_id"] for s in body["seeds"]] == [seeded.p1.paper_id]
    assert [p["paper_id"] for p in body["papers_needing_notes"]] == [seeded.p2.paper_id]
    assert body["lenses"] == []


def test_unknown_graph_is_graph_not_found(server, seeded):
    for name, args in (("graph_show", {"graph": "no_such_graph"}),
                       ("paper_list", {"graph": "no_such_graph"}),
                       ("search", {"query": "gadget", "graph": "no_such_graph"})):
        assert err(call(server, name, args))["code"] == "graph_not_found", name


# --- papers ------------------------------------------------------------------------------------

def test_paper_list_filters(server, seeded):
    every = ok(call(server, "paper_list"))
    assert every["total"] == 4
    in_graph = ok(call(server, "paper_list", {"graph": GRAPH}))
    assert [p["paper_id"] for p in in_graph["papers"]] == [seeded.p1.paper_id, seeded.p2.paper_id,
                                                           seeded.p3.paper_id]
    assert in_graph["papers"][0]["is_seed"] is True and in_graph["papers"][0]["has_current_note"] is True
    frontier = ok(call(server, "paper_list", {"graph": GRAPH, "status": "frontier"}))
    assert [p["paper_id"] for p in frontier["papers"]] == [seeded.p3.paper_id]
    needing = ok(call(server, "paper_list", {"graph": GRAPH, "status": "member", "needs": "note"}))
    assert [p["paper_id"] for p in needing["papers"]] == [seeded.p2.paper_id]
    titled = ok(call(server, "paper_list", {"query": "SUBSIDIES AND"}))
    assert [p["paper_id"] for p in titled["papers"]] == [seeded.p1.paper_id]
    page = ok(call(server, "paper_list", {"limit": 3, "offset": 0}))
    assert len(page["papers"]) == 3 and page["next_offset"] == 3


def test_paper_list_refuses_member_status_without_graph(server, seeded):
    assert err(call(server, "paper_list", {"status": "member"}))["code"] == "invalid_argument"
    assert err(call(server, "paper_list", {"needs": "everything"}))["code"] == "invalid_argument"


def test_paper_show_note_claims_and_pages(server, seeded):
    body = ok(call(server, "paper_show", {"paper_id": seeded.p1.paper_id}))
    assert body["title"] == "Invented gadget subsidies and μ"
    assert body["notes_hidden"] is False
    note = body["current_note"]
    assert note["note_id"] == seeded.note_id
    labels = {c["label"] for c in note["claims"] if c["status"] == "found"}
    assert {"two-step estimator", "no anticipation", "output rises"} <= labels
    found = [c for c in note["claims"] if c["status"] == "found"]
    assert all(c["anchors"] for c in found)
    assert [s["heading"] for s in body["sections"]] == ["Introduction", "Method", "Results", "References"]


def test_paper_show_unknown_paper(server, seeded):
    assert err(call(server, "paper_show", {"paper_id": "lp_missing"}))["code"] == "paper_not_found"


def test_paper_read_ranges_and_sections(server, seeded):
    text = seeded.p1.text
    first = ok(call(server, "paper_read", {"paper_id": seeded.p1.paper_id, "max_chars": 40}))
    assert first["text"] == text[:40] and first["next_start"] == 40 and first["total_chars"] == len(text)
    mid = ok(call(server, "paper_read", {"paper_id": seeded.p1.paper_id, "start": 10, "end": 30}))
    assert mid["text"] == text[10:30] and mid["next_start"] is None
    sec = ok(call(server, "paper_read", {"paper_id": seeded.p1.paper_id, "section": "method"}))
    assert sec["text"].startswith("## Method") and METHOD_QUOTE in sec["text"]
    assert sec["section"]["heading"] == "Method"


def test_paper_read_refusals(server, seeded):
    pid = seeded.p1.paper_id
    assert err(call(server, "paper_read", {"paper_id": pid, "section": "Nowhere"}))["code"] == "invalid_argument"
    assert err(call(server, "paper_read", {"paper_id": pid, "start": 9, "end": 3}))["code"] == "invalid_argument"
    assert err(call(server, "paper_read", {"paper_id": pid, "max_chars": 10**6}))["code"] == "invalid_argument"


# --- search ------------------------------------------------------------------------------------

def test_search_spans_claims_notes(server, seeded):
    claims = ok(call(server, "search", {"query": "two-step estimator", "kind": "claims"}))
    assert {h["paper_ids"][0] for h in claims["hits"]} == {seeded.p1.paper_id}
    notes = ok(call(server, "search", {"query": "gadget subsidy effect", "kind": "notes"}))
    assert [h["note_id"] for h in notes["hits"]] == [seeded.note_id]
    spans = ok(call(server, "search", {"query": INTRO_QUOTE[:30], "kind": "spans", "graph": GRAPH}))
    assert all(set(h["paper_ids"]) <= {seeded.p1.paper_id, seeded.p2.paper_id} for h in spans["hits"])
    assert err(call(server, "search", {"query": "x", "kind": "everything"}))["code"] == "invalid_argument"


# --- lenses ------------------------------------------------------------------------------------

def test_lens_results_graph_scope(server, seeded, home):
    _lens_run(home, seeded)
    body = ok(call(server, "lens_results", {"lens_id": LENS["lens_id"], "graph": GRAPH}))
    assert {r["paper_id"] for r in body["results"]} == {seeded.p1.paper_id, seeded.p2.paper_id}
    assert body["papers_without_current_results"] == []
    assert body["lens_chosen_for_graph"] is False
    assert err(call(server, "lens_results", {"lens_id": "no_lens"}))["code"] == "work_refused"


# --- agent audience ----------------------------------------------------------------------------

def test_hidden_notes_stay_hidden_everywhere(server, seeded, home):
    _lens_run(home, seeded)
    _hide(home, seeded.p1.paper_id)
    pid = seeded.p1.paper_id
    show = ok(call(server, "paper_show", {"paper_id": pid}))
    assert show["notes_hidden"] is True and show["current_note"] is None
    assert [a["kind"] for a in show["attention"]] == ["pdf_title_miss"]
    assert ok(call(server, "search", {"query": "two-step estimator", "kind": "claims"}))["hits"] == []
    assert ok(call(server, "search", {"query": "gadget subsidy effect", "kind": "notes"}))["hits"] == []
    lens = ok(call(server, "lens_results", {"lens_id": LENS["lens_id"]}))
    assert pid not in {r["paper_id"] for r in lens["results"]}
    # the paper text itself is never hidden
    assert ok(call(server, "paper_read", {"paper_id": pid, "max_chars": 10}))["text"]


@pytest.mark.parametrize("name", ["graph_show", "paper_show"])
def test_missing_argument_is_a_tool_error(server, seeded, name):
    assert call(server, name, {}).isError is True
