"""Stage 5 slice R3 (design §3.2): the concept tools log one event each, with the item shapes
records/path.py reads, and a quote returned by concept_show counts as read for
record_answer. Reuses the stage 4 invented concept world. Invented text only."""

from __future__ import annotations

import json

from test_v3_server_concepts import GRAPH, p, read_resource, world  # noqa: F401 (fixture)
from v3_mcp_support import call, ok, home, server  # noqa: F401 (fixtures)

from seedgraph.library import store as libstore
from seedgraph.mcp.server import RECORD_REMINDER

NAME = "sprocket-rules-0915"


def events(home) -> list[dict]:
    lib = libstore.open_library_at(home / "library")
    try:
        rows = lib.conn.execute("SELECT * FROM traversal_events ORDER BY seq").fetchall()
        return [{**dict(r), "args": json.loads(r["args_json"]), "items": json.loads(r["items_json"])} for r in rows]
    finally:
        lib.close()


def test_concept_search_logs_handles_with_example_claims(server, world, home):
    body = ok(call(server, "concept_search", {"query": "sprocket weights balance", "session": NAME}))
    assert body["session"] == NAME and body["record_reminder"] == RECORD_REMINDER
    (e,) = events(home)
    assert (e["tool"], e["kind"]) == ("concept_search", "search")
    assert e["args"] == {"query": "sprocket weights balance", "k": 20, "mode": "keyword"}
    first, hit = e["items"][0], body["hits"][0]
    assert first == {"kind": "concept", "id": hit["handle"], "claim_ids": [hit["example"]["claim_id"]],
                     "paper_ids": [hit["example"]["paper"]["paper_id"]]}
    assert [i["id"] for i in e["items"]] == [h["handle"] for h in body["hits"]]


def test_concept_show_logs_claims_and_the_spans_whose_quotes_it_returned(server, world, home):
    body = ok(call(server, "concept_show", {"handle": "method/auto wobble", "graph": GRAPH, "session": NAME}))
    assert body["session"] == NAME and "record_reminder" not in body
    (e,) = events(home)
    assert (e["tool"], e["kind"]) == ("concept_show", "open_concept")
    assert e["args"] == {"handle": "method/auto wobble", "graph": GRAPH, "claims_offset": 0, "claims_limit": 20}
    claim_items = [i for i in e["items"] if "claim_id" in i]
    assert [i["claim_id"] for i in claim_items] == [c["claim_id"] for c in body["claims"]]
    for item, c in zip(claim_items, body["claims"], strict=True):
        assert item["span_ids"] == ([c["quote"]["span_id"]] if c["quote"] else [])
        assert item["paper_id"] == c["paper"]["paper_id"]
    link_spans = {q["span_id"] for link in body["links"] for q in link["quotes"] if q["text"]}
    assert link_spans <= {s for i in e["items"] if "claim_id" not in i for s in i["span_ids"]}


def test_a_chooser_logs_an_event_with_no_items(server, world, home):
    from concepts._world import World

    from seedgraph.concepts import refresh as rf

    lib = libstore.open_library_at(home / "library")
    try:
        w = World(lib)
        w.split("assumption", "identification", {"s1": "exogenous variation", "s2": "unique solution"})
        for paper, sense in ((p(0, 1), "s1"), (p(1, 2), "s2")):
            cid = lib.conn.execute("SELECT claim_id FROM current_claims WHERE paper_id = ? AND normalized_label = "
                                   "'identification'", (paper,)).fetchone()[0]
            w.assign("assumption", "identification", cid, sense)
        rf.refresh(lib)
    finally:
        lib.close()
    body = ok(call(server, "concept_show", {"handle": "assumption/identification"}))
    assert body["kind"] == "chooser"
    (e,) = events(home)
    assert e["items"] == []


def test_literatures_and_hubs_log_concept_handles(server, world, home):
    lits = ok(call(server, "literatures", {"graph": GRAPH, "session": NAME}))
    hubs = ok(call(server, "concept_hubs", {"graph": GRAPH, "family": "assumption", "session": NAME}))
    assert lits["record_reminder"] == hubs["record_reminder"] == RECORD_REMINDER
    a, b = events(home)
    assert (a["tool"], a["kind"], a["args"]) == ("literatures", "search", {"graph": GRAPH})
    assert (b["tool"], b["kind"], b["args"]) == ("concept_hubs", "search",
                                                  {"graph": GRAPH, "family": "assumption", "limit": 30})
    top = {c["handle"] for lit in lits["literatures"] for c in lit["top_concepts"]}
    assert top and top <= {i["id"] for i in a["items"]}
    assert {i["kind"] for i in a["items"] + b["items"]} == {"concept"}
    assert [i["id"] for i in b["items"]][:len(hubs["families"]["assumption"])] == \
        [h["handle"] for h in hubs["families"]["assumption"]]


def test_the_hubs_resource_is_not_logged(server, world, home):
    assert read_resource(server, f"seedgraph://graphs/{GRAPH}/hubs")["graph"] == GRAPH
    assert events(home) == []


def test_a_quote_seen_in_concept_show_counts_as_read(server, world, home):
    show = ok(call(server, "concept_show", {"handle": "assumption/sprocket balance", "session": NAME}))
    quoted = next(c for c in show["claims"] if c["quote"])
    reply = ok(call(server, "record_answer", {"question": "q", "answer": "a", "session": NAME,
                                              "evidence": [{"span_id": quoted["quote"]["span_id"]},
                                                           {"claim_id": quoted["claim_id"]}]}))
    assert reply["flags"] == [] and reply["counts"]["cited"] == 2
