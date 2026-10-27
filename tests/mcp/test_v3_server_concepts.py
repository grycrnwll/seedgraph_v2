"""v3 MCP concept tools (stage 4 design §9, slice C9) over an invented library: concept search
with its mode marker, concept pages with senses, merges and links, literatures and bridges,
and the hub list (tool and resource). Agent audience throughout: claims on a paper whose
notes are hidden support nothing. Invented text only."""

from __future__ import annotations

import json

import anyio
import pytest
from concepts._embed import index_fts  # tests/ is on sys.path (rootdir conftest)
from concepts._world import World, a, m
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import AnyUrl
from v3_mcp_support import call, err, ok, home, server  # noqa: F401 (fixtures)

from seedgraph.concepts import refresh as rf
from seedgraph.library import attention, graphs
from seedgraph.library import literatures as lits
from seedgraph.library import store as libstore

GRAPH = "invented_gadgets"
GROUPS, SIZE = 3, 5


def p(c: int, i: int) -> str:
    return f"lp_{c}_{i}"


EXTRA = {
    p(0, 0): [a("sprocket balance", "Invented: sprocket weights equalise the widget groups."),
              m("serial wobble", "Invented: residual wobble persists across periods.")],
    p(1, 0): [a("sprocket balance", "Invented: sprocket balance holds after weighting the gadgets.")],
    p(0, 1): [a("identification", "Invented: a lever shock gives identification of the widget effect.")],
    p(1, 2): [a("identification", "Invented: the gadget system has a unique solution, so identification holds.")],
    p(1, 1): [m("widget lag", "Invented: the widget lag captures delayed gadget response.")],
    p(2, 1): [m("widget lag order", "Invented: choose the widget lag order by an invented criterion.")],
    p(2, 2): [m("auto wobble", "Invented: the wobble depends on its own past values.")],
    p(2, 3): [a("secret flange", "Invented: the secret flange is calibrated before the widget test.")],
}


@pytest.fixture
def world(home):
    lib = libstore.open_library_at(home / "library")
    try:
        w = World(lib)
        for c in range(GROUPS):
            for i in range(SIZE):
                claims = [a(f"sprocket rule {c}", f"Invented: sprocket rule {c} holds in paper {i}.")]
                w.note(p(c, i), claims + EXTRA.get(p(c, i), []))
        with lib.write() as conn:
            for c in range(GROUPS):
                for i in range(SIZE):
                    for j in range(i + 1, SIZE):
                        conn.execute("INSERT INTO citation_links (source_paper_id, target_paper_id, provenance, "
                                     "confidence, created_at) VALUES (?, ?, 'manual_override', 1.0, '2026-01-01')",
                                     (p(c, i), p(c, j)))
        w.decide("merge", "widget lag", "widget lag order", name="widget lag")
        did = w.decide("link", "auto wobble", "serial wobble", by="agent")
        with lib.write() as conn:
            conn.execute("UPDATE concept_decisions SET note = ? WHERE decision_id = ?",
                         ("history: a residual property later modelled as a process", did))
        with graphs.create_graph(lib, GRAPH) as g:
            for c in range(GROUPS):
                for i in range(SIZE):
                    graphs.set_membership(g, p(c, i), "member")
            lits.regroup(g, stability_seeds=())
            first = lits.literatures(g)[0]
            lits.rename_literature(g, first.literature_id, "Researcher sprocket studies")
        index_fts(lib)
        rf.refresh(lib)
        yield w
    finally:
        lib.close()


def hide(home, paper_id: str) -> None:
    lib = libstore.open_library_at(home / "library")
    try:
        attention.raise_item(lib, "pdf_title_miss", {"paper_id": paper_id}, paper_id=paper_id,
                             detail="the PDF title does not match", hides_notes=True)
    finally:
        lib.close()


def read_resource(server, uri: str) -> dict:
    async def _driver():
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            return await client.read_resource(AnyUrl(uri))

    res = anyio.run(_driver)
    return json.loads(res.contents[0].text)


def handles(body: dict) -> list[str]:
    return [h["handle"] for h in body["hits"]]


# --- concept_search -------------------------------------------------------------------------------

def test_concept_search_returns_sense_hits_with_the_mode_marker(server, world):
    body = ok(call(server, "concept_search", {"query": "sprocket weights balance"}))
    assert body["search_mode"] == "keyword" and body["embeddings_adopted"] is False
    hit = body["hits"][0]
    assert hit["handle"] == "assumption/sprocket balance"
    assert hit["family"] == "assumption" and hit["sense_id"] == "s0"
    assert hit["papers"] == 2 and hit["claims"] == 2
    assert hit["gloss"] is None and hit["sense_checked"] is False
    ex = hit["example"]
    assert ex["paper"]["paper_id"] in (p(0, 0), p(1, 0)) and ex["quote"] and ex["trust"] == "paper states"


def test_concept_search_filters_and_empty_replies_keep_the_marker(server, world):
    body = ok(call(server, "concept_search", {"query": "wobble", "family": "assumption"}))
    assert body["hits"] == [] and body["search_mode"] == "keyword"
    body = ok(call(server, "concept_search", {"query": "wobble", "family": "method", "k": 1}))
    assert len(body["hits"]) == 1 and body["hits"][0]["family"] == "method"
    assert err(call(server, "concept_search", {"query": "wobble", "family": "nonsense"}))["code"] == "invalid_argument"
    assert err(call(server, "concept_search", {"query": "wobble", "graph": "nope"}))["code"] == "graph_not_found"
    assert err(call(server, "concept_search", {"query": "wobble", "k": 0}))["code"] == "invalid_argument"


def test_concept_search_in_a_graph_counts_only_members(server, world, home):
    lib = libstore.open_library_at(home / "library")
    try:
        with graphs.create_graph(lib, "small_graph") as g:
            graphs.set_membership(g, p(0, 0), "member")
    finally:
        lib.close()
    body = ok(call(server, "concept_search", {"query": "sprocket balance", "graph": "small_graph"}))
    hit = next(h for h in body["hits"] if h["handle"] == "assumption/sprocket balance")
    assert hit["papers"] == 1 and hit["example"]["paper"]["paper_id"] == p(0, 0)


def test_hidden_papers_support_nothing_in_search(server, world, home):
    assert "assumption/secret flange" in handles(ok(call(server, "concept_search", {"query": "secret flange"})))
    hide(home, p(2, 3))
    hide(home, p(1, 0))
    body = ok(call(server, "concept_search", {"query": "secret flange sprocket balance"}))
    assert "assumption/secret flange" not in handles(body)
    hit = next(h for h in body["hits"] if h["handle"] == "assumption/sprocket balance")
    assert hit["papers"] == 1 and hit["example"]["paper"]["paper_id"] == p(0, 0)
    assert p(2, 3) not in json.dumps(body) and p(1, 0) not in json.dumps(body)


# --- concept_show ---------------------------------------------------------------------------------

def test_concept_show_lists_members_claims_merges_and_rules(server, world):
    body = ok(call(server, "concept_show", {"handle": "method/widget lag order"}))
    assert body["kind"] == "concept"
    c = body["concept"]
    assert c["name"] == "widget lag" and c["family"] == "method" and c["papers"] == 2
    assert sorted(mb["handle"] for mb in body["members"]) == ["method/widget lag", "method/widget lag order"]
    (merge,) = body["merges"]
    assert merge["decided_by"] == "researcher" and {merge["label_a"], merge["label_b"]} == {"widget lag",
                                                                                          "widget lag order"}
    assert {cl["paper"]["paper_id"] for cl in body["claims"]} == {p(1, 1), p(2, 1)}
    cl = body["claims"][0]
    assert cl["claim_id"] and cl["quote"]["span_id"] and cl["quote"]["text"] and cl["trust"] == "paper states"
    rules = " ".join(body["agent_rules"]).lower()
    assert "researcher" in rules and "sense not checked" in rules


def test_concept_show_links_carry_their_note_and_quotes(server, world):
    body = ok(call(server, "concept_show", {"handle": "method/serial wobble"}))
    (link,) = body["links"]
    assert link["other"]["handle"] == "method/auto wobble"
    assert link["note"].startswith("history:") and link["note_by"] == "agent inferred"
    assert link["decided_by"] == "agent"


def test_concept_show_in_a_graph_names_literatures(server, world):
    body = ok(call(server, "concept_show", {"handle": "assumption/sprocket balance", "graph": GRAPH}))
    names = sorted(lit["name"] for lit in body["literatures"])
    assert len(names) == 2 and "Researcher sprocket studies" in names
    assert body["concept"]["sense_status"] == "same words, sense not checked"


def test_concept_show_handles_choosers_paging_and_refusals(server, world, home):
    lib = libstore.open_library_at(home / "library")
    try:
        w = World(lib)
        w.split("assumption", "identification", {"s1": "causal: source of exogenous variation",
                                                 "s2": "theory: unique solution of the system"})
        for paper, sense in ((p(0, 1), "s1"), (p(1, 2), "s2")):
            cid = lib.conn.execute("SELECT claim_id FROM current_claims WHERE paper_id = ? AND normalized_label = "
                                   "'identification'", (paper,)).fetchone()[0]
            w.assign("assumption", "identification", cid, sense)
        rf.refresh(lib)
    finally:
        lib.close()
    body = ok(call(server, "concept_show", {"handle": "assumption/identification"}))
    assert body["kind"] == "chooser"
    assert [s["handle"] for s in body["senses"]] == ["assumption/identification~s1", "assumption/identification~s2"]
    assert body["senses"][0]["gloss_by"] == "researcher"
    paged = ok(call(server, "concept_show", {"handle": "assumption/sprocket rule 0", "claims_limit": 2,
                                             "claims_offset": 1}))
    assert len(paged["claims"]) == 2 and paged["claims_total"] == SIZE
    assert err(call(server, "concept_show", {"handle": "no slash"}))["code"] == "invalid_argument"
    assert err(call(server, "concept_show", {"handle": "assumption/no such label"}))["code"] == "concept_not_found"


def test_hidden_papers_support_nothing_in_concept_show(server, world, home):
    hide(home, p(2, 3))
    hide(home, p(1, 0))
    assert err(call(server, "concept_show", {"handle": "assumption/secret flange"}))["code"] == "concept_not_found"
    body = ok(call(server, "concept_show", {"handle": "assumption/sprocket balance"}))
    assert body["concept"]["papers"] == 1
    assert {cl["paper"]["paper_id"] for cl in body["claims"]} == {p(0, 0)}
    assert p(1, 0) not in json.dumps(body)


# --- literatures, hubs ----------------------------------------------------------------------------

def test_literatures_lists_names_sizes_top_concepts_and_bridges(server, world):
    body = ok(call(server, "literatures", {"graph": GRAPH}))
    assert sorted(lit["size"] for lit in body["literatures"]) == [SIZE] * GROUPS
    named = [lit for lit in body["literatures"] if lit["name_source"] == "researcher"]
    assert [lit["name"] for lit in named] == ["Researcher sprocket studies"]
    assert all(lit["top_concepts"] for lit in body["literatures"])
    bridges = {b["kind"]: b for b in body["bridges"]}
    assert bridges["concept"]["handle"] == "method/widget lag" and bridges["concept"]["literature_count"] == 2
    assert {bridges["link"]["a"]["handle"], bridges["link"]["b"]["handle"]} == {"method/auto wobble",
                                                                              "method/serial wobble"}
    unchecked = {u["handle"]: u for u in body["unchecked"]}
    assert set(unchecked) == {"assumption/sprocket balance", "assumption/identification"}
    assert unchecked["assumption/identification"]["reason"] == "ambiguous label"
    assert "agent inferred" in json.dumps(body["rules"])


def test_a_confirmed_sense_becomes_a_bridge(server, world, home):
    lib = libstore.open_library_at(home / "library")
    try:
        with lib.write() as conn:
            conn.execute("INSERT INTO concept_senses (family, label_key, sense_id, gloss, decided_by, state, "
                         "decided_at) VALUES ('assumption', 'sprocket balance', 's0', NULL, 'agent', 'confirmed', "
                         "'2026-01-02')")
        rf.refresh(lib)
    finally:
        lib.close()
    body = ok(call(server, "literatures", {"graph": GRAPH}))
    assert "assumption/sprocket balance" in [b.get("handle") for b in body["bridges"]]
    assert "assumption/sprocket balance" not in [u["handle"] for u in body["unchecked"]]


def test_literatures_without_a_grouping_say_what_to_do(server, world, home):
    lib = libstore.open_library_at(home / "library")
    try:
        with graphs.create_graph(lib, "ungrouped") as g:
            graphs.set_membership(g, p(0, 0), "member")
    finally:
        lib.close()
    body = ok(call(server, "literatures", {"graph": "ungrouped"}))
    assert body["literatures"] == [] and body["bridges"] == [] and body["next_action"]
    assert err(call(server, "literatures", {"graph": "nope"}))["code"] == "graph_not_found"


def test_concept_hubs_tool_and_resource(server, world, home):
    body = ok(call(server, "concept_hubs", {"graph": GRAPH}))
    top = body["families"]["assumption"][0]
    assert top["handle"].startswith("assumption/sprocket rule") and top["papers"] == SIZE
    assert top["literature_count"] == 1
    assert "method" in body["families"] and body["bridges"]
    assert body == read_resource(server, f"seedgraph://graphs/{GRAPH}/hubs")
    only = ok(call(server, "concept_hubs", {"family": "method", "limit": 2}))
    assert list(only["families"]) == ["method"] and len(only["families"]["method"]) == 2
    hide(home, p(2, 3))
    assert "secret flange" not in json.dumps(ok(call(server, "concept_hubs", {"limit": 100})))


def test_adopted_embeddings_fall_back_to_keyword_and_keep_the_audience(server, world, home):
    from seedgraph.concepts import search as cs

    lib = libstore.open_library_at(home / "library")
    try:
        cs.set_adoption(lib, adopted=True)
    finally:
        lib.close()
    hide(home, p(1, 0))
    body = ok(call(server, "concept_search", {"query": "sprocket balance"}))
    assert body["embeddings_adopted"] is True
    assert body["search_mode"] == "keyword" and body["fallback"] == "no_vectors"
    assert "assumption/sprocket balance" in handles(body) and p(1, 0) not in json.dumps(body)
    empty = ok(call(server, "concept_search", {"query": "zzzz nothing matches"}))
    assert empty["hits"] == [] and empty["search_mode"] == "keyword" and empty["fallback"] == "no_vectors"


def test_a_label_supported_only_by_hidden_notes_leaves_the_concept_page(server, world, home):
    lib = libstore.open_library_at(home / "library")
    try:
        World(lib).decide("merge", "secret flange", "sprocket balance", name="sprocket balance")
        rf.refresh(lib)
    finally:
        lib.close()
    before = ok(call(server, "concept_show", {"handle": "assumption/sprocket balance"}))
    assert "secret flange" in json.dumps(before)
    hide(home, p(2, 3))
    body = ok(call(server, "concept_show", {"handle": "assumption/sprocket balance"}))
    assert "secret flange" not in json.dumps(body) and p(2, 3) not in json.dumps(body)
    assert body["merges"] == []
