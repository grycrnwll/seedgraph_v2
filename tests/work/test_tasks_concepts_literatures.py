"""``concepts:name_literatures`` (stage 4 design §6.3): one package per graph grouping; agent
names are stored as agent inferred, researcher renames win, and a regroup between
packaging and saving drops the stale names. Invented text only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from concepts._world import World, a  # tests/ is on sys.path (rootdir conftest)

from seedgraph.library import graphs
from seedgraph.library import literatures as lits
from seedgraph.work import routine
from seedgraph.work.claims import WorkerRef
from seedgraph.work.store import WorkError
from seedgraph.work.tasks_concepts import load_manifest

from .conftest import T0, at

W1 = WorkerRef("interactive", "w1", client="claude")
GROUPS, SIZE = 3, 5


@pytest.fixture
def graph_world(lib):
    w = World(lib)
    for c in range(GROUPS):
        for i in range(SIZE):
            w.note(f"lp_{c}_{i}", [a(f"sprocket rule {c}", f"Invented: sprocket rule {c} holds in paper {i}.")])
    with lib.write() as conn:
        for c in range(GROUPS):
            for i in range(SIZE):
                for j in range(i + 1, SIZE):
                    conn.execute("INSERT INTO citation_links (source_paper_id, target_paper_id, provenance, "
                                 "confidence, created_at) VALUES (?, ?, 'manual_override', 1.0, '2026-01-01')",
                                 (f"lp_{c}_{i}", f"lp_{c}_{j}"))
    g = graphs.create_graph(lib, "invented_graph")
    for c in range(GROUPS):
        for i in range(SIZE):
            graphs.set_membership(g, f"lp_{c}_{i}", "member")
    lits.regroup(g, stability_seeds=())
    yield g
    g.close()


def test_a_naming_package_lists_each_literature_and_names_are_saved(lib, graph_world):
    g = graph_world
    first = lits.literatures(g)[0]
    lits.rename_literature(g, first.literature_id, "Researcher name")  # a researcher name wins: not offered
    view = routine.create_batch(lib, "concepts:name_literatures", graph="invented_graph", now=T0)
    assert [i.subject_label for i in view.items] == [f"literatures of invented_graph (run {first.run_id})"]
    msg = routine.claim_next(lib, view.batch_id, W1, now=at(1)).message
    text = Path(msg["text_path"]).read_text(encoding="utf-8")
    assert text.count("## ") == GROUPS - 1 and "An invented paper" in text and "sprocket rule" in text
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"items": [
        {"item": 1, "name": "One two three four five six seven"}, {"item": 2, "name": "Fine"}]}, now=at(2))
    assert res.status == "invalid" and res.errors == ["item 1: name has more than 6 words"]
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"items": [
        {"item": 1, "name": "Invented sprocket studies", "description": "Papers on invented sprockets."},
        {"item": 2, "name": "Invented gadget studies"}]}, now=at(3))
    assert res.status == "accepted" and res.paper_status == "done"
    named = {lit.literature_id: lit for lit in lits.literatures(g)}
    assert named[first.literature_id].name == "Researcher name"
    agent = [lit for lit in named.values() if lit.name_source == "agent"]
    assert sorted(lit.name for lit in agent) == ["Invented gadget studies", "Invented sprocket studies"]


def test_researcher_renames_win_over_a_later_agent_name(lib, graph_world):
    g = graph_world
    view = routine.create_batch(lib, "concepts:name_literatures", graph="invented_graph", now=T0)
    msg = routine.claim_next(lib, view.batch_id, W1, now=at(1)).message
    lid = load_manifest(lib, view.batch_id, msg["subject_id"])["items"][0]["literature"]["literature_id"]
    lits.rename_literature(g, lid, "Researcher name")
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"items": [
        {"item": n, "name": f"Agent name {n}"} for n in range(1, GROUPS + 1)]}, now=at(2))
    assert res.status == "accepted"
    lit = next(x for x in lits.literatures(g) if x.literature_id == lid)
    assert lit.name == "Researcher name" and lit.agent_name == "Agent name 1"


def test_a_regroup_after_packaging_drops_the_stale_names(lib, graph_world):
    g = graph_world
    view = routine.create_batch(lib, "concepts:name_literatures", graph="invented_graph", now=T0)
    msg = routine.claim_next(lib, view.batch_id, W1, now=at(1)).message
    lits.regroup(g, stability_seeds=())  # new run: literature ids change
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"items": [
        {"item": n, "name": f"Agent name {n}"} for n in range(1, GROUPS + 1)]}, now=at(2))
    assert res.status == "accepted"
    assert all(lit.agent_name is None for lit in lits.literatures(g))
    detail = lib.conn.execute("SELECT detail_json FROM batch_events WHERE batch_id = ? AND kind = 'concepts_saved'",
                              (view.batch_id,)).fetchone()[0]
    assert json.loads(detail)["dropped"] == GROUPS


def test_naming_needs_a_graph(lib):
    with pytest.raises(WorkError, match="context graph"):
        routine.create_batch(lib, "concepts:name_literatures", now=T0)
