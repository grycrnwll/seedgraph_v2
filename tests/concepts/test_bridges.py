"""Graph slice and bridges (stage 4 design §6.1, §6.3), shared by the web graph pages and
the MCP concept tools: concepts summed from senses inside a graph's members, the slice
weight, bridges across literatures, and the "same words, sense not checked" list.
Ambiguous and unchecked labels never count as bridges. Invented text only.

``build_world(lib)`` is reused by the web graph page tests.
"""

from __future__ import annotations

import math

import pytest

from concepts._embed import index_fts
from concepts._world import World, a, m
from seedgraph.concepts import bridges as br
from seedgraph.concepts import guardrails
from seedgraph.concepts import refresh as rf
from seedgraph.library import attention, graphs
from seedgraph.library import literatures as lits
from seedgraph.library import store as libstore

GRAPH = "invented_gadgets"
GROUPS, SIZE = 3, 5
OUTSIDE = ("lp_out_1", "lp_out_2")


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


def build_world(lib, *, graph: str = GRAPH, shuffle: bool = False) -> World:
    """Three literatures of five papers in ``graph`` (citation cliques), plus two held papers
    outside the graph: ``lp_out_1`` shares ``sprocket rule 0`` and ``sprocket balance``,
    ``lp_out_2`` shares ``auto wobble``. ``shuffle`` writes papers and links in another
    order (same content)."""
    w = World(lib)
    order = [(c, i) for c in range(GROUPS) for i in range(SIZE)]
    if shuffle:
        order = order[::-1]
    for c, i in order:
        claims = [a(f"sprocket rule {c}", f"Invented: sprocket rule {c} holds in paper {i}.")]
        w.note(p(c, i), claims + EXTRA.get(p(c, i), []))
    w.note(OUTSIDE[0], [a("sprocket rule 0", "Invented: sprocket rule 0 holds out of the graph."),
                        a("sprocket balance", "Invented: sprocket balance holds out of the graph.")])
    w.note(OUTSIDE[1], [m("auto wobble", "Invented: the wobble feeds on itself out of the graph.")])
    pairs = [(p(c, i), p(c, j)) for c in range(GROUPS) for i in range(SIZE) for j in range(i + 1, SIZE)]
    if shuffle:
        pairs = pairs[::-1]
    with lib.write() as conn:
        for s, t in pairs:
            conn.execute("INSERT INTO citation_links (source_paper_id, target_paper_id, provenance, "
                         "confidence, created_at) VALUES (?, ?, 'manual_override', 1.0, '2026-01-01')", (s, t))
    w.decide("merge", "widget lag", "widget lag order", name="widget lag")
    did = w.decide("link", "auto wobble", "serial wobble", by="agent")
    with lib.write() as conn:
        conn.execute("UPDATE concept_decisions SET note = ? WHERE decision_id = ?",
                     ("history: a residual property later modelled as a process", did))
    with graphs.create_graph(lib, graph, display_name="Invented gadgets") as g:
        for c, i in order:
            graphs.set_membership(g, p(c, i), "member", is_seed=(i == 0 and c == 0))
        lits.regroup(g, stability_seeds=())
    index_fts(lib)
    rf.refresh(lib)
    return w


@pytest.fixture
def lib(tmp_path):
    library = libstore.open_library_at(tmp_path / "home" / "library")
    try:
        build_world(library)
        yield library
    finally:
        library.close()


def confirm(lib, family: str, label: str, sense: str = "s0") -> None:
    with lib.write() as conn:
        conn.execute("INSERT INTO concept_senses (family, label_key, sense_id, gloss, decided_by, state, decided_at) "
                     "VALUES (?, ?, ?, 'an invented gloss', 'researcher', 'confirmed', '2026-01-02')",
                     (family, label, sense))
    rf.refresh(lib)


def bridge_handles(lib, *, agent_only: bool = False) -> tuple[list[str], dict[str, str]]:
    s = br.load_slice(lib, GRAPH, agent_only=agent_only)
    bridges, unchecked = br.bridges(lib.conn, s)
    names = [b.get("handle") or "+".join(sorted((b["a"]["handle"], b["b"]["handle"]))) for b in bridges]
    return names, {u["handle"]: u["reason"] for u in unchecked}


def test_bridges_span_literatures_and_unchecked_or_ambiguous_labels_never_count(lib):
    names, unchecked = bridge_handles(lib)
    assert names == ["method/widget lag", "method/auto wobble+method/serial wobble"]
    assert unchecked == {"assumption/sprocket balance": br.UNCHECKED,
                         "assumption/identification": br.AMBIGUOUS}


def test_a_confirmed_sense_counts_as_a_bridge(lib):
    confirm(lib, "assumption", "sprocket balance")
    names, unchecked = bridge_handles(lib)
    assert "assumption/sprocket balance" in names and "assumption/sprocket balance" not in unchecked


def test_the_graph_ambiguous_overrides_apply_to_its_slice(lib):
    guardrails.add_ambiguous(lib, "sprocket balance", graph=GRAPH)
    guardrails.mark_single_sense(lib, "identification", GRAPH)
    rf.refresh(lib)
    names, unchecked = bridge_handles(lib)
    assert unchecked["assumption/sprocket balance"] == br.AMBIGUOUS
    # single-sense in this graph: no longer "ambiguous", but still unchecked, so still no bridge
    assert unchecked["assumption/identification"] == br.UNCHECKED
    assert "assumption/identification" not in names


def test_the_slice_sums_senses_inside_members_with_its_weight(lib):
    s = br.load_slice(lib, GRAPH, agent_only=False)
    rows = {c.handle: c for c in br.slice_concepts(s)}
    rule0 = rows["assumption/sprocket rule 0"]
    assert (rule0.papers, rule0.claims) == (SIZE, SIZE)  # lp_out_1 is outside the graph
    assert rule0.weight == pytest.approx(math.log(GROUPS * SIZE / SIZE))
    lag = rows["method/widget lag"]
    assert lag.papers == 2 and lag.literature_count == 2 and lag.is_bridge
    assert rows["assumption/sprocket balance"].literature_count == 0  # unchecked across two: counts nowhere
    assert not rows["assumption/sprocket balance"].is_bridge


def test_the_agent_audience_drops_hidden_papers_and_the_researcher_view_keeps_them(lib):
    attention.raise_item(lib, "pdf_title_miss", {"paper_id": p(2, 3)}, paper_id=p(2, 3),
                         detail="the invented PDF title does not match", hides_notes=True)
    rf.refresh(lib)
    agent = {c.handle for c in br.slice_concepts(br.load_slice(lib, GRAPH, agent_only=True))}
    researcher = {c.handle for c in br.slice_concepts(br.load_slice(lib, GRAPH, agent_only=False))}
    assert "assumption/secret flange" not in agent and "assumption/secret flange" in researcher


def test_a_graph_with_no_grouping_has_no_bridges(lib):
    with graphs.create_graph(lib, "ungrouped") as g:
        graphs.set_membership(g, p(0, 0), "member")
    s = br.load_slice(lib, "ungrouped")
    assert s.run is None and br.bridges(lib.conn, s) == ([], [])
