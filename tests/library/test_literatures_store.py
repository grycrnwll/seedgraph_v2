"""Literatures stored per graph: regroup, fingerprint staleness, renames carried by
Jaccard, the researcher's fewer / more step (v3 design §6.2)."""

from __future__ import annotations

import pytest

from seedgraph.library import graphs, store
from seedgraph.library import literatures as lits

from .test_store import NOW, add_paper, add_source_file, attach_file

GROUPS = 4
SIZE = 6


def _pid(c: int, i: int) -> str:
    return f"lp_{c}_{i:02d}"


def _link(conn, src, dst):
    conn.execute(
        "INSERT INTO citation_links (source_paper_id, target_paper_id, provenance, confidence, "
        "created_at) VALUES (?, ?, 'parsed_bibliography', 1.0, ?)", (src, dst, NOW))


def _hold(conn, pid, n):
    add_paper(conn, pid)
    attach_file(conn, pid, add_source_file(conn, n))


@pytest.fixture
def lib(tmp_path):
    library = store.open_library(tmp_path)
    n = 0
    with library.write() as conn:
        for c in range(GROUPS):
            for i in range(SIZE):
                n += 1
                _hold(conn, _pid(c, i), n)
            for i in range(SIZE):
                for j in range(i + 1, SIZE):
                    _link(conn, _pid(c, i), _pid(c, j))
        for c in range(GROUPS):
            _link(conn, _pid(c, 0), _pid((c + 1) % GROUPS, 1))
        for k, pid in enumerate(("lp_alone", "lp_late"), start=1):
            _hold(conn, pid, 100 + k)
    yield library
    library.close()


@pytest.fixture
def g(lib):
    graph = graphs.create_graph(lib, "diff_in_diff")
    for c in range(GROUPS):
        for i in range(SIZE):
            graphs.set_membership(graph, _pid(c, i), "member")
    graphs.set_membership(graph, "lp_alone", "member")
    yield graph
    graph.close()


def _by_first(g):
    return {lit.members[0]: lit for lit in lits.literatures(g)}


def test_regroup_stores_the_planned_literatures(g):
    assert lits.latest_run(g) is None
    assert lits.is_stale(g)
    plan = lits.plan_for_graph(g, stability_seeds=(1, 2))
    run = lits.regroup(g, stability_seeds=(1, 2), built_at=NOW)
    assert run.resolution == plan.resolution and run.chosen_by == "rule" and run.rule_met
    assert run.ari_mean == pytest.approx(1.0)
    assert run.citation_sha256 == plan.fingerprint
    stored = lits.literatures(g)
    assert [lit.members for lit in stored] == list(plan.groups)
    assert [lit.ordinal for lit in stored] == [1, 2, 3, 4]
    assert all(lit.size == SIZE for lit in stored)
    assert stored[0].name == f"unnamed ({SIZE} papers)" and stored[0].name_source is None
    assert lits.unplaced(g) == ["lp_alone"]
    assert not lits.is_stale(g)


def test_paper_lookup_covers_placed_papers_only(g):
    assert lits.paper_literatures(g) == {}
    lits.regroup(g, stability_seeds=())
    lookup = lits.paper_literatures(g)
    assert len(lookup) == GROUPS * SIZE and "lp_alone" not in lookup
    assert lookup[_pid(2, 3)].members == tuple(_pid(2, i) for i in range(SIZE))


def test_planning_writes_nothing(g):
    lits.plan_for_graph(g, stability_seeds=())
    assert lits.latest_run(g) is None
    assert lits.literatures(g) == []


def test_fingerprint_staleness_follows_membership_and_citations(lib, g):
    lits.regroup(g, stability_seeds=())
    assert not lits.is_stale(g)
    graphs.set_membership(g, "lp_late", "member")  # an uncited member
    assert lits.is_stale(g)
    lits.regroup(g, stability_seeds=())
    assert not lits.is_stale(g)
    assert lits.unplaced(g) == ["lp_alone", "lp_late"]
    with lib.write() as conn:  # a new in-graph citation decided in the library
        _link(conn, "lp_alone", _pid(0, 3))
    assert lits.is_stale(g)
    # frontier papers do not count
    lits.regroup(g, stability_seeds=())
    with lib.write() as conn:
        add_paper(conn, "lp_cited_only")
        _link(conn, _pid(1, 2), "lp_cited_only")
    graphs.set_membership(g, "lp_cited_only", "frontier")
    assert not lits.is_stale(g)


def test_names_and_renames_are_carried_by_jaccard(lib, g):
    lits.regroup(g, stability_seeds=())
    first = _by_first(g)
    renamed = first[_pid(0, 0)]
    agent_only = first[_pid(1, 0)]
    both = first[_pid(2, 0)]
    lits.rename_literature(g, renamed.literature_id, "Invented staggered timing")
    lits.set_agent_name(g, agent_only.literature_id, "Invented event studies", "invented line")
    lits.set_agent_name(g, both.literature_id, "Agent words")
    lits.rename_literature(g, both.literature_id, "Researcher words")

    # grow group 0 by one paper (Jaccard 6/7 with its old self) and regroup
    with lib.write() as conn:
        for j in range(SIZE):
            _link(conn, "lp_late", _pid(0, j))
    graphs.set_membership(g, "lp_late", "member")
    lits.regroup(g, stability_seeds=())

    after = {lit.members[0]: lit for lit in lits.literatures(g)}
    grown = next(lit for lit in after.values() if "lp_late" in lit.members)
    assert grown.size == SIZE + 1
    assert grown.name == "Invented staggered timing" and grown.name_source == "researcher"
    assert grown.literature_id != renamed.literature_id
    carried = after[_pid(1, 0)]
    assert (carried.name, carried.name_source) == ("Invented event studies", "agent")
    assert carried.agent_description == "invented line"
    kept = after[_pid(2, 0)]
    assert (kept.name, kept.agent_name, kept.name_source) == ("Researcher words", "Agent words", "researcher")
    assert after[_pid(3, 0)].name == f"unnamed ({SIZE} papers)"


def test_unmatched_literatures_start_unnamed(lib, g):
    lits.regroup(g, stability_seeds=())
    for lit in lits.literatures(g):
        lits.rename_literature(g, lit.literature_id, f"name {lit.ordinal}")
    # drop four papers of every group (and so the ring): Jaccard 2/6 < 0.5, and the
    # disconnected pairs left behind fall unplaced
    for c in range(GROUPS):
        for i in range(1, SIZE - 1):
            graphs.remove_membership(g, _pid(c, i))
    lits.regroup(g, stability_seeds=())
    assert lits.literatures(g) == []
    assert len(lits.unplaced(g)) == GROUPS * 2 + 1


def test_rename_refuses_unknown_or_empty(g):
    lits.regroup(g, stability_seeds=())
    lit = lits.literatures(g)[0]
    with pytest.raises(graphs.GraphError):
        lits.rename_literature(g, "nope", "x")
    with pytest.raises(graphs.GraphError):
        lits.rename_literature(g, lit.literature_id, "   ")
    lits.rename_literature(g, lit.literature_id, "Mine")
    lits.rename_literature(g, lit.literature_id, None)  # clears back to the agent / unnamed
    assert lits.literatures(g)[0].name_source is None


def test_fewer_or_more_steps_one_rung_and_is_kept_on_regroup(lib, g):
    rule = lits.regroup(g, stability_seeds=())
    i = lits.LADDER.index(rule.resolution)
    more = lits.step_resolution(g, "more", stability_seeds=())
    assert (more.resolution, more.chosen_by) == (lits.LADDER[i + 1], "researcher")
    # a later regroup (after a change) keeps the researcher's rung
    graphs.set_membership(g, "lp_late", "member")
    again = lits.regroup(g, stability_seeds=())
    assert (again.resolution, again.chosen_by) == (lits.LADDER[i + 1], "researcher")
    back = lits.regroup(g, reset_to_rule=True, stability_seeds=())
    assert (back.resolution, back.chosen_by) == (rule.resolution, "rule")


def test_step_past_the_end_of_the_ladder_is_refused(g):
    lits.regroup(g, resolution=lits.LADDER[0], stability_seeds=())
    with pytest.raises(graphs.GraphError):
        lits.step_resolution(g, "fewer", stability_seeds=())
    with pytest.raises(graphs.GraphError):
        lits.step_resolution(g, "sideways", stability_seeds=())  # type: ignore[arg-type]
