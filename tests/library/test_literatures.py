"""Literatures: seeded Louvain over the in-graph citation view (v3 design §6.2).

Pure planning core: determinism, the resolution ladder and size rule, unplaced papers,
the stability (ARI) report. Graph-bound storage lives in test_literatures_store.py.
"""

from __future__ import annotations

import random

import pytest

from seedgraph.library import literatures as lits


def _cliques(k: int, n: int, prefix: str = "p") -> tuple[list[str], list[tuple[str, str]]]:
    """k planted cliques of n papers, joined in a ring by one citation each."""
    members: list[str] = []
    edges: list[tuple[str, str]] = []
    groups = []
    for c in range(k):
        g = [f"{prefix}{c}_{i:02d}" for i in range(n)]
        groups.append(g)
        members += g
        edges += [(a, b) for i, a in enumerate(g) for b in g[i + 1:]]
    for c in range(k):
        edges.append((groups[c][0], groups[(c + 1) % k][1]))
    return members, edges


# --- stability measure -------------------------------------------------------


def test_ari_is_one_for_identical_partitions_up_to_relabelling():
    a = {"x": 0, "y": 0, "z": 1, "w": 1}
    b = {"x": 7, "y": 7, "z": 3, "w": 3}
    assert lits.adjusted_rand_index(a, b) == pytest.approx(1.0)


def test_ari_worked_example():
    # Hand-worked contingency: a = {0,0,0,1,1,1}, b = {0,0,1,1,2,2}
    # cells (0,0)=2 (0,1)=1 (1,1)=1 (1,2)=2 -> sum C(n,2) = 1+0+0+1 = 2
    # rows 3,3 -> 3+3 = 6; cols 2,2,2 -> 1+1+1 = 3; C(6,2) = 15
    # expected = 6*3/15 = 1.2; max = (6+3)/2 = 4.5; ARI = (2-1.2)/(4.5-1.2) = 0.8/3.3
    nodes = "abcdef"
    a = dict(zip(nodes, [0, 0, 0, 1, 1, 1]))
    b = dict(zip(nodes, [0, 0, 1, 1, 2, 2]))
    assert lits.adjusted_rand_index(a, b) == pytest.approx(0.8 / 3.3)


# --- determinism -------------------------------------------------------------


def test_plan_is_identical_under_shuffled_and_duplicated_input():
    members, edges = _cliques(4, 6)
    base = lits.plan_literatures(members, edges)
    rng = random.Random(11)
    for _ in range(5):
        m = members[:]
        e = edges[:] + [(b, a) for a, b in edges[:7]]  # reversed duplicates collapse
        rng.shuffle(m)
        rng.shuffle(e)
        again = lits.plan_literatures(m, e)
        assert again.groups == base.groups
        assert again.unplaced == base.unplaced
        assert again.resolution == base.resolution
        assert again.fingerprint == base.fingerprint


# --- ladder and size rule ----------------------------------------------------


def test_choose_rung_takes_the_lowest_resolution_meeting_the_rule():
    rungs = [
        lits.Rung(0.5, sizes=(20, 6, 6), modularity=0.30),  # largest 20/32 > 30%
        lits.Rung(0.75, sizes=(9, 9, 8, 6), modularity=0.40),  # 9/32 ok, 4 groups >= 5
        lits.Rung(1.0, sizes=(8, 8, 8, 8), modularity=0.45),  # also ok, but higher
    ]
    chosen, rule_met = lits.choose_rung(rungs)
    assert (chosen.resolution, rule_met) == (0.75, True)


def test_choose_rung_needs_three_literatures_of_min_size():
    rungs = [
        lits.Rung(0.5, sizes=(6, 6), modularity=0.2),  # only 2 literatures
        lits.Rung(1.0, sizes=(5, 5, 5, 4), modularity=0.3),  # 5/15 > 30%... placed = 15
        lits.Rung(1.5, sizes=(5, 5, 5, 5), modularity=0.25),  # 25%, 4 of size 5
    ]
    chosen, rule_met = lits.choose_rung(rungs)
    assert (chosen.resolution, rule_met) == (1.5, True)


def test_choose_rung_falls_back_to_best_modularity():
    rungs = [
        lits.Rung(0.5, sizes=(30,), modularity=0.10),
        lits.Rung(1.0, sizes=(20, 10), modularity=0.35),
        lits.Rung(2.0, sizes=(15, 15), modularity=0.20),
    ]
    chosen, rule_met = lits.choose_rung(rungs)
    assert (chosen.resolution, rule_met) == (1.0, False)


def test_planted_cliques_are_recovered_as_literatures():
    members, edges = _cliques(4, 6)
    plan = lits.plan_literatures(members, edges)
    assert plan.rule_met
    assert plan.resolution in lits.LADDER
    expected = sorted(tuple(f"p{c}_{i:02d}" for i in range(6)) for c in range(4))
    assert sorted(plan.groups) == expected
    # every lower rung failed the rule
    for r in plan.rungs:
        if r.resolution < plan.resolution:
            assert not lits.meets_size_rule(r.sizes)


def test_explicit_resolution_overrides_the_ladder():
    members, edges = _cliques(4, 6)
    plan = lits.plan_literatures(members, edges, resolution=2.0)
    assert plan.resolution == 2.0


# --- unplaced ----------------------------------------------------------------


def test_uncited_members_and_small_communities_are_unplaced():
    members, edges = _cliques(4, 6)
    # an isolated member, and a 3-paper triangle hanging off nothing
    members += ["z_alone", "t1", "t2", "t3"]
    edges += [("t1", "t2"), ("t2", "t3"), ("t1", "t3")]
    plan = lits.plan_literatures(members, edges)
    assert plan.unplaced == ("t1", "t2", "t3", "z_alone")
    placed = {p for g in plan.groups for p in g}
    assert placed.isdisjoint(plan.unplaced)
    assert len(placed) == 24


def test_groups_are_ordered_by_size_then_smallest_member():
    members, edges = _cliques(3, 5, prefix="b")
    m2, e2 = _cliques(1, 7, prefix="a")
    # join the big clique to the ring so all share one component
    plan = lits.plan_literatures(members + m2, edges + e2 + [("a0_00", "b0_00")])
    sizes = [len(g) for g in plan.groups]
    assert sizes == sorted(sizes, reverse=True)
    assert plan.groups[0][0] == "a0_00"


def test_empty_and_edgeless_graphs_plan_without_error():
    empty = lits.plan_literatures([], [])
    assert empty.groups == () and empty.unplaced == ()
    lonely = lits.plan_literatures(["b", "a"], [])
    assert lonely.groups == () and lonely.unplaced == ("a", "b")


# --- stability report --------------------------------------------------------


def test_stability_report_is_one_for_unambiguous_structure():
    members, edges = _cliques(4, 8)
    plan = lits.plan_literatures(members, edges, stability_seeds=(1, 2, 3))
    assert plan.ari_mean == pytest.approx(1.0)
    assert plan.nmi_mean == pytest.approx(1.0)
    assert plan.stability_seeds == (1, 2, 3)


def test_stability_can_be_skipped():
    members, edges = _cliques(4, 6)
    plan = lits.plan_literatures(members, edges, stability_seeds=())
    assert plan.ari_mean is None


# --- fingerprint -------------------------------------------------------------


def test_fingerprint_changes_with_membership_or_citations_only():
    members, edges = _cliques(3, 5)
    fp = lits.citation_fingerprint(members, edges)
    assert lits.citation_fingerprint(list(reversed(members)), [(b, a) for a, b in edges]) == fp
    assert lits.citation_fingerprint(members + ["new_uncited"], edges) != fp
    assert lits.citation_fingerprint(members, edges[:-1]) != fp
