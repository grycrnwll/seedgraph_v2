"""Guardrails (stage 4 design §2.2-§2.3): vetoes, seeds, the ambiguous list with per-graph
overrides. Invented text only."""

from __future__ import annotations

import pytest

from seedgraph.concepts import guardrails as g
from seedgraph.concepts.model import SenseKey
from seedgraph.concepts.refresh import refresh
from seedgraph.library import store

from ._world import World, a, m


@pytest.fixture
def lib(tmp_path):
    library = store.open_library_at(tmp_path / "library")
    yield library
    library.close()


@pytest.fixture
def w(lib):
    return World(lib)


def K(label, family="assumption", sense="s0"):
    return SenseKey(family, label, sense)


# --- pure vetoes ----------------------------------------------------------------------------------

def test_type_fence_refuses_cross_family_merges():
    assert g.merge_veto([K("x")], [K("y", "method")], g.VetoIndex([])) == "type_fence"


def test_never_merge_is_transitive_over_the_whole_group():
    vetoes = g.VetoIndex([g.Veto("never_merge", "ate", "att")])
    assert g.merge_veto([K("ate"), K("average effect")], [K("att")], vetoes) == "never_merge"
    assert g.merge_veto([K("ate")], [K("average effect")], vetoes) is None


def test_never_connect_blocks_links_but_never_merge_does_not():
    vetoes = g.VetoIndex([g.Veto("never_merge", "ate", "att"), g.Veto("never_connect", "x", "y")])
    assert g.link_veto([K("ate")], [K("att")], vetoes) is None
    assert g.link_veto([K("x")], [K("y")], vetoes) == "never_connect"
    assert g.merge_veto([K("x")], [K("y")], vetoes) == "never_connect"


def test_veto_without_a_sense_covers_every_sense_and_with_one_only_that_sense():
    all_senses = g.VetoIndex([g.Veto("never_merge", "identification", "uniqueness")])
    assert g.merge_veto([K("identification", sense="s2")], [K("uniqueness")], all_senses) == "never_merge"
    one = g.VetoIndex([g.Veto("never_merge", "identification", "uniqueness", sense_a="s1")])
    assert g.merge_veto([K("identification", sense="s2")], [K("uniqueness")], one) is None
    assert g.merge_veto([K("identification", sense="s1")], [K("uniqueness")], one) == "never_merge"


def test_stoplist_is_compared_over_the_resulting_group():
    assert g.discriminating_conflict(["widget model v1", "widget model v2"])
    assert not g.discriminating_conflict(["widget model v1", "widget model v1 assumption"])
    assert g.merge_veto([K("widget model v1"), K("widget model")], [K("widget model v2")], g.VetoIndex([])) \
        == "stoplist"
    assert g.merge_veto([K("a v1")], [K("a v2")], g.VetoIndex([]), stoplist=False) is None


def test_closure():
    assert g.closure_ok("Parallel trends", ["parallel trends", "common trends"])
    assert not g.closure_ok("trend assumptions", ["parallel trends", "common trends"])


# --- vetoes inside refresh ------------------------------------------------------------------------

def test_transitive_never_merge_veto_in_refresh(lib, w):
    w.note("lp_p1", [a("strict exogeneity"), a("sequential exogeneity"), a("predetermined regressors")])
    w.decide("never_merge", "sequential exogeneity", "strict exogeneity")
    first = w.decide("merge", "predetermined regressors", "strict exogeneity", by="agent",
                     at="2026-01-01T00:00:00")
    second = w.decide("merge", "predetermined regressors", "sequential exogeneity", by="agent",
                      at="2026-01-02T00:00:00")
    rep = refresh(lib)
    assert w.concept_of("assumption", "predetermined regressors") == w.concept_of("assumption", "strict exogeneity")
    assert w.concept_of("assumption", "sequential exogeneity") != w.concept_of("assumption", "strict exogeneity")
    assert [(o.decision_id, o.reason) for o in rep.blocked] == [(second, "never_merge")]
    assert w.state(first)[0] == "applied"


def test_seed_pairs_stay_apart_even_through_acronym_folds(lib, w):
    w.note("lp_p1", [m("ate"), m("average treatment effect"), m("att"),
                     m("average treatment effect on the treated"), m("did"), m("ddd"),
                     m("difference in differences")])
    w.note("lp_p2", [a("heterogeneity"), a("heteroskedasticity")])
    w.decide("merge", "ate", "att", by="agent", state="pending_apply")
    w.decide("merge", "heterogeneity", "heteroskedasticity", by="agent", state="pending_apply")
    refresh(lib)
    seeds = lib.conn.execute("SELECT label_a, label_b FROM concept_decisions WHERE origin = 'seed' "
                             "AND decided_by = 'researcher' AND kind = 'never_merge'").fetchall()
    assert ("ate", "att") in [tuple(r) for r in seeds]
    assert w.concept_of("method", "ate") == w.concept_of("method", "average treatment effect")
    assert w.concept_of("method", "ate") != w.concept_of("method", "att")
    assert w.concept_of("method", "did") == w.concept_of("method", "difference in differences")
    assert w.concept_of("method", "did") != w.concept_of("method", "ddd")
    assert w.concept_of("assumption", "heterogeneity") != w.concept_of("assumption", "heteroskedasticity")


def test_seeds_are_written_once_and_a_researcher_delete_sticks(lib, w):
    w.note("lp_p1", [m("ate")])
    refresh(lib)
    n = lib.conn.execute("SELECT COUNT(*) FROM concept_decisions WHERE origin = 'seed'").fetchone()[0]
    assert n == len(g.SEED_NEVER_MERGE)
    with lib.write() as conn:
        conn.execute("DELETE FROM concept_decisions WHERE label_a = 'ate' AND label_b = 'att'")
    w.note("lp_p2", [m("att")])
    refresh(lib)
    assert lib.conn.execute("SELECT COUNT(*) FROM concept_decisions WHERE origin = 'seed'").fetchone()[0] == n - 1


def test_a_seed_never_overwrites_an_existing_decision_on_the_pair(lib, w):
    w.note("lp_p1", [m("ate"), m("att")])
    did = w.decide("merge", "ate", "att", by="researcher", origin="v2:test_proj", state="pending_apply")
    refresh(lib)
    assert lib.conn.execute("SELECT COUNT(*) FROM concept_decisions WHERE label_a = 'ate' AND label_b = 'att'"
                            ).fetchone()[0] == 1
    assert w.state(did)[0] == "applied"


# --- ambiguous labels -----------------------------------------------------------------------------

def test_head_patterns_match_generic_variants_only():
    assert g.pattern_matches("exogeneity", "exogeneity", "head")
    assert g.pattern_matches("exogeneity assumption", "exogeneity", "head")
    assert g.pattern_matches("exogeneity limitation", "exogeneity", "head")
    assert not g.pattern_matches("strict exogeneity", "exogeneity", "head")
    assert g.pattern_matches("identification limitation", "identification limitation", "label")
    assert not g.pattern_matches("identification", "identification limitation", "label")


def test_seed_list_stands_in_before_seeding_and_is_stored_after(lib, w):
    lst = g.effective_ambiguous(lib.conn)
    assert lst.is_ambiguous("identification") and lst.is_ambiguous("selection bias")
    w.note("lp_p1", [m("ate")])
    refresh(lib)
    rows = lib.conn.execute("SELECT pattern, scope, decided_by, origin FROM ambiguous_labels").fetchall()
    assert len(rows) == len(g.SEED_AMBIGUOUS)
    assert {(r[1], r[2], r[3]) for r in rows} == {("", "researcher", "seed")}


def test_library_list_is_editable_with_provenance(lib):
    g.add_ambiguous(lib, "Instrument", match="head", note="invented reason")
    lst = g.effective_ambiguous(lib.conn)
    hit = lst.matches("instrument condition")
    assert hit and (hit.pattern, hit.decided_by, hit.origin, hit.note) == \
        ("instrument", "researcher", "researcher", "invented reason")
    assert g.remove_ambiguous(lib, "instrument")
    assert not g.effective_ambiguous(lib.conn).is_ambiguous("instrument")


def test_graph_overrides_add_labels_and_mark_single_sense(lib, w):
    w.note("lp_p1", [m("ate")])
    refresh(lib)  # seeds stored
    g.add_ambiguous(lib, "widget design", graph="did_graph")
    g.mark_single_sense(lib, "identification", "did_graph", note="narrow invented graph")
    lib_list = g.effective_ambiguous(lib.conn)
    graph_list = g.effective_ambiguous(lib.conn, graph="did_graph")
    other = g.effective_ambiguous(lib.conn, graph="other_graph")
    assert lib_list.is_ambiguous("identification") and not lib_list.is_ambiguous("widget design")
    assert not graph_list.is_ambiguous("identification") and not graph_list.is_ambiguous("identification assumption")
    assert graph_list.is_ambiguous("widget design") and graph_list.is_ambiguous("exogeneity")
    assert other.is_ambiguous("identification") and not other.is_ambiguous("widget design")
    with pytest.raises(ValueError):
        g.mark_single_sense(lib, "identification", "")
    row = lib.conn.execute("SELECT match, effect, decided_by, note FROM ambiguous_labels "
                           "WHERE pattern = 'identification' AND scope = 'did_graph'").fetchone()
    assert tuple(row) == ("head", "single_sense", "researcher", "narrow invented graph")


def test_library_refresh_uses_the_library_list_not_graph_overrides(lib, w):
    w.note("lp_p1", [a("identification"), a("identification assumption")])
    refresh(lib)
    g.mark_single_sense(lib, "identification", "did_graph")
    refresh(lib)
    assert w.concept_of("assumption", "identification") != w.concept_of("assumption", "identification assumption")
