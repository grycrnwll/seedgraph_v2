"""Candidate generation for the concept tasks (stage 4 design §3.2, §4.1): lexical groups for
``concepts:merge``, neighbour pairs and sense checks for ``concepts:bridge``. Invented text only."""

from __future__ import annotations

import pytest

from seedgraph.concepts import candidates as cand
from seedgraph.concepts import refresh as rf
from seedgraph.concepts.model import SenseKey
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


def A(label: str, sense: str = "s0") -> SenseKey:
    return SenseKey("assumption", label, sense)


def pairs(cands) -> set[frozenset]:
    return {frozenset((c.a.label_key, c.b.label_key)) for c in cands if c.kind == "pair"}


def test_lexical_groups_pair_word_variants_across_papers(lib, w):
    w.note("lp_p1", [a("widget monotonicity"), a("gadget overlap")])
    w.note("lp_p2", [a("monotonicity of widgets"), a("gadget overlap rule")])
    w.note("lp_p3", [a("widget monotone response"), m("sprocket estimator")])
    rf.refresh(lib)
    got = cand.merge_candidates(cand.load(lib.conn))
    assert frozenset(("gadget overlap", "gadget overlap rule")) in pairs(got)
    assert all(c.a.family == c.b.family for c in got)
    # every pair belongs to a numbered lexical group
    assert all(c.group is not None for c in got)


def test_candidates_skip_decided_and_never_connect_pairs(lib, w):
    w.note("lp_p1", [a("gadget overlap"), a("widget monotone shape"), a("sprocket balance")])
    w.note("lp_p2", [a("gadget overlap rule"), a("widget monotone shape rule"), a("sprocket balance rule")])
    w.decide("unrelated", "gadget overlap", "gadget overlap rule", by="agent", family="assumption")
    w.decide("never_connect", "widget monotone shape", "widget monotone shape rule")
    rf.refresh(lib)
    got = pairs(cand.merge_candidates(cand.load(lib.conn)))
    assert frozenset(("gadget overlap", "gadget overlap rule")) not in got
    assert frozenset(("widget monotone shape", "widget monotone shape rule")) not in got
    assert frozenset(("sprocket balance", "sprocket balance rule")) in got


def test_candidates_skip_pairs_already_in_one_concept_and_pairs_inside_one_paper(lib, w):
    w.note("lp_p1", [a("gadget overlap"), a("gadget overlap rule"), a("sprocket balance")])
    w.note("lp_p2", [a("sprocket balance rule")])
    w.decide("merge", "sprocket balance", "sprocket balance rule", by="researcher", family="assumption")
    rf.refresh(lib)
    got = pairs(cand.merge_candidates(cand.load(lib.conn)))
    assert frozenset(("gadget overlap", "gadget overlap rule")) not in got  # one paper only
    assert frozenset(("sprocket balance", "sprocket balance rule")) not in got  # already merged


def test_needs_rejudge_decisions_are_judged_again(lib, w):
    w.note("lp_p1", [a("gadget overlap")])
    w.note("lp_p2", [a("gadget overlap rule")])
    w.decide("unrelated", "gadget overlap", "gadget overlap rule", by="agent", family="assumption",
             state="needs_rejudge")
    rf.refresh(lib)
    assert frozenset(("gadget overlap", "gadget overlap rule")) in pairs(cand.merge_candidates(cand.load(lib.conn)))


def test_big_lexical_groups_are_cut(lib, w):
    labels = [f"gadget overlap variant{i}" for i in range(20)]
    for i, lab in enumerate(labels):
        w.note(f"lp_q{i:02d}", [a(lab)])
    rf.refresh(lib)
    got = cand.merge_candidates(cand.load(lib.conn))
    by_group: dict[int, set[str]] = {}
    for c in got:
        by_group.setdefault(c.group, set()).update((c.a.label_key, c.b.label_key))
    assert by_group and max(len(v) for v in by_group.values()) <= cand.MAX_GROUP


def test_load_works_on_a_read_only_connection(tmp_path, w, lib):
    import sqlite3

    w.note("lp_p1", [a("gadget overlap")])
    w.note("lp_p2", [a("gadget overlap rule")])
    ro = sqlite3.connect(f"{lib.db_path.resolve().as_uri()}?mode=ro", uri=True)
    ro.row_factory = sqlite3.Row
    try:
        state = cand.load(ro)  # never refreshed: plans in memory, writes nothing
        assert frozenset(("gadget overlap", "gadget overlap rule")) in pairs(cand.merge_candidates(state))
    finally:
        ro.close()


# --- bridge ---------------------------------------------------------------------------------------

BRIDGE = dict(families=("assumption",))


def test_bridge_pairs_come_from_claim_text_neighbours(lib, w):
    w.note("lp_p1", [a("serial correlation", "Invented: residual draws of the sprocket series correlate over time.")])
    w.note("lp_p2", [a("autoregressive errors", "Invented: residual draws of the sprocket series follow lagged values.")])
    w.note("lp_p3", [a("gadget overlap", "Invented: every widget has positive treatment probability.")])
    rf.refresh(lib)
    got = cand.bridge_candidates(cand.load(lib.conn), **BRIDGE)
    assert frozenset(("autoregressive errors", "serial correlation")) in pairs(got)


def test_bridge_orders_sense_checks_first_then_citation_distant_pairs(lib, w):
    t = "Invented: the sprocket shock is unrelated to the widget draw."
    w.note("lp_p1", [a("exogeneity", t), a("sprocket independence", "Invented: cog lever flange pivots freely.")])
    w.note("lp_p2", [a("exogeneity", t), a("sprocket shock exclusion", "Invented: cog lever flange stays put.")])
    w.note("lp_p3", [a("sprocket shock separation", "Invented: cog lever flange rotates apart.")])
    with lib.write() as conn:  # p1 cites p2: distance 1; p3 is unconnected
        conn.execute("INSERT INTO citation_links (source_paper_id, target_paper_id, provenance, confidence, "
                     "created_at) VALUES ('lp_p1', 'lp_p2', 'manual_override', 1.0, '2026-01-01')")
    rf.refresh(lib)
    got = cand.bridge_candidates(cand.load(lib.conn), **BRIDGE)
    kinds = [c.kind for c in got]
    assert kinds[0] == "sense_check" and got[0].a == A("exogeneity")
    assert "sense_check" not in kinds[kinds.index("pair"):]
    dist = [cand.distance_rank(c.distance) for c in got if c.kind == "pair"]
    assert dist == sorted(dist, reverse=True)
    assert None in [c.distance for c in got if c.kind == "pair"]  # an infinite-distance pair exists


def test_bridge_skips_checked_senses_and_decided_pairs(lib, w):
    t = "Invented: the sprocket shock is unrelated to the widget draw."
    w.note("lp_p1", [a("exogeneity", t), a("sprocket independence", t)])
    w.note("lp_p2", [a("exogeneity", t), a("sprocket shock exclusion", t)])
    with lib.write() as conn:
        conn.execute("INSERT INTO concept_senses (family, label_key, sense_id, decided_by, state, decided_at) "
                     "VALUES ('assumption', 'exogeneity', 's0', 'agent', 'confirmed', '2026-01-01')")
    w.decide("link", "sprocket independence", "sprocket shock exclusion", by="agent", family="assumption")
    rf.refresh(lib)
    got = cand.bridge_candidates(cand.load(lib.conn), **BRIDGE)
    assert not [c for c in got if c.kind == "sense_check" and c.a.label_key == "exogeneity"]
    assert frozenset(("sprocket independence", "sprocket shock exclusion")) not in pairs(got)


def test_bridge_crosses_into_adjacent_families_only_for_links(lib, w):
    t = "Invented: lagged sprocket outputs predict the current sprocket output."
    w.note("lp_p1", [a("sprocket persistence", t)])
    w.note("lp_p2", [m("sprocket autoregression", t + " estimated")])
    rf.refresh(lib)
    got = cand.bridge_candidates(cand.load(lib.conn), families=("assumption",))
    cross = [c for c in got if c.kind == "pair" and c.a.family != c.b.family]
    assert cross and {cross[0].a.family, cross[0].b.family} == {"assumption", "method"}
    # the method pass does not repeat the assumption-method pair
    later = cand.bridge_candidates(cand.load(lib.conn), families=("method",))
    assert not [c for c in later if c.kind == "pair" and c.a.family != c.b.family]


def test_effective_ambiguous_list_library_wide_versus_graph_scoped(lib, w):
    from seedgraph.concepts import guardrails
    from seedgraph.library import graphs

    t = "Invented: the sprocket shock is unrelated to the widget draw."
    w.note("lp_p1", [a("exogeneity", t)])
    w.note("lp_p2", [a("exogeneity", t + " Again.")])
    w.note("lp_p3", [a("exogeneity", t + " Once more.")])
    rf.refresh(lib)
    gr = graphs.create_graph(lib, "invented_graph")
    for pid in ("lp_p1", "lp_p2"):
        graphs.set_membership(gr, pid, "member")
    gr.close()
    guardrails.mark_single_sense(lib, "exogeneity", "invented_graph")

    def checks(state):
        return [c for c in cand.bridge_candidates(state, **BRIDGE, sense_checks="ambiguous") if c.kind == "sense_check"]

    library_wide = cand.load(lib.conn)
    assert [c.a for c in checks(library_wide)] == [A("exogeneity")]   # library list: still ambiguous
    assert library_wide.senses[A("exogeneity")].papers == {"lp_p1", "lp_p2", "lp_p3"}
    scoped = cand.load(lib.conn, graph="invented_graph", library_root=lib.root)
    assert scoped.senses[A("exogeneity")].papers == {"lp_p1", "lp_p2"}  # the graph's slice
    assert not scoped.senses[A("exogeneity")].ambiguous and checks(scoped) == []
