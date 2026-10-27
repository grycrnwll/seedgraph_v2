"""Deterministic refresh (stage 4 design §1.4-§1.5, §2.4): rule merges, migrated decisions,
byte-stable rebuilds, staleness and sense assignments across re-notes. Invented text only."""

from __future__ import annotations

import sqlite3

import pytest

from seedgraph.concepts import refresh as rf
from seedgraph.library import store

from ._world import World, a, m

DERIVED = ("claim_concepts", "concepts", "concept_members", "concept_links")


@pytest.fixture
def lib(tmp_path):
    library = store.open_library_at(tmp_path / "library")
    yield library
    library.close()


@pytest.fixture
def w(lib):
    return World(lib)


def dump(lib) -> dict[str, list[tuple]]:
    out = {t: [tuple(r) for r in lib.conn.execute(f"SELECT * FROM {t} ORDER BY 1, 2, 3")] for t in DERIVED}
    out["decisions"] = [tuple(r) for r in lib.conn.execute(
        "SELECT decision_id, kind, family, label_a, label_b, decided_by, state, veto_reason "
        "FROM concept_decisions ORDER BY decision_id")]
    return out


# --- rule merges ----------------------------------------------------------------------------------

def test_suffix_fold_needs_closure(lib, w):
    w.note("lp_p1", [a("parallel trends"), a("parallel trends assumption"), a("overlap condition")])
    w.note("lp_p2", [m("widget weighting"), m("widget weighting hypothesis")])
    rep = rf.refresh(lib)
    assert w.concept_of("assumption", "parallel trends") == w.concept_of("assumption", "parallel trends assumption")
    # "overlap" is not itself a label in the family: no fold onto an invented name
    assert w.concept_of("assumption", "overlap condition") is not None
    assert w.concept_of("assumption", "overlap") is None
    assert w.concept_of("method", "widget weighting") == w.concept_of("method", "widget weighting hypothesis")
    assert rep.rule_merges_added == 2
    rows = lib.conn.execute("SELECT family, label_a, label_b, decided_by, state FROM concept_decisions "
                            "WHERE decided_by = 'rule' ORDER BY family").fetchall()
    assert [tuple(r) for r in rows] == [
        ("assumption", "parallel trends", "parallel trends assumption", "rule", "applied"),
        ("method", "widget weighting", "widget weighting hypothesis", "rule", "applied")]


def test_acronym_fold_within_a_family_only(lib, w):
    w.note("lp_p1", [m("iv"), m("instrumental variables"), a("sutva"),
                     {"type": "result", "label": "instrumental variables", "text": "Invented result."}])
    rf.refresh(lib)
    assert w.concept_of("method", "iv") == w.concept_of("method", "instrumental variables")
    assert w.concept_of("result", "instrumental variables") != w.concept_of("method", "instrumental variables")


def test_rule_merges_skip_ambiguous_labels(lib, w):
    w.note("lp_p1", [a("exogeneity"), a("exogeneity assumption"), a("strict exogeneity"),
                     a("strict exogeneity assumption")])
    rf.refresh(lib)
    assert w.concept_of("assumption", "exogeneity") != w.concept_of("assumption", "exogeneity assumption")
    # "strict exogeneity" is not matched by the head pattern, so its suffix fold applies
    assert w.concept_of("assumption", "strict exogeneity") == \
        w.concept_of("assumption", "strict exogeneity assumption")
    cid = w.concept_of("assumption", "exogeneity")
    assert w.concept(cid)["needs_sense_check"] == 1
    assert w.concept(w.concept_of("assumption", "strict exogeneity"))["needs_sense_check"] == 0


def test_a_split_rule_merge_stays_split(lib, w):
    w.note("lp_p1", [a("parallel trends"), a("parallel trends assumption")])
    rf.refresh(lib)
    with lib.write() as conn:  # "split" in Recent merges: undone + never_connect
        conn.execute("UPDATE concept_decisions SET state = 'undone' WHERE decided_by = 'rule'")
    w.decide("never_connect", "parallel trends", "parallel trends assumption", family="assumption")
    rf.refresh(lib)
    assert w.concept_of("assumption", "parallel trends") != w.concept_of("assumption", "parallel trends assumption")


def test_stoplist_blocks_rule_merges(lib, w):
    w.note("lp_p1", [m("bwp"), m("big widget panel"), m("base widget panel"), m("widget model 2"),
                     m("widget model 2 assumption")])
    rf.refresh(lib)
    # both expansions fold onto the acronym, but big (large) and base conflict: never one concept
    assert w.concept_of("method", "big widget panel") != w.concept_of("method", "base widget panel")
    acronym = w.concept_of("method", "bwp")
    assert [w.concept_of("method", x) == acronym for x in ("big widget panel", "base widget panel")].count(True) == 1
    # the suffix fold keeps the same version token: allowed
    assert w.concept_of("method", "widget model 2") == w.concept_of("method", "widget model 2 assumption")


# --- migrated decisions ---------------------------------------------------------------------------

def test_migrated_decisions_applied_and_vetoed_rows_listed(lib, w):
    w.note("lp_p1", [a("parallel trends"), a("common trends"), a("selection on observables"),
                     a("unconfoundedness"), m("widget estimator v1"), m("widget estimator v2"),
                     m("two step widgets")])
    w.note("lp_p2", [a("no anticipation"), a("exogeneity"), a("ignorability")])
    pend = dict(state="pending_apply", origin="v2:test_proj")
    ok = w.decide("merge", "common trends", "parallel trends", by="agent", **pend)
    res = w.decide("merge", "selection on observables", "unconfoundedness", by="researcher", **pend)
    never = w.decide("never_merge", "ignorability", "no anticipation", by="researcher", **pend)
    fence = w.decide("merge", "no anticipation", "two step widgets", by="agent", **pend)
    stop = w.decide("merge", "widget estimator v1", "widget estimator v2", by="agent", **pend)
    amb = w.decide("merge", "exogeneity", "ignorability", by="agent", **pend)
    absent = w.decide("merge", "never seen label", "parallel trends", by="agent", **pend)

    rep = rf.refresh(lib)
    assert w.state(ok) == ("applied", None)
    assert w.state(res) == ("applied", None)
    assert w.state(never) == ("applied", None)
    assert w.state(fence) == ("vetoed", "type_fence")
    assert w.state(stop) == ("vetoed", "stoplist")
    assert w.state(amb) == ("vetoed", "ambiguous_label")
    assert w.state(absent) == ("pending_apply", None)
    assert {o.decision_id: o.reason for o in rep.migrated_vetoed} == {
        fence: "type_fence", stop: "stoplist", amb: "ambiguous_label"}
    assert [o.decision_id for o in rep.migrated_pending] == [absent]
    assert w.concept_of("assumption", "common trends") == w.concept_of("assumption", "parallel trends")
    assert w.concept_of("method", "widget estimator v1") != w.concept_of("method", "widget estimator v2")


def test_researcher_merge_across_families_applies_in_each_family_with_both_labels(lib, w):
    w.note("lp_p1", [a("widget rank"), a("widget order"), m("widget rank"), m("widget order"),
                     {"type": "data", "label": "widget rank", "text": "Invented data."}])
    w.decide("merge", "widget order", "widget rank", by="researcher", state="pending_apply")
    rf.refresh(lib)
    assert w.concept_of("assumption", "widget order") == w.concept_of("assumption", "widget rank")
    assert w.concept_of("method", "widget order") == w.concept_of("method", "widget rank")
    assert w.concept_of("assumption", "widget rank") != w.concept_of("method", "widget rank")


def test_contradicting_researcher_rows_newest_wins(lib, w):
    w.note("lp_p1", [a("common trends"), a("parallel trends")])
    old = w.decide("merge", "common trends", "parallel trends", state="pending_apply", at="2024-01-01T00:00:00")
    new = w.decide("never_merge", "common trends", "parallel trends", state="pending_apply",
                   at="2025-01-01T00:00:00")
    rep = rf.refresh(lib)
    assert w.state(old)[0] == "undone" and w.state(new)[0] == "applied"
    assert [o.decision_id for o in rep.superseded] == [old]
    assert w.concept_of("assumption", "common trends") != w.concept_of("assumption", "parallel trends")


# --- determinism and staleness --------------------------------------------------------------------

def test_rebuild_is_byte_stable(lib, w):
    w.note("lp_p1", [a("parallel trends"), a("parallel trends assumption"), a("common trends"), m("did")])
    w.note("lp_p2", [m("difference in differences"), a("parallel trends"), m("iv"), m("instrumental variables")])
    w.decide("merge", "common trends", "parallel trends", by="agent", state="applied")
    w.decide("link", "common trends", "no anticipation", by="agent", state="applied")
    rep = rf.refresh(lib, now="2026-02-01T00:00:00+00:00")
    assert rep.rule_merges_added == 3  # suffix fold, did, iv: every rule row is stored
    stored = {tuple(r) for r in lib.conn.execute(
        "SELECT label_a, label_b FROM concept_decisions WHERE decided_by = 'rule'")}
    assert stored == {("parallel trends", "parallel trends assumption"), ("did", "difference in differences"),
                      ("instrumental variables", "iv")}
    first = dump(lib)
    rf.refresh(lib, now="2026-03-01T00:00:00+00:00")
    assert dump(lib) == first
    # a fresh plan against the same stored state writes nothing different
    p = rf.plan(lib.conn, now="2026-04-01T00:00:00+00:00")
    assert not p.new_rule_decisions and not p.state_updates and not p.assignment_rewrites
    assert p.concepts == [tuple(r) for r in lib.conn.execute("SELECT * FROM concepts ORDER BY concept_id")]


def test_out_of_date_tracks_claim_changes_and_marks(lib, w):
    w.note("lp_p1", [a("parallel trends")])
    assert rf.out_of_date(lib.conn).changed == ("lp_p1",)
    rf.refresh(lib)
    assert not rf.out_of_date(lib.conn).stale
    w.note("lp_p2", [a("parallel trends")])
    with lib.write() as conn:
        conn.execute("INSERT INTO concept_refresh (paper_id, reason, since) VALUES ('lp_p1', 'invented', 'x')")
    ood = rf.out_of_date(lib.conn)
    assert ood.changed == ("lp_p2",) and ood.marked == ("lp_p1",) and ood.papers == ("lp_p1", "lp_p2")
    assert rf.refresh_if_stale(lib) is not None
    assert rf.refresh_if_stale(lib) is None


def test_links_join_concepts_and_vanish_inside_one_concept(lib, w):
    w.note("lp_p1", [a("serial correlation"), m("autoregression"), a("common trends"), a("parallel trends")])
    w.decide("link", "autoregression", "serial correlation", by="agent")  # family NULL: no common family
    with lib.write() as conn:
        conn.execute("INSERT INTO concept_decisions (decision_id, kind, family, family_b, label_a, label_b, note, "
                     "decided_by, state, decided_at) VALUES ('cd_drift', 'link', 'method', 'assumption', "
                     "'autoregression', 'serial correlation', 'drift: invented history note', 'researcher', "
                     "'applied', '2026-01-01')")
    w.decide("link", "common trends", "parallel trends", family="assumption", by="agent")
    w.decide("merge", "common trends", "parallel trends", family="assumption", by="researcher")
    rf.refresh(lib)
    links = [tuple(r) for r in lib.conn.execute("SELECT decision_id, note FROM concept_links")]
    assert links == [("cd_drift", "drift: invented history note")]


# --- sense assignments ----------------------------------------------------------------------------

def test_sense_assignments_survive_a_lightly_reworded_renote(lib, w):
    ids = w.note("lp_p1", [m("identification", "Identification comes from the invented lottery draw.",
                             start=10),
                           a("parallel trends")])
    ids2 = w.note("lp_p2", [m("identification", "Identification means the widget system has one solution.")])
    w.split("method", "identification", {"s1": "source of exogenous variation (invented)",
                                         "s2": "unique solution of a system (invented)"})
    w.assign("method", "identification", ids["identification"], "s1")
    w.assign("method", "identification", ids2["identification"], "s2")
    rf.refresh(lib)
    assert lib.conn.execute("SELECT sense_id FROM claim_concepts WHERE claim_id = ?",
                            (ids["identification"],)).fetchone()[0] == "s1"

    # re-note p1: new claim ids, lightly reworded text, a different quote position
    new = w.note("lp_p1", [m("identification", "Identification comes from the invented lottery draws.",
                             start=400),
                           a("parallel trends"), m("identification strategy", "An unrelated invented claim.")])
    rep = rf.refresh(lib)
    assert rep.reassigned == 1
    row = lib.conn.execute("SELECT sense_id, concept_id FROM claim_concepts WHERE claim_id = ?",
                           (new["identification"],)).fetchone()
    assert row[0] == "s1" and row[1] == w.concept_of("method", "identification", "s1")
    assert rf.plan(lib.conn).assignment_rewrites == []  # re-keyed: the next refresh is a no-op

    # a new, unassigned claim under the split label needs a sense
    w.note("lp_p3", [m("identification", "A third invented use of the word.")])
    rep = rf.refresh(lib)
    assert rep.needs_sense_claims == 1
    assert lib.conn.execute("SELECT COUNT(*) FROM claim_concepts WHERE sense_id IS NULL "
                            "AND concept_id IS NULL").fetchone()[0] == 1


# --- dry run --------------------------------------------------------------------------------------

def test_plan_writes_nothing_on_a_read_only_connection(tmp_path, lib, w):
    w.note("lp_p1", [a("parallel trends"), a("parallel trends assumption")])
    w.decide("merge", "common trends", "parallel trends", by="agent", state="pending_apply")
    lib.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    path = lib.db_path
    ro = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        for t in ("claim_concepts", "concepts", "concept_senses", "sense_assignments", "ambiguous_labels"):
            assert ro.execute("SELECT COUNT(*) FROM " + t).fetchone()[0] == 0
        p = rf.plan(ro)
        assert p.seed and p.report.rule_merges_added == 1 and p.report.concepts == 1
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("DELETE FROM concepts")
    finally:
        ro.close()
