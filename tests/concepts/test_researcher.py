"""Researcher concept decisions (stage 4 design §2.4, §7, §11): merge, link with the
researcher's own note, never connect / never merge, split a merge or link (records never
connect) and undo it, split a label into senses and undo that, the short queue, and the
frozen review sample with its marks. Invented text only."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter

import pytest

from seedgraph.concepts import refresh as rf
from seedgraph.concepts import researcher as rs
from seedgraph.concepts.model import SenseKey
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


def derived(lib) -> dict[str, list[tuple]]:
    return {t: [tuple(r) for r in lib.conn.execute(f"SELECT * FROM {t} ORDER BY 1, 2, 3")] for t in DERIVED}


def A(label: str, sense: str = "s0") -> SenseKey:
    return SenseKey("assumption", label, sense)


def _two_labels(w, lib):
    w.note("lp_p1", [a("widget overlap", "Invented: every widget has a twin.")])
    w.note("lp_p2", [a("gadget overlap", "Invented: every gadget has a twin.")])
    rf.refresh(lib)


# --- merge ----------------------------------------------------------------------------------------

def test_researcher_merge_joins_the_two_concepts(lib, w):
    _two_labels(w, lib)
    assert w.concept_of("assumption", "widget overlap") != w.concept_of("assumption", "gadget overlap")
    out = rs.merge(lib, A("widget overlap"), A("gadget overlap"), name="gadget overlap")
    assert w.concept_of("assumption", "widget overlap") == w.concept_of("assumption", "gadget overlap")
    row = lib.conn.execute("SELECT kind, decided_by, state, name FROM concept_decisions WHERE decision_id = ?",
                           (out.decision_id,)).fetchone()
    assert tuple(row) == ("merge", "researcher", "applied", "gadget overlap")
    assert w.concept(w.concept_of("assumption", "widget overlap"))["name"] == "gadget overlap"


def test_researcher_merge_refuses_across_families(lib, w):
    w.note("lp_p1", [a("widget overlap"), m("widget estimator")])
    rf.refresh(lib)
    with pytest.raises(rs.ResearcherError, match="famil"):
        rs.merge(lib, A("widget overlap"), SenseKey("method", "widget estimator"))


def test_researcher_merge_overrides_a_direct_never_merge(lib, w):
    _two_labels(w, lib)
    nm = w.decide("never_merge", "widget overlap", "gadget overlap")
    rs.merge(lib, A("widget overlap"), A("gadget overlap"))
    assert w.concept_of("assumption", "widget overlap") == w.concept_of("assumption", "gadget overlap")
    assert w.state(nm)[0] == "undone"


def test_researcher_merge_refuses_a_transitive_never_merge(lib, w):
    w.note("lp_p1", [a("widget overlap"), a("gadget overlap"), a("doohickey overlap")])
    rf.refresh(lib)
    w.decide("never_merge", "doohickey overlap", "widget overlap")
    rs.merge(lib, A("gadget overlap"), A("doohickey overlap"))
    with pytest.raises(rs.ResearcherError, match="never"):
        rs.merge(lib, A("widget overlap"), A("gadget overlap"))
    assert w.concept_of("assumption", "widget overlap") != w.concept_of("assumption", "gadget overlap")


# --- link and never connect ------------------------------------------------------------------------

def test_researcher_link_keeps_both_concepts_with_the_note_and_a_quote_per_side(lib, w):
    _two_labels(w, lib)
    with pytest.raises(rs.ResearcherError, match="note"):
        rs.link(lib, A("widget overlap"), A("gadget overlap"), note="  ")
    out = rs.link(lib, A("widget overlap"), A("gadget overlap"), note="Invented: widgets drifted into gadgets.")
    ca, cb = w.concept_of("assumption", "widget overlap"), w.concept_of("assumption", "gadget overlap")
    assert ca != cb
    links = lib.conn.execute("SELECT concept_a, concept_b, note, decided_by FROM concept_links").fetchall()
    assert [tuple(r) for r in links] == [(*sorted((ca, cb)), "Invented: widgets drifted into gadgets.",
                                          "researcher")]
    qa, qb = lib.conn.execute("SELECT quote_a_json, quote_b_json FROM concept_decisions WHERE decision_id = ?",
                              (out.decision_id,)).fetchone()
    assert {json.loads(qa)["paper_id"], json.loads(qb)["paper_id"]} == {"lp_p1", "lp_p2"}


def test_never_connect_drops_an_agent_link_and_undo_brings_it_back(lib, w):
    _two_labels(w, lib)
    agent = w.decide("link", "gadget overlap", "widget overlap", by="agent", family="assumption")
    rf.refresh(lib)
    assert lib.conn.execute("SELECT COUNT(*) FROM concept_links").fetchone()[0] == 1
    out = rs.never_connect(lib, A("widget overlap"), A("gadget overlap"))
    assert lib.conn.execute("SELECT COUNT(*) FROM concept_links").fetchone()[0] == 0
    assert w.state(agent)[0] == "undone"
    rs.undo(lib, out.decision_id)
    assert w.state(agent)[0] == "applied"
    assert lib.conn.execute("SELECT COUNT(*) FROM concept_links").fetchone()[0] == 1


# --- split a merge / undo (round trip) -------------------------------------------------------------

def test_split_an_agent_merge_records_never_connect_and_undo_round_trips(lib, w):
    _two_labels(w, lib)
    did = w.decide("merge", "gadget overlap", "widget overlap", by="agent", family="assumption")
    rf.refresh(lib)
    merged = derived(lib)
    assert w.concept_of("assumption", "widget overlap") == w.concept_of("assumption", "gadget overlap")
    out = rs.split_decision(lib, did)
    assert w.concept_of("assumption", "widget overlap") != w.concept_of("assumption", "gadget overlap")
    assert w.state(did)[0] == "undone"
    nc = lib.conn.execute("SELECT kind, decided_by, state FROM concept_decisions WHERE decision_id = ?",
                          (out.decision_id,)).fetchone()
    assert tuple(nc) == ("never_connect", "researcher", "applied")
    with pytest.raises(rs.ResearcherError):
        rs.split_decision(lib, did)
    rs.undo(lib, out.decision_id)
    assert w.state(did)[0] == "applied"
    assert derived(lib) == merged


def test_split_removes_an_applied_link(lib, w):
    _two_labels(w, lib)
    did = w.decide("link", "gadget overlap", "widget overlap", by="agent", family="assumption")
    rf.refresh(lib)
    rs.split_decision(lib, did)
    assert lib.conn.execute("SELECT COUNT(*) FROM concept_links").fetchone()[0] == 0


def test_agent_decisions_are_split_not_undone(lib, w):
    _two_labels(w, lib)
    did = w.decide("merge", "gadget overlap", "widget overlap", by="agent", family="assumption")
    rf.refresh(lib)
    with pytest.raises(rs.ResearcherError, match="split"):
        rs.undo(lib, did)


def test_undo_a_researcher_merge_restores_the_overridden_never_merge(lib, w):
    _two_labels(w, lib)
    nm = w.decide("never_merge", "widget overlap", "gadget overlap")
    before = derived(lib)
    out = rs.merge(lib, A("widget overlap"), A("gadget overlap"))
    rs.undo(lib, out.decision_id)
    assert w.state(nm) == ("applied", None)
    assert derived(lib) == before


# --- split a label into senses / undo (round trip) -------------------------------------------------

def test_split_label_assigns_claims_and_undo_round_trips(lib, w):
    c1 = w.note("lp_p1", [a("widget rank", "Invented: the widget matrix has full rank.")])["widget rank"]
    c2 = w.note("lp_p2", [a("widget rank", "Invented: widgets are ranked by size.")])["widget rank"]
    agent = w.decide("merge", "gadget rank", "widget rank", by="agent", family="assumption")
    rf.refresh(lib)
    before = derived(lib)
    rs.split_label(lib, "assumption", "widget rank",
                   [("full rank of a matrix", [c1]), ("an ordering by size", [c2])])
    rows = {r[0]: r[1] for r in lib.conn.execute("SELECT claim_id, sense_id FROM claim_concepts")}
    assert rows == {c1: "s1", c2: "s2"}
    assert w.concept_of("assumption", "widget rank", "s1") != w.concept_of("assumption", "widget rank", "s2")
    assert w.concept_of("assumption", "widget rank") is None
    glosses = [r[0] for r in lib.conn.execute("SELECT gloss FROM concepts WHERE family = 'assumption' "
                                              "ORDER BY gloss")]
    assert glosses == ["an ordering by size", "full rank of a matrix"]
    assert w.state(agent)[0] == "needs_rejudge"
    with pytest.raises(rs.ResearcherError, match="already split"):
        rs.split_label(lib, "assumption", "widget rank", [("x", [c1]), ("y", [c2])])
    rs.undo_label_split(lib, "assumption", "widget rank")
    assert derived(lib) == before
    assert lib.conn.execute("SELECT COUNT(*) FROM sense_assignments").fetchone()[0] == 2  # kept for history


def test_split_label_refuses_bad_input(lib, w):
    c1 = w.note("lp_p1", [a("widget rank")])["widget rank"]
    rf.refresh(lib)
    with pytest.raises(rs.ResearcherError, match="two senses"):
        rs.split_label(lib, "assumption", "widget rank", [("one", [c1])])
    with pytest.raises(rs.ResearcherError, match="gloss"):
        rs.split_label(lib, "assumption", "widget rank", [("", [c1]), ("two", [])])
    with pytest.raises(rs.ResearcherError, match="one sense"):
        rs.split_label(lib, "assumption", "widget rank", [("one", [c1]), ("two", [c1])])
    with pytest.raises(rs.ResearcherError, match="not claims"):
        rs.split_label(lib, "assumption", "widget rank", [("one", [c1]), ("two", ["cl_nope"])])


# --- the queue -------------------------------------------------------------------------------------

def _agent(lib, w, kind, x, y, *, state="queued", reason=None, batch="b_1", note=None, family="assumption"):
    did = w.decide(kind, x, y, by="agent", state=state, family=family)
    with lib.write() as conn:
        conn.execute("UPDATE concept_decisions SET veto_reason = ?, batch_id = ?, note = ?, confidence = 'medium' "
                     "WHERE decision_id = ?", (reason, batch, note, did))
    return did


def test_queue_lists_queued_pairs_and_senses_by_reach(lib, w):
    _two_labels(w, lib)
    w.note("lp_p3", [a("doohickey overlap"), a("widget overlap")])
    did = _agent(lib, w, "merge", "gadget overlap", "widget overlap", reason="low_confidence")
    unsure = _agent(lib, w, "unrelated", "doohickey overlap", "gadget overlap", reason="unsure")
    with lib.write() as conn:
        conn.execute("INSERT INTO concept_senses (family, label_key, sense_id, gloss, decided_by, state, decided_at) "
                     "VALUES ('assumption', 'widget overlap', 's0', NULL, 'agent', 'queued', '2026-01-01')")
    rf.refresh(lib)
    items = rs.queue(lib.conn)
    assert {i.item_id for i in items} == {did, unsure, "sense:assumption/widget overlap"}
    by_id = {i.item_id: i for i in items}
    assert by_id[did].actions == ("accept", "reject", "link")
    assert by_id[unsure].proposal == "unsure" and by_id[unsure].actions == ("merge", "link", "unrelated")
    assert by_id["sense:assumption/widget overlap"].actions == ("accept", "reject")
    assert items[0].item_id == "sense:assumption/widget overlap"  # widget overlap: 2 papers


def test_queue_accept_turns_the_proposal_into_a_researcher_merge(lib, w):
    _two_labels(w, lib)
    did = _agent(lib, w, "merge", "gadget overlap", "widget overlap", reason="reach")
    rf.refresh(lib)
    assert w.concept_of("assumption", "widget overlap") != w.concept_of("assumption", "gadget overlap")
    rs.settle(lib, did, "accept")
    assert w.concept_of("assumption", "widget overlap") == w.concept_of("assumption", "gadget overlap")
    row = lib.conn.execute("SELECT decided_by, state, batch_id FROM concept_decisions WHERE decision_id = ?",
                           (did,)).fetchone()
    assert tuple(row) == ("researcher", "applied", "b_1")
    assert rs.queue(lib.conn) == []


def test_queue_reject_records_unrelated(lib, w):
    _two_labels(w, lib)
    did = _agent(lib, w, "merge", "gadget overlap", "widget overlap", reason="reach")
    rf.refresh(lib)
    out = rs.settle(lib, did, "reject")
    assert w.state(did)[0] == "undone"
    kind, by = lib.conn.execute("SELECT kind, decided_by FROM concept_decisions WHERE decision_id = ?",
                                (out.decision_id,)).fetchone()
    assert (kind, by) == ("unrelated", "researcher")
    assert rs.queue(lib.conn) == []


def test_queue_link_instead_needs_the_researchers_note(lib, w):
    _two_labels(w, lib)
    did = _agent(lib, w, "merge", "gadget overlap", "widget overlap", reason="reach")
    rf.refresh(lib)
    with pytest.raises(rs.ResearcherError, match="note"):
        rs.settle(lib, did, "link")
    rs.settle(lib, did, "link", note="Invented: close, but gadgets are larger.")
    assert w.concept_of("assumption", "widget overlap") != w.concept_of("assumption", "gadget overlap")
    assert tuple(lib.conn.execute("SELECT note, decided_by FROM concept_links").fetchone()) == \
        ("Invented: close, but gadgets are larger.", "researcher")
    assert rs.queue(lib.conn) == []


def test_queue_unsure_cannot_be_accepted(lib, w):
    _two_labels(w, lib)
    did = _agent(lib, w, "unrelated", "gadget overlap", "widget overlap", reason="unsure")
    rf.refresh(lib)
    with pytest.raises(rs.ResearcherError, match="unsure"):
        rs.settle(lib, did, "accept")
    rs.settle(lib, did, "merge")
    assert w.concept_of("assumption", "widget overlap") == w.concept_of("assumption", "gadget overlap")


def test_queue_sense_items_accept_and_reject(lib, w):
    c1 = w.note("lp_p1", [a("widget rank", "Invented: full rank.")])["widget rank"]
    w.note("lp_p2", [a("widget rank", "Invented: ranked by size.")])
    rf.refresh(lib)
    w.split("assumption", "widget rank", {"s1": "full rank", "s2": "ordering"}, by="agent")
    with lib.write() as conn:
        conn.execute("UPDATE concept_senses SET state = 'queued' WHERE label_key = 'widget rank'")
    w.assign("assumption", "widget rank", c1, "s1", by="agent")
    rf.refresh(lib)
    assert w.concept_of("assumption", "widget rank") is not None  # a queued split is not applied
    rs.settle(lib, "sense:assumption/widget rank", "accept")
    assert w.concept_of("assumption", "widget rank", "s1") is not None
    assert w.concept_of("assumption", "widget rank") is None
    assert rs.queue(lib.conn) == []
    with pytest.raises(rs.ResearcherError):
        rs.settle(lib, "sense:assumption/widget rank", "reject")


# --- review sample ---------------------------------------------------------------------------------

def _many_agent_rows(lib, w, n_merge=40, n_link=30):
    labels = [f"widget {i:02d}" for i in range(n_merge + n_link + 1)]
    w.note("lp_p1", [a(x) for x in labels])
    ids = []
    for i in range(n_merge):
        ids.append(_agent(lib, w, "merge", labels[i], labels[i + 1], state="applied"))
    for i in range(n_merge, n_merge + n_link):
        ids.append(_agent(lib, w, "link", labels[i], labels[i + 1], state="applied", note="Invented note."))
    # rule rows and batch-less (migrated) agent rows never enter the sample
    w.decide("merge", "widget 00", "widget 70", by="rule", family="assumption")
    w.decide("merge", "widget 01", "widget 70", by="agent", family="assumption")
    rf.refresh(lib)
    return ids


def test_review_sample_is_deterministic_frozen_and_excludes_rule_and_migrated_rows(lib, w):
    ids = _many_agent_rows(lib, w)
    s1 = rs.create_review_sample(lib, n=50, seed=7)
    s2 = rs.create_review_sample(lib, n=50, seed=7)
    assert s1 == s2 and len(s1.decision_ids) == 50
    assert set(s1.decision_ids) <= set(ids)
    kinds = Counter(lib.conn.execute("SELECT kind FROM concept_decisions WHERE decision_id = ?",
                                     (d,)).fetchone()[0] for d in s1.decision_ids)
    assert kinds["merge"] > 0 and kinds["link"] > 0
    assert rs.create_review_sample(lib, n=50, seed=8).decision_ids != s1.decision_ids
    _agent(lib, w, "merge", "widget 69", "widget 70", state="applied")
    assert rs.review_sample(lib.conn, s1.sample_id) == s1  # frozen


def test_review_marks_are_stored_and_summarised(lib, w):
    _many_agent_rows(lib, w, n_merge=3, n_link=2)
    s = rs.create_review_sample(lib, n=50, seed=0)
    kind = {d: lib.conn.execute("SELECT kind FROM concept_decisions WHERE decision_id = ?", (d,)).fetchone()[0]
            for d in s.decision_ids}
    merges = [d for d in s.decision_ids if kind[d] == "merge"]
    links = [d for d in s.decision_ids if kind[d] == "link"]
    rs.mark(lib, s.sample_id, merges[0], "right")
    rs.mark(lib, s.sample_id, merges[1], "wrong", note="Invented: different things.")
    rs.mark(lib, s.sample_id, merges[2], "should_be_link")
    rs.mark(lib, s.sample_id, links[0], "right")
    rs.mark(lib, s.sample_id, links[1], "should_be_merge", note="Invented: same thing.")
    with pytest.raises(rs.ResearcherError, match="not a mark"):
        rs.mark(lib, s.sample_id, links[0], "should_be_link")
    stored = lib.conn.execute("SELECT mark, note FROM concept_reviews WHERE decision_id = ?",
                              (links[1],)).fetchone()
    assert stored[0] == "wrong" and stored[1].startswith(rs.SHOULD_BE_MERGE_PREFIX)
    assert rs.marks(lib.conn, s.sample_id)[links[1]] == ("should_be_merge", "Invented: same thing.")
    summ = rs.review_summary(lib.conn, s.sample_id)
    assert summ.marked == 5 and summ.size == 5
    assert summ.by_mark == {"right": 2, "should_be_link": 1, "should_be_merge": 1, "wrong": 1}
    assert summ.by_verdict["merge"] == {"right": 1, "should_be_link": 1, "wrong": 1}
    assert summ.wrong_merges == (merges[1],)
    assert summ.gate_passed is False  # 1 wrong of 3 merges is over 10%
    rs.mark(lib, s.sample_id, merges[1], "right")  # re-marking replaces
    assert rs.review_summary(lib.conn, s.sample_id).gate_passed is True


def test_the_reviews_table_has_no_should_be_merge_mark(lib):
    with pytest.raises(sqlite3.IntegrityError), lib.write() as conn:
        conn.execute("INSERT INTO concept_reviews (sample_id, decision_id, mark, marked_at) "
                     "VALUES ('rs_x', 'cd_x', 'should_be_merge', '2026-01-01')")


# --- settings lists --------------------------------------------------------------------------------

def test_never_merge_list_add_and_remove(lib, w):
    _two_labels(w, lib)
    out = rs.add_never_merge(lib, "Widget overlap", "gadget overlap", note="Invented reason.")
    assert w.state(out.decision_id)[0] == "applied"
    rs.remove_never_merge(lib, out.decision_id)
    assert w.state(out.decision_id)[0] == "undone"
    with pytest.raises(rs.ResearcherError):
        rs.remove_never_merge(lib, out.decision_id)


def test_ambiguous_list_edits_refresh_the_sense_check_flag(lib, w):
    w.note("lp_p1", [a("widget overlap")])
    rf.refresh(lib)
    cid = w.concept_of("assumption", "widget overlap")
    assert w.concept(cid)["needs_sense_check"] == 0
    rs.edit_ambiguous(lib, "add", "widget overlap")
    assert w.concept(cid)["needs_sense_check"] == 1
    rs.edit_ambiguous(lib, "single_sense", "widget overlap", graph="widgets")
    rows = lib.conn.execute("SELECT scope, effect FROM ambiguous_labels WHERE pattern = 'widget overlap' "
                            "ORDER BY scope").fetchall()
    assert [tuple(r) for r in rows] == [("", "ambiguous"), ("widgets", "single_sense")]
    rs.edit_ambiguous(lib, "remove", "widget overlap")
    assert w.concept(cid)["needs_sense_check"] == 0
    with pytest.raises(rs.ResearcherError):
        rs.edit_ambiguous(lib, "single_sense", "widget overlap")
