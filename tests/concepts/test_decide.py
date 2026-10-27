"""Deciding agent answers (stage 4 design §2.4, §3.5): the unclear rules, researcher
override, save-time conflicts, split invalidation. Invented text only."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from seedgraph.concepts import decide
from seedgraph.concepts import refresh as rf
from seedgraph.library import store
from seedgraph.work import routine
from seedgraph.work.claims import WorkerRef
from seedgraph.work.tasks_concepts import load_manifest

from ._world import World, a

T0 = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)
W1 = WorkerRef("interactive", "w1", client="claude")


@pytest.fixture
def lib(tmp_path):
    library = store.open_library_at(tmp_path / "library")
    yield library
    library.close()


@pytest.fixture
def w(lib):
    return World(lib)


# --- the unclear rules (pure) ---------------------------------------------------------------------

@pytest.mark.parametrize("verdict, conf, kw, expected", [
    ("merge", "high", {}, ("applied", None)),
    ("unsure", "high", {}, ("queued", "unsure")),
    ("link", "low", {}, ("queued", "low_confidence")),
    ("merge", "medium", {"ambiguous": True}, ("queued", "ambiguous")),
    ("link", "high", {"ambiguous": True}, ("applied", None)),
    ("merge", "high", {"papers_a": 5, "papers_b": 5}, ("queued", "reach")),
    ("merge", "high", {"papers_a": 5, "papers_b": 4}, ("applied", None)),
    ("link", "high", {"papers_a": 9, "papers_b": 9}, ("applied", None)),     # links never held for reach
    ("split", "high", {"papers_a": 9, "papers_b": 9}, ("applied", None)),    # nor splits
    ("unrelated", "medium", {"disagreement": True}, ("queued", "disagreement")),
    ("merge", "high", {"save_veto": "never_merge"}, ("queued", "save_veto:never_merge")),
])
def test_unclear_classification(verdict, conf, kw, expected):
    assert decide.classify(verdict, conf, **kw) == expected
    assert decide.REACH_MIN == 5


def _pair_manifest(label_a, key_a, label_b, key_b, family_b="assumption"):
    def side(label, key, family, ref):
        return {"family": family, "label_key": key, "sense_id": "s0", "label": label, "papers": 1,
                "ambiguous": False, "refs": {ref: {"paper_id": "lp_x", "claim_id": ref, "span_id": None,
                                                   "claim_fingerprint": ref}}}
    return {"items": [{"item": 1, "kind": "pair", "family": "assumption", "group": None, "distance": None,
                       "reason": None, "a": side(label_a, key_a, "assumption", "c1"),
                       "b": side(label_b, key_b, family_b, "c2"), "literature": None}]}


SPLIT = {"items": [{"item": 1, "verdict": "split", "confidence": "high",
                    "senses": [{"gloss": "Invented: one", "claims": ["c1"]}, {"gloss": "Invented: two", "claims": ["c2"]}]}]}


def test_a_split_on_a_pair_is_allowed_when_the_labels_are_equal_after_label_normalisation():
    manifest = _pair_manifest("Weak Sprocket-Balance", "Weak Sprocket-Balance", "weak sprocket balance",
                              "weak sprocket balance")
    assert decide.validate(manifest, SPLIT) == []


def test_a_split_on_a_pair_of_one_label_is_saved_as_senses_of_that_label(lib, w):
    view, msg, manifest = merge_batch(lib, w)
    item = manifest["items"][0]
    variant = item["a"]["label"].upper().replace(" ", "-")
    item["b"]["label"], item["b"]["label_key"] = variant, variant
    ra, rb = list(item["a"]["refs"])[0], list(item["b"]["refs"])[0]
    ans = {"items": [{"item": 1, "verdict": "split", "confidence": "high",
                      "senses": [{"gloss": "Invented: one", "claims": [ra]}, {"gloss": "Invented: two", "claims": [rb]}]}]}
    assert decide.validate(manifest, ans, decide.Guard.load(lib.conn, None)) == []
    with lib.write() as conn:
        rep = decide.save_package(conn, manifest, ans)
    assert rep.by_verdict == {"split": 1} and rep.applied == 1
    rows = lib.conn.execute("SELECT label_key, sense_id, state FROM concept_senses WHERE decided_by = 'agent' "
                            "ORDER BY sense_id").fetchall()
    assert [tuple(r) for r in rows] == [(item["a"]["label_key"], "s1", "confirmed"),
                                        (item["a"]["label_key"], "s2", "confirmed")]


def test_a_split_on_a_pair_with_different_labels_or_families_is_refused():
    for manifest in (_pair_manifest("weak sprocket balance", "weak sprocket balance", "weak sprocket rule",
                                    "weak sprocket rule"),
                     _pair_manifest("weak sprocket balance", "weak sprocket balance", "weak sprocket balance",
                                    "weak sprocket balance", family_b="method")):
        errs = decide.validate(manifest, SPLIT)
        assert errs and "split on a pair needs both sides to share one label" in errs[0]


def _concepts_x_y_merged_and_z(lib, w):
    w.note("lp_p1", [a("sprocket monotone"), a("gadget drift")])
    w.note("lp_p2", [a("sprocket monotone rule"), a("gadget drift")])
    w.decide("unrelated", "gadget drift", "sprocket monotone", by="agent", family="assumption")
    w.decide("merge", "sprocket monotone", "sprocket monotone rule", by="researcher", family="assumption")
    rf.refresh(lib)


def _save_one(lib, x, z, ans):
    with lib.write() as conn:
        return decide.save_package(conn, _pair_manifest(x, x, z, z), {"items": [{"item": 1, **ans}]})


def test_a_different_verdict_on_another_member_pair_of_two_decided_concepts_is_queued(lib, w):
    _concepts_x_y_merged_and_z(lib, w)
    rep = _save_one(lib, "sprocket monotone rule", "gadget drift",
                    {"verdict": "merge", "confidence": "high", "name": "gadget drift"})
    assert rep.queue_reasons == {"disagreement": 1} and rep.applied == 0


def test_the_same_verdict_on_another_member_pair_of_two_decided_concepts_is_dropped(lib, w):
    _concepts_x_y_merged_and_z(lib, w)
    rep = _save_one(lib, "sprocket monotone rule", "gadget drift", {"verdict": "unrelated", "confidence": "high"})
    assert rep.dropped == 1 and rep.feedback[0]["reason"] == "already_decided"
    assert len(agent_rows(lib)) == 1


def test_a_link_or_unrelated_verdict_inside_one_concept_is_queued(lib, w):
    # merged by an agent (a researcher merge would drop the verdict as researcher_decided)
    w.note("lp_p1", [a("sprocket monotone")])
    w.note("lp_p2", [a("sprocket monotone rule")])
    w.decide("merge", "sprocket monotone", "sprocket monotone rule", by="agent", family="assumption",
             name="sprocket monotone")
    rf.refresh(lib)
    rep = _save_one(lib, "sprocket monotone", "sprocket monotone rule", {"verdict": "unrelated", "confidence": "high"})
    assert rep.queue_reasons == {"disagreement": 1} and rep.applied == 0


def test_a_pair_on_the_default_sense_of_a_label_split_since_the_package_was_frozen_is_dropped(lib, w):
    _concepts_x_y_merged_and_z(lib, w)
    with lib.write() as conn:
        for sid in ("s1", "s2"):
            conn.execute("INSERT INTO concept_senses (family, label_key, sense_id, gloss, decided_by, state, "
                         "decided_at) VALUES ('assumption', 'gadget drift', ?, 'Invented.', 'agent', 'confirmed', "
                         "'2026-01-01')", (sid,))
    rep = _save_one(lib, "sprocket monotone rule", "gadget drift", {"verdict": "link", "confidence": "high",
                                                                     "note": "Invented."})
    assert rep.dropped == 1 and rep.feedback[0]["reason"] == "label_split" and rep.applied == 0


def test_two_items_of_one_package_on_the_same_concept_pair_do_not_both_apply(lib, w):
    w.note("lp_p1", [a("sprocket monotone"), a("gadget drift")])
    w.note("lp_p2", [a("sprocket monotone rule"), a("gadget drift")])
    w.decide("merge", "sprocket monotone", "sprocket monotone rule", by="researcher", family="assumption")
    rf.refresh(lib)
    first = _pair_manifest("sprocket monotone", "sprocket monotone", "gadget drift", "gadget drift")
    second = _pair_manifest("sprocket monotone rule", "sprocket monotone rule", "gadget drift", "gadget drift")
    second["items"][0]["item"] = 2
    manifest = {"items": first["items"] + second["items"]}
    with lib.write() as conn:
        rep = decide.save_package(conn, manifest, {"items": [
            {"item": 1, "verdict": "unrelated", "confidence": "high"},
            {"item": 2, "verdict": "merge", "confidence": "high", "name": "gadget drift"}]})
    assert rep.applied == 1 and rep.queue_reasons == {"disagreement": 1}


# --- through a real package -----------------------------------------------------------------------

def merge_batch(lib, w, left="sprocket balance", right="sprocket balance rule"):
    w.note("lp_p1", [a(left, "Invented: sprocket weights equalise the widget groups.")])
    w.note("lp_p2", [a(right, "Invented: sprocket balance holds after weighting.")])
    view = routine.create_batch(lib, "concepts:merge", now=T0)
    reply = routine.claim_next(lib, view.batch_id, W1, now=T0)
    msg = reply.message
    manifest = load_manifest(lib, view.batch_id, msg["subject_id"])
    return view, msg, manifest


def agent_rows(lib):
    return [dict(r) for r in lib.conn.execute(
        "SELECT kind, state, veto_reason, name FROM concept_decisions WHERE decided_by = 'agent' ORDER BY kind")]


def merge_errors(lib, manifest, guard=None):
    guard = guard or decide.Guard.load(lib.conn, None)
    item = manifest["items"][0]
    return decide.validate(manifest, {"items": [
        {"item": item["item"], "verdict": "merge", "confidence": "high", "name": item["a"]["label"]}]}, guard)


def test_a_stoplist_veto_names_the_size_word_on_each_side(lib, w):
    view, msg, manifest = merge_batch(lib, w, left="large sprocket panel", right="small sprocket panel")
    (item,) = manifest["items"]
    la, lb = item["a"]["label"], item["b"]["label"]
    big, other = (la, lb) if "large" in la else (lb, la)
    assert merge_errors(lib, manifest) == [
        f"item 1: merge vetoed: the labels differ in a size word: {big!r} has 'large', {other!r} has 'small'; "
        "answer link, unrelated or unsure"]


def test_a_size_word_only_one_label_has_is_no_veto(lib, w):
    view, msg, manifest = merge_batch(lib, w, left="sprocket drift",
                                      right="sprocket drift (large T)")
    assert merge_errors(lib, manifest) == []


def test_a_type_fence_veto_names_both_families(lib, w):
    view, msg, manifest = merge_batch(lib, w)
    manifest["items"][0]["b"]["family"] = "method"
    la, lb = manifest["items"][0]["a"]["label"], manifest["items"][0]["b"]["label"]
    assert merge_errors(lib, manifest) == [
        f"item 1: merge vetoed: different families: {la!r} is an assumption, {lb!r} is a method, and merges "
        "stay within one family; answer link, unrelated or unsure"]


def test_a_never_merge_veto_names_the_pair(lib, w):
    view, msg, manifest = merge_batch(lib, w)
    w.decide("never_merge", "sprocket balance", "sprocket balance rule", by="researcher")
    assert merge_errors(lib, manifest) == [
        "item 1: merge vetoed: 'sprocket balance' and 'sprocket balance rule' are a never-merge pair; "
        "answer link, unrelated or unsure"]


def test_a_never_merge_veto_through_another_member_says_so(lib, w):
    view, msg, manifest = merge_batch(lib, w)
    w.note("lp_p5", [a("sprocket parity", "Invented: sprocket parity.")])
    w.decide("merge", "sprocket balance rule", "sprocket parity", by="researcher", family="assumption")
    w.decide("never_merge", "sprocket balance", "sprocket parity", by="researcher")
    rf.refresh(lib)
    assert merge_errors(lib, manifest) == [
        "item 1: merge vetoed: 'sprocket balance' and 'sprocket parity' are a never-merge pair, and this merge "
        "would put them in one concept; answer link, unrelated or unsure"]


def test_an_unsure_answer_is_stored_queued_so_the_pair_is_not_judged_again(lib, w):
    view, msg, manifest = merge_batch(lib, w)
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"items": [
        {"item": 1, "verdict": "unsure", "confidence": "low", "why": "Invented: too short."}]}, now=T0)
    assert res.status == "accepted"
    assert agent_rows(lib) == [{"kind": "unrelated", "state": "queued", "veto_reason": "unsure", "name": None}]


def test_an_ambiguous_label_without_high_confidence_is_queued(lib, w):
    w.note("lp_p3", [a("selection", "Invented: widgets chose their gadgets.")])
    w.note("lp_p4", [a("selection rule", "Invented: gadgets were assigned by a widget rule.")])
    view = routine.create_batch(lib, "concepts:merge", now=T0)
    msg = routine.claim_next(lib, view.batch_id, W1, now=T0).message
    manifest = load_manifest(lib, view.batch_id, msg["subject_id"])
    (item,) = manifest["items"]
    qa, qb = list(item["a"]["refs"])[0], list(item["b"]["refs"])[0]
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"items": [
        {"item": 1, "verdict": "link", "confidence": "medium", "note": "Invented: self versus rule.",
         "quote_a": qa, "quote_b": qb}]}, now=T0)
    assert res.status == "accepted"
    assert agent_rows(lib) == [{"kind": "link", "state": "queued", "veto_reason": "ambiguous", "name": None}]
    assert lib.conn.execute("SELECT COUNT(*) FROM concept_links").fetchone()[0] == 0  # queued: not applied


def test_researcher_decisions_override_agent_proposals(lib, w):
    view, msg, manifest = merge_batch(lib, w)
    w.decide("unrelated", "sprocket balance", "sprocket balance rule", by="researcher", family="assumption")
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"items": [
        {"item": 1, "verdict": "merge", "confidence": "high", "name": "sprocket balance"}]}, now=T0)
    assert res.status == "accepted"
    assert [x["reason"] for x in res.warnings] == ["researcher_decided"]
    assert agent_rows(lib) == []  # dropped, never queued
    assert w.concept_of("assumption", "sprocket balance") != w.concept_of("assumption", "sprocket balance rule")
    event = lib.conn.execute("SELECT detail_json FROM batch_events WHERE batch_id = ? AND kind = 'concepts_saved'",
                             (view.batch_id,)).fetchone()[0]
    assert json.loads(event)["dropped"] == 1


def test_a_save_time_conflict_goes_to_the_queue(lib, w):
    view, msg, manifest = merge_batch(lib, w)
    # concurrently: a researcher merge joins the rule label to a label that may never merge with the left one
    w.note("lp_p5", [a("sprocket parity", "Invented: sprocket parity.")])
    w.decide("merge", "sprocket balance rule", "sprocket parity", by="researcher", family="assumption")
    w.decide("never_merge", "sprocket balance", "sprocket parity", by="researcher")
    rf.refresh(lib)
    with lib.write() as conn:
        rep = decide.save_package(conn, manifest, {"items": [
            {"item": 1, "verdict": "merge", "confidence": "high", "name": "sprocket balance"}]},
            batch_id=view.batch_id)
    assert rep.queued == 1 and rep.vetoed == 1 and rep.applied == 0
    assert agent_rows(lib) == [{"kind": "merge", "state": "queued", "veto_reason": "save_veto:never_merge",
                                "name": "sprocket balance"}]
    assert w.concept_of("assumption", "sprocket balance") != w.concept_of("assumption", "sprocket balance rule")


def test_disagreeing_with_an_earlier_agent_decision_is_queued(lib, w):
    view, msg, manifest = merge_batch(lib, w)
    w.decide("link", "sprocket balance", "sprocket balance rule", by="agent", family="assumption")
    with lib.write() as conn:
        rep = decide.save_package(conn, manifest, {"items": [
            {"item": 1, "verdict": "merge", "confidence": "high", "name": "sprocket balance"}]})
    assert rep.queue_reasons == {"disagreement": 1}


def test_a_cited_claim_that_is_no_longer_live_refuses_the_item(lib, w):
    view, msg, manifest = merge_batch(lib, w)
    (item,) = manifest["items"]
    qa, qb = list(item["a"]["refs"])[0], list(item["b"]["refs"])[0]
    w.note("lp_p1", [a("sprocket balance", "Invented: a re-note with other words entirely.", start=900)])
    with lib.write() as conn:
        rep = decide.save_package(conn, manifest, {"items": [
            {"item": 1, "verdict": "link", "confidence": "high", "note": "Invented.", "quote_a": qa,
             "quote_b": qb}]})
    assert rep.dropped == 1 and rep.feedback[0]["reason"] == "stale_ref"
    assert agent_rows(lib) == []


def test_an_applied_split_sends_agent_decisions_on_the_default_sense_to_rejudge(lib, w):
    t = "Invented: the sprocket shock is unrelated to the widget draw."
    w.note("lp_p1", [a("exogeneity", t), a("gadget shock", "Invented: gadget shock.")])
    w.note("lp_p2", [a("exogeneity", t + " Again.")])
    did = w.decide("link", "exogeneity", "gadget shock", by="agent", family="assumption")
    view = routine.create_batch(lib, "concepts:bridge", options={"families": "assumption", "sense_checks": "all"},
                                now=T0)
    first = view.items[0].subject_id
    assert first.startswith("assumption-checks")
    manifest = load_manifest(lib, view.batch_id, first)
    refs = list(manifest["items"][0]["a"]["refs"])
    with lib.write() as conn:
        decide.save_package(conn, manifest, {"items": [
            {"item": 1, "verdict": "split", "confidence": "high",
             "senses": [{"gloss": "one", "claims": [refs[0]]}, {"gloss": "two", "claims": [refs[1]]}]}]})
    assert w.state(did)[0] == "needs_rejudge"


def test_concepts_audit_counts_violations_only(lib, w):
    w.note("lp_p1", [a("ate"), a("att")])
    w.note("lp_p2", [a("ate")])
    rf.refresh(lib)
    res = decide.audit(lib.conn)
    assert res.ok and set(res.checks) >= {"spans_families", "seed_pairs_together", "stoplist_conflict"}
    # a researcher merge across the seed never-merge pair is refused by refresh, so the audit stays clean
    w.decide("merge", "ate", "att", by="researcher", family="assumption")
    rf.refresh(lib)
    assert decide.audit(lib.conn).checks["seed_pairs_together"] == 0
