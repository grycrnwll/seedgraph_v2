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
