"""Batch state and the pause model: four pause states, precedence, continue, stop (design §1.6)."""

from __future__ import annotations

import sqlite3

import pytest

from seedgraph.library import store as libstore
from seedgraph.work import claims, state, store

from .conftest import TTL, at, make_batch

ALICE = claims.WorkerRef(worker_kind="interactive", worker_id="w_alice", client="claude")
BOB = claims.WorkerRef(worker_kind="background", worker_id="w_bob", client="codex")


def _state(lib, batch_id, t):
    return state.batch_state(lib, batch_id, now=at(t))


# --- the four pause states (and working / finished) -----------------------------------------

def test_a_fresh_batch_is_waiting_for_an_agent(lib):
    batch_id = make_batch(lib, parts=(1, 1))
    s = _state(lib, batch_id, 0)
    assert s.state == "waiting"
    assert s.label == "Waiting for an agent"
    assert s.packages == {"open": 2}
    assert s.items == {"queued": 2}
    assert not s.paused


def test_a_live_claim_is_working(lib):
    batch_id = make_batch(lib, parts=(1, 1, 1))
    claims.claim_package(lib, batch_id, ALICE, now=at(0))
    claims.claim_package(lib, batch_id, ALICE, now=at(0))
    claims.claim_package(lib, batch_id, BOB, now=at(0))
    s = _state(lib, batch_id, 1)
    assert s.state == "working"
    assert s.label == "Working: 2 agents"
    assert s.live_claims == 3
    assert s.workers == ("w_alice", "w_bob")


def test_an_expired_claim_shows_claim_expired(lib):
    batch_id = make_batch(lib, parts=(1, 1))
    first = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    assert _state(lib, batch_id, TTL - 1).state == "working"
    s = _state(lib, batch_id, TTL)
    assert s.state == "claim_expired"
    assert s.label == "Paused: a worker stopped responding; its package is back in the pool"
    assert s.expired_packets == (first.package.packet_id,)
    assert s.packages == {"open": 2}


def test_claiming_again_clears_the_derived_claim_expired_state(lib):
    batch_id = make_batch(lib, parts=(1,))
    claims.claim_package(lib, batch_id, ALICE, now=at(0))
    r = claims.claim_package(lib, batch_id, BOB, now=at(TTL + 1))  # claim_expired does not refuse
    assert r.status == "claimed"
    assert _state(lib, batch_id, TTL + 2).state == "working"


def test_stopped_state(lib):
    batch_id = make_batch(lib, parts=(1,))
    s = state.stop(lib, batch_id, mode="finish", now=at(1))
    assert s.state == "stopped" and s.label == "Stopped (finishing current papers: all finished)" and s.paused
    assert s.stop_mode == "finish"


def test_usage_limit_with_and_without_a_reset_time(lib):
    batch_id = make_batch(lib, parts=(1,))
    s = state.report_limit(lib, batch_id, ALICE, resets_at="2026-03-01T14:00:00Z",
                           message="client reported a weekly limit", now=at(1))
    assert s.state == "usage_limit"
    assert s.resets_at == "2026-03-01T14:00:00.000000+00:00"
    assert s.label == "Paused: usage limit, resets 2026-03-01T14:00:00.000000+00:00"
    assert s.pause_detail == "client reported a weekly limit"

    other = make_batch(lib, parts=(1,))
    s2 = state.report_limit(lib, other, ALICE, resets_at=None, message=None, now=at(1))
    assert s2.state == "usage_limit"
    assert s2.resets_at is None
    assert s2.label == "Paused: usage limit, resets: unknown"


def test_a_bad_reset_time_is_refused(lib):
    batch_id = make_batch(lib)
    with pytest.raises(store.WorkError):
        state.report_limit(lib, batch_id, ALICE, resets_at="tomorrow-ish", now=at(1))


def test_finished_when_nothing_is_open_or_claimed(lib):
    batch_id = make_batch(lib, parts=(1,))
    with lib.write() as conn:
        store.insert_item(conn, batch_id, 2, "lp_held", state="held_back", reason="text garbled", now=at(0))
    r = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    claims.complete_claim(lib, r.package.packet_id, r.claim.claim_id, now=at(1))
    with lib.write() as conn:
        store.update_item(conn, batch_id, 1, state="done", result_ref="note_x", now=at(1))
    s = _state(lib, batch_id, 2)
    assert s.state == "finished"
    assert s.label == "Finished: 1 done / 1 held back / 0 failed"


# --- precedence: stopped > usage limit > claim expired > waiting ---------------------------

def test_stopped_beats_usage_limit_in_either_order(lib):
    a = make_batch(lib)
    state.report_limit(lib, a, ALICE, resets_at=None, now=at(0))
    assert state.stop(lib, a, mode="finish", now=at(1)).state == "stopped"

    b = make_batch(lib)
    state.stop(lib, b, mode="finish", now=at(0))
    s = state.report_limit(lib, b, ALICE, resets_at="2026-03-02T00:00:00+00:00", now=at(1))
    assert s.state == "stopped"


def test_usage_limit_beats_an_expired_claim(lib):
    batch_id = make_batch(lib, parts=(1, 1))
    claims.claim_package(lib, batch_id, BOB, now=at(0))
    assert _state(lib, batch_id, TTL).state == "claim_expired"
    assert state.report_limit(lib, batch_id, ALICE, now=at(TTL + 1)).state == "usage_limit"


def test_stopped_beats_a_live_claim_and_an_expired_one(lib):
    batch_id = make_batch(lib, parts=(1, 1))
    claims.claim_package(lib, batch_id, ALICE, now=at(0))
    claims.claim_package(lib, batch_id, BOB, now=at(TTL - 10))
    s = state.stop(lib, batch_id, mode="finish", now=at(TTL + 1))  # Alice's expired, Bob's live
    assert s.state == "stopped"
    assert s.label == "Stopped (finishing current papers: 1 claim left)"


def test_claim_expired_beats_waiting_and_continue_acknowledges_it(lib):
    batch_id = make_batch(lib, parts=(1, 1))
    claims.claim_package(lib, batch_id, ALICE, now=at(0))
    assert _state(lib, batch_id, TTL).state == "claim_expired"
    s = state.continue_batch(lib, batch_id, now=at(TTL + 1))
    assert s.state == "waiting"
    assert s.expired_packets == ()
    (claim,) = store.list_claims(lib.conn, batch_id=batch_id)
    assert claim.acknowledged is True


def test_a_later_expiry_after_continue_shows_again(lib):
    batch_id = make_batch(lib, parts=(1,))
    claims.claim_package(lib, batch_id, ALICE, now=at(0))
    state.continue_batch(lib, batch_id, now=at(TTL + 1))
    claims.claim_package(lib, batch_id, BOB, now=at(TTL + 2))
    assert _state(lib, batch_id, 2 * TTL + 2).state == "claim_expired"


# --- continue -----------------------------------------------------------------------------

def test_continue_clears_a_stop_and_claiming_works_again(lib):
    batch_id = make_batch(lib, parts=(1,))
    state.stop(lib, batch_id, mode="finish", now=at(0))
    refused = claims.claim_package(lib, batch_id, ALICE, now=at(1))
    assert refused.status == "paused"
    assert refused.pause() == {"status": "paused", "state": "stopped",
                               "label": "Stopped (finishing current papers: all finished)",
                               "resets_at": None, "next_action": refused.state.next_action}
    s = state.continue_batch(lib, batch_id, now=at(2))
    assert s.state == "waiting"
    assert s.pause_kind is None and s.stop_mode is None
    assert claims.claim_package(lib, batch_id, ALICE, now=at(3)).status == "claimed"


def test_continue_clears_a_usage_limit_from_a_different_agent(lib, tmp_path):
    batch_id = make_batch(lib, parts=(1, 1))
    state.report_limit(lib, batch_id, ALICE, resets_at="2026-03-01T14:00:00+00:00", now=at(0))
    refused = claims.claim_package(lib, batch_id, BOB, now=at(1))
    assert refused.status == "paused"
    assert refused.pause()["resets_at"] == "2026-03-01T14:00:00.000000+00:00"
    other = libstore.open_library_at(lib.root)  # another agent's own connection
    try:
        s = state.continue_batch(other, batch_id, now=at(2))
        assert s.state == "waiting" and s.resets_at is None and s.pause_detail is None
        assert claims.claim_package(other, batch_id, BOB, now=at(3)).status == "claimed"
    finally:
        other.close()


def test_report_limit_releases_only_the_reporting_workers_claims(lib):
    batch_id = make_batch(lib, parts=(1, 1))
    a = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    b = claims.claim_package(lib, batch_id, BOB, now=at(0))
    s = state.report_limit(lib, batch_id, ALICE, now=at(1))
    assert s.live_claims == 1 and s.workers == ("w_bob",)
    with pytest.raises(claims.ClaimRefused) as exc:
        claims.renew_claim(lib, a.package.packet_id, a.claim.claim_id, now=at(2))
    assert exc.value.code == "claim_limit"
    claims.renew_claim(lib, b.package.packet_id, b.claim.claim_id, now=at(2))  # Bob may finish


# --- stop finish vs now -------------------------------------------------------------------

def test_stop_finish_lets_live_claims_finish_but_no_new_claims(lib):
    batch_id = make_batch(lib, parts=(1, 1))
    a = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    state.stop(lib, batch_id, mode="finish", now=at(1))
    assert claims.claim_package(lib, batch_id, BOB, now=at(2)).status == "paused"
    renewed = claims.renew_claim(lib, a.package.packet_id, a.claim.claim_id, now=at(3))
    assert renewed.live
    claims.complete_claim(lib, a.package.packet_id, a.claim.claim_id, now=at(4))
    assert store.get_package(lib.conn, a.package.packet_id).state == "saved"
    s = _state(lib, batch_id, 5)
    assert s.state == "stopped" and s.label == "Stopped (finishing current papers: all finished)"


def test_a_stopped_batch_whose_last_claim_finishes_is_finished(lib):
    batch_id = make_batch(lib, parts=(1,))
    a = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    state.stop(lib, batch_id, mode="finish", now=at(1))
    claims.complete_claim(lib, a.package.packet_id, a.claim.claim_id, now=at(2))
    r = claims.claim_package(lib, batch_id, BOB, now=at(3))
    assert r.status == "finished" and r.state.state == "finished"


def test_the_schema_refuses_a_stop_mode_without_a_stop(lib):
    batch_id = make_batch(lib)
    with pytest.raises(sqlite3.IntegrityError), lib.write() as conn:
        conn.execute("UPDATE batches SET pause_kind = NULL, stop_mode = 'now' WHERE batch_id = ?", (batch_id,))


def test_stop_now_releases_live_claims_and_refuses_their_submits(lib):
    batch_id = make_batch(lib, parts=(1, 1))
    a = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    s = state.stop(lib, batch_id, mode="now", now=at(1))
    assert s.state == "stopped" and s.live_claims == 0
    assert s.packages == {"open": 2}
    (row,) = store.list_claims(lib.conn, packet_id=a.package.packet_id)
    assert row.release_reason == "stopped"
    for fn in (claims.renew_claim, claims.complete_claim):
        with pytest.raises(claims.ClaimRefused) as exc:
            fn(lib, a.package.packet_id, a.claim.claim_id, now=at(2))
        assert exc.value.code == "claim_stopped"
    # still refused after continue: a stop-now claim is never revived
    state.continue_batch(lib, batch_id, now=at(3))
    with pytest.raises(claims.ClaimRefused):
        claims.renew_claim(lib, a.package.packet_id, a.claim.claim_id, now=at(4))


def test_stop_now_also_refuses_reviving_an_expired_claim(lib):
    batch_id = make_batch(lib)
    a = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    state.stop(lib, batch_id, mode="now", now=at(TTL + 1))
    with pytest.raises(claims.ClaimRefused) as exc:
        claims.renew_claim(lib, a.package.packet_id, a.claim.claim_id, now=at(TTL + 2))
    assert exc.value.code == "batch_stopped"


def test_an_unknown_stop_mode_is_refused(lib):
    batch_id = make_batch(lib)
    with pytest.raises(store.WorkError):
        state.stop(lib, batch_id, mode="later", now=at(0))


# --- lazy expiry on reads, events ---------------------------------------------------------

def test_a_status_read_that_expires_nothing_does_not_dirty_the_library(tmp_path):
    lib = libstore.open_library_at(tmp_path / "library", migration=True)  # writes leave dirty alone
    try:
        batch_id = make_batch(lib)
    finally:
        lib.close()
    lib = libstore.open_library_at(tmp_path / "library")
    try:
        assert not libstore.is_dirty(lib.conn)
        state.batch_state(lib, batch_id, now=at(0))
        assert not libstore.is_dirty(lib.conn)
    finally:
        lib.close()


def test_pause_and_continue_events_are_logged(lib):
    batch_id = make_batch(lib)
    state.report_limit(lib, batch_id, ALICE, resets_at=None, message="limit", now=at(1))
    state.stop(lib, batch_id, mode="now", now=at(2))
    state.continue_batch(lib, batch_id, now=at(3))
    kinds = [(e.seq, e.kind) for e in store.list_events(lib.conn, batch_id)]
    assert kinds == [(1, "batch_created"), (2, "usage_limit"), (3, "stopped"), (4, "continued")]
    assert store.list_events(lib.conn, batch_id)[-1].detail == {"cleared": "stopped", "acknowledged": 0}


# --- next_action wording and the stop mode -------------------------------------------------

def test_working_with_an_open_package_says_claim_the_next_one(lib):
    batch_id = make_batch(lib, parts=(1, 1))
    claims.claim_package(lib, batch_id, ALICE, now=at(0))
    assert _state(lib, batch_id, 1).next_action == "Claim the next package."


def test_working_with_every_package_claimed_does_not_say_claim(lib):
    batch_id = make_batch(lib, parts=(1,))
    claims.claim_package(lib, batch_id, ALICE, now=at(0))
    s = _state(lib, batch_id, 1)
    assert s.state == "working"
    assert "Claim the next package" not in s.next_action
    assert "All open packages are claimed" in s.next_action


def test_stop_finish_tells_claim_holders_to_finish_and_others_to_stop(lib):
    batch_id = make_batch(lib, parts=(1, 1))
    a = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    s = state.stop(lib, batch_id, mode="finish", now=at(1))
    assert s.label == "Stopped (finishing current papers: 1 claim left)"
    assert "If you hold a claim, finish and submit your current package" in s.next_action
    assert "Without a claim, stop working and report your counts" in s.next_action
    claims.complete_claim(lib, a.package.packet_id, a.claim.claim_id, now=at(2))
    after = _state(lib, batch_id, 3)
    assert after.label == "Stopped (finishing current papers: all finished)"
    assert after.stop_mode == "finish"
    assert "finish and submit" not in after.next_action


def test_stop_now_is_labelled_stopped_now(lib):
    batch_id = make_batch(lib, parts=(1,))
    claims.claim_package(lib, batch_id, ALICE, now=at(0))
    s = state.stop(lib, batch_id, mode="now", now=at(1))
    assert s.label == "Stopped now" and s.stop_mode == "now"
    assert "finish and submit" not in s.next_action


# --- item state follows its packages back to the pool --------------------------------------

def test_an_expired_claim_puts_its_item_back_in_the_queue(lib):
    batch_id = make_batch(lib, parts=(1, 1))
    claims.claim_package(lib, batch_id, ALICE, now=at(0))
    claims.claim_package(lib, batch_id, BOB, now=at(0))
    assert _state(lib, batch_id, 1).items == {"in_progress": 2}
    s = _state(lib, batch_id, TTL)
    assert s.packages == {"open": 2}
    assert s.items == {"queued": 2}


def test_a_given_back_package_puts_its_item_back_in_the_queue(lib):
    batch_id = make_batch(lib, parts=(1,))
    a = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    claims.release_claim(lib, a.package.packet_id, a.claim.claim_id, now=at(1))
    assert _state(lib, batch_id, 2).items == {"queued": 1}


def test_stop_now_puts_the_released_items_back_in_the_queue(lib):
    batch_id = make_batch(lib, parts=(1, 1))
    claims.claim_package(lib, batch_id, ALICE, now=at(0))
    s = state.stop(lib, batch_id, mode="now", now=at(1))
    assert s.items == {"queued": 2}


def test_a_usage_limit_puts_the_released_items_back_in_the_queue(lib):
    batch_id = make_batch(lib, parts=(1,))
    claims.claim_package(lib, batch_id, ALICE, now=at(0))
    assert state.report_limit(lib, batch_id, ALICE, now=at(1)).items == {"queued": 1}


def test_a_paper_with_a_saved_part_stays_in_progress_when_its_other_part_returns(lib):
    batch_id = make_batch(lib, parts=(2,))
    p1 = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    claims.complete_claim(lib, p1.package.packet_id, p1.claim.claim_id, now=at(1))
    p2 = claims.claim_package(lib, batch_id, ALICE, now=at(2))
    claims.release_claim(lib, p2.package.packet_id, p2.claim.claim_id, now=at(3))
    assert _state(lib, batch_id, 4).items == {"in_progress": 1}


def test_a_revived_late_submit_puts_its_item_back_in_progress(lib):
    batch_id = make_batch(lib, parts=(1,))
    a = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    assert _state(lib, batch_id, TTL + 1).items == {"queued": 1}
    claims.renew_claim(lib, a.package.packet_id, a.claim.claim_id, now=at(TTL + 2))
    assert _state(lib, batch_id, TTL + 3).items == {"in_progress": 1}
