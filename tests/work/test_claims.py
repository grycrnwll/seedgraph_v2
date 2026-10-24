"""Package claims: one live claim per package, lazy expiry, claim_id checks (design §1.4)."""

from __future__ import annotations

import sqlite3
import threading

import pytest

from seedgraph.library import store as libstore
from seedgraph.work import claims, store

from .conftest import TTL, at, make_batch

ALICE = claims.WorkerRef(worker_kind="interactive", worker_id="w_alice", client="claude")
BOB = claims.WorkerRef(worker_kind="interactive", worker_id="w_bob", client="codex")


def test_claim_returns_the_lowest_open_package_with_a_claim_id_and_expiry(lib):
    batch_id = make_batch(lib, parts=(1, 1))
    result = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    assert result.status == "claimed"
    first = store.list_packages(lib.conn, batch_id)[0]
    assert result.package.packet_id == first.packet_id
    assert result.claim.claim_id.startswith("pc_")
    assert result.claim.expires_at == store.fmt_ts(at(TTL))
    assert store.get_package(lib.conn, first.packet_id).state == "claimed"
    assert store.get_item(lib.conn, batch_id, 1).state == "in_progress"


def test_two_workers_in_sequence_get_different_packages(lib):
    batch_id = make_batch(lib, parts=(1, 1))
    a = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    b = claims.claim_package(lib, batch_id, BOB, now=at(1))
    assert a.package.packet_id != b.package.packet_id
    c = claims.claim_package(lib, batch_id, BOB, now=at(2))
    assert c.status == "no_package"


def test_two_concurrent_claimers_never_get_one_package(lib, tmp_path):
    n_each = 6
    batch_id = make_batch(lib, parts=(1,) * (2 * n_each))
    got: dict[str, list[str]] = {}
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def work(worker):
        handle = libstore.open_library_at(lib.root)  # its own connection, as a second process would
        try:
            barrier.wait()
            for i in range(n_each + 2):
                r = claims.claim_package(handle, batch_id, worker, now=at(i))
                if r.status == "claimed":
                    got.setdefault(worker.worker_id, []).append(r.package.packet_id)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)
        finally:
            handle.close()

    threads = [threading.Thread(target=work, args=(w,)) for w in (ALICE, BOB)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    claimed = [p for ids in got.values() for p in ids]
    assert len(claimed) == 2 * n_each
    assert len(set(claimed)) == len(claimed)


def test_the_database_refuses_a_second_live_claim_on_one_package(lib):
    batch_id = make_batch(lib)
    result = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    with pytest.raises(sqlite3.IntegrityError), lib.write() as conn:
        conn.execute(
            "INSERT INTO package_claims (claim_id, packet_id, claim_no, worker_id, worker_kind, "
            "claimed_at, expires_at) VALUES ('pc_x', ?, 2, 'w_x', 'interactive', 'a', 'b')",
            (result.package.packet_id,),
        )


def test_a_worker_is_offered_further_parts_of_the_paper_it_holds(lib):
    batch_id = make_batch(lib, parts=(2, 2))
    a1 = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    b1 = claims.claim_package(lib, batch_id, BOB, now=at(1))
    b2 = claims.claim_package(lib, batch_id, BOB, now=at(2))
    a2 = claims.claim_package(lib, batch_id, ALICE, now=at(3))
    assert (a1.package.ordinal, a1.package.part) == (1, 1)
    assert (b1.package.ordinal, b1.package.part) == (1, 2)  # lowest open: Bob holds nothing yet
    assert (b2.package.ordinal, b2.package.part) == (2, 1)
    assert (a2.package.ordinal, a2.package.part) == (2, 2)


def test_preference_for_own_paper_beats_the_lowest_ordinal(lib):
    batch_id = make_batch(lib, parts=(1, 2))
    claims.claim_package(lib, batch_id, ALICE, paper_id="lp_02", now=at(0))
    bob = claims.claim_package(lib, batch_id, BOB, now=at(1))
    alice = claims.claim_package(lib, batch_id, ALICE, now=at(2))
    assert bob.package.ordinal == 1
    assert (alice.package.ordinal, alice.package.part) == (2, 2)


def test_claim_for_a_given_paper(lib):
    batch_id = make_batch(lib, parts=(1, 1, 1))
    r = claims.claim_package(lib, batch_id, ALICE, paper_id="lp_03", now=at(0))
    assert r.package.ordinal == 3
    again = claims.claim_package(lib, batch_id, BOB, paper_id="lp_03", now=at(1))
    assert again.status == "no_package"


def test_expiry_returns_the_package_to_the_pool(lib):
    batch_id = make_batch(lib)
    first = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    assert claims.claim_package(lib, batch_id, BOB, now=at(TTL - 1)).status == "no_package"
    second = claims.claim_package(lib, batch_id, BOB, now=at(TTL))
    assert second.status == "claimed"
    assert second.package.packet_id == first.package.packet_id
    rows = store.list_claims(lib.conn, packet_id=first.package.packet_id)
    assert [(c.claim_no, c.worker_id, c.release_reason) for c in rows] == [
        (1, "w_alice", "expired"), (2, "w_bob", None)]
    assert rows[0].released_at == store.fmt_ts(at(TTL))


def test_a_late_submit_with_a_stale_claim_is_refused_after_re_claim(lib):
    batch_id = make_batch(lib)
    alice = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    bob = claims.claim_package(lib, batch_id, BOB, now=at(TTL + 5))
    with pytest.raises(claims.ClaimRefused) as exc:
        claims.renew_claim(lib, alice.package.packet_id, alice.claim.claim_id, now=at(TTL + 6))
    assert exc.value.code == "claim_reclaimed"
    assert "re-claimed by another worker" in str(exc.value)
    # the new holder is unaffected
    renewed = claims.renew_claim(lib, bob.package.packet_id, bob.claim.claim_id, now=at(TTL + 7))
    assert renewed.claim_id == bob.claim.claim_id


def test_a_late_submit_revives_an_expired_claim_nobody_re_claimed(lib):
    batch_id = make_batch(lib)
    alice = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    revived = claims.renew_claim(lib, alice.package.packet_id, alice.claim.claim_id, now=at(TTL + 30))
    assert revived.claim_id == alice.claim.claim_id
    assert revived.released_at is None and revived.release_reason is None
    assert revived.expires_at == store.fmt_ts(at(2 * TTL + 30))
    assert store.get_package(lib.conn, alice.package.packet_id).state == "claimed"
    assert claims.claim_package(lib, batch_id, BOB, now=at(TTL + 31)).status == "no_package"


def test_renew_pushes_expiry_forward(lib):
    batch_id = make_batch(lib)
    alice = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    claims.renew_claim(lib, alice.package.packet_id, alice.claim.claim_id, now=at(TTL - 10))
    assert claims.claim_package(lib, batch_id, BOB, now=at(TTL + 10)).status == "no_package"


def test_a_wrong_or_unknown_claim_id_is_refused(lib):
    batch_id = make_batch(lib, parts=(1, 1))
    a = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    b = claims.claim_package(lib, batch_id, BOB, now=at(0))
    with pytest.raises(claims.ClaimRefused) as exc:
        claims.renew_claim(lib, a.package.packet_id, b.claim.claim_id, now=at(1))
    assert exc.value.code == "claim_unknown"
    with pytest.raises(claims.ClaimRefused):
        claims.renew_claim(lib, a.package.packet_id, "pc_nope", now=at(1))


def test_release_hands_the_package_back(lib):
    batch_id = make_batch(lib)
    a = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    claims.release_claim(lib, a.package.packet_id, a.claim.claim_id, now=at(5))
    b = claims.claim_package(lib, batch_id, BOB, now=at(6))
    assert b.package.packet_id == a.package.packet_id
    with pytest.raises(claims.ClaimRefused) as exc:
        claims.renew_claim(lib, a.package.packet_id, a.claim.claim_id, now=at(7))
    assert exc.value.code == "claim_released"


def test_release_can_void_the_package(lib):
    batch_id = make_batch(lib)
    a = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    claims.release_claim(lib, a.package.packet_id, a.claim.claim_id, package_state="void", now=at(5))
    assert store.get_package(lib.conn, a.package.packet_id).state == "void"
    assert claims.claim_package(lib, batch_id, BOB, now=at(6)).status == "finished"


def test_complete_claim_saves_the_package_and_ends_the_claim(lib):
    batch_id = make_batch(lib)
    a = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    claims.complete_claim(lib, a.package.packet_id, a.claim.claim_id, now=at(5))
    assert store.get_package(lib.conn, a.package.packet_id).state == "saved"
    (row,) = store.list_claims(lib.conn, packet_id=a.package.packet_id)
    assert row.release_reason == "submitted"
    with pytest.raises(claims.ClaimRefused) as exc:
        claims.renew_claim(lib, a.package.packet_id, a.claim.claim_id, now=at(6))
    assert exc.value.code == "package_saved"
    assert claims.claim_package(lib, batch_id, BOB, now=at(7)).status == "finished"


def test_events_log_claims_and_expiry_in_order(lib):
    batch_id = make_batch(lib)
    a = claims.claim_package(lib, batch_id, ALICE, now=at(0))
    claims.claim_package(lib, batch_id, BOB, now=at(TTL + 1))
    events = store.list_events(lib.conn, batch_id)
    assert [(e.seq, e.kind, e.worker_id) for e in events] == [
        (1, "batch_created", None), (2, "claimed", "w_alice"),
        (3, "claim_expired", "w_alice"), (4, "claimed", "w_bob")]
    assert events[1].packet_id == a.package.packet_id
    assert events[1].detail["claim_id"] == a.claim.claim_id


def test_unknown_batch_is_an_error(lib):
    with pytest.raises(store.WorkError):
        claims.claim_package(lib, "b_missing", ALICE, now=at(0))
