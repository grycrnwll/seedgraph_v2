"""Spot-check marks (design §5.3, §5.5; slice R4): validation, the last mark counts, the
finished sample, and the derived "fixed" check that leaves accuracy unchanged."""

from __future__ import annotations

import pytest

from seedgraph.notes import overlay
from seedgraph.spotcheck import accuracy, marks
from seedgraph.spotcheck import sample as sp

PAST = "2020-01-01T00:00:00.000000+00:00"
LATER = "2021-01-01T00:00:00.000000+00:00"


@pytest.fixture
def method_item(world, lib):
    """A one-item sample holding a paper's method claim."""
    pid = world.noted_paper()
    s = sp.create_sample(lib, n=1, seed=1, filters={"claim_type": "method"})
    return pid, s.sample_id, s.items[0].claim_id


def test_mark_refusals(method_item, lib):
    _, sid, cid = method_item
    with pytest.raises(sp.SpotCheckError, match="not a mark"):
        marks.mark(lib, sid, cid, "half")
    with pytest.raises(sp.SpotCheckError, match="unknown reason"):
        marks.mark(lib, sid, cid, "wrong", reasons=["bad vibes"])
    with pytest.raises(sp.SpotCheckError, match="partly right or wrong"):
        marks.mark(lib, sid, cid, "right", reasons=["misreads"])
    with pytest.raises(sp.SpotCheckError, match="not in that"):
        marks.mark(lib, sid, "claim_other", "right")
    with pytest.raises(sp.SpotCheckError, match="not in that"):
        marks.fixed(lib, "sc_none", cid)


def test_the_last_mark_counts(method_item, lib):
    _, sid, cid = method_item
    item = marks.mark(lib, sid, cid, "wrong", reasons=["not_in_paper", "wrong_type", "wrong_type"],
                      note="  quote is   about\nserial correlation ")
    assert item.mark == "wrong" and item.reasons == ("wrong_type", "not_in_paper")
    assert item.note == "quote is about serial correlation" and item.marked_at
    item = marks.mark(lib, sid, cid, "right")
    assert item.mark == "right" and item.reasons == () and item.note is None
    rows = lib.conn.execute("SELECT mark FROM spot_check_items WHERE claim_id = ?", (cid,)).fetchall()
    assert [r[0] for r in rows] == ["right"]
    view = lib.conn.execute("SELECT n_right, n_partly, n_wrong, n_marked FROM spot_check_accuracy").fetchall()
    assert [tuple(r) for r in view] == [(1, 0, 0, 1)]


def test_latest_mark_across_samples_counts_once(world, lib):
    world.noted_paper()
    a = sp.create_sample(lib, n=1, seed=1, filters={"claim_type": "method"})
    b = sp.create_sample(lib, n=1, seed=2, filters={"claim_type": "method"})  # still unmarked: drawn again
    cid = a.items[0].claim_id
    assert b.items[0].claim_id == cid and a.sample_id != b.sample_id
    marks.mark(lib, a.sample_id, cid, "wrong", now=PAST)
    marks.mark(lib, b.sample_id, cid, "partly", now=LATER)
    view = lib.conn.execute("SELECT n_right, n_partly, n_wrong, n_marked FROM spot_check_accuracy").fetchall()
    assert [tuple(r) for r in view] == [(0, 1, 0, 1)]


def test_a_sample_is_finished_when_every_item_is_marked(world, lib):
    world.noted_paper()
    s = sp.create_sample(lib, n=2, seed=3)
    marks.mark(lib, s.sample_id, s.items[0].claim_id, "right")
    assert sp.sample(lib, s.sample_id).finished_at is None
    marks.mark(lib, s.sample_id, s.items[1].claim_id, "partly", now=LATER)
    done = sp.sample(lib, s.sample_id)
    assert done.finished_at == LATER and done.n_marked == 2
    assert done.counts == {"right": 1, "partly": 1, "wrong": 0, "unmarked": 0}


def test_an_edit_after_the_mark_is_fixed_and_leaves_accuracy_unchanged(method_item, lib):
    pid, sid, cid = method_item
    assert not marks.fixed(lib, sid, cid)  # unmarked
    marks.mark(lib, sid, cid, "wrong", reasons=["wrong_type"], now=PAST)
    before = accuracy.accuracy(lib)
    assert not marks.fixed(lib, sid, cid)
    overlay.save_edit(lib, pid, {"op": "edit", "target": {"claim_id": cid},
                                 "set": {"claim_type": "result", "claim_text": "Earnings rise after the programme."}},
                      expected_version=0)
    assert marks.fixed(lib, sid, cid)
    item = sp.sample(lib, sid).item(cid)
    assert item.fixed and item.edited and item.claim_type == "method" and item.mark == "wrong"
    assert accuracy.accuracy(lib) == before


def test_an_edit_carried_forward_after_the_mark_is_not_a_fix(method_item, lib):
    pid, sid, cid = method_item
    overlay.save_edit(lib, pid, {"op": "edit", "target": {"claim_id": cid},
                                 "set": {"claim_text": "An edited design."}}, expected_version=0)
    with lib.write() as conn:  # the edit predates the mark
        conn.execute("UPDATE overlay_versions SET created_at = ?", (PAST,))
    marks.mark(lib, sid, cid, "partly", now=LATER)
    overlay.save_notes(lib, pid, "Invented researcher notes.", expected_version=1)  # ops unchanged
    assert not marks.fixed(lib, sid, cid)
    # another claim's edit is not this claim's fix
    other = lib.conn.execute("SELECT claim_id FROM extracted_claims WHERE field_key = 'main_results[0]'").fetchone()[0]
    overlay.save_edit(lib, pid, {"op": "hide", "target": {"claim_id": other}}, expected_version=2)
    assert not marks.fixed(lib, sid, cid)
    overlay.save_edit(lib, pid, {"op": "edit", "target": {"claim_id": cid},
                                 "set": {"claim_text": "Edited again after the mark."}}, expected_version=3)
    assert marks.fixed(lib, sid, cid)


def test_editing_a_sibling_that_quotes_the_same_sentence_is_not_a_fix(world, lib):
    from .conftest import raw_note

    pid = world.paper()
    saved = world.note(pid, raw=raw_note(results=2))
    claims = world.claims(saved.note_id)
    first, second = claims["main_results[0]"], claims["main_results[1]"]
    s = sp.create_sample(lib, n=2, seed=1, filters={"claim_type": "result"})
    assert {i.claim_id for i in s.items} == {first, second}
    marks.mark(lib, s.sample_id, first, "wrong", now=PAST)
    overlay.save_edit(lib, pid, {"op": "edit", "target": {"claim_id": second},
                                 "set": {"claim_text": "An edited second result."}}, expected_version=0)
    assert not marks.fixed(lib, s.sample_id, first)


def test_hiding_after_the_mark_is_a_fix(method_item, lib):
    pid, sid, cid = method_item
    marks.mark(lib, sid, cid, "wrong", now=PAST)
    overlay.save_edit(lib, pid, {"op": "hide", "target": {"claim_id": cid}}, expected_version=0)
    item = sp.sample(lib, sid).item(cid)
    assert item.fixed and item.hidden


def test_labels():
    assert marks.MARKS == ("right", "partly", "wrong")
    assert marks.MARK_LABELS["partly"] == "partly right"
    assert set(marks.REASON_LABELS) == set(marks.REASONS)
    assert marks.FIX_MARKS == ("partly", "wrong")
