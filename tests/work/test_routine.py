"""The routine (design §1.1-§1.5, §2): create batch -> claim -> submit -> current note.
Tests drive the public functions of ``seedgraph.work.routine`` only. Invented text."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seedgraph.notes import current_note_id
from seedgraph.work import routine
from seedgraph.work.claims import WorkerRef

from .routine_support import (
    MU_QUOTE,
    REFERENCE_QUOTE,
    good_note,
    make_paper,
    open_lib,
)

W1 = WorkerRef("interactive", "w1", client="claude")
W2 = WorkerRef("interactive", "w2", client="codex")


@pytest.fixture
def lib(tmp_path):
    library = open_lib(tmp_path)
    yield library
    library.close()


def test_whole_paper_is_one_package_without_its_reference_list(lib):
    paper = make_paper(lib, 1)
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    assert [i.state for i in view.items] == ["queued"]
    assert view.items[0].part_count == 1

    reply = routine.claim_next(lib, view.batch_id, W1)
    assert reply.status == "claimed"
    msg = reply.message
    assert msg["part"] == 1 and msg["part_count"] == 1
    assert msg["paper_id"] == paper.paper_id
    text = Path(msg["text_path"]).read_bytes().decode("utf-8")
    assert "[references omitted]" in text
    assert REFERENCE_QUOTE not in text
    assert "## Method" in text and "## Results" in text
    for key in ("instructions_path", "labels_path"):
        assert Path(msg[key]).is_file()
    assert msg["result_path"].endswith(f"{msg['packet_id']}.result.json")
    assert msg["est_tokens"] == len(text) // 4


def test_accepted_submission_makes_the_current_note(lib):
    paper = make_paper(lib, 1)
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    msg = routine.claim_next(lib, view.batch_id, W1).message

    result = routine.submit(lib, msg["packet_id"], msg["claim_id"], note=good_note(), model="m-1")
    assert result.status == "accepted", result.as_dict()
    assert result.paper_status == "done"
    assert result.quotes_found == result.quotes_anchored == 5
    assert result.note_id == current_note_id(lib.conn, paper.paper_id)
    final = routine.batch_view(lib, view.batch_id)
    assert final.state.state == "finished"
    assert final.items[0].state == "done"
    assert final.items[0].result_ref == result.note_id


def test_submit_reads_the_note_from_a_file(lib):
    paper = make_paper(lib, 1)
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    msg = routine.claim_next(lib, view.batch_id, W1).message
    # a BOM, as Windows agents often write
    Path(msg["result_path"]).write_bytes(b"\xef\xbb\xbf" + json.dumps(good_note()).encode("utf-8"))
    result = routine.submit(lib, msg["packet_id"], msg["claim_id"], note_path=Path(msg["result_path"]))
    assert result.status == "accepted"


def test_lookalike_quote_needs_revision_and_keeps_the_claim(lib):
    paper = make_paper(lib, 1)
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    msg = routine.claim_next(lib, view.batch_id, W1).message

    wrong = MU_QUOTE.replace("μ", "µ")
    result = routine.submit(lib, msg["packet_id"], msg["claim_id"], note=good_note(mu_quote=wrong))
    assert result.status == "needs_revision"
    assert result.paper_status == "in_progress"
    (failed,) = result.failed_quotes
    assert failed["field_key"] == "main_results[1]"
    assert failed["reason"] == "lookalike_characters"
    assert failed["markdown_slice"] == MU_QUOTE
    assert "U+03BC in text, U+00B5 in quote" in failed["hint"]
    assert current_note_id(lib.conn, paper.paper_id) is None
    # the claim is still live: another worker gets nothing, and a resubmit is accepted
    assert routine.claim_next(lib, view.batch_id, W2).status in ("no_package", "finished")
    again = routine.submit(lib, msg["packet_id"], msg["claim_id"], note=good_note())
    assert again.status == "accepted"


def test_accept_unanchored_downgrades_and_is_reported(lib):
    paper = make_paper(lib, 1)
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    msg = routine.claim_next(lib, view.batch_id, W1).message
    note = good_note(mu_quote="This sentence is not anywhere in the invented paper at all.")
    result = routine.submit(lib, msg["packet_id"], msg["claim_id"], note=note, accept_unanchored=True)
    assert result.status == "accepted"
    assert result.accepted_unanchored == 1
    status = lib.conn.execute(
        "SELECT status FROM extracted_claims WHERE structured_note_id = ? AND field_key = 'main_results[1]'",
        (result.note_id,)).fetchone()[0]
    assert status == "ambiguous"
    assert routine.batch_view(lib, view.batch_id).accepted_unanchored == 1


def test_invalid_note_keeps_the_claim(lib):
    paper = make_paper(lib, 1)
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    msg = routine.claim_next(lib, view.batch_id, W1).message
    note = good_note()
    del note["assumptions"][0]["plain_words"]
    result = routine.submit(lib, msg["packet_id"], msg["claim_id"], note=note)
    assert result.status == "invalid"
    assert any("plain_words" in e for e in result.errors)
    assert routine.submit(lib, msg["packet_id"], msg["claim_id"], note=good_note()).status == "accepted"


def test_quote_in_reference_list_anchors_with_a_warning(lib):
    paper = make_paper(lib, 1)
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    msg = routine.claim_next(lib, view.batch_id, W1).message
    result = routine.submit(lib, msg["packet_id"], msg["claim_id"], note=good_note(mu_quote=REFERENCE_QUOTE))
    assert result.status == "accepted"
    assert any(w.get("reason") == "in_references" for w in result.warnings)


def test_replies_are_json_serialisable(lib):
    paper = make_paper(lib, 1)
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    json.dumps(view.as_dict())
    reply = routine.claim_next(lib, view.batch_id, W1)
    json.dumps(reply.as_dict())
    msg = reply.message
    wrong = MU_QUOTE.replace("μ", "µ")
    json.dumps(routine.submit(lib, msg["packet_id"], msg["claim_id"], note=good_note(mu_quote=wrong)).as_dict())
    json.dumps(routine.submit(lib, msg["packet_id"], msg["claim_id"], note=good_note()).as_dict())
    routine.stop(lib, view.batch_id, mode="finish")
    paper2 = make_paper(lib, 2)
    other = routine.create_batch(lib, "notes", paper_ids=[paper2.paper_id])
    routine.stop(lib, other.batch_id, mode="finish")
    paused = routine.claim_next(lib, other.batch_id, W2)
    assert paused.status == "paused"
    json.dumps(paused.as_dict())
    json.dumps([v.as_dict() for v in routine.list_batches(lib)])
