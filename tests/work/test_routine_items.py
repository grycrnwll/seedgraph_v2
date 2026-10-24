"""Item states at creation and at save (design §1.2, §3, §5.3-§5.4): preserved, held back by
the gate or a title miss (with the accept_text override), failed, stale, conflict; the
selection language; release; the work commands."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from seedgraph.library import attention, graphs, textquality
from seedgraph.notes import current_note_id
from seedgraph.work import routine
from seedgraph.work.claims import ClaimRefused, WorkerRef
from seedgraph.work.commands import work_commands
from seedgraph.work.store import WorkError

from .routine_support import filler, good_note, make_paper, open_lib, reconvert

W1 = WorkerRef("interactive", "w1", client="claude")

# Family A mojibake inside words, many times: the gate calls this garbled.
GARBLED = "# Garbled\n\n" + " ".join(
    f"The identiÖed e§ect number {i} is di§erent and reÖnes the Örst estimate." for i in range(60)) + "\n"
THIN = "# Scan\n\nPage one.\n"


@pytest.fixture
def lib(tmp_path):
    library = open_lib(tmp_path)
    yield library
    library.close()


def _note_paper(lib, paper):
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    msg = routine.claim_next(lib, view.batch_id, W1).message
    return routine.submit(lib, msg["packet_id"], msg["claim_id"], note=good_note())


def test_fresh_note_is_preserved_unless_replace(lib):
    paper = make_paper(lib, 1)
    first = _note_paper(lib, paper)
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    assert view.items[0].state == "preserved"
    assert view.state.state == "finished"
    again = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id], options={"replace": True})
    assert again.items[0].state == "queued"
    msg = routine.claim_next(lib, again.batch_id, W1).message
    second = routine.submit(lib, msg["packet_id"], msg["claim_id"], note=good_note())
    assert second.paper_status == "done" and second.note_id != first.note_id
    assert current_note_id(lib.conn, paper.paper_id) == second.note_id


def test_garbled_text_is_held_back_with_an_attention_item_and_override(lib):
    paper = make_paper(lib, 1, text=GARBLED, sections=[])
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    (item,) = view.items
    assert item.state == "held_back" and item.reason.startswith("text_garbled")
    assert "reconvert --ocr" in item.reason
    att = attention.get_item(lib, item.attention_item_id)
    assert att.kind == "text_garbled" and att.status == "open" and not att.hides_notes
    cached = lib.conn.execute("SELECT status, checker_version FROM text_quality WHERE markdown_id = ?",
                              (paper.markdown_id,)).fetchone()
    assert tuple(cached) == ("garbled", textquality.CHECKER_VERSION)
    assert routine.claim_next(lib, view.batch_id, W1).status == "finished"

    attention.resolve_item(lib, att.item_id, "accept_text")
    override = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    assert override.items[0].state == "queued"


def test_text_missing_is_held_back(lib):
    paper = make_paper(lib, 1, text=THIN, sections=[])
    (item,) = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id]).items
    assert item.state == "held_back" and item.reason.startswith("text_missing")


def test_stale_cache_version_is_reassessed(lib):
    paper = make_paper(lib, 1)
    with lib.write() as conn:
        conn.execute("INSERT INTO text_quality (markdown_id, status, score_json, checker_version, assessed_at) "
                     "VALUES (?, 'garbled', '{}', 'textquality-0', 'then')", (paper.markdown_id,))
    (item,) = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id]).items
    assert item.state == "queued"


def test_open_title_miss_holds_the_paper_back(lib):
    paper = make_paper(lib, 1)
    miss = attention.raise_item(lib, "pdf_title_miss", {"paper_id": paper.paper_id}, paper_id=paper.paper_id,
                                detail="this PDF may not be this paper", hides_notes=True)
    (item,) = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id]).items
    assert item.state == "held_back" and item.reason.startswith("pdf_title_miss")
    assert item.attention_item_id == miss.item_id


def test_unknown_and_unconverted_papers_fail(lib):
    paper = make_paper(lib, 1)
    with lib.write() as conn:
        conn.execute("INSERT INTO papers (paper_id, title, created_at) VALUES ('lp_cited', 'Cited', 'x')")
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id, "lp_cited", "lp_nowhere"])
    assert [i.state for i in view.items] == ["queued", "failed", "failed"]


def test_reconverted_paper_becomes_stale(lib):
    paper = make_paper(lib, 1)
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    msg = routine.claim_next(lib, view.batch_id, W1).message
    reconvert(lib, paper, paper.text + "\nAn appended invented line.\n")
    result = routine.submit(lib, msg["packet_id"], msg["claim_id"], note=good_note())
    assert result.status == "accepted" and result.paper_status == "stale"
    assert current_note_id(lib.conn, paper.paper_id) is None
    item = routine.batch_view(lib, view.batch_id).items[0]
    assert item.state == "stale" and "re-converted" in item.reason


def test_current_note_changed_since_creation_is_a_conflict(lib):
    paper = make_paper(lib, 1)
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    msg = routine.claim_next(lib, view.batch_id, W1).message
    other = _note_paper(lib, paper)  # another batch notes the paper first
    result = routine.submit(lib, msg["packet_id"], msg["claim_id"], note=good_note())
    assert result.status == "accepted" and result.paper_status == "conflict"
    assert current_note_id(lib.conn, paper.paper_id) == other.note_id
    assert routine.batch_view(lib, view.batch_id).items[0].state == "conflict"


def test_resubmit_after_save_is_refused(lib):
    paper = make_paper(lib, 1)
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    msg = routine.claim_next(lib, view.batch_id, W1).message
    routine.submit(lib, msg["packet_id"], msg["claim_id"], note=good_note())
    with pytest.raises(ClaimRefused) as exc:
        routine.submit(lib, msg["packet_id"], msg["claim_id"], note=good_note())
    assert exc.value.code == "package_saved"


def test_release_give_back_and_paper_problem(lib):
    a, b = make_paper(lib, 1), make_paper(lib, 2)
    view = routine.create_batch(lib, "notes", paper_ids=[a.paper_id, b.paper_id])
    msg = routine.claim_next(lib, view.batch_id, W1).message
    after = routine.release(lib, msg["packet_id"], msg["claim_id"], "give_back")
    assert after.state.packages == {"open": 2}
    msg = routine.claim_next(lib, view.batch_id, W1).message
    after = routine.release(lib, msg["packet_id"], msg["claim_id"], "paper_problem", message="unreadable tables")
    assert after.items[0].state == "failed" and "unreadable tables" in after.items[0].reason
    assert routine.claim_next(lib, view.batch_id, W1).message["paper_id"] == b.paper_id
    with pytest.raises(WorkError):
        routine.release(lib, msg["packet_id"], msg["claim_id"], "because")


def test_selection_query_in_graph_without_current_note(lib):
    a, b, _outside = make_paper(lib, 1), make_paper(lib, 2), make_paper(lib, 3)
    g = graphs.create_graph(lib, "diff_in_diff")
    for p in (a, b):
        graphs.set_membership(g, p.paper_id, "member")
    g.close()
    _note_paper(lib, b)
    view = routine.create_batch(lib, "notes", query="held papers in graph diff_in_diff without a current note")
    assert [i.subject_id for i in view.items] == [a.paper_id]
    assert view.selection_query == "held papers in graph diff_in_diff without a current note"
    assert view.graph == "diff_in_diff"
    with pytest.raises(WorkError):
        routine.create_batch(lib, "notes", query="all the papers please")
    reconvert(lib, b, b.text + "\nMore invented text.\n")
    stale = routine.create_batch(lib, "notes", query="held with stale note")
    assert [i.subject_id for i in stale.items] == [b.paper_id]


def test_batch_reply_carries_the_work_commands(lib):
    paper = make_paper(lib, 1)
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    cmds = view.as_dict()["commands"]
    assert cmds["claude_code"] == f"/seedgraph-v3-work {view.batch_id}"
    assert cmds["codex"] == (
        "Use the seedgraph-v3-work skill and the seedgraph_v3 MCP tools to work Seedgraph batch "
        f"{view.batch_id} until it is finished or paused. Submit results with note_path; follow the feedback.")
    assert work_commands(view.batch_id) == view.commands


def test_options_are_checked(lib):
    paper = make_paper(lib, 1)
    with pytest.raises(WorkError):
        routine.create_batch(lib, "notes", paper_ids=[paper.paper_id], options={"claim_tll_s": 5})
    with pytest.raises(WorkError):
        routine.create_batch(lib, "notes", paper_ids=[paper.paper_id, paper.paper_id])
    with pytest.raises(WorkError):
        routine.create_batch(lib, "no_such_task", paper_ids=[paper.paper_id])


def test_pauses_and_expiry_through_the_routine(lib):
    t0 = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)
    a, b = make_paper(lib, 1), make_paper(lib, 2)
    view = routine.create_batch(lib, "notes", paper_ids=[a.paper_id, b.paper_id], options={"claim_ttl_s": 60},
                                now=t0)
    msg = routine.claim_next(lib, view.batch_id, W1, now=t0).message
    later = t0 + timedelta(seconds=120)
    assert routine.batch_view(lib, view.batch_id, now=later).state.state == "claim_expired"
    w2 = WorkerRef("interactive", "w2")
    again = routine.claim_next(lib, view.batch_id, w2, now=later).message
    assert again["packet_id"] == msg["packet_id"]
    with pytest.raises(ClaimRefused) as exc:
        routine.submit(lib, msg["packet_id"], msg["claim_id"], note=good_note(), now=later)
    assert exc.value.code == "claim_reclaimed"

    stopped = routine.stop(lib, view.batch_id, mode="now", now=later)
    assert stopped.state.state == "stopped"
    assert routine.claim_next(lib, view.batch_id, W1, now=later).status == "paused"
    resumed = routine.continue_batch(lib, view.batch_id, now=later)
    assert resumed.state.state == "waiting"
    limited = routine.report_limit(lib, view.batch_id, W1, resets_at="2026-03-01T15:00:00Z", now=later)
    assert limited.state.state == "usage_limit"
    paused = routine.claim_next(lib, view.batch_id, W1, now=later).as_dict()
    assert paused["status"] == "paused" and paused["resets_at"].startswith("2026-03-01T15")


def test_filler_is_long_enough_to_pass_the_gate():
    assert textquality.assess_text(filler("x", 40)).status == "pass"
