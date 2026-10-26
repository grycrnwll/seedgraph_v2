"""The "Write notes" estimate (stage 3 design §2.2): paper text tokens from the packaging
estimator, a per-paper range of agent tokens, the papers that will be skipped and why, and
no writes. Invented text only."""

from __future__ import annotations

import pytest

from seedgraph.library import attention, textquality
from seedgraph.work import estimate, routine
from seedgraph.work.claims import WorkerRef
from seedgraph.work.store import WorkError

from .routine_support import REFERENCE_QUOTE, good_note, make_paper


def _note(lib, paper):
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    msg = routine.claim_next(lib, view.batch_id, WorkerRef("interactive", "est")).message
    assert routine.submit(lib, msg["packet_id"], msg["claim_id"], note=good_note()).paper_status == "done"


def _cache_verdict(lib, paper, status):
    with lib.write() as conn:
        conn.execute("INSERT INTO text_quality (markdown_id, status, score_json, checker_version, assessed_at) "
                     "VALUES (?, ?, '{\"summary\": \"invented verdict\"}', ?, 'then')",
                     (paper.markdown_id, status, textquality.CHECKER_VERSION))


def test_text_tokens_are_chars_over_four_after_stripping_references(lib):
    paper = make_paper(lib, 1)
    est = estimate.estimate(lib, "notes", [paper.paper_id])
    (p,) = est.papers
    body = paper.text[:paper.text.index("## References")]
    assert REFERENCE_QUOTE not in body
    stripped = body + "[references omitted]\n"
    assert p.state == "queued" and p.reason is None and p.parts == 1
    assert p.text_tokens == len(stripped) // 4
    assert est.text_tokens == p.text_tokens
    assert 0 < p.agent_tokens_low <= p.agent_tokens_high
    assert (est.agent_tokens_low, est.agent_tokens_high) == (p.agent_tokens_low, p.agent_tokens_high)


def test_the_range_scales_with_the_paper(lib):
    short, long_ = make_paper(lib, 1), make_paper(lib, 2, text="# Long\n\n" + "Invented words here. " * 2000)
    est = estimate.estimate(lib, "notes", [short.paper_id, long_.paper_id])
    a, b = est.papers
    assert b.text_tokens > a.text_tokens and b.agent_tokens_low > a.agent_tokens_low
    assert est.text_tokens == a.text_tokens + b.text_tokens
    assert est.agent_tokens_high == a.agent_tokens_high + b.agent_tokens_high
    assert [p.paper_id for p in est.papers] == [short.paper_id, long_.paper_id]  # selection order


def test_skipped_papers_carry_reasons_and_are_left_out_of_totals(lib):
    noted, titled, garbled, queued = (make_paper(lib, n) for n in (1, 2, 3, 4))
    _note(lib, noted)
    attention.raise_item(lib, "pdf_title_miss", {"paper_id": titled.paper_id}, paper_id=titled.paper_id,
                         detail="this PDF may not be this paper", hides_notes=True)
    _cache_verdict(lib, garbled, "garbled")
    ids = [noted.paper_id, titled.paper_id, garbled.paper_id, queued.paper_id, "lp_missing"]
    est = estimate.estimate(lib, "notes", ids)
    states = {p.paper_id: (p.state, p.reason) for p in est.papers}
    assert states[noted.paper_id][0] == "preserved"
    assert states[titled.paper_id][0] == "held_back" and states[titled.paper_id][1].startswith("pdf_title_miss")
    assert states[garbled.paper_id][0] == "held_back" and states[garbled.paper_id][1].startswith("text_garbled")
    assert states["lp_missing"][0] == "failed"
    assert states[queued.paper_id] == ("queued", None)
    assert [p.paper_id for p in est.queued] == [queued.paper_id]
    assert {p.paper_id for p in est.skipped} == set(ids) - {queued.paper_id}
    assert all(p.text_tokens == 0 for p in est.skipped)
    assert est.text_tokens == next(p for p in est.papers if p.paper_id == queued.paper_id).text_tokens


def test_replace_counts_a_noted_paper(lib):
    paper = make_paper(lib, 1)
    _note(lib, paper)
    (p,) = estimate.estimate(lib, "notes", [paper.paper_id], replace=True).papers
    assert p.state == "queued" and p.text_tokens > 0


def test_an_accepted_garbled_text_is_counted(lib):
    paper = make_paper(lib, 1)
    _cache_verdict(lib, paper, "garbled")
    item = attention.raise_item(lib, "text_garbled", {"paper_id": paper.paper_id, "markdown_id": paper.markdown_id},
                                paper_id=paper.paper_id, detail="invented", hides_notes=False)
    attention.resolve_item(lib, item.item_id, "accept_text")
    (p,) = estimate.estimate(lib, "notes", [paper.paper_id]).papers
    assert p.state == "queued"


def test_an_unchecked_text_is_counted_and_flagged(lib):
    paper = make_paper(lib, 1)
    (p,) = estimate.estimate(lib, "notes", [paper.paper_id]).papers
    assert p.state == "queued" and p.text_checked is False
    _cache_verdict(lib, paper, "pass")
    (p,) = estimate.estimate(lib, "notes", [paper.paper_id]).papers
    assert p.text_checked is True


def test_the_estimate_writes_nothing(lib):
    papers = [make_paper(lib, n) for n in (1, 2)]
    before = lib.conn.total_changes
    estimate.estimate(lib, "notes", [p.paper_id for p in papers])
    assert lib.conn.total_changes == before
    assert lib.conn.execute("SELECT COUNT(*) FROM text_quality").fetchone()[0] == 0
    assert lib.conn.execute("SELECT COUNT(*) FROM attention_items").fetchone()[0] == 0


def test_duplicates_are_counted_once_and_unknown_tasks_refused(lib):
    paper = make_paper(lib, 1)
    assert len(estimate.estimate(lib, "notes", [paper.paper_id, paper.paper_id]).papers) == 1
    with pytest.raises(WorkError):
        estimate.estimate(lib, "metadata", [paper.paper_id])
    with pytest.raises(WorkError):
        estimate.estimate(lib, "no_such_task", [paper.paper_id])
