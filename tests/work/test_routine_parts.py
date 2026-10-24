"""Long papers in parts (design §1.3): split only over the threshold, at section starts;
part 1 owns the scalars; the note becomes current only when every part is saved."""

from __future__ import annotations

from pathlib import Path

import pytest

from seedgraph.notes import current_note_id
from seedgraph.work import packages, routine
from seedgraph.work.claims import WorkerRef

from .routine_support import (
    ASSUMPTION_QUOTE,
    INTRO_QUOTE,
    METHOD_QUOTE,
    RESULT_QUOTE,
    claim,
    empty_note,
    make_paper,
    open_lib,
    sections_markdown,
)

W1 = WorkerRef("interactive", "w1", client="claude")
W2 = WorkerRef("background", "w2", client="claude")
SPLIT = {"split_threshold_tokens": 300, "part_target_tokens": 300}


@pytest.fixture
def lib(tmp_path):
    library = open_lib(tmp_path)
    yield library
    library.close()


# --- the pure split ---------------------------------------------------------------------------

def test_under_the_threshold_is_one_part():
    text, sections = sections_markdown()
    refs = [(s, e) for _, k, s, e in sections if k == "references"]
    (part,) = packages.plan_parts(text, refs, [s for _, _, s, _ in sections])
    assert (part.part, part.part_count, part.start_char, part.end_char) == (1, 1, 0, len(text))
    assert part.text == packages.strip_references(text, refs)


def test_over_the_threshold_splits_at_section_starts():
    text, sections = sections_markdown()
    refs = [(s, e) for _, k, s, e in sections if k == "references"]
    starts = [s for _, _, s, _ in sections]
    parts = packages.plan_parts(text, refs, starts, **SPLIT)
    assert len(parts) == 3
    assert [p.part_count for p in parts] == [3, 3, 3]
    assert parts[0].start_char == 0 and parts[-1].end_char == len(text)
    assert [p.start_char for p in parts[1:]] == [sections[1][2], sections[2][2]]
    assert parts[0].text.startswith("# Introduction")
    assert parts[1].text.startswith("## Method")
    assert parts[2].text.startswith("## Results") and "[references omitted]" in parts[2].text
    for p in parts:  # every part's text is its slice of the full markdown, references cut
        assert p.text.replace("[references omitted]\n", "") in text


def test_an_oversized_section_falls_back_to_paragraph_boundaries():
    paras = "\n\n".join(f"Paragraph {i} " + "invented words " * 40 for i in range(8))
    text = "# Only section\n\n" + paras + "\n"
    parts = packages.plan_parts(text, [], [0], split_threshold_tokens=200, part_target_tokens=400)
    assert len(parts) >= 2
    for p in parts[1:]:
        assert text[p.start_char - 2:p.start_char] == "\n\n"  # cut at a paragraph boundary


# --- parts through the routine ----------------------------------------------------------------

def _split_batch(lib):
    paper = make_paper(lib, 1)
    view = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id], options=SPLIT)
    return paper, view


def test_a_long_paper_gets_one_package_per_part(lib):
    _, view = _split_batch(lib)
    assert view.items[0].part_count == 3
    assert view.state.packages == {"open": 3}
    instructions = (Path(view.work_dir) / "instructions.md").read_text(encoding="utf-8")
    assert "Long papers in parts" in instructions


def test_later_part_scalars_are_dropped_with_feedback(lib):
    _, view = _split_batch(lib)
    first = routine.claim_next(lib, view.batch_id, W1).message
    second = routine.claim_next(lib, view.batch_id, W1).message
    assert (first["part"], second["part"]) == (1, 2)  # one worker keeps one long paper
    # a found estimand without plain_words would be invalid in part 1; in part 2 it is dropped
    note = empty_note(
        research_question=claim("Restated question.", "a quote that is nowhere", "restated"),
        estimand_or_target_object=claim("An estimand.", "another quote that is nowhere", "estimand"),
        method_or_model=[claim("A two-step estimator.", METHOD_QUOTE, "two-step estimator",
                               method_type="estimator", plain_words="Compare workshops.")])
    result = routine.submit(lib, second["packet_id"], second["claim_id"], note=note)
    assert result.status == "accepted", result.as_dict()
    dropped = [w for w in result.warnings if "scalars belong to part 1" in w["message"]]
    assert {w["field_key"] for w in dropped} == {"research_question", "estimand_or_target_object"}
    assert result.quotes_found == 1 and result.claims_kept == 1


def test_note_is_current_only_when_every_part_is_saved(lib):
    paper, view = _split_batch(lib)
    p1 = routine.claim_next(lib, view.batch_id, W1).message
    p2 = routine.claim_next(lib, view.batch_id, W2).message
    p3 = routine.claim_next(lib, view.batch_id, W2).message
    assert sorted([p1["part"], p2["part"], p3["part"]]) == [1, 2, 3]
    by_part = {m["part"]: m for m in (p1, p2, p3)}
    notes = {
        1: empty_note(research_question=claim("Question.", INTRO_QUOTE, "gadget subsidy effect")),
        2: empty_note(assumptions=[claim("No anticipation.", ASSUMPTION_QUOTE, "no anticipation",
                                         assumption_type="identification", plain_words="Nobody acts early.")]),
        3: empty_note(main_results=[claim("Output rises.", RESULT_QUOTE, "output rises", result_type="empirical")]),
    }
    for part in (3, 1):
        r = routine.submit(lib, by_part[part]["packet_id"], by_part[part]["claim_id"], note=notes[part])
        assert r.status == "accepted" and r.paper_status == "in_progress"
        assert current_note_id(lib.conn, paper.paper_id) is None
    assert r.parts_done == 2
    last = routine.submit(lib, by_part[2]["packet_id"], by_part[2]["claim_id"], note=notes[2])
    assert last.paper_status == "done" and last.parts_done == 3
    assert last.note_id == current_note_id(lib.conn, paper.paper_id)
    rows = lib.conn.execute(
        "SELECT field_key, status, source_chunk_index FROM extracted_claims WHERE structured_note_id = ? "
        "AND status = 'found' ORDER BY field_key", (last.note_id,)).fetchall()
    assert [tuple(r) for r in rows] == [("assumptions[0]", "found", 1), ("main_results[0]", "found", 2),
                                        ("research_question", "found", 0)]
    mode, chunks = lib.conn.execute(
        "SELECT r.extraction_mode, r.chunk_count FROM structured_notes n JOIN extraction_runs r "
        "ON r.extraction_run_id = n.extraction_run_id WHERE n.note_id = ?", (last.note_id,)).fetchone()
    assert (mode, chunks) == ("agent_parts", 3)
