"""`library textcheck`: the gate's verdict per paper, and against the golden-set
category (design §5.6: acceptance is verdict == category). Invented text only."""

from __future__ import annotations

import pytest
from work.routine_support import make_paper

from seedgraph.library import textcheck
from seedgraph.library.store import open_library_at

from .test_reconvert import garbled_md


@pytest.fixture
def lib(tmp_path):
    library = open_library_at(tmp_path / "home" / "library")
    try:
        yield library
    finally:
        library.close()


def test_papers_get_the_gate_verdict_and_it_is_cached(lib):
    clean = make_paper(lib, 1)
    bad = make_paper(lib, 2, text=garbled_md(), sections=[])

    first = {r.paper_id: r for r in textcheck.check_papers(lib, [clean.paper_id, bad.paper_id])}
    assert first[clean.paper_id].status == "pass" and first[bad.paper_id].status == "garbled"
    assert first[bad.paper_id].markdown_id == bad.markdown_id and first[bad.paper_id].summary
    again = textcheck.check_papers(lib, [bad.paper_id])
    assert again[0].cached is True
    assert textcheck.check_papers(lib, [bad.paper_id], force=True)[0].cached is False


def test_no_ids_means_every_held_paper_with_text(lib):
    make_paper(lib, 1)
    make_paper(lib, 2)
    assert [r.paper_id for r in textcheck.check_papers(lib, None)] == ["lp_test0001", "lp_test0002"]


def test_a_paper_without_text_is_reported_not_raised(lib):
    from datetime import datetime, timezone

    with lib.write() as conn:
        conn.execute("INSERT INTO papers (paper_id, title, created_at) VALUES (?, ?, ?)",
                     ("lp_nopdf", "An invented paper with no PDF", datetime.now(timezone.utc).isoformat()))
    [row] = textcheck.check_papers(lib, ["lp_nopdf"])
    assert row.status == "no_markdown" and row.markdown_id is None


def test_golden_verdict_matches_category(lib):
    clean = make_paper(lib, 1)
    bad = make_paper(lib, 2, text=garbled_md(), sections=[])
    manifest = [
        {"id": "GS01", "category": "math", "v2_markdown_id": clean.markdown_id},
        {"id": "GS06", "category": "garbled", "v2_markdown_id": bad.markdown_id},
        {"id": "GS07", "category": "garbled", "v2_markdown_id": clean.markdown_id},  # a disagreement
        {"id": "GS09", "category": "scan", "v2_markdown_id": "md_" + "0" * 64},  # not in the library
    ]
    rows = {r.golden_id: r for r in textcheck.check_golden(lib, manifest)}

    assert (rows["GS01"].status, rows["GS01"].expected, rows["GS01"].agrees) == ("pass", "pass", True)
    assert (rows["GS06"].status, rows["GS06"].expected, rows["GS06"].agrees) == ("garbled", "garbled", True)
    assert rows["GS07"].agrees is False
    assert rows["GS09"].status == "missing" and rows["GS09"].agrees is False
