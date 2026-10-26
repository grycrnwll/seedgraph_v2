"""Quote → page lookup and the selection anchor (design §3.1, §5). Synthetic PDFs and
invented text only."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from seedgraph.cache.marker_backend import MarkerResult
from seedgraph.library import files
from seedgraph.library.pages import anchor_selection, quote_locators
from seedgraph.library.store import open_library_at
from seedgraph.library.textfold import fold
from seedgraph.notes import ClaimDraft, NoteProvenance, save_note

from .test_files import text_pdf

PAGES = [
    "The invented bakery opens at dawn. Its ovens warm slowly.",
    "A made-up flour index rises in winter. The index falls in spring.",
    "Bakers in the story share one long table.",
]
DELIM = "-" * 48


def paged_md(pages: list[str], *, first: int = 0, numbers: list[int] | None = None) -> str:
    """Markdown with Marker page delimiters (``{N}`` + 48 dashes, N 0-based)."""
    nums = numbers if numbers is not None else list(range(first, first + len(pages)))
    return "".join(f"\n\n{{{n}}}{DELIM}\n\n{body}" for n, body in zip(nums, pages)) + "\n"


def plain_md(pages: list[str]) -> str:
    return "# Invented Bakery\n\n" + "\n\n".join(pages) + "\n"


class FixedBackend:
    """A fake converter returning ``markdown`` (changeable between conversions)."""

    version = "fake-fixed-1"

    def __init__(self, markdown: str):
        self.markdown = markdown

    def __call__(self, pdf_path, cfg):
        return MarkerResult(markdown=self.markdown, block_json=None, warnings=[])


@pytest.fixture
def lib(tmp_path):
    library = open_library_at(tmp_path / "home" / "library")
    try:
        yield library
    finally:
        library.close()


def make_paper(lib, tmp_path, markdown: str, *, pdf_pages: list[str] = PAGES,
               paper_id: str = "lp_00000000000000000042", name: str = "a.pdf") -> tuple[str, str, str]:
    """A held paper with a synthetic PDF converted to ``markdown``: (paper, file, md)."""
    now = datetime.now(timezone.utc).isoformat()
    with lib.write() as conn:
        conn.execute("INSERT INTO papers (paper_id, title, created_at) VALUES (?, ?, ?)",
                     (paper_id, "Invented Bakery", now))
    res = files.attach_pdf(lib, paper_id, text_pdf(tmp_path / name, pdf_pages, marker=paper_id))
    outcome = files.convert_file(lib, res.source_file_id, backend=FixedBackend(markdown))
    assert outcome.status == "converted"
    return paper_id, res.source_file_id, outcome.markdown_id


def reconvert_to(lib, source_file_id: str, markdown: str) -> str:
    outcome = files.convert_file(lib, source_file_id, backend=FixedBackend(markdown), force=True)
    assert outcome.status == "converted"
    return outcome.markdown_id


def save_quotes(lib, paper_id: str, md_id: str, quotes: list[str | None], *,
                claim_type: str = "main_result") -> list[str]:
    """Save a note whose claims quote ``quotes`` (None = no quote); returns claim ids in order."""
    drafts = [ClaimDraft(claim_type=claim_type, field_key=f"main_results[{i}]",
                         status="found" if q else "not_found", epistemic_type="llm_extracted",
                         normalized_label=f"invented label {i}", claim_text=f"Invented claim {i}.",
                         exact_quote=q, confidence=0.7)
              for i, q in enumerate(quotes)]
    mh = lib.conn.execute("SELECT markdown_hash FROM markdown_documents WHERE markdown_id = ?",
                          (md_id,)).fetchone()[0]
    with lib.write() as conn:
        saved = save_note(conn, paper_id=paper_id, markdown_id=md_id, markdown_hash=mh, drafts=drafts,
                          note_text="An invented note.", archetype="empirical",
                          raw_note_json=json.dumps({"q": quotes}),
                          provenance=NoteProvenance(client="invented-agent", model="m-1"))
    assert saved.quotes_anchored == sum(1 for q in quotes if q)
    rows = lib.conn.execute(
        "SELECT claim_id, field_key FROM extracted_claims WHERE structured_note_id = ?",
        (saved.note_id,)).fetchall()
    by_key = {r[1]: r[0] for r in rows}
    return [by_key[f"main_results[{i}]"] for i in range(len(quotes))]


# --- quote locators -------------------------------------------------------------------


def test_marker_pages_come_from_the_span(lib, tmp_path):
    pid, _, md = make_paper(lib, tmp_path, paged_md(PAGES))
    c1, c2, c3 = save_quotes(lib, pid, md, ["Its ovens warm slowly.", "The index falls in spring.", None])

    locs = {loc.claim_id: loc for loc in quote_locators(lib, pid)}

    assert set(locs) == {c1, c2}  # a claim without a quote has no locator
    assert (locs[c1].page, locs[c1].page_end, locs[c1].how) == (1, 1, "marker")
    assert (locs[c2].page, locs[c2].how) == (2, "marker")
    assert locs[c1].needle == fold("Its ovens warm slowly.").text == "itsovenswarmslowly"
    assert locs[c1].span_id and locs[c1].warning is None


def test_a_span_without_pages_is_reanchored_into_the_current_paged_markdown(lib, tmp_path):
    pid, sf, old_md = make_paper(lib, tmp_path, plain_md(PAGES))
    [c1] = save_quotes(lib, pid, old_md, ["The index falls in spring."])
    reconvert_to(lib, sf, paged_md(PAGES))

    [loc] = quote_locators(lib, pid)

    assert (loc.claim_id, loc.page, loc.page_end, loc.how) == (c1, 2, 2, "reanchored")


def test_no_page_information_is_never_guessed(lib, tmp_path):
    pid, _, md = make_paper(lib, tmp_path, plain_md(PAGES))
    save_quotes(lib, pid, md, ["Bakers in the story share one long table."])

    [loc] = quote_locators(lib, pid)

    assert (loc.page, loc.page_end, loc.how) == (None, None, "none")


def test_a_quote_that_is_not_unique_in_the_current_markdown_is_not_reanchored(lib, tmp_path):
    pid, sf, old_md = make_paper(lib, tmp_path, plain_md(PAGES))
    save_quotes(lib, pid, old_md, ["Its ovens warm slowly."])
    reconvert_to(lib, sf, paged_md(PAGES + ["Its ovens warm slowly."]))

    [loc] = quote_locators(lib, pid)

    assert (loc.page, loc.how) == (None, "none")


def test_a_page_beyond_the_pdf_is_reported_as_none_with_a_warning(lib, tmp_path):
    md_text = paged_md(PAGES, numbers=[0, 1, 7])  # the markdown claims an 8th page
    pid, _, md = make_paper(lib, tmp_path, md_text)
    save_quotes(lib, pid, md, ["Bakers in the story share one long table."])

    [loc] = quote_locators(lib, pid)

    assert (loc.page, loc.how) == (None, "none")
    assert "8" in loc.warning and "3" in loc.warning


def test_without_the_current_claims_view_the_current_notes_claims_are_read(lib, tmp_path):
    pid, _, md = make_paper(lib, tmp_path, paged_md(PAGES))
    save_quotes(lib, pid, md, ["Its ovens warm slowly."])
    with_view = quote_locators(lib, pid)
    with lib.write() as conn:
        conn.execute("DROP VIEW IF EXISTS current_claims")

    assert quote_locators(lib, pid) == with_view != []


def test_lens_claims_have_no_locator(lib, tmp_path):
    pid, _, md = make_paper(lib, tmp_path, paged_md(PAGES))
    save_quotes(lib, pid, md, ["Its ovens warm slowly."], claim_type="lens:invented")

    assert quote_locators(lib, pid) == []


# --- selection anchor ------------------------------------------------------------------


def test_a_selection_anchors_to_the_verbatim_markdown_widened_to_whole_words(lib, tmp_path):
    pages = ["The invented bakery opens at dawn. Its **ovens** warm slo-\nwly.", PAGES[1], PAGES[2]]
    md_text = paged_md(pages)
    pid, _, md = make_paper(lib, tmp_path, md_text)

    res = anchor_selection(lib, pid, 1, "ts  ovens\nwarm slo")

    assert res.status == "ok" and res.matches == 1
    assert res.quote == "Its **ovens** warm slo-\nwly"
    assert md_text[res.start_char:res.end_char] == res.quote
    assert res.markdown_id == md


def test_a_selection_across_a_page_break_anchors_in_the_widened_slice(lib, tmp_path):
    md_text = paged_md(PAGES)
    pid, _, _ = make_paper(lib, tmp_path, md_text)

    res = anchor_selection(lib, pid, 1, "warm slowly. A made-up")

    assert res.status == "ok"
    assert res.quote.startswith("warm slowly.") and res.quote.endswith("A made-up")
    assert md_text[res.start_char:res.end_char] == res.quote


def test_a_selection_that_appears_twice_is_refused(lib, tmp_path):
    pid, _, _ = make_paper(lib, tmp_path, paged_md(PAGES))

    res = anchor_selection(lib, pid, 2, "index")

    assert (res.status, res.matches, res.quote) == ("ambiguous", 2, None)


def test_a_selection_unique_near_its_page_but_repeated_elsewhere_is_refused(lib, tmp_path):
    pages = PAGES + ["Filler page four.", "Filler page five.", "Its ovens warm slowly."]
    pid, _, _ = make_paper(lib, tmp_path, paged_md(pages), pdf_pages=pages)

    res = anchor_selection(lib, pid, 1, "Its ovens warm slowly")

    assert res.status == "ambiguous" and res.quote is None


def test_a_selection_not_in_the_markdown_is_not_found(lib, tmp_path):
    pid, _, _ = make_paper(lib, tmp_path, paged_md(PAGES))

    assert anchor_selection(lib, pid, 1, "a sentence nobody wrote").status == "not_found"
    assert anchor_selection(lib, pid, 1, " ;; ").status == "empty"


def test_a_paper_without_page_markers_refuses_selection_anchoring(lib, tmp_path):
    pid, _, _ = make_paper(lib, tmp_path, plain_md(PAGES))

    assert anchor_selection(lib, pid, 1, "Its ovens warm slowly").status == "no_markers"
