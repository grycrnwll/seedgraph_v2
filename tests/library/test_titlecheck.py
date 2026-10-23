"""Behaviour of the shared title matcher (design §7). All titles and text are invented."""

from __future__ import annotations

from seedgraph.library.titlecheck import (
    MISS_THRESHOLD,
    PASS_THRESHOLD,
    check_pdf_text,
    first_pages_from_markdown,
    first_pages_from_pdf,
    title_similarity,
)

TITLE = "Bayesian Thermostats for Coastal Bakeries"
FILLER = " ".join(f"word{i}" for i in range(400))


def test_no_text_is_unknown_never_a_miss():
    assert check_pdf_text(TITLE, None, []).status == "unknown"
    assert check_pdf_text(TITLE, None, ["", "  \n"]).status == "unknown"


def test_no_title_is_unknown():
    assert check_pdf_text(None, None, ["Some text about ovens."]).status == "unknown"
    assert check_pdf_text("  ", None, ["Some text about ovens."]).status == "unknown"


def test_exact_title_with_markup_and_line_break_hyphenation_passes():
    page = "# **Bayesian Thermo-\n  stats for Coastal** Bakeries<sup>1</sup>\n\nJane Doe"
    result = check_pdf_text(TITLE, None, [page])
    assert (result.status, result.via, result.score) == ("pass", "exact", 1.0)


def test_unicode_ligatures_and_diacritics_are_normalised():
    page = "Eﬃcient Café Scheduling under Uncertainty"
    result = check_pdf_text("Efficient Cafe Scheduling under Uncertainty", None, [page])
    assert (result.status, result.via) == ("pass", "exact")


def test_doi_on_page_passes_even_when_title_differs():
    page = "Journal of Invented Results\nhttps://doi.org/10.9999/JIR.2020.\n001 \n Unrelated heading"
    result = check_pdf_text("A Different Title Entirely Here", "https://doi.org/10.9999/jir.2020.001", [page])
    assert (result.status, result.via) == ("pass", "doi")


def test_doi_prefix_of_a_longer_doi_does_not_count():
    page = "doi: 10.9999/abc.12345 Something else about pastries"
    result = check_pdf_text("Rainfall and Mortgage Defaults Revisited", "10.9999/abc.1234", [page])
    assert result.via != "doi"


def test_mojibake_fi_ffi_ff_glyphs_match_as_a_variant():
    # Marker maps ligatures to Latin-1 lookalikes; the page text is never rewritten.
    page = "IdentiÖcation of E¢ cient Di¤erence Estimators in …nite Samples"
    result = check_pdf_text("Identification of Efficient Difference Estimators in Finite Samples", None, [page])
    assert (result.status, result.via) == ("pass", "exact")
    assert "folded" in result.evidence


def test_fold_does_not_touch_real_capital_letters():
    # A name starting with Ö followed by an uppercase letter or space is left alone, so the
    # original variant still matches the original title.
    page = "ÖSTERGAARD AND THE LAKE: A HISTORY OF FISHERIES"
    assert check_pdf_text("Östergaard and the Lake: A History of Fisheries", None, [page]).status == "pass"


def test_mojibake_title_matches_clean_page():
    result = check_pdf_text("SpeciÖcation Tests for Bakery Demand", None, ["Specification Tests for Bakery Demand"])
    assert result.status == "pass"


def test_hyphenated_compound_matches_its_closed_form():
    page = "Classification of Nonparametric Pastry Functions in Longitudinal Bakery Models " + FILLER
    result = check_pdf_text("Classification of Non-Parametric Pastry Functions in Longitudinal Bakery Models", None, [page])
    assert result.status == "pass"
    reverse = check_pdf_text("Nonparametric Pastry Functions", None, ["Non-parametric pastry functions " + FILLER])
    assert reverse.status == "pass"


def test_tolerant_match_with_reordered_title_words_passes():
    page = "Coastal Bakeries and Bayesian Thermostats: evidence " + FILLER
    result = check_pdf_text(TITLE, None, [page])
    assert result.status == "pass"
    assert result.via == "tolerant"
    assert result.score >= PASS_THRESHOLD


def test_scattered_title_words_across_a_long_page_do_not_pass():
    # Each title word appears, but far apart: no single window covers them.
    words = ["bayesian", "thermostats", "coastal", "bakeries"]
    page = " ".join(f"{w} {FILLER}" for w in words)
    result = check_pdf_text(TITLE, None, [page])
    assert result.status == "miss"
    assert result.score < MISS_THRESHOLD


def test_partial_overlap_is_weak():
    page = "Bayesian Thermostats in Alpine Dairies " + FILLER
    result = check_pdf_text(TITLE, None, [page])
    assert result.status == "weak"
    assert MISS_THRESHOLD <= result.score < PASS_THRESHOLD


def test_unrelated_page_is_a_miss():
    page = "Volcanic Soil and Grape Yields in Ancient Vineyards " + FILLER
    result = check_pdf_text(TITLE, None, [page])
    assert result.status == "miss"
    assert result.score == 0.0


def test_short_title_needs_every_content_token():
    page = "Ovens Pricing " + FILLER  # 2 of 3 content tokens
    result = check_pdf_text("Ovens, Flour and Pricing", None, [page])
    assert result.status != "pass"
    page_all = "Pricing of Flour Ovens " + FILLER
    assert check_pdf_text("Ovens, Flour and Pricing", None, [page_all]).status == "pass"


def test_title_found_on_a_later_first_page_passes():
    pages = ["Cover sheet. Working paper series.", "Table of contents", TITLE]
    assert check_pdf_text(TITLE, None, pages).status == "pass"


# --- first pages from markdown -------------------------------------------------------


def _paged(*bodies: str) -> str:
    return "".join(f"\n\n{{{i}}}{'-' * 48}\n\n{body}" for i, body in enumerate(bodies))


def test_first_pages_split_on_marker_page_markers():
    md = _paged("page zero", "page one", "page two", "page three")
    pages = first_pages_from_markdown(md)
    assert [p.strip() for p in pages] == ["page zero", "page one", "page two"]
    assert [p.strip() for p in first_pages_from_markdown(md, n=1)] == ["page zero"]


def test_title_beyond_the_first_pages_is_not_seen():
    md = _paged("cover", "blank", "contents", TITLE)
    assert check_pdf_text(TITLE, None, first_pages_from_markdown(md)).status == "miss"


def test_markdown_without_markers_uses_the_first_8000_characters():
    md = TITLE + " " + "x" * 9000 + " tail-marker-words"
    pages = first_pages_from_markdown(md)
    assert len(pages) == 1
    assert len(pages[0]) == 8000
    assert first_pages_from_markdown("") == []


def test_unreadable_pdf_gives_none_and_no_text_is_unknown(tmp_path):
    bogus = tmp_path / "not_a_pdf.pdf"
    bogus.write_bytes(b"this is not a pdf")
    assert first_pages_from_pdf(bogus) is None
    assert check_pdf_text(TITLE, None, first_pages_from_pdf(bogus) or []).status == "unknown"


# --- title vs title -------------------------------------------------------------------


def test_title_similarity_is_symmetric_and_ignores_stopwords_and_case():
    assert title_similarity("The Ovens of Paris", "ovens paris") == 1.0
    a, b = "Bayesian Thermostats for Coastal Bakeries", "Bayesian Thermostats in Alpine Dairies"
    assert title_similarity(a, b) == title_similarity(b, a)
    assert 0.0 < title_similarity(a, b) < 1.0
    assert title_similarity(a, None) == 0.0
