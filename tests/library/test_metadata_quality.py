"""Title cleaning and the generic-title stoplist (docs/V3_UI_DESIGN.md §7). Pure; invented titles."""

from __future__ import annotations

import pytest

from seedgraph.library.metadata_quality import clean_title, generic_title_reason


@pytest.mark.parametrize(("raw", "clean"), [
    ("Invented Ovens and the Morning Loaf†", "Invented Ovens and the Morning Loaf"),
    ("Invented Ovens and the Morning Loaf ‡", "Invented Ovens and the Morning Loaf"),
    ("Invented Ovens and the Morning Loaf*", "Invented Ovens and the Morning Loaf"),
    ("Invented Ovens and the Morning Loaf§¶", "Invented Ovens and the Morning Loaf"),
    ("Invented Ovens and the Morning Loaf¹", "Invented Ovens and the Morning Loaf"),
    ("Invented Ovens and the Morning Loaf²³", "Invented Ovens and the Morning Loaf"),
    ("Invented Ovens and the Morning Loaf<sup>1</sup>", "Invented Ovens and the Morning Loaf"),
    ("Invented Ovens and the Morning Loaf<sup>*,†</sup>.", "Invented Ovens and the Morning Loaf"),
    ("  Invented   Ovens\nand the  Morning Loaf ;", "Invented Ovens and the Morning Loaf"),
    ("Invented Ovens: A Survey,", "Invented Ovens: A Survey"),
])
def test_footnote_marks_and_stray_punctuation_are_stripped(raw, clean):
    assert clean_title(raw) == clean


@pytest.mark.parametrize("keep", [
    "Invented Markets after COVID-19",
    "Bakery Platforms in Web 2.0",
    "Do Invented Ovens Work?",
    "Invented Ovens!",
    "The 1999 Invented Loaf Survey",
])
def test_ordinary_digits_and_question_marks_survive(keep):
    assert clean_title(keep) == keep


def test_clean_title_of_nothing_is_none():
    assert clean_title(None) is None
    assert clean_title("  † ") is None


@pytest.mark.parametrize("generic", [
    "Review", "INTRODUCTION", "Contents", "Table of Contents", "Abstract", "Front matter",
    "Preface", "Acknowledgements", "Editorial", "Book review", "Article", "Untitled",
    "Working Paper", "Chapter 3", "chapter IV", "Introduction†", "Ab",
])
def test_generic_titles_have_a_reason(generic):
    assert generic_title_reason(generic) is not None


def test_a_title_equal_to_the_venue_is_generic():
    assert generic_title_reason("Journal of Invented Results", venue="journal of invented results") is not None


@pytest.mark.parametrize("ok", [
    "A Review of Invented Ovens", "Introduction to Invented Bakeries", "Invented Ovens and the Morning Loaf",
])
def test_real_titles_are_not_generic(ok):
    assert generic_title_reason(ok) is None
    assert generic_title_reason(ok, venue="Journal of Invented Results") is None
