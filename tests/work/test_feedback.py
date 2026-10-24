"""Failed-quote feedback (design §1.5). All text is invented.

Confusable characters are written as escapes so editors cannot silently normalise them.
"""

from __future__ import annotations

import pytest

from seedgraph.anchor import locate_quote
from seedgraph.work.feedback import diagnose_quote

MICRO = "µ"  # MICRO SIGN
MU = "μ"  # GREEK SMALL LETTER MU

MARKDOWN = (
    "# Ovens on the coast\n\n"
    "We estimate the mean " + MU + " of oven heat across seasons and find that "
    "the bakers’ rule is stable in every town we visit.\n\n"
    "Prices respond to demand—mostly in the summer months—and stay put in winter. "
    "The effect is large in harbour towns and small inland. The effect is large.\n\n"
    "Our sample covers forty shops that report their sales each week, which lets us "
    "track changes in the oven tem-\nperature over time.\n\n"
    "Some passages came out garbled: the parameter is identiÖed under di§erent "
    "conditions with e¢ ciency gains.\n\n"
    "## References\n\nBaker, A. (2001). The bakers’ rule. Journal of Ovens.\n"
)
REFS = [(MARKDOWN.index("## References"), len(MARKDOWN))]


def _check_slice_anchors(result):
    """Any slice handed to the agent must itself anchor with the unchanged matcher."""
    if result.markdown_slice is not None:
        assert locate_quote(MARKDOWN, result.markdown_slice) is not None


def test_exact_unique_quote_anchors():
    result = diagnose_quote("the mean " + MU + " of oven heat", MARKDOWN, reference_ranges=REFS)
    assert result.anchored
    assert result.reason is None
    assert result.span == locate_quote(MARKDOWN, "the mean " + MU + " of oven heat")


@pytest.mark.parametrize("quote", [None, "", "   "])
def test_empty_quote(quote):
    result = diagnose_quote(quote, MARKDOWN)
    assert not result.anchored
    assert result.reason == "empty"
    assert "ambiguous" in result.hint


def test_multiple_matches_reports_count():
    result = diagnose_quote("The effect is large", MARKDOWN)
    assert not result.anchored
    assert result.reason == "multiple_matches"
    assert result.match_count == 2
    assert "unique" in result.hint


def test_micro_sign_versus_greek_mu():
    result = diagnose_quote("the mean " + MICRO + " of oven heat", MARKDOWN)
    assert not result.anchored
    assert result.reason == "lookalike_characters"
    assert result.markdown_slice == "the mean " + MU + " of oven heat"
    assert [d.describe() for d in result.differences] == ["U+03BC in text, U+00B5 in quote"]
    assert "U+03BC in text, U+00B5 in quote" in result.hint
    _check_slice_anchors(result)


def test_straight_versus_curly_quotes():
    result = diagnose_quote("the bakers' rule is stable", MARKDOWN)
    assert result.reason == "lookalike_characters"
    assert result.markdown_slice == "the bakers’ rule is stable"
    assert [d.describe() for d in result.differences] == ["U+2019 in text, U+0027 in quote"]
    _check_slice_anchors(result)


def test_dash_lookalike():
    result = diagnose_quote("demand-mostly in the summer months", MARKDOWN)
    assert result.reason == "lookalike_characters"
    assert result.markdown_slice == "demand—mostly in the summer months"
    _check_slice_anchors(result)


def test_ligature_and_no_break_space_lookalikes():
    md = "We ﬁnd a 5 percent rise."
    result = diagnose_quote("find a 5 percent rise", md)
    assert result.reason == "lookalike_characters"
    assert result.markdown_slice == "ﬁnd a 5 percent rise"
    assert locate_quote(md, result.markdown_slice) is not None


def test_whitespace_and_hyphenated_line_break():
    result = diagnose_quote("track changes in the oven temperature over time", MARKDOWN)
    assert result.reason == "whitespace_or_linebreak"
    assert result.markdown_slice == "track changes in the oven tem-\nperature over time"
    _check_slice_anchors(result)


def test_collapsed_whitespace():
    result = diagnose_quote("Our sample  covers forty\nshops", MARKDOWN)
    assert result.reason == "whitespace_or_linebreak"
    assert result.markdown_slice == "Our sample covers forty shops"


def test_repaired_mojibake_points_to_the_garbled_text():
    result = diagnose_quote(
        "the parameter is identified under different conditions with efficiency gains",
        MARKDOWN,
    )
    assert result.reason == "mojibake"
    assert result.markdown_slice == (
        "the parameter is identiÖed under di§erent conditions with e¢ ciency gains"
    )
    _check_slice_anchors(result)


def test_match_inside_references_still_anchors_with_a_warning():
    result = diagnose_quote("Journal of Ovens", MARKDOWN, reference_ranges=REFS)
    assert result.anchored
    assert result.reason == "in_references"
    assert result.span is not None


def test_not_found_offers_the_nearest_passage():
    result = diagnose_quote("forty shops report their sales every single week", MARKDOWN)
    assert not result.anchored
    assert result.reason == "not_found"
    assert result.markdown_slice is not None
    assert "forty shops" in result.markdown_slice
    _check_slice_anchors(result)


def test_not_found_without_a_good_passage():
    result = diagnose_quote("volcanic glass melts quickly under pressure", MARKDOWN)
    assert result.reason == "not_found"
    assert result.markdown_slice is None


def test_folded_but_ambiguous_match_gives_no_slice():
    md = "the bakers’ rule. And again the bakers’ rule."
    result = diagnose_quote("the bakers' rule", md)
    assert result.reason == "lookalike_characters"
    assert result.match_count == 2
    assert result.markdown_slice is None
    assert "unique" in result.hint


def test_diagnosis_serialises():
    d = diagnose_quote("the mean " + MICRO + " of oven heat", MARKDOWN).as_dict()
    assert d["reason"] == "lookalike_characters"
    assert d["differences"] == [{"text": "U+03BC", "quote": "U+00B5"}]
