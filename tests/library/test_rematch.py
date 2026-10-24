"""Quote re-matching ladder (design §5.5 step 4). All text is invented."""

from __future__ import annotations

from seedgraph.library.rematch import match_quote

CLEAN = (
    "The first step of the invented estimator is identified under different "
    "conditions, and its efficiency reflects the made-up sample design."
)


def test_an_exact_unique_quote_matches_exactly():
    m = match_quote("its efficiency reflects", CLEAN)
    assert m is not None and m.method == "exact"
    assert CLEAN[m.start:m.end] == "its efficiency reflects"


def test_a_folded_old_quote_matches_clean_text():
    old = "The Örst step of the invented estimator is identiÖed under di§erent conditions"
    m = match_quote(old, CLEAN)
    assert m is not None and m.method == "folded"
    assert CLEAN[m.start:m.end] == (
        "The first step of the invented estimator is identified under different conditions")


def test_family_a_ffi_with_a_space_and_fl_fold():
    m = match_quote("its e¢ ciency reáects", CLEAN)
    assert m is not None and m.method == "folded"
    assert CLEAN[m.start:m.end] == "its efficiency reflects"


def test_a_quote_with_different_line_breaks_matches_whitespace_normalised():
    text = "The invented\nestimator   is identified\n\nunder different conditions."
    m = match_quote("estimator is identiÖed under\ndi§erent", text)
    assert m is not None and m.method == "whitespace"
    assert text[m.start:m.end] == "estimator   is identified\n\nunder different"


def test_a_missing_quote_does_not_match():
    assert match_quote("a sentence the new text never had", CLEAN) is None


def test_an_ambiguous_quote_does_not_match():
    assert match_quote("the", "the cat and the dog") is None
    assert match_quote("a  b", "a b and a\nb") is None
