"""Behaviour of the text-quality gate (design §5.2). All text is invented."""

from __future__ import annotations

from seedgraph.library.textquality import assess_text

CLEAN = (
    "We study how bakeries along the coast adjust their ovens when the weather turns. "
    "The model has two kinds of bakers and a simple rule for prices. "
    "Our data come from a survey of small shops over many seasons. "
)


def _words(n: int) -> str:
    """About ``n`` clean words of invented prose."""
    base = CLEAN.split()
    return " ".join(base[i % len(base)] for i in range(n))


def test_clean_text_passes():
    result = assess_text(_words(3000))
    assert result.status == "pass"
    assert result.doc_rate == 0.0


def test_family_a_garbled_text_is_flagged():
    garbled = (
        "The parameter is identiÖed under di§erent conditions. "
        "We gain e¢ ciency when the model reáects the authorís view. "
        "The Örst step is speciÖcation. "
    ) * 10
    result = assess_text(_words(1500) + " " + garbled)
    assert result.status == "garbled"
    assert result.hits_by_signal["Ö"] >= 20
    assert result.hits_by_signal["§"] >= 10
    assert result.hits_by_signal["¢"] >= 10
    assert result.examples


def test_family_b_garbled_text_is_flagged():
    garbled = "The …rst di¤erence is coe¢cient times e¤ect in…nity. " * 10
    result = assess_text(_words(1500) + " " + garbled)
    assert result.status == "garbled"
    assert result.hits_by_signal["…"] >= 20
    assert result.hits_by_signal["¤"] >= 20
    assert result.hits_by_signal["¢"] >= 10


def test_accented_names_and_ordinary_symbols_pass():
    names = (
        "As shown by Sánchez and Martínez, and later by García, Galán and Gómez, the effect holds. "
        "Rodríguez agrees; see Section § 3 and the price of 5¢ per loaf. "
        "Weiß reports a ratio of Þ and the list goes on… and on. "
    ) * 15
    # A capitalised name starting with Ö is a strong hit by the design's rule, so one
    # such name is fine but a page full of them is not part of this fixture.
    result = assess_text(_words(2000) + " " + names + " Work by Ölmez also agrees.")
    assert result.status == "pass"
    assert result.corroborating_hits >= 60  # in-word á and í are counted ...
    assert result.hits_by_signal.get("á", 0) >= 15
    assert result.hits_by_signal.get("§", 0) == 0  # ... but symbols in context are not


def test_converter_garbage_is_reported_but_never_garbled():
    junk = "The (cid:12)(cid:7) value \ufffd is \ue001 here.\n| a | \ufffd |\n" * 30
    result = assess_text(_words(1000) + "\n" + junk)
    assert result.status == "pass"
    assert result.strong_hits == 0
    assert result.hits_by_signal["(cid:N)"] == 60
    assert result.hits_by_signal["U+FFFD"] == 60
    assert result.hits_by_signal["private_use"] == 30
    assert result.corroborating_hits == 150
    assert any("\ufffd" in ex for ex in result.examples)
    assert "replacement characters" in result.summary
    assert "in tables" in result.summary


def test_converter_garbage_does_not_add_to_ligature_hits():
    pages = [_words(500) + " \ufffd\ufffd\ufffd \ue001\ue001"] * 10
    result = assess_text(_paged(pages))
    assert result.bad_pages == 0
    assert result.status == "pass"


def _paged(pages: list[str]) -> str:
    return "".join(f"{{{i}}}{'-' * 48}\n\n{p}\n\n" for i, p in enumerate(pages))


def test_page_rule_catches_a_garbled_section():
    bad = _words(500) + " identiÖed and di§erent"  # 2 strong hits in ~500 words: a bad page
    pages = [_words(500)] * 16 + [bad] * 4
    result = assess_text(_paged(pages))
    assert result.pages == 20
    assert result.strong_hits == 8 and result.doc_rate < 1.0  # the document rule alone misses it
    assert result.bad_pages == 4
    assert result.status == "garbled"


def test_page_rule_needs_twenty_percent_of_pages():
    bad = _words(500) + " identiÖed and di§erent"
    result = assess_text(_paged([_words(500)] * 17 + [bad] * 3))
    assert result.bad_pages == 3
    assert result.status == "pass"


def test_short_pages_are_never_bad():
    bad = _words(40) + " identiÖed and di§erent"  # under 50 words
    result = assess_text(_paged([_words(500)] * 6 + [bad] * 4))
    assert result.bad_pages == 0


def test_no_text_layer():
    pages = ["Page 3", "", "Figure 2", _words(10), "", _words(500)]
    assert assess_text(_paged(pages)).status == "no_text"
    assert assess_text("").status == "no_text"


def test_text_without_page_markers_is_one_page():
    result = assess_text(_words(800))
    assert result.pages == 1
    assert result.status == "pass"


def test_reference_list_is_not_scored():
    body = _words(2000)
    refs = "\n\n## References\n\n" + "Ö§¢… identiÖed, di§erent, e¢ ciency. " * 20
    md = body + refs
    assert assess_text(md).status == "garbled"
    result = assess_text(md, reference_ranges=[(len(body), len(md))])
    assert result.status == "pass"
    assert result.strong_hits == 0


def test_result_is_explainable_and_versioned():
    result = assess_text(_words(1500) + " identiÖed di§erent" * 10)
    assert result.checker_version
    assert "Ö" in result.summary
    assert any("identiÖed" in ex for ex in result.examples)
    assert result.as_dict()["status"] == "garbled"


# --- repeated text: a corroborating signal that never gates ---------------------------------

def test_degenerate_repetition_is_reported_but_does_not_change_the_verdict():
    result = assess_text(_words(1500) + "\n\n" + "the gadget estimator converges " * 12 + "\n\n" + _words(200))
    assert result.status == "pass"
    assert result.repeated_runs == 1
    assert "repeated text: 1 run" in result.summary
    assert "the gadget estimator converges" in result.summary


def test_a_repeated_line_is_reported():
    line = "Invented Journal of Gadget Studies, volume nine\n"
    result = assess_text(_words(1500) + "\n" + line * 6 + _words(100))
    assert result.repeated_runs == 1


def test_clean_text_has_no_repeated_runs_and_an_unchanged_summary():
    result = assess_text(_words(3000))
    assert result.repeated_runs == 0
    assert result.summary == "pass: no mojibake signals"


def test_numbers_tables_and_short_repeats_are_not_repetition():
    table = "| 0 | 0 | 0 | 0 | 0 | 0 | 0 |\n" * 3
    numbers = "0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00\n"
    few = "very " * 4 + "good\n"
    result = assess_text(_words(1500) + "\n" + table + numbers + few)
    assert result.repeated_runs == 0


def test_repetition_never_turns_garbled_text_into_a_pass_or_back():
    garbled = _words(1500) + " identiÖed di§erent" * 10
    plain = assess_text(garbled)
    repeated = assess_text(garbled + "\n" + "same words again " * 10)
    assert plain.status == repeated.status == "garbled"
    assert repeated.repeated_runs == plain.repeated_runs + 1


def test_checker_version_was_bumped_for_the_repetition_signal():
    assert assess_text("x").checker_version == "textquality-3"
