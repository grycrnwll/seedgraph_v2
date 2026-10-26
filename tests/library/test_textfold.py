"""The one text fold shared by highlight, selection and measurement (design §3.2).

The vectors in ``tests/fixtures/textfold_vectors.json`` are shared with ``fold.js``
(``test_textfold_js.py``), so the two implementations cannot drift. Invented text only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seedgraph.library.textfold import find_passage, fold

VECTORS = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "textfold_vectors.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("vec", VECTORS["fold"], ids=lambda v: v["name"])
def test_fold_vectors(vec):
    assert fold(vec["raw"]).text == vec["folded"]


@pytest.mark.parametrize("vec", VECTORS["map"], ids=lambda v: v["name"])
def test_index_map_vectors(vec):
    folded = fold(vec["raw"])
    assert folded.text == vec["folded"]
    assert [list(s) for s in zip(folded.starts, folded.ends)] == vec["spans"]


@pytest.mark.parametrize("vec", VECTORS["find"], ids=lambda v: v["name"])
def test_find_passage_vectors(vec):
    assert [list(h) for h in find_passage(vec["haystack"], vec["needle"])] == vec["hits"]


def test_every_folded_character_maps_back_to_raw_text_that_folds_to_it():
    raw = ("The invented ﬁeld trial of a bak­ery, identiÖed by its oven-\n"
           "timer <sup>3</sup>, shows [7](#page-2-0) a reáected café crust.")
    folded = fold(raw)
    for k, ch in enumerate(folded.text):
        piece = raw[folded.starts[k]:folded.ends[k]]
        assert ch in fold(piece).text or ch in fold(raw[max(0, folded.starts[k] - 1):folded.ends[k] + 1]).text


def test_raw_span_of_a_folded_match_is_the_verbatim_passage():
    raw = "Before it. The made-up **oven** timer\nrings twice; after it."
    folded = fold(raw)
    needle = fold("the made-up oven timer rings twice").text
    [(s, e)] = find_passage(folded.text, needle)
    start, end = folded.raw_span(s, e)
    assert raw[start:end] == "The made-up **oven** timer\nrings twice"


def test_raw_span_of_an_empty_range_is_refused():
    with pytest.raises(ValueError):
        fold("abc").raw_span(1, 1)
