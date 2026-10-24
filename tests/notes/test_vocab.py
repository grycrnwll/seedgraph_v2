"""Fixed note vocabularies (design §4.1): variants map, unknown -> other, never a rejection."""

from __future__ import annotations

import pytest

from seedgraph.notes import vocab


@pytest.mark.parametrize(
    ("group", "raw", "expected"),
    [
        ("method_type", "estimator", "estimator"),
        ("method_type", "Monte Carlo simulation", "simulation"),
        ("method_type", "simulation study", "simulation"),
        ("method_type", "structural model", "model"),
        ("method_type", "Research_Design", "research_design"),
        ("method_type", "difference-in-differences", "research_design"),
        ("method_type", "bootstrap", "inference"),
        ("method_type", "asymptotic analysis", "asymptotic_theory"),
        ("result_type", "simulation finding", "simulation"),
        ("result_type", "Monte Carlo result", "simulation"),
        ("result_type", "asymptotic normality", "asymptotic"),
        ("result_type", "empirical finding", "empirical"),
        ("result_type", "theorem", "theoretical"),
        ("result_type", "policy implication", "policy"),
        ("limitation_type", "external_validity", "external_validity"),
        ("limitation_type", "data limitation", "data"),
        ("limitation_type", "measurement error", "measurement"),
        ("limitation_type", "scope condition", "scope"),
        ("limitation_type", "model misspecification", "model_specification"),
        ("assumption_type", "identifying", "identification"),
        ("assumption_type", "regularity condition", "regularity"),
        ("assumption_type", "general", "general"),
    ],
)
def test_known_variants_map_to_the_fixed_value(group, raw, expected):
    match = vocab.map_label(group, raw)
    assert match.value == expected
    assert match.known is True


@pytest.mark.parametrize(
    "raw", ["model selection", "model fit", "model comparison", "model selection criterion"]
)
def test_model_words_that_are_not_models_do_not_become_model(raw):
    assert vocab.map_label("method_type", raw).value != "model"


def test_unknown_value_becomes_other_and_is_flagged():
    match = vocab.map_label("method_type", "a wholly invented procedure name")
    assert match.value == "other"
    assert match.known is False
    assert match.raw == "a wholly invented procedure name"


def test_unknown_assumption_type_folds_to_general_and_is_flagged():
    match = vocab.map_label("assumption_type", "something invented")
    assert match.value == "general"
    assert match.known is False


def test_missing_value_is_not_flagged():
    match = vocab.map_label("result_type", None)
    assert match.value is None
    assert match.known is True


def test_every_group_lists_other_except_assumptions():
    for group, values in vocab.VOCABULARIES.items():
        if group == "assumption_type":
            assert values == ("identification", "regularity", "general")
        else:
            assert values[-1] == "other"


def test_unknown_group_is_an_error():
    with pytest.raises(KeyError):
        vocab.map_label("colour_type", "red")
