"""Schema 2.0.0 note format: validation, task checks and normalisation (design §3, §4).

All note text here is invented.
"""

from __future__ import annotations

import json

from seedgraph.notes import format as fmt


def _assumption(**kw):
    base = {
        "claim_text": "Invented units do not react before the invented programme starts.",
        "normalized_label": "no anticipation",
        "status": "found",
        "assumption_type": "identification",
        "plain_words": "Nobody changes what they do before the change begins.",
        "exact_quote": "an invented quote about anticipation",
    }
    base.update(kw)
    return base


def _method(**kw):
    base = {
        "claim_text": "An invented two-step widget estimator.",
        "normalized_label": "widget estimator",
        "status": "found",
        "method_type": "estimator",
        "plain_words": "Estimate the widget count in two passes.",
        "exact_quote": "an invented quote about the estimator",
    }
    base.update(kw)
    return base


def _estimand(**kw):
    base = {
        "claim_text": "The invented average widget effect.",
        "normalized_label": "average widget effect",
        "status": "found",
        "plain_words": "How much widgets change on average.",
        "exact_quote": "an invented quote about the target",
    }
    base.update(kw)
    return base


def _note(**kw):
    base = {
        "research_question": {"claim_text": "Do invented widgets grow?", "status": "found",
                              "exact_quote": "do invented widgets grow"},
        "estimand_or_target_object": _estimand(),
        "method_or_model": [_method()],
        "assumptions": [_assumption()],
    }
    base.update(kw)
    return base


# --- validation ---------------------------------------------------------------------------

def test_validate_accepts_a_dict_and_raw_text_with_preamble():
    note, errors = fmt.validate_note(_note())
    assert errors == [] and note is not None
    note2, errors2 = fmt.validate_note("Here is the note:\n```json\n" + json.dumps(_note()) + "\n```")
    assert errors2 == [] and note2 == note


def test_validate_reports_unusable_json():
    note, errors = fmt.validate_note("no json here")
    assert note is None and errors


def test_validate_coerces_a_list_valued_type_string():
    note, errors = fmt.validate_note(_note(method_or_model=[_method(method_type=["estimator"])]))
    assert errors == []
    assert note.method_or_model[0].method_type == "estimator"


def test_epistemic_type_is_not_in_the_agent_envelope_and_is_ignored_if_sent():
    sent = _note(assumptions=[_assumption(epistemic_type="user_validated")])
    note, errors = fmt.validate_note(sent)
    assert errors == []
    assert "epistemic_type" not in note.assumptions[0].model_dump()
    assert "epistemic_type" not in fmt.ClaimEnvelope.model_fields


def test_schema_version_is_2_0_0_on_the_same_schema_id():
    assert fmt.SCHEMA_ID == "default_research_note_v1"
    assert fmt.SCHEMA_VERSION == "2.0.0"
    assert fmt.PROMPT_VERSION == "2.0.0"


# --- task checks --------------------------------------------------------------------------

def _check(raw):
    note, errors = fmt.validate_note(raw)
    assert errors == []
    return fmt.check_note(note)


def test_a_complete_note_has_no_errors_or_warnings():
    result = _check(_note())
    assert result.errors == [] and result.warnings == []
    assert result.ok


def test_plain_words_required_for_found_assumption_method_and_estimand():
    result = _check(_note(
        assumptions=[_assumption(plain_words=None)],
        method_or_model=[_method(plain_words="  ")],
        estimand_or_target_object=_estimand(plain_words=None),
    ))
    paths = {e.path for e in result.errors}
    assert paths == {"assumptions[0].plain_words", "method_or_model[0].plain_words",
                     "estimand_or_target_object.plain_words"}
    assert all("plain" in e.message for e in result.errors)
    assert not result.ok


def test_a_status_read_as_found_still_needs_plain_words():
    result = _check(_note(assumptions=[_assumption(status="Found", plain_words=None)]))
    assert [e.path for e in result.errors] == ["assumptions[0].plain_words"]


def test_plain_words_not_required_when_the_claim_is_not_found():
    result = _check(_note(
        assumptions=[_assumption(status="ambiguous", plain_words=None)],
        estimand_or_target_object={"status": "not_found"},
    ))
    assert result.errors == []


def test_inferred_claim_without_explanation_is_an_error_not_a_crash():
    result = _check(_note(assumptions=[_assumption(assertion_status="inferred")]))
    assert [e.path for e in result.errors] == ["assumptions[0].inferred_explanation"]


def test_unknown_type_is_a_warning_listing_allowed_values_never_an_error():
    result = _check(_note(method_or_model=[_method(method_type="a wholly invented procedure")]))
    assert result.errors == []
    (warning,) = result.warnings
    assert warning.path == "method_or_model[0].method_type"
    assert "other" in warning.message and "estimator" in warning.message


def test_known_variant_gives_no_warning():
    result = _check(_note(method_or_model=[_method(method_type="Monte Carlo simulation")]))
    assert result.warnings == []


# --- normalisation ------------------------------------------------------------------------

def _drafts(*raws):
    notes = [fmt.validate_note(r)[0] for r in raws]
    return fmt.normalize_parts(notes)


def _by_key(norm):
    return {d.field_key: d for d in norm.drafts}


def test_epistemic_type_is_derived_from_assertion_status():
    norm = _drafts(_note(assumptions=[
        _assumption(epistemic_type="user_validated"),
        _assumption(assertion_status="inferred", inferred_explanation="invented reasoning",
                    epistemic_type="llm_extracted"),
    ]))
    d = _by_key(norm)
    assert d["assumptions[0]"].epistemic_type == "llm_extracted"
    assert d["assumptions[1]"].epistemic_type == "llm_inferred"


def test_types_are_mapped_through_the_vocabulary():
    norm = _drafts(_note(
        method_or_model=[_method(method_type="structural model"),
                         _method(method_type="Monte Carlo simulation"),
                         _method(method_type="model selection")],
        assumptions=[_assumption(assumption_type="regularity condition")],
        main_results=[{"claim_text": "Invented finding.", "status": "found",
                       "result_type": "simulation finding", "exact_quote": "q"}],
        limitations=[{"claim_text": "Invented limit.", "status": "found",
                      "limitation_type": "a wholly invented kind", "exact_quote": "q2"}],
    ))
    d = _by_key(norm)
    assert (d["method_or_model[0]"].claim_type, d["method_or_model[0]"].claim_subtype) == ("model", "model")
    assert (d["method_or_model[1]"].claim_type, d["method_or_model[1]"].claim_subtype) == ("method", "simulation")
    assert d["method_or_model[2]"].claim_type == "method"
    assert d["assumptions[0]"].claim_type == "regularity_condition"
    assert d["assumptions[0]"].claim_subtype is None
    assert d["main_results[0]"].claim_subtype == "simulation"
    assert d["limitations[0]"].claim_subtype == "other"


def test_plain_words_travel_on_the_drafts():
    d = _by_key(_drafts(_note()))
    assert d["assumptions[0]"].plain_words == "Nobody changes what they do before the change begins."
    assert d["method_or_model[0]"].plain_words == "Estimate the widget count in two passes."
    assert d["estimand_or_target_object"].plain_words == "How much widgets change on average."
    assert d["research_question"].plain_words is None


def test_absent_fields_give_one_not_found_row_each():
    norm = _drafts({})
    keys = [(d.field_key, d.status) for d in norm.drafts]
    assert ("research_question", "not_found") in keys
    assert ("method_or_model", "not_found") in keys
    assert len(keys) == 11
    assert norm.archetype == "empty"


def test_parts_take_scalars_from_part_one_and_concatenate_arrays():
    part1 = _note(main_results=[])
    part2 = {
        "research_question": {"claim_text": "A restated invented question.", "status": "found",
                              "exact_quote": "restated"},
        "method_or_model": [_method(normalized_label="second widget method")],
        "main_results": [{"claim_text": "Invented later result.", "status": "found",
                          "result_type": "empirical", "exact_quote": "later"}],
    }
    norm = _drafts(part1, part2)
    d = _by_key(norm)
    assert d["research_question"].claim_text == "Do invented widgets grow?"
    assert d["research_question"].source_chunk_index == 0
    assert d["method_or_model[0]"].source_chunk_index == 0
    assert d["method_or_model[1]"].normalized_label == "second widget method"
    assert d["method_or_model[1]"].source_chunk_index == 1
    assert d["main_results[0]"].source_chunk_index == 1
    assert "main_results" not in d  # no not_found row when a later part has elements
    assert sum(1 for x in norm.drafts if x.field_key == "research_question") == 1


def test_single_part_has_no_chunk_index():
    d = _by_key(_drafts(_note()))
    assert d["method_or_model[0]"].source_chunk_index is None
