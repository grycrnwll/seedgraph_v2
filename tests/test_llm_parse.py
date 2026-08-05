r"""Shared LLM JSON parser (``llm/parse.py``; Build C chunk 1, D1).

One robust brace-depth parser, four adopters. These tests pin the behaviors the
consolidation was bought for: fence stripping, prose tolerance, string-literal
brace safety, and first-balanced-object-wins — strictly better than the greedy
``\{.*\}`` regexes it replaced, which glued two objects into one unparseable
match (and metadata's find/rfind bracketing, which did the same).
"""

from __future__ import annotations

import json

from seedgraph.llm.parse import extract_json_object, parse_json_object

# --------------------------------------------------------------------------
# extract_json_object / parse_json_object core behaviors
# --------------------------------------------------------------------------


def test_fenced_json():
    obj = {"title": "T", "n": 1, "nested": {"k": [1, 2]}}
    raw = "```json\n" + json.dumps(obj) + "\n```"
    assert extract_json_object(raw) == json.dumps(obj)
    assert parse_json_object(raw) == obj


def test_prose_preamble_and_trailing_tokens():
    raw = 'Sure, here is the note:\n{"a": 1}\nHope that helps!'
    assert extract_json_object(raw) == '{"a": 1}'
    assert parse_json_object(raw) == {"a": 1}


def test_braces_inside_string_literals():
    raw = 'Model says: {"quote": "an { open and a } close", "n": 2} trailing'
    assert parse_json_object(raw) == {"quote": "an { open and a } close", "n": 2}


def test_escaped_quotes_inside_string_literals():
    raw = 'x {"k": "he said \\"hi\\" {"} y'
    assert parse_json_object(raw) == {"k": 'he said "hi" {'}


def test_two_objects_first_balanced_wins():
    # The greedy ``\{.*\}`` regexes matched across BOTH objects and failed to
    # decode (None); the brace-depth scan recovers the first object.
    raw = '{"first": 1} and then {"second": 2}'
    assert extract_json_object(raw) == '{"first": 1}'
    assert parse_json_object(raw) == {"first": 1}


def test_non_dict_top_level():
    assert parse_json_object("[1, 2, 3]") is None  # array, no object anywhere
    assert parse_json_object("42") is None
    assert parse_json_object('"just a string"') is None


def test_unbalanced_or_garbage_object():
    assert extract_json_object('{"never": "closed"') is None
    assert parse_json_object('{"never": "closed"') is None
    assert parse_json_object("{not json}") is None  # balanced but undecodable


def test_empty_and_none_ish_input():
    assert extract_json_object("") is None
    assert extract_json_object(None) is None  # type: ignore[arg-type]
    assert parse_json_object("") is None
    assert parse_json_object(None) is None  # type: ignore[arg-type]
    assert parse_json_object("   \n  ") is None
    assert parse_json_object("no braces at all") is None


# --------------------------------------------------------------------------
# the four adopters share the one implementation (own return shapes kept)
# --------------------------------------------------------------------------


def test_validator_reexport_is_the_shared_impl():
    from seedgraph.extraction import validator
    from seedgraph.llm import parse

    assert validator.extract_json_object is parse.extract_json_object


def test_adopters_first_object_wins_with_own_shapes():
    # Two concatenated objects: the pre-consolidation greedy regexes (compose,
    # llm_propose) and find/rfind bracketing (metadata) all returned None here.
    two = '{"synonym_groups": [["a", "b"]]} {"junk": true}'

    from seedgraph.answer.compose import _parse_json
    from seedgraph.extraction.metadata import parse_biblio_json
    from seedgraph.semantic.llm_propose import _parse_groups

    assert _parse_json(two) == {"synonym_groups": [["a", "b"]]}
    assert parse_biblio_json(two) == {"synonym_groups": [["a", "b"]]}
    assert _parse_groups(two) == ({"synonym_groups": [["a", "b"]]}, None)
    # llm_propose keeps its (parsed, error) executor-parse shape on failure.
    assert _parse_groups("no json here") == (None, "invalid json")
