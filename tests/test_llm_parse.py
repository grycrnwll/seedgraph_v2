r"""Shared LLM JSON parser (``llm/parse.py``; Build C chunk 1, D1).

One robust brace-depth parser; since stage 2 its adopters are the work routine and
the note format (the v2 adopters were deleted). These tests pin the behaviors the
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
# the protocol adopters share the one implementation
# --------------------------------------------------------------------------


def test_protocol_adopters_share_the_one_implementation():
    from seedgraph.llm import parse
    from seedgraph.notes import format as note_format
    from seedgraph.work import routine

    assert routine.extract_json_object is parse.extract_json_object
    assert note_format.extract_json_object is parse.extract_json_object
