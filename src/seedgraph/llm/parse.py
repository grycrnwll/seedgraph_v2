r"""Shared tolerant JSON-object parsing for LLM output (Build C chunk 1, D1).

One robust parser, four adopters — ``extraction/validator.validate_note``,
``extraction/metadata.parse_biblio_json``, ``answer/compose._parse_json``,
``semantic/llm_propose._parse_groups`` — consolidating the four independently
duplicated fence-strip parsers (the v1 duplication lesson).

:func:`extract_json_object` is the string-aware brace-depth scanner (moved
verbatim from ``extraction/validator.py``): it returns the FIRST balanced
``{...}`` substring, tolerating prose preamble, ```json fences, trailing
tokens, and braces inside string literals — strictly better than the greedy
``\{.*\}`` regexes it replaces (which glued two objects into one unparseable
match). :func:`parse_json_object` adds ``json.loads`` + a top-level dict
check. Callers keep their own post-parse semantics and return shapes.
"""

from __future__ import annotations

import json

__all__ = ["extract_json_object", "parse_json_object"]


def extract_json_object(raw_text: str) -> str | None:
    """Return the first balanced ``{...}`` JSON substring of ``raw_text``, or None.

    Brace-depth scan that ignores braces inside string literals — handles a model
    that wraps JSON in prose or a ```json fence.
    """
    if not raw_text:
        return None
    start = raw_text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw_text)):
        ch = raw_text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw_text[start : i + 1]
    return None


def parse_json_object(raw_text: str) -> dict | None:
    """Extract + ``json.loads`` the first balanced JSON object, or ``None``.

    Returns the parsed top-level ``dict``, or ``None`` when no balanced object
    exists, the candidate fails to decode, or the decoded value is not an
    object. Callers needing granular error strings (e.g. the validator's repair
    prompt) use :func:`extract_json_object` and decode themselves.
    """
    candidate = extract_json_object(raw_text)
    if candidate is None:
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
