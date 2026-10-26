"""The task-specific ``instructions.md`` text of a notes batch (design §1.2, §4.4).

The work skill carries the generic claim -> read -> submit loop, so this text holds
only what is specific to the notes task at schema 2.0.0: the role, the rules, a
compact field table (instead of a JSON Schema), the fixed vocabularies, the quote
rules, the labels rule and one worked example. Pydantic stays the validator.

The text is deterministic (no dates, no ids), so a batch's ``instructions_sha256`` is
the same for every batch of the same version. :func:`render_instructions` returns the
text; the protocol (P4) writes the file.
"""

from __future__ import annotations

import json

from .format import PROMPT_VERSION, SCHEMA_ID, SCHEMA_VERSION
from .vocab import VOCABULARIES

#: Version of this instructions text. Bump on any wording change (the text is
#: byte-stable for a given version). Independent of PROMPT_VERSION, which is stamped on
#: saved notes. 2.1.0: reading tips, submit names, formula quote tip, errata rule,
#: assumption_type guidance.
INSTRUCTIONS_VERSION = "2.1.0"

#: One line per ``assumption_type`` value (tested to cover the vocabulary).
_ASSUMPTION_GUIDANCE: dict[str, str] = {
    "identification": "what makes the target quantity recoverable from the data: parallel trends, "
                      "unconfoundedness, exclusion restrictions, overlap, no anticipation, monotonicity",
    "regularity": "technical conditions for estimation and inference: sparsity, smoothness, moment conditions, "
                  "continuity or differentiability, mixing or dependence limits, rate conditions",
    "general": "anything else the results rely on: modelling, behavioural or functional-form assumptions "
               "that are neither of the above",
}
#: Invented worked example. It passes ``validate_note`` + ``check_note`` with no
#: errors or warnings (tested).
WORKED_EXAMPLE: dict = {
    "research_question": {
        "claim_text": "Does the invented widget subsidy raise widget output?",
        "normalized_label": "effect of widget subsidy on output",
        "status": "found", "assertion_status": "stated", "confidence": 0.9,
        "exact_quote": "We ask whether the widget subsidy raises output.",
    },
    "main_contribution": {
        "claim_text": "A new two-step estimator for subsidy effects with staggered timing.",
        "normalized_label": "two-step staggered subsidy estimator",
        "status": "found", "assertion_status": "stated", "confidence": 0.8,
        "exact_quote": "Our main contribution is a two-step estimator",
    },
    "method_or_model": [{
        "claim_text": "Difference-in-differences with staggered adoption, estimated in two steps.",
        "normalized_label": "difference-in-differences",
        "method_type": "research_design",
        "plain_words": "Compare firms that got the subsidy with firms that did not, before and after.",
        "status": "found", "assertion_status": "stated", "confidence": 0.9,
        "exact_quote": "we compare subsidised and unsubsidised firms before and after adoption",
    }],
    "data_sources": [{
        "claim_text": "An invented census of widget firms, 2001-2010.",
        "normalized_label": "widget firm census",
        "status": "found", "assertion_status": "stated", "confidence": 0.9,
        "exact_quote": "the census of widget firms covers 2001 to 2010",
    }],
    "setting": None,
    "estimand_or_target_object": {
        "claim_text": "The average effect of the subsidy on treated firms.",
        "normalized_label": "average treatment effect on the treated",
        "plain_words": "How much output changed, on average, for the firms that got the subsidy.",
        "status": "found", "assertion_status": "stated", "confidence": 0.8,
        "exact_quote": "the average effect on subsidised firms",
    },
    "assumptions": [{
        "claim_text": "Without the subsidy, treated and control firms would have followed parallel output paths.",
        "normalized_label": "parallel trends",
        "assumption_type": "identification",
        "plain_words": "Both groups of firms would have grown alike if nobody got the subsidy.",
        "status": "found", "assertion_status": "stated", "confidence": 0.9,
        "exact_quote": "output would have evolved in parallel absent the subsidy",
    }],
    "main_results": [{
        "claim_text": "The subsidy raises output by about 4 percent.",
        "normalized_label": "subsidy raises output",
        "result_type": "empirical",
        "status": "found", "assertion_status": "stated", "confidence": 0.8,
        "exact_quote": "output rises by about 4 percent",
    }],
    "limitations": [{
        "claim_text": "The authors suggest the effect may not carry over to larger firms.",
        "normalized_label": "limited to small firms",
        "limitation_type": "external_validity",
        "status": "found", "assertion_status": "inferred",
        "inferred_explanation": "The sample has only small firms; the paper does not say so as a limitation.",
        "confidence": 0.5,
        "exact_quote": "all firms in the census have fewer than 50 employees",
    }],
    "robustness_checks": [],
    "open_questions": [],
}

_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("research_question", "claim or null", "the question the paper asks"),
    ("main_contribution", "claim or null", "what the paper adds"),
    ("setting", "claim or null", "where and when (population, market, period)"),
    ("estimand_or_target_object", "claim or null; needs `plain_words`",
     "the quantity or object the paper targets"),
    ("method_or_model", "list; each needs `method_type`, `plain_words`",
     "methods, estimators, designs and models used or proposed"),
    ("data_sources", "list", "named data sets or data-generating set-ups"),
    ("assumptions", "list; each needs `assumption_type`, `plain_words`",
     "assumptions the results rely on"),
    ("main_results", "list; each has `result_type`", "the main findings"),
    ("limitations", "list; each has `limitation_type`", "stated or clear limitations"),
    ("robustness_checks", "list", "robustness and sensitivity checks"),
    ("open_questions", "list", "questions the paper leaves open"),
)

_CLAIM_KEYS: tuple[tuple[str, str], ...] = (
    ("claim_text", "the claim in your words, one or two sentences"),
    ("normalized_label", "a short standard name (see the labels rule)"),
    ("status", "`found`, `not_found`, `ambiguous` or `not_applicable`"),
    ("assertion_status", "`stated` (the paper says it) or `inferred` (you concluded it)"),
    ("inferred_explanation", "required when `assertion_status` is `inferred`: how you inferred it"),
    ("confidence", "a number from 0 to 1"),
    ("exact_quote", "required for `found`: a verbatim passage from the paper (quote rules)"),
    ("plain_words", "assumptions, methods/models and the estimand only, required when `found`: "
                    "one or two sentences that state it without the paper's notation"),
)


def _vocab_lines() -> list[str]:
    return [f"- `{field}`: " + ", ".join(f"`{v}`" for v in values)
            for field, values in VOCABULARIES.items()]


def render_instructions(*, has_parts: bool = False) -> str:
    """The ``instructions.md`` text. ``has_parts``: the batch contains split papers."""
    lines: list[str] = [
        f"# Notes task: `{SCHEMA_ID}` (schema {SCHEMA_VERSION}, prompt {PROMPT_VERSION}, "
        f"instructions {INSTRUCTIONS_VERSION})",
        "",
        "You write one structured research note for the paper text in your package file. "
        "Write the note as one JSON object, to the package's `result_path`, and submit that file "
        "(MCP: `work_submit` with `note_path`; CLI: `work submit` with `--note-file`).",
        "",
        "## Reading the package",
        "",
        "- The package text file can be large. Read it in ranges, about 300 lines at a time, "
        "until you reach the end. The package gives its `lines` (line count).",
        "- `est_tokens` is approximate (formula-heavy text often takes more tokens).",
        "",
        "## Rules",
        "",
        "- Use only the package text. Do not use outside knowledge of the paper.",
        "- If the text includes an erratum or corrigendum, use the corrected values and say in "
        "the claim's `claim_text` that they come from the correction.",
        "- Scalars may be null, arrays may be empty. Do not invent content to fill a field.",
        "- One array element per distinct method, data source, assumption, result, "
        "limitation, robustness check or open question.",
        "- Type fields take one value from the fixed lists below. If none fits, use `other` "
        "(`general` for assumptions); an unknown value is saved as that and you get a warning.",
        "- Labels rule: `labels.json` (beside this file) lists labels already used in the "
        "library, per group. If one of these labels names the same concept, use it exactly; "
        "otherwise use the standard community name.",
        "- Write no other keys. Seedgraph fills provenance itself.",
        "",
        "## Quote rules",
        "",
        "- Copy `exact_quote` character for character from the package text: same letters, "
        "symbols, accents, dashes, quote marks, spaces and line breaks.",
        "- Never repair characters, even when the text looks garbled. Copy what is there.",
        "- Make each quote long enough to be unique in the paper (one sentence or clause is "
        "usually enough).",
        "- In formula-heavy text, prefer a plain-prose passage that states the claim. If the "
        "quote must contain math, copy inline math exactly, including the spaces around it "
        "(the text often has two spaces before and after `$...$`).",
        "- A quote that cannot be found comes back with the exact passage to copy. If no "
        "passage supports the claim, set `status` to `ambiguous`.",
        "- The text marks removed reference lists with `[references omitted]`; never quote "
        "that marker.",
        "",
        "## Fields",
        "",
        "| field | shape | what |",
        "|---|---|---|",
        *[f"| `{name}` | {shape} | {what} |" for name, shape, what in _FIELDS],
        "",
        "Every claim (a scalar, or one array element) is an object with these keys:",
        "",
        "| key | meaning |",
        "|---|---|",
        *[f"| `{key}` | {meaning} |" for key, meaning in _CLAIM_KEYS],
        "",
        "## Fixed vocabularies",
        "",
        *_vocab_lines(),
        "",
        "Choosing `assumption_type`:",
        "",
        *[f"- `assumption_type` = `{value}`: {_ASSUMPTION_GUIDANCE[value]}."
          for value in VOCABULARIES["assumption_type"]],
        "",
    ]
    if has_parts:
        lines += [
            "## Long papers in parts",
            "",
            "A long paper is split into parts (`part` k of n in the package). Part 1 holds the "
            "abstract and introduction and owns the four scalars: `research_question`, "
            "`main_contribution`, `setting` and `estimand_or_target_object`. In parts 2..n, "
            "leave those four null (a scalar sent there is dropped) and do not restate the "
            "paper's research question as a result. Every part fills the arrays with what its "
            "own text contains, and sends an empty array when it has none.",
            "",
        ]
    lines += [
        "## Worked example (invented paper)",
        "",
        "```json",
        json.dumps(WORKED_EXAMPLE, ensure_ascii=False, indent=1),
        "```",
        "",
    ]
    return "\n".join(lines)
