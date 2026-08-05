"""Default-note prompt builder (plan §5/§8, step 5).

Builds the system + user prompt for ``default_research_note_v1``: injects the
schema field list, the closed status vocabulary, the stated-vs-inferred rule
(``assertion_status``), and the hard requirement that every substantive ``found``
claim carry a VERBATIM ``exact_quote`` copied from the markdown (the phase_3
``ensure_span`` anchor — plan §8). Output is constrained to a single JSON object
so the validator can bracket-extract and Pydantic-validate it.
"""

from __future__ import annotations

from ..vocab import AssertionStatus, StatusValue
from .schema import PROMPT_VERSION, SCHEMA_ID

# Field-agnostic schema field map injected into the user prompt (plan §4.0). Names
# match DefaultNoteV1; the model returns one JSON object keyed by these fields.
_SCHEMA_FIELDS: list[tuple[str, str]] = [
    ("research_question", "scalar object — the paper's central research question"),
    ("main_contribution", "scalar object — the headline contribution"),
    ("method_or_model", "array — each item has a `method_type` (e.g. regression, model, design)"),
    ("data_sources", "array — each item names a dataset / data source"),
    ("setting", "scalar object — the empirical or theoretical setting / context"),
    ("estimand_or_target_object", "scalar object — the target object of inference"),
    ("assumptions", "array — each item has an `assumption_type` (identification, regularity, general)"),
    ("main_results", "array — each item has a `result_type`"),
    ("limitations", "array — each item has a `limitation_type`"),
    ("robustness_checks", "array — each item is a robustness / sensitivity check"),
    ("open_questions", "array — each item is an open question the paper raises"),
]

_STATUS_VOCAB = ", ".join(s.value for s in StatusValue)
_ASSERTION_VOCAB = ", ".join(a.value for a in AssertionStatus)

_SYSTEM_PROMPT = (
    "You are a meticulous research-note extractor. Read the paper markdown and emit "
    "EXACTLY ONE JSON object matching the default research-note schema. Output JSON "
    "ONLY — no prose, no markdown fences, no commentary.\n"
    "\n"
    "Rules:\n"
    f"- Every field/element carries `status` (one of: {_STATUS_VOCAB}). Use `not_found` "
    "when the paper does not contain the field, `not_applicable` when the field cannot "
    "apply, `ambiguous` when the text is unclear.\n"
    f"- Every field/element carries `assertion_status` (one of: {_ASSERTION_VOCAB}). "
    "Use `stated` when the author explicitly states it; use `inferred` when you infer it "
    "from context. An `inferred` claim MUST include a non-empty `inferred_explanation`.\n"
    "- For every substantive `found` claim, include `exact_quote`: a VERBATIM substring "
    "copied character-for-character from the markdown that supports the claim. Do NOT "
    "paraphrase the quote. If you cannot find a verbatim supporting quote, set status to "
    "`ambiguous` instead of `found`.\n"
    "- `confidence` is a number in [0, 1].\n"
    "- `normalized_label` MUST be the STANDARD community canonical name of the "
    "concept — NOT this paper's idiosyncratic phrasing. Use the standard name so the "
    "SAME concept gets the SAME label across different papers. Examples: write "
    "`Masked Language Modeling` even if the paper calls it \"the MLM objective\"; "
    "write `difference-in-differences` even if the paper says \"our two-way DiD "
    "design\"; write `Monte Carlo simulation` even if the paper writes \"our MC "
    "experiments\". This rule applies ONLY to `normalized_label` — `claim_text` and "
    "`exact_quote` keep the paper's own language faithfully.\n"
    "- Be field-agnostic: do not invent economics-specific structure; just fill the "
    "generic research-note fields.\n"
)


def build_prompt(markdown_text: str, *, schema_id: str = SCHEMA_ID) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for the default note extraction.

    The system prompt fixes the role, the JSON-only output contract, the status
    vocabulary, the stated/inferred rule, and the verbatim-quote requirement; the
    user prompt carries the schema field descriptions plus ``markdown_text``.
    Field-agnostic (no econ-specific instructions — doc 05 §1).
    """
    field_lines = "\n".join(f"- {name}: {desc}" for name, desc in _SCHEMA_FIELDS)
    user_prompt = (
        f"Schema id: {schema_id}\n"
        "Return one JSON object with these top-level fields (omit none — use a "
        "`not_found` status object / empty array when absent):\n"
        f"{field_lines}\n"
        "\n"
        "Each claim object may carry: claim_text, normalized_label, status, "
        "assertion_status, inferred_explanation, confidence, exact_quote. "
        "`normalized_label` must be the standard community canonical name "
        "(same concept, same label across papers).\n"
        "\n"
        "=== PAPER MARKDOWN START ===\n"
        f"{markdown_text}\n"
        "=== PAPER MARKDOWN END ===\n"
    )
    return _SYSTEM_PROMPT, user_prompt


def build_repair_prompt(prior_response: str, errors: list[str]) -> str:
    """Build the single repair re-prompt (plan §8 / step 6).

    Echoes the schema contract and the concrete validation ``errors`` from the
    first attempt and asks the model to return corrected JSON only. Used at most
    once per run; a second failure is terminal (``extraction_failed``).
    """
    error_lines = "\n".join(f"- {e}" for e in errors) or "- the output was not valid JSON"
    return (
        "Your previous response could not be parsed/validated. Return ONE corrected "
        "JSON object ONLY (no prose, no fences) for the default research-note schema.\n"
        "\n"
        "Validation errors:\n"
        f"{error_lines}\n"
        "\n"
        "Previous response (for reference):\n"
        f"{prior_response}\n"
    )


def prompt_fingerprint() -> str:
    """Return the deterministic prompt identity used for staleness (== PROMPT_VERSION).

    Trivial passthrough kept so the runner has one call site for the prompt
    invalidation key; a future template-hash scheme can replace the body without
    touching callers.
    """
    return PROMPT_VERSION
