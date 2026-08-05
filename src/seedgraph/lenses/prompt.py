"""Render the strict-JSON lens extraction prompt from a `LensDefinition` (plan §5/§8).

The prompt embeds the lens `object_type`, positive/negative anchors, the
`output_schema` as the required JSON shape, and the `evidence_policy`, and
demands: verbatim quotes for every ``found`` record, and explicit
``not_found`` records carrying ``searched_sections[]`` (docs/06 §6). The routed
profile must satisfy the source-text-leaves-machine policy for the work's
`access_class` (the runner gates routing; this module only builds text).

`LENS_PROMPT_VERSION` is stamped onto each `extraction_runs.prompt_version`
(plan §7) so a prompt change participates in provenance.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import LensDefinition

LENS_PROMPT_VERSION = "lens_extraction_v1"

_SYSTEM = (
    "You are a precise, citation-faithful extraction engine. You return ONLY a "
    "single JSON object and never invent text that is not present verbatim in the "
    "supplied document."
)


def _schema_shape(lens: "LensDefinition") -> dict:
    """A human-readable JSON description of the lens output_schema fields."""
    shape: dict[str, str] = {}
    for key, spec in lens.output_schema.items():
        descr = spec.field_type
        if spec.allowed_values:
            descr += " one of " + json.dumps(spec.allowed_values)
        if spec.required:
            descr += " (required)"
        shape[key] = descr
    return shape


def build_lens_prompt(
    lens: "LensDefinition",
    *,
    markdown_text: str,
    sections: list[str] | None = None,
) -> tuple[str, str]:
    """Build the strict-JSON extraction instruction for ``lens`` over one work's
    markdown; return ``(system_prompt, user_prompt)``.

    Renders anchors + `output_schema` (as a required JSON shape) + the
    `evidence_policy` (verbatim-quote requirement, not-found contract). When
    ``sections`` is given they bound/aid the not-found ``searched_sections[]``
    reporting. The runner routes the result through the `project_lens_extraction`
    task (doc 13 §5/§6).
    """
    section_hint = ""
    if sections:
        section_hint = (
            "\nThe document is organized into these sections (use them for "
            "`section` and `searched_sections`):\n- " + "\n- ".join(sections) + "\n"
        )

    policy = lens.evidence_policy
    user = f"""Extract every instance of the object_type `{lens.object_type}` from the document below.

POSITIVE anchors (terms that signal a relevant instance):
- {chr(10).join('- ' + a for a in lens.positive_anchors).lstrip('- ')}

NEGATIVE anchors (signals to EXCLUDE — do NOT extract these):
- {chr(10).join('- ' + a for a in lens.negative_anchors).lstrip('- ') or '(none)'}

For each found instance produce one record with EXACTLY these fields:
{json.dumps(_schema_shape(lens), indent=2)}

Rules:
- Every found record MUST carry a verbatim quote copied character-for-character
  from the document (the `*_verbatim` / quote field). Do not paraphrase.
- `stated_or_inferred` is `stated` when the document asserts it explicitly and
  `inferred` when you deduce it. {'An `inferred` record MUST carry a non-empty `notes` explanation.' if policy.inferred_requires_explanation else ''}
- Leave `evidence_span_ids` as an empty array; the system fills it after anchoring.
{section_hint}
Return a SINGLE JSON object of this exact shape:
{{
  "records": [ {{ ...one object per found instance... }} ],
  "not_found": {{ "searched_sections": [..section names searched..], "notes": "...", "confidence": 0.0 }}
}}
If there are NO instances, return an empty `records` array and a populated
`not_found` object (record_not_found={str(policy.record_not_found).lower()}).

DOCUMENT:
{markdown_text}
"""
    return _SYSTEM, user
