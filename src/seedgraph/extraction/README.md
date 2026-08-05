# The Reader: what Seedgraph extracts from a paper

This is the step where Seedgraph reads a paper and turns it into structured notes.
It is deliberately **not** a "summarize this paper" step. It is a fixed set of
questions, answered with cited evidence, with an explicit record of what the paper
does *not* say.

If you only read one thing: the reader is built on the premise that *"summarize
this paper"* is a bad instruction. An open-ended summary optimizes for sounding
like a summary — it compresses lossily, quietly launders inference as fact, and
gives you no handle on what it dropped or invented. This step replaces that with a
protocol: **answer these questions, quote your evidence, and declare when you
can't.**

---

## 1. The questions it answers

Every paper is read against the same eleven-field protocol (schema id
`default_research_note_v1`). This is the reading checklist a careful referee or
PhD student applies to any empirical paper:

| Field | What it captures |
| --- | --- |
| `research_question` | The paper's central question |
| `main_contribution` | The headline contribution |
| `method_or_model` | Each method / model / design used (carries a `method_type`) |
| `data_sources` | Each dataset or data source |
| `setting` | The empirical or theoretical setting / context |
| `estimand_or_target_object` | The target object of inference |
| `assumptions` | Each assumption, typed `identification` / `regularity` / `general` |
| `main_results` | Each principal result (carries a `result_type`) |
| `limitations` | Each stated limitation (carries a `limitation_type`) |
| `robustness_checks` | Each robustness / sensitivity check |
| `open_questions` | Each open question the paper raises |

The field list alone is not the point — you could paste it into a chat prompt.
What makes the output trustworthy for research is the discipline attached to every
answer, below.

---

## 2. The guarantees on every answer

Each answer carries a per-claim provenance envelope. These are the properties a
one-off chat summary cannot give you:

- **Verbatim evidence or nothing.** Every substantive `found` claim must include
  an `exact_quote`: a character-for-character substring copied from the paper. If
  the model cannot find a supporting quote, it is required to mark the claim
  `ambiguous` instead of asserting it. This is the structural brake on
  hallucination — a claim without a real quote does not get to be `found`.

- **Absence is recorded, never dropped.** A field the paper does not address is
  returned with status `not_found` / `not_applicable` / `ambiguous`, not silently
  omitted. For a literature review, *"this paper states no identification
  assumption"* is often more valuable than any positive claim. A summary can never
  tell you what wasn't there.

- **Stated vs. inferred is tracked.** Each claim carries `assertion_status`:
  `stated` when the author says it explicitly, `inferred` when the reader concludes
  it from context. An `inferred` claim must include an `inferred_explanation`. You
  always know whether you are reading the paper or reading between its lines.

- **Confidence is reported.** Every claim carries a `confidence` in `[0, 1]`.

- **Labels are canonical across papers.** `normalized_label` carries the *standard
  community name* of a concept — `difference-in-differences`, not "our two-way DiD
  design"; `Masked Language Modeling`, not "the MLM objective." The same concept
  gets the same label across papers, which is what lets the graph layer merge
  concepts by construction. The paper's own wording is preserved faithfully in
  `claim_text` and `exact_quote`; only the label is normalized.

### The status vocabulary

```
found            the paper contains the field, with a verbatim quote
not_found        the paper does not address it
ambiguous        the text is unclear, or no verbatim quote could be anchored
not_applicable   the field cannot apply to this paper
extraction_failed the extractor could not produce a valid answer
```

### The per-claim envelope

| Key | Meaning |
| --- | --- |
| `claim_text` | The claim, in the paper's own language |
| `normalized_label` | Standard community canonical name of the concept |
| `status` | One of the five status values above |
| `assertion_status` | `stated` / `inferred` |
| `inferred_explanation` | Required when `assertion_status = inferred` |
| `epistemic_type` | Origin tier — `llm_extracted` or `llm_inferred` for reader notes |
| `confidence` | Model-reported confidence in `[0, 1]` |
| `exact_quote` | Verbatim supporting substring (required for `found` claims) |

### A worked example

A slice of the note for an illustrative difference-in-differences study. The
quotes are schematic — they show the *shape* of an answer, not excerpts from a
specific paper. Note three things: the `found` result is anchored to a verbatim
quote; the identification assumption is `inferred` (the authors never name it) and
so carries an explanation; and `estimand_or_target_object` is recorded as
`not_found` rather than dropped.

```json
{
  "method_or_model": [
    {
      "claim_text": "Compares employment in the treated state against a neighboring control state, before and after a minimum-wage increase.",
      "normalized_label": "difference-in-differences",
      "method_type": "empirical_design",
      "status": "found",
      "assertion_status": "stated",
      "confidence": 0.95,
      "exact_quote": "We estimate the employment effect using a difference-in-differences comparison across the state border."
    }
  ],
  "assumptions": [
    {
      "claim_text": "Absent the policy, employment in the two states would have moved in parallel.",
      "normalized_label": "parallel trends",
      "assumption_type": "identification",
      "status": "found",
      "assertion_status": "inferred",
      "inferred_explanation": "The paper never uses the term 'parallel trends', but the DiD estimate is only valid under it, and the authors defend the neighboring-state control on exactly those grounds.",
      "confidence": 0.71,
      "exact_quote": "The neighboring state provides a comparison whose labor market tracks the treated state's."
    }
  ],
  "estimand_or_target_object": {
    "status": "not_found",
    "assertion_status": null
  }
}
```

A "summarize this paper" prompt would have written a fluent paragraph that stated
the parallel-trends assumption as if the authors had, and would simply never have
mentioned that the estimand was left implicit. This step makes both facts legible.

---

## 3. Why a schema beats "summarize this paper"

| | "Summarize this paper" | Seedgraph reader |
| --- | --- | --- |
| Task the model optimizes | Produce fluent-sounding prose | Answer fixed questions, cite evidence |
| Hallucination | Unconstrained | A claim needs a verbatim quote or it isn't `found` |
| What's missing | Invisible | Recorded as `not_found` / `not_applicable` |
| Fact vs. inference | Blurred | Separated (`stated` vs. `inferred`) |
| Comparability across papers | None | Same eleven fields, canonical labels |
| Reproducibility | None | Versioned + content-addressed (see §5) |

The difference is the difference between a vibe and a measurement.

---

## 4. This is an opinionated default — extend it with a lens

The eleven fields are **field-agnostic in implementation** (the code carries no
economics-specific vocabulary), but the *choice* of fields — estimand,
identification assumptions, robustness checks — is a distinctly empirical-research
reading protocol. That is a feature: it is what a careful empirical reading looks
like, not a view from nowhere.

When your project needs a different or finer protocol — a theory paper's lemmas, a
lab paper's outcome measures, a specific sub-field's vocabulary — you attach a
**lens** rather than editing this schema. Lenses layer project-specific extraction
on top of the field-agnostic default.

---

## 5. Reproducibility and provenance

Every extraction run is stamped so a note can be reproduced and invalidated:

- `schema_id` = `default_research_note_v1`, plus `SCHEMA_VERSION` and
  `PROMPT_VERSION` (both currently `1.1.0`).
- model, provider, temperature, and the **hash of the source markdown**.

Bumping `SCHEMA_VERSION` or `PROMPT_VERSION` deliberately invalidates and re-rolls
the cached note for every paper, so a corpus never silently mixes protocol
versions. Most "summarize my paper" tools cannot tell you which protocol produced a
given note; here the note is content-addressed to the exact schema, prompt, model,
and source text that produced it.

### A note on access and sharing

Structured notes derived from full text are treated as **local, private**
artifacts by default (see [`CONTENT_ACCESS_POLICY.md`](../../../CONTENT_ACCESS_POLICY.md)).
Only `open_access` / `metadata_only` material passes the export gate. Seedgraph is
a local-first research harness for papers you already have lawful access to — not a
full-text repository or a paywall bypass.

---

## 6. Where this lives in the code

| Concern | File |
| --- | --- |
| Field schema (Pydantic) + version constants | [`schema.py`](schema.py) |
| Reader prompt (rules, status vocab, quote requirement) | [`prompt.py`](prompt.py) |
| Controlled vocabularies (status, assertion, epistemic type) | [`../vocab.py`](../vocab.py) |

> Maintainer note: the field table in §1 mirrors `_SCHEMA_FIELDS` in `prompt.py`
> and `DefaultNoteV1` in `schema.py`. If you change the schema, update this table
> in the same commit (or wire a parity test) so the public description can't drift
> from the code.
