"""The ``instructions.md`` of the concept tasks (stage 4 design §3.3-§3.4, §4.2, §6.3).

Written once per batch. Every example is invented (textbook terms or nonsense words).
Kept as Python text, not a resource file, so no package-data entry is needed.
"""

from __future__ import annotations

#: Bumped whenever the instructions change what a worker should answer. It is the concept
#: tasks' version (``tasks_concepts.TASK_VERSION``); batches created at an earlier listed
#: version still load (``tasks_concepts.ACCEPTED_VERSIONS``).
INSTRUCTIONS_VERSION = "1.1.0"

_VERSION_LINE = f"Instructions version {INSTRUCTIONS_VERSION}.\n\n"

_COMMON = """\
## How to answer

Read this file once per batch. Each package lists numbered items. Each side of an item shows its family,
its label, its paper count and one to four example claims. Every example has a short claim ref (`c17`), the
claim text, its plain words when present, a quote from the paper, and the paper's short title and year.
A package may repeat a claim under different refs (the same claim shown on two items); cite whichever ref
the item lists.

- Answer **every item exactly once**, by its number.
- Cite claims **by ref only**. Never write quotes: the refs point at quotes that are already anchored.
- Judge **meaning**, from the claims. A shared word is not evidence. Labels marked "ambiguous label" are
  used in several senses across the literature; for them a merge or a link must cite a ref from **each**
  side (`quote_a` and `quote_b`).
- When unsure between merge and link, answer **link**.
- When unsure between link and unrelated, answer **unrelated**.
- `unsure` is an honest answer: the item goes to the researcher's queue.
- Researcher decisions always win; you never see pairs the researcher has decided.
- Write the result file as one JSON object. If your client exposes token usage, write it to a JSON file and
  pass `--usage-file` (CLI) or `usage` (MCP) when you submit; otherwise skip it.

## Answer schema

```json
{"items": [
  {"item": 3, "verdict": "merge | link | split | unrelated | same_sense | unsure",
   "confidence": "high | medium | low",
   "name": "<one of the item's labels; merge only>",
   "note": "<= 200 chars; link only: the difference, or the history for drift>",
   "quote_a": "c17", "quote_b": "c42",
   "senses": [{"gloss": "<= 120 chars", "claims": ["c17", "c18"]}, {"gloss": "...", "claims": ["c19"]}],
   "why": "<= 200 chars, optional"}
]}
```

## Verdicts

| verdict | meaning | needs |
|---|---|---|
| `merge` | true synonyms in the **same sense** (same family only) | `name`: one of the item's labels |
| `link` | closely related: near-equivalents with a difference that matters, or drift | `note` (the difference or the history), `quote_a` (a ref of side A), `quote_b` (a ref of side B) |
| `split` | one label used in different senses | `senses`: at least two, each with a gloss; every listed claim of the label in exactly one sense |
| `unrelated` | not near-equivalents: no connection, or merely related topics (see below) | nothing; stored so the pair is never judged again |
| `same_sense` | sense checks only: every listed claim uses the label in one sense | nothing |
| `unsure` | you cannot tell from the claims | nothing; `why` helps the researcher |

A pair takes `merge`, `link`, `split` (only when both sides share one label), `unrelated` or `unsure`. A sense
check takes `same_sense`, `split` or `unsure`. A merge that breaks a guardrail (different families, a
never-merge pair, a version or size word that differs) comes back with the reason: answer link, unrelated or
unsure instead; the refusal names the word, the families or the pair that triggered it.

## When is it a link?

A link is for **near-equivalents whose difference matters**: near-synonyms that drifted apart in use (serial
correlation and autoregression), an assumption and its weaker or stronger variant, two names for nearly the
same estimand. Most pairs that share a topic are **not** links.

The test: if swapping one label for the other in a typical sentence from either paper would be nearly
right, but a careful reader would object to something specific, it is a link (the `note` says what they
would object to). If the two are merely on related topics, are used together, or one is an input to the
other, answer `unrelated`.

## Worked examples (invented)

- **merge**: A "propensity score" (c1: *the probability of treatment given covariates*) and B "treatment
  probability score" (c2: *estimated chance of treatment given the covariates*):
  `{"item": 1, "verdict": "merge", "confidence": "high", "name": "propensity score"}`
- **link (difference)**: A "ATE" (c3) and B "ATT" (c4) measure the effect over different populations:
  `{"item": 2, "verdict": "link", "confidence": "high", "note": "ATT averages over the treated only; ATE over everyone", "quote_a": "c3", "quote_b": "c4"}`
- **link (drift)**: A "serial correlation" (c5: *residuals correlate over time*) and B "autoregression"
  (c6: *the series depends on its own lags*): a property of residuals and a model that converged in use:
  `{"item": 3, "verdict": "link", "confidence": "medium", "note": "history: residual property, later modelled as an autoregressive process", "quote_a": "c5", "quote_b": "c6"}`
- **split**: the label "identification" with c7 (*a source of exogenous variation for a causal reading*) and
  c8 (*the system of equations has a unique solution*):
  `{"item": 4, "verdict": "split", "confidence": "high", "senses": [{"gloss": "causal: source of exogenous variation", "claims": ["c7"]}, {"gloss": "econometric theory: unique solution of the system", "claims": ["c8"]}]}`
- **same_sense**: every listed claim of "sprocket balance" means the same weighting property:
  `{"item": 5, "verdict": "same_sense", "confidence": "high"}`
- **unrelated**: A "gadget overlap" (c9) and B "widget lag order" (c10):
  `{"item": 6, "verdict": "unrelated", "confidence": "high"}`
- **unrelated (topical)**: A "widget lag order" (c11: *how many lags of the widget the model keeps*) and B
  "widget unit root" (c12: *the widget series is not stationary*): same topic, often tested together, but
  swapping one for the other is simply wrong:
  `{"item": 7, "verdict": "unrelated", "confidence": "high"}`
- **unrelated (topical)**: A "sprocket instrument" (c13: *sprocket prices shift gadget supply*) and B "first
  stage strength" (c14: *how well the instrument predicts gadget supply*): one is an input to the other:
  `{"item": 8, "verdict": "unrelated", "confidence": "high"}`
- **unrelated (topical)**: A "gadget clustering" (c15: *errors correlate within gadget plants*) and B "plant
  fixed effects" (c16: *a dummy for each gadget plant*): used together in one design, not the same idea:
  `{"item": 9, "verdict": "unrelated", "confidence": "medium"}`
- **unsure**: `{"item": 10, "verdict": "unsure", "confidence": "low", "why": "the claims are too short to tell"}`
"""

MERGE = """\
# Tidy labels (concepts:merge)

Items are pairs of labels that look alike on the page: spelling variants, word order, acronyms, extra words.
They are grouped into lexical groups; judge each pair on its own claims. Most pairs are either the same idea
(merge) or a near-equivalent with a difference (link); some are unrelated despite the shared words.

""" + _VERSION_LINE + _COMMON

BRIDGE = """\
# Find bridges (concepts:bridge)

Items are **sense checks** (one label, several example claims from different papers: is it one sense, or
several?) and **pairs** of senses whose claims read alike, often from papers far apart in the citation graph.
Pairs across families (an assumption and a method) can only be linked, never merged.

""" + _VERSION_LINE + _COMMON

NAME_LITERATURES = """\
# Name literatures (concepts:name_literatures)

Each item is one literature of a context graph: a group of papers that cite each other. It lists its most
cited papers (title and year) and its most distinctive concepts. Give each literature a short name a
researcher would recognise, and a one-line description. Names are labelled "agent inferred"; a researcher's
own name always wins.

""" + _VERSION_LINE + """\
- Answer every item exactly once.
- `name`: at most 6 words. `description`: one line, at most 200 characters.

```json
{"items": [{"item": 1, "name": "Invented panel widget estimators", "description": "Papers on estimating widget effects from panels of gadgets."}]}
```
"""

BY_TASK = {"merge": MERGE, "bridge": BRIDGE, "name_literatures": NAME_LITERATURES}
