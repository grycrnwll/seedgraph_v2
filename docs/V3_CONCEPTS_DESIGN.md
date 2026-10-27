# v3 Stage 4 design: concepts and graphs

Status: design for build. No code yet. This document covers v3 stage 4:
- library concepts built from claims, with **senses**: one label can carry several meanings;
- merges, closely related links and sense splits, with their guardrails;
- the agent concept tasks on the one work protocol;
- meaning-based search and the embeddings adoption test;
- literatures and bridges per context graph;
- the concept and graph screens, and guided corpus growth;
- MCP and skill updates, deletions, the acceptance plan, build slices and open questions.

It uses the glossary terms from `CONTEXT.md` (library, context graph, concept, merge, closely related link,
bridge, literature, batch, worker, researcher assertion) and the deep-module vocabulary from `codebase-design`
(module, interface, depth, seam, adapter). `docs/V3_LIBRARY_DESIGN.md` (stage 1),
`docs/V3_AGENT_PROTOCOL_DESIGN.md` (stage 2) and `docs/V3_UI_DESIGN.md` (stage 3) are binding for storage, the
work protocol and the web stack.

The repo is public. Real labels, counts, test cases, cost figures and review results live in the v3 home, never
in this document or the repo. Sizes and costs below are formulas with named parameters. Examples use textbook
terms only (ATE, serial correlation, exogeneity).

## 0. Decided before this stage

From ticket 11 / ADR-0013 (binding), and from researcher input given during this design:
- Concepts are library bridges. Every specific concept is kept. There is no theme tree.
- **Polysemy is real.** One label can carry several senses. "Identification" can mean the source of exogenous
  variation behind a causal reading (causal inference), or the unique solution of a system of equations
  (econometric theory). A concept keyed by label alone would create false bridges. This is the ticket 11 risk
  ("library-wide decisions may lose nuance") now realised. **Concept identity is label + sense.**
- **Drift is real.** Near-synonyms can converge over time, e.g. serial correlation (a property of residuals) and
  autoregression (a model). Such pairs get a closely related link whose note records the history. They are not
  merged.
- Guardrails are hard vetoes: the type fence, the never-merge list, the ambiguous-label rule, closure (the merged
  name is a label used in the corpus) and the stoplist of version and size words. Researcher decisions override
  agent decisions.

## 1. Concept model in the library

### 1.1 Layers

Each layer is derived from the one below, plus stored decisions:

1. **Claims**: `current_claims` (stage 3). This is the researcher's current interpretation: overlay edits
   included, hidden claims excluded. Researcher-added claims count too.
2. **Labels**: `(family, label_key)`. Here `label_key = semantic.merge.normalize(normalized_label)`.
3. **Senses**: `(family, label_key, sense_id)`. Every label starts with one sense, `s0`, which has no gloss. A
   split replaces `s0` with `s1..sn`, each with a one-line gloss.
4. **Concepts**: groups of senses joined by merges, within one family. A concept's name is one of its member
   labels (closure).
5. **Links**: closely related links between two concepts, each with a note and a quote from each side.

**Type families, not claim types.** The type fence applies to a *family*. A fence on `claim_type` would split
one idea across the three assumption claim types, and across method and model, which the migrated data shows
happening. The families:

| family | claim types | bridge order |
|---|---|---|
| `assumption` | `identification_assumption`, `general_assumption`, `regularity_condition` | 1 |
| `estimand` | `estimand` | 2 |
| `method` | `method`, `model` | 3 |
| `result` | `result` | 4 |
| `data` | `data` | 5 |
| `limitation`, `robustness_check`, `open_question` | one family each | on request |
| `paper_scalar` | `research_question`, `main_contribution`, `setting` | never bridged; shown on paper pages only |

The claim type stays visible on every member claim. The family table is a constant in `concepts/families.py`,
and old free-text subtypes map through the stage 2 variant table.

### 1.2 Schema: `db/schema/library/0010_concepts.sql`

Stored, authoritative rows (decisions and inputs):

| table | key columns | notes |
|---|---|---|
| `concept_decisions` | **rebuilt** (SQLite cannot ALTER a CHECK): `decision_id` (hash of kind + subjects), `kind` (`merge`, `link`, `never_merge`, `never_connect`, `unrelated`), `family` NULL (NULL = every family where both labels exist), `label_a, sense_a, label_b, sense_b` (sense NULL = the default sense; pair ordered `(label, sense)` a < b), `decided_by` (`researcher`, `agent`, `rule`), `state` (`applied`, `pending_apply`, `queued`, `undone`, `vetoed`, `needs_rejudge`), `note` (difference or history note), `quote_a_json`, `quote_b_json` (`{paper_id, span_id, claim_fingerprint}`), `confidence`, `veto_reason`, `batch_id`, `packet_id`, `model`, `origin`, `decided_at` | carried rows are copied across in the same file, unchanged in meaning |
| `concept_senses` | PK(`family, label_key, sense_id`); `gloss`, `decided_by`, `state` (`default`, `confirmed`, `queued`), `decision_ref`, `origin`, `decided_at` | rows exist only for split or sense-checked labels. A label with no row has the implicit `s0`, unconfirmed |
| `sense_assignments` | PK(`family, label_key, paper_id, claim_fingerprint`); `sense_id`, `decided_by`, `decided_at` | which sense a claim belongs to under a split label. Keyed by the stage 3 claim fingerprint (`notes/identity.py`), **never** by `claim_id`, which changes on every save. Fingerprints are matched by `identity.match` (tiered, not equality): refresh re-matches old assignments onto the paper's new claims, and an unmatched claim becomes "needs sense". Link quotes are re-matched the same way; `span_id` is a hint, not the key |
| `ambiguous_labels` | PK(`pattern`); `match` (`label`, `head`), `note`, `decided_by`, `origin`, `created_at` | §2.3 |
| `concept_vectors` | PK(`text_sha256, model`); `dims`, `vector` (float32 LE blob), `created_at` | §5. Only if embeddings are adopted; the table always exists |
| `concept_reviews` | PK(`sample_id, decision_id`); `mark` (`right`, `wrong`, `should_be_link`, `should_be_split`), `note`, `marked_at` | §11 |
| `concept_build_state` | PK(`paper_id`); `claims_sha256`, `built_at` | what each paper's claims looked like at the last refresh (§1.5) |

Derived rows, rebuilt deterministically by `concepts.refresh` and never edited by hand:

| table | columns |
|---|---|
| `claim_concepts` | `claim_id, paper_id, family, label_key, sense_id` (NULL = needs sense), `concept_id`, `supports` (0/1), `agent_visible` (0/1) |
| `concepts` | `concept_id` ('cn_' + hash of family + smallest member sense key), `family`, `name`, `gloss`, `paper_count`, `claim_count`, `quote_count`, `supported` (0/1), `weight` (`log(N_papers / paper_count)`), `needs_sense_check` (0/1) |
| `concept_members` | `concept_id, family, label_key, sense_id` |
| `concept_links` | `concept_a, concept_b, decision_id, note, decided_by` |

**Handles.** `concept_id` changes when a merge or split reshapes a group. The stable public handle is a member
sense key, written `family/label_key[~sense_id]`. It always resolves to the current concept, so URLs, MCP
arguments and stage 5 traversal records stay valid. A label that was later split resolves to a "this label has
n senses" chooser.

### 1.3 Grounding

- A claim **supports** its sense when it is live: trust `paper_states` (an anchored agent quote), or a
  researcher claim of kind `quote` or `assertion`. Hidden claims are already absent from `current_claims`.
  `agent_inferred` claims are listed, labelled "agent inferred, no quote", but support nothing.
- A concept is shown while at least one supporting claim exists (`supported = 1`). An unsupported concept leaves
  every view, search and bridge count. Its decisions are kept, so it comes back intact when support returns.
- **Agent audience**: `agent_visible` drops claims on papers with an open `hides_notes` attention item and claims
  of labelled note versions (stage 1 §2). Agent-facing views count only agent-visible support.
- Counts are shown everywhere: papers, claims and anchored quotes. Every gloss, link note and literature name
  written by an agent carries the "agent inferred" label. Researcher text carries "researcher".

### 1.4 Rule merges (deterministic, decided_by `rule`)

These are applied by `refresh` and shown in "recent merges", where each can be split:
- **Generic suffix fold**: within a family, `X assumption` / `X condition` / `X restriction` / `X hypothesis`
  folds onto `X`, but only when `X` is itself a label used in that family (closure).
- **Acronym fold**: `merge.is_acronym_expansion` within a family.

Rule merges never apply to ambiguous labels (§2.3). The v2 token-overlap fold is **not** a rule merge; lexical
overlap only generates candidates (§3.2).

### 1.5 Refresh

- **Deterministic refresh** (`concepts.refresh(lib)`) recomputes every derived table from `current_claims`, the
  decisions, senses and assignments. It is a pure function of the stored state, so a rebuild is byte-stable. Its
  cost scales with the number of claims (seconds), so there is no partial derived update to get wrong.
- **Out of date** = papers whose `current_claims` fingerprint differs from `concept_build_state`, plus
  `concept_refresh` rows (stage 3 overlay saves, merges, reassignments). Notes batches do not have to remember to
  mark anything. Home shows "concepts out of date (n papers changed)" with a **Refresh** button. Every concept
  task and concept view refreshes first when the state is stale. Refresh clears `concept_refresh`.
- **New papers** feed AI work incrementally. Candidate search and judging (§3, §4) consider only senses that have
  a new supporting paper, or that are new since the last pass of that task. Senses already judged against each
  other are never judged again.
- **Full rebuild on request** (`concepts rebuild --rejudge`): re-runs candidate search over every sense. It marks
  agent decisions `needs_rejudge` only when the researcher ticks "re-judge agent decisions". Researcher and rule
  decisions are never re-judged.
- **Splits invalidate.** When a label is split, every decision that names its default sense is set to
  `needs_rejudge`. The next pass re-judges it at sense level. Researcher decisions are offered for re-assignment
  in the queue rather than re-judged.

## 2. Merges, links, splits and guardrails

### 2.1 What each connection means

| connection | meaning | effect |
|---|---|---|
| **merge** | true synonyms in the **same sense** | the senses join one concept |
| **closely related link** | near-equivalents with a difference that matters, or drift | both concepts stay; the link carries a one-line note (difference or history) and a quote from each side |
| **split** | one label, several senses | `s0` is replaced by `s1..sn` with glosses; each supporting claim is assigned to one sense |
| **unrelated** | no connection | stored, so the pair is never judged again |
| **never_merge** | researcher or seeded list: may be linked, never merged | veto |
| **never_connect** | recorded by "split" on a merge or link: neither merged nor linked again | veto |

When unsure between merge and link, link. Links may cross families (drift often crosses from an assumption to a
model). Merges never cross families.

### 2.2 Guardrails (hard vetoes, checked on every agent proposal at save)

1. **Type fence**: a merge whose senses are in different families is refused.
2. **Never-merge list**: no merge across a `never_merge` pair, and no merge or link across a `never_connect`
   pair. This is checked **transitively**: a merge that would put a never-merge pair into one concept through
   other members is refused. The seed rows are `decided_by = 'researcher'`, `origin = 'seed'`: ATE ≠ ATT,
   DiD ≠ DDD, heterogeneity ≠ heteroskedasticity.
3. **Ambiguous-label rule** (§2.3).
4. **Closure**: the concept name must be one of the member labels.
5. **Stoplist**: v2's `canon.DISCRIMINATING_TOKENS` (versions, sizes, numerals, static/dynamic). A merge whose
   members differ in one of these tokens is refused. The tokens are compared over the whole resulting group.

**Removed as a veto**: v2's token-overlap precondition (Jaccard ≥ 0.6 or an acronym match). It would block
every merge across notation, which is the whole point of bridging. It survives only as a candidate generator.

A vetoed merge is never silently turned into something else. The package gets `needs_revision` with the reason,
and the worker answers link, unrelated or unsure (§3.4). A veto that only appears at save time (a concurrent
package changed the groups) becomes a queue item instead, because the worker cannot fix it.

### 2.3 Ambiguous labels

`ambiguous_labels` is a seeded, editable list, maintained like the never-merge list on a settings sub-page.
- **Patterns**:
  - `label`: an exact `label_key`;
  - `head`: a head word that matches every label whose content tokens are that word plus generic words
    (assumption, condition, limitation, analysis, test, bias, ...). For example the head `exogeneity` matches
    "exogeneity", "exogeneity assumption" and "exogeneity limitation", but not "strict exogeneity".
- **Seed** (generic econometrics vocabulary, editable): identification, efficiency, consistency, structural,
  robustness, heterogeneity, endogeneity, exogeneity, independence, selection, stationarity, identification
  limitation. The real-library check that motivated these stays in the v3 home.
- **Effects**:
  - no rule merges;
  - no merge, link or bridge on the label alone: an agent must compare example claims from both sides, chosen
    from different papers and literatures;
  - an unsure or low-confidence result on an ambiguous label always goes to the researcher's queue;
  - until a sense check confirms a single sense or splits the label, its default sense is
    `needs_sense_check = 1`. It then counts for nothing across literatures (§6.3).

**Cross-literature identity needs evidence, for every label.** Within one literature a label is taken as
single-sense by default; a narrow corpus is usually univocal. A sense whose supporting papers fall in two or more
literatures of some graph, or that sit at citation distance ≥ 3 (§4.3), is shown as "same words, sense not
checked" until a sense check confirms it. Sense checks are cheap: one package line per label. They are part of
the bridge pass (§4).

### 2.4 Applying, undoing, the researcher's queue

- **Applied automatically**: an agent proposal that passes every guardrail and is not "unclear" is stored with
  `state = 'applied'` and appears in **Recent merges** (merges, links and splits, newest first, filterable by
  batch). Its actions:
  - **Split** a merge: `undone` + `never_connect`;
  - **Remove** a link: `undone` + `never_connect`;
  - **Undo a split**: the senses collapse back to `s0`, the assignments are kept for history, and agent
    decisions naming `s1..sn` become `needs_rejudge`.
- **Genuinely unclear** goes to the queue (`state = 'queued'`). A proposal is unclear when any of these holds:
  1. the agent answered `unsure`, or gave `confidence = low`;
  2. it touches an ambiguous label and is not `high` confidence;
  3. **reach**: it is a merge where both sides have at least `REACH_MIN` papers (default 5), because a wrong
     merge between two established concepts is costly. A link or split is never held back for reach;
  4. **disagreement**: it contradicts an existing agent decision on the same subjects;
  5. a save-time veto (§2.2).

  The target is a queue of about 5% of proposals or less. The constants live in `concepts/decide.py` and the
  acceptance report states the observed share. The queue page shows both sides' claims with quotes and offers
  merge, link (with a note), split, unrelated and skip. Ordering is by reach.
- **Researcher decisions always win.** They are stored as `decided_by = 'researcher'`. An agent proposal that
  contradicts one is dropped at save, with feedback, and never queued. The researcher can merge, link with their
  own note (labelled theirs), split a label into senses (assigning claims with checkboxes), or split any agent
  merge or link.
- **Migrated decisions** (stage 1 §6.10) are applied by the first refresh, with family NULL and default senses:
  - user must_link → researcher `merge`;
  - user cannot_link → researcher `never_merge`;
  - the bulk agent merges → agent `merge`, `state = 'applied'`, shown in Recent merges with origin
    "migrated from v2".

  A migrated merge that a guardrail now vetoes is set to `vetoed` and listed in the refresh report, not
  applied. That covers a type fence or stoplist conflict, or an ambiguous label. Researcher rows are never
  vetoed; a researcher merge across families applies in each family where both labels exist.

## 3. The agent concept tasks on the one protocol

### 3.1 The seam, and the change it needs

One adapter module, `work/tasks_concepts.py`, registers `register_prefix("concepts", ConceptTask)`. Its
arguments are `merge`, `bridge` and `name_literatures`. They share one decision schema, one `check` and one
`save`.

Stage 2 said `concepts:*` would "need no protocol change". That holds for claims, expiry, pauses, submit and
feedback. It does not hold for selection and display, which assume papers today. The minimal seam change,
owned by slice C3:

| place | today | change |
|---|---|---|
| `routine._select`, `tasks.parse_query` | paper ids or a paper query | an optional `TaskType.select(lib, options) -> list[str]`; when a task defines it, `create_batch` calls it and paper arguments are refused |
| `routine._package_message`, batch views | `papers.title/year` of `subject_id` | an optional `TaskType.subject_label(lib, subject_id) -> str`; default is the paper lookup |
| `MAX_PAPERS` | 1000 papers | `MAX_SUBJECTS`, same value; the message says "subjects" |
| `work/estimate.py` | per-paper text tokens | an optional `TaskType.estimate(lib, options) -> Estimate`; concept tasks use their dry-run (§4.5) |

`subject_kind = 'concept_package'`. Eligibility is always `queued`, with no markdown binding; the packet id
already takes `markdown_hash` NULL. The paper-only staleness check in `routine` is already guarded by
`subject_kind == 'paper'`. The batch page shows the subject label ("assumption pairs 41–90").

### 3.2 Candidate generation (no AI cost)

Run by `concepts/candidates.py` at batch creation, then frozen into the packages:
- **Lexical groups** (`concepts:merge`): within a family, labels joined by token Jaccard ≥ `TAU_LEX` (0.5) on
  content tokens (generic words removed), or by acronym expansion. This is v2's `canon.candidate_clusters`. Big
  groups are cut to at most 12 labels, keeping the strongest edges.
- **Meaning neighbours** (`concepts:bridge`): for each sense, its `K` nearest senses (default 5) above a
  similarity floor. The search is in the same family, plus the adjacent family pairs `assumption↔method` and
  `estimand↔method`; across families only links are allowed. Search is by embeddings if adopted (§5), else by
  FTS5 over the sense's claims (§5.4).
- **Sense checks** (`concepts:bridge`): every ambiguous label with support in at least 2 papers, and every sense
  whose papers span literatures or citation distance ≥ 3 (§2.3) and that has not been checked.
- **Within-label separation** (embeddings only): a label whose claim vectors form two clusters with low
  cross-similarity is queued for a sense check even if it is not on the ambiguous list.
- **Dropped before packaging**: pairs inside one paper only, pairs with any stored decision (any kind, either
  maker), never-connect pairs, and unsupported senses.

### 3.3 Packages

A package is the stage 2 message plus a markdown file (`work/<batch_id>/<packet_id>.md`). Items are numbered.
Each side of an item shows:
- family and label;
- one to three example claims, from different papers and different literatures where possible. Each example
  has the claim text, its plain words if present, an anchored quote (at most 240 characters, centred on the
  span), and the paper's short title, year and literature name;
- the paper count.

Every example carries a short claim ref (`c17`). Instructions and the answer schema live once in
`instructions.md`. Package size: `PAIRS_PER_PACKAGE` (default 50) pairs, or 25 sense checks, or 12 lexical
groups.

**Text used when plain words are missing.** Most migrated notes lack plain words (stage 2 §4.3). Each example
therefore always shows label + claim text + quote, with plain words added when present. The quote is the
paper's own words and carries the notation. The same text is what meaning search embeds (§5.2). Re-noting for
plain words is an open question (§13 Q2), not a prerequisite.

### 3.4 Answer schema and validation

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

`check` (pure):
- every item answered exactly once;
- the verdict is allowed for the item kind. A sense-check item takes `same_sense` or `split`, plus `unsure`. A
  pair takes the five verdicts; `split` on a pair means "these are different senses of one label", allowed only
  when both sides share a label;
- refs belong to the item and to the right side;
- a link has a note and one quote per side; a split covers every listed claim exactly once, with at least two
  senses and a gloss each;
- the guardrails (§2.2), evaluated against the library state at check time.

Failures return `needs_revision` with a per-item list, for example "item 3: merge vetoed: different families
(assumption vs method); answer link, unrelated or unsure". The package claim renews, as in stage 2.

**No anchor preview.** Concept tasks cite claims and never write quotes, so `check` declares no `QuoteRef`.
`save` stores each cited quote as `{paper_id, span_id, claim_fingerprint}`, so a link's quotes survive a
re-note. If a cited span is no longer live at save, that item is refused with feedback.

### 3.5 Save

In the protocol's save transaction:
1. write `concept_decisions` / `concept_senses` / `sense_assignments` with `decided_by = 'agent'`, the batch,
   packet and model;
2. classify each item (applied / queued, §2.4);
3. run `concepts.refresh` for the touched labels' concepts (cheap; the full refresh stays the definition);
4. append a batch event with counts (applied, queued, vetoed, unrelated).

### 3.6 `concepts:merge` vs `concepts:bridge`

`merge` is the cheap first pass: lexical groups, mostly spelling and wording variants. It runs before `bridge`,
so bridge candidates are computed on deduplicated concepts. `bridge` judges meaning-level pairs and sense
checks. Both appear in Settings → Concepts as "Tidy labels" and "Find bridges", each with the dry-run estimate
(§4.5).

## 4. Bridge finding (`concepts:bridge`)

### 4.1 Pass order

- **Families** in the table order (§1.1): assumptions, then estimands, then methods; results and data in a
  second pass; the rest only on request. Single-paper senses are included.
- **Within a family**: sense checks first, because they change the subjects. Then pairs by **descending
  citation distance** (`∞`, then 3, then 2, then 1), with descending similarity as the tie-break.
- A batch carries one family-pass, so pausing between passes is natural.

### 4.2 Verdicts

Same sense → merge. Closely related → link with a difference or history note and a quote from each side.
Different senses of one label → split. Unrelated → stored. Unsure → queue. A drift pair (serial correlation ↔
autoregression type) is a link whose note gives the history. The instructions include an invented worked
example of each verdict.

### 4.3 Citation distance

This is computed on the **library** citation graph, undirected, over `citation_links`, with cited-only papers
as nodes. Two papers that cite a common reference are therefore at distance 2. The distance between two senses
is the minimum over their supporting paper pairs: a pair is "near" if any of its papers are near. BFS runs from
the smaller side, capped at depth 3, with `∞` meaning "> 3".

### 4.4 Cost model

The cost model is computed by `concepts/estimate.py`. Its terms:
- `P` pairs (+ `S` sense checks, + `G` lexical groups);
- `t_side` ≈ (label + claim text + plain words + quote + paper line) / 4 tokens per example, and `e` examples
  per side;
- `t_item = 2·e·t_side + t_answer`;
- `t_pkg = n_items·t_item + t_instr_amortised`;
- agent tokens ≈ `ρ · Σ t_pkg`, where `ρ` is the agent-overhead factor measured in stage 2 (agent tokens ÷
  package text tokens), shown as a range.

The ticket 11 estimate (about 10k pairs, about 200 packages, 2–4M tokens) implies an all-in 200–400 tokens per
pair. That is reachable only with `e = 1` for ordinary senses, quotes capped at 240 characters, and compact
answers. So the defaults are: `e = 1`, rising to `e = 2..3` only for ambiguous labels, sense checks and senses
with at least `REACH_MIN` papers; and `K = 5`. The dry run reports where the estimate lands for the current
library. If it exceeds the researcher's budget (§13 Q3), lower `K`, the families included or the citation
distance floor, never the evidence per item.

### 4.5 Dry run and budget

`concepts estimate --task bridge [--family ...] [--k 5] [--budget-tokens N]` (CLI and the Settings → Concepts
panel) builds the candidate list without writing anything. It reports:
- pairs, sense checks and packages per family;
- the token range;
- the share of pairs at each citation distance.

With a budget, it lists the cut-off: "the first n pairs by priority fit". `create_batch` takes the same
`budget_tokens` and freezes only the pairs that fit; the rest wait for the next pass. Actual usage is taken from
`package_submissions.usage_json` when the worker reports it. The work skill asks the agent to pass `usage`. When
usage is missing, the estimate stands in, labelled as an estimate.

## 5. Embeddings (tentative) and search

### 5.1 Model

Ollama `nomic-embed-text` (v1.5, 137M parameters, 768 dimensions, about 274 MB download). Its context window is
2k tokens, ample for one claim context. It needs task prefixes: `search_document:` when indexing and
`search_query:` when searching. The alternative is `mxbai-embed-large` (335M, 1024 dimensions, about 670 MB,
512-token context). The adoption test (§5.5) runs nomic first; mxbai is tried only if nomic fails narrowly.
Nothing is downloaded without the researcher's go-ahead (§13 Q4). Ollama stays a direct client
(`llm/ollama.embed(model, texts)`), not a batch worker (stage 2 §6).

### 5.2 What is embedded

**Claim contexts, not bare labels.** Each supported claim is embedded as
`"<family>: <label>. <claim text>. <plain words>"`. Sense vectors are the normalised mean of their supporting
claims' vectors, and concept vectors the mean over their senses.
- Senses of one homonymous label separate, because their claim texts differ.
- Drifted near-synonyms land close, because their contexts are alike.

Bare-label vectors are never stored.

### 5.3 Storage and search

- `concept_vectors` is keyed by `sha256(embed text)` + model, never by `claim_id`. Re-notes that keep the text
  reuse the vector, and a changed text re-embeds only that claim. Embedding is a JobRunner job after each
  refresh ("embedding n new claims").
- **Search uses pinned `numpy`** (a new app dependency; a binary wheel, no BLAS threading concerns at this size).
  A brute-force cosine over a matrix of sense vectors per family is loaded once per process and invalidated by
  the refresh fingerprint. It holds `N_claims × 768 × 4` bytes of claim vectors plus the sense matrix. Pure
  Python is too slow for candidate generation, and a vector extension (sqlite-vec) adds a native SQLite
  extension load for no gain at this size. numpy is added only if embeddings are adopted.

### 5.4 Keyword search (always present; the fallback)

FTS5 `claim_fts` (label, claim text and plain words; stage 2 §1.7). The concept query is the sense's label plus
the top distinctive terms of its claims (bm25 over the sense's own claims). Hits are aggregated to senses by
best score. The same interface serves both modes:

```python
concepts.search.find(lib, query: str | SenseKey, *, family=None, graph=None, k=20,
                     mode: 'auto' | 'meaning' | 'keyword' = 'auto') -> list[SenseHit]  # hit.mode says which ran
```

`auto` uses meaning when embeddings are adopted and Ollama is reachable, else keyword. The fallback is silent
except for a "keyword search" marker on results.

### 5.5 The adoption test (two-sided)

- **Test cases** are written by the researcher (from a proposed list) into
  `<home>/evaluations/concept_search_cases.yaml` **before** any run. They are never tuned on, and a change after
  the first run is versioned and reported. Each case names labels plus claim refs (paper id + fingerprint):
  - **partner cases**: sense A ↔ sense B that should be found, synonyms or links (drift pairs included);
  - **homonym probes**: two claim sets that share a label or a head word but differ in sense (one label used
    two ways, or two labels sharing a word such as "identification"), plus a same-sense partner for set 1
    (another label with the same meaning).
- The eval runner ranks the cases' **own claim sets** (a set's vector is the mean of its claims; its keyword
  query is built as in §5.4 from its claims), not stored senses. At test time the labels are not split yet.
- The runner is pure Python: a few dozen queries against the cases' sets plus every sense of the families
  involved is fast enough without numpy, so the test needs no dependency that adoption would decide.
- **Metrics**, both modes on the same frozen library state:
  - (a) **partner recall@10**: the share of partner cases where B is among A's top 10, plus MRR;
  - (b) **separation**: the share of homonym probes where set 1's partner ranks above set 2 in set 1's
    neighbour list. Keyword search scores about zero on (b) by construction (same words), so (b)
    carries much of the "clearly beats".
- **Pass rule** (all three):
  1. separation ≥ 0.8;
  2. recall@10 ≥ keyword's, and embeddings find at least `max(3, 25% of cases)` partners that keyword misses;
  3. no over-merge signal: no homonym probe has set 2 as set 1's top-1 neighbour.
- If the rule fails, keyword search stays, embeddings are not adopted, numpy is not added and the
  within-label separation trigger (§3.2) is off. The ambiguous list and the agent's sense checks still guard
  polysemy.
- The report (`<home>/acceptance/stage4/search_test.json` + `.md`) gives both modes side by side, per case.

## 6. Graph slice, literatures and bridges

### 6.1 Graph slice

`concepts.slice(lib, graph) -> Slice` gives the senses and concepts present in the graph's member papers, with:
- per-slice paper, claim and quote counts;
- a slice weight `log(N_members / slice_papers)`;
- each concept's literatures.

Concepts are summed from senses: a split label shows as separate concepts in the slice, and in a narrow graph
often only one sense is present. Nothing is stored per graph.

### 6.2 Literatures (`library/literatures.py`, `db/schema/graph/0002_literatures.sql`)

- **Input**: the in-graph citation view (members only, undirected, deduplicated).
- **Method**: seeded Louvain (`networkx.community.louvain_communities`, seed 0, nodes relabelled in sorted id
  order: v2's determinism recipe from `graph/analyze.py`). Leiden is not in networkx, and its packages would add
  native dependencies for little gain at these sizes.
- **Resolution ladder** `γ ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}`. Choose the lowest `γ` that satisfies the size
  rule:
  - the largest literature holds ≤ 30% of placed papers;
  - at least 3 literatures have ≥ `MIN_LIT` (5) papers.

  If none does, take the `γ` with the best modularity. The researcher can step "fewer / more literatures" (one
  rung), and the choice is stored.
- **Unplaced**: members with no in-graph citation, and communities smaller than `MIN_LIT`, form "unplaced". It
  is shown as a list, not a literature, and does not count toward bridges.
- **Stability**: re-run with seeds 1–9 at the chosen `γ` and report the mean adjusted Rand index against seed 0
  (and NMI) on the graph page and in the acceptance report. A low score (< 0.6) shows a "groupings are
  unstable" note. Random-pair agreement is not used: obviously separate pairs inflate it.
- **Tables**:
  - `literature_runs` (`run_id, algorithm, resolution, seed, modularity, ari_mean, citation_sha256, built_at`);
  - `literatures` (`literature_id, run_id, ordinal, size, agent_name, agent_description, researcher_name,
    named_at`);
  - `literature_members` (`literature_id, paper_id`).
- **Rebuild** happens when the in-graph citation fingerprint changes and the researcher clicks "regroup". New
  literatures are matched to the old ones by member Jaccard ≥ 0.5, which carries names and renames forward.
  Unmatched literatures get "unnamed (n papers)" until named.

### 6.3 Naming (`concepts:name_literatures`) and bridges

- **Naming**: one package per graph regroup. Each literature shows its 10 most-cited-within papers (title and
  year) and its 8 most distinctive confirmed concepts (slice tf-idf). The agent returns `{literature_id, name
  (≤ 6 words), description (≤ 1 line)}`, labelled agent inferred. A researcher rename overrides and survives
  regrouping.
- **Bridge** (computed, never stored): a concept with support in at least 2 literatures of the graph, counting
  only senses that are confirmed or single-literature (§2.3); or a closely related link whose two concepts
  are in different literatures. Unchecked same-words concepts are listed apart, as "same words, sense not
  checked", with a "check senses" action that creates a small bridge batch.
- **Overview** "Literatures and bridges": one card per literature (name, size, top concepts), then bridges
  ranked by the number of literatures spanned, then by mean citation distance.
- **Hub list**: the most common supported concepts per family in the graph (v2's sheet), each with paper count,
  literature spread and sense gloss.

## 7. Getting around

- **Concept search** (`/concepts?q=`): a meaning or keyword search (§5.4) with a mode marker, filterable by
  family and graph. Results are sense-level, showing the gloss, counts and the best example claim.
- **Concept page** (`/concepts/{family}/{label_key}[?sense=]`):
  - name, gloss, family, counts, trust labels;
  - **members**: labels and senses, with their decision origin (rule, agent, researcher, migrated);
  - **supporting claims** with quotes (each opens the paper page at the highlighted passage), paged and grouped
    by paper. Claims needing a sense are shown under "unsorted";
  - **merges and links**, with notes, quotes from both sides and who decided them;
  - **often appears with**, computed on demand: concepts co-supported in the same papers, ranked by lift, top
    10, never stored;
  - papers, and literatures per graph;
  - actions: merge with…, link to…, split into senses, split / remove (on agent decisions).
- **Suggestions outside the graph** (on a concept page opened from a graph): "n papers elsewhere in the library
  share this sense" (or a linked concept), listing title, year and graphs, with **Add to graph**. Held papers
  join as members, cited-only ones as frontier. Nothing is added automatically. The same list feeds guided
  growth (§8.2).

## 8. Context graph screens

### 8.1 List and detail

The list keeps stage 3's counts and adds literatures and bridges. The detail page gets four tabs:
- **Papers**: stage 3's membership and seeds table, plus a literature column;
- **Concepts**: the slice, the hub list and the bridges overview;
- **Graph**: §8.3;
- **Grow**: §8.2.

The stage 3 right-hand column becomes a "next steps" panel: concepts out of date, unnamed literatures, open
growth reviews, queue items touching this graph.

### 8.2 Guided corpus growth (`library/growth.py`)

Acquisition beyond seeds is a convenience. It is never automatic, and researcher-supplied PDFs are always
welcome.
1. **Seeds**: pick seeds (default: the graph's seed flags).
2. **Depth and cap**: depth 1–2, at most `cap` new papers per generation. The pre-run estimate is v2's
   `acquisition/preview.preview_acquisition` (no network).
3. **Walk** (a JobRunner job): the outbound references of seeds and members. Two sources are used:
   - library `reference_entries` resolved to papers; reference lists not yet parsed are parsed with the kept
     `citation/bib_parser` and resolved by `citation/resolve` through the provider chain;
   - provider `referenced_works` through the cached chain.

   Found papers are ranked by how many graph members cite them (v2 walk ranking) and become **cited-only**
   library papers (identity rules from stage 1). New citation links go to `citation_links`. Mid-band resolutions
   become citation-review attention items, the producer stage 3 open question 11 waited for.
4. **Review found papers**: a table with title, year, "cited by n members", OA available (yes / no / unknown)
   and "already held". It also lists the **outside-graph suggestions** from concepts (held library papers that
   share confirmed senses or links, §7). Each row gets a tick box.
5. **Approve**:
   - ticked held papers join as members;
   - ticked cited-only papers join the frontier, and those with an OA copy are fetched (`acquisition/fetch`, OA
     only, never paywalled) and converted;
   - the rest appear in the stage 3 **needs a PDF** list, where the researcher can drop their own PDF.

   Approving never starts a notes batch; the Home checklist offers it.

### 8.3 The 2D graph

**Choice: server-rendered SVG.** The layout is computed in Python with `networkx.spring_layout` (seeded, fixed
iterations), so it is deterministic and testable. Nodes and edges are drawn as SVG elements with presentation
attributes and CSS classes. Inline `style` attributes and scripts are not used, so the stage 3 CSP holds
unchanged. Nodes are `<a>` links with a `<title>` tooltip.
- **Concept view** (default): the top `N` (40) supported concepts by slice weight × papers. An edge is drawn
  when two concepts are co-supported in ≥ 2 papers, computed on demand; closely related links are drawn dashed.
  Nodes are coloured by the literature where they are most supported, and bridges get a ring.
- **Citation view**: member papers coloured by literature, unplaced papers in grey, in-graph citations as
  edges. Above `MAX_SVG_NODES` (800) only placed papers are drawn, with a count of the rest.
- The toggle is an htmx swap of the SVG partial.

Rejected: vendored cytoscape.js or sigma. They would bring a pinned JS library, a hash entry and a client
layout, which gives nondeterministic pictures and more to test, for interactivity the researcher has not asked
for. That decision can be revisited if zoom and pan become a need.

### 8.4 Lens choice

The graph detail gains a "Lenses" panel: the library lenses (`lenses/registry`) with a choose / unchoose toggle
(`library.graphs.choose_lens`, plus a new `unchoose_lens`), and a "run lens on members missing it" link that
pre-selects a `lens:<id>` batch.

## 9. MCP and skill updates

New tools in `mcp/tools_concepts.py`. They are thin adapters over `concepts.*`, and each reply states
`search_mode` and labels agent-inferred text:

| tool | arguments | returns |
|---|---|---|
| `concept_search` | `query, family?, graph?, limit=20` | sense hits: handle, name, gloss, counts, best example claim with quote, mode |
| `concept_show` | `handle, graph?, claims_offset=0, claims_limit=20` | members and senses, supporting claims with quotes and trust, merges and links with notes and quotes, papers, literatures, often-appears-with |
| `literatures_overview` | `graph` | literatures (name, agent or researcher, size, top concepts), bridges, unchecked same-words concepts, stability |
| `concept_hubs` | `graph?, family?, limit=30` | the hub list |

The hub list is also exposed as the MCP resource `seedgraph://graphs/{slug}/hubs`. The consultation skill's
"Find your footing" gains:
- start with `literatures_overview` and the hubs;
- search concepts before spans when the question is about an idea;
- never treat a concept marked "sense not checked" as evidence of a cross-literature connection;
- say "agent inferred" for glosses, link notes and literature names.

The **work skill** (`seedgraph_v3_work_skill.md`) gains a short "Concept packages" section:
- read `instructions.md` once;
- answer with claim refs only, and never invent quotes;
- link when unsure between merge and link;
- `unsure` is an honest answer;
- pass `usage` on submit.

The loop is unchanged.

## 10. Deletions and reuse

Confirmed: v3 has no 3D view and no stored co-occurrence edges. The v2 web UI was deleted in stage 2, and the
co-occurrence code lives only in the reserved `semantic/edges.py`, `semantic/graph_build.py` and
`semantic/export.py`.

| v2 code | handling in stage 4 |
|---|---|
| `semantic/merge.py` | **keep in place** (pure; used by `migrate/plan.py` and `notes/identity.py`) |
| `semantic/canon.py` | move `DISCRIMINATING_TOKENS`, `discriminating_conflict` and `candidate_clusters` into `concepts/guardrails.py` and `concepts/candidates.py`; drop `passes_guardrails` (the token-overlap veto); delete the file |
| `semantic/{__init__,access,audit,concepts,current,edges,export,graph_build,proposer,query}.py` | delete: v2 schema, co-occurrence edges, broader/narrower edges, the labels-only proposer seam, export gate, audit log |
| `graph/analyze.py` | port the seeded Louvain helper into `library/literatures.py`; delete `graph/` |
| `citation/bib_parser.py`, `citation/resolve.py` | **keep** (pure parsing and strong-id-gated resolution, used by growth); move `_OVERLAP_STOPWORDS` per stage 3 §9 |
| `citation/{edges,parsed_bib,project_edges}.py` | delete (project.db and ORM; growth writes `reference_entries` / `citation_links` itself) |
| `acquisition/preview.py` | **keep** (pure growth estimate) |
| `acquisition/fetch.py` | already kept |
| `acquisition/{walk,resolve,backfill,bridge,service,doctor_reconcile}.py` | delete. The walk ranking and the title backfill are rewritten library-native in `library/growth.py`; the rest is ORM or project bound |
| `extraction/metadata.py` | delete (the id scan lives in `work/tasks_metadata.py`) |
| `db/models_project`, `sqlmodel` | delete with the semantic and lens-registry v2 tests that use them; drop `sqlmodel` when the reachability test shows no importer |
| `vocab.py` edge-type entries (`broader_than`, `narrower_than`, `related_to`, `contrasts_with`, `co_occurs_with`) | delete |
| `project/review.py` concept-edge review kinds | delete with `project/` remnants, if unreachable |
| `networkx` | **keep** (Louvain, spring layout, BFS) |

`tests/test_reachability.py`: `RESERVED` shrinks to empty during the stage. `KEEP` loses `lenses.registry`
(reachable from the lens panel) and `db.models_project`. The last slice asserts `RESERVED == {}`.

## 11. Acceptance plan

Run on the real migrated library in the v3 home; reports go to `<home>/acceptance/stage4/`.
1. **Embeddings vs keyword** (§5.5) on the agreed cases, side by side, with the adoption verdict.
2. **50 merges, links and splits reviewed.**
   - Sampling is stratified: from the first `merge` and `bridge` batches (rule merges excluded), split across
     verdict (merge / link / split) and family, with ambiguous-label items over-sampled to at least 10. It uses a
     seeded random draw, with the seed stored.
   - A review screen (`/concepts/review/{sample_id}`) shows each item's sides and quotes. The marks are right,
     wrong, should have been a link, and should have been a split, plus a note; they are stored in
     `concept_reviews`.
   - The report gives counts per mark, per verdict and per family, and lists the wrong merges.
   - **Gate**: wrong merges ≤ 3 of the sampled merges (or ≤ 10% if fewer than 30 merges are sampled). A miss
     means tightening the unclear thresholds and re-running, not adopting.
3. **Cost vs estimate**: the first bridge pass reports pairs, sense checks, packages and tokens (reported, or
   marked estimated), beside the dry-run estimate and the ticket 11 figures.
4. **Guardrails, automated** (`tests/acceptance/test_concept_guardrails.py`, also a CLI check
   `concepts check`) over the real library after the passes:
   - no concept spans families;
   - no concept contains both sides of a never-merge pair, or of a never-connect pair as merge or link;
   - no concept name is outside its members;
   - no merged group has a stoplist conflict;
   - no ambiguous label is merged without an agent or researcher sense decision;
   - every bridge counts only confirmed or single-literature senses;
   - the seed pairs (ATE/ATT, DiD/DDD, heterogeneity/heteroskedasticity) are in different concepts.
5. **Walk-through** for sign-off: search by meaning, open a concept with senses, split a merge in Recent
   merges, settle a queue item, regroup and rename a literature, grow a graph by one generation, toggle the 2D
   graph.

## 12. Build slices

Each slice is one agent, test-first, with strict file ownership. Fixtures are invented (stage 1 helpers); no
private text. Shared files (`pyproject.toml`, `uv.lock`, `cli.py`, `commands/__init__.py`, `.gitignore`) are
touched only by the slice named for them.

| # | Slice | Depends | Owns | Tests first |
|---|---|---|---|---|
| C1 | **Schema and model**: `0010_concepts.sql` (rebuild of `concept_decisions` carrying rows), `concepts/{families,model,refresh,guardrails}.py`, the seed never-merge and ambiguous lists | — | `db/schema/library/0010_concepts.sql`, `src/seedgraph/concepts/{__init__,families,model,refresh,guardrails}.py`, `tests/concepts/test_model*.py`, `test_refresh*.py`, `test_guardrails*.py` | grounding (hidden, inferred, agent-visible); rule merges with closure; ambiguous labels skip rule merges; migrated decisions applied, vetoed rows listed; byte-stable rebuild; handles resolve after merge and split; sense assignments survive a lightly reworded re-note via `identity.match`; transitive never-merge veto; migrate tests stay green without editing `src/seedgraph/migrate/`, with `expected.json` regenerated by `accept.expected_template` for the new tables and columns (as stages 2 and 3 did) |
| C2 | **Literatures**: `db/schema/graph/0002_literatures.sql`, `library/literatures.py` | — | those files, `tests/library/test_literatures*.py` | determinism under shuffled input; ladder and size rule; unplaced; ARI report; renames carried by Jaccard; fingerprint staleness |
| C3 | **Protocol seam**: `TaskType.select/subject_label/estimate`, `MAX_SUBJECTS` | — | `work/{routine,tasks,estimate}.py` (these hooks only), `web/views/batch.py` (subject label), `tests/work/test_task_seam*.py` | a fake non-paper task creates, claims, submits and finishes a batch; paper tasks unchanged (full stage 2 suite green) |
| C4 | **Search and embeddings**: `concepts/search.py`, `concepts/vectors.py`, `llm/ollama.embed`, the evaluation runner | C1 | those files, `commands/concepts.py` (`search`, `eval-search`) and its registration in `cli.py` / `commands/__init__.py`, `pyproject.toml` + `uv.lock` (numpy, only after adoption), `tests/concepts/test_search*.py` | keyword aggregation to senses; vector cache keyed by text hash + model; fake embedder separates homonyms in a synthetic case; metrics and pass rule on invented cases; graceful keyword fallback |
| C5 | **Concept tasks**: `work/tasks_concepts.py`, `concepts/{candidates,decide,estimate,packages}.py`, instructions | C1, C3 (C4 for meaning candidates; keyword path first) | those files, `commands/concepts.py` (`estimate`, `check`), `tests/work/test_tasks_concepts*.py` | candidates skip decided and never-connect pairs; packages render refs; each verdict validates; veto feedback; unclear classification; researcher override; save-time conflict → queue; dry-run matches created batch; budget cut |
| C6 | **Growth**: `library/growth.py`, reference parsing for new papers, citation-review items | C2 | `src/seedgraph/library/growth.py`, edits to `citation/{bib_parser,resolve}.py` imports only, `tests/library/test_growth*.py` | fake provider chain: ranking by member citations; cited-only creation via stage 1 identity; OA only; needs-a-PDF fallthrough; idempotent re-walk |
| C7 | **Concept pages**: search, concept page, recent merges, queue, review sample, settings lists | C1, C4, C5 | `web/views/concepts.py`, `web/routes/concepts.py`, `web/templates/concepts/*`, tests | each POST calls one deep function; split/undo round-trips; queue actions; review marks stored |
| C8 | **Graph pages**: tabs, slice, overview, SVG graph, growth wizard, lens panel | C2, C6, C7 | `web/views/graphs.py`, `web/routes/graphs.py`, `web/templates/graphs/*`, `library/graphs.py` (`unchoose_lens`), tests | deterministic SVG for a fixture; no inline style or script (CSP test); growth steps; outside-graph suggestions |
| C9 | **MCP and skills** | C4, C5, C2 | `mcp/tools_concepts.py`, `mcp/server.py` (registration line), `agent_setup/*.md`, tests | in-memory MCP calls; agent audience rules; mode marker; hubs resource |
| C10 | **Deletions and reachability** | C1–C9 | deleted modules and tests, `tests/test_reachability.py`, `vocab.py`, `pyproject.toml` (drops) | `RESERVED == {}`; full suite green |
| C11 | **Acceptance harness** | C10 | `tests/acceptance/stage4/`, `commands/concepts.py` (`review-sample`) | opt-in; writes reports to the home |

**Waves (2–3 at a time).** Wave 1: C1, C2, C3. Wave 2: C4, C5 (keyword candidates first), C6. Wave 3: C7, C8.
Then C9, C10 and C11. The critical path is C1 → C5 → C7 → C8 → C10. `commands/concepts.py` is split by
function between C4, C5 and C11, which merge in sequence. `pyproject.toml` goes to C4 (numpy, if adopted) and
C10 (drops), never in the same wave.

## 13. Open questions for the researcher

Each has a recommendation (*R*).
1. **Test cases for the adoption test.** *R:* start from the proposed list of partner cases and homonym probes
   (kept in the v3 home, never here). Add or drop cases before the first run, then freeze them. Aim for 10–15
   partner cases and 6–10 homonym probes, with as many as possible crossing literatures or graphs.
2. **Re-note for plain words before bridge finding?** Plain words exist only on the few papers re-noted in v3.
   A full re-note costs a stage-2-sized notes batch per paper. *R:* not before the first pass. Claim text +
   quote carry the meaning (§3.3). Re-note only the papers behind queue items or wrong-merge marks, and
   re-assess after the 50-item review.
3. **Bridge-pass budget.** *R:* approve the first pass at the dry-run's upper estimate for assumptions,
   estimands and methods (expected within the ticket 11 range). Run results and data only after the review
   gate.
4. **Embedding model download.** *R:* allow `nomic-embed-text` (about 274 MB) for the test. Keep
   `mxbai-embed-large` (about 670 MB) in reserve. Delete both if the test fails.
5. **The ambiguous-label seed list.** *R:* accept the generic seed (§2.3) and edit it on the settings page. The
   real-library examples behind it are in the hand-back report.
6. **Stoplist additions for econometrics** (e.g. strict / weak / sequential / local): *R:* do not add them to
   the stoplist. They would block merges where the words do not discriminate. Use never-merge rows (strict ≠
   weak exogeneity) instead.
7. **Reach threshold** for queuing merges between established concepts (`REACH_MIN = 5`): *R:* accept, and tune
   after the first pass's queue share is known.
8. **Migrated agent merges that a guardrail now vetoes**: *R:* list them and leave them unapplied, rather than
   override the guardrail.
9. **Literature size rule** (largest ≤ 30%, at least 3 literatures of ≥ 5): *R:* accept as the default, with the
   fewer / more control per graph.
10. **2D graph as server-rendered SVG** (no zoom or pan): *R:* accept; revisit only if you miss zoom.
