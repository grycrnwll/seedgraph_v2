# v3 Stage 2 design: one agent work protocol

Status: design for build. No code yet. This document covers v3 stage 2: one package → work → check → save
protocol for every AI task, the batch and package state model in `library.db`, task types, the kept
validate-and-anchor boundary, note format changes, the text-quality gate, Ollama as a worker, the v3 MCP
server and its registration during the build, the deletions from v2, the acceptance plan and the build slices.
It uses the glossary terms from `CONTEXT.md` (library, context graph, batch, worker, background agent,
interactive session, held paper) and the deep-module vocabulary from `codebase-design` (module, interface,
depth, seam, adapter). Stage 1's `docs/V3_LIBRARY_DESIGN.md` is binding for storage.

The repo is public. Real counts, token measurements, calibration output and acceptance reports live in the
v3 home, never in this document or the repo. Papers are referred to only by golden-set id (GSnn).

Words used here:
- **Package**: the unit a worker claims: one paper's text, or one part of a long paper's text, plus a short
  message. Its id is the `packet_id` (the name is kept from v2 and the spec).
- **Package claim**: a worker's time-limited hold on one package. Always written "package claim", because
  "claim" alone means a note claim (`extracted_claims`).
- **Submission**: a worker's result for one package.

## 1. Protocol state model

### 1.1 The routine

Create batch (gate, references stripped, split only when long) → claim a package → read its file → submit →
feedback (`needs_revision` loops back to submit; expiry returns the package to the pool) → when every part of
the paper is saved, one new note version becomes current.

Every step is one function of one deep module, `seedgraph.work` (§11). The MCP tools (§7), the CLI mirror
(§10.3) and the Ollama worker loop (§6) are thin adapters over the same seven calls:

```python
create_batch(lib, task: str, *, paper_ids: Sequence[str] | None = None, query: str | None = None,
             graph: str | None = None, options: BatchOptions | None = None) -> BatchView
claim_next(lib, batch_id, worker: WorkerRef, *, paper_id: str | None = None) -> ClaimResult
submit(lib, packet_id, claim_id, *, note: dict | str | None = None, note_path: Path | None = None,
       accept_unanchored: bool = False, model: str | None = None, usage: dict | None = None) -> SubmitResult
release(lib, packet_id, claim_id, reason: str) -> None
stop(lib, batch_id, *, mode: 'finish' | 'now') -> BatchView
report_limit(lib, batch_id, worker: WorkerRef, *, resets_at: str | None, message: str | None) -> BatchView
continue_batch(lib, batch_id) -> BatchView
batch_view(lib, batch_id) -> BatchView     # read-only, also used by status/list
```

### 1.2 Batch

A batch is one task type over a chosen set of library papers ("subjects").
- **Selection** is either explicit `paper_ids` (ordered, distinct, 1–1000) or a `query` in a small, fixed
  selection language, resolved once at creation and frozen:
  `held [in graph <slug>] [without current note | with stale note | needing metadata | lens <id> missing]`.
  Examples: "held papers in graph diff_in_diff without notes", "held needing metadata". The query text and the
  resolved ids are both stored, so the batch is reproducible and the UI can show what was asked.
- **Instructions and schema are sent once**: at creation, seedgraph writes `work/<batch_id>/instructions.md`
  (role, rules, the field table, vocabularies, an example) and `work/<batch_id>/labels.json` (§4.2). Packages
  only point at them. The work skill (§7.3) carries the generic loop, so the instructions hold only what is
  task- and version-specific.
- **Options** (`BatchOptions`, stored as JSON): `claim_ttl_s` (default 1800), `split_threshold_tokens`
  (default 100 000), `part_target_tokens` (default 60 000), `replace` (re-note papers whose current note is
  already fresh), `model` (optional hint; seedgraph names none by default), `auto_continue_after_limit`
  (stage 6, default off), `stop_after_papers` (stage 6, optional).
- **Eligibility per paper** at creation: `preserved` (a fresh current note exists and `replace` is off),
  `held_back` (text-quality gate or open `pdf_title_miss` item, §5.4), `failed` (no markdown, unreadable
  blob), otherwise `queued`. Held-back and failed papers are listed in the creation reply with their reason;
  they never get packages.

### 1.3 Packages: whole paper by default

- **Text**: the paper's current markdown minus the reference list, cut out by the deterministic
  `document_sections.section_kind = 'references'` ranges (v2's `_strip_references`, fail-open when there are
  no sections). Each cut is replaced by a visible `[references omitted]` marker. Quotes are always anchored
  against the full stored markdown, so the cut never affects anchoring.
- **One form only**: a package is a short JSON message plus a file path. The message carries `packet_id`,
  `claim_id`, `batch_id`, `paper_id`, title and year, `part` (k of n), `text_path`, `instructions_path`,
  `labels_path`, `result_path` (where to write the note: `work/<batch_id>/<packet_id>.result.json`),
  `est_tokens` and `claim_expires_at`. The text file is `work/<batch_id>/<packet_id>.md`. The tool reply never
  contains paper text, so it never arrives inline in one session and as a persisted file in another.
- **Split only over the threshold**: if the stripped text is over `split_threshold_tokens` (estimated as
  characters / 4, as v2 `llm.tokens`), it is cut into `ceil(tokens / part_target_tokens)` near-equal parts at
  section boundaries (falling back to paragraph boundaries inside an oversized section). Most papers are well
  under the threshold; GS08 is the golden-set paper that exercises the split path.
- **How parts make ONE note without a scalar merge**: the note's fields have fixed owners.
  - Part 1 owns the four scalars (`research_question`, `main_contribution`, `setting`,
    `estimand_or_target_object`). It holds the abstract and introduction, where these are stated.
  - Every part may contribute array elements (methods, data, assumptions, results, limitations, robustness
    checks, open questions) found in its own text.
  - A later part that sends a scalar gets feedback ("scalars belong to part 1") and the scalar is dropped. A
    part with no elements of a field just sends an empty array; no part is asked to write `not_found` rows for
    text it cannot see.
  - The assembled note is part 1's scalars plus every part's arrays concatenated in part order, with
    `source_chunk_index` = part index on each claim row. Nothing is guessed or merged; parts can be worked by
    different workers in parallel.
  - The instructions for parts 2..n say what part 1 is for, so a part-2 worker does not restate the paper's
    research question as a result.

### 1.4 Package claims and expiry

- `claim_next` runs in one `BEGIN IMMEDIATE` transaction:
  1. expire lapsed package claims (lazy expiry; there is no daemon, and every MCP server process is its own
     writer, so every protocol call and every status read expires first);
  2. refuse with a pause reply if the batch is stopped or limit-paused (§1.6);
  3. pick the open package with the lowest `(paper ordinal, part)`, preferring further parts of a paper this
     worker already holds or finished (so one agent usually keeps one long paper), or the given `paper_id`;
  4. insert a `package_claims` row with `expires_at = now + claim_ttl_s` and return the package message with
     its `claim_id`. `submit` and `release` require that `claim_id` and check it is the package's live claim,
     so a late holder is told apart from the new holder on any transport, even when several subagents share
     one MCP server process.
- SQLite's writer lock makes two simultaneous `claim_next` calls get different packages. No collision is
  possible without a second writer bypassing the module.
- A claim **expires** after `claim_ttl_s` without a submit. The package returns to the pool and the claim row
  keeps `release_reason = 'expired'`. Every submit, including a `needs_revision` reply, renews the claim.
- A submit on an expired claim is accepted (the claim revives) if nobody has re-claimed the package. It is
  refused, with "claim expired and the package was re-claimed by another worker", if someone has.
- `release` hands a package back early, for example when an agent sees its session ending.
- **Worker identity** (`WorkerRef`): `worker_kind` (`interactive`, `background`, `ollama`), `client`, and a
  `worker_id` for reporting. The MCP server mints one per server process, and `work_claim` accepts an optional
  `worker` name so subagents sharing a process can report separately; the CLI takes `--worker`. Correctness
  never rests on `worker_id`: that is the `claim_id`'s job.

### 1.5 Submit and feedback

`submit(packet_id, claim_id, note | note_path, …)`: the server looks up the binding (batch, paper, markdown
id and hash, part, attempt) from `packet_id`. The worker never retypes hashes. `note_path` is the package's
`result_path` or any absolute path to a `.json` file holding the note JSON alone (no envelope). An inline
`note` is equally valid and is the route for stage 6 background agents, which get no file-write tool.

1. **Validate**: `validate_note` (kept as-is: bracket extraction plus Pydantic), then the task type's own
   checks (vocabularies, part ownership, required plain-words statements, §4). Errors give the reply
   `status: "invalid"` with the list. The package claim is kept and renewed.
2. **Anchor preview** (pure, no writes): every `found` claim's `exact_quote` is located in the full markdown
   with the unchanged exact-unique matcher (`anchor.locate_quote`). A failed quote gets a reason:

   | reason | detection | feedback gives |
   |---|---|---|
   | `not_found` | no exact match, and none of the diagnoses below match | the nearest passage by token overlap, if one scores well |
   | `multiple_matches` | two or more exact matches | the count; "extend the quote until it is unique" |
   | `lookalike_characters` | matches after a confusable fold: NFKC (ligatures), µ U+00B5 ↔ μ U+03BC, curly ↔ straight quotes, en/em dash and minus ↔ hyphen, no-break or thin space ↔ space | the exact markdown slice to copy, and each differing code point (`U+00B5 in text, U+03BC in quote`) |
   | `whitespace_or_linebreak` | matches after collapsing whitespace and joining `-\n` hyphenation | the exact slice to copy |
   | `mojibake` | matches only after folding the markdown with the mojibake table (§5.2): the agent "repaired" garbled text | the exact garbled slice to copy (only reachable on a paper let through by an override) |
   | `in_references` | the match lies inside the stripped reference list | a warning only; it still anchors |
   | `empty` | `found` with no quote | "add a verbatim quote or set status to ambiguous" |

   The matcher used for saving is **not loosened**. A diagnosis only explains a failure and hands the agent
   the exact text to copy. This keeps "quotes anchored" comparable with the live run's figure.
3. **Outcome**:
   - All found quotes anchor (or `accept_unanchored=true`): the submission is **saved** for the package
     (`status: "accepted"`). Unanchored quotes are downgraded to `ambiguous` at save time, exactly as v2's
     `save_normalized_note` does.
   - Some quotes fail and `accept_unanchored` is false: `status: "needs_revision"`. Nothing is saved, the
     claim is renewed, and the agent may resubmit the same `packet_id` as often as it likes before expiry.
   - The reply always carries `claims_kept` (claim rows the note would produce), `quotes_found`,
     `quotes_anchored`, `failed_quotes[]` (`field_key`, `reason`, `hint`, `markdown_slice`) and `warnings[]`.
4. **Completion**: when every package of the paper has a saved submission, the paper is assembled (§1.3) and
   saved through the kept boundary (§3) in the same transaction. The reply then carries `note_id` and
   `paper_status: "done"`. Before that it carries `paper_status: "in_progress"` and `parts_done k/n`. **A note
   becomes current only when the whole paper is complete.** Saved part submissions of an unfinished paper are
   invisible to search.
5. **Resubmit after completion** is not a revision. It needs a new batch (`replace`), which makes a new
   version. The completed version is never edited in place.
6. **Reported failures**: `release(packet_id, claim_id, reason)` with `reason` in `paper_problem` (the paper cannot be
   noted, e.g. unreadable text; the paper becomes `failed` with the worker's message) or `give_back` (the
   package returns to the pool). Session-wide trouble goes through `report_limit` (§1.6), not through submit.

### 1.6 Pause states and "continue"

The batch shows exactly one state. Two pause states are stored; two are derived from claims, so they can
never be stale:

| state | stored or derived | set by | shown as |
|---|---|---|---|
| **stopped on purpose** | stored: `pause_kind = 'stopped'` | `stop(mode)`: `finish` lets live claims finish; `now` releases them and refuses their submits | "Stopped" |
| **usage limit hit** | stored: `pause_kind = 'usage_limit'`, `resets_at` (UTC, or NULL if unknown), `pause_detail` | `report_limit` from an interactive agent (the skill tells it to call this when its client reports a limit), or the stage 6 launcher's `rate_limit_event`; the reporting worker's claims are released at once | "Paused: usage limit, resets <time>" / "resets: unknown" / "out of API credits" |
| **claim expired** | derived | at least one package whose latest claim expired without a submit and that is not claimed again, with no live claims on the batch | "Paused: a worker stopped responding; its package is back in the pool" |
| **waiting for an agent** | derived | open packages, no live claims, none of the above | "Waiting for an agent" |
| *(working)* | derived | at least one live claim | "Working: n agents" |
| *(finished)* | derived | no open or claimed packages | "Finished: done / held back / failed" |

Precedence when several apply: stopped > usage limit > claim expired > waiting. A claim refused because of a
pause returns `{status: "paused", state, resets_at, next_action}` in plain words.

**Continue** is the only resume. `continue_batch` clears a stored pause and acknowledges expired-claim flags
(`package_claims.acknowledged = 1`). Claiming also clears the derived "claim expired" state naturally,
because the batch then has a live claim. There is no `resume` flag, and `claim_next` never needs one.

### 1.7 Where it lives: `db/schema/library/0008_work_protocol.sql`

The batch progress lives in `library.db`, so any mix of interactive sessions, background agents and Ollama
can join a batch. Every table follows stage 1's conventions: TEXT ids, `origin`, explicit ordering, no rowid
tie-breaks.

| table | key columns | notes |
|---|---|---|
| `batches` | `batch_id` ('b_' + time-ordered hex), `task_type`, `task_version`, `selection_query`, `selection_json`, `graph_slug` NULL, `options_json`, `instructions_sha256`, `labels_sha256`, `pause_kind` NULL/'stopped'/'usage_limit', `stop_mode`, `resets_at`, `pause_detail`, `created_at`, `created_by_client`, `origin` | one row per batch |
| `batch_items` | `batch_id, ordinal` PK; `subject_kind` ('paper', or a stage 4 kind), `subject_id`, `state` (`queued`, `held_back`, `preserved`, `failed`, `in_progress`, `done`, `stale`, `conflict`), `markdown_id`, `markdown_hash`, `expected_current_note_id`, `part_count`, `result_ref` (note id, lens run id, …), `reason`, `attention_item_id` | UNIQUE(`batch_id, subject_kind, subject_id`) |
| `work_packages` | `packet_id` PK (sha256 of the binding: batch, subject, markdown hash, part, part range, task version, attempt), `batch_id, ordinal, part, part_count, start_char, end_char, est_tokens, text_path, state` (`open`, `claimed`, `saved`, `void`), `attempt` | |
| `package_claims` | `claim_id` PK, `packet_id`, `worker_id, worker_kind, client, model`, `claimed_at, expires_at, released_at, release_reason` (`submitted`, `expired`, `released`, `stopped`, `limit`), `acknowledged` | the live claim is the row with `released_at IS NULL`; a partial unique index enforces at most one per package |
| `package_submissions` | `submission_id` PK, `packet_id, claim_id, received_at, payload_json, payload_sha256, outcome` (`accepted`, `needs_revision`, `invalid`), `feedback_json`, `usage_json` | the note is stored **once** (v2 stored `note` and `payload`); only the latest `accepted` row per package is used |
| `batch_events` | `batch_id, seq` PK; `at, kind, packet_id, worker_id, detail_json` | append-only feed for the stage 3 activity view and stage 6 batch page |
| `text_quality` | `markdown_id` PK, `status` (`pass`, `garbled`, `no_text`), `score_json`, `checker_version`, `assessed_at` | gate cache (§5) |
| `extracted_claims.plain_words` | `ALTER TABLE … ADD COLUMN plain_words TEXT` | §4.3 |
| `claim_fts` | dropped and recreated with a `plain_words` column, then repopulated | FTS5 cannot ALTER; done once, here, so stage 4's keyword fallback can match plain words |

Package text files live under `<home>/work/<batch_id>/`. This is the only folder the stage 6 lockdown lets
background agents read. Old batch folders are cleaned by `seedgraph work prune --finished-before <date>`,
never automatically.

Every protocol write goes through `library.store.transaction` and sets `library_meta.dirty = 1`. A later
`migrate run` therefore needs `--replace-dirty`, which is intended: re-noting is real library work (open
question 6).

**Interaction with the stage 1 migration.** `migrate` applies every numbered library schema file to its
staging build, so a migrated library gains the (empty) protocol tables, `plain_words` and the recreated
`claim_fts`. `migrate/apply.py` rebuilds FTS with explicit column lists, so the extra `claim_fts` column stays
NULL for carried claims and runs stay byte-identical. `report.json` gains `library.<table>` keys for the new
tables (all zero), so the researcher's `expected.json` is regenerated with `accept.expected_template`. P1
must keep stage 1's migrate tests green without editing `src/seedgraph/migrate/`.

## 2. Task types on one routine

### 2.1 The seam

The protocol module owns batches, packages, package claims, expiry, pauses, feedback assembly and the save
transaction. What varies per task sits behind one interface, `TaskType`, registered by name:

```python
class TaskType(Protocol):
    name: str                       # 'notes', 'metadata', 'lens:<lens_id>', later 'concepts:bridges'
    version: str                    # schema+prompt version, or lens definition hash
    subject_kind: str               # 'paper' (stage 4 may add another)
    allowed_workers: frozenset[str] # e.g. {'interactive', 'background'} or {'ollama', 'interactive', 'background'}
    def resolve(self, lib, selection) -> list[str]                      # query → subject ids
    def eligibility(self, lib, subject_id, options) -> Eligibility      # queued / preserved / held_back / failed
    def plan(self, lib, subject_id, options) -> list[PackagePlan]       # text slices (+ split)
    def instructions(self, lib, batch) -> Instructions                  # instructions.md, labels.json
    def check(self, lib, package, payload) -> CheckResult               # pure: errors, warnings, quote list
    def save(self, conn, item, parts: list[AcceptedPart]) -> SaveResult # inside the protocol's transaction
```

Three adapters exist in stage 2 (notes, metadata, lens), so the seam is real, not hypothetical. Anchor
preview and anchoring are protocol services that `check` and `save` call. A task type cannot skip them for
quotes it declares.

### 2.2 The adapters

| task | packages | output | `save` | default workers |
|---|---|---|---|---|
| `notes` | whole paper minus references; parts above the threshold | `DefaultNoteV1` (format §4) | assemble parts → `normalize_note` → `save_note` (§3) | interactive, background |
| `metadata` | the first ~2 600 characters of the markdown (the cover and title page, as v2 `extraction/metadata.py`) plus the deterministic DOI/arXiv hits from the same region | `BiblioMeta` (title, authors, year, venue, doi, arxiv) | fill empty fields only; a researcher-set field is never overwritten; `title_source = 'agent'` or `'ollama'`; new ids into `identifiers`; resolves the paper's `metadata_needed` attention item | **ollama**, interactive, background |
| `lens:<id>` | whole paper as for notes; the lens prompt (`lenses/prompt.build_lens_prompt`) and record schema go in `instructions.md` | lens records | ported from `lenses/runner._project_found_record`: lens `extraction_runs` row (lens id + definition hash), `lens_outputs`, projected claims and anchors | interactive, background |
| `concepts:*` | **stage 4 hook only**: `subject_kind` such as `concept_pair_group`, packages of label pairs with quotes from both papers | merge / link / reject decisions | writes `concept_decisions` | interactive, background |

The lens task ports the v2 lens behaviour as it is: same YAML definitions, same record validation, same
evidence policy and the same "current = latest run matching the definition hash". Only the dispatch changes,
from the API runner to packages. Lens batches are chosen per graph (`graph_lenses`) by passing `graph`.

Adding a task type means writing one adapter and one registry line. The protocol, MCP tools, CLI and skill do
not change; the skill's loop is task-agnostic and reads the task's `instructions.md`.

## 3. The validate-and-anchor boundary

Kept unchanged in meaning: `validate_note` (bracket extraction, tolerant string coercion, Pydantic),
`normalize_note` (one claim row per scalar and per array element, explicit `not_found` rows, decision 47
assumption folding, archetype) and the save boundary (anchor every found quote through the one span writer,
downgrade unanchored ones, write run, note, claims, anchors and FTS in one transaction).

Adapted from `project.db` to `library.db`:
- `save_normalized_note` moves to `seedgraph/notes/save.py` as `save_note(conn, *, paper_id, markdown_id,
  markdown_hash, drafts, note_text, archetype, raw_note_json, provenance, expected_current, extraction_mode,
  part_count, batch_id) -> SavedNote`. It takes an open library connection inside the protocol's
  `BEGIN IMMEDIATE`, not a SQLModel session plus a `cache.db` connection.
- `ensure_span` gets a library port (`seedgraph/notes/spans.py`): it reads the markdown from the library blob
  store by `markdown_id`, writes `evidence_spans`/`span_fts` keyed by `markdown_id` (no `work_id`, stage 1
  §2.1), deduplicates on `(markdown_id, quote_hash)` and keeps `anchor.assert_invariant`. `access_class` is
  provenance only (ADR-0012), so the most-restrictive reconciliation stays as data but gates nothing.
- **Versions** (stage 1 §6.6): `version_no = MAX(version_no) + 1` for `(paper_id, schema_id)`, assigned
  inside the same transaction as the compare-and-activate. `latest_note_id` is ported onto `note_current`.
  If the current note changed since the batch was created (`expected_current_note_id`), the item becomes
  `conflict` and nothing is saved. `replace` is the only way to supersede a fresh note.
- **Staleness check before save**: the paper's primary file still points at the package's `markdown_id`.
  Otherwise the item becomes `stale` ("the paper was re-converted; create a new batch").
- **Schema id stays `default_research_note_v1`**, with `SCHEMA_VERSION` and `PROMPT_VERSION` bumped to 2.0.0
  (§4). A new id would give each paper two "current" notes under `note_current`'s per-`schema_id` rule.
- `extraction_runs` for agent notes: `provider` = client, `access_mode = 'agent_session'`,
  `extraction_mode` = `agent_whole` or `agent_parts`, `chunk_count` = parts, `run_id` = batch id,
  `input_tokens`/`output_tokens` from the reported usage when given.
- **Stage 3 overlay**: `save_note` never reads or writes `note_overlays`. Each completed paper is a new
  extraction run, which is exactly what ADR-0011's overlay replays onto. Stage 3 changes `note_current` into
  "extraction + overlay" and designs claim identity. Nothing in stage 2 assumes claims are stable across
  versions.
- Agent-audience hiding (open `hides_notes` items, labelled versions) is unchanged. A note saved by the
  protocol for a paper with an open `pdf_title_miss` item stays hidden until the item is resolved, but the gate
  normally keeps such papers out of batches anyway (§5.4).

## 4. Note format changes (`default_research_note_v1`, schema 2.0.0)

### 4.1 Fixed vocabularies

What the migrated v2 notes show, read-only:
- **`epistemic_type` is never filled by agents.** In every carried raw note it is null. The stored column is
  derived by `normalize_note` from `assertion_status` (`llm_inferred` iff `inferred`, else `llm_extracted`).
  The agent-facing field only confuses agents (ticket 04 friction 5).
- `assumption_type` is folded by decision 47 into three claim types (identification, regularity, everything
  else). Any value beyond the first two is discarded today.
- `method_type`, `result_type` and `limitation_type` are free strings stored as `claim_subtype`. They have
  long tails of near-duplicate spellings ("simulation" / "simulation result" / "simulation finding").

Decisions:
- **Remove `epistemic_type` from the agent-facing envelope.** It is ignored if sent (`extra="ignore"`). The
  column stays, derived as today, because it is also how researcher content is labelled (`user_supplied`,
  `user_validated`) from stage 3.
- **`assumption_type`**: `identification | regularity | general`. This is exactly what is stored, now stated.
- **`method_type`**: `estimator | regression | research_design | identification | inference | model |
  simulation | asymptotic_theory | computation | decomposition | empirical_application | data_analysis |
  other`. `model` keeps v2's meaning: the claim type becomes `model` instead of `method`.
- **`result_type`**: `empirical | theoretical | identification | asymptotic | simulation | heterogeneity |
  mechanism | descriptive | methodological | policy | other`.
- **`limitation_type`**: `scope | external_validity | identification | measurement | data | inference |
  model_specification | method | computational | interpretation | other`.
- **An unknown value is not a rejection.** A known variant ("Monte Carlo simulation", "asymptotic
  normality") is mapped by a small table in `seedgraph/notes/vocab.py`. Anything else becomes `other`, with a
  feedback warning listing the allowed values, so a worker never loops on a label.
- **Old notes are not rewritten.** Read-side facets (stage 3 filters, stage 4 concept typing) map old free
  text through the same variant table, so old and new notes facet together.
- The vocabularies were drafted from the observed values, and a keyword mapping of the observed values onto
  them covers most claims. The calibration numbers stay in the v3 home. A vocabulary change bumps
  `SCHEMA_VERSION`.

### 4.2 Label consistency

At batch creation, `labels.json` gets, per label group, the library's top ~100 `normalized_label`s, ranked by
the number of distinct papers with a current note (agent audience) that use them. The groups are: methods and
models, each assumption type, estimands, data sources, results, limitations, robustness checks. The file is
frozen for the batch, so every package in a batch sees the same list and the instructions file stays
byte-stable. The rule in the instructions: "If one of these labels names the same concept, use it exactly;
otherwise use the standard community name." This replaces ticket 04's missing hand-over of earlier chunks'
labels.

### 4.3 Plain-words statements

`Assumption`, `MethodOrModel` and the `estimand_or_target_object` scalar gain `plain_words`: one or two
sentences stating the claim without the paper's notation, e.g. "units do not change behaviour before
treatment starts" next to the label "no anticipation". It is **required** for `found` claims of those three
kinds. A missing one is a validation error with a pointed message; such errors are cheap to fix on resubmit.
It is stored in `extracted_claims.plain_words` and indexed in `claim_fts` (§1.7). Stage 4's meaning-based
bridge search uses it; old notes gain it only when re-noted.

### 4.4 Instruction fixes in the same version

`instructions.md` replaces the `anyOf[string, null]` JSON Schema with a compact field table and one worked
example (Pydantic stays the validator); "omit none" becomes "scalars may be null, arrays may be empty"; quote
rules say copy character for character, never repair characters, and make each quote long enough to be unique.

## 5. Text-quality gate

Before packaging, hold back papers whose markdown is garbled, flag them for re-conversion with Marker's
forced OCR, and spend no tokens on them. It must hold back GS06 (the garbled live-run paper) and, by its
golden-set category, GS07, and pass every other golden-set paper (§5.6).

### 5.2 Detector (`seedgraph/library/textquality.py`, pure)

```python
assess_text(markdown: str) -> TextQuality
  # .status: 'pass' | 'garbled' | 'no_text'; .doc_rate; .bad_pages; .pages; .hits_by_signal; .examples
```

**Mojibake families**, in one table, `seedgraph/library/mojibake.py`, used by the gate, quote feedback and
quote re-matching. Stage 1's title matcher keeps its own fold table unchanged in stage 2: switching it would
change stored PDF-check results and its calibration, for no stage 2 benefit. Its table lacks family A's `§`
and `á`; unifying the two is a later, separately calibrated change.
- **Family A** (seen in GS06 and GS07): `Ö`→fi (`identiÖed`), `§`→ff (`di§erent`), `¢`+space→ffi
  (`e¢ ciency`), `á`→fl (`reáects`), `í`→apostrophe before `s` (`authorís`).
- **Family B** (seen elsewhere in the library): `…`→fi (`…rst`), `¤`→ff (`di¤erence`), `¢`→ffi.
- **Strong signals**: `Ö`, `§`, `¢`, `…`, `¤` touching a lowercase ASCII letter on both sides, or at word
  start followed by two or more lowercase letters (`Örst`). Also U+FFFD, `(cid:N)` runs and private-use-area
  code points (common converter garbage, none in the golden set).
- **Corroborating signals only**: in-word `á` and `í` are real letters in names and non-English words. They
  are counted and reported but never trigger the gate alone.
- `Þ` (a closing parenthesis in math) and `ß` are not signals (stage 1 checked these).

**Score**
- The text scored is the reference-stripped text (what would be sent), so names in a reference list never
  count.
- Words = `[A-Za-z]{2,}` tokens. `doc_rate` = strong hits per 1 000 words.
- Pages = Marker page markers (`^\{N\}-{10,}$`). A page is **bad** when it has at least 50 words, at least 2
  strong hits, and at least 2.0 strong hits per 1 000 words. Without page markers the whole text is one page.
- **garbled** if (`doc_rate ≥ 1.0` and strong hits ≥ 10) or (bad pages ≥ max(3, 20 % of pages)). The page
  rule catches papers garbled only in some sections, such as an appendix in another font.
- **no_text** if the median page has fewer than 30 words (a scan with no usable text layer). It is handled
  like garbled: forced OCR.
- Constants are module-level and test-covered. A design-time read-only pass over the library found a wide
  gap: clean papers have essentially no strong hits (every non-garbled golden-set paper has none), while the
  garbled ones are an order of magnitude above the threshold. The S4 slice's calibration script reruns this
  and writes the distribution to the v3 home, never to the repo. It must confirm the thresholds sit in that
  gap. The detector deliberately does no dictionary lookup: it is cheap, deterministic and explainable
  ("40 `Ö` inside words, 13 `§` inside words; examples: …").

### 5.3 When it runs

- After every conversion (`library.files.convert_file` calls it and caches the result in `text_quality`), so
  stage 3's "held back (garbled)" library view needs nothing new.
- At batch creation, for papers without a cached result for their current markdown (e.g. migrated papers).
  `checker_version` in the cache forces a re-assessment when the detector changes.

### 5.4 Holding back and the override

- A `garbled` or `no_text` result raises an attention item `text_garbled` / `text_missing` (stage 1 table,
  new kinds, `hides_notes = 0`: existing notes stay visible, their quotes are verbatim copies of the
  garbage). The batch item is `held_back` with the reason and the suggested fix.
- Batch creation also holds back papers with an open `pdf_title_miss` or `shared_pdf` item hiding notes.
  Noting a possibly wrong PDF would waste tokens.
- **Override**: the researcher resolves the item with `accept_text` ("note anyway"). The gate then passes
  that `markdown_id`, and quote feedback's `mojibake` diagnosis (§1.5) helps the agent copy the garbled
  characters.

### 5.5 Forced-OCR re-conversion, quote re-matching, re-note offer

`seedgraph library reconvert --ocr <paper_id>…` (CLI; a button in stage 3):
1. `files.convert_file(lib, source_file_id, config=MarkerConfig(force_ocr=True), force=True)` through the
   stage 0 warm pool. `force_ocr` is already in the converter protocol's options, so the ADR-0014 contract is
   unchanged. The result is a new `markdown_documents` row with its own fingerprint; sections and spans are
   indexed.
2. The gate runs on the new markdown. If it still fails, the paper keeps its old markdown and the item stays
   open with "forced OCR did not help". The researcher can override or replace the PDF.
3. If it passes, `paper_files.markdown_id` switches to the new markdown and the item is resolved.
4. **Quote re-matching** for the paper's current note. For each anchored quote, try in order: an exact
   unique match; then the quote folded through the mojibake table (old quotes copied the garbage, so the raw
   string cannot match clean OCR text) and an exact unique match; then a whitespace-normalised match. Report
   `rematched / previously anchored`.
5. If **every** previously anchored quote re-matches, the note is carried to the new markdown as a new
   version (`extraction_mode = 'rematch'`, `parent_extraction_run_id` = the old run, same claim text,
   anchors on the new spans). Otherwise the old version stays current on its old markdown (the blob is kept)
   and an attention item `renote_offered` records the match rate and offers a notes batch for the paper.
   Nothing is re-noted automatically.

**Golden-set note.** Stage 0's golden check compares a fresh default conversion against the stored v2
markdown. A forced-OCR conversion of GS06/GS07 will differ from the stored markdown, by design. Golden checks
keep running only the default conversion path; forced OCR is never part of them.

### 5.6 The golden-set conflict, resolved

The spec asks the gate to "pass every golden-set PDF" and also to hold back the garbled live-run paper. That
paper is GS06, which is in the golden set, and GS07 shares its mojibake. The two checks measure different
things:
- the **golden set measures conversion fidelity**: same markdown as v2, page markers, quotes, equations,
  tables, speed. GS06/GS07 pass it, because the converter faithfully reproduces the same garbage;
- the **gate measures text quality**: is this text worth sending to an agent?

`golden_set.json` already labels GS06 and GS07 with the category `garbled`. **Acceptance for the gate is
therefore: the gate's verdict equals the golden-set category.** GS06 and GS07 are held back, and the other ten
pass. The ten include the two old scans (OCR text, not garbled) and GS08 (very long). This is recorded as open
question 1, with this as the recommendation.

## 6. Ollama as a worker

- `seedgraph work ollama <batch_id> [--model M] [--max N]` runs an in-process loop: `claim_next` with
  `worker_kind = 'ollama'` → build the prompt from `instructions.md` plus the package text → the Ollama client
  with JSON output → `submit` → repeat until the batch has no package Ollama may take, or it is paused or
  stopped. It uses the same functions as every other worker, never a side path.
- `claim_next` only offers an Ollama worker packages of task types whose `allowed_workers` include `ollama`.
  In v3 that is `metadata` only (ticket 14 open item 3). Notes, lenses, merges and bridge judgements stay with
  Claude or Codex. Concept embeddings (stage 4, tentative) are not a batch task; they are a direct Ollama
  client call.
- A model or connection failure releases the package (`give_back`) and stops the loop with the error. It
  never pauses the batch for other workers.
- `metadata` batches are the remedy for the "metadata needed" attention items from the migration (stage 1
  open question 2).

**What stays in `src/seedgraph/llm/`**
- `ollama.py`: the Ollama HTTP client, from `llm/providers/ollama.py` plus `_http.py`, reduced to
  `complete_json(model, system, user, schema) -> str` and `embed(model, texts)` (stage 4).
- `parse.py`: the JSON object extractor that `validate_note` re-exports.
- `tokens.py`: `estimate_tokens` (chars / 4), used for package estimates.

**Deleted** (with their tests and CLI groups): `llm/providers/{anthropic,openai,gemini}.py`, `backend.py`
(the multi-provider backend), `routing.py` (task routing, fallback hops and the content-access send-gate
`assert_external_content_allowed`), `cost.py`, `usage.py`, `executor.py` (structured-call seam with budget
preflight and usage logging), `profiles.py` (access profiles), `secrets.py` (key references and keyring
lookup), `llm_capabilities.yaml`, top-level `budget.py`, the `llm` and `secrets` CLI groups, and the
`keyring` dependency. Scholarly-API keys (Semantic Scholar, CORE, OpenAlex) resolved today through
`llm.secrets.resolve_named_secret` move to environment variables read in `providers/__init__.py` (open
question 4).

## 7. MCP server v3

### 7.1 Command and module

`seedgraph mcp serve` (`commands/mcp.py`, a thin adapter) starts a stdio FastMCP server named `seedgraph`. The
registration name is separate, §7.4. The v2 `src/seedgraph/mcp/` package is replaced in place: it is built on
`project.db` handles and has no v3 data to serve. v2 keeps running from the main checkout. Imports stay lazy
so a cold start is fast (target under 3 s; checked by a smoke test). `--redact-private` is removed
(ADR-0012).

### 7.2 Tools

The v2 server registers 21 tools (`version`, nine project/run/lens tools, five concept/search/graph tools,
`ask`, `work_read`, and four `extraction_*`). v3:

| tool | replaces | kind |
|---|---|---|
| `version` | `version` | read |
| `graph_list` | `project_list` | read |
| `graph_show(graph)` | `project_dashboard`, `corpus_status`, `list_documents` | read: membership counts, seeds, lenses, papers needing notes |
| `paper_list(graph?, status?, needs?, query?, offset, limit)` | `list_documents` | read |
| `paper_show(paper_id)` | — | read: metadata, files, current note summary, claims with anchor pages, attention items |
| `search(query, kind, graph?, paper_id?, section_kind?, limit)` | `search_spans` | read: stage 1 `library.search`, agent audience |
| `paper_read(paper_id, start?, end? \| section?)` | `work_read` | read: bounded markdown slices |
| `lens_results(lens_id, graph?, status?)` | `lens_results`, `lens_staleness` | read |
| `batch_create(task, paper_ids? \| query?, graph?, options?)` | `extraction_prepare` | mutation |
| `batch_status(batch_id, offset, limit)` / `batch_list(active_only)` | `extraction_status`, `run_list`, `run_events` | read |
| `work_claim(batch_id, paper_id?, worker?)` | `extraction_next` | mutation |
| `work_submit(packet_id, claim_id, note? \| note_path?, accept_unanchored?, model?, usage?)` | `extraction_submit` | mutation |
| `work_release(packet_id, claim_id, reason)` | `extraction_submit(failure=…)` | mutation |
| `batch_stop(batch_id, mode)` / `batch_continue(batch_id)` / `batch_report_limit(batch_id, resets_at?, message?)` | `resume=true` | mutation |

Every parameter that named a project now names a `graph` (slug). Dropped without replacement in stage 2:
`review_list` (stage 3 screens), `concepts_overview`, `concepts_list`, `concept_show`, `graph_analyze`
(stage 4 brings new concept tools), and `ask` (folded into the agent, ADR-0009; stage 5 adds the
record-answer tools). Mutation tools carry `readOnlyHint: false` so clients can ask for approval. Errors keep
v2's structured `{code, message, detail}` shape plus a plain-words `next_action`; a pause is a normal reply.

### 7.3 Skills

- `agent_setup/seedgraph_skill.md` (consultation) is rewritten for the library and graph tools: graph not
  project, `search` and `paper_read` instead of `search_spans` and `work_read`, and no `ask`.
- `agent_setup/seedgraph_work_skill.md` replaces `seedgraph_extract_skill.md`. It holds the task-agnostic loop:
  claim → read `instructions.md` once per batch → read the package file → write the note to `result_path` →
  `work_submit` with `note_path` (or inline `note` when it cannot write files) → on `needs_revision` copy the given slices and resubmit → claim again; on a
  client limit, call `batch_report_limit`; stop when the reply says paused or finished, and report counts. It
  also says how to fan out (§10.3). It never prepares a batch unless asked.
- **Build-time names avoid v2 collisions**: during stages 2–6 the skills install as `seedgraph-v3` and
  `seedgraph-v3-work` (Claude Code `~/.claude/skills/`, Codex `~/.codex/skills/`), leaving v2's
  `seedgraph` and `seedgraph-extract` untouched. Stage 7 renames them with the MCP entries.

### 7.4 Registration during the build (to confirm with the researcher at stage 2 start)

Default, as the spec assumes: v3 is added **alongside** v2 as `seedgraph_v3`; v2 keeps `seedgraph` until
stage 7. `seedgraph setup agents --client claude|codex|both` prints the exact change first (the command line
or TOML block, and the skill files) and writes only with `--write`, after the researcher has seen it. It
never edits other entries.

**Launch command.** The interim command `seedgraph-v3.exe` exists only in the worktree's venv (not on PATH).
`uv run` is slow and depends on the working directory. The registration therefore uses the absolute
executable:
- command: `<worktree>\.venv\Scripts\seedgraph-v3.exe`
- args: `["mcp", "serve"]`
- env: `SEEDGRAPH_HOME=<user home>\.seedgraph-v3` (explicit, so a stray environment can never point it at
  the v2 home), `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8` (ticket 04: Windows needs UTF-8 on stdio).

**Claude Code** (user scope):
`claude mcp add --scope user seedgraph_v3 --env SEEDGRAPH_HOME=… --env PYTHONUTF8=1 --env PYTHONIOENCODING=utf-8 -- <exe> mcp serve`

**Codex** (`~/.codex/config.toml`, the piece v2's `agent-setup --client codex` never wrote):
```toml
[mcp_servers.seedgraph_v3]
command = '<worktree>\.venv\Scripts\seedgraph-v3.exe'
args = ["mcp", "serve"]
startup_timeout_sec = 60          # Codex default is 10 s; a cold start can exceed it
[mcp_servers.seedgraph_v3.env]
SEEDGRAPH_HOME = '<user home>\.seedgraph-v3'
PYTHONUTF8 = "1"
PYTHONIOENCODING = "utf-8"
```
Setup writes this through `codex mcp add` where the flags allow it, and edits the TOML only for the timeout.
It shows both before writing. The worktree path is fragile if the worktree moves. Stage 7 re-points all
entries to `seedgraph mcp serve` on PATH (ticket 13 #9).

## 8. The copyable "work on this batch" command

`batch_create` returns, and `seedgraph work command <batch_id>` prints, two strings. Stage 3 shows them on
the batch page:
- **Claude Code**: `/seedgraph-v3-work <batch_id>`. The skill takes the batch id as its argument and has
  `disable-model-invocation: true`, so it runs only when typed.
- **Codex**: a prompt: `Use the seedgraph-v3-work skill and the seedgraph_v3 MCP tools to work Seedgraph
  batch <batch_id> until it is finished or paused. Write each result to its result_path and submit that
  file (MCP note_path, CLI --note-file); follow the feedback.`
  Codex skill invocation syntax varies by version; a plain prompt naming the skill works in all of them.

After stage 7 both drop the `-v3`/`_v3` suffixes. The strings are generated from one template in
`seedgraph/work/commands.py`, so the UI, the CLI and the batch reply never drift.

## 9. Deletions from v2 in this stage

The deletion rule is to delete in the stage that replaces the code. Removing `llm/` breaks importers that are
replaced later. Each case is decided here:

| v2 code | replaced by | importers of deleted modules | handling in stage 2 |
|---|---|---|---|
| `extraction/runner.py` (API whole-paper), `extraction/chunked_runner.py`, `extraction/reduce.py` (scalar merge), `extraction/chunker.py` | protocol + `notes` task | `extract` CLI group | delete, with the `extract` group and their tests; `save_normalized_note` and `_strip_references` are ported first (§3) |
| `extraction/agent.py`, `agent_cli.py` | protocol | `agent-extract` CLI, `mcp/tools_extraction.py` | delete after the port (the packet path is adapted, not discarded) |
| `lenses/runner.py`, `lenses/fallback.py` | `lens:<id>` task | `lens` CLI group | delete the runner and `results.py` (replaced by the adapter's library-backed results); keep `lenses/{schema,registry,prompt}.py` (definitions, validation, prompt) |
| `extraction/metadata.py` (LLM via executor) | `metadata` task | corpus backfill command | keep the deterministic id scan and `BiblioMeta`; drop the executor path |
| `llm/` hosted layer, `budget.py`, `llm_capabilities.yaml`, `keyring` | Ollama client + agents | `answer/compose.py`, `project/service.py`, `doctor.py`, `semantic/{__init__,llm_propose}.py`, `providers/__init__.py`, `work_read.py`, `web/*`, `mcp/tools_read.py`, `cli.py` groups `llm`/`secrets` | see the rows below |
| `answer/` (`ask` pipeline and ranker) | agent + stage 5 records | `answer` CLI, `ask` MCP tool | delete now (spec "deleted overall"; ADR-0009). Stage 1's `v2_answers/` export already preserves old answers |
| `semantic/llm_propose.py` (labels-only LLM merge proposer) | stage 4 agent proposals | `semantic/concepts.py` | delete now and keep `NullProposer`. Stage 4 deletes the rest; this is one stage early because its only dependency goes now |
| content-access send-gate (`assert_external_content_allowed`, `confirm_external`, `redact_private`, `is_shareable` gating in MCP and `work_read`) | nothing (ADR-0012) | `work_read.py`, `mcp/*`, `semantic/access.py`, `graph/export.py` | remove the gate; `access_class` stays as provenance; `graph.json` export keeps its own sharing rule until stage 4 replaces it |
| `doctor.py` LLM checks, `project/service.py` budget display | — | `doctor` CLI | trim to the Ollama reachability check; budget display goes |
| v2 MCP server `mcp/*` | v3 MCP (§7) | `mcp serve` | replaced in place |
| `agent_setup` extraction skill | work skill | `agent-setup` CLI | replaced by `seedgraph setup agents` |
| `resume` flag, one-pending-packet `next`, three delivery forms, note stored twice | protocol | — | gone with `extraction/agent.py` |
| `eval/` harness | nothing | `eval` CLI | delete now: it depends on the API runners (spec "deleted overall") |

**The old web UI (`web/`, `api/`, the `serve` command).** It imports `llm.routing`, `llm.cost`,
`llm.profiles`, `llm.secrets` and `llm.executor` (planner, routes, ui), and the extraction runners. Stage 1 §9
already established that it is not pointed at the v3 home, so it serves no v3 data. Keeping it alive would
mean rewriting its LLM settings and planner pages only to delete them in stage 3. **Decision: delete `web/`,
`api/` and the old `serve` command in stage 2.** This deviates from the deletion rule, because the stage 2
deletions remove what the UI is built on. The stage 2 working app is the CLI plus the MCP server. Stage 3
adds the new `seedgraph serve`, and the v2 UI keeps running from the main checkout on v2 data until stage 7.
The `prototype/v3-ui` branch is unaffected. This is open question 2.

Other old groups (`corpus`, `cite`, `sections`, `spans`, `concepts`, `graph`, `review`, `project`, `cache`)
stay, trimmed of LLM imports, until their replacement stage.

## 10. Acceptance plan

### 10.1 What is run

All on the real v3 home, after a fresh `migrate accept`. The papers are referred to by golden-set id; their
library paper ids are found through `golden_set.json`'s PDF hash → `source_files` → `paper_files`.

1. **Gate**: `seedgraph library textcheck --golden` assesses every golden-set paper's current markdown.
   Pass = verdict equals category (GS06 and GS07 `garbled`, the other ten `pass`).
2. **Re-note the three live-run papers** (GS06, GS09, GS10) in one `notes` batch with `replace`:
   - GS06 is held back by the gate at creation (this is part of the check).
   - `library reconvert --ocr` on GS06; it must pass the gate on the new markdown. The quote re-match report
     is recorded, and the expected `renote_offered` item appears if matching drops.
   - A second batch notes GS06 on the OCR markdown. If forced OCR does not cure it, GS06 is noted by
     researcher override, and the report says so.
3. **Tokens**, measured the way ticket 04 did: one subagent per paper, reading the subagent's reported total
   tokens at its end, from the orchestrating session, so the comparison with the ticket 04 per-chunk and
   per-paper baseline is like for like. Workers also pass `usage` to `work_submit` when their client
   exposes it; those figures are recorded beside the totals. Report per paper: tokens, parts, wall time. If
   whole-paper packages do not save tokens, the knob is `split_threshold_tokens`, not the protocol
   (ADR-0010).
4. **Anchoring** at least matches the live run, where every found quote anchored. The figures sit in the
   acceptance report in the home, not here. Report
   `quotes_anchored / quotes_found` per paper after the last accepted submission, plus every failed quote seen
   during revision, with its reason. Also report how many submissions needed a revision.
5. **Parallel agents, no collisions**: a batch of at least six papers (the three above plus three needing
   notes; GS08 recommended, to exercise the split) worked by two or more workers at once. Pass:
   - no package is claimed by two live claims (checked from `package_claims`);
   - every paper has exactly one new version;
   - `batch_events` shows interleaving.
6. **Claim expiry**: a batch with `claim_ttl_s = 120`. A worker claims and stops; after expiry the batch
   shows "claim expired", another worker claims the same package, and the late submit from the first worker
   is refused.
7. **Codex interactive**: at least one paper noted from a Codex session over `seedgraph_v3` using the
   copyable prompt.
8. **Pause states**: each of the four shown correctly in `batch_status` and continued: `stop finish`,
   `stop now`, a `report_limit` with a reset time and one without, an expired claim, and an idle batch.
   Continue works across agents (one agent stops, a different agent continues).
9. **Ollama**: a `metadata` batch over a few "metadata needed" papers, worked by `seedgraph work ollama`. A
   notes package is never offered to it.

### 10.2 Who runs the workers

Background agents arrive in stage 6, so stage 2 acceptance uses **interactive sessions**: the researcher's
Claude Code session acting as orchestrator, and a Codex session for item 7.

**Can subagents spawned by the orchestrating session be workers?** Yes: ticket 04 ran each paper in a
subagent over MCP. But `seedgraph_v3` may not be registered when the build first needs a real run, and
subagents share their parent's MCP server process (harmless for correctness thanks to `claim_id`, but
reporting needs the `worker` name).

### 10.3 Recommendation: a CLI mirror of the protocol tools

`seedgraph work …` mirrors the MCP protocol tools one to one, so any shell-capable agent (a Claude Code
subagent, Codex, a script) can work a batch with no MCP registration:

```
seedgraph-v3 work create notes --paper ID … | --query "held in graph diff_in_diff without current note" [--replace] [--ttl 1800]
seedgraph-v3 work claim  BATCH --worker NAME [--client claude] [--paper ID]
seedgraph-v3 work submit PACKET --claim CLAIM --note-file PATH [--accept-unanchored] [--model M] [--usage-file PATH]
seedgraph-v3 work release PACKET --claim CLAIM --reason give_back|paper_problem
seedgraph-v3 work status BATCH | list | stop BATCH --mode finish|now | continue BATCH | limit BATCH [--resets-at T]
seedgraph-v3 work ollama BATCH | command BATCH | prune --finished-before DATE
```

- One command per call, inputs as arguments or file paths, JSON on stdout. The worktree guard refuses
  compound or env-prefixed shell commands (ticket 04), so nothing needs `VAR=… cmd` or pipes.
- UTF-8 is forced inside the program (`sys.stdout.reconfigure`), so `PYTHONIOENCODING` is not needed.
- `--worker NAME` gives each subagent its own name in reports; `claim` prints the `claim_id` that `submit`
  and `release` need.
- `work` is part of the CLI core (setup, migrate, serve, mcp serve, convert, maintenance, and now work), so
  stage 3's CLI trim keeps it (open question 9).
- The orchestrator creates the batch, then spawns N subagents, each told: "run `seedgraph-v3 work claim
  <batch> --worker w<i>`, read the files it names, write the note to a file, run `work submit`, repeat until
  paused or finished". The same works over MCP once `seedgraph_v3` is registered, which item 7 needs anyway.

## 11. Build slices

Each slice is sized for one agent, written test-first, and owns its files exclusively. Shared files
(`cli.py`, `commands/__init__.py`, `pyproject.toml`, `.gitignore`) are touched only by the slice named for
them. Tests use synthetic fixtures with invented text, built with stage 1's library fixture helpers.

**New beside old, one swap.** P1–P7 only add new modules next to the v2 code; they edit no v2 module that
v2 code or tests still import. P8 is the single slice that re-points `cli.py`, deletes the §9 list and moves
the new modules into their final names. The full suite stays green after every slice.

| # | Slice | Depends on | Owns | Tests first |
|---|---|---|---|---|
| P1 | **Protocol schema and state core**: `0008_work_protocol.sql`; `seedgraph/work/{__init__,store,state,claims}.py` (batches, items, packages, package claims, lazy expiry, pause model, events) | — | `db/schema/library/0008_work_protocol.sql`, `src/seedgraph/work/{__init__,store,state,claims}.py`, `tests/work/test_state*.py`, `test_claims*.py` | two claimers never get one package; expiry returns a package; late submit refused after re-claim; four pause states and precedence; continue clears; stop finish/now |
| P2 | **Notes boundary on the library**: `seedgraph/notes/{save,spans,vocab,format}.py` (port of `save_normalized_note` and `ensure_span`; schema 2.0.0 envelopes, validator, normalizer and instructions as new modules; vocab tables; plain words). `extraction/*` stays untouched for v2 code until P8 | — | `src/seedgraph/notes/`, `tests/notes/` | version_no max+1 under conflict; anchors keep the invariant; unanchored → ambiguous; vocab mapping and `other`; plain words required; `epistemic_type` derived and ignored if sent |
| P3 | **Anchor feedback and the text-quality gate (pure)**: `seedgraph/work/feedback.py` (reasons table §1.5), `library/mojibake.py` (shared table), `library/textquality.py`, a calibration script writing to the v3 home; titlecheck untouched | — | those files, `tests/work/test_feedback*.py`, `tests/library/test_textquality*.py` | each failure reason incl. µ/μ and curly quotes; garbled family A and B fixtures flagged; accented-name fixture passes; page rule; no_text |
| P4 | **Task types and the routine**: `work/{tasks,routine,packages,commands}.py`; `notes`, `metadata`, `lens` adapters; reference stripping and the split; instructions and labels files; assembly of parts; gate and title-miss hold-back | P1, P2, P3 | `src/seedgraph/work/{tasks,routine,packages,commands}.py`, `src/seedgraph/work/tasks_*.py` (the lens adapter reads `lenses/{schema,registry,prompt}.py` without editing them and has its own library-backed results), `tests/work/test_routine*.py` | whole paper = one package; >threshold splits at sections; scalars only from part 1; note current only when all parts saved; needs_revision keeps the claim; stale and conflict items; held-back listed; lens port keeps the v2 record rules |
| P5 | **Re-conversion and re-matching**: `library reconvert --ocr` path in `library/files.py` callers, `library/rematch.py`, gate hook after conversion, `renote_offered` items | P3, P2 | `src/seedgraph/library/{rematch,reconvert}.py`, `src/seedgraph/library/files.py` (gate hook after conversion), `tests/library/test_rematch*.py` | folded old quotes re-match clean text; full re-match → rematch version; a drop → offer, old version stays current; a failed OCR keeps the old markdown |
| P6 | **Ollama worker**: new `llm/ollama.py` client beside the old layer; worker loop (deleting the hosted layer and moving scholarly keys to env vars happen in P8) | P4 | `src/seedgraph/llm/ollama.py`, `src/seedgraph/work/ollama_worker.py`, `tests/llm/` | Ollama offered only metadata packages; failure releases, never pauses; fake Ollama server fixture |
| P7 | **MCP server v3, skills and setup**: new modules in `mcp/` (`server_v3.py`, `tools_library.py`, `tools_work.py`) and `agent_setup/` (v3 skill files, `register.py` for `setup agents`, show then `--write`) beside the v2 ones | P4 | those new files, `tests/mcp/`, `tests/agent_setup/` | in-memory MCP transport per tool; `graph` parameters; cold-start smoke (opt-in `mcp_client` marker); setup renders the exact Claude/Codex change and writes nothing without `--write` |
| P8 | **Swap, deletions and wiring**: `commands/work.py`, `commands/mcp.py`, `library textcheck/reconvert` subcommands, `cli.py` registration; the §9 deletions with their CLI groups and tests; the hosted `llm/` layer and keyring; scholarly keys to env vars; new modules moved to final names | P5, P6, P7 | `src/seedgraph/commands/{work,mcp}.py`, `commands/library.py` (new subcommands), `cli.py`, `pyproject.toml`, `providers/__init__.py`, every deleted module and test | smoke per command; JSON stdout; UTF-8 forced; the full suite green after the deletions |

**Parallelism.** P1, P2 and P3 run in parallel; their files are disjoint. Then P4, alone, which is the
critical join. Then P5, P6 and P7 in parallel. Then P8. The critical path is P1/P2 → P4 → P7 → P8.

**Real acceptance run** (gate 2), after P8: the researcher confirms the `seedgraph_v3` registration (§7.4),
then the §10.1 steps run in order. Reports go to `<home>/work/acceptance/<run>/report.md`: gate table, token
table, anchoring table, collision check, pause-state screenshots or JSON, and the Codex note id. The v3
library is dirtied by design. The pre-acceptance `migrate accept` dump is the reference if a clean migration
must be rebuilt.

## 12. Open questions for the researcher

Each has a recommendation (*R*).

1. **Gate vs golden set**: GS06/GS07 are in the golden set and garbled. *R:* the golden set keeps measuring
   conversion fidelity; gate acceptance is "verdict equals the golden-set category" (GS06/GS07 held back, the
   other ten pass).
2. **Old web UI in stage 2**: it depends on the deleted LLM layer and serves no v3 data. *R:* delete `web/`,
   `api/` and the old `serve` in stage 2 (a recorded deviation from the deletion rule). The CLI and MCP are
   the stage 2 app, and the v2 UI keeps running from the main checkout.
3. **MCP registration during the build**. *R:* the spec default: add `seedgraph_v3` alongside v2 in Claude
   Code (user scope) and Codex, with the absolute worktree executable, explicit `SEEDGRAPH_HOME`, UTF-8 env
   and a 60 s Codex startup timeout, shown before writing. Skills install as `seedgraph-v3` and
   `seedgraph-v3-work` until stage 7.
4. **Scholarly-API keys** (Semantic Scholar, CORE, OpenAlex) currently go through the keyring code being
   deleted. *R:* environment variables only. ADR-0012's "no stored keys" then holds literally, and the
   `keyring` dependency goes.
5. **Plain-words statements required, or optional?** *R:* required for found assumption, method and
   estimand claims. A missing one costs one resubmit, and stage 4's bridges depend on it.
6. **Re-noting dirties the library**, so a later `migrate run` needs `--replace-dirty`. *R:* accept it.
   Stage 2 acceptance is the first real library work, and the pre-acceptance dump stays as the reference.
7. **Failed quotes after revision**: allow `accept_unanchored` (downgrade to ambiguous, as v2), or refuse?
   *R:* allow, but report it per batch. Some math-heavy quotes cannot be made unique, and refusing would
   strand papers.
8. **If forced OCR does not cure GS06**: *R:* note it by override for acceptance, record the result, and do
   not block stage 2. Pursuing another converter belongs to ADR-0014's upgrade path.
9. **`work` in the CLI core**: ticket 14's core list predates the protocol. *R:* add `work` to the core,
   because it is the shell route for any agent and for the Ollama worker.
10. **Claim TTL**: *R:* 30 minutes by default, set per batch. Whole-paper packages take much longer than v2's
    chunks; if real runs show expiries on healthy agents, raise the default rather than adding heartbeats.
11. **The `*_type` vocabularies** (§4.1). *R:* adopt as drafted, with `other` and the variant table. Review
    the `other` share after the first real batches and extend only with evidence.
