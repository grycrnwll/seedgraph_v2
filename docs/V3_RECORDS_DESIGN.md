# v3 Stage 5 design: records and spot-check

Status: design for build. No code yet. This document covers v3 stage 5:
- traversal records: the session log, how a session is told apart, the record-answer tool and its citation
  check, and "path without answer";
- where records are stored, and how one record belongs to several context graphs;
- the spot-check: the sampling rule, the marks, "fix it now", and accuracy per claim type and per model;
- the Records and Spot-check screens, and records in the Home activity feed;
- MCP and skill updates, deletions, the acceptance plan, build slices and open questions.

It uses the glossary terms from `CONTEXT.md` (library, context graph, traversal record, spot-check, current
interpretation, researcher-edited note, researcher assertion, worker, interactive session) and the deep-module
vocabulary from `codebase-design` (module, interface, depth, seam, adapter). `docs/V3_LIBRARY_DESIGN.md`
(stage 1), `docs/V3_AGENT_PROTOCOL_DESIGN.md` (stage 2), `docs/V3_UI_DESIGN.md` (stage 3) and
`docs/V3_CONCEPTS_DESIGN.md` (stage 4) are binding for storage, the MCP server, the web stack and claims.

The repo is public. Real questions, answers, records, model names with their counts, spot-check marks and
acceptance reports live in the v3 home, never in this document or the repo. Examples use textbook terms only
(ATE, parallel trends, serial correlation).

**Two meanings of "session".** SPEC calls a spot-check run a "session of about 20 claims". This document calls
that a **spot-check sample**, and keeps **session** for an agent's line of work over MCP.

## 0. Decided before this stage

From ticket 07 (all resolutions binding), ticket 08 (screens), ticket 09 #8, ticket 14 open item 5, ADR-0009:
- A recorded answer owns every search and read since the previous recorded answer **in the same session**.
  There is no "start record" call. This rule is applied literally: a long pause does not split ownership.
- The record-answer tool takes the question, the answer, the cited evidence and the agent's judgement, kept
  apart. Each citation is checked against the session log. Evidence not read is flagged "cited without being
  read".
- A record shows each search (query, results), each paper or section opened, and each item tagged *found but
  not opened*, *opened but not cited* or *cited*. v2's dispositions (`shown`, `cut_rank`, `cut_budget`) are
  retired.
- The skill tells the agent to record every corpus-based answer. Every search result carries a one-line
  reminder. An unanswered path shows as "path without answer".
- A record belongs to the graph(s) it searched. One spanning two graphs appears in both, marked.
- Records are kept indefinitely: star, annotate, delete, export to markdown.
- Only corpus-backed answers produce records. The UI never claims to show every question asked.
- Spot-check: on-demand samples of about 20 claims, spread across claim type and model. Marks are right, partly
  right, wrong. "Fix it now" opens the edit. Accuracy counts the agent's original claim, not the fix. Accuracy
  per type and per model is stored in the library and always shown with its count.
- Stage 3 §4.3 promised that past records store the claim text they used and the `(overlay_id, version_no)`, so
  later edits never rewrite them (ticket 09 #6).
- **v3 data is disposable until stage 7** (stage 3 §0). Records and spot-check marks written before the
  switch-over are test data, like v3 edits. See open question 9.

## 1. What the code says (findings that shape this design)

### 1.1 How `mcp serve` runs

- The v3 server is a FastMCP server over **stdio** (`mcp/server.py:186-194`, `build_server(...).run("stdio")`).
  Each client that registers it starts its own server process. One process serves one connection.
- `V3Context` mints one `worker_id` per process (`mcp/server.py:117-121`). It is the natural connection key.
- The client name comes from the MCP `initialize` handshake (`client_name`, `mcp/server.py:136-148`): `claude`,
  `codex`, or the client's own name.
- Tools are registered in three modules: `tools_library.py` (`graph_list`, `graph_show`, `paper_list`,
  `paper_show`, `search`, `paper_read`, `lens_results`, lines 354-416), `tools_concepts.py` (`concept_search`,
  `concept_show`, `literatures`, `concept_hubs`, and the hubs resource, lines 506-552) and `tools_work.py`
  (the protocol). The read tools take **no MCP `Context`** today. Only the work tools do, through the
  `_worker(name, mcp_ctx)` pattern (`tools_work.py:29-32`).
- The brief's `search_spans` and `work_read` are v2 tool names. In v3 they are `search` and `paper_read`.

### 1.2 Subagents, parallel agents, Codex, restarts

- **Claude Code subagents share their parent's MCP server process.** Stage 2 recorded this from the ticket 04
  live run (`V3_AGENT_PROTOCOL_DESIGN.md` §1.4 and §10.2: "subagents share their parent's MCP server process").
  That is why `work_claim` takes an optional `worker` name (`tools_work.py:82-91`). This design could not
  re-verify it first-hand (listing the processes was refused in the design session), so the acceptance plan
  includes a live identity probe (§10, item 5). If subagents share a connection, "everything since the last
  recorded answer in this connection" would let one subagent's answer own another subagent's reads.
- **Parallel interactive agents** (two Claude Code windows, or Claude Code and Codex) each start their own
  server process. They are already told apart by connection.
- **Codex interactive** (stage 2 acceptance item 7) also starts its own process and reports `codex` in the
  handshake. Whatever is designed must work with no Codex-specific feature.
- **Reconnects and restarts** (`/mcp` reconnect, a crashed server, a new Claude Code window) start a new
  process. A connection-keyed session therefore splits at every restart. Reads before the restart would be
  "path without answer", and an answer after it would flag those reads as "cited without being read".
- The CLI mirror (`seedgraph-v3 library search`, `commands/library.py:100`) is not an MCP call and is not
  logged. The consultation skill already sends agents to the MCP tools.

### 1.3 Which tool returns what text

This decides what "read" means (§3.3):
- `search(kind=spans)` returns each hit's full `exact_quote` (`library/search.py:229-260`). `kind=claims` returns
  claim text; `kind=notes` returns note hits.
- `paper_show` returns metadata, the section outline and the current claims with their text, plus each claim's
  anchor **span ids and pages, but no quote text** (`tools_library.py:208-222`).
- `paper_read` returns a bounded slice of the paper's markdown by character range or section.
- `concept_show` returns supporting claims with their quotes; `concept_search` returns one example claim with
  its quote per hit (both through `_claim_rows`, `tools_concepts.py:124-134`).

### 1.4 Model per claim, and the original beneath an overlay

- Model is recorded per **extraction run**, not per claim: `extraction_runs.model_name`, `provider`,
  `access_mode` (`db/schema/library/0004_notes.sql:14-42`). Each `extracted_claims` row carries its
  `extraction_run_id` (line 84).
- **Migrated v2 runs carry a model name.** Some of them carry a client label instead of a model (a v2 mailbox
  run recorded the client that did the work).
- **v3 agent runs so far carry none.** `notes/save.py:175-182` writes `provenance.model`, which comes from
  `work_submit(model=...)` (`work/routine.py:446`). The work skill says "pass `model` if your client shows it"
  (`seedgraph_v3_work_skill.md:33`), and the agents did not. So the "unknown model" stratum is mainly for
  v3-native notes, not migrated ones. The fix-forward is a skill wording change (§8.2).
- **The original claim is always readable.** An overlay never edits `extracted_claims`. For an agent claim,
  `interpretation_claims.claim_id` is the base claim id (`0009_note_overlays.sql:47-66`), so the agent's
  original type, label and text are one join away. Added claims have `rc_` ids and are researcher claims.
  The paper page already shows "agent's original" (`web/templates/paper/_notes.html:72-77`).
- `current_claims` excludes hidden claims (`0009_note_overlays.sql:114-137`). The spot-check must not sample
  from it, or claims the researcher hid would drop out of the accuracy figure (§5.1).

### 1.5 What v2 code is left to delete

Stage 2 slice P8 (`a27873b`) already deleted the `ask` pipeline and ranker (`answer/`), `AnswerTrace` and its
dispositions (`answer/trace.py`, the trace views and tests) and the eval harness (`eval/`), with their CLI
groups and MCP tools (stage 2 §9). The v3 server registers no `ask` tool. The `ask` tool visible in a Claude
Code session today belongs to the **v2** `seedgraph` server from the `v2-final` checkout; stage 7 removes that
registration. What remains is listed in §9.

## 2. Storage: records live in the library

### 2.1 The conflict

Stage 1 (§3) reserved `traversal_records` and `traversal_events` in each graph's `graph.db`. Ticket 05 and
`CONTEXT.md` say a context graph owns its traversal records. Ticket 07 #5 says a record that spans two graphs
appears in both, marked.

### 2.2 Decision: one record in `library.db`, linked to its graphs

Records, events and sessions are stored once in `library.db`. A link table `record_graphs` says which graphs a
record belongs to. A graph "owns" its records through that link: the graph's Records tab lists them, and a
record's graphs are shown on it.

Why not per-graph copies:
- **Events are written before any graph is known.** Most read calls name no graph: `paper_show`,
  `paper_read`, and any library-wide `search`. A per-graph log would have nowhere to put them.
- **One record, one row.** Star, annotate and delete act once. Two copies would need syncing, and "appears in
  both, marked" is a query over the link table.
- **Fewer writers.** Every MCP call already opens `library.db`. Writing into `graph.db` would add a second
  database (and a second lock) to every read call, for every graph searched.
- **Kept indefinitely.** Records outlive graph renames and deletions. A deleted graph's records stay, marked
  "graph deleted".

This amends stage 1 §3, ticket 05 and the `CONTEXT.md` wording, as ADR-0013 did for concepts. The
graph-schema comment that reserves the two tables (`db/schema/graph/0001_graph.sql:6`) is in an applied
migration and is left as it is. `CONTEXT.md` is not in the worktree, so the glossary edit goes to the
orchestrator (§9.3).

## 3. Traversal records

### 3.1 Session identity

A **session** is one stream of logged calls. Its key is chosen in this order:
1. **Named**: the call passes `session`, a short name the agent chose. The stream key is `(client, name)`. It
   is scoped to the library, not to the connection, so it survives reconnects and server restarts.
2. **Connection** (default): no `session`. The stream key is the connection: the process's `worker_id`. On
   stdio this is one process, so it is also one client window. The session is **named by its start date and
   time** (`2026-11-04 14:32`, local time; a second one in the same minute gets `-2`), and that is how it is
   listed and indexed within each context graph (researcher, 2026-11-04). The name is a label only: two
   windows started in the same minute stay apart by connection.

Name rules: `[a-z0-9][a-z0-9-]{2,47}`, optionally followed by `/` and a helper suffix of the same form (one
level). An invalid name is refused with a plain `next_action`.

**Helpers (subagents).** A name with a suffix, `parallel-trends-1432/a`, is a **helper stream** of the parent
`parallel-trends-1432`:
- A helper stream opens no record of its own. Its events go into the **parent's** open record (opening one if
  there is none). So the parent's answer owns its own and its helpers' calls since its previous answer. The
  path shows which helper made each call.
- A helper that calls `record_answer` itself gets its own record. Its own events since its previous answer
  move from the parent's open record into the new one. This is the only time an event changes record.
- The citation check (§3.3) looks at the parent's whole family log, so a passage a helper opened counts as
  read for the parent's answer. It is tagged "opened by helper a".

**What the skill tells the agent** (§8.1): pick a name at the start of each question thread (a topic slug plus
the clock time, `parallel-trends-1432`), and pass it on every seedgraph call for that question and its
follow-ups. When you start subagents to search for you, give each a helper name `<yours>/a`, `<yours>/b` and
tell it to pass that. A subagent that was given no name picks its own.

**Hard cases:**

| case | named | fallback (no name) |
|---|---|---|
| one interactive agent | one stream per question thread | one stream per window: works, one thread at a time |
| subagents sharing the parent's connection | parent and helpers apart; helpers' reads count for the parent | all in one stream: a helper pattern still works; **independent** subagents answering separate questions swallow each other's reads |
| two windows, or Claude Code and Codex | apart by client and name | apart by connection |
| the same name in two windows | a warning on the record: "session name used from two connections" | — |
| reconnect or server restart | the stream continues | the stream splits; earlier reads become a path without answer |
| Codex interactive | works (no Codex-specific feature) | works |

Every event stores the connection id, client and name, so a mixed stream can always be inspected. A record
whose stream is a connection fallback says "session told apart by connection only". An identity probe (R1)
also logs the **keys** (never the values) of each request's MCP `_meta`, so the acceptance run can see whether
Claude Code marks subagent calls at all. If it does, that can become a third key later (open question 1).

### 3.2 The session log

Logged tools (all read-only for the researcher's data; they keep `readOnlyHint: true`):

| tool | event kind | what is stored |
|---|---|---|
| `search` | search | query, kind, graph, paper, section kind; hits in rank order: kind, id, paper id, a 160-character preview |
| `paper_list` with `query` | search | query, graph; paper ids in order |
| `concept_search` | search | query, family, graph, mode; sense handles and the example claim ids |
| `literatures`, `concept_hubs` | search | graph, family; concept handles |
| `paper_show` | open paper | paper id; the claim ids returned (with their anchor span ids) |
| `paper_read` | open text | paper id, markdown id, the returned character range, section id |
| `concept_show` | open concept | handle, graph; the claim ids and span ids whose quotes were returned |
| `lens_results` | search | lens id, graph; paper ids |

Not logged: `graph_list`, `graph_show`, `version` (orientation, no evidence), and every work-protocol tool. A
worker noting papers creates no path.

Rules:
- Logging happens after the tool body succeeds, in the same library handle, as a short write
  (`busy_timeout` as stage 1). A failed write never fails the read. It is counted in memory and written with
  the next successful event as "n calls could not be logged". The record shows that count.
- Paper text is never stored in events, only ids, ranges and short previews. Cited text is frozen in the record
  (§3.4).
- The audience is the agent's. Events hold what the agent was shown, so hidden notes stay out.

### 3.3 "Read", and the citation check

An item can be a **paper**, a **span** or a **claim**. Its tag in a record comes from the strongest thing that
happened to it in the session:
- **found**: it appeared in a search-kind reply.
- **opened**: an open-kind reply returned its text:
  - a span is opened when a `paper_read` slice of the same markdown covers at least half of it, or when
    `concept_show` returned its quote;
  - a claim is opened when `paper_show` or `concept_show` returned it, or when its anchor span is opened;
  - a paper is opened when `paper_show` or `paper_read` was called on it.
- **cited**: it is in the answer's evidence.

A span id seen only as an anchor in `paper_show` was not shown as text, so it counts as nothing.

**Citation check** (at record time, against the session family's whole log up to this answer, not only the
record's own events):
- cited and opened → fine; tagged "opened in this record" or "opened earlier in this session".
- cited and found only → flagged **"cited without being read (seen in search results only)"**. SPEC's check is
  "a span the agent never opened"; a search hit is "found", not "opened" (ticket 07 #3). The skill already says
  to read the passage before relying on it.
- cited and never returned → flagged **"cited without being read"**.
- an id that does not exist in the library → the call is refused with the unknown ids, so a typo is fixed
  rather than stored.

The check result is frozen with the record. Deleting an earlier record later does not change it.

### 3.4 The record-answer tool

```
record_answer(question, answer, evidence, judgement?, session?)
  evidence: [{"span_id": ...} | {"claim_id": ...} | {"paper_id": ..., "page"?: int}, "why"?: str]
```

- `question`: the researcher's question as the agent understood it (≤ 2 000 characters).
- `answer`: what the corpus says, as given to the researcher (≤ 20 000).
- `judgement`: the agent's own view, kept apart and labelled "agent's judgement" everywhere (≤ 5 000).
- `evidence`: 1 to 100 items. Span or paper ids, as SPEC says, plus **claim ids**, because agents cite claims
  from `search(kind=claims)` and `paper_show`. A claim resolves to its paper and anchor span.
- Reply: `record_id`, the counts (searches, opened, cited), the flags, the page path `/records/<record_id>`
  (the MCP process does not know the web server's port), and a `next_action`
  when something is flagged ("tell the researcher which citations were not read, or read them and record
  again").

On save, in one transaction:
1. resolve the stream (§3.1) and its open record;
2. run the citation check (§3.3);
3. **freeze the evidence**: for a span, the quote, page and paper; for a claim, its type, label and text as
   the agent saw it (current interpretation), its trust label and `(overlay_id, version_no)` when the paper is
   researcher-edited; for a paper, title, year and the page given;
4. write question, answer, judgement and evidence; set the record `answered`;
5. link the record's graphs (§3.6).

A record that owns no events is allowed. It happens when the agent answers a follow-up from what it already
read. Its citations are checked against the session log, so they pass.

The agent may record again after reading what it missed. The new record owns the new reads. The first stays,
flagged. Nothing is ever amended in place.

### 3.5 Open records and "path without answer"

The first logged event of a stream with no open record **opens** a record (`state = open`). `record_answer`
answers it. The next event opens the next one. So ownership is exactly ticket 07 #1, with no claiming step.
Helper streams use their parent's open record (§3.1).

An open record is shown as:
- **in progress** while its last event is less than `PATH_IDLE` (30 minutes) old and its connection is live;
- **path without answer** once it is idle that long, or its connection closed (the stdio server marks its
  connection closed when stdin ends).

This is display only. If the agent does answer later in the same session, the same record becomes answered.
A long gap inside a record is shown in the path ("40 minutes later").

### 3.6 Graphs of a record

- The graphs named as `graph` in the record's events, how `searched`.
- If none was named (library-wide searches only), the graphs that hold a cited paper as a member, how
  `cited_member`, shown as "library-wide search; cites papers in this graph".
- A record with two or more graphs is marked "spans n graphs: A, B" in every list.
- A record with no graph appears under **Library-wide** in the Records list.

### 3.7 Schema: `db/schema/library/0011_records.sql`

| table | key columns | notes |
|---|---|---|
| `record_sessions` | `session_id` ('ss_' + hash of key), `key_kind` (`named`, `connection`), `client`, `name`, `parent_session_id` NULL, `first_at`, `last_at` | one row per stream; a helper points to its parent |
| `record_connections` | `connection_id` (the process `worker_id`), `client`, `started_at`, `closed_at` NULL, `meta_keys_json` | `closed_at` set when stdin ends; `meta_keys_json` is the identity probe |
| `traversal_records` | `record_id` ('tr_' + hex), `session_id`, `state` (`open`, `answered`), `opened_at`, `last_event_at`, `answered_at`, `question`, `answer`, `judgement`, `evidence_json` (frozen, §3.4), `flags_json`, `counts_json`, `unlogged_calls`, `starred` (0/1), `annotation`, `annotated_at`, `record_version` (1), `seedgraph_version` | `record_version` lets readers tolerate later fields (the v2 trace's additive rule, kept) |
| `traversal_events` | `event_id`, `record_id`, `session_id`, `seq`, `at`, `tool`, `kind` (`search`, `open_paper`, `open_text`, `open_concept`), `connection_id`, `args_json`, `items_json` | content immutable; `record_id` changes at most once, when a helper records its own answer (§3.1); `items_json` holds ids, ranges and previews only |
| `record_graphs` | PK(`record_id`, `slug`); `how` (`searched`, `cited_member`) | ticket 07 #5 |

The item tags are **derived**, never stored: `records.path(events, evidence)` is a pure function, so the
record page and the export can never disagree. Deleting a record deletes its events and links.

### 3.8 Export to markdown

`records.export_markdown(lib, record_id) -> str`, offered as a download on the record page:
- question; answer; "Agent's judgement" as its own section;
- cited passages: each frozen quote or claim text, with paper title, year and page, its trust label, and any
  flag;
- the path: each search with its query and the hits (tagged), each paper or section opened, in order;
- the record's graphs, client, session name, dates, and the researcher's annotation.

No PDF or full paper text goes into an export.

## 4. Records module (`src/seedgraph/records/`)

| module | interface | hides |
|---|---|---|
| `log.py` | `log_event(lib, call: CallInfo, tool, kind, args, items) -> None`, `close_connection(lib, connection_id)` | stream resolution, opening records, the unlogged-call count, busy handling |
| `sessions.py` | `resolve(lib, call: CallInfo) -> Stream`, `family(lib, session_id) -> list[str]` | the key order, name validation, helpers, the two-connection warning |
| `answer.py` | `record_answer(lib, call: CallInfo, question, answer, evidence, judgement) -> RecordResult` | id resolution, the citation check, evidence freezing, graph links |
| `path.py` | `path(events, evidence) -> Path` (pure), `state(record, now, connection) -> str` | the found / opened / cited rules, "path without answer" |
| `export.py` | `export_markdown(lib, record_id) -> str` | layout |
| `manage.py` | `star`, `annotate`, `delete`, `list_records(lib, filters)` | the list query |

`CallInfo` is `(connection_id, client, session_name, meta_keys)`, built once per call by the MCP adapter.

## 5. Spot-check

### 5.1 Population

Every agent claim of each paper's **current note**: `extracted_claims` whose `structured_note_id` is the
paper's `note_current.note_id`. For a researcher-edited paper that is the overlay's base note. So:
- **edited claims are included**, judged on the agent's original wording;
- **hidden claims are included**, shown with "you hid this claim". Leaving them out would drop the claims the
  researcher already found wanting, and push accuracy up;
- **researcher-added claims are excluded** (not the agent's);
- **lens claims are excluded** in stage 5 (`structured_note_id` NULL);
- claims already marked in an earlier sample are excluded, so each claim is marked once.

Each claim has a **claim type** (the original `claim_type`) and a **model key**:
- `model_name` as recorded on its extraction run;
- else `"<provider>, model not recorded"` (for example a v3 Claude Code run saved without `model`);
- else `"model not recorded"`.

A recorded value that names a client rather than a model is kept as recorded (open question 7).

### 5.2 The sampling rule

Inputs: size `n` (default 20), a seed (drawn and stored), and optional filters (graph, claim type, model).
1. **Cells** are `(claim type, model key)` pairs with at least one unmarked claim.
2. **Draw one claim at a time.** Pick the cell with the **fewest marks so far** (all earlier samples plus this
   one's draws). Ties go to the model with fewer marks, then the type with fewer marks, then a seeded random
   choice.
3. Within the cell, pick a seeded random unmarked claim, skipping papers that already have **2 claims** in
   this sample.
4. Stop at `n`, or when no cell can give a claim.

This spreads a sample across claim types and models, and over several samples it fills the thinnest cells
first. That is where a per-type or per-model figure is least certain. It is deterministic: the same library
state and seed give the same sample. It reuses the stage 4 review-sample pattern
(`concepts/researcher.py:723-760`: sorted population, `random.Random(seed)`, a sample id hashed from seed, size
and ids), but stores the sample in a table instead of `library_meta`, because items carry marks.

Per-cell sample shares are not proportional to the library's mix, so the figures are reported as follows.

### 5.3 Marks

Each item shows:
- the paper (title, year, PDF button);
- the agent's original claim: type, label, text, plain words;
- its quote, with the page, and "show in PDF" (the stage 3 highlight);
- "you edited this" (with the current version) or "you hid this claim" when that applies;
- the model key and the extraction date.

The researcher marks **right**, **partly right** or **wrong**, with optional reasons (wrong type, text
misreads the paper, quote does not support it, not in the paper) and a note. A mark can be changed. The last
mark counts.

**"Fix it now"** appears after partly right or wrong. It opens the paper page with that claim's edit form open
(`/papers/{paper_id}?edit={claim_id}&from=spot-check:{sample_id}`) and a "back to the spot-check" link. The
edit is the existing stage 3 overlay save (`notes.overlay.save_edit`). The mark stays on the original
`extracted_claims` row, which an overlay never changes, so accuracy keeps counting the agent's claim. If the
paper has been re-noted since the sample was drawn, the item says "this claim is from an earlier extraction",
and "fix it now" opens the paper page at the current note.

### 5.4 Accuracy

- Stored in the library as the view `spot_check_accuracy` over the marks: per claim type, per model key, and
  per (type, model): counts of right, partly right and wrong, and the total marked.
- Every figure is shown with its count: "right 14 of 20 (70%); partly right 4; wrong 2". The headline is the
  share marked right. Partly right is shown beside it, never folded in as half a point (open question 4).
- Per-type and per-model figures are raw shares of the marks in that row.
- The overall line is weighted by each cell's share of the population and says so ("weighted to the
  library's mix of types and models"), because the sampling rule over-draws thin cells.
- A row with fewer than 5 marks shows "too few marks to say".

### 5.5 Schema: `db/schema/library/0012_spot_check.sql`

| table | key columns | notes |
|---|---|---|
| `spot_check_samples` | `sample_id` ('sc_' + hash of seed, size, ids), `seed`, `size`, `filters_json`, `created_at`, `finished_at` NULL | one row per spot-check sample |
| `spot_check_items` | PK(`sample_id`, `claim_id`); `ordinal`, `paper_id`, `extraction_run_id`, `claim_type`, `model_key`, `claim_fingerprint`, `mark` NULL (`right`, `partly`, `wrong`), `reasons_json`, `note`, `marked_at` | `claim_id` is the immutable `extracted_claims` id; the fingerprint (stage 3 `notes/identity.py`) finds the claim in a later note. "Fixed" is derived, not stored: an `overlay_versions` row for the paper after `marked_at` whose ops touch the claim's fingerprint. So "fix it now" needs no hook in the notes routes |
| view `spot_check_accuracy` | `claim_type`, `model_key`, `n_right`, `n_partly`, `n_wrong`, `n_marked` | latest mark per claim |

### 5.6 Spot-check module (`src/seedgraph/spotcheck/`)

| module | interface | hides |
|---|---|---|
| `sample.py` | `create_sample(lib, *, n=20, seed=None, filters=None) -> Sample`, `sample(lib, sample_id)` | the population, model keys, the least-marked-first draw |
| `marks.py` | `mark(lib, sample_id, claim_id, value, *, reasons=(), note=None)`, `fixed(lib, sample_id, claim_id) -> bool` | validation, last-mark rule, the derived "fixed" check |
| `accuracy.py` | `accuracy(lib, *, by=('type', 'model')) -> AccuracyTable` | raw rows, the weighted overall line, the "too few" rule |

## 6. Screens

All pages follow stage 3: thin routes, one view per GET, one deep function per POST, no inline style or
script (CSP).

### 6.1 Records (`/records`)

- A standing line at the top: "Only answers an agent recorded from your library appear here, with paths that
  ended without an answer. Questions answered without searching the library do not."
- **List**: newest first, paged. Each row: when; the question, or "path without answer", or "in progress";
  graphs (with "spans 2 graphs" marked) or "Library-wide"; client and session name (the start date and time
  when the agent gave none); counts (searches, opened,
  cited); flags; star.
- **Filters**: graph (including Library-wide), state (answered, path without answer, in progress), starred,
  flagged, client, and a text match on the question.

### 6.2 Record detail (`/records/{record_id}`)

- Question; answer; **Agent's judgement** in its own box, labelled as such.
- **Cited evidence**: the frozen quote or claim text, paper title, year and page, trust label ("paper
  states", "agent inferred", "researcher"), and any flag in plain words. Each quote opens the paper page at the
  highlighted passage. A claim whose paper has been edited since shows "edited since this answer" with a link
  to the current version.
- **Path**: a timeline of searches (query, kind, graph, hits) and opens (paper, section or range), each with
  the helper name when there is one. Below it, the **items table**: every item tagged *found but not opened*,
  *opened but not cited* or *cited*.
- Notes: "session told apart by connection only", "session name used from two connections", "n calls could
  not be logged", "graph deleted".
- Actions: star, annotate (one free-text box), export to markdown, delete (with a confirm step).

### 6.3 Graph Records tab

The graph detail page (stage 4 §8.1) gains a fifth tab, **Records** (`/graphs/{slug}/records`): the same list,
filtered to that graph, with the "cites papers in this graph" mark for library-wide records.

### 6.4 Home activity feed

`views.home.activity` adds the latest records: "record: <question, shortened> (3 cited, 1 flag)" and "path
without answer (4 searches)", linked to the record. Open records still in progress are left out of the feed.

### 6.5 Spot-check (`/spot-check`)

- **Accuracy**: tables per claim type and per model, each figure with its count (§5.4), and the weighted
  overall line.
- **Start a spot-check**: size (default 20), optional graph, claim type and model filters. "Start" creates the
  sample.
- **Earlier samples**: date, size, how many marked, link.
- **Sample page** (`/spot-check/{sample_id}`): the items in order (§5.3), each with its mark buttons, reasons
  and note; marks save by htmx swap; a running line for this sample ("marked 12 of 20: right 9, partly 2,
  wrong 1", ticket 08's running figure); "fix it now" after partly right or wrong, and "fixed" once an edit
  follows.

### 6.6 The paper page change

`/papers/{id}` accepts `edit={claim_id}`: that claim's existing edit form renders open and the page scrolls to
it. For a hidden claim it also turns on "show hidden", so the form is there (the researcher may unhide or edit
it). `from=spot-check:{sample_id}` adds the "back to the spot-check" link. Nothing else on the page changes.

## 7. MCP updates

New module `mcp/tools_records.py`:

| tool | arguments | returns |
|---|---|---|
| `record_answer` | `question, answer, evidence, judgement?, session?` | `record_id`, counts, flags, `next_action`, the page path |

`record_answer` is a mutation (`readOnlyHint: false`). It writes only records.

Changes to the existing read tools (§3.2):
- each logged tool gains `session: str | None = None` and an injected `mcp_ctx: Context | None`, following the
  `_worker` pattern in `tools_work.py`;
- one helper, `call_info(ctx, mcp_ctx, session)`, builds the `CallInfo`; one wrapper logs after success;
- search-kind replies carry one line, `record_reminder`: "If you answer the researcher from these results,
  call record_answer once with the question, your answer, the ids you cite and your own judgement apart.";
- every logged reply echoes `session` (the name in use, or the start date and time for an unnamed connection);
- `main()` marks the connection closed when the stdio transport ends.

The researcher is asked to approve `record_answer` once in Claude Code ("always allow"). `setup agents` prints
that hint; it does not edit Claude Code permissions.

## 8. Skill updates

### 8.1 Consultation skill (`seedgraph_v3_skill.md`)

A new section, **"Record your answer"**:
- When you answer the researcher from the library, call `record_answer` once per answer, at the end. Do not
  record answers that did not use the library.
- Choose a session name at the start of each question thread (topic slug + clock time, for example
  `parallel-trends-1432`) and pass it as `session` on every seedgraph call for that question and its
  follow-ups.
- When you start subagents to search for you, give each a helper name `<yours>/a`, `<yours>/b`, and tell it
  to pass it. If you are a subagent and were given no name, choose your own.
- Cite only what you opened: read the passage with `paper_read` (or see its quote in `concept_show`) before
  you cite a span. A search hit alone counts as found, not read.
- Put corpus statements in `answer` and your own view in `judgement`.
- If the reply flags a citation, tell the researcher, or read it and record again.

"Find your footing" and "Ground a claim" stay as they are.

### 8.2 Work skill (`seedgraph_v3_work_skill.md`)

One wording change: "Pass `model` on every submit: your model id as your client states it (for example in
your system prompt). Spot-check accuracy is reported per model." The current wording ("if your client shows
it") let every v3 run so far save without a model (§1.4).

## 9. Deletions and reuse

### 9.1 Delete in stage 5

The `ask` pipeline, the ranker, `AnswerTrace`, its dispositions and the eval harness code were deleted in stage
2 (`a27873b`). What is left:

| what | where | handling |
|---|---|---|
| answer-harness enums `QueryType`, `AnswerCategory`, `AnswerMode` | `vocab.py:494-530` | delete (no importer outside `vocab.py`) |
| `AnswerConfig`, `AnswerPolicy`, `ContentPolicy.external_llm_for_answer_generation`, `GlobalConfig.answer` / `.answer_policy` | `config/models.py:60, 82, 85-122, 360-364, 410` (the class bodies, the fields and the ceiling entry), exports in `config/__init__.py` | delete; `extra="ignore"` keeps old `config.yaml` files loading |
| `answer` / `answer_policy` merge keys | `config/loader.py:366` | delete |
| answer-policy tests | `tests/test_config.py:101-104, 167-171, 194-196` | delete |
| v2 eval fixtures (answer envelopes, leak cases, a mini project) | `evals/` (11 tracked files) | delete; nothing imports them |
| empty package dirs holding only untracked `__pycache__` | `src/seedgraph/{answer,api,eval}/` | remove from disk |

Kept on purpose:
- `migrate/answers.py` (the stage 1 `v2_answers/` export, which preserves v2's saved answers and traces);
- the `answer_policy` block in the v2 home fixture (`tests/migrate/fixtures/v2home.py:250`): it is frozen v2
  input;
- `audit_records` in the v2 **project-scope** schema (`db/schema/project/0012_evaluation.sql`,
  `0016_audit_payload.sql`): a migration chain is not edited piecemeal. The project scope as a whole is read
  only by `tests/test_migrations.py` today, and goes with v2's `project.db` support at stage 7 (open
  question 10).

### 9.2 Reachability

`RESERVED` and `KEEP` are unchanged (`tests/test_reachability.py:37-44`). The scan cannot see dead code inside
a module that is still reachable (`vocab.py` is imported everywhere), so the deletion slice adds:
- `DELETED_IN_STAGE5`, a test that the enums and config classes above are gone (by name, through `ast`), and
  that `evals/` does not exist;
- `records` and `spotcheck` are reached from the MCP server and the web routes, so they need no allowlist.

### 9.3 For the orchestrator (outside the worktree)

`CONTEXT.md` edits, proposed:
- **Traversal record**: "...stored once in the library and listed under each context graph it searched; a
  record spanning two graphs appears in both, marked."
- **Context graph**: "owns ... traversal records" becomes "lists the traversal records that searched it".
- **AnswerTrace**, **Disposition**, **Unsaved answer**, **MCP surface**: mark as v2 only.
- New: **Session** (one agent's stream of logged calls, told apart by a session name or, failing that, by its
  connection, which is then named by its start date and time) and **Spot-check sample**.

### 9.4 Reused

| need | reused from |
|---|---|
| connection id, client name, structured errors | `mcp/server.py` (`V3Context.worker_id`, `client_name`, `tool_error`) |
| optional name + injected context | `tools_work.py` `_worker` pattern |
| agent-audience reads | `tools_library.py`, `tools_concepts.py` unchanged bodies |
| frozen claim text and version | `current_claims` (`overlay_id`, `version_no`), `notes.interpretation.current` |
| claim fingerprint, matching | `notes/identity.py` (`match`, line 220) |
| quote → page highlight | stage 3 paper page and `library/pages.py` |
| claim edit | `notes.overlay.save_edit` and the existing edit form |
| seeded sample pattern | `concepts/researcher.py` review sample |
| web shell, feed, tabs | `web/views/home.py` (`activity`), `views/graphs.py` tabs, `routes/placeholders.py` (replaced) |

## 10. Acceptance plan

Run on the real migrated library in the v3 home. Reports go to `<home>/acceptance/stage5/`. Questions,
answers and marks stay there.

1. **Five real questions.**
   - Before the run, the researcher writes five questions they would really ask, on the migrated DiD and
     land_value graphs, into `<home>/acceptance/stage5/questions.yaml`. At least one is asked from **Codex**,
     the rest from Claude Code. One uses two helper subagents.
   - The researcher asks each question in a fresh agent session with the consultation skill.
   - After each answer, the researcher opens the record and fills the per-record checklist that
     `records acceptance-report` writes to `<home>/acceptance/stage5/records_check.md`: the question is right; the answer matches what the agent said; every search
     appears with its query and hits; every paper or section opened appears; the tags are right; the cited
     passages are the ones the agent named; judgement is apart; the graphs are right.
   - An automated cross-check (`records check-transcript`, Claude Code only) reads the local session transcript
     and compares its seedgraph tool calls with the record's events, by count and order.
   - **Pass**: five records marked correct, and the cross-check finds no missing or extra event.
2. **A planted unread citation is caught** (automated, `tests/acceptance/stage5/test_planted_citation.py`,
   opt-in, against a copy of the real library through an in-memory MCP client):
   - search, read one passage, then cite a span of another paper that no reply returned → "cited without
     being read";
   - cite a span that was only in search hits → "cited without being read (seen in search results only)";
   - cite a span seen only as a `paper_show` anchor → "cited without being read";
   - cite the passage that was read → no flag.
   One live variant: the researcher asks the agent to cite a named span without opening it.
3. **Path without answer**: automated (calls, no `record_answer`, the clock moved past `PATH_IDLE`, then a
   closed connection) and live (the researcher stops an agent mid-search). Both show "path without answer" in
   Records and on Home.
4. **Session identity probe**: a parent plus two parallel helper subagents with names gives the parent's
   record with helper-tagged reads and no cross-ownership. The same run without names shows one connection
   stream, marked "session told apart by connection only". A server restart mid-question with a name keeps one
   record. The probe's `_meta` keys are reported.
5. **Spot-check end to end**: a 20-claim sample on the real library. Every item is marked. One wrong claim is
   fixed through "fix it now", and the accuracy figure is the same before and after the fix. Accuracy per type
   and per model is shown with counts. Report: `spotcheck.json` + `.md` (marks per cell, the seed, the
   weighted overall line).
6. **Walk-through** for sign-off: filter the Records list by graph, open a record spanning two graphs, star,
   annotate and export one, delete one; start a spot-check, mark, fix one, read the accuracy table.

## 11. Build slices

Each slice is one agent, test-first, with strict file ownership. Fixtures are invented (stage 1 helpers); no
private text. Shared files (`pyproject.toml`, `uv.lock`, `cli.py`, `commands/__init__.py`, `.gitignore`) are
touched only by the slice named for them.

| # | Slice | Depends | Owns | Tests first |
|---|---|---|---|---|
| R1 | **Schema and session log**: both migrations, `records/{__init__,sessions,log}.py` | — | `db/schema/library/0011_records.sql`, `db/schema/library/0012_spot_check.sql`, `src/seedgraph/records/{__init__,sessions,log}.py`, any migrate-test expectations for the new tables, `tests/records/test_sessions*.py`, `test_log*.py` | key order (named, connection); name validation; helpers point to parents; same name from two connections warns; restart keeps a named stream and splits a connection one; first event opens a record; unlogged-call count; migrate tests green without editing `src/seedgraph/migrate/` |
| R2 | **Answer, path, export, manage**: `records/{answer,path,export,manage}.py` | R1 | those files, `tests/records/test_answer*.py`, `test_path*.py`, `test_export*.py` | found / opened / cited rules incl. the half-span overlap and anchors-only; each flag; unknown ids refused; zero-event record passes on earlier reads; helper events join the parent's open record, and move to the helper's own record when the helper answers (no helper record left as a path without answer); frozen claim text and `(overlay_id, version_no)` survive a later edit; graph links (`searched`, `cited_member`, none); idle and closed → path without answer; export layout; delete removes events |
| R3 | **MCP and skills**: `tools_records.py`, logging in the read tools, reminder line, connection close, skill text | R1, R2 | `mcp/tools_records.py`, `mcp/tools_library.py`, `mcp/tools_concepts.py`, `mcp/server.py` (registration, `call_info`, close hook), `agent_setup/*.md`, `tests/mcp/test_records*.py` | in-memory MCP: every logged tool writes one event and echoes `session`; unlogged tools write none; reminder on search-kind replies only; a failed log write still answers; `record_answer` round-trip; agent audience unchanged (stage 2–4 MCP suites green) |
| R4 | **Spot-check core**: `spotcheck/{__init__,sample,marks,accuracy}.py` | R1 | those files, `tests/spotcheck/` | population includes hidden and edited, excludes added and lens claims and marked ones; model key fallbacks; least-marked-first with ties; 2-per-paper cap; same state + seed → same sample; last mark counts; an overlay edit after a mark shows as fixed and leaves the figure unchanged; weighted overall; "too few" rule |
| R5 | **Records screens**: list, detail, graph tab, Home feed; replaces the placeholders | R2 | `web/views/records.py`, `web/routes/records.py`, `web/templates/records/*`, `web/views/home.py` (`activity` only), `web/views/graphs.py` (tabs constant only), deletes `web/routes/placeholders.py` and `web/templates/placeholders/`, tests | each POST calls one deep function; filters; "spans 2 graphs" mark; judgement labelled; standing scope line; export download; feed entries; CSP test |
| R6 | **Spot-check screens and "fix it now"** | R4, R5 | `web/views/spotcheck.py`, `web/routes/spotcheck.py`, `web/templates/spotcheck/*`, `web/views/paper.py` + `web/routes/papers.py` + `web/templates/paper/_notes.html` (the `edit=` / `from=` parameters only), tests | start, mark and change a mark by htmx; "fix it now" opens the right form, including for a hidden claim; back link; earlier-extraction notice; running per-sample counts; accuracy tables with counts |
| R7 | **Deletions and reachability**: §9.1, §9.2 | — | `vocab.py` (the enums), `config/{models,loader,__init__}.py` (answer parts), `tests/test_config.py` (those tests), `evals/`, `tests/test_reachability.py` | `DELETED_IN_STAGE5` green; full suite green |
| R8 | **Acceptance harness**: planted citation, path without answer, identity probe, transcript cross-check, the per-record checklist, spot-check report | R3, R5, R6, R7 | `tests/acceptance/stage5/`, `src/seedgraph/commands/records.py` (`acceptance-report`, `check-transcript`) and its registration in `cli.py` / `commands/__init__.py` | opt-in; writes reports to the home; the transcript reader on an invented transcript |

**Waves (2–3 at a time).** Wave 1: R1, R7. Wave 2: R2, R4. Wave 3: R3, R5. Wave 4: R6. Then R8 and the real
acceptance run. The critical path is R1 → R2 → R5 → R6 → R8. R1 owns both migrations, so only one slice
touches the migrate-test expectations. R6 waits for R5 because R5 removes the `/spot-check` placeholder route,
which would otherwise be mounted first (`routes/placeholders.py` sorts before `spotcheck.py`). R3 is the only
slice that edits `tools_library.py` and `tools_concepts.py`.

## 12. Disagreements recorded

- **Storage** (§2): stage 1 §3 reserved the record tables in `graph.db`, and ticket 05 / `CONTEXT.md` say the
  context graph owns records. Ticket 07 #5 ("appears in both, marked") wins: one record in `library.db`, linked
  to its graphs. Glossary edits go to the orchestrator (§9.3).
- **Deletions**: SPEC lists the `ask` pipeline, ranker, dispositions and eval harness under stage 5. Stage 2 (P8)
  already deleted them, as stage 2 §9 planned. Stage 5 deletes only the remnants (§9.1).
- **Evidence ids**: SPEC and ticket 07 #2 say "span or paper ids". Claim ids are accepted too, resolved to their
  span. This extends the ticket; it does not contradict it.
- **"Session"**: SPEC uses it for a spot-check run. This document says "spot-check sample".
- **Model "unknown" stratum**: the brief expected migrated claims to lack a model. In the real library it is
  the other way round: migrated runs carry one, v3 runs do not (§1.4).

## 13. Open questions for the researcher

Each has a recommendation (*R*). **All twelve were settled by the researcher on 2026-11-04**: each *R* is
adopted, with question 1 amended as recorded there.
1. **How a session is told apart.** Options: the connection only (simple, but subagents on one connection and
   restarts go wrong); an agent-chosen session name with helper names and the connection as fallback; or a
   client-supplied marker, if Claude Code turns out to send one. *R:* the session name with the connection as
   fallback (§3.1). Revisit after the identity probe reports the `_meta` keys. **Decided:** the session name
   stays optional; by default a session is indexed by its start date and time within the context graph
   (§3.1).
2. **A citation seen only in search results.** Flag it "cited without being read", or only tag it? *R:* flag it,
   with "(seen in search results only)". That matches SPEC's "never opened" check and the skill's "read before
   relying". If the five real questions show this flag on answers you consider fine, soften it to a tag.
3. **Library-wide records in graph lists.** *R:* show them in each graph that holds a cited paper, marked
   "library-wide search", as well as under Library-wide.
4. **The accuracy headline.** *R:* the share marked right, with partly right shown beside it and never counted
   as half.
5. **Sample size and rule.** *R:* 20 per sample, least-marked cell first, at most 2 claims per paper. A sample
   takes about 15–20 minutes.
6. **Hidden and edited claims in the population.** *R:* include both. Leaving out hidden claims would make
   accuracy look better than the agent was.
7. **Model labels that name a client, not a model** (some migrated runs; v3 runs saved without a model).
   *R:* keep each as recorded, as its own stratum, so no figure is guessed. The work skill change (§8.2) stops
   new runs from adding to "model not recorded".
8. **Lens claims in spot-checks.** *R:* not in stage 5. Add a lens filter when a lens is in real use.
9. **Records and marks before stage 7.** v3 data is disposable until the switch-over (stage 3 §0). *R:* treat
   them as test data. Spot-check marks are keyed by claim ids that the migration carries from v2, plus
   fingerprints, so stage 7 can decide to carry them if the figures are worth keeping.
10. **v2 project-scope schema** (it still holds the eval harness's `audit_records`). *R:* leave it to stage 7,
    which removes v2 `project.db` support as a whole.
11. **Path idle time** (`PATH_IDLE` = 30 minutes before an unanswered path shows as "path without answer").
    *R:* accept; it changes only the label, never which record owns what.
12. **Approving `record_answer` in Claude Code.** *R:* approve it once with "always allow". Seedgraph does not
    edit Claude Code's permission settings.
