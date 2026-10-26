# v3 Stage 3 design: UI core (library, paper page, note editing, batches)

Status: design for build. No code yet. This document covers v3 stage 3: the web stack and module layout, the
sidebar shell and its pages, quote → PDF highlighting and how it is measured, researcher note editing as an
overlay (ADR-0011), claim identity across extractions and replay, quote-backed added claims, adding papers in
the UI, the Ollama metadata quality fixes carried over from stage 2, security, the CLI trim and deletions, the
acceptance plan, the build slices and the open questions. It uses the glossary terms from `CONTEXT.md` (library,
context graph, held paper, cited-only paper, batch, worker, researcher-edited note, researcher notes, researcher
assertion, current interpretation, extraction freshness) and the deep-module vocabulary from `codebase-design`
(module, interface, depth, seam, adapter). `docs/V3_LIBRARY_DESIGN.md` (stage 1) and
`docs/V3_AGENT_PROTOCOL_DESIGN.md` (stage 2) are binding for storage and for the work protocol.

The repo is public. Real counts, highlight measurements, the prototype test set, acceptance PDFs and reports
live in the v3 home, never in this document or the repo. The UI prototype branch `prototype/v3-ui` holds private
note text and PDF copies. It was read for its structure only; nothing from it is copied into the repo.

## 0. Decided before this stage

**v3 data is disposable until stage 7** (researcher, 2026-10-26, option A):
- At stage 7 a fresh migration from v2 runs, and then the researcher switches over.
- Everything written natively in v3 before then is test data: researcher edits, attention resolutions, notes
  written in v3. A re-migration moves it aside with the rest of the library (`migration/replaced/`, stage 1
  §1.3). It is not replayed.
- Real editing stays in v2 until the switch.
- Consequence for the overlay design (§4): overlays are still keyed by stable ids (`paper_id`, the base
  `extraction_run_id`, and a claim fingerprint that does not depend on row ids). An export or replay feature
  therefore stays possible later. This stage designs and builds no replay-across-migration step.

## 1. Web stack and module layout

### 1.1 Stack

- **Starlette + Jinja2**, served by **uvicorn**. Pages are rendered on the server. FastAPI is not used: its value
  is JSON APIs and OpenAPI, while these routes take HTML forms and return HTML, so FastAPI would add a layer
  without adding depth. Dependencies re-added in `pyproject.toml`: `starlette`, `jinja2`, `uvicorn` (plain, not
  `[standard]`), and `python-multipart` (uploads). Exact versions are pinned in `uv.lock`.
- **htmx** handles the live parts: polling of job and batch progress, partial swaps after edits, and the layout
  toggle. **No JS framework.** The page's own JS is a few small modules: the PDF pane controller, the text fold,
  the CSRF header hook, and selection capture.
- **PDF.js** renders the PDF pane: the generic viewer components (`pdf.mjs`, `pdf.worker.mjs`,
  `pdf_viewer.mjs` and `.css`) with the text layer, plus `cmaps/`, `standard_fonts/` and `LICENSE`. The version
  is chosen and pinned when the U4 slice is built: it must be a released version at least two weeks old, with
  ES-module builds. The vendored set is everything that build loads at runtime (including any `wasm/` or
  `iccs/` folder), not a fixed file list. Vendored assets are recorded in `web/static/vendor/VERSIONS.json` (name, version, source
  URL, sha256 per file). A test checks every file's hash, so an asset cannot drift silently. Nothing loads from
  a CDN at runtime.
- **`seedgraph serve [--port 8765] [--no-open] [--root PATH]`** (`commands/serve.py`, a thin adapter). It always
  binds `127.0.0.1` and has no `--host` option. It opens the browser unless `--no-open`. If the port is busy it
  fails with a plain message; it never picks another port silently. There is no login. CSRF protection covers
  every mutating request (§8).

### 1.2 Module layout

The deep modules stay in `library/`, `notes/` and `work/`, and stage 3 adds the new behaviour there. `web/`
contains only adapters and one view-model layer.

```
src/seedgraph/web/
  __init__.py
  app.py            create_app(home, *, pool=None, chain=None, clock=None) -> Starlette
                    the one wiring point: middleware, routes, templates, jobs; tests build it with fakes
  security.py       LoopbackGuard (client + Host check), CSRF (per-process token), security headers (§8)
  jobs.py           JobRunner: background ingest / convert / DOI lookup / Ollama metadata (§6)
  views/            THE view-model layer: pure functions (Library, args) -> frozen dataclasses for templates
    home.py         checklist per graph, activity feed
    library.py      list views, counts, estimates
    paper.py        header, badges, interpretation with trust labels, quote locators, history, compare
    batch.py        batch list and detail (wraps work.routine.batch_view)
    review.py       attention, duplicates, citation review
    settings.py     paths, converter, Ollama, provider keys present (names only)
    graphs.py       graph list and minimal detail
  routes/           thin adapters: parse request -> call ONE deep function or view -> render a template
    home.py library.py papers.py notes.py batches.py review.py graphs.py settings.py files.py placeholders.py
  templates/
    base.html  _sidebar.html  _flash.html
    pages/      home, library, paper, batch, batches, graphs, graph, review_*, settings, records, spotcheck
    partials/   jobs, batch_items, claim_row, claim_edit, overlay_history, compare_row, add_claim, ...
  static/
    app.css
    js/csrf.js  js/layout.js  js/pdfpane.js  js/fold.js  js/select.js
    vendor/htmx/  vendor/pdfjs/  vendor/VERSIONS.json
```

Rules:
- **Routes never run SQL** and never import `sqlite3`. A GET calls one `views.*` function. A POST calls one deep
  function: `notes.overlay.save_edit`, `library.ingest.add_dois`, `work.routine.create_batch`, `work.routine.stop`,
  `library.attention.resolve_item`, and so on. Then it re-renders through a view. A test checks route modules
  for `sqlite3` and `execute(` imports.
- **Templates never call the library.** They receive view dataclasses only.
- **Views are the only place where data is composed for display.** They are pure reads, so they can be tested
  directly against a fixture library without HTTP.
- Each request opens its own read connection, or reuses the thread's (`Library` is not shared across threads).
  Writes go through the deep modules' `lib.write()`, with `busy_timeout` set (stage 1 store).

New deep modules (all test-first, no web imports):

| module | interface (small) | hides |
|---|---|---|
| `notes/overlay.py` | `save_edit(lib, paper_id, op, *, expected_version) -> OverlayVersion`, `revert(lib, paper_id, to_version, *, expected_version)`, `history(lib, paper_id)`, `set_researcher_notes(...)` | op validation, versioning, quote anchoring for added claims, FTS refresh, "concepts out of date" mark |
| `notes/interpretation.py` | `current(lib, paper_id, *, audience) -> Interpretation`, `reindex(lib, paper_id)` | extraction + overlay merge, trust labels, hidden-from-agent rules |
| `notes/identity.py` | `fingerprint(claim_row, markdown) -> ClaimFingerprint`, `match(old: Sequence[ClaimFingerprint], new) -> Matching` | the "same claim" rules (§4.4), pure |
| `notes/replay.py` | `plan(lib, paper_id) -> ReplayPlan`, `accept(lib, plan, decisions) -> OverlayVersion` | side-by-side comparison, re-keying ops onto the new extraction |
| `library/textfold.py` | `fold(text) -> Folded` (folded string plus an index map back to the raw offsets) | the normalisation that highlight, selection and measurement share (§3.2) |
| `library/pages.py` | `quote_locators(lib, paper_id) -> list[QuoteLocator]`, `anchor_selection(lib, paper_id, page, text) -> AnchorResult` | page lookup, re-anchoring, unique match within a page range |
| `library/pagecheck.py` | `run(lib, home, *, set_name, mode) -> PageReport` | the highlight measurement (§3.4) |
| `library/ingest.py` | `add_pdfs(lib, paths, *, source)`, `add_dois(lib, dois, *, chain, fetch)` | matching to existing papers, paper creation, attach, queueing, OA fetch |
| `library/metadata_quality.py` | `clean_title(s)`, `generic_title_reason(s) -> str or None` | footnote marks, generic-title stoplist (§7), pure |
| `library/metadata_verify.py` | `verify(lib, paper_id, *, chain) -> VerifyOutcome` | DOI → OpenAlex/Crossref, tolerant comparison, preferring verified values |
| `work/estimate.py` | `estimate(lib, task, paper_ids) -> Estimate` | text tokens per paper (chars / 4 after stripping references) and the per-paper range |

### 1.3 CLI trim

After stage 3 the core CLI is: `setup` (incl. `setup agents`), `migrate`, `serve`, `mcp serve`, `convert`,
`work`, `golden`, `library` (the "maintenance" group in the spec: status, search, show, check, attention,
remove, delete, refetch-metadata, prune-replaced, textcheck, reconvert, graphs, and the new `pagecheck`), and
`version`. The trim happens in the last slice (U9) and edits `cli.py` and `commands/__init__.py` only there.
`cli.py` becomes about 30 lines: the UTF-8 stream fix, the app, `version`, and `commands.register(app)`.

Old groups deleted with their tests: `corpus`, `cite`, `sections`, `spans`, `lens` (v2 project lens files;
lens choice comes back in the stage 4 UI), `concepts`, `graph`, `cache`, `project`, `review`, `doctor`
(project-based checks; the Ollama and converter checks move to the Settings page and `setup`), and the
top-level `work-read` (replaced by MCP `paper_read`). Deleting `concepts` and `graph` here is a **recorded
deviation** from the deletion rule, because their replacement arrives in stage 4. They read only `project.db`,
which v3 never writes, so they serve no v3 data. This is the same reasoning stage 2 used for the old web UI
(open question 7).

## 2. Sidebar shell and pages

The base layout has a persistent left sidebar: **Home · Library · Context graphs · Records · Spot-check ·
Settings**. A "Review" entry appears under Library with a count badge while attention items are open. The top bar
holds a search box (library search, researcher audience) and a jobs indicator that shows conversions running.
Every paper title anywhere carries a small **PDF** button (opens `/papers/{id}/pdf` in a new tab) next to the
link to its paper page.

| page | URL | built from | live parts |
|---|---|---|---|
| Home | `/` | `views.home` | activity feed polls every 10 s while a batch is working |
| Library | `/library?view=&graph=&q=&page=` | `views.library` | job strip polls every 2 s while jobs run |
| Paper | `/papers/{id}?layout=split\|pdf\|notes` | `views.paper` | edit forms swap claim rows; history swaps |
| Batches / batch | `/batches`, `/batches/{id}` | `views.batch` | items poll every 5 s until finished |
| Context graphs | `/graphs`, `/graphs/{slug}` | `views.graphs` | none |
| Review | `/review/attention?kind=`, `/review/duplicates`, `/review/citations` | `views.review` | resolve swaps the row |
| Settings | `/settings` | `views.settings` | "check again" buttons |
| Records, Spot-check | `/records`, `/spot-check` | static placeholders ("arrives in stage 5") | — |

### 2.1 Home

- **"What's next" checklist per graph.** Each line is derived, never stored, so it cannot go stale. Lines appear
  only while true, each with its action link:
  - "n member papers need notes" (no current note, or a stale one) → Write notes, pre-selected;
  - "n papers held back" (text gate or title miss) → Library, held-back view;
  - "n attention items on this graph's papers" → Review, filtered;
  - "batch X paused: usage limit, resets <time>" / "stopped" / "a worker stopped responding" → batch page;
  - "n papers have a newer extraction to review" → those paper pages;
  - "concepts out of date (n papers changed)": shown now, acted on in stage 4;
  - stage 4 adds the growth steps (seeds → depth → review found papers).
- **Activity feed**: the latest `batch_events` across batches (claimed, submitted, accepted, needs revision,
  held back, paused, continued), finished ingest and conversion jobs, and overlay saves. Each entry shows the
  batch's single state label from `work.state` (stage 2 §1.6), so pause states read exactly as on the batch
  page. Stage 5 adds recent records.

### 2.2 Library

- **Views**: `held` (default), `cited-only`, `needs notes` (held, no current or a stale note), `held back`
  (open `text_garbled` / `text_missing` / `pdf_title_miss` / `shared_pdf` items, with the reason and the
  `accept_text` action where it applies), and `all`. Each can be filtered by graph and narrowed by a title search.
  Columns: title (+ PDF button), year, authors, graphs, note state (none / current / stale / edited / newer
  extraction), quotes anchored/total, badges.
- **Add papers** panel (§6): a drop zone for PDFs, a folder picker (`<input webkitdirectory>`) with a "or type
  a folder path" field for large folders (read in place, not uploaded), and a textarea for DOIs, one per line.
  Results appear as a job strip: queued → converting → converted → metadata → done, with the standing PDF check
  result per file ("PDF matches the paper" / "this PDF may not be this paper" / "could not check").
- **Write notes**: select rows (or "all in this view") → a confirmation panel. It shows the papers, those that
  will be skipped (preserved, held back, failed) with reasons, and the estimate from `work.estimate`: total paper
  text tokens and a per-paper range of agent tokens. The range factor is a constant in `work/estimate.py`,
  calibrated from the stage 2 measurements, which stay in the home. Confirm calls `work.routine.create_batch`
  and goes to the batch page. "Start in background" appears in stage 6; until then the batch page's copyable
  commands are the launch path.

### 2.3 Context graphs (minimal until stage 4)

- A list with display name, member, seed and frontier counts, and papers needing notes.
- A detail page with the membership table (member / frontier / excluded, seed flag), remove from graph
  (`library.lifecycle.remove_from_graph`), and the lenses chosen (read only). The 2D graph, literatures and
  growth arrive in stage 4. The page leaves a right-hand column for them.

### 2.4 Paper page

- **Header**: title (with its provenance, §7), authors, year, venue, identifiers, and badges: held / cited-only,
  note state, "edited", "newer extraction available", converter version, "no page information", and open
  attention items with links. Also the graphs the paper is in, and the actions **Open PDF**, **Write notes**
  (a one-paper batch; `replace` when a current note exists), **Add to graph** (a select menu;
  `library.graphs.set_membership`, refused for cited-only papers as stage 1 requires) and **Edit metadata**
  (sets `title_source = 'user'`).
- **Layout toggle**: Side by side (default: PDF left, notes right) · PDF · Notes. The choice is a query
  parameter, remembered in `localStorage` (wrapped in try/catch; the page works without it).
- **Notes pane**: the current interpretation (§4.3) grouped by claim type in the fixed order, then the
  researcher notes box. Each claim shows a trust label chip (**paper states** / **agent inferred** /
  **researcher**), its text, its label, and its quote with **"p. N"**. Clicking the quote or the page chip jumps
  the PDF pane and highlights the passage (§3). Edit controls sit on each claim (§4.2). "History" opens the
  version list with revert. Under an edited note, "Newer extraction available — compare" opens the replay view
  (§4.5).
- **PDF pane**: PDF.js viewer with page navigation, zoom and find. A text selection shows an "Add claim from
  selection" button (§5). A paper without a PDF shows "cited-only: add its PDF" with the add-papers controls.
- **Text view**: a toggle inside the notes pane shows the markdown of the current page range as plain text with
  offsets. It is the fallback for selection (§5) and for papers without page markers.

### 2.5 Batches

- **List**: every batch with task, selection text, state label, done / held back / failed counts, created.
- **Detail** (`work.routine.batch_view`):
  - the state label and next action in plain words, and stop mode;
  - the two copyable commands (Claude Code, Codex) from `work/commands.py`, each with a copy button;
  - per-paper rows: state (queued / claimed by worker since t / done / held back / failed / preserved / stale /
    conflict), parts done k/n, anchored/total quotes for done papers (from the saved note's claims and
    `claim_spans`), and the held-back or failed reason with a link to the attention item;
  - each row expands to show its claim history (worker, client, claimed at, how it ended: live, submitted,
    expired, released, stopped, limit), from the stage 2 claim history;
  - **Stop: finish current papers** / **Stop now** (`work.routine.stop`), and **Continue**
    (`work.routine.continue_batch`), each a POST with CSRF;
  - `accepted_unanchored` is shown when non-zero ("n quotes were saved as ambiguous after revision").
- Stage 6 adds running agents with elapsed time, tokens and the last error. The detail template reserves a slot
  for them.

### 2.6 Review screens

One page per kind of queue. Each row shows its evidence and its resolve actions. Every action is a POST that
calls `library.attention.resolve_item` (or a duplicate or citation function) with a fixed `resolution` value:

| queue | source | actions (resolution) |
|---|---|---|
| Held back: text | `text_garbled`, `text_missing` | **Note anyway** (`accept_text`), **Re-convert with OCR** (`library.reconvert`), Replace PDF |
| PDF may not be this paper | `pdf_title_miss`, `shared_pdf` | **It is this paper** (`confirmed`), Detach PDF, Attach the right PDF |
| Identity | `identity_check`, `identity_collision` | **Same paper** (merge, `duplicate_decisions` `same`), **Different** (`different`) |
| Titles and metadata | `malformed_title`, `metadata_needed`, `metadata_generic_title`, `metadata_doi_mismatch` | Edit metadata, Re-fetch by DOI, Run Ollama metadata |
| Re-note offers | `renote_offered` | Write notes (one-paper batch), Dismiss |
| Duplicates | `identity_check` raised by ingest (below) and by the migration | Same / Different (recorded once for every graph) |
| Citation review | migrated `citation_review_decisions` (read only) | Undo a rejection |
| Anything else | any other kind (e.g. migration-only kinds such as test-converter markdown) | generic row: detail, subject JSON, **Mark resolved** (`resolved_by_researcher`) |

The migration's attention list is the same `attention_items` table, so the migrated items show up here with
no special path. Items with `hides_notes = 1` are marked "notes hidden from agents until resolved". A test
collects every attention kind raised anywhere in `src/` and asserts that each maps to a review row, falling
back to the generic row.
- **Duplicates from ingest**: when metadata fills a new paper's title, `library.ingest` runs the tolerant title
  plus year match (`library.titlecheck`) against existing papers. A hit raises `identity_check`, so the paper is
  never merged silently.
- **Citation review**: no v3 module generates new citation-resolution candidates yet; they are produced when
  reference lists are resolved, which is stage 4's guided growth. Stage 3 shows the migrated decisions and lets
  a rejection be undone. The candidate queue arrives with its producer (open question 11).

### 2.7 Settings (minimal)

The v3 home path, library size, graphs folder; the converter (locked Marker version, slots, GPUs, running
converters, "check"); Ollama (reachable, models present, the metadata model setting); which scholarly-API
environment variables are set (names only, never values); and the MCP registration status for Claude Code and
Codex (read only, from `setup agents`' show mode). Nothing on this page writes secrets.

## 3. Quote → PDF highlight

### 3.1 Page lookup (server, `library/pages.py`)

`quote_locators(lib, paper_id)` returns one `QuoteLocator` per anchored claim in the current interpretation:
`claim_id, span_id, page, page_end, how, needle`.
- **`how = 'marker'`**: `evidence_spans.page_start/page_end`, which stage 2's `ensure_span` fills from Marker's
  page delimiters (`segment.page_for_range`). Spans carried from v2 have the same columns.
- **`how = 'reanchored'`**: the span's markdown has no delimiters, but the paper's current markdown, which is
  from the same PDF, does. The quote is relocated with `anchor.reanchor` (exact and unique only), and the page
  comes from the current markdown.
- **`how = 'none'`**: no page information. The quote shows "no page" and the paper header shows "no page
  information: re-convert to get pages" (`library.reconvert`, with notes kept by re-matching, stage 2 §5.5).
  **A page is never guessed** from character proportions or neighbouring quotes.
- A page beyond the PDF's page count (from pypdfium2, cached per source file) is reported as `how = 'none'` with
  a warning. It would mean the markdown and the PDF disagree.
- `needle` is the quote's folded form (§3.2), sent to the browser so client and server search the same string.

### 3.2 One normalisation, two implementations

`library/textfold.py` and `static/js/fold.js` implement the same fold. Each returns the folded string plus an
index map from each folded character to its raw offset:
1. Unicode NFKC (ligatures such as `ﬁ` → `fi`, full-width forms), then lower case.
2. Drop soft hyphens (U+00AD) and zero-width characters. Join line-end hyphenation (`-` followed by optional
   spaces and a newline, then a letter).
3. Drop markdown and markup that the PDF does not print: `*`, `_`, `#`, backticks, `<sup>`/`<sub>`/`<span>`
   tags, link brackets, table pipes. Drop inline `$…$` math (the PDF's glyphs for it are not comparable text).
4. Keep only `[0-9a-z]` and Unicode letters. Whitespace and punctuation are removed entirely, which absorbs
   PDF.js's per-item spacing, line breaks and curly vs straight quotes.

Shared test vectors (`tests/fixtures/textfold_vectors.json`, invented text) are run by pytest against the
Python fold and by the Playwright suite against `fold.js` in the browser, so the two cannot drift.

### 3.3 Highlight in the browser (`pdfpane.js`)

On a quote click:
1. Scroll the PDF.js viewer to `page`, and wait for that page's text layer to render (`textlayerrendered`).
2. Fold the concatenated text-content items of `page` (and `page_end` when it differs), keeping an index map
   back to (text-layer span, offset).
3. Search for the full `needle`. If it is not found, search for three 30-character windows (start, middle,
   end) and take the range from the first hit to the last, if they are in order and within 1.5× the needle's
   length.
4. If found: build a DOM `Range` over the mapped text-layer nodes, wrap them in `mark.sg-hl`, and scroll the
   first one into view. If not found on the page(s), try the neighbouring pages once, then fall back to a
   **page-only highlight** (outline the page) with "exact passage not found on this page; showing the page".

A rough check on the five prototype test PDFs (pypdfium2 text, this fold, three windows) found most quotes
on their assigned page and **none on a different page**. Nearly all misses came from one PDF whose text layer
does not match its printed text (likely a font-encoding problem), which is the case the page-only fallback is
for. The figures are kept in the home.

### 3.4 Measurement: `seedgraph library pagecheck`

**What the prototype measured.** The prototype's export script did this for every non-lens claim of each picked
work that had a quote:
- it took the claim's first anchor quote, all note versions included;
- it located the quote with a first-occurrence `find` in the markdown, falling back to the stored span offset;
- it took the page as the last Marker delimiter (`{N}` plus 48 dashes) before that position, plus 1;
- it counted a quote as "paged" when a page resulted.

The ticket's figures (all quotes paged on the five test PDFs; the larger paged/total count across the noted
papers) are therefore **page-assigned counts**. Nobody checked them against the PDF. The browser pane only
jumped to the page (`#page=N` in an iframe). This stage keeps that figure as the acceptance bar the spec names,
and adds verification against the PDF on top.

**The sets** live in `<home>/ui/highlight_sets.json` and are created once by hand when U3 is built. No repo code
reads the prototype branch:
- `proto5`: the sha256 of each of the five PDFs in the prototype branch's `pdfs/` folder (computed from
  `git show` output), resolved through `source_files` → `paper_files`;
- `proto31`: the v2 work ids of the prototype's noted papers (the `work_id` of each entry in `snapshot.js`
  `notes`) plus `v2_project`, resolved through `v2_id_map`. It also lists the **claim-id prefixes of every quoted
  claim** in those entries; the snapshot keeps an 8-hex prefix of each claim id, and stage 1 carried claim ids
  verbatim. Prototype mode counts exactly those claims. Two prototype works merge into one library paper (a
  known shared-PDF defect), and pooling their claims would otherwise give a larger population than the
  prototype counted. The temporary copy of `snapshot.js` used to build the file is deleted afterwards.
- `<home>/ui/highlight_expected.json` holds the prototype's figures as the bars to compare against
  (researcher-owned, like stage 1's `expected.json`).

**Two modes**:
- `prototype`: exactly the prototype's claims (by the stored prefixes; for `proto5`, all quoted claims of those
  papers' notes carried from the prototype's project), rank-0 anchor, page from the stored span. This is the
  like-for-like comparison with the prototype's counting. Stored span
  offsets are authoritative, so this mode is at least as strict as the prototype's first-occurrence find.
- `current`: the anchored claims of the current interpretation (§4.3). It is what the researcher actually sees,
  so it differs in what it counts: current notes only, library-wide note choice, and merged papers.

**Three tiers per quote**, reported per paper and in total:
1. **page assigned** (`how` = marker / reanchored / none);
2. **passage verified**: the folded quote (or the windows) is found in pypdfium2's text of the assigned page(s);
3. **page-only**: assigned but not verifiable. Separately, **page wrong**: found only on a different page. Any
   page-wrong quote fails the check.

Papers with no page markers are listed by paper id with their converter version and offered for re-conversion,
never guessed. The report goes to `<home>/ui/pagecheck/<timestamp>/report.{md,json}`. In a read-only check of
the current library during design, the `current` mode counts fewer quotes than `prototype` mode, for the reasons
above, and found a few more marker-less papers than the prototype reported. Those were markdowns imported from
v2 without pagination, plus one note made on an older markdown, which the `reanchored` path covers.

## 4. Note editing as an overlay (ADR-0011, ticket 09)

### 4.1 Schema: `db/schema/library/0009_note_overlays.sql`

| object | columns | notes |
|---|---|---|
| `note_overlays` | `overlay_id` PK ('ov_' + hex), `paper_id`, `schema_id`, `base_note_id`, `base_extraction_run_id`, `head_version_no`, `created_at`, `updated_at`, `origin` | UNIQUE(`paper_id, schema_id`); one overlay per paper note schema |
| `overlay_versions` | PK(`overlay_id, version_no`); `base_note_id`, `base_extraction_run_id`, `ops_json` (the **complete** op set at this version), `researcher_notes`, `action` ('edit', 'hide', 'unhide', 'add', 'remove_added', 'notes', 'revert', 'accept_extraction'), `summary`, `reverted_to` NULL, `created_at` | immutable; revert = a new version copying an old op set; the source of truth |
| `interpretation_claims` | `claim_id` (base claim id, or 'rc_' + hex for an added claim), `paper_id`, `overlay_id`, `version_no`, `claim_type`, `claim_subtype`, `field_key`, `normalized_label`, `claim_text`, `plain_words`, `status`, `trust` ('paper_states', 'agent_inferred', 'researcher'), `edited` (0/1), `span_id` NULL | **derived** and rebuildable from the head version; rows exist only for overlaid papers |
| `interpretation_notes` | `paper_id, schema_id` PK, `note_text` | derived: the note text recomputed from the interpretation plus the researcher notes |
| `concept_refresh` | `paper_id` PK, `reason`, `since` | "concepts out of date": stage 4 refreshes only these papers' concepts and clears them |
| VIEW `note_latest` | the old `note_current` definition (highest unlabelled version) | **extraction freshness** reads this |
| VIEW `note_current` | for an overlaid (paper, schema): the overlay's base note; otherwise `note_latest` | the **current interpretation's** note |
| VIEW `current_claims` | `interpretation_claims` for overlaid papers ∪ `extracted_claims` of `note_current` for all others | what every claim reader uses |

Why the views are split this way: stage 2 makes a new note version current as soon as its paper completes.
For an edited note that would break "nothing changes until the researcher accepts". So an overlay **pins** its
base, and `note_current` follows the pin. `work/tasks_notes.py` (`current_note`, `note_freshness`) and
`notes/save.py` (`expected_current`) are about the extraction, so they move to `note_latest`. A re-note of an
edited paper is then judged fresh or stale, and conflict-checked, on extractions alone: the new version saves
normally and becomes `note_latest`, while `note_current` stays pinned. Papers with no overlay behave exactly as
in stage 2.

Ops in `ops_json` (validated by `notes/overlay.py`):
- `{op: "edit", target: ClaimRef, set: {claim_text?, normalized_label?, claim_type?}}`
- `{op: "hide", target: ClaimRef}`
- `{op: "add", add_id: "rc_…", claim_type, normalized_label, claim_text, kind: "quote" | "assertion",
  quote?: {markdown_id, markdown_hash, start_char, end_char, quote_hash, span_id}}`
- `ClaimRef = {claim_id, fingerprint}`, where `claim_id` names the row in the base extraction and `fingerprint`
  is computed **at edit time** from the agent's original claim (§4.4). Replay therefore never needs the old
  rows, and the stable-id rule in §0 holds.

Stage 1 compatibility: 0009 only adds tables and redefines two views. With no overlays, `note_current` is
identical to before, so the migration dump stays byte-identical. `report.json` gains zero-count keys for the new
tables, and `expected.json` is regenerated with `accept.expected_template`, as stage 2 did. U1 must keep every
stage 1 migrate test green without editing `src/seedgraph/migrate/`.

### 4.2 Saving, versions, revert

- Every edit control posts one op with the `expected_version` it was rendered from. `save_edit`:
  1. checks the version (a stale form gets "this note changed in another tab; reload");
  2. validates the op: the claim type must be in the fixed vocabularies of stage 2 §4.1; a label must be
     non-empty; a quote-backed add must anchor exactly and uniquely (§5);
  3. creates the overlay on the first edit, with base = the current `note_current`;
  4. writes version n+1 with the full op set;
  5. rebuilds `interpretation_claims` and `interpretation_notes` for the paper;
  6. replaces the paper's `claim_fts` and `note_fts` rows;
  7. upserts `concept_refresh`.

  All of this happens in **one** library transaction. Search, MCP and the paper page therefore see the edit at
  once ("search updates immediately").
- **History** lists versions newest first, with action, summary and time. **Revert to version k** writes a new
  version whose op set is k's (`reverted_to = k`). Nothing is ever deleted, so a revert can itself be reverted.
- **Researcher notes box**: free text, saved as a version (`action = 'notes'`), indexed in `note_fts` with the
  note text, and shown to agents labelled "researcher notes".
- **Deleting an overlay** happens only with the paper (`library.lifecycle`, stage 1 §5). `plan_delete` already
  reserves "has researcher edits". U1 fills it in from `note_overlays`.

### 4.3 Current interpretation and trust labels

`notes/interpretation.current(lib, paper_id, audience)` is the one read path for the paper page, MCP
`paper_show`, MCP and CLI `search` (claims and notes), and the views. It returns the claims of `current_claims`
with:
- **trust**:
  - **paper states**: an agent claim with an anchored quote;
  - **agent inferred**: an agent claim without one (`ambiguous`, inferred, or `not_found` scalars);
  - **researcher**: any added claim, and any edited claim. An edited claim shows "edited by researcher" and keeps
    the agent's original one click away. A researcher assertion shows "researcher assertion — not the paper's
    words". A quote-backed researcher claim shows its quote as the paper's words and its claim text as the
    researcher's.
- **hidden** claims are excluded for every audience except the researcher's own "show hidden" toggle;
- for the **agent audience**, stage 1's rules still apply (labelled versions and `hides_notes` items hide notes).

MCP output gains `trust`, `edited_by_researcher` and `researcher_notes` fields. The consultation skill
(`agent_setup/seedgraph_v3_skill.md`) gains three rules: follow the researcher's version on conflict and say
so; never present a researcher assertion as the paper's text; say "researcher" when citing researcher
content. Past traversal records (stage 5) store the claim text they used and the `(overlay_id, version_no)`,
so later edits never rewrite them.

### 4.4 Claim identity across extractions

`claim_id` is minted fresh at every save, and `field_key` carries a position (`assumptions[3]`), so neither is
stable. A **fingerprint** is computed from the claim and its markdown:
`claim_type`, `field_root` (the field name without its index), `label_fold` (`semantic.merge.normalize`, the
same normaliser stage 4's concepts use), `text_tokens` (the folded claim text as a token set), and, for an
anchored claim, `quote_hash`, `markdown_hash` and `(start_char, end_char)`.

**Same claim** means the same `claim_type`, compared on the agent's original type even if the researcher
re-typed the claim, plus one of these tiers:

| tier | rule |
|---|---|
| S | a scalar field (`research_question`, `main_contribution`, `setting`, `estimand_or_target_object`): same `field_root`, which is unique per note |
| 1 | quotes overlap (character IoU ≥ 0.5 on the same markdown; across markdowns, after relocating the old quote with `anchor.reanchor`) **and** `label_fold` equal |
| 2 | quotes overlap **and** token Jaccard of the claim texts ≥ 0.5 |
| 3 | neither side anchored, `label_fold` equal **and** token Jaccard ≥ 0.5 |

**Matching** is one-to-one and deterministic. Candidate pairs are sorted by (tier, −overlap, −Jaccard, new
`field_key` order, new `claim_id`) and taken greedily. If two candidates tie on everything except the final id
tie-breakers, the pair is marked **ambiguous** and shown for the researcher to choose. It is never picked
silently. The thresholds are constants in `notes/identity.py`, tested on invented fixtures and reported by the
replay check (§10.2).

**Unmatched edits are never dropped.** An edit whose claim has no match appears in the replay view under "your
edits that no longer match a claim". The choices are: keep it as a researcher claim (quote-backed if its quote
still anchors exactly in the new markdown, otherwise an assertion); attach it to a chosen new claim; or discard
it. If the researcher does not choose before accepting, the default is **keep**. A hide whose claim did not
reappear is reported ("n hidden claims are not in the new extraction") and needs no decision.

### 4.5 Replay on re-extraction

- **Trigger**: an overlaid paper whose `note_latest.version_no` > the overlay base's version. The notes pane
  shows "Newer extraction available — compare". Extraction freshness stays a property of the extraction
  (`note_latest` and its markdown).
- **`replay.plan(lib, paper_id) -> ReplayPlan`** is pure and writes nothing. It returns a side-by-side list:
  - **kept edits**: matched, showing the new agent text beside the researcher's version, which wins;
  - **still hidden**: matched hides;
  - **added claims kept**: each quote-backed one is re-checked for an exact unique anchor in the new markdown,
    and one that no longer anchors is flagged and offered as an assertion;
  - **unchanged claims**: matched with no op; they take the new extraction's wording;
  - **new agent claims**: unmatched new claims, each with accept/dismiss, plus "accept all" / "dismiss all";
  - **gone**: unmatched old claims with no op ("no longer in the new extraction");
  - **ambiguous** matches and **unmatched edits** needing a decision.

  The plan carries a fingerprint of the overlay head and of `note_latest`.
- **`replay.accept(lib, plan, decisions)`** refuses a stale plan. It then writes one new overlay version with
  `action = 'accept_extraction'`, base = the new note, ops re-keyed to the new claim ids, and dismissed new
  claims recorded as `hide` ops. It rebuilds the interpretation and FTS and marks `concept_refresh`, all in one
  transaction.
- **Nothing changes until accept**: before it, `note_current`, `current_claims`, search, MCP and the paper page
  are unchanged. §10.2 asserts this byte for byte.
- An earlier accept can be undone by reverting to the version before it, which restores the old base.

## 5. Quote-backed added claims

1. In the PDF pane the researcher selects a passage in the text layer. `select.js` reads
   `window.getSelection().toString()` and the page number of the selection's start node, and shows "Add claim
   from selection".
2. A POST to `/papers/{id}/anchor` with `{page, text}` calls `library.pages.anchor_selection`. It:
   - folds the selection (§3.2);
   - folds the markdown slice for that page (from the delimiters), widened by one page on each side for
     selections across a page break;
   - requires **exactly one** match of the folded selection;
   - maps the match back to raw markdown offsets through the index map, widens it to the enclosing word
     boundaries, and returns `markdown[start:end]`.

   The quote stored is always the **verbatim markdown slice** (the anchor of record, `anchor.assert_invariant`),
   never the PDF's text.
3. The add form shows that exact slice for confirmation, with claim type (fixed vocabulary), label and claim
   text. Saving calls `save_edit` with an `add` op. Inside that transaction, `notes/spans.ensure_span` creates or
   reuses the evidence span (filling `page_start` from the delimiters), and `anchor.locate_quote` confirms the
   slice is unique in the whole markdown.
4. **Fallbacks**: if there is no match, or more than one ("the selection appears 2 times on this page; select a
   longer passage"), or the paper has no page markers, the researcher selects in the **text view** instead. It
   renders the markdown with `data-offset` attributes, so a selection maps straight to offsets. The exact
   unique-anchor rule still applies there.
5. A **researcher assertion** has no selection: a form with type, label and text, saved as `kind = 'assertion'`,
   and always labelled as the researcher's.

## 6. Adding papers in the UI

- **Upload, drop, folder** → `POST /library/add` (multipart; only `.pdf` files that start with `%PDF`; at most
  200 MB each). Files stream to `<home>/tmp/upload/<job>/` under a generated name; the uploaded filename is never
  used as a path. A typed folder path must be an existing directory, is walked for `*.pdf`, and is read in place.
  Each file becomes a job for `library.ingest.add_pdfs`, which:
  1. hashes the file; a file already attached to a paper reports that paper and stops;
  2. scans the first pages for identifiers (`tasks_metadata.deterministic_identifiers`);
  3. matches an existing paper on a strong id only, so a **cited-only paper becomes held** when its PDF arrives;
  4. otherwise creates a paper with no title and a `metadata_needed` item;
  5. calls `library.files.attach_pdf` (standing check plus conversion queue, one transaction, stage 1 §7.4).
- **DOIs** → `library.ingest.add_dois`. It normalises each DOI. A DOI already in `identifiers` reports the
  existing paper. Otherwise it calls `ProviderChain.by_doi` (OpenAlex, then Crossref, then Semantic Scholar;
  keys from environment variables, stage 2 open question 4; answers cached in `provider_cache.db`) and creates a
  **cited-only** paper with `title_source` = the provider. Then it tries `acquisition.fetch.fetch_oa`, which
  covers open-access copies only and never paywalled ones. A fetched PDF goes through `add_pdfs` with
  `access_class = open_access` and `acquisition_method = oa_fetch`, so the paper becomes held. All network calls
  run outside write transactions.
- **Jobs** (`web/jobs.py`): one in-process `JobRunner` with a small thread pool:
  - **conversion**: `pool.size` threads calling `library.files.convert_file`, so both warm converters stay busy;
  - **metadata**: one thread for provider lookups and `metadata_verify`. When Ollama is reachable it also creates
    a `metadata` batch for new untitled papers and runs `work.ollama_worker.run_ollama_worker` on it. When Ollama
    is not reachable, the `metadata_needed` item stays and the Library says so.

  `JobRunner` itself is generic: `submit(kind, fn, *args) -> job_id`, `jobs(active_only)`. U7's library routes
  wire in the ingest, DOI and metadata jobs.
  Each thread opens its own `Library`. The job registry is in memory (state, step, message, paper ids). The
  durable truth is the database: `conversion_queue`, `paper_files`, attention items. On start the server
  resumes anything left in `conversion_queue`, so a restart loses nothing. The browser polls `/library/jobs`
  (an htmx partial) every 2 s while any job runs.
- **One process owns the converters.** `get_pool()` is one pool per process, and stage 0 has no lock across
  processes. While `serve` runs, it holds `<home>/run/converter.lock` (its PID, released at exit, and ignored
  when that PID is gone). `seedgraph convert` and `library reconvert` refuse while the lock is held: "seedgraph
  serve is running; add or re-convert PDFs in the UI, or stop it first". So two processes never start
  converters on the same GPUs.
- **After conversion**: `convert_file` already re-checks an `unknown` PDF check on the markdown and runs the text
  gate. After metadata fills a title, the job calls `library.files.recheck_pdf`, because the check at attach
  time had no title to look for. The result shows in the job strip and on the paper page. A miss is a flag, never
  a block.

## 7. Ollama metadata quality fixes

All of these live in `library/metadata_quality.py` (pure) and `library/metadata_verify.py`, and are used by
`work/tasks_metadata.py` and the ingest jobs:
- **Clean the title before saving**: strip footnote and affiliation marks (`†`, `‡`, `§`, `¶`, `*`, trailing
  superscript digits and `<sup>…</sup>`), collapse whitespace, and trim trailing punctuation.
- **Reject generic titles**: a stoplist ("review", "introduction", "contents", "table of contents", "abstract",
  "front matter", "preface", "acknowledgements", "editorial", "article", "untitled", "working paper",
  "chapter N"), a title equal to the venue, or one shorter than 3 characters. The value is **not saved**. A
  `metadata_generic_title` attention item records the proposed value and its source, and the item stays open.
- **Verify a DOI**. When Ollama returns a DOI (or the deterministic scan finds one), `metadata_verify.verify`
  runs after the submit commits, outside the routine's write transaction. It looks the DOI up with
  `ProviderChain.by_doi` and compares titles with the tolerant matcher (`library.titlecheck`):
  - provider title ≈ Ollama title → **prefer the provider's values** (title, year, venue, authors), with
    `title_source = 'openalex'` or `'crossref'`;
  - they differ, but the provider title is found in the PDF's first pages → the provider wins (Ollama misread
    the title);
  - they differ and the provider title is not in the PDF → the DOI likely came from the reference list. The DOI
    row (`source = 'ollama'`) is removed, the Ollama title stays as `title_source = 'ollama'` (unverified), and a
    `metadata_doi_mismatch` item is raised.
- **Provenance on the paper page**: "Title from: v2 / OpenAlex (verified by DOI) / Crossref (verified by DOI) /
  Ollama (unverified) / you". The `metadata_json` evidence (the DOI, the matcher score) is shown on hover.
- `seedgraph library refetch-metadata [--malformed] [--paper ID]`, a stub since stage 1, is implemented on
  `metadata_verify` (open question 5 of stage 1).

## 8. Security

- **Bind**: uvicorn binds `127.0.0.1` only, and no option changes it.
- **Client check** (`LoopbackGuard`, the outermost ASGI middleware): a request whose `scope["client"]` host is
  not `127.0.0.1` or `::1` gets 403. This is defence in depth against port forwarding and proxies.
- **Host check** on **every** request, GETs included, because PDF and note routes expose private text to DNS
  rebinding. `Host` must be `127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>`. Anything else gets 421 with
  a plain message.
- **CSRF without accounts or cookies**:
  - at start the server mints a per-process secret token (`secrets.token_urlsafe(32)`);
  - the base template puts it in `<meta name="csrf-token">`, every form carries it as a hidden field, and
    `csrf.js` adds it as `X-CSRF-Token` to every htmx request;
  - every non-GET/HEAD request must carry the matching token (`hmac.compare_digest`), and, when the headers are
    present, `Origin` must equal the server's own origin and `Sec-Fetch-Site` must be `same-origin` or `none`;
  - a cross-site page cannot read the token, because same-origin policy and the Host check block it;
  - the token rotates on restart, so a stale tab gets 403 and the banner "seedgraph restarted; reload the page".
  This needs no session and **no cookies at all**.
- **GET never mutates.** A route-table test asserts that every route with side effects is POST-only.
- **Headers**: `Content-Security-Policy: default-src 'self'; script-src 'self'; worker-src 'self';
  img-src 'self' data: blob:; style-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'`,
  plus `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`. There are no inline scripts (the
  token is read from the meta tag). htmx's injected indicator style is turned off (`htmx.config.includeIndicatorStyles
  = false`; the CSS lives in `app.css`). Whether the pinned PDF.js build needs more (e.g. `wasm-unsafe-eval` for
  image decoders) is settled when U4 is built, not assumed. The Playwright pass (U10) fails on any CSP violation
  in the console and on any request that leaves localhost.
- **Content**: Jinja2 autoescape is on. Markdown and note text render as escaped text, never as HTML. The PDF
  route serves only library blobs looked up by `paper_id`. No route takes a file path, except the folder-ingest
  form, which only reads PDFs.
- **Gone**: session cookies and content-access gates (ADR-0012). `access_class` stays as provenance.

## 9. Deletions in this stage

The CLI groups in §1.3 go first. Packages go only when nothing in the core still imports them. Instead of
deleting modules by guesswork, U9 adds **`tests/test_reachability.py`**: it scans imports statically from the
entry points (`cli`, `mcp.server`, `web.app`, the converter worker) and fails if any module under
`src/seedgraph/` is unreachable and not on a short allowlist (the calibration scripts). The table records the
decisions known now. The test catches the rest.

| v2 code | depends on it today (core) | handling |
|---|---|---|
| CLI groups `corpus`, `cite`, `sections`, `spans`, `lens`, `concepts`, `graph`, `cache`, `project`, `review`, `doctor`, `work-read` | nothing in the core | delete with their tests (§1.3) |
| `acquisition/{service,resolve,walk,backfill,bridge,preview,links,doctor_reconcile}.py` | `extraction/metadata.py`, `citation/parsed_bib.py` (both go too) | delete |
| `acquisition/fetch.py` | `providers/*` (`OACandidate`), `library/ingest.py` (new) | **keep** |
| `providers/` | `library/ingest.py`, `library/metadata_verify.py` | **keep** |
| `project/identity.py` | `library/identity.py`, `library/titlecheck.py`, `migrate/plan.py` | **keep in place** (pure); the rest of `project/` goes |
| `citation/resolve.py` | `library/titlecheck.py` (stopword set only) | move `_OVERLAP_STOPWORDS` into `titlecheck.py`; delete `citation/` |
| `sections/parser.py` | `library/files.py`, `golden/metrics.py` | **keep**; `sections/store.py` goes if unreachable |
| `spans/store.py` | `anchor.py` (type hint only) | replace the hint with a small Protocol in `anchor.py`; delete `spans/` |
| `fts/search.py` | `library/search.py`, `migrate/parity.py` | **keep**; `fts/schema.py` goes if unreachable |
| `semantic/merge.py` | `migrate/plan.py`, `notes/identity.py` (new) | **keep**; the rest of `semantic/` and `graph/` go with `concepts`/`graph` (deviation, §1.3) |
| `db/migrations.py` | `library/store.py` | **keep**; `db/{adapter,engine,bootstrap,connection,models_project,project_models,fts}.py` and `db/schema/project` go if unreachable (migrate uses its frozen `v2_schema/`) |
| `extraction/metadata.py` | nothing after the trim | delete (the deterministic id scan already lives in `work/tasks_metadata.py`) |
| `lenses/{schema,registry,prompt}.py` | `work/tasks_lens.py` | **keep** |
| `cache_access.py`, `doctor.py`, `doctor_spans.py`, `work_read.py`, `work_read_cli.py`, `run.py`, `progress.py`, `display.py` | v2 CLI only | delete |
| `sqlmodel`, `networkx` dependencies | v2 models, `graph/` | drop from `pyproject.toml` when the reachability test shows no importer |

The v2 web UI was already deleted in stage 2. Nothing from v2's `web/` comes back; leftover `__pycache__`
folders are removed.

## 10. Acceptance plan

All runs use the real v3 home, which is disposable (§0), after a fresh `migrate accept`. Its dump is the
reference. Reports go to `<home>/ui/acceptance/<run>/report.md`.

### 10.1 Test layers: Playwright for the walk-through, HTTP tests for everything else

- **HTTP-level tests** (Starlette `TestClient`, the default suite, fast): every route's status and partial
  swaps, CSRF and Host refusal, views against fixture libraries, overlay, replay and identity unit tests, and
  ingest with a fake converter and fake providers.
- **Playwright** (a dev-only dependency group `browser`, pinned in `uv.lock`; Chromium installed once with
  `playwright install chromium`; tests marked `browser`, opt-in). A real browser is the only way to test PDF.js
  rendering, text-layer highlighting, selection capture and `fold.js`. **Recommendation: Playwright for the
  walk-through and the highlight check, HTTP tests for the rest.** htmx-level HTTP tests alone cannot show that a
  highlight lands or that a selection anchors.

### 10.2 Checks

1. **Scripted walk-through** (Playwright, `seedgraph serve --root <v3 home> --no-open`) on the migrated
   `diff_in_diff` graph:
   1. Library → drop one PDF. It is an open-access PDF the researcher chose, not yet in the library, identified
      by its hash in `<home>/ui/acceptance/inputs.json`.
   2. The job strip goes from converting to converted, the standing check result shows, and metadata fills.
   3. Add it to `diff_in_diff`, then Write notes. The batch page shows the copyable commands. A Claude Code
      subagent works the batch through the CLI mirror (`seedgraph-v3 work claim/submit`, stage 2 §10.3). The
      page's polling shows claimed → done with anchored/total quotes.
   4. Open the paper page. Click three quotes: each scrolls to its page and a `mark.sg-hl` appears on that page.
   5. Re-word, re-label and re-type one claim; hide one; add an assertion; add a quote-backed claim by selecting
      text in the PDF pane.
   6. Search: the edited text is found, the hidden claim is gone, MCP `paper_show` labels the researcher
      content.
   7. Revert to the version before the hide; the claim is back. History lists every version.
   8. From the Library list, the PDF button opens the PDF.
2. **Edits survive re-extraction** on a migrated paper with a current note (chosen in `inputs.json`):
   1. Add, edit and hide claims over HTTP.
   2. Snapshot the interpretation from `notes.interpretation.current` (agent and researcher audiences): claim
      ids, text, labels, types, trust, the hidden set and the researcher notes. Also snapshot the **sets** of
      (id, text) returned by `library search` for three fixed queries. Scores and order are not compared:
      bm25 statistics shift when the re-note adds FTS rows (stage 1 §8.2).
   3. Create a `notes` batch with `replace` for the paper. A Claude Code subagent re-notes it through the CLI
      mirror.
   4. Assert:
      - the interpretation snapshot is **byte-identical**, so nothing changed before accept;
      - the search sets are equal;
      - the claims in MCP `paper_show` are unchanged;
      - the freshness indicators changed as expected ("Newer extraction available" shows; `note_latest` is
        new);
      - the replay plan lists the edit as kept, the hide as still hidden and the added claims as kept.
   5. Accept all.
   6. Assert all three kinds of edit survive, new agent claims are present, `note_current` now points to the new
      base, and the matching tiers used are reported.
3. **Highlights land**:
   - `seedgraph library pagecheck --set proto5 --mode prototype` and `--set proto31 --mode prototype` against
     `highlight_expected.json`. Gate: proto5 page-assigned equals the expected total. proto31 counts the same
     claim population as the prototype, and its page-assigned count **and** ratio are each at least the
     prototype's. Page-wrong = 0. Papers without markers are listed.
   - The `current` mode and the verified tier are reported.
   - A Playwright pass over proto5 runs `pdfpane.js`'s finder for every quote and reports its agreement with the
     server-side verified tier.
4. **Localhost and CSRF**:
   - ASGI tests: a non-loopback client gets 403; a foreign Host gets 421 on GET and POST; a POST without a token,
     with a wrong token, or with a foreign Origin gets 403; a correct POST gets 200.
   - A route-table test: no GET mutates.
   - A live test starts `serve`, asserts the listening socket is `127.0.0.1`, and asserts a connection to the
     machine's LAN address is refused.

## 11. Build slices

Each slice is sized for one agent, written test-first, and owns its files exclusively. Shared files
(`pyproject.toml`, `cli.py`, `commands/__init__.py`, `.gitignore`, `uv.lock`) are touched only by the slice
named for them. Tests use invented text and synthetic fixtures (stage 1 library fixture helpers). No private
text or PDF is ever a fixture.

| # | Slice | Depends on | Owns | Tests first |
|---|---|---|---|---|
| U1 | **Overlay core**: `0009_note_overlays.sql`, views, `notes/overlay.py`, `notes/interpretation.py`, FTS refresh, `concept_refresh`; `tasks_notes`/`save` move to `note_latest`; `lifecycle.plan_delete` reports edits | — | `db/schema/library/0009_note_overlays.sql`, `src/seedgraph/notes/{overlay,interpretation}.py`, edits to `work/tasks_notes.py`, `notes/save.py`, `library/lifecycle.py`, `tests/notes/test_overlay*.py`, `test_interpretation*.py` | no overlay → `note_current` unchanged and migrate dump identical; version per save; revert is a new version; stale `expected_version` refused; hidden excluded; trust labels; FTS updated in the same transaction; a re-note of an edited paper leaves `note_current` pinned and `note_latest` new |
| U2 | **Claim identity and replay**: `notes/identity.py`, `notes/replay.py` | U1 | those files, `tests/notes/test_identity*.py`, `test_replay*.py` | each tier; one-to-one; deterministic ties; ambiguous flagged; re-typed claims match on the original type; cross-markdown re-anchor; unmatched edit default keep; plan writes nothing; stale plan refused; accept re-keys ops; dismissed → hide |
| U3 | **Text fold, pages, pagecheck**: `library/{textfold,pages,pagecheck}.py`, `static/js/fold.js`, shared vectors, `library pagecheck` subcommand | — | those files, `commands/library.py` (pagecheck only), `tests/library/test_textfold*.py`, `test_pages*.py`, `test_pagecheck*.py`, `tests/fixtures/textfold_vectors.json` | fold vectors (ligatures, soft hyphen, line-end hyphenation, markup, math); index map round-trip; marker / reanchored / none; never guessed; selection anchor unique or refused; tiers incl. page-wrong; prototype vs current modes on a fixture |
| U4 | **Web shell and security**: `web/{app,security,jobs}.py`, `base.html`, sidebar, static layout, vendored htmx and PDF.js with `VERSIONS.json`, `commands/serve.py`, deps | — | `src/seedgraph/web/{__init__,app,security,jobs}.py`, `web/templates/{base,_sidebar,_flash}.html`, `web/static/{app.css,js/csrf.js,js/layout.js,vendor/}`, `commands/serve.py`, `pyproject.toml`, `uv.lock` | loopback, Host and CSRF matrix; no cookies set; CSP header; vendor hashes; JobRunner resumes the conversion queue; per-thread connections |
| U5 | **Ingest and metadata quality**: `library/{ingest,metadata_quality,metadata_verify}.py`, `work/tasks_metadata.py` (clean, reject), a post-commit hook (`TaskType.after_commit`, called by `work/routine.py` after the submit transaction), `work/ollama_worker.py`, `library refetch-metadata`, the converter lock in `convert`/`reconvert` | — | those files, `work/{routine,tasks}.py` (the hook only), `library/reconvert.py` and `commands/convert.py` (the lock check), `commands/library.py` (refetch-metadata only; U3 edits a different function in the same file, so U3 and U5 merge in sequence), `tests/library/test_ingest*.py`, `test_metadata_*.py` | cited-only → held on a strong id; new paper + `metadata_needed`; duplicate file reported; DOI paste cited-only, then held on an OA fetch (fake chain and fetch); generic title → item, not saved; footnote marks stripped; DOI verify prefers provider values; reference-list DOI removed with an item |
| U6 | **Interpretation consumers**: `library/search.py` and MCP `tools_library.py` read `current_claims` / `interpretation`; skill rules; `work.estimate` | U1 | `library/search.py`, `mcp/tools_library.py`, `agent_setup/seedgraph_v3_skill.md`, `work/estimate.py`, tests | search finds edited text and added claims, not hidden ones; agent audience rules kept; MCP fields; stable order kept |
| U7 | **Library, Home, Batch, Review, Graphs, Settings pages**: views + routes + templates | U4, U5, U6 | `web/views/{home,library,batch,review,graphs,settings}.py`, `web/routes/{home,library,batches,review,graphs,settings,placeholders}.py`, matching templates and partials | views on fixture libraries; each POST calls its one deep function; polling partials; pause labels match `work.state`; resolve actions incl. `accept_text` |
| U8 | **Paper page, PDF pane, editing UI** | U1, U2, U3, U4, U6 | `web/views/paper.py`, `web/routes/{papers,notes,files}.py`, paper templates and partials, `static/js/{pdfpane,select}.js` | header and badges; layout toggle; quote locators rendered; edit/hide/add/revert round-trips; replay view and accept; text-view fallback; PDF route serves blobs by `paper_id` only |
| U9 | **CLI trim, deletions, reachability**: §1.3 and §9 | U7, U8 | `cli.py`, `commands/__init__.py`, every deleted module and test, `tests/test_reachability.py`, `pyproject.toml` (dependency drops, after U4) | smoke test per remaining command; reachability green; every `seedgraph-v3 <command>` named in `agent_setup/*.md` resolves in the Typer app; full suite green |
| U10 | **Acceptance harness**: Playwright group, walk-through, replay check, highlight browser pass, live bind test | U8, U9 | `tests/acceptance/`, `pyproject.toml` (`browser` group, after U9) | runs opt-in; writes reports to the home |

**Parallelism (2–3 at a time).** Wave 1: U1, U3, U4. Wave 2: U2, U5, U6. Wave 3: U7, U8. Then U9, then U10
and the real acceptance run. The critical path is U1 → U2 → U8 → U9 → U10. The full suite stays green after
every slice. U4 and U9 both touch `pyproject.toml`, and U5 and U3 both touch `commands/library.py`, but never in
the same wave.

## 12. Open questions for the researcher

Each has a recommendation (*R*). Question (a) of the brief is decided (§0).
1. **Trust label for an edited agent claim**: show "researcher" (edited), or keep "paper states" with an edit
   mark? *R:* "researcher", with "edited from the agent's claim" and the original one click away. The quote
   still shows as the paper's words.
2. **Unmatched edits at accept**: default keep, or force a decision? *R:* default keep as a researcher claim.
   Nothing the researcher wrote is lost silently.
3. **Matched claims you did not edit**: should they take the new extraction's wording on accept? *R:* yes. That
   is what a fresh extraction is for, and the side-by-side view shows every change before accept.
4. **The highlight gate**: the prototype's figures are page-assigned counts, never checked against the PDF.
   *R:* gate on what the spec names (page assigned, prototype mode) plus "no page wrong"; report the
   verified tier and the current mode without gating them in stage 3.
5. **PDFs with no identifiable DOI from a folder drop**: *R:* create new papers with `metadata_needed` and run
   Ollama metadata automatically when it is reachable. A title Ollama finds is verified by DOI where one exists.
6. **DOI paste fetches open-access PDFs automatically**: *R:* yes, open access only, with the source shown;
   nothing paywalled is ever fetched.
7. **Deleting the `concepts` and `graph` CLI groups in stage 3**, one stage early: *R:* accept. They read only
   v2's `project.db` and serve no v3 data; v2 keeps them in the main checkout.
8. **"Maintenance" in the CLI core**: *R:* read it as the existing `library` group (status, check, attention,
   textcheck, reconvert, pagecheck, refetch-metadata, prune-replaced).
9. **Playwright as a dev-only dependency** (it downloads a Chromium build once): *R:* yes, in a `browser`
   dependency group, opt-in tests only.
10. **Editing metadata in the UI** (`title_source = 'user'`): *R:* yes, a small form on the paper page. Researcher
    values are never overwritten by Ollama or a provider.
11. **Citation review candidates**: nothing in v3 generates new ones until reference lists are resolved again.
    *R:* stage 3 shows migrated decisions only, and stage 4's guided growth builds the candidate queue together
    with its producer.
12. **`seedgraph convert` while the UI runs**: *R:* refuse, and point to the UI, so converters never compete for
    the same GPUs. Using the UI's job queue from the CLI is not worth a second channel.
