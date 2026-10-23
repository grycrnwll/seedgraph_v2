# v3 Stage 1 design: the library and the migration

Status: design for build. No code yet. This document covers v3 stage 1: the library data model, context graph
storage, paper identity, lifecycle rules, the v2 → v3 migration, the standing PDF check, search parity, the
CLI layout, and the build slices. It uses the glossary terms from `CONTEXT.md` (library, context graph, held
paper, cited-only paper, researcher-edited note) and the deep-module vocabulary from `codebase-design` (module,
interface, depth, seam, adapter).

The repo is public. Real counts, query sets, attention lists and reports live in the v3 home, never in this
document or the repo.

## 1. Store layout

### 1.1 Decision: one `library.db` that absorbs stage 0's cache tables

The library is a single SQLite file, `library/library.db`. Stage 0's `cache.db` tables (`source_files`,
`conversion_runs`, `markdown_documents`, `cache_events`) move into it with their DDL unchanged. Blobs stay as
content-addressed files next to it.

Reasons:
- **Real foreign keys.** v2's soft ids across `project.db`/`cache.db` let the defects the migration must repair
  arise unchecked. In one file, `paper_files → source_files → markdown_documents → sections/spans` is enforced.
- **One transaction** for attach PDF + PDF check + queue conversion.
- **Deletion test.** `cache.db` existed so projects could share conversions; the library now owns sharing, so a
  separate cache store would be a pass-through.
- **One canonical dump and one backup unit.** ADR-0008 rejects one DB for *all context graphs*, not merging the
  per-paper stores, which share one lifecycle (the paper's).

The exception is **`library/provider_cache.db`**, a separate, disposable file holding stage 0's `provider_cache`
DDL. Provider responses are a TTL cache with `fetched_at` churn. They are not library facts, and they are
excluded from the canonical dump. The migration copies v2's rows across so that offline metadata lookups (§7.3)
work.

Consequence for stage 0 code: `cache/ingest.py` and `cache/convert.py` gain a store-location parameter (an open
library connection plus a blob root) instead of resolving `<home>/cache/cache.db` themselves. The `convert`
slice (§11, S4) does this re-pointing, so a conversion is recorded in exactly one place. Old v2-shaped code that
still calls them with `root=` keeps its `cache.db` behaviour on its own test fixtures until it is deleted
(§9). `golden/v2.py` reads v2's `cache.db` and is unaffected.

### 1.2 One DB per context graph

Each context graph is `graphs/<slug>/graph.db`. It holds only graph-owned rows (§3) and refers to library paper
ids softly. Readers `ATTACH` the library read-only to join. Deleting or rebuilding a graph deletes one folder.

### 1.3 The v3 home

```
~/.seedgraph-v3/                      (resolve_home(); --root / $SEEDGRAPH_HOME override)
  golden_set.json                     stage 0, never touched by stage 1
  golden_reports/                     stage 0, never touched by stage 1
  library/                            OWNED BY: library (swapped in whole by migrate)
    library.db
    provider_cache.db
    pdfs/<sha256>.pdf
    markdown/<sha256>.md
  graphs/                             OWNED BY: library (swapped in whole by migrate)
    <slug>/graph.db
  migration/                          OWNED BY: migrate
    staging/<run_id>/                 a build in progress: library/, graphs/, v2_answers/
    runs/<run_id>/                    report.md, report.json, attention.md, dump.jsonl.gz,
                                      dump.sha256, v2_guard_before.json, v2_guard_after.json
    replaced/<timestamp>/             previous library/, graphs/, v2_answers/ moved aside, never deleted
    acceptance/expected.json          expected counts and the known-difference allowances (researcher-owned)
    parity/queries.json               fixed query set (researcher-owned)
    parity/v2_expected.json           v2 results captured from the snapshot
  v2_answers/                         OWNED BY: migrate (swapped in whole)
    <slug>/<answer_id>.md, index.md
```

**Hard rule: the migration never deletes or rewrites anything outside `migration/`, except for the three
folders it owns (`library/`, `graphs/` and `v2_answers/`).** Swap procedure:
1. Build everything into `migration/staging/<run_id>/`, which is on the same volume, so renames are atomic.
2. Verify the staging build: integrity check, FTS row counts, canonical dump written.
3. If a live `library/` exists, refuse unless (a) its `library_meta.origin` is a migration run **and** (b)
   `library_meta.dirty = 0`. Any non-migration write sets `dirty = 1`. `--replace-dirty` overrides this and
   says what will be moved.
4. Rename the live `library/`, `graphs/` and `v2_answers/` into `migration/replaced/<timestamp>/`. Then rename
   the staging folders into place. A crash between renames leaves both sets on disk, and `migrate` reports the
   half-swap on its next start.

`golden_set.json`, `golden_reports/` and any unknown file in the home are never listed, moved or deleted.
`migration/replaced/` is cleaned only by an explicit `seedgraph library prune-replaced`.

## 2. Library schema

Conventions:
- Every table is authored in numbered `db/schema/library/NNNN_*.sql` files (decision D6 carries over). No
  `create_all`.
- Ids are TEXT and either content-derived or carried verbatim from v2. No new uuid4 ids are minted by the
  migration.
- Every row the migration creates has `origin` = `'v2:<slug>'` or `'migration'`, so later writes are
  distinguishable.
- Timestamps carried from v2 keep v2's value. The migration's own timestamps live only in `migration_runs` and
  in the columns the dump excludes (§4.5).
- No `rowid` tie-breaks anywhere. Ordering is always by explicit columns.

### 2.1 Tables

| Group | Table | Key columns / notes | From v2 |
|---|---|---|---|
| meta | `schema_migrations` | as v2 | runner |
| | `library_meta` | `key, value`: `origin`, `dirty`, `migration_run_id`, `v2_home` | new |
| | `migration_runs` | `run_id, started_at, finished_at, v2_fingerprint, dump_sha256, code_version` (excluded from the dump) | new |
| | `v2_id_map` | `v2_project, v2_table, v2_id → v3_table, v3_id` (traceability; used by parity) | new |
| papers | `papers` | `paper_id` ('lp_' + 20 hex, §4.1), `title, title_source, authors_json, year, venue, abstract, oa_status, created_at, origin` | `works` (merged) |
| | `identifiers` | `paper_id, id_type, id_value, source`; UNIQUE(`id_type, id_value`); types doi/openalex/arxiv/s2/ssrn | `identifiers` + works id columns |
| | `paper_status` (VIEW) | `held` iff a `paper_files` row exists; otherwise `cited_only` | derived |
| files | `source_files` | stage 0 DDL verbatim (`'sf_'+sha256`) | `cache.source_files` |
| | `conversion_runs` | stage 0 DDL verbatim; `converter_version` per run | `cache.conversion_runs` (no `pending` rows) |
| | `markdown_documents` | stage 0 DDL verbatim (`'md_'+sha256`) | `cache.markdown_documents` |
| | `cache_events` | stage 0 DDL verbatim (file/conversion log) | `cache.cache_events` |
| | `paper_files` | `paper_id, source_file_id, markdown_id NULL, role ('primary'), pdf_check_status, pdf_check_json, attached_at`; PK(`paper_id, source_file_id`); **many-to-many**, so a shared PDF can exist while flagged | `work_source_files` |
| | `paper_converter` (VIEW) | paper → primary file → current markdown → its run's `converter_version` | derived |
| text | `document_sections` | v2 DDL keyed by `markdown_id`, **`work_id` dropped** (text belongs to a markdown, not a paper) | `document_sections` (dedupe by id) |
| | `evidence_spans` | v2 DDL keyed by `markdown_id`, **`work_id` dropped**; deduped (§4.4) | `evidence_spans` |
| | `span_fts` | FTS5 as v2 (unicode61, remove_diacritics 2, prefix 2 3) without `work_id`; papers are reached via `paper_files.markdown_id` | rebuilt |
| notes | `extraction_runs` | v2 columns incl. chunk/lens columns; `work_id` → `paper_id` | `extraction_runs` |
| | `structured_notes` | v2 columns plus `version_no` (1..n per `paper_id, schema_id`), `version_label` (NULL, `'wrong_pdf'`, `'other_markdown'`) | `structured_notes` |
| | `note_current` (VIEW) | highest `version_no` per `(paper_id, schema_id)` whose `version_label` IS NULL | replaces `CURRENT_CLAIMS_CTE` |
| | `extracted_claims` | v2 columns (incl. `source_chunk_index`); `work_id` → `paper_id` | `extracted_claims` |
| | `claim_spans` | quote anchors, as v2 (span ids remapped by dedupe) | `claim_spans` |
| | `claim_fts`, `note_fts` | FTS5 as v2 | rebuilt |
| | *(stage 3)* `note_overlays`, `overlay_versions` | ADR-0011 overlay on a specific `extraction_run_id`. Reserved names, not created in stage 1. `note_current` becomes "extraction + overlay" in stage 3 | — |
| citations | `reference_entries` | v2 columns; `citing_work_id` → `citing_paper_id`, `resolved_work_id` → `resolved_paper_id`; plus `ordinal` | `reference_entries` (one list per paper, §6.7) |
| | `citation_links` | `source_paper_id, target_paper_id, provenance, confidence, reference_id`; UNIQUE(`source, target, provenance`); v2's `run_id` dropped (library fact) | `citation_edges` (deduped) |
| | `citation_review_decisions` | `reference_id, decision ('reject'), rejected_record_json, reason, decided_at, origin` | resolved `citation_resolution` review items |
| identity | `duplicate_decisions` | `paper_a, paper_b` (a < b), `verdict ('same', 'different')`, `decided_by ('researcher', 'rule')`, `reason, decided_at, origin` | resolved `duplicate_candidate` items + migration merges |
| lenses | `lens_definitions` | `lens_id, name, object_type, status, definition_hash, definition_yaml, created_at` (the YAML text lives in the DB, not in a project folder) | `lenses` + `projects/<slug>/lenses/*.yaml` |
| | `lens_outputs` | v2 columns; per paper and `extraction_run.lens_definition_hash` (= lens version) | `lens_outputs` |
| concepts | `concept_decisions` | `decision_id` (hash of kind+pair), `kind ('merge', 'never_merge', 'link')`, `label_a, label_b` (normalized, a < b), `decided_by ('researcher', 'agent')`, `state ('pending_apply', 'applied', 'undone')`, `note, origin, decided_at` | `concept_constraints` |
| | *(stage 4)* `concepts`, `concept_members`, `concept_links`, `claim_concepts` | reserved; built from claims in stage 4 | rebuilt, not carried |
| attention | `attention_items` | `item_id` (hash of kind + subject), `kind, paper_id NULL, subject_json, detail, hides_notes (0/1), status ('open', 'resolved'), resolution, created_at, resolved_at` | new (§7) |
| conversion queue | `conversion_queue` | `source_file_id, reason ('not_converted', 'relink_failed'), queued_at` | new |

Notes:
- **Held vs cited-only is derived**, never stored, so it cannot drift. A cited-only paper becomes held when a
  `paper_files` row is added.
- **Hidden from agents.** A note is hidden from agent-facing search when its paper has an open attention item
  with `hides_notes = 1`, or when the note has a non-NULL `version_label`. Researcher-facing views show
  everything, labelled.
- **Converter version per paper** is the `paper_converter` view over stage 0's per-run `converter_version`.
- `access_class` columns are carried as provenance only (ADR-0012).
- **Dropped, not carried**: `llm_usage_events`, `llm_key_refs`, `audit_records` (the eval harness),
  `project_graph_edges`/`edge_spans` (co-occurrence edges), `concepts`/`concept_aliases`/`claim_concepts`
  (rebuilt in stage 4), and every open `review_queue` item (ticket 13 #7).

## 3. Context graph schema (`graphs/<slug>/graph.db`)

| Table | Columns | Notes |
|---|---|---|
| `schema_migrations` | | runner |
| `graph_meta` | `key, value`: `slug, display_name, created_at, origin` | the rename offered at migration sets `display_name`; the slug is fixed |
| `graph_papers` | `paper_id` PK, `status ('member', 'frontier', 'excluded')`, `is_seed`, `reason`, `added_at`, `origin` | v2 `project_documents`: included → member, metadata_only → frontier, excluded → excluded (an exclusion is the graph's own decision) |
| `graph_lenses` | `lens_id`, `chosen_at` | which library lenses apply |
| *(stage 4)* `literatures`, `literature_members` | name, agent-inferred label, researcher rename | reserved, not created |
| *(stage 5)* `traversal_records`, `traversal_events` | | reserved, not created |

**In-graph citation view: derived, not stored.** It is `citation_links` restricted to pairs where both ends are
graph members (or frontier, for the frontier view), computed through ATTACH. A citation decision is made once
in the library and every graph sees it at once. If it proves slow, a rebuildable cache table can be added
later. It would never be authoritative.

Members must be held papers. Adding a cited-only paper as a member is refused, because it belongs on the
frontier until its PDF is added.

## 4. Identity

### 4.1 Library paper id

`paper_id = 'lp_' + sha256(key)[:20]` (hex), where `key` is the **first** of:
1. `doi:<normalized doi>`, else `openalex:<W…>`, else `arxiv:<id>`. This is the winning strong id of the merged
   group (§4.2). Normalization is `project/identity.normalize_id` plus the arXiv-DOI fold (`strong_ids`).
2. `pdf:<sha256>` of the paper's primary PDF (a held paper with no strong id).
3. `title:<normalize_title(title)>|<year or ''>` (a cited-only paper with no id).
4. `v2:<slug>:<work_id>` (no id, no PDF and no usable title; v2 ids are fixed in the snapshot).

Within a merged group that holds several ids of the winning type (alias DOIs, see §4.2), the
lexicographically smallest one wins.

**The id is minted once and stored.** It is never re-derived. A paper that later gains a DOI keeps its id, and
the new DOI is added to `identifiers`. The derivation only defines what a given migration produces, and what
`add paper` produces for a new paper.

Collisions: if a new paper's derived id already exists with different ids, append `:2`, `:3`, and so on to
the key (deterministic), and raise an attention item.

### 4.2 Merge rules (ticket 13 #3, made precise)

The migration builds merge groups over all carried v2 works (held and cited-only) with a union-find over these
keys, applied in a fixed order: DOI, then OpenAlex, then arXiv, then identical PDF hash.

- **Conflict blocks a merge.** A union that would put two *different* values of the same id type (two DOIs,
  two OpenAlex ids, two arXiv ids) into one group is not made. Precedence ordering never resolves a conflict.
  This includes the identical-PDF union: two works with different DOIs sharing one PDF stay **separate papers
  that share one `source_file`**, and the case goes on the attention list as "different papers sharing one
  PDF" (§7.3). Preprint-vs-published pairs land here as well. The researcher decides; nothing is auto-merged.
- **No-id works** merge only if (a) they share a PDF with an identified work in a non-conflicting group, or
  (b) they match an identified work on `normalize_title` **and** year (exact equality of both, and the match
  must be unique). Otherwise the work becomes its own paper and goes on **"check these"**, with the nearest
  title candidates listed (§4.3).
- **A researcher decision beats a rule.** A pair recorded as "not duplicates" in v2's resolved
  `duplicate_candidate` items is never merged by rule. A rule that would merge it is skipped and listed in the
  report.
- **Every automatic merge is recorded** as a `duplicate_decisions` row (`decided_by = 'rule'`, `reason` =
  which key joined them). The researcher can split it later.
- **Field winner within a group** (title, authors, year, venue, abstract): take the value from the group member
  in the highest-precedence project (second_test > land_value > test_proj > oct2026test > dbg_*), then the
  lowest v2 `work_id`. A value repaired in §7.3 overrides this.

### 4.3 "Check these" list

"Check these" is an attention item, `kind = 'identity_check'`, raised for:
- a no-id held work that matched nothing (it stays a separate paper);
- a no-id work whose title+year matched more than one paper (ambiguous, no merge);
- a blocked conflict merge that is not a shared-PDF case (e.g. two DOIs reached through OpenAlex);
- a rule merge that was skipped because of a researcher's "not duplicates" decision.

Each item carries the candidates and the evidence: shared keys, title similarity from the shared matcher (§7),
and years.

### 4.4 Ids of carried rows

- **Carried verbatim** (random in v2, but fixed and globally unique): `note_id`, `claim_id`,
  `extraction_run_id`, `reference_id`, `lens_output_id`, `conversion_run_id`, `source_file_id`,
  `markdown_id`. Only `work_id` is remapped to `paper_id`.
- **Sections and auto spans** have deterministic ids (`sec_`/auto span hashes over `markdown_hash` + position),
  so the same markdown in two projects yields the same id. Because sections and spans carry no paper id, two
  papers sharing one markdown (the shared-PDF defect) share its sections and spans without conflict. Dedupe by
  id. Rows that are equal apart from v2's `work_id` are one row; any other difference is a migration error.
- **Manual spans** have random ids. Dedupe by `(markdown_id, start_char, end_char, span_kind)`. The canonical
  span is the smallest `span_id`. The others go into `v2_id_map`, and `claim_spans` is re-pointed.
- **Citation links** are deduped on `(source_paper_id, target_paper_id, provenance)`, keeping the highest
  confidence, with ties broken by the smallest `reference_id`.

### 4.5 Re-run identity and the canonical dump

Hard requirement: two from-scratch runs over unchanged v2 data produce **byte-identical canonical dumps**.

How:
- **No network** inside `migrate`. Metadata repair uses v2's `provider_cache` only (§7.3). A live re-fetch is a
  separate command.
- **No wall-clock time in data.** Rows created by the migration take their `created_at` from the v2 row they
  derive from. Where there is none (attention items, rule decisions), they take the **snapshot time** (§6.1),
  which is excluded from the dump.
- **Sorted inputs.** Every stage iterates in sorted order (project precedence, then v2 id). Inserts happen in
  primary-key order, so rowids and AUTOINCREMENT values are stable.
- Inputs from the filesystem (PDF and markdown trees, answer files) are read in sorted path order.

**Canonical dump** (`migration/runs/<run_id>/dump.jsonl.gz`, plus `dump.sha256`):
- For `library.db` and each `graph.db`, in table-name order: one JSON line per row, `json.dumps(row,
  sort_keys=True, ensure_ascii=False)`, `ORDER BY` the primary key (or all columns when there is no PK).
- FTS tables are dumped by content (`SELECT rowid-free columns ... ORDER BY` the id column). FTS shadow
  tables, `schema_migrations.applied_at`, `migration_runs`, `library_meta.migration_run_id` and the
  snapshot-time columns listed in `migrate/dump.py:EXCLUDED` are left out. That list is short, explicit and
  test-covered.
- A file manifest of `library/pdfs`, `library/markdown` and `v2_answers/`: relative path, size, sha256, sorted.
- `seedgraph migrate verify --against <run_id>` diffs two dumps and names the first differing table and row.

## 5. Lifecycle rules (ticket 05)

`seedgraph.library.lifecycle` has a two-call interface:

- `remove_from_graph(graph, paper_id)` deletes one `graph_papers` row. Nothing in the library changes. If the
  paper was a seed, the result says so.
- `plan_delete(paper_id) -> DeletePlan`. This is pure and writes nothing. It lists every graph still using the
  paper (member, frontier or excluded), the counts of notes, versions, claims, anchors, lens outputs,
  references and incoming citation links that would go, whether a researcher-edited note exists (from stage 3),
  and whether the PDF blob is shared with another paper.
- `delete_paper(plan, confirm=True)` executes exactly that plan and refuses if the state changed since (the plan
  carries a fingerprint). It removes the paper's held parts (files link, notes, versions, claims, anchors, lens
  outputs, its reference list) and its `graph_papers` rows in every graph; blobs go only when no other paper
  uses them. If other papers' reference lists cite it, the `papers` row and identifiers stay as a cited-only
  paper, because those citation facts belong to the citing papers.

The CLI and the UI always call `plan_delete` first and show its warning ("used by graphs X, Y; has researcher
edits"). **Researcher-edited notes go only with the paper.** No other operation deletes an overlay:
re-extraction, re-conversion, version pruning, `remove_from_graph` and `migrate` (which moves the old library
aside rather than deleting it) all leave it in place.

## 6. Migration design

`seedgraph.migrate` is one deep module. Its interface is:

```python
run_migration(v2_home: Path, v3_home: Path, *, graph_names: Mapping[str, str] | None = None,
              replace_dirty: bool = False) -> MigrationResult   # report path, dump sha, counts, attention count
```

Internally it is a pipeline of pure stages over an in-memory **snapshot** and a **plan**. Only the `apply` and
`files` stages touch disk. Tests drive `run_migration` on synthetic v2 homes. The plan builder is also testable
alone, because it is a pure function from snapshot to plan: an internal seam, not part of the interface.

### 6.1 Stages

1. **Preflight.**
   - Check that the v2 home exists and the schema versions are the expected ones (`project.db` and `cache.db`
     at the versions the frozen v2 schema has).
   - Check v2 processes. **Refuse on a v2 writer** (`serve`, extraction or build runners). **Warn only** on
     the read-only v2 MCP server, which agent sessions keep connected. The real gate is the logical and tree
     hashes in §6.4.
   - Check that no v3 process holds `library/` or `graphs/` open (`serve`, `mcp serve`). Windows refuses to
     rename directories with open handles, so the swap would fail half-way.
   - Check free disk space on the v3 home's volume: at least `2 × (pdfs + markdown) + 1 GB`. A staging copy
     and the live copy can coexist.
2. **v2 guard (before).** Record a byte hash and a logical hash for each v2 file (§6.4).
3. **Snapshot.** For each v2 DB (cache, then each project in precedence order), open
   `file:…?mode=ro` and run `BEGIN`. Every SELECT the migration needs runs inside that **one read
   transaction**, followed by `COMMIT`. Each DB is internally consistent. Cross-DB consistency rests on the
   no-v2-process preflight. The snapshot records each DB's `PRAGMA data_version`, schema version and logical
   hash. The snapshot time (UTC) is taken once here.
4. **Plan (pure).** Builds identity groups and paper ids (§4), graph membership (§6.2), note version order
   (§6.6), reference-list choice (§6.7), decision mapping (§6.8–6.10) and defect classification (§7.3). The
   output is a `MigrationPlan` value. Running the plan builder twice on the same snapshot gives equal plans.
5. **Files.** Copy each needed PDF and markdown blob into staging. **Copy, never hardlink**, so v3 can never
   alter a v2 file. Copy through `copy_file_atomic` and verify sha256 against the content address.
   Every cached PDF is copied (every cached PDF joins the library), plus every markdown referenced by
   a carried `markdown_documents` row.
6. **Apply.** Create a fresh staging `library.db` and `graph.db`s through the numbered migrations, then insert
   the plan in key order. Then rebuild the FTS tables and run `PRAGMA integrity_check` and
   `foreign_key_check`.
7. **Defect repair and attention** (§7.3). This is applied as part of the plan; the attention items are
   written here.
8. **v2 answers export** (§6.11).
9. **Dump and report.** Write the canonical dump, `report.md`/`report.json` and `attention.md` into
   `migration/runs/<run_id>/`.
10. **Swap** (§1.3).
11. **v2 guard (after).** Recompute and compare. A difference fails the run loudly, even though v3 is already
    swapped in. The run is marked failed in the report and the previous library is still in `replaced/`.

### 6.2 Which v2 projects become which graphs

| v2 project | v3 outcome |
|---|---|
| `second_test` | becomes the **DiD graph** (slug `did`; display name offered at migration). Members = its included works after merging, seeds kept, frontier = its metadata_only works (§6.3) |
| `test_proj` | **not a graph**. Every held paper of test_proj that is not already a member of the DiD graph after identity merging **joins the DiD graph as a member** (`origin = 'v2:test_proj'`). The report lists them. Its notes, decisions and reference lists are carried into the library |
| `land_value` | becomes its own graph (slug `land-value`; display name offered). Its excluded works become `excluded` rows |
| `oct2026test` | not a graph. Its held papers become library papers in no graph. Its notes are carried as versions |
| `dbg_probe`, `dbg_probe2` | not graphs, nothing carried (synthetic fixtures, no PDFs) |
| every cached PDF | joins the library. A PDF held by no project becomes a held paper with **no metadata** (title unknown). It is listed under "metadata needed", and its PDF check is `unknown` |

Renames are offered as `--name did="…" --name land-value="…"`. Display names are stored data; slugs are fixed,
so re-runs stay identical.

### 6.3 Cited-only papers carried

A v2 metadata-only or excluded work is carried as a cited-only paper only if at least one of these holds:
- it is the target of a carried citation link or a resolved reference;
- it is in a carried graph's frontier or exclusions;
- it is the subject of a carried decision.

Merging applies to them too (§4.2). Unreferenced metadata-only works of non-graph projects (for example
oct2026test's) are not carried. The report counts them.

### 6.4 Proving v2 unchanged

For each v2 project and the cache:
- **Byte hash**: sha256 of `*.db` and `*.db-wal` (not `-shm`). Missing files are recorded as missing.
- **Logical hash**: sha256 of a canonical dump of every v2 table, read in one read transaction.
- **Tree hash**: sha256 over the sorted `(relative path, size, sha256)` of `cache/pdfs` and `cache/markdown`.
  Mtimes are never used.

Pass means all three are equal before and after. If the logical and tree hashes are equal but a byte hash
differs, the result is "WAL checkpoint by another process", not a v2 write. It is reported as a warning with
the preflight's process list. A logical or tree difference is a failure. The migration's own connections are
`mode=ro` and never checkpoint.

### 6.5 PDFs and markdown

The disk check runs in the preflight. Blobs keep their content-addressed names
(`pdfs/<sha256>.pdf`, `markdown/<sha256>.md`). `storage_uri` stays relative to `library/`. v2's
`cache/marker/` side artifacts are not carried; they stay in the archived v2 home.

A source file with no successful markdown goes into `conversion_queue` (`not_converted`) and the report's "not
converted" list. The migration never converts, because conversion is GPU-dependent and would make runs
non-deterministic. `seedgraph convert --queued` does it afterwards.

Carried conversion runs: all `success` and `failed` rows, but not `pending` ones (in-flight leftovers).
Markdown produced by a test/fake converter version is carried, and an attention item is raised if any paper
uses it.

### 6.6 Notes, claims and anchors: version order

For each `(paper_id, schema_id)`, every carried note is a version. `version_no` is assigned in ascending order
of `(created_at, project precedence, note_id)`. The highest unlabelled version is current. There is no quality
judgement (ticket 05).
- Lens-origin runs (`lens_id` set) are versioned per `(paper_id, lens_id)` the same way.
- A note made on markdown that is not the paper's own (§7.3, "other markdown") gets
  `version_label = 'other_markdown'`. A note on a confirmed wrong PDF gets `'wrong_pdf'`. Labelled versions are
  never current and are hidden from agent search.
- Claims, `claim_spans` and the claims' extraction runs travel with their note. Spans are deduped (§4.4).
  Anchors are re-checked with `anchor.assert_invariant` against the carried markdown. A failure is a migration
  error, not a silent drop.

The current note is chosen **library-wide**. For a paper held in several projects, the current note can come
from a project other than the graph's source project. This is a named parity difference (§8.3).

### 6.7 Reference lists, citations, citation reviews

- **One reference list per paper.** It is taken from the highest-precedence project whose list was parsed from
  the paper's current markdown. Ties go to precedence. `ordinal` is the entry's order in that list.
- Decisions recorded against another project's list are **mapped** onto the chosen list by `(normalized raw
  reference text, ordinal)`, falling back to normalized raw text alone when that is unique. Unmapped decisions
  are counted and listed in the report. They are never silently dropped.
- `citation_edges` from every project are remapped to paper ids and deduped (§4.4). Edges whose ends were not
  carried are counted in the report.
- Resolved `citation_resolution` review items become `citation_review_decisions`. Open items are not carried.

### 6.8 Duplicate decisions

Resolved `duplicate_candidate` items become `duplicate_decisions` with `decided_by = 'researcher'`. They are
mapped to paper ids through the incoming record's ids and the conflicting work ids in the payload. They feed
§4.2 as "never merge" constraints. Rule merges are added as `decided_by = 'rule'` rows.

### 6.9 Lens results

Lens YAML from `projects/<slug>/lenses/` goes into `lens_definitions` (keyed by `lens_id` + `definition_hash`).
Lens extraction runs, claims and `lens_outputs` are carried per paper and lens version. A graph whose v2
project ran a lens gets a `graph_lenses` row. Lenses are ported as they are (no redesign).

### 6.10 Concept decisions

`concept_constraints` rows become `concept_decisions`:
- `source = 'user'` must_link → `kind = 'merge', decided_by = 'researcher'`.
- `source = 'user'` cannot_link → `kind = 'never_merge', decided_by = 'researcher'`.
- the bulk must_links (v2 `source` other than `user`) → `kind = 'merge', decided_by = 'agent'`. These appear
  in the stage 4 undo list.

All rows start as `state = 'pending_apply'`; stage 4 applies them. Pairs are normalized (a < b) and deduped
across projects, with researcher rows winning over agent rows for the same pair. A **contradiction** (merge in
one project, never_merge in another, both from the researcher) raises an attention item and both rows are kept.
test_proj's resolved concept-merge review items are already reflected in its constraints, so they are not
carried a second time.

### 6.11 v2 answers export

Every saved answer in a carried project (`projects/<slug>/answers/*.json` plus `.trace.json`) is rendered once
to `v2_answers/<slug>/<answer_id>.md`: question, answer, cited evidence and trace summary, and the original JSON
in a fenced block. There is also an `index.md` sorted by answer time. Answers are not turned into traversal
records. Run folders, the mailbox, staging and backups are not carried.

### 6.12 Report

`report.md` (human) and `report.json` (machine, used by acceptance) account for **every v2 item**: each count is
split as *carried*, *merged into*, *not carried (reason)*, or *flagged*. Sections:
- totals per table;
- graph membership, including the folded-in test_proj papers by title;
- identity merges and blocked merges;
- not-converted papers;
- unmapped decisions;
- defects and repairs;
- the v2 guard result;
- the dump sha.

`report.json` also has a flat `totals` map (dotted keys, integers) that the acceptance check compares with the
researcher's `migration/acceptance/expected.json`. Every key is always present, zero included, so an expectation
never fails just because a count was empty. `accept.expected_template(report)` lists every key of a run.
"Carried projects" are all v2 projects except `dbg_*`. "Held" means a work has a PDF attached in that project.

| key | meaning |
|---|---|
| `library.<table>` | rows in each library table |
| `library.pdfs` | cached PDFs in the library |
| `library.pdfs_held_by_project` / `library.pdfs_held_by_none` | of those, PDFs attached to some carried v2 work in any project / attached to none (these become papers with no metadata) |
| `graph.<slug>.member` / `.frontier` / `.excluded` / `.seeds` | graph membership by status, and seed count |
| `graph.<slug>.folded_in` | papers folded into the graph from another project (test_proj into `did`) |
| `papers.held_in_multiple_projects` | library papers whose merged v2 works were held in more than one graph or fold-in project (second_test, land_value, test_proj) |
| `papers.held_in_multiple_projects_incl_oct` | the same over every carried project (adds library-only projects such as oct2026test) |
| `notes.multi_project_pairs` | papers whose note versions come from more than one v2 project |
| `works.no_strong_id` | carried v2 works with no DOI, OpenAlex or arXiv id |
| `works.no_strong_id.merged_by_rule` / `.check_these` | of those, merged into another work by rule (PDF, title + year) / left unmerged with their paper on "check these" (`identity_check`) |
| `merges.papers`, `merges.blocked` | papers made from more than one v2 work; merges blocked by conflicting ids |
| `concepts.<kind>.<decided_by>` | carried concept decisions (`merge` / `never_merge` × `researcher` / `agent`) |
| `concepts.v2.<project>.<kind>.<decided_by>` | v2 `concept_constraints` rows of that project reflected in a decision (carried or merged into one), by the row's own kind and maker (`user` → researcher, bulk `llm_merge` → agent) |
| `concepts.contradictions` | pairs with contradicting researcher decisions |
| `citations.decided_total_v2` | resolved citation-resolution review items in carried projects |
| `citations.decided_carried` / `.collapsed_onto_existing` / `.decided_unmapped` / `.decided_reflected_in_reference` | a partition of that total: carried as a decision row / mapped onto a decision already carried for the same reference / no counterpart on the kept reference list / an accept, reflected in the carried reference entry |
| `citations.review_decisions`, `citations.unmapped_decisions`, `citations.edges_dropped` | decision rows; all unmapped decisions (citation and duplicate); citation edges not carried |
| `answers.exported` / `answers.run_scoped_not_carried` | saved v2 answers exported to markdown / answers in run folders, not carried |
| `attention.total`, `attention.hides_notes`, `attention.<kind>` | attention items in all, those hiding notes, and the count for every kind |
| `v2.<source>.<table>.rows` / `.carried` / `.merged` / `.rebuilt` / `.not_carried` | the per-table reconciliation of every v2 row |
| `current_elsewhere`, `not_converted`, `labelled_versions`, `repairs.titles`, `repairs.relinks` | the matching report sections' list lengths |
| `errors`, `unaccounted` | migration errors; v2 rows with no disposition (must be 0) |

Report files go only to the v3 home.

## 7. Defect detection and the standing PDF check

### 7.1 One shared title matcher

`seedgraph.library.titlecheck` is a pure module and a deep one: callers give a title, an optional DOI and page
texts, and get a verdict.

```python
check_pdf_text(title: str | None, doi: str | None, pages: Sequence[str]) -> PdfCheck
  # PdfCheck.status: 'pass' | 'weak' | 'miss' | 'unknown'; .score; .via ('doi', 'exact', 'tolerant'); .evidence
first_pages_from_markdown(md: str, n: int = 3) -> list[str]
first_pages_from_pdf(pdf_path: Path, n: int = 3) -> list[str] | None     # pypdfium2 text layer
```

The migration, the standing check, "check these" evidence and the stage 3 library-wide checks all use it.

### 7.2 Tolerant match

**Normalization** (applied to the title and to the page text):
1. NFKC, which expands Unicode ligatures (ﬁ, ﬂ, ﬀ, ﬃ).
2. Strip markdown and HTML markup: `<sup>…</sup>`, `**`, `#`, link syntax.
3. Join line-break hyphenation (`-\s*\n\s*` → '').
4. NFKD and drop combining marks, as in `identity.normalize_title`.
5. Lowercase, map every non-`[a-z0-9]` run to a space, and split.

**Mojibake variants.** Marker output on some PDFs maps ligature glyphs to Latin-1 lookalikes. In v2 markdown,
`Ö` stands for "fi" (e.g. `speciÖcation`) and `¢` followed by a space stands for "ffi" (e.g. `e¢ ciency`). These
characters are also real letters or symbols (`Ö`, `Ø` in names; `Þ` in math), so the text is **never
rewritten**. Instead the page text is matched as a *variant set*:
- the original;
- a folded copy: `¢ ?` → `ffi`, and `Ö` → `fi` only when it touches a lowercase letter.

The best score over the variants wins. The fold table is a small constant, extended only with evidence. The same
variants are applied to the title, because v2 titles can carry `�` or mojibake from PDF metadata.

**Scoring.**
- DOI: the normalized DOI appears in the whitespace-stripped, lowercased page text → `pass` via `doi`.
- Exact: the normalized title's token sequence appears contiguously → `pass` via `exact`.
- Tolerant: let T be the title's content tokens (`citation/resolve._content_tokens` rules: stopwords dropped,
  length > 1). Slide a window of `2·|T| + 10` tokens over the page tokens. The score is the best fraction of T
  covered within one window. The window stops a long first page from matching scattered words.
  - score ≥ 0.80 → `pass` via `tolerant`;
  - 0.30 ≤ score < 0.80 → `weak` (reported, not an attention item);
  - score < 0.30 → `miss` (zero-hit: attention item, notes hidden).
  - A title with ≤ 3 content tokens needs all of them for `pass`.
- No title, or no text at all (a scan with no text layer, or empty pages) → `unknown`. This is never a miss.

The thresholds are constants in the module, set from a run over the real v2 data. A throwaway pass during
design put both known mix-up cases at 0.25, only 0.05 under the miss threshold. The S2 slice re-runs this
calibration in the v3 home and must confirm the miss threshold against the cases scoring 0.30–0.50. It
records the distribution in the report. The acceptance check then explains each project's weak and miss
counts.

**First pages.** The first 3 pages.
- Markdown: split on Marker page markers, lines matching `^\{N\}-{10,}$` with N 0-based. Take pages 0–2. With
  no markers (external imports), use the first 8,000 characters.
- PDF: the pypdfium2 text layer of pages 1–3.

The migration checks the markdown pages and falls back to the PDF text when the markdown check is not `pass`.
The best result wins. The standing check runs **before conversion**, so it uses the PDF text layer only. If
there is no text layer, the status is `unknown`, and it is checked again on the markdown after conversion.

### 7.3 Defects: classification and handling

| Defect | Detection | Handling |
|---|---|---|
| Double import (two works, one paper) | same group under §4.2 with no conflict | merged automatically; rule `duplicate_decisions` row; report |
| Different papers sharing one PDF | one `source_file` linked to ≥ 2 papers after §4.2 (a blocked conflict) | both links kept; attention `shared_pdf`. Each paper whose title check is `miss` against that PDF gets `hides_notes = 1` |
| Work pointing at another PDF's markdown | `work_source_files.markdown_hash` → markdown whose `source_file_id` ≠ the work's PDF | re-link to the PDF's own latest successful markdown. If there is none, queue for conversion. Notes made on the foreign markdown get `version_label = 'other_markdown'`. Report |
| Zero-hit title check | §7.2 `miss` | attention `pdf_title_miss`, `hides_notes = 1` until the researcher confirms or fixes it |
| Malformed title | empty; filename-like (`.pdf`, no spaces with `_`/`-`, hash-like); contains `�`; length < 10 or > 300; publisher boilerplate | first try v2 `provider_cache` (`by_doi` responses) offline. If a title is found, replace it and set `title_source = 'provider_cache'`. Otherwise attention `malformed_title`. Live re-fetch: `seedgraph library refetch-metadata --malformed` after migration |
| Unheld PDF, no metadata | cached PDF used by no project | held paper with no title; attention `metadata_needed` |
| Not converted | source file without successful markdown | `conversion_queue`; report list |

Confirming a paper (`seedgraph library attention resolve <item> --confirm`) clears the item and un-hides its
notes. Fixing a wrong PDF (detach, attach the right one) relabels the old notes `wrong_pdf` and leaves the paper
needing notes.

### 7.4 The standing PDF check

`library.files.attach_pdf(paper_id, pdf_path)` always runs `check_pdf_text(title, doi, first_pages_from_pdf())`
and stores the result in `paper_files.pdf_check_*`. A `miss` raises a `pdf_title_miss` attention item: "this PDF
may not be this paper". It never blocks attach, conversion or notes, because books and scans fail legitimately.
After conversion, an `unknown` result is re-checked on the markdown.

## 8. Search and parity

### 8.1 Library search

`seedgraph.library.search` has one interface for spans, claims and notes:

```python
search(conn, query: str, *, kind: 'spans' | 'claims' | 'notes', graph: str | None = None,
       paper_id: str | None = None, section_kind: str | None = None, audience: 'agent' | 'researcher' = 'agent',
       limit: int | None = 20) -> list[Hit]
```

- **Matching** is unchanged from v2: a phrase-safe `MATCH` (`fts/search.build_match_query`) with unicode61 and
  no stemming, ordered by `bm25`.
- **Graph scope**: `graph=` restricts span hits to markdowns that are the current `paper_files.markdown_id` of a
  graph member (via ATTACH). A span hit reports every member paper on that markdown. The frontier has no text,
  so only members can match.
- **Current only**: claims and notes come only from `note_current` (and current lens runs).
- **Agent audience** drops notes and claims hidden by §2 (open `hides_notes` attention items, labelled
  versions). The researcher audience shows them, labelled.
- **Stable ordering**: `ORDER BY rank, paper_id, start_char, span_id` (or the claim or note id), so equal
  scores never reorder between runs.

### 8.2 Parity: what "same results as v2" means

FTS5's `bm25` uses statistics over the whole table. The library index holds every paper, so scores and the
order of equal matches **will** differ from v2's per-project index even when the matches are identical. Under
`LIMIT 20` the top-k set itself can shift. Parity is therefore defined on match **sets**, not ranks:

- **Spans.** For each query q, let `V2(q)` be the full match set (no limit) of v2's `fts/search.search_spans`
  run on the second_test snapshot. Let `V3(q)` be the full match set of v3 `search(kind='spans',
  graph='did')`. Both are keyed by `(markdown_hash, start_char, end_char)`, not by work or span id. Parity
  holds when `V3(q) = map(V2(q)) ∪ Allowed⁺(q) − Allowed⁻(q)`.
- **Claims and notes.** The baseline is v2's `answer/retrieve` path: `CURRENT_CLAIMS_CTE` plus the
  `inclusion_status <> 'excluded'` filter over `claim_fts` and `note_fts`. Claims are keyed by `claim_id` and
  notes by `note_id`, both carried verbatim.
- **Rank** is reported, not gated: top-20 overlap and Kendall τ per query, for information.

### 8.3 Named known differences

Each difference is classified by a rule from the migration's own data. Anything unclassified fails parity.

| Class | Effect | Source of truth |
|---|---|---|
| `folded_in` | extra hits from test_proj papers that joined the DiD graph | `graph_papers.origin = 'v2:test_proj'` |
| `hidden` | missing note and claim hits on papers with `hides_notes` items, and labelled versions | `attention_items`, `version_label` |
| `merged` | two v2 works → one paper: duplicate span hits collapse (same key); claim and note hits of the non-current version vanish | `duplicate_decisions` (rule) |
| `current_elsewhere` | for papers held in several projects, the current note or claims come from the most recent version, which may be another project's | `structured_notes.version_no` + origin |
| `relinked` | hits change for works re-linked from a foreign markdown to their own | report repair list |
| `spans_from_other_projects` | extra span hits: manual spans created in another project on the same markdown | span origin in `v2_id_map` |

### 8.4 How parity is tested

- **Real data, in the v3 home only.** `migration/parity/queries.json` is a fixed list of about 30–50 queries,
  chosen with the researcher. A starting set is sampled from the most frequent claim labels and technical
  phrases, then edited.
  - `seedgraph migrate parity --capture` runs the v2 baseline SQL against the read-only v2 snapshot and writes
    `parity/v2_expected.json`.
  - `seedgraph migrate parity --check` runs v3 search and writes `runs/<run_id>/parity.md`: per query, the
    matched, allowed (by class) and unexplained counts. Any unexplained difference exits non-zero.
- **Synthetic, in the repo.** A fixture builder writes small v2-shaped homes using the **frozen v2 schema**
  (§9). The fixture has two projects with an overlapping paper and diverging notes, a double import, a shared
  PDF with conflicting DOIs, a foreign-markdown work, a no-id paper, a mojibake title and a wrong-PDF paper.
  Tests assert each parity class is produced and classified, and that an injected unexplained difference fails.
  The fixtures contain only invented text.

## 9. v2 code in the v3 branch during stage 1

Decision already taken; recorded here:
- **The migration writes no `project.db`.** v3 data lives only in `library.db` and the `graph.db`s.
- The old `project.db`-based code (the web UI, MCP server, extraction runners, lens runner, `ask`, eval) stays
  in the branch with its tests passing on **their own fixtures**, until the stage that replaces it (stage 2 or
  3) deletes it. It is not pointed at the v3 home.
- **No compatibility layer.** Nothing translates `library.db` into `project.db` shapes or back.
- **Frozen v2 schema.** The migration and its fixtures need v2's schema after the old code is gone, so stage 1
  copies `db/schema/project/*.sql` and `db/schema/cache/*.sql` into `seedgraph/migrate/v2_schema/` (read-only
  data). `golden/v2.py` keeps reading v2 directly.
- Replaced in stage 1, and so deleted in stage 1 (the deletion rule):
  - the old top-level `migrate` command (v2 schema upgrade; its test in `tests/test_cli.py` goes with it);
  - the old top-level `search --project`.
  Both names are taken by the new core commands. Nothing else is deleted in stage 1.

## 10. CLI module layout

`seedgraph/cli.py` already exists, so a `seedgraph/cli/` package cannot sit beside it. The new commands live
in a new package, **`seedgraph/commands/`**. `cli.py` remains the entry point (`seedgraph-v3 =
"seedgraph.cli:app"`) and becomes a registration module as old groups are deleted stage by stage.

```
seedgraph/commands/
  __init__.py      register(app): mounts the core commands onto the root app (one call from cli.py)
  _common.py       --root option, error → exit-code mapping, table/JSON rendering (shared, tiny)
  setup.py         `seedgraph setup`: create the v3 home and library.db, check the converter env and GPUs
  migrate.py       `seedgraph migrate run|verify|parity|accept`
  convert.py       `seedgraph convert [PDF…] [--queued] [--paper ID]`
  library.py       `seedgraph library status|search|show|check|attention|remove|delete|refetch-metadata|prune-replaced`
                   `seedgraph library graphs list|show|rename`
  (stage 2) mcp.py → `seedgraph mcp serve` replaces the old mcp group
  (stage 3) serve.py → `seedgraph serve` replaces the old serve
```

Rules:
- **Every command module is a thin adapter.** It parses options, calls **one** function of a deep module, and
  renders. Logic lives in `seedgraph.library` and `seedgraph.migrate`, which tests exercise directly. CLI tests
  are smoke tests only.
- Deep modules behind them:
  - `seedgraph.library`: `store` (open, migrate schema, dirty flag), `identity` (pure key and id derivation,
    merge grouping), `titlecheck` (pure), `files` (attach, ingest, convert via the stage 0 pool), `search`,
    `lifecycle`, `graphs`, `attention`;
  - `seedgraph.migrate`: `run_migration` is the interface. Its internals are `snapshot`, `plan`, `apply`,
    `files`, `defects`, `answers`, `dump`, `v2guard`, `report`, `parity` and `accept`.
- `golden` stays as it is (stage 0). Old groups (`corpus`, `cite`, `extract`, `lens`, `concepts`, `graph`,
  `eval`, `llm`, `secrets`, `answer`, `agent-extract`, `project`, `review`, `cache`, `serve`, `mcp`) remain
  until their replacement stage. `seedgraph library graphs` is used rather than `graph` because the old `graph`
  group stays until stage 4.

## 11. Build plan for stage 1

Each slice is sized for one agent and written test-first. Each slice owns its files exclusively. Shared files
(`cli.py`, `.gitignore`, `pyproject.toml`) are touched only by the slice named for them. A slice may read but
never edit another slice's files.

| # | Slice | Depends on | Owns | Tests first |
|---|---|---|---|---|
| S1 | **Library schema and store**: `db/schema/library/*.sql`, graph schema, `library/store.py`, `library/graphs.py` (create, open, membership), views, dirty flag | — | `db/schema/library/`, `db/schema/graph/`, `src/seedgraph/library/{__init__,store,graphs}.py`, `tests/library/test_store*.py` | schema applies; FK enforced; views; membership refuses cited-only members |
| S2 | **Identity and title matcher (pure)**: id derivation, merge grouping with conflict blocking, title matcher with variants, calibration script (calibration output stays in the v3 home) | — | `library/identity.py`, `library/titlecheck.py`, `tests/library/test_identity*.py`, `test_titlecheck*.py` | determinism; conflict blocks the PDF union; researcher never-merge; mojibake variants; window scoring; `unknown` on empty |
| S3 | **Frozen v2 schema and synthetic v2 fixture builder** | — | `migrate/v2_schema/`, `tests/migrate/fixtures/` (builder plus invented text) | builder produces each defect class; fixtures load with the frozen schema |
| S4 | **Files and conversion into the library**: re-point `cache/ingest.py` and `cache/convert.py` to take a store location; `library/files.py` (attach plus the standing check, convert queue) | S1, S2 | `library/files.py`, `cache/ingest.py`, `cache/convert.py`, `tests/library/test_files*.py` | attach records the check; a miss flags, never blocks; conversion lands in `library.db`; old `cache.db` tests still pass |
| S5 | **Search and lifecycle**: `library/search.py`, `library/lifecycle.py`, `library/attention.py` | S1 | those three files plus tests | graph scope; agent audience hides; stable order; `plan_delete` names graphs; delete refuses a stale plan; overlay-safety placeholder test |
| S6 | **Migration core**: `snapshot`, `plan`, `apply`, `files`, `defects`, `dump`, `v2guard` | S1, S2, S3 | `src/seedgraph/migrate/{__init__,snapshot,plan,apply,files,defects,dump,v2guard}.py`, `tests/migrate/test_core*.py` | run twice → identical dump; v2 fixture hashes unchanged; each merge and defect rule; note version order; reference decision mapping; the swap never touches unknown files or the golden files; disk-space refusal |
| S7 | **Answers export, report, parity, acceptance** | S5, S6 | `migrate/{answers,report,parity,accept}.py`, `tests/migrate/test_report*.py`, `test_parity*.py` | the report accounts for every fixture item; parity classes; an unexplained diff fails; acceptance reads expected values from a home file |
| S8 | **CLI core and stage-1 deletions**: `commands/`, registration in `cli.py`, delete the old `migrate`/`search` and their tests | S4, S5, S7 | `src/seedgraph/commands/`, `cli.py`, `tests/test_cli.py` (old-migrate test removed), `tests/commands/` | smoke tests per command; `--root` respected; exit codes |

**Parallelism.** S1, S2 and S3 run in parallel. Then S4, S5 and S6 run in parallel; their files are disjoint,
and S6 uses S1's store API but not S4's or S5's. Then S7, then S8. The critical path is S1 → S6 → S7 → S8.

**Final real-data acceptance run** (gate 2):
`seedgraph migrate accept --v2-home ~/.seedgraph` runs these steps in order:
1. preflight (no v2 writer; no v3 process holding the library);
2. `migrate run`;
3. a second `migrate run` into a fresh staging folder, comparing dump shas;
4. the v2 guard;
5. `parity --check`;
6. a comparison of `report.json` against `migration/acceptance/expected.json`.

`expected.json` holds the expected counts with tolerances, and the expected attention item kinds with counts.
The researcher reviews and owns it; it lives in the v3 home. The command prints one pass/fail table and writes
`runs/<run_id>/acceptance.md`.

## 12. Open questions for the researcher

Each has a recommendation (*R*).
1. **Conflicting-DOI shared PDFs** (same paper in two versions, or a true mix-up). *R:* never auto-merge; attention
   list with title-check evidence per side; resolve by hand (few cases).
2. **Unheld cached PDFs.** *R:* held papers with no metadata under "metadata needed"; fill via the stage 2 metadata
   task, not in the migration.
3. **oct2026test's held papers.** *R:* library papers in no graph, notes as versions; its unreferenced
   metadata-only works are not carried.
4. **Current note across projects.** "Most recent wins" makes another project's note current in the DiD graph for
   some papers. *R:* keep the ticket's rule; list those papers in the report for manual version switching.
5. **Malformed titles without cached metadata.** *R:* keep `migrate` offline; run `library refetch-metadata
   --malformed` once after acceptance as an ordinary later write.
6. **v2 answers outside second_test.** *R:* export every saved answer in a carried project, under its slug.
7. **Graph slugs and names.** *R:* fixed slugs `did` and `land-value`; display names given at migration.
