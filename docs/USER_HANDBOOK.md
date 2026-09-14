# Seedgraph v2 — User Handbook

Operational guide: install, run, and operate Seedgraph day to day. For the one-page
happy path, see **[QUICKSTART.md](QUICKSTART.md)**.

## 1. What Seedgraph is

Seedgraph is a local-first research-grounding system: it turns a researcher-chosen
scholarly corpus into a provenance-preserving context shell so an LLM (or a human)
can answer questions with traceable evidence instead of hallucinated citations.
The one mental model to keep: **the graph routes attention; source-linked evidence
spans carry the factual accountability.** A citation/concept graph tells you where
to look; every claim resolves back to an exact, verbatim quote in a specific paper.

## 2. Install & prerequisites

| Requirement | Command / note |
|---|---|
| Python | >= 3.11 |
| Package (editable) | `pip install -e .` |
| Real PDF->markdown (Marker) | `pip install -e .[marker]` — heavy, GPU-friendly. Without it, `cache convert` / uploads have no working backend. |
| Local LLM (optional but recommended) | Install [Ollama](https://ollama.com), then `ollama pull qwen3.5` — this is the model verified working against the local-first routes (the shipped default profile's `model` field is `llama3`; pull whichever model you point `local_ollama_default` at). |
| Hosted LLM (optional) | Set `ANTHROPIC_API_KEY` in your environment for the `anthropic_api_default` profile. No OpenAI adapter ships yet — an `openai` profile is treated as unavailable. |
| Browser test extra (dev only) | `pip install -e .[browser]` + `playwright install` — not needed to use the app. |

Check the install with:

```powershell
seedgraph version
seedgraph doctor
```

## 3. Running the app

```powershell
seedgraph serve
```

| Flag | Default | Effect |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8765` | Preferred port; auto-increments past a busy one. |
| `--allow-remote` | off | Required to bind a non-loopback host. |
| `--root` | `~/.seedgraph` | Override the seedgraph home directory. |
| `--open-browser` | off | Opens the one-time `/auth` URL automatically. |

**Run it from the interpreter that has `marker` installed** — e.g. the machine's
global `G:\python313` if that's where you `pip install -e .[marker]`'d. If you
`serve` from a different interpreter, uploads ingest fine but never convert (no
error banner today — check the corpus page's queued/converting state, or `doctor`).

On start it prints:

```text
seedgraph web UI — open this one-time sign-in URL:
  http://127.0.0.1:8765/auth?t=<token>
```

Opening that URL once exchanges the token for an HttpOnly, `SameSite=Strict`
session cookie (`sg_session`) and redirects to `/ui`; the token never reappears in
the address bar. All subsequent requests are gated by
`require_local_session`: the client must be loopback **and** present the cookie.

**Honest `--allow-remote` note:** the flag only permits the *bind* to a
non-loopback interface. The per-request gate above still requires the client to be
loopback, so **remote clients cannot reach any private-evidence screen even with
`--allow-remote`** — remote LAN access is not actually enabled today; the flag
exists for future use and prints a warning when used.

## 4. End-to-end workflow

Ordered pipeline; each stage names its CLI command(s), prerequisite, and gotcha.

### 4.1 Create a project

```powershell
seedgraph project new <slug> [--name NAME] [--description DESC]
```
Scaffolds `project.db` + `project.yaml` + `runs/`. Slug must match `^[a-z0-9._-]+$`.
UI equivalent: the "New project" form on `/ui` (`POST /ui/projects`).

### 4.2 Add / upload seeds

CLI (by identifier):
```powershell
seedgraph project add <slug> --doi 10.xxxx/example --seed [--title T] [--year Y] [--author A ...]
```
CLI (upload a PDF you already have, with a known work id):
```powershell
seedgraph corpus upload <slug> <work_id> <path/to.pdf>
```
UI: per-project drag-and-drop / file-picker upload on the corpus page
(`POST /ui/projects/{slug}/upload`). Each accepted PDF (must start with `%PDF`)
becomes a seed work, is ingested inline, and enqueued for conversion.
**Gotcha:** uploaded PDFs are stamped `user_supplied_private` — see §7.

### 4.3 Background marker conversion queue

Automatic after upload/ingest — no command to run. One subprocess per GPU slot
converts PDFs to markdown; the corpus page polls `GET
/ui/projects/{slug}/upload/status` and shows queued/converting/failed.
Concurrency = GPU count by default (see §8 env vars). **Prerequisite:** the
`marker` extra installed in the interpreter running `serve` (§3).

### 4.4 Metadata backfill (identify)

```powershell
seedgraph corpus identify <slug> [--work-id ID ...] [--force]
```
Runs automatically after conversion; this command re-runs it (e.g. after a
conversion fix, or with `--force` to override an already-titled work). Extracts
title/authors/venue/year + a cover-region DOI/arXiv via the local model, writing
through the identity layer (id collisions route to `review`, never a silent
merge). **Prerequisite:** the work has converted markdown.

### 4.5 Resolve / walk / acquire (the network phase)

```powershell
seedgraph corpus resolve <slug>
seedgraph corpus walk <slug> [--depth 2] [--per-gen-cap 50]
seedgraph corpus acquire <slug> [--promote]
seedgraph corpus run <slug> [--depth 2] [--per-gen-cap 50] [--promote]   # umbrella: resolve -> walk -> acquire, one run_id
seedgraph corpus status <slug>
```
The sole network phase: resolves each work's identifiers, walks outbound
`referenced_works` N generations deep, and fetches open-access PDFs. Provider
order: OpenAlex -> Crossref -> Unpaywall -> Semantic Scholar -> CORE.
**Gotcha:** CLI-only today — not yet in the UI run planner. Unpaywall
needs a `contact_email` (a top-level field in `config.yaml`, global or
per-project — see §8); CORE needs a key or it sits out (see §7/§9).
Acquisition is **open-access only** — no paywall bypass; unavailable works
stay `metadata_only` and can be filled by manual upload (§4.2).

### 4.6 Sections + spans (evidence layer)

```powershell
seedgraph sections build <slug> --work <work_id>      # or --markdown-id <id>
seedgraph spans index <slug> --work <work_id>
```
Parses ATX headings into `document_sections`, then materializes paragraph-level
`evidence_spans` + the `span_fts` search index. **Prerequisite:** the work has
converted markdown. **UI equivalent:** the "Sections build" plan task loops this
over every included work with markdown.

### 4.7 Citation graph (both tiers)

```powershell
seedgraph cite project <slug>                 # provider-projection tier (offline, no network)
seedgraph cite build <slug> [--run-id ID] [--open-world]
seedgraph cite run <slug> [--open-world]      # umbrella: project -> build, one run_id
seedgraph cite table <slug> [--all]
seedgraph cite parse <slug> [--work ID] [--run-id ID] [--force]   # parsed-bibliography tier
seedgraph cite disagreements <slug> [--run-id ID]
```
Two edge tiers: tier 2 projects `provider_reference` edges offline from the
provider cache (no network call — `cite project`/`build`/`run`/`table`); tier 3b
(`cite parse`) parses each work's references section into resolved
`parsed_bibliography` edges. **Prerequisite for `cite parse`:** run
`sections build` first (it reads `document_sections`). `cite table` shows the
closed-world included->included matrix by default; `--all` adds edges to
`metadata_only` stub targets. `cite disagreements` surfaces (source, target) pairs
carrying 2+ distinct provenances.

### 4.8 Note extraction (default schema)

```powershell
seedgraph extract notes <slug> [--work ID] [--schema ID] [--profile ID] [--force] [--confirm-external] [--dry-run] [--chunked] [--yes]
seedgraph extract notes-chunked <slug> [--work ID] [--schema ID] [--profile ID] [--overlap-tokens 128] [--force] [--confirm-external] [--dry-run] [--run-id ID]
```
Runs `default_research_note_v1` over each included work's markdown.
`--chunked` on `extract notes` auto-routes oversize papers to the map-reduce
chunked extractor (or run `extract notes-chunked` directly over the
`skipped_oversize` worklist). `--confirm-external` permits private full text to
leave the machine on an external profile (see §7). `--dry-run` estimates cost and
writes nothing.

### 4.9 Concepts

```powershell
seedgraph concepts build <slug> [--profile ID] [--no-llm] [--tau 0.6]
seedgraph concepts list <slug> [--type T] [--status S]
seedgraph concepts show <slug> <concept_id>
```
Extract -> cluster (token-Jaccard, `--tau`) -> (LLM-propose synonyms unless
`--no-llm`) -> guard -> merge; writes `concepts`/`aliases`/`claim_concepts` and
`discusses`/`co_occurs_with` edges. `concepts show` is the milestone query:
linked papers, claims, and evidence spans for one concept.

### 4.10 Project lenses

```powershell
seedgraph lens new <lens_id> --from-template TEMPLATE [--project SLUG]
seedgraph lens list [--project SLUG]
seedgraph lens validate <lens_id> [--project SLUG]
seedgraph lens show <lens_id> [--project SLUG]
seedgraph lens calibrate <lens_id> [--sample 3] [--work ID ...] [--project SLUG]
seedgraph lens run <lens_id> [--all] [--work ID ...] [--force] [--project SLUG]
seedgraph lens results <lens_id> [--status found|not_found|all] [--json] [--project SLUG]
seedgraph lens status <lens_id> [--project SLUG]
```
Project-specific extraction beyond the default schema (e.g. "regularity
conditions"). `--project` defaults to the current directory's project if it's a
project dir. Calibrate before a full `run --all`; a lens promotes to `active` on
first full run (immutable definition thereafter).

### 4.11 Graph export

```powershell
seedgraph graph export <slug> [--run-id ID] [--format json|graphml|csv] [--allow-private]
```
Exports the computed concept/citation view through a default-deny access guard;
`--allow-private` includes private concepts/edges.

### 4.12 Ask

```powershell
seedgraph ask "<question>" --project <slug> [--mode project_only|allow_outside] [--no-llm] [--json] [--save] [--limit 40]
```
Deterministic FTS5 + relational retrieval, then one self-declaring LLM call with
code-enforced faithfulness (no invented citations, abstains rather than emitting
unsupported prose). **`--no-llm` works with zero LLM configured** — it degrades to
a ranked, cited evidence list with no prose.

**Every ask now persists, unconditionally.** Both the `AnswerEnvelope` and a sibling
`AnswerTrace` — a versioned capture of the pipeline's intermediate state for that
answer — are written on every CLI ask, every web ask, and every MCP `ask` call.
`--save` is kept for backward compatibility but is now a redundant no-op: the answer
persists with or without it. The CLI echoes both paths on every call, including
before the JSON body under `--json`:
```text
saved projects/<slug>/answers/<answer_id>.json
saved projects/<slug>/answers/<answer_id>.trace.json
```
Ad-hoc asks (CLI, web GET/POST, MCP) land under `projects/{slug}/answers/`; an ask
made as part of a build run lands under `projects/{slug}/runs/{run_id}/answers/`
instead — the `run_id` only ever controls this nesting path (see §5).

**Trace view.** The web UI links every rendered answer to a trace page at
`/ui/projects/{slug}/answers/{answer_id}/trace` (run-scoped answers use
`/ui/projects/{slug}/runs/{run_id}/answers/{answer_id}/trace`). It shows the
deterministic query classification, the full ranked-candidate table, and — for
citation-shaped questions — an explorable subgraph of the citation neighborhood the
answer drew on. An answer from before this feature shipped has no trace; the page
says so plainly instead of erroring.

**Export.** To share one answer's reasoning outside a running server:
```powershell
seedgraph answer trace-export <answer_id> --project <slug> [--out PATH]
```
renders the same page as a single self-contained HTML file (vendored JS inlined, no
`/static/` or network references) that opens with `seedgraph serve` down. Example:
```powershell
seedgraph answer trace-export ans_3f2a9c1e6b774e3a9c2e1a9f7d0e88b1 --project my_project --out trace.html
```

**Reading a trace.** Every ranked candidate carries one of three dispositions:
`shown` (its evidence reached the prompt), `cut_rank` (dropped by the rank-limit
before token budgeting ran), or `cut_budget` (survived rank-truncation but was
dropped while packing the prompt because the token budget ran out — read from the
budget loop's own break point, not inferred, so it's exact). The header also puts
the deterministic classification (`spec.protocol_hint`) next to the LLM-declared
`query_type` from the envelope and flags it when they diverge — a gap that was
invisible before this feature. The subgraph pane, present only for citation-shaped
answers, renders the neighborhood actually captured (seeds, neighbors,
recommendation works); click a node to see why it's there.

### 4.13 Search (exact-term)

```powershell
seedgraph search "<query>" --project <slug> [--work ID] [--section-kind KIND] [--limit 20]
```
FTS5 exact-term span search (no stemming) — the deterministic building block `ask`
also uses.

### 4.14 Review queue

```powershell
seedgraph review list <slug>
seedgraph review resolve <slug> <item_id> <approve|reject|merge|re_resolve|exclude|split>
```
Every ambiguous identity/anchor/merge decision routes here instead of silently
guessing.

### 4.15 Other useful commands (project / cache / spans utilities)

Not pipeline stages, but useful day to day:

```powershell
seedgraph project list
seedgraph project show <slug> [--status included|metadata_only|excluded ...]
seedgraph project set-status <slug> <work_id> <included|metadata_only|excluded> [--reason R]

seedgraph cache stat                                  # counts + bytes + resolved cache root
seedgraph cache show <file_hash-or-markdown_hash>      # provenance/manifest for a cached artifact
seedgraph cache ingest <path> [--access-class C] [--acquisition-method M]   # hash/dedup/store, no convert
seedgraph cache convert <file_hash-or-source_file_id> [--use-llm/--no-llm] [--allow-external-llm] [--force]
seedgraph cache add <path> [--access-class C] [--acquisition-method M] [--use-llm/--no-llm] [--force]  # ingest+convert in one step

seedgraph spans create <slug> --markdown-id <id> (--quote "..." | --start N --end N) [--kind manual]
seedgraph spans get <slug> <span_id> [--json]
seedgraph spans verify <slug> [<span_id> | --all]      # re-checks exact_quote against current markdown
seedgraph spans reanchor <slug> --work <work_id>        # relocate stale spans after a re-conversion
```

`cache add`/`convert --use-llm` enable Marker's LLM-assisted math correction (not
yet wired through the web upload path); `--allow-external-llm`
is required alongside it for a private document (see §7).

## 5. Storage layout & the two scopes

Two SQLite scopes under `~/.seedgraph` (or `--root` / `$SEEDGRAPH_HOME`):

```text
~/.seedgraph/
  config.yaml                     # global config overlay (see §6)
  cache/
    cache.db                      # CONTENT scope: source_files, markdown, provider_cache
                                   # content-addressed ids: sf_<sha256(pdf)>, md_<sha256(markdown)>
  projects/
    <slug>/
      project.db                  # PROJECT scope: works, project_documents, evidence_spans,
                                   #   document_sections, structured_notes, concepts,
                                   #   project_graph_edges, review_queue, llm_key_refs,
                                   #   llm_usage_events, ...
      project.yaml                # per-project override patch (thin overlay, not a full dump)
      runs/
        <run_id>/
          manifest.json           # per-stage counts/provenance for this run
          graph.json              # citation/concept graph export
          events.jsonl            # background-job progress log (UI polls this)
      answers/                    # {answer_id}.json + .trace.json per ask (§4.12,
                                   #   always-on — no longer gated by `--save`)
      lenses/
        <lens_id>.yaml            # lens definitions
```

`cache.db` is content-addressed and **deduped across projects** — the same PDF
uploaded to two projects converts once. `project.db` holds all project-specific
state and is safe to delete independently of the cache (works must be re-added).
Both scopes run with `foreign_keys=ON` + WAL.

## 6. Config & LLM routing

Merge order (project overrides win): built-in defaults -> `~/.seedgraph/config.yaml`
(global) -> `projects/{slug}/project.yaml` (project). A project may only **choose**
a routing profile per task, never define a new profile or redirect an
endpoint/key — that stays machine-global (a hand-edited `project.yaml` carrying
`llm.profiles`, `provider`, `env_var`, `base_url`, or `key_source` is stripped/
rejected).

Shipped default profiles (`~/.seedgraph/config.yaml` `llm.profiles`):

| profile_id | provider | access_mode | model |
|---|---|---|---|
| `anthropic_api_default` | anthropic | api_key (`ANTHROPIC_API_KEY`) | `claude-sonnet-4-6` |
| `local_ollama_default` | ollama | local | `llama3` (repoint to `qwen3.5` — the verified-working local model) |
| `local_embedding_model` | ollama | local | `nomic-embed-text` |
| `no_llm` | none | disabled | — |

Shipped default routes (`llm.routes.<task>`): `note_extraction` prefers
`local_ollama_default` -> falls back to `anthropic_api_default`;
`answer_generation` prefers `anthropic_api_default` -> falls back to `no_llm`;
`semantic_graph_extraction` / `project_lens_extraction` /
`metadata_extraction` prefer local Ollama -> `no_llm`; `reference_parsing` /
`reranking` are `no_llm` with a genuine deterministic path.

**Cheap-model option for concept canonicalization** — an opt-in profile
`anthropic_api_cheap` (anthropic, `claude-haiku-4-5`, same `ANTHROPIC_API_KEY`)
ships alongside `anthropic_api_default`, restricted to
`semantic_graph_extraction` only. Concept canon is a small controlled-vocabulary
subset-selection task over pre-clustered labels whose correctness is carried by
the deterministic guardrails (label closure, veto stoplist, type fence), not the
model, so the cheaper/faster Haiku-class model ($1/$5 per MTok vs. sonnet's
$3/$15) is strictly better there. No default route references it (ADR-0006
local-first stands); opt in per project with
`seedgraph llm route set --task semantic_graph_extraction --preferred anthropic_api_cheap --project SLUG`.
Its `allowed_tasks` fence means routing any *other* task to it simply falls
through to that route's fallback.

**Let a Claude Code subagent be the model** — the `mailbox` provider makes no API
call: it writes each request to a file and blocks until an answer file appears, so
any agent with disk access can answer. The shipped profile `claude_code_subagent`
(provider `mailbox`, model `claude-code-subagent`, local, no key) is allowed
`note_extraction`, `semantic_graph_extraction` and `metadata_extraction`. Opt in with
`seedgraph llm route set --task note_extraction --preferred claude_code_subagent`.
Protocol: seedgraph writes `<dir>/requests/<id>.json` (`system_prompt`,
`user_prompt`, `model`, `temperature`, `max_tokens`); the agent writes the whole
answer — nothing else — as `<dir>/responses/<id>.txt` (or `<id>.json` with a `text`
key) in a single write, and seedgraph moves the request to `<dir>/done/`. The
mailbox directory is the profile's `base_url`, else `SEEDGRAPH_MAILBOX_DIR`, else
`~/.seedgraph/mailbox`.

**To change a route** — three ways, easiest first:
1. **CLI** — `seedgraph llm route set`:
   ```powershell
   seedgraph llm route set --task note_extraction --preferred anthropic_api_default [--fallback local_ollama_default]
   seedgraph llm route set --task note_extraction --preferred anthropic_api_default --fallback ""   # clear fallback
   seedgraph llm route set --task answer_generation --preferred local_ollama_default --project SLUG # project override
   ```
   `--task` must be an existing route; omit `--fallback` to keep the current one, pass `""` to clear it. Without `--project` it writes the global `config.yaml`; with `--project` a project override. An unknown profile is rejected before anything is written.
2. **Settings UI** — `/ui/settings` (global) or a project's Settings tab (per-field
   override/inherit, `POST /ui/projects/{slug}/settings`) lets you pick the
   preferred/fallback profile per task.
3. **Hand-edit** `~/.seedgraph/config.yaml`:
   ```yaml
   llm:
     routes:
       note_extraction:
         preferred_profile: anthropic_api_default
         fallback_profile: local_ollama_default
   ```

Inspect the live routing table and profile availability:
```powershell
seedgraph llm profiles list [--project SLUG]
```

Key references (never values) and preflight cost:
```powershell
seedgraph llm keys list --project SLUG
seedgraph llm keys set --project SLUG --provider anthropic --env-var ANTHROPIC_API_KEY [--access-mode api_key] [--key-source environment]
seedgraph llm keys remove --project SLUG --provider PROV [--env-var VAR]
seedgraph llm keys test --project SLUG [--provider PROV] [--smoke]   # --smoke makes ONE real (paid) call
seedgraph llm usage --project SLUG [--group-by task|model|provider|date] [--month YYYY-MM]
seedgraph llm estimate --project SLUG --task TASK --input-tokens N [--output-tokens N] [--access-class open_access]
```
`llm keys set` only stores an env-var **name**, never a secret value — the real
key lives in your shell environment (`ANTHROPIC_API_KEY=...`).

## 7. Privacy & content-access gates

Every source has an access class; **uploaded PDFs are `user_supplied_private`**.
Global policy (`~/.seedgraph/config.yaml` `content_policy`) is a **tighten-only
ceiling** — a project may make a gate stricter but never loosen it:

| Gate | Default | What it controls |
|---|---|---|
| `content_policy.external_llm_for_private_full_text` | `false` | May a private paper's *full text* leave the machine to an external (non-local) LLM? Off by default — private full text stays local unless you flip this, and `--confirm-external` on `extract notes` is still required per-run. |
| `content_policy.external_llm_for_answer_generation` | `false` | May *bounded evidence fragments* (not full text) from a private source leave the machine at answer time? Separate, narrower gate than the one above. |
| `content_policy.allow_external_llm` | `true` | Master external-LLM switch. |

A project override can only AND these down, never flip them back on. See
`CONTENT_ACCESS_POLICY.md` for the full access-class model (open access vs.
user-supplied-private vs. future licensed content) and product-wording rules.

## 8. Environment variable reference

| Name | Purpose | Default |
|---|---|---|
| `SEEDGRAPH_HOME` | Override the seedgraph home directory (same as `--root`). | `~/.seedgraph` |
| `ANTHROPIC_API_KEY` | API key resolved for the `anthropic_api_default` profile's `env_var`. | unset |
| `SEEDGRAPH_MARKER_SLOTS` | Marker-conversion worker pool size (concurrency cap). | `torch.cuda.device_count()`, else `1` |
| `SEEDGRAPH_MARKER_TIMEOUT` | Per-conversion subprocess timeout (seconds). | `1800` |
| `SEEDGRAPH_LOG_LEVEL` | Log level. | `INFO` |
| `CORE_API_KEY` / `SEEDGRAPH_CORE_KEY` | CORE provider key (either name works; without one, CORE sits out of the acquisition chain). | unset |
| `SEEDGRAPH_FAKE_PROVIDERS` | Test/offline-demo only — swaps in a fake OA provider chain + fake marker backend for `corpus`/`cite`. Not for normal use. | unset |
| `SEEDGRAPH_FAKE_MARKER` | Test-only — fake Marker backend for `cache convert`. Not for normal use. | unset |

**Polite pool:** set a top-level `contact_email: your@email` in
`~/.seedgraph/config.yaml` (or per-project) to activate the polite pool for
OpenAlex, Crossref, and Unpaywall. Unpaywall in particular needs it — it
runs at lower priority (or refuses) without one.

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Uploads stay `queued`/never convert; no error banner | `seedgraph serve` is running under a Python interpreter without the `marker` extra | Reinstall/run `serve` from the interpreter where you ran `pip install -e .[marker]` (e.g. a dedicated `G:\python313`). |
| `corpus resolve`/`walk`/`acquire` never uses Unpaywall or CORE | Missing `contact_email` (polite-pool identifier — see §8) or missing `CORE_API_KEY`/`SEEDGRAPH_CORE_KEY` | Set a top-level `contact_email: your@email` in `config.yaml` (global or per-project) to activate the OpenAlex/Crossref/Unpaywall polite pool; set `CORE_API_KEY` for CORE. |
| `extract notes` / `ask` refuses on a private PDF, or silently degrades to `no_llm`/retrieval-only | `content_policy.external_llm_for_private_full_text` (or `..._for_answer_generation`) is `false`, and the preferred profile is external | This is the intended fail-closed gate (§7) — flip the policy explicitly (global or project, tighten-only in the project direction) or use a local profile instead of an external one; per-run `--confirm-external` on `extract notes` is also required. |
| Marker crashes with a `c10::Half` overflow / CUDA error | Two conversions ran concurrently in one process | Already handled by design — each conversion runs in its own subprocess pinned to one GPU via `CUDA_VISIBLE_DEVICES`; this should not recur. If it does, it's a real bug, not a config issue. |
| `--allow-remote` set, but a remote browser gets 403 on every screen | Working as designed — the per-request loopback+cookie gate (`require_local_session`) blocks all non-loopback clients regardless of bind (§3). | Remote private access is intentionally not enabled; there is no supported workaround today. |
| `seedgraph llm profiles list` shows a profile as unavailable | No API key resolves for that `env_var`, or (for `openai`) no adapter ships | Set the env var (`llm keys set` + your shell env), or use a shipped provider (`anthropic`, `ollama`). |
| `cite parse` errors or finds nothing | `sections build` was never run for that work | Run `sections build --work <id>` first (§4.6), then `cite parse`. |
| `ask` returns `[insufficient evidence]` | Retrieval found nothing above the evidence floor, or `--no-llm` with no strong matches | Broaden the question, check `search`/`corpus status`/`spans index` ran, or lower expectations for `--no-llm` mode (retrieval-only never fabricates prose). |

## 10. MCP server

Seedgraph ships an **MCP (Model Context Protocol) server** — `seedgraph mcp serve` — that
exposes the read/query surface to any MCP host (Claude Code, Claude Desktop, the MCP
Inspector). It is a thin in-process adapter over the same service layer the CLI uses: the
CLI stays the complete surface, and MCP is a consumer of it, like the `serve` web UI. It
speaks JSON-RPC over stdin/stdout and is spawned by the host, not run by you.

The full setup and registration walkthrough is in §10.4 below.

### 10.1 What the server exposes

**16 tools**, all read-only:

| Group | Tools |
|---|---|
| canary | `version` |
| project / corpus reads (9) | `project_list`, `project_dashboard`, `list_documents`, `corpus_status`, `review_list`, `run_list`, `run_events`, `lens_results`, `lens_staleness` |
| concept / search / graph (5) | `concepts_overview`, `concepts_list`, `concept_show`, `search_spans`, `graph_analyze` |
| answer (1) | `ask` |

Every tool except `project_list` requires an explicit `project` slug — the server never
guesses one. Job launch and write/verdict tools (`job_launch`, `review_resolve`,
`set_inclusion_status`) are **v2 and deliberately not registered**: nothing on this surface
can mutate a project.

**2 resources and 1 resource template:**

| URI | Listed under |
|---|---|
| `seedgraph://docs/quickstart` | `resources/list` |
| `seedgraph://docs/user-handbook` | `resources/list` |
| `seedgraph://projects/{slug}/graph.json` | `resources/templates/list` |

`graph.json` carries a `{slug}` parameter, which makes it a **resource template** — it is
absent from the plain resource list and appears only in the template list (in the Inspector:
**Resources → Resource Templates**). It serves the *public* graph export through the
default-deny export guard, with `allow_private` hard-false and no parameter that can relax
it. **No resource serves `cache.db` content** (PDF or markdown full text) — that stays out
of MCP scope entirely.

The two docs resources read `docs/` **repo-relative**, so they need the editable install
below. From a non-editable wheel (which does not ship `docs/`) they return a clean
`doc_not_found` error instead of content.

**1 prompt:** `semantic_query(term, project=None)` — emits the overview-first concept
workflow (`concepts_overview` → typed slice → `concept_show` → `ask` → `graph_analyze`)
together with the honesty rules the answer must carry: `weight` is IDF-style
discriminativeness rather than importance, concept selection is judgment rather than a
similarity score, semantic reach is not citation reach, quotes ground in evidence spans
rather than concept labels, `metadata_only` means thin evidence, and a negative answer is a
valid answer.

### 10.2 Cost posture

**The MCP surface cannot spend money by accident.** `ask` defaults to `no_llm=true` — the
free, local, deterministic retrieval floor: FTS5 retrieval and ranking returning a cited
evidence list with no prose and no model call. On that path the tool loads no config,
resolves no route, and runs no budget query.

Reaching a paid model requires **both**:

1. an explicit **`no_llm=false`** (no other argument, and no default, gets there); **and**
2. passing the **budget handshake**. This gate is *conditional*, not a confirm-every-call
   prompt: before dispatch the server prices the prospective call and folds it into your
   month-to-date spend. It fails closed with `budget_confirmation_required` only if the
   estimate clears `budget.require_confirmation_above_usd` **or** the projection crosses
   `budget.monthly_soft_limit_usd`. A cheap call under both thresholds just proceeds. When
   it does fire, the error carries `estimate_usd`, `monthly_spend_usd`, `monthly_limit_usd`,
   `profile`, and which trigger tripped — re-call with **`confirm_spend=true`** to proceed.
   Confirming when nothing needed confirming is a harmless no-op.

Note this is a **consent gate, not a cap**: the spend cap is advisory (dogfooding found a
run exceed it), so the handshake is what actually stands between the MCP surface and an
unintended paid call.

A budget block after that point is not an error but an **honest degrade**: the call
succeeds, `mode` comes back `retrieval_only`, and `budget_exceeded` appears in the
envelope's `warnings`. Read `warnings`, not just the error channel. A retrieval-only
envelope is a real answer, not a failure.

**Every `ask` call persists**, same as the CLI (§4.12): both the envelope and its
`AnswerTrace` are written under `projects/{slug}/answers/`, unredacted on disk
regardless of `--redact-private` (which governs only the tool's response payload,
never the files). This reverses the original stance that an answer is a query, not
a run (decisions 16/35, amended not repealed — `run_id` still only controls the
nesting path). The response shape is unaffected; the returned `answer_id` is the
lookup key for the files.

### 10.3 The `--redact-private` valve

Starting the server with `--redact-private` blanks full-text-derived fields for
**non-shareable** works across every tool response: span `exact_quote` / `quote_text`,
claim `claim_text`, and `ask` citation `quote`. Metadata (titles, labels, years, ids,
counts, concept labels) is never touched.

What it does **not** cover, and why that is fine:

- **The `graph.json` resource.** Its export guard is strictly stronger — it *drops*
  non-shareable concepts and edges entirely rather than blanking a field on a row that
  still ships. There is nothing left for the filter to withhold.
- **The load-bearing access gates.** `--redact-private` is a **best-effort valve, not a
  security boundary.** The real gates are the content-access policy gates (§7), which apply
  regardless of this flag. Default off gives the same visibility the `/seedgraph` skill has
  today; turn it on when a host or transcript may be shared more widely than the corpus.

### 10.4 Registration

**Prerequisite** — install the extra in the interpreter the host will run:

```powershell
<repo>\.venv\Scripts\python.exe -m pip install -e ".[mcp]"
<repo>\.venv\Scripts\python.exe -m seedgraph mcp serve --help
```

The base package never imports the MCP SDK, so a no-extra install is unaffected; the
command exits 1 with an install hint if the extra is missing.

**Claude Code, user scope** (everything after `--` is the command line):

```powershell
claude mcp add seedgraph -- <repo>\.venv\Scripts\python.exe -m seedgraph mcp serve --root C:\Users\<you>\.seedgraph
```

**Claude Code, project scope** — a `.mcp.json` at the working-directory root. Copy
`.mcp.json.example` from the repo root and edit the paths (backslashes doubled: JSON):

```json
{
  "mcpServers": {
    "seedgraph": {
      "command": "<repo>\\.venv\\Scripts\\python.exe",
      "args": ["-m", "seedgraph", "mcp", "serve", "--root", "C:\\Users\\<you>\\.seedgraph"]
    }
  }
}
```

**Claude Desktop** — edit (create if absent)
`%APPDATA%\Claude\claude_desktop_config.json`, i.e.
`C:\Users\<you>\AppData\Roaming\Claude\claude_desktop_config.json`, with the same
`mcpServers` block, then **fully restart Claude Desktop from the tray** (closing the window
is not enough).

Prefer the **`--root` flag** over a `SEEDGRAPH_HOME` env var: MCP hosts pass a filtered
environment to the child process, so the env var may not survive; the flag always does. Add
`"--redact-private"` to `args` to enable the valve (§10.3).

Verify in Claude Code with `/mcp` — `seedgraph` should list as connected, with tools named
`mcp__seedgraph__project_list` and so on.

### 10.5 Smoke-test with the MCP Inspector

Requires Node. This spawns the server and opens a browser UI to invoke everything by hand:

```powershell
npx @modelcontextprotocol/inspector D:\documents\research\seedgraph_v2\.venv\Scripts\python.exe -m seedgraph mcp serve
```

Sequence: `version` → `project_list` → `project_dashboard` on a real slug →
`concepts_overview` → `concepts_list` → `ask` with `no_llm=true`. Then load the
`seedgraph://docs/quickstart` resource and render the `semantic_query` prompt.

**Looking for `graph.json`?** It will not be in the Resources list. Because its URI carries
`{slug}` it is a resource *template* — open **Resources → Resource Templates**, pick
`project_graph_json`, fill in `slug`, then read.

### 10.6 Reading errors from a client

Tools raise a structured `{"code", "message", "detail"}` body, but the SDK **re-wraps** it,
so what arrives is prefixed:

```
Error executing tool project_dashboard: {"code": "project_not_found", ...}
```

Resource errors are wrapped the same way (and in SDK 1.28.1 the resource prefix arrives
*doubled*). A client branching on `code` must therefore **slice from the first `{` and
parse that** — never assume the JSON starts at index 0, and never match on the prefix text.
Codes: `project_not_found`, `invalid_mode`, `concept_not_found`, `no_citation_run`,
`no_graph_run`, `doc_not_found`, `budget_confirmation_required`, `gate_refused`.

Note also that `serverInfo.version` reports the **MCP SDK** version, not seedgraph's — call
the `version` tool for the package version.

Real stdio was verified end-to-end on Windows 11 (CPython 3.13, MCP SDK 1.28.1) with no
code change required; frames terminate `\r\n` rather than bare `\n`, which both official
SDKs handle. Two opt-in `mcp_client`-marked tests pin this
(`pytest tests/test_mcp_server.py -m mcp_client`); the default suite stays subprocess-free.
