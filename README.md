# Seedgraph v2

Seedgraph is a **local-first grounded reasoning colleague** for research. It turns a
researcher-selected scholarly corpus into a project-specific conceptual frame that an LLM
reasons *inside* — so the assistant contributes like a colleague who has read the
literature, not like a search box that returns facts.

The atomic unit of value is a **conceptual orientation**, not a retrieved passage. A fact
lookup is the degenerate, shortest-possible case; the point is that the assistant's
reasoning runs inside *this* project's actual conceptual landscape — its concepts, citation
lineage, assumptions, disagreements, and gaps — instead of the assistant's generic training
prior. A colleague who has read the papers does not fetch quotes on demand; they have views,
they notice things, and they can say *"these two literatures are more connected than people
admit, and here is the evidence."* That is the target behavior.

> **Seedgraph is Claude Code for a research corpus.** Just as a coding agent reaches for
> `grep` without being told, an assistant wired to a Seedgraph project reaches for the
> project graph on any research-substantive turn — automatically (see *the ambient trigger*
> below). The failure mode it is designed against is **silent non-use**: answering from the
> training prior and never noticing the topic lives in the corpus.

## What "colleague, not lookup" means in practice

- **Orientation is the deliverable.** The graph routes attention; the assistant reasons over
  *this* corpus's structure, not the open sprawl of its training.
- **Grounded, labeled judgment is first-class.** The target answer is two parts: (1) what the
  corpus actually says — grounded, evidence attached, honest negatives included ("no paper
  states X") — then (2) the assistant's own inference, explicitly marked as judgment, with the
  evidence trail shown. A colleague is trusted because they *show their work*, not because they
  never venture a view. Surfacing **latent structure** the literature hasn't made explicit
  (a link between two concepts via a shared third) is exactly the kind of contribution
  Seedgraph exists to enable.
- **Three trust tiers, always labeled.** Navigate primarily by the **semantic overlay**
  (LLM-inferred concepts + relations — a *hypothesis to check*, never fact alone); drill into
  the **citation graph** (deterministic lineage) and **evidence spans** (verbatim,
  quote-hashed — ground truth) to support any claim. A semantic edge is never stated as fact
  without a span behind it.
- **Provenance is the guardrail, not the product.** Evidence spans and content-access gates
  are what *earn* the colleague trust — they are not the thing being sold.

**Success** splits accordingly: a **testable trust floor** (substrate fidelity + judgment
honesty + calibration — auditable) and an **in-use value ceiling** (does the corpus-oriented
assistant reason measurably better than a corpus-blind one; does it help the researcher
think) — judged in use, deliberately never reduced to a benchmark.

## The substrate — an honest compression of the corpus

The colleague reasons over a deterministic artifact lineage. Each stage is provenanced and
locally stored; the graph is a faithful, auditable map, not ground truth:

```text
source PDF / HTML
    ↓
local conversion artifact  (Marker → markdown + block metadata)
    ↓
evidence spans  (verbatim, quote-hashed)
    ↓
typed extracted claims
    ↓
structured notes
    ↓
citation graph (deterministic)  +  project semantic overlay (LLM-inferred)
    ↓
the colleague's grounded reasoning
```

The graph routes attention. Source-linked evidence carries factual accountability. The engine
below this framing — conversion, spans, identity/merge, the deterministic citation graph, the
semantic overlay, FTS5 — is deep and unchanged by the colleague reframe; it is the accurate
map the colleague reads.

> **Zoom in on one stage:** how the reader turns a paper into typed claims and
> structured notes — and why a fixed extraction schema beats asking an LLM to
> *"summarize this paper"* — is documented in
> [`src/seedgraph/extraction/README.md`](src/seedgraph/extraction/README.md).

## Requirements

- **Python ≥ 3.11.** SQLite ships with Python — no separate database server.
- **Core dependencies install automatically** (`typer`, `fastapi`, `pydantic`, `sqlmodel`,
  `httpx`, `networkx`, `jinja2` + `uvicorn` for the web UI, `keyring` for OS-native secrets).
- **Optional extras:** `.[marker]` for real PDF→markdown conversion (heavy, GPU-friendly — the
  test suite uses a fake and never needs it); `.[dev]` for the test suite; `.[browser]` for the
  Playwright UI smoke tests.
- **An LLM backend is optional.** The entire deterministic spine runs with `--no-llm`. For
  synthesized answers, concept proposal, and note extraction, wire either a **local model**
  (Ollama, e.g. `qwen3.5`) or an **API key** (`ANTHROPIC_API_KEY`, OpenAI, or Gemini). Local is
  the default; nothing assumes a paid subscription grants programmatic access.

## Installation

```bash
git clone https://github.com/grycrnwll/seedgraph_v2.git
cd seedgraph_v2
pip install -e .            # core (installs the `seedgraph` CLI)
pip install -e .[marker]    # optional: real PDF→markdown conversion (heavy, GPU-friendly)
```

Optionally wire a model for the non-deterministic paths:

```bash
ollama pull qwen3.5             # local, or…
export ANTHROPIC_API_KEY=...    # hosted fallback (PowerShell: $env:ANTHROPIC_API_KEY = "…")
```

Verify the install with `seedgraph version` and `seedgraph doctor`.

## Quick start

Build a small corpus from the CLI, then hand it to the colleague. (The web UI — `seedgraph
serve` — does the same with drag-and-drop seed upload.) Every step below is deterministic and
runs without a model; `--no-llm` is only needed where an LLM would otherwise be used.

```bash
# 1. create a project and add a seed paper
seedgraph project new my_project
seedgraph project add my_project --doi 10.xxxx/example --seed

# 2. grow + resolve the corpus against live providers (the one network phase)
seedgraph corpus run my_project --depth 2          # resolve → walk → acquire

# 3. index evidence spans, then build the deterministic citation graph
seedgraph sections build my_project --work <work_id>  # per work
seedgraph spans index my_project --work <work_id>     # per work (the UI plan task loops over all)
seedgraph cite run my_project

# 4. build the concept overlay (drop --no-llm once a model is wired)
seedgraph concepts build my_project --no-llm

# 5. ask — retrieval-only floor with --no-llm; synthesized + cited without it
seedgraph ask "What identification assumptions do these papers rely on?" \
  --project my_project --no-llm

# 6. wire the ambient colleague into your working directory (see next section)
seedgraph project agent-setup my_project --dir .
```

Every `ask` above persists the answer envelope + trace pair unconditionally — `--save` is a
no-op — and echoes both `saved <path>` lines; the pair is also inspectable on disk (web view +
`answer trace-export`). See `docs/USER_HANDBOOK.md` §4.12.

Full happy-path walkthrough: **`docs/QUICKSTART.md`**. Gotchas, config, storage layout, and
privacy gates: **`docs/USER_HANDBOOK.md`**.

## How you use it — the ambient trigger

Seedgraph is **ambient**: it shapes the assistant's contributions on essentially every
research-substantive turn about the project, without being explicitly invoked per fact.
Because the assistant cannot hold a 500-paper corpus in context, "ambient" means **reliable
self-triggering**, installed by two carriers:

- a per-project `CLAUDE.md` norm block (always-loaded project memory), and
- a proactive `seedgraph` skill whose description *is* the trigger.

One command wires both into a working directory:

```bash
seedgraph project agent-setup <project-slug> --dir <your research working dir>
```

Then you open a session in that directory and just work — the assistant reaches into the
corpus on its own. The trigger drives the CLI (the CLI is the complete surface; every other
consumer is thin). An MCP server exposing the same surface as schema-validated tools is
also available — see the next section.

## MCP server

`seedgraph mcp serve` exposes the read/query surface over the **Model Context Protocol**, so
any MCP host (Claude Code, Claude Desktop, the MCP Inspector) gets schema-validated tools
instead of shell-parsed CLI output. Like the web UI, it is a thin consumer of the one service
layer — the CLI remains the complete surface.

```powershell
pip install -e ".[mcp]"                                   # the optional extra
claude mcp add seedgraph -- <python> -m seedgraph mcp serve   # register with Claude Code
```

It serves **16 read-only tools** (project/corpus reads, the concept-overlay and search/graph
tools, and `ask`), two docs resources plus a `graph.json` resource template, and the
`semantic_query` prompt that carries the honesty rules. Nothing on the surface can mutate a
project, and `ask` defaults to `no_llm=true` — the free retrieval floor — so nothing spends
unless you explicitly pass `no_llm=false`. The budget-confirmation handshake is **conditional**:
it fires only when the project's budget policy arms `require_confirmation_above_usd` or the
monthly soft limit is crossed — with no policy armed, a paid call proceeds unprompted, so arm
that setting before using the paid path (the spend cap is advisory; a hard stop is not yet
implemented).

> "Read-only" means no project mutation — `ask` still writes its answer + trace artifact
> files to disk on every call (see above); that is artifact persistence, not project state.

Setup, registration snippets, the `--redact-private` valve, Inspector smoke steps, and the
client-side error contract: **`docs/USER_HANDBOOK.md` §10**. A copyable project-scope config
ships as `.mcp.json.example`.

## Architecture

Seedgraph v2 is Python-first and local-first.

- Core language: Python.
- CLI: Typer (the complete surface; the API/UI/MCP are thin consumers of one service layer).
- Local API/web surface: FastAPI (a localhost read/curation window into the same substrate).
- Database: SQLite (two scopes — a machine content cache and one `project.db` per project).
- Local full-text search: SQLite FTS5.
- Graph algorithms and construction: NetworkX.
- PDF conversion backend: Marker.
- Future hosted/institutional target: PostgreSQL + pgvector.

The durable source of truth is the relational artifact lineage and project state. Graphs are
computed or exported views over those records.

## LLM access posture

Seedgraph v2 treats LLM subscriptions, API keys, and local models as distinct access modes.

```text
Subscriptions:  human-facing application entitlements.
API keys:       programmatic execution credentials.
Local models:   local execution backends controlled by the user.
```

Automated Seedgraph tasks require an API key, an official provider integration, a local model
backend, or the no-LLM fallback. The design does not assume a consumer subscription grants
programmatic model access. The deterministic / no-LLM spine (resolve → walk → convert → spans →
citation graph → concept overlay → FTS search → `ask --no-llm`) is functional end-to-end
without any model call.

## Storage posture

Seedgraph v2 is local-first by default.

```text
Local private content cache
    PDFs, Marker output, markdown, spans, notes, embeddings, semantic overlays.
    Scope: one user / one machine / one lawful research environment.

Project overlay
    Corpus decisions, selected papers, project lenses, extracted claims, graph edges, review state.

Optional shared metadata cache
    Public metadata, identifiers, citation metadata, open-access links, schema templates.

Future licensed content layer
    Only if publisher or institutional entitlements exist. Access-controlled and license-aware.
```

Do not cross-user-share PDFs, converted markdown, full-text embeddings, evidence spans, or
detailed notes derived from restricted full text by default. Grounding a colleague *raises*
the value of these gates; it does not relax them.

## Documentation

- `docs/QUICKSTART.md` — the one-page happy-path walkthrough.
- `docs/USER_HANDBOOK.md` — config, storage layout, and the full command reference (MCP setup in §10).
- `CONTENT_ACCESS_POLICY.md` — the access-class model and content-sharing rules.
- `src/seedgraph/extraction/README.md` — what the reader extracts from each paper, and why (the schema-driven-extraction explainer).
