# Quickstart

Zero to a converted, walked, graphed corpus you can `ask`. Details, gotchas, and
config live in **[USER_HANDBOOK.md](USER_HANDBOOK.md)** — this page is the happy
path only.

Windows/PowerShell shown; commands are otherwise shell-portable.

## 0. Install

```powershell
pip install -e .
pip install -e .[marker]   # real PDF->markdown conversion (heavy, GPU-friendly)
```

Pull a local model for Ollama (or set `ANTHROPIC_API_KEY` for the hosted fallback
— see handbook §6):

```powershell
ollama pull qwen3.5
```

## 1. Start the app

Run this from the Python interpreter that has the `marker` extra installed
(uploads silently fail to convert otherwise — handbook §9):

```powershell
seedgraph serve --open-browser
```

Open the printed one-time `http://127.0.0.1:8765/auth?t=...` link once to set your
session cookie.

## 2. Create a project + add seeds

Via the UI: `/ui` → "New project" → open it → drag-and-drop seed PDFs onto the
corpus page. Or headless from the CLI:

```powershell
seedgraph project new my_project
seedgraph project add my_project --doi 10.xxxx/example --seed
```

## 3. Wait for conversion, then backfill metadata

The marker queue converts uploads in the background (corpus page shows
queued/converting). Then:

```powershell
seedgraph corpus identify my_project
```

## 4. Grow + resolve the corpus (network phase)

```powershell
seedgraph corpus run my_project --depth 2
```

(Umbrella for `resolve` → `walk` → `acquire` under one run.)

## 5. Index evidence + build the citation graph

```powershell
seedgraph sections build my_project --work <work_id>
seedgraph spans index my_project --work <work_id>
seedgraph cite run my_project
```

Repeat `sections build` / `spans index` per included work with markdown (or use
the UI's "Sections build" plan task, which loops over all of them).

## 6. Concepts

```powershell
seedgraph concepts build my_project --no-llm
```

Drop `--no-llm` once a profile (local Ollama or `ANTHROPIC_API_KEY`) is wired.

## 7. Ask

```powershell
seedgraph ask "What identification assumptions do these papers rely on?" --project my_project --no-llm
```

Drop `--no-llm` for a synthesized, cited answer once an LLM profile is available.
Every ask leaves an inspectable trace (web view + `answer trace-export`) — see
**[USER_HANDBOOK.md §4.12](USER_HANDBOOK.md#412-ask)**.

For install prerequisites, storage layout, privacy gates, and troubleshooting,
see **[USER_HANDBOOK.md](USER_HANDBOOK.md)**.
