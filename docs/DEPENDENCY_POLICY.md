# Dependency policy

Seedgraph runs in two separate, locked Python environments. Both use Python 3.13,
managed by uv through each project's `.python-version` file.

## The two environments

| Environment | Where | Holds | Lock file |
|---|---|---|---|
| App | repo root | seedgraph: CLI, MCP server, database | `uv.lock` |
| Converter | `converter/` | Marker, Surya, torch (CUDA build) | `converter/uv.lock` |

They are joined by one narrow contract: **PDF in, markdown + page info out.**
The converter never imports seedgraph and never touches its database; the app
saves what the converter returns. The app environment never contains torch,
Marker or Surya.

They are two independent uv projects, not a uv workspace, so each has its own
resolution, lock and `.venv`.

## Pinning

Every package in both environments is pinned exactly in its `uv.lock`. The lock
files are committed. `pyproject.toml` states the direct requirements; the lock
decides the exact versions that get installed.

The converter's working set is Marker 1.10.2, Surya 0.17.1 and torch
2.11.0+cu128 (from the PyTorch CUDA 12.8 index).

## When versions change

- **No automatic updates.** Nothing is upgraded by a bot, a schedule or a
  routine `uv lock --upgrade`.
- **App environment:** bumped on request or for a security fix. An agent makes
  the change, re-locks, runs the full test suite, and the researcher approves
  before it is merged.
- **Converter environment:** bumped only after the new versions pass the
  golden-set check (page counts, references found, quote re-matching, equations
  and tables, speed). Newer torch builds (for example CUDA 13) are adopted the same
  way.
- **Marker 2** is only a later option, and only if it does not force every user
  to install Docker (or a similar non-Python service).

## Running the converter

Always call the converter with the lock as-is:

```
uv run --frozen --project converter ...
```

Plain `uv run` re-locks and re-syncs the environment when it thinks it is out of
date, which can silently change what is installed. `--frozen` (or `--no-sync`)
uses exactly what the lock says.

The app reaches the converter only through warm worker processes, one per GPU,
over a stdin/stdout pipe. The message format is in
[CONVERTER_PROTOCOL.md](CONVERTER_PROTOCOL.md).

## Keys and local services (environment variables only)

The app stores no secrets and has no keyring dependency (removed in v3 stage 2,
with the hosted-LLM layer). Scholarly-API keys are read from environment
variables only; every one is optional. `seedgraph-v3 setup` and the Settings page list which are set
(names only, never values).

| Variable | Used for |
|---|---|
| `SEEDGRAPH_CONTACT_EMAIL` | Polite-pool contact email for OpenAlex, Crossref and Unpaywall. Not a secret; `contact_email` in `config.yaml` is the fallback. |
| `S2_API_KEY` | Semantic Scholar API key (raises the rate limit). |
| `OPENALEX_API_KEY` | OpenAlex API key. |
| `CORE_API_KEY` | CORE API key; CORE joins the provider chain only when it is set. `SEEDGRAPH_CORE_KEY` is the legacy name. |
| `SEEDGRAPH_OLLAMA_URL` / `OLLAMA_HOST` | Where the local Ollama server is (`work ollama`); default `http://localhost:11434`. |
| `SEEDGRAPH_OLLAMA_MODEL` | Default model for `work ollama`. |

The config keys `s2_api_key`, `core_key` and `openalex_key`, and keyring
entries, are no longer read.
