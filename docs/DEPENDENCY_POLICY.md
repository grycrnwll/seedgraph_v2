# Dependency policy

Seedgraph runs in two separate, locked Python environments. Both use Python 3.13,
managed by uv through each project's `.python-version` file.

## The two environments

| Environment | Where | Holds | Lock file |
|---|---|---|---|
| App | repo root | seedgraph: CLI, web UI, MCP server, database | `uv.lock` |
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
