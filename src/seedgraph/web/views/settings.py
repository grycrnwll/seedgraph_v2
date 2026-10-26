"""Settings (design §2.7): paths, converter, Ollama and the scholarly-API environment
variables (names only). Pure reads: nothing here starts a converter, probes a GPU or
calls Ollama; the Ollama result shown is the last "check again" (``ollama_check``).

Rule chosen where the design is silent: the MCP registration status is not shown,
because reading it means opening the agents' own config files (which may hold secrets).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...library.store import Library


@dataclass(frozen=True)
class EnvVar:
    name: str
    purpose: str
    present: bool


@dataclass(frozen=True)
class Converter:
    locked_version: str | None
    started: bool               # the pool exists (the server's job runner resolved it)
    slots: int | None
    gpus: tuple[str, ...]
    running: int | None
    lock_holder: int | None     # PID of the process holding <home>/run/converter.lock
    slots_env: str | None


@dataclass(frozen=True)
class Ollama:
    base_url: str
    model: str | None
    checked: bool
    ok: bool = False
    models: tuple[str, ...] = ()
    error: str | None = None

    @property
    def model_present(self) -> bool | None:
        if not self.checked or not self.ok or not self.model:
            return None
        return any(m == self.model or m.split(":")[0] == self.model for m in self.models)


@dataclass(frozen=True)
class SettingsView:
    home: str
    library: str
    graphs: str
    library_db_bytes: int
    papers_held: int
    papers_cited_only: int
    converter: Converter
    ollama: Ollama
    env: tuple[EnvVar, ...]


def _converter(home: Path, pool: Any) -> Converter:
    from ...conversion import pool as pool_mod
    from ..jobs import converter_lock_holder

    gpus: tuple[str, ...] = ()
    running = None
    if pool is not None:
        gpus = tuple(str(getattr(c, "gpu", None) or "CPU") for c in getattr(pool, "converters", ()))
        count = getattr(pool, "running_count", None)
        running = count() if callable(count) else None
    return Converter(locked_version=pool_mod.locked_marker_version(), started=pool is not None,
                     slots=int(pool.size) if pool is not None else None, gpus=gpus, running=running,
                     lock_holder=converter_lock_holder(home),
                     slots_env=os.environ.get("SEEDGRAPH_MARKER_SLOTS") or None)


def _ollama(check: Any) -> Ollama:
    from ...llm import ollama

    base = ollama.resolve_base_url()
    model = ollama.default_model()
    if check is None:
        return Ollama(base_url=base, model=model, checked=False)
    return Ollama(base_url=check.base_url or base, model=model, checked=True, ok=bool(check.ok),
                  models=tuple(check.models or ()), error=check.error)


def page(lib: Library, home: Path, *, pool: Any = None, ollama_check: Any = None) -> SettingsView:
    """``pool``: the converter pool if one already exists (never resolved here)."""
    from ...providers import SCHOLARLY_ENV_VARS

    counts = dict(lib.conn.execute("SELECT status, COUNT(*) FROM paper_status GROUP BY status").fetchall())
    db = Path(lib.root) / "library.db"
    env = tuple(EnvVar(name, purpose, bool((os.environ.get(name) or "").strip()))
                for name, purpose in SCHOLARLY_ENV_VARS.items())
    return SettingsView(
        home=str(home), library=str(lib.root), graphs=str(lib.graphs_root),
        library_db_bytes=db.stat().st_size if db.is_file() else 0,
        papers_held=int(counts.get("held", 0)), papers_cited_only=int(counts.get("cited_only", 0)),
        converter=_converter(home, pool), ollama=_ollama(ollama_check), env=env)
