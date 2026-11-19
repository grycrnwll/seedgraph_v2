"""An invented library for the R6 spot-check screen tests (stage 5 design §5, §6.5-6.6): six
noted papers made with the R4 World in ``tests/spotcheck/conftest.py``; textbook terms only.
Paper ``did`` (model ``invented-model-a``, in graph ``did-reading``) and ``ate`` (model
``invented-model-b``) are the two the tests name; four more (two per model) let a sample reach
a dozen claims, since a sample holds at most two claims of a paper. Each paper has five agent
claims: a research question, a method, two assumptions (one with no quote) and a result."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from spotcheck.conftest import World

from seedgraph.library.store import open_library_at
from seedgraph.spotcheck import sample as sample_mod

T0 = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)
MODEL_A, MODEL_B = "invented-model-a", "invented-model-b"
GRAPH = "did-reading"
HEADERS = {"hx-request": "true"}


@dataclass
class SpotFixture:
    home: Path
    did: str
    ate: str
    claims: dict[str, dict[str, str]]    # paper id -> field_key -> claim id

    def claim(self, paper: str, field: str) -> str:
        return self.claims[paper][field]


def clock(at: datetime = T0):
    return lambda: at


def build(home: Path) -> SpotFixture:
    lib = open_library_at(home / "library")
    try:
        w = World(lib)
        did, ate = w.paper(title="An invented DiD study"), w.paper(title="An invented matching study")
        a, b = w.note(did, model=MODEL_A), w.note(ate, model=MODEL_B)
        w.graph(GRAPH, [did])
        claims = {did: w.claims(a.note_id), ate: w.claims(b.note_id)}
        for i, model in enumerate((MODEL_A, MODEL_B, MODEL_A, MODEL_B)):   # a sample holds at most 2 claims a paper
            pid = w.paper(title=f"An invented extra study {i + 1}")
            claims[pid] = w.claims(w.note(pid, model=model).note_id)
        return SpotFixture(home, did, ate, claims)
    finally:
        lib.close()


def with_lib(home: Path, fn):
    lib = open_library_at(home / "library")
    try:
        return fn(lib)
    finally:
        lib.close()


def draw(home: Path, *, n: int = 10, seed: int = 1, **filters):
    """A stored sample (the deep function, as a script would call it)."""
    return with_lib(home, lambda lib: sample_mod.create_sample(lib, n=n, seed=seed, filters=filters))
