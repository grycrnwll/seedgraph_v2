"""Stage 5 acceptance fixtures (design §10; slice R8).

Opt-in against a real home:

    SEEDGRAPH_STAGE5_HOME=C:/path/to/home  uv run pytest tests/acceptance/stage5

The automated probes (planted citation, path without answer, identity) run on a **copy** of
that home's library, under pytest's tmp folder; the real home is only read (``mode=ro``).
Every probe also runs on an invented library, always, so the harness itself is tested.
Assertion messages carry counts and ids, never labels or paper text.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from records._world import Agent, standard

from seedgraph.library import store as libstore

HOME_ENV = "SEEDGRAPH_STAGE5_HOME"


@pytest.fixture
def invented_home(tmp_path) -> Path:
    """A home whose library holds two invented noted papers (textbook terms)."""
    home = tmp_path / "invented_home"
    lib = libstore.open_library_at(home / "library")
    try:
        standard(lib)
    finally:
        lib.close()
    return home


@pytest.fixture
def real_home() -> Path:
    value = os.environ.get(HOME_ENV)
    if not value:
        pytest.skip(f"set {HOME_ENV} to a v3 home to run")
    home = Path(value)
    if not libstore.library_db_path(libstore.library_dir(home)).is_file():
        pytest.fail(f"no library under {home}")
    return home


@pytest.fixture
def work_home(tmp_path, monkeypatch) -> Path:
    """Where a probe's copy of a library goes; ``SEEDGRAPH_HOME`` points there too."""
    home = tmp_path / "probe_copy"
    home.mkdir()
    monkeypatch.setenv("SEEDGRAPH_HOME", str(home))
    return home


__all__ = ["Agent"]
