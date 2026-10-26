"""`seedgraph work ollama` verifies Ollama's titles by DOI: it passes the provider chain
(stage 3 U5 gap). The offline fake provider (``SEEDGRAPH_FAKE_PROVIDERS``) keeps this
test off the network. Invented text only."""

from __future__ import annotations

import json

import pytest
from work.routine_support import make_paper

from seedgraph.cli import app
from seedgraph.library import store as libstore


@pytest.fixture
def home(isolated_home):
    lib = libstore.open_library_at(isolated_home / "library")
    try:
        make_paper(lib, 1)
    finally:
        lib.close()
    return isolated_home


def test_ollama_passes_the_default_provider_chain_and_closes_its_cache(runner, home, monkeypatch):
    from seedgraph.providers.fake import FakeProvider
    from seedgraph.work import ollama_worker

    monkeypatch.setenv("SEEDGRAPH_FAKE_PROVIDERS", "1")
    seen = {}

    def fake_run(lib, batch_id, *, model, max_packages=None, chain=None, **kw):
        seen["chain"] = chain
        seen["providers"] = list(chain.providers) if chain is not None else []
        return ollama_worker.OllamaRunReport(batch_id=batch_id, worker_id="ollama-test", model=model)

    monkeypatch.setattr(ollama_worker, "run_ollama_worker", fake_run)
    created = runner.invoke(app, ["work", "create", "metadata", "--paper", "lp_test0001", "--root", str(home)])
    batch = json.loads(created.stdout)["batch_id"]
    result = runner.invoke(app, ["work", "ollama", batch, "--model", "invented:1b", "--root", str(home)])

    assert result.exit_code == 0, result.output
    chain = seen["chain"]
    assert chain is not None
    assert seen["providers"] and all(isinstance(p, FakeProvider) for p in seen["providers"])
    with pytest.raises(Exception):  # the chain's cache connection was closed afterwards
        chain.cache_engine.execute("SELECT 1")
