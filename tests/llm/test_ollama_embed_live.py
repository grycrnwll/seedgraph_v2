"""Opt-in smoke against a real local Ollama embedding model (``pytest -m llm_network``;
deselected by default). Model from ``SEEDGRAPH_EMBED_MODEL``, else ``nomic-embed-text``."""

from __future__ import annotations

import os

import pytest

from seedgraph.llm import ollama

pytestmark = pytest.mark.llm_network


def test_a_real_daemon_embeds_a_batch():
    model = os.environ.get("SEEDGRAPH_EMBED_MODEL", "nomic-embed-text")
    reach = ollama.check_reachable()
    if not reach.ok or not any(m.split(":")[0] == model.split(":")[0] for m in reach.models):
        pytest.skip(f"no local Ollama with {model}: {reach.error or 'model not pulled'}")
    vecs = ollama.embed(model, ["search_document: an invented claim", "search_query: invented"])
    assert len(vecs) == 2 and len(vecs[0]) == len(vecs[1]) > 0
