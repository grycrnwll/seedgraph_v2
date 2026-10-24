"""Opt-in smoke against a real local Ollama (``pytest -m llm_network``; deselected by
default). Model from ``SEEDGRAPH_OLLAMA_MODEL``, else the first listed model."""

from __future__ import annotations

import json

import pytest

from seedgraph.llm import ollama

pytestmark = pytest.mark.llm_network


def test_a_real_daemon_answers_json():
    reach = ollama.check_reachable()
    if not reach.ok or not reach.models:
        pytest.skip(f"no local Ollama with a model: {reach.error or 'no models'}")
    model = ollama.default_model() or reach.models[0]
    text = ollama.complete_json(model, "Answer with a JSON object.",
                                'Return {"title": "An Invented Title"} exactly.',
                                {"type": "object", "properties": {"title": {"type": "string"}},
                                 "required": ["title"]})
    assert isinstance(json.loads(text), dict)
