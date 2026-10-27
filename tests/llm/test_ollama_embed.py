"""``llm/ollama.embed`` (stage 4 design §5.1): one batched ``/api/embed`` call, typed
errors. An in-process httpx transport, no real network. Invented text only."""

from __future__ import annotations

import json

import httpx
import pytest

from seedgraph.llm import ollama

from .fake_ollama import closed_port_url


def _transport(handler):
    seen: list[dict] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append({"path": request.url.path, "body": json.loads(request.content or b"null")})
        return handler(request)

    return httpx.MockTransport(wrapped), seen


def test_embed_sends_one_batch_and_returns_the_vectors_in_order():
    transport, seen = _transport(lambda r: httpx.Response(200, json={
        "model": "invented-embed", "embeddings": [[1.0, 0.0], [0.0, 1.0]]}))
    client = ollama.OllamaClient("http://invented.host:1", transport=transport)
    vectors = client.embed("invented-embed", ["search_document: alpha", "search_document: beta"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert seen == [{"path": "/api/embed", "body": {
        "model": "invented-embed", "input": ["search_document: alpha", "search_document: beta"],
        "truncate": True}}]


def test_embed_of_nothing_makes_no_call():
    transport, seen = _transport(lambda r: httpx.Response(500))
    assert ollama.OllamaClient("http://invented.host:1", transport=transport).embed("m", []) == []
    assert seen == []


def test_a_missing_embedding_model_is_model_missing():
    transport, _ = _transport(lambda r: httpx.Response(404, json={"error": "model 'x' not found"}))
    with pytest.raises(ollama.OllamaError) as err:
        ollama.OllamaClient("http://invented.host:1", transport=transport).embed("x", ["a"])
    assert err.value.code == "model_missing"


def test_a_reply_with_the_wrong_count_is_invalid():
    transport, _ = _transport(lambda r: httpx.Response(200, json={"embeddings": [[1.0]]}))
    with pytest.raises(ollama.OllamaError) as err:
        ollama.OllamaClient("http://invented.host:1", transport=transport).embed("m", ["a", "b"])
    assert err.value.code == "invalid_response"


def test_module_level_embed_reports_an_unreachable_daemon():
    with pytest.raises(ollama.OllamaError) as err:
        ollama.embed("m", ["a"], base_url=closed_port_url(), timeout=5)
    assert err.value.code == "unreachable"
