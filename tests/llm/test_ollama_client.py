"""The Ollama HTTP client (design §6): JSON-format chat, base URL resolution,
typed errors, reachability for doctor. A fake daemon on loopback, no real network."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from seedgraph.llm import ollama

from .fake_ollama import chat_reply, closed_port_url


def test_complete_json_sends_a_json_chat_and_returns_the_text(fake_ollama):
    fake_ollama.script(chat_reply('{"title": "An Invented Title"}', prompt_tokens=50, eval_tokens=9))
    client = ollama.OllamaClient(fake_ollama.url)
    reply = client.complete_json("invented-model", "You extract metadata.", "Cover page text.")
    assert json.loads(reply.text) == {"title": "An Invented Title"}
    assert reply.usage == {"input_tokens": 50, "output_tokens": 9}
    (body,) = fake_ollama.chats
    assert body["model"] == "invented-model"
    assert body["stream"] is False and body["format"] == "json"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"] == "Cover page text."


def test_a_schema_is_sent_as_the_format(fake_ollama):
    schema = {"type": "object", "properties": {"title": {"type": ["string", "null"]}}, "required": ["title"]}
    fake_ollama.script(chat_reply('{"title": null}'))
    ollama.OllamaClient(fake_ollama.url).complete_json("invented-model", "s", "u", schema)
    assert fake_ollama.chats[0]["format"] == schema


def test_module_level_complete_json_returns_the_text(fake_ollama):
    fake_ollama.script(chat_reply('{"year": 2019}'))
    assert ollama.complete_json("invented-model", "s", "u", base_url=fake_ollama.url) == '{"year": 2019}'


def test_a_closed_port_is_unreachable():
    with pytest.raises(ollama.OllamaError) as err:
        ollama.OllamaClient(closed_port_url(), timeout=5).complete_json("invented-model", "s", "u")
    assert err.value.code == "unreachable"


def test_a_slow_daemon_times_out(fake_ollama):
    fake_ollama.script(1.0)
    with pytest.raises(ollama.OllamaError) as err:
        ollama.OllamaClient(fake_ollama.url, timeout=0.2).complete_json("invented-model", "s", "u")
    assert err.value.code == "timeout"


def test_a_missing_model_is_model_missing(fake_ollama):
    fake_ollama.script((404, {"error": "model 'nope' not found"}))
    with pytest.raises(ollama.OllamaError) as err:
        ollama.OllamaClient(fake_ollama.url).complete_json("nope", "s", "u")
    assert err.value.code == "model_missing" and "not found" in str(err.value)


def test_a_server_error_and_a_non_json_body_are_typed(fake_ollama):
    fake_ollama.script((500, {"error": "invented crash"}), (200, "not json"))
    client = ollama.OllamaClient(fake_ollama.url)
    with pytest.raises(ollama.OllamaError) as first:
        client.complete_json("invented-model", "s", "u")
    with pytest.raises(ollama.OllamaError) as second:
        client.complete_json("invented-model", "s", "u")
    assert (first.value.code, second.value.code) == ("http_error", "invalid_response")


def test_empty_content_falls_back_to_thinking_else_stays_empty(fake_ollama):
    fake_ollama.script(
        (200, {"model": "m", "message": {"content": "", "thinking": '{"title": "T"}'}, "done": True}),
        (200, {"model": "m", "message": {"content": "  "}, "done": True}),
    )
    client = ollama.OllamaClient(fake_ollama.url)
    assert client.complete_json("m", "s", "u").text == '{"title": "T"}'
    empty = client.complete_json("m", "s", "u")
    assert empty.text.strip() == "" and empty.usage == {}


def test_check_lists_models_or_reports_unreachable(fake_ollama):
    ok = ollama.check_reachable(fake_ollama.url)
    assert ok.ok and ok.models == ["invented-model:latest"]
    down = ollama.check_reachable(closed_port_url())
    assert not down.ok and down.models == [] and "unreachable" in down.error


@pytest.mark.parametrize(("env", "expected"), [
    ({}, "http://localhost:11434"),
    ({"SEEDGRAPH_OLLAMA_URL": "http://gadget-box:9999/"}, "http://gadget-box:9999"),
    ({"OLLAMA_HOST": "0.0.0.0"}, "http://127.0.0.1:11434"),
    ({"OLLAMA_HOST": "gadget-box:1234"}, "http://gadget-box:1234"),
    ({"OLLAMA_HOST": "gadget-box:1234", "SEEDGRAPH_OLLAMA_URL": "http://first:1"}, "http://first:1"),
])
def test_base_url_resolution(monkeypatch, env, expected):
    for name in ("SEEDGRAPH_OLLAMA_URL", "OLLAMA_HOST"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert ollama.resolve_base_url() == expected
    assert ollama.resolve_base_url("http://explicit:2/") == "http://explicit:2"


def _imports(path: str) -> set[str]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            found.add(("." * node.level) + (node.module or ""))
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
    return found


def test_the_client_does_not_import_the_hosted_layer():
    """P8 deletes the hosted layer; this module must stand alone (a source-level check,
    since ``seedgraph/llm/__init__.py`` still imports that layer until P8)."""
    assert not {m for m in _imports(ollama.__file__) if m.startswith(".") or "seedgraph" in m}
