"""Provider adapter tests (Track 1, Build Stage B) — Ollama + Anthropic mocked.

Fully offline: every HTTP call is served by an ``httpx.MockTransport`` injected
into the adapter, so no key, no daemon, and no network are required. A raising
transport must surface as a typed :class:`ProviderError` (never a bare exception)
so the executor can degrade. Real-network smokes live behind ``-m llm_network``.
"""

from __future__ import annotations

import httpx
import pytest

from seedgraph.llm.providers import ProviderError, classify_http_error
from seedgraph.llm.providers.anthropic import AnthropicBackend
from seedgraph.llm.providers.gemini import GeminiBackend
from seedgraph.llm.providers.ollama import OllamaBackend
from seedgraph.llm.providers.openai import OpenAIBackend


def _transport(handler):
    return httpx.MockTransport(handler)


# --- classify_http_error ----------------------------------------------------

def test_classify_http_error_mapping():
    assert classify_http_error(401) == ("auth_error", False)
    assert classify_http_error(403) == ("auth_error", False)
    assert classify_http_error(429) == ("rate_limited", True)
    assert classify_http_error(408) == ("timeout", True)
    assert classify_http_error(500) == ("provider_unavailable", True)
    assert classify_http_error(503) == ("provider_unavailable", True)
    assert classify_http_error(400) == ("invalid_response", False)
    assert classify_http_error(422) == ("invalid_response", False)


# --- Ollama -----------------------------------------------------------------

def test_ollama_chat_success_parses_text_and_tokens():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "hello from ollama"},
                "prompt_eval_count": 11,
                "eval_count": 7,
            },
        )

    backend = OllamaBackend(model="llama3", transport=_transport(handler))
    out = backend.complete("sys", "user", model="llama3")
    assert out.text == "hello from ollama"
    assert (out.input_tokens, out.output_tokens) == (11, 7)
    assert seen["url"].endswith("/api/chat")


def test_ollama_connect_error_is_provider_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    backend = OllamaBackend(model="llama3", transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("sys", "user")
    assert ei.value.code == "provider_unavailable"
    assert ei.value.retryable is True


def test_ollama_missing_model_404_is_invalid_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'ghost' not found"})

    backend = OllamaBackend(model="ghost", transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("sys", "user")
    # 404 is a generic client error -> invalid_response (non-retryable, triggers fallback).
    assert ei.value.code == "invalid_response"


def test_ollama_honors_custom_base_url():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"message": {"content": "x"}})

    backend = OllamaBackend(model="m", base_url="http://host:9999", transport=_transport(handler))
    backend.complete("s", "u")
    assert seen["url"] == "http://host:9999/api/chat"


def test_ollama_sends_think_false_in_body():
    # must-fix #3a: the shared adapter must disable hidden reasoning so a thinking
    # model (qwen3) returns a direct answer instead of spending the budget thinking.
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen["body"] = _json.loads(request.content.decode())
        return httpx.Response(200, json={"message": {"content": "ok"}})

    backend = OllamaBackend(model="qwen3", transport=_transport(handler))
    backend.complete("sys", "user")
    assert seen["body"]["think"] is False


def test_ollama_empty_content_falls_back_to_thinking():
    # must-fix #3b: an older daemon that ignored think:false may leave content empty
    # and route the answer into message.thinking — use that as the completion.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "   ", "thinking": "the answer is 42"},
                "prompt_eval_count": 3,
                "eval_count": 4,
            },
        )

    backend = OllamaBackend(model="qwen3", transport=_transport(handler))
    out = backend.complete("sys", "user")
    assert out.text == "the answer is 42"
    assert (out.input_tokens, out.output_tokens) == (3, 4)


def test_ollama_empty_content_and_no_thinking_is_invalid_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant", "content": ""}})

    backend = OllamaBackend(model="qwen3", transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("sys", "user")
    assert ei.value.code == "invalid_response"


# --- Anthropic --------------------------------------------------------------

def test_anthropic_messages_success_and_headers():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        seen["url"] = str(request.url)
        import json as _json

        seen["body"] = _json.loads(request.content.decode())
        return httpx.Response(
            200,
            headers={"request-id": "req_abc"},
            json={
                "content": [{"type": "text", "text": "answer text"}],
                "usage": {"input_tokens": 20, "output_tokens": 5},
            },
        )

    backend = AnthropicBackend(model="claude-sonnet-4-6", api_key="sk-test", transport=_transport(handler))
    out = backend.complete("be helpful", "the question", model="claude-sonnet-4-6", max_tokens=512)
    assert out.text == "answer text"
    assert (out.input_tokens, out.output_tokens) == (20, 5)
    assert seen["url"].endswith("/v1/messages")
    assert seen["headers"]["x-api-key"] == "sk-test"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    # top-level system + required max_tokens (skill-confirmed shape).
    assert seen["body"]["system"] == "be helpful"
    assert seen["body"]["max_tokens"] == 512
    assert seen["body"]["messages"] == [{"role": "user", "content": "the question"}]


def test_anthropic_no_key_is_auth_error_without_network():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        calls["n"] += 1
        return httpx.Response(200, json={})

    backend = AnthropicBackend(model="m", api_key=None, transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "auth_error"
    assert calls["n"] == 0  # never hit the wire


def test_anthropic_401_is_auth_error_not_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"type": "authentication_error"}})

    backend = AnthropicBackend(model="m", api_key="bad", transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "auth_error"
    assert ei.value.retryable is False
    assert ei.value.status_code == 401


def test_anthropic_429_is_rate_limited_with_retry_after():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "2"}, json={"error": {}})

    backend = AnthropicBackend(model="m", api_key="k", transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "rate_limited"
    assert ei.value.retryable is True
    assert ei.value.retry_after == 2.0


def test_anthropic_500_is_provider_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {}})

    backend = AnthropicBackend(model="m", api_key="k", transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "provider_unavailable"


def test_raising_transport_degrades_to_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("boom in the transport")

    backend = AnthropicBackend(model="m", api_key="k", transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    # A non-httpx exception from the transport is still wrapped (no traceback escapes).
    assert ei.value.code == "provider_unavailable"


# --- finish_reason normalization (Build C chunk 2, design D2) -----------------
#
# Each adapter maps its provider's native stop signal onto the normalized
# "length" | "stop" | None vocabulary; an absent/unknown signal fails open (None).

def _static(json_body: dict) -> httpx.MockTransport:
    return _transport(lambda request: httpx.Response(200, json=json_body))


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [("max_tokens", "length"), ("end_turn", "stop"), ("stop_sequence", "stop"),
     ("tool_use", None), (None, None)],
)
def test_anthropic_finish_reason_mapping(stop_reason, expected):
    body: dict = {"content": [{"type": "text", "text": "answer"}]}
    if stop_reason is not None:
        body["stop_reason"] = stop_reason
    backend = AnthropicBackend(model="m", api_key="k", transport=_static(body))
    assert backend.complete("s", "u").finish_reason == expected


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [("length", "length"), ("stop", "stop"), ("content_filter", None), (None, None)],
)
def test_openai_finish_reason_mapping(finish_reason, expected):
    choice: dict = {"message": {"role": "assistant", "content": "answer"}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    backend = OpenAIBackend(model="m", api_key="k", transport=_static({"choices": [choice]}))
    assert backend.complete("s", "u").finish_reason == expected


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [("MAX_TOKENS", "length"), ("STOP", "stop"), ("RECITATION", None), (None, None)],
)
def test_gemini_finish_reason_mapping(finish_reason, expected):
    candidate: dict = {"content": {"parts": [{"text": "answer"}]}}
    if finish_reason is not None:
        candidate["finishReason"] = finish_reason
    backend = GeminiBackend(model="m", api_key="k", transport=_static({"candidates": [candidate]}))
    assert backend.complete("s", "u").finish_reason == expected


@pytest.mark.parametrize(
    ("done_reason", "expected"),
    [("length", "length"), ("stop", "stop"), ("load", None), (None, None)],
)
def test_ollama_finish_reason_mapping(done_reason, expected):
    body: dict = {"message": {"role": "assistant", "content": "answer"}}
    if done_reason is not None:
        body["done_reason"] = done_reason
    backend = OllamaBackend(model="m", transport=_static(body))
    assert backend.complete("s", "u").finish_reason == expected
