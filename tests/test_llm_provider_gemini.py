"""Gemini adapter tests (ADR-0004) — fully offline via ``httpx.MockTransport``.

Covers the generateContent request shape (header auth, system_instruction,
generationConfig), thinking-token accounting, the two Gemini quirks (safety
blocks on a 200; invalid key via HTTP 400 ``API_KEY_INVALID``), and the
registry/availability wiring that makes a ``gemini`` profile selectable. Also
pins that ``post_json``'s default classification is unchanged for the other
adapters (the ``classify=`` hook is opt-in).
"""

from __future__ import annotations

import json

import httpx
import pytest

from seedgraph.config.models import LLMProfile
from seedgraph.llm.backend import SUPPORTED_PROVIDERS, default_backend
from seedgraph.llm.profiles import is_profile_available
from seedgraph.llm.providers import ProviderError
from seedgraph.llm.providers._http import post_json
from seedgraph.llm.providers.gemini import GeminiBackend


def _transport(handler):
    return httpx.MockTransport(handler)


def _ok_response(parts=None, usage=None, finish_reason="STOP"):
    candidate: dict = {"finishReason": finish_reason}
    if parts is not None:
        candidate["content"] = {"role": "model", "parts": parts}
    body: dict = {"candidates": [candidate]}
    if usage is not None:
        body["usageMetadata"] = usage
    return httpx.Response(200, json=body)


# --- request/response shape ---------------------------------------------------

def test_gemini_success_parses_multipart_text_tokens_and_headers():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content.decode())
        return _ok_response(
            parts=[{"text": "first "}, {"text": "second"}],
            usage={"promptTokenCount": 30, "candidatesTokenCount": 7},
        )

    backend = GeminiBackend(
        model="gemini-3.5-flash", api_key="g-test", transport=_transport(handler)
    )
    out = backend.complete("be helpful", "the question", max_tokens=512, temperature=0.2)
    assert out.text == "first second"
    assert (out.input_tokens, out.output_tokens) == (30, 7)
    assert seen["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-3.5-flash:generateContent"
    )
    # Header auth only — the key must never ride the URL (?key= leaks into logs).
    assert seen["headers"]["x-goog-api-key"] == "g-test"
    assert "key=" not in seen["url"]
    assert seen["body"]["system_instruction"] == {"parts": [{"text": "be helpful"}]}
    assert seen["body"]["contents"] == [
        {"role": "user", "parts": [{"text": "the question"}]}
    ]
    assert seen["body"]["generationConfig"] == {
        "temperature": 0.2,
        "maxOutputTokens": 512,
    }


def test_gemini_thoughts_tokens_sum_into_output():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response(
            parts=[{"text": "answer"}],
            usage={
                "promptTokenCount": 10,
                "candidatesTokenCount": 4,
                "thoughtsTokenCount": 6,
            },
        )

    backend = GeminiBackend(model="m", api_key="k", transport=_transport(handler))
    out = backend.complete("s", "u")
    # Thinking tokens bill as output — the budget gate depends on the sum.
    assert out.output_tokens == 10
    assert out.input_tokens == 10


def test_gemini_usage_absent_falls_back_to_estimates():
    backend = GeminiBackend(
        model="m", api_key="k", transport=_transport(lambda r: _ok_response(parts=[{"text": "answer"}]))
    )
    out = backend.complete("sys", "user")
    assert out.text == "answer"
    assert out.input_tokens > 0
    assert out.output_tokens > 0


def test_gemini_no_key_is_auth_error_without_network():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        calls["n"] += 1
        return _ok_response(parts=[{"text": "x"}])

    backend = GeminiBackend(model="m", api_key=None, transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "auth_error"
    assert calls["n"] == 0  # never hit the wire


# --- safety-block quirk (200 with no usable text) ------------------------------

def test_gemini_prompt_block_no_candidates_is_invalid_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}
        )

    backend = GeminiBackend(model="m", api_key="k", transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "invalid_response"
    assert ei.value.retryable is False
    assert "PROHIBITED_CONTENT" in ei.value.message


def test_gemini_safety_finish_reason_without_parts_is_invalid_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_response(parts=None, finish_reason="SAFETY")

    backend = GeminiBackend(model="m", api_key="k", transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "invalid_response"
    assert ei.value.retryable is False
    assert "SAFETY" in ei.value.message


# --- error classification (classify= hook + shared defaults) -------------------

def test_gemini_400_api_key_invalid_is_auth_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "status": "INVALID_ARGUMENT",
                    "details": [{"reason": "API_KEY_INVALID"}],
                }
            },
        )

    backend = GeminiBackend(model="m", api_key="bad", transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "auth_error"
    assert ei.value.retryable is False
    assert ei.value.status_code == 400


def test_gemini_plain_400_stays_invalid_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"status": "INVALID_ARGUMENT"}})

    backend = GeminiBackend(model="m", api_key="k", transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "invalid_response"


def test_gemini_429_is_rate_limited_and_5xx_unavailable():
    def handler_429(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "2"}, json={"error": {}})

    backend = GeminiBackend(model="m", api_key="k", transport=_transport(handler_429))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "rate_limited"
    assert ei.value.retryable is True
    assert ei.value.retry_after == 2.0

    def handler_503(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {}})

    backend = GeminiBackend(model="m", api_key="k", transport=_transport(handler_503))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "provider_unavailable"
    assert ei.value.retryable is True


def test_post_json_default_classification_unchanged_without_classify():
    # Regression pin for the other adapters: no classify= means the shared
    # status map still decides (a 401 is auth_error, a 400 invalid_response).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {}})

    with pytest.raises(ProviderError) as ei:
        post_json(
            "https://example.invalid/x",
            headers={},
            json_body={},
            transport=_transport(handler),
            provider="test",
        )
    assert ei.value.code == "auth_error"


# --- registry / availability --------------------------------------------------

def test_default_backend_gemini_returns_adapter():
    backend = default_backend(provider="gemini", model="m", api_key="k")
    assert isinstance(backend, GeminiBackend)
    assert "gemini" in SUPPORTED_PROVIDERS


def _gemini_profile(**overrides) -> LLMProfile:
    fields = {
        "profile_id": "gemini_test",
        "provider": "gemini",
        "access_mode": "api_key",
        "model": "gemini-3.5-flash",
        "env_var": "GEMINI_API_KEY",
    }
    fields.update(overrides)
    return LLMProfile(**fields)


def test_gemini_profile_available_iff_key_resolves(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-live")
    assert is_profile_available(_gemini_profile()) is True
    monkeypatch.delenv("GEMINI_API_KEY")
    assert is_profile_available(_gemini_profile()) is False
