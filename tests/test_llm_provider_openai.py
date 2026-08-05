"""OpenAI adapter tests (ADR-0003) — fully offline via ``httpx.MockTransport``.

Covers the hosted/local body split (``max_completion_tokens`` + no temperature
hosted; ``max_tokens`` + temperature + keyless against a custom ``base_url``),
error classification through the shared helper, and the registry/availability
wiring that makes an ``openai`` profile selectable.
"""

from __future__ import annotations

import json

import httpx
import pytest

from seedgraph.config.models import LLMProfile
from seedgraph.llm.backend import SUPPORTED_PROVIDERS, default_backend
from seedgraph.llm.profiles import is_profile_available
from seedgraph.llm.providers import ProviderError
from seedgraph.llm.providers.openai import OpenAIBackend


def _transport(handler):
    return httpx.MockTransport(handler)


def _ok_response(text="answer text", usage=None):
    body = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


# --- request/response shape ---------------------------------------------------

def test_openai_success_parses_text_tokens_headers_and_hosted_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content.decode())
        return _ok_response(usage={"prompt_tokens": 20, "completion_tokens": 5})

    backend = OpenAIBackend(model="gpt-5.4-mini", api_key="sk-test", transport=_transport(handler))
    out = backend.complete("be helpful", "the question", max_tokens=512, temperature=0.7)
    assert out.text == "answer text"
    assert (out.input_tokens, out.output_tokens) == (20, 5)
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["headers"]["authorization"] == "Bearer sk-test"
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "the question"},
    ]
    # Hosted branch: max_completion_tokens only; temperature omitted (current-gen
    # hosted models reject max_tokens / non-default temperature).
    assert seen["body"]["max_completion_tokens"] == 512
    assert "max_tokens" not in seen["body"]
    assert "temperature" not in seen["body"]


def test_openai_custom_base_url_is_keyless_local_with_legacy_body():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content.decode())
        return _ok_response("local answer")

    backend = OpenAIBackend(
        model="qwen2.5-32b",
        base_url="http://127.0.0.1:1234/v1",
        transport=_transport(handler),
    )
    out = backend.complete("sys", "user", max_tokens=256, temperature=0.0)
    assert out.text == "local answer"
    assert seen["url"] == "http://127.0.0.1:1234/v1/chat/completions"
    assert "authorization" not in seen["headers"]
    # Local branch: legacy fields the OpenAI-compatible servers understand.
    assert seen["body"]["max_tokens"] == 256
    assert seen["body"]["temperature"] == 0.0
    assert "max_completion_tokens" not in seen["body"]


def test_openai_no_key_default_url_is_auth_error_without_network():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        calls["n"] += 1
        return _ok_response()

    backend = OpenAIBackend(model="m", api_key=None, transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "auth_error"
    assert calls["n"] == 0  # never hit the wire


def test_openai_usage_absent_falls_back_to_estimates():
    backend = OpenAIBackend(model="m", api_key="k", transport=_transport(lambda r: _ok_response()))
    out = backend.complete("sys", "user")
    assert out.text == "answer text"
    assert out.input_tokens > 0
    assert out.output_tokens > 0


def test_openai_missing_choices_is_invalid_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "chat.completion"})

    backend = OpenAIBackend(model="m", api_key="k", transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "invalid_response"


def test_openai_empty_content_is_invalid_response():
    backend = OpenAIBackend(model="m", api_key="k", transport=_transport(lambda r: _ok_response("   ")))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "invalid_response"


# --- error classification (shared classify_http_error) -----------------------

def test_openai_401_is_auth_error_not_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"code": "invalid_api_key"}})

    backend = OpenAIBackend(model="m", api_key="bad", transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "auth_error"
    assert ei.value.retryable is False
    assert ei.value.status_code == 401


def test_openai_429_is_rate_limited_with_retry_after():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "3"}, json={"error": {}})

    backend = OpenAIBackend(model="m", api_key="k", transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "rate_limited"
    assert ei.value.retryable is True
    assert ei.value.retry_after == 3.0


def test_openai_500_is_provider_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {}})

    backend = OpenAIBackend(model="m", api_key="k", transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "provider_unavailable"


def test_openai_connect_error_is_provider_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    backend = OpenAIBackend(model="m", api_key="k", transport=_transport(handler))
    with pytest.raises(ProviderError) as ei:
        backend.complete("s", "u")
    assert ei.value.code == "provider_unavailable"
    assert ei.value.retryable is True


# --- registry / availability --------------------------------------------------

def test_default_backend_openai_returns_adapter():
    backend = default_backend(provider="openai", model="m", api_key="k")
    assert isinstance(backend, OpenAIBackend)
    assert "openai" in SUPPORTED_PROVIDERS


def _openai_profile(**overrides) -> LLMProfile:
    fields = {
        "profile_id": "openai_test",
        "provider": "openai",
        "access_mode": "api_key",
        "model": "gpt-5.4-mini",
        "env_var": "OPENAI_API_KEY",
    }
    fields.update(overrides)
    return LLMProfile(**fields)


def test_openai_profile_available_iff_key_resolves(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live")
    assert is_profile_available(_openai_profile()) is True
    monkeypatch.delenv("OPENAI_API_KEY")
    assert is_profile_available(_openai_profile()) is False


def test_openai_local_profile_available_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    profile = _openai_profile(
        access_mode="local", is_local=True, base_url="http://127.0.0.1:1234/v1", env_var=None
    )
    assert is_profile_available(profile) is True
