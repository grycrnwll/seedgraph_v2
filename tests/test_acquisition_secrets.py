"""Chunk-5 acquisition-key unification (ADR-0001 §acquisition, decision D2).

``build_default_providers`` is the single acquisition resolution seam:
config attr -> keyring ``seedgraph/{service}/default`` -> env var. Provider
classes stay BYO-key and secret-ignorant. The suite-wide autouse fixture nulls
``_keyring_get``; tests here re-patch it with a dict-backed fake (their later
patch wins), never touching the real OS store.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from seedgraph.config.models import GlobalConfig
from seedgraph.llm import secrets
from seedgraph.providers import build_default_providers


def _fake_ring(monkeypatch, entries: dict[str, str]) -> None:
    monkeypatch.setattr(secrets, "_keyring_get", entries.get)


def _by_attr(providers, attr):
    hits = [p for p in providers if hasattr(p, attr)]
    assert hits, f"no provider exposing .{attr} in the chain"
    return hits[0]


# --------------------------------------------------------------------------
# Resolution precedence: config attr > keyring > env
# --------------------------------------------------------------------------
def test_config_attr_beats_keyring_and_env(monkeypatch):
    _fake_ring(monkeypatch, {"seedgraph/s2/default": "ring-s2", "seedgraph/core/default": "ring-core"})
    monkeypatch.setenv("S2_API_KEY", "env-s2")
    monkeypatch.setenv("CORE_API_KEY", "env-core")
    cfg = {"s2_api_key": "cfg-s2", "core_key": "cfg-core"}

    providers = build_default_providers(cfg)
    assert _by_attr(providers, "s2_api_key").s2_api_key == "cfg-s2"
    assert _by_attr(providers, "core_key").core_key == "cfg-core"


def test_keyring_beats_env(monkeypatch):
    _fake_ring(monkeypatch, {"seedgraph/s2/default": "ring-s2", "seedgraph/core/default": "ring-core"})
    monkeypatch.setenv("S2_API_KEY", "env-s2")
    monkeypatch.setenv("CORE_API_KEY", "env-core")

    providers = build_default_providers(GlobalConfig())
    assert _by_attr(providers, "s2_api_key").s2_api_key == "ring-s2"
    assert _by_attr(providers, "core_key").core_key == "ring-core"


def test_env_is_last_resort(monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "env-s2")
    monkeypatch.setenv("CORE_API_KEY", "env-core")

    providers = build_default_providers(GlobalConfig())
    assert _by_attr(providers, "s2_api_key").s2_api_key == "env-s2"
    assert _by_attr(providers, "core_key").core_key == "env-core"


def test_legacy_seedgraph_core_key_still_works(monkeypatch):
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    monkeypatch.setenv("SEEDGRAPH_CORE_KEY", "legacy-core")

    providers = build_default_providers(GlobalConfig())
    assert _by_attr(providers, "core_key").core_key == "legacy-core"


def test_no_core_key_means_no_core_provider(monkeypatch):
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    monkeypatch.delenv("SEEDGRAPH_CORE_KEY", raising=False)

    providers = build_default_providers(GlobalConfig())
    assert not [p for p in providers if hasattr(p, "core_key")]


# --------------------------------------------------------------------------
# OpenAlex premium key: chain -> provider -> request query param
# --------------------------------------------------------------------------
def test_openalex_key_resolves_from_keyring(monkeypatch):
    _fake_ring(monkeypatch, {"seedgraph/openalex/default": "ring-oa"})

    providers = build_default_providers(GlobalConfig())
    assert _by_attr(providers, "api_key").api_key == "ring-oa"


def test_openalex_key_env_fallback(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "env-oa")

    providers = build_default_providers(GlobalConfig())
    assert _by_attr(providers, "api_key").api_key == "env-oa"


def _openalex_with_capture(api_key):
    import httpx

    from seedgraph.providers.openalex import OpenAlexProvider

    captured: list = []

    def _handler(request):
        captured.append(request)
        return httpx.Response(200, content=json.dumps({"id": "https://openalex.org/W1", "title": "T"}))

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    provider = OpenAlexProvider(contact_email="me@example.org", api_key=api_key, client=client)
    return provider, captured


def test_openalex_premium_key_sent_as_query_param():
    provider, captured = _openalex_with_capture("pk-123")
    rec = asyncio.run(provider.by_doi("10.1234/abc"))
    assert rec and rec.get("openalex_id") == "W1"
    params = dict(captured[0].url.params)
    assert params["api_key"] == "pk-123"
    assert params["mailto"] == "me@example.org"  # polite pool untouched


def test_openalex_no_key_no_param():
    provider, captured = _openalex_with_capture(None)
    asyncio.run(provider.by_doi("10.1234/abc"))
    assert "api_key" not in dict(captured[0].url.params)
