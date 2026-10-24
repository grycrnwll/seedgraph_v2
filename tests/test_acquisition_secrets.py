"""Scholarly-API keys come from environment variables only (stage 2).

``build_default_providers`` is the single acquisition resolution seam: keys are read
from ``S2_API_KEY``, ``OPENALEX_API_KEY`` and ``CORE_API_KEY`` (legacy
``SEEDGRAPH_CORE_KEY``); the polite-pool contact email from ``SEEDGRAPH_CONTACT_EMAIL``,
else ``contact_email`` in config. Config key attributes and the OS keyring are no
longer consulted. Provider classes stay key-ignorant. All values are invented.
"""

from __future__ import annotations

import asyncio
import json

from seedgraph.config.models import GlobalConfig
from seedgraph.providers import SCHOLARLY_ENV_VARS, build_default_providers, env_key


def _by_attr(providers, attr):
    hits = [p for p in providers if hasattr(p, attr)]
    assert hits, f"no provider exposing .{attr} in the chain"
    return hits[0]


# --------------------------------------------------------------------------
# Environment only
# --------------------------------------------------------------------------
def test_env_keys_reach_the_providers(monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "env-s2")
    monkeypatch.setenv("CORE_API_KEY", "env-core")
    monkeypatch.setenv("OPENALEX_API_KEY", "env-oa")

    providers = build_default_providers(GlobalConfig())
    assert _by_attr(providers, "s2_api_key").s2_api_key == "env-s2"
    assert _by_attr(providers, "core_key").core_key == "env-core"
    assert _by_attr(providers, "api_key").api_key == "env-oa"


def test_config_key_attributes_are_ignored(monkeypatch):
    cfg = {"s2_api_key": "cfg-s2", "core_key": "cfg-core", "openalex_key": "cfg-oa"}

    providers = build_default_providers(cfg)
    assert _by_attr(providers, "s2_api_key").s2_api_key is None
    assert _by_attr(providers, "api_key").api_key is None
    assert not [p for p in providers if hasattr(p, "core_key")]


def test_env_beats_config_key_attributes(monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "env-s2")
    providers = build_default_providers({"s2_api_key": "cfg-s2"})
    assert _by_attr(providers, "s2_api_key").s2_api_key == "env-s2"


def test_blank_env_value_counts_as_unset(monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "   ")
    assert env_key("S2_API_KEY") is None
    assert _by_attr(build_default_providers(GlobalConfig()), "s2_api_key").s2_api_key is None


def test_legacy_seedgraph_core_key_still_works(monkeypatch):
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    monkeypatch.setenv("SEEDGRAPH_CORE_KEY", "legacy-core")

    providers = build_default_providers(GlobalConfig())
    assert _by_attr(providers, "core_key").core_key == "legacy-core"


def test_core_api_key_beats_the_legacy_name(monkeypatch):
    monkeypatch.setenv("CORE_API_KEY", "env-core")
    monkeypatch.setenv("SEEDGRAPH_CORE_KEY", "legacy-core")
    assert _by_attr(build_default_providers(GlobalConfig()), "core_key").core_key == "env-core"


def test_no_core_key_means_no_core_provider(monkeypatch):
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    monkeypatch.delenv("SEEDGRAPH_CORE_KEY", raising=False)

    providers = build_default_providers(GlobalConfig())
    assert not [p for p in providers if hasattr(p, "core_key")]


# --------------------------------------------------------------------------
# Contact email: env first, config fallback
# --------------------------------------------------------------------------
def test_contact_email_env_beats_config(monkeypatch):
    monkeypatch.setenv("SEEDGRAPH_CONTACT_EMAIL", "env@example.invalid")
    providers = build_default_providers({"contact_email": "cfg@example.invalid"})
    assert _by_attr(providers, "contact_email").contact_email == "env@example.invalid"


def test_contact_email_falls_back_to_config(monkeypatch):
    monkeypatch.delenv("SEEDGRAPH_CONTACT_EMAIL", raising=False)
    providers = build_default_providers({"contact_email": "cfg@example.invalid"})
    assert _by_attr(providers, "contact_email").contact_email == "cfg@example.invalid"


def test_env_var_names_are_documented():
    assert set(SCHOLARLY_ENV_VARS) == {"SEEDGRAPH_CONTACT_EMAIL", "S2_API_KEY", "OPENALEX_API_KEY",
                                       "CORE_API_KEY"}


# --------------------------------------------------------------------------
# OpenAlex premium key: provider -> request query param
# --------------------------------------------------------------------------
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
