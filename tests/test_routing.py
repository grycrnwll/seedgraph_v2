import pytest

from seedgraph.config.models import (
    ContentPolicy,
    LLMConfig,
    LLMProfile,
    ProjectConfig,
    TaskRoute,
)
from seedgraph.errors import ConfigError
from seedgraph.llm.profiles import is_profile_available
from seedgraph.llm.routing import NoLlmRoute, Route, resolve_route


def _cfg(profiles, routes, policy=None):
    return ProjectConfig(
        slug="t",
        llm=LLMConfig(profiles=profiles, routes=routes),
        content_policy=policy or ContentPolicy(),
    )


def test_content_gate_blocks_private_full_text(monkeypatch):
    monkeypatch.setenv("EXTKEY", "secret")
    cfg = _cfg(
        {"ext": LLMProfile(profile_id="ext", provider="anthropic", env_var="EXTKEY")},
        {"note": TaskRoute(task_type="note", preferred_profile="ext", requires_source_text=True)},
        ContentPolicy(external_llm_for_private_full_text=False),
    )
    with pytest.raises(ConfigError) as exc:
        resolve_route("note", "user_supplied_private", cfg)
    message = str(exc.value)
    assert "source text" in message
    assert "local LLM profile" in message


def test_content_gate_allows_when_policy_permits(monkeypatch):
    monkeypatch.setenv("EXTKEY", "secret")
    cfg = _cfg(
        {"ext": LLMProfile(profile_id="ext", provider="anthropic", env_var="EXTKEY")},
        {"note": TaskRoute(task_type="note", preferred_profile="ext", requires_source_text=True)},
        ContentPolicy(external_llm_for_private_full_text=True),
    )
    route = resolve_route("note", "user_supplied_private", cfg)
    assert isinstance(route, Route)
    assert route.external_full_text is True


def test_no_llm_route_carries_deterministic_fallback():
    cfg = _cfg(
        {"no_llm": LLMProfile(profile_id="no_llm", provider="none", access_mode="none")},
        {
            "ref": TaskRoute(task_type="ref", preferred_profile="no_llm", deterministic_fallback=True),
            "only": TaskRoute(task_type="only", preferred_profile="no_llm", deterministic_fallback=False),
        },
    )
    ref = resolve_route("ref", "open_access", cfg)
    only = resolve_route("only", "open_access", cfg)
    assert isinstance(ref, NoLlmRoute) and ref.deterministic_fallback is True
    assert isinstance(only, NoLlmRoute) and only.deterministic_fallback is False


def test_preferred_unavailable_falls_back(monkeypatch):
    monkeypatch.delenv("MISSINGKEY", raising=False)
    cfg = _cfg(
        {
            "ext": LLMProfile(profile_id="ext", provider="anthropic", env_var="MISSINGKEY"),
            "local": LLMProfile(profile_id="local", provider="ollama", access_mode="local", is_local=True),
        },
        {"t": TaskRoute(task_type="t", preferred_profile="ext", fallback_profile="local")},
    )
    route = resolve_route("t", "open_access", cfg)
    assert isinstance(route, Route)
    assert route.profile_id == "local"


def test_embeddings_external_fallback_open_access_only(monkeypatch):
    monkeypatch.setenv("EMBKEY", "x")
    # provider must be one with a shipped adapter (registry-aware availability,
    # review #1) for the profile to be "available"; the gate logic itself is
    # provider-agnostic. (OpenAI would be rejected as unavailable — see
    # test_registry_aware_availability_rejects_unshipped_provider below.)
    cfg = _cfg(
        {"ext_embed": LLMProfile(profile_id="ext_embed", provider="anthropic", env_var="EMBKEY")},
        {
            "embeddings": TaskRoute(
                task_type="embeddings",
                preferred_profile="ext_embed",
                requires_source_text=True,
                external_fallback_access_class="open_access",
            )
        },
    )
    # open_access external embedding fallback is permitted
    assert isinstance(resolve_route("embeddings", "open_access", cfg), Route)
    # any other access class is blocked (symmetric mirror of the private gate)
    with pytest.raises(ConfigError):
        resolve_route("embeddings", "metadata_only", cfg)


def test_registry_aware_availability_rejects_unshipped_provider(monkeypatch):
    """A profile whose provider has no shipped adapter is NEVER 'available' even
    with a key set, so resolve_route falls through to the fallback and never
    silently dispatches to it. (OpenAI shipped in ADR-0003 — mistral stands in
    as the unshipped provider.)"""
    monkeypatch.setenv("MISTRALKEY", "secret")
    mistral = LLMProfile(profile_id="mst", provider="mistral", env_var="MISTRALKEY")
    assert is_profile_available(mistral) is False  # key present, still unavailable

    cfg = _cfg(
        {
            "mst": mistral,
            "local": LLMProfile(
                profile_id="local", provider="ollama", access_mode="local", is_local=True
            ),
        },
        {"t": TaskRoute(task_type="t", preferred_profile="mst", fallback_profile="local")},
    )
    route = resolve_route("t", "open_access", cfg)
    assert isinstance(route, Route)
    assert route.profile_id == "local"  # fell through past the unavailable profile


def test_default_routes_are_local_first():
    """Stage A default-route flips: note_extraction + answer_generation off OpenAI."""
    from seedgraph.config.models import default_routes

    routes = default_routes()
    assert routes["note_extraction"].preferred_profile == "local_ollama_default"
    assert routes["note_extraction"].fallback_profile == "anthropic_api_default"
    assert routes["answer_generation"].preferred_profile == "anthropic_api_default"
    assert routes["answer_generation"].fallback_profile == "no_llm"
    assert routes["semantic_graph_extraction"].preferred_profile == "local_ollama_default"
    assert routes["semantic_graph_extraction"].deterministic_fallback is True
