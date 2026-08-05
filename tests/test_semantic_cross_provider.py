"""Cross-provider semantic-graph acceptance test (ADR-0003/0004, chunk 7).

For each shipped provider adapter (anthropic, ollama, openai, gemini), the REAL
adapter is driven through the full proposer path — ``_ExternalConceptProposer``
→ ``run_llm`` (route resolution + content gate + dispatch) → ``adapter.complete``
— with an ``httpx.MockTransport`` serving that provider's NATIVE response
envelope wrapping the strict ``{"synonym_groups": ...}`` JSON. This proves the
semantic graph's synonym proposal parses end-to-end against every provider's
wire shape, fully offline.

Hosted providers exercise the real opt-in a project makes (ADR-0006): the
profile's env key present + ``content_policy.external_llm_for_private_full_text``
on (semantic labels are full-text-derived → restricted class by default).
"""

from __future__ import annotations

import json

import httpx
import pytest

from seedgraph.config.models import GlobalConfig
from seedgraph.llm.providers.anthropic import AnthropicBackend
from seedgraph.llm.providers.gemini import GeminiBackend
from seedgraph.llm.providers.ollama import OllamaBackend
from seedgraph.llm.providers.openai import OpenAIBackend
from seedgraph.semantic.llm_propose import make_proposer

_GROUPS_JSON = json.dumps(
    {"synonym_groups": [["gmm", "generalized method of moments"]]}
)


def _anthropic_envelope() -> dict:
    return {
        "content": [{"type": "text", "text": _GROUPS_JSON}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _ollama_envelope() -> dict:
    return {
        "message": {"role": "assistant", "content": _GROUPS_JSON},
        "prompt_eval_count": 10,
        "eval_count": 5,
    }


def _openai_envelope() -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": _GROUPS_JSON}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _gemini_envelope() -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": _GROUPS_JSON}]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
    }


# (provider, default profile id, adapter class, ctor kwargs, env var to satisfy
#  availability or None for local, native envelope factory, endpoint suffix)
_CASES = [
    (
        "anthropic",
        "anthropic_api_default",
        AnthropicBackend,
        {"api_key": "test-key"},
        "ANTHROPIC_API_KEY",
        _anthropic_envelope,
        "/v1/messages",
    ),
    (
        "ollama",
        "local_ollama_default",
        OllamaBackend,
        {},
        None,
        _ollama_envelope,
        "/api/chat",
    ),
    (
        "openai",
        "openai_api_default",
        OpenAIBackend,
        {"api_key": "test-key"},
        "OPENAI_API_KEY",
        _openai_envelope,
        "/chat/completions",
    ),
    (
        "gemini",
        "gemini_api_default",
        GeminiBackend,
        {"api_key": "test-key"},
        "GEMINI_API_KEY",
        _gemini_envelope,
        ":generateContent",
    ),
]


@pytest.mark.parametrize(
    "provider, profile_id, backend_cls, ctor_kwargs, env_var, envelope, suffix",
    _CASES,
    ids=[case[0] for case in _CASES],
)
def test_semantic_proposal_parses_every_provider_envelope(
    monkeypatch, provider, profile_id, backend_cls, ctor_kwargs, env_var, envelope, suffix
):
    if env_var is not None:
        monkeypatch.setenv(env_var, "test-key")

    cfg = GlobalConfig()
    # The per-project external opt-in a real user makes for hosted providers
    # (labels are restricted-class; local ollama never consults the gate).
    cfg.content_policy.external_llm_for_private_full_text = True
    profile = cfg.llm.profiles[profile_id]

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=envelope())

    backend = backend_cls(
        model=profile.model, transport=httpx.MockTransport(handler), **ctor_kwargs
    )
    proposer = make_proposer(profile, config=cfg, backend=backend)

    groups = proposer.propose(["gmm", "generalized method of moments", "ols"])

    assert groups == [["gmm", "generalized method of moments"]], (
        f"{provider}: expected the synonym group to survive the full "
        f"run_llm -> adapter -> parse path (got {groups!r})"
    )
    assert seen["url"].split("?")[0].endswith(suffix) or suffix in seen["url"]


def test_semantic_gate_reroutes_external_to_local_without_opt_in(monkeypatch):
    """Without the per-project opt-in, requesting a hosted profile for the
    semantic task re-routes to the local profile (labels never leave the
    machine) — pins ADR-0006's local-first posture. The pinned test backend is
    dispatched under the LOCAL route's identity, which is what the usage/audit
    trail would record."""
    from seedgraph.llm.executor import run_llm
    from seedgraph.semantic.llm_propose import _PROPOSE_SYSTEM, _parse_groups

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    cfg = GlobalConfig()  # external_llm_for_private_full_text defaults False

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_openai_envelope())

    backend = OpenAIBackend(
        model="gpt-5.4-mini",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    result = run_llm(
        "semantic_graph_extraction",
        _PROPOSE_SYSTEM,
        "- gmm\n- generalized method of moments",
        access_class="user_supplied_private",
        config=cfg,
        profile_id_override="openai_api_default",
        backend=backend,
        parse=_parse_groups,
        log=False,
    )
    assert result.profile_id == "local_ollama_default"
    assert result.external_full_text is False
