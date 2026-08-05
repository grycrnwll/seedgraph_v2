"""Opt-in real-provider smoke tests (``-m llm_network``).

Deselected by default (see ``addopts`` in pyproject). Each test SELF-SKIPS when
its provider is unavailable, so ``python -m pytest -m llm_network`` runs cleanly
with no provider configured AND exercises the REAL wire format
(``default_backend`` -> adapter -> ``httpx``) when a local Ollama daemon is up
and/or ``ANTHROPIC_API_KEY`` is set. This is the live counterpart to the
offline ``MockTransport`` coverage in ``test_llm_providers.py``.

Run live::

    # local Ollama (start `ollama serve`, `ollama pull llama3`)
    python -m pytest -m llm_network -q -rs
    # hosted providers
    ANTHROPIC_API_KEY=sk-... python -m pytest -m llm_network -q -rs
    OPENAI_API_KEY=sk-...    python -m pytest -m llm_network -q -rs
    GEMINI_API_KEY=...       python -m pytest -m llm_network -q -rs

Override the models with ``SEEDGRAPH_SMOKE_OLLAMA_MODEL`` /
``SEEDGRAPH_SMOKE_ANTHROPIC_MODEL`` / ``SEEDGRAPH_SMOKE_OPENAI_MODEL`` /
``SEEDGRAPH_SMOKE_GEMINI_MODEL`` if the defaults are not pulled / not valid
for your account.
"""

from __future__ import annotations

import json
import os
import urllib.request

import pytest

from seedgraph.llm.backend import LLMCompletion, default_backend

pytestmark = pytest.mark.llm_network

_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Captured at import (collection) time: the autouse ``isolated_home`` fixture
# scrubs provider keys from the environment before each test body runs, so
# reading ``os.environ`` inside a smoke would KeyError on a live run.
_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
_OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
_GEMINI_KEY = os.environ.get("GEMINI_API_KEY")


def _ollama_model() -> str | None:
    """Return a pulled Ollama model name (env override, else the configured
    default if present, else the first available), or ``None`` if the daemon is
    unreachable or has no models."""
    try:
        with urllib.request.urlopen(_OLLAMA_HOST + "/api/tags", timeout=2) as resp:
            available = [m["name"] for m in (json.load(resp).get("models") or [])]
    except Exception:
        return None
    if not available:
        return None
    want = os.environ.get("SEEDGRAPH_SMOKE_OLLAMA_MODEL")
    if want:
        # accept a bare name matching "name" or "name:tag"
        for name in available:
            if name == want or name.split(":", 1)[0] == want:
                return name
        return None
    for default in ("llama3", "llama3.1"):
        for name in available:
            if name.split(":", 1)[0] == default:
                return name
    return available[0]


def test_ollama_real_completion():
    """A real Ollama /api/chat round-trip returns non-empty text + token counts."""
    model = _ollama_model()
    if model is None:
        pytest.skip(f"no reachable Ollama daemon/model at {_OLLAMA_HOST}")
    backend = default_backend(provider="ollama", model=model, base_url=_OLLAMA_HOST)
    out = backend.complete(
        "You are terse. Answer in one word.",
        "Reply with the single word: pong.",
        model=model,
        max_tokens=16,
    )
    assert isinstance(out, LLMCompletion)
    assert out.text.strip(), "Ollama returned empty text"
    # Ollama reports prompt_eval_count / eval_count; both should be non-negative.
    assert out.input_tokens >= 0 and out.output_tokens >= 0


@pytest.mark.skipif(not _ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
def test_anthropic_real_completion():
    """A real Anthropic /v1/messages round-trip returns text + usage tokens."""
    model = os.environ.get("SEEDGRAPH_SMOKE_ANTHROPIC_MODEL", "claude-sonnet-4-6")
    backend = default_backend(
        provider="anthropic",
        model=model,
        api_key=_ANTHROPIC_KEY,
    )
    out = backend.complete(
        "You are terse. Answer in one word.",
        "Reply with the single word: pong.",
        model=model,
        max_tokens=16,
    )
    assert isinstance(out, LLMCompletion)
    assert out.text.strip(), "Anthropic returned empty text"
    assert out.input_tokens > 0 and out.output_tokens > 0


@pytest.mark.skipif(not _OPENAI_KEY, reason="OPENAI_API_KEY not set")
def test_openai_real_completion():
    """A real OpenAI /v1/chat/completions round-trip returns text + usage tokens."""
    model = os.environ.get("SEEDGRAPH_SMOKE_OPENAI_MODEL", "gpt-5.4-mini")
    backend = default_backend(provider="openai", model=model, api_key=_OPENAI_KEY)
    out = backend.complete(
        "You are terse. Answer in one word.",
        "Reply with the single word: pong.",
        model=model,
        max_tokens=64,
    )
    assert isinstance(out, LLMCompletion)
    assert out.text.strip(), "OpenAI returned empty text"
    assert out.input_tokens > 0 and out.output_tokens > 0


@pytest.mark.skipif(not _GEMINI_KEY, reason="GEMINI_API_KEY not set")
def test_gemini_real_completion():
    """A real Gemini generateContent round-trip returns text + usage tokens.

    ``max_tokens`` is generous: flash-class models may spend output budget on
    hidden thinking (billed as output, ADR-0004) before emitting text."""
    model = os.environ.get("SEEDGRAPH_SMOKE_GEMINI_MODEL", "gemini-3.5-flash")
    backend = default_backend(provider="gemini", model=model, api_key=_GEMINI_KEY)
    out = backend.complete(
        "You are terse. Answer in one word.",
        "Reply with the single word: pong.",
        model=model,
        max_tokens=256,
    )
    assert isinstance(out, LLMCompletion)
    assert out.text.strip(), "Gemini returned empty text"
    assert out.input_tokens > 0 and out.output_tokens > 0
