"""LLM backend seam (phase_4 / Track 1) — offline-first, like ``FakeMarkerBackend``.

The thin :class:`LLMBackend` ``Protocol`` + a deterministic :class:`FakeLLMBackend`
for tests, plus a lazy provider registry (:func:`default_backend`) that resolves a
real adapter (Ollama / Anthropic / OpenAI / Gemini / mailbox) by provider name. Unknown / ``none`` providers
resolve to :class:`StubRealBackend`, which raises an actionable
"configure a profile/key" error if ever invoked. Provider adapters live in
``llm/providers/*`` and import ``httpx`` lazily inside their methods, so importing
this module touches no network. Tests INJECT :class:`FakeLLMBackend` (or a
MockTransport-backed adapter) so pytest never touches the network, a key, or a
real provider. ``SUPPORTED_PROVIDERS`` (the registry keys plus the no-backend
``none``/``local`` modes) is the registry-aware availability source for
``profiles.is_profile_available``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..errors import ConfigError
from .tokens import estimate_tokens


@dataclass
class LLMCompletion:
    """One backend completion: the raw text plus the token accounting.

    ``finish_reason`` is the provider's normalized stop signal — ``"length"``
    (output cut at max_tokens), ``"stop"`` (natural stop), or ``None`` when the
    provider sent no recognizable signal. Additive and fail-open: a ``None``
    behaves exactly like the pre-field world (Build C chunk 2, design D2)."""

    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str | None = None


@runtime_checkable
class LLMBackend(Protocol):
    """A pluggable text-completion backend (the seam phase_4 dispatches through)."""

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMCompletion:
        """Return an :class:`LLMCompletion` for the given prompts."""
        ...


@dataclass
class FakeLLMBackend:
    """Deterministic offline backend returning canned JSON + token counts (tests).

    ``response`` is the text returned for every call; ``responses`` (if given) is a
    queue consumed in order (e.g. a bad first response then a good repaired one),
    falling back to ``response`` / the last item once exhausted. ``response`` may be
    a ``dict`` (JSON-encoded automatically) or a ``str``. ``finish_reasons`` (if
    given) is a parallel queue of :attr:`LLMCompletion.finish_reason` values popped
    per call (``None`` once exhausted — the fail-open default). Every call is
    recorded in :attr:`calls` so a test can assert that NO source text was
    dispatched when a gate refuses (content/context/budget/no-llm).
    """

    response: object = None
    responses: list[object] | None = None
    finish_reasons: list[str | None] | None = None
    calls: list[dict] = field(default_factory=list)

    def _as_text(self, value: object) -> str:
        if value is None:
            return "{}"
        if isinstance(value, str):
            return value
        return json.dumps(value)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMCompletion:
        self.calls.append(
            {
                "system": system_prompt,
                "user": user_prompt,
                "model": model,
                "temperature": temperature,
            }
        )
        if self.responses:
            value = self.responses.pop(0)
        else:
            value = self.response
        finish_reason = self.finish_reasons.pop(0) if self.finish_reasons else None
        text = self._as_text(value)
        input_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
        output_tokens = estimate_tokens(text)
        return LLMCompletion(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )


class StubRealBackend:
    """Lazy real-backend placeholder — raises an actionable error if invoked.

    Resolved for an unknown / ``none`` provider (no shipped adapter). If a task is
    dispatched against such a profile with no usable backend, this raises a clear
    "configure a profile/key" message instead of silently fabricating output. The
    executor catches this (``config_error`` → ``skipped_no_llm``) so callers degrade,
    never crash. Never hit in pytest's success path (tests inject the fake)."""

    def __init__(self, *, provider: str | None = None, model: str | None = None) -> None:
        self.provider = provider
        self.model = model

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMCompletion:
        raise ConfigError(
            "no LLM backend client is configured for this task "
            f"(provider={self.provider!r}, model={model or self.model!r}). Configure an "
            "LLM access profile and key (or run a local Ollama model), or inject a "
            "backend, to dispatch real calls."
        )


# Provider registry (Track 1): provider name -> shipped adapter.
_PROVIDER_REGISTRY = {"ollama", "anthropic", "openai", "gemini", "mailbox"}

# Registry-aware availability source (consumed by profiles.is_profile_available):
# a provider is usable iff it has a shipped adapter here, OR it is a no-backend
# mode (none/local) handled by the local path.
SUPPORTED_PROVIDERS = _PROVIDER_REGISTRY | {"none", "local"}


def default_backend(
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    transport=None,
    timeout: float | None = None,
) -> LLMBackend:
    """Resolve a real backend for ``provider`` (lazy adapter import).

    ``ollama`` / ``anthropic`` / ``openai`` / ``gemini`` return their HTTP adapter (``base_url``
    / ``api_key`` / ``transport`` injected centrally by the executor — adapters
    never read env); ``mailbox`` returns the file-mailbox adapter (``base_url`` is
    the mailbox directory). Anything else (incl. ``none``) returns the
    :class:`StubRealBackend`. Adapter modules import ``httpx`` lazily, so this
    stays offline at import time."""
    name = (provider or "").lower()
    kwargs: dict = {"model": model}
    if transport is not None:
        kwargs["transport"] = transport
    if timeout is not None:
        kwargs["timeout"] = timeout
    if name == "ollama":
        from .providers.ollama import OllamaBackend

        return OllamaBackend(base_url=base_url, **kwargs)
    if name == "anthropic":
        from .providers.anthropic import AnthropicBackend

        return AnthropicBackend(api_key=api_key, base_url=base_url, **kwargs)
    if name == "openai":
        from .providers.openai import OpenAIBackend

        return OpenAIBackend(api_key=api_key, base_url=base_url, **kwargs)
    if name == "gemini":
        from .providers.gemini import GeminiBackend

        return GeminiBackend(api_key=api_key, base_url=base_url, **kwargs)
    if name == "mailbox":
        from .providers.mailbox import MailboxBackend

        return MailboxBackend(base_url=base_url, model=model)
    return StubRealBackend(provider=provider, model=model)
