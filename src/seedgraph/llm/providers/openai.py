"""OpenAI Chat Completions adapter (ADR-0003).

``POST {base_url or https://api.openai.com/v1}/chat/completions`` with
``Authorization: Bearer`` auth and the system prompt as a ``system`` role
message. Chat Completions (not the Responses API) is deliberate: it is the wire
format every OpenAI-compatible local server (LM Studio / vLLM / llama.cpp)
speaks, so a ``provider: openai`` profile with a custom ``base_url`` covers
local backends with no extra adapter. Text is read from
``choices[0].message.content``; tokens from ``usage.prompt_tokens`` /
``usage.completion_tokens``. ``httpx`` is imported inside :func:`post_json`;
the api key is injected at construction (the adapter never reads the
environment).
"""

from __future__ import annotations

from typing import Optional

from ..backend import LLMCompletion
from ..tokens import estimate_tokens
from ._http import ProviderError, post_json

API_URL = "https://api.openai.com/v1/chat/completions"
# Cheap-tier default per ADR-0005 (pricing verified in llm_capabilities.yaml);
# the model id is supplied per-profile — this adapter is model-agnostic.
DEFAULT_MODEL = "gpt-5.4-mini"


class OpenAIBackend:
    """OpenAI Chat Completions backend (Bearer auth; hosted/local body split)."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        transport=None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self._api_key = api_key
        # A custom base_url (e.g. http://127.0.0.1:1234/v1) is an OpenAI-compatible
        # local server: keyless is allowed and the legacy body fields apply.
        self._hosted = base_url is None
        self._url = (base_url.rstrip("/") + "/chat/completions") if base_url else API_URL
        self._transport = transport
        self._timeout = timeout

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMCompletion:
        if self._hosted and not self._api_key:
            # Resolved centrally; an empty key here means the route was selected
            # without an available secret — surface as auth_error (no network).
            raise ProviderError(
                "auth_error",
                "no OpenAI API key configured for this profile",
                retryable=False,
                provider="openai",
            )
        use_model = model or self.model or DEFAULT_MODEL
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        body: dict = {
            "model": use_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._hosted:
            # Hosted current-gen (reasoning-family) models reject ``max_tokens``
            # and any non-default ``temperature`` with a 400 — send only
            # ``max_completion_tokens`` and omit temperature entirely.
            body["max_completion_tokens"] = max_tokens
        else:
            # Local OpenAI-compatible servers predate ``max_completion_tokens``
            # and honor temperature (determinism matters for JSON extraction).
            body["max_tokens"] = max_tokens
            body["temperature"] = temperature
        data = post_json(
            self._url,
            headers=headers,
            json_body=body,
            timeout=self._timeout,
            transport=self._transport,
            provider="openai",
        )
        text = _first_text(data)
        if text is None:
            raise ProviderError(
                "invalid_response",
                "openai response missing choices[0].message.content",
                retryable=False,
                provider="openai",
                request_id=data.get("_request_id"),
            )
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        if not isinstance(input_tokens, int):
            input_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
        if not isinstance(output_tokens, int):
            output_tokens = estimate_tokens(text)
        return LLMCompletion(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=_finish_reason(data),
        )


def _finish_reason(data: dict) -> Optional[str]:
    """Normalize ``choices[0].finish_reason`` to ``"length" | "stop" | None``.

    ``length`` means the output was cut at the token budget (the executor's terser
    retry keys off it); ``stop`` is a natural stop. Anything else (absent,
    ``content_filter``, ``tool_calls``, unknown) is ``None`` — fail-open."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    reason = choices[0].get("finish_reason")
    if reason == "length":
        return "length"
    if reason == "stop":
        return "stop"
    return None


def _first_text(data: dict) -> Optional[str]:
    """Return ``choices[0].message.content`` when it is a non-empty string."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    return content
