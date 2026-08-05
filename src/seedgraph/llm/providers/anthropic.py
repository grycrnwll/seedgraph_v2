"""Anthropic Messages API adapter (hosted fallback for note extraction).

``POST https://api.anthropic.com/v1/messages`` with ``x-api-key`` +
``anthropic-version`` headers, a top-level ``system`` field and the required
``max_tokens`` (shape per the claude-api skill). Text is read from
``content[0].text``; tokens from ``usage.input_tokens`` / ``usage.output_tokens``.
``httpx`` is imported inside :func:`post_json`; the api key is injected at
construction (the adapter never reads the environment).
"""

from __future__ import annotations

from typing import Optional

from ..backend import LLMCompletion
from ..tokens import estimate_tokens
from ._http import ProviderError, post_json

API_URL = "https://api.anthropic.com/v1/messages"
# Stable Messages API version header (claude-api skill). The model id is supplied
# per-profile (config default: claude-sonnet-4-6); this adapter is model-agnostic.
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicBackend:
    """Anthropic Messages backend (x-api-key auth; top-level system + max_tokens)."""

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
        self._url = (base_url.rstrip("/") + "/v1/messages") if base_url else API_URL
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
        if not self._api_key:
            # Resolved centrally; an empty key here means the route was selected
            # without an available secret — surface as auth_error (no network).
            raise ProviderError(
                "auth_error",
                "no Anthropic API key configured for this profile",
                retryable=False,
                provider="anthropic",
            )
        use_model = model or self.model or DEFAULT_MODEL
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        body = {
            "model": use_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        data = post_json(
            self._url,
            headers=headers,
            json_body=body,
            timeout=self._timeout,
            transport=self._transport,
            provider="anthropic",
        )
        text = _first_text(data)
        if text is None:
            raise ProviderError(
                "invalid_response",
                "anthropic response missing content[].text",
                retryable=False,
                provider="anthropic",
                request_id=data.get("_request_id"),
            )
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
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
    """Normalize ``stop_reason`` to ``"length" | "stop" | None`` (fail-open).

    ``max_tokens`` means the output was cut at the budget (the executor's terser
    retry keys off ``"length"``); ``end_turn``/``stop_sequence`` are natural stops.
    Anything else (absent, unknown) is ``None`` — behaves like today."""
    reason = data.get("stop_reason")
    if reason == "max_tokens":
        return "length"
    if reason in ("end_turn", "stop_sequence"):
        return "stop"
    return None


def _first_text(data: dict) -> Optional[str]:
    """Concatenate the text of every ``type=="text"`` content block."""
    content = data.get("content")
    if not isinstance(content, list):
        return None
    parts = [
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    ]
    if not parts:
        return None
    return "".join(parts)
