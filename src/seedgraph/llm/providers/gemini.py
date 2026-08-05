"""Gemini generateContent adapter (ADR-0004).

``POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent``
with ``x-goog-api-key`` header auth — never the ``?key=`` query param, which
would leak the key into logs and the request-id audit trail. Text is the
concatenation of ``candidates[0].content.parts[*].text``; tokens from
``usageMetadata`` (``thoughtsTokenCount`` is summed into output — thinking
tokens bill as output and the budget gate consumes these numbers). Gemini
quirks handled here per ADR-0004: a 200 can carry no candidates (safety block)
and an invalid key surfaces as HTTP 400 ``API_KEY_INVALID`` rather than 401.
``httpx`` is imported inside :func:`post_json`; the api key is injected at
construction (the adapter never reads the environment).
"""

from __future__ import annotations

from typing import Optional

from ..backend import LLMCompletion
from ..tokens import estimate_tokens
from ._http import ProviderError, classify_http_error, post_json

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
# Cheap-tier default per ADR-0005 (pricing verified in llm_capabilities.yaml);
# the model id is supplied per-profile — this adapter is model-agnostic.
DEFAULT_MODEL = "gemini-3.5-flash"


def _classify_gemini(status: int, body_text: str) -> tuple[str, bool]:
    """Gemini signals an invalid key via HTTP 400 with ``API_KEY_INVALID`` in the
    error details — promote that to auth_error so the executor doesn't misfile
    it as a malformed request; everything else uses the shared status map."""
    if status == 400 and "API_KEY_INVALID" in (body_text or ""):
        return ("auth_error", False)
    return classify_http_error(status)


class GeminiBackend:
    """Gemini generateContent backend (x-goog-api-key auth; hosted-only)."""

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
        self._root = base_url.rstrip("/") if base_url else API_ROOT
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
                "no Gemini API key configured for this profile",
                retryable=False,
                provider="gemini",
            )
        use_model = model or self.model or DEFAULT_MODEL
        headers = {
            "x-goog-api-key": self._api_key,
            "content-type": "application/json",
        }
        body = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        data = post_json(
            f"{self._root}/models/{use_model}:generateContent",
            headers=headers,
            json_body=body,
            timeout=self._timeout,
            transport=self._transport,
            provider="gemini",
            classify=_classify_gemini,
        )
        text = _candidate_text(data)
        if text is None:
            # A 200 with no usable text is a safety block or an empty candidate
            # (promptFeedback.blockReason / finishReason=SAFETY) — degrade, don't
            # retry: the same prompt will block again.
            raise ProviderError(
                "invalid_response",
                _block_message(data),
                retryable=False,
                provider="gemini",
                request_id=data.get("_request_id"),
            )
        usage = data.get("usageMetadata") if isinstance(data.get("usageMetadata"), dict) else {}
        input_tokens = usage.get("promptTokenCount")
        output_tokens = usage.get("candidatesTokenCount")
        thoughts = usage.get("thoughtsTokenCount")
        if isinstance(output_tokens, int) and isinstance(thoughts, int):
            # Thinking tokens bill as output tokens — the budget gate consumes
            # these numbers, so the sum is load-bearing for cost accounting.
            output_tokens += thoughts
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
    """Normalize ``candidates[0].finishReason`` to ``"length" | "stop" | None``.

    ``MAX_TOKENS`` means the output was cut at the budget (the executor's terser
    retry keys off ``"length"``); ``STOP`` is a natural stop. Anything else
    (absent, ``SAFETY``, ``RECITATION``, unknown) is ``None`` — fail-open."""
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        return None
    reason = candidates[0].get("finishReason")
    if reason == "MAX_TOKENS":
        return "length"
    if reason == "STOP":
        return "stop"
    return None


def _candidate_text(data: dict) -> Optional[str]:
    """Concatenate ``candidates[0].content.parts[*].text``; None when absent/empty."""
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return None
    content = first.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return None
    texts = [
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    joined = "".join(texts)
    if not joined.strip():
        return None
    return joined


def _block_message(data: dict) -> str:
    """An actionable message naming the block/finish reason when present."""
    feedback = data.get("promptFeedback")
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        return f"gemini blocked the prompt (blockReason={feedback['blockReason']})"
    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        reason = candidates[0].get("finishReason")
        if reason:
            return f"gemini candidate has no text parts (finishReason={reason})"
    return "gemini response has no candidates with text parts"
