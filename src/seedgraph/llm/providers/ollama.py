"""Ollama provider adapter (local-first note extraction).

``POST {base_url}/api/chat`` with ``stream:false``. A ConnectError (daemon down)
or a 404 (model not pulled) surfaces as a :class:`ProviderError`
(``provider_unavailable`` / ``invalid_response``) so the executor's runtime
fallback chain re-routes to the hosted Anthropic profile (content gate re-applied)
instead of crashing. ``httpx`` is imported inside :meth:`complete`.
"""

from __future__ import annotations

from typing import Optional

from ..backend import LLMCompletion
from ..tokens import estimate_tokens
from ._http import ProviderError, post_json

DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class OllamaBackend:
    """Local Ollama chat backend (no key; honors a custom ``base_url``)."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        transport=None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
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
        use_model = model or self.model
        body = {
            "model": use_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            # Thinking models (e.g. qwen3) otherwise spend the whole ``num_predict``
            # budget on hidden reasoning and emit EMPTY content; ``think: false`` forces
            # a direct answer. Non-thinking models ignore the flag (must-fix #3a).
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        data = post_json(
            f"{self.base_url}/api/chat",
            headers={"content-type": "application/json"},
            json_body=body,
            timeout=self._timeout,
            transport=self._transport,
            provider="ollama",
        )
        message = data.get("message")
        if not isinstance(message, dict):
            raise ProviderError(
                "invalid_response",
                "ollama response missing message",
                retryable=False,
                provider="ollama",
                request_id=data.get("_request_id"),
            )
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        # Belt-and-suspenders: an older daemon that ignored ``think: false`` may still
        # route the answer into ``message.thinking`` and leave ``content`` empty; use
        # the thinking text as the completion in that case (must-fix #3b).
        if not text.strip():
            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                text = thinking
        if not text.strip():
            raise ProviderError(
                "invalid_response",
                "ollama response missing message.content",
                retryable=False,
                provider="ollama",
                request_id=data.get("_request_id"),
            )
        input_tokens = data.get("prompt_eval_count")
        output_tokens = data.get("eval_count")
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
    """Normalize ``done_reason`` to ``"length" | "stop" | None`` (fail-open).

    ``length`` means the output was cut at ``num_predict`` (the executor's terser
    retry keys off it); ``stop`` is a natural stop. Anything else (absent on older
    daemons, ``load``, unknown) is ``None`` — behaves like today."""
    reason = data.get("done_reason")
    if reason == "length":
        return "length"
    if reason == "stop":
        return "stop"
    return None
