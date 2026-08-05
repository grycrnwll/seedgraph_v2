"""Shared HTTP helper for provider adapters (Track 1).

``httpx`` is imported inside :func:`post_json` so importing this module is
network-dep-free. :class:`ProviderError` is the single typed error every adapter
raises on any transport / HTTP failure; the executor maps its ``code`` onto the
status taxonomy. :func:`classify_http_error` is the status-code → (code,
retryable) mapping (plan §Providers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderError(Exception):
    """A typed provider failure (never a bare exception leaks to a caller).

    ``code`` is one of the executor ``LLMError`` codes
    (``auth_error|rate_limited|timeout|provider_unavailable|invalid_response``).
    ``retryable`` gates the executor's transport retry; ``retry_after`` (seconds)
    honors a 429 ``Retry-After``. ``request_id`` is the provider request id when
    the response carried one (audit only — never a body).
    """

    code: str
    message: str
    retryable: bool = False
    provider: Optional[str] = None
    status_code: Optional[int] = None
    retry_after: Optional[float] = None
    request_id: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.code}: {self.message}"


def classify_http_error(status: int, retry_after: Optional[str] = None) -> tuple[str, bool]:
    """Map an HTTP status to ``(LLMError.code, retryable)`` (plan §Providers).

    401/403 → auth_error (no retry); 429 → rate_limited (retry, honor
    Retry-After); 408 → timeout (retry); 5xx → provider_unavailable (retry);
    400/422 → invalid_response (no retry — a malformed request/response). Any
    other 4xx is treated as a non-retryable invalid_response.
    """
    if status in (401, 403):
        return ("auth_error", False)
    if status == 429:
        return ("rate_limited", True)
    if status == 408:
        return ("timeout", True)
    if 500 <= status < 600:
        return ("provider_unavailable", True)
    if status in (400, 422):
        return ("invalid_response", False)
    return ("invalid_response", False)


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def post_json(
    url: str,
    *,
    headers: dict,
    json_body: dict,
    timeout: float = 120.0,
    transport=None,
    provider: Optional[str] = None,
    classify=None,
) -> dict:
    """POST ``json_body`` to ``url`` and return the decoded JSON dict.

    ``transport`` (an ``httpx.MockTransport`` in tests) is injected so the whole
    chain stays offline by default. Any transport error, non-2xx status, or
    non-JSON body raises a :class:`ProviderError` — the method body never lets a
    raw ``httpx`` / unexpected exception escape, so the executor always degrades
    instead of crashing (the three no-crash regressions).

    ``classify`` (optional) is a ``(status_code, body_text) -> (code, retryable)``
    override for providers whose error semantics don't fit the shared status map
    (e.g. Gemini signals an invalid key via HTTP 400, ADR-0004). Default:
    :func:`classify_http_error`.
    """
    import httpx

    client_kwargs: dict = {"timeout": timeout}
    if transport is not None:
        client_kwargs["transport"] = transport
    try:
        with httpx.Client(**client_kwargs) as client:
            response = client.post(url, headers=headers, json=json_body)
    except httpx.TimeoutException as exc:  # pragma: no cover - exercised via MockTransport
        raise ProviderError("timeout", str(exc), retryable=True, provider=provider) from exc
    except httpx.HTTPError as exc:  # ConnectError, ReadError, etc.
        raise ProviderError(
            "provider_unavailable", str(exc), retryable=True, provider=provider
        ) from exc
    except Exception as exc:  # noqa: BLE001 - a raising MockTransport must still degrade
        raise ProviderError(
            "provider_unavailable", str(exc), retryable=True, provider=provider
        ) from exc

    request_id = response.headers.get("request-id") or response.headers.get("x-request-id")
    if response.status_code >= 400:
        retry_after_hdr = response.headers.get("retry-after")
        if classify is not None:
            code, retryable = classify(response.status_code, response.text)
        else:
            code, retryable = classify_http_error(response.status_code, retry_after_hdr)
        raise ProviderError(
            code,
            f"HTTP {response.status_code} from {provider or url}",
            retryable=retryable,
            provider=provider,
            status_code=response.status_code,
            retry_after=_parse_retry_after(retry_after_hdr),
            request_id=request_id,
        )
    try:
        data = response.json()
    except Exception as exc:  # noqa: BLE001 - non-JSON body is an invalid response
        raise ProviderError(
            "invalid_response",
            "provider returned a non-JSON body",
            retryable=False,
            provider=provider,
            status_code=response.status_code,
            request_id=request_id,
        ) from exc
    if not isinstance(data, dict):
        raise ProviderError(
            "invalid_response",
            "provider returned a non-object JSON body",
            retryable=False,
            provider=provider,
            status_code=response.status_code,
            request_id=request_id,
        )
    data["_request_id"] = request_id
    return data
