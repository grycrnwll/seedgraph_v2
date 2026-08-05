"""Per-provider transient retry (Build D chunk 1, decision D-1).

A single 429/5xx/transport blip must NOT count against a provider's circuit
breaker: each provider's HTTP helper (``_get_json`` / ``_fetch`` / ``_search``)
wraps its request in :func:`retry_transient`, so the chain's breaker only ever
sees *exhausted-transient* failures — it still opens after ``fail_max`` such
failures, preserving offline degradation.

Stdlib-only by design: tenacity was rejected (v2 hand-rolls the identical
pattern at ``llm/executor.py:_call_with_retry`` and ``base.py`` pins "no
anyio/tenacity"). ``httpx`` is imported ONLY inside function bodies (the
package must import without the optional network dep — base.py's rule). This
helper sits BELOW the breaker and ABOVE the httpx transport, so an injected
``client``/``MockTransport`` sees every attempt (the transport seam stays
injectable for cassette recording).

A plain 404 miss is NEVER retried — providers return ``None`` before raising
the :class:`TransientHTTPStatus` marker, so a miss stays a miss.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Awaitable, Callable, Optional

#: Statuses retried as transient upstream failures. 404 is a plain miss and is
#: never classified here; other 4xx are permanent and raise immediately.
TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})


class TransientHTTPStatus(Exception):
    """Marker for a transient HTTP status (429/5xx) from a provider request.

    Raised by the provider HTTP helpers so :func:`retry_transient` retries it;
    on exhaustion it propagates to ``ProviderChain._dispatch``, which records
    ONE breaker failure for the whole retried call.
    """

    def __init__(self, message: str, *, status_code: int, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def _retry_after_seconds(resp: Any) -> Optional[float]:
    """Parse a numeric ``Retry-After`` header (seconds), or ``None``.

    ponytail: the HTTP-date form is ignored (falls back to computed backoff) —
    none of the five providers emits it; parse dates only if one ever does.
    """
    raw = resp.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(str(raw).strip()))
    except ValueError:
        return None


def raise_for_transient(resp: Any, provider: str, detail: str = "") -> None:
    """Raise :class:`TransientHTTPStatus` if ``resp`` has a transient status.

    No-op otherwise — the caller keeps its own 404-miss and permanent->
    ``httpx.HTTPError`` handling unchanged.
    """
    if resp.status_code in TRANSIENT_STATUSES:
        raise TransientHTTPStatus(
            f"{provider} HTTP {resp.status_code} {detail}".rstrip(),
            status_code=resp.status_code,
            retry_after=_retry_after_seconds(resp),
        )


async def _sleep(seconds: float) -> None:  # pragma: no cover - patched in tests
    """Backoff sleep seam (mirrors ``llm/executor._sleep`` for test patching)."""
    await asyncio.sleep(seconds)


async def retry_transient(
    fn: Callable[[], Awaitable[Any]],
    *,
    attempts: int = 3,
    base: float = 0.2,
    cap: float = 2.0,
) -> Any:
    """Await ``fn()`` retrying transient failures; re-raise on exhaustion.

    Retries ``httpx.TransportError`` and :class:`TransientHTTPStatus` with
    exponential backoff + jitter (``min(cap, base * 2**attempt + jitter)``),
    honoring a numeric ``Retry-After`` when the marker carries one. Anything
    else (permanent HTTP errors, a returned ``None`` miss) passes straight
    through. An adversarially huge ``Retry-After`` is honored as-is — the
    chain's per-call ``asyncio.wait_for`` (60s) is the ceiling and turns it
    into a breaker failure.
    """
    import httpx

    for attempt in range(attempts):
        try:
            return await fn()
        except (httpx.TransportError, TransientHTTPStatus) as exc:
            if attempt == attempts - 1:
                raise
            retry_after = getattr(exc, "retry_after", None)
            if retry_after is not None:
                delay = retry_after
            else:
                delay = min(cap, base * (2 ** attempt) + random.uniform(0, base))
            await _sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover - loop returns or raises
