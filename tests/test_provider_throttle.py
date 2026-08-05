"""Polite request throttle on the Semantic Scholar provider.

The provider spaces outbound HTTP calls (the ``_get_json`` chokepoint) at least
``min_request_interval`` seconds apart so a back-to-back re-check pass issues at
~1 req/s instead of spraying the API. Uses a tiny interval + an instant
MockTransport so nothing sleeps for real seconds.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from seedgraph.providers.semantic_scholar import SemanticScholarProvider


def _instant_client() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_get_json_throttle_spaces_requests():
    interval = 0.05
    n = 4

    async def _drive(min_interval: float) -> float:
        client = _instant_client()
        provider = SemanticScholarProvider(client=client, min_request_interval=min_interval)
        try:
            start = time.monotonic()
            for _ in range(n):
                await provider._get_json("/paper/x", params={"fields": "title"})
            return time.monotonic() - start
        finally:
            await client.aclose()

    # Throttled: n sequential calls are spaced >= interval apart, so the total
    # elapsed is at least (n-1) intervals (the first call goes immediately).
    elapsed = asyncio.run(_drive(interval))
    assert elapsed >= (n - 1) * interval, f"expected >= {(n - 1) * interval}s, got {elapsed}s"

    # Disabled: min_request_interval=0 skips the gate entirely -> near-zero.
    elapsed_off = asyncio.run(_drive(0))
    assert elapsed_off < interval, f"throttle should be disabled, got {elapsed_off}s"
