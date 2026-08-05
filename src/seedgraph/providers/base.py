"""Provider chain core (ported from v1 ``providers/base.py``) — phase_5b §6.3.

The thin async surface the walk/fetch consume:

* :class:`CitationProvider` — the structural Protocol every concrete provider
  satisfies.
* :class:`CircuitBreaker` — per-provider breaker; an open provider is skipped so
  the chain falls through (offline resilience, decision 73).
* :class:`ProviderChain` — tries providers in order, rides the ``cache.db
  provider_cache`` (this phase is the SOLE writer; decisions 39/64/73), activates
  the polite pool from ``contact_email``, threads a ``run_id``, and persists via
  :mod:`seedgraph.cache.provider_cache` under the PINNED §6.5 request-key grammar.

No-LLM, offline-resilient: a fresh cache hit needs no network. ``httpx`` is
referenced ONLY inside concrete providers' method bodies (kept out of module
import so the package imports without the optional network dep installed). The
chain itself uses stdlib ``asyncio`` for the per-call timeout (no anyio/tenacity).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

from ..cache.provider_cache import (
    cache_get,
    cache_put,
    key_by_doi,
    key_by_openalex_ids,
    key_by_title,
    key_oa_candidates,
    key_referenced_works,
)

if TYPE_CHECKING:
    import sqlite3

    from ..acquisition.fetch import OACandidate
    from ..config.models import GlobalConfig

#: Wall-clock cap (seconds) on a SINGLE provider call (mirrors v1
#: DEFAULT_REQUEST_TIMEOUT): httpx's timeout is per-read, not total, so a hung host
#: would otherwise wedge the chain. asyncio.wait_for turns it into a breaker failure.
DEFAULT_REQUEST_TIMEOUT: float = 60.0


@runtime_checkable
class CitationProvider(Protocol):
    """Structural contract for a metadata provider (ported v1).

    A concrete provider implements the subset it supports; the chain probes each
    in order and skips missing-method / breaker-open providers.
    """

    name: str

    async def by_doi(self, doi: str) -> dict | None: ...
    async def by_title(self, title: str, year: int | None = None) -> dict | None: ...
    async def referenced_works(self, record: dict) -> list[dict]: ...
    async def oa_pdf_candidates(self, record: dict) -> list["OACandidate"]: ...


class CircuitBreaker:
    """Per-provider circuit breaker (ported v1).

    Opens after ``fail_max`` consecutive failures and half-opens after
    ``reset_timeout`` seconds; :meth:`is_open` lets the chain skip a tripped
    provider and fall through to the next (decision 73).
    """

    def __init__(self, *, fail_max: int = 3, reset_timeout: float = 60.0) -> None:
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self._failures = 0
        self._opened_at: float | None = None

    def record_success(self) -> None:
        """Reset the consecutive-failure counter and close the breaker."""
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        """Increment the failure counter; open the breaker at ``fail_max``."""
        self._failures += 1
        if self._failures >= self.fail_max:
            self._opened_at = time.monotonic()

    def is_open(self) -> bool:
        """True while the breaker is open (within ``reset_timeout`` of tripping)."""
        if self._opened_at is None:
            return False
        if (time.monotonic() - self._opened_at) >= self.reset_timeout:
            return False  # cool-down elapsed -> allow a half-open probe
        return True


class ProviderChain:
    """Ordered provider fallback over ``cache.db provider_cache`` (§6.3, ported v1).

    Implements decisions 39/64/73: a fresh ``provider_cache`` hit short-circuits
    the network (offline-resilient); ``contact_email`` activates the polite pool;
    ``s2_api_key`` / CORE keys are optional BYO; ``run_id`` is threaded for manifest
    provenance. Cache keys come from the §6.5 builders in
    :mod:`seedgraph.cache.provider_cache`; this class owns NO schema.

    ``cache_engine`` is the cache HOME root (``Path``/``str``/``None`` — opened via
    phase_1's ``open_cache_db``) OR a live ``sqlite3.Connection`` (used as-is). A
    bare-engine root is migrated once (``init_cache_db``) so ``provider_cache``
    exists before the first read.
    """

    def __init__(
        self,
        providers,
        *,
        cache_engine,
        run_id: str | None = None,
        contact_email: str | None = None,
        s2_api_key: str | None = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        use_cache: bool = True,
        fail_max: int = 3,
        reset_timeout: float = 60.0,
    ) -> None:
        self.providers = list(providers)
        self.cache_engine = cache_engine  # cache.db HOME root OR sqlite3.Connection
        self.run_id = run_id
        self.contact_email = contact_email
        self.s2_api_key = s2_api_key
        self.request_timeout = request_timeout
        self.use_cache = use_cache
        self._cache_ready = False
        self._breakers: dict[str, CircuitBreaker] = {
            p.name: CircuitBreaker(fail_max=fail_max, reset_timeout=reset_timeout)
            for p in self.providers
        }
        self.call_counts: dict[str, int] = {}

    @classmethod
    def from_config(
        cls, config: "GlobalConfig", *, cache_engine, run_id: str | None = None
    ) -> "ProviderChain":
        """Build a chain from config: default providers + ``contact_email`` (polite
        pool) + optional ``S2_API_KEY`` / ``core_key`` (decisions 39/73)."""
        from . import build_default_providers

        contact_email = _config_get(config, "contact_email")
        s2_api_key = _config_get(config, "s2_api_key") or _config_get(config, "S2_API_KEY")
        return cls(
            build_default_providers(config),
            cache_engine=cache_engine,
            run_id=run_id,
            contact_email=contact_email,
            s2_api_key=s2_api_key,
        )

    # ------------------------------------------------------------------ breakers
    def breaker(self, name: str) -> CircuitBreaker:
        bp = self._breakers.get(name)
        if bp is None:
            bp = CircuitBreaker()
            self._breakers[name] = bp
        return bp

    # --------------------------------------------------------------- cache conn
    def _open_conn(self) -> "tuple[sqlite3.Connection, bool]":
        """Return ``(conn, owns)`` over ``cache.db`` (owns -> the caller closes it)."""
        import sqlite3

        if isinstance(self.cache_engine, sqlite3.Connection):
            return self.cache_engine, False
        from ..cache.db import init_cache_db
        from ..db.connection import open_cache_db

        if not self._cache_ready:
            init_cache_db(self.cache_engine)
            self._cache_ready = True
        return open_cache_db(self.cache_engine), True

    def _cache_get(self, provider: str, request_key: str) -> Any | None:
        if not self.use_cache:
            return None
        conn, owns = self._open_conn()
        try:
            return cache_get(conn, provider=provider, request_key=request_key)
        finally:
            if owns:
                conn.close()

    def _cache_put(self, provider: str, request_key: str, response: Any) -> None:
        if not self.use_cache:
            return
        conn, owns = self._open_conn()
        try:
            cache_put(conn, provider=provider, request_key=request_key, response=response)
        finally:
            if owns:
                conn.close()

    # --------------------------------------------------------- call dispatch
    async def _dispatch(
        self, method: str, request_key: str, *args: Any, cache_empty: bool = True
    ) -> Any:
        """Fan ``method`` across providers in order; return the first truthy hit.

        For each provider exposing ``method`` whose breaker is closed: a fresh
        cached response (truthy) ends the fan-out; a cached NEGATIVE (empty list)
        falls through to the next provider; a live call's result is cached (truthy
        or negative) and a truthy one is returned. A raised exception records a
        breaker failure and falls through. Each call is bounded by
        ``request_timeout`` via ``asyncio.wait_for`` (a hang -> breaker failure).

        ``cache_empty=False`` (the batched :meth:`by_openalex_ids` path, D-8)
        skips ONLY the negative write: an empty result is still returned, but a
        transient whole-batch outage is never pinned as a 30-day cached negative
        that would starve every subsequent backfill/enrichment pass. Breaker and
        cache-hit behavior are otherwise identical to the stock path.
        """
        last_error: Optional[Exception] = None
        for provider in self.providers:
            fn = getattr(provider, method, None)
            if fn is None:
                continue
            bp = self.breaker(provider.name)
            if bp.is_open():
                continue

            cached = self._cache_get(provider.name, request_key)
            if cached is not None:
                if cached:
                    return cached
                continue  # cached negative -> try the next provider

            try:
                result = await asyncio.wait_for(fn(*args), timeout=self.request_timeout)
            except Exception as exc:  # noqa: BLE001 - breaker reacts to any error
                bp.record_failure()
                last_error = exc
                continue

            self.call_counts[provider.name] = self.call_counts.get(provider.name, 0) + 1
            bp.record_success()
            if result:
                self._cache_put(provider.name, request_key, result)
                return result
            if cache_empty:  # D-8: the batched verb never negative-caches an empty batch
                self._cache_put(provider.name, request_key, result if result is not None else [])

        if last_error is not None and all(
            self.breaker(p.name).is_open() for p in self.providers
        ):
            raise last_error
        return None

    # ----------------------------------------------------- protocol fan-out
    async def by_doi(self, doi: str) -> dict | None:
        """First provider that resolves ``doi`` wins; cached under ``key_by_doi`` (§6.5)."""
        return await self._dispatch("by_doi", key_by_doi(doi), doi)

    async def by_title(self, title: str, year: int | None = None) -> dict | None:
        """Title(+year) lookup across the chain; cached under ``key_by_title`` (§6.5)."""
        return await self._dispatch("by_title", key_by_title(title, year), title, year)

    async def by_openalex_ids(self, ids: list[str]) -> list[dict]:
        """Batched multi-W-id lookup; cached under ``key_by_openalex_ids`` (the
        ADDITIVE §6.5 namespace, Build D ch7).

        Rides :meth:`_dispatch` with the ONE deliberate difference of
        ``cache_empty=False`` (D-8): an EMPTY batch result is returned but never
        written as a cached negative — a whole-batch outage negative-cached for
        the 30-day TTL would starve every subsequent title-backfill/enrichment
        pass, while the stale ``by_doi:`` negatives this verb exists to bypass
        stay in their own namespace. Breaker and cache-hit behavior match
        :meth:`by_doi`. Empty input short-circuits to ``[]`` with no cache read
        and no provider call.
        """
        if not ids:
            return []
        result = await self._dispatch(
            "by_openalex_ids", key_by_openalex_ids(ids), ids, cache_empty=False
        )
        return list(result) if result else []

    async def referenced_works(self, record: dict) -> list[dict]:
        """Outbound reference list for ``record``; cached under
        ``key_referenced_works`` (§6.5) — the load-bearing cross-phase key phase_2
        reconstructs offline."""
        result = await self._dispatch("referenced_works", key_referenced_works(record), record)
        return list(result) if result else []

    async def oa_pdf_candidates(self, record: dict) -> list["OACandidate"]:
        """Cross-provider OA candidate gather; cached per provider under
        ``key_oa_candidates`` (§6.5). Unlike :meth:`_dispatch`'s first-hit, this asks
        EVERY closed-breaker provider and merges their candidates, deduped by URL."""
        from ..acquisition.fetch import OACandidate

        request_key = key_oa_candidates(record)
        merged: list[OACandidate] = []
        seen: set[str] = set()

        def _absorb(items) -> None:
            for cand in items:
                url = (getattr(cand, "url", None) or "").strip().rstrip("/")
                key = url or f"landing:{(getattr(cand, 'landing_url', None) or '').strip()}"
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                merged.append(cand)

        for provider in self.providers:
            fn = getattr(provider, "oa_pdf_candidates", None)
            if fn is None:
                continue
            bp = self.breaker(provider.name)
            if bp.is_open():
                continue

            cached = self._cache_get(provider.name, request_key)
            if cached is not None:
                _absorb(_candidates_from_cache(cached))
                continue

            try:
                result = await asyncio.wait_for(fn(record), timeout=self.request_timeout)
            except Exception:  # noqa: BLE001 - breaker reacts to any error
                bp.record_failure()
                continue

            self.call_counts[provider.name] = self.call_counts.get(provider.name, 0) + 1
            bp.record_success()
            result = list(result or [])
            self._cache_put(provider.name, request_key, [_candidate_to_cache(c) for c in result])
            _absorb(result)

        return merged


# ===========================================================================
# Internal helpers
# ===========================================================================
def _config_get(config: Any, key: str) -> Optional[str]:
    """Read ``key`` from a config object (attribute or mapping), or ``None``."""
    if config is None:
        return None
    if isinstance(config, dict):
        val = config.get(key)
    else:
        val = getattr(config, key, None)
        if val is None:
            # config.content_policy / nested objects may carry contact_email
            cp = getattr(config, "content_policy", None)
            val = getattr(cp, key, None) if cp is not None else None
    if val in (None, ""):
        return None
    return str(val)


def _candidate_to_cache(candidate) -> dict[str, Any]:
    """Serialize an :class:`OACandidate` to a JSON-safe dict for ``provider_cache``."""
    return {
        "url": candidate.url,
        "landing_url": candidate.landing_url,
        "host_type": candidate.host_type,
        "version": candidate.version,
        "provider": candidate.provider,
        "is_oa_asserted": candidate.is_oa_asserted,
    }


def _candidates_from_cache(cached: Any) -> list:
    """Rebuild :class:`OACandidate` objects from a cached JSON list (tolerant)."""
    from ..acquisition.fetch import OACandidate

    out = []
    for row in cached or []:
        if not isinstance(row, dict):
            continue
        out.append(
            OACandidate(
                url=row.get("url"),
                landing_url=row.get("landing_url"),
                host_type=row.get("host_type"),
                version=row.get("version"),
                provider=row.get("provider") or "",
                is_oa_asserted=bool(row.get("is_oa_asserted")),
            )
        )
    return out
