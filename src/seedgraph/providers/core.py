"""CORE provider (ported from v1) — BYO-key, LAST in the chain.

CORE aggregates OA full-text repositories; it requires a ``core_key`` and is tried
last (decision 73). OA-only. ``httpx`` is imported ONLY inside method bodies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ._retry import raise_for_transient, retry_transient

if TYPE_CHECKING:
    from .fetch import OACandidate  # noqa: F401  (annotation only)

_API_BASE = "https://api.core.ac.uk/v3"


class CoreProvider:
    """CORE ``CitationProvider`` (ported v1; BYO ``core_key``, last in chain)."""

    name = "core"

    def __init__(
        self,
        *,
        core_key: str | None = None,
        contact_email: str | None = None,
        request_timeout: float = 60.0,
        client: Any = None,
        base_url: str = _API_BASE,
    ) -> None:
        self.core_key = core_key
        self.contact_email = contact_email
        self.request_timeout = request_timeout
        self.base_url = base_url.rstrip("/")
        self._client = client

    def _get_client(self):
        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.request_timeout)
        return self._client

    async def _search(self, query: str) -> Optional[dict]:
        import httpx

        if not self.core_key:
            return None
        client = self._get_client()

        async def _attempt() -> Optional[dict]:
            resp = await client.get(
                f"{self.base_url}/search/works",
                params={"q": query, "limit": "1"},
                headers={"Authorization": f"Bearer {self.core_key}"},
            )
            if resp.status_code == 404:
                return None  # plain miss — never retried
            raise_for_transient(resp, self.name)  # 429/5xx -> retried marker
            if resp.status_code >= 400:
                raise httpx.HTTPError(f"core HTTP {resp.status_code}")
            return resp.json()

        return await retry_transient(_attempt)

    async def by_title(self, title: str, year: int | None = None) -> Optional[dict]:
        title = (title or "").strip()
        if not title:
            return None
        data = await self._search(f'title:"{title}"')
        results = (data or {}).get("results") if isinstance(data, dict) else None
        if not results:
            return None
        first = results[0]
        record: dict = {}
        if first.get("title"):
            record["title"] = first["title"]
        if isinstance(first.get("yearPublished"), int):
            record["year"] = first["yearPublished"]
        return record or None

    async def oa_pdf_candidates(self, record: dict) -> list["OACandidate"]:
        from ..acquisition.fetch import OACandidate

        title = record.get("title") if isinstance(record, dict) else None
        if not title:
            return []
        data = await self._search(f'title:"{title}"')
        results = (data or {}).get("results") if isinstance(data, dict) else None
        if not results:
            return []
        url = results[0].get("downloadUrl")
        if not url:
            return []
        return [
            OACandidate(url=str(url), host_type="repository", provider="core", is_oa_asserted=True)
        ]
