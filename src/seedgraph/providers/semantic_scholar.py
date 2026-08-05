"""Semantic Scholar (S2) provider (ported from v1) — metadata, references, OA PDF.

Supplies metadata, the outbound reference list, and S2's ``openAccessPdf`` when
present. An optional ``S2_API_KEY`` raises the rate limit (BYO, decision 73).
``httpx`` is imported ONLY inside method bodies.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Optional

from ..project.identity import normalize_id
from ._retry import raise_for_transient, retry_transient

if TYPE_CHECKING:
    from .fetch import OACandidate  # noqa: F401  (annotation only)

_API_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "externalIds,title,year,authors,openAccessPdf"


class SemanticScholarProvider:
    """Semantic Scholar ``CitationProvider`` (ported v1)."""

    name = "semantic_scholar"

    def __init__(
        self,
        *,
        s2_api_key: str | None = None,
        contact_email: str | None = None,
        request_timeout: float = 60.0,
        client: Any = None,
        base_url: str = _API_BASE,
        min_request_interval: float = 1.0,
    ) -> None:
        self.s2_api_key = s2_api_key
        self.contact_email = contact_email
        self.request_timeout = request_timeout
        self.base_url = base_url.rstrip("/")
        self._client = client
        # Proactive polite throttle: outbound HTTP calls are spaced >= this many
        # seconds apart (default 1.0 = 1 req/s, matching the BYO key limit).
        # <= 0 disables the gate entirely.
        self._min_request_interval = min_request_interval
        self._rate_lock = asyncio.Lock()
        self._next_allowed = 0.0

    def _get_client(self):
        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.request_timeout)
        return self._client

    @property
    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.s2_api_key} if self.s2_api_key else {}

    async def _get_json(self, path: str, *, params: dict | None = None) -> Optional[dict]:
        import httpx

        # Polite rate gate: serialize *scheduling* under the lock so concurrent
        # coroutines each reserve a slot >= interval apart (a true issue rate),
        # while the HTTP request below runs after the lock releases so in-flight
        # requests can still overlap.
        if self._min_request_interval > 0:
            async with self._rate_lock:
                now = time.monotonic()
                wait = self._next_allowed - now
                if wait > 0:
                    await asyncio.sleep(wait)
                self._next_allowed = time.monotonic() + self._min_request_interval

        client = self._get_client()

        async def _attempt() -> Optional[dict]:
            resp = await client.get(
                f"{self.base_url}{path}", params=params or {}, headers=self._headers
            )
            if resp.status_code == 404:
                return None  # plain miss — never retried
            raise_for_transient(resp, self.name, path)  # 429/5xx -> retried marker
            if resp.status_code >= 400:
                raise httpx.HTTPError(f"semantic_scholar HTTP {resp.status_code} for {path}")
            return resp.json()

        return await retry_transient(_attempt)

    async def by_doi(self, doi: str) -> Optional[dict]:
        norm = normalize_id("doi", doi)
        if not norm:
            return None
        data = await self._get_json(f"/paper/DOI:{norm}", params={"fields": _FIELDS})
        return _paper_to_record(data) if data else None

    async def by_title(self, title: str, year: int | None = None) -> Optional[dict]:
        title = (title or "").strip()
        if not title:
            return None
        data = await self._get_json(
            "/paper/search", params={"query": title, "limit": "1", "fields": _FIELDS}
        )
        items = (data or {}).get("data") if isinstance(data, dict) else None
        if not items:
            return None
        return _paper_to_record(items[0])

    async def referenced_works(self, record: dict) -> list[dict]:
        doi = normalize_id("doi", record.get("doi"))
        if not doi:
            return []
        data = await self._get_json(
            f"/paper/DOI:{doi}/references", params={"fields": "externalIds,title", "limit": "200"}
        )
        items = (data or {}).get("data") if isinstance(data, dict) else None
        if not items:
            return []
        out: list[dict] = []
        for item in items:
            cited = item.get("citedPaper") if isinstance(item, dict) else None
            rec = _paper_to_record(cited)
            if rec:
                out.append(rec)
        return out

    async def oa_pdf_candidates(self, record: dict) -> list["OACandidate"]:
        from ..acquisition.fetch import OACandidate

        doi = normalize_id("doi", record.get("doi"))
        if not doi:
            return []
        data = await self._get_json(f"/paper/DOI:{doi}", params={"fields": "openAccessPdf"})
        oa = (data or {}).get("openAccessPdf") if isinstance(data, dict) else None
        if not isinstance(oa, dict) or not oa.get("url"):
            return []
        return [
            OACandidate(
                url=str(oa["url"]),
                host_type=None,
                provider="semantic_scholar",
                is_oa_asserted=True,
            )
        ]


def _paper_to_record(paper: Optional[dict]) -> Optional[dict]:
    if not isinstance(paper, dict):
        return None
    record: dict = {}
    ext = paper.get("externalIds") or {}
    if isinstance(ext, dict):
        doi = normalize_id("doi", ext.get("DOI"))
        if doi:
            record["doi"] = doi
        if ext.get("ArXiv"):
            record["arxiv_id"] = str(ext["ArXiv"])
    if paper.get("title"):
        record["title"] = paper["title"]
    if isinstance(paper.get("year"), int):
        record["year"] = paper["year"]
    authors = [a.get("name") for a in (paper.get("authors") or []) if isinstance(a, dict) and a.get("name")]
    if authors:
        record["authors"] = authors
    return record or None
