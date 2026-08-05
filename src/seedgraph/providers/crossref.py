"""Crossref provider (ported from v1) — DOI + title metadata fallback.

Used to confirm/canonicalize DOIs and resolve title-only seeds when OpenAlex
misses. ``httpx`` is imported ONLY inside method bodies; ``contact_email`` activates
the polite pool. A caller may inject an ``httpx.AsyncClient`` for offline tests.
"""

from __future__ import annotations

from typing import Any, Optional

from ..project.identity import normalize_id
from ._retry import raise_for_transient, retry_transient

_API_BASE = "https://api.crossref.org"


class CrossrefProvider:
    """Crossref ``CitationProvider`` (ported v1)."""

    name = "crossref"

    def __init__(
        self,
        *,
        contact_email: str | None = None,
        request_timeout: float = 60.0,
        client: Any = None,
        base_url: str = _API_BASE,
    ) -> None:
        self.contact_email = contact_email
        self.request_timeout = request_timeout
        self.base_url = base_url.rstrip("/")
        self._client = client

    def _get_client(self):
        import httpx

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.request_timeout)
        return self._client

    @property
    def _polite_params(self) -> dict[str, str]:
        return {"mailto": self.contact_email} if self.contact_email else {}

    async def _get_json(self, path: str, *, params: dict | None = None) -> Optional[dict]:
        import httpx

        client = self._get_client()
        query = dict(self._polite_params)
        if params:
            query.update(params)

        async def _attempt() -> Optional[dict]:
            resp = await client.get(f"{self.base_url}{path}", params=query)
            if resp.status_code == 404:
                return None  # plain miss — never retried
            raise_for_transient(resp, self.name, path)  # 429/5xx -> retried marker
            if resp.status_code >= 400:
                raise httpx.HTTPError(f"crossref HTTP {resp.status_code} for {path}")
            return resp.json()

        return await retry_transient(_attempt)

    async def by_doi(self, doi: str) -> Optional[dict]:
        norm = normalize_id("doi", doi)
        if not norm:
            return None
        data = await self._get_json(f"/works/{norm}")
        message = (data or {}).get("message") if isinstance(data, dict) else None
        return _message_to_record(message) if message else None

    async def by_title(self, title: str, year: int | None = None) -> Optional[dict]:
        title = (title or "").strip()
        if not title:
            return None
        data = await self._get_json(
            "/works", params={"query.bibliographic": title, "rows": "1"}
        )
        items = ((data or {}).get("message") or {}).get("items") if isinstance(data, dict) else None
        if not items:
            return None
        return _message_to_record(items[0])


def _message_to_record(message: dict) -> Optional[dict]:
    if not isinstance(message, dict):
        return None
    record: dict = {}
    doi = normalize_id("doi", message.get("DOI"))
    if doi:
        record["doi"] = doi
    titles = message.get("title")
    if isinstance(titles, list) and titles:
        record["title"] = titles[0]
    elif isinstance(titles, str):
        record["title"] = titles
    issued = (message.get("issued") or {}).get("date-parts") if isinstance(message, dict) else None
    if isinstance(issued, list) and issued and isinstance(issued[0], list) and issued[0]:
        year = issued[0][0]
        if isinstance(year, int):
            record["year"] = year
    authors = []
    for a in message.get("author") or []:
        if not isinstance(a, dict):
            continue
        name = " ".join(p for p in (a.get("given"), a.get("family")) if p)
        if name:
            authors.append(name)
    if authors:
        record["authors"] = authors
    return record or None
