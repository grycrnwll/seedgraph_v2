"""Unpaywall provider (ported from v1) — OA location lookup by DOI.

Unpaywall is OA-only: given a DOI it returns the best OA location(s). It never
provides paywalled full text. ``httpx`` is imported ONLY inside method bodies;
``contact_email`` is the required polite-pool identifier (without one this provider
degrades to a clean miss rather than hitting the API).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..project.identity import normalize_id
from ._retry import raise_for_transient, retry_transient

if TYPE_CHECKING:
    from .fetch import OACandidate  # noqa: F401  (annotation only)

_API_BASE = "https://api.unpaywall.org/v2"


class UnpaywallProvider:
    """Unpaywall ``CitationProvider`` (ported v1; OA candidates by DOI)."""

    name = "unpaywall"

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

    async def _fetch(self, doi: str) -> Optional[dict]:
        import httpx

        if not self.contact_email:
            return None
        norm = normalize_id("doi", doi)
        if not norm:
            return None
        client = self._get_client()

        async def _attempt() -> Optional[dict]:
            resp = await client.get(
                f"{self.base_url}/{norm}", params={"email": self.contact_email}
            )
            if resp.status_code == 404:
                return None  # plain miss — never retried
            raise_for_transient(resp, self.name, norm)  # 429/5xx -> retried marker
            if resp.status_code >= 400:
                raise httpx.HTTPError(f"unpaywall HTTP {resp.status_code} for {norm}")
            return resp.json()

        return await retry_transient(_attempt)

    async def by_doi(self, doi: str) -> Optional[dict]:
        payload = await self._fetch(doi)
        if not payload:
            return None
        record: dict = {}
        norm = normalize_id("doi", payload.get("doi"))
        if norm:
            record["doi"] = norm
        if payload.get("title"):
            record["title"] = str(payload["title"])
        if payload.get("year") is not None:
            try:
                record["year"] = int(payload["year"])
            except (TypeError, ValueError):
                pass
        # Build D ch10: Unpaywall's payload already carries the OA color at the
        # top level — stop dropping it (it feeds the triage split via works.oa_status).
        if payload.get("oa_status"):
            record["oa_status"] = str(payload["oa_status"])
        return record or None

    async def oa_pdf_candidates(self, record: dict) -> list["OACandidate"]:
        from ..acquisition.fetch import OACandidate

        doi = record.get("doi") if isinstance(record, dict) else None
        if not doi:
            return []
        payload = await self._fetch(str(doi))
        if not payload:
            return []
        out: list[OACandidate] = []
        seen: set[tuple] = set()
        locations = list(payload.get("oa_locations") or [])
        best = payload.get("best_oa_location")
        if isinstance(best, dict):
            locations.append(best)
        for loc in locations:
            if not isinstance(loc, dict):
                continue
            url = loc.get("url_for_pdf") or None
            landing = loc.get("url") or None
            key = (url, landing)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                OACandidate(
                    url=str(url) if url else None,
                    landing_url=str(landing) if landing else None,
                    host_type=loc.get("host_type") or None,
                    version=loc.get("version") or None,
                    provider="unpaywall",
                    is_oa_asserted=True,
                )
            )
        return out
