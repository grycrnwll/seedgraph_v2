"""Deterministic offline ``FakeProvider`` (opt-in via ``SEEDGRAPH_FAKE_PROVIDERS``).

A pure in-memory provider + a ``%PDF``-serving fake ``httpx`` transport so the
``seedgraph corpus`` CLI can round-trip a full ``resolve -> walk -> acquire`` flow
**offline, with no network and no LLM** (the build-plan VERIFY step). Tests inject
their own mocks; this exists only so the CLI is exercisable without live APIs. It
is NEVER active unless the env flag is set (``build_default_providers`` /
``acquisition.service.corpus_io`` consult it).
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from ..cache.provider_cache import canonical_request_id


def _wid(seed: str) -> str:
    return "W" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10].upper()


class FakeProvider:
    """Offline ``CitationProvider`` test/demo double (no network)."""

    name = "openalex"

    async def by_doi(self, doi: str) -> Optional[dict]:
        text = str(doi).strip()
        if text.upper().startswith("W"):
            # An OpenAlex W-id lookup (the walk's DOI backfill path).
            return {
                "openalex_id": text.upper(),
                "doi": f"10.fake/{text.lower()}",
                "title": f"Referenced work {text}",
                "year": 2019,
            }
        return {
            "doi": text.lower(),
            "openalex_id": _wid(text),
            "title": f"Seed work {text}",
            "year": 2020,
        }

    async def by_title(self, title: str, year: int | None = None) -> Optional[dict]:
        return None

    async def by_openalex_ids(self, ids: list[str]) -> list[dict]:
        # Batched W-id verb (Build D ch7-9): mirrors the by_doi W-id branch above
        # so the offline CLI round-trip exercises the walk's batched enrichment
        # prefetch (ch8) and `corpus backfill-titles` (ch9) with no network.
        out: list[dict] = []
        for wid in ids or []:
            text = str(wid).strip().upper()
            if text.startswith("W"):
                out.append(
                    {
                        "openalex_id": text,
                        "doi": f"10.fake/{text.lower()}",
                        "title": f"Referenced work {text}",
                        "year": 2019,
                    }
                )
        return out

    async def referenced_works(self, record: dict) -> list[dict]:
        src = canonical_request_id(record)
        return [{"openalex_id": _wid(src + "#1")}, {"openalex_id": _wid(src + "#2")}]

    async def oa_pdf_candidates(self, record: dict) -> list[Any]:
        # No provider-served OA copy; arXiv-direct (handled by fetch_oa) is the only
        # OA route in the fake flow, so non-arXiv works fall to requires_user_upload.
        return []


def fake_http_client():
    """An ``httpx.AsyncClient`` whose transport returns a tiny ``%PDF`` for arXiv
    hosts and 404 otherwise (offline OA download for the CLI verify)."""
    import httpx

    def _handler(request: httpx.Request) -> httpx.Response:
        host = (request.url.host or "").lower()
        if "arxiv.org" in host:
            return httpx.Response(200, content=b"%PDF-1.4 fake arxiv pdf body\n%%EOF\n")
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(_handler))
