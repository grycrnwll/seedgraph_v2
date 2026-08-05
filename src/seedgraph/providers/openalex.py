"""OpenAlex provider (ported from v1) — the primary metadata + references source.

OpenAlex supplies the canonical record, the outbound ``referenced_works`` id list
(the spine of the corpus-growth walk), and OA location candidates. ``httpx`` is
imported ONLY inside method bodies; the polite pool is activated by ``contact_email``.
A test/caller may inject an ``httpx.AsyncClient`` (wired to a ``MockTransport``) so
the provider runs fully offline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..project.identity import ARXIV_DOI_RE, normalize_id
from ._retry import raise_for_transient, retry_transient

if TYPE_CHECKING:
    from ..acquisition.fetch import OACandidate  # noqa: F401  (annotation only)

_API_BASE = "https://api.openalex.org"


def _normalize_openalex_id(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    for prefix in ("https://openalex.org/", "http://openalex.org/", "openalex.org/"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    text = text.strip().upper()
    return text or None


def _arxiv_id_from_input(value: Any) -> Optional[str]:
    """Extract a normalized arXiv id from arXiv-shaped input, or ``None``.

    Accepts bare new-style ids (``2401.01234``, optional ``vN`` suffix),
    ``arXiv:`` / ``arxiv.org/abs|pdf/`` forms, and the arXiv-minted DOI surface
    form (``10.48550/arXiv.{id}``). Detection operates on RAW input shapes, so
    it tolerates either land order with Build A's identity-layer 10.48550 fold
    (which strips the DOI form before it ever reaches the provider). Anything
    else — plain DOIs included — returns ``None`` so the caller's DOI path
    handles it.
    """
    if not value:
        return None
    # Lazy import: keeps the provider import-light (the same convention as the
    # OACandidate import inside _oa_candidates below).
    from ..acquisition.fetch import ARXIV_BARE_RE

    text = str(value).strip()
    doi_alias = ARXIV_DOI_RE.match(text)
    if doi_alias:
        text = doi_alias.group(1).strip()
    elif "arxiv" not in text.lower() and not ARXIV_BARE_RE.match(text):
        return None
    norm = normalize_id("arxiv", text)
    # Only new-style NNNN.NNNNN ids mint the predictable ``10.48550/arXiv.{id}``
    # DOI we construct; anything else (old-style ``hep-th/…``, arXiv-adjacent
    # DOIs) falls back to the caller's DOI path.
    if norm and ARXIV_BARE_RE.match(norm):
        return norm
    return None


class OpenAlexProvider:
    """OpenAlex ``CitationProvider`` (ported v1)."""

    name = "openalex"

    def __init__(
        self,
        *,
        contact_email: str | None = None,
        api_key: str | None = None,
        request_timeout: float = 60.0,
        client: Any = None,
        base_url: str = _API_BASE,
    ) -> None:
        self.contact_email = contact_email
        self.api_key = api_key  # OpenAlex premium; sent as the `api_key` query param
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
        if self.api_key:
            query["api_key"] = self.api_key
        if params:
            query.update(params)

        async def _attempt() -> Optional[dict]:
            resp = await client.get(f"{self.base_url}{path}", params=query)
            if resp.status_code == 404:
                return None  # plain miss, not an error — never retried
            raise_for_transient(resp, self.name, path)  # 429/5xx -> retried marker
            if resp.status_code >= 400:
                raise httpx.HTTPError(f"openalex HTTP {resp.status_code} for {path}")
            return resp.json()

        return await retry_transient(_attempt)

    async def _fetch_work(self, record: dict) -> Optional[dict]:
        oa = _normalize_openalex_id(record.get("openalex_id") or record.get("openalex"))
        if oa and oa.startswith("W"):
            return await self._get_json(f"/works/{oa}")
        doi = normalize_id("doi", record.get("doi"))
        if doi:
            return await self._get_json(f"/works/doi:{doi}")
        arxiv = _arxiv_id_from_input(record.get("arxiv_id") or record.get("arxiv"))
        if arxiv:
            # arXiv-only record: resolve via the arXiv-minted DOI (see by_doi's
            # comment for why this endpoint, and never landing_page_url.search).
            return await self._get_json(f"/works/doi:10.48550/arXiv.{arxiv}")
        return None

    async def by_doi(self, doi: str) -> Optional[dict]:
        oa = _normalize_openalex_id(doi)
        if oa and oa.startswith("W"):
            data = await self._get_json(f"/works/{oa}")
            return _work_to_record(data) if data else None
        arxiv = _arxiv_id_from_input(doi)
        if arxiv:
            # arXiv mints a DOI for every paper under the ``10.48550/arXiv.``
            # prefix, and OpenAlex indexes works by that DOI. Resolve via the
            # canonical ``/works/doi:`` endpoint: a hit returns the work, a
            # miss is a clean 404 (-> None -> the chain falls through to a bare
            # arXiv record). The older ``locations.landing_page_url.search``
            # filter was deprecated and now returns HTTP 400 (which would raise
            # and zero out provider_reference expansion for arXiv seeds).
            data = await self._get_json(f"/works/doi:10.48550/arXiv.{arxiv}")
            return _work_to_record(data) if data else None
        norm = normalize_id("doi", doi)
        if norm:
            data = await self._get_json(f"/works/doi:{norm}")
            return _work_to_record(data) if data else None
        return None

    async def by_title(self, title: str, year: int | None = None) -> Optional[dict]:
        title = (title or "").strip()
        if not title:
            return None
        filters = [f"title.search:{title}"]
        if year:
            filters.append(f"publication_year:{int(year)}")
        data = await self._get_json("/works", params={"filter": ",".join(filters), "per-page": "1"})
        results = (data or {}).get("results") if isinstance(data, dict) else None
        if not results:
            return None
        return _work_to_record(results[0])

    async def referenced_works(self, record: dict) -> list[dict]:
        work = await self._fetch_work(record)
        if not work:
            return []
        out: list[dict] = []
        for ref in work.get("referenced_works") or []:
            oa = _normalize_openalex_id(ref)
            if oa:
                out.append({"openalex_id": oa})
        return out

    async def oa_pdf_candidates(self, record: dict) -> list["OACandidate"]:
        work = await self._fetch_work(record)
        if not work:
            return []
        return _oa_candidates(work)

    # ------------------------------------------- additive: batched W-id fetch
    async def by_openalex_ids(self, ids: list[str]) -> list[dict]:
        """Batch-resolve OpenAlex ``W…`` ids to shaped records in ONE call.

        The efficiency hook (Build D ch7, ported v1) for BOTH downstream
        consumers: the walk's DOI-less reference enrichment (ch8) and the
        post-walk title backfill (ch9) — titling/enriching N stubs costs
        ceil(N/50) HTTP calls instead of N. Purely ADDITIVE: it does not touch
        :meth:`referenced_works` or the pinned reference path.

        Each ``W…`` id is normalized and de-duplicated (order preserved); the
        batch is fetched in ONE request via the OpenAlex
        ``filter=ids.openalex:W1|W2|…`` form with ``per-page=200`` and a narrow
        ``select``. The select deliberately carries ``doi`` on top of v1's
        title-only field list so the walk-side consumer keeps the r2-1 §6
        per-ref DOI backfill — a naive narrow substitution would silently drop
        it. The caller chunks the id list (≤~50/batch) so a single page (200)
        always covers a batch; this method issues exactly one HTTP call per
        invocation, riding the retried :meth:`_get_json` (ch1).

        Returns shaped records (one per indexed work the API returned) in
        whatever order OpenAlex pages them; missing ids simply do not appear.
        Returns ``[]`` for an empty/blank id list WITHOUT a network call.

        Cache hygiene (D-8): the chain namespaces this verb under its own
        ``by_openalex_ids:`` cache key — DISTINCT from ``by_doi:``, whose
        30-day negative entries pinned the untitled state in the first place —
        and never caches an empty batch result.
        """
        seen: set[str] = set()
        norm_ids: list[str] = []
        for raw in ids or []:
            oa = _normalize_openalex_id(raw)
            if oa and oa.startswith("W") and oa not in seen:
                seen.add(oa)
                norm_ids.append(oa)
        if not norm_ids:
            return []

        params = {
            "filter": f"ids.openalex:{'|'.join(norm_ids)}",
            "per-page": "200",
            "select": "id,doi,title,publication_year,authorships",
        }
        data = await self._get_json("/works", params=params)
        results = (data or {}).get("results") if isinstance(data, dict) else None
        if not results:
            return []
        out: list[dict] = []
        for work in results:
            record = _work_to_record(work)
            if record:
                out.append(record)
        return out


# --- response shaping -------------------------------------------------------
def _work_to_record(work: Optional[dict]) -> Optional[dict]:
    if not work or not isinstance(work, dict):
        return None
    record: dict = {}
    oa = _normalize_openalex_id(work.get("id"))
    if oa:
        record["openalex_id"] = oa
    doi = normalize_id("doi", work.get("doi"))
    if doi:
        record["doi"] = doi
    title = work.get("title") or work.get("display_name")
    if title:
        record["title"] = title
    year = work.get("publication_year")
    if isinstance(year, int):
        record["year"] = year
    authors = [
        a.get("author", {}).get("display_name")
        for a in (work.get("authorships") or [])
        if isinstance(a, dict) and isinstance(a.get("author"), dict)
    ]
    authors = [a for a in authors if a]
    if authors:
        record["authors"] = authors
    # Build D ch10 (D-10): shaping happens HERE because the chain caches the
    # SHAPED record — pre-existing cache entries simply lack the two fields
    # (harmless: nullable columns, backfilled empty-only on the next fresh fetch).
    abstract = _abstract_from_inverted_index(work.get("abstract_inverted_index"))
    if abstract:
        record["abstract"] = abstract
    oa_status = _oa_status(work)
    if oa_status:
        record["oa_status"] = oa_status
    return record or None


def _abstract_from_inverted_index(inverted: Any) -> Optional[str]:
    """Reconstruct an abstract string from OpenAlex's ``abstract_inverted_index``.

    OpenAlex does not ship plain abstracts; it stores them as
    ``{word: [positions...]}``. This re-orders the words by position to recover
    the running text (ported v1). Returns ``None`` when absent or malformed.
    The stored abstract is LOCAL-ONLY (CONTENT_ACCESS_POLICY.md:42) — the D8
    export allowlist (``vocab.PROVIDER_SHAREABLE_FIELDS``) excludes it.
    """
    if not isinstance(inverted, dict) or not inverted:
        return None
    positioned: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                positioned.append((pos, word))
    if not positioned:
        return None
    positioned.sort(key=lambda p: p[0])
    return " ".join(word for _, word in positioned)


def _oa_status(work: dict) -> Optional[str]:
    """Return the OA color (``gold``/``green``/``hybrid``/``bronze``/``diamond``/
    ``closed``) from the work's ``open_access`` object, or ``None`` (ported v1)."""
    oa = work.get("open_access")
    if isinstance(oa, dict):
        status = oa.get("oa_status")
        if status:
            return str(status)
    return None


def _oa_candidates(work: dict) -> list["OACandidate"]:
    from ..acquisition.fetch import OACandidate

    out: list[OACandidate] = []
    seen: set[tuple] = set()

    def _loc(loc) -> None:
        if not isinstance(loc, dict) or not loc.get("is_oa"):
            return
        source = loc.get("source")
        host_type = source.get("type") if isinstance(source, dict) else None
        key = (loc.get("pdf_url"), loc.get("landing_page_url"))
        if key in seen:
            return
        seen.add(key)
        out.append(
            OACandidate(
                url=loc.get("pdf_url") or None,
                landing_url=loc.get("landing_page_url") or None,
                host_type=host_type,
                version=loc.get("version") or None,
                provider="openalex",
                is_oa_asserted=True,
            )
        )

    for loc in work.get("locations") or []:
        _loc(loc)
    for curated in ("best_oa_location", "primary_location"):
        _loc(work.get(curated))
    return out
