"""OA acquisition (ported v1 ``fetch.py``) — OA-only, paywall-safe (§6.3).

Gather cross-provider OA candidates -> repository-first rank -> ``%PDF`` validate
-> return downloaded bytes XOR a stub (never a paywall fetch, never raises for a
plain no-OA/closed record). The constructed-vs-asserted-OA gate is the ported
``_PREPRINT_HOSTS`` allowlist; SSRN is link-only (no constructor, never
allowlisted); arXiv resolves via id-normalize + a deterministic PDF URL (doc 01 §7;
plan §2); landing->PDF derivers (bioRxiv/medRxiv, OSF, NBER — build D chunk 3)
recover a direct PDF guess from a landing-only copy, gated as constructed URLs. Ingestion/conversion are done by the service via phase_1 ``ingest_file`` /
``convert_source_file`` — there is NO bespoke hashing/storage here. ``httpx`` is
referenced ONLY inside method bodies.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse

from ..project.identity import ARXIV_DOI_RE, normalize_id

if TYPE_CHECKING:
    from ..providers.base import ProviderChain

#: Hard cap on OA candidates tried for ONE paper (plan §1).
MAX_OA_CANDIDATES: int = 5

#: Hard ceiling on a downloaded PDF (bytes) — a guard against a mis-routed OA URL
#: streaming something enormous (ported v1). 200 MiB is far above any real paper
#: PDF. Enforced INCREMENTALLY by the streamed :func:`_download` (D-3 — a post-hoc
#: ``len(resp.content)`` check would have already buffered the bytes); overflow
#: aborts that candidate and falls through to the next.
MAX_PDF_BYTES: int = 200 * 1024 * 1024

#: Default per-paper wall-clock deadline (seconds) over candidate resolution PLUS
#: the whole download loop (D-2). httpx timeouts are per-read, not total, so a
#: server that trickles bytes (or holds a keep-alive) resets them indefinitely —
#: observed live in v1: a publisher OA endpoint wedged a whole run for 67 minutes.
#: Only ONE shared wall-clock deadline bounds a hostile host; a fresh per-candidate
#: timeout would multiply the wedge by :data:`MAX_OA_CANDIDATES`.
DEFAULT_PAPER_TIMEOUT: float = 300.0

#: A real PDF begins with ``%PDF-`` (scan the first KiB — the spec tolerates a
#: little leading junk). An OA URL is frequently an HTML landing page; saving that
#: as a PDF would crash Marker, so the magic-check turns it into a clean STUB.
_PDF_MAGIC: bytes = b"%PDF-"

_ARXIV_PDF_BASE = "https://arxiv.org/pdf"
ARXIV_BARE_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")

#: FROZEN preprint/repository host allowlist (ported v1, plan §3a). A CONSTRUCTED or
#: DERIVED fetch URL (one no provider vouched for as OA — ``is_oa_asserted=False``)
#: is downloaded ONLY when its host is in this set; ``%PDF`` still validates the
#: bytes. Asserted-OA URLs (a provider's ``pdf_url``) bypass this gate — the OA
#: assertion is their sanction. SSRN is DELIBERATELY EXCLUDED and has no
#: constructor: link-only, permanent.
_PREPRINT_HOSTS: frozenset[str] = frozenset(
    {
        "arxiv.org",
        "biorxiv.org",
        "medrxiv.org",
        "chemrxiv.org",
        "osf.io",
        "europepmc.org",
        "www.ncbi.nlm.nih.gov",
        "nber.org",
        "econstor.eu",
        "core.ac.uk",
    }
)

_VERSION_RANK: dict[str, int] = {
    "publishedversion": 3,
    "acceptedversion": 2,
    "submittedversion": 1,
}


@dataclass
class OACandidate:
    """A candidate OA location (ported v1 §6.3 fields)."""

    url: Optional[str] = None
    landing_url: Optional[str] = None
    host_type: Optional[str] = None
    version: Optional[str] = None
    provider: str = ""
    is_oa_asserted: bool = False


@dataclass
class FetchResult:
    """Outcome of an OA fetch attempt.

    ``status`` is ``'downloaded'`` (``content`` set, ``candidate`` the winning
    location), ``'stub'`` (no-OA / SSRN / paywall / all-clean-non-PDF — ``content``
    is ``None``, no file written, never raises), or ``'failed'`` (TRANSIENT: at
    least one candidate raised a transport error — or the per-paper deadline
    expired — and none produced a PDF; RETURNED, never raised, so one hostile host
    cannot abort the whole corpus pass; no bridge row is written, so a re-run
    naturally retries — D-4). ``candidates_tried`` records the ranked attempts
    for diagnostics.
    """

    status: str
    content: Optional[bytes] = None
    candidate: Optional[OACandidate] = None
    landing_url: Optional[str] = None
    candidates_tried: list = field(default_factory=list)


# --- helpers ----------------------------------------------------------------
def _looks_like_pdf(data: bytes) -> bool:
    return _PDF_MAGIC in data[:1024]


def _host_of(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        host = urlparse(str(url)).hostname
    except (ValueError, TypeError):
        return None
    if not host:
        return None
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def _host_in_allowlist(url: Optional[str]) -> bool:
    host = _host_of(url)
    if host is None:
        return False
    if host in _PREPRINT_HOSTS:
        return True
    return host == "osf.io" or host.endswith(".osf.io")


def _arxiv_id_from_record(record: dict) -> Optional[str]:
    raw = record.get("arxiv_id") or record.get("arxiv")
    if raw:
        norm = normalize_id("arxiv", raw)
        if norm and not norm.lower().startswith("10."):
            return norm
    doi = record.get("doi")
    if doi:
        m = ARXIV_DOI_RE.match(str(doi).strip())
        if m:
            return normalize_id("arxiv", m.group(1)) or m.group(1)
    return None


def arxiv_pdf_url(record: dict) -> Optional[str]:
    """Build a direct arXiv OA PDF URL for ``record`` (or ``None``).

    arXiv is open-access by construction, so a normalized arXiv id always yields a
    legitimately-OA PDF URL (``https://arxiv.org/pdf/{id}``).
    """
    arxiv = _arxiv_id_from_record(record)
    return f"{_ARXIV_PDF_BASE}/{arxiv}" if arxiv else None


def _version_rank(version: Optional[str]) -> int:
    if not version:
        return 0
    return _VERSION_RANK.get(str(version).strip().lower(), 0)


def _is_repository(candidate: OACandidate) -> bool:
    if (candidate.host_type or "").strip().lower() == "repository":
        return True
    return _host_in_allowlist(candidate.url) or _host_in_allowlist(candidate.landing_url)


def _candidate_tier(candidate: OACandidate) -> int:
    if candidate.provider == "arxiv":
        return 0
    if _is_repository(candidate):
        return 1
    return 2


# ---------------------------------------------------------------------------
# Landing->PDF derivation + the NBER DOI constructor (ported v1, plan D chunk 3)
# ---------------------------------------------------------------------------
# These are the GENERALIZED arXiv trick: a derived URL is a guess (no provider
# vouched for it), so each one is emitted with ``is_oa_asserted=False`` and is
# fetched ONLY if its host passes ``_host_in_allowlist`` (the gate in
# :func:`fetch_oa`) and the bytes pass ``%PDF``. They are pure string ops (no
# API call), fired from ``_resolve_oa_candidates`` over the gathered candidates'
# ``landing_url``s — and a derived URL that duplicates an already-present
# ``pdf_url`` is removed by the normalized-URL dedupe, so a derived candidate is
# only TRIED when a higher-ranked real copy failed. bioRxiv/medRxiv and OSF
# derive from the LANDING URL only (no DOI-only construction — ``10.1101`` is
# shared by both servers and the OSF guid is not on the record, critic G3/G5);
# NBER is the one DOI constructor.

#: A bioRxiv/medRxiv landing URL carrying a version segment (``…/{doi}v{N}``). The
#: versioned form serves the PDF at ``….full.pdf``; the versionless form often
#: serves HTML, so a ``vN`` segment is REQUIRED (critic G3).
_BIORXIV_VERSIONED_RE = re.compile(r"/10\.1101/[^/?#]+v\d+", re.IGNORECASE)

#: An NBER working-paper DOI (``10.3386/w{n}``) → the captured working-paper number.
_NBER_DOI_RE = re.compile(r"(?i)^10\.3386/w(\d+)$")


def _derive_biorxiv_pdf(landing_url: Optional[str]) -> Optional[str]:
    """bioRxiv/medRxiv ``…/{doi}vN`` landing → ``{landing}.full.pdf`` (else ``None``).

    Requires a ``vN`` version segment (critic G3: versionless often serves HTML).
    The host must be biorxiv/medrxiv; the allowlist gate re-checks at fetch time.
    """
    host = _host_of(landing_url)
    if host not in ("biorxiv.org", "medrxiv.org"):
        return None
    landing = str(landing_url).split("#", 1)[0].split("?", 1)[0].rstrip("/")
    if not _BIORXIV_VERSIONED_RE.search(landing):
        return None
    return f"{landing}.full.pdf"


def _derive_osf_pdf(landing_url: Optional[str]) -> Optional[str]:
    """OSF ``osf.io/{guid}`` landing → ``{landing}/download`` (else ``None``).

    Covers the whole OSF family (PsyArXiv / SocArXiv / engrXiv via ``*.osf.io``).
    Derived from the landing URL only — the guid is not on the record (critic G5).
    """
    if not _host_in_allowlist(landing_url):
        return None
    host = _host_of(landing_url)
    if host != "osf.io" and not (host or "").endswith(".osf.io"):
        return None
    parsed = urlparse(str(landing_url))
    # Need a single guid path segment (osf.io/{guid}); deeper/blank paths refused
    # (critic G5 — a deep path is a project page/file view, not a preprint guid).
    segments = [s for s in parsed.path.split("/") if s]
    if len(segments) != 1 or segments[0].lower() == "download":
        return None
    base = f"{parsed.scheme or 'https'}://{parsed.netloc}/{segments[0]}"
    return f"{base}/download"


def _derive_nber_pdf(record: dict) -> Optional[str]:
    """NBER DOI (``10.3386/w{n}``) → the working-paper PDF URL (else ``None``).

    The one DOI constructor (critic-surviving — a clean unambiguous prefix). An
    off path simply 404s and falls through harmlessly (candidate loop + ``%PDF``).
    """
    doi = record.get("doi")
    if not doi:
        return None
    m = _NBER_DOI_RE.match(str(doi).strip())
    if not m:
        return None
    n = m.group(1)
    return f"https://www.nber.org/system/files/working_papers/w{n}/w{n}.pdf"


def _derive_candidates(record: dict, gathered: "list[OACandidate]") -> list[OACandidate]:
    """Build derived (constructed) PDF candidates from ``gathered`` + ``record``.

    Pure string ops, no network. Returns NEW tier-1 candidates with
    ``is_oa_asserted=False`` (gated by the host allowlist + ``%PDF`` at fetch time):

    * bioRxiv/medRxiv — each gathered candidate's ``landing_url`` on a versioned
      ``biorxiv``/``medrxiv`` page → ``….full.pdf`` (G3).
    * OSF — each ``landing_url`` ``osf.io/{guid}`` → ``…/download`` (G5).
    * NBER — the record's ``10.3386/w{n}`` DOI → the working-paper PDF (one
      constructor). Always emitted (allowlist + ``%PDF`` make an off path harmless).
    """
    derived: list[OACandidate] = []

    def _emit(url: Optional[str], landing: Optional[str], provider: str) -> None:
        if not url:
            return
        derived.append(
            OACandidate(
                url=url,
                landing_url=landing,
                host_type="repository",
                provider=provider,
                is_oa_asserted=False,
            )
        )

    for cand in gathered:
        landing = cand.landing_url
        _emit(_derive_biorxiv_pdf(landing), landing, "biorxiv_derived")
        _emit(_derive_osf_pdf(landing), landing, "osf_derived")

    _emit(_derive_nber_pdf(record), None, "nber_derived")
    return derived


async def _resolve_oa_candidates(record: dict, chain: "ProviderChain | None") -> list[OACandidate]:
    """Resolve the RANKED OA candidate list for ``record`` (OA-only).

    Gathers arXiv-direct (a guaranteed real PDF, OA by construction; tier 0) + the
    provider chain's cross-provider candidate fan-out + derived landing->PDF
    guesses (:func:`_derive_candidates` — ``is_oa_asserted=False``, so the fetch
    gate applies), drops landing-only copies, ranks repository-first
    (``(tier, -version_rank)``), dedupes by normalized URL, and caps at
    :data:`MAX_OA_CANDIDATES`. Returns ``[]`` when no OA PDF exists.
    """
    candidates: list[OACandidate] = []

    arxiv_url = arxiv_pdf_url(record)
    if arxiv_url:
        candidates.append(OACandidate(url=arxiv_url, provider="arxiv", is_oa_asserted=True))

    if chain is not None and hasattr(chain, "oa_pdf_candidates"):
        gathered = await chain.oa_pdf_candidates(dict(record))
        candidates.extend(gathered or [])

    # Landing->PDF derivation (build D chunk 3): synthesize direct-PDF guesses
    # from the gathered candidates' landing_urls + the record's NBER DOI. Runs
    # AFTER the chain gather and BEFORE the downloadable filter/dedupe below, so
    # a derived URL duplicating a real (earlier-gathered, stable-sorted) pdf_url
    # dedupes away and derived candidates still ride the fetch-time allowlist gate.
    candidates.extend(_derive_candidates(record, candidates))

    downloadable = [c for c in candidates if c.url]
    downloadable.sort(key=lambda c: (_candidate_tier(c), -_version_rank(c.version)))

    deduped: list[OACandidate] = []
    seen: set[str] = set()
    for cand in downloadable:
        key = (cand.url or "").strip().rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
        if len(deduped) >= MAX_OA_CANDIDATES:
            break
    return deduped


async def _download(url: str, client: Any) -> bytes:
    """GET ``url`` STREAMED, accumulating against :data:`MAX_PDF_BYTES` (D-3).

    Raises ``httpx.HTTPError`` on any transport failure, non-2xx response, or
    byte-cap overflow, so the caller maps all three to the same per-candidate
    fall-through. The cap is checked incrementally per chunk — an oversized body is
    aborted mid-flight, never fully buffered (the rejected post-hoc
    ``len(resp.content)`` check guards nothing).
    """
    import httpx

    total = 0
    chunks: list[bytes] = []
    async with client.stream("GET", url, follow_redirects=True) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            total += len(chunk)
            if total > MAX_PDF_BYTES:
                raise httpx.HTTPError(
                    f"OA PDF at {url} exceeds {MAX_PDF_BYTES} bytes ({total}+)"
                )
            chunks.append(chunk)
    return b"".join(chunks)


async def fetch_oa(
    record: dict,
    *,
    chain: "ProviderChain | None" = None,
    client: Any = None,
    timeout: float = 60.0,
    paper_timeout: float = DEFAULT_PAPER_TIMEOUT,
) -> FetchResult:
    """Gather/rank/``%PDF``-validate OA candidates for ``record``; return bytes, a
    stub, or a transient failure (§6.4 step 3).

    OA-only and paywall-safe: a CONSTRUCTED non-allowlisted host (``is_oa_asserted``
    False) is never fetched; SSRN is link-only; an asserted-OA location is trusted.
    Each downloaded body is ``%PDF``-validated; a non-PDF/HTML body falls through to
    the next candidate. Returns a ``stub`` (``content=None``, no file, no raise) when
    no real OA PDF is available, or ``'failed'`` (D-4 — returned, never raised) when
    at least one candidate transport-errored (or ``paper_timeout`` expired) and none
    produced a PDF. ``paper_timeout`` is the SINGLE shared per-paper wall-clock
    deadline over candidate resolution plus the whole download loop (D-2, the v1
    67-minute-wedge scar — see :data:`DEFAULT_PAPER_TIMEOUT`); ``timeout`` remains
    the per-read httpx timeout for an owned client. ``client`` is an injectable
    ``httpx.AsyncClient`` (tests wire one to an ``httpx.MockTransport``); a shared
    injected client survives one paper's deadline cancellation (httpx returns or
    discards the cancelled connection).
    """
    import httpx

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    tried: list[str] = []
    winner: Optional[tuple[bytes, OACandidate]] = None
    last_landing: Optional[str] = None
    last_error: Optional[Exception] = None

    async def _resolve_and_download() -> None:
        nonlocal winner, last_landing, last_error
        candidates = await _resolve_oa_candidates(record, chain)
        for cand in candidates:
            last_landing = cand.landing_url or last_landing
            # Allowlist GATE: a CONSTRUCTED/DERIVED URL no provider vouched for as OA
            # is fetched ONLY when its host is allowlisted (no blind grabs). An
            # asserted-OA URL bypasses the gate.
            if not cand.is_oa_asserted and not _host_in_allowlist(cand.url):
                continue
            tried.append(cand.url or "")
            # A transport error (dead link / non-2xx / MAX_PDF_BYTES overflow) on a
            # HIGHER-ranked candidate must NOT abort the loop and fail a paper that
            # has a live lower-ranked copy: catch httpx.HTTPError per candidate and
            # fall through, remembering the error so an all-error paper reports
            # 'failed' (not 'stub') below. The catch is NARROW: the per-paper
            # deadline's cancellation is NOT an httpx.HTTPError, so the wedge guard
            # still trips fast on a hung host and is never swallowed here.
            try:
                data = await _download(cand.url, client)
            except httpx.HTTPError as exc:
                last_error = exc
                continue  # dead link / non-2xx / oversize -> try the next candidate
            if not _looks_like_pdf(data):
                continue  # HTML landing page / non-PDF body -> try next (clean miss)
            winner = (data, cand)
            break

    timed_out = False
    try:
        # ONE shared per-paper wall-clock deadline over candidate RESOLUTION plus
        # the WHOLE download loop (ported v1 fetch.py — plan D-2), NOT a fresh
        # timeout per candidate: httpx timeouts are per-read, not total, so a server
        # that trickles bytes (or holds a keep-alive) resets them indefinitely
        # (observed live in v1: a publisher OA endpoint wedged a whole run for
        # 67 minutes), and a per-candidate cap would multiply the wedge by
        # MAX_OA_CANDIDATES. One deadline bounds TOTAL wall-clock for the paper.
        await asyncio.wait_for(_resolve_and_download(), timeout=paper_timeout)
    except TimeoutError:  # asyncio.TimeoutError is this same class on 3.11+
        timed_out = True  # hostile/trickling host -> honest transient 'failed'
    finally:
        if owns_client:
            await client.aclose()

    if winner is None:
        if timed_out or last_error is not None:
            # TRANSIENT failure: >=1 candidate transport-errored (or the deadline
            # expired) and none produced a PDF. Returned as 'failed', never raised
            # (D-4): v2's corpus loop has no retryable-job router, and raising would
            # abort the whole pass. No bridge row is written, so a re-run retries.
            return FetchResult(
                status="failed", content=None, candidate=None,
                landing_url=last_landing, candidates_tried=tried,
            )
        # 'stub' stays reserved for the PERMANENT shapes: no candidates at all, or
        # every tried candidate returned a clean non-PDF body (landing page).
        return FetchResult(
            status="stub", content=None, candidate=None,
            landing_url=last_landing, candidates_tried=tried,
        )
    data, cand = winner
    return FetchResult(
        status="downloaded", content=data, candidate=cand,
        landing_url=cand.landing_url, candidates_tried=tried,
    )
