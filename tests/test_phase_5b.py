"""Phase 5b — Acquisition & Resolution — acceptance tests (§11).

Every plan §11 test is implemented and runs fully OFFLINE: providers are mocked
(simple in-memory doubles; OA downloads ride an ``httpx.MockTransport``), nothing
touches the real network or an LLM. The top-of-file imports double as a
whole-phase import-cleanliness check.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.dialects import sqlite as sqlite_dialect

# Import the full phase surface so collection validates import-cleanliness.
from seedgraph.acquisition.fetch import FetchResult, OACandidate, fetch_oa  # noqa: F401
from seedgraph.cache.db import init_cache_db
from seedgraph.cache.provider_cache import (  # noqa: F401
    DEFAULT_TTL_SECONDS,
    ProviderCache,
    cache_get,
    cache_is_fresh,
    cache_put,
    canonical_request_id,
    key_by_doi,
    key_by_title,
    key_oa_candidates,
    key_referenced_works,
)
from seedgraph.db.connection import open_cache_db
from seedgraph.providers import build_default_providers  # noqa: F401
from seedgraph.providers.base import CircuitBreaker, CitationProvider, ProviderChain  # noqa: F401


# --------------------------------------------------------------------------
# Offline mock provider
# --------------------------------------------------------------------------
class MockProvider:
    """In-memory provider double (no network). Keys reference/oa maps by
    ``canonical_request_id`` so they match the chain's §6.5 cache keys."""

    def __init__(
        self,
        name: str = "openalex",
        *,
        by_doi: dict | None = None,
        by_title: dict | None = None,
        refs: dict | None = None,
        oa: dict | None = None,
        fail_on: set[str] | None = None,
    ) -> None:
        self.name = name
        self.by_doi_map = by_doi or {}
        self.by_title_map = by_title or {}
        self.refs = refs or {}
        self.oa = oa or {}
        self.fail_on = set(fail_on or ())
        self.calls: list = []

    async def by_doi(self, doi):
        self.calls.append(("by_doi", doi))
        if "by_doi" in self.fail_on:
            raise RuntimeError("simulated transport error")
        return self.by_doi_map.get(doi)

    async def by_title(self, title, year=None):
        self.calls.append(("by_title", title, year))
        if "by_title" in self.fail_on:
            raise RuntimeError("simulated transport error")
        return self.by_title_map.get(title)

    async def referenced_works(self, record):
        self.calls.append(("referenced_works", canonical_request_id(record)))
        if "referenced_works" in self.fail_on:
            raise RuntimeError("simulated transport error")
        return list(self.refs.get(canonical_request_id(record), []))

    async def oa_pdf_candidates(self, record):
        self.calls.append(("oa", canonical_request_id(record)))
        if "oa" in self.fail_on:
            raise RuntimeError("simulated transport error")
        return list(self.oa.get(canonical_request_id(record), []))


def _chain(providers, **kw):
    kw.setdefault("cache_engine", None)  # None root -> resolves SEEDGRAPH_HOME (test env)
    return ProviderChain(list(providers), **kw)


# --------------------------------------------------------------------------
# test_provider_cache
# --------------------------------------------------------------------------
def test_provider_cache():
    init_cache_db(None)
    conn = open_cache_db(None)
    try:
        cache_put(conn, provider="openalex", request_key="by_doi:doi=10.x/y", response={"a": 1})
        assert cache_get(conn, provider="openalex", request_key="by_doi:doi=10.x/y") == {"a": 1}

        # cache_is_fresh honors the 30-day TTL.
        now = datetime.now(timezone.utc)
        fresh = (now - timedelta(days=1)).isoformat()
        stale = (now - timedelta(days=40)).isoformat()
        assert cache_is_fresh(fetched_at=fresh, ttl_seconds=DEFAULT_TTL_SECONDS) is True
        assert cache_is_fresh(fetched_at=stale, ttl_seconds=DEFAULT_TTL_SECONDS) is False

        # A stale row re-fetches (cache_get returns None past the TTL).
        conn.execute(
            "UPDATE provider_cache SET fetched_at = ? WHERE request_key = ?",
            (stale, "by_doi:doi=10.x/y"),
        )
        conn.commit()
        assert cache_get(conn, provider="openalex", request_key="by_doi:doi=10.x/y") is None

        # A negative (empty list) hit is cached and returned as-is (NOT None) so a
        # known miss is not re-queried within the TTL.
        cache_put(conn, provider="openalex", request_key="referenced_works:src=doi=10.z", response=[])
        assert cache_get(conn, provider="openalex", request_key="referenced_works:src=doi=10.z") == []
    finally:
        conn.close()


# --------------------------------------------------------------------------
# test_request_key_contract (must-fix #2)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# test_provider_chain
# --------------------------------------------------------------------------
def test_provider_chain():
    # (a) Ordered fallback skips a breaker-open provider.
    pa = MockProvider("pa", by_doi={"10.1/x": {"doi": "10.1/x", "src": "a"}})
    pb = MockProvider("pb", by_doi={"10.1/x": {"doi": "10.1/x", "src": "b"}})
    chain = _chain([pa, pb], fail_max=1, use_cache=False)
    chain.breaker("pa").record_failure()  # fail_max=1 -> opens immediately
    assert chain.breaker("pa").is_open()
    out = asyncio.run(chain.by_doi("10.1/x"))
    assert out["src"] == "b"
    assert pa.calls == []  # pa was skipped (breaker open)

    # (b) A transport error trips the breaker and falls through to the next provider.
    pa2 = MockProvider("pa", fail_on={"by_doi"})
    pb2 = MockProvider("pb", by_doi={"10.2/y": {"doi": "10.2/y", "src": "b"}})
    chain2 = _chain([pa2, pb2], fail_max=1, use_cache=False)
    out2 = asyncio.run(chain2.by_doi("10.2/y"))
    assert out2["src"] == "b"
    assert chain2.breaker("pa").is_open()  # the error tripped pa's breaker

    # (c) A fresh cached hit needs no network (the provider is never called).
    init_cache_db(None)
    conn = open_cache_db(None)
    try:
        cache_put(conn, provider="pa", request_key=key_by_doi("10.3/z"), response={"doi": "10.3/z", "src": "cache"})
    finally:
        conn.close()
    pa3 = MockProvider("pa", fail_on={"by_doi"})  # would RAISE if called
    chain3 = _chain([pa3])
    out3 = asyncio.run(chain3.by_doi("10.3/z"))
    assert out3["src"] == "cache"
    assert pa3.calls == []  # served from cache, no network


# --------------------------------------------------------------------------
# test_resolve_rubric
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# walk fixtures
# --------------------------------------------------------------------------
def _walk_chain():
    # Gen2: C and D are each cited TWICE (by both A and B); E once (by B only). With a
    # per_gen_cap below the candidate count the cap must keep the top-cited (C, D) and
    # prune E from the next frontier — while E still materializes as a works row.
    refs = {
        "openalex=w_s": [{"openalex_id": "W_A"}, {"openalex_id": "W_B"}],
        "openalex=w_a": [{"openalex_id": "W_C"}, {"openalex_id": "W_D"}],
        "openalex=w_b": [{"openalex_id": "W_C"}, {"openalex_id": "W_D"}, {"openalex_id": "W_E"}],
    }
    by_doi = {
        "W_A": {"openalex_id": "W_A", "doi": "10.a", "title": "Paper A", "year": 2018},
        "W_B": {"openalex_id": "W_B", "doi": "10.b", "title": "Paper B", "year": 2017},
        "W_C": {"openalex_id": "W_C", "doi": "10.c", "title": "Paper C", "year": 2016},
        "W_D": {"openalex_id": "W_D", "doi": "10.d", "title": "Paper D", "year": 2015},
        "W_E": {"openalex_id": "W_E", "doi": "10.e", "title": "Paper E", "year": 2014},
    }
    return _chain([MockProvider("openalex", refs=refs, by_doi=by_doi)])


# --------------------------------------------------------------------------
# test_walk_corpus_growth
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# test_doi_backfill
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# test_walk_identity_collapse
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# test_fetch_oa
# --------------------------------------------------------------------------
class _OAChain:
    def __init__(self, candidates):
        self._c = candidates

    async def oa_pdf_candidates(self, record):
        return list(self._c)


def _mock_client(responses: dict, requested: list):
    """``httpx.AsyncClient`` whose MockTransport serves ``responses`` (url -> bytes
    or None for 404) and records every requested URL into ``requested``."""
    import httpx

    def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requested.append(url)
        body = responses.get(url)
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(_handler))


def test_fetch_oa():
    pdf = b"%PDF-1.5 real pdf body"
    html = b"<html><body>landing page</body></html>"

    # (a) An asserted-OA candidate returning %PDF -> downloaded.
    requested: list = []
    client = _mock_client({"https://repo.example.org/a.pdf": pdf}, requested)
    chain = _OAChain([OACandidate(url="https://repo.example.org/a.pdf", host_type="repository", is_oa_asserted=True)])
    res = asyncio.run(fetch_oa({"doi": "10.1/a"}, chain=chain, client=client))
    assert res.status == "downloaded" and res.content.startswith(b"%PDF")

    # (b) A non-PDF/HTML body falls through to the next candidate.
    requested = []
    client = _mock_client(
        {"https://pub.example.org/land": html, "https://repo.example.org/b.pdf": pdf}, requested
    )
    chain = _OAChain([
        OACandidate(url="https://pub.example.org/land", is_oa_asserted=True),
        OACandidate(url="https://repo.example.org/b.pdf", host_type="repository", is_oa_asserted=True),
    ])
    res = asyncio.run(fetch_oa({"doi": "10.1/b"}, chain=chain, client=client))
    assert res.status == "downloaded" and res.content == pdf

    # (c) No OA -> stub (no file, no raise).
    requested = []
    client = _mock_client({}, requested)
    res = asyncio.run(fetch_oa({"doi": "10.1/c"}, chain=_OAChain([]), client=client))
    assert res.status == "stub" and res.content is None

    # (d) A CONSTRUCTED non-allowlisted host (is_oa_asserted=False) is NEVER fetched.
    requested = []
    client = _mock_client({"https://evil.example.com/x.pdf": pdf}, requested)
    chain = _OAChain([OACandidate(url="https://evil.example.com/x.pdf", is_oa_asserted=False)])
    res = asyncio.run(fetch_oa({"doi": "10.1/d"}, chain=chain, client=client))
    assert res.status == "stub"
    assert "https://evil.example.com/x.pdf" not in requested  # gated out, never attempted

    # (e) SSRN is link-only: a constructed SSRN candidate is never fetched -> stub.
    requested = []
    client = _mock_client({"https://papers.ssrn.com/x.pdf": pdf}, requested)
    chain = _OAChain([OACandidate(url="https://papers.ssrn.com/x.pdf", is_oa_asserted=False)])
    res = asyncio.run(fetch_oa({"ssrn_id": "123"}, chain=chain, client=client))
    assert res.status == "stub"
    assert "https://papers.ssrn.com/x.pdf" not in requested


# --------------------------------------------------------------------------
# test_acquire_bridge_single_primary (must-fix #3)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# test_manual_upload
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# test_import_markdown_and_bridge
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# test_bridge_reconcile
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# test_offline_resilience
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# test_manifest_merge (D5)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# test_acquisition_milestone (doc 01 §3 steps 3-6; no network, no LLM)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# D6 ORM-vs-migration PARITY (provider_cache + work_source_files)
# --------------------------------------------------------------------------
def _affinity(type_str: str) -> str:
    t = type_str.upper()
    if "INT" in t:
        return "INTEGER"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "TEXT"


def _assert_table_parity(insp, table_name, model):
    orm_cols = {c.name: c for c in model.__table__.columns}
    mig_cols = {c["name"]: c for c in insp.get_columns(table_name)}
    assert set(orm_cols) == set(mig_cols), f"{table_name}: column set drift"

    mig_pk = set(insp.get_pk_constraint(table_name)["constrained_columns"])
    orm_pk = {c.name for c in model.__table__.columns if c.primary_key}
    assert mig_pk == orm_pk, f"{table_name}: PK drift"

    orm_dialect = sqlite_dialect.dialect()
    for name, orm_col in orm_cols.items():
        mig_col = mig_cols[name]
        orm_aff = _affinity(str(orm_col.type.compile(dialect=orm_dialect)))
        mig_aff = _affinity(str(mig_col["type"]))
        assert orm_aff == mig_aff, f"{table_name}.{name}: type {orm_aff} != {mig_aff}"
        if name not in mig_pk:
            assert orm_col.nullable == mig_col["nullable"], f"{table_name}.{name}: nullability drift"

    mig_ix = {tuple(ix["column_names"]) for ix in insp.get_indexes(table_name)}
    orm_ix = {tuple(c.name for c in ix.columns) for ix in model.__table__.indexes}
    assert mig_ix == orm_ix, f"{table_name}: index column-sets drift ({mig_ix} != {orm_ix})"


# --------------------------------------------------------------------------
# Opt-in real-network smoke (deselected by default — `pytest -m network`).
# --------------------------------------------------------------------------
@pytest.mark.network
def test_openalex_polite_pool_smoke():
    """One real OpenAlex DOI resolution confirming the polite-pool wiring."""
    from seedgraph.providers.openalex import OpenAlexProvider

    rec = asyncio.run(OpenAlexProvider(contact_email="ci@example.com").by_doi("10.7717/peerj.4375"))
    assert rec and rec.get("doi")


# --------------------------------------------------------------------------
# contact_email polite-pool wiring (offline — construction only, no network)
# --------------------------------------------------------------------------
def test_contact_email_reaches_providers():
    """A top-level ``contact_email`` on the config must reach every polite-pool
    provider in the constructed chain; absent it, providers default to ``None``.
    Construction only (no ``.resolve()``), so nothing hits the network."""
    from seedgraph.config.models import GlobalConfig

    # (a) GlobalConfig must PRESERVE contact_email (the bug: extra="ignore" dropped it).
    cfg = GlobalConfig(contact_email="me@example.org")
    assert cfg.contact_email == "me@example.org"

    providers = build_default_providers(cfg)
    assert providers, "expected a non-empty provider chain"
    # OpenAlex / Crossref / Unpaywall / Semantic Scholar all take contact_email.
    polite = [p for p in providers if hasattr(p, "contact_email")]
    assert polite, "expected polite-pool providers exposing .contact_email"
    assert all(p.contact_email == "me@example.org" for p in polite)

    # (b) Default (no email configured) yields None on the providers.
    default_providers = build_default_providers(GlobalConfig())
    assert all(
        p.contact_email is None for p in default_providers if hasattr(p, "contact_email")
    )


# --------------------------------------------------------------------------
# Build D chunk 1 — per-provider transient retry (below the breaker; D-1)
# --------------------------------------------------------------------------
def _seq_client(responses: list, seen: list):
    """``httpx.AsyncClient`` over a MockTransport serving ``responses`` in order.

    Each entry is ``(status, json_body_or_None, headers_or_None)``; the last
    entry repeats for any further requests. Every request URL is appended to
    ``seen``. A ``"transport"`` status raises ``httpx.ConnectError`` instead.
    """
    import httpx

    def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        status, body, headers = responses[min(len(seen) - 1, len(responses) - 1)]
        if status == "transport":
            raise httpx.ConnectError("simulated transport blip", request=request)
        if body is None:
            return httpx.Response(status, headers=headers or {})
        return httpx.Response(status, json=body, headers=headers or {})

    return httpx.AsyncClient(transport=httpx.MockTransport(_handler))


def _no_sleep(monkeypatch) -> list:
    """Patch the retry backoff sleep seam; return the list of recorded delays."""
    from seedgraph.providers import _retry

    sleeps: list = []

    async def _fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(_retry, "_sleep", _fake_sleep)
    return sleeps


def test_retry_transient_429_blip_never_reaches_breaker(monkeypatch):
    """429, 429, 200 -> the provider returns the record and the chain records
    ZERO breaker failures (the blip is absorbed below the breaker)."""
    from seedgraph.providers.openalex import OpenAlexProvider

    sleeps = _no_sleep(monkeypatch)
    work = {"id": "https://openalex.org/W1", "title": "Paper X", "publication_year": 2020}
    seen: list = []
    client = _seq_client([(429, None, None), (429, None, None), (200, work, None)], seen)
    chain = _chain([OpenAlexProvider(client=client)], use_cache=False)

    out = asyncio.run(chain.by_doi("10.1/x"))

    assert out and out["title"] == "Paper X"
    assert len(seen) == 3  # two transient hits + the success
    assert chain.breaker("openalex")._failures == 0
    assert not chain.breaker("openalex").is_open()
    assert len(sleeps) == 2  # backed off between attempts


def test_retry_never_retries_404(monkeypatch):
    """A 404 is a plain miss: exactly ONE request, no sleep, ``None`` returned."""
    from seedgraph.providers.openalex import OpenAlexProvider

    sleeps = _no_sleep(monkeypatch)
    seen: list = []
    client = _seq_client([(404, None, None)], seen)

    out = asyncio.run(OpenAlexProvider(client=client).by_doi("10.1/miss"))

    assert out is None  # a plain miss stays a miss
    assert len(seen) == 1  # never retried
    assert sleeps == []


def test_retry_exhaustion_raises_and_breaker_counts_one(monkeypatch):
    """Persistent 503: the helper raises after 3 attempts; the chain records ONE
    breaker failure per exhausted call and opens only after fail_max (=3) such
    calls — offline degradation preserved."""
    from seedgraph.providers._retry import TransientHTTPStatus
    from seedgraph.providers.openalex import OpenAlexProvider

    _no_sleep(monkeypatch)
    seen: list = []
    client = _seq_client([(503, None, None)], seen)
    provider = OpenAlexProvider(client=client)

    # Direct provider call: re-raises the marker after exactly 3 attempts.
    with pytest.raises(TransientHTTPStatus):
        asyncio.run(provider.by_doi("10.1/x"))
    assert len(seen) == 3

    # Through the chain: one exhausted call == one breaker failure.
    seen.clear()
    chain = _chain([provider], use_cache=False)
    for expected_failures in (1, 2):
        assert asyncio.run(chain.by_doi("10.1/x")) is None
        assert chain.breaker("openalex")._failures == expected_failures
        assert not chain.breaker("openalex").is_open()
    # Third exhausted call trips the breaker; the sole-provider chain re-raises.
    with pytest.raises(TransientHTTPStatus):
        asyncio.run(chain.by_doi("10.1/x"))
    assert chain.breaker("openalex").is_open()
    assert len(seen) == 9  # 3 chain calls x 3 attempts each


def test_retry_after_header_honored(monkeypatch):
    """A numeric ``Retry-After`` replaces the computed backoff verbatim."""
    from seedgraph.providers.openalex import OpenAlexProvider

    sleeps = _no_sleep(monkeypatch)
    work = {"id": "https://openalex.org/W2", "title": "Paper Y"}
    seen: list = []
    client = _seq_client([(429, None, {"Retry-After": "0"}), (200, work, None)], seen)

    out = asyncio.run(OpenAlexProvider(client=client).by_doi("10.1/y"))

    assert out and out["title"] == "Paper Y"
    assert sleeps == [0.0]  # header value used, not the exponential backoff
    assert len(seen) == 2


def test_retry_transport_error(monkeypatch):
    """An httpx transport blip (ConnectError) is retried like a transient status."""
    from seedgraph.providers.openalex import OpenAlexProvider

    _no_sleep(monkeypatch)
    work = {"id": "https://openalex.org/W3", "title": "Paper Z"}
    seen: list = []
    client = _seq_client([("transport", None, None), (200, work, None)], seen)

    out = asyncio.run(OpenAlexProvider(client=client).by_doi("10.1/z"))

    assert out and out["title"] == "Paper Z"
    assert len(seen) == 2


def test_retry_wired_into_all_providers(monkeypatch):
    """Every provider HTTP helper rides ``retry_transient``: a single 429 blip
    followed by a 200 yields the record with exactly two requests."""
    from seedgraph.providers.core import CoreProvider
    from seedgraph.providers.crossref import CrossrefProvider
    from seedgraph.providers.openalex import OpenAlexProvider
    from seedgraph.providers.semantic_scholar import SemanticScholarProvider
    from seedgraph.providers.unpaywall import UnpaywallProvider

    _no_sleep(monkeypatch)
    cases = [
        (lambda c: OpenAlexProvider(client=c).by_doi("10.1/x"),
         {"id": "https://openalex.org/W1", "title": "T"}),
        (lambda c: CrossrefProvider(client=c).by_doi("10.1/x"),
         {"message": {"DOI": "10.1/x", "title": ["T"]}}),
        (lambda c: SemanticScholarProvider(client=c).by_doi("10.1/x"),
         {"externalIds": {"DOI": "10.1/x"}, "title": "T"}),
        (lambda c: UnpaywallProvider(contact_email="e@x.org", client=c).by_doi("10.1/x"),
         {"doi": "10.1/x", "title": "T"}),
        (lambda c: CoreProvider(core_key="k", client=c).by_title("T"),
         {"results": [{"title": "T", "yearPublished": 2020}]}),
    ]
    for call, body in cases:
        seen: list = []
        client = _seq_client([(429, None, None), (200, body, None)], seen)
        rec = asyncio.run(call(client))
        assert rec and rec.get("title") == "T"
        assert len(seen) == 2


# --------------------------------------------------------------------------
# Build D chunk 2 — per-paper deadline, size cap, failed outcome (D-2/D-3/D-4)
# --------------------------------------------------------------------------
def _status_client(routes: dict, requested: list):
    """``httpx.AsyncClient`` over a MockTransport mapping url -> ``(status, body)``
    (unknown urls 404). Every requested URL is appended to ``requested``."""
    import httpx

    def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requested.append(url)
        status, body = routes.get(url, (404, b""))
        return httpx.Response(status, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(_handler))


def test_fetch_503_fall_through_and_all_503_failed():
    """A 503 on a higher-ranked candidate falls through to a live lower-ranked copy
    (fall-through preserved); when EVERY tried candidate transport-errors the
    result is the honest transient ``'failed'`` — never a ``'stub'`` mislabel; a
    clean non-PDF 200 on every candidate stays a permanent ``'stub'``."""
    pdf = b"%PDF-1.5 lower ranked copy"
    chain = _OAChain([
        OACandidate(url="https://a.example.org/x.pdf", host_type="repository", is_oa_asserted=True),
        OACandidate(url="https://b.example.org/x.pdf", is_oa_asserted=True),
    ])

    # (a) 503 then a live PDF -> downloaded (fall-through preserved).
    requested: list = []
    client = _status_client(
        {"https://a.example.org/x.pdf": (503, b""), "https://b.example.org/x.pdf": (200, pdf)},
        requested,
    )
    res = asyncio.run(fetch_oa({"doi": "10.1/ft"}, chain=chain, client=client))
    assert res.status == "downloaded" and res.content == pdf
    assert requested == ["https://a.example.org/x.pdf", "https://b.example.org/x.pdf"]

    # (b) all-503 -> transient 'failed' (>=1 transport error, no winner).
    requested = []
    client = _status_client(
        {"https://a.example.org/x.pdf": (503, b""), "https://b.example.org/x.pdf": (503, b"")},
        requested,
    )
    res = asyncio.run(fetch_oa({"doi": "10.1/all503"}, chain=chain, client=client))
    assert res.status == "failed" and res.content is None
    assert len(requested) == 2  # both candidates were tried before failing

    # (c) all-clean-non-PDF 200s -> still a permanent 'stub' (no transport error).
    requested = []
    html = b"<html>landing page</html>"
    client = _status_client(
        {"https://a.example.org/x.pdf": (200, html), "https://b.example.org/x.pdf": (200, html)},
        requested,
    )
    res = asyncio.run(fetch_oa({"doi": "10.1/clean"}, chain=chain, client=client))
    assert res.status == "stub"


def test_fetch_oversized_body_skips_candidate(monkeypatch):
    """An oversized body aborts THAT candidate against the incremental byte cap
    (D-3) and falls through; an all-oversized paper routes transient 'failed'
    (v1 grouped the overflow raise with the transport errors)."""
    from seedgraph.acquisition import fetch as fetch_mod

    monkeypatch.setattr(fetch_mod, "MAX_PDF_BYTES", 64)
    big = b"%PDF-1.5 " + b"x" * 200  # 209 bytes > the 64-byte test cap
    small = b"%PDF-1.5 small ok"

    requested: list = []
    client = _status_client(
        {"https://a.example.org/big.pdf": (200, big), "https://b.example.org/ok.pdf": (200, small)},
        requested,
    )
    chain = _OAChain([
        OACandidate(url="https://a.example.org/big.pdf", host_type="repository", is_oa_asserted=True),
        OACandidate(url="https://b.example.org/ok.pdf", is_oa_asserted=True),
    ])
    res = asyncio.run(fetch_oa({"doi": "10.1/big"}, chain=chain, client=client))
    assert res.status == "downloaded" and res.content == small  # oversize skipped

    # Only the oversized candidate -> 'failed' (the copy exists; the guard refused it).
    client = _status_client({"https://a.example.org/big.pdf": (200, big)}, [])
    chain = _OAChain([OACandidate(url="https://a.example.org/big.pdf", is_oa_asserted=True)])
    res = asyncio.run(fetch_oa({"doi": "10.1/big2"}, chain=chain, client=client))
    assert res.status == "failed"


def test_fetch_hostile_stream_paper_deadline_failed_fast():
    """A never-completing stream (trickle-byte/keep-alive host) is bounded by the
    SINGLE per-paper wall-clock deadline (D-2, the v1 67-minute-wedge scar) and
    reports transient 'failed' fast — httpx per-read timeouts alone never fire."""
    import time

    import httpx

    async def _handler(request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()  # hostile host: never completes
        raise AssertionError("unreachable")

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    chain = _OAChain([OACandidate(url="https://hostile.example.org/x.pdf", is_oa_asserted=True)])

    t0 = time.monotonic()
    res = asyncio.run(
        fetch_oa({"doi": "10.1/hang"}, chain=chain, client=client, paper_timeout=0.1)
    )
    elapsed = time.monotonic() - t0
    assert res.status == "failed"
    assert elapsed < 5.0, f"per-paper deadline must trip fast; took {elapsed:.1f}s"


def test_injected_client_survives_paper_timeout():
    """The D-2 cancellation risk, pinned by name: one paper hitting the per-paper
    ``wait_for`` deadline must not poison a SHARED injected ``AsyncClient`` — the
    next paper's download on the same client (the ``_acquire_async`` shape) still
    succeeds cleanly."""
    import httpx

    pdf = b"%PDF-1.5 next paper body"
    requested: list = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requested.append(url)
        if "hostile" in url:
            await asyncio.Event().wait()  # first paper wedges forever
        return httpx.Response(200, content=pdf)

    client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    hostile = _OAChain([OACandidate(url="https://hostile.example.org/a.pdf", is_oa_asserted=True)])
    healthy = _OAChain([OACandidate(url="https://repo.example.org/b.pdf", is_oa_asserted=True)])

    async def _two_papers():
        first = await fetch_oa(
            {"doi": "10.1/wedge"}, chain=hostile, client=client, paper_timeout=0.1
        )
        second = await fetch_oa(
            {"doi": "10.1/ok"}, chain=healthy, client=client, paper_timeout=30.0
        )
        return first, second

    first, second = asyncio.run(_two_papers())
    assert first.status == "failed"
    assert second.status == "downloaded" and second.content == pdf
    assert "https://repo.example.org/b.pdf" in requested


# --------------------------------------------------------------------------
# Build D chunk 3 — landing->PDF derivers (bioRxiv/medRxiv, OSF, NBER)
# --------------------------------------------------------------------------
from seedgraph.acquisition.fetch import (  # noqa: E402  (grouped with the chunk-3 tests)
    _derive_biorxiv_pdf,
    _derive_candidates,
    _derive_nber_pdf,
    _derive_osf_pdf,
    _resolve_oa_candidates,
)

_BIORXIV_LANDING = "https://www.biorxiv.org/content/10.1101/833400v1"
_BIORXIV_PDF = "https://www.biorxiv.org/content/10.1101/833400v1.full.pdf"
_NBER_PDF = "https://www.nber.org/system/files/working_papers/w12345/w12345.pdf"


def test_derive_biorxiv_landing_to_full_pdf():
    """A versioned bioRxiv/medRxiv landing -> ``.full.pdf``; versionless refused
    (critic G3: versionless often serves HTML); non-matching hosts -> None."""
    assert _derive_biorxiv_pdf(_BIORXIV_LANDING) == _BIORXIV_PDF
    assert (
        _derive_biorxiv_pdf("https://www.medrxiv.org/content/10.1101/2020.01.01v2")
        == "https://www.medrxiv.org/content/10.1101/2020.01.01v2.full.pdf"
    )
    # No version segment -> no derivation (versionless often serves HTML).
    assert _derive_biorxiv_pdf("https://www.biorxiv.org/content/10.1101/833400") is None
    # Not a biorxiv/medrxiv host -> None (incl. SSRN: link-only, never derived).
    assert _derive_biorxiv_pdf("https://example.org/10.1101/833400v1") is None
    assert _derive_biorxiv_pdf("https://papers.ssrn.com/sol3/abc") is None
    assert _derive_biorxiv_pdf(None) is None


def test_derive_osf_landing_to_download():
    """An OSF ``osf.io/{guid}`` landing -> ``/download`` (whole OSF family, G5);
    deep paths refused; non-OSF hosts -> None."""
    assert _derive_osf_pdf("https://osf.io/ab12c") == "https://osf.io/ab12c/download"
    assert _derive_osf_pdf("https://osf.io/ab12c/") == "https://osf.io/ab12c/download"
    # The OSF sub-domain family (PsyArXiv/SocArXiv/...) rides osf.io.
    assert _derive_osf_pdf("https://psyarxiv.osf.io/xy9z") == (
        "https://psyarxiv.osf.io/xy9z/download"
    )
    # Deep path refused (critic G5: only a single guid segment is a preprint page).
    assert _derive_osf_pdf("https://osf.io/ab12c/files/x") is None
    # An already-/download path -> None (no /download/download doubling).
    assert _derive_osf_pdf("https://osf.io/ab12c/download") is None
    # Not an OSF host -> None (incl. SSRN).
    assert _derive_osf_pdf("https://example.org/ab12c") is None
    assert _derive_osf_pdf("https://papers.ssrn.com/sol3/abc") is None
    assert _derive_osf_pdf(None) is None


def test_derive_nber_doi_to_pdf():
    """An NBER ``10.3386/w{n}`` DOI -> the working-paper PDF URL (the one DOI
    constructor); any other DOI (or no DOI) is never constructed."""
    assert _derive_nber_pdf({"doi": "10.3386/w12345"}) == _NBER_PDF
    assert _derive_nber_pdf({"doi": "10.1371/journal.pone.0000308"}) is None
    assert _derive_nber_pdf({"title": "no doi"}) is None


def test_derive_candidates_emit_shape_and_dedupe_against_real_pdf_url():
    """``_derive_candidates`` emits constructed (``is_oa_asserted=False``) tier-1
    candidates; hooked BEFORE the dedupe in ``_resolve_oa_candidates``, a derived
    URL duplicating a real asserted ``pdf_url`` dedupes away (the real copy wins)."""
    landing_only = OACandidate(
        url=None, landing_url=_BIORXIV_LANDING, host_type="repository",
        provider="openalex", is_oa_asserted=True,
    )
    derived = _derive_candidates({"doi": "10.1101/833400"}, [landing_only])
    assert [(c.url, c.provider, c.is_oa_asserted) for c in derived] == [
        (_BIORXIV_PDF, "biorxiv_derived", False)
    ]

    # End-to-end through _resolve_oa_candidates: the provider ALSO asserted the
    # same URL as a real pdf_url -> the derived duplicate dedupes away and the
    # surviving candidate is the asserted-OA one.
    asserted = OACandidate(
        url=_BIORXIV_PDF, landing_url=_BIORXIV_LANDING, host_type="repository",
        provider="unpaywall", is_oa_asserted=True,
    )
    chain = _OAChain([asserted])
    resolved = asyncio.run(_resolve_oa_candidates({"doi": "10.1101/833400"}, chain))
    assert [c.url for c in resolved] == [_BIORXIV_PDF]
    assert resolved[0].is_oa_asserted is True and resolved[0].provider == "unpaywall"


def test_landing_only_biorxiv_derived_and_fetched():
    """End-to-end recovery: a record whose ONLY candidate is a versioned bioRxiv
    landing page (no pdf_url) -> a derived ``.full.pdf`` is constructed, passes
    the allowlist gate (biorxiv.org is allowlisted), and downloads under
    MockTransport — the is_oa_asserted=False gate is now non-vestigial."""
    pdf = b"%PDF-1.5 biorxiv derived body"
    requested: list = []
    client = _mock_client({_BIORXIV_PDF: pdf}, requested)
    chain = _OAChain([
        OACandidate(url=None, landing_url=_BIORXIV_LANDING, host_type="repository",
                    provider="openalex", is_oa_asserted=True),
    ])
    res = asyncio.run(fetch_oa({"doi": "10.1101/833400"}, chain=chain, client=client))
    assert res.status == "downloaded" and res.content == pdf
    assert res.candidate.provider == "biorxiv_derived"
    assert res.candidate.is_oa_asserted is False
    assert res.landing_url == _BIORXIV_LANDING
    assert requested == [_BIORXIV_PDF]  # only the derived URL was fetched


def test_derived_candidate_rides_allowlist_gate(monkeypatch):
    """Derived candidates pass through the SAME fetch-time host gate as any other
    constructed URL: with nber.org allowlisted the derived NBER PDF downloads;
    with it removed from the allowlist the derived candidate is NEVER requested."""
    from seedgraph.acquisition import fetch as fetch_mod

    pdf = b"%PDF-1.5 nber wp body"
    record = {"doi": "10.3386/w12345"}
    chain = _OAChain([])  # no provider candidates at all; only the DOI constructor

    # (a) Default allowlist: nber.org is allowlisted -> derived candidate fetched.
    requested: list = []
    client = _mock_client({_NBER_PDF: pdf}, requested)
    res = asyncio.run(fetch_oa(record, chain=chain, client=client))
    assert res.status == "downloaded" and res.content == pdf
    assert res.candidate.provider == "nber_derived"
    assert requested == [_NBER_PDF]

    # (b) Host NOT allowlisted -> the derived (is_oa_asserted=False) candidate is
    # gated out, never requested; the paper stubs (no transport call at all).
    monkeypatch.setattr(fetch_mod, "_PREPRINT_HOSTS", frozenset({"arxiv.org"}))
    requested = []
    client = _mock_client({_NBER_PDF: pdf}, requested)
    res = asyncio.run(fetch_oa(record, chain=chain, client=client))
    assert res.status == "stub"
    assert requested == []  # the gate prevented any download
    assert res.candidates_tried == []


# --------------------------------------------------------------------------
# Build D chunk 4 — OpenAlex arXiv resolution branch (minted-DOI endpoint)
# --------------------------------------------------------------------------
def _oa_json_client(routes: dict, seen: list):
    """``httpx.AsyncClient`` over a MockTransport mapping exact url -> JSON body
    (unknown urls 404). Every requested URL is appended to ``seen``."""
    import httpx

    def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen.append(url)
        body = routes.get(url)
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, json=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(_handler))


_ARXIV_MINTED_ENDPOINT = "https://api.openalex.org/works/doi:10.48550/arXiv.2401.01234"
_ARXIV_WORK = {
    "id": "https://openalex.org/W4400000000",
    "doi": "https://doi.org/10.48550/arxiv.2401.01234",
    "title": "An arXiv-Only Paper",
    "publication_year": 2024,
    "authorships": [{"author": {"display_name": "A. Author"}}],
    "referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"],
}


def test_by_doi_bare_arxiv_id_resolves_via_minted_doi():
    """A bare ``NNNN.NNNNN`` arXiv id resolves via ``/works/doi:10.48550/arXiv.{id}``
    and returns a shaped record — no longer a dead-end None."""
    from seedgraph.providers.openalex import OpenAlexProvider

    seen: list = []
    client = _oa_json_client({_ARXIV_MINTED_ENDPOINT: _ARXIV_WORK}, seen)

    rec = asyncio.run(OpenAlexProvider(client=client).by_doi("2401.01234"))

    assert rec == {
        "openalex_id": "W4400000000",
        "doi": "10.48550/arxiv.2401.01234",
        "title": "An arXiv-Only Paper",
        "year": 2024,
        "authors": ["A. Author"],
    }
    assert seen == [_ARXIV_MINTED_ENDPOINT]  # exactly one call, the minted-DOI route


def test_by_doi_arxiv_surface_forms_normalize_to_one_endpoint():
    """``arXiv:`` label, version suffix, abs-URL, and the arXiv-minted DOI surface
    form all normalize to the SAME minted-DOI endpoint (raw-shape detection —
    tolerates either land order with Build A's identity-layer fold)."""
    from seedgraph.providers.openalex import OpenAlexProvider

    forms = [
        "arXiv:2401.01234",
        "2401.01234v3",
        "arxiv.org/abs/2401.01234",
        "10.48550/arXiv.2401.01234",
        "10.48550/arxiv.2401.01234v2",
    ]
    for form in forms:
        seen: list = []
        client = _oa_json_client({_ARXIV_MINTED_ENDPOINT: _ARXIV_WORK}, seen)
        rec = asyncio.run(OpenAlexProvider(client=client).by_doi(form))
        assert rec and rec["openalex_id"] == "W4400000000", form
        assert seen == [_ARXIV_MINTED_ENDPOINT], form


def test_referenced_works_from_arxiv_only_record():
    """An arXiv-only record (``{"arxiv_id": …}``) is reference-expanded: the walk
    gets the refs list instead of ``[]`` from arXiv seeds."""
    from seedgraph.providers.openalex import OpenAlexProvider

    seen: list = []
    client = _oa_json_client({_ARXIV_MINTED_ENDPOINT: _ARXIV_WORK}, seen)

    refs = asyncio.run(
        OpenAlexProvider(client=client).referenced_works({"arxiv_id": "2401.01234"})
    )

    assert refs == [{"openalex_id": "W1"}, {"openalex_id": "W2"}]
    assert seen == [_ARXIV_MINTED_ENDPOINT]


def test_by_doi_arxiv_miss_is_clean_404_none():
    """A miss on the minted-DOI endpoint is a clean 404 -> ``None`` (never retried,
    no fall-through to a bogus plain-DOI query)."""
    from seedgraph.providers.openalex import OpenAlexProvider

    seen: list = []
    client = _oa_json_client({}, seen)

    out = asyncio.run(OpenAlexProvider(client=client).by_doi("2401.99999"))

    assert out is None
    assert seen == ["https://api.openalex.org/works/doi:10.48550/arXiv.2401.99999"]


def test_plain_doi_still_routes_through_doi_endpoint_unchanged():
    """Plain DOIs — including arXiv-adjacent ones that merely contain 'arxiv' —
    keep today's ``/works/doi:{doi}`` routing, in both by_doi and _fetch_work."""
    from seedgraph.providers.openalex import OpenAlexProvider

    plain = {
        "id": "https://openalex.org/W9",
        "title": "Plain DOI Paper",
        "publication_year": 2007,
        "referenced_works": ["https://openalex.org/W7"],
    }
    for doi in ("10.1371/journal.pone.0000308", "10.1016/j.arxiv-adjacent.001"):
        endpoint = f"https://api.openalex.org/works/doi:{doi}"
        seen: list = []
        client = _oa_json_client({endpoint: plain}, seen)
        rec = asyncio.run(OpenAlexProvider(client=client).by_doi(doi))
        assert rec and rec["openalex_id"] == "W9", doi
        assert seen == [endpoint], doi  # never the 10.48550 minted route

    # _fetch_work (the referenced_works spine) keeps DOI-first routing too.
    doi = "10.1371/journal.pone.0000308"
    endpoint = f"https://api.openalex.org/works/doi:{doi}"
    seen = []
    client = _oa_json_client({endpoint: plain}, seen)
    refs = asyncio.run(
        OpenAlexProvider(client=client).referenced_works({"doi": doi, "arxiv_id": "2401.01234"})
    )
    assert refs == [{"openalex_id": "W7"}]
    assert seen == [endpoint]  # the real DOI wins over the arxiv_id sibling


# --------------------------------------------------------------------------
# Build D chunk 5 — budgeted, value-ordered acquire pass (D-5 / D-6)
# --------------------------------------------------------------------------
_BUDGET_URLS = {
    "seed": "https://arxiv.org/pdf/2101.00001",
    "low": "https://arxiv.org/pdf/2101.00002",
    "high": "https://arxiv.org/pdf/2101.00003",
}
_BUDGET_BODIES = {
    _BUDGET_URLS["seed"]: b"%PDF-1.5 seed body",
    _BUDGET_URLS["low"]: b"%PDF-1.5 low-count body",
    _BUDGET_URLS["high"]: b"%PDF-1.5 high-count body",
}


# --------------------------------------------------------------------------
# Build D chunk 6 — per-paper acquisition progress events (D-7)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Build F chunk 10 — resolve-loop progress rider + invariant re-verification
# (the piece Build D chunk 6 left untouched: the per-item step in resolve_corpus,
#  built on D's exact Progress/emit hook signature. F must ACTIVELY re-verify the
#  event-silent CLI invariant risk 7's crash-sweep + chunk 9's interrupted-sweep
#  both rest on — never assume it.)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Build D chunk 7 — batched W-id fetch: by_openalex_ids (D-8)
# --------------------------------------------------------------------------
def _batch_client(works_by_id: dict, filters_seen: list):
    """``httpx.AsyncClient`` over a MockTransport serving the batched
    ``/works?filter=ids.openalex:…`` endpoint. Each HTTP call appends its
    requested W-id list to ``filters_seen`` (so ``len(filters_seen)`` counts
    calls) and returns the matching work objects. Pins the request shape:
    ``per-page=200`` and the widened select carrying ``doi`` (r2-1 §6)."""
    import httpx

    def _handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        flt = params.get("filter") or ""
        assert flt.startswith("ids.openalex:"), flt
        assert params.get("per-page") == "200"
        assert params.get("select") == "id,doi,title,publication_year,authorships"
        wids = flt[len("ids.openalex:"):].split("|")
        filters_seen.append(wids)
        results = [works_by_id[w] for w in wids if w in works_by_id]
        return httpx.Response(200, json={"results": results})

    return httpx.AsyncClient(transport=httpx.MockTransport(_handler))


def test_by_openalex_ids_ceil_batches_105_ids():
    """105 W-ids chunked ≤50/batch cost exactly ceil(105/50)=3 HTTP calls (not
    105 — mirrors v1 test_title_backfill.py); each invocation is ONE request,
    batches cover every id exactly once, and every record comes back shaped
    with the widened select's ``doi`` (the r2-1 §6 per-ref DOI backfill)."""
    import math

    from seedgraph.providers.openalex import OpenAlexProvider

    wids = [f"W{1000 + i:07d}" for i in range(105)]
    works = {
        w: {
            "id": f"https://openalex.org/{w}",
            "doi": f"https://doi.org/10.99/{w.lower()}",
            "title": f"Title for {w}",
            "publication_year": 2020,
            "authorships": [{"author": {"display_name": "A. Author"}}],
        }
        for w in wids
    }
    filters_seen: list = []
    provider = OpenAlexProvider(client=_batch_client(works, filters_seen))

    async def _run() -> list:
        out: list = []
        for start in range(0, len(wids), 50):
            out.extend(await provider.by_openalex_ids(wids[start:start + 50]))
        return out

    records = asyncio.run(_run())

    assert len(filters_seen) == math.ceil(105 / 50) == 3  # ceil(N/50) calls, not N
    assert all(len(batch) <= 50 for batch in filters_seen)
    assert sorted(w for batch in filters_seen for w in batch) == sorted(wids)
    assert len(records) == 105
    by_id = {r["openalex_id"]: r for r in records}
    assert by_id["W0001000"] == {
        "openalex_id": "W0001000",
        "doi": "10.99/w0001000",  # doi carried through the widened select
        "title": "Title for W0001000",
        "year": 2020,
        "authors": ["A. Author"],
    }


def test_by_openalex_ids_normalizes_dedupes_and_empty_is_no_network():
    """Surface forms (URL prefix, lowercase ``w``) normalize and dedupe into one
    filter id each; non-W garbage is dropped; empty/blank input returns ``[]``
    with ZERO network calls."""
    from seedgraph.providers.openalex import OpenAlexProvider

    works = {
        "W1": {"id": "https://openalex.org/W1", "title": "One"},
        "W2": {"id": "https://openalex.org/W2", "title": "Two"},
    }
    filters_seen: list = []
    provider = OpenAlexProvider(client=_batch_client(works, filters_seen))

    recs = asyncio.run(
        provider.by_openalex_ids(
            ["https://openalex.org/W1", "w1", "W1", "W2", "10.1/not-a-wid", ""]
        )
    )
    assert filters_seen == [["W1", "W2"]]  # deduped, normalized, order preserved
    assert [r["openalex_id"] for r in recs] == ["W1", "W2"]

    # Empty / garbage-only input: [] without touching the transport.
    filters_seen.clear()
    assert asyncio.run(provider.by_openalex_ids([])) == []
    assert asyncio.run(provider.by_openalex_ids(["", "  ", "10.1/x"])) == []
    assert filters_seen == []


def test_key_by_openalex_ids_deterministic_order_insensitive():
    """The additive key builder is deterministic and order-insensitive: surface
    forms normalize, duplicates collapse, ids sort — one id set, one key. The
    namespace is distinct from ``by_doi:`` (the D-8 starvation guard is
    structural), and the pinned §6.5 grammar gains only this builder."""
    from seedgraph.cache.provider_cache import key_by_openalex_ids

    k1 = key_by_openalex_ids(["W2", "W1"])
    k2 = key_by_openalex_ids(["w1", "https://openalex.org/W2", "W1"])
    assert k1 == k2 == "by_openalex_ids:ids=w1|w2"
    assert key_by_openalex_ids([]) == "by_openalex_ids:ids="
    assert key_by_openalex_ids(["10.1/not-a-wid"]) == "by_openalex_ids:ids="
    assert not k1.startswith("by_doi:")  # its own namespace, never by_doi's


class _BatchChainProvider:
    """Chain-level double exposing only the batched verb; counts invocations."""

    name = "openalex"

    def __init__(self, records: list) -> None:
        self.records = records
        self.calls = 0

    async def by_openalex_ids(self, ids):
        self.calls += 1
        return list(self.records)


def test_chain_by_openalex_ids_empty_not_cached_truthy_cached():
    """Chain dispatch for the batched verb (D-8): a TRUTHY batch result is
    cached (second call = cache hit, zero provider calls — behavior identical
    to the stock path); an EMPTY batch result is NEVER written to
    ``provider_cache``, so the next pass asks again instead of starving on a
    30-day negative; and a pre-seeded stale ``by_doi`` NEGATIVE for the same
    work cannot block the batch (distinct namespace)."""
    init_cache_db(None)
    from seedgraph.cache.provider_cache import key_by_openalex_ids

    # Pre-poison the cache exactly the way a live corpus does: the 30-day
    # negative by_doi entry that pinned the untitled state in the first place.
    conn = open_cache_db(None)
    try:
        cache_put(conn, provider="openalex", request_key=key_by_doi("10.1/starved"), response=[])
    finally:
        conn.close()

    # (a) Empty batch result -> [] returned, NOTHING cached, provider re-asked.
    empty = _BatchChainProvider([])
    chain = _chain([empty])
    assert asyncio.run(chain.by_openalex_ids(["W1", "W2"])) == []
    assert asyncio.run(chain.by_openalex_ids(["W2", "W1"])) == []
    assert empty.calls == 2  # not starved by a cached negative
    assert chain.breaker("openalex")._failures == 0  # an empty batch is not a failure
    conn = open_cache_db(None)
    try:
        assert cache_get(
            conn, provider="openalex", request_key=key_by_openalex_ids(["W1", "W2"])
        ) is None
        count = conn.execute(
            "SELECT COUNT(*) FROM provider_cache WHERE request_key LIKE 'by_openalex_ids:%'"
        ).fetchone()[0]
        assert count == 0  # no row at all — not even a stale one
    finally:
        conn.close()

    # (b) Truthy batch result -> cached; a reordered/surface-form re-ask hits
    # the SAME key and never reaches the provider (cache-hit behavior identical).
    rec = {"openalex_id": "W1", "doi": "10.1/starved", "title": "Recovered Title"}
    full = _BatchChainProvider([rec])
    chain2 = _chain([full])
    assert asyncio.run(chain2.by_openalex_ids(["W1"])) == [rec]
    assert asyncio.run(chain2.by_openalex_ids(["https://openalex.org/W1", "w1"])) == [rec]
    assert full.calls == 1
    assert chain2.breaker("openalex")._failures == 0


def test_chain_by_openalex_ids_empty_input_and_breaker():
    """Empty input short-circuits at the chain (no dispatch, no provider call);
    a raising provider records ONE breaker failure and the chain returns ``[]``
    — breaker behavior identical to the stock path."""
    provider = _BatchChainProvider([{"openalex_id": "W1", "title": "T"}])
    chain = _chain([provider], use_cache=False)
    assert asyncio.run(chain.by_openalex_ids([])) == []
    assert provider.calls == 0

    class _Boom:
        name = "openalex"

        async def by_openalex_ids(self, ids):
            raise RuntimeError("simulated whole-batch outage")

    chain2 = _chain([_Boom()], use_cache=False)
    assert asyncio.run(chain2.by_openalex_ids(["W1"])) == []
    assert chain2.breaker("openalex")._failures == 1
    assert not chain2.breaker("openalex").is_open()


# --------------------------------------------------------------------------
# Walk-side batched enrichment + residue counters (Build D ch8)
# --------------------------------------------------------------------------
class _EnrichChain:
    """Chain-level double for the ch8 walk-side batched enrichment: serves ONE
    reference list for any source, answers the batched verb from ``batch``
    (ids absent from it simply do not appear — the real OpenAlex shape), and
    answers the per-ref fallback from ``by_doi`` — counting both verbs."""

    def __init__(
        self,
        refs: list,
        *,
        batch: list | None = None,
        by_doi: dict | None = None,
        by_doi_raises: bool = False,
    ) -> None:
        self._refs = list(refs)
        self._batch = {r["openalex_id"]: r for r in (batch or [])}
        self._by_doi = dict(by_doi or {})
        self._by_doi_raises = by_doi_raises
        self.batch_calls: list[list[str]] = []
        self.by_doi_calls: list = []

    async def referenced_works(self, record):
        return [dict(r) for r in self._refs]

    async def by_openalex_ids(self, ids):
        self.batch_calls.append(list(ids))
        return [dict(self._batch[i]) for i in ids if i in self._batch]

    async def by_doi(self, wid):
        self.by_doi_calls.append(wid)
        if self._by_doi_raises:
            raise RuntimeError("simulated transport error")
        return self._by_doi.get(wid)


_ENRICH_RECS = {
    "W_A": {"openalex_id": "W_A", "doi": "10.a", "title": "Paper A", "year": 2018},
    "W_B": {"openalex_id": "W_B", "doi": "10.b", "title": "Paper B", "year": 2017},
    "W_C": {"openalex_id": "W_C", "doi": "10.c", "title": "Paper C", "year": 2016},
}
_ENRICH_REFS = [{"openalex_id": "W_A"}, {"openalex_id": "W_B"}, {"openalex_id": "W_C"}]


# --------------------------------------------------------------------------
# Build D chunk 10 — abstract + oa_status shaping, works columns, triage state
# --------------------------------------------------------------------------
def test_abstract_inverted_index_reconstruction():
    """Position-sort rehydration of OpenAlex's ``abstract_inverted_index``
    (ported v1); malformed/absent input -> ``None``, never a raise."""
    from seedgraph.providers.openalex import _abstract_from_inverted_index as rehydrate

    assert rehydrate({"world": [1], "Hello": [0]}) == "Hello world"
    # interleaved repeated words re-order by position, not by key.
    assert rehydrate({"b": [1, 3], "a": [0, 2]}) == "a b a b"
    # malformed -> None: absent, empty, non-dict, positions not a list, no int positions.
    assert rehydrate(None) is None
    assert rehydrate({}) is None
    assert rehydrate("not a dict") is None
    assert rehydrate(["not", "a", "dict"]) is None
    assert rehydrate({"w": "positions-not-a-list"}) is None
    assert rehydrate({"w": [None, "x", 1.5]}) is None


def test_work_to_record_carries_abstract_and_oa_status():
    """The SHAPED record (what the chain caches) carries ``abstract`` and
    ``oa_status``; both are omitted when the payload lacks them (columns stay NULL)."""
    from seedgraph.providers.openalex import _work_to_record

    rec = _work_to_record(
        {
            "id": "https://openalex.org/W77",
            "title": "Shaped",
            "publication_year": 2020,
            "abstract_inverted_index": {"study": [2], "We": [0], "the": [1], "system.": [3]},
            "open_access": {"is_oa": True, "oa_status": "hybrid"},
        }
    )
    assert rec["abstract"] == "We the study system."
    assert rec["oa_status"] == "hybrid"

    bare = _work_to_record({"id": "https://openalex.org/W78", "title": "Bare"})
    assert "abstract" not in bare and "oa_status" not in bare


def test_openalex_by_doi_shaped_record_includes_abstract_and_oa_status():
    """End-to-end through the provider: ``by_doi`` returns the shaped record WITH
    the two new fields (shaping lives in ``_work_to_record`` because the chain
    caches the shaped record — D-10)."""
    from seedgraph.providers.openalex import OpenAlexProvider

    endpoint = "https://api.openalex.org/works/doi:10.5/shaped"
    payload = {
        "id": "https://openalex.org/W501",
        "doi": "https://doi.org/10.5/shaped",
        "title": "Shaped End To End",
        "publication_year": 2019,
        "abstract_inverted_index": {"Second": [1], "First": [0]},
        "open_access": {"oa_status": "gold"},
    }
    seen: list = []
    client = _oa_json_client({endpoint: payload}, seen)

    rec = asyncio.run(OpenAlexProvider(client=client).by_doi("10.5/shaped"))

    assert rec["abstract"] == "First Second"
    assert rec["oa_status"] == "gold"
    assert seen == [endpoint]


def test_unpaywall_by_doi_carries_oa_status():
    """Unpaywall's payload carries the OA color at the top level — ``by_doi`` no
    longer drops it (one field in its record shaping)."""
    import httpx

    from seedgraph.providers.unpaywall import UnpaywallProvider

    payload = {"doi": "10.4/UP", "title": "Unpaywalled", "year": "2021", "oa_status": "bronze"}
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    )

    rec = asyncio.run(
        UnpaywallProvider(contact_email="ci@example.com", client=client).by_doi("10.4/UP")
    )

    assert rec == {"doi": "10.4/up", "title": "Unpaywalled", "year": 2021, "oa_status": "bronze"}


# ---------------------------------------------------------------------------
# Build D ch13 (D-13): no-network closed-form acquisition budget preview.
# ---------------------------------------------------------------------------


def test_preview_acquisition_closed_form_table():
    """``preview_acquisition`` is pure arithmetic: expected_papers = seeds +
    cap×depth, ~3 API calls/paper, ~1.9 MB disk/paper; negatives clamp to 0
    (the v1 ``preview_budget`` behavior, minus the LLM lines)."""
    from seedgraph.acquisition.preview import preview_acquisition

    table = [
        # (seeds, depth, cap) -> expected_papers
        (0, 0, 0, 0),
        (5, 0, 50, 5),  # depth 0: seeds only, no expansion
        (2, 2, 50, 102),  # the planner-conversion defaults over 2 seeds
        (5, 2, 50, 105),
        (1, 3, 10, 31),
        (2, 1, 0, 2),  # cap 0: generations add nothing
        (-3, -1, -5, 0),  # negatives clamp to 0
    ]
    for seeds, depth, cap, papers in table:
        pv = preview_acquisition(seeds, depth, cap)
        assert pv["expected_papers"] == papers, (seeds, depth, cap)
        assert pv["api_calls"] == papers * 3, (seeds, depth, cap)
        assert pv["est_disk_bytes"] == papers * 1_900_000, (seeds, depth, cap)
        assert pv["est_disk_mb"] == round(papers * 1.9, 1), (seeds, depth, cap)
        # Echo-back keys the renderers use, clamped like the arithmetic inputs.
        assert pv["seed_count"] == max(0, seeds)
        assert pv["depth"] == max(0, depth)
        assert pv["cap"] == max(0, cap)
