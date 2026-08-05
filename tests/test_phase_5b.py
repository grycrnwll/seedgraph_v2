"""Phase 5b — Acquisition & Resolution — acceptance tests (§11).

Every plan §11 test is implemented and runs fully OFFLINE: providers are mocked
(simple in-memory doubles; OA downloads ride an ``httpx.MockTransport``), nothing
touches the real network or an LLM. The top-of-file imports double as a
whole-phase import-cleanliness check.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects import sqlite as sqlite_dialect
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

# Import the full phase surface so collection validates import-cleanliness.
from seedgraph.acquisition import (  # noqa: F401
    acquire_corpus,
    manual_upload,
    resolve_corpus,
    resolve_work_markdown,
    run_corpus,
    walk_corpus,
)
from seedgraph.acquisition.bridge import (  # noqa: F401
    WorkSourceFile,
    backfill_markdown,
    derive_acquisition_state,
    write_bridge,
)
from seedgraph.acquisition.doctor_reconcile import (  # noqa: F401
    ReconcileFinding,
    cross_db_bridge_reconcile,
    current_markdown_for_source,
    foreign_key_check_all,
    open_cache_ro,
    reconcile_bridge,
)
from seedgraph.acquisition.fetch import FetchResult, OACandidate, fetch_oa  # noqa: F401
from seedgraph.acquisition.resolve import (  # noqa: F401
    SIM_RESOLVE,
    SIM_REVIEW,
    ResolutionOutcome,
    ResolveReport,
    resolve_record,
    score_title_match,
)
from seedgraph.acquisition.service import (  # noqa: F401
    AcquireReport,
    IngestFolderReport,
    ingest_folder,
    ingest_one_pdf,
)
from seedgraph.acquisition.walk import WalkReport, expand_references  # noqa: F401
from seedgraph.cache.db import init_cache_db
from seedgraph.cache.marker_backend import FakeMarkerBackend
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
from seedgraph.db import engine as db_engine
from seedgraph.db.connection import open_cache_db
from seedgraph.db.project_models import Identifier, ProjectDocument, ReviewQueueItem, Work
from seedgraph.project import review, service
from seedgraph.providers import build_default_providers  # noqa: F401
from seedgraph.providers.base import CircuitBreaker, CitationProvider, ProviderChain  # noqa: F401

_SCHEMA_DIR = Path(db_engine.__file__).resolve().parent / "schema"


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
def test_request_key_contract():
    # Stable, id-preference (openalex > doi) lowercased grammar.
    assert canonical_request_id({"openalex_id": "W123", "doi": "10.1/X"}) == "openalex=w123"
    assert canonical_request_id({"doi": "10.1/X"}) == "doi=10.1/x"
    assert key_referenced_works({"openalex_id": "W123"}) == "referenced_works:src=openalex=w123"

    # Build A ch2 — deliberate key move for the 10.48550 family (grammar note in
    # provider_cache §6.5 block): the arXiv-DOI alias keys as arxiv=<id>, agreeing
    # with the identity/fetch fold; the arXiv normalizer applies to the captured id.
    assert canonical_request_id({"doi": "10.48550/arXiv.2401.01234"}) == "arxiv=2401.01234"
    assert canonical_request_id({"doi": "10.48550/arxiv.2401.01234v2"}) == "arxiv=2401.01234"
    assert canonical_request_id({"arxiv_id": "2401.01234"}) == "arxiv=2401.01234"

    # phase_2's reconstruction over a stored Work row computes the SAME key.
    h = service.create_project("keycontract")
    service.add_work(h, ids={"doi": "10.1/x", "openalex": "W123"}, title="Paper")
    with Session(h.engine) as s:
        work = s.exec(select(Work)).one()
        key_from_work = key_referenced_works(work)
    key_from_dict = key_referenced_works({"openalex_id": "W123", "doi": "10.1/x"})
    assert key_from_work == key_from_dict == "referenced_works:src=openalex=w123"

    # Round-trip: cache a reference list under the work's key; reconstruct + read it.
    init_cache_db(None)
    conn = open_cache_db(None)
    try:
        cache_put(conn, provider="openalex", request_key=key_from_work, response=[{"openalex_id": "W9"}])
        reconstructed = key_referenced_works(work)
        assert cache_get(conn, provider="openalex", request_key=reconstructed) == [{"openalex_id": "W9"}]
    finally:
        conn.close()


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
def test_resolve_rubric():
    h = service.create_project("rubric")
    service.add_work(h, ids={"doi": "10.1/strong"}, title="Strong Id Paper")
    service.add_work(h, title="alpha beta gamma delta", year=2020)         # title resolve
    service.add_work(h, title="alpha beta gamma midband", year=2019)       # mid-band
    service.add_work(h, title="alpha lonely token here", year=2021)        # sub-threshold

    by_title = {
        "alpha beta gamma delta": {"title": "alpha beta gamma delta", "year": 2020, "doi": "10.9/resolved"},
        "alpha beta gamma midband": {"title": "alpha beta gamma midband extra", "year": 2019},
        "alpha lonely token here": {"title": "zeta eta theta iota", "year": 2021},
    }
    chain = _chain([MockProvider("openalex", by_title=by_title)], use_cache=False)

    def resolve(title_or_doi):
        with Session(h.engine, expire_on_commit=False) as s:
            work = s.exec(select(Work).where(Work.canonical_title == title_or_doi)).one()
            outcome = asyncio.run(resolve_record(chain, s, work=work))
            s.commit()
        return outcome

    # (a) strong-id DOI -> confidence 1.0, resolved, resolution_source='doi'.
    strong = resolve("Strong Id Paper")
    assert strong.status == "resolved" and strong.confidence == 1.0 and strong.resolution_source == "doi"
    with Session(h.engine) as s:
        idrow = s.exec(select(Identifier).where(Identifier.id_value == "10.1/strong")).one()
        assert idrow.resolution_source == "doi" and idrow.confidence == 1.0

    # (b) title+year sim >= 0.90 & year ok -> resolved (confidence == sim == 1.0 here).
    titled = resolve("alpha beta gamma delta")
    assert titled.status == "resolved" and titled.confidence >= SIM_RESOLVE

    # (c) sim in [0.65, 0.90) -> review_queue (title_year_ambiguous), ambiguous, NO merge.
    mid = resolve("alpha beta gamma midband")
    assert mid.status == "ambiguous" and mid.review_kind == "title_year_ambiguous"
    assert SIM_REVIEW <= mid.confidence < SIM_RESOLVE
    assert len(review.list_open(h)) == 1

    # (d) sim < 0.65 -> unresolved, no work mutation (no identifiers attached).
    low = resolve("alpha lonely token here")
    assert low.status == "unresolved"
    with Session(h.engine) as s:
        low_work = s.exec(select(Work).where(Work.canonical_title == "alpha lonely token here")).one()
        assert s.exec(select(Identifier).where(Identifier.work_id == low_work.work_id)).all() == []


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
def test_walk_corpus_growth(tmp_path):
    h = service.create_project("walkgrowth")
    service.add_work(h, ids={"doi": "10.s/seed", "openalex": "W_S"}, title="Seed", is_seed=True)
    chain = _walk_chain()
    frontier_dir = tmp_path / "frontier"

    report = asyncio.run(
        walk_corpus(h, chain, run_id="run-walk", depth=2, per_gen_cap=2, frontier_dir=frontier_dir)
    )

    with Session(h.engine) as s:
        works = s.exec(select(Work)).all()
        docs = s.exec(
            select(ProjectDocument).where(ProjectDocument.inclusion_reason == "citation_walk")
        ).all()
        e_work = s.exec(select(Work).where(Work.openalex_id == "W_E")).one()
    # EVERY discovered target is a works row (seed + A,B,C,D,E); membership is
    # metadata_only / citation_walk. E materialized even though the cap pruned it.
    assert len(works) == 6
    assert len(docs) == 5
    assert all(d.inclusion_status == "metadata_only" for d in docs)
    assert report.discovered == 5

    # gen0=1 seed; gen1=[A,B]; gen2 ranked [C,D,E] capped to per_gen_cap=2 -> [C,D].
    assert report.frontier_sizes == [1, 2, 2]

    # The cap kept the TOP-cited (count 2) and pruned the once-cited E from the frontier.
    gen2 = json.loads((frontier_dir / "gen_2.json").read_text(encoding="utf-8"))
    counts = [r["citation_count"] for r in gen2["ranked"]]
    assert counts[:2] == [2, 2] and counts[-1] == 1
    assert len(gen2["frontier"]) == 2
    assert e_work.work_id not in gen2["frontier"]

    # ZERO citation_edges rows (phase_2 owns those — binding r2-1 §4).
    pconn = sqlite3.connect(str(h.db_path))
    try:
        assert pconn.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0] == 0
    finally:
        pconn.close()

    # Reference lists are in provider_cache under referenced_works:src=<id>.
    cconn = open_cache_db(None)
    try:
        n = cconn.execute(
            "SELECT COUNT(*) FROM provider_cache WHERE request_key LIKE 'referenced_works:src=%'"
        ).fetchone()[0]
    finally:
        cconn.close()
    assert n >= 3  # S, A, B were expanded


# --------------------------------------------------------------------------
# test_doi_backfill
# --------------------------------------------------------------------------
def test_doi_backfill():
    h = service.create_project("doibackfill")
    service.add_work(h, ids={"openalex": "W_S"}, title="Seed", is_seed=True)
    refs = {"openalex=w_s": [{"openalex_id": "W_A"}, {"openalex_id": "W_A"}]}  # duplicate ref
    by_doi = {"W_A": {"openalex_id": "W_A", "doi": "10.a", "title": "Paper A", "year": 2018}}
    chain = _chain([MockProvider("openalex", refs=refs, by_doi=by_doi)])

    with Session(h.engine, expire_on_commit=False) as s:
        seed = s.exec(select(Work).where(Work.openalex_id == "W_S")).one()
        cited = asyncio.run(expand_references(chain, s, source_work_id=seed.work_id, run_id="r"))
        s.commit()

    assert len(cited) == 1  # the duplicate ref collapsed to one work
    with Session(h.engine) as s:
        a_rows = s.exec(select(Work).where(Work.openalex_id == "W_A")).all()
        assert len(a_rows) == 1
        assert a_rows[0].doi == "10.a"  # DOI backfilled from the OpenAlex record


# --------------------------------------------------------------------------
# test_walk_identity_collapse
# --------------------------------------------------------------------------
def test_walk_identity_collapse():
    h = service.create_project("collapse")
    service.add_work(h, ids={"openalex": "W_S"}, title="Seed", is_seed=True)
    refs = {
        "openalex=w_s": [
            {"openalex_id": "W_X", "doi": "10.x"},
            {"doi": "10.x"},               # same DOI -> collapses to one work
            {"title": "Shared Title"},
            {"title": "Shared Title"},     # title-only collision -> duplicate_candidate
        ]
    }
    chain = _chain([MockProvider("openalex", refs=refs)])

    with Session(h.engine, expire_on_commit=False) as s:
        seed = s.exec(select(Work).where(Work.openalex_id == "W_S")).one()
        asyncio.run(expand_references(chain, s, source_work_id=seed.work_id, run_id="r"))
        s.commit()

    with Session(h.engine) as s:
        doi_works = s.exec(select(Work).where(Work.doi == "10.x")).all()
        assert len(doi_works) == 1  # DOI collapse via upsert_work
        items = s.exec(select(ReviewQueueItem)).all()
        reasons = {json.loads(i.payload)["reason"] for i in items if i.payload}
        assert "title_collision" in reasons  # title hash NEVER auto-merges


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
def test_acquire_bridge_single_primary():
    h = service.create_project("acqbridge")
    work = service.add_work(h, ids={"arxiv": "2101.00001"}, title="Arxiv Paper", is_seed=True)
    wid = work.work_id

    pdf = b"%PDF-1.4 arxiv body one"
    requested: list = []
    client = _mock_client({"https://arxiv.org/pdf/2101.00001": pdf}, requested)
    backend = FakeMarkerBackend()
    chain = _chain([MockProvider("openalex")])  # no provider OA; arXiv-direct is used

    report = acquire_corpus(
        h, chain, run_id="run-acq", statuses=("included",),
        cache_root=None, http_client=client, backend=backend,
    )
    assert report.acquired == 1
    assert backend.call_count == 1  # convert ran once

    with Session(h.engine) as s:
        rows = s.exec(select(WorkSourceFile).where(WorkSourceFile.work_id == wid)).all()
        assert len(rows) == 1
        assert rows[0].markdown_id is not None and rows[0].markdown_hash is not None
        doc = s.get(ProjectDocument, wid)
        assert doc.access_status == "open_access"

    # Re-run is idempotent: bridge reconciles ok -> no second download/convert.
    client2 = _mock_client({"https://arxiv.org/pdf/2101.00001": pdf}, [])
    report2 = acquire_corpus(
        h, chain, run_id="run-acq2", statuses=("included",),
        cache_root=None, http_client=client2, backend=backend,
    )
    assert report2.already_cached == 1 and report2.acquired == 0
    assert backend.call_count == 1  # convert did NOT run again
    with Session(h.engine) as s:
        assert len(s.exec(select(WorkSourceFile).where(WorkSourceFile.work_id == wid)).all()) == 1

    # A replacement acquisition (different file, same work) upserts in place: still 1 row.
    with Session(h.engine, expire_on_commit=False) as s:
        write_bridge(
            s, work_id=wid, source_file_id="sf_replacement", file_hash="replacement",
            markdown_id="md_repl", markdown_hash="repl", acquisition_method="manual_upload",
        )
        s.commit()
    with Session(h.engine) as s:
        rows = s.exec(select(WorkSourceFile).where(WorkSourceFile.work_id == wid)).all()
        assert len(rows) == 1 and rows[0].file_hash == "replacement"  # newest wins

    # A SECOND bridge row for the same work_id hits UNIQUE(work_id) -> IntegrityError.
    now = datetime.now(timezone.utc).isoformat()
    with Session(h.engine) as s:
        s.add(WorkSourceFile(
            work_id=wid, source_file_id="sf_dup", file_hash="dup",
            acquisition_method="manual_upload", created_at=now, updated_at=now,
        ))
        with pytest.raises(IntegrityError):
            s.commit()


# --------------------------------------------------------------------------
# test_manual_upload
# --------------------------------------------------------------------------
def test_manual_upload(tmp_path):
    h = service.create_project("manual")
    work = service.add_work(h, title="Private Paper", inclusion_status="metadata_only")
    pdf_path = tmp_path / "upload.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 user supplied private body text here")

    row = manual_upload(h, work_id=work.work_id, pdf_path=pdf_path, backend=FakeMarkerBackend())
    assert row.acquisition_method == "manual_upload"
    assert row.markdown_id is not None

    with Session(h.engine) as s:
        bridges = s.exec(select(WorkSourceFile).where(WorkSourceFile.work_id == work.work_id)).all()
        assert len(bridges) == 1
        doc = s.get(ProjectDocument, work.work_id)
        assert doc.access_status == "user_supplied_private"
        # Provided => included: a converted+bridged upload joins the local corpus by
        # default (metadata_only -> included, inclusion_reason='provided_pdf').
        assert doc.inclusion_status == "included"
        assert doc.inclusion_reason == "provided_pdf"

    # Opt-out: promote=False preserves the prior (metadata_only) membership.
    work2 = service.add_work(h, title="Kept Meta Paper", inclusion_status="metadata_only")
    pdf2 = tmp_path / "upload2.pdf"
    pdf2.write_bytes(b"%PDF-1.4 another user supplied private body of words here")
    manual_upload(
        h, work_id=work2.work_id, pdf_path=pdf2, backend=FakeMarkerBackend(), promote=False
    )
    with Session(h.engine) as s:
        doc2 = s.get(ProjectDocument, work2.work_id)
        assert doc2.inclusion_status == "metadata_only"
        assert doc2.access_status == "user_supplied_private"

    # The cache source_files row carries the private access class (authority of record).
    cconn = open_cache_db(None)
    try:
        ac = cconn.execute(
            "SELECT access_class FROM source_files WHERE source_file_id = ?", (row.source_file_id,)
        ).fetchone()[0]
    finally:
        cconn.close()
    assert ac == "user_supplied_private"


# --------------------------------------------------------------------------
# test_import_markdown_and_bridge
# --------------------------------------------------------------------------
def test_import_markdown_and_bridge(tmp_path):
    """GPU-free import of externally-converted markdown: no Marker, a
    markdown_documents row indistinguishable from a converted one, bridged +
    promoted, and retrievable via the same resolve path a converted doc uses."""
    from seedgraph import cache_access
    from seedgraph.acquisition.bridge import resolve_work_markdown
    from seedgraph.acquisition.service import import_markdown_and_bridge

    h = service.create_project("importmd")
    work = service.add_work(h, title="Externally Converted Paper",
                            inclusion_status="metadata_only")
    pdf_path = tmp_path / "provided.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 provided pdf provenance only, never converted")
    markdown_text = "# Externally Converted Paper\n\n## Intro\n\nBody words here already.\n"
    md_path = tmp_path / "provided.md"
    md_path.write_text(markdown_text, encoding="utf-8")

    # No Marker backend is passed anywhere; a spy backend would never be called.
    row = import_markdown_and_bridge(
        h, work_id=work.work_id, pdf_path=pdf_path, markdown_path=md_path
    )
    assert row.acquisition_method == "manual_upload"
    assert row.markdown_id is not None

    # The stored markdown_documents row equals the provided markdown, and its
    # conversion_runs provenance is HONEST that Marker never ran.
    cconn = open_cache_db(None)
    try:
        mrow = cconn.execute(
            "SELECT storage_uri, conversion_status, conversion_run_id "
            "FROM markdown_documents WHERE markdown_id = ?", (row.markdown_id,)
        ).fetchone()
        assert mrow is not None
        assert mrow[1] == "success"
        conv = cconn.execute(
            "SELECT converter_name, run_status FROM conversion_runs "
            "WHERE conversion_run_id = ?", (mrow[2],)
        ).fetchone()
        assert (conv[0], conv[1]) == ("external_import", "success")
    finally:
        cconn.close()

    # Bridge links the work; work promoted metadata_only -> included.
    with Session(h.engine) as s:
        bridges = s.exec(
            select(WorkSourceFile).where(WorkSourceFile.work_id == work.work_id)
        ).all()
        assert len(bridges) == 1
        assert resolve_work_markdown(s, work_id=work.work_id) == (
            row.markdown_id, row.markdown_hash
        )
        doc = s.get(ProjectDocument, work.work_id)
        assert doc.inclusion_status == "included"
        assert doc.inclusion_reason == "provided_pdf"
        assert doc.access_status == "user_supplied_private"

    # Retrievable via the SAME resolve path a converted doc uses (sections build /
    # cite parse read markdown this way) — content equals what we provided.
    cconn = cache_access.open_cache_ro(None)
    try:
        mdrow = cache_access.read_markdown(cconn, None, row.markdown_id)
    finally:
        cconn.close()
    assert mdrow is not None
    assert mdrow.text == markdown_text
    assert mdrow.access_class == "user_supplied_private"

    # Idempotent: a second import for the same (already-bridged) work is a no-op.
    row2 = import_markdown_and_bridge(
        h, work_id=work.work_id, pdf_path=pdf_path, markdown_path=md_path
    )
    assert row2.markdown_id == row.markdown_id


# --------------------------------------------------------------------------
# test_bridge_reconcile
# --------------------------------------------------------------------------
def test_bridge_reconcile(tmp_path):
    from seedgraph.cache.convert import convert_source_file
    from seedgraph.cache.ingest import ingest_file
    from seedgraph.vocab import AccessClass, AcquisitionMethod

    h = service.create_project("reconcile")
    w_ok = service.add_work(h, ids={"doi": "10.1/ok"}, title="OK Work")
    w_stale = service.add_work(h, ids={"doi": "10.1/stale"}, title="Stale Work")
    w_missing = service.add_work(h, ids={"doi": "10.1/missing"}, title="Missing Work")

    def _ingest(content: bytes):
        p = tmp_path / f"{abs(hash(content))}.pdf"
        p.write_bytes(content)
        return ingest_file(p, access_class=AccessClass.open_access,
                           acquisition_method=AcquisitionMethod.open_access_fetch, root=None)

    # ok: ingest + convert, bridge to the current markdown.
    src_ok = _ingest(b"%PDF-1.4 ok body content here words")
    md_ok = convert_source_file(src_ok.source_file_id, backend=FakeMarkerBackend(), root=None)

    # stale: bridge to an OLDER markdown, then a reconversion mints a newer one.
    src_st = _ingest(b"%PDF-1.4 stale body content here words")
    md_a = convert_source_file(
        src_st.source_file_id,
        backend=FakeMarkerBackend(markdown="# Alpha\n\n## Sec\n\nalpha body words here.\n"),
        root=None,
    )
    md_b = convert_source_file(
        src_st.source_file_id,
        backend=FakeMarkerBackend(markdown="# Beta\n\n## Sec\n\nbeta body words here now.\n"),
        force=True, root=None,
    )
    assert md_a.markdown_hash != md_b.markdown_hash

    with Session(h.engine, expire_on_commit=False) as s:
        write_bridge(s, work_id=w_ok.work_id, source_file_id=src_ok.source_file_id,
                     file_hash=src_ok.file_hash, markdown_id=md_ok.markdown_id,
                     markdown_hash=md_ok.markdown_hash, acquisition_method="open_access_fetch")
        write_bridge(s, work_id=w_stale.work_id, source_file_id=src_st.source_file_id,
                     file_hash=src_st.file_hash, markdown_id=md_a.markdown_id,
                     markdown_hash=md_a.markdown_hash, acquisition_method="open_access_fetch")
        write_bridge(s, work_id=w_missing.work_id, source_file_id="sf_deadbeef",
                     file_hash="deadbeef", markdown_id="md_nope", markdown_hash="nope",
                     acquisition_method="open_access_fetch")
        s.commit()

    findings = {f.work_id: f.status for f in reconcile_bridge(h.engine, None)}
    assert findings[w_ok.work_id] == "ok"
    assert findings[w_stale.work_id] == "stale"
    assert findings[w_missing.work_id] == "missing"

    # foreign_key_check_all reports clean on a healthy project.db (+ cache.db).
    assert foreign_key_check_all(h.engine, None) == []


# --------------------------------------------------------------------------
# test_offline_resilience
# --------------------------------------------------------------------------
def test_offline_resilience():
    h = service.create_project("offline")
    service.add_work(h, ids={"openalex": "W_S"}, title="Seed", is_seed=True)

    # Pre-warm provider_cache with the seed's reference list (DOI already present so
    # NO by_doi enrichment / network is needed).
    seed_record = {"openalex_id": "W_S"}
    init_cache_db(None)
    conn = open_cache_db(None)
    try:
        cache_put(conn, provider="openalex", request_key=key_referenced_works(seed_record),
                  response=[{"openalex_id": "W_A", "doi": "10.a"}])
    finally:
        conn.close()

    # The provider RAISES on every call (network down) — the walk must build from cache.
    chain = _chain([MockProvider("openalex", fail_on={"referenced_works", "by_doi", "by_title"})])
    report = asyncio.run(walk_corpus(h, chain, run_id="run-offline", depth=1))
    assert report.discovered >= 1
    with Session(h.engine) as s:
        assert s.exec(select(Work).where(Work.openalex_id == "W_A")).first() is not None


# --------------------------------------------------------------------------
# test_manifest_merge (D5)
# --------------------------------------------------------------------------
def test_manifest_merge(monkeypatch):
    from seedgraph import run as run_mod
    from seedgraph.errors import SeedgraphError
    from seedgraph.paths import project_dir

    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret-value")
    rid = run_mod.ensure_run("manifestmerge")

    # Two stages write DISJOINT top-level sections into one manifest.
    run_mod.update_manifest("manifestmerge", rid, {"walk": {"discovered": 3, "frontier_sizes": [1, 2]}})
    run_mod.update_manifest(
        "manifestmerge", rid,
        {"acquisition": {"acquired": 1, "access_class_summary": {"open_access": 1}}},
    )

    path = project_dir("manifestmerge") / "runs" / rid / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sections = manifest["sections"]
    assert sections["walk"]["discovered"] == 3
    assert sections["acquisition"]["access_class_summary"] == {"open_access": 1}
    assert "super-secret-value" not in path.read_text(encoding="utf-8")  # no secret leaked

    # A second DIFFERENT-stage write to an already-written section is a contract
    # violation (D5 — no deep-merge, no cross-stage accumulation).
    with pytest.raises(SeedgraphError):
        run_mod.update_manifest("manifestmerge", rid, {"walk": {"discovered": 99}})


# --------------------------------------------------------------------------
# test_acquisition_milestone (doc 01 §3 steps 3-6; no network, no LLM)
# --------------------------------------------------------------------------
def test_acquisition_milestone():
    from seedgraph.providers.fake import FakeProvider, fake_http_client

    h = service.create_project("acq_demo")
    service.add_work(h, ids={"doi": "10.x/seed1"}, title="Seed DOI", is_seed=True)
    service.add_work(h, ids={"arxiv": "2101.00001"}, title="Seed Arxiv", is_seed=True)

    chain = _chain([FakeProvider()])
    out = run_corpus(
        h, chain, depth=1, per_gen_cap=25, root=None,
        http_client=fake_http_client(), backend=FakeMarkerBackend(),
    )

    # Seeds resolved (confidence filled).
    with Session(h.engine) as s:
        seed_ids = s.exec(select(Identifier)).all()
        assert any(i.confidence == 1.0 for i in seed_ids)

        # >=1 metadata-only walked work with inclusion_reason='citation_walk'.
        walked = s.exec(
            select(ProjectDocument).where(
                ProjectDocument.inclusion_status == "metadata_only",
                ProjectDocument.inclusion_reason == "citation_walk",
            )
        ).all()
        assert len(walked) >= 1

        # >=1 OA paper acquired with a complete single-primary bridge (markdown filled).
        bridged = s.exec(select(WorkSourceFile).where(WorkSourceFile.markdown_id.is_not(None))).all()
        assert len(bridged) >= 1

        # A no-OA seed reports requires_user_upload (included, no bridge).
        doi_seed = s.exec(select(Work).where(Work.doi == "10.x/seed1")).one()
        assert derive_acquisition_state(s, None, doi_seed.work_id) == "requires_user_upload"

        # ZERO citation_edges rows (phase_2's job).
        pconn = sqlite3.connect(str(h.db_path))
        try:
            assert pconn.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0] == 0
        finally:
            pconn.close()

    assert out["acquisition"].acquired >= 1
    assert out["acquisition"].requires_upload >= 1

    # provider_cache populated under the §6.5 keys.
    cconn = open_cache_db(None)
    try:
        refs = cconn.execute(
            "SELECT COUNT(*) FROM provider_cache WHERE request_key LIKE 'referenced_works:src=%'"
        ).fetchone()[0]
    finally:
        cconn.close()
    assert refs >= 1

    # doctor reports the bridge ok.
    results = cross_db_bridge_reconcile(None, "acq_demo")
    recon = next(r for r in results if r.name == "cross_db_bridge_reconcile")
    assert recon.ok


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


def test_phase5b_schema_parity(tmp_path):
    # provider_cache (cache scope) — apply only 0003 to a fresh db, then reflect.
    cache_sql = (_SCHEMA_DIR / "cache" / "0003_provider_cache.sql").read_text(encoding="utf-8")
    cdb = tmp_path / "cache_mig.db"
    conn = sqlite3.connect(str(cdb))
    conn.executescript(cache_sql)
    conn.commit()
    conn.close()
    insp = inspect(create_engine(f"sqlite:///{cdb.as_posix()}"))
    assert "provider_cache" in insp.get_table_names()
    _assert_table_parity(insp, "provider_cache", ProviderCache)
    uq = [set(u["column_names"]) for u in insp.get_unique_constraints("provider_cache")]
    assert {"provider", "request_key"} in uq

    # work_source_files (project scope) — needs works (0002) for its FK.
    proj_db = tmp_path / "proj_mig.db"
    conn = sqlite3.connect(str(proj_db))
    conn.executescript((_SCHEMA_DIR / "project" / "0002_project_model.sql").read_text(encoding="utf-8"))
    conn.executescript((_SCHEMA_DIR / "project" / "0003_acquisition.sql").read_text(encoding="utf-8"))
    conn.commit()
    conn.close()
    insp = inspect(create_engine(f"sqlite:///{proj_db.as_posix()}"))
    assert "work_source_files" in insp.get_table_names()
    _assert_table_parity(insp, "work_source_files", WorkSourceFile)
    uq = [set(u["column_names"]) for u in insp.get_unique_constraints("work_source_files")]
    assert {"work_id"} in uq
    fks = insp.get_foreign_keys("work_source_files")
    assert any(fk["referred_table"] == "works" and fk["constrained_columns"] == ["work_id"] for fk in fks)


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


def test_acquire_transient_failure_counts_failed_no_bridge():
    """All-transport-error acquisition tallies ``AcquireReport.failed`` (the
    previously-dead field) + ``state_summary['failed']``, converts nothing, and
    writes NO bridge row — so a re-run naturally retries (D-4)."""
    h = service.create_project("acqfailed")
    work = service.add_work(h, ids={"arxiv": "2103.00003"}, title="Flaky Paper", is_seed=True)

    requested: list = []
    client = _status_client({"https://arxiv.org/pdf/2103.00003": (503, b"")}, requested)
    backend = FakeMarkerBackend()
    chain = _chain([MockProvider("openalex")])  # no provider OA; arXiv-direct only

    report = acquire_corpus(
        h, chain, run_id="run-acqfailed", statuses=("included",),
        cache_root=None, http_client=client, backend=backend,
    )
    assert report.failed == 1
    assert report.requires_upload == 0 and report.acquired == 0
    assert report.state_summary.get("failed") == 1
    assert report.to_section()["failed"] == 1  # honest run-state accounting
    assert backend.call_count == 0  # nothing converted
    with Session(h.engine) as s:
        rows = s.exec(
            select(WorkSourceFile).where(WorkSourceFile.work_id == work.work_id)
        ).all()
        assert rows == []  # no bridge row -> a re-run retries


def test_acquire_conversion_error_counts_failed_and_continues():
    """A single OA PDF that converts to garbage (the ``validate_markdown`` gate,
    raising ``ConversionError``) is counted ``failed`` and the pass CONTINUES —
    exactly like a transient fetch failure — instead of the ConversionError
    propagating out of ``_acquire_async`` and aborting the whole corpus pass. A
    subsequent good work in the same pass is still ``acquired``."""
    from seedgraph.cache.marker_backend import MarkerResult

    class _SelectiveBackend:
        """Returns empty (garbage) markdown for the PDF whose bytes carry
        ``bad_marker`` -> ``convert_source_file`` raises ``ConversionError`` for
        that work only; good markdown (passes the gate) for every other PDF."""

        version = "fake-0.0.0"

        def __init__(self, bad_marker: bytes) -> None:
            self.bad_marker = bad_marker
            self.call_count = 0

        def __call__(self, pdf_path, cfg):
            self.call_count += 1
            data = Path(pdf_path).read_bytes()
            markdown = (
                ""
                if self.bad_marker in data
                else "# Good\n\n## Section\n\nGood body words here now.\n"
            )
            return MarkerResult(markdown=markdown, block_json=None, warnings=[])

    h = service.create_project("acqconvfail")
    good = service.add_work(h, ids={"arxiv": "2201.00001"}, title="Good Paper", is_seed=True)
    bad = service.add_work(h, ids={"arxiv": "2201.00002"}, title="Bad Paper", is_seed=True)

    pdf_good = b"%PDF-1.4 good body GOODMARK"
    pdf_bad = b"%PDF-1.4 bad body BADMARK"
    requested: list = []
    client = _mock_client(
        {
            "https://arxiv.org/pdf/2201.00001": pdf_good,
            "https://arxiv.org/pdf/2201.00002": pdf_bad,
        },
        requested,
    )
    backend = _SelectiveBackend(bad_marker=b"BADMARK")
    chain = _chain([MockProvider("openalex")])  # arXiv-direct only

    # The pass must COMPLETE (no ConversionError propagates out).
    report = acquire_corpus(
        h, chain, run_id="run-acqconvfail", statuses=("included",),
        cache_root=None, http_client=client, backend=backend,
    )

    assert report.failed == 1  # the garbage-converting work
    assert report.acquired == 1  # the good work still lands
    assert report.state_summary.get("failed") == 1
    assert backend.call_count == 2  # both works reached the convert step

    with Session(h.engine) as s:
        good_rows = s.exec(
            select(WorkSourceFile).where(WorkSourceFile.work_id == good.work_id)
        ).all()
        bad_rows = s.exec(
            select(WorkSourceFile).where(WorkSourceFile.work_id == bad.work_id)
        ).all()
    assert len(good_rows) == 1 and good_rows[0].markdown_id is not None
    assert bad_rows == []  # no bridge row for the failed convert -> a re-run retries


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


def _budget_project(name: str):
    """Three works: one seed + two walk-cited ``metadata_only`` works, with
    ``created_at`` pinned to DISTINCT values in a DIFFERENT order than insertion
    (low < high < seed) — so a created_at-order assertion is unambiguous and any
    seed-first / hint reorder is provably the sort, never accident."""
    h = service.create_project(name)
    seed = service.add_work(h, ids={"arxiv": "2101.00001"}, title="Seed", is_seed=True)
    low = service.add_work(
        h, ids={"arxiv": "2101.00002"}, title="Cited low", inclusion_status="metadata_only"
    )
    high = service.add_work(
        h, ids={"arxiv": "2101.00003"}, title="Cited high", inclusion_status="metadata_only"
    )
    stamps = {
        low.work_id: "2026-01-01T00:00:00+00:00",
        high.work_id: "2026-01-02T00:00:00+00:00",
        seed.work_id: "2026-01-03T00:00:00+00:00",
    }
    with Session(h.engine, expire_on_commit=False) as s:
        for wid, ts in stamps.items():
            w = s.get(Work, wid)
            w.created_at = ts
            s.add(w)
        s.commit()
    return h, seed.work_id, low.work_id, high.work_id


def test_acquire_budget_value_ordered_max_papers():
    """With an ``order_hint`` (the run_corpus path) targets sort seeds first, then
    hint count desc, work_id tie-break (D-5); ``max_papers=2`` attempts exactly the
    seed + the count-5 work and COUNTS the count-1 work ``skipped_budget`` (D-6) —
    never silently skipped."""
    from seedgraph.acquisition.service import _acquire_async

    h, seed_id, low_id, high_id = _budget_project("budgetorder")
    requested: list = []
    client = _mock_client(dict(_BUDGET_BODIES), requested)
    chain = _chain([MockProvider("openalex")])  # no provider OA; arXiv-direct URLs

    report = asyncio.run(
        _acquire_async(
            h, chain, run_id="run-budget-order", cache_root=None,
            http_client=client, backend=FakeMarkerBackend(),
            max_papers=2, order_hint={high_id: 5, low_id: 1},
        )
    )
    # Seed first (despite the LATEST created_at), then the count-5 work; the
    # count-1 work was cut by the budget — accounted, not attempted.
    assert requested == [_BUDGET_URLS["seed"], _BUDGET_URLS["high"]]
    assert report.acquired == 2
    assert report.skipped_budget == 1
    assert report.to_section()["skipped_budget"] == 1  # additive section key


def test_acquire_no_hint_created_at_order_preserved():
    """Standalone ``corpus acquire`` (no ``order_hint``): today's ``created_at``
    order is untouched — even the seed does not jump ahead (D-5: the hint-or-
    created_at split; no in-degree fallback)."""
    h, seed_id, low_id, high_id = _budget_project("budgetnohint")
    requested: list = []
    client = _mock_client(dict(_BUDGET_BODIES), requested)
    chain = _chain([MockProvider("openalex")])

    report = acquire_corpus(
        h, chain, run_id="run-budget-nohint", cache_root=None,
        http_client=client, backend=FakeMarkerBackend(),
    )
    # created_at order low < high < seed (insertion order was seed, low, high —
    # this pins ORDER BY created_at specifically, not rowid/insertion order).
    assert requested == [_BUDGET_URLS["low"], _BUDGET_URLS["high"], _BUDGET_URLS["seed"]]
    assert report.acquired == 3
    assert report.skipped_budget == 0


def test_acquire_max_papers_zero_attempts_nothing():
    """``max_papers=0`` attempts NOTHING (``is not None`` check — 0 is a real
    budget, not "unbounded"; v1 semantics ported verbatim, D-6): every target is
    counted ``skipped_budget``."""
    h, *_ = _budget_project("budgetzero")
    requested: list = []
    client = _mock_client(dict(_BUDGET_BODIES), requested)
    chain = _chain([MockProvider("openalex")])

    report = acquire_corpus(
        h, chain, run_id="run-budget-zero", cache_root=None,
        http_client=client, backend=FakeMarkerBackend(), max_papers=0,
    )
    assert requested == []  # zero attempts
    assert report.acquired == 0 and report.requires_upload == 0 and report.failed == 0
    assert report.skipped_budget == 3


def test_acquire_budget_seconds_zero_attempts_nothing():
    """``budget_seconds=0.0`` likewise attempts nothing (``is not None``, not
    truthiness): the whole pass is counted ``skipped_budget``."""
    h, *_ = _budget_project("budgetsecszero")
    requested: list = []
    client = _mock_client(dict(_BUDGET_BODIES), requested)
    chain = _chain([MockProvider("openalex")])

    report = acquire_corpus(
        h, chain, run_id="run-budget-secs", cache_root=None,
        http_client=client, backend=FakeMarkerBackend(), budget_seconds=0.0,
    )
    assert requested == []
    assert report.acquired == 0
    assert report.skipped_budget == 3


def test_acquire_default_budget_manifest_section_shape():
    """Default ``max_papers=None`` / ``budget_seconds=None`` leave behavior
    unchanged (every target attempted, created_at order) and the ``acquisition``
    manifest section keeps today's shape plus ONLY the additive ``skipped_budget``
    key (D5 — the section is solely owned by this stage)."""
    from seedgraph import run as run_mod
    from seedgraph.paths import project_dir

    h, *_ = _budget_project("budgetdefault")
    rid = run_mod.ensure_run("budgetdefault")
    requested: list = []
    client = _mock_client(dict(_BUDGET_BODIES), requested)
    chain = _chain([MockProvider("openalex")])

    report = acquire_corpus(
        h, chain, run_id=rid, cache_root=None,
        http_client=client, backend=FakeMarkerBackend(),
    )
    assert len(requested) == 3 and report.acquired == 3  # unbounded, as today

    manifest_path = project_dir("budgetdefault") / "runs" / rid / "manifest.json"
    section = json.loads(manifest_path.read_text(encoding="utf-8"))["sections"]["acquisition"]
    assert set(section) == {
        "acquired", "already_cached", "requires_upload", "failed",
        "skipped_budget", "access_class_summary", "state_summary",
    }
    assert section["skipped_budget"] == 0
    assert section["acquired"] == 3


def test_walk_citation_counts_hint_in_memory_only():
    """The walk accumulates the per-generation ``next_counts`` onto
    ``WalkReport.citation_counts`` (the D-5 value-ordering hint) and does NOT
    serialize it: the pinned ``walk`` manifest section shape is unchanged."""
    h = service.create_project("walkcounts")
    service.add_work(h, ids={"doi": "10.s/seed", "openalex": "W_S"}, title="Seed", is_seed=True)
    chain = _walk_chain()

    report = asyncio.run(walk_corpus(h, chain, run_id="run-walkcounts", depth=2, per_gen_cap=2))

    with Session(h.engine) as s:
        by_oa = {w.openalex_id: w.work_id for w in s.exec(select(Work)).all()}
    # gen1: A,B cited once each by the seed; gen2: C,D cited twice (by A and B),
    # E once (by B only) — the SAME counts that ranked each frontier, accumulated.
    assert report.citation_counts == {
        by_oa["W_A"]: 1,
        by_oa["W_B"]: 1,
        by_oa["W_C"]: 2,
        by_oa["W_D"]: 2,
        by_oa["W_E"]: 1,
    }
    # In-memory only: citation_counts never serializes; the walk section carries
    # exactly today's keys plus the additive ch8 enrichment/residue counters (D5).
    assert set(report.to_section()) == {
        "depth", "discovered", "frontier_sizes", "per_provider_calls",
        "enriched", "enrichment_failed", "untitled_after_walk",
    }


def test_run_corpus_threads_walk_hint_and_budgets(monkeypatch):
    """``run_corpus`` threads ``walk_report.citation_counts`` into the acquire
    pass as its ``order_hint`` and passes ``max_papers``/``budget_seconds``
    through (the chunk-5 seam)."""
    from seedgraph.acquisition import service as service_mod

    h = service.create_project("runthreads")
    service.add_work(h, ids={"doi": "10.s/seed", "openalex": "W_S"}, title="Seed", is_seed=True)
    chain = _walk_chain()

    captured: dict = {}

    async def _fake_acquire(h2, chain2, **kw):
        captured.update(kw)
        return AcquireReport()

    monkeypatch.setattr(service_mod, "_acquire_async", _fake_acquire)
    out = service_mod.run_corpus(h, chain, depth=1, per_gen_cap=2, max_papers=7, budget_seconds=1.5)

    assert captured["max_papers"] == 7
    assert captured["budget_seconds"] == 1.5
    assert captured["order_hint"] is out["walk"].citation_counts  # the SAME dict
    assert captured["order_hint"]  # depth=1 walk counted A and B once each


def test_cli_corpus_budget_flags(monkeypatch):
    """``--max-papers`` / ``--budget-seconds`` parse on ``corpus acquire`` and
    ``corpus run``, and the echo accounts ``skipped_budget`` (never silent)."""
    from typer.testing import CliRunner

    from seedgraph.cli import app

    monkeypatch.setenv("SEEDGRAPH_FAKE_PROVIDERS", "1")
    h = service.create_project("clibudget")
    service.add_work(h, ids={"arxiv": "2101.00001"}, title="Seed", is_seed=True)

    runner = CliRunner()
    res = runner.invoke(app, ["corpus", "acquire", "clibudget", "--max-papers", "0"])
    assert res.exit_code == 0, res.output
    assert "skipped_budget=1" in res.output
    assert "acquired=0" in res.output

    res2 = runner.invoke(
        app,
        ["corpus", "run", "clibudget", "--depth", "1",
         "--max-papers", "0", "--budget-seconds", "60"],
    )
    assert res2.exit_code == 0, res2.output
    assert "skipped_budget=" in res2.output
    assert "acquired=0" in res2.output


# --------------------------------------------------------------------------
# Build D chunk 6 — per-paper acquisition progress events (D-7)
# --------------------------------------------------------------------------
def test_acquire_progress_steps_once_per_target():
    """``_acquire_async`` steps its ``progress`` once per target with
    ``detail=work_id`` and the running ``acquired/failed/requires_upload/
    skipped_budget`` counters — budget-cut targets INCLUDED (accounted frames,
    never silent). Emit-backed here (the job path), so the frames land in
    events.jsonl in the exact ``append_event`` shape the run page polls."""
    from seedgraph.acquisition.service import _acquire_async
    from seedgraph.progress import Progress, make_emitter
    from seedgraph.run import ensure_run, read_events

    h, seed_id, low_id, high_id = _budget_project("progacq")
    rid = ensure_run("progacq")
    emit = make_emitter("progacq", rid, phase="conversion")
    prog = Progress(0, "works", emit=emit, echo=False)

    client = _mock_client(dict(_BUDGET_BODIES), [])
    chain = _chain([MockProvider("openalex")])
    report = asyncio.run(
        _acquire_async(
            h, chain, run_id=rid, cache_root=None,
            http_client=client, backend=FakeMarkerBackend(),
            max_papers=1, order_hint={high_id: 5, low_id: 1},
            progress=prog,
        )
    )

    frames = [e for e in read_events("progacq", rid) if e["event"] == "progress"]
    assert len(frames) == 3  # one per target — the 2 budget-cut works still step
    assert [f["data"]["done"] for f in frames] == [1, 2, 3]
    # The pass reconciled the caller's placeholder total from its own select.
    assert all(f["data"]["total"] == 3 for f in frames)
    # detail=work_id rides the message; the hint sort put the seed first.
    assert seed_id in frames[0]["message"]
    assert high_id in frames[1]["message"] and low_id in frames[2]["message"]
    # Running outcome counters: 1 acquired (the seed), then 2 skipped_budget.
    assert frames[0]["data"]["acquired"] == 1 and frames[0]["data"]["skipped_budget"] == 0
    assert frames[2]["data"]["skipped_budget"] == 2
    assert frames[2]["data"]["failed"] == 0 and frames[2]["data"]["requires_upload"] == 0
    assert report.acquired == 1 and report.skipped_budget == 2


def test_walk_progress_steps_per_source_per_generation():
    """``walk_corpus`` steps once per expanded source work per generation and
    grows ``progress.total`` by each generation's frontier size (the walk's
    universe is only known generation-by-generation)."""
    from seedgraph.progress import Progress, make_emitter
    from seedgraph.run import ensure_run, read_events

    h = service.create_project("progwalk")
    seed = service.add_work(h, ids={"doi": "10.s/seed", "openalex": "W_S"}, title="Seed", is_seed=True)
    chain = _walk_chain()
    rid = ensure_run("progwalk")
    emit = make_emitter("progwalk", rid, phase="conversion")
    prog = Progress(0, "works", emit=emit, echo=False)

    asyncio.run(walk_corpus(h, chain, run_id=rid, depth=2, per_gen_cap=2, progress=prog))

    frames = [e for e in read_events("progwalk", rid) if e["event"] == "progress"]
    # gen1 expands the seed (frontier of 1); gen2 the capped frontier (A and B).
    assert [f["data"]["done"] for f in frames] == [1, 2, 3]
    assert [f["data"]["total"] for f in frames] == [1, 3, 3]
    assert [f["data"]["generation"] for f in frames] == [1, 2, 2]
    assert seed.work_id in frames[0]["message"] and frames[0]["data"]["cited"] == 2
    assert sorted(f["data"]["cited"] for f in frames[1:]) == [2, 3]  # A cites 2, B cites 3


def test_planner_conversion_job_emits_per_paper_progress(monkeypatch):
    """The planner's conversion job threads emit-backed :class:`Progress` handles
    for BOTH stages through ``run_corpus``: every walk source and every acquire
    target lands as a ``progress`` event with ``done``/``total`` in ``data`` (the
    run page's existing poller shape) — no new progress file (D-7)."""
    from seedgraph.web.planner import _job_conversion, plan_job

    monkeypatch.setenv("SEEDGRAPH_FAKE_PROVIDERS", "1")
    h = service.create_project("plannerprog")
    service.add_work(h, ids={"arxiv": "2101.00001"}, title="Seed", is_seed=True)

    events: list = []

    def emit(event, message="", *, level="info", **data):
        events.append({"event": event, "message": message, "level": level, "data": data})
        return len(events) - 1

    emit.run_id = "run-planner-prog"

    fn = _job_conversion(plan_job(h, "conversion"), {})
    fn(emit, h)

    frames = [e for e in events if e["event"] == "progress"]
    assert frames and all(
        isinstance(f["data"].get("done"), int) and isinstance(f["data"].get("total"), int)
        for f in frames
    )
    # Walk frames (per source per generation) vs acquire frames (per target).
    walk_frames = [f for f in frames if "generation" in f["data"]]
    acq_frames = [f for f in frames if "acquired" in f["data"]]
    assert len(walk_frames) == 3  # fake walk depth=2: gen1 the seed, gen2 its two refs
    assert [f["data"]["done"] for f in acq_frames] == list(range(1, len(acq_frames) + 1))
    assert all(f["data"]["total"] == len(acq_frames) for f in acq_frames)
    # Per-paper: every target work steps exactly once (detail=work_id in message).
    with Session(h.engine) as s:
        all_ids = {w.work_id for w in s.exec(select(Work)).all()}
    stepped = {f["message"].split(" — ")[1].split("  ")[0] for f in acq_frames}
    assert stepped == all_ids
    # The two coarse conversion breadcrumbs are unchanged.
    assert [e["event"] for e in events if e["event"] == "conversion"] == ["conversion", "conversion"]


def test_cli_corpus_verbs_heartbeat_but_no_progress_events(monkeypatch):
    """CLI corpus verbs keep the stdout N/M heartbeat but append NO events.jsonl
    events (``emit=None``): a CLI run must never leave a non-terminal ``progress``
    event that strands ``_status_from_events`` on ``running`` (the D-7 hazard)."""
    from typer.testing import CliRunner

    from seedgraph.cli import app
    from seedgraph.paths import project_dir
    from seedgraph.run import _status_from_events, read_events

    monkeypatch.setenv("SEEDGRAPH_FAKE_PROVIDERS", "1")
    h = service.create_project("cliprogress")
    service.add_work(h, ids={"arxiv": "2101.00001"}, title="Seed", is_seed=True)

    runner = CliRunner()
    res = runner.invoke(app, ["corpus", "run", "cliprogress", "--depth", "1"])
    assert res.exit_code == 0, res.output
    # Stdout heartbeat: the walk's 1/1 source frame + the acquire pass's N/M frames.
    assert "1/1 works" in res.output
    assert "3/3 works" in res.output

    # The standalone verbs heartbeat too (walk grows total; acquire reconciles it).
    res2 = runner.invoke(app, ["corpus", "acquire", "cliprogress"])
    assert res2.exit_code == 0, res2.output
    assert "3/3 works" in res2.output
    res3 = runner.invoke(app, ["corpus", "walk", "cliprogress", "--depth", "1"])
    assert res3.exit_code == 0, res3.output
    assert " works — " in res3.output

    # NO run of this project carries ANY event — let alone a stranding 'progress'.
    runs_dir = project_dir("cliprogress") / "runs"
    run_ids = [p.name for p in runs_dir.iterdir() if p.is_dir()]
    assert run_ids  # the verbs really did mint runs
    for rid in run_ids:
        events = read_events("cliprogress", rid)
        assert events == []
        assert _status_from_events(events) is None  # never 'running'


# --------------------------------------------------------------------------
# Build F chunk 10 — resolve-loop progress rider + invariant re-verification
# (the piece Build D chunk 6 left untouched: the per-item step in resolve_corpus,
#  built on D's exact Progress/emit hook signature. F must ACTIVELY re-verify the
#  event-silent CLI invariant risk 7's crash-sweep + chunk 9's interrupted-sweep
#  both rest on — never assume it.)
# --------------------------------------------------------------------------
def test_resolve_progress_steps_once_per_work():
    """The resolve-loop rider steps its ``progress`` exactly once per work with
    ``detail=work_id`` and the running rubric counters, reconciling the caller's
    placeholder total from the worklist (mirrors ``_acquire_async`` / ``walk_corpus``,
    D-7). Emit-backed here, so the N frames land in events.jsonl at ``done``/``total``
    = 1/N..N/N — the exact ``append_event`` shape the run page polls (acceptance 10)."""
    from seedgraph.acquisition.resolve import resolve_corpus
    from seedgraph.progress import Progress, make_emitter
    from seedgraph.run import ensure_run, read_events

    # Three works (seed included + two metadata_only), all strong-id (arXiv) hits
    # -> resolved deterministically with NO network (canonical_key short-circuit).
    h, seed_id, low_id, high_id = _budget_project("progresolve")
    rid = ensure_run("progresolve")
    emit = make_emitter("progresolve", rid, phase="conversion")
    prog = Progress(0, "works", emit=emit, echo=False)

    chain = _chain([MockProvider("openalex")])  # no maps needed — strong-id path
    report = asyncio.run(resolve_corpus(h, chain, run_id=rid, progress=prog))

    frames = [e for e in read_events("progresolve", rid) if e["event"] == "progress"]
    assert len(frames) == 3  # exactly one frame per work
    assert [f["data"]["done"] for f in frames] == [1, 2, 3]
    # The rider reconciled the caller's placeholder total (0) from its own select.
    assert all(f["data"]["total"] == 3 for f in frames)
    # Running rubric counter climbs as each strong-id work resolves.
    assert [f["data"]["resolved"] for f in frames] == [1, 2, 3]
    assert all(f["data"]["ambiguous"] == 0 and f["data"]["unresolved"] == 0 for f in frames)
    # detail=work_id rides the message; created_at order is low < high < seed.
    stepped = [f["message"].split(" — ")[1].split("  ")[0] for f in frames]
    assert stepped == [low_id, high_id, seed_id]
    assert report.resolved == 3


def test_resolve_no_hook_emits_nothing():
    """No hook (``progress=None``, the default) — and a stdout-only ``emit=None``
    Progress — append ZERO events.jsonl events: the rider is opt-in and event-silent
    unless an emit-backed Progress is passed."""
    from seedgraph.acquisition.resolve import resolve_corpus
    from seedgraph.progress import Progress
    from seedgraph.run import ensure_run, read_events

    h, *_ = _budget_project("progresolve_none")
    chain = _chain([MockProvider("openalex")])

    # (a) progress=None (default): the loop never steps.
    rid_none = ensure_run("progresolve_none")
    asyncio.run(resolve_corpus(h, chain, run_id=rid_none, progress=None))
    assert read_events("progresolve_none", rid_none) == []

    # (b) a stdout-only Progress (emit=None): steps, but nothing touches the fs.
    rid_stdout = ensure_run("progresolve_none")
    asyncio.run(
        resolve_corpus(
            h, chain, run_id=rid_stdout, progress=Progress(0, "works", echo=False)
        )
    )
    assert read_events("progresolve_none", rid_stdout) == []


def test_planner_conversion_progress_frames_land_in_events_jsonl(monkeypatch):
    """Re-verify (never assume) Build D chunk 6's mechanism END-TO-END: the planner
    conversion job driven by a REAL events.jsonl-backed emit writes its walk +
    acquire ``progress`` frames into ``runs/{run_id}/events.jsonl`` in the
    ``append_event`` shape the run page's existing poller reads — proving the frames
    persist to the substrate, not merely reach the emit callable (part b)."""
    from seedgraph.progress import make_emitter
    from seedgraph.run import ensure_run, read_events
    from seedgraph.web.planner import _job_conversion, plan_job

    monkeypatch.setenv("SEEDGRAPH_FAKE_PROVIDERS", "1")
    h = service.create_project("plannerland")
    service.add_work(h, ids={"arxiv": "2101.00001"}, title="Seed", is_seed=True)

    rid = ensure_run("plannerland")
    emit = make_emitter("plannerland", rid, phase="conversion")
    fn = _job_conversion(plan_job(h, "conversion"), {})
    fn(emit, h)

    frames = [e for e in read_events("plannerland", rid) if e["event"] == "progress"]
    assert frames  # the mechanism actually landed frames on disk
    assert all(
        isinstance(f["data"].get("done"), int) and isinstance(f["data"].get("total"), int)
        for f in frames
    )
    walk_frames = [f for f in frames if "generation" in f["data"]]
    acq_frames = [f for f in frames if "acquired" in f["data"]]
    assert walk_frames and acq_frames  # both D stages persisted their frames
    assert [f["data"]["done"] for f in acq_frames] == list(range(1, len(acq_frames) + 1))
    assert all(f["data"]["total"] == len(acq_frames) for f in acq_frames)


def test_cli_corpus_resolve_appends_zero_events(monkeypatch):
    """INVARIANT PIN: the CLI ``corpus resolve`` verb resolves with NO progress hook,
    so the resolve-loop rider never fires on the CLI path and NO event — least of all
    a stranding non-terminal ``progress`` — ever lands in events.jsonl. This is the
    invariant risk 7's accepted crash-sweep rationale AND chunk 9's interrupted-sweep
    safety both depend on (the sweep can only touch web-job runs precisely because CLI
    builds emit no job events); Build F re-verifies it rather than assuming it."""
    from typer.testing import CliRunner

    from seedgraph.cli import app
    from seedgraph.paths import project_dir
    from seedgraph.run import _status_from_events, read_events

    monkeypatch.setenv("SEEDGRAPH_FAKE_PROVIDERS", "1")
    h = service.create_project("cliresolve")
    service.add_work(h, ids={"arxiv": "2101.00001"}, title="Seed", is_seed=True)

    runner = CliRunner()
    res = runner.invoke(app, ["corpus", "resolve", "cliresolve"])
    assert res.exit_code == 0, res.output
    assert "resolved=" in res.output  # the verb really ran the resolve pass

    runs_dir = project_dir("cliresolve") / "runs"
    run_ids = [p.name for p in runs_dir.iterdir() if p.is_dir()]
    assert run_ids  # the verb minted a run
    for rid in run_ids:
        events = read_events("cliresolve", rid)
        assert events == []
        assert _status_from_events(events) is None  # never 'running'


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


def test_walk_enrichment_batches_per_source():
    """3 DOI-less refs on one source -> exactly ONE ``by_openalex_ids`` call and
    ZERO ``by_doi`` calls (the ch8 politeness fix), while every ref still folds
    DOI+title — the r2-1 §6 backfill preserved through the batch path."""
    h = service.create_project("enrichbatch")
    service.add_work(h, ids={"openalex": "W_S"}, title="Seed", is_seed=True)
    chain = _EnrichChain(_ENRICH_REFS, batch=list(_ENRICH_RECS.values()))
    report = WalkReport()

    with Session(h.engine, expire_on_commit=False) as s:
        seed = s.exec(select(Work).where(Work.openalex_id == "W_S")).one()
        cited = asyncio.run(
            expand_references(chain, s, source_work_id=seed.work_id, run_id="r", report=report)
        )
        s.commit()

    assert chain.batch_calls == [["W_A", "W_B", "W_C"]]  # ONE batched call per source
    assert chain.by_doi_calls == []  # zero in-band single-record calls
    assert len(cited) == 3
    assert report.enriched == 3 and report.enrichment_failed == 0
    with Session(h.engine) as s:
        for oa, rec in _ENRICH_RECS.items():
            w = s.exec(select(Work).where(Work.openalex_id == oa)).one()
            assert w.doi == rec["doi"]  # r2-1 §6 DOI backfill through the batch
            assert w.canonical_title == rec["title"]


def test_walk_enrichment_batch_miss_falls_back_to_by_doi():
    """A ref the batch did not return falls back to ``by_doi`` exactly ONCE —
    multi-provider recovery (and the negative-cache semantics for genuinely
    missing works) preserved for batch misses only."""
    h = service.create_project("enrichmiss")
    service.add_work(h, ids={"openalex": "W_S"}, title="Seed", is_seed=True)
    chain = _EnrichChain(
        _ENRICH_REFS,
        batch=[_ENRICH_RECS["W_A"], _ENRICH_RECS["W_B"]],  # W_C absent from the batch
        by_doi={"W_C": _ENRICH_RECS["W_C"]},
    )
    report = WalkReport()

    with Session(h.engine, expire_on_commit=False) as s:
        seed = s.exec(select(Work).where(Work.openalex_id == "W_S")).one()
        asyncio.run(
            expand_references(chain, s, source_work_id=seed.work_id, run_id="r", report=report)
        )
        s.commit()

    assert chain.batch_calls == [["W_A", "W_B", "W_C"]]
    assert chain.by_doi_calls == ["W_C"]  # ONLY the batch miss, exactly once
    assert report.enriched == 3 and report.enrichment_failed == 0
    with Session(h.engine) as s:
        w = s.exec(select(Work).where(Work.openalex_id == "W_C")).one()
        assert w.doi == "10.c" and w.canonical_title == "Paper C"


def test_walk_enrichment_failure_counted_and_untitled_residue():
    """Batch AND fallback both fail for one ref: ``enrichment_failed==1`` (a
    measured number, not the old silent swallow), the stub still upserts
    title-less (existing behavior preserved), and ``untitled_after_walk``
    counts it — all three land as additive keys of the walk manifest section."""
    from seedgraph import run as run_mod
    from seedgraph.paths import project_dir

    h = service.create_project("enrichfail")
    service.add_work(h, ids={"openalex": "W_S"}, title="Seed", is_seed=True)
    chain = _EnrichChain(
        _ENRICH_REFS,
        batch=[_ENRICH_RECS["W_A"], _ENRICH_RECS["W_B"]],  # W_C absent from the batch...
        by_doi_raises=True,  # ...and its per-ref fallback raises
    )
    rid = run_mod.ensure_run("enrichfail")

    report = asyncio.run(walk_corpus(h, chain, run_id=rid, depth=1))

    assert report.enriched == 2
    assert report.enrichment_failed == 1
    assert report.untitled_after_walk == 1
    with Session(h.engine) as s:
        c = s.exec(select(Work).where(Work.openalex_id == "W_C")).one()
        assert not c.canonical_title  # stub upserted title-less, as before

    manifest_path = project_dir("enrichfail") / "runs" / rid / "manifest.json"
    section = json.loads(manifest_path.read_text(encoding="utf-8"))["sections"]["walk"]
    assert section["enriched"] == 2
    assert section["enrichment_failed"] == 1
    assert section["untitled_after_walk"] == 1


def test_walk_enrichment_happy_path_counters():
    """Happy path through ``walk_corpus``: every DOI-less ref enriches from the
    batch (``enriched==N``), nothing fails, and the post-walk residue count is
    zero — the manifest's fully-titled-corpus signal."""
    h = service.create_project("enrichhappy")
    service.add_work(h, ids={"openalex": "W_S"}, title="Seed", is_seed=True)
    chain = _EnrichChain(_ENRICH_REFS, batch=list(_ENRICH_RECS.values()))

    report = asyncio.run(walk_corpus(h, chain, run_id="run-enrichhappy", depth=1))

    assert report.enriched == 3
    assert report.enrichment_failed == 0
    assert report.untitled_after_walk == 0
    assert chain.by_doi_calls == []
    section = report.to_section()
    assert section["enriched"] == 3 and section["untitled_after_walk"] == 0


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


def test_walk_ref_to_incoming_passes_triage_metadata_through():
    """The walk's ref->incoming mapping passes ``abstract``/``oa_status`` through
    to the identity fold — walk-discovered stubs ARE the unfetched papers the
    triage metadata exists for (a ``by_doi``-enriched ref carries both)."""
    from seedgraph.acquisition.walk import _ref_to_incoming

    incoming = _ref_to_incoming(
        {"openalex_id": "W1", "doi": "10.9/x", "title": "T",
         "abstract": "Ref abstract.", "oa_status": "green"}
    )
    assert incoming["abstract"] == "Ref abstract."
    assert incoming["oa_status"] == "green"
    # absent fields stay absent (empty-only downstream — nothing to fill).
    bare = _ref_to_incoming({"openalex_id": "W2"})
    assert "abstract" not in bare and "oa_status" not in bare


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


def test_abstract_oa_status_persist_empty_only_through_upsert():
    """Both fields persist through ``upsert_work`` with EMPTY-ONLY semantics: a
    second record never overwrites a present value; a later record fills a NULL."""
    from seedgraph.project import identity

    h = service.create_project("oaupsert")
    with Session(h.engine, expire_on_commit=False) as s:
        work, outcome = identity.upsert_work(
            s,
            {"doi": "10.3/abs", "title": "Abstracted",
             "abstract": "First abstract.", "oa_status": "green"},
        )
        s.commit()
        assert outcome == "created"
        assert work.abstract == "First abstract."
        assert work.oa_status == "green"

        # A second record for the SAME work never overwrites (empty-only).
        work2, outcome2 = identity.upsert_work(
            s,
            {"doi": "10.3/abs", "title": "Abstracted",
             "abstract": "Different abstract.", "oa_status": "closed"},
        )
        s.commit()
        assert outcome2 == "merged" and work2.work_id == work.work_id
        assert work2.abstract == "First abstract."
        assert work2.oa_status == "green"

        # Empty-only FILL: a work created without the fields gains them on merge
        # (the pre-ch10 cache-entry case: old shaped records lack the fields, the
        # next fresh fetch backfills them).
        bare, _ = identity.upsert_work(s, {"doi": "10.3/bare", "title": "Bare"})
        s.commit()
        assert bare.abstract is None and bare.oa_status is None
        filled, _ = identity.upsert_work(
            s, {"doi": "10.3/bare", "abstract": "Late abstract.", "oa_status": "bronze"}
        )
        s.commit()
        assert filled.abstract == "Late abstract." and filled.oa_status == "bronze"


def test_derive_acquisition_state_available_open_access_table():
    """Table test for the ch10 triage state: an UN-bridged, non-excluded work with
    an open OA color derives ``available_open_access`` (the previously dead vocab
    state — "OA exists, re-run/budget will get it"); ``closed``/absent colors keep
    the pre-ch10 states; excluded and bridged works never take the branch."""
    init_cache_db(None)
    h = service.create_project("oatriage")

    def _mk(doi: str, inclusion: str, oa: "str | None") -> str:
        work = service.add_work(h, ids={"doi": doi}, title=f"T {doi}", inclusion_status=inclusion)
        if oa is not None:
            with Session(h.engine, expire_on_commit=False) as s:
                row = s.get(Work, work.work_id)
                row.oa_status = oa
                s.add(row)
                s.commit()
        return work.work_id

    cases = [
        # (doi, inclusion, oa_status, expected state)
        ("10.1/gold", "included", "gold", "available_open_access"),
        ("10.1/green", "metadata_only", "green", "available_open_access"),
        ("10.1/hybrid", "metadata_only", "hybrid", "available_open_access"),
        # diamond: the deliberate superset over §4.3's list (a real OpenAlex value).
        ("10.1/diamond", "metadata_only", "diamond", "available_open_access"),
        ("10.1/closedinc", "included", "closed", "requires_user_upload"),
        ("10.1/closedmeta", "metadata_only", "closed", "metadata_only"),
        ("10.1/noneinc", "included", None, "requires_user_upload"),
        ("10.1/nonemeta", "metadata_only", None, "metadata_only"),
        ("10.1/excl", "excluded", "gold", "excluded"),
    ]
    wids = {doi: _mk(doi, inclusion, oa) for doi, inclusion, oa, _expected in cases}

    with Session(h.engine) as s:
        for doi, _inclusion, _oa, expected in cases:
            assert derive_acquisition_state(s, None, wids[doi]) == expected, doi

        # Bridged works never derive available_open_access, whatever the color:
        # a markdown-bearing bridge takes the reconcile path ('failed' against this
        # empty cache.db); a pdf-only bridge keeps the pre-ch10 no-markdown states.
        write_bridge(
            s, work_id=wids["10.1/gold"], source_file_id="sf_1", file_hash="fh_1",
            markdown_id="md_1", markdown_hash="mh_1",
            acquisition_method="open_access_fetch",
        )
        write_bridge(
            s, work_id=wids["10.1/green"], source_file_id="sf_2", file_hash="fh_2",
            acquisition_method="open_access_fetch",
        )
        s.commit()
        assert derive_acquisition_state(s, None, wids["10.1/gold"]) == "failed"
        assert derive_acquisition_state(s, None, wids["10.1/green"]) == "metadata_only"


def test_corpus_rows_emit_oa_status():
    """``corpus_rows`` emits the OA color (additive key) and the derived state
    reflects the ch10 triage split — the ch12 frontier pane's read-model."""
    from seedgraph.acquisition.service import corpus_rows

    h = service.create_project("oarows")
    w_open = service.add_work(
        h, ids={"doi": "10.2/open"}, title="OA row", inclusion_status="metadata_only"
    )
    service.add_work(
        h, ids={"doi": "10.2/plain"}, title="Plain row", inclusion_status="metadata_only"
    )
    with Session(h.engine, expire_on_commit=False) as s:
        row = s.get(Work, w_open.work_id)
        row.oa_status = "hybrid"
        s.add(row)
        s.commit()

    by_id = {r["work_id"]: r for r in corpus_rows(h)}
    assert by_id[w_open.work_id]["oa_status"] == "hybrid"
    assert by_id[w_open.work_id]["acquisition_state"] == "available_open_access"
    plain = next(r for r in by_id.values() if r["work_id"] != w_open.work_id)
    assert plain["oa_status"] is None
    assert plain["acquisition_state"] == "metadata_only"


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
