"""Build D ch9 — post-walk title backfill (§5.10): offline read-repair tests.

Mirrors v1's ``tests/test_title_backfill.py`` — deliberately NOT
``tests/test_metadata_backfill.py``, which is the LLM metadata suite for
``extraction/metadata.py``. Everything here is fully OFFLINE: an injected fake
batch provider rides a REAL :class:`ProviderChain` over the isolated test-home
``provider_cache`` (so the load-bearing cache-hygiene assertions exercise the
real §6.5 key namespaces), no network, no LLM.

Covers the ch9 contract:

* untitled W-id stub + fake batch provider -> title filled IN PLACE
  (``filled == 1``, same ``work_id``, no duplicate row);
* second run -> ``queried == 0`` and ZERO provider calls (provably idempotent);
* provider returns nothing -> ``still_missing == 1`` and the work unchanged
  (and, D-8, the empty batch is never negative-cached — the next pass re-asks);
* an already-titled work is never selected or overwritten;
* a NEGATIVE ``by_doi`` cache entry for the same W-id does not block the
  backfill (the load-bearing distinct-namespace assertion);
* the empty-STRING title form is selected and filled (the v1 lesson);
* ceil(N/50) batched calls for N > 50 untitled ids;
* ``run_corpus`` runs the pass fail-soft between walk and acquire and writes
  the additive ``title_backfill`` manifest section once;
* CLI smoke: ``seedgraph corpus backfill-titles <slug>`` prints the four
  counters.
"""

from __future__ import annotations

import asyncio
import json
import math

from sqlmodel import Session, select

from seedgraph.acquisition.backfill import BackfillResult, backfill_titles
from seedgraph.cache.provider_cache import cache_get, cache_put, key_by_doi
from seedgraph.db.connection import open_cache_db
from seedgraph.db.project_models import Work
from seedgraph.project import service
from seedgraph.providers.base import ProviderChain


# --------------------------------------------------------------------------
# Offline fake provider — exposes ONLY the batched backfill verb.
# --------------------------------------------------------------------------
class _RecordingProvider:
    """Backfill provider double (mirrors v1's ``_RecordingProvider``).

    * Returns a titled record for every known W-id in a requested batch and
      records each ``by_openalex_ids`` id list in :attr:`batches` so a test can
      assert ``ceil(N / batch_size)`` calls and the exact partition.
    * ``by_doi`` RAISES if ever called — the backfill must ride the DISTINCT
      batched namespace and NEVER fall back to the (negative-cached) ``by_doi``
      route. A green test passes the raise-on-touch guard only because that
      verb is never invoked.
    """

    name = "openalex"

    def __init__(self, titles: dict[str, dict]) -> None:
        self._titles = dict(titles)
        self.batches: list[list[str]] = []

    async def by_openalex_ids(self, ids: list[str]) -> list[dict]:
        self.batches.append(list(ids))
        out: list[dict] = []
        for wid in ids:
            extra = self._titles.get(wid)
            if extra is not None:
                rec = {"openalex_id": wid}
                rec.update(extra)
                out.append(rec)
        return out

    async def by_doi(self, doi):  # pragma: no cover - the guard itself
        raise AssertionError(
            "backfill must use the batched by_openalex_ids namespace, not by_doi"
        )


def _chain(provider, **kw) -> ProviderChain:
    kw.setdefault("cache_engine", None)  # None -> the isolated SEEDGRAPH_HOME cache.db
    return ProviderChain([provider], **kw)


def _seed_untitled(h, wid: str) -> str:
    """Insert one untitled-with-W-id work (the walk-stub shape); return work_id."""
    work = service.add_work(
        h, ids={"openalex": wid},
        inclusion_status="metadata_only", inclusion_reason="citation_walk",
    )
    assert work.openalex_id == wid
    assert not (work.canonical_title or "").strip(), "fixture must start title-less"
    return work.work_id


def _works(h) -> list:
    with Session(h.engine) as s:
        return list(s.exec(select(Work)).all())


def _get_work(h, work_id: str):
    with Session(h.engine) as s:
        return s.get(Work, work_id)


# ==========================================================================
# Fill in place, no duplicate, idempotent re-run
# ==========================================================================
def test_backfill_fills_same_row_no_dup_and_is_idempotent():
    """An untitled W-id stub is filled IN PLACE (same work_id, no duplicate) and
    a second pass is a no-op: ``queried == 0`` and zero provider calls."""
    h = service.create_project("backfillfill")
    wid = "W2741809807"
    work_id = _seed_untitled(h, wid)
    assert len(_works(h)) == 1

    provider = _RecordingProvider(
        {
            wid: {
                "title": "Heterogeneous Treatment Effects in Panel Data",
                "year": 2021,
                "authors": ["Pedro H. C. Sant'Anna", "Brantly Callaway"],
            }
        }
    )
    chain = _chain(provider)

    result = asyncio.run(backfill_titles(h, chain))
    assert isinstance(result, BackfillResult)
    assert (result.queried, result.fetched, result.filled, result.still_missing) == (1, 1, 1, 0)
    # The manifest section shape is pinned to exactly the four counters.
    assert result.to_section() == {
        "queried": 1, "fetched": 1, "filled": 1, "still_missing": 0,
    }

    works = _works(h)
    assert len(works) == 1, "backfill must fill IN PLACE, never insert a duplicate"
    filled = works[0]
    assert filled.work_id == work_id, "work_id must be unchanged by the backfill"
    assert filled.openalex_id == wid
    assert filled.canonical_title == "Heterogeneous Treatment Effects in Panel Data"
    assert filled.year == 2021
    assert filled.authors == ["Pedro H. C. Sant'Anna", "Brantly Callaway"]

    # Second pass: idempotent — the now-titled row is no longer selected, so no
    # batch is issued at all (provably queried == 0, not merely re-filled).
    provider.batches.clear()
    rerun = asyncio.run(backfill_titles(h, chain))
    assert (rerun.queried, rerun.fetched, rerun.filled, rerun.still_missing) == (0, 0, 0, 0)
    assert provider.batches == [], "idempotent re-run must not issue a provider call"
    again = _get_work(h, work_id)
    assert again.canonical_title == "Heterogeneous Treatment Effects in Panel Data"
    assert len(_works(h)) == 1


def test_backfill_never_overwrites_an_existing_title():
    """An already-titled W-id work is never even selected (queried == 0), so a
    mischievous provider that WOULD return a different title is never asked."""
    h = service.create_project("backfillkeep")
    wid = "W3000000001"
    good = service.add_work(h, ids={"openalex": wid}, title="The Real Stored Title", year=2019)

    provider = _RecordingProvider({wid: {"title": "A WRONG Replacement Title"}})
    chain = _chain(provider)

    result = asyncio.run(backfill_titles(h, chain))
    assert result.queried == 0, "an already-titled row is not in the backfill universe"
    assert provider.batches == [], "no batch is issued when nothing is untitled"

    after = _get_work(h, good.work_id)
    assert after.canonical_title == "The Real Stored Title"
    assert len(_works(h)) == 1


def test_backfill_excludes_parked_duplicate_that_does_not_own_its_wid():
    """A parked duplicate_candidate MIRRORS a W-id whose ``identifiers`` row
    another work owns; the fold resolves through ``identifiers``, so the pass
    could never fill that row — it is excluded from the selection up front.
    The owning row is filled in place, the parked row is untouched, and the
    re-run still reports ``queried == 0`` with zero provider calls (the
    module-docstring idempotence claim survives a parked duplicate)."""
    from seedgraph.db.project_models import Identifier

    h = service.create_project("backfillparked")
    wid = "W4300000007"
    owner = service.add_work(
        h, ids={"openalex": wid, "doi": "10.1/owner"},
        inclusion_status="metadata_only", inclusion_reason="citation_walk",
    )
    # Same W-id, conflicting DOI -> upsert demotes to duplicate_candidate: the
    # parked row mirrors wid on its scalar column but the identifiers row for
    # ("openalex", wid) stays owned by `owner`.
    parked = service.add_work(
        h, ids={"openalex": wid, "doi": "10.1/parked"},
        inclusion_status="metadata_only", inclusion_reason="citation_walk",
    )
    assert parked.work_id != owner.work_id
    assert parked.openalex_id == wid and not (parked.canonical_title or "").strip()
    with Session(h.engine) as s:
        row = s.exec(
            select(Identifier).where(
                Identifier.id_type == "openalex", Identifier.id_value == wid
            )
        ).one()
        assert row.work_id == owner.work_id, "fixture: owner must own the W-id"

    provider = _RecordingProvider({wid: {"title": "The Owner Title", "year": 2022}})
    chain = _chain(provider)

    # Only the OWNING untitled row is in the universe: queried == 1, not 2.
    result = asyncio.run(backfill_titles(h, chain))
    assert (result.queried, result.fetched, result.filled, result.still_missing) == (1, 1, 1, 0)
    assert provider.batches == [[wid]], "the shared W-id is fetched exactly once"
    assert _get_work(h, owner.work_id).canonical_title == "The Owner Title"
    still_parked = _get_work(h, parked.work_id)
    assert still_parked.canonical_title is None, "the parked row is not this pass's to fill"
    assert len(_works(h)) == 2, "the fold must not mint a third row"

    # Idempotent DESPITE the untitled parked row: it never enters the
    # selection, so the re-run is queried == 0 and issues no provider call.
    provider.batches.clear()
    rerun = asyncio.run(backfill_titles(h, chain))
    assert (rerun.queried, rerun.fetched, rerun.filled, rerun.still_missing) == (0, 0, 0, 0)
    assert provider.batches == []


def test_backfill_provider_returns_nothing_counts_still_missing():
    """A W-id the provider cannot title stays exactly as it was and is counted
    ``still_missing`` — and (D-8) the empty batch is never negative-cached, so
    the NEXT pass asks the provider again instead of starving for 30 days."""
    h = service.create_project("backfillmiss")
    wid = "W4100000001"
    work_id = _seed_untitled(h, wid)

    provider = _RecordingProvider({})  # knows no titles -> empty batch results
    chain = _chain(provider)

    result = asyncio.run(backfill_titles(h, chain))
    assert (result.queried, result.fetched, result.filled, result.still_missing) == (1, 0, 0, 1)
    unchanged = _get_work(h, work_id)
    assert unchanged.canonical_title is None, "a missed work must be left unchanged"
    assert len(_works(h)) == 1

    # The empty batch was NOT written as a cached negative: the second pass
    # issues a fresh provider call for the same ids (the backfill universe is
    # never starved by its own misses — D-8).
    result2 = asyncio.run(backfill_titles(h, chain))
    assert result2.still_missing == 1
    assert provider.batches == [[wid], [wid]]


def test_backfill_fills_empty_string_title_too():
    """The v1 lesson: a once-NULL title persisted as ``""`` by an intermediate
    upsert is still "untitled" — selected AND fillable (the ch9 identity fold
    treats blank-only as empty; a present title is still never overwritten)."""
    h = service.create_project("backfillblank")
    wid = "W4150000002"
    work_id = _seed_untitled(h, wid)
    with Session(h.engine, expire_on_commit=False) as s:
        w = s.get(Work, work_id)
        w.canonical_title = ""  # the intermediate-upsert artifact
        s.add(w)
        s.commit()

    provider = _RecordingProvider({wid: {"title": "Recovered From Blank", "year": 2015}})
    chain = _chain(provider)

    result = asyncio.run(backfill_titles(h, chain))
    assert (result.queried, result.fetched, result.filled, result.still_missing) == (1, 1, 1, 0)
    assert _get_work(h, work_id).canonical_title == "Recovered From Blank"


# ==========================================================================
# Batching: ceil(N/50) calls, never N
# ==========================================================================
def test_backfill_batches_in_ceil_n_over_50_calls():
    """105 untitled W-ids drive exactly ceil(105/50)=3 batched calls (NOT 105),
    each batch <= 50 ids, covering every id exactly once — all filled."""
    h = service.create_project("backfillbatch")
    n = 105
    wids = [f"W{1000 + i:07d}" for i in range(n)]
    for wid in wids:
        _seed_untitled(h, wid)
    assert len(_works(h)) == n

    provider = _RecordingProvider(
        {wid: {"title": f"Title for {wid}", "year": 2020} for wid in wids}
    )
    chain = _chain(provider)

    result = asyncio.run(backfill_titles(h, chain, batch_size=50))

    assert len(provider.batches) == math.ceil(n / 50) == 3
    assert all(len(b) <= 50 for b in provider.batches), "no batch may exceed batch_size"
    seen = [wid for batch in provider.batches for wid in batch]
    assert sorted(seen) == sorted(wids), "batches must cover every W-id exactly once"

    assert (result.queried, result.fetched, result.filled, result.still_missing) == (n, n, n, 0)
    works = _works(h)
    assert len(works) == n
    assert all((w.canonical_title or "").strip() for w in works)


# ==========================================================================
# THE load-bearing cache-hygiene assertion: a negative by_doi entry for the
# same W-id must never pin the untitled state (distinct ch7 namespace).
# ==========================================================================
def test_backfill_negative_by_doi_cache_does_not_block():
    """Poison the cache exactly the way a live walk does — the per-ref by_doi
    fallback missed and ``_dispatch`` stored a 30-day NEGATIVE under ``by_doi:``
    for the W-id — then prove the backfill still fills the title: it fetches via
    the DISTINCT ``by_openalex_ids:`` namespace (never touching ``by_doi``; the
    provider raises if it did) and leaves the stale negative row untouched."""
    from seedgraph.cache.db import init_cache_db

    h = service.create_project("backfillcache")
    wid = "W4200000042"
    work_id = _seed_untitled(h, wid)

    init_cache_db(None)
    conn = open_cache_db(None)
    try:
        cache_put(conn, provider="openalex", request_key=key_by_doi(wid), response=[])
    finally:
        conn.close()

    provider = _RecordingProvider({wid: {"title": "A Recovered Real Title", "year": 2018}})
    chain = _chain(provider)

    result = asyncio.run(backfill_titles(h, chain))

    assert result.filled == 1, "the negative by_doi cache entry must not block the backfill"
    assert provider.batches == [[wid]], "fetch must go through the batched namespace"
    assert _get_work(h, work_id).canonical_title == "A Recovered Real Title"

    # The stale by_doi negative is bypassed, not rewritten: it still stores [].
    conn = open_cache_db(None)
    try:
        assert cache_get(conn, provider="openalex", request_key=key_by_doi(wid)) == []
    finally:
        conn.close()


# ==========================================================================
# run_corpus integration: fail-soft pass between walk and acquire + manifest
# ==========================================================================
class _WalkLosesTitlesProvider:
    """The exact §5.10 sequence ch9 repairs, end-to-end: one DOI-less ref whose
    batch prefetch AND ``by_doi`` fallback both miss during the walk (the stub
    lands untitled and the miss IS negative-cached under ``by_doi:``), then a
    recovered batch afterwards. Because the walk's empty batch was never
    negative-cached (D-8), the backfill's identical batch re-asks and fills."""

    name = "openalex"

    def __init__(self) -> None:
        self.batch_calls: list[list[str]] = []

    async def by_doi(self, doi):
        if str(doi) == "10.s/seed":
            return {"doi": "10.s/seed", "openalex_id": "W_SEED", "title": "Seed"}
        return None  # the W-id fallback miss -> _dispatch caches a negative

    async def by_title(self, title, year=None):
        return None

    async def referenced_works(self, record):
        return [{"openalex_id": "W_LOST"}]

    async def by_openalex_ids(self, ids):
        self.batch_calls.append(list(ids))
        if len(self.batch_calls) == 1:
            return []  # the walk-time prefetch outage: the title is "lost"
        return [{"openalex_id": wid, "title": f"Recovered {wid}"} for wid in ids]

    async def oa_pdf_candidates(self, record):
        return []


def test_run_corpus_backfills_between_walk_and_acquire():
    """``run_corpus`` repairs a walk-lost title before the acquire pass and
    writes the four counters ONCE as the additive ``title_backfill`` manifest
    section — even though the walk's fallback pinned a by_doi NEGATIVE for the
    same W-id (the in-vivo proof of the ch7 distinct-namespace design)."""
    from seedgraph.acquisition.service import run_corpus
    from seedgraph.paths import project_dir

    h = service.create_project("backfillrun")
    service.add_work(h, ids={"doi": "10.s/seed"}, title="Seed Title", is_seed=True)

    provider = _WalkLosesTitlesProvider()
    chain = _chain(provider)
    out = run_corpus(h, chain, depth=1, per_gen_cap=10)

    # Walk lost the title (measured, ch8) …
    assert out["walk"].untitled_after_walk == 1
    # … the walk's per-ref fallback really did pin a 30-day by_doi negative …
    conn = open_cache_db(None)
    try:
        assert cache_get(conn, provider="openalex", request_key=key_by_doi("W_LOST")) == []
    finally:
        conn.close()
    # … and the ch9 pass repaired it anyway, through its own namespace: the
    # walk's empty batch (call 1) was never cached, so the backfill (call 2)
    # re-asked the SAME ids and filled the row.
    assert provider.batch_calls == [["W_LOST"], ["W_LOST"]]
    tb = out["title_backfill"]
    assert (tb.queried, tb.fetched, tb.filled, tb.still_missing) == (1, 1, 1, 0)
    with Session(h.engine) as s:
        lost = s.exec(select(Work).where(Work.openalex_id == "W_LOST")).one()
        assert lost.canonical_title == "Recovered W_LOST"

    # The additive manifest section, written once, exactly the four counters;
    # the neighbouring stage sections are untouched (D5 disjoint ownership).
    manifest_path = project_dir("backfillrun") / "runs" / out["run_id"] / "manifest.json"
    sections = json.loads(manifest_path.read_text(encoding="utf-8"))["sections"]
    assert sections["title_backfill"] == {
        "queried": 1, "fetched": 1, "filled": 1, "still_missing": 0,
    }
    assert "walk" in sections and "acquisition" in sections and "resolution" in sections


def test_run_corpus_backfill_is_fail_soft(monkeypatch):
    """A raising backfill pass must not cost the run its acquire stage: the run
    completes, ``title_backfill`` is ``None`` (and its manifest section absent),
    and the acquisition section is still written."""
    from seedgraph.acquisition import service as service_mod
    from seedgraph.paths import project_dir

    h = service.create_project("backfillsoft")
    service.add_work(h, ids={"doi": "10.s/seed"}, title="Seed Title", is_seed=True)

    async def _boom(h2, chain2, **kw):
        raise RuntimeError("simulated backfill outage")

    monkeypatch.setattr(service_mod, "backfill_titles", _boom)
    chain = _chain(_WalkLosesTitlesProvider())
    out = service_mod.run_corpus(h, chain, depth=1, per_gen_cap=10)

    assert out["title_backfill"] is None
    assert out["acquisition"] is not None  # acquire still ran after the failure

    manifest_path = project_dir("backfillsoft") / "runs" / out["run_id"] / "manifest.json"
    sections = json.loads(manifest_path.read_text(encoding="utf-8"))["sections"]
    assert "title_backfill" not in sections
    assert "acquisition" in sections and "walk" in sections


# ==========================================================================
# CLI smoke: `seedgraph corpus backfill-titles <slug>` prints the counters
# ==========================================================================
def test_cli_backfill_titles_prints_counters(monkeypatch):
    """The verb round-trips offline under SEEDGRAPH_FAKE_PROVIDERS and prints
    the four counters; the untitled W-id work really is titled by the pass."""
    from typer.testing import CliRunner

    from seedgraph.cli import app

    monkeypatch.setenv("SEEDGRAPH_FAKE_PROVIDERS", "1")
    h = service.create_project("clibackfill")
    work_id = _seed_untitled(h, "W12345")

    runner = CliRunner()
    res = runner.invoke(app, ["corpus", "backfill-titles", "clibackfill"])
    assert res.exit_code == 0, res.output
    assert "queried=1" in res.output
    assert "fetched=1" in res.output
    assert "filled=1" in res.output
    assert "still_missing=0" in res.output
    assert _get_work(h, work_id).canonical_title == "Referenced work W12345"

    # Idempotent from the CLI too.
    res2 = runner.invoke(app, ["corpus", "backfill-titles", "clibackfill"])
    assert res2.exit_code == 0, res2.output
    assert "queried=0" in res2.output and "still_missing=0" in res2.output
