"""Provider cassette harness tests (Build F chunk 11).

Three offline layers plus the opt-in recorder:

* scrubber + replay unit tests (fully offline, ``RecordingTransport`` wrapping an
  ``httpx.MockTransport`` inner) proving query params redact, secret headers
  drop, replay serves same-key exchanges in recorded order, and an unrecorded
  request fails closed;
* an offline end-to-end proof that a real provider + Build-D retry consumes a
  recorded 429-then-200 same-key sequence faithfully;
* one replay test per shipped provider (OpenAlex, Crossref, Unpaywall, S2, CORE)
  that self-skips while its committed cassette is absent, so the chunk is green
  before the dev-machine recording session;
* the ``-m network`` recorder itself, deselected by default and a no-op unless
  ``SEEDGRAPH_VCR_RECORD=1``.
"""

from __future__ import annotations

import asyncio
import json
import os

import httpx
import pytest

from seedgraph.providers.core import CoreProvider
from seedgraph.providers.crossref import CrossrefProvider
from seedgraph.providers.openalex import OpenAlexProvider
from seedgraph.providers.semantic_scholar import SemanticScholarProvider
from seedgraph.providers.unpaywall import UnpaywallProvider
from support.cassette import (
    RecordingTransport,
    ReplayTransport,
    UnrecordedRequestError,
    cassette_path,
    recording_enabled,
)

# Snapshot the environment at IMPORT — before conftest's autouse ``isolated_home``
# strips API keys per-test — so the opt-in recorder can still reach BYO-key
# providers (CORE, and S2's higher rate limit) on the dev machine.
_IMPORT_ENV = dict(os.environ)

_EMAIL = "ci@example.com"
# A stable, long-published open-access DOI (the same one the phase_5b polite-pool
# smoke uses) so a recorded by_doi resolves to a real record on the dev machine.
_DOI = "10.7717/peerj.4375"


# --- shared scenarios (one per provider; used by BOTH replay and recorder) ---
def _scenarios():
    """(name, build(client) -> provider, call(provider) -> awaitable[record])."""
    return [
        (
            "openalex",
            lambda c: OpenAlexProvider(contact_email=_EMAIL, client=c),
            lambda p: p.by_doi(_DOI),
        ),
        (
            "crossref",
            lambda c: CrossrefProvider(contact_email=_EMAIL, client=c),
            lambda p: p.by_doi(_DOI),
        ),
        (
            "unpaywall",
            lambda c: UnpaywallProvider(contact_email=_EMAIL, client=c),
            lambda p: p.by_doi(_DOI),
        ),
        (
            "semantic_scholar",
            lambda c: SemanticScholarProvider(
                contact_email=_EMAIL, s2_api_key=_IMPORT_ENV.get("S2_API_KEY"), client=c
            ),
            lambda p: p.by_doi(_DOI),
        ),
        (
            "core",
            lambda c: CoreProvider(
                # A placeholder key on replay (the real Authorization header is
                # dropped, not part of the match key); the real key on record.
                core_key=_IMPORT_ENV.get("CORE_API_KEY") or "REPLAY",
                contact_email=_EMAIL,
                client=c,
            ),
            lambda p: p.by_title("Deep learning"),
        ),
    ]


_SCENARIOS = _scenarios()
_IDS = [s[0] for s in _SCENARIOS]


# ---------------------------------------------------------------------------
# Offline scrubber + replay unit tests
# ---------------------------------------------------------------------------
def test_recording_scrubs_query_and_drops_headers(tmp_path):
    """A request carrying ``mailto=`` + ``x-api-key``/``authorization`` records
    with the query param REDACTED and NO headers field at all (request headers
    are never recorded) — fully offline (RecordingTransport over a MockTransport
    inner)."""
    path = tmp_path / "scrub.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "https://openalex.org/W1", "ok": True})

    async def _drive():
        client = httpx.AsyncClient(transport=RecordingTransport(httpx.MockTransport(handler), path))
        try:
            return await client.get(
                "https://api.openalex.org/works/W1",
                params={"mailto": "secret@example.com", "filter": "x"},
                headers={"x-api-key": "supersecret", "authorization": "Bearer tok"},
            )
        finally:
            await client.aclose()

    resp = asyncio.run(_drive())
    # The downstream client still sees the untouched 200 body.
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    recorded = json.loads(path.read_text(encoding="utf-8"))
    assert len(recorded) == 1
    ex = recorded[0]
    assert "mailto=REDACTED" in ex["url"]
    assert "secret@example.com" not in ex["url"]
    assert "filter=x" in ex["url"]  # non-secret query param preserved
    assert "headers" not in ex  # request headers are never recorded at all
    assert set(ex) == {"method", "url", "status", "body"}

    # Whole-file grep (mirrors the acceptance pre-commit grep): no secret leaks.
    raw = path.read_text(encoding="utf-8")
    assert "supersecret" not in raw
    assert "secret@example.com" not in raw


def test_recorder_refuses_to_write_guarded_env_key_value(tmp_path, monkeypatch):
    """Bodies are unscrubbed by design, so the write path is the body-side net:
    a response body echoing a live S2_API_KEY value refuses to serialize."""
    monkeypatch.setenv("S2_API_KEY", "sk-live-cassette-guard-9x7")
    path = tmp_path / "leak.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"echo": "sk-live-cassette-guard-9x7"})

    async def _drive():
        client = httpx.AsyncClient(transport=RecordingTransport(httpx.MockTransport(handler), path))
        try:
            await client.get("https://api.semanticscholar.org/graph/v1/paper/x")
        finally:
            await client.aclose()

    with pytest.raises(RuntimeError, match="S2_API_KEY"):
        asyncio.run(_drive())
    assert not path.exists()  # nothing was written


def test_replay_round_trip_and_exhaustion(tmp_path):
    """Replay serves the recorded body; the mailto scrub collapses distinct live
    emails to one key; a second same-key request (only one recorded) fails closed."""
    path = tmp_path / "rt.json"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"doi": "10.1/x", "title": "T"})

    async def _record():
        client = httpx.AsyncClient(transport=RecordingTransport(httpx.MockTransport(handler), path))
        try:
            r = await client.get("https://api.crossref.org/works/10.1/x?mailto=me@x.org")
            return r.json()
        finally:
            await client.aclose()

    assert asyncio.run(_record())["doi"] == "10.1/x"

    async def _replay_twice():
        client = httpx.AsyncClient(transport=ReplayTransport(path))
        try:
            # Different live email -> same scrubbed key -> served.
            first = await client.get("https://api.crossref.org/works/10.1/x?mailto=other@y.org")
            with pytest.raises(UnrecordedRequestError):
                await client.get("https://api.crossref.org/works/10.1/x?mailto=other@y.org")
            return first.json()
        finally:
            await client.aclose()

    assert asyncio.run(_replay_twice()) == {"doi": "10.1/x", "title": "T"}


def test_replay_serves_same_key_exchanges_in_recorded_order(tmp_path):
    """Two same-key exchanges (429 then 200) replay in recorded order — the
    property that makes a Build-D retry sequence replay faithfully."""
    path = tmp_path / "seq.json"
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429 if calls["n"] == 1 else 200, json={"n": calls["n"]})

    url = "https://api.openalex.org/works/W1?mailto=me@x.org"

    async def _record():
        client = httpx.AsyncClient(transport=RecordingTransport(httpx.MockTransport(handler), path))
        try:
            await client.get(url)
            await client.get(url)
        finally:
            await client.aclose()

    asyncio.run(_record())

    async def _replay():
        client = httpx.AsyncClient(transport=ReplayTransport(path))
        try:
            first = await client.get(url)
            second = await client.get(url)
            return first.status_code, second.status_code
        finally:
            await client.aclose()

    assert asyncio.run(_replay()) == (429, 200)


def test_replay_raises_on_unrecorded_request(tmp_path):
    """Fail-closed: a request the cassette never saw raises (record_mode=none)."""
    path = tmp_path / "one.json"
    path.write_text(
        json.dumps(
            [
                {
                    "method": "GET",
                    "url": "https://api.openalex.org/works/W1?mailto=REDACTED",
                    "status": 200,
                    "body": "{\"ok\": true}",
                }
            ]
        ),
        encoding="utf-8",
    )

    async def _drive():
        client = httpx.AsyncClient(transport=ReplayTransport(path))
        try:
            await client.get("https://api.openalex.org/works/W999?mailto=REDACTED")
        finally:
            await client.aclose()

    with pytest.raises(UnrecordedRequestError):
        asyncio.run(_drive())


def test_provider_retry_replays_same_key_sequence(tmp_path, monkeypatch):
    """End-to-end offline: a recorded 429-then-200 same-key sequence, replayed
    through the REAL OpenAlex provider + Build-D transient retry, yields the
    shaped record and consumes both exchanges (retry-faithful replay)."""
    from seedgraph.providers import _retry

    sleeps: list = []

    async def _fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(_retry, "_sleep", _fake_sleep)

    work = {"id": "https://openalex.org/W1", "title": "Paper X", "publication_year": 2020}
    path = tmp_path / "openalex_seq.json"
    path.write_text(
        json.dumps(
            [
                {
                    "method": "GET",
                    "url": "https://api.openalex.org/works/doi:10.1/x",
                    "status": 429,
                    "body": "{}",
                },
                {
                    "method": "GET",
                    "url": "https://api.openalex.org/works/doi:10.1/x",
                    "status": 200,
                    "body": json.dumps(work),
                },
            ]
        ),
        encoding="utf-8",
    )

    async def _drive():
        client = httpx.AsyncClient(transport=ReplayTransport(path))
        try:
            return await OpenAlexProvider(client=client).by_doi("10.1/x")
        finally:
            await client.aclose()

    out = asyncio.run(_drive())
    assert out and out["title"] == "Paper X"  # replayed body parsed by the real shaper
    assert len(sleeps) == 1  # exactly one backoff between the 429 and the 200


# ---------------------------------------------------------------------------
# Per-provider replay (self-skips until the committed cassette exists)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name, build, call", _SCENARIOS, ids=_IDS)
def test_provider_cassette_replays(name, build, call):
    path = cassette_path(name)
    if not path.exists():
        pytest.skip(
            f"cassette {path.name} not recorded yet — run "
            f"`SEEDGRAPH_VCR_RECORD=1 pytest -m network tests/test_provider_cassettes.py`"
        )

    async def _drive():
        client = httpx.AsyncClient(transport=ReplayTransport(path))
        try:
            return await call(build(client))
        finally:
            await client.aclose()

    record = asyncio.run(_drive())
    assert record is not None and isinstance(record, dict), (
        f"{name}: replaying the committed cassette must yield a shaped record "
        f"(a break here is provider-shape drift the MockTransport doubles miss); "
        f"got {record!r}"
    )


# ---------------------------------------------------------------------------
# The opt-in recorder (deselected by default; no-op unless SEEDGRAPH_VCR_RECORD=1)
# ---------------------------------------------------------------------------
@pytest.mark.network
@pytest.mark.parametrize("name, build, call", _SCENARIOS, ids=_IDS)
def test_record_provider_cassette(name, build, call):
    if not recording_enabled():
        pytest.skip("recording is opt-in: set SEEDGRAPH_VCR_RECORD=1 to (re)record")

    async def _drive():
        client = httpx.AsyncClient(
            transport=RecordingTransport(httpx.AsyncHTTPTransport(), cassette_path(name))
        )
        try:
            return await call(build(client))
        finally:
            await client.aclose()

    record = asyncio.run(_drive())
    # If this is None the fixture input didn't resolve live (or the provider had
    # no key and issued no request) — pick a resolving DOI/title / provide a key.
    assert record is not None, f"{name}: recorded call did not resolve to a record"
    assert cassette_path(name).exists()
