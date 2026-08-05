"""Record-once / replay-offline provider cassette harness (Build F chunk 11).

TEST-TREE ONLY — this module lives under ``tests/`` and is NEVER shipped in the
wheel. It gives the provider layer a committed record/replay cassette at the
existing injectable-client seam (``providers/*`` construct ``httpx.AsyncClient``
objects and accept an injected ``client=``), so committed cassettes catch
provider-shape drift that the hand-authored ``httpx.MockTransport`` doubles
cannot. This is deliberately NOT vcrpy: the injectable-client seam makes a small
``httpx.AsyncBaseTransport`` pair sufficient, and the only part of the v1
prototype worth porting verbatim is the scrub-filter list.

Two transports, both ``httpx.AsyncBaseTransport`` (providers await
``client.get`` on an ``httpx.AsyncClient``, so a sync ``httpx.BaseTransport``
cannot be injected at this seam — the async method is the one that runs):

* :class:`RecordingTransport` wraps a REAL transport, forwards every request,
  and appends a scrubbed ``{method, url, status, body}`` exchange to a JSON
  cassette. Recording is opt-in: a caller only builds one when
  :func:`recording_enabled` (``SEEDGRAPH_VCR_RECORD=1``) under ``pytest -m
  network``.
* :class:`ReplayTransport` serves recorded exchanges matched on
  ``(method, scheme, host, path, sorted scrubbed query)`` and raises
  :class:`UnrecordedRequestError` on any unrecorded request — vcrpy
  ``record_mode="none"`` semantics (offline, fail-closed). Same-key exchanges
  are served in RECORDED ORDER, so a Build-D retry (a 429-then-200 sequence
  recorded as two same-key exchanges) replays faithfully.

Scrub filters: query params ``mailto,email,api_key,apikey`` are redacted to
``REDACTED`` (list ported verbatim from v1 ``tests/providers/vcr_config.py``).
Request headers are NOT RECORDED AT ALL — :class:`ReplayTransport` never reads
them, and recording nothing is strictly safer than dropping a pinned name list
(a future provider's unlisted auth header cannot leak a value that is never
written). The write path additionally refuses to serialize a cassette whose
text contains a live ``S2_API_KEY``/``CORE_API_KEY`` value (response bodies are
otherwise unscrubbed by design). The committed cassettes must still pass the
acceptance grep (``grep -ri "mailto=\\|x-api-key\\|authorization"`` matches only
``REDACTED``) before commit — CI runs the same grep as a backstop.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

#: Directory holding committed cassettes (offline replay). Recorded on the dev
#: machine under ``pytest -m network``; committed after the scrub grep.
CASSETTE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "provider_cassettes"

#: Query-string params scrubbed from every recorded request URL — the polite-pool
#: email and provider keys may ride the query string (OpenAlex/Crossref/Unpaywall/S2).
FILTER_QUERY_PARAMS = ("mailto", "email", "api_key", "apikey")

#: Env vars whose live values must never appear in a serialized cassette
#: (auth rides headers for these providers; bodies are unscrubbed by design).
GUARDED_ENV_KEYS = ("S2_API_KEY", "CORE_API_KEY")

#: Placeholder substituted for any scrubbed query value.
REDACTED = "REDACTED"


def recording_enabled() -> bool:
    """``True`` iff ``SEEDGRAPH_VCR_RECORD=1`` — the opt-in dev-machine record mode.

    Recording only ever runs opt-in, under ``pytest -m network`` with live
    network; the default offline suite replays committed cassettes.
    """
    return os.environ.get("SEEDGRAPH_VCR_RECORD") == "1"


def cassette_path(name: str) -> Path:
    """Return the committed cassette path for provider ``name``."""
    return CASSETTE_DIR / f"{name}.json"


class UnrecordedRequestError(RuntimeError):
    """Raised by :class:`ReplayTransport` for a request with no recorded match.

    Fail-closed replay (``record_mode="none"``): a request the cassette never
    saw is an error, never a silent network fall-through.
    """


# --- scrubbing --------------------------------------------------------------
def _scrub_query(params: httpx.QueryParams) -> httpx.QueryParams:
    """Redact the pinned secret query params to ``REDACTED`` (idempotent)."""
    out = params
    for name in FILTER_QUERY_PARAMS:
        if name in out:
            out = out.set(name, REDACTED)
    return out


def _scrub_url(url: httpx.URL) -> httpx.URL:
    """Return ``url`` with its secret query params redacted."""
    scrubbed = _scrub_query(url.params)
    return url.copy_with(query=str(scrubbed).encode("utf-8"))


def _match_key(method: str, url: httpx.URL) -> tuple:
    """The replay match key: ``(method, scheme, host, path, sorted scrubbed query)``.

    Scrubbing is applied to both sides (recorded URL and live request), so a
    live ``mailto=me@x.org`` and its recorded ``mailto=REDACTED`` collapse to
    the same key. Port is deliberately excluded (per the proposal's match tuple).
    """
    scrubbed = _scrub_url(url)
    return (
        method.upper(),
        scrubbed.scheme,
        scrubbed.host,
        scrubbed.path,
        tuple(sorted(scrubbed.params.multi_items())),
    )


# --- recording --------------------------------------------------------------
class RecordingTransport(httpx.AsyncBaseTransport):
    """Forwarding transport that appends scrubbed exchanges to a JSON cassette.

    Wraps a real ``inner`` transport (e.g. ``httpx.AsyncHTTPTransport()``);
    every request is forwarded, the response body is buffered, and a scrubbed
    ``{method, url, status, body}`` exchange is appended to ``path``. Request
    headers are never recorded (replay does not read them).
    The full cassette is rewritten on each exchange so the file is always
    complete without a teardown hook, and a run that never issues a request
    never creates the file (so a keyless provider's absent cassette self-skips
    replay). The response returned downstream is a fresh buffered copy with the
    decoded body and no ``content-encoding`` (so the client never double-decodes).
    """

    def __init__(self, inner: httpx.AsyncBaseTransport, path: str | os.PathLike) -> None:
        self._inner = inner
        self._path = Path(path)
        self._exchanges: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        await response.aread()
        body = response.content  # decoded (content decoders already applied)
        await response.aclose()

        self._exchanges.append(
            {
                "method": request.method.upper(),
                "url": str(_scrub_url(request.url)),
                "status": response.status_code,
                "body": body.decode("utf-8", errors="replace"),
            }
        )
        self._write()

        return httpx.Response(
            status_code=response.status_code,
            content=body,
            request=request,
        )

    def _write(self) -> None:
        text = json.dumps(self._exchanges, indent=2, ensure_ascii=False) + "\n"
        for env_key in GUARDED_ENV_KEYS:
            value = os.environ.get(env_key)
            if value and value in text:
                raise RuntimeError(
                    f"refusing to write cassette {self._path.name}: the live value of "
                    f"{env_key} appears in the serialized exchanges (a response body "
                    "echoed the key?). Scrub the exchange or unset the key and re-record."
                )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(text, encoding="utf-8")

    async def aclose(self) -> None:
        await self._inner.aclose()


# --- replay -----------------------------------------------------------------
class ReplayTransport(httpx.AsyncBaseTransport):
    """Offline transport serving recorded exchanges; raises on an unrecorded one.

    Loads the cassette at ``path``, indexes exchanges by :func:`_match_key` in
    RECORDED ORDER, and serves each once. Repeated same-key requests consume the
    recorded exchanges front-to-back — so a Build-D retry (429 recorded first,
    200 second) replays faithfully. An exhausted or absent key raises
    :class:`UnrecordedRequestError` (``record_mode="none"``).
    """

    def __init__(self, path: str | os.PathLike) -> None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self._remaining: dict[tuple, list[dict[str, Any]]] = {}
        for exchange in raw:
            key = _match_key(exchange["method"], httpx.URL(exchange["url"]))
            self._remaining.setdefault(key, []).append(exchange)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        key = _match_key(request.method, request.url)
        queue = self._remaining.get(key)
        if not queue:
            raise UnrecordedRequestError(
                "no recorded cassette exchange for "
                f"{request.method} {str(_scrub_url(request.url))!r} "
                "(replay is fail-closed; re-record with SEEDGRAPH_VCR_RECORD=1 "
                "under `pytest -m network`)"
            )
        exchange = queue.pop(0)
        return httpx.Response(
            status_code=exchange["status"],
            content=exchange["body"].encode("utf-8"),
            request=request,
        )
