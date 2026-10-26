"""Localhost-only, CSRF-protected, cookie-free serving (docs/V3_UI_DESIGN.md §8).

Three pure ASGI middlewares, applied outermost first by :func:`secure`:

1. :class:`SecurityHeaders` - adds the CSP, ``nosniff`` and ``no-referrer`` headers to
   every response (refusals, 404s and static files included) and strips any
   ``Set-Cookie``: the UI sets no cookies at all.
2. :class:`LoopbackGuard` - 403 unless ``scope["client"]`` is ``127.0.0.1`` / ``::1``
   (a missing client fails closed); then 421 unless ``Host`` is exactly
   ``127.0.0.1:<port>``, ``localhost:<port>`` or ``[::1]:<port>`` - on every request,
   GETs included, against DNS rebinding.
3. :class:`CSRFGuard` - every request other than GET/HEAD must carry the per-process
   token, either as the ``X-CSRF-Token`` header (``csrf.js`` adds it to every htmx
   request) or as the ``csrf_token`` field of an urlencoded form body. When present,
   ``Origin`` must equal ``http://<the validated Host>`` and ``Sec-Fetch-Site`` must be
   ``same-origin`` or ``none``.

Rules chosen where the design is silent:
- A multipart body must send the token as the header (uploads go through htmx); the
  body is never buffered to look for it, so a 200 MB upload is not held in memory.
- An urlencoded body is read (at most :data:`FORM_LIMIT` bytes, else 413) and then
  replayed to the endpoint unchanged.
- A missing or wrong token answers 403 with :data:`RESTART_HINT` (a stale tab after a
  restart is the usual cause); a foreign Origin / Sec-Fetch-Site answers 403 with
  :data:`CROSS_SITE_MESSAGE`, so the page banner can tell them apart.
- ``'wasm-unsafe-eval'`` is in ``script-src``: the pinned PDF.js worker decodes JPEG 2000
  images with an embedded WebAssembly module. It allows compiling WebAssembly only,
  not JavaScript ``eval``.
"""

from __future__ import annotations

import hmac
import secrets
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any
from urllib.parse import parse_qs

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

LOOPBACK_CLIENTS = frozenset({"127.0.0.1", "::1"})
SAFE_METHODS = frozenset({"GET", "HEAD"})
CSRF_HEADER = "x-csrf-token"
CSRF_FIELD = "csrf_token"
FORM_LIMIT = 1024 * 1024
OK_FETCH_SITES = frozenset({"same-origin", "none"})

RESTART_HINT = "seedgraph restarted; reload the page"
CROSS_SITE_MESSAGE = "refused: this request did not come from the seedgraph page"
NOT_LOCAL_MESSAGE = "refused: seedgraph serves this computer only"
FOREIGN_HOST_MESSAGE = "refused: unknown Host header; open seedgraph at http://127.0.0.1:{port}/"

CSP = ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; "
       "img-src 'self' data: blob:; style-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
       "form-action 'self'")
SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"content-security-policy", CSP.encode()),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
)


def new_token() -> str:
    """A fresh per-process CSRF secret (rotates on every restart)."""
    return secrets.token_urlsafe(32)


def allowed_hosts(port: int) -> frozenset[str]:
    return frozenset({f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"})


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return value.decode("latin-1")
    return None


async def _plain(send: Send, status: int, text: str) -> None:
    body = text.encode("utf-8")
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"text/plain; charset=utf-8"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


async def _refuse(scope: Scope, receive: Receive, send: Send, status: int, text: str) -> None:
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1008})
    else:
        await _plain(send, status, text)


class SecurityHeaders:
    """Adds :data:`SECURITY_HEADERS` to every HTTP response and drops ``Set-Cookie``."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        names = {k for k, _ in SECURITY_HEADERS} | {b"set-cookie"}

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", ()) if k.lower() not in names]
                message["headers"] = headers + list(SECURITY_HEADERS)
            await send(message)

        await self.app(scope, receive, send_with_headers)


class LoopbackGuard:
    """403 for a non-loopback client, 421 for a foreign ``Host`` (every method)."""

    def __init__(self, app: ASGIApp, *, port: int) -> None:
        self.app = app
        self.port = port
        self.hosts = allowed_hosts(port)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):  # lifespan passes through
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        if not client or client[0] not in LOOPBACK_CLIENTS:
            await _refuse(scope, receive, send, 403, NOT_LOCAL_MESSAGE)
            return
        host = (_header(scope, b"host") or "").lower()
        if host not in self.hosts:
            await _refuse(scope, receive, send, 421, FOREIGN_HOST_MESSAGE.format(port=self.port))
            return
        await self.app(scope, receive, send)


class CSRFGuard:
    """Token + Origin + Sec-Fetch-Site checks on every request other than GET/HEAD.

    Must sit inside :class:`LoopbackGuard`, which has already validated ``Host``.
    """

    def __init__(self, app: ASGIApp, *, token: str) -> None:
        self.app = app
        self.token = token

    def _token_ok(self, candidate: str | None) -> bool:
        return candidate is not None and hmac.compare_digest(candidate.encode(), self.token.encode())

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":  # the UI has none; nothing may bypass the token
            await _refuse(scope, receive, send, 403, RESTART_HINT)
            return
        if scope["type"] != "http" or scope["method"] in SAFE_METHODS:
            await self.app(scope, receive, send)
            return

        origin = _header(scope, b"origin")
        own_origin = f"http://{(_header(scope, b'host') or '').lower()}"
        if origin is not None and origin.lower() != own_origin:
            await _plain(send, 403, CROSS_SITE_MESSAGE)
            return
        site = _header(scope, b"sec-fetch-site")
        if site is not None and site.lower() not in OK_FETCH_SITES:
            await _plain(send, 403, CROSS_SITE_MESSAGE)
            return

        if self._token_ok(_header(scope, CSRF_HEADER.encode())):
            await self.app(scope, receive, send)
            return

        ctype = (_header(scope, b"content-type") or "").split(";")[0].strip().lower()
        if ctype != "application/x-www-form-urlencoded":
            await _plain(send, 403, RESTART_HINT)
            return
        body = bytearray()
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body += message.get("body", b"")
            more = message.get("more_body", False)
            if len(body) > FORM_LIMIT:
                await _plain(send, 413, "refused: form too large")
                return
        fields = parse_qs(bytes(body).decode("latin-1"), keep_blank_values=True)
        if not self._token_ok((fields.get(CSRF_FIELD) or [None])[0]):
            await _plain(send, 403, RESTART_HINT)
            return

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def secure(app: ASGIApp, *, port: int, token: str) -> ASGIApp:
    """Wrap ``app``: headers (outermost), then loopback/Host, then CSRF."""
    return SecurityHeaders(LoopbackGuard(CSRFGuard(app, token=token), port=port))
