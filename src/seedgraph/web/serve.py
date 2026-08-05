"""Localhost serve-security primitives for ``seedgraph serve`` (Track 2 shell).

Pure, socket-server-free helpers so the bind guard, the free-port search, the
one-time ``/auth`` bootstrap, the per-request local-session gate, and the CSRF
synchronizer check are all unit-testable without standing up uvicorn.

Security model (plan cross-cutting #1):

* ``serve`` prints a one-time URL ``http://127.0.0.1:{port}/auth?t={session_token}``.
* ``GET /auth`` validates ``t == app.state.session_token`` (constant-time), sets an
  **HttpOnly, SameSite=Strict** cookie ``sg_session`` and 302→``/ui``. The token
  never reappears in the address bar; there are no query tokens on normal navigation.
* ``require_local_session`` is the authoritative per-request content gate: the
  client must be loopback **and** present the matching cookie; fail-closed (403)
  when ``request.client is None``. Applied to every private-evidence GET and every
  mutating POST.
* ``require_csrf`` adds a synchronizer token to mutating POSTs (defense in depth on
  top of SameSite=Strict + loopback + no-CORS).

``app.state.remote_bind`` (set by ``--allow-remote``) is only a UI banner hint, not
the gate — the loopback + cookie check above is authoritative regardless of bind.
"""

from __future__ import annotations

import socket
from hmac import compare_digest

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

#: Hosts treated as loopback for both the bind guard and the per-request gate.
LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})

#: Session cookie name (HttpOnly, SameSite=Strict, path=/).
SESSION_COOKIE = "sg_session"


def bind_is_allowed(host: str, allow_remote: bool) -> bool:
    """Return whether binding ``host`` is permitted.

    Loopback hosts are always allowed; any other (remotely reachable) interface
    requires an explicit ``allow_remote`` opt-in.
    """
    return bool(allow_remote) or host in LOOPBACK_HOSTS


def find_free_port(host: str, port: int, attempts: int = 20) -> int:
    """Return the first bindable port at or above ``port`` (up to ``attempts`` tries).

    Probes ``host:port``, ``host:port+1`` … with a throwaway socket; the first
    candidate that binds cleanly is returned. Raises :class:`OSError` if none of the
    ``attempts`` candidates are free.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    for offset in range(max(1, attempts)):
        candidate = port + offset
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, candidate))
            except OSError:
                continue
            return candidate
    raise OSError(f"no free port in [{port}, {port + max(1, attempts)}) on {host!r}")


def require_local_session(request: Request) -> None:
    """FastAPI dependency: authoritative per-request loopback + session-cookie gate.

    Passes only when the client is loopback **and** presents a ``sg_session`` cookie
    that constant-time-matches ``app.state.session_token``. Fail-closed (403) when
    ``request.client is None`` (e.g. a non-HTTP transport), a non-loopback host, or a
    missing / mismatched cookie.
    """
    client = request.client
    if client is None or client.host not in LOOPBACK_HOSTS:
        raise HTTPException(status_code=403, detail="local session required")
    token = getattr(request.app.state, "session_token", "") or ""
    cookie = request.cookies.get(SESSION_COOKIE, "")
    if not token or not compare_digest(cookie, token):
        raise HTTPException(status_code=403, detail="local session required")


async def require_csrf(request: Request) -> None:
    """FastAPI dependency: validate the synchronizer CSRF token on a mutating POST.

    The server renders ``app.state.csrf_token`` into a hidden ``_csrf`` form field;
    this checks the posted value constant-time. Defense in depth on top of
    SameSite=Strict + loopback + no-CORS (no JS / query token needed).
    """
    form = await request.form()
    posted = str(form.get("_csrf", ""))
    token = getattr(request.app.state, "csrf_token", "") or ""
    if not token or not compare_digest(posted, token):
        raise HTTPException(status_code=403, detail="invalid CSRF token")


auth_router = APIRouter(tags=["auth"])


@auth_router.get("/auth")
def auth(request: Request, t: str = "") -> RedirectResponse:
    """One-time bootstrap: exchange the session token in ``?t=`` for the cookie.

    Validates ``t`` against ``app.state.session_token`` (constant-time). On match,
    sets the HttpOnly + SameSite=Strict ``sg_session`` cookie and 302-redirects to
    ``/ui`` (so the token never persists in the address bar); else 403.
    """
    token = getattr(request.app.state, "session_token", "") or ""
    if not token or not compare_digest(t, token):
        raise HTTPException(status_code=403, detail="invalid or missing session token")
    response = RedirectResponse(url="/ui", status_code=302)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        samesite="strict",
        path="/",
    )
    return response
