"""Stage A web shell: serve security primitives + ``/auth`` + ``/ui`` project list.

Offline / keyless: no real socket server (uvicorn is never started); the FastAPI
app is exercised via Starlette's ``TestClient`` and ``require_local_session`` is
unit-tested against crafted request stand-ins (cross-cutting #1 test seam — we do
not rely on ``TestClient`` loopback quirks for the gate's behavior).
"""

from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from seedgraph.api.app import app
from seedgraph.project import service
from seedgraph.run import ensure_run
from seedgraph.web import serve


# --------------------------------------------------------------------------
# bind_is_allowed
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "host,allow_remote,expected",
    [
        ("127.0.0.1", False, True),
        ("::1", False, True),
        ("localhost", False, True),
        ("0.0.0.0", False, False),
        ("192.168.1.10", False, False),
        ("0.0.0.0", True, True),
        ("192.168.1.10", True, True),
    ],
)
def test_bind_is_allowed(host, allow_remote, expected):
    assert serve.bind_is_allowed(host, allow_remote) is expected


# --------------------------------------------------------------------------
# find_free_port
# --------------------------------------------------------------------------

def test_find_free_port_skips_occupied():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        taken = occupied.getsockname()[1]
        found = serve.find_free_port("127.0.0.1", taken, attempts=20)
        assert found > taken
        # The returned port is actually bindable.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", found))


def test_find_free_port_returns_same_when_free():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        free = sock.getsockname()[1]
    # Port released on context exit (no listen/accept ⇒ no TIME_WAIT).
    assert serve.find_free_port("127.0.0.1", free, attempts=1) == free


def test_find_free_port_exhausted_raises():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        taken = occupied.getsockname()[1]
        with pytest.raises(OSError):
            serve.find_free_port("127.0.0.1", taken, attempts=1)


# --------------------------------------------------------------------------
# GET /auth — one-time bootstrap → HttpOnly SameSite=Strict cookie + 302
# --------------------------------------------------------------------------

def test_auth_sets_cookie_and_redirects():
    client = TestClient(app)
    token = app.state.session_token
    resp = client.get(f"/auth?t={token}", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/ui"
    set_cookie = resp.headers["set-cookie"]
    assert "sg_session=" in set_cookie
    assert "httponly" in set_cookie.lower()
    assert "samesite=strict" in set_cookie.lower()
    # The cookie carries the session token; it is never echoed back into a URL.
    assert client.cookies.get("sg_session") == token


def test_auth_rejects_wrong_token():
    client = TestClient(app)
    resp = client.get("/auth?t=not-the-token", follow_redirects=False)
    assert resp.status_code == 403


def test_auth_rejects_missing_token():
    client = TestClient(app)
    resp = client.get("/auth", follow_redirects=False)
    assert resp.status_code == 403


# --------------------------------------------------------------------------
# require_local_session — crafted request stand-ins (cross-cutting #1 seam)
# --------------------------------------------------------------------------

def _fake_request(host, cookie_value, token):
    client = SimpleNamespace(host=host) if host is not None else None
    cookies = {} if cookie_value is None else {serve.SESSION_COOKIE: cookie_value}
    app_ns = SimpleNamespace(state=SimpleNamespace(session_token=token))
    return SimpleNamespace(client=client, cookies=cookies, app=app_ns)


def test_require_local_session_passes_loopback_with_cookie():
    token = "session-token-xyz"
    for host in ("127.0.0.1", "::1", "localhost"):
        assert serve.require_local_session(_fake_request(host, token, token)) is None


def test_require_local_session_rejects_missing_cookie():
    with pytest.raises(HTTPException) as ei:
        serve.require_local_session(_fake_request("127.0.0.1", None, "tok"))
    assert ei.value.status_code == 403


def test_require_local_session_rejects_wrong_cookie():
    with pytest.raises(HTTPException) as ei:
        serve.require_local_session(_fake_request("127.0.0.1", "wrong", "tok"))
    assert ei.value.status_code == 403


def test_require_local_session_rejects_non_loopback():
    with pytest.raises(HTTPException) as ei:
        serve.require_local_session(_fake_request("10.0.0.5", "tok", "tok"))
    assert ei.value.status_code == 403


def test_require_local_session_rejects_client_none():
    # Fail-closed when request.client is None (non-HTTP transport).
    with pytest.raises(HTTPException) as ei:
        serve.require_local_session(_fake_request(None, "tok", "tok"))
    assert ei.value.status_code == 403


# --------------------------------------------------------------------------
# /ui project-list + graph placeholder
# --------------------------------------------------------------------------

def test_ui_project_list_renders_empty():
    client = TestClient(app)
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "Projects" in resp.text


def test_ui_project_list_renders_projects():
    service.create_project("webproj")
    client = TestClient(app)
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "webproj" in resp.text
    assert "/ui/projects/webproj/graph" in resp.text


def test_ui_graph_placeholder_renders_no_runs_page():
    service.create_project("webnoruns")
    client = TestClient(app)
    resp = client.get("/ui/projects/webnoruns/graph")
    assert resp.status_code == 200
    assert "no runs" in resp.text.lower()


def test_ui_graph_placeholder_redirects_to_latest_run():
    service.create_project("webrun")
    run_id = ensure_run("webrun")
    client = TestClient(app)
    resp = client.get("/ui/projects/webrun/graph", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == f"/ui/projects/webrun/runs/{run_id}/graph"


# --------------------------------------------------------------------------
# Private-evidence routes are actually GATED by require_local_session
# (regression for the "Depends applied to zero routes" content-access leak).
# --------------------------------------------------------------------------

#: Every private-evidence GET that must refuse a non-loopback / cookieless client.
_GATED_ROUTES = [
    "/projects/gateproj/documents",
    "/projects/gateproj/concepts/concept::x",
    "/projects/gateproj/answer?q=hi",
    "/projects/gateproj/runs/r1/graph.json",
]


@pytest.mark.parametrize("path", _GATED_ROUTES)
def test_private_routes_refuse_without_session(path):
    # TestClient's client host is "testclient" (non-loopback) and carries no
    # sg_session cookie, so the loopback+cookie gate must fail-closed with 403 —
    # never the route body (404/200). This proves the gate is wired, not just defined.
    service.create_project("gateproj")
    client = TestClient(app)
    assert client.get(path).status_code == 403


@pytest.mark.parametrize("path", _GATED_ROUTES)
def test_private_routes_reachable_with_dependency_override(path):
    # The dependency-override seam (cross-cutting #1) lets a test bypass the loopback
    # gate; the request then reaches the route body (anything but a 403 gate refusal).
    service.create_project("gateproj")
    app.dependency_overrides[serve.require_local_session] = lambda: None
    try:
        client = TestClient(app)
        assert client.get(path).status_code != 403
    finally:
        app.dependency_overrides.pop(serve.require_local_session, None)
