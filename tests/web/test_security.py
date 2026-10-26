"""Localhost, Host and CSRF matrix; no cookies; security headers (design §8, §10.2 item 4).

The seam is the ASGI app built by ``create_app``: requests go in, status codes and
headers come out. A tiny test-only form route checks that a POST that passes the CSRF
check still reaches the endpoint with its form fields intact.
"""

from __future__ import annotations

import re

import pytest
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from .conftest import ORIGIN, PORT


async def _echo(request: Request):
    form = await request.form()
    return PlainTextResponse(f"note={form.get('note', '')}")


ECHO = [Route("/test/echo", _echo, methods=["POST"])]


@pytest.fixture
def app(make_app):
    return make_app(routes=ECHO)


def token_of(client) -> str:
    html = client.get("/").text
    m = re.search(r'<meta name="csrf-token" content="([^"]+)"', html)
    assert m, "base page carries the csrf meta tag"
    return m.group(1)


# --- client (loopback) check -----------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_loopback_clients_are_served(app, client_for, host):
    assert client_for(app, client=(host, 50000)).get("/").status_code == 200


@pytest.mark.parametrize("host", ["192.168.1.20", "10.0.0.5", "testclient"])
def test_a_non_loopback_client_gets_403(app, client_for, host):
    r = client_for(app, client=(host, 50000)).get("/")
    assert r.status_code == 403


def test_a_request_with_no_client_address_is_refused(app):
    import asyncio

    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "method": "GET", "path": "/", "raw_path": b"/", "query_string": b"",
             "headers": [(b"host", f"127.0.0.1:{PORT}".encode())], "client": None,
             "server": ("127.0.0.1", PORT), "scheme": "http", "http_version": "1.1",
             "root_path": "", "asgi": {"version": "3.0"}}
    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 403


# --- Host check ------------------------------------------------------------------------


@pytest.mark.parametrize("host", [f"127.0.0.1:{PORT}", f"localhost:{PORT}", f"[::1]:{PORT}"])
def test_own_host_names_are_accepted(app, client_for, host):
    assert client_for(app, base_url=f"http://{host}").get("/").status_code == 200


@pytest.mark.parametrize("host", ["evil.example:8765", f"127.0.0.1:{PORT + 1}", "localhost",
                                  "127.0.0.1", "rebind.example"])
def test_a_foreign_host_gets_421_on_get(app, client_for, host):
    r = client_for(app, base_url=f"http://{host}").get("/")
    assert r.status_code == 421
    assert "host" in r.text.lower()


def test_a_foreign_host_gets_421_on_post_even_with_a_token(app, client_for):
    token = token_of(client_for(app))
    r = client_for(app, base_url="http://evil.example:8765").post(
        "/test/echo", data={"csrf_token": token, "note": "x"})
    assert r.status_code == 421


def test_static_files_are_behind_the_host_check_too(app, client_for):
    assert client_for(app).get("/static/app.css").status_code == 200
    assert client_for(app, base_url="http://evil.example:8765").get("/static/app.css").status_code == 421


# --- CSRF ------------------------------------------------------------------------------


def test_a_post_with_the_form_token_reaches_the_route_with_its_fields(app, client_for):
    c = client_for(app)
    r = c.post("/test/echo", data={"csrf_token": token_of(c), "note": "hello"})
    assert r.status_code == 200
    assert r.text == "note=hello"


def test_a_post_with_the_header_token_is_accepted(app, client_for):
    c = client_for(app)
    r = c.post("/test/echo", data={"note": "via htmx"}, headers={"X-CSRF-Token": token_of(c)})
    assert (r.status_code, r.text) == (200, "note=via htmx")


def test_a_multipart_post_with_the_header_token_is_accepted(app, client_for):
    c = client_for(app)
    r = c.post("/test/echo", data={"note": "upload"}, files={"f": ("a.pdf", b"%PDF-1.4\n")},
               headers={"X-CSRF-Token": token_of(c)})
    assert (r.status_code, r.text) == (200, "note=upload")


def test_a_post_without_a_token_gets_403_with_the_restart_hint(app, client_for):
    r = client_for(app).post("/test/echo", data={"note": "x"})
    assert r.status_code == 403
    assert "seedgraph restarted; reload the page" in r.text


def test_a_post_with_a_wrong_token_gets_403(app, client_for):
    c = client_for(app)
    token_of(c)
    assert c.post("/test/echo", data={"csrf_token": "wrong", "note": "x"}).status_code == 403
    assert c.post("/test/echo", headers={"X-CSRF-Token": "wrong"}).status_code == 403


def test_a_multipart_post_with_only_a_form_token_is_refused(app, client_for):
    """Uploads must send the header (htmx does); the body is never buffered."""
    c = client_for(app)
    r = c.post("/test/echo", data={"csrf_token": token_of(c), "note": "x"},
               files={"f": ("a.pdf", b"%PDF-1.4\n")})
    assert r.status_code == 403


def test_the_token_rotates_with_the_process(make_app, client_for):
    old = token_of(client_for(make_app(routes=ECHO)))
    fresh = client_for(make_app(routes=ECHO))
    assert token_of(fresh) != old
    assert fresh.post("/test/echo", data={"csrf_token": old}).status_code == 403


@pytest.mark.parametrize("origin", ["http://evil.example", "null", f"http://localhost:{PORT}",
                                    f"https://127.0.0.1:{PORT}", f"http://127.0.0.1:{PORT + 1}"])
def test_a_foreign_origin_gets_403_even_with_the_token(app, client_for, origin):
    c = client_for(app)
    r = c.post("/test/echo", data={"csrf_token": token_of(c)}, headers={"Origin": origin})
    assert r.status_code == 403


def test_the_own_origin_is_accepted(app, client_for):
    c = client_for(app)
    r = c.post("/test/echo", data={"csrf_token": token_of(c), "note": "o"},
               headers={"Origin": ORIGIN})
    assert r.status_code == 200


@pytest.mark.parametrize("site,ok", [("same-origin", True), ("none", True), ("same-site", False),
                                     ("cross-site", False)])
def test_sec_fetch_site_must_be_same_origin_or_none(app, client_for, site, ok):
    c = client_for(app)
    r = c.post("/test/echo", data={"csrf_token": token_of(c)}, headers={"Sec-Fetch-Site": site})
    assert (r.status_code == 200) is ok


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE"])
def test_every_mutating_method_needs_the_token(app, client_for, method):
    assert client_for(app).request(method, "/test/echo").status_code == 403


def test_head_and_get_need_no_token(app, client_for):
    c = client_for(app)
    assert c.get("/").status_code == 200
    assert c.head("/").status_code == 200


# --- headers and cookies ---------------------------------------------------------------

CSP = ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; "
       "img-src 'self' data: blob:; style-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
       "form-action 'self'")


def _responses(app, client_for):
    ok = client_for(app)
    yield ok.get("/")
    yield ok.get("/static/app.css")
    yield ok.get("/no-such-page")
    yield ok.post("/test/echo", data={"csrf_token": token_of(ok), "note": "x"})
    yield ok.post("/test/echo", data={"note": "no token"})
    yield client_for(app, client=("10.0.0.9", 1)).get("/")
    yield client_for(app, base_url="http://evil.example:8765").get("/")


def test_every_response_carries_the_csp_and_hardening_headers(app, client_for):
    for r in _responses(app, client_for):
        assert r.headers["content-security-policy"] == CSP, r.request.url
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["referrer-policy"] == "no-referrer"


def test_no_response_sets_a_cookie(app, client_for):
    for r in _responses(app, client_for):
        assert "set-cookie" not in r.headers, r.request.url
        assert not r.cookies


def test_a_route_that_tries_to_set_a_cookie_has_it_stripped(make_app, client_for):
    async def sneaky(request):
        resp = PlainTextResponse("x")
        resp.set_cookie("sid", "1")
        return resp

    r = client_for(make_app(routes=[Route("/test/sneaky", sneaky)])).get("/test/sneaky")
    assert r.status_code == 200 and "set-cookie" not in r.headers


@pytest.mark.parametrize("path,ctype", [
    ("/static/vendor/pdfjs/pdf.min.mjs", "text/javascript"),
    ("/static/vendor/htmx/htmx.min.js", "text/javascript"),
    ("/static/js/csrf.js", "text/javascript"),
    ("/static/app.css", "text/css"),
])
def test_scripts_are_served_with_a_javascript_type(app, client_for, path, ctype):
    """With nosniff, a module served as text/plain would not run."""
    r = client_for(app).get(path)
    assert r.status_code == 200
    assert r.headers["content-type"].split(";")[0] == ctype


# --- templates -------------------------------------------------------------------------


def test_the_base_page_has_no_inline_script_or_style(app, client_for):
    html = client_for(app).get("/").text
    for m in re.finditer(r"<script\b([^>]*)>(.*?)</script>", html, re.S):
        assert "src=" in m.group(1) and m.group(2).strip() == ""
    assert "<style" not in html
    assert " style=" not in html
    assert not re.search(r"\son[a-z]+=", html)  # no onclick= etc.
    assert "hx-on" not in html


def test_the_base_page_turns_off_htmx_eval_and_indicator_styles(app, client_for):
    html = client_for(app).get("/").text
    m = re.search(r"<meta name=\"htmx-config\" content='([^']+)'", html)
    assert m
    import json

    cfg = json.loads(m.group(1))
    assert cfg["includeIndicatorStyles"] is False and cfg["allowEval"] is False


def test_the_sidebar_lists_the_six_sections(app, client_for):
    html = client_for(app).get("/").text
    nav = re.search(r'<nav class="sg-sidebar".*?</nav>', html, re.S).group(0)
    labels = re.findall(r'<a [^>]*href="([^"]+)"[^>]*>\s*([^<]+?)\s*<', nav)
    assert labels[:6] == [("/", "Home"), ("/library", "Library"), ("/graphs", "Context graphs"),
                          ("/records", "Records"), ("/spot-check", "Spot-check"),
                          ("/settings", "Settings")]
