"""The app shell (design §1.2, §2, §8): route discovery, flash messages, and the
route-table rule that GET never mutates."""

from __future__ import annotations

import re
import sqlite3

import pytest
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route

from seedgraph.web.app import Flash, create_app, discovered_routes, get_web

from .conftest import PORT


def test_the_shell_home_renders_with_the_sidebar_and_marks_home_active(make_app, client_for):
    html = client_for(make_app()).get("/").text
    assert 'aria-current="page"' in re.search(r'<a [^>]*href="/"[^>]*>', html).group(0)
    assert "no jobs running" in html


def test_a_routes_module_claiming_root_replaces_the_shell(make_app, client_for):
    def home(request):
        return PlainTextResponse("custom home")

    assert client_for(make_app(routes=[Route("/", home)])).get("/").text == "custom home"


def test_discovered_routes_come_from_the_routes_package():
    # Every discovered route must be a real Starlette route; none ships with U4.
    assert all(isinstance(r, (Route, Mount)) for r in discovered_routes())


def test_a_flash_survives_one_redirect_and_is_then_consumed(make_app, client_for):
    def do(request):
        return get_web(request).redirect("/?x=1", flash=[Flash("Saved <b>it</b>", "success")])

    def page(request):
        return get_web(request).render(request, "base.html", {"page_title": "Done"})

    c = client_for(make_app(routes=[Route("/test/do", do, methods=["POST"]), Route("/done", page)]))
    r = c.post("/test/do", headers={"X-CSRF-Token": get_web_token(c)})
    assert r.status_code == 303
    target = r.headers["location"]
    assert target.startswith("/?x=1&flash=")
    first = c.get(target).text
    assert "sg-flash-success" in first
    assert "Saved &lt;b&gt;it&lt;/b&gt;" in first  # autoescaped
    assert "sg-flash-success" not in c.get(target).text


def test_redirect_refuses_a_non_local_target(make_app):
    app = make_app()
    web = app.state.web
    for bad in ("http://evil.example/", "//evil.example/x"):
        with pytest.raises(ValueError):
            web.redirect(bad)


def get_web_token(client) -> str:
    return re.search(r'name="csrf-token" content="([^"]+)"', client.get("/").text).group(1)


# --- GET never mutates -----------------------------------------------------------------


def _dump(db_path) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        return list(conn.iterdump())
    finally:
        conn.close()


def _get_paths(app) -> list[str]:
    paths = []
    for r in app.routes:
        if isinstance(r, Route) and "GET" in (r.methods or ()):
            paths.append(re.sub(r"\{[^}]+\}", "x", r.path))
    return paths


def test_no_route_accepts_both_get_and_a_mutating_method(make_app):
    for r in make_app().routes:
        if isinstance(r, Route):
            methods = set(r.methods or ())
            assert not ("GET" in methods and methods - {"GET", "HEAD"}), r.path


def test_get_on_every_route_leaves_the_library_unchanged(home, make_app, client_for):
    app = make_app()
    db = home / "library" / "library.db"
    before = _dump(db)
    c = client_for(app)
    paths = _get_paths(app)
    assert "/" in paths
    for path in paths:
        c.get(path)
    assert _dump(db) == before


def test_create_app_needs_no_converter_pool_until_jobs_start(home, monkeypatch, client_for):
    import seedgraph.conversion.pool as pool_mod

    def boom():
        raise AssertionError("the converter pool must not be built by building or requesting the app")

    monkeypatch.setattr(pool_mod, "_POOL", None)
    monkeypatch.setattr(pool_mod, "pool_from_environment", boom)
    app = create_app(home, port=PORT)
    assert client_for(app).get("/").status_code == 200
    with pytest.raises(AssertionError):  # the guard is live: resolving the pool trips it
        app.state.web.jobs.pool


def test_the_app_lifespan_starts_the_job_runner_and_resumes_the_queue(home, tmp_path, client_for):
    from .test_jobs import FakePool, queue_papers, queued_now

    queue_papers(home / "library", tmp_path, 1)
    pool = FakePool(size=1)
    app = create_app(home, port=PORT, pool=pool)
    with client_for(app) as c:
        assert app.state.web.jobs.wait_idle(timeout=30)
        assert c.get("/").status_code == 200
    assert len(pool.calls) == 1
    assert queued_now(home / "library") == []
