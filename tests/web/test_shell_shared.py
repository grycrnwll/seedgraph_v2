"""Two shell-wide behaviours every page relies on (U9 shared fixes):

* the sidebar's Review badge shows the open attention count on every page, even one
  whose route does not pass ``review_open`` itself;
* the job runner exposes an already-built converter pool without ever building one,
  so the Settings GET cannot probe GPUs.
"""

from __future__ import annotations

from starlette.routing import Route

from seedgraph.library import attention
from seedgraph.library.store import open_library_at

from .test_jobs import FakePool


def _bare_page(request):
    from seedgraph.web.app import get_web

    # A route that knows nothing about the Review badge.
    return get_web(request).render(request, "base.html", {"page_title": "Bare"})


def _counted_page(request):
    from seedgraph.web.app import get_web

    return get_web(request).render(request, "base.html", {"page_title": "Counted", "review_open": 7})


def _raise_items(home, n):
    lib = open_library_at(home / "library")
    try:
        for i in range(n):
            attention.raise_item(lib, "garbled_text", {"file": f"invented-{i}"}, detail="invented")
    finally:
        lib.close()


def test_a_page_that_does_not_pass_the_count_still_shows_the_review_badge(home, make_app, client_for):
    _raise_items(home, 3)
    html = client_for(make_app(routes=[Route("/bare", _bare_page)])).get("/bare").text
    assert 'href="/review/attention"' in html
    assert '<span class="sg-badge">3</span>' in html


def test_no_badge_when_nothing_is_open(home, make_app, client_for):
    html = client_for(make_app(routes=[Route("/bare", _bare_page)])).get("/bare").text
    assert 'href="/review/attention"' not in html


def test_a_count_passed_by_the_page_is_kept(home, make_app, client_for):
    _raise_items(home, 1)
    html = client_for(make_app(routes=[Route("/counted", _counted_page)])).get("/counted").text
    assert '<span class="sg-badge">7</span>' in html


def test_pool_if_started_is_none_until_a_pool_exists(tmp_path, monkeypatch):
    import seedgraph.conversion as conversion
    from seedgraph.web.jobs import JobRunner

    def boom(*a, **k):
        raise AssertionError("pool_if_started must never build the pool")

    monkeypatch.setattr(conversion, "get_pool", boom)
    lib_dir = tmp_path / "library"
    open_library_at(lib_dir).close()
    assert JobRunner(lib_dir).pool_if_started() is None
    pool = FakePool(size=1)
    assert JobRunner(lib_dir, pool=pool).pool_if_started() is pool
