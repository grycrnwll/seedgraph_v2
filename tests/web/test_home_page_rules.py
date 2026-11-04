"""Rules every U7 page keeps (design §1.2, §8): GET never mutates the fixture library, no
inline script or style, the CSRF field first in every POST form, and route modules run
no SQL."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from .pages_support import GRAPH, build, no_inline_script_or_style, post_forms_lead_with_csrf

U7_ROUTE_MODULES = ("home", "library", "batches", "review", "graphs", "settings", "placeholders", "records",
                    "_pages")


@pytest.fixture
def fx(home):
    return build(home)


def _pages(fx) -> list[str]:
    return ["/", "/home/activity", "/library", "/library?view=cited-only", "/library?view=needs-notes",
            "/library?view=held-back", "/library?view=needs-pdf", "/library?view=all&q=widget",
            f"/library?graph={GRAPH}", "/library/jobs", f"/library/write-notes?paper={fx.no_note}",
            f"/library/write-notes?view=needs-notes&graph={GRAPH}", "/batches", f"/batches/{fx.done_batch}",
            f"/batches/{fx.open_batch}", f"/batches/{fx.open_batch}/items", "/review/attention",
            "/review/duplicates", "/review/citations", "/graphs", f"/graphs/{GRAPH}", "/settings", "/records",
            "/spot-check"]


def _dump(db) -> list[str]:
    conn = sqlite3.connect(db)
    try:
        return list(conn.iterdump())
    finally:
        conn.close()


def test_get_on_every_page_leaves_the_fixture_library_unchanged(fx, make_app, client_for):
    db = fx.home / "library" / "library.db"
    before = _dump(db)
    c = client_for(make_app())
    for path in _pages(fx):
        assert c.get(path).status_code == 200, path
    assert _dump(db) == before


def test_every_page_has_no_inline_script_or_style_and_csrf_first(fx, make_app, client_for):
    c = client_for(make_app())
    for path in _pages(fx):
        html = c.get(path).text
        assert no_inline_script_or_style(html), path
        assert post_forms_lead_with_csrf(html), path
        if "<html" in html:  # full pages (not the polled partials) carry the page stylesheet
            assert 'href="/static/pages.css"' in html, path


def test_route_modules_run_no_sql():
    root = Path(__file__).resolve().parents[2] / "src" / "seedgraph" / "web" / "routes"
    for name in U7_ROUTE_MODULES:
        text = (root / f"{name}.py").read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import sqlite3|from sqlite3)", text, re.M), name
        assert "execute(" not in text, name
