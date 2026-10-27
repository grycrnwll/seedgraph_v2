"""Graph page Lenses tab (stage 4 design §8.4): the library lenses with choose / unchoose
for the graph, members missing each chosen lens, and the command that runs it on them.
Invented text only."""

from __future__ import annotations

import pytest

from seedgraph.library import graphs
from seedgraph.library.store import open_library_at

from .pages_support import (
    GRAPH,
    build,
    no_inline_script_or_style,
    post_forms_lead_with_csrf,
    token,
)

NOW = "2026-01-01T00:00:00+00:00"


@pytest.fixture
def fx(home):
    fx = build(home)
    lib = open_library_at(home / "library")
    try:
        with lib.write() as conn:
            for lens_id, name in (("widget_lens", "Invented widget lens"), ("gadget_lens", "Invented gadget lens")):
                conn.execute("INSERT INTO lens_definitions (lens_id, name, object_type, status, definition_hash, "
                             "created_at) VALUES (?, ?, 'paper', 'active', ?, ?)", (lens_id, name, f"h_{lens_id}", NOW))
    finally:
        lib.close()
    return fx


def chosen(home):
    lib = open_library_at(home / "library")
    try:
        with graphs.open_graph(lib, GRAPH) as g:
            return graphs.chosen_lenses(g)
    finally:
        lib.close()


def test_the_lenses_tab_lists_library_lenses_with_choose_buttons(fx, make_app, client_for):
    html = client_for(make_app()).get(f"/graphs/{GRAPH}/lenses").text
    assert "Invented widget lens" in html and "Invented gadget lens" in html
    assert html.count(f'action="/graphs/{GRAPH}/lenses/choose"') == 2
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)


def test_choose_then_unchoose_a_lens(fx, make_app, client_for, monkeypatch):
    calls = []
    real_choose, real_unchoose = graphs.choose_lens, graphs.unchoose_lens
    monkeypatch.setattr(graphs, "choose_lens", lambda g, lid, **kw: (calls.append(("choose", lid)),
                                                                      real_choose(g, lid, **kw))[1])
    monkeypatch.setattr(graphs, "unchoose_lens", lambda g, lid: (calls.append(("unchoose", lid)),
                                                                 real_unchoose(g, lid))[1])
    c = client_for(make_app())
    assert c.post(f"/graphs/{GRAPH}/lenses/choose", data={"lens_id": "widget_lens"}).status_code == 403
    r = c.post(f"/graphs/{GRAPH}/lenses/choose", data={"csrf_token": token(c), "lens_id": "widget_lens"})
    assert r.status_code == 303 and chosen(fx.home) == ["widget_lens"]
    html = c.get(r.headers["location"]).text
    assert "3" in html and f"seedgraph work create lens:widget_lens --graph {GRAPH}" in html  # 3 members lack it
    c.post(f"/graphs/{GRAPH}/lenses/unchoose", data={"csrf_token": token(c), "lens_id": "widget_lens"})
    assert chosen(fx.home) == []
    assert calls == [("choose", "widget_lens"), ("unchoose", "widget_lens")]


def test_an_unknown_lens_cannot_be_chosen(fx, make_app, client_for):
    c = client_for(make_app())
    r = c.post(f"/graphs/{GRAPH}/lenses/choose", data={"csrf_token": token(c), "lens_id": "no_such_lens"})
    assert "sg-flash-error" in c.get(r.headers["location"]).text and chosen(fx.home) == []
