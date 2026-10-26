"""Context graphs (design §2.3, minimal until stage 4): the list, the per-graph page with
members and frontier, add to graph and remove from graph."""

from __future__ import annotations

import re

import pytest

from seedgraph.library import graphs, lifecycle
from seedgraph.library.store import open_library_at
from seedgraph.web.views import graphs as graphs_view

from .pages_support import GRAPH, build, no_inline_script_or_style, post_forms_lead_with_csrf, token


@pytest.fixture
def fx(home):
    return build(home)


def members(home):
    lib = open_library_at(home / "library")
    try:
        with graphs.open_graph(lib, GRAPH) as g:
            return {p.paper_id: p.status for p in graphs.membership(g)}
    finally:
        lib.close()


def test_the_graph_list_view_counts_members_seeds_frontier_and_needing_notes(fx):
    lib = open_library_at(fx.home / "library")
    try:
        (row,) = graphs_view.list_page(lib)
    finally:
        lib.close()
    assert (row.slug, row.display_name) == (GRAPH, "Invented widgets")
    assert (row.members, row.seeds, row.frontier) == (3, 1, 1)
    assert row.needs_notes == 2  # lp_test0002 and lp_test0003 have no note


def test_the_graphs_page_lists_each_graph_with_a_link(fx, make_app, client_for):
    html = client_for(make_app()).get("/graphs").text
    assert f'href="/graphs/{GRAPH}"' in html and "Invented widgets" in html
    assert 'href="/graphs" aria-current="page"' in html


def test_a_graph_page_shows_members_frontier_seed_and_pdf_buttons(fx, make_app, client_for):
    html = client_for(make_app()).get(f"/graphs/{GRAPH}").text
    assert f'href="/papers/{fx.noted}"' in html and f'href="/papers/{fx.noted}/pdf"' in html
    assert f'href="/papers/{fx.cited}"' in html and f'href="/papers/{fx.cited}/pdf"' not in html  # cited-only
    assert "seed" in html and "frontier" in html
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)


def test_an_unknown_graph_is_404(fx, make_app, client_for):
    assert client_for(make_app()).get("/graphs/nope").status_code == 404


def test_remove_from_graph_calls_lifecycle_once_and_needs_csrf(fx, make_app, client_for, monkeypatch):
    calls = []
    real = lifecycle.remove_from_graph

    def spy(lib, slug, paper_id):
        calls.append((slug, paper_id))
        return real(lib, slug, paper_id)

    monkeypatch.setattr(lifecycle, "remove_from_graph", spy)
    c = client_for(make_app())
    assert c.post(f"/graphs/{GRAPH}/remove", data={"paper_id": fx.garbled}).status_code == 403
    r = c.post(f"/graphs/{GRAPH}/remove", data={"csrf_token": token(c), "paper_id": fx.garbled})
    assert r.status_code == 303 and calls == [(GRAPH, fx.garbled)]
    assert fx.garbled not in members(fx.home)
    assert "removed" in c.get(r.headers["location"]).text


def test_add_to_graph_refuses_a_cited_only_member_but_takes_it_on_the_frontier(fx, make_app, client_for):
    c = client_for(make_app())
    t = token(c)
    r = c.post(f"/graphs/{GRAPH}/add", data={"csrf_token": t, "paper_id": fx.cited_no_doi, "status": "member"})
    assert r.status_code == 303 and fx.cited_no_doi not in members(fx.home)
    assert re.search(r"sg-flash-error[^>]*>[^<]*cited-only", c.get(r.headers["location"]).text)
    c.post(f"/graphs/{GRAPH}/add", data={"csrf_token": t, "paper_id": fx.cited_no_doi, "status": "frontier"})
    assert members(fx.home)[fx.cited_no_doi] == "frontier"
