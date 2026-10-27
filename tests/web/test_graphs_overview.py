"""Graph page Overview tab (stage 4 design §6.2, §6.3, §8.1): counts, literatures with names,
rename, regroup and "fewer / more literatures", the staleness banner, and bridges. Ambiguous
and unchecked labels are listed apart and never as bridges. Invented text only."""

from __future__ import annotations

import re

import pytest
from concepts.test_bridges import GRAPH, build_world, p

from seedgraph.library import graphs
from seedgraph.library import literatures as lits
from seedgraph.library.store import open_library_at

from .pages_support import no_inline_script_or_style, post_forms_lead_with_csrf, token


@pytest.fixture
def world(home):
    lib = open_library_at(home / "library")
    try:
        build_world(lib)
    finally:
        lib.close()
    return home


def literatures_of(home):
    lib = open_library_at(home / "library")
    try:
        with graphs.open_graph(lib, GRAPH) as g:
            return lits.literatures(g), lits.latest_run(g)
    finally:
        lib.close()


def section(html: str, heading: str) -> str:
    m = re.search(rf"<h2[^>]*>{heading}</h2>(.*?)(?=<h2|</main>|$)", html, re.S)
    assert m, heading
    return m.group(1)


def test_the_overview_shows_counts_tabs_literatures_and_bridges(world, make_app, client_for):
    html = client_for(make_app()).get(f"/graphs/{GRAPH}").text
    assert "15 members" in html and "1 seed" in html
    for tab in ("concepts", "graph", "grow", "lenses"):
        assert f'href="/graphs/{GRAPH}/{tab}"' in html
    lit_html = section(html, "Literatures")
    assert lit_html.count("unnamed (5 papers)") == 3
    bridges = section(html, "Bridges")
    assert "widget lag" in bridges and "auto wobble" in bridges and "serial wobble" in bridges
    assert 'href="/concepts/method/widget%20lag"' in bridges
    assert "sprocket balance" not in bridges and "identification" not in bridges
    unchecked = section(html, "Same words, sense not checked")
    assert "sprocket balance" in unchecked and "identification" in unchecked and "ambiguous label" in unchecked
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)


def test_the_papers_table_gets_a_literature_column(world, make_app, client_for):
    html = client_for(make_app()).get(f"/graphs/{GRAPH}").text
    row = re.search(rf'<tr>(?:(?!</tr>).)*href="/papers/{p(0, 0)}"(?:(?!</tr>).)*</tr>', html, re.S).group(0)
    assert "unnamed (5 papers)" in row


def test_renaming_a_literature_stores_the_researchers_name(world, make_app, client_for):
    (first, *_), _ = literatures_of(world)
    c = client_for(make_app())
    data = {"literature_id": first.literature_id, "name": "Invented sprocket studies"}
    assert c.post(f"/graphs/{GRAPH}/literatures/rename", data=data).status_code == 403
    r = c.post(f"/graphs/{GRAPH}/literatures/rename", data={"csrf_token": token(c), **data})
    assert r.status_code == 303
    (first, *_), _ = literatures_of(world)
    assert (first.name, first.name_source) == ("Invented sprocket studies", "researcher")
    html = c.get(r.headers["location"]).text
    assert "Invented sprocket studies" in html and "researcher" in html


def test_an_empty_rename_is_refused_with_a_message(world, make_app, client_for):
    (first, *_), _ = literatures_of(world)
    c = client_for(make_app())
    r = c.post(f"/graphs/{GRAPH}/literatures/rename",
               data={"csrf_token": token(c), "literature_id": first.literature_id, "name": "  "})
    assert "sg-flash-error" in c.get(r.headers["location"]).text
    assert literatures_of(world)[0][0].researcher_name is None


def test_fewer_and_more_literatures_step_the_resolution_and_keep_the_choice(world, make_app, client_for):
    _, before = literatures_of(world)
    c = client_for(make_app())
    r = c.post(f"/graphs/{GRAPH}/literatures/step", data={"csrf_token": token(c), "direction": "more"})
    assert r.status_code == 303
    _, after = literatures_of(world)
    assert after.resolution > before.resolution and after.chosen_by == "researcher"
    c.post(f"/graphs/{GRAPH}/literatures/step", data={"csrf_token": token(c), "direction": "fewer"})
    assert literatures_of(world)[1].resolution == before.resolution


def test_a_changed_graph_shows_the_staleness_banner_until_regrouped(world, make_app, client_for):
    lib = open_library_at(world / "library")
    try:
        with graphs.open_graph(lib, GRAPH) as g:
            graphs.remove_membership(g, p(2, 4))
    finally:
        lib.close()
    c = client_for(make_app())
    assert "Literatures are out of date" in c.get(f"/graphs/{GRAPH}").text
    r = c.post(f"/graphs/{GRAPH}/literatures/regroup", data={"csrf_token": token(c)})
    assert r.status_code == 303
    assert "Literatures are out of date" not in c.get(r.headers["location"]).text


def test_a_graph_without_a_grouping_offers_to_group(world, make_app, client_for):
    lib = open_library_at(world / "library")
    try:
        with graphs.create_graph(lib, "fresh") as g:
            graphs.set_membership(g, p(0, 0), "member")
    finally:
        lib.close()
    html = client_for(make_app()).get("/graphs/fresh").text
    assert "No literatures yet" in html and 'action="/graphs/fresh/literatures/regroup"' in html
