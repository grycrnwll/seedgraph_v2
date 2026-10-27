"""The 2D graph (stage 4 design §8.3): a deterministic server-rendered SVG with classes and
presentation attributes only (CSP), a concept view (top concepts, edges at >= 2 shared
papers, dashed links, bridge rings) and a citation view coloured by literature, toggled by
an htmx swap with a plain link fallback. Invented text only."""

from __future__ import annotations

import re

import pytest
from concepts._world import a
from concepts.test_bridges import GRAPH, GROUPS, SIZE, build_world, p

from seedgraph.concepts import refresh as rf
from seedgraph.library import graphs
from seedgraph.library.store import open_library_at
from seedgraph.web.views import graph_svg

from .pages_support import no_inline_script_or_style

EXTRA = "lp_extra"


def make_home(tmp_path, name: str, *, shuffle: bool = False, extra: bool = True):
    home = tmp_path / name
    lib = open_library_at(home / "library")
    try:
        w = build_world(lib, shuffle=shuffle)
        if extra:  # a member sharing sprocket rule 0 and sprocket balance with lp_0_0: one solid edge
            w.note(EXTRA, [a("sprocket rule 0", "Invented: sprocket rule 0 holds in the extra paper."),
                           a("sprocket balance", "Invented: sprocket balance holds in the extra paper.")])
            with graphs.open_graph(lib, GRAPH) as g:
                graphs.set_membership(g, EXTRA, "member")
            rf.refresh(lib)
    finally:
        lib.close()
    return home


def svg_of(html: str) -> str:
    m = re.search(r'<svg xmlns="http://www.w3.org/2000/svg" class="sg-graph-svg.*?</svg>', html, re.S)
    assert m
    return m.group(0)


def nodes(svg: str) -> dict[str, str]:
    return dict(re.findall(r'data-node="([^"]+)"><circle class="([^"]+)"', svg))


def edges(svg: str) -> list[tuple[str, str, bool]]:
    return [(x, y, "sg-edge-link" in cls) for cls, x, y in
            re.findall(r'<line class="([^"]+)" data-a="([^"]+)" data-b="([^"]+)"', svg)]


def names(svg: str) -> dict[str, str]:
    """data-node -> concept name (from the <text> label)."""
    return dict(re.findall(r'data-node="([^"]+)">.*?<text [^>]*>([^<]+)</text>', svg))


@pytest.fixture
def home_a(tmp_path):
    return make_home(tmp_path, "a")


def page(make_app_for, home, client_for, path):
    return client_for(make_app_for(home)).get(path).text


@pytest.fixture
def app_for():
    from seedgraph.web.app import create_app

    return lambda home: create_app(home, port=8765)


def test_the_concept_svg_is_byte_identical_across_renders_and_insertion_order(tmp_path, app_for, client_for):
    first = make_home(tmp_path, "a")
    second = make_home(tmp_path, "b", shuffle=True)
    one = svg_of(page(app_for, first, client_for, f"/graphs/{GRAPH}/graph"))
    again = svg_of(page(app_for, first, client_for, f"/graphs/{GRAPH}/graph"))
    other = svg_of(page(app_for, second, client_for, f"/graphs/{GRAPH}/graph"))
    assert one == again == other
    cites = [svg_of(page(app_for, h, client_for, f"/graphs/{GRAPH}/graph?view=citation")) for h in (first, second)]
    assert cites[0] == cites[1]


def test_the_concept_view_draws_edges_only_at_two_shared_papers_and_links_dashed(home_a, app_for, client_for):
    svg = svg_of(page(app_for, home_a, client_for, f"/graphs/{GRAPH}/graph"))
    label = names(svg)
    drawn = {tuple(sorted((label[x], label[y]))): dashed for x, y, dashed in edges(svg)}
    assert drawn == {("sprocket balance", "sprocket rule 0"): False,  # lp_0_0 and lp_extra
                     ("auto wobble", "serial wobble"): True}  # a closely related link, one paper each
    rings = {label[n] for n, cls in nodes(svg).items() if "sg-bridge" in cls}
    assert rings == {"widget lag"}
    assert 'href="/concepts/method/widget%20lag"' in svg and "<title>widget lag (method): 2 papers, a bridge" in svg


def test_the_concept_view_keeps_the_top_n(home_a):
    lib = open_library_at(home_a / "library")
    try:
        g = graph_svg.concept_graph(lib, GRAPH, top_n=3)
    finally:
        lib.close()
    # weight x papers: log(16/6)*6 and log(16/5)*5 beat every two-paper concept's log(8)*2
    assert sorted(n.label for n in g.nodes) == ["sprocket rule 0", "sprocket rule 1", "sprocket rule 2"]


def test_the_citation_view_colours_papers_by_literature_and_unplaced_grey(home_a, app_for, client_for):
    svg = svg_of(page(app_for, home_a, client_for, f"/graphs/{GRAPH}/graph?view=citation"))
    drawn = nodes(svg)
    assert len(drawn) == GROUPS * SIZE + 1
    assert drawn[EXTRA] == "sg-lit-none"  # joined after the grouping: unplaced
    classes = {drawn[p(c, i)] for c in range(GROUPS) for i in range(SIZE)}
    assert classes == {"sg-lit-1", "sg-lit-2", "sg-lit-3"}
    for c in range(GROUPS):
        assert len({drawn[p(c, i)] for i in range(SIZE)}) == 1  # one literature per citation clique
    assert len(edges(svg)) == GROUPS * SIZE * (SIZE - 1) // 2
    assert f'href="/papers/{p(0, 0)}"' in svg


def test_above_the_node_limit_only_placed_papers_are_drawn(home_a):
    lib = open_library_at(home_a / "library")
    try:
        g = graph_svg.citation_graph(lib, GRAPH, max_nodes=10)
    finally:
        lib.close()
    assert len(g.nodes) == GROUPS * SIZE and g.omitted == 1


def test_the_graph_tab_has_no_inline_style_or_script_and_toggles_by_htmx(home_a, app_for, client_for):
    c = client_for(app_for(home_a))
    html = c.get(f"/graphs/{GRAPH}/graph").text
    assert no_inline_script_or_style(html)
    assert f'href="/graphs/{GRAPH}/graph?view=citation"' in html
    assert f'hx-get="/graphs/{GRAPH}/graph/svg?view=citation"' in html
    partial = c.get(f"/graphs/{GRAPH}/graph/svg?view=citation", headers={"hx-request": "true"}).text
    assert no_inline_script_or_style(partial) and "sg-graph-citation" in partial and "<html" not in partial
    assert c.get("/graphs/nope/graph").status_code == 404


def test_the_layout_is_independent_of_input_order():
    nodes_ = ["c", "a", "b", "d"]
    edges_ = [("a", "b", 2.0), ("c", "d", 1.0), ("b", "c", 1.0)]
    assert graph_svg.spring_layout(nodes_, edges_) == graph_svg.spring_layout(nodes_[::-1], edges_[::-1])
