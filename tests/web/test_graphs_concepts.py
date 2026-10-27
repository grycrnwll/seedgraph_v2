"""Graph page Concepts tab (stage 4 design §6.1, §7): the graph slice (top concepts per
family with counts, slice weights and sense glosses, linking to the concept pages) and the
outside-graph suggestions ("n papers elsewhere share this ...") with Add after a
confirmation step. Suggestions come only from confirmed senses or links. Invented text."""

from __future__ import annotations

import re

import pytest
from concepts.test_bridges import GRAPH, OUTSIDE, build_world, confirm

from seedgraph.library import graphs
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


def with_lib(home, fn):
    lib = open_library_at(home / "library")
    try:
        return fn(lib)
    finally:
        lib.close()


def membership(home):
    def read(lib):
        with graphs.open_graph(lib, GRAPH) as g:
            return {r.paper_id: r.status for r in graphs.membership(g)}
    return with_lib(home, read)


def concept_row(html: str, name: str) -> str:
    m = re.search(rf"<tr>(?:(?!</tr>).)*>{re.escape(name)}</a>(?:(?!</tr>).)*</tr>", html, re.S)
    assert m, name
    return m.group(0)


def test_the_slice_lists_top_concepts_per_family_with_counts_weights_and_links(world, make_app, client_for):
    html = client_for(make_app()).get(f"/graphs/{GRAPH}/concepts").text
    assert re.search(r"<h2[^>]*>assumption</h2>", html) and re.search(r"<h2[^>]*>method</h2>", html)
    row = concept_row(html, "sprocket rule 0")
    assert 'href="/concepts/assumption/sprocket%20rule%200"' in row
    assert ">5<" in row and "1.10" in row  # 5 papers of 15 members: log(3)
    assert "bridge" in concept_row(html, "widget lag")
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)


def test_sense_glosses_are_shown_with_who_wrote_them(world, make_app, client_for):
    with_lib(world, lambda lib: confirm(lib, "assumption", "sprocket rule 1"))
    html = client_for(make_app()).get(f"/graphs/{GRAPH}/concepts").text
    row = concept_row(html, "sprocket rule 1")
    assert "an invented gloss" in row and "researcher" in row


def test_unchecked_senses_suggest_nothing_outside_the_graph(world, make_app, client_for):
    html = client_for(make_app()).get(f"/graphs/{GRAPH}/concepts").text
    assert "elsewhere" not in html and OUTSIDE[0] not in html and OUTSIDE[1] not in html


def test_a_confirmed_sense_suggests_papers_elsewhere_that_share_it(world, make_app, client_for):
    with_lib(world, lambda lib: confirm(lib, "assumption", "sprocket rule 0"))
    html = client_for(make_app()).get(f"/graphs/{GRAPH}/concepts").text
    row = concept_row(html, "sprocket rule 0")
    assert "1 paper elsewhere shares this" in row
    assert f'href="/graphs/{GRAPH}/suggestions/confirm?paper={OUTSIDE[0]}"' in row
    assert OUTSIDE[1] not in row


def test_a_link_carries_suggestions_to_the_related_concept(world, make_app, client_for):
    with_lib(world, lambda lib: confirm(lib, "method", "auto wobble"))
    html = client_for(make_app()).get(f"/graphs/{GRAPH}/concepts").text
    assert OUTSIDE[1] in concept_row(html, "auto wobble")
    serial = concept_row(html, "serial wobble")
    assert OUTSIDE[1] in serial and "via auto wobble" in serial


def test_adding_a_suggestion_asks_first_then_adds_a_held_paper_as_a_member(world, make_app, client_for,
                                                                          monkeypatch):
    with_lib(world, lambda lib: confirm(lib, "assumption", "sprocket rule 0"))
    calls = []
    real = graphs.set_membership

    def spy(g, paper_id, status, **kw):
        calls.append((paper_id, status))
        return real(g, paper_id, status, **kw)

    monkeypatch.setattr(graphs, "set_membership", spy)
    c = client_for(make_app())
    page = c.get(f"/graphs/{GRAPH}/suggestions/confirm?paper={OUTSIDE[0]}")
    assert page.status_code == 200 and "as a member" in page.text and calls == []
    assert no_inline_script_or_style(page.text) and post_forms_lead_with_csrf(page.text)
    data = {"paper_id": OUTSIDE[0]}
    assert c.post(f"/graphs/{GRAPH}/suggestions/add", data=data).status_code == 403
    r = c.post(f"/graphs/{GRAPH}/suggestions/add", data={"csrf_token": token(c), **data})
    assert r.status_code == 303 and calls == [(OUTSIDE[0], "member")]
    assert membership(world)[OUTSIDE[0]] == "member"
    assert "added" in c.get(r.headers["location"]).text


def test_a_paper_that_is_not_a_suggestion_cannot_be_added_that_way(world, make_app, client_for):
    c = client_for(make_app())
    assert c.get(f"/graphs/{GRAPH}/suggestions/confirm?paper={OUTSIDE[0]}").status_code == 404
    r = c.post(f"/graphs/{GRAPH}/suggestions/add", data={"csrf_token": token(c), "paper_id": OUTSIDE[0]})
    assert r.status_code == 303 and OUTSIDE[0] not in membership(world)
    assert "sg-flash-error" in c.get(r.headers["location"]).text


def test_an_excluded_paper_is_never_suggested_again(world, make_app, client_for):
    def setup(lib):
        confirm(lib, "assumption", "sprocket rule 0")
        with graphs.open_graph(lib, GRAPH) as g:
            graphs.set_membership(g, OUTSIDE[0], "excluded")
    with_lib(world, setup)
    html = client_for(make_app()).get(f"/graphs/{GRAPH}/concepts").text
    assert f"paper={OUTSIDE[0]}" not in html
