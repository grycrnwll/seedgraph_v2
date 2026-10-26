"""The Library page (design §2.2): list views, the "Needs a PDF" view with DOI links,
the title filter and note search, and "Write notes" (estimate -> batch -> batch page)."""

from __future__ import annotations

import re

import pytest

from seedgraph.library.store import open_library_at
from seedgraph.web.views import library as library_view
from seedgraph.work import estimate, routine

from .pages_support import DOI, GRAPH, build, no_inline_script_or_style, post_forms_lead_with_csrf, token


@pytest.fixture
def fx(home):
    return build(home)


def rows_of(html: str) -> list[str]:
    return re.findall(r'<tr[^>]*data-paper="([^"]+)"', html)


def test_the_view_function_lists_held_papers_with_note_state_and_quotes(fx):
    lib = open_library_at(fx.home / "library")
    try:
        view = library_view.page(lib)
    finally:
        lib.close()
    by_id = {r.paper_id: r for r in view.rows}
    assert set(by_id) == {fx.noted, fx.no_note, fx.garbled}
    assert by_id[fx.noted].note_state == "current" and by_id[fx.no_note].note_state == "none"
    assert (by_id[fx.noted].quotes_anchored, by_id[fx.noted].quotes_total) == (5, 5)
    assert by_id[fx.noted].graphs == (GRAPH,)
    assert view.counts == {"held": 3, "cited-only": 2, "needs-notes": 2, "held-back": 2, "needs-pdf": 2, "all": 5}


def test_the_held_view_is_the_default_and_carries_pdf_buttons(fx, make_app, client_for):
    html = client_for(make_app()).get("/library").text
    assert sorted(rows_of(html)) == sorted([fx.noted, fx.no_note, fx.garbled])
    assert f'href="/papers/{fx.noted}/pdf"' in html and "5/5" in html
    assert 'href="/library" aria-current="page"' in html
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)


def test_cited_only_and_needs_notes_views(fx, make_app, client_for):
    c = client_for(make_app())
    cited = c.get("/library?view=cited-only").text
    assert sorted(rows_of(cited)) == [fx.cited, fx.cited_no_doi]
    assert f'href="/papers/{fx.cited}/pdf"' not in cited
    assert sorted(rows_of(c.get("/library?view=needs-notes").text)) == [fx.no_note, fx.garbled]


def test_the_held_back_view_shows_reasons_and_accept_text_where_it_applies(fx, make_app, client_for):
    html = client_for(make_app()).get("/library?view=held-back").text
    assert sorted(rows_of(html)) == [fx.no_note, fx.garbled]
    assert "the invented text layer looks garbled" in html
    assert f'action="/review/attention/{fx.garbled_item}/resolve"' in html and 'value="accept_text"' in html
    assert f'action="/review/attention/{fx.title_miss_item}/resolve"' not in html  # no accept_text for a PDF miss


def test_needs_a_pdf_lists_cited_only_papers_with_doi_links_in_a_new_tab(fx, make_app, client_for):
    c = client_for(make_app())
    html = c.get("/library?view=needs-pdf").text
    link = re.search(r'<a [^>]*href="https://doi.org/' + re.escape(DOI) + r'"[^>]*>', html).group(0)
    assert 'target="_blank"' in link and 'rel="noopener noreferrer"' in link
    assert "Another cited invented treatise" in html  # listed even without a DOI
    on_graph = c.get(f"/library?view=needs-pdf&graph={GRAPH}").text
    assert DOI in on_graph and "Another cited invented treatise" not in on_graph
    frontier = c.get("/library?view=needs-pdf&frontier=1").text
    assert DOI in frontier and "Another cited invented treatise" not in frontier


def test_an_unknown_graph_filter_is_404(fx, make_app, client_for):
    assert client_for(make_app()).get("/library?graph=nope").status_code == 404


def test_the_title_filter_narrows_and_the_search_finds_note_text(fx, make_app, client_for):
    c = client_for(make_app())
    assert rows_of(c.get("/library?q=garbled").text) == [fx.garbled]
    html = c.get("/library?view=all&q=Output+rises").text
    assert "Output rises." in html and "paper states" in html


def test_graph_filter_on_the_held_view(fx, make_app, client_for):
    assert sorted(rows_of(client_for(make_app()).get(f"/library?graph={GRAPH}").text)) == sorted(
        [fx.noted, fx.no_note, fx.garbled])


# --- Write notes -----------------------------------------------------------------------------


def test_write_notes_shows_the_estimate_and_the_skipped_papers(fx, make_app, client_for, monkeypatch):
    calls = []
    real = estimate.estimate

    def spy(lib, task, paper_ids, *, replace=False):
        calls.append((task, list(paper_ids), replace))
        return real(lib, task, paper_ids, replace=replace)

    monkeypatch.setattr(estimate, "estimate", spy)
    html = client_for(make_app()).get(f"/library/write-notes?paper={fx.no_note}&paper={fx.noted}").text
    assert calls == [("notes", [fx.no_note, fx.noted], False)]
    assert "agent tokens" in html and "text tokens" in html
    assert "skipped" in html.lower() and fx.noted in html  # already has a current note: preserved
    assert f'name="paper" value="{fx.no_note}"' in html
    assert f'name="paper" value="{fx.noted}"' not in html
    assert post_forms_lead_with_csrf(html) and no_inline_script_or_style(html)


def test_write_notes_for_all_in_a_view(fx, make_app, client_for):
    html = client_for(make_app()).get(f"/library/write-notes?view=needs-notes&graph={GRAPH}").text
    assert f'name="paper" value="{fx.no_note}"' in html


def test_write_notes_with_nothing_selected_says_so(fx, make_app, client_for):
    r = client_for(make_app()).get("/library/write-notes")
    assert r.status_code == 200 and "Select papers" in r.text


def test_confirm_creates_one_batch_and_goes_to_its_page(fx, make_app, client_for, monkeypatch):
    calls = []
    real = routine.create_batch

    def spy(lib, task, **kw):
        calls.append((task, kw.get("paper_ids"), kw.get("options")))
        return real(lib, task, **kw)

    monkeypatch.setattr(routine, "create_batch", spy)
    c = client_for(make_app())
    assert c.post("/library/write-notes", data={"paper": fx.garbled}).status_code == 403
    r = c.post("/library/write-notes", data={"csrf_token": token(c), "paper": [fx.garbled, fx.no_note]})
    assert r.status_code == 303 and len(calls) == 1
    assert calls[0][0] == "notes" and calls[0][1] == [fx.garbled, fx.no_note]
    assert calls[0][2] == {"replace": False}
    assert re.match(r"/batches/b_[0-9a-f]+", r.headers["location"])
    assert "claude" in c.get(r.headers["location"]).text.lower()
