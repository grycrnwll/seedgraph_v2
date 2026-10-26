"""Note editing on the paper page (docs/V3_UI_DESIGN.md §4, §5): edit / hide / add / revert
round-trips through POST with CSRF, stale versions refused, selection anchoring, the text
view fallback. Invented text only."""

from __future__ import annotations

import re

import pytest

from seedgraph.notes import interpretation, overlay

from . import paper_support as ps


@pytest.fixture
def fx(home, tmp_path):
    return ps.build(home, tmp_path)


@pytest.fixture
def client(fx, make_app, client_for):
    return client_for(make_app())


def head_version(fx) -> int:
    with fx.open() as lib:
        return overlay.head_version_no(lib.conn, fx.held)


def interp(fx, **kw):
    with fx.open() as lib:
        return interpretation.current(lib, fx.held, "researcher", **kw)


def claim(fx, claim_id, **kw):
    return next(c for c in interp(fx, **kw).claims if c.claim_id == claim_id)


def notes_post(client, fx, action, data, **kw):
    return ps.post(client, f"/papers/{fx.held}/notes/{action}", data, **kw)


# --- edit ------------------------------------------------------------------------------------


def test_an_edit_round_trips_and_shows_the_researcher_label_with_the_original(client, fx):
    cid = fx.claim("method_or_model[0]")
    r = notes_post(client, fx, "edit", {"expected_version": 0, "claim_id": cid, "claim_type": "method",
                                        "normalized_label": "widget estimator",
                                        "claim_text": "A quuxifying widget tally."})
    assert r.status_code == 200
    html = r.text
    assert html.lstrip().startswith('<div id="sg-notes"')  # the whole pane, for the htmx swap
    assert 'data-expected-version="1"' in html
    assert "A quuxifying widget tally." in html
    assert "edited from the agent&#39;s claim" in html
    assert re.search(r"agent's original</summary>.*A two-step widget estimator\.", html, re.S)
    c = claim(fx, cid)
    assert c.trust == "researcher" and c.claim_text == "A quuxifying widget tally."
    assert c.original.claim_text == "A two-step widget estimator."


def test_relabel_and_retype_send_only_the_changed_fields(client, fx):
    cid = fx.claim("method_or_model[0]")
    notes_post(client, fx, "edit", {"expected_version": 0, "claim_id": cid, "claim_type": "model",
                                    "normalized_label": "widget model", "claim_text": "A two-step widget estimator."})
    with fx.open() as lib:
        [op] = overlay.head(lib.conn, fx.held).ops
    assert op["set"] == {"claim_type": "model", "normalized_label": "widget model"}


def test_a_claim_without_a_label_can_be_reworded(client, fx):
    cid = fx.claim("research_question")
    html = client.get(f"/papers/{fx.held}/notes").text
    form = re.search(rf'value="{cid}">.*?</form>', html, re.S).group(0)
    assert re.search(r'name="normalized_label" value=""(?! required)', form)  # no browser block
    notes_post(client, fx, "edit", {"expected_version": 0, "claim_id": cid, "claim_type": "research_question",
                                    "normalized_label": "", "claim_text": "Do invented widgets really grow?"})
    with fx.open() as lib:
        [op] = overlay.head(lib.conn, fx.held).ops
    assert op["set"] == {"claim_text": "Do invented widgets really grow?"}


def test_the_pane_carries_one_version_input_for_forms_outside_it(client, fx):
    notes_post(client, fx, "hide", {"expected_version": 0, "claim_id": fx.claim("assumptions[1]")})
    html = client.get(f"/papers/{fx.held}/notes").text
    assert '<input type="hidden" id="sg-notes-version" name="expected_version" value="1">' in html
    anchor = ps.post(client, f"/papers/{fx.held}/anchor", {"page": 3, "text": "frumious index"}).text
    assert 'hx-include="#sg-notes-version"' in anchor


def test_an_unchanged_edit_writes_no_version(client, fx):
    cid = fx.claim("method_or_model[0]")
    r = notes_post(client, fx, "edit", {"expected_version": 0, "claim_id": cid, "claim_type": "method",
                                        "normalized_label": "widget estimator",
                                        "claim_text": "A two-step widget estimator."})
    assert "nothing changed" in r.text
    assert head_version(fx) == 0


def test_a_stale_expected_version_is_refused_and_writes_nothing(client, fx):
    cid = fx.claim("method_or_model[0]")
    notes_post(client, fx, "edit", {"expected_version": 0, "claim_id": cid, "claim_type": "method",
                                    "normalized_label": "widget estimator", "claim_text": "First tab."})
    r = notes_post(client, fx, "edit", {"expected_version": 0, "claim_id": cid, "claim_type": "method",
                                        "normalized_label": "widget estimator", "claim_text": "Second tab."})
    assert r.status_code == 200
    assert "this note changed in another tab; reload" in r.text
    assert 'data-expected-version="1"' in r.text  # re-rendered at the current version
    assert head_version(fx) == 1
    assert claim(fx, cid).claim_text == "First tab."


def test_a_plain_form_post_redirects_back_with_a_flash(client, fx):
    cid = fx.claim("assumptions[1]")
    r = notes_post(client, fx, "hide", {"expected_version": 0, "claim_id": cid}, htmx=False)
    assert r.status_code == 303 and r.headers["location"].startswith(f"/papers/{fx.held}")
    assert "claim hidden" in client.get(r.headers["location"]).text


# --- hide / unhide / add / remove / notes / revert --------------------------------------------


def test_hide_then_show_hidden_then_unhide(client, fx):
    cid = fx.claim("assumptions[1]")
    html = notes_post(client, fx, "hide", {"expected_version": 0, "claim_id": cid}).text
    assert f'data-claim-id="{cid}"' not in html
    html = client.get(f"/papers/{fx.held}/notes?show_hidden=1").text
    assert re.search(rf'data-claim-id="{cid}".*?data-hidden>hidden<', html, re.S)
    assert f'action="/papers/{fx.held}/notes/unhide"' in html
    html = notes_post(client, fx, "unhide", {"expected_version": 1, "claim_id": cid, "show_hidden": "1"}).text
    assert 'name="show_hidden" value="1"' in html  # the toggle survives the swap
    assert not claim(fx, cid, show_hidden=True).hidden


def test_add_an_assertion_is_labelled_as_the_researchers(client, fx):
    html = notes_post(client, fx, "add", {"expected_version": 0, "kind": "assertion", "claim_type": "limitation",
                                          "normalized_label": "frumious caveat",
                                          "claim_text": "An invented caveat about frumious widgets."}).text
    assert "An invented caveat about frumious widgets." in html
    assert "researcher assertion — not the paper&#39;s words" in html
    added = next(c for c in interp(fx).claims if c.kind == "assertion")
    assert added.trust == "researcher"
    html = notes_post(client, fx, "remove", {"expected_version": 1, "claim_id": added.claim_id}).text
    assert "An invented caveat about frumious widgets." not in html


def test_an_invalid_add_is_refused_with_its_reason(client, fx):
    html = notes_post(client, fx, "add", {"expected_version": 0, "kind": "assertion", "claim_type": "nonsense",
                                          "normalized_label": "x", "claim_text": "y"}).text
    assert "not saved: unknown claim type" in html
    assert head_version(fx) == 0


def test_researcher_notes_are_saved_and_shown(client, fx):
    html = notes_post(client, fx, "researcher-notes", {"expected_version": 0,
                                                       "text": "My invented reading of this paper."}).text
    assert "My invented reading of this paper.</textarea>" in html
    assert interp(fx).researcher_notes == "My invented reading of this paper."


def test_history_lists_versions_and_revert_restores_a_hidden_claim(client, fx):
    cid = fx.claim("assumptions[1]")
    notes_post(client, fx, "researcher-notes", {"expected_version": 0, "text": "Invented."})
    html = notes_post(client, fx, "hide", {"expected_version": 1, "claim_id": cid}).text
    assert 'data-version="2"' in html and 'data-version="1"' in html
    assert "Revert to v1" in html
    html = notes_post(client, fx, "revert", {"expected_version": 2, "to_version": 1}).text
    assert f'data-claim-id="{cid}"' in html
    assert "reverted to version 1" in html
    assert 'data-version="3"' in html
    with fx.open() as lib:
        assert overlay.head(lib.conn, fx.held).reverted_to == 1


def test_every_notes_post_requires_the_csrf_token(client, fx):
    cid = fx.claim("assumptions[1]")
    for action in ("edit", "hide", "unhide", "add", "remove", "researcher-notes", "revert"):
        r = notes_post(client, fx, action, {"expected_version": 0, "claim_id": cid}, csrf=False)
        assert r.status_code == 403, action
    for path in ("anchor", "anchor-text", "replay/accept", "metadata", "graphs", "write-notes", "add-pdf"):
        r = ps.post(client, f"/papers/{fx.held}/{path}", {"page": 1}, csrf=False)
        assert r.status_code == 403, path
    assert head_version(fx) == 0


# --- quote-backed claims: selection anchoring ---------------------------------------------


def test_a_unique_pdf_selection_anchors_and_offers_the_add_form(client, fx):
    r = ps.post(client, f"/papers/{fx.held}/anchor", {"page": 3, "text": "frumious index appears once"})
    assert r.status_code == 200
    html = r.text
    assert 'data-anchor-status="ok"' in html
    assert "data-anchor-quote>The frumious index appears once in" not in html  # widened to words only
    assert re.search(r"data-anchor-quote>frumious index appears once</blockquote>", html)
    form = re.search(r'<form method="post" action="/papers/[^"]+/notes/add"[^>]*>\s*(<input[^>]*>)', html)
    assert form and 'name="csrf_token"' in form.group(1)
    start = int(re.search(r'name="start_char" value="(\d+)"', html).group(1))
    end = int(re.search(r'name="end_char" value="(\d+)"', html).group(1))
    md_id = re.search(r'name="markdown_id" value="([^"]+)"', html).group(1)
    assert md_id == fx.markdown_id

    html = notes_post(client, fx, "add", {"expected_version": 0, "kind": "quote", "claim_type": "result",
                                          "normalized_label": "frumious once", "claim_text": "My reading.",
                                          "markdown_id": md_id, "start_char": start, "end_char": end}).text
    assert "quote-backed claim added" in html
    added = next(c for c in interp(fx).claims if c.kind == "quote")
    assert added.trust == "researcher" and added.span_id
    assert re.search(r"frumious index appears once.*?>p\. 3<", html, re.S)  # the quote has its page


def test_an_ambiguous_pdf_selection_is_refused_with_a_message(client, fx):
    html = ps.post(client, f"/papers/{fx.held}/anchor", {"page": 3, "text": "Widgets grow by an invented amount."}).text
    assert 'data-anchor-status="ambiguous"' in html
    assert "appears 2 times on this page; select a longer passage" in html
    assert "text view" in html


def test_a_selection_not_on_the_page_is_not_found(client, fx):
    html = ps.post(client, f"/papers/{fx.held}/anchor", {"page": 1, "text": "Nothing like this is in the paper"}).text
    assert 'data-anchor-status="not_found"' in html


def test_a_paper_without_markers_points_to_the_text_view(client, fx):
    html = ps.post(client, f"/papers/{fx.plain}/anchor", {"page": 1, "text": "invented widgets"}).text
    assert 'data-anchor-status="no_markers"' in html and "text view" in html


# --- the text view fallback ----------------------------------------------------------------


def test_the_text_view_shows_one_page_with_offsets(client, fx):
    html = client.get(f"/papers/{fx.held}/text?page=2").text
    assert ps.METHOD_QUOTE in html
    assert ps.RQ_QUOTE not in html
    assert '<strong aria-current="page">2</strong>' in html
    with fx.open() as lib:
        md = lib.conn.execute("SELECT storage_uri FROM markdown_documents WHERE markdown_id = ?",
                              (fx.markdown_id,)).fetchone()[0]
        text = (fx.lib_dir / md).read_text(encoding="utf-8")
    for off, chunk in re.findall(r'data-offset="(\d+)">([^<]*)<', html):
        assert text[int(off):int(off) + len(chunk)] == chunk.replace("&#39;", "'")


def test_the_text_view_of_a_paper_without_markers_shows_the_whole_text(client, fx):
    html = client.get(f"/papers/{fx.plain}/text").text
    assert "No page markers" in html
    assert ps.RQ_QUOTE in html and ps.RESULT_QUOTE in html


def test_a_text_view_selection_anchors_by_offset_and_refuses_duplicates(client, fx):
    html = client.get(f"/papers/{fx.plain}/text").text
    m = next((int(o), c) for o, c in re.findall(r'data-offset="(\d+)">([^<]*)<', html) if ps.RESULT_QUOTE in c)
    start = m[0] + m[1].index("frumious") + 2  # mid-word: widened to whole words
    end = start + len("umious index appears")
    ok = ps.post(client, f"/papers/{fx.plain}/anchor-text", {"start": start, "end": end}).text
    assert 'data-anchor-status="ok"' in ok
    assert "data-anchor-quote>frumious index appears</blockquote>" in ok

    dup = next((int(o), c) for o, c in re.findall(r'data-offset="(\d+)">([^<]*)<', html)
               if "Widgets grow by" in c)
    s = dup[0] + dup[1].index("Widgets grow by")
    amb = ps.post(client, f"/papers/{fx.plain}/anchor-text", {"start": s, "end": s + len("Widgets grow by")}).text
    assert 'data-anchor-status="ambiguous"' in amb
