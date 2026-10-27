"""The claim edit fields on the paper page notes pane are full-width, auto-growing fields
(researcher request after the stage 3 walk-through): claim text, plain words and the
researcher notes are textareas marked for auto-grow; plain words can be edited.
Invented text only."""

from __future__ import annotations

import re

import pytest

from seedgraph.notes import interpretation

from . import paper_support as ps
from .pages_support import no_inline_script_or_style, post_forms_lead_with_csrf


@pytest.fixture
def fx(home, tmp_path):
    return ps.build(home, tmp_path)


@pytest.fixture
def client(fx, make_app, client_for):
    return client_for(make_app())


def _edit_form(html: str, claim_id: str) -> str:
    m = re.search(rf'id="claim-{claim_id}".*?<form[^>]*/notes/edit"(.*?)</form>', html, re.S)
    assert m, "no edit form for the claim"
    return m.group(1)


def test_the_edit_form_uses_auto_growing_textareas_for_long_text(client, fx):
    cid = fx.claim("method_or_model[0]")
    html = client.get(f"/papers/{fx.held}").text
    form = _edit_form(html, cid)
    assert re.search(r'<textarea[^>]*name="claim_text"[^>]*data-autogrow', form)
    assert re.search(r'<textarea[^>]*name="plain_words"[^>]*data-autogrow', form)
    assert "Count the widgets twice and average." in form  # the current plain words, prefilled
    assert re.search(r'<input type="text" name="normalized_label"', form)
    assert re.search(r'<select name="claim_type"', form)
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)


def test_researcher_notes_and_the_add_assertion_text_auto_grow(client, fx):
    html = client.get(f"/papers/{fx.held}").text
    assert re.search(r'<textarea[^>]*name="text"[^>]*data-autogrow', html)
    add = re.search(r'/notes/add"(.*?)</form>', html, re.S).group(1)
    assert re.search(r'<textarea[^>]*name="claim_text"[^>]*data-autogrow', add)


def test_the_edit_forms_stack_in_their_own_class(client, fx):
    html = client.get(f"/papers/{fx.held}").text
    assert 'class="sg-form sg-claim-form"' in html


def test_plain_words_can_be_edited(client, fx):
    cid = fx.claim("method_or_model[0]")
    r = ps.post(client, f"/papers/{fx.held}/notes/edit",
                {"expected_version": 0, "claim_id": cid, "claim_type": "method",
                 "normalized_label": "widget estimator", "claim_text": "A two-step widget estimator.",
                 "plain_words": "Tally each widget two times, then take the mean."})
    assert r.status_code == 200
    assert "Tally each widget two times, then take the mean." in r.text
    with fx.open() as lib:
        c = next(c for c in interpretation.current(lib, fx.held, "researcher").claims if c.claim_id == cid)
    assert c.plain_words == "Tally each widget two times, then take the mean."
    assert c.claim_text == "A two-step widget estimator." and c.trust == "researcher"


def test_paper_css_makes_fields_fill_the_notes_pane():
    from seedgraph.web.app import STATIC_DIR

    css = (STATIC_DIR / "paper.css").read_text(encoding="utf-8")
    assert "field-sizing: content" in css
    assert re.search(r"\.sg-claim-edit\[open\]\s*\{[^}]*flex:\s*1 1 100%", css)
