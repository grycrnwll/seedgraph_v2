""""Fix it now" (stage 5 design §5.3, §6.6, §11 R6; slice R6): the link after partly right or
wrong, the paper page's ``edit=`` / ``from=`` parameters (the claim's edit form open, "show
hidden" for a hidden claim, the back link), the "fixed" chip once an edit follows, the
earlier-extraction notice, and the figure staying unchanged. Invented fixtures only."""

from __future__ import annotations

import html as htmllib
import re

import pytest

from seedgraph.notes import overlay
from seedgraph.spotcheck import accuracy as accuracy_mod
from seedgraph.spotcheck import sample as sample_mod

from .pages_support import no_inline_script_or_style, post_forms_lead_with_csrf, token
from .spotcheck_support import HEADERS, build, clock, draw, with_lib


@pytest.fixture
def sx(home):
    return build(home)


@pytest.fixture
def c(sx, make_app, client_for):
    return client_for(make_app(clock=clock()))


def mark(c, sid, cid, value, **extra):
    return c.post(f"/spot-check/{sid}/mark/{cid}", data={"csrf_token": token(c), "value": value, **extra},
                  headers=HEADERS)


def item_html(html: str, claim_id: str) -> str:
    m = re.search(r'<article class="sg-sc-item[^"]*" id="item-%s".*?</article>' % claim_id, html, re.S)
    assert m, claim_id
    return m.group(0)


def claim_article(html: str, claim_id: str) -> str | None:
    m = re.search(r'<article class="sg-claim[^"]*" id="claim-%s".*?</article>' % claim_id, html, re.S)
    return m.group(0) if m else None


def one_method(sx):
    """A one-item sample holding the DiD paper's method claim."""
    s = draw(sx.home, n=1, claim_type="method", graph="did-reading")
    return s.sample_id, s.items[0].claim_id


def fix_link(html: str, claim_id: str) -> str | None:
    m = re.search(r'<a class="sg-sc-fix-link" href="([^"]+)"', item_html(html, claim_id))
    return htmllib.unescape(m.group(1)) if m else None


# --- the link -------------------------------------------------------------------------------------------

def test_fix_it_now_shows_only_after_partly_right_or_wrong(sx, c):
    sid, cid = one_method(sx)
    assert fix_link(c.get(f"/spot-check/{sid}").text, cid) is None
    assert fix_link(mark(c, sid, cid, "right").text, cid) is None
    for value in ("partly", "wrong"):
        r = mark(c, sid, cid, value)
        assert fix_link(r.text, cid) == f"/papers/{sx.did}?edit={cid}&from=spot-check:{sid}#claim-{cid}", value
    assert fix_link(mark(c, sid, cid, "right").text, cid) is None       # changed back: no longer offered


def test_fix_it_now_opens_that_claims_edit_form_and_a_back_link(sx, c):
    sid, cid = one_method(sx)
    mark(c, sid, cid, "wrong")
    link = fix_link(c.get(f"/spot-check/{sid}").text, cid)
    r = c.get(link.split("#")[0])
    assert r.status_code == 200
    html = r.text
    assert f'<a href="/spot-check/{sid}" data-back-to-spot-check>Back to the spot-check</a>' in html
    art = claim_article(html, cid)
    assert art is not None and '<details class="sg-claim-edit" open' in art
    # only that claim's form is open
    assert html.count('<details class="sg-claim-edit" open') == 1
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)
    # the form is the ordinary stage 3 edit form
    assert f'hx-post="/papers/{sx.did}/notes/edit"' in art and f'name="claim_id" value="{cid}"' in art
    # without the parameters nothing is open and there is no back link
    plain = c.get(f"/papers/{sx.did}").text
    assert "sg-claim-edit\" open" not in plain and "Back to the spot-check" not in plain


def test_the_edit_parameter_ignores_an_unknown_claim_and_the_from_parameter_a_bad_sample(sx, c):
    r = c.get(f"/papers/{sx.did}?edit=claim_nope&from=spot-check:../x")
    assert r.status_code == 200
    assert 'sg-claim-edit" open' not in r.text and "Back to the spot-check" not in r.text
    r = c.get(f"/papers/{sx.did}?from=elsewhere:sc_abc123")
    assert "Back to the spot-check" not in r.text
    # a claim of another paper is not on this page
    other = sx.claim(sx.ate, "method_or_model[0]")
    assert 'sg-claim-edit" open' not in c.get(f"/papers/{sx.did}?edit={other}").text


def test_editing_through_the_opened_form_marks_the_item_fixed_and_keeps_the_figure(sx, c):
    sid, cid = one_method(sx)
    mark(c, sid, cid, "wrong", reasons=["wrong_type"])
    before = with_lib(sx.home, accuracy_mod.accuracy)
    page = c.get(f"/papers/{sx.did}?edit={cid}&from=spot-check:{sid}")
    version = re.search(r'name="expected_version" value="(\d+)"', page.text).group(1)
    r = c.post(f"/papers/{sx.did}/notes/edit", headers=HEADERS,
               data={"csrf_token": token(c), "expected_version": version, "claim_id": cid,
                     "claim_type": "result", "claim_text": "An edited claim of the researcher."})
    assert "edit saved" in r.text
    after = htmllib.unescape(c.get(f"/spot-check/{sid}").text)
    item = item_html(after, cid)
    assert 'data-fixed' in item and "you edited this" in item
    assert "A two-period difference-in-differences design." in item         # the agent's original
    assert "An edited claim of the researcher." not in item
    assert "Marked <strong>wrong</strong>" in item
    got = with_lib(sx.home, lambda lib: sample_mod.sample(lib, sid).item(cid))
    assert got.fixed and got.claim_type == "method"
    assert with_lib(sx.home, accuracy_mod.accuracy) == before


# --- hidden claims --------------------------------------------------------------------------------------

def test_a_hidden_claim_says_so_and_fix_it_now_shows_hidden_claims_with_the_form_open(sx, c):
    sid, cid = one_method(sx)
    overlay.save_edit(c.app.state.web.library(), sx.did, {"op": "hide", "target": {"claim_id": cid}},
                      expected_version=0)
    html = htmllib.unescape(mark(c, sid, cid, "partly").text)
    item = item_html(html, cid)
    assert "you hid this claim" in item and "A two-period difference-in-differences design." in item
    link = fix_link(html, cid)
    assert link == f"/papers/{sx.did}?edit={cid}&from=spot-check:{sid}#claim-{cid}"
    # the paper page turns on "show hidden" itself, so the form is there
    assert claim_article(c.get(f"/papers/{sx.did}").text, cid) is None
    page = c.get(link.split("#")[0])
    art = claim_article(page.text, cid)
    assert art is not None and "sg-claim-hidden" in art and '<details class="sg-claim-edit" open' in art
    assert 'name="show_hidden" value="1"' in art                       # saves keep hidden claims shown
    assert "Unhide" in art and "Hide hidden claims" in page.text
    assert "Back to the spot-check" in page.text


def test_hiding_after_the_mark_counts_as_fixed(sx, c):
    sid, cid = one_method(sx)
    mark(c, sid, cid, "wrong")
    overlay.save_edit(c.app.state.web.library(), sx.did, {"op": "hide", "target": {"claim_id": cid}},
                      expected_version=0)
    item = item_html(c.get(f"/spot-check/{sid}").text, cid)
    assert "data-fixed" in item and "you hid this claim" in item


# --- an earlier extraction ----------------------------------------------------------------------------------

def test_an_item_from_an_earlier_extraction_opens_the_paper_at_the_current_note(sx, c):
    sid, cid = one_method(sx)
    mark(c, sid, cid, "wrong")
    lib = c.app.state.web.library()
    from spotcheck.conftest import World

    World(lib).note(sx.did)                       # the paper is re-noted: the sampled claim is no longer current
    html = htmllib.unescape(c.get(f"/spot-check/{sid}").text)
    item = item_html(html, cid)
    assert "this claim is from an earlier extraction" in item
    assert "A two-period difference-in-differences design." in item
    assert "show in PDF" not in item
    assert fix_link(html, cid) == f"/papers/{sx.did}?from=spot-check:{sid}"
    page = c.get(f"/papers/{sx.did}?from=spot-check:{sid}")
    assert page.status_code == 200 and "Back to the spot-check" in page.text
    assert 'sg-claim-edit" open' not in page.text


def test_a_claim_of_the_current_note_has_no_earlier_extraction_notice(sx, c):
    sid, cid = one_method(sx)
    assert "earlier extraction" not in c.get(f"/spot-check/{sid}").text
