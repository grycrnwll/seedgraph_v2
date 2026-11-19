"""Spot-check screens (stage 5 design §5.3-5.4, §6.5, §13 Q4-Q8; slice R6): the accuracy
tables with their counts, starting a sample, the earlier samples, one sample's items, marking
and changing a mark by htmx with the running line, refusals, the page rules (GETs never write,
CSP, CSRF). "Fix it now" is in test_spotcheck_fix.py. Invented fixtures only."""

from __future__ import annotations

import html as htmllib
import re
import sqlite3

import pytest

from seedgraph.spotcheck import accuracy as accuracy_mod
from seedgraph.spotcheck import marks
from seedgraph.spotcheck import sample as sample_mod

from .pages_support import no_inline_script_or_style, post_forms_lead_with_csrf, token
from .spotcheck_support import GRAPH, HEADERS, MODEL_A, MODEL_B, build, clock, draw, with_lib


@pytest.fixture
def sx(home):
    return build(home)


@pytest.fixture
def c(sx, make_app, client_for):
    return client_for(make_app(clock=clock()))


def text(r) -> str:
    return htmllib.unescape(r.text)


def mark_post(c, sid: str, cid: str, value: str, *, htmx: bool = True, **extra):
    data = {"csrf_token": token(c), "value": value, **extra}
    return c.post(f"/spot-check/{sid}/mark/{cid}", data=data, headers=HEADERS if htmx else {})


def item_html(html: str, claim_id: str) -> str:
    m = re.search(r'<article class="sg-sc-item[^"]*" id="item-%s".*?</article>' % claim_id, html, re.S)
    assert m, claim_id
    return m.group(0)


def stored(home, sid: str, cid: str):
    return with_lib(home, lambda lib: sample_mod.sample(lib, sid).item(cid))


# --- the index ----------------------------------------------------------------------------------

def test_the_index_shows_accuracy_start_form_and_no_samples_yet(sx, c):
    r = c.get("/spot-check")
    assert r.status_code == 200
    html = text(r)
    assert 'href="/spot-check" aria-current="page"' in r.text
    assert "arrives in stage 5" not in html
    assert "too few marks to say (0 marked)" in html
    assert "No marks yet." in html and "No samples yet." in html
    # the start form: size default 20, graph / type / model choices from the library
    assert re.search(r'name="size"[^>]*value="20"', html)
    assert f'<option value="{GRAPH}">' in html
    assert f'<option value="{MODEL_A}">' in html and f'<option value="{MODEL_B}">' in html
    for t in ("research_question", "method", "identification_assumption", "result"):
        assert f'<option value="{t}">' in html, t
    assert 'action="/spot-check/start"' in html


def test_accuracy_tables_show_every_figure_with_its_count(sx, c):
    s = draw(sx.home, n=12)
    assert len(s.items) == 12
    for i, item in enumerate(s.items):
        mark_post(c, s.sample_id, item.claim_id, ("right", "right", "partly", "wrong")[i % 4])
    html = text(c.get("/spot-check"))
    table = with_lib(sx.home, accuracy_mod.accuracy)
    rows = (*table.by_type, *table.by_model)
    assert any(not r.too_few for r in table.by_model)         # a model with 5 or more marks has real figures
    assert any(r.too_few for r in table.by_type)              # a claim type with fewer says so
    for row in rows:                                          # every row of the deep module, with its own count
        assert row.summary in html
    assert re.search(r"right \d+ of \d+ \(\d+%\); partly right \d+; wrong \d+", html)
    assert "too few marks to say (" in html and " marked)" in html
    assert table.overall.summary in html and "weighted to the library's mix of types and models" in html


def test_earlier_samples_list_newest_first_with_marked_counts(sx, c):
    first = draw(sx.home, n=2, seed=1)
    second = draw(sx.home, n=3, seed=2)
    mark_post(c, first.sample_id, first.items[0].claim_id, "right")
    html = c.get("/spot-check").text
    ids = re.findall(r'<tr id="sample-(sc_[0-9a-f]+)">', html)
    assert set(ids) == {first.sample_id, second.sample_id}
    row = re.search(r'<tr id="sample-%s">(.*?)</tr>' % first.sample_id, html, re.S).group(1)
    assert "1 of 2" in row and f'href="/spot-check/{first.sample_id}"' in row


# --- starting a sample -------------------------------------------------------------------------------

def test_start_creates_a_sample_with_the_filters_and_goes_to_it(sx, c):
    r = c.post("/spot-check/start", data={"csrf_token": token(c), "size": "4", "graph": "", "claim_type": "method",
                                          "model": ""})
    assert r.status_code == 303
    sid = r.headers["location"].split("?")[0].rsplit("/", 1)[1]
    s = with_lib(sx.home, lambda lib: sample_mod.sample(lib, sid))
    assert s is not None and s.size == 4 and len(s.items) == 4
    assert {i.claim_type for i in s.items} == {"method"}
    assert s.filters == {"claim_type": "method"}
    page = c.get(r.headers["location"])
    assert page.status_code == 200 and "started a sample of 2 claims" in page.text


def test_start_defaults_the_size_to_twenty_and_applies_graph_and_model_filters(sx, c):
    r = c.post("/spot-check/start", data={"csrf_token": token(c), "graph": GRAPH})
    sid = r.headers["location"].split("?")[0].rsplit("/", 1)[1]
    s = with_lib(sx.home, lambda lib: sample_mod.sample(lib, sid))
    assert s.size == 20 and {i.paper_id for i in s.items} == {sx.did}
    r = c.post("/spot-check/start", data={"csrf_token": token(c), "size": "20", "model": MODEL_B})
    sid = r.headers["location"].split("?")[0].rsplit("/", 1)[1]
    assert {i.model_key for i in with_lib(sx.home, lambda lib: sample_mod.sample(lib, sid)).items} == {MODEL_B}


@pytest.mark.parametrize("size", ["0", "-3", "abc", "1000", "2.5"])
def test_start_refuses_a_bad_size_and_draws_nothing(sx, c, size):
    r = c.post("/spot-check/start", data={"csrf_token": token(c), "size": size})
    assert r.status_code == 303 and r.headers["location"].startswith("/spot-check?")
    assert "whole number from 1 to 100" in c.get(r.headers["location"]).text
    assert with_lib(sx.home, sample_mod.samples) == []


def test_start_with_nothing_to_draw_says_why(sx, c):
    r = c.post("/spot-check/start", data={"csrf_token": token(c), "size": "5", "model": "no such model"})
    assert r.status_code == 303 and r.headers["location"].startswith("/spot-check?")
    assert "no unmarked agent claims to spot-check with these filters" in text(c.get(r.headers["location"]))
    assert with_lib(sx.home, sample_mod.samples) == []


def test_each_post_needs_the_csrf_token(sx, c):
    s = draw(sx.home, n=1)
    assert c.post("/spot-check/start", data={"size": "3"}).status_code == 403
    assert c.post(f"/spot-check/{s.sample_id}/mark/{s.items[0].claim_id}", data={"value": "right"}).status_code == 403
    assert with_lib(sx.home, lambda lib: len(sample_mod.samples(lib))) == 1


def test_start_calls_one_deep_function(sx, make_app, client_for, monkeypatch):
    calls = []
    real = sample_mod.create_sample

    def spy(lib, **kw):
        calls.append(kw)
        return real(lib, **kw)

    monkeypatch.setattr(sample_mod, "create_sample", spy)
    c = client_for(make_app(clock=clock()))
    c.post("/spot-check/start", data={"csrf_token": token(c), "size": "3", "claim_type": "result"})
    assert len(calls) == 1 and calls[0]["n"] == 3 and calls[0]["filters"]["claim_type"] == "result"
    assert calls[0]["now"].startswith("2026-03-01T09:00:00")   # the app clock is passed through


# --- one sample ---------------------------------------------------------------------------------------

def test_the_sample_page_shows_the_original_claims_in_draw_order(sx, c):
    s = draw(sx.home, n=6)
    r = c.get(f"/spot-check/{s.sample_id}")
    assert r.status_code == 200
    html = text(r)
    assert "marked 0 of 6: right 0, partly right 0, wrong 0" in html
    assert 'href="/spot-check"' in html and "Back to the spot-check" in html
    order = re.findall(r'<article class="sg-sc-item[^"]*" id="item-([^"]+)"', html)
    assert order == [i.claim_id for i in s.items]
    method = next(i for i in s.items if i.claim_type == "method")
    one = item_html(html, method.claim_id)
    assert "A two-period difference-in-differences design." in one
    assert "difference in differences" in one and "Compare the change in treated units" in one
    assert "We use a difference-in-differences design with two periods." in one      # the quote
    assert method.model_key in one and "extracted " in one
    assert method.title in one and f'href="/papers/{method.paper_id}/pdf"' in one    # title and PDF button
    assert f'href="/papers/{method.paper_id}#claim-{method.claim_id}"' in one       # show in PDF
    for v in ("right", "partly", "wrong"):
        assert f'data-mark-button="{v}"' in one
    for reason in ("wrong type", "text misreads the paper", "quote does not support it", "not in the paper"):
        assert reason in one
    assert "Fix it now" not in one and "data-fixed" not in one


def test_an_item_without_a_quote_or_text_still_renders(sx, c):
    s = draw(sx.home, n=12, claim_type="identification_assumption")
    serial = next(i for i in s.items if i.normalized_label == "no serial correlation")
    assert serial.quote is None
    conn = sqlite3.connect(sx.home / "library" / "library.db")
    try:
        conn.execute("UPDATE extracted_claims SET claim_text = NULL, normalized_label = NULL, plain_words = NULL "
                     "WHERE claim_id = ?", (serial.claim_id,))
        conn.commit()
    finally:
        conn.close()
    r = c.get(f"/spot-check/{s.sample_id}")
    assert r.status_code == 200
    assert "gave no text or quote" in item_html(text(r), serial.claim_id)


def test_unknown_sample_and_claim_are_404(sx, c):
    s = draw(sx.home, n=1)
    assert c.get("/spot-check/sc_none").status_code == 404
    assert mark_post(c, "sc_none", s.items[0].claim_id, "right").status_code == 404
    assert mark_post(c, s.sample_id, "claim_other", "right").status_code == 404


# --- marking ------------------------------------------------------------------------------------------

def test_marking_by_htmx_returns_the_item_and_the_running_line(sx, c):
    s = draw(sx.home, n=4)
    first, second = s.items[0], s.items[1]
    r = mark_post(c, s.sample_id, first.claim_id, "right")
    assert r.status_code == 200
    html = text(r)
    assert "<html" not in html and "<article" in html            # a partial, not a page
    assert f'id="item-{first.claim_id}"' in html and 'data-mark="right"' in html
    assert 'aria-pressed="true"' in item_html(html, first.claim_id)
    assert re.search(r'<p id="sc-running"[^>]*hx-swap-oob="true"[^>]*>marked 1 of 4: right 1, partly right 0, wrong 0</p>',
                     html)
    mark_post(c, s.sample_id, second.claim_id, "wrong", reasons=["not_in_paper", "wrong_type"], note="  invented\nnote ")
    got = stored(sx.home, s.sample_id, second.claim_id)
    assert got.mark == "wrong" and got.reasons == ("wrong_type", "not_in_paper") and got.note == "invented note"
    page = text(c.get(f"/spot-check/{s.sample_id}"))
    assert "marked 2 of 4: right 1, partly right 0, wrong 1" in page
    assert "Marked <strong>wrong</strong> (wrong type, not in the paper)" in item_html(page, second.claim_id)
    assert "invented note" in item_html(page, second.claim_id)


def test_changing_a_mark_replaces_it_and_the_last_mark_counts(sx, c):
    s = draw(sx.home, n=3)
    cid = s.items[0].claim_id
    mark_post(c, s.sample_id, cid, "wrong", reasons=["misreads"])
    r = mark_post(c, s.sample_id, cid, "partly")
    assert 'data-mark="partly"' in r.text
    assert "marked 1 of 3: right 0, partly right 1, wrong 0" in text(r)
    got = stored(sx.home, s.sample_id, cid)
    assert got.mark == "partly" and got.reasons == ()
    marks_in_db = with_lib(sx.home, lambda lib: lib.conn.execute(
        "SELECT COUNT(*) FROM spot_check_items WHERE mark IS NOT NULL").fetchone()[0])
    assert marks_in_db == 1


def test_a_refused_mark_says_why_and_changes_nothing(sx, c):
    s = draw(sx.home, n=2)
    cid = s.items[0].claim_id
    r = mark_post(c, s.sample_id, cid, "right", reasons=["misreads"])
    assert r.status_code == 200
    assert "not saved:" in text(r) and "partly right or wrong" in text(r)
    assert stored(sx.home, s.sample_id, cid).mark is None
    r = mark_post(c, s.sample_id, cid, "half")
    assert "not saved:" in text(r) and "not a mark" in text(r)
    assert stored(sx.home, s.sample_id, cid).mark is None


def test_marking_without_htmx_redirects_back_to_the_item(sx, c):
    s = draw(sx.home, n=2)
    cid = s.items[1].claim_id
    r = mark_post(c, s.sample_id, cid, "right", htmx=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith(f"/spot-check/{s.sample_id}?flash=")
    assert r.headers["location"].endswith(f"#item-{cid}")
    assert stored(sx.home, s.sample_id, cid).mark == "right"


def test_a_mark_calls_one_deep_function_with_the_app_clock(sx, make_app, client_for, monkeypatch):
    s = draw(sx.home, n=2)
    calls = []
    real = marks.mark

    def spy(lib, sample_id, claim_id, value, **kw):
        calls.append((sample_id, claim_id, value, kw))
        return real(lib, sample_id, claim_id, value, **kw)

    monkeypatch.setattr(marks, "mark", spy)
    c = client_for(make_app(clock=clock()))
    mark_post(c, s.sample_id, s.items[0].claim_id, "partly", reasons=["misreads"], note="x")
    assert len(calls) == 1
    sid, cid, value, kw = calls[0]
    assert (sid, cid, value) == (s.sample_id, s.items[0].claim_id, "partly")
    assert kw["reasons"] == ["misreads"] and kw["note"] == "x" and kw["now"].startswith("2026-03-01T09:00:00")
    assert stored(sx.home, s.sample_id, cid).marked_at.startswith("2026-03-01T09:00:00")


def test_the_last_mark_finishes_the_sample(sx, c):
    s = draw(sx.home, n=2)
    for i in s.items:
        mark_post(c, s.sample_id, i.claim_id, "right")
    assert "· finished" in text(c.get(f"/spot-check/{s.sample_id}"))
    assert "(finished)" in text(c.get("/spot-check"))


# --- page rules -------------------------------------------------------------------------------------

def test_gets_never_write_and_pages_follow_the_page_rules(sx, make_app, client_for):
    s = draw(sx.home, n=4)
    c = client_for(make_app(clock=clock()))
    mark_post(c, s.sample_id, s.items[0].claim_id, "wrong")
    db = sx.home / "library" / "library.db"

    def dump():
        conn = sqlite3.connect(db)
        try:
            return list(conn.iterdump())
        finally:
            conn.close()

    before = dump()
    for path in ("/spot-check", f"/spot-check/{s.sample_id}"):
        r = c.get(path)
        assert r.status_code == 200
        assert no_inline_script_or_style(r.text), path
        assert post_forms_lead_with_csrf(r.text), path
        assert 'href="/static/spotcheck.css"' in r.text
    assert dump() == before
