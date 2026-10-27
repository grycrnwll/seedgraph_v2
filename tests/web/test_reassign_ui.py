""""This isn't the paper" on the paper page and in Review: choose the paper the PDF belongs to
(cover title suggestions + title search) or a new paper, preview the plan, confirm with CSRF.
Invented text only."""

from __future__ import annotations

import re

import pytest

from seedgraph.library import attention, files, graphs
from seedgraph.library.store import is_held, open_library_at

from . import paper_support as ps
from .pages_support import no_inline_script_or_style, post_forms_lead_with_csrf

SRC, TGT, DECOY, HELD = "lp_src", "lp_tgt", "lp_decoy", "lp_held"
TGT_TITLE = "Harbour Crane Productivity Under Invented Tides"
WRONG = [f"{TGT_TITLE}. Ada Inventor. Journal of Invented Ports 2029.", "Invented cranes lift boxes.", "The end."]


class Fx:
    def __init__(self, home, sf, item_id):
        self.home, self.sf, self.item_id = home, sf, item_id

    def open(self):
        return open_library_at(self.home / "library")


@pytest.fixture
def fx(home, tmp_path):
    lib = open_library_at(home / "library")
    try:
        ps._paper(lib, SRC, "Seasonal Patterns in Invented Lighthouse Maintenance")
        ps._paper(lib, TGT, TGT_TITLE, title_source="openalex")
        ps._paper(lib, DECOY, "Harbour Crane Scheduling in Invented Ports")
        ps._paper(lib, HELD, "Ledger Methods for Invented Granaries")
        res = files.attach_pdf(lib, SRC, ps.text_pdf(tmp_path / "wrong.pdf", WRONG))
        files.attach_pdf(lib, HELD, ps.text_pdf(tmp_path / "right.pdf", ["Ledger Methods for Invented Granaries"]))
        with graphs.create_graph(lib, "harbours") as g:
            graphs.set_membership(g, SRC, "member")
    finally:
        lib.close()
    assert res.attention_item_id
    return Fx(home, res.source_file_id, res.attention_item_id)


@pytest.fixture
def client(fx, make_app, client_for):
    return client_for(make_app())


def hidden(html, name):
    m = re.search(rf'name="{name}" value="([^"]*)"', html)
    return m.group(1) if m else None


def confirm_data(html):
    return {k: hidden(html, k) or "" for k in ("file", "to", "new", "fingerprint", "next")}


def held(fx, pid):
    with fx.open() as lib:
        return is_held(lib.conn, pid)


def check_page(html):
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)


def test_the_paper_page_offers_this_is_not_the_papers_pdf(client, fx):
    html = client.get(f"/papers/{SRC}").text
    assert f'href="/papers/{SRC}/reassign-pdf"' in html
    check_page(html)
    assert f'/papers/{TGT}/reassign-pdf"' not in client.get(f"/papers/{TGT}").text  # cited-only: no PDF


def test_the_chooser_ranks_the_cover_title_first_and_searches_titles(client, fx):
    r = client.get(f"/papers/{SRC}/reassign-pdf")
    assert r.status_code == 200
    html = r.text
    rows = re.findall(r'data-suggestion="(lp_\w+)"', html)
    assert rows[0] == TGT and DECOY in rows and SRC not in rows
    assert "1.00" in html
    assert f"/papers/{SRC}/reassign-pdf/preview?file={fx.sf}&amp;to={TGT}" in html
    assert f"/papers/{SRC}/reassign-pdf/preview?file={fx.sf}&amp;new=1" in html
    assert re.search(r'<form[^>]*method="get"[^>]*>.*name="q"', html, re.S)
    check_page(html)
    found = re.findall(r'data-search-hit="(lp_\w+)"', client.get(f"/papers/{SRC}/reassign-pdf?q=granaries").text)
    assert found == [HELD]


def test_the_preview_shows_what_moves_and_is_read_only(client, fx):
    r = client.get(f"/papers/{SRC}/reassign-pdf/preview?file={fx.sf}&to={TGT}")
    assert r.status_code == 200
    html = r.text
    assert TGT_TITLE in html and "becomes held" in html and "becomes cited-only" in html
    assert "harbours" in html and "frontier" in html and "Add to graph" in html
    assert hidden(html, "fingerprint") and "Confirm move" in html
    check_page(html)
    assert held(fx, SRC) and not held(fx, TGT)


def test_confirm_needs_the_csrf_token(client, fx):
    html = client.get(f"/papers/{SRC}/reassign-pdf/preview?file={fx.sf}&to={TGT}").text
    r = ps.post(client, f"/papers/{SRC}/reassign-pdf", confirm_data(html), csrf=False, htmx=False)
    assert r.status_code == 403
    assert held(fx, SRC)


def test_confirm_from_the_paper_page_moves_the_pdf(client, fx):
    html = client.get(f"/papers/{SRC}/reassign-pdf/preview?file={fx.sf}&to={TGT}").text
    r = ps.post(client, f"/papers/{SRC}/reassign-pdf", confirm_data(html), htmx=False)
    assert r.status_code == 303 and r.headers["location"].startswith(f"/papers/{TGT}")
    assert held(fx, TGT) and not held(fx, SRC)
    with fx.open() as lib:
        assert attention.get_item(lib, fx.item_id).resolution == "wrong_pdf_moved"


def test_a_stale_preview_is_refused_with_409(client, fx):
    html = client.get(f"/papers/{SRC}/reassign-pdf/preview?file={fx.sf}&to={TGT}").text
    data = {**confirm_data(html), "fingerprint": "0" * 64}
    r = ps.post(client, f"/papers/{SRC}/reassign-pdf", data, htmx=False)
    assert r.status_code == 409 and "changed since" in r.text
    check_page(r.text)
    assert held(fx, SRC)


def test_a_target_with_another_pdf_is_blocked(client, fx):
    html = client.get(f"/papers/{SRC}/reassign-pdf/preview?file={fx.sf}&to={HELD}").text
    assert "already has a different PDF" in html and "merge" in html and "Confirm move" not in html
    r = ps.post(client, f"/papers/{SRC}/reassign-pdf",
                {"file": fx.sf, "to": HELD, "fingerprint": "x"}, htmx=False)
    assert r.status_code == 409
    assert held(fx, SRC)


def test_the_new_paper_path_from_the_paper_page(client, fx):
    html = client.get(f"/papers/{SRC}/reassign-pdf/preview?file={fx.sf}&new=1").text
    assert "new paper" in html and "metadata" in html
    r = ps.post(client, f"/papers/{SRC}/reassign-pdf", confirm_data(html), htmx=False)
    assert r.status_code == 303
    new_id = re.match(r"/papers/(lp_\w+)", r.headers["location"]).group(1)
    assert new_id not in (SRC, TGT) and held(fx, new_id) and not held(fx, SRC)


def test_review_offers_wrong_pdf_and_confirm_returns_to_review(client, fx):
    page = client.get("/review/attention?kind=pdf_title_miss").text
    link = f"/papers/{SRC}/reassign-pdf?file={fx.sf}&amp;next=/review/attention%3Fkind%3Dpdf_title_miss"
    assert link in page
    assert 'value="wrong_pdf_moved"' not in page  # a link to the chooser, never a bare resolution
    check_page(page)
    chooser = client.get(link.replace("&amp;", "&")).text
    assert "next=%2Freview%2Fattention%3Fkind%3Dpdf_title_miss" in chooser
    preview = client.get(f"/papers/{SRC}/reassign-pdf/preview?file={fx.sf}&to={TGT}"
                         f"&next=/review/attention%3Fkind%3Dpdf_title_miss").text
    assert hidden(preview, "next") == "/review/attention?kind=pdf_title_miss"
    r = ps.post(client, f"/papers/{SRC}/reassign-pdf", confirm_data(preview), htmx=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/review/attention?kind=pdf_title_miss")
    assert held(fx, TGT)
    assert fx.item_id not in client.get("/review/attention?kind=pdf_title_miss").text


def test_a_bare_wrong_pdf_resolution_is_not_accepted(client, fx):
    r = ps.post(client, f"/review/attention/{fx.item_id}/resolve", {"resolution": "wrong_pdf_moved"}, htmx=False)
    assert r.status_code == 400


def test_a_shared_pdf_row_offers_wrong_pdf_for_each_holder(client, fx, tmp_path):
    with fx.open() as lib:
        files.attach_pdf(lib, TGT, ps.text_pdf(tmp_path / "again.pdf", WRONG))
        attention.raise_item(lib, "shared_pdf", {"source_file_id": fx.sf}, detail="{}")
    page = client.get("/review/attention?kind=shared_pdf").text
    for pid in (SRC, TGT):
        assert f"/papers/{pid}/reassign-pdf?file={fx.sf}" in page, pid
    check_page(page)


def test_a_blocked_move_offers_a_real_merge(client, fx):
    html = client.get(f"/papers/{SRC}/reassign-pdf/preview?file={fx.sf}&to={HELD}").text
    assert f'action="/papers/{SRC}/reassign-pdf/merge"' in html and "Merge these papers" in html
    check_page(html)
    r = ps.post(client, f"/papers/{SRC}/reassign-pdf/merge", {"to": HELD}, htmx=False)
    assert r.status_code == 303
    m = re.match(r"/review/merge\?item=(att_\w+)&with=" + HELD, r.headers["location"])
    assert m, r.headers["location"]
    merge_page = client.get(r.headers["location"].split("&flash=")[0])
    assert merge_page.status_code == 200 and "Confirm merge" in merge_page.text
    assert HELD in client.get("/review/duplicates").text
