"""Review -> duplicates: "Same paper" opens the merge preview and Confirm merges the two papers;
"Different papers" records the verdict. Fixture: ``pages_support.build`` (lp_cited2 may be the
same paper as lp_cited1). Invented text only."""

from __future__ import annotations

import re

import pytest

from seedgraph.library import attention, merge
from seedgraph.library.store import open_library_at, paper_exists

from .pages_support import build, no_inline_script_or_style, post_forms_lead_with_csrf, token


@pytest.fixture
def fx(home):
    return build(home)


def with_lib(home, fn):
    lib = open_library_at(home / "library")
    try:
        return fn(lib)
    finally:
        lib.close()


def preview_url(fx, **extra):
    q = "&".join(f"{k}={v}" for k, v in {"item": fx.identity_item, "with": fx.cited, **extra}.items())
    return f"/review/merge?{q}"


def hidden(html, name):
    m = re.search(rf'name="{name}" value="([^"]*)"', html)
    return m.group(1) if m else None


def confirm_data(c, html):
    return {"csrf_token": token(c), "item": hidden(html, "item"), "with": hidden(html, "with"),
            "keep": hidden(html, "keep"),
            "other": hidden(html, "other"), "primary": hidden(html, "primary") or "",
            "fingerprint": hidden(html, "fingerprint")}


def test_the_identity_row_offers_the_merge_preview_not_a_bare_same(fx, make_app, client_for):
    html = client_for(make_app()).get("/review/duplicates").text
    assert f'href="/review/merge?item={fx.identity_item}&amp;with={fx.cited}"' in html
    assert 'value="same"' not in html and 'value="different"' in html
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)


def test_a_migrated_identity_check_offers_a_merge_with_its_evidence_candidates(fx, make_app, client_for):
    def raise_migrated(lib):
        return attention.raise_item(
            lib, "identity_check", {"reason": "no_match", "paper_id": fx.cited_no_doi, "key_type": None,
                                    "key_value": None},
            paper_id=fx.cited_no_doi, origin="migration",
            detail=f'{{"candidates": [{{"paper_id": "{fx.cited}", "title": "x", "similarity": 0.9}}]}}').item_id
    item_id = with_lib(fx.home, raise_migrated)
    c = client_for(make_app())
    assert f'href="/review/merge?item={item_id}&amp;with={fx.cited}"' in c.get("/review/duplicates").text
    assert c.get(f"/review/merge?item={item_id}&with={fx.cited}").status_code == 200


def test_the_preview_shows_the_survivor_and_what_moves(fx, make_app, client_for):
    r = client_for(make_app()).get(preview_url(fx))
    assert r.status_code == 200
    html = r.text
    # lp_cited1 has a DOI, so it survives by default; lp_cited2 is merged into it
    assert hidden(html, "keep") == fx.cited and hidden(html, "other") == fx.cited_no_doi
    assert "A cited invented treatise" in html and "Another cited invented treatise" in html
    assert "Confirm merge" in html and hidden(html, "fingerprint")
    assert f"keep={fx.cited_no_doi}" in html  # the swap link
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)


def test_the_preview_is_read_only(fx, make_app, client_for):
    client_for(make_app()).get(preview_url(fx))
    assert with_lib(fx.home, lambda lib: paper_exists(lib.conn, fx.cited_no_doi))


def test_confirm_needs_the_csrf_token(fx, make_app, client_for):
    c = client_for(make_app())
    data = confirm_data(c, c.get(preview_url(fx)).text)
    data.pop("csrf_token")
    assert c.post("/review/merge", data=data).status_code == 403
    assert with_lib(fx.home, lambda lib: paper_exists(lib.conn, fx.cited_no_doi))


def test_confirm_merges_resolves_the_item_and_records_the_decision(fx, make_app, client_for):
    c = client_for(make_app())
    r = c.post("/review/merge", data=confirm_data(c, c.get(preview_url(fx)).text))
    assert r.status_code == 303 and r.headers["location"].startswith("/review/duplicates")

    def check(lib):
        assert not paper_exists(lib.conn, fx.cited_no_doi) and paper_exists(lib.conn, fx.cited)
        item = attention.get_item(lib, fx.identity_item)
        assert (item.status, item.resolution) == ("resolved", "same")
        row = lib.conn.execute("SELECT verdict, decided_by FROM duplicate_decisions WHERE paper_a = ? AND "
                               "paper_b = ?", (fx.cited, fx.cited_no_doi)).fetchone()
        assert tuple(row) == ("same", "researcher")
        assert merge.resolve_merged(lib.conn, fx.cited_no_doi) == fx.cited
    with_lib(fx.home, check)
    page = c.get("/review/duplicates").text
    assert fx.identity_item not in page.split("Decisions on record")[0]


def test_the_researcher_can_keep_the_other_id(fx, make_app, client_for):
    c = client_for(make_app())
    html = c.get(preview_url(fx, keep=fx.cited_no_doi)).text
    assert hidden(html, "keep") == fx.cited_no_doi
    assert c.post("/review/merge", data=confirm_data(c, html)).status_code == 303
    assert with_lib(fx.home, lambda lib: (paper_exists(lib.conn, fx.cited_no_doi),
                                          paper_exists(lib.conn, fx.cited))) == (True, False)


def test_a_stale_preview_is_refused_with_a_link_to_review_again(fx, make_app, client_for):
    c = client_for(make_app())
    data = confirm_data(c, c.get(preview_url(fx)).text)

    def change(lib):
        with lib.write() as conn:
            conn.execute("UPDATE papers SET year = 2032 WHERE paper_id = ?", (fx.cited_no_doi,))
            conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value) VALUES (?, 'arxiv', '2031.00009')",
                         (fx.cited_no_doi,))
    with_lib(fx.home, change)
    r = c.post("/review/merge", data=data)
    assert r.status_code == 409 and "/review/merge?" in r.text
    assert with_lib(fx.home, lambda lib: paper_exists(lib.conn, fx.cited_no_doi))


def test_a_blocked_merge_shows_the_reason_and_no_confirm(fx, make_app, client_for):
    def conflict(lib):
        with lib.write() as conn:
            conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value) VALUES (?, 'doi', '10.9999/other.1')",
                         (fx.cited_no_doi,))
    with_lib(fx.home, conflict)
    c = client_for(make_app())
    html = c.get(preview_url(fx)).text
    assert "two ids of one type cannot be merged" in html and "Confirm merge" not in html


def test_a_preview_for_a_paper_the_item_does_not_name_is_404(fx, make_app, client_for):
    r = client_for(make_app()).get(f"/review/merge?item={fx.identity_item}&with={fx.noted}")
    assert r.status_code == 404


def test_different_papers_records_the_verdict(fx, make_app, client_for):
    c = client_for(make_app())
    r = c.post(f"/review/attention/{fx.identity_item}/resolve",
               data={"csrf_token": token(c), "resolution": "different"})
    assert r.status_code == 303

    def check(lib):
        assert attention.get_item(lib, fx.identity_item).resolution == "different"
        assert merge.decided_different(lib.conn, fx.cited, fx.cited_no_doi)
    with_lib(fx.home, check)


def test_a_bare_same_resolution_is_refused(fx, make_app, client_for):
    c = client_for(make_app())
    r = c.post(f"/review/attention/{fx.identity_item}/resolve", data={"csrf_token": token(c), "resolution": "same"})
    assert r.status_code == 400
