"""Review screens (design §2.6): the attention list grouped by kind with fixed-resolution
actions (``accept_text`` included), duplicate / identity checks, and the migrated citation
decisions (read only)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from seedgraph.library import attention
from seedgraph.library.store import open_library_at
from seedgraph.web.views import review as review_view

from .pages_support import GRAPH, build, no_inline_script_or_style, post_forms_lead_with_csrf, token


@pytest.fixture
def fx(home):
    return build(home)


@pytest.fixture
def spy(monkeypatch):
    calls = []
    real = attention.resolve_item

    def wrapped(lib, item_id, resolution, **kw):
        calls.append((item_id, resolution))
        return real(lib, item_id, resolution, **kw)

    monkeypatch.setattr(attention, "resolve_item", wrapped)
    return calls


def item(home, item_id):
    lib = open_library_at(home / "library")
    try:
        return attention.get_item(lib, item_id)
    finally:
        lib.close()


def test_the_attention_view_groups_items_by_queue(fx):
    lib = open_library_at(fx.home / "library")
    try:
        view = review_view.attention_page(lib)
    finally:
        lib.close()
    groups = {g.queue.key: [i.item_id for i in g.items] for g in view.groups}
    assert groups["text"] == [fx.garbled_item]
    assert groups["pdf"] == [fx.title_miss_item]
    assert groups["identity"] == [fx.identity_item]
    assert groups["renote"] == [fx.renote_item]
    assert groups["other"] == [fx.other_item]


def test_the_attention_page_shows_actions_and_the_hidden_notes_mark(fx, make_app, client_for):
    html = client_for(make_app()).get("/review/attention").text
    assert "Note anyway" in html and 'value="accept_text"' in html
    assert "It is this paper" in html and 'value="confirmed"' in html
    assert 'value="same"' in html and 'value="different"' in html
    assert 'value="dismissed"' in html and 'value="resolved_by_researcher"' in html
    assert "notes hidden from agents until resolved" in html
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)


def test_the_kind_and_graph_filters_narrow_the_list(fx, make_app, client_for):
    c = client_for(make_app())
    only = c.get("/review/attention?kind=text_garbled").text
    assert fx.garbled_item in only and fx.title_miss_item not in only
    on_graph = c.get(f"/review/attention?graph={GRAPH}").text
    assert fx.garbled_item in on_graph and fx.identity_item not in on_graph and fx.other_item not in on_graph


def test_accept_text_resolves_the_item_through_resolve_item(fx, make_app, client_for, spy):
    c = client_for(make_app())
    url = f"/review/attention/{fx.garbled_item}/resolve"
    assert c.post(url, data={"resolution": "accept_text"}).status_code == 403
    r = c.post(url, data={"csrf_token": token(c), "resolution": "accept_text"})
    assert r.status_code == 303
    assert spy == [(fx.garbled_item, "accept_text")]
    assert item(fx.home, fx.garbled_item).resolution == "accept_text"


@pytest.mark.parametrize("attr,resolution", [("title_miss_item", "confirmed"), ("identity_item", "different"),
                                             ("identity_item", "same"), ("renote_item", "dismissed"),
                                             ("other_item", "resolved_by_researcher")])
def test_each_queue_resolves_with_its_fixed_resolution(fx, make_app, client_for, spy, attr, resolution):
    item_id = getattr(fx, attr)
    c = client_for(make_app())
    c.post(f"/review/attention/{item_id}/resolve", data={"csrf_token": token(c), "resolution": resolution})
    assert spy == [(item_id, resolution)]
    assert item(fx.home, item_id).status == "resolved"


def test_a_resolution_outside_the_kinds_actions_is_refused(fx, make_app, client_for, spy):
    c = client_for(make_app())
    r = c.post(f"/review/attention/{fx.title_miss_item}/resolve",
               data={"csrf_token": token(c), "resolution": "accept_text"})
    assert r.status_code == 400 and spy == []
    assert item(fx.home, fx.title_miss_item).status == "open"


def test_an_htmx_resolve_swaps_the_row(fx, make_app, client_for, spy):
    c = client_for(make_app())
    r = c.post(f"/review/attention/{fx.renote_item}/resolve", data={"resolution": "dismissed"},
               headers={"X-CSRF-Token": token(c), "HX-Request": "true"})
    assert r.status_code == 200
    assert r.text.lstrip().startswith("<tr") and "resolved" in r.text and "<html" not in r.text


def test_resolving_an_unknown_item_is_404(fx, make_app, client_for):
    c = client_for(make_app())
    r = c.post("/review/attention/att_nope/resolve", data={"csrf_token": token(c), "resolution": "confirmed"})
    assert r.status_code == 404


def test_accept_text_from_the_library_returns_to_the_library(fx, make_app, client_for):
    c = client_for(make_app())
    r = c.post(f"/review/attention/{fx.garbled_item}/resolve",
               data={"csrf_token": token(c), "resolution": "accept_text", "next": "/library?view=held-back"})
    assert r.headers["location"].startswith("/library?view=held-back")


def test_the_duplicates_page_shows_identity_checks_and_migrated_decisions(fx, make_app, client_for):
    html = client_for(make_app()).get("/review/duplicates").text
    assert fx.identity_item in html and f'href="/papers/{fx.cited}"' in html
    assert "lp_aaa" in html and "different" in html and "invented reason" in html
    assert post_forms_lead_with_csrf(html)


def test_the_citation_decisions_are_read_only(fx, make_app, client_for):
    html = client_for(make_app()).get("/review/citations").text
    assert "A made-up treatise" in html and "invented wrong match" in html
    assert "<form" not in html.split('id="sg-content"', 1)[1]


def _kinds_raised_in_src() -> set[str]:
    src = Path(__file__).resolve().parents[2] / "src" / "seedgraph"
    kinds: set[str] = set()
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        kinds |= set(re.findall(r'\battn\(\s*"([a-z_]+)"', text))
        kinds |= set(re.findall(r'\braise_item\(\s*\w+,\s*"([a-z_]+)"', text))
        kinds |= set(re.findall(r'^\s*[A-Z_]*(?:KIND|NEEDED|CHECK|TITLE)\s*=\s*"([a-z_]+)"', text, re.M))
        for block in re.findall(r"_GATE_KINDS\s*=\s*\{([^}]*)\}", text):
            kinds |= set(re.findall(r':\s*"([a-z_]+)"', block))
    return kinds


def test_every_attention_kind_raised_in_src_gets_a_resolvable_row(home, make_app, client_for):
    kinds = _kinds_raised_in_src()
    assert {"text_garbled", "pdf_title_miss", "identity_check", "metadata_needed", "renote_offered"} <= kinds
    lib = open_library_at(home / "library")
    try:
        ids = {k: attention.raise_item(lib, k, {"probe": k}, detail=f"invented {k}").item_id for k in kinds}
    finally:
        lib.close()
    html = client_for(make_app()).get("/review/attention").text
    for kind, item_id in ids.items():
        assert f'action="/review/attention/{item_id}/resolve"' in html, kind
