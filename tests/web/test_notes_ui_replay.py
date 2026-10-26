"""The replay view (docs/V3_UI_DESIGN.md §4.5): "Newer extraction available" banner, side by
side plan, nothing changes until accept, accept applies the plan, stale plans refused.
Invented text only."""

from __future__ import annotations

import re
import sqlite3

import pytest

from seedgraph.notes import interpretation, overlay

from . import paper_support as ps


def newer_raw() -> dict:
    raw = ps.raw_note(method_text="A two-step estimator that double counts widgets.")
    raw["main_results"].append({"claim_text": "Widgets grow by an invented amount.", "normalized_label": "growth",
                                "status": "inferred", "result_type": "main",
                                "inferred_explanation": "invented reason"})
    return raw


@pytest.fixture
def fx(home, tmp_path):
    fx = ps.build(home, tmp_path)
    method, counting = fx.claim("method_or_model[0]"), fx.claim("assumptions[1]")
    with fx.open() as lib:
        overlay.save_edit(lib, fx.held, {"op": "edit", "target": {"claim_id": method},
                                         "set": {"claim_text": "A quuxifying widget tally."}}, expected_version=0)
        overlay.save_edit(lib, fx.held, {"op": "hide", "target": {"claim_id": counting}}, expected_version=1)
        overlay.save_edit(lib, fx.held, {"op": "add", "kind": "assertion", "claim_type": "limitation",
                                         "normalized_label": "frumious caveat",
                                         "claim_text": "An invented researcher caveat."}, expected_version=2)
        ps.save_note(lib, fx.held, fx.markdown_id, newer_raw())
    return fx


@pytest.fixture
def client(fx, make_app, client_for):
    return client_for(make_app())


def dump(fx):
    conn = sqlite3.connect(fx.lib_dir / "library.db")
    try:
        return list(conn.iterdump())
    finally:
        conn.close()


def test_the_page_offers_the_comparison_and_still_shows_the_edited_note(client, fx):
    html = client.get(f"/papers/{fx.held}").text
    assert f'Newer extraction available — <a href="/papers/{fx.held}/replay">compare</a>' in html
    assert 'data-badge="newer-extraction"' in html
    assert "A quuxifying widget tally." in html
    assert "double counts widgets" not in html


def test_the_replay_view_lists_the_plan_side_by_side_and_writes_nothing(client, fx):
    before = dump(fx)
    r = client.get(f"/papers/{fx.held}/replay")
    assert r.status_code == 200
    html = r.text
    assert re.search(r'data-op-status="kept".*A quuxifying widget tally\..*A two-step widget estimator\..*'
                     r"A two-step estimator that double counts widgets\.", html, re.S)
    assert 'data-op-status="still_hidden"' in html
    assert 'data-op-status="added_kept"' in html and "An invented researcher caveat." in html
    assert re.search(r'data-new-claim="[^"]+">.*Widgets grow by an invented amount\.', html, re.S)
    assert "1 edits kept · 1 still hidden · 1 new agent claims" in html
    m = re.search(r'<form method="post" action="/papers/[^"]+/replay/accept"[^>]*>\s*(<input[^>]*>)', html)
    assert m and 'name="csrf_token"' in m.group(1)
    assert dump(fx) == before


def _form(html: str) -> dict:
    data = dict(re.findall(r'<input type="hidden" name="(head_version_no|new_note_id)" value="([^"]+)"', html))
    data["accept_new"] = re.findall(r'name="accept_new" value="([^"]+)" checked', html)
    return data


def test_accept_all_applies_the_plan(client, fx):
    data = _form(client.get(f"/papers/{fx.held}/replay").text)
    r = ps.post(client, f"/papers/{fx.held}/replay/accept", {**data, "mode": "all"}, htmx=False)
    assert r.status_code == 303 and r.headers["location"].startswith(f"/papers/{fx.held}?")
    assert "newer extraction accepted" in client.get(r.headers["location"]).text
    with fx.open() as lib:
        i = interpretation.current(lib, fx.held, "researcher")
        assert overlay.head(lib.conn, fx.held).action == "accept_extraction"
    texts = {c.claim_text for c in i.claims}
    assert "A quuxifying widget tally." in texts               # the edit survives
    assert "An invented researcher caveat." in texts           # the added claim survives
    assert "Widgets are counted without error." not in texts   # still hidden
    assert "Widgets grow by an invented amount." in texts      # the new agent claim
    assert not i.newer_extraction


def test_dismiss_and_discard_decisions_are_applied(client, fx):
    html = client.get(f"/papers/{fx.held}/replay").text
    data = _form(html)
    edit_key = re.search(r'data-op-key="(edit:[^"]+)" data-op-status="kept"', html).group(1)
    r = ps.post(client, f"/papers/{fx.held}/replay/accept",
                {**data, "mode": "dismiss_all", f"decision:{edit_key}": "discard"}, htmx=False)
    assert r.status_code == 303
    with fx.open() as lib:
        texts = {c.claim_text for c in interpretation.current(lib, fx.held, "researcher").claims}
    assert "A quuxifying widget tally." not in texts
    assert "A two-step estimator that double counts widgets." in texts  # the new wording
    assert "Widgets grow by an invented amount." not in texts          # dismissed


def test_a_stale_comparison_is_refused(client, fx):
    data = _form(client.get(f"/papers/{fx.held}/replay").text)
    with fx.open() as lib:  # another tab saves meanwhile
        overlay.save_notes(lib, fx.held, "Another tab.", expected_version=3)
    before = dump(fx)
    r = ps.post(client, f"/papers/{fx.held}/replay/accept", {**data, "mode": "all"}, htmx=False)
    assert r.headers["location"].startswith(f"/papers/{fx.held}/replay?")
    assert "this note changed in another tab; reload" in client.get(r.headers["location"]).text
    assert dump(fx) == before


def test_a_paper_without_a_newer_extraction_has_no_replay(home, tmp_path, make_app, client_for):
    fx = ps.build(home, tmp_path)
    c = client_for(make_app())
    r = c.get(f"/papers/{fx.held}/replay")
    assert r.status_code == 404 and "no newer extraction" in r.text
