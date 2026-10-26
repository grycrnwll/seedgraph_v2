"""The edits-survive scripts end to end on a synthetic fixture library (invented text):
setup over HTTP with CSRF, the pre-re-note self-test, a simulated re-note, the step 4
checks, accept all and the step 6 checks."""

from __future__ import annotations

import io
import json

import pytest

from web import paper_support as ps

from . import edits_survive_check as check
from . import edits_survive_setup as setup
from . import harness as h


@pytest.fixture
def fixture_home(tmp_path):
    home = tmp_path / "home"
    fx = ps.build(home, tmp_path)
    return home, fx


def _setup(home, fx):
    client = h.PageClient.inprocess(home)
    try:
        before = setup.run_setup(home, fx.held, client, run_name="r1", out=io.StringIO())
    finally:
        client.close()
    return before


def _check(home, **kw):
    client = h.PageClient.inprocess(home)
    try:
        return check.run_check(home, h.acceptance_dir(home) / "r1", client, out=io.StringIO(), **kw)
    finally:
        client.close()


def _renote(fx):
    with fx.open() as lib:
        return ps.save_note(lib, fx.held, fx.markdown_id,
                            ps.raw_note(method_text="A two-step widget estimator, re-noted."))


def test_setup_writes_before_json_and_a_queued_replace_batch(fixture_home):
    home, fx = fixture_home
    before = _setup(home, fx)
    run = h.acceptance_dir(home) / "r1"
    saved = json.loads((run / "before.json").read_text(encoding="utf-8"))
    assert saved["paper_id"] == fx.held
    assert set(saved["ops"]) == {"edit", "hide", "add_assertion", "add_quote", "researcher_notes"}
    assert saved["setup_checks"]["edit_found_by_search"]
    assert saved["setup_checks"]["added_found_by_search"]
    assert saved["setup_checks"]["hidden_not_found_by_search"]
    assert saved["setup_checks"]["paper_show_edit_trust"] == "researcher"
    assert saved["setup_checks"]["paper_show_hidden_absent"]
    assert saved["invariant_unchanged_by_batch"]
    assert saved["expected_to_change"]["newer_extraction"] is False
    assert [i["state"] for i in saved["batch"]["items"]] == ["queued"]
    assert json.loads(saved["batch"]["options_json"]).get("replace") is True
    assert saved["batch"]["detail_shows_command"]
    inputs = h.load_inputs(home)["edits_survive"]
    assert inputs == {"paper_id": fx.held, "run": "r1", "batch_id": before["batch"]["batch_id"]}


def test_pre_mode_passes_before_any_renote(fixture_home):
    home, fx = fixture_home
    _setup(home, fx)
    checks, result = _check(home, pre=True)
    assert checks.ok, checks.rows


def test_check_before_renote_fails_on_freshness(fixture_home):
    home, fx = fixture_home
    _setup(home, fx)
    checks, _ = _check(home)
    failed = {r["check"] for r in checks.rows if not r["ok"]}
    assert "note_latest is new" in failed
    assert "interpretation byte-identical (agent + researcher)" not in failed


def test_renote_then_check_then_accept(fixture_home):
    home, fx = fixture_home
    _setup(home, fx)
    _renote(fx)
    checks, result = _check(home)
    assert checks.ok, [r for r in checks.rows if not r["ok"]]
    assert result["tiers"]["edit"] is not None
    checks, result = _check(home, accept=True)
    assert checks.ok, [r for r in checks.rows if not r["ok"]]
    names = {r["check"] for r in checks.rows}
    assert "after accept: edit survives (text, label, type)" in names
    assert "after accept: note_current points to the new base" in names


def test_another_papers_new_note_is_reported_not_gated(fixture_home):
    home, fx = fixture_home
    before = _setup(home, fx)
    with fx.open() as lib:  # another paper gains a claim that matches the hidden-label query
        md = lib.conn.execute("SELECT markdown_id FROM paper_files WHERE paper_id = ?", (fx.plain,)).fetchone()[0]
        ps.save_note(lib, fx.plain, md)
        with lib.write() as conn:  # let agents see it (the fixture holds it back: no page markers)
            conn.execute("UPDATE attention_items SET status = 'resolved' WHERE paper_id = ?", (fx.plain,))
    _renote(fx)
    checks, result = _check(home)
    assert checks.ok, [r for r in checks.rows if not r["ok"]]
    q2 = before["queries"][1]["query"]
    info = [i for i in result["info"] if i["query"] == q2]
    assert info and fx.plain in info[0]["papers"], (q2, result["info"], before["invariant"]["search"][1])


def test_an_edit_after_setup_breaks_the_invariant(fixture_home):
    home, fx = fixture_home
    before = _setup(home, fx)
    client = h.PageClient.inprocess(home)
    try:
        version = h.pane_version(client.get(f"/papers/{fx.held}").text)
        client.post(f"/papers/{fx.held}/notes/unhide",
                    {"claim_id": before["ops"]["hide"]["claim_id"], "expected_version": version})
    finally:
        client.close()
    _renote(fx)
    checks, _ = _check(home)
    failed = {r["check"] for r in checks.rows if not r["ok"]}
    assert "interpretation byte-identical (agent + researcher)" in failed


def test_inputs_template_holds_no_real_values():
    data = json.loads(h.TEMPLATE.read_text(encoding="utf-8"))
    assert data["edits_survive"]["paper_id"] == ""
    assert data["walkthrough"]["pdf_sha256"] == ""
    assert data["walkthrough"]["pdf_path"] == ""
    assert "lp_" not in h.TEMPLATE.read_text(encoding="utf-8")


def test_canonical_is_order_independent_for_keys():
    assert h.canonical({"b": 1, "a": [2, 3]}) == h.canonical({"a": [2, 3], "b": 1})
    assert h.sha256({"x": "é"}) == h.sha256({"x": "é"})
