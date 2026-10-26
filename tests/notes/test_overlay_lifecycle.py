"""Overlays and the rest of the library: deleting a paper (stage 1 §5, stage 3 §4.2) and a
replay-style base change (the hook U2's accept uses). Invented text only."""

from __future__ import annotations

import pytest

from seedgraph.library import lifecycle
from seedgraph.notes import overlay

from .test_overlay import claim_id, raw_note, rows, save_raw


@pytest.fixture
def edited(lib, paper):
    saved = save_raw(lib, paper)
    m = claim_id(lib, saved.note_id, "method_or_model[0]")
    overlay.save_edit(lib, paper.paper_id, {"op": "edit", "target": {"claim_id": m},
                                            "set": {"claim_text": "A quuxifying tally."}}, expected_version=0)
    overlay.save_edit(lib, paper.paper_id, {
        "op": "add", "kind": "assertion", "claim_type": "limitation", "normalized_label": "frumious",
        "claim_text": "An invented frumious limitation."}, expected_version=1)
    return paper, saved


def test_plan_delete_reports_researcher_edits(lib, paper, edited):
    plan = lifecycle.plan_delete(lib, paper.paper_id)
    assert plan.has_researcher_edits
    assert "has researcher edits, which will be deleted" in plan.warnings


def test_plan_delete_without_edits_is_unchanged(lib, paper):
    save_raw(lib, paper)
    plan = lifecycle.plan_delete(lib, paper.paper_id)
    assert not plan.has_researcher_edits
    assert not any("researcher" in w for w in plan.warnings)


def test_a_new_edit_makes_a_delete_plan_stale(lib, paper, edited):
    plan = lifecycle.plan_delete(lib, paper.paper_id)
    overlay.save_notes(lib, paper.paper_id, "invented", expected_version=2)
    with pytest.raises(lifecycle.StalePlanError):
        lifecycle.delete_paper(lib, plan)


def test_delete_paper_removes_overlays_derived_rows_and_their_fts(lib, paper, edited):
    lifecycle.delete_paper(lib, lifecycle.plan_delete(lib, paper.paper_id))
    for table in ("note_overlays", "overlay_versions", "interpretation_claims", "interpretation_notes",
                  "concept_refresh", "structured_notes"):
        assert lib.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
    assert rows(lib.conn, "SELECT claim_id FROM claim_fts WHERE claim_fts MATCH 'frumious'") == []
    assert lib.conn.execute("SELECT COUNT(*) FROM claim_fts").fetchone()[0] == 0
    assert lib.conn.execute("SELECT COUNT(*) FROM note_fts").fetchone()[0] == 0


def test_moving_the_base_restores_the_old_bases_agent_index(lib, paper, edited):
    """What U2's accept does: write a version on a new base in the caller's transaction."""
    _, saved = edited
    renote = save_raw(lib, paper, raw_note("A newer agent wording."))
    old_m = claim_id(lib, saved.note_id, "method_or_model[0]")
    with lib.write() as conn:
        v = overlay.write_version(conn, paper.paper_id, ops=[], researcher_notes=None,
                                  action="accept_extraction", expected_version=2,
                                  base_note_id=renote.note_id)
    assert v.base_note_id == renote.note_id and v.base_extraction_run_id == renote.extraction_run_id
    assert lib.conn.execute("SELECT note_id FROM note_current").fetchone()[0] == renote.note_id
    # the old base is indexed as the agent wrote it again; the added claim is gone
    assert rows(lib.conn, "SELECT claim_text FROM claim_fts WHERE claim_id = ?", old_m) == [
        ("A two-step widget estimator.",)]
    assert rows(lib.conn, "SELECT claim_id FROM claim_fts WHERE claim_fts MATCH 'frumious'") == []
    # reverting to the version before the accept restores the old base and its edits
    overlay.revert(lib, paper.paper_id, 2, expected_version=3)
    assert lib.conn.execute("SELECT note_id FROM note_current").fetchone()[0] == saved.note_id
    assert rows(lib.conn, "SELECT claim_id FROM claim_fts WHERE claim_fts MATCH 'quuxifying'") == [(old_m,)]
    assert len(rows(lib.conn, "SELECT claim_id FROM claim_fts WHERE claim_fts MATCH 'frumious'")) == 1
