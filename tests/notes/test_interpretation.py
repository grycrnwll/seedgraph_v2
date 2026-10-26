"""The current interpretation (stage 3 design §4.3): the one reader for the paper page, search
and MCP. Trust labels, hidden claims, audiences, researcher notes. Invented text only."""

from __future__ import annotations

import pytest

from seedgraph.notes import interpretation, overlay

from .test_overlay import RESULT_QUOTE, claim_id, raw_note, save_raw


@pytest.fixture
def noted(lib, paper):
    return paper, save_raw(lib, paper)


def by_field(interp):
    return {c.field_key: c for c in interp.claims}


def test_no_overlay_reads_the_current_note_with_agent_trust(lib, noted):
    paper, saved = noted
    interp = interpretation.current(lib, paper.paper_id, "agent")
    assert interp.note_id == saved.note_id and interp.overlay_id is None and interp.version_no is None
    assert interp.researcher_notes is None and not interp.newer_extraction
    claims = by_field(interp)
    assert claims["method_or_model[0]"].trust == "paper_states"
    assert claims["method_or_model[0]"].trust_label == "paper states"
    assert claims["method_or_model[0]"].span_id is not None
    assert claims["assumptions[1]"].trust == "agent_inferred"  # inferred, no quote
    assert claims["setting"].trust == "agent_inferred"          # a not_found scalar
    assert not any(c.edited_by_researcher for c in interp.claims)
    # stable order: the note's field order
    assert [c.field_key for c in interp.claims][:4] == [
        "research_question", "main_contribution", "setting", "estimand_or_target_object"]
    assert interp.note_text == lib.conn.execute(
        "SELECT note_text FROM structured_notes WHERE note_id = ?", (saved.note_id,)).fetchone()[0]


def test_a_paper_without_a_note_has_no_interpretation(lib, paper):
    assert interpretation.current(lib, paper.paper_id, "researcher") is None


def test_trust_labels_for_edited_added_and_agent_claims(lib, noted):
    paper, saved = noted
    m = claim_id(lib, saved.note_id, "method_or_model[0]")
    v = overlay.save_edit(lib, paper.paper_id, {"op": "edit", "target": {"claim_id": m},
                                                "set": {"claim_text": "Re-worded by the researcher."}},
                          expected_version=0)
    v = overlay.save_edit(lib, paper.paper_id, {
        "op": "add", "kind": "assertion", "claim_type": "limitation", "normalized_label": "small sample",
        "claim_text": "Only invented widgets."}, expected_version=v.version_no)
    start = paper.text.index(RESULT_QUOTE)
    v = overlay.save_edit(lib, paper.paper_id, {
        "op": "add", "kind": "quote", "claim_type": "result", "normalized_label": "micro sign",
        "claim_text": "A micro sign appears.",
        "quote": {"markdown_id": paper.markdown_id, "start_char": start, "end_char": start + len(RESULT_QUOTE)}},
        expected_version=v.version_no)
    interp = interpretation.current(lib, paper.paper_id, "agent")
    assert interp.overlay_id == v.overlay_id and interp.version_no == 3
    claims = by_field(interp)
    edited = claims["method_or_model[0]"]
    assert edited.trust == "researcher" and edited.edited_by_researcher
    assert edited.trust_note == "edited from the agent's claim"
    assert edited.claim_text == "Re-worded by the researcher."
    assert edited.original.claim_text == "A two-step widget estimator."  # one click away
    assert edited.span_id is not None  # the quote still shows as the paper's words
    assertion = claims["added[0]"]
    assert assertion.trust == "researcher" and assertion.kind == "assertion" and assertion.span_id is None
    assert assertion.trust_note == "researcher assertion — not the paper's words"
    quoted = claims["added[1]"]
    assert quoted.kind == "quote" and quoted.span_id is not None and quoted.original is None
    assert claims["assumptions[0]"].trust == "paper_states"
    assert claims["assumptions[1]"].trust == "agent_inferred"
    assert [c.field_key for c in interp.claims][-2:] == ["added[0]", "added[1]"]


def test_hidden_claims_are_excluded_except_for_the_researchers_toggle(lib, noted):
    paper, saved = noted
    a = claim_id(lib, saved.note_id, "assumptions[0]")
    overlay.save_edit(lib, paper.paper_id, {"op": "hide", "target": {"claim_id": a}}, expected_version=0)
    for audience in ("agent", "researcher"):
        assert a not in {c.claim_id for c in interpretation.current(lib, paper.paper_id, audience).claims}
    shown = interpretation.current(lib, paper.paper_id, "researcher", show_hidden=True)
    [hidden] = [c for c in shown.claims if c.claim_id == a]
    assert hidden.hidden
    with pytest.raises(ValueError):
        interpretation.current(lib, paper.paper_id, "agent", show_hidden=True)


def test_researcher_notes_reach_every_audience(lib, noted):
    paper, saved = noted
    overlay.save_notes(lib, paper.paper_id, "Invented reminder for later.", expected_version=0)
    for audience in ("agent", "researcher"):
        interp = interpretation.current(lib, paper.paper_id, audience)
        assert interp.researcher_notes == "Invented reminder for later."
        assert "Researcher notes: Invented reminder for later." in interp.note_text


def test_the_agent_audience_keeps_stage1_hiding_rules(lib, noted):
    paper, saved = noted
    with lib.write() as conn:
        conn.execute(
            "INSERT INTO attention_items (item_id, kind, paper_id, subject_json, detail, status, hides_notes, "
            "created_at) VALUES ('att_invented', 'wrong_pdf', ?, '{}', 'invented', 'open', 1, '2026-01-01')",
            (paper.paper_id,))
    assert interpretation.current(lib, paper.paper_id, "agent") is None
    assert interpretation.current(lib, paper.paper_id, "researcher") is not None


def test_a_newer_extraction_is_flagged_but_changes_nothing(lib, noted):
    paper, saved = noted
    m = claim_id(lib, saved.note_id, "method_or_model[0]")
    overlay.save_edit(lib, paper.paper_id, {"op": "edit", "target": {"claim_id": m},
                                            "set": {"claim_text": "Researcher text."}}, expected_version=0)
    before = interpretation.current(lib, paper.paper_id, "agent")
    save_raw(lib, paper, raw_note("A newer agent wording."))
    after = interpretation.current(lib, paper.paper_id, "agent")
    assert after.newer_extraction and not before.newer_extraction
    assert after.claims == before.claims and after.note_id == before.note_id
