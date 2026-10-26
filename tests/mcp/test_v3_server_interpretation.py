"""MCP paper_show / search return the current interpretation (stage 3 design §4.3, §11 U6):
trust labels, the edited_by_researcher marker with the agent's original one level down,
added claims, hidden claims left out, researcher notes. Invented text only."""

from __future__ import annotations

from v3_mcp_support import call, ok, home, seeded, server  # noqa: F401 (fixtures)

from seedgraph.library import store as libstore
from seedgraph.notes import overlay


def _edit(home, seeded):
    """Edit the method claim, hide the assumption, add an assertion and researcher notes."""
    lib = libstore.open_library_at(home / "library")
    try:
        def cid(label):
            return lib.conn.execute("SELECT claim_id FROM extracted_claims WHERE structured_note_id = ? "
                                    "AND normalized_label = ?", (seeded.note_id, label)).fetchone()[0]
        method, assumption = cid("two-step estimator"), cid("no anticipation")
        v = overlay.save_edit(lib, seeded.p1.paper_id, {"op": "edit", "target": {"claim_id": method},
                                                        "set": {"claim_text": "A wallaby-weighted estimator."}},
                              expected_version=0)
        v = overlay.save_edit(lib, seeded.p1.paper_id, {"op": "hide", "target": {"claim_id": assumption}},
                              expected_version=v.version_no)
        v = overlay.save_edit(lib, seeded.p1.paper_id, {
            "op": "add", "kind": "assertion", "claim_type": "limitation", "normalized_label": "narrow sample",
            "claim_text": "Only bandicoot workshops are covered."}, expected_version=v.version_no)
        v = overlay.save_notes(lib, seeded.p1.paper_id, "Compare with the koala survey.",
                               expected_version=v.version_no)
        return method, assumption, v
    finally:
        lib.close()


def test_paper_show_returns_the_interpretation_with_trust_labels(server, seeded, home):
    method, assumption, v = _edit(home, seeded)
    note = ok(call(server, "paper_show", {"paper_id": seeded.p1.paper_id}))["current_note"]
    assert note["note_id"] == seeded.note_id
    assert note["edited_by_researcher"] is True
    assert (note["overlay_id"], note["overlay_version"]) == (v.overlay_id, v.version_no)
    assert note["researcher_notes"] == "Compare with the koala survey."
    claims = {c["claim_id"]: c for c in note["claims"]}
    assert assumption not in claims  # hidden claims are left out
    edited = claims[method]
    assert edited["claim_text"] == "A wallaby-weighted estimator."
    assert edited["edited_by_researcher"] is True and edited["trust"] == "researcher"
    assert edited["trust_note"] == "edited by researcher"
    assert edited["original"] == {"claim_type": "method", "label": "two-step estimator",
                                  "claim_text": "A two-step estimator."}
    assert edited["anchors"]  # the quote is still the paper's words
    [added] = [c for c in note["claims"] if c["claim_id"].startswith("rc_")]
    assert added["claim_kind"] == "assertion" and added["trust_label"] == "researcher"
    assert added["trust_note"] == "researcher assertion — not the paper's words"
    assert added["anchors"] == [] and "original" not in added
    agent = next(c for c in note["claims"] if c["label"] == "gadget subsidy effect")
    assert agent["trust"] == "paper_states" and agent["edited_by_researcher"] is False
    assert agent["trust_note"] is None and "original" not in agent


def test_paper_show_without_edits_labels_agent_trust(server, seeded):
    note = ok(call(server, "paper_show", {"paper_id": seeded.p1.paper_id}))["current_note"]
    assert note["edited_by_researcher"] is False and note["researcher_notes"] is None
    assert note["overlay_id"] is None
    assert {c["trust"] for c in note["claims"] if c["status"] == "found"} == {"paper_states"}
    assert not any(c["edited_by_researcher"] for c in note["claims"])


def test_search_returns_interpretation_hits_with_trust(server, seeded, home):
    method, assumption, _ = _edit(home, seeded)
    [hit] = ok(call(server, "search", {"query": "wallaby-weighted estimator", "kind": "claims"}))["hits"]
    assert hit["claim_id"] == method and hit["text"] == "A wallaby-weighted estimator."
    assert hit["trust"] == "researcher" and hit["edited_by_researcher"] is True
    assert hit["trust_note"] == "edited by researcher"
    assert hit["original"]["claim_text"] == "A two-step estimator."
    # the label was not edited; the agent wording of the text no longer matches
    assert ok(call(server, "search", {"query": "A two-step estimator", "kind": "claims"}))["hits"] == []
    assert ok(call(server, "search", {"query": "no anticipation", "kind": "claims"}))["hits"] == []
    [added] = ok(call(server, "search", {"query": "bandicoot workshops", "kind": "claims"}))["hits"]
    assert added["claim_id"].startswith("rc_") and added["claim_kind"] == "assertion"
    assert added["trust_note"] == "researcher assertion — not the paper's words"
    assert "page_start" not in added and "original" not in added
    [note] = ok(call(server, "search", {"query": "koala survey", "kind": "notes"}))["hits"]
    assert note["note_id"] == seeded.note_id and note["edited_by_researcher"] is True
    assert "Researcher notes: Compare with the koala survey." in note["text"]


def test_unedited_search_hits_carry_agent_trust(server, seeded):
    [hit] = ok(call(server, "search", {"query": "two-step estimator", "kind": "claims"}))["hits"]
    assert hit["trust"] == "paper_states" and hit["trust_label"] == "paper states"
    assert hit["edited_by_researcher"] is False and "original" not in hit
