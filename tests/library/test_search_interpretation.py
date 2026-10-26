"""Search reads the current interpretation (stage 3 design §4.3, §11 U6): edited wording,
added claims, hidden claims, trust labels, interpretation note text, agent audience rules
and stable order. Invented text only."""

from __future__ import annotations

import pytest
from notes.conftest import MARKDOWN, make_paper
from notes.test_overlay import RESULT_QUOTE, claim_id, raw_note, save_raw

from seedgraph.library import attention, search, store
from seedgraph.notes import overlay


@pytest.fixture
def lib(tmp_path):
    library = store.open_library_at(tmp_path / "library")
    yield library
    library.close()


@pytest.fixture
def noted(lib):
    paper = make_paper(lib)
    return paper, save_raw(lib, paper)


def edit(lib, paper, note_id, field_key, text, version=0):
    cid = claim_id(lib, note_id, field_key)
    return cid, overlay.save_edit(lib, paper.paper_id, {"op": "edit", "target": {"claim_id": cid},
                                                        "set": {"claim_text": text}},
                                  expected_version=version)


def add(lib, paper, version, **op):
    return overlay.save_edit(lib, paper.paper_id, {"op": "add", **op}, expected_version=version)


def claims(lib, query, audience="agent"):
    return search.search(lib, query, kind="claims", audience=audience, limit=None)


# --- claims ------------------------------------------------------------------------------------

def test_an_edited_claim_is_found_by_its_new_wording_and_shows_it(lib, noted):
    paper, saved = noted
    cid, _ = edit(lib, paper, saved.note_id, "method_or_model[0]", "A zigzag gadget tally.")
    hits = claims(lib, "zigzag gadget tally")
    assert [h.claim_id for h in hits] == [cid]
    hit = hits[0]
    assert hit.text == "A zigzag gadget tally."
    assert hit.trust == "researcher" and hit.trust_label == "researcher"
    assert hit.edited_by_researcher and hit.trust_note == "edited by researcher"
    assert hit.original is not None and hit.original.claim_text == "A two-step widget estimator."
    assert hit.page_start is None or isinstance(hit.page_start, int)
    assert hit.span_id is not None  # the quote still anchors the claim
    # the agent's wording no longer matches the interpretation
    assert claims(lib, "two-step widget estimator") == []


def test_added_claims_are_found_with_researcher_trust(lib, noted):
    paper, saved = noted
    v = add(lib, paper, 0, kind="assertion", claim_type="limitation", normalized_label="tiny sample",
            claim_text="Only seven quokka widgets were counted.")
    start = paper.text.index(RESULT_QUOTE)
    v = add(lib, paper, v.version_no, kind="quote", claim_type="result", normalized_label="quokka sign",
            claim_text="A quokka sign shows up.",
            quote={"markdown_id": paper.markdown_id, "start_char": start, "end_char": start + len(RESULT_QUOTE)})
    [assertion] = claims(lib, "seven quokka widgets")
    assert assertion.claim_id.startswith("rc_") and assertion.paper_id == paper.paper_id
    assert assertion.note_id == saved.note_id
    assert assertion.trust == "researcher" and assertion.claim_kind == "assertion"
    assert assertion.trust_note == "researcher assertion — not the paper's words"
    assert assertion.span_id is None and assertion.page_start is None  # never the paper's words
    assert not assertion.edited_by_researcher and assertion.original is None
    assert (assertion.overlay_id, assertion.version_no) == (v.overlay_id, v.version_no)
    [quoted] = claims(lib, "quokka sign shows")
    assert quoted.claim_kind == "quote" and quoted.trust == "researcher" and quoted.trust_note is None
    assert quoted.span_id is not None and quoted.markdown_id == paper.markdown_id


def test_hidden_claims_are_found_by_no_audience(lib, noted):
    paper, saved = noted
    cid = claim_id(lib, saved.note_id, "assumptions[0]")
    overlay.save_edit(lib, paper.paper_id, {"op": "hide", "target": {"claim_id": cid}}, expected_version=0)
    for audience in ("agent", "researcher"):
        assert cid not in {h.claim_id for h in claims(lib, "no anticipation", audience)}
    # an unedited claim of the same paper keeps its agent trust
    [kept] = claims(lib, "counted without error")
    assert kept.trust == "agent_inferred" and kept.trust_label == "agent inferred"
    assert not kept.edited_by_researcher and kept.claim_kind == "agent" and kept.overlay_id is not None


def test_unedited_papers_get_the_derived_trust(lib, noted):
    [found] = claims(lib, "invented widgets grow")
    assert found.trust == "paper_states" and found.claim_kind == "agent" and found.overlay_id is None
    [inferred] = claims(lib, "counted without error")
    assert inferred.trust == "agent_inferred" and inferred.trust_note is None


# --- notes -------------------------------------------------------------------------------------

def test_a_note_hit_shows_the_interpretation_text_with_researcher_notes(lib, noted):
    paper, saved = noted
    _, v = edit(lib, paper, saved.note_id, "method_or_model[0]", "A zigzag gadget tally.")
    v = overlay.save_notes(lib, paper.paper_id, "Check the wombat appendix.", expected_version=v.version_no)
    [hit] = search.search(lib, "wombat appendix", kind="notes")
    assert hit.note_id == saved.note_id and hit.edited_by_researcher
    assert "zigzag gadget tally" in hit.text and "Researcher notes: Check the wombat appendix." in hit.text
    assert "two-step widget estimator" not in hit.text
    assert (hit.overlay_id, hit.version_no) == (v.overlay_id, v.version_no)
    assert search.search(lib, "two-step widget estimator", kind="notes") == []


def test_an_unedited_note_hit_is_unchanged(lib, noted):
    paper, saved = noted
    [hit] = search.search(lib, "two-step widget estimator", kind="notes")
    stored = lib.conn.execute("SELECT note_text FROM structured_notes WHERE note_id = ?",
                              (saved.note_id,)).fetchone()[0]
    assert hit.text == stored and not hit.edited_by_researcher and hit.overlay_id is None


# --- agent audience ----------------------------------------------------------------------------

def test_a_hides_notes_item_hides_added_and_edited_claims_from_agents(lib, noted):
    paper, saved = noted
    _, v = edit(lib, paper, saved.note_id, "method_or_model[0]", "A zigzag gadget tally.")
    add(lib, paper, v.version_no, kind="assertion", claim_type="limitation", normalized_label="tiny",
        claim_text="Only seven quokka widgets were counted.")
    attention.raise_item(lib, "pdf_title_miss", {"paper_id": paper.paper_id}, paper_id=paper.paper_id,
                         detail="invented", hides_notes=True)
    for q in ("zigzag gadget tally", "seven quokka widgets"):
        assert claims(lib, q) == []
        [hit] = claims(lib, q, "researcher")
        assert hit.hidden
    assert search.search(lib, "zigzag gadget tally", kind="notes") == []


def test_a_newer_extraction_of_an_edited_paper_is_not_current_for_agents(lib, noted):
    paper, saved = noted
    edit(lib, paper, saved.note_id, "method_or_model[0]", "A zigzag gadget tally.")
    newer = save_raw(lib, paper, raw_note(method_text="A platypus widget estimator."))
    # the overlay pins its base: the re-note is note_latest but not current
    assert claims(lib, "platypus widget estimator") == []
    [hit] = claims(lib, "platypus widget estimator", "researcher")
    assert hit.note_id == newer.note_id and not hit.is_current and hit.trust == "paper_states"
    assert [h.claim_id for h in claims(lib, "zigzag gadget tally")] == [
        claim_id(lib, saved.note_id, "method_or_model[0]")]


# --- order -------------------------------------------------------------------------------------

def test_order_is_stable_on_ties_across_added_and_agent_claims(lib):
    papers = [make_paper(lib, paper_id="lp_invented0000000002", n=2, text=MARKDOWN + "\nSecond copy.\n"),
              make_paper(lib, paper_id="lp_invented0000000001", n=1)]
    for p in papers:
        save_raw(lib, p)
        v = 0
        for _ in range(2):
            v = add(lib, p, v, kind="assertion", claim_type="limitation", normalized_label="echidna",
                    claim_text="Echidna echidna.").version_no
    hits = claims(lib, "echidna")
    keys = [(h.rank, h.paper_id, h.claim_id) for h in hits]
    assert len(hits) == 4 and keys == sorted(keys)
    assert [h.paper_id for h in hits[:2]] == ["lp_invented0000000001"] * 2
    assert hits == claims(lib, "echidna")
    assert search.match_set(lib, "echidna", kind="claims") == frozenset(h.claim_id for h in hits)
