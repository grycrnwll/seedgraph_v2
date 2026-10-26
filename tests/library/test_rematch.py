"""Quote re-matching ladder (design §5.5 step 4). All text is invented."""

from __future__ import annotations

from seedgraph.library.rematch import match_quote

CLEAN = (
    "The first step of the invented estimator is identified under different "
    "conditions, and its efficiency reflects the made-up sample design."
)


def test_an_exact_unique_quote_matches_exactly():
    m = match_quote("its efficiency reflects", CLEAN)
    assert m is not None and m.method == "exact"
    assert CLEAN[m.start:m.end] == "its efficiency reflects"


def test_a_folded_old_quote_matches_clean_text():
    old = "The Örst step of the invented estimator is identiÖed under di§erent conditions"
    m = match_quote(old, CLEAN)
    assert m is not None and m.method == "folded"
    assert CLEAN[m.start:m.end] == (
        "The first step of the invented estimator is identified under different conditions")


def test_family_a_ffi_with_a_space_and_fl_fold():
    m = match_quote("its e¢ ciency reáects", CLEAN)
    assert m is not None and m.method == "folded"
    assert CLEAN[m.start:m.end] == "its efficiency reflects"


def test_a_quote_with_different_line_breaks_matches_whitespace_normalised():
    text = "The invented\nestimator   is identified\n\nunder different conditions."
    m = match_quote("estimator is identiÖed under\ndi§erent", text)
    assert m is not None and m.method == "whitespace"
    assert text[m.start:m.end] == "estimator   is identified\n\nunder different"


def test_a_missing_quote_does_not_match():
    assert match_quote("a sentence the new text never had", CLEAN) is None


def test_an_ambiguous_quote_does_not_match():
    assert match_quote("the", "the cat and the dog") is None
    assert match_quote("a  b", "a b and a\nb") is None


# --- rematch of a researcher-edited paper (stage 3 §4.1) --------------------------------------

def test_rematch_of_an_edited_paper_with_a_newer_renote_carries_the_latest_extraction(tmp_path):
    """The interpretation is pinned to the overlay base; a rematch carries the newest
    extraction and is conflict-checked against it (``note_latest``)."""
    from notes.conftest import add_markdown_file, make_paper
    from notes.test_overlay import claim_id, raw_note, save_raw

    from seedgraph.library import store
    from seedgraph.library.rematch import rematch
    from seedgraph.notes import interpretation, overlay, save

    lib = store.open_library_at(tmp_path / "library")
    try:
        paper = make_paper(lib)
        v1 = save_raw(lib, paper)
        overlay.save_edit(lib, paper.paper_id, {
            "op": "edit", "target": {"claim_id": claim_id(lib, v1.note_id, "method_or_model[0]")},
            "set": {"claim_text": "An invented researcher wording."}}, expected_version=0)
        renote = save_raw(lib, paper, raw_note("A newer invented agent wording."))
        before = interpretation.current(lib, paper.paper_id, "researcher")

        new_text = "A preface from the invented re-conversion.\n\n" + paper.text
        with lib.write() as conn:
            md_b, _ = add_markdown_file(lib, conn, paper.source_file_id, new_text, 2)
            conn.execute("UPDATE paper_files SET markdown_id = ? WHERE paper_id = ?", (md_b, paper.paper_id))
        result = rematch(lib, paper.paper_id, paper.markdown_id, md_b)

        assert result.status == "carried" and result.old_note_id == renote.note_id
        assert save.latest_note_id(lib.conn, paper.paper_id) == result.note_id
        carried = lib.conn.execute(
            "SELECT claim_text FROM extracted_claims WHERE structured_note_id = ? AND field_key = ?",
            (result.note_id, "method_or_model[0]")).fetchone()[0]
        assert carried == "A newer invented agent wording."
        assert save.current_note_id(lib.conn, paper.paper_id) == v1.note_id
        after = interpretation.current(lib, paper.paper_id, "researcher")
        assert after.claims == before.claims and after.newer_extraction
    finally:
        lib.close()
