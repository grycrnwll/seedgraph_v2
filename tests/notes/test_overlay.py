"""Note overlays (stage 3 design §4.1-4.2, ADR-0011): schema, versions, revert, stale saves,
hide/add/notes, FTS in the same transaction, and the pin on re-notes. Invented text only."""

from __future__ import annotations

import dataclasses
import json

import pytest

from seedgraph.notes import format as fmt
from seedgraph.notes import overlay, save

METHOD_QUOTE = "Our two-step widget estimator counts widgets twice and averages."
ASSUMPTION_QUOTE = "Units do not anticipate the invented programme before it starts."
RESULT_QUOTE = "The micro-sign µ appears once in this invented text."


def raw_note(method_text="A two-step widget estimator."):
    return {
        "research_question": {"claim_text": "Do invented widgets grow?", "status": "found",
                              "exact_quote": "We study whether invented widgets grow in a made-up market."},
        "method_or_model": [{"claim_text": method_text,
                             "normalized_label": "widget estimator", "status": "found",
                             "method_type": "estimator",
                             "plain_words": "Count the widgets twice and average.",
                             "exact_quote": METHOD_QUOTE}],
        "assumptions": [{"claim_text": "No anticipation of the invented programme.",
                         "normalized_label": "no anticipation", "status": "found",
                         "assumption_type": "identification",
                         "exact_quote": ASSUMPTION_QUOTE},
                        {"claim_text": "Widgets are counted without error.",
                         "normalized_label": "exact counting", "status": "inferred",
                         "assumption_type": "general",
                         "inferred_explanation": "invented reason"}],
    }


def save_raw(lib, paper, raw=None, **kw):
    raw = raw or raw_note()
    note, errors = fmt.validate_note(raw)
    assert errors == []
    norm = fmt.normalize_parts([note])
    with lib.write() as conn:
        return save.save_note(
            conn, paper_id=paper.paper_id, markdown_id=paper.markdown_id,
            markdown_hash=paper.markdown_hash, drafts=norm.drafts, note_text=norm.note_text,
            archetype=norm.archetype, raw_note_json=json.dumps(raw),
            provenance=save.NoteProvenance(client="claude", model="invented-model"), **kw)


def claim_id(lib, note_id, field_key):
    return lib.conn.execute(
        "SELECT claim_id FROM extracted_claims WHERE structured_note_id = ? AND field_key = ?",
        (note_id, field_key)).fetchone()[0]


def old_note_current(conn):
    """The stage 1 definition of note_current, verbatim."""
    return conn.execute(
        "SELECT n.* FROM structured_notes n WHERE n.version_label IS NULL AND n.version_no = ("
        " SELECT MAX(o.version_no) FROM structured_notes o WHERE o.paper_id = n.paper_id"
        " AND o.schema_id IS n.schema_id AND o.version_label IS NULL) ORDER BY n.note_id").fetchall()


def rows(conn, sql, *args):
    return [tuple(r) for r in conn.execute(sql, args).fetchall()]


@pytest.fixture
def noted(lib, paper):
    saved = save_raw(lib, paper)
    return paper, saved


# --- no overlay: nothing changes ------------------------------------------------------------

def test_without_overlays_note_current_is_the_stage1_definition(lib, paper):
    first = save_raw(lib, paper)
    save_raw(lib, paper)
    save_raw(lib, paper)
    with lib.write() as conn:  # a labelled newest version is never current
        conn.execute("UPDATE structured_notes SET version_label = 'wrong_pdf' "
                     "WHERE paper_id = ? AND version_no = 3", (paper.paper_id,))
    expected = [tuple(r) for r in old_note_current(lib.conn)]
    assert expected and expected[0][0] != first.note_id
    assert rows(lib.conn, "SELECT * FROM note_current ORDER BY note_id") == expected
    assert rows(lib.conn, "SELECT * FROM note_latest ORDER BY note_id") == expected


def test_without_overlays_the_migration_dump_gains_no_rows(lib, noted):
    from seedgraph.migrate import dump

    tables = {json.loads(line)["table"] for line in dump.dump_db_lines(lib.db_path, "library")}
    assert "structured_notes" in tables
    assert not tables & {"note_overlays", "overlay_versions", "interpretation_claims",
                         "interpretation_notes", "concept_refresh"}


def test_without_overlays_current_claims_are_the_current_notes_claims(lib, noted):
    paper, saved = noted
    got = rows(lib.conn, "SELECT claim_id, claim_text, trust, edited FROM current_claims ORDER BY claim_id")
    want = rows(lib.conn, "SELECT claim_id, claim_text FROM extracted_claims "
                          "WHERE structured_note_id = ? ORDER BY claim_id", saved.note_id)
    assert [g[:2] for g in got] == want
    assert {g[3] for g in got} == {0}
    trust = {r[0]: r[1] for r in rows(lib.conn, "SELECT field_key, trust FROM current_claims")}
    assert trust["method_or_model[0]"] == "paper_states"
    assert trust["assumptions[1]"] == "agent_inferred"
    assert trust["setting"] == "agent_inferred"  # a not_found scalar


# --- saving: versions, stale forms, revert ---------------------------------------------------

def edit(lib, paper, cid, expected, **set_):
    return overlay.save_edit(lib, paper.paper_id, {"op": "edit", "target": {"claim_id": cid}, "set": set_},
                             expected_version=expected)


def current_ids(lib):
    return {r[0] for r in rows(lib.conn, "SELECT claim_id FROM current_claims")}


def test_first_edit_creates_the_overlay_pinned_to_the_current_note(lib, noted):
    paper, saved = noted
    cid = claim_id(lib, saved.note_id, "method_or_model[0]")
    v = edit(lib, paper, cid, 0, claim_text="An invented re-worded estimator.")
    assert v.version_no == 1 and v.action == "edit"
    head = overlay.head(lib.conn, paper.paper_id)
    assert head.version_no == 1 and head.base_note_id == saved.note_id
    assert head.base_extraction_run_id == saved.extraction_run_id
    [op] = head.ops
    assert op["op"] == "edit" and op["target"]["claim_id"] == cid
    fp = op["target"]["fingerprint"]
    assert fp["claim_type"] == "method" and fp["field_root"] == "method_or_model"
    assert fp["label_fold"] == "widget estimator"
    assert fp["text_tokens"] == ["a", "estimator", "step", "two", "widget"]
    assert fp["quote_hash"] and fp["start_char"] == paper.text.index(METHOD_QUOTE)
    assert fp["markdown_hash"] == paper.markdown_hash
    text = lib.conn.execute("SELECT claim_text FROM current_claims WHERE claim_id = ?", (cid,)).fetchone()[0]
    assert text == "An invented re-worded estimator."
    # the agent's original is untouched
    assert lib.conn.execute("SELECT claim_text FROM extracted_claims WHERE claim_id = ?",
                            (cid,)).fetchone()[0] == "A two-step widget estimator."


def test_every_save_is_a_new_version_with_the_complete_op_set(lib, noted):
    paper, saved = noted
    m = claim_id(lib, saved.note_id, "method_or_model[0]")
    a = claim_id(lib, saved.note_id, "assumptions[0]")
    edit(lib, paper, m, 0, claim_text="Re-worded once.")
    edit(lib, paper, m, 1, normalized_label="widget counter")
    v3 = overlay.save_edit(lib, paper.paper_id, {"op": "hide", "target": {"claim_id": a}}, expected_version=2)
    assert v3.version_no == 3
    hist = overlay.history(lib.conn, paper.paper_id)
    assert [h.version_no for h in hist] == [3, 2, 1]
    assert [h.action for h in hist] == ["hide", "edit", "edit"]
    ops = overlay.head(lib.conn, paper.paper_id).ops
    assert len(ops) == 2  # one merged edit op per claim, plus the hide
    [e] = [o for o in ops if o["op"] == "edit"]
    assert e["set"] == {"claim_text": "Re-worded once.", "normalized_label": "widget counter"}
    assert overlay.version(lib.conn, paper.paper_id, 1).ops[0]["set"] == {"claim_text": "Re-worded once."}


def test_a_stale_expected_version_is_refused_and_writes_nothing(lib, noted):
    paper, saved = noted
    m = claim_id(lib, saved.note_id, "method_or_model[0]")
    edit(lib, paper, m, 0, claim_text="First tab.")
    with pytest.raises(overlay.OverlayConflict, match="another tab"):
        edit(lib, paper, m, 0, claim_text="Second tab, rendered before the first save.")
    with pytest.raises(overlay.OverlayConflict):
        edit(lib, paper, m, 5, claim_text="From the future.")
    assert overlay.head(lib.conn, paper.paper_id).version_no == 1
    assert lib.conn.execute("SELECT COUNT(*) FROM overlay_versions").fetchone()[0] == 1


def test_revert_is_a_new_version_copying_an_old_op_set(lib, noted):
    paper, saved = noted
    m = claim_id(lib, saved.note_id, "method_or_model[0]")
    a = claim_id(lib, saved.note_id, "assumptions[0]")
    edit(lib, paper, m, 0, claim_text="Re-worded.")
    overlay.save_edit(lib, paper.paper_id, {"op": "hide", "target": {"claim_id": a}}, expected_version=1)
    assert a not in current_ids(lib)
    v = overlay.revert(lib, paper.paper_id, 1, expected_version=2)
    assert v.version_no == 3 and v.action == "revert" and v.reverted_to == 1
    assert a in current_ids(lib)
    assert overlay.head(lib.conn, paper.paper_id).ops == overlay.version(lib.conn, paper.paper_id, 1).ops
    # a revert can itself be reverted; nothing was deleted
    overlay.revert(lib, paper.paper_id, 2, expected_version=3)
    assert a not in current_ids(lib)
    assert [h.version_no for h in overlay.history(lib.conn, paper.paper_id)] == [4, 3, 2, 1]
    with pytest.raises(overlay.OverlayConflict):
        overlay.revert(lib, paper.paper_id, 1, expected_version=3)
    with pytest.raises(overlay.OverlayError):
        overlay.revert(lib, paper.paper_id, 9, expected_version=4)


# --- ops: edit, hide, add, notes -----------------------------------------------------------

def test_edit_can_reword_relabel_and_retype(lib, noted):
    paper, saved = noted
    m = claim_id(lib, saved.note_id, "method_or_model[0]")
    edit(lib, paper, m, 0, claim_text="Re-worded.", normalized_label="widget counter", claim_type="model")
    r = lib.conn.execute("SELECT claim_text, normalized_label, claim_type, edited, trust "
                         "FROM current_claims WHERE claim_id = ?", (m,)).fetchone()
    assert tuple(r) == ("Re-worded.", "widget counter", "model", 1, "researcher")
    # the fingerprint keeps the agent's original type
    assert overlay.head(lib.conn, paper.paper_id).ops[0]["target"]["fingerprint"]["claim_type"] == "method"


@pytest.mark.parametrize("op, message", [
    ({"op": "edit", "set": {"claim_type": "astrology"}}, "claim type"),
    ({"op": "edit", "set": {"normalized_label": "  "}}, "label"),
    ({"op": "edit", "set": {}}, "nothing"),
    ({"op": "edit", "set": {"status": "found"}}, "cannot set"),
    ({"op": "frobnicate"}, "unknown op"),
])
def test_invalid_ops_are_refused(lib, noted, op, message):
    paper, saved = noted
    op = dict(op, target={"claim_id": claim_id(lib, saved.note_id, "method_or_model[0]")})
    with pytest.raises(overlay.OverlayError, match=message):
        overlay.save_edit(lib, paper.paper_id, op, expected_version=0)
    assert overlay.head(lib.conn, paper.paper_id) is None


def test_a_target_outside_the_base_note_is_refused(lib, noted):
    paper, saved = noted
    with pytest.raises(overlay.OverlayError, match="not a claim of"):
        edit(lib, paper, "claim_invented_missing", 0, claim_text="x")


def test_a_paper_without_a_current_note_cannot_be_edited(lib, paper):
    with pytest.raises(overlay.OverlayError, match="no current note"):
        overlay.save_notes(lib, paper.paper_id, "invented notes", expected_version=0)


def test_hide_and_unhide(lib, noted):
    paper, saved = noted
    a = claim_id(lib, saved.note_id, "assumptions[0]")
    overlay.save_edit(lib, paper.paper_id, {"op": "hide", "target": {"claim_id": a}}, expected_version=0)
    assert a not in current_ids(lib)
    assert lib.conn.execute("SELECT hidden FROM interpretation_claims WHERE claim_id = ?", (a,)).fetchone()[0] == 1
    overlay.save_edit(lib, paper.paper_id, {"op": "unhide", "target": {"claim_id": a}}, expected_version=1)
    assert a in current_ids(lib)
    assert overlay.head(lib.conn, paper.paper_id).ops == []


def test_add_assertion_and_quote_backed_claim(lib, noted):
    paper, saved = noted
    v1 = overlay.save_edit(lib, paper.paper_id, {
        "op": "add", "kind": "assertion", "claim_type": "limitation",
        "normalized_label": "small sample", "claim_text": "Only invented widgets were counted."},
        expected_version=0)
    start = paper.text.index(RESULT_QUOTE)
    v2 = overlay.save_edit(lib, paper.paper_id, {
        "op": "add", "kind": "quote", "claim_type": "result",
        "normalized_label": "micro sign", "claim_text": "The text mentions a micro sign.",
        "quote": {"markdown_id": paper.markdown_id, "start_char": start,
                  "end_char": start + len(RESULT_QUOTE)}},
        expected_version=v1.version_no)
    added = rows(lib.conn, "SELECT claim_id, kind, trust, span_id FROM current_claims "
                           "WHERE claim_id LIKE 'rc_%'")
    assert len(added) == 2
    kinds = {r[1]: r for r in added}
    assert kinds["assertion"][2] == "researcher" and kinds["assertion"][3] is None
    assert kinds["quote"][2] == "researcher" and kinds["quote"][3] is not None
    span = lib.conn.execute("SELECT exact_quote FROM evidence_spans WHERE span_id = ?",
                            (kinds["quote"][3],)).fetchone()[0]
    assert span == RESULT_QUOTE
    [q] = [o for o in overlay.head(lib.conn, paper.paper_id).ops if o.get("kind") == "quote"]
    assert q["add_id"] == kinds["quote"][0]
    assert set(q["quote"]) == {"markdown_id", "markdown_hash", "start_char", "end_char", "quote_hash", "span_id"}
    overlay.save_edit(lib, paper.paper_id, {"op": "remove_added", "add_id": q["add_id"]},
                      expected_version=v2.version_no)
    assert q["add_id"] not in current_ids(lib)


def test_an_added_claim_can_be_edited(lib, noted):
    paper, saved = noted
    overlay.save_edit(lib, paper.paper_id, {
        "op": "add", "kind": "assertion", "claim_type": "limitation",
        "normalized_label": "small sample", "claim_text": "Only invented widgets."}, expected_version=0)
    [rc] = [r[0] for r in rows(lib.conn, "SELECT claim_id FROM current_claims WHERE claim_id LIKE 'rc_%'")]
    edit(lib, paper, rc, 1, claim_text="Only a few invented widgets.")
    assert lib.conn.execute("SELECT claim_text FROM current_claims WHERE claim_id = ?",
                            (rc,)).fetchone()[0] == "Only a few invented widgets."
    assert len(overlay.head(lib.conn, paper.paper_id).ops) == 1


@pytest.mark.parametrize("quote, message", [
    ("Widgets grow by an invented amount.", "uniquely"),       # occurs twice
    (None, "anchor"),
])
def test_a_quote_backed_add_must_anchor_exactly_and_uniquely(lib, noted, quote, message):
    paper, saved = noted
    if quote is None:  # offsets outside the markdown
        span = (len(paper.text) + 5, len(paper.text) + 20)
    else:
        start = paper.text.index(quote)
        span = (start, start + len(quote))
    with pytest.raises(overlay.OverlayError, match=message):
        overlay.save_edit(lib, paper.paper_id, {
            "op": "add", "kind": "quote", "claim_type": "result", "normalized_label": "growth",
            "claim_text": "Widgets grow.",
            "quote": {"markdown_id": paper.markdown_id, "start_char": span[0], "end_char": span[1]}},
            expected_version=0)
    assert overlay.head(lib.conn, paper.paper_id) is None


def test_researcher_notes_are_a_version_and_searchable(lib, noted):
    paper, saved = noted
    v = overlay.save_notes(lib, paper.paper_id, "Check the zorblax appendix.", expected_version=0)
    assert v.action == "notes"
    assert overlay.head(lib.conn, paper.paper_id).researcher_notes == "Check the zorblax appendix."
    assert rows(lib.conn, "SELECT note_id FROM note_fts WHERE note_fts MATCH 'zorblax'") == [(saved.note_id,)]


# --- FTS and concept_refresh in the same transaction ----------------------------------------

def test_fts_follows_the_interpretation(lib, noted):
    paper, saved = noted
    m = claim_id(lib, saved.note_id, "method_or_model[0]")
    a = claim_id(lib, saved.note_id, "assumptions[0]")

    def claim_hits(word):
        return {r[0] for r in rows(lib.conn, "SELECT claim_id FROM claim_fts WHERE claim_fts MATCH ?", word)}

    def note_hits(word):
        return {r[0] for r in rows(lib.conn, "SELECT note_id FROM note_fts WHERE note_fts MATCH ?", word)}

    assert m in claim_hits("estimator")
    edit(lib, paper, m, 0, claim_text="A quuxifying widget tally.")
    overlay.save_edit(lib, paper.paper_id, {"op": "hide", "target": {"claim_id": a}}, expected_version=1)
    v = overlay.save_edit(lib, paper.paper_id, {
        "op": "add", "kind": "assertion", "claim_type": "limitation", "normalized_label": "frumious",
        "claim_text": "An invented frumious limitation."}, expected_version=2)
    assert claim_hits("quuxifying") == {m}
    assert a not in claim_hits("anticipation")
    [rc] = claim_hits("frumious")
    assert rc.startswith("rc_")
    assert note_hits("quuxifying") == {saved.note_id} and note_hits("frumious") == {saved.note_id}
    assert note_hits("anticipation") == set()
    # one FTS row per claim / note, never duplicates
    assert lib.conn.execute("SELECT COUNT(*) FROM claim_fts WHERE claim_id = ?", (m,)).fetchone()[0] == 1
    assert lib.conn.execute("SELECT COUNT(*) FROM note_fts WHERE note_id = ?", (saved.note_id,)).fetchone()[0] == 1
    # removing the added claim removes its FTS row; unhiding restores the agent's row
    v = overlay.save_edit(lib, paper.paper_id, {"op": "remove_added", "add_id": rc}, expected_version=v.version_no)
    assert claim_hits("frumious") == set()
    overlay.save_edit(lib, paper.paper_id, {"op": "unhide", "target": {"claim_id": a}},
                      expected_version=v.version_no)
    assert a in claim_hits("anticipation")


def test_a_failure_late_in_the_save_rolls_back_version_derived_rows_and_fts(lib, noted, monkeypatch):
    paper, saved = noted
    m = claim_id(lib, saved.note_id, "method_or_model[0]")

    def boom(*a, **k):
        raise RuntimeError("invented failure after the FTS write")

    monkeypatch.setattr(overlay, "mark_concepts_out_of_date", boom)
    with pytest.raises(RuntimeError):
        edit(lib, paper, m, 0, claim_text="A quuxifying widget tally.")
    assert overlay.head(lib.conn, paper.paper_id) is None
    assert lib.conn.execute("SELECT COUNT(*) FROM interpretation_claims").fetchone()[0] == 0
    assert rows(lib.conn, "SELECT claim_id FROM claim_fts WHERE claim_fts MATCH 'quuxifying'") == []
    assert lib.conn.execute("SELECT COUNT(*) FROM claim_fts WHERE claim_id = ?", (m,)).fetchone()[0] == 1


def test_a_save_marks_the_papers_concepts_out_of_date(lib, noted):
    paper, saved = noted
    assert lib.conn.execute("SELECT COUNT(*) FROM concept_refresh").fetchone()[0] == 0
    overlay.save_notes(lib, paper.paper_id, "invented", expected_version=0)
    since = lib.conn.execute("SELECT since FROM concept_refresh WHERE paper_id = ?",
                             (paper.paper_id,)).fetchone()[0]
    overlay.save_notes(lib, paper.paper_id, "invented again", expected_version=1)
    assert rows(lib.conn, "SELECT paper_id, since FROM concept_refresh") == [(paper.paper_id, since)]


# --- the pin: a re-note of an edited paper ---------------------------------------------------

def test_a_renote_of_an_edited_paper_leaves_note_current_pinned_and_note_latest_new(lib, noted):
    from seedgraph.notes import interpretation
    from seedgraph.work import tasks_notes

    paper, saved = noted
    m = claim_id(lib, saved.note_id, "method_or_model[0]")
    edit(lib, paper, m, 0, claim_text="A researcher wording.")
    before = interpretation.current(lib, paper.paper_id, "researcher")
    current_before = rows(lib.conn, "SELECT * FROM current_claims ORDER BY claim_id")
    fts_before = rows(lib.conn, "SELECT claim_id FROM claim_fts WHERE claim_fts MATCH 'researcher'")
    assert tasks_notes.current_note(lib.conn, paper.paper_id)[0] == saved.note_id
    renote = save_raw(lib, paper, raw_note("A newer agent wording."), expected_current=saved.note_id)
    assert renote.version_no == 2
    assert lib.conn.execute("SELECT note_id FROM note_latest").fetchone()[0] == renote.note_id
    assert lib.conn.execute("SELECT note_id FROM note_current").fetchone()[0] == saved.note_id
    after = interpretation.current(lib, paper.paper_id, "researcher")
    assert after.newer_extraction and not before.newer_extraction  # "newer extraction available"
    assert dataclasses.replace(after, newer_extraction=False) == before
    assert rows(lib.conn, "SELECT * FROM current_claims ORDER BY claim_id") == current_before
    assert rows(lib.conn, "SELECT claim_id FROM claim_fts WHERE claim_fts MATCH 'researcher'") == fts_before
    # extraction freshness and the next conflict check read note_latest
    assert tasks_notes.current_note(lib.conn, paper.paper_id)[0] == renote.note_id
    assert tasks_notes.note_freshness(lib.conn, paper.paper_id) == "fresh"
    assert save.latest_note_id(lib.conn, paper.paper_id) == renote.note_id
    with pytest.raises(save.NoteConflict):
        save_raw(lib, paper, expected_current=saved.note_id)
    # later edits keep the pin and leave the re-note's FTS rows alone
    newer = claim_id(lib, renote.note_id, "method_or_model[0]")
    edit(lib, paper, m, 1, claim_text="A second researcher wording.")
    assert overlay.head(lib.conn, paper.paper_id).base_note_id == saved.note_id
    assert lib.conn.execute("SELECT note_id FROM note_current").fetchone()[0] == saved.note_id
    assert rows(lib.conn, "SELECT claim_text FROM claim_fts WHERE claim_id = ?", newer) == [("A newer agent wording.",)]
