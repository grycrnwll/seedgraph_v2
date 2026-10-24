"""save_note: one new version per save, anchoring, downgrade, conflict and staleness (design §3)."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from seedgraph import anchor
from seedgraph.library import store
from seedgraph.notes import format as fmt
from seedgraph.notes import save

from .conftest import add_markdown_file

GOOD_QUOTE = "Our two-step widget estimator counts widgets twice and averages."
ASSUMPTION_QUOTE = "Units do not anticipate the invented programme before it starts."


def _note(method_quote=GOOD_QUOTE, assumption_quote=ASSUMPTION_QUOTE):
    return {
        "research_question": {"claim_text": "Do invented widgets grow?", "status": "found",
                              "exact_quote": "We study whether invented widgets grow in a made-up market."},
        "method_or_model": [{"claim_text": "A two-step widget estimator.",
                             "normalized_label": "widget estimator", "status": "found",
                             "method_type": "estimator",
                             "plain_words": "Count the widgets twice and average.",
                             "exact_quote": method_quote}],
        "assumptions": [{"claim_text": "No anticipation of the invented programme.",
                         "normalized_label": "no anticipation", "status": "found",
                         "assumption_type": "identification",
                         "plain_words": "Nobody reacts before the programme starts.",
                         "exact_quote": assumption_quote}],
    }


def _save(conn, paper, raw=None, **kw):
    note, errors = fmt.validate_note(raw or _note())
    assert errors == []
    norm = fmt.normalize_parts([note])
    args = dict(
        paper_id=paper.paper_id, markdown_id=paper.markdown_id, markdown_hash=paper.markdown_hash,
        drafts=norm.drafts, note_text=norm.note_text, archetype=norm.archetype,
        raw_note_json=json.dumps(raw or _note()),
        provenance=save.NoteProvenance(client="claude", model="invented-model",
                                       input_tokens=10, output_tokens=5),
        extraction_mode="agent_whole", part_count=1, batch_id="b_invented",
    )
    args.update(kw)
    return save.save_note(conn, **args)


def _count(lib, table):
    return lib.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _claim(lib, note_id, field_key):
    return lib.conn.execute(
        "SELECT * FROM extracted_claims WHERE structured_note_id = ? AND field_key = ?",
        (note_id, field_key)).fetchone()


def test_first_save_is_version_one_and_current(lib, paper):
    with lib.write() as conn:
        saved = _save(conn, paper)
    assert saved.version_no == 1
    current = lib.conn.execute("SELECT note_id, schema_version, prompt_version FROM note_current "
                               "WHERE paper_id = ?", (paper.paper_id,)).fetchone()
    assert tuple(current) == (saved.note_id, "2.0.0", "2.0.0")
    assert saved.claim_count == _count(lib, "extracted_claims")
    assert saved.quotes_found == 3 and saved.quotes_anchored == 3


def test_found_quotes_anchor_and_keep_the_invariant(lib, paper):
    with lib.write() as conn:
        saved = _save(conn, paper)
    rows = lib.conn.execute(
        "SELECT e.* FROM claim_spans cs JOIN evidence_spans e ON e.span_id = cs.span_id "
        "JOIN extracted_claims c ON c.claim_id = cs.claim_id WHERE c.structured_note_id = ?",
        (saved.note_id,)).fetchall()
    assert len(rows) == 3
    for row in rows:
        anchor.assert_invariant(paper.text, SimpleNamespace(**dict(row)))
    assert _claim(lib, saved.note_id, "method_or_model[0]")["status"] == "found"


def test_unanchored_quote_becomes_ambiguous(lib, paper):
    raw = _note(method_quote="a sentence that is not in the invented text",
                assumption_quote="Widgets grow by an invented amount.")  # occurs twice
    with lib.write() as conn:
        saved = _save(conn, paper, raw=raw)
    for key in ("method_or_model[0]", "assumptions[0]"):
        claim = _claim(lib, saved.note_id, key)
        assert claim["status"] == "ambiguous"
        assert "could not be anchored" in claim["inferred_explanation"]
        assert lib.conn.execute("SELECT COUNT(*) FROM claim_spans WHERE claim_id = ?",
                                (claim["claim_id"],)).fetchone()[0] == 0
    assert saved.quotes_found == 3 and saved.quotes_anchored == 1


def test_plain_words_are_stored_and_searchable(lib, paper):
    with lib.write() as conn:
        saved = _save(conn, paper)
    claim = _claim(lib, saved.note_id, "assumptions[0]")
    assert claim["plain_words"] == "Nobody reacts before the programme starts."
    hits = lib.conn.execute("SELECT claim_id FROM claim_fts WHERE claim_fts MATCH 'plain_words:reacts'").fetchall()
    assert [h[0] for h in hits] == [claim["claim_id"]]
    assert lib.conn.execute("SELECT COUNT(*) FROM note_fts WHERE note_id = ?",
                            (saved.note_id,)).fetchone()[0] == 1


def test_epistemic_type_column_is_derived(lib, paper):
    raw = _note()
    raw["assumptions"][0].update(assertion_status="inferred", inferred_explanation="invented why",
                                 epistemic_type="user_validated")
    with lib.write() as conn:
        saved = _save(conn, paper, raw=raw)
    assert _claim(lib, saved.note_id, "assumptions[0]")["epistemic_type"] == "llm_inferred"
    assert _claim(lib, saved.note_id, "method_or_model[0]")["epistemic_type"] == "llm_extracted"


def test_run_records_agent_provenance(lib, paper):
    with lib.write() as conn:
        saved = _save(conn, paper, extraction_mode="agent_parts", part_count=3)
    run = lib.conn.execute("SELECT * FROM extraction_runs WHERE extraction_run_id = ?",
                           (saved.extraction_run_id,)).fetchone()
    assert run["provider"] == "claude" and run["model_name"] == "invented-model"
    assert run["access_mode"] == "agent_session"
    assert run["extraction_mode"] == "agent_parts" and run["chunk_count"] == 3
    assert run["run_id"] == "b_invented" and run["run_status"] == "success"
    assert (run["input_tokens"], run["output_tokens"]) == (10, 5)
    assert run["schema_id"] == "default_research_note_v1"


def test_version_no_is_max_plus_one_including_labelled_versions(lib, paper):
    with lib.write() as conn:
        first = _save(conn, paper)
        conn.execute("UPDATE structured_notes SET version_label = 'wrong_pdf' WHERE note_id = ?",
                     (first.note_id,))
        second = _save(conn, paper)
    assert (first.version_no, second.version_no) == (1, 2)
    with lib.write() as conn:
        third = _save(conn, paper, expected_current=second.note_id)
    assert third.version_no == 3
    current = lib.conn.execute("SELECT note_id FROM note_current WHERE paper_id = ?",
                               (paper.paper_id,)).fetchone()[0]
    assert current == third.note_id


def test_stale_expected_current_is_a_conflict_and_writes_nothing(lib, paper):
    with lib.write() as conn:
        _save(conn, paper, expected_current=None)
    before = {t: _count(lib, t) for t in ("extraction_runs", "structured_notes",
                                          "extracted_claims", "claim_spans", "evidence_spans")}
    other = store.open_library_at(lib.root)
    try:
        with pytest.raises(save.NoteConflict):
            with other.write() as conn:
                _save(conn, paper, expected_current=None)  # prepared before the first save
    finally:
        other.close()
    assert {t: _count(lib, t) for t in before} == before


def test_concurrent_saves_get_distinct_versions(lib, paper):
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def worker():
        mine = store.open_library_at(lib.root)
        try:
            barrier.wait()
            with mine.write() as conn:
                _save(conn, paper)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)
        finally:
            mine.close()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    versions = [r[0] for r in lib.conn.execute(
        "SELECT version_no FROM structured_notes WHERE paper_id = ? ORDER BY version_no",
        (paper.paper_id,))]
    assert versions == [1, 2, 3, 4]


def test_reconverted_paper_is_stale_and_writes_nothing(lib, paper):
    with lib.write() as conn:
        new_md, _ = add_markdown_file(lib, conn, paper.source_file_id, "invented new text\n", 9)
        conn.execute("UPDATE paper_files SET markdown_id = ? WHERE paper_id = ?",
                     (new_md, paper.paper_id))
    with pytest.raises(save.NoteStale):
        with lib.write() as conn:
            _save(conn, paper)
    assert _count(lib, "structured_notes") == 0


def test_save_outside_a_transaction_is_refused(lib, paper):
    with pytest.raises(save.NoteSaveError):
        _save(lib.conn, paper)
    assert _count(lib, "structured_notes") == 0


def test_shared_quote_is_one_span_with_two_anchors(lib, paper):
    raw = _note(assumption_quote=GOOD_QUOTE)
    with lib.write() as conn:
        _save(conn, paper, raw=raw)
    assert _count(lib, "evidence_spans") == 2
    assert _count(lib, "claim_spans") == 3


def test_not_found_rows_are_saved_without_anchors(lib, paper):
    with lib.write() as conn:
        saved = _save(conn, paper)
    claim = _claim(lib, saved.note_id, "data_sources")
    assert claim["status"] == "not_found" and claim["claim_type"] == "data"


def test_nothing_is_saved_when_the_paper_is_unknown(lib, paper):
    with pytest.raises(save.NoteStale):
        with lib.write() as conn:
            _save(conn, paper, paper_id="lp_unknown")
