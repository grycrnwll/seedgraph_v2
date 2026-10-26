"""instructions.md content and labels.json input for a notes batch (design §4.2, §4.4)."""

from __future__ import annotations

import json
import re

from seedgraph.library import attention
from seedgraph.notes import format as fmt
from seedgraph.notes import instructions, labels, vocab

from .conftest import MARKDOWN, NOW, make_paper


def _example(text):
    (block,) = re.findall(r"```json\n(.*?)\n```", text, flags=re.S)
    return block


def test_instructions_are_byte_stable():
    assert instructions.render_instructions() == instructions.render_instructions()


def test_worked_example_passes_validation_and_checks_cleanly():
    note, errors = fmt.validate_note(_example(instructions.render_instructions()))
    assert errors == []
    result = fmt.check_note(note)
    assert result.errors == [] and result.warnings == []


def test_field_table_names_every_field_and_vocabulary():
    text = instructions.render_instructions()
    for name in fmt.NoteV2.model_fields:
        assert f"`{name}`" in text
    for values in vocab.VOCABULARIES.values():
        for value in values:
            assert f"`{value}`" in text
    assert "plain_words" in text
    assert "epistemic_type" not in text
    assert "anyOf" not in text


def test_rules_state_null_scalars_quote_rules_and_labels():
    text = instructions.render_instructions().lower()
    assert "scalars may be null, arrays may be empty" in text
    assert "character for character" in text
    assert "never repair" in text
    assert "unique" in text
    assert "labels.json" in text


def test_header_names_the_instructions_version():
    text = instructions.render_instructions()
    assert instructions.INSTRUCTIONS_VERSION == "2.1.0"
    assert f"instructions {instructions.INSTRUCTIONS_VERSION}" in text.splitlines()[0]


def test_reading_tips_cover_long_files_and_approximate_token_counts():
    text = instructions.render_instructions()
    assert "about 300 lines at a time" in text
    assert "`est_tokens` is approximate" in text
    assert "`lines`" in text


def test_submit_names_the_mcp_and_cli_ways():
    text = instructions.render_instructions()
    assert "`note_path`" in text and "`--note-file`" in text


def test_quote_tip_for_formula_heavy_text():
    text = instructions.render_instructions()
    assert "prefer a plain-prose passage" in text
    assert "inline math exactly, including the spaces around it" in text


def test_errata_rule():
    text = instructions.render_instructions().lower()
    assert "erratum or corrigendum" in text
    assert "use the corrected values" in text


def test_assumption_type_guidance_has_one_line_per_value():
    lines = instructions.render_instructions().splitlines()
    for value in vocab.VOCABULARIES["assumption_type"]:
        (line,) = [x for x in lines if x.startswith(f"- `assumption_type` = `{value}`")]
        assert len(line) > 40
    (regularity,) = [x for x in lines if x.startswith("- `assumption_type` = `regularity`")]
    for example in ("sparsity", "smoothness", "moment conditions"):
        assert example in regularity


def test_part_rules_only_when_the_batch_has_parts():
    plain = instructions.render_instructions()
    parts = instructions.render_instructions(has_parts=True)
    assert "part 1" not in plain.lower()
    assert "part 1" in parts.lower() and "research_question" in parts


# --- labels ------------------------------------------------------------------------------

def _note_with_labels(conn, paper_id, n, version_no, items, *, label=None):
    run, note = f"extr_l{n}", f"note_l{n}"
    conn.execute(
        "INSERT INTO extraction_runs (extraction_run_id, paper_id, markdown_id, markdown_hash, "
        "schema_id, schema_version, prompt_version, run_status, created_at) "
        "VALUES (?, ?, 'md_x', 'x', 'default_research_note_v1', '2.0.0', '2.0.0', 'success', ?)",
        (run, paper_id, NOW))
    conn.execute(
        "INSERT INTO structured_notes (note_id, extraction_run_id, paper_id, markdown_id, "
        "markdown_hash, schema_id, schema_version, prompt_version, raw_note_json, note_text, "
        "version_no, version_label, created_at) VALUES (?, ?, ?, 'md_x', 'x', "
        "'default_research_note_v1', '2.0.0', '2.0.0', '{}', 'invented', ?, ?, ?)",
        (note, run, paper_id, version_no, label, NOW))
    for i, (claim_type, lab, status) in enumerate(items):
        conn.execute(
            "INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, "
            "paper_id, claim_type, field_key, normalized_label, status, epistemic_type, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'llm_extracted', ?)",
            (f"c_{n}_{i}", note, run, paper_id, claim_type, f"k{i}", lab, status, NOW))


def test_labels_rank_by_papers_with_a_current_visible_note(lib):
    papers = [make_paper(lib, paper_id=f"lp_invented00000000{i:02d}", n=i,
                         text=MARKDOWN + f"Invented copy {i}.").paper_id
              for i in range(1, 5)]
    with lib.write() as conn:
        # paper 1: an old version uses "old label", the current one does not
        _note_with_labels(conn, papers[0], 1, 1, [("method", "old label", "found")])
        _note_with_labels(conn, papers[0], 2, 2, [("method", "widget estimator", "found"),
                                                  ("model", "gadget model", "found"),
                                                  ("identification_assumption", "no anticipation", "found")])
        _note_with_labels(conn, papers[1], 3, 1, [("method", "widget estimator", "found"),
                                                  ("method", "widget estimator", "found"),
                                                  ("method", "unseen label", "ambiguous")])
        # paper 3: its only version is labelled wrong_pdf -> not current
        _note_with_labels(conn, papers[2], 4, 1, [("method", "wrong pdf label", "found")],
                          label="wrong_pdf")
        # paper 4: hidden by an open hides_notes item
        _note_with_labels(conn, papers[3], 5, 1, [("method", "hidden label", "found")])
    attention.raise_item(lib, kind="pdf_title_miss", paper_id=papers[3],
                         subject={"paper_id": papers[3]}, detail="invented", hides_notes=True)

    groups = labels.label_groups(lib.conn)
    assert groups["methods_and_models"] == ["widget estimator", "gadget model"]
    assert groups["identification_assumptions"] == ["no anticipation"]
    assert groups["results"] == []
    assert set(groups) == set(labels.LABEL_GROUPS)


def test_label_variants_are_listed_once_under_the_most_used_spelling(lib):
    papers = [make_paper(lib, paper_id=f"lp_invented00000000{i:02d}", n=i,
                         text=MARKDOWN + f"Invented copy {i}.").paper_id
              for i in range(1, 5)]
    ident = "identification_assumption"
    with lib.write() as conn:
        _note_with_labels(conn, papers[0], 1, 1, [(ident, "Parallel Trends", "found"),
                                                  ("method", "Event study", "found"),
                                                  ("method", "difference-in-differences", "found")])
        _note_with_labels(conn, papers[1], 2, 1, [(ident, "parallel trends", "found"),
                                                  (ident, "parallel  trends.", "found"),
                                                  ("method", "event study", "found"),
                                                  ("method", "Difference in differences", "found")])
        _note_with_labels(conn, papers[2], 3, 1, [(ident, "parallel trends", "found"),
                                                  (ident, "no anticipation", "found"),
                                                  ("method", "difference in differences", "found")])
        _note_with_labels(conn, papers[3], 4, 1, [(ident, "No anticipation", "found"),
                                                  ("method", "difference in differences", "found")])
    groups = labels.label_groups(lib.conn)
    # three papers use some spelling of parallel trends (paper 2 twice: counted once)
    assert groups["identification_assumptions"] == ["parallel trends", "No anticipation"]
    # hyphen and space spellings are one label; a tie picks the first spelling in sort order
    assert groups["methods_and_models"] == ["difference in differences", "Event study"]


def test_label_limit_applies_after_grouping(lib):
    papers = [make_paper(lib, paper_id=f"lp_invented00000000{i:02d}", n=i,
                         text=MARKDOWN + f"Invented copy {i}.").paper_id
              for i in range(1, 6)]
    with lib.write() as conn:
        _note_with_labels(conn, papers[0], 1, 1, [("method", "Gadget Model", "found")])
        _note_with_labels(conn, papers[1], 2, 1, [("method", "gadget model", "found")])
        _note_with_labels(conn, papers[2], 3, 1, [("method", "gadget model;", "found")])
        _note_with_labels(conn, papers[3], 4, 1, [("method", "widget model", "found")])
        _note_with_labels(conn, papers[4], 5, 1, [("method", "widget model", "found")])
    assert labels.label_groups(lib.conn, limit=1)["methods_and_models"] == ["Gadget Model"]


def test_labels_json_is_stable(lib):
    a = labels.labels_json(lib.conn)
    assert a == labels.labels_json(lib.conn)
    assert set(json.loads(a)) == set(labels.LABEL_GROUPS)
