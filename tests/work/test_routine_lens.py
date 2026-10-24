"""The ``lens:<id>`` task (design §2.2): the v2 lens behaviour on packages. Record rules
kept from v2 ``lenses/runner._project_found_record``: anchor first, then validate; a
found record with a span becomes a claim plus a ``found`` output; a type/enum failure or
an inferred record without notes is ``extraction_failed`` (no claim); a span-less found
record is ``ambiguous`` (no claim); no records -> one ``not_found`` output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seedgraph.lenses.prompt import LENS_PROMPT_VERSION
from seedgraph.lenses.schema import LensDefinition
from seedgraph.work import routine, tasks_lens
from seedgraph.work.claims import WorkerRef

from .routine_support import ASSUMPTION_QUOTE, METHOD_QUOTE, make_paper, open_lib

W1 = WorkerRef("interactive", "w1", client="codex")

LENS = {
    "lens_id": "invented_conditions_v1",
    "name": "Invented conditions",
    "object_type": "assumption",
    "positive_anchors": ["anticipate", "estimator"],
    "negative_anchors": ["related literature"],
    "output_schema": {
        "condition_text_verbatim": "string",
        "normalized_condition_label": "string",
        "condition_type": {"type": "array", "allowed_values": ["identification", "other"]},
        "section": "string",
        "stated_or_inferred": {"allowed_values": ["stated", "inferred"]},
        "evidence_span_ids": {"type": "array"},
        "confidence": "number",
        "notes": "string",
    },
}


def record(quote, label, **extra):
    return {"condition_text_verbatim": quote, "normalized_condition_label": label,
            "condition_type": ["identification"], "section": "Method", "stated_or_inferred": "stated",
            "evidence_span_ids": [], "confidence": 0.7, **extra}


@pytest.fixture
def lib(tmp_path):
    library = open_lib(tmp_path)
    yield library
    library.close()


@pytest.fixture
def lens(lib):
    return tasks_lens.register_lens(lib, LensDefinition.from_dict(LENS), status="active")


def _claim(lib, paper, **opts):
    view = routine.create_batch(lib, f"lens:{LENS['lens_id']}", paper_ids=[paper.paper_id], **opts)
    reply = routine.claim_next(lib, view.batch_id, W1)
    return view, reply.message


def test_lens_instructions_carry_the_lens_prompt(lib, lens):
    paper = make_paper(lib, 1)
    view, msg = _claim(lib, paper)
    assert view.task_version == lens.definition_hash()
    text = Path(msg["instructions_path"]).read_text(encoding="utf-8")
    assert "`assumption`" in text and "anticipate" in text and "condition_type" in text
    assert "DOCUMENT" not in text.split("## Lens prompt")[0]


def test_lens_records_follow_the_v2_rules(lib, lens):
    paper = make_paper(lib, 1)
    view, msg = _claim(lib, paper)
    payload = {"records": [
        record(ASSUMPTION_QUOTE, "no anticipation"),
        record(METHOD_QUOTE, "bad enum", condition_type=["not-a-type"]),
        record(METHOD_QUOTE, "inferred without notes", stated_or_inferred="inferred"),
        record("A sentence the invented paper never says.", "unanchored"),
    ], "not_found": {"searched_sections": ["Method"]}}
    first = routine.submit(lib, msg["packet_id"], msg["claim_id"], note=payload)
    assert first.status == "needs_revision"
    assert [f["field_key"] for f in first.failed_quotes] == ["assumption[3]"]
    assert sum("extraction_failed" in w["message"] for w in first.warnings) == 2

    done = routine.submit(lib, msg["packet_id"], msg["claim_id"], note=payload, accept_unanchored=True)
    assert done.status == "accepted" and done.paper_status == "done"
    run = lib.conn.execute(
        "SELECT lens_id, lens_definition_hash, schema_id, prompt_version, schema_version, run_id, paper_id "
        "FROM extraction_runs WHERE extraction_run_id = ?", (done.note_id,)).fetchone()
    assert tuple(run) == (LENS["lens_id"], lens.definition_hash(), None, LENS_PROMPT_VERSION,
                          tasks_lens.LENS_SCHEMA_VERSION, view.batch_id, paper.paper_id)
    outputs = lib.conn.execute(
        "SELECT status, claim_id IS NOT NULL, fields_json FROM lens_outputs WHERE extraction_run_id = ? "
        "ORDER BY created_at, lens_output_id", (done.note_id,)).fetchall()
    by_label = {json.loads(f)["normalized_condition_label"]: (s, bool(c)) for s, c, f in outputs}
    assert by_label == {"no anticipation": ("found", True), "bad enum": ("extraction_failed", False),
                        "inferred without notes": ("extraction_failed", False),
                        "unanchored": ("ambiguous", False)}
    claim = lib.conn.execute(
        "SELECT claim_type, field_key, claim_text, normalized_label, structured_note_id, epistemic_type "
        "FROM extracted_claims WHERE extraction_run_id = ?", (done.note_id,)).fetchone()
    assert tuple(claim) == ("general_assumption", "assumption[0]", ASSUMPTION_QUOTE, "no anticipation",
                            None, "llm_extracted")
    assert lib.conn.execute("SELECT COUNT(*) FROM claim_spans cs JOIN extracted_claims c USING (claim_id) "
                            "WHERE c.extraction_run_id = ?", (done.note_id,)).fetchone()[0] == 1

    results = tasks_lens.lens_results(lib, LENS["lens_id"])
    assert [(r["paper_id"], r["normalized_label"]) for r in results] == [(paper.paper_id, "no anticipation")]
    assert len(tasks_lens.lens_results(lib, LENS["lens_id"], status="all")) == 4


def test_no_records_writes_one_not_found_output(lib, lens):
    paper = make_paper(lib, 1)
    _, msg = _claim(lib, paper)
    nf = {"records": [], "not_found": {"searched_sections": ["Method", "Results"], "notes": "none", "confidence": 0.6}}
    done = routine.submit(lib, msg["packet_id"], msg["claim_id"], note=nf)
    assert done.paper_status == "done"
    (status, fields) = lib.conn.execute(
        "SELECT status, fields_json FROM lens_outputs WHERE extraction_run_id = ?", (done.note_id,)).fetchone()
    assert status == "not_found"
    assert json.loads(fields) == {"status": "not_found", "searched_sections": ["Method", "Results"],
                                  "notes": "none", "confidence": 0.6}


def test_a_current_lens_run_is_preserved_and_the_query_finds_missing(lib, lens):
    a, b = make_paper(lib, 1), make_paper(lib, 2)
    _, msg = _claim(lib, a)
    routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"records": []})
    again = routine.create_batch(lib, f"lens:{LENS['lens_id']}", paper_ids=[a.paper_id])
    assert again.items[0].state == "preserved"
    missing = routine.create_batch(lib, f"lens:{LENS['lens_id']}", query=f"held lens {LENS['lens_id']} missing")
    assert [i.subject_id for i in missing.items] == [b.paper_id]


def test_a_payload_without_records_list_is_invalid(lib, lens):
    paper = make_paper(lib, 1)
    _, msg = _claim(lib, paper)
    bad = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"records": "none"})
    assert bad.status == "invalid"


def test_ollama_is_not_offered_lens_packages(lib, lens):
    paper = make_paper(lib, 1)
    view = routine.create_batch(lib, f"lens:{LENS['lens_id']}", paper_ids=[paper.paper_id])
    reply = routine.claim_next(lib, view.batch_id, WorkerRef("ollama", "o1"))
    assert reply.status == "not_allowed"
    assert routine.batch_view(lib, view.batch_id).state.live_claims == 0


def test_a_batch_keeps_its_lens_version_when_a_new_one_is_registered(lib, lens):
    paper = make_paper(lib, 1)
    view, msg = _claim(lib, paper)
    v2 = tasks_lens.register_lens(lib, LensDefinition.from_dict({**LENS, "positive_anchors": ["gadget"]}))
    assert v2.definition_hash() != lens.definition_hash()
    with lib.write() as conn:
        conn.execute("UPDATE lens_definitions SET status = 'archived' WHERE definition_hash = ?",
                     (lens.definition_hash(),))
    done = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"records": [record(ASSUMPTION_QUOTE, "x")]})
    assert done.paper_status == "done"
    (stamped,) = lib.conn.execute("SELECT lens_definition_hash FROM extraction_runs WHERE extraction_run_id = ?",
                                  (done.note_id,)).fetchone()
    assert stamped == lens.definition_hash() == view.task_version
