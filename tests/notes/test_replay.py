"""Replay of researcher edits on a re-extraction (stage 3 design §4.4-4.5): the plan writes
nothing, accept re-keys ops onto the new base, refuses a stale plan, and keeps what the
researcher wrote. Invented text only."""

from __future__ import annotations

import dataclasses

import pytest

from seedgraph.notes import interpretation, overlay, replay

from .conftest import add_markdown_file
from .test_overlay import ASSUMPTION_QUOTE, METHOD_QUOTE, RESULT_QUOTE, claim_id, raw_note, save_raw

RQ_QUOTE = "We study whether invented widgets grow in a made-up market."


def raw_v2(*, method_text="A two-step estimator that double counts widgets.", method_label="widget estimator",
           method_quote=METHOD_QUOTE, extra_assumptions=(), exact_counting=True, result=True):
    """A re-extraction: new wording, an assumption inserted first (indexes shift), a new result."""
    assumptions = [{"claim_text": "Widget units are stable over time.", "normalized_label": "stable units",
                    "status": "inferred", "assumption_type": "general",
                    "inferred_explanation": "invented reason"},
                   {"claim_text": "Units do not anticipate the invented programme.",
                    "normalized_label": "no anticipation", "status": "found",
                    "assumption_type": "identification", "exact_quote": ASSUMPTION_QUOTE},
                   *extra_assumptions]
    if exact_counting:
        assumptions.append({"claim_text": "Widgets are counted without any error.",
                            "normalized_label": "exact counting", "status": "inferred",
                            "assumption_type": "general", "inferred_explanation": "invented reason"})
    raw = {
        "research_question": {"claim_text": "Do invented widgets grow in a made-up market?", "status": "found",
                              "exact_quote": RQ_QUOTE},
        "method_or_model": [{"claim_text": method_text, "normalized_label": method_label, "status": "found",
                             "method_type": "estimator", "exact_quote": method_quote}],
        "assumptions": assumptions,
    }
    if result:
        raw["main_results"] = [{"claim_text": "A micro-sign appears in the invented text.",
                                "normalized_label": "micro sign", "status": "found", "result_type": "main",
                                "exact_quote": RESULT_QUOTE}]
    return raw


def edit(lib, paper, cid, expected, **set_):
    return overlay.save_edit(lib, paper.paper_id, {"op": "edit", "target": {"claim_id": cid}, "set": set_},
                             expected_version=expected)


def hide(lib, paper, cid, expected):
    return overlay.save_edit(lib, paper.paper_id, {"op": "hide", "target": {"claim_id": cid}},
                             expected_version=expected)


def add_assertion(lib, paper, expected, text="An invented frumious limitation."):
    return overlay.save_edit(lib, paper.paper_id, {
        "op": "add", "kind": "assertion", "claim_type": "limitation", "normalized_label": "frumious",
        "claim_text": text}, expected_version=expected)


def by_text(interp, text):
    return next(c for c in interp.claims if c.claim_text == text)


def dump(lib):
    return list(lib.conn.iterdump())


@pytest.fixture
def v1(lib, paper):
    return save_raw(lib, paper)


@pytest.fixture
def edited(lib, paper, v1):
    """Edit the method, hide 'exact counting', add an assertion: overlay version 3."""
    m = claim_id(lib, v1.note_id, "method_or_model[0]")
    ec = claim_id(lib, v1.note_id, "assumptions[1]")
    edit(lib, paper, m, 0, claim_text="A quuxifying tally.")
    hide(lib, paper, ec, 1)
    add_assertion(lib, paper, 2)
    return {"method": m, "exact_counting": ec}


def op_by_key(plan, key):
    return next(o for o in plan.ops if o.key == key)


# --- the end-to-end path ---------------------------------------------------------------------

def test_edit_hide_and_add_survive_a_reextraction_after_accept(lib, paper, v1, edited):
    before = interpretation.current(lib, paper.paper_id, "researcher", show_hidden=True)
    v2 = save_raw(lib, paper, raw_v2())  # notes.save: the re-note becomes note_latest only

    after_save = interpretation.current(lib, paper.paper_id, "researcher", show_hidden=True)
    assert after_save.claims == before.claims and after_save.note_text == before.note_text
    assert after_save.note_id == v1.note_id and after_save.newer_extraction

    snapshot = dump(lib)
    plan = replay.plan_replay(lib, paper.paper_id)
    assert dump(lib) == snapshot  # the plan writes nothing
    assert (plan.base_note_id, plan.new_note_id) == (v1.note_id, v2.note_id)
    assert op_by_key(plan, f"edit:{edited['method']}").status == "kept"
    assert op_by_key(plan, f"hide:{edited['exact_counting']}").status == "still_hidden"
    [added] = [o for o in plan.ops if o.op["op"] == "add"]
    assert added.status == "added_kept"
    assert {c.normalized_label for c in plan.new_claims} == {"stable units", "micro sign"}
    changed = {c.field_key: c for c in plan.changed}
    assert changed["research_question"].new_text == "Do invented widgets grow in a made-up market?"

    v = replay.accept(lib, paper.paper_id, plan)
    assert v.action == "accept_extraction" and v.base_note_id == v2.note_id

    now = interpretation.current(lib, paper.paper_id, "researcher", show_hidden=True)
    assert now.note_id == v2.note_id and not now.newer_extraction
    new_m = claim_id(lib, v2.note_id, "method_or_model[0]")
    tally = by_text(now, "A quuxifying tally.")
    assert tally.claim_id == new_m and tally.edited_by_researcher and tally.trust == "researcher"
    assert tally.original.claim_text == "A two-step estimator that double counts widgets."
    counting = next(c for c in now.claims if c.normalized_label == "exact counting")
    assert counting.hidden and counting.claim_id == claim_id(lib, v2.note_id, "assumptions[2]")
    assert by_text(now, "An invented frumious limitation.").kind == "assertion"
    assert by_text(now, "A micro-sign appears in the invented text.").trust == "paper_states"
    assert by_text(now, "Widget units are stable over time.").hidden is False
    # a matched claim the researcher did not edit takes the new wording
    assert by_text(now, "Do invented widgets grow in a made-up market?").trust == "paper_states"
    # the re-keyed ops carry fingerprints of the new agent claims, for the next replay
    edit_op = next(o for o in v.ops if o["op"] == "edit")
    assert edit_op["target"] == {"claim_id": new_m, "fingerprint": overlay.fingerprint(lib.conn, new_m)}
    assert v.researcher_notes is None


def test_nothing_changes_before_accept_and_revert_undoes_an_accept(lib, paper, v1, edited):
    save_raw(lib, paper, raw_v2())
    plan = replay.plan_replay(lib, paper.paper_id)
    replay.accept(lib, paper.paper_id, plan)
    overlay.revert(lib, paper.paper_id, 3, expected_version=4)
    now = interpretation.current(lib, paper.paper_id, "researcher")
    assert now.note_id == v1.note_id
    assert by_text(now, "A quuxifying tally.").claim_id == edited["method"]


def test_researcher_notes_carry_over(lib, paper, v1, edited):
    overlay.save_notes(lib, paper.paper_id, "Invented margin note.", expected_version=3)
    save_raw(lib, paper, raw_v2())
    v = replay.accept(lib, paper.paper_id, replay.plan_replay(lib, paper.paper_id))
    assert v.researcher_notes == "Invented margin note."


# --- refusals --------------------------------------------------------------------------------

def test_no_plan_without_an_overlay_or_without_a_newer_extraction(lib, paper, v1):
    with pytest.raises(replay.ReplayError):
        replay.plan_replay(lib, paper.paper_id)
    edit(lib, paper, claim_id(lib, v1.note_id, "method_or_model[0]"), 0, claim_text="Invented.")
    with pytest.raises(replay.ReplayError):
        replay.plan_replay(lib, paper.paper_id)


def test_a_plan_made_before_another_edit_is_stale(lib, paper, v1, edited):
    save_raw(lib, paper, raw_v2())
    plan = replay.plan_replay(lib, paper.paper_id)
    overlay.save_notes(lib, paper.paper_id, "later", expected_version=3)
    with pytest.raises(replay.ReplayStale):
        replay.accept(lib, paper.paper_id, plan)
    assert overlay.head_version_no(lib.conn, paper.paper_id) == 4


def test_a_plan_made_before_a_further_reextraction_is_stale(lib, paper, v1, edited):
    save_raw(lib, paper, raw_v2())
    plan = replay.plan_replay(lib, paper.paper_id)
    save_raw(lib, paper, raw_v2(method_text="Yet another invented wording."))
    snapshot = dump(lib)
    with pytest.raises(replay.ReplayStale):
        replay.accept(lib, paper.paper_id, plan)
    assert dump(lib) == snapshot


def test_a_plan_for_another_paper_is_refused(lib, paper, v1, edited):
    save_raw(lib, paper, raw_v2())
    plan = replay.plan_replay(lib, paper.paper_id)
    with pytest.raises(replay.ReplayError):
        replay.accept(lib, "lp_someotherpaper0001", plan)


# --- new agent claims ------------------------------------------------------------------------

def test_dismissed_new_claims_become_hides(lib, paper, v1, edited):
    v2 = save_raw(lib, paper, raw_v2())
    plan = replay.plan_replay(lib, paper.paper_id)
    stable = claim_id(lib, v2.note_id, "assumptions[0]")
    v = replay.accept(lib, paper.paper_id, plan, dismiss=[stable])
    hides = {o["target"]["claim_id"] for o in v.ops if o["op"] == "hide"}
    assert stable in hides
    now = interpretation.current(lib, paper.paper_id, "researcher")
    assert all(c.claim_id != stable for c in now.claims)


def test_accepting_only_some_new_claims_hides_the_rest(lib, paper, v1, edited):
    v2 = save_raw(lib, paper, raw_v2())
    plan = replay.plan_replay(lib, paper.paper_id)
    result = claim_id(lib, v2.note_id, "main_results[0]")
    stable = claim_id(lib, v2.note_id, "assumptions[0]")
    v = replay.accept(lib, paper.paper_id, plan, accept_new=[result])
    hides = {o["target"]["claim_id"] for o in v.ops if o["op"] == "hide"}
    assert stable in hides and result not in hides


def test_dismissing_a_claim_that_is_not_in_the_new_extraction_is_refused(lib, paper, v1, edited):
    save_raw(lib, paper, raw_v2())
    plan = replay.plan_replay(lib, paper.paper_id)
    with pytest.raises(replay.ReplayError):
        replay.accept(lib, paper.paper_id, plan, dismiss=[edited["method"]])


# --- unmatched and ambiguous edits -----------------------------------------------------------

def test_an_unmatched_edit_is_kept_as_a_quote_backed_researcher_claim_by_default(lib, paper, v1, edited):
    # the method claim disappears from the new extraction; its quote still anchors
    raw = raw_v2()
    raw["method_or_model"] = []
    save_raw(lib, paper, raw)
    plan = replay.plan_replay(lib, paper.paper_id)
    op = op_by_key(plan, f"edit:{edited['method']}")
    assert op.status == "unmatched" and op.keep_as == "quote"
    replay.accept(lib, paper.paper_id, plan)
    now = interpretation.current(lib, paper.paper_id, "researcher")
    kept = by_text(now, "A quuxifying tally.")
    assert kept.kind == "quote" and kept.trust == "researcher" and kept.claim_id.startswith("rc_")
    assert kept.normalized_label == "widget estimator" and kept.claim_type == "method"
    span = lib.conn.execute("SELECT exact_quote FROM evidence_spans WHERE span_id = ?",
                            (kept.span_id,)).fetchone()[0]
    assert span == METHOD_QUOTE


def test_an_unmatched_edit_whose_quote_no_longer_anchors_is_kept_as_an_assertion(lib, paper, v1):
    old_rq = claim_id(lib, v1.note_id, "assumptions[1]")  # unanchored 'exact counting'
    edit(lib, paper, old_rq, 0, claim_text="Counting is invented and exact.")
    save_raw(lib, paper, raw_v2(exact_counting=False))
    plan = replay.plan_replay(lib, paper.paper_id)
    op = op_by_key(plan, f"edit:{old_rq}")
    assert (op.status, op.keep_as) == ("unmatched", "assertion")
    replay.accept(lib, paper.paper_id, plan)
    kept = by_text(interpretation.current(lib, paper.paper_id, "researcher"), "Counting is invented and exact.")
    assert kept.kind == "assertion" and kept.normalized_label == "exact counting"


def test_an_unmatched_edit_can_be_discarded_or_attached(lib, paper, v1, edited):
    raw = raw_v2()
    raw["method_or_model"] = []
    v2 = save_raw(lib, paper, raw)
    plan = replay.plan_replay(lib, paper.paper_id)
    key = f"edit:{edited['method']}"
    v = replay.accept(lib, paper.paper_id, plan, discard=[key])
    assert not any(o.get("claim_text") == "A quuxifying tally." for o in v.ops)
    overlay.revert(lib, paper.paper_id, 3, expected_version=4)

    plan = replay.plan_replay(lib, paper.paper_id)
    stable = claim_id(lib, v2.note_id, "assumptions[0]")
    v = replay.accept(lib, paper.paper_id, plan, attach={edited["method"]: stable})
    [e] = [o for o in v.ops if o["op"] == "edit"]
    assert e["target"]["claim_id"] == stable and e["set"] == {"claim_text": "A quuxifying tally."}


def test_a_hidden_claim_that_is_gone_is_reported_and_not_re_added(lib, paper, v1, edited):
    edit(lib, paper, edited["exact_counting"], 3, claim_text="Invented exact counts.")
    save_raw(lib, paper, raw_v2(exact_counting=False))
    plan = replay.plan_replay(lib, paper.paper_id)
    assert op_by_key(plan, f"hide:{edited['exact_counting']}").status == "hide_gone"
    assert op_by_key(plan, f"edit:{edited['exact_counting']}").status == "hide_gone"
    assert plan.hides_gone == 1
    v = replay.accept(lib, paper.paper_id, plan)
    assert not any(o.get("claim_text") == "Invented exact counts." for o in v.ops)


def test_an_ambiguous_match_is_flagged_and_needs_a_choice(lib, paper, v1, edited):
    twin = {"claim_text": "Widgets are counted without any error.", "normalized_label": "exact counting",
            "status": "inferred", "assumption_type": "general", "inferred_explanation": "another reason"}
    v2 = save_raw(lib, paper, raw_v2(extra_assumptions=[twin]))
    plan = replay.plan_replay(lib, paper.paper_id)
    op = op_by_key(plan, f"hide:{edited['exact_counting']}")
    a, b = claim_id(lib, v2.note_id, "assumptions[2]"), claim_id(lib, v2.note_id, "assumptions[3]")
    assert op.status == "ambiguous" and set(op.candidates) == {a, b}
    assert a not in {c.claim_id for c in plan.new_claims}
    v = replay.accept(lib, paper.paper_id, plan, attach={edited["exact_counting"]: b})
    hides = {o["target"]["claim_id"] for o in v.ops if o["op"] == "hide"}
    assert b in hides and a not in hides


def test_an_undecided_ambiguous_edit_is_kept_unless_the_claim_was_hidden(lib, paper, v1, edited):
    edit(lib, paper, edited["exact_counting"], 3, claim_text="Invented exact counts.")
    twin = {"claim_text": "Widgets are counted without any error.", "normalized_label": "exact counting",
            "status": "inferred", "assumption_type": "general", "inferred_explanation": "another reason"}
    save_raw(lib, paper, raw_v2(extra_assumptions=[twin]))
    plan = replay.plan_replay(lib, paper.paper_id)
    assert op_by_key(plan, f"edit:{edited['exact_counting']}").status == "ambiguous"
    v = replay.accept(lib, paper.paper_id, plan)  # hidden and undecided: neither re-hidden nor re-added
    assert not any(o.get("claim_text") == "Invented exact counts." for o in v.ops)
    overlay.revert(lib, paper.paper_id, 4, expected_version=5)
    plan = replay.plan_replay(lib, paper.paper_id)
    v = replay.accept(lib, paper.paper_id, plan, discard=[f"hide:{edited['exact_counting']}"])
    [kept] = [o for o in v.ops if o.get("claim_text") == "Invented exact counts."]
    assert kept["op"] == "add" and kept["kind"] == "assertion"


def test_a_retyped_claim_matches_on_the_agents_original_type(lib, paper, v1):
    ident = claim_id(lib, v1.note_id, "assumptions[0]")  # identification_assumption
    edit(lib, paper, ident, 0, claim_type="general_assumption")
    decoy = {"claim_text": "Units do not anticipate the invented programme.", "normalized_label": "no anticipation",
             "status": "found", "assumption_type": "general", "exact_quote": ASSUMPTION_QUOTE}
    v2 = save_raw(lib, paper, raw_v2(extra_assumptions=[decoy]))
    plan = replay.plan_replay(lib, paper.paper_id)
    op = op_by_key(plan, f"edit:{ident}")
    assert op.status == "kept" and op.new_claim_id == claim_id(lib, v2.note_id, "assumptions[1]")
    v = replay.accept(lib, paper.paper_id, plan)
    now = interpretation.current(lib, paper.paper_id, "researcher")
    retyped = next(c for c in now.claims if c.claim_id == op.new_claim_id)
    assert retyped.claim_type == "general_assumption" and retyped.original.claim_type == "identification_assumption"
    assert v.ops[0]["set"] == {"claim_type": "general_assumption"}


# --- cross-markdown --------------------------------------------------------------------------

def test_a_reextraction_on_another_markdown_re_anchors_old_quotes_through_the_text(lib, paper, v1, edited):
    # the new conversion adds a preface (all offsets shift) and re-wraps the method sentence
    new_text = ("Preface added by the invented re-conversion.\n\n"
                + paper.text.replace(METHOD_QUOTE, METHOD_QUOTE.replace(" counts ", "\ncounts ")))
    with lib.write() as conn:
        md_b, hash_b = add_markdown_file(lib, conn, paper.source_file_id, new_text, 2)
        conn.execute("UPDATE paper_files SET markdown_id = ? WHERE paper_id = ?", (md_b, paper.paper_id))
    paper_b = dataclasses.replace(paper, markdown_id=md_b, markdown_hash=hash_b, text=new_text)
    wrapped = METHOD_QUOTE.replace(" counts ", "\ncounts ")
    # a new label, so this is tier 2: the relocated quotes overlap and the texts are similar
    v2 = save_raw(lib, paper_b, raw_v2(method_quote=wrapped, method_label="double counting",
                                       method_text="A two-step widget estimator, double counting."))

    plan = replay.plan_replay(lib, paper.paper_id)
    assert plan.new_markdown_id == md_b
    op = op_by_key(plan, f"edit:{edited['method']}")
    assert op.status == "kept" and op.new_claim_id == claim_id(lib, v2.note_id, "method_or_model[0]")
    replay.accept(lib, paper.paper_id, plan)
    now = interpretation.current(lib, paper.paper_id, "researcher")
    assert by_text(now, "A quuxifying tally.").claim_id == op.new_claim_id


def test_a_quote_backed_add_is_relocated_onto_the_new_markdown(lib, paper, v1):
    start = paper.text.index(RESULT_QUOTE)
    overlay.save_edit(lib, paper.paper_id, {
        "op": "add", "kind": "quote", "claim_type": "result", "normalized_label": "micro sign",
        "claim_text": "Researcher reading of the micro-sign.",
        "quote": {"markdown_id": paper.markdown_id, "start_char": start, "end_char": start + len(RESULT_QUOTE)}},
        expected_version=0)
    new_text = "Preface added by the invented re-conversion.\n\n" + paper.text
    with lib.write() as conn:
        md_b, hash_b = add_markdown_file(lib, conn, paper.source_file_id, new_text, 2)
        conn.execute("UPDATE paper_files SET markdown_id = ? WHERE paper_id = ?", (md_b, paper.paper_id))
    save_raw(lib, dataclasses.replace(paper, markdown_id=md_b, markdown_hash=hash_b, text=new_text),
             raw_v2(result=False))
    plan = replay.plan_replay(lib, paper.paper_id)
    [add] = [o for o in plan.ops if o.op["op"] == "add"]
    assert add.status == "added_kept" and add.quote["markdown_id"] == md_b
    replay.accept(lib, paper.paper_id, plan)
    kept = by_text(interpretation.current(lib, paper.paper_id, "researcher"), "Researcher reading of the micro-sign.")
    row = lib.conn.execute("SELECT markdown_id, exact_quote FROM evidence_spans WHERE span_id = ?",
                           (kept.span_id,)).fetchone()
    assert tuple(row) == (md_b, RESULT_QUOTE)


def test_a_quote_backed_add_that_no_longer_anchors_becomes_an_assertion(lib, paper, v1):
    start = paper.text.index(RESULT_QUOTE)
    overlay.save_edit(lib, paper.paper_id, {
        "op": "add", "kind": "quote", "claim_type": "result", "normalized_label": "micro sign",
        "claim_text": "Researcher reading of the micro-sign.",
        "quote": {"markdown_id": paper.markdown_id, "start_char": start, "end_char": start + len(RESULT_QUOTE)}},
        expected_version=0)
    new_text = paper.text.replace(RESULT_QUOTE, "A sentence the invented re-conversion rewrote.")
    with lib.write() as conn:
        md_b, hash_b = add_markdown_file(lib, conn, paper.source_file_id, new_text, 2)
        conn.execute("UPDATE paper_files SET markdown_id = ? WHERE paper_id = ?", (md_b, paper.paper_id))
    save_raw(lib, dataclasses.replace(paper, markdown_id=md_b, markdown_hash=hash_b, text=new_text),
             raw_v2(result=False))
    plan = replay.plan_replay(lib, paper.paper_id)
    [add] = [o for o in plan.ops if o.op["op"] == "add"]
    assert (add.status, add.keep_as) == ("added_unanchored", "assertion")
    replay.accept(lib, paper.paper_id, plan)
    kept = by_text(interpretation.current(lib, paper.paper_id, "researcher"), "Researcher reading of the micro-sign.")
    assert kept.kind == "assertion" and kept.span_id is None
