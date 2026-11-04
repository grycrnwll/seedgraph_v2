"""Recording an answer (stage 5 design §3.3, §3.4, §3.6; slice R2): the citation check and its
flags, refused ids, records with no events, helpers, frozen evidence that survives a later
edit, and the graphs of a record. Invented papers and textbook terms only."""

from __future__ import annotations

import json

import pytest

from seedgraph.records import answer as A
from seedgraph.records import log, sessions

from ._world import Agent, claim_hit, span_hit, standard


@pytest.fixture
def lib(tmp_path):
    from seedgraph.library import store

    lib = store.open_library_at(tmp_path / "library")
    yield lib
    lib.close()


@pytest.fixture
def world(lib):
    return standard(lib)


def records(lib):
    return [dict(r) for r in lib.conn.execute("SELECT * FROM traversal_records ORDER BY opened_at, record_id")]


def record(lib, record_id):
    return dict(lib.conn.execute("SELECT * FROM traversal_records WHERE record_id = ?", (record_id,)).fetchone())


def events_of(lib, record_id):
    return [dict(r) for r in lib.conn.execute("SELECT * FROM traversal_events WHERE record_id = ? ORDER BY seq",
                                              (record_id,))]


def graphs_of(lib, record_id):
    return sorted(tuple(r) for r in lib.conn.execute("SELECT slug, how FROM record_graphs WHERE record_id = ?",
                                                     (record_id,)))


# --- the citation check -----------------------------------------------------------------------

def test_a_cited_passage_that_was_read_passes(lib, world):
    w, did, _ = world
    sp = w.span(did, "parallel")
    a = Agent(lib)
    a.search(hits=[span_hit(sp, did)])
    a.paper_read(did, *did.at("parallel"))
    res = a.answer([{"span_id": sp, "why": "states the assumption"}])
    assert res.flags == () and res.next_action is None
    assert res.counts == {"searches": 1, "opened": 1, "cited": 1}
    assert res.page == f"/records/{res.record_id}"
    [ev] = res.evidence
    assert ev["check"] == "opened_here" and ev["check_label"] == "opened in this record" and ev["flag"] is None


def test_each_unread_citation_is_flagged(lib, world):
    w, did, ate = world
    read_sp, hit_sp, anchor_sp = w.span(did, "parallel"), w.span(did, "serial"), w.span(ate, "matching")
    never_sp = w.span(ate, "unconfounded")
    a = Agent(lib)
    a.search(hits=[span_hit(hit_sp, did)])
    a.paper_read(did, *did.at("parallel"))
    a.paper_show(ate, [ate.claims["method_or_model[0]"]], anchors={ate.claims["method_or_model[0]"]: [anchor_sp]})
    res = a.answer([{"span_id": read_sp}, {"span_id": hit_sp}, {"span_id": anchor_sp}, {"span_id": never_sp}])
    assert [f["flag"] for f in res.flags] == [
        "cited without being read (seen in search results only)", "cited without being read",
        "cited without being read"]
    assert [f["evidence"] for f in res.flags] == [1, 2, 3]
    assert res.next_action == ("Tell the researcher which citations were not read, or read them and record "
                               "again.")
    stored = record(lib, res.record_id)
    assert json.loads(stored["flags_json"]) == list(res.flags)
    assert [e["check"] for e in json.loads(stored["evidence_json"])] == [
        "opened_here", "found_only", "not_returned", "not_returned"]


def test_a_claim_is_read_when_paper_show_returned_it(lib, world):
    w, did, _ = world
    c = did.claims["assumptions[0]"]
    a = Agent(lib)
    a.paper_show(did)
    res = a.answer([{"claim_id": c}])
    assert res.flags == ()


def test_a_claim_seen_only_as_a_search_hit_is_flagged(lib, world):
    w, did, _ = world
    c = did.claims["assumptions[0]"]
    a = Agent(lib)
    a.search(kind="claims", hits=[claim_hit(c, did)])
    res = a.answer([{"claim_id": c}])
    assert [f["flag"] for f in res.flags] == ["cited without being read (seen in search results only)"]


def test_a_claim_is_read_when_its_anchor_passage_was_read(lib, world):
    w, did, _ = world
    a = Agent(lib)
    a.paper_read(did, *did.at("parallel"))
    assert a.answer([{"claim_id": did.claims["assumptions[0]"]}]).flags == ()


def test_a_paper_is_read_when_it_was_opened(lib, world):
    w, did, ate = world
    a = Agent(lib)
    a.paper_show(did)
    a.search(kind="papers", hits=[{"kind": "paper", "id": ate.paper_id}])
    res = a.answer([{"paper_id": did.paper_id, "page": 2}, {"paper_id": ate.paper_id}])
    assert [f["evidence"] for f in res.flags] == [1]
    first = res.evidence[0]
    assert (first["title"], first["year"], first["page"]) == ("Invented county minimum wages", 2019, 2)


def test_unknown_ids_are_refused_and_nothing_is_written(lib, world):
    w, did, _ = world
    a = Agent(lib)
    a.search()
    before = records(lib)
    with pytest.raises(A.UnknownEvidence) as exc:
        a.answer([{"span_id": "span_nope"}, {"claim_id": "c_nope"}, {"paper_id": "lp_nope"},
                  {"paper_id": did.paper_id}])
    assert exc.value.unknown == ("span_nope", "c_nope", "lp_nope")
    assert exc.value.code == "unknown_evidence" and "span_nope" in str(exc.value)
    assert exc.value.next_action
    assert records(lib) == before


@pytest.mark.parametrize("evidence, why", [
    ([], "1 to 100"),
    ([{"paper_id": "lp_x"}] * 101, "1 to 100"),
    ([{"span_id": "a", "claim_id": "b"}], "exactly one"),
    ([{}], "exactly one"),
    (["sp_1"], "object"),
    ([{"span_id": "a", "page": 2}], "page"),
    ([{"paper_id": "a", "page": "2"}], "page"),
    ([{"paper_id": "a", "page": 0}], "page"),
    ([{"paper_id": "a", "why": 3}], "why"),
    ([{"paper_id": "a", "colour": "red"}], "colour"),
    ([{"paper_id": ""}], "exactly one"),
])
def test_malformed_evidence_is_refused(lib, world, evidence, why):
    with pytest.raises(A.RecordAnswerError) as exc:
        Agent(lib).answer(evidence)
    assert why in str(exc.value) and exc.value.next_action


@pytest.mark.parametrize("field, value, why", [
    ("question", "", "question"), ("question", "x" * 2001, "2000"), ("answer", " ", "answer"),
    ("answer", "x" * 20001, "20000"), ("judgement", "x" * 5001, "5000"),
])
def test_text_limits_are_enforced(lib, world, field, value, why):
    w, did, _ = world
    kw = {"question": "Is the ATE identified?", "answer": "Yes.", "judgement": None, field: value}
    with pytest.raises(A.RecordAnswerError) as exc:
        Agent(lib).answer([{"paper_id": did.paper_id}], **kw)
    assert why in str(exc.value)


def test_an_invalid_session_name_is_refused(lib, world):
    w, did, _ = world
    with pytest.raises(sessions.SessionNameError):
        Agent(lib, name="Bad Name").answer([{"paper_id": did.paper_id}])


# --- what the record holds --------------------------------------------------------------------

def test_the_open_record_is_answered_and_owns_its_events(lib, world):
    w, did, _ = world
    a = Agent(lib)
    a.search()
    a.paper_show(did)
    [open_rec] = records(lib)
    res = a.answer([{"paper_id": did.paper_id}], question="Is parallel trends plausible?",
                   answer="The authors argue so.", judgement="I find the pre-trends thin.")
    assert res.record_id == open_rec["record_id"]
    rec = record(lib, res.record_id)
    assert rec["state"] == "answered" and rec["answered_at"] == "2026-11-04T14:35:05.000000+00:00"
    assert (rec["question"], rec["answer"], rec["judgement"]) == (
        "Is parallel trends plausible?", "The authors argue so.", "I find the pre-trends thin.")
    assert json.loads(rec["counts_json"]) == {"searches": 1, "opened": 1, "cited": 1}
    assert len(events_of(lib, res.record_id)) == 2
    a.search()
    assert [r["state"] for r in records(lib)] == ["answered", "open"]


def test_a_record_with_no_events_passes_on_earlier_reads(lib, world):
    w, did, _ = world
    sp = w.span(did, "parallel")
    a = Agent(lib, name="parallel-trends-1432")
    a.paper_read(did, *did.at("parallel"))
    first = a.answer([{"span_id": sp}])
    follow = a.answer([{"span_id": sp}], question="And the follow-up?")
    assert follow.record_id != first.record_id and follow.flags == ()
    assert follow.evidence[0]["check_label"] == "opened earlier in this session"
    assert follow.counts == {"searches": 0, "opened": 0, "cited": 1}
    rec = record(lib, follow.record_id)
    assert events_of(lib, follow.record_id) == [] and rec["state"] == "answered"
    assert rec["opened_at"] == rec["last_event_at"] == rec["answered_at"]


def test_another_session_s_reads_do_not_count(lib, world):
    w, did, _ = world
    sp = w.span(did, "parallel")
    Agent(lib, name="ate-review").paper_read(did, *did.at("parallel"))
    res = Agent(lib, name="parallel-trends-1432").answer([{"span_id": sp}])
    assert [f["flag"] for f in res.flags] == ["cited without being read"]


def test_recording_again_owns_only_the_new_reads_and_the_first_stays_flagged(lib, world):
    w, did, _ = world
    sp = w.span(did, "parallel")
    a = Agent(lib)
    a.search(hits=[span_hit(sp, did)])
    first = a.answer([{"span_id": sp}])
    a.paper_read(did, *did.at("parallel"))
    second = a.answer([{"span_id": sp}])
    assert len(first.flags) == 1 and second.flags == ()
    assert [e["tool"] for e in events_of(lib, second.record_id)] == ["paper_read"]
    assert json.loads(record(lib, first.record_id)["flags_json"]) == list(first.flags)


# --- helpers ----------------------------------------------------------------------------------

def test_helper_reads_count_for_the_parent_and_are_tagged(lib, world):
    w, did, _ = world
    sp = w.span(did, "parallel")
    parent = Agent(lib, name="parallel-trends-1432")
    helper = Agent(lib, name="parallel-trends-1432/a")
    parent.search()
    helper.paper_read(did, *did.at("parallel"))
    res = parent.answer([{"span_id": sp}])
    assert res.flags == ()
    assert res.evidence[0]["opened_by"] == ["a"]
    assert {e["session_id"] for e in events_of(lib, res.record_id)} == {
        sessions.named_session_id("claude", "parallel-trends-1432"),
        sessions.named_session_id("claude", "parallel-trends-1432/a")}


def test_a_helper_that_answers_takes_its_own_events_into_its_own_record(lib, world):
    w, did, _ = world
    sp = w.span(did, "parallel")
    parent = Agent(lib, name="parallel-trends-1432")
    helper = Agent(lib, name="parallel-trends-1432/a")
    parent.search()
    helper.search(hits=[span_hit(sp, did)])
    helper.paper_read(did, *did.at("parallel"))
    [parent_open] = records(lib)
    res = helper.answer([{"span_id": sp}])
    assert res.record_id != parent_open["record_id"] and res.flags == ()
    rec = record(lib, res.record_id)
    assert rec["session_id"] == sessions.named_session_id("claude", "parallel-trends-1432/a")
    assert [e["tool"] for e in events_of(lib, res.record_id)] == ["search", "paper_read"]
    assert rec["opened_at"] == events_of(lib, res.record_id)[0]["at"]
    assert rec["last_event_at"] == events_of(lib, res.record_id)[-1]["at"]
    left = events_of(lib, parent_open["record_id"])
    assert [e["tool"] for e in left] == ["search"]
    assert record(lib, parent_open["record_id"])["last_event_at"] == left[-1]["at"]


def test_no_helper_record_is_left_as_a_path_without_answer(lib, world):
    w, did, _ = world
    sp = w.span(did, "parallel")
    helper = Agent(lib, name="parallel-trends-1432/a")
    helper.paper_read(did, *did.at("parallel"))
    res = helper.answer([{"span_id": sp}])
    assert [(r["record_id"], r["state"]) for r in records(lib)] == [(res.record_id, "answered")]
    parent = sessions.named_session_id("claude", "parallel-trends-1432")
    assert log.open_record(lib.conn, parent) is None


def test_a_helper_answering_keeps_the_parent_s_unlogged_count(lib, world):
    w, did, _ = world
    helper = Agent(lib, name="parallel-trends-1432/a")
    helper.search()
    with lib.write() as conn:
        conn.execute("UPDATE traversal_records SET unlogged_calls = 2")
    res = helper.answer([{"paper_id": did.paper_id}])
    assert record(lib, res.record_id)["unlogged_calls"] == 2


def test_after_a_helper_answers_its_next_call_joins_the_parent_again(lib, world):
    w, did, _ = world
    helper = Agent(lib, name="parallel-trends-1432/a")
    helper.search()
    first = helper.answer([{"paper_id": did.paper_id}])
    helper.search()
    parent = sessions.named_session_id("claude", "parallel-trends-1432")
    assert log.open_record(lib.conn, parent) not in (None, first.record_id)


# --- frozen evidence --------------------------------------------------------------------------

def test_a_span_is_frozen_with_its_quote_page_and_paper(lib, world):
    w, did, _ = world
    sp = w.span(did, "did_result")
    a = Agent(lib)
    a.paper_read(did, *did.at("did_result"))
    [ev] = a.answer([{"span_id": sp, "why": "the headline estimate"}]).evidence
    assert ev["kind"] == "span" and ev["id"] == sp and ev["paper_id"] == did.paper_id
    assert ev["quote"] == "The estimated ATE on teen employment is small and not significant."
    assert (ev["page_start"], ev["page_end"]) == (2, 2)
    assert (ev["title"], ev["year"], ev["why"]) == ("Invented county minimum wages", 2019, "the headline estimate")


def test_a_claim_is_frozen_as_the_agent_saw_it(lib, world):
    w, did, _ = world
    c = did.claims["assumptions[0]"]
    a = Agent(lib)
    a.paper_show(did)
    [ev] = a.answer([{"claim_id": c}]).evidence
    assert (ev["claim_type"], ev["label"], ev["text"]) == (
        "identification_assumption", "parallel trends", "Parallel trends between treated and control counties.")
    assert ev["trust"] == "paper_states" and ev["trust_label"] == "paper states"
    assert ev["overlay_id"] is None and ev["version_no"] is None and ev["current"] is True
    assert ev["quote"].startswith("Treated and control counties") and ev["page_start"] == 1
    assert ev["span_id"] == w.span(did, "parallel")


def test_frozen_claim_text_and_version_survive_a_later_edit(lib, world):
    w, did, _ = world
    c = did.claims["assumptions[0]"]
    w.edit(did, c, 0, claim_text="Parallel pre-trends in employment.")
    a = Agent(lib)
    a.paper_show(did)
    res = a.answer([{"claim_id": c}])
    [ev] = res.evidence
    assert ev["text"] == "Parallel pre-trends in employment." and ev["trust"] == "researcher"
    assert ev["trust_note"] == "edited from the agent's claim"
    overlay_id = lib.conn.execute("SELECT overlay_id FROM note_overlays WHERE paper_id = ?",
                                  (did.paper_id,)).fetchone()[0]
    assert (ev["overlay_id"], ev["version_no"]) == (overlay_id, 1)
    w.edit(did, c, 1, claim_text="Something else entirely.")
    stored = json.loads(record(lib, res.record_id)["evidence_json"])
    assert stored[0]["text"] == "Parallel pre-trends in employment." and stored[0]["version_no"] == 1


def test_a_hidden_claim_is_not_refused_and_is_frozen_from_the_agent_s_original(lib, world):
    from seedgraph.notes import overlay

    w, did, _ = world
    c = did.claims["assumptions[1]"]
    overlay.save_edit(lib, did.paper_id, {"op": "hide", "target": {"claim_id": c}}, expected_version=0)
    a = Agent(lib)
    a.search(kind="claims", hits=[claim_hit(c, did)])
    [ev] = a.answer([{"claim_id": c}]).evidence
    assert ev["current"] is False and ev["text"] == "No spillovers across county lines."
    assert ev["note"] == "not in the paper's current interpretation"


# --- graphs of a record -----------------------------------------------------------------------

def test_graphs_named_in_the_record_s_events_are_searched(lib, world):
    w, did, ate = world
    w.graph("did-reading", did)
    w.graph("matching", ate)
    a = Agent(lib)
    a.search(graph="did-reading")
    a.search(graph="matching")
    a.search()
    res = a.answer([{"paper_id": ate.paper_id}])
    assert graphs_of(lib, res.record_id) == [("did-reading", "searched"), ("matching", "searched")]
    assert res.graphs == (("did-reading", "searched"), ("matching", "searched"))


def test_a_library_wide_record_links_the_graphs_holding_a_cited_paper(lib, world):
    w, did, ate = world
    w.graph("did-reading", did)
    w.graph("both", did, ate)
    w.graph("other", ate)
    a = Agent(lib)
    a.paper_show(did)
    res = a.answer([{"paper_id": did.paper_id}])
    assert graphs_of(lib, res.record_id) == [("both", "cited_member"), ("did-reading", "cited_member")]


def test_a_record_citing_no_graph_member_has_no_graph(lib, world):
    w, did, ate = world
    w.graph("matching", ate)
    a = Agent(lib)
    a.paper_show(did)
    res = a.answer([{"paper_id": did.paper_id}])
    assert graphs_of(lib, res.record_id) == [] and res.graphs == ()


def test_a_claim_or_span_cites_its_paper_for_membership(lib, world):
    w, did, ate = world
    w.graph("matching", ate)
    a = Agent(lib)
    a.paper_read(ate, *ate.at("matching"))
    res = a.answer([{"span_id": w.span(ate, "matching")}])
    assert graphs_of(lib, res.record_id) == [("matching", "cited_member")]


# --- session notes ----------------------------------------------------------------------------

def test_a_name_used_from_two_live_connections_is_flagged_on_the_record(lib, world):
    w, did, _ = world
    one = Agent(lib, "mcp-aaaa0001", name="parallel-trends-1432")
    two = Agent(lib, "mcp-bbbb0002", name="parallel-trends-1432")
    one.search()
    two.paper_show(did)
    res = two.answer([{"paper_id": did.paper_id}])
    assert res.flags == ({"kind": "session", "flag": "session name used from two connections"},)
    assert res.next_action is None  # nothing to re-read: the record notes it


def test_the_reply_carries_the_session_name(lib, world):
    w, did, _ = world
    res = Agent(lib, name="parallel-trends-1432").answer([{"paper_id": did.paper_id}])
    assert res.session == "parallel-trends-1432"
    unnamed = Agent(lib, "mcp-cccc0003").answer([{"paper_id": did.paper_id}])
    assert unnamed.session == "2026-11-04 14:33"
    reply = unnamed.to_reply()
    assert set(reply) == {"record_id", "session", "counts", "flags", "page", "next_action"}
