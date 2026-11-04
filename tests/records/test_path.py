"""The path of a record (stage 5 design §3.3, §3.5; slice R2): the found / opened / cited rules
as a pure function over events and evidence, the half-span overlap, anchors that count as
nothing, the citation check, gaps, and "in progress" vs "path without answer". Invented ids."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from seedgraph.records import path as P

T0 = datetime(2026, 11, 4, 14, 32, 5, tzinfo=timezone.utc)


def ts(minutes: float = 0) -> str:
    return (T0 + timedelta(minutes=minutes)).isoformat(timespec="microseconds")


def ev(seq, kind, items=(), *, tool=None, args=None, minutes=None, record_id="tr_a", session_id="ss_p",
       helper=None):
    tool = tool or {"search": "search", "open_paper": "paper_show", "open_text": "paper_read",
                    "open_concept": "concept_show"}[kind]
    return P.Event(seq=seq, at=ts(seq if minutes is None else minutes), tool=tool, kind=kind,
                   session_id=session_id, connection_id="mcp-1", args=args or {}, items=list(items),
                   record_id=record_id, helper=helper)


def span_hit(span_id, paper_id="lp_did"):
    return {"kind": "span", "id": span_id, "paper_id": paper_id, "preview": "x"}


def read(seq, start, end, *, md="md_did", paper_id="lp_did", **kw):
    return ev(seq, "open_text", [{"paper_id": paper_id, "markdown_id": md, "start": start, "end": end}],
              args={"paper_id": paper_id, "start": start, "end": end}, **kw)


LOOKUP = P.Lookup(
    spans={"sp_1": P.SpanPlace("md_did", 100, 200, "lp_did"), "sp_2": P.SpanPlace("md_did", 300, 340, "lp_did"),
           "sp_z": P.SpanPlace("md_did", 500, 500, "lp_did"), "sp_x": P.SpanPlace("md_ate", 0, 50, "lp_ate")},
    anchors={"c_pt": ("sp_1",), "c_sc": ("sp_2",)},
    claim_papers={"c_pt": "lp_did", "c_sc": "lp_did"})


def tags(events, evidence=(), lookup=LOOKUP):
    return {(i.kind, i.id): i.tag for i in P.path(events, evidence, lookup=lookup).items}


def levels(events, lookup=LOOKUP):
    return {k: s.level for k, s in P.seen(events, lookup).items()}


# --- found ------------------------------------------------------------------------------------

def test_a_search_hit_is_found_and_so_is_its_paper():
    got = levels([ev(1, "search", [span_hit("sp_1")])], P.Lookup())
    assert got == {("span", "sp_1"): "found", ("paper", "lp_did"): "found"}


def test_claim_paper_and_concept_hits_are_found():
    got = levels([ev(1, "search", [{"kind": "claim", "id": "c_pt", "paper_id": "lp_did"},
                                   {"kind": "paper", "id": "lp_ate"},
                                   {"kind": "concept", "id": "parallel-trends", "claim_ids": ["c_sc"],
                                    "paper_ids": ["lp_x"]}], tool="concept_search")])
    assert got == {("claim", "c_pt"): "found", ("paper", "lp_did"): "found", ("paper", "lp_ate"): "found",
                   ("claim", "c_sc"): "found", ("paper", "lp_x"): "found"}


# --- opened -----------------------------------------------------------------------------------

def test_paper_show_opens_the_paper_and_its_claims_but_not_their_anchor_spans():
    got = levels([ev(1, "open_paper", [{"claim_id": "c_pt", "span_ids": ["sp_1"]}], args={"paper_id": "lp_did"})])
    assert got == {("paper", "lp_did"): "opened", ("claim", "c_pt"): "opened"}
    assert ("span", "sp_1") not in got


def test_paper_read_opens_a_span_it_covers_at_least_half_of():
    assert levels([read(1, 150, 400)])[("span", "sp_1")] == "opened"   # exactly half of sp_1
    assert levels([read(1, 0, 149)]).get(("span", "sp_1")) is None      # 49 of 100: not enough
    assert levels([read(1, 151, 400)]).get(("span", "sp_1")) is None    # 49 of 100
    assert levels([read(1, 0, 1000)])[("span", "sp_2")] == "opened"


def test_paper_read_on_another_markdown_opens_nothing_of_this_one():
    got = levels([read(1, 0, 1000, md="md_other")])
    assert ("span", "sp_1") not in got and got[("paper", "lp_did")] == "opened"


def test_a_zero_length_span_is_opened_only_when_inside_the_slice():
    assert levels([read(1, 400, 600)])[("span", "sp_z")] == "opened"
    assert levels([read(1, 0, 499)]).get(("span", "sp_z")) is None


def test_the_returned_range_counts_not_the_requested_one():
    e = ev(1, "open_text", [{"paper_id": "lp_did", "markdown_id": "md_did", "start": 0, "end": 120}],
           args={"paper_id": "lp_did", "start": 0, "end": 1000})
    assert levels([e]).get(("span", "sp_1")) is None


def test_a_claim_is_opened_when_its_anchor_span_is_read():
    assert levels([read(1, 100, 200)])[("claim", "c_pt")] == "opened"


def test_a_claim_is_found_when_its_anchor_span_was_a_hit():
    assert levels([ev(1, "search", [span_hit("sp_2")])])[("claim", "c_sc")] == "found"


def test_concept_show_opens_its_claims_and_their_quoted_spans():
    got = levels([ev(1, "open_concept", [{"claim_id": "c_pt", "span_ids": ["sp_1"]}], args={"handle": "pt"})])
    assert got[("claim", "c_pt")] == "opened" and got[("span", "sp_1")] == "opened"
    assert got.get(("paper", "lp_did")) != "opened"


def test_the_strongest_thing_wins_in_any_order():
    events = [read(1, 100, 200), ev(2, "search", [span_hit("sp_1")])]
    assert levels(events)[("span", "sp_1")] == "opened"


# --- tags in a path ---------------------------------------------------------------------------

def test_items_are_tagged_found_opened_or_cited():
    events = [ev(1, "search", [span_hit("sp_1"), span_hit("sp_2")]), read(2, 100, 200)]
    evidence = [{"kind": "span", "id": "sp_2"}]
    got = tags(events, evidence)
    assert got[("span", "sp_1")] == "opened" and got[("span", "sp_2")] == "cited"
    assert got[("paper", "lp_did")] == "opened"
    labels = {i.id: i.label for i in P.path(events, evidence, lookup=LOOKUP).items}
    assert labels == {"sp_1": "opened but not cited", "sp_2": "cited", "lp_did": "opened but not cited",
                      "c_pt": "opened but not cited", "c_sc": "found but not opened"}


def test_a_cited_item_never_returned_is_listed_after_the_rest():
    p = P.path([ev(1, "search", [span_hit("sp_1")])], [{"kind": "span", "id": "sp_x"}], lookup=LOOKUP)
    assert [(i.id, i.tag, i.opened, i.found) for i in p.items][-1] == ("sp_x", "cited", False, False)


def test_path_counts_searches_opens_and_citations():
    events = [ev(1, "search"), ev(2, "search"), read(3, 0, 10), ev(4, "open_paper", args={"paper_id": "lp_did"})]
    assert P.path(events, [{"kind": "paper", "id": "lp_did"}], lookup=LOOKUP).counts == {
        "searches": 2, "opened": 2, "cited": 1}


def test_a_long_gap_is_shown_in_the_path():
    events = [ev(1, "search", minutes=0), ev(2, "search", minutes=3), ev(3, "search", minutes=43)]
    steps = P.path(events, (), lookup=LOOKUP).steps
    assert [s.gap for s in steps] == [None, None, "40 minutes later"]
    assert P.gap_text(timedelta(minutes=95)) == "1 hour 35 minutes later"


def test_helper_steps_name_the_helper():
    steps = P.path([ev(1, "search", helper="a"), ev(2, "search")], (), lookup=LOOKUP).steps
    assert [s.event.helper for s in steps] == ["a", None]


def test_path_is_pure_and_ordered_by_seq():
    events = [ev(2, "search", [span_hit("sp_2")]), ev(1, "search", [span_hit("sp_1")])]
    first = P.path(events, (), lookup=LOOKUP)
    assert [s.event.seq for s in first.steps] == [1, 2]
    assert first == P.path(list(reversed(events)), (), lookup=LOOKUP)


# --- the citation check -----------------------------------------------------------------------

def check(events, kind, item_id, *, record_id="tr_a", session_id="ss_p"):
    return P.check_citation(P.seen(events, LOOKUP), (kind, item_id), record_id=record_id,
                            session_id=session_id)


def test_cited_and_opened_in_this_record_is_fine():
    c = check([read(1, 100, 200)], "span", "sp_1")
    assert (c.status, c.label, c.flag) == ("opened_here", "opened in this record", None)


def test_cited_and_opened_in_an_earlier_record_of_the_session_is_fine():
    c = check([read(1, 100, 200, record_id="tr_old")], "span", "sp_1")
    assert (c.status, c.label, c.flag) == ("opened_earlier", "opened earlier in this session", None)


def test_cited_and_found_only_is_flagged_seen_in_search_results():
    c = check([ev(1, "search", [span_hit("sp_1")])], "span", "sp_1")
    assert c.status == "found_only" and c.flag == "cited without being read (seen in search results only)"


def test_cited_and_never_returned_is_flagged():
    c = check([read(1, 100, 200)], "span", "sp_x")
    assert c.status == "not_returned" and c.flag == "cited without being read"


def test_a_span_seen_only_as_a_paper_show_anchor_is_cited_without_being_read():
    c = check([ev(1, "open_paper", [{"claim_id": "c_pt", "span_ids": ["sp_1"]}], args={"paper_id": "lp_did"})],
              "span", "sp_1")
    assert c.flag == "cited without being read"


def test_a_passage_a_helper_opened_counts_and_is_tagged():
    c = check([read(1, 100, 200, session_id="ss_p_a", helper="a")], "span", "sp_1")
    assert c.flag is None and c.opened_by == ("a",) and "opened by helper a" in c.notes


def test_own_opening_is_not_tagged_as_a_helper():
    c = check([read(1, 100, 200, session_id="ss_p_a", helper="a")], "span", "sp_1", session_id="ss_p_a")
    assert c.opened_by == () and c.notes == ()


# --- state ------------------------------------------------------------------------------------

def rec(state="open", last=0):
    return {"state": state, "last_event_at": ts(last)}


def test_state_answered():
    assert P.state(rec("answered"), T0 + timedelta(days=9), None) == "answered"


def test_state_in_progress_while_recent_and_live():
    assert P.state(rec(last=0), T0 + timedelta(minutes=29), {"closed_at": None}) == "in_progress"
    assert P.state(rec(last=0), T0 + timedelta(minutes=29), None) == "in_progress"


def test_state_path_without_answer_once_idle():
    assert P.PATH_IDLE == timedelta(minutes=30)
    assert P.state(rec(last=0), T0 + timedelta(minutes=30), {"closed_at": None}) == "path_without_answer"


def test_state_path_without_answer_once_the_connection_closed():
    assert P.state(rec(last=0), T0 + timedelta(minutes=1), {"closed_at": ts(0.5)}) == "path_without_answer"
    assert P.STATE_LABELS["path_without_answer"] == "path without answer"


def test_state_refuses_an_unknown_record_state():
    with pytest.raises(ValueError):
        P.state(rec("weird"), T0, None)
