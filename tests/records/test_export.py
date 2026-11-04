"""Export to markdown (stage 5 design §3.8; slice R2): question, answer, the agent's judgement
apart, cited passages with paper, year, page, trust label and flag, the tagged path, the
record's graphs, client, session, dates and annotation; never paper text beyond what was
cited. Invented papers and textbook terms only."""

from __future__ import annotations

from datetime import timedelta

import pytest

from seedgraph.records import export, manage

from ._world import DID_TEXT, QUOTES, UTC, Agent, span_hit, standard


@pytest.fixture
def lib(tmp_path):
    from seedgraph.library import store

    lib = store.open_library_at(tmp_path / "library")
    yield lib
    lib.close()


@pytest.fixture
def world(lib):
    return standard(lib)


@pytest.fixture
def answered(lib, world):
    w, did, ate = world
    w.graph("did-reading", did)
    read_sp, hit_sp = w.span(did, "parallel"), w.span(did, "serial")
    parent = Agent(lib, name="parallel-trends-1432")
    helper = Agent(lib, name="parallel-trends-1432/a", start=parent.clock)
    parent.search("parallel trends", hits=[span_hit(read_sp, did), span_hit(hit_sp, did)], graph="did-reading")
    helper.clock = parent.clock
    helper.paper_read(did, *did.at("parallel"))
    parent.clock = helper.clock + timedelta(minutes=39)
    parent.paper_show(ate, [ate.claims["assumptions[0]"]])
    res = parent.answer([{"span_id": read_sp, "why": "states the assumption"}, {"span_id": hit_sp},
                         {"claim_id": ate.claims["assumptions[0]"]}, {"paper_id": ate.paper_id, "page": 3}],
                        question="Is parallel trends plausible in the county design?",
                        answer="The authors assume parallel trends between treated and control counties.",
                        judgement="Plausible, but the pre-period is short.")
    manage.annotate(lib, res.record_id, "Ask about event-study leads.", now=parent.clock)
    return res, did, ate


def test_export_layout(lib, answered):
    res, did, ate = answered
    md = export.export_markdown(lib, res.record_id, tz=UTC)
    heads = [line for line in md.splitlines() if line.startswith("#")]
    assert heads == ["# Is parallel trends plausible in the county design?", "## Answer", "## Agent's judgement",
                     "## Cited evidence", "## Path", "## Items", "## Annotation"]
    assert "The authors assume parallel trends between treated and control counties." in md
    judgement = md.split("## Agent's judgement")[1].split("##")[0]
    assert "Plausible, but the pre-period is short." in judgement
    assert "Plausible, but" not in md.split("## Agent's judgement")[0]


def test_export_header_names_session_client_graphs_and_dates(lib, answered):
    res, _, _ = answered
    md = export.export_markdown(lib, res.record_id, tz=UTC)
    head = md.split("## Answer")[0]
    assert f"- Record: `{res.record_id}`" in head
    assert "- Session: parallel-trends-1432 (claude)" in head
    assert "- State: answered" in head
    assert "- Graphs: did-reading (searched)" in head
    assert "- Opened: 2026-11-04 14:33 UTC" in head and "- Answered: 2026-11-04 15:15 UTC" in head


def test_export_cited_passages_carry_paper_page_trust_and_flags(lib, answered):
    res, did, ate = answered
    cited = export.export_markdown(lib, res.record_id, tz=UTC).split("## Cited evidence")[1].split("## Path")[0]
    entries = [e for e in cited.split("\n\n") if e.strip()]
    assert len(entries) == 4
    first, second, claim, paper = entries
    assert "1. Invented county minimum wages (2019), page 1" in first
    assert f"> {QUOTES['parallel']}" in first and "Why: states the assumption" in first
    assert "opened in this record (opened by helper a)" in first and "Flag" not in first
    assert "**Flag: cited without being read (seen in search results only)**" in second
    assert "Claim (identification_assumption, unconfoundedness): Selection on observables." in claim
    assert "paper states" in claim and "Invented matching for training (2015)" in claim
    assert "4. Invented matching for training (2015), page 3" in paper


def test_export_path_lists_searches_with_tagged_hits_and_opens_in_order(lib, answered):
    res, did, _ = answered
    p = export.export_markdown(lib, res.record_id, tz=UTC).split("## Path")[1].split("## Items")[0]
    lines = [line for line in p.splitlines() if line.strip()]
    assert lines[0].startswith('1. 14:33 search "parallel trends" (spans; graph did-reading): 2 hits')
    assert lines[1].strip().endswith(": cited") and lines[2].strip().endswith(": cited")
    assert lines[3].startswith("2. 14:34 paper_read Invented county minimum wages (2019)")
    assert "(helper a)" in lines[3]
    assert lines[4].strip() == "_40 minutes later_"
    assert lines[5].startswith("3. 15:14 paper_show Invented matching for training (2015)")


def test_export_items_table_tags_every_item(lib, answered):
    res, did, ate = answered
    items = export.export_markdown(lib, res.record_id, tz=UTC).split("## Items")[1].split("## Annotation")[0]
    assert "| item | paper | tag |" in items
    assert f"| paper `{did.paper_id}` | Invented county minimum wages (2019) | opened but not cited |" in items


def test_export_has_no_paper_text_beyond_what_was_cited(lib, answered):
    res, _, _ = answered
    md = export.export_markdown(lib, res.record_id, tz=UTC)
    assert QUOTES["did_result"] not in md and "an invented preview" not in md
    assert DID_TEXT not in md


def test_export_annotation_last(lib, answered):
    res, _, _ = answered
    md = export.export_markdown(lib, res.record_id, tz=UTC)
    assert md.rstrip().endswith("Ask about event-study leads.")


def test_export_a_path_without_answer(lib, world):
    w, did, _ = world
    a = Agent(lib)
    a.search("serial correlation")
    a.paper_show(did)
    [row] = manage.list_records(lib, now=a.clock).rows
    md = export.export_markdown(lib, row.record_id, now=a.clock + timedelta(hours=1), tz=UTC)
    assert md.startswith("# Path without answer")
    assert "## Answer" not in md and "## Cited evidence" not in md and "## Annotation" not in md
    assert "- Session: 2026-11-04 14:33 (claude)" in md
    assert "- Notes: session told apart by connection only" in md
    assert "- Graphs: Library-wide" in md
    assert '1. 14:33 search "serial correlation" (spans): 0 hits' in md


def test_export_refuses_an_unknown_record(lib):
    with pytest.raises(manage.RecordNotFound):
        export.export_markdown(lib, "tr_nope")
