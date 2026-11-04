"""Managing records (stage 5 design §3.5-3.7, §6.1-6.3; slice R2): the list and its filters,
"in progress" vs "path without answer", graphs and the library-wide mark, the record notes,
star, annotate and delete (events and graph links go with the record). Invented papers."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from seedgraph.records import log, manage

from ._world import ATE_TEXT, T0, Agent, span_hit, standard


@pytest.fixture
def lib(tmp_path):
    from seedgraph.library import store

    lib = store.open_library_at(tmp_path / "library")
    yield lib
    lib.close()


@pytest.fixture
def world(lib):
    return standard(lib)


def ids(page):
    return [r.record_id for r in page.rows]


# --- the list -----------------------------------------------------------------------------------

def test_list_is_newest_first_with_counts_session_and_state(lib, world):
    w, did, _ = world
    a = Agent(lib, name="parallel-trends-1432")
    a.search()
    a.paper_show(did)
    first = a.answer([{"paper_id": did.paper_id}], question="Is parallel trends plausible?")
    a.search()
    page = manage.list_records(lib, now=a.clock + timedelta(minutes=1))
    assert page.total == 2 and len(page.rows) == 2
    newest, older = page.rows
    assert older.record_id == first.record_id
    assert (older.state, older.question, older.title) == ("answered", "Is parallel trends plausible?",
                                                          "Is parallel trends plausible?")
    assert older.counts == {"searches": 1, "opened": 1, "cited": 1} and older.flags == ()
    assert (older.session_name, older.client, older.key_kind) == ("parallel-trends-1432", "claude", "named")
    assert (newest.state, newest.title) == ("in_progress", "in progress")
    assert newest.counts == {"searches": 1, "opened": 0, "cited": 0}


def test_an_idle_or_closed_open_record_is_a_path_without_answer(lib, world):
    a = Agent(lib)
    a.search()
    assert manage.list_records(lib, now=a.clock + timedelta(minutes=29)).rows[0].state == "in_progress"
    assert manage.list_records(lib, now=a.clock + timedelta(minutes=30)).rows[0].state == "path_without_answer"
    log.close_connection(lib, a.conn_id, now=a.clock + timedelta(minutes=1))
    row = manage.list_records(lib, now=a.clock + timedelta(minutes=2)).rows[0]
    assert (row.state, row.title) == ("path_without_answer", "path without answer")


def test_the_same_record_becomes_answered_if_the_agent_answers_later(lib, world):
    w, did, _ = world
    a = Agent(lib)
    a.paper_show(did)
    [open_row] = manage.list_records(lib, now=a.clock + timedelta(hours=2)).rows
    a.tick(120)
    res = a.answer([{"paper_id": did.paper_id}])
    assert res.record_id == open_row.record_id
    assert manage.list_records(lib, now=a.clock).rows[0].state == "answered"


def test_filters_state_starred_flagged_client_and_text(lib, world):
    w, did, _ = world
    sp = w.span(did, "parallel")
    a = Agent(lib, "mcp-aaaa0001")
    a.search(hits=[span_hit(sp, did)])
    flagged = a.answer([{"span_id": sp}], question="Does serial correlation matter?")
    a.paper_show(did)
    clean = a.answer([{"paper_id": did.paper_id}], question="What is the ATE?")
    b = Agent(lib, "mcp-bbbb0002", client="codex")
    b.search()
    now = b.clock + timedelta(minutes=1)
    manage.star(lib, clean.record_id)
    F = manage.Filters
    assert ids(manage.list_records(lib, F(state="answered"), now=now)) == [clean.record_id, flagged.record_id]
    assert len(ids(manage.list_records(lib, F(state="in_progress"), now=now))) == 1
    assert ids(manage.list_records(lib, F(state="path_without_answer"), now=now)) == []
    assert ids(manage.list_records(lib, F(starred=True), now=now)) == [clean.record_id]
    assert ids(manage.list_records(lib, F(flagged=True), now=now)) == [flagged.record_id]
    assert ids(manage.list_records(lib, F(flagged=False, state="answered"), now=now)) == [clean.record_id]
    assert len(ids(manage.list_records(lib, F(client="codex"), now=now))) == 1
    assert ids(manage.list_records(lib, F(text="serial CORRELATION"), now=now)) == [flagged.record_id]
    row = manage.list_records(lib, F(flagged=True), now=now).rows[0]
    assert row.flags == ("cited without being read (seen in search results only)",)


def test_an_unknown_state_filter_is_refused(lib):
    with pytest.raises(ValueError):
        manage.list_records(lib, manage.Filters(state="done"))


def test_paging(lib, world):
    w, did, _ = world
    a = Agent(lib)
    made = [a.answer([{"paper_id": did.paper_id}], question=f"Question {n}?").record_id for n in range(5)]
    page = manage.list_records(lib, limit=2, offset=2, now=a.clock)
    assert page.total == 5 and ids(page) == list(reversed(made))[2:4]
    assert (page.offset, page.limit) == (2, 2)


# --- graphs in the list -------------------------------------------------------------------------

def _graph_world(lib, world):
    w, did, ate = world
    w.graph("did-reading", did)
    w.graph("matching", ate)
    a = Agent(lib)
    a.search(graph="did-reading")
    a.search(graph="matching")
    a.paper_show(ate)
    both = a.answer([{"paper_id": ate.paper_id}], question="Two graphs?")
    a.paper_show(did)
    wide = a.answer([{"paper_id": did.paper_id}], question="Library-wide?")
    free = w.paper("lp_free000000000000001", ATE_TEXT + "\nAn invented coda.\n", "An ungraphed paper", 2020)
    a.paper_show(free)
    none = a.answer([{"paper_id": free.paper_id}], question="Nothing?")
    return a, both, wide, none


def test_a_record_spanning_two_graphs_is_marked(lib, world):
    a, both, wide, none = _graph_world(lib, world)
    rows = {r.record_id: r for r in manage.list_records(lib, now=a.clock).rows}
    assert rows[both.record_id].spans_label == "spans 2 graphs: did-reading, matching"
    assert rows[wide.record_id].spans_label is None
    assert [(g.slug, g.how) for g in rows[wide.record_id].graphs] == [("did-reading", "cited_member")]


def test_the_graph_filter_includes_library_wide_records_that_cite_its_papers(lib, world):
    a, both, wide, none = _graph_world(lib, world)
    page = manage.list_records(lib, manage.Filters(graph="did-reading"), now=a.clock)
    assert ids(page) == [wide.record_id, both.record_id]
    marks = {r.record_id: r.graph_mark("did-reading") for r in page.rows}
    assert marks == {wide.record_id: "library-wide search; cites papers in this graph", both.record_id: None}


def test_library_wide_lists_records_that_searched_no_graph(lib, world):
    a, both, wide, none = _graph_world(lib, world)
    page = manage.list_records(lib, manage.Filters(graph=manage.LIBRARY_WIDE), now=a.clock)
    assert ids(page) == [none.record_id, wide.record_id]
    assert [r.graphs for r in page.rows][0] == ()


def test_a_deleted_graph_is_marked(lib, world):
    import shutil

    a, both, wide, none = _graph_world(lib, world)
    shutil.rmtree(lib.graphs_root / "matching")
    row = next(r for r in manage.list_records(lib, now=a.clock).rows if r.record_id == both.record_id)
    assert [(g.slug, g.deleted) for g in row.graphs] == [("did-reading", False), ("matching", True)]
    assert "graph deleted: matching" in manage.get_record(lib, both.record_id, now=a.clock).notes


# --- one record ---------------------------------------------------------------------------------

def test_get_record_has_the_path_evidence_and_notes(lib, world):
    w, did, _ = world
    sp = w.span(did, "parallel")
    a = Agent(lib)
    a.search(hits=[span_hit(sp, did)])
    a.paper_read(did, *did.at("parallel"))
    res = a.answer([{"span_id": sp}], judgement="Plausible but untested.")
    d = manage.get_record(lib, res.record_id, now=a.clock)
    assert d.row.state == "answered" and d.judgement == "Plausible but untested." and d.answer
    assert [s.event.tool for s in d.path.steps] == ["search", "paper_read"]
    assert {(i.kind, i.id): i.tag for i in d.path.items}[("span", sp)] == "cited"
    assert d.evidence[0]["quote"].startswith("Treated and control")
    assert d.papers[did.paper_id] == ("Invented county minimum wages", 2019)
    assert "session told apart by connection only" in d.notes


def test_notes_name_unlogged_calls_and_two_connections(lib, world):
    w, did, _ = world
    one = Agent(lib, "mcp-aaaa0001", name="parallel-trends-1432")
    two = Agent(lib, "mcp-bbbb0002", name="parallel-trends-1432")
    one.search()
    two.search()
    with lib.write() as conn:
        conn.execute("UPDATE traversal_records SET unlogged_calls = 3")
    [row] = manage.list_records(lib, now=two.clock).rows
    d = manage.get_record(lib, row.record_id, now=two.clock)
    assert "session name used from two connections" in d.notes
    assert "3 calls could not be logged" in d.notes
    assert "session told apart by connection only" not in d.notes


def test_get_record_refuses_an_unknown_id(lib):
    with pytest.raises(manage.RecordNotFound):
        manage.get_record(lib, "tr_nope")


def test_the_page_tags_come_from_the_record_s_own_events(lib, world):
    """A follow-up answered from earlier reads: no events of its own, so its cited item is
    cited and nothing else is listed; the frozen check still says it was read earlier."""
    w, did, _ = world
    sp = w.span(did, "parallel")
    a = Agent(lib, name="parallel-trends-1432")
    a.paper_read(did, *did.at("parallel"))
    a.answer([{"span_id": sp}])
    follow = a.answer([{"span_id": sp}], question="Follow-up?")
    d = manage.get_record(lib, follow.record_id, now=a.clock)
    assert d.path.steps == () and [(i.id, i.tag) for i in d.path.items] == [(sp, "cited")]
    assert d.evidence[0]["check_label"] == "opened earlier in this session"


# --- star, annotate, delete ---------------------------------------------------------------------

def test_star_and_unstar(lib, world):
    w, did, _ = world
    rid = Agent(lib).answer([{"paper_id": did.paper_id}]).record_id
    manage.star(lib, rid)
    assert manage.get_record(lib, rid).row.starred is True
    manage.star(lib, rid, False)
    assert manage.get_record(lib, rid).row.starred is False


def test_annotate_sets_and_clears(lib, world):
    w, did, _ = world
    rid = Agent(lib).answer([{"paper_id": did.paper_id}]).record_id
    manage.annotate(lib, rid, "Check the pre-trends figure.", now=T0)
    rec = lib.conn.execute("SELECT annotation, annotated_at FROM traversal_records WHERE record_id = ?",
                           (rid,)).fetchone()
    assert tuple(rec) == ("Check the pre-trends figure.", "2026-11-04T14:32:05.000000+00:00")
    assert manage.get_record(lib, rid).row.annotation == "Check the pre-trends figure."
    manage.annotate(lib, rid, "  ")
    assert tuple(lib.conn.execute("SELECT annotation, annotated_at FROM traversal_records WHERE record_id = ?",
                                  (rid,)).fetchone()) == (None, None)
    with pytest.raises(manage.RecordError):
        manage.annotate(lib, rid, "x" * (manage.MAX_ANNOTATION + 1))


def test_actions_refuse_an_unknown_record(lib):
    for act in (lambda: manage.star(lib, "tr_nope"), lambda: manage.annotate(lib, "tr_nope", "x"),
                lambda: manage.delete(lib, "tr_nope")):
        with pytest.raises(manage.RecordNotFound):
            act()


def test_delete_removes_events_and_graph_links(lib, world):
    w, did, _ = world
    w.graph("did-reading", did)
    a = Agent(lib)
    a.search(graph="did-reading")
    a.paper_show(did)
    rid = a.answer([{"paper_id": did.paper_id}]).record_id
    a.search()
    manage.delete(lib, rid)
    q = lambda sql: lib.conn.execute(sql, (rid,)).fetchone()[0]  # noqa: E731
    assert q("SELECT COUNT(*) FROM traversal_records WHERE record_id = ?") == 0
    assert q("SELECT COUNT(*) FROM traversal_events WHERE record_id = ?") == 0
    assert q("SELECT COUNT(*) FROM record_graphs WHERE record_id = ?") == 0
    assert lib.conn.execute("SELECT COUNT(*) FROM traversal_events").fetchone()[0] == 1


def test_star_annotate_and_delete_leave_the_library_clean(lib, world):
    # Records are test data until stage 7 (design §13 Q9), like the session log.
    w, did, _ = world
    rid = Agent(lib).answer([{"paper_id": did.paper_id}]).record_id
    lib.conn.execute("UPDATE library_meta SET value = '0' WHERE key = 'dirty'")
    manage.star(lib, rid)
    manage.annotate(lib, rid, "Check the pre-trends figure.", now=T0)
    manage.delete(lib, rid)
    assert lib.conn.execute("SELECT value FROM library_meta WHERE key = 'dirty'").fetchone()[0] == "0"


def test_deleting_an_earlier_record_never_changes_a_later_check(lib, world):
    w, did, _ = world
    sp = w.span(did, "parallel")
    a = Agent(lib, name="parallel-trends-1432")
    a.paper_read(did, *did.at("parallel"))
    first = a.answer([{"span_id": sp}])
    later = a.answer([{"span_id": sp}], question="Follow-up?")
    manage.delete(lib, first.record_id)
    stored = lib.conn.execute("SELECT flags_json, evidence_json FROM traversal_records WHERE record_id = ?",
                              (later.record_id,)).fetchone()
    assert json.loads(stored[0]) == [] and json.loads(stored[1])[0]["check"] == "opened_earlier"
