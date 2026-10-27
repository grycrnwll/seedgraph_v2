"""Concept pages (stage 4 design §7, §2.4, §11.2; slice C7): GETs never write, no inline
script or style, CSRF on every POST, each POST calls one deep function, split / undo
round trips, queue actions, review marks stored and summarised, and grounding hides
unsupported concepts. Invented text only."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from seedgraph.concepts import refresh as rf
from seedgraph.concepts import researcher as rs
from seedgraph.library.store import open_library_at

from .concepts_support import GRAPH, build
from .pages_support import no_inline_script_or_style, post_forms_lead_with_csrf, token


@pytest.fixture
def fx(home):
    return build(home)


@pytest.fixture
def c(fx, make_app, client_for):
    return client_for(make_app())


def _lib(fx):
    return open_library_at(fx.home / "library")


def _dump(fx) -> list[str]:
    conn = sqlite3.connect(fx.home / "library" / "library.db")
    try:
        return list(conn.iterdump())
    finally:
        conn.close()


def _pages(fx) -> list[str]:
    return ["/concepts", "/concepts?q=widget", "/concepts?q=rank&family=assumption",
            f"/concepts?q=widget&graph={GRAPH}", "/concepts/assumption/widget%20overlap",
            "/concepts/assumption/gizmo%20rank", "/concepts/assumption/cog%20rank",
            "/concepts/assumption/cog%20rank?sense=s1", "/concepts/recent", "/concepts/recent?batch=b_c7",
            "/concepts/queue", "/concepts/review", f"/concepts/review/{fx.sample_id}", "/concepts/lists",
            "/concepts/batches", "/concepts/batches?task=merge&family=assumption&budget_tokens=100000",
            "/concepts/batches?task=bridge&family=assumption&k=3"]


# --- page rules ---------------------------------------------------------------------------------

def test_get_on_every_concept_page_leaves_the_library_unchanged(fx, c):
    before = _dump(fx)
    for path in _pages(fx):
        assert c.get(path).status_code == 200, path
    assert _dump(fx) == before


def test_get_on_a_stale_library_writes_nothing_and_offers_refresh(fx, c):
    from concepts._world import World, a

    lib = _lib(fx)
    try:
        World(lib).note("lp_c4", [a("gadget overlap", "Invented: a newly noted gadget claim.")])
    finally:
        lib.close()
    before = _dump(fx)
    for path in _pages(fx):
        assert c.get(path).status_code == 200, path
    assert _dump(fx) == before
    html = c.get("/concepts").text
    assert "Concepts out of date (1 paper changed)" in html and 'action="/concepts/refresh"' in html
    c.post("/concepts/refresh", data={"csrf_token": token(c), "next": "/concepts"})
    assert "Concepts out of date" not in c.get("/concepts").text


def test_concept_page_forms_post_to_encoded_paths(fx, c):
    html = c.get("/concepts/assumption/widget%20overlap").text
    assert 'action="/concepts/assumption/widget%20overlap/merge"' in html
    assert "%2520" not in html and 'action="/concepts/assumption/widget overlap' not in html


def test_every_concept_page_has_no_inline_script_or_style_and_csrf_first(fx, c):
    for path in _pages(fx):
        html = c.get(path).text
        assert no_inline_script_or_style(html), path
        assert post_forms_lead_with_csrf(html), path
        assert 'href="/static/pages.css"' in html, path
        assert 'href="/concepts"' in html  # the sidebar entry


def test_the_route_module_runs_no_sql():
    text = (Path(__file__).resolve().parents[2] / "src" / "seedgraph" / "web" / "routes" / "concepts.py").read_text(
        encoding="utf-8")
    assert not re.search(r"^\s*(import sqlite3|from sqlite3)", text, re.MULTILINE)
    assert "execute(" not in text


# --- one deep function per POST, CSRF -------------------------------------------------------------

def _posts(fx) -> list[tuple[str, dict, str, str]]:
    """(path, form, module attribute, deep function)."""
    page = "/concepts/assumption/widget%20overlap"
    return [
        ("/concepts/refresh", {}, "seedgraph.concepts.refresh", "refresh"),
        (f"{page}/merge", {"other": "doohickey rank"}, "seedgraph.concepts.researcher", "merge"),
        (f"{page}/link", {"other": "doohickey rank", "note": "Invented note."}, "seedgraph.concepts.researcher",
         "link"),
        (f"{page}/never-connect", {"other": "doohickey rank"}, "seedgraph.concepts.researcher", "never_connect"),
        ("/concepts/assumption/doohickey%20rank/split-senses", {"gloss_1": "x", "gloss_2": "y"},
         "seedgraph.concepts.researcher", "split_label"),
        ("/concepts/assumption/cog%20rank/undo-split", {}, "seedgraph.concepts.researcher", "undo_label_split"),
        ("/concepts/decisions/split", {"decision": fx.merge_id}, "seedgraph.concepts.researcher", "split_decision"),
        ("/concepts/decisions/undo", {"decision": fx.merge_id}, "seedgraph.concepts.researcher", "undo"),
        ("/concepts/queue/settle", {"item": fx.queued_merge, "action": "reject"}, "seedgraph.concepts.researcher",
         "settle"),
        ("/concepts/review", {"size": "50", "seed": "1"}, "seedgraph.concepts.researcher", "create_review_sample"),
        (f"/concepts/review/{fx.sample_id}/mark", {"decision": fx.merge_id, "mark": "right"},
         "seedgraph.concepts.researcher", "mark"),
        ("/concepts/lists/ambiguous", {"action": "add", "pattern": "gizmo rank"}, "seedgraph.concepts.researcher",
         "edit_ambiguous"),
        ("/concepts/lists/never-merge", {"label_a": "gizmo rank", "label_b": "cog rank"},
         "seedgraph.concepts.researcher", "add_never_merge"),
        ("/concepts/lists/never-merge/remove", {"decision": "cd_x"}, "seedgraph.concepts.researcher",
         "remove_never_merge"),
        ("/concepts/batches", {"task": "bridge", "family": "assumption"}, "seedgraph.work.routine", "create_batch"),
    ]


def test_every_post_needs_csrf(fx, c):
    before = _dump(fx)
    for path, data, _, _ in _posts(fx):
        assert c.post(path, data=data).status_code == 403, path
    assert _dump(fx) == before


def test_every_post_calls_one_deep_function(fx, c, monkeypatch):
    import importlib

    for path, data, module, name in _posts(fx):
        calls = []

        def fake(*args, _name=name, _calls=calls, **kw):
            _calls.append((args, kw))
            if _name == "create_review_sample":
                return rs.ReviewSample("rs_fake", 0, 1, "now", ("cd_x",))
            if _name == "refresh":
                return rf.RefreshReport()
            if _name == "create_batch":
                raise ValueError("stopped by the test")
            if _name == "mark":
                return "right"
            return rs.Outcome(None, "done")

        monkeypatch.setattr(importlib.import_module(module), name, fake)
        r = c.post(path, data={"csrf_token": token(c), **data})
        assert r.status_code == 303, path
        assert len(calls) == 1, path
        monkeypatch.undo()


# --- search and grounding -------------------------------------------------------------------------

def test_search_shows_the_mode_and_sense_level_hits(fx, c):
    html = c.get("/concepts?q=widget").text
    assert "keyword search" in html
    assert "/concepts/assumption/widget%20overlap" in html or "/concepts/assumption/gadget%20overlap" in html


def test_grounding_hides_concepts_without_live_support(fx, c):
    assert c.get("/concepts/method/unanchored%20estimator").status_code == 404
    html = c.get("/concepts?q=unanchored%20estimator").text
    assert "/concepts/method/unanchored" not in html and "No supported concept matches." in html
    assert c.get("/concepts/notafamily/widget").status_code == 404
    assert c.get("/concepts/assumption/no%20such%20label").status_code == 404


def test_a_split_label_shows_a_sense_chooser_with_glosses(fx, c):
    html = c.get("/concepts/assumption/cog%20rank").text
    assert "full rank of a matrix" in html and "an ordering by size" in html
    assert "?sense=s1" in html and "?sense=s2" in html


def test_the_concept_page_shows_members_claims_links_and_literatures(fx, c):
    html = c.get("/concepts/assumption/widget%20overlap").text
    assert "gadget overlap" in html and "widget overlap" in html          # members of the merge
    assert "merged by agent inferred" in html
    assert "Invented: each widget has a twin widget." in html             # supporting claim + quote
    assert "/papers/lp_c1#claim-" in html
    assert "Often appears with" in html and "doohickey rank" in html      # co-supported on lp_c1, lp_c2
    assert "Invented concept widgets" in html                             # literatures per graph
    link = c.get("/concepts/assumption/gizmo%20rank").text
    assert "Invented: gizmo rank is the older name." in link and "agent inferred" in link
    assert link.count("<blockquote>") >= 2                                # a quote from each side


# --- researcher actions through the pages -----------------------------------------------------------

def test_merge_from_the_concept_page_then_undo(fx, c):
    page = "/concepts/assumption/widget%20overlap"
    r = c.post(f"{page}/merge", data={"csrf_token": token(c), "other": "sprocket condition"})
    assert r.status_code == 303
    assert "sprocket condition" in c.get(page).text
    lib = _lib(fx)
    try:
        did = lib.conn.execute("SELECT decision_id FROM concept_decisions WHERE kind = 'merge' AND "
                               "decided_by = 'researcher'").fetchone()[0]
    finally:
        lib.close()
    c.post("/concepts/decisions/undo", data={"csrf_token": token(c), "decision": did})
    assert "sprocket condition" not in c.get(page).text.split("Supporting claims")[0]


def test_a_refused_merge_flashes_the_reason(fx, c):
    r = c.post("/concepts/assumption/widget%20overlap/merge",
               data={"csrf_token": token(c), "other": "method/unanchored estimator"})
    assert r.status_code == 303
    assert "not done" in c.get(r.headers["location"]).text


def test_split_a_merge_in_recent_merges_and_undo_round_trips(fx, c):
    def concepts_state():
        lib = _lib(fx)
        try:
            return [tuple(x) for x in lib.conn.execute("SELECT * FROM concept_members ORDER BY 1, 2, 3, 4")]
        finally:
            lib.close()

    before = concepts_state()
    assert "Split" in c.get("/concepts/recent").text
    r = c.post("/concepts/decisions/split", data={"csrf_token": token(c), "decision": fx.merge_id})
    assert r.status_code == 303 and r.headers["location"].startswith("/concepts/recent")
    after = concepts_state()
    assert after != before
    recent = c.get("/concepts/recent").text
    assert "never connect" in recent
    lib = _lib(fx)
    try:
        nc = lib.conn.execute("SELECT decision_id FROM concept_decisions WHERE origin = ?",
                              (f"split:{fx.merge_id}",)).fetchone()[0]
    finally:
        lib.close()
    c.post("/concepts/decisions/undo", data={"csrf_token": token(c), "decision": nc})
    assert concepts_state() == before


def test_split_senses_from_the_concept_page(fx, c):
    page = "/concepts/assumption/doohickey%20rank"
    html = c.get(page).text
    ids = re.findall(r'name="claim_(cl_[^"]+)"', html)
    assert len(ids) == 2
    r = c.post(f"{page}/split-senses", data={"csrf_token": token(c), "gloss_1": "Invented sense one",
                                             "gloss_2": "Invented sense two", f"claim_{ids[0]}": "1",
                                             f"claim_{ids[1]}": "2"})
    assert r.status_code == 303
    chooser = c.get(page).text
    assert "Invented sense one" in chooser and "Invented sense two" in chooser
    c.post(f"{page}/undo-split", data={"csrf_token": token(c), "sense": "s1"})
    assert "Supporting claims" in c.get(page).text


# --- queue ----------------------------------------------------------------------------------------

def test_the_queue_shows_both_sides_and_settles_items(fx, c):
    html = c.get("/concepts/queue").text
    assert "Why it is here: reach" in html and "Why it is here: unsure" in html
    assert "Link instead" in html
    assert "Invented: sprockets turn freely." in html
    r = c.post("/concepts/queue/settle", data={"csrf_token": token(c), "item": fx.queued_merge, "action": "link",
                                               "note": "Invented: related, not the same."})
    assert r.status_code == 303
    r = c.post("/concepts/queue/settle", data={"csrf_token": token(c), "item": fx.queued_unsure,
                                               "action": "accept"})
    assert "not done" in c.get(r.headers["location"]).text  # unsure cannot be accepted
    c.post("/concepts/queue/settle", data={"csrf_token": token(c), "item": fx.queued_unsure, "action": "unrelated"})
    assert "The queue is empty." in c.get("/concepts/queue").text
    assert "Invented: related, not the same." in c.get("/concepts/assumption/widget%20overlap").text


# --- review sample --------------------------------------------------------------------------------

def test_review_marks_are_stored_and_summarised_on_the_page(fx, c):
    url = f"/concepts/review/{fx.sample_id}"
    html = c.get(url).text
    assert "Should have been a link" in html and "Should have been a merge" in html
    c.post(f"{url}/mark", data={"csrf_token": token(c), "decision": fx.merge_id, "mark": "wrong",
                                "note": "Invented: not the same."})
    c.post(f"{url}/mark", data={"csrf_token": token(c), "decision": fx.link_id, "mark": "should_be_merge"})
    r = c.post(f"{url}/mark", data={"csrf_token": token(c), "decision": fx.link_id, "mark": "should_be_link"})
    assert "not done" in c.get(r.headers["location"].split("#")[0]).text
    lib = _lib(fx)
    try:
        summ = rs.review_summary(lib.conn, fx.sample_id)
    finally:
        lib.close()
    assert summ.by_mark == {"should_be_merge": 1, "wrong": 1}
    page = c.get(url).text
    assert "gate missed" in page and "should be merge: 1" in page
    index = c.get("/concepts/review").text
    assert "2 of 2" in index


def test_drawing_a_sample_redirects_to_it(fx, c):
    r = c.post("/concepts/review", data={"csrf_token": token(c), "size": "1", "seed": "3"})
    assert r.status_code == 303 and r.headers["location"].startswith("/concepts/review/rs_")
    assert c.get(r.headers["location"].split("?")[0]).status_code == 200


# --- lists ----------------------------------------------------------------------------------------

def test_the_lists_edit_ambiguous_labels_and_never_merge_pairs(fx, c):
    html = c.get("/concepts/lists").text
    assert "identification" in html and "heteroskedasticity" in html  # seed lists
    c.post("/concepts/lists/ambiguous", data={"csrf_token": token(c), "action": "add", "pattern": "Gizmo rank"})
    c.post("/concepts/lists/ambiguous", data={"csrf_token": token(c), "action": "single_sense",
                                              "pattern": "identification", "graph": GRAPH})
    c.post("/concepts/lists/never-merge", data={"csrf_token": token(c), "label_a": "gizmo rank",
                                                "label_b": "doohickey rank", "note": "Invented reason."})
    html = c.get("/concepts/lists").text
    assert "gizmo rank" in html and "single-sense here" in html and "Invented reason." in html
    assert "sense not checked" in c.get("/concepts/assumption/gizmo%20rank").text


# --- batches --------------------------------------------------------------------------------------

def test_the_batch_panel_shows_the_estimate_and_the_command(fx, c):
    html = c.get("/concepts/batches?task=merge&family=assumption&budget_tokens=100000").text
    assert "Dry-run estimate" in html
    assert "seedgraph work create concepts:merge --option families=assumption --option budget_tokens=100000" in html


def test_creating_a_concept_batch_redirects_to_the_batch(fx, c):
    r = c.post("/concepts/batches", data={"csrf_token": token(c), "task": "bridge", "family": "assumption"})
    assert r.status_code == 303
    loc = r.headers["location"]
    assert loc.startswith("/batches/b_")
    assert c.get(loc.split("?")[0]).status_code == 200
