"""Records screens (stage 5 design §6.1-6.3, §13 Q3; slice R5): the list with its standing
scope line and filters, the record page (judgement boxed and labelled, cited evidence with
trust labels and "edited since this answer", the path and items table, notes), star /
annotate / export / delete, and the graph Records tab. Invented records only."""

from __future__ import annotations

import re
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from seedgraph.records import export, manage

from .pages_support import no_inline_script_or_style, post_forms_lead_with_csrf, token
from .records_support import ATE_GRAPH, DID_GRAPH, JUDGEMENT, NOW, build, clock, with_lib

SCOPE_LINE = ("Only answers an agent recorded from your library appear here, with paths that ended without an "
              "answer. Questions answered without searching the library do not.")


@pytest.fixture
def rx(home):
    return build(home)


@pytest.fixture
def c(rx, make_app, client_for):
    return client_for(make_app(clock=clock()))


def listed(html: str) -> list[str]:
    """Record ids linked from list rows, in page order."""
    return list(dict.fromkeys(re.findall(r'<a class="sg-record-link" href="/records/(tr_[0-9a-f]+)"', html)))


def row_html(html: str, record_id: str) -> str:
    m = re.search(r'<tr id="record-%s">(.*?)</tr>' % record_id, html, re.S)
    assert m, record_id
    return m.group(1)


# --- the list -----------------------------------------------------------------------------------

def test_the_list_shows_the_scope_line_and_every_record_newest_first(rx, c):
    r = c.get("/records")
    assert r.status_code == 200
    html = r.text
    assert SCOPE_LINE.replace("'", "&#39;") in html or SCOPE_LINE in html
    assert listed(html) == [rx.in_progress, rx.flagged, rx.both, rx.no_answer]
    assert 'href="/records" aria-current="page"' in html
    assert "arrives in stage 5" not in html


def test_each_row_has_title_graphs_session_counts_flags(rx, c):
    html = c.get("/records").text
    both = row_html(html, rx.both)
    assert "Which graphs discuss two invented designs?" in both
    assert "spans 2 graphs" in both and "DiD reading" in both and "Matching estimators" in both
    assert "parallel-trends-1432" in both and "claude" in both
    assert "2 searches" in both and "1 opened" in both and "1 cited" in both

    flagged = row_html(html, rx.flagged)
    assert "Library-wide" in flagged and "cited without being read" in flagged
    assert "library-wide search; cites papers in DiD reading" in flagged
    assert "3 cited" in flagged

    no_answer = row_html(html, rx.no_answer)
    name = with_lib(rx.home, lambda lib: manage.get_record(lib, rx.no_answer, now=NOW).row.session_name)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}(-\d+)?", name)
    assert "path without answer" in no_answer and name in no_answer and "codex" in no_answer
    assert "Library-wide" in no_answer
    assert "in progress" in row_html(html, rx.in_progress)


@pytest.mark.parametrize("query,expected", [
    (f"graph={DID_GRAPH}", ["flagged", "both"]),
    (f"graph={ATE_GRAPH}", ["both"]),
    ("graph=%3Alibrary-wide", ["in_progress", "flagged", "no_answer"]),
    ("state=answered", ["flagged", "both"]),
    ("state=path_without_answer", ["no_answer"]),
    ("state=in_progress", ["in_progress"]),
    ("flagged=1", ["flagged"]),
    ("flagged=0", ["in_progress", "both", "no_answer"]),
    ("client=codex", ["no_answer"]),
    ("q=INVENTED+county", ["flagged"]),
])
def test_filters(rx, c, query, expected):
    html = c.get(f"/records?{query}").text
    assert listed(html) == [getattr(rx, name) for name in expected], query


def test_a_bad_filter_value_is_ignored_not_an_error(rx, c):
    r = c.get("/records?state=bogus&offset=x&graph=")
    assert r.status_code == 200 and len(listed(r.text)) == 4


def test_the_filter_form_offers_graphs_library_wide_states_and_clients(rx, c):
    html = c.get("/records").text
    form = re.search(r'<form[^>]*method="get"[^>]*action="/records".*?</form>', html, re.S).group(0)
    for value in (DID_GRAPH, ATE_GRAPH, ":library-wide", "answered", "path_without_answer", "in_progress",
                  "claude", "codex"):
        assert f'value="{value}"' in form, value
    assert 'name="starred"' in form and 'name="flagged"' in form and 'name="q"' in form


def test_paging_keeps_the_filters(rx, make_app, client_for, monkeypatch):
    from seedgraph.web.views import records as rv

    monkeypatch.setattr(rv, "PAGE_SIZE", 1)
    c = client_for(make_app(clock=clock()))
    html = c.get("/records?state=answered").text
    assert listed(html) == [rx.flagged]
    assert 'href="/records?state=answered&amp;offset=1"' in html
    assert listed(c.get("/records?state=answered&offset=1").text) == [rx.both]


# --- the record page ----------------------------------------------------------------------------

def test_the_record_page_has_question_answer_and_the_judgement_boxed_and_labelled(rx, c):
    html = c.get(f"/records/{rx.flagged}").text
    assert "Is parallel trends plausible in the invented county study?" in html
    assert "The paper assumes parallel trends between treated and control counties." in html
    box = re.search(r'<section class="sg-judgement"[^>]*>(.*?)</section>', html, re.S)
    assert box, "the judgement has its own box"
    assert "<h2" in box.group(1) and "Agent's judgement</h2>" in box.group(1) and JUDGEMENT in box.group(1)
    assert JUDGEMENT not in html.replace(box.group(0), "")


def test_cited_evidence_shows_quote_paper_page_trust_flag_and_links(rx, c):
    html = c.get(f"/records/{rx.flagged}").text
    ev = re.search(r'<section class="sg-section sg-evidence".*?</section>', html, re.S).group(0)
    assert "Treated and control counties would have followed parallel trends without the policy." in ev
    assert "Invented county minimum wages" in ev and "2019" in ev and "p. 1" in ev
    assert "paper states" in ev
    assert "Parallel trends between treated and control counties." in ev  # the frozen claim text
    assert "cited without being read" in ev
    assert "the assumption in the paper&#39;s words" in ev
    assert f'href="/papers/{rx.did.paper_id}#claim-{rx.claim_id}"' in ev
    assert "edited since this answer" not in ev


def test_the_path_timeline_and_items_table(rx, c):
    html = c.get(f"/records/{rx.flagged}").text
    path = re.search(r'<section class="sg-section sg-path".*?</section>', html, re.S).group(0)
    assert "search" in path and "parallel trends" in path
    assert "paper_read" in path and "paper_show" in path and "Invented county minimum wages" in path
    items = re.search(r'<table class="sg-table sg-items".*?</table>', html, re.S).group(0)
    assert "cited" in items and "opened but not cited" in items
    assert f"<code>{rx.span_id}</code>" in items


def test_a_path_without_answer_page_and_its_notes(rx, c):
    html = c.get(f"/records/{rx.no_answer}").text
    assert "path without answer" in html.lower()
    assert "session told apart by connection only" in html
    assert "matching estimators" in html
    assert "found but not opened" in html
    assert "sg-judgement" not in html


def test_an_unknown_record_is_404(rx, c):
    assert c.get("/records/tr_00000000").status_code == 404
    assert c.get("/records/not-a-record").status_code == 404
    assert c.get("/records/tr_00000000/export").status_code == 404
    assert c.get("/records/tr_00000000/delete").status_code == 404


def _edited_world(home):
    from records._world import Agent

    rx = build(home)

    def go(lib):
        from records._world import World
        w = World(lib)
        # an edit after the flagged answer
        w.edit(rx.did, rx.claim_id, 0, claim_text="Parallel pre-trends in invented employment.")
        # a record answered after that edit, with no change since
        a = Agent(lib, "mcp-dddd0004", name="after-edit-1500", start=NOW - timedelta(minutes=20))
        a.paper_show(rx.did)
        return a.answer([{"claim_id": rx.claim_id}], question="Answered after the edit?").record_id

    return rx, with_lib(home, go)


def test_edited_since_this_answer_is_shown_only_when_the_paper_changed_after(home, make_app, client_for):
    rx, after = _edited_world(home)
    c = client_for(make_app(clock=clock()))
    before = c.get(f"/records/{rx.flagged}").text
    assert "edited since this answer" in before
    assert "Parallel trends between treated and control counties." in before  # still the frozen text
    assert f'href="/papers/{rx.did.paper_id}#claim-{rx.claim_id}"' in before
    assert "edited since this answer" not in c.get(f"/records/{after}").text


# --- actions ------------------------------------------------------------------------------------

def _spy(monkeypatch, name):
    calls = []
    real = getattr(manage, name)

    def spy(*args, **kw):
        calls.append(args[1:])
        return real(*args, **kw)

    monkeypatch.setattr(manage, name, spy)
    return calls


def test_the_record_page_offers_star_annotate_export_and_delete(rx, c):
    html = c.get(f"/records/{rx.flagged}").text
    assert f'action="/records/{rx.flagged}/star"' in html
    assert f'action="/records/{rx.flagged}/annotate"' in html
    assert f'href="/records/{rx.flagged}/export"' in html
    assert f'href="/records/{rx.flagged}/delete"' in html
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)


def test_star_and_unstar_call_manage_star_once_each(rx, c, monkeypatch):
    calls = _spy(monkeypatch, "star")
    assert c.post(f"/records/{rx.both}/star", data={"starred": "1"}).status_code == 403
    r = c.post(f"/records/{rx.both}/star", data={"csrf_token": token(c), "starred": "1"})
    assert r.status_code == 303 and r.headers["location"].startswith(f"/records/{rx.both}")
    assert listed(c.get("/records?starred=1").text) == [rx.both]
    c.post(f"/records/{rx.both}/star", data={"csrf_token": token(c), "starred": "0", "next": "/records?q=x"})
    assert listed(c.get("/records?starred=1").text) == []
    assert calls == [(rx.both, True), (rx.both, False)]


def test_star_from_the_list_returns_to_the_list(rx, c):
    html = c.get("/records?state=answered").text
    assert f'action="/records/{rx.both}/star"' in row_html(html, rx.both)
    r = c.post(f"/records/{rx.both}/star", data={"csrf_token": token(c), "starred": "1",
                                                  "next": "/records?state=answered"})
    assert r.headers["location"].startswith("/records?state=answered")


def test_annotate_saves_shows_and_clears(rx, c, monkeypatch):
    calls = _spy(monkeypatch, "annotate")
    r = c.post(f"/records/{rx.flagged}/annotate", data={"csrf_token": token(c), "text": "An invented remark."})
    assert r.status_code == 303
    assert "An invented remark." in c.get(r.headers["location"]).text
    c.post(f"/records/{rx.flagged}/annotate", data={"csrf_token": token(c), "text": ""})
    assert with_lib(rx.home, lambda lib: manage.get_record(lib, rx.flagged, now=NOW).row.annotation) is None
    assert [a[0] for a in calls] == [rx.flagged, rx.flagged]


def test_a_refused_annotation_flashes_the_reason(rx, c):
    r = c.post(f"/records/{rx.flagged}/annotate",
               data={"csrf_token": token(c), "text": "x" * (manage.MAX_ANNOTATION + 1)})
    html = c.get(r.headers["location"]).text
    assert "sg-flash-error" in html and "limit" in html


def test_delete_asks_first_then_calls_manage_delete(rx, c, monkeypatch):
    calls = _spy(monkeypatch, "delete")
    confirm = c.get(f"/records/{rx.both}/delete")
    assert confirm.status_code == 200 and calls == []
    assert f'action="/records/{rx.both}/delete"' in confirm.text
    assert "Which graphs discuss two invented designs?" in confirm.text
    assert post_forms_lead_with_csrf(confirm.text)
    assert c.post(f"/records/{rx.both}/delete").status_code == 403
    r = c.post(f"/records/{rx.both}/delete", data={"csrf_token": token(c)})
    assert r.status_code == 303 and r.headers["location"].startswith("/records")
    assert calls == [(rx.both,)]
    assert rx.both not in listed(c.get("/records").text)
    assert c.get(f"/records/{rx.both}").status_code == 404


def test_export_downloads_the_markdown(rx, c, monkeypatch):
    calls = []
    real = export.export_markdown
    monkeypatch.setattr(export, "export_markdown", lambda lib, rid, **kw: (calls.append(rid), real(lib, rid, **kw))[1])
    r = c.get(f"/records/{rx.flagged}/export")
    assert r.status_code == 200 and calls == [rx.flagged]
    assert r.headers["content-type"].startswith("text/markdown")
    assert r.headers["content-disposition"] == f'attachment; filename="{rx.flagged}.md"'
    assert r.text.startswith("# Is parallel trends plausible in the invented county study?")
    assert "## Agent's judgement" in r.text and JUDGEMENT in r.text


# --- the graph Records tab ----------------------------------------------------------------------

def test_the_graph_page_has_a_records_tab(rx, c):
    html = c.get(f"/graphs/{DID_GRAPH}").text
    assert f'href="/graphs/{DID_GRAPH}/records"' in html


def test_the_graph_records_tab_lists_its_records_and_marks_library_wide_ones(rx, c):
    r = c.get(f"/graphs/{DID_GRAPH}/records")
    assert r.status_code == 200
    html = r.text
    assert f'href="/graphs/{DID_GRAPH}/records" aria-current="page"' in html
    assert listed(html) == [rx.flagged, rx.both]
    assert "library-wide search; cites papers in this graph" in row_html(html, rx.flagged)
    assert "library-wide search" not in row_html(html, rx.both)
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)
    assert listed(c.get(f"/graphs/{DID_GRAPH}/records?state=path_without_answer").text) == []
    assert listed(c.get(f"/graphs/{ATE_GRAPH}/records").text) == [rx.both]


def test_an_unknown_graph_records_tab_is_404(rx, c):
    assert c.get("/graphs/no-such-graph/records").status_code == 404


def test_a_deleted_graph_is_named_by_its_slug_and_marked(rx, c):
    import shutil

    shutil.rmtree(with_lib(rx.home, lambda lib: lib.graphs_root) / ATE_GRAPH)
    both = row_html(c.get("/records").text, rx.both)
    assert ATE_GRAPH in both and "graph deleted" in both
    assert "graph deleted: matching" in c.get(f"/records/{rx.both}").text


# --- page rules ---------------------------------------------------------------------------------

def _pages(rx) -> list[str]:
    return ["/records", f"/records?graph={DID_GRAPH}&state=answered&starred=1&flagged=1&client=claude&q=x",
            f"/records/{rx.flagged}", f"/records/{rx.both}", f"/records/{rx.no_answer}",
            f"/records/{rx.in_progress}", f"/records/{rx.both}/delete", f"/graphs/{DID_GRAPH}/records", "/"]


def _dump(db: Path) -> list[str]:
    conn = sqlite3.connect(db)
    try:
        return list(conn.iterdump())
    finally:
        conn.close()


def test_every_records_page_keeps_the_csp_and_csrf_rules_and_never_writes(rx, c):
    db = rx.home / "library" / "library.db"
    before = _dump(db)
    for path in _pages(rx):
        r = c.get(path)
        assert r.status_code == 200, path
        assert no_inline_script_or_style(r.text), path
        assert post_forms_lead_with_csrf(r.text), path
        assert 'href="/static/pages.css"' in r.text, path
    c.get(f"/records/{rx.flagged}/export")
    assert _dump(db) == before


def test_the_records_route_module_runs_no_sql_and_calls_records_through_their_modules():
    text = (Path(__file__).resolve().parents[2] / "src" / "seedgraph" / "web" / "routes" / "records.py"
            ).read_text(encoding="utf-8")
    assert "execute(" not in text and "sqlite3" not in text
    assert not re.search(r"from \S*records\.manage import|from \S*records\.export import", text)
