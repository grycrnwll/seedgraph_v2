"""The Home activity feed's records (stage 5 design §6.4; slice R5): the latest answered
records and paths without answer, linked to the record; records still in progress are
left out. Invented records only."""

from __future__ import annotations

from datetime import timedelta

import pytest

from records._world import Agent

from seedgraph.web.views import home as home_view

from .records_support import NOW, build, clock, with_lib


@pytest.fixture
def rx(home):
    return build(home)


def records_in(view):
    return {e.href: e for e in view.entries if e.what == "record"}


def test_the_feed_has_answered_records_and_paths_without_answer(rx):
    feed = records_in(with_lib(rx.home, lambda lib: home_view.activity(lib, now=NOW)))
    assert set(feed) == {f"/records/{rx.both}", f"/records/{rx.flagged}", f"/records/{rx.no_answer}"}
    assert feed[f"/records/{rx.both}"].text == "record: Which graphs discuss two invented designs? (1 cited)"
    assert feed[f"/records/{rx.flagged}"].text == (
        "record: Is parallel trends plausible in the invented county study? (3 cited, 1 flag)")
    assert feed[f"/records/{rx.no_answer}"].text == "path without answer (1 search)"


def test_in_progress_records_are_left_out(rx):
    feed = records_in(with_lib(rx.home, lambda lib: home_view.activity(lib, now=NOW)))
    assert f"/records/{rx.in_progress}" not in feed
    later = records_in(with_lib(rx.home, lambda lib: home_view.activity(lib, now=NOW + timedelta(hours=1))))
    assert f"/records/{rx.in_progress}" in later  # once idle it is a path without answer


def test_a_long_question_is_shortened_and_entries_are_newest_first(rx):
    long_q = "Does the invented minimum wage " + "really " * 30 + "change teen employment?"

    def go(lib):
        a = Agent(lib, "mcp-eeee0005", name="long-question-1600", start=NOW - timedelta(minutes=10))
        a.paper_show(rx.did)
        return a.answer([{"paper_id": rx.did.paper_id}], question=long_q).record_id

    rid = with_lib(rx.home, go)
    view = with_lib(rx.home, lambda lib: home_view.activity(lib, now=NOW))
    e = records_in(view)[f"/records/{rid}"]
    assert e.text.startswith("record: Does the invented minimum wage really") and e.text.endswith("… (1 cited)")
    assert len(e.text) < 120
    ats = [x.at for x in view.entries]
    assert ats == sorted(ats, reverse=True)
    assert view.entries[0].href == f"/records/{rid}"


def test_the_home_page_links_the_records(rx, make_app, client_for):
    html = client_for(make_app(clock=clock())).get("/").text
    assert f'href="/records/{rx.flagged}"' in html
    assert "path without answer (1 search)" in html
    assert f'href="/records/{rx.in_progress}"' not in html
