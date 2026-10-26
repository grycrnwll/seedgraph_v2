"""Home (design §2.1): the per-graph "what's next" checklist (derived, lines only while
true) and the activity feed of batch progress with work.state's pause labels."""

from __future__ import annotations

import re

import pytest

from seedgraph.library.store import open_library_at
from seedgraph.web.views import home as home_view
from seedgraph.work import routine, state
from seedgraph.work.claims import WorkerRef

from .pages_support import GRAPH, build, no_inline_script_or_style, post_forms_lead_with_csrf


@pytest.fixture
def fx(home):
    return build(home)


def with_lib(home, fn):
    lib = open_library_at(home / "library")
    try:
        return fn(lib)
    finally:
        lib.close()


def lines_of(home) -> dict[str, tuple[str, str]]:
    view = with_lib(home, lambda lib: home_view.page(lib))
    (g,) = view.graphs
    assert g.slug == GRAPH
    return {line.key: (line.text, line.href) for line in g.lines}


def test_the_checklist_lists_what_is_true_with_action_links(fx):
    lines = lines_of(fx.home)
    assert lines["needs_notes"] == ("2 member papers need notes",
                                    f"/library/write-notes?view=needs-notes&graph={GRAPH}")
    assert lines["held_back"] == ("2 papers held back", f"/library?view=held-back&graph={GRAPH}")
    assert lines["attention"] == ("3 attention items on this graph's papers", f"/review/attention?graph={GRAPH}")
    assert "paused" not in " ".join(k for k in lines)
    assert "concepts" not in lines


def test_a_paused_batch_and_stale_concepts_appear_and_go_away(fx):
    with_lib(fx.home, lambda lib: routine.stop(lib, fx.open_batch, mode="now"))
    def stale_concepts(lib):
        with lib.write() as conn:
            conn.execute("INSERT INTO concept_refresh (paper_id, reason, since) VALUES (?, 'edited', '2026-01-01')",
                         (fx.noted,))

    with_lib(fx.home, stale_concepts)
    lines = lines_of(fx.home)
    label = with_lib(fx.home, lambda lib: state.batch_state(lib, fx.open_batch).label)
    assert lines[f"batch:{fx.open_batch}"] == (f"batch {fx.open_batch}: {label}", f"/batches/{fx.open_batch}")
    assert lines["concepts"][0] == "concepts out of date (1 paper changed)"
    with_lib(fx.home, lambda lib: routine.continue_batch(lib, fx.open_batch))
    assert f"batch:{fx.open_batch}" not in lines_of(fx.home)


def test_the_home_page_renders_the_checklist_and_feed(fx, make_app, client_for):
    html = client_for(make_app()).get("/").text
    assert "Invented widgets" in html and "2 member papers need notes" in html
    assert f'href="/library/write-notes?view=needs-notes&amp;graph={GRAPH}"' in html
    assert 'href="/" aria-current="page"' in html
    assert f'href="/batches/{fx.done_batch}"' in html  # the feed links its batches
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)


def test_the_feed_shows_batch_events_with_the_state_label(fx):
    view = with_lib(fx.home, lambda lib: home_view.activity(lib))
    kinds = [e.what for e in view.entries if e.batch_id == fx.done_batch]
    assert "submitted" in kinds and "claimed" in kinds
    labels = {e.batch_id: e.label for e in view.entries if e.batch_id}
    assert labels[fx.done_batch] == with_lib(fx.home, lambda lib: state.batch_state(lib, fx.done_batch).label)


def test_the_feed_polls_every_10s_only_while_a_batch_is_working(fx, make_app, client_for):
    c = client_for(make_app())
    idle = c.get("/home/activity").text
    assert "<html" not in idle and "hx-trigger" not in idle
    with_lib(fx.home, lambda lib: routine.claim_next(lib, fx.open_batch, WorkerRef("interactive", "w9")))
    working = c.get("/home/activity").text
    assert 'hx-trigger="every 10s"' in working and 'hx-get="/home/activity"' in working
    assert re.search(r"Working: 1 agent", working)


def test_home_without_graphs_points_to_the_library(make_app, client_for):
    html = client_for(make_app()).get("/").text
    assert "No context graphs yet" in html and 'href="/library"' in html
