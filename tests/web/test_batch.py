"""Batches (design §2.5): the list and the detail page (per-paper state, anchored/total
quotes, claim history, copyable commands, stop / continue), the polling partial, and the
pause labels exactly as ``work.state`` words them."""

from __future__ import annotations

from datetime import timedelta

import pytest

from seedgraph.library.store import open_library_at
from seedgraph.web.views import batch as batch_view
from seedgraph.work import routine, state
from seedgraph.work.claims import WorkerRef
from seedgraph.work.commands import work_commands

from .pages_support import T0, build, no_inline_script_or_style, post_forms_lead_with_csrf, token


@pytest.fixture
def fx(home):
    return build(home)


def with_lib(home, fn):
    lib = open_library_at(home / "library")
    try:
        return fn(lib)
    finally:
        lib.close()


def label_of(home, batch_id, now=None):
    return with_lib(home, lambda lib: state.batch_state(lib, batch_id, now=now).label)


def test_the_batch_list_shows_each_batch_with_its_state_and_counts(fx, make_app, client_for):
    rows = with_lib(fx.home, lambda lib: batch_view.list_page(lib))
    by_id = {r.batch_id: r for r in rows}
    assert by_id[fx.done_batch].done == 1 and by_id[fx.done_batch].state == "finished"
    assert by_id[fx.open_batch].label == "Waiting for an agent"
    html = client_for(make_app()).get("/batches").text
    assert f'href="/batches/{fx.done_batch}"' in html and "Waiting for an agent" in html
    assert no_inline_script_or_style(html)


def test_the_detail_page_shows_rows_quotes_claim_history_and_commands(fx, make_app, client_for):
    html = client_for(make_app()).get(f"/batches/{fx.done_batch}").text
    cmds = work_commands(fx.done_batch)
    assert cmds.claude_code in html
    assert "data-copy" in html  # copy buttons, handled by a static script
    assert f'href="/papers/{fx.noted}"' in html and f'href="/papers/{fx.noted}/pdf"' in html
    assert "5/5 quotes anchored" in html  # good_note's five quoted claims all anchor
    assert "w1" in html and "claude" in html and "submitted" in html  # claim history
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)


def test_a_live_claim_shows_who_holds_the_paper(fx, make_app, client_for):
    with_lib(fx.home, lambda lib: routine.claim_next(lib, fx.open_batch, WorkerRef("interactive", "w2"), now=T0))
    html = client_for(make_app(clock=lambda: T0 + timedelta(seconds=5))).get(f"/batches/{fx.open_batch}").text
    assert "claimed by w2" in html
    assert label_of(fx.home, fx.open_batch, now=T0 + timedelta(seconds=5)) in html


def test_an_expired_claim_reads_with_work_states_label(fx, make_app, client_for):
    with_lib(fx.home, lambda lib: routine.claim_next(lib, fx.open_batch, WorkerRef("interactive", "w2"), now=T0))
    later = T0 + timedelta(days=2)
    html = client_for(make_app(clock=lambda: later)).get(f"/batches/{fx.open_batch}").text
    expected = label_of(fx.home, fx.open_batch, now=later)
    assert expected.startswith("Paused: a worker stopped responding") and expected in html


def test_a_usage_limit_reads_with_work_states_label(fx, make_app, client_for):
    with_lib(fx.home, lambda lib: routine.report_limit(lib, fx.open_batch, WorkerRef("interactive", "w2"),
                                                        resets_at="2031-01-01T00:00:00+00:00", now=T0))
    c = client_for(make_app())
    expected = label_of(fx.home, fx.open_batch)
    assert expected.startswith("Paused: usage limit, resets 2031")
    assert expected in c.get(f"/batches/{fx.open_batch}").text
    assert expected in c.get("/batches").text


@pytest.mark.parametrize("mode,label", [("finish", "Stopped (finishing current papers"), ("now", "Stopped now")])
def test_stop_calls_routine_stop_once_and_needs_csrf(fx, make_app, client_for, monkeypatch, mode, label):
    calls = []
    real = routine.stop

    def spy(lib, batch_id, *, mode, now=None):
        calls.append((batch_id, mode))
        return real(lib, batch_id, mode=mode, now=now)

    monkeypatch.setattr(routine, "stop", spy)
    c = client_for(make_app())
    url = f"/batches/{fx.open_batch}/stop"
    assert c.post(url, data={"mode": mode}).status_code == 403
    r = c.post(url, data={"csrf_token": token(c), "mode": mode})
    assert r.status_code == 303 and calls == [(fx.open_batch, mode)]
    assert label in c.get(r.headers["location"]).text
    assert label_of(fx.home, fx.open_batch).startswith(label)


def test_continue_calls_routine_continue_once(fx, make_app, client_for, monkeypatch):
    with_lib(fx.home, lambda lib: routine.stop(lib, fx.open_batch, mode="now"))
    calls = []
    real = routine.continue_batch

    def spy(lib, batch_id, *, now=None):
        calls.append(batch_id)
        return real(lib, batch_id, now=now)

    monkeypatch.setattr(routine, "continue_batch", spy)
    c = client_for(make_app())
    r = c.post(f"/batches/{fx.open_batch}/continue", data={"csrf_token": token(c)})
    assert r.status_code == 303 and calls == [fx.open_batch]
    assert label_of(fx.home, fx.open_batch) == "Waiting for an agent"


def test_a_bad_stop_mode_is_refused(fx, make_app, client_for):
    c = client_for(make_app())
    assert c.post(f"/batches/{fx.open_batch}/stop", data={"csrf_token": token(c), "mode": "later"}).status_code == 400


def test_the_items_partial_polls_until_the_batch_finishes(fx, make_app, client_for):
    c = client_for(make_app())
    working = c.get(f"/batches/{fx.open_batch}/items")
    assert working.status_code == 200 and "<html" not in working.text
    assert 'hx-trigger="every 5s"' in working.text
    assert f'hx-get="/batches/{fx.open_batch}/items"' in working.text
    finished = c.get(f"/batches/{fx.done_batch}/items").text
    assert "hx-trigger" not in finished


def test_unknown_batches_are_404(fx, make_app, client_for):
    c = client_for(make_app())
    for path in ("/batches/b_nope", "/batches/b_nope/items"):
        assert c.get(path).status_code == 404
    assert c.post("/batches/b_nope/stop", data={"csrf_token": token(c), "mode": "now"}).status_code == 404
