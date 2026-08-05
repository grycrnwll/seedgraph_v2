"""Track 2: the corpus-page aggregate conversion bar (render + JS contract).

Mirrors ``test_web_create_upload.py``'s fixture (loopback gate override + fixed CSRF
+ ``SEEDGRAPH_FAKE_MARKER``). Asserts that ``ui_corpus`` renders the ``data-conv-bar``
progress element folded from ``conversion_summary(rows, marker_queue.status(slug))``,
and that the ``/upload/status`` endpoint the JS polls still returns the flat
``{work_id: {state, detail}}`` map (unchanged shape — the bar is computed, not served).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from seedgraph.acquisition.service import corpus_rows
from seedgraph.api.app import app
from seedgraph.project import service
from seedgraph.web import marker_queue, serve

CSRF = "fixed-csrf-token-for-tests"
PDF_BYTES = b"%PDF-1.4\n%fake seed pdf for the corpus-bar test\n"


@pytest.fixture
def authed(monkeypatch):
    monkeypatch.setenv("SEEDGRAPH_FAKE_MARKER", "1")
    app.dependency_overrides[serve.require_local_session] = lambda: None
    prev = getattr(app.state, "csrf_token", None)
    app.state.csrf_token = CSRF
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(serve.require_local_session, None)
        app.state.csrf_token = prev


def _in_process_convert(monkeypatch):
    from seedgraph.acquisition.service import convert_and_bridge

    def _run(slug, work_id, source_file_id, file_hash, *, root=None, gpu_index=None):
        h2 = service.open_project(slug, root=root)
        convert_and_bridge(
            h2, work_id=work_id, source_file_id=source_file_id,
            file_hash=file_hash, cache_root=root,
        )

    monkeypatch.setattr(marker_queue, "_run_conversion", _run)
    marker_queue._reset_for_tests()


def test_corpus_bar_renders_converted_over_total(authed, monkeypatch):
    """One converted seed + one held-`queued` conversion → the bar shows 1/2."""
    _in_process_convert(monkeypatch)
    h = service.create_project("convbar")

    # Upload + drain one seed → one converted (has_markdown) work.
    resp = authed.post(
        "/ui/projects/convbar/upload",
        data={"_csrf": CSRF},
        files={"files": ("paper_one.pdf", PDF_BYTES, "application/pdf")},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert marker_queue.wait_idle(timeout=60) is True
    assert sum(1 for r in corpus_rows(h) if r["has_markdown"]) == 1

    # Hold one conversion 'queued' (not on the FIFO → the pool never drains it).
    with marker_queue._LOCK:
        marker_queue._STATUS[("convbar", "work_pending")] = {"state": "queued", "detail": ""}

    page = authed.get("/ui/projects/convbar/corpus")
    assert page.status_code == 200
    body = page.text
    # The aggregate bar: converted=1, pending=1 → total=2.
    assert "data-conv-bar" in body
    assert 'max="2"' in body
    assert 'value="1"' in body
    assert "1/2 converted" in body

    # The poller endpoint the JS reads still returns the FLAT map, unchanged shape.
    status = authed.get("/ui/projects/convbar/upload/status")
    assert status.status_code == 200
    assert status.json()["work_pending"] == {"state": "queued", "detail": ""}


def test_corpus_bar_absent_when_nothing_to_convert(authed, monkeypatch):
    """No markdown + empty queue → total==0 and failed==0 → no bar rendered."""
    _in_process_convert(monkeypatch)
    service.create_project("nobar")
    page = authed.get("/ui/projects/nobar/corpus")
    assert page.status_code == 200
    assert "data-conv-bar" not in page.text
