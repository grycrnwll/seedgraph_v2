"""Track 2: the create-project + seed-PDF-upload ``/ui`` forms.

Both POSTs are fail-closed twice — ``require_local_session`` (loopback + session
cookie) AND ``require_csrf`` (``_csrf == app.state.csrf_token``). Offline / keyless:
the FastAPI surface is exercised via Starlette's ``TestClient`` with the loopback
gate replaced by the ``dependency_overrides`` seam and the CSRF token pinned to a
fixed value. ``SEEDGRAPH_FAKE_MARKER=1`` makes PDF→markdown conversion deterministic
(the offline :class:`FakeMarkerBackend`), so an uploaded seed gains ``has_markdown``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from seedgraph.acquisition.service import corpus_rows
from seedgraph.api.app import app
from seedgraph.project import service
from seedgraph.web import marker_queue, serve

CSRF = "fixed-csrf-token-for-tests"
PDF_BYTES = b"%PDF-1.4\n%fake seed pdf for the upload test\n"


@pytest.fixture
def authed(monkeypatch):
    """TestClient with the loopback gate overridden, a fixed CSRF token, fake marker."""
    monkeypatch.setenv("SEEDGRAPH_FAKE_MARKER", "1")
    app.dependency_overrides[serve.require_local_session] = lambda: None
    prev = getattr(app.state, "csrf_token", None)
    app.state.csrf_token = CSRF
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(serve.require_local_session, None)
        app.state.csrf_token = prev


# --------------------------------------------------------------------------
# 1. POST /ui/projects creates the project and 303→ its corpus.
# --------------------------------------------------------------------------

def test_create_project_makes_project_and_redirects_to_corpus(authed):
    assert "uitest" not in service.list_projects()

    resp = authed.post(
        "/ui/projects",
        data={"_csrf": CSRF, "slug": "uitest", "name": "UI Test", "description": "made via UI"},
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text[:200]
    assert resp.headers["location"] == "/ui/projects/uitest/corpus"

    # The project now exists (service truth + the landing page lists it).
    assert "uitest" in service.list_projects()
    landing = authed.get("/ui")
    assert "uitest" in landing.text

    # Following the redirect lands on the (now-reachable) corpus page.
    followed = authed.post(
        "/ui/projects",
        data={"_csrf": CSRF, "slug": "uitest2"},
        follow_redirects=True,
    )
    assert followed.status_code == 200
    assert str(followed.url).endswith("/ui/projects/uitest2/corpus")


# --------------------------------------------------------------------------
# 2. POST /ui/projects/{slug}/upload makes a converted seed work.
# --------------------------------------------------------------------------

def test_upload_pdf_creates_converted_seed_work(authed, monkeypatch):
    from seedgraph.acquisition.service import convert_and_bridge

    # The upload route now enqueues a per-GPU SUBPROCESS conversion; monkeypatch the
    # dispatch seam to an in-process convert so the test stays fast + torch-free.
    def _in_process(slug, work_id, source_file_id, file_hash, *, root=None, gpu_index=None):
        h2 = service.open_project(slug, root=root)
        convert_and_bridge(
            h2, work_id=work_id, source_file_id=source_file_id,
            file_hash=file_hash, cache_root=root,
        )

    monkeypatch.setattr(marker_queue, "_run_conversion", _in_process)
    marker_queue._reset_for_tests()

    h = service.create_project("uploadproj")
    assert corpus_rows(h) == []  # empty corpus before upload

    resp = authed.post(
        "/ui/projects/uploadproj/upload",
        data={"_csrf": CSRF},
        files={"files": ("seed_paper_one.pdf", PDF_BYTES, "application/pdf")},
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text[:200]
    assert resp.headers["location"] == (
        "/ui/projects/uploadproj/corpus?queued=1&failed=0"
    )

    # Conversion now runs on the background queue -- drain it before asserting markdown.
    assert marker_queue.wait_idle(timeout=60) is True
    rows = corpus_rows(h)
    assert len(rows) == 1
    row = rows[0]
    assert row["is_seed"] is True
    assert row["inclusion_status"] == "included"
    assert row["title"] == "seed paper one"  # _/- → spaces, extension dropped
    assert row["has_markdown"] is True  # fake marker converted it deterministically


# --------------------------------------------------------------------------
# 3. Both POSTs 403 without the _csrf field (CSRF gate holds).
# --------------------------------------------------------------------------

def test_posts_reject_missing_csrf(authed):
    service.create_project("csrfproj")

    create = authed.post(
        "/ui/projects", data={"slug": "no_csrf"}, follow_redirects=False
    )
    assert create.status_code == 403
    assert "no_csrf" not in service.list_projects()  # nothing was created

    upload = authed.post(
        "/ui/projects/csrfproj/upload",
        files={"files": ("x.pdf", PDF_BYTES, "application/pdf")},
        follow_redirects=False,
    )
    assert upload.status_code == 403
    assert corpus_rows(service.open_project("csrfproj")) == []


# --------------------------------------------------------------------------
# 4. A non-PDF upload is skipped — no work is created.
# --------------------------------------------------------------------------

def test_non_pdf_upload_is_skipped(authed):
    h = service.create_project("skipproj")

    resp = authed.post(
        "/ui/projects/skipproj/upload",
        data={"_csrf": CSRF},
        files={"files": ("notes.txt", b"this is not a pdf", "text/plain")},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == (
        "/ui/projects/skipproj/corpus?queued=0&failed=1"
    )
    assert corpus_rows(h) == []  # nothing created for a non-PDF
