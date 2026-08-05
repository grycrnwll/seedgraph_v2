"""Track 2: the ``/ui/projects/{slug}/ingest-folder`` bulk auto-match drop-zone.

Mirrors ``test_web_create_upload.py``: the loopback gate is replaced via
``dependency_overrides`` and the CSRF token is pinned; ``SEEDGRAPH_FAKE_MARKER=1``
makes PDF→markdown deterministic (the offline :class:`FakeMarkerBackend`, whose
default markdown H1 is ``Title``). Conversion is SYNCHRONOUS on this route (no
marker queue), so an uploaded PDF is matched + bridged within the request.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from seedgraph.acquisition.service import corpus_rows
from seedgraph.api.app import app
from seedgraph.project import review, service
from seedgraph.web import serve

CSRF = "fixed-csrf-token-for-tests"
PDF_BYTES = b"%PDF-1.4\n%fake pdf for the ingest-folder test\n"


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
# 1. A pending metadata_only work is auto-matched + bridged by an uploaded PDF.
# --------------------------------------------------------------------------
def test_ingest_folder_matches_and_bridges(authed):
    h = service.create_project("ingmatch")
    # The fake-marker default markdown H1 is "Title" -> matches this work by title.
    service.add_work(h, title="Title", inclusion_status="metadata_only")

    resp = authed.post(
        "/ui/projects/ingmatch/ingest-folder",
        data={"_csrf": CSRF},
        files={"files": ("some_arbitrary_name.pdf", PDF_BYTES, "application/pdf")},
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text[:200]
    assert "matched=1" in resp.headers["location"]

    rows = corpus_rows(h)
    assert len(rows) == 1
    assert rows[0]["has_markdown"] is True
    assert rows[0]["inclusion_status"] == "included"  # promoted from metadata_only
    assert rows[0]["access_status"] == "user_supplied_private"


# --------------------------------------------------------------------------
# 2. A PDF matching no pending work routes to review (unmatched_upload).
# --------------------------------------------------------------------------
def test_ingest_folder_miss_routes_to_review(authed):
    h = service.create_project("ingmiss")  # no work titled "Title"

    resp = authed.post(
        "/ui/projects/ingmiss/ingest-folder",
        data={"_csrf": CSRF},
        files={"files": ("miss.pdf", PDF_BYTES, "application/pdf")},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "review=1" in resp.headers["location"]

    items = review.list_open(h)
    assert len(items) == 1
    assert items[0].item_type == "unmatched_upload"


# --------------------------------------------------------------------------
# 3. The CSRF gate holds (403 without _csrf; nothing bridged).
# --------------------------------------------------------------------------
def test_ingest_folder_rejects_missing_csrf(authed):
    h = service.create_project("ingcsrf")
    service.add_work(h, title="Title", inclusion_status="metadata_only")

    resp = authed.post(
        "/ui/projects/ingcsrf/ingest-folder",
        files={"files": ("x.pdf", PDF_BYTES, "application/pdf")},
        follow_redirects=False,
    )
    assert resp.status_code == 403
    rows = corpus_rows(h)
    assert rows[0]["has_markdown"] is False  # nothing was bridged


# --------------------------------------------------------------------------
# 4. A non-PDF upload is skipped (failed), no work touched.
# --------------------------------------------------------------------------
def test_ingest_folder_skips_non_pdf(authed):
    h = service.create_project("ingskip")

    resp = authed.post(
        "/ui/projects/ingskip/ingest-folder",
        data={"_csrf": CSRF},
        files={"files": ("notes.txt", b"this is not a pdf", "text/plain")},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert "failed=1" in resp.headers["location"]
    assert corpus_rows(h) == []
