"""Track 2 Stage A: read-only ``/ui`` screens + ``/api`` JSON helpers + the
service-extraction parity these screens are built on.

Offline / keyless: the FastAPI surface is exercised via Starlette's ``TestClient``;
the gated screens render only under the ``require_local_session`` dependency
override (cross-cutting #1 seam) and fail-closed (403) without it. Every screen is
rendered over a throwaway ``SEEDGRAPH_HOME`` (the ``isolated_home`` autouse fixture).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from seedgraph.api.app import app
from seedgraph.config.loader import load_global_config, write_global_config
from seedgraph.errors import ConfigError
from seedgraph.project import service
from seedgraph.web import serve


@pytest.fixture
def screen_project():
    """A project with one included work (so the corpus drawer has a row to render)."""
    h = service.create_project("screenproj")
    w = service.add_work(h, ids={"doi": "10.1/x"}, title="Sample Paper", inclusion_status="included")
    return "screenproj", w.work_id


def _override():
    app.dependency_overrides[serve.require_local_session] = lambda: None


def _clear():
    app.dependency_overrides.pop(serve.require_local_session, None)


def _paths(slug: str, work_id: str) -> list[tuple[str, str]]:
    """(path, known marker) for every gated read-only screen + JSON helper."""
    return [
        ("/ui/setup", "First-run setup"),
        (f"/ui/projects/{slug}", "Dashboard"),
        (f"/ui/projects/{slug}/corpus", "Corpus"),
        (f"/ui/projects/{slug}/corpus/{work_id}", "Document detail"),
        (f"/ui/projects/{slug}/ask", "Ask"),
        (f"/ui/projects/{slug}/ask?q=parallel%20trends", "Citations"),
        (f"/ui/projects/{slug}/lenses", "Lenses"),
        (f"/ui/projects/{slug}/lenses/regularity_v1", "Coverage"),
        ("/ui/settings", "Settings"),
        (f"/ui/projects/{slug}/settings", "Settings"),
    ]


def _api_paths(slug: str) -> list[str]:
    return [
        f"/api/projects/{slug}/dashboard",
        f"/api/projects/{slug}/corpus",
        f"/api/projects/{slug}/lenses",
        f"/api/projects/{slug}/lenses/regularity_v1",
        "/api/settings",
        f"/api/projects/{slug}/settings",
    ]


# --------------------------------------------------------------------------
# Each gated GET renders (200 + marker) under the session override …
# --------------------------------------------------------------------------

def test_screens_render_with_session(screen_project):
    slug, work_id = screen_project
    _override()
    try:
        client = TestClient(app)
        for path, marker in _paths(slug, work_id):
            resp = client.get(path)
            assert resp.status_code == 200, (path, resp.text[:200])
            assert marker in resp.text, (path, marker)
    finally:
        _clear()


def test_api_helpers_return_json_with_session(screen_project):
    slug, _ = screen_project
    _override()
    try:
        client = TestClient(app)
        for path in _api_paths(slug):
            resp = client.get(path)
            assert resp.status_code == 200, (path, resp.text[:200])
            assert resp.headers["content-type"].startswith("application/json")
    finally:
        _clear()


# --------------------------------------------------------------------------
# … and fail-closed (403) without it (the gate is actually wired).
# --------------------------------------------------------------------------

def test_screens_refuse_without_session(screen_project):
    slug, work_id = screen_project
    client = TestClient(app)  # host "testclient" (non-loopback), no sg_session cookie
    for path, _marker in _paths(slug, work_id):
        assert client.get(path).status_code == 403, path
    for path in _api_paths(slug):
        assert client.get(path).status_code == 403, path


def test_project_list_landing_stays_ungated():
    # The /ui project list shows slugs + links only (no private evidence), so it is
    # the reachable entry point even without a cookie; every screen it links to is gated.
    service.create_project("landingproj")
    client = TestClient(app)
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "landingproj" in resp.text


def test_corpus_drawer_unknown_work_is_404_not_path_leak(screen_project):
    slug, _ = screen_project
    _override()
    try:
        client = TestClient(app)
        resp = client.get(f"/ui/projects/{slug}/corpus/work_does_not_exist")
        assert resp.status_code == 404
        assert "work_does_not_exist" not in resp.text  # never echo the work id into a path
    finally:
        _clear()


# --------------------------------------------------------------------------
# Service-extraction parity (the UI and CLI share one source of truth).
# --------------------------------------------------------------------------

def test_corpus_rows_carries_metadata_and_flags(screen_project):
    from seedgraph.acquisition.service import corpus_rows

    slug, work_id = screen_project
    h = service.open_project(slug)
    rows = corpus_rows(h)
    assert len(rows) == 1
    row = rows[0]
    # the dense CLI columns …
    for key in ("work_id", "title", "inclusion_status", "access_status", "acquisition_state", "has_markdown"):
        assert key in row
    # … plus the enrichment flags the screen needs.
    for key in ("has_citations", "has_extraction", "has_concepts", "has_review"):
        assert row[key] is False
    assert row["work_id"] == work_id
    assert row["has_markdown"] is False


def test_project_dashboard_shape(screen_project):
    slug, _ = screen_project
    h = service.open_project(slug)
    dash = service.project_dashboard(h)
    assert dash["status_counts"].get("included") == 1
    assert dash["extraction"]["works_included"] == 1
    assert dash["open_reviews"] == 0
    assert isinstance(dash["models"], list) and dash["models"]
    assert "monthly_spend_usd" in dash["budget"]
    assert set(dash["doctor"]) >= {"ok", "n_fail", "n_warn"}


def test_settings_view_never_exposes_a_raw_secret(monkeypatch, screen_project):
    from seedgraph.web.routes import settings_view

    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret-value-xyz")
    view = settings_view()
    # key presence is by NAME + a boolean, never the value.
    blob = repr(view)
    assert "super-secret-value-xyz" not in blob
    anthropic = next(p for p in view["profiles"] if p["provider"] == "anthropic")
    assert anthropic["env_var"] == "ANTHROPIC_API_KEY"
    assert anthropic["key_present"] is True


def test_write_global_config_round_trips_and_validates():
    cfg = write_global_config({"log_level": "DEBUG"})
    assert cfg.log_level == "DEBUG"
    # persisted as a thin overlay that load re-merges over the defaults.
    assert load_global_config().log_level == "DEBUG"


def test_write_global_config_rejects_raw_secret():
    with pytest.raises(ConfigError):
        write_global_config({"llm": {"profiles": {"x": {"api_key": "sk-leak"}}}})


def test_write_global_config_rejects_invalid_patch():
    # a structurally invalid patch raises and writes nothing.
    with pytest.raises(ConfigError):
        write_global_config({"llm": {"routes": {"t": {"task_type": "t", "preferred_profile": "ghost"}}}})
    assert load_global_config().log_level == "INFO"  # unchanged


# --------------------------------------------------------------------------
# Build D ch12 — missing-paper frontier pane (K-of-N, triage sort, per-row
# upload, resolver links, escaping). Gap scan §5.6 gaps 2+3.
# --------------------------------------------------------------------------

FRONTIER_CSRF = "frontier-csrf-token"
FRONTIER_PDF = b"%PDF-1.4\n%frontier provided-work pdf\n"
FRONTIER_PDF2 = b"%PDF-1.4\n%frontier per-row-upload pdf (distinct hash)\n"


@pytest.fixture
def frontier_client(monkeypatch):
    """Authed TestClient: session override + pinned CSRF + fake marker backend."""
    monkeypatch.setenv("SEEDGRAPH_FAKE_MARKER", "1")
    _override()
    prev = getattr(app.state, "csrf_token", None)
    app.state.csrf_token = FRONTIER_CSRF
    try:
        yield TestClient(app)
    finally:
        _clear()
        app.state.csrf_token = prev


@pytest.fixture
def frontier_project(tmp_path, monkeypatch):
    """One provided + two missing works: an OA-available stub (created FIRST so
    the pane's rank sort — not insertion order — must move it last) and a
    genuinely-paywalled included work (``requires_user_upload``)."""
    from sqlmodel import Session

    from seedgraph.acquisition.service import manual_upload
    from seedgraph.db.project_models import Work

    monkeypatch.setenv("SEEDGRAPH_FAKE_MARKER", "1")
    h = service.create_project("frontierproj")
    provided = service.add_work(
        h, ids={"doi": "10.1/provided"}, title="Provided Paper", inclusion_status="included"
    )
    pdf = tmp_path / "provided.pdf"
    pdf.write_bytes(FRONTIER_PDF)
    manual_upload(h, work_id=provided.work_id, pdf_path=pdf)  # → has_markdown=True
    oa = service.add_work(
        h, ids={"doi": "10.1/oa"}, title="Open Access Paper", inclusion_status="metadata_only"
    )
    with Session(h.engine) as session:
        w = session.get(Work, oa.work_id)
        w.oa_status = "gold"  # un-bridged + open OA color → available_open_access
        session.add(w)
        session.commit()
    paywalled = service.add_work(
        h, ids={"doi": "10.1/pay"}, title="Paywalled Paper", inclusion_status="included"
    )
    return h, provided.work_id, paywalled.work_id, oa.work_id


def test_frontier_pane_renders_k_of_n_and_rows(frontier_project, frontier_client):
    _h, provided, paywalled, oa = frontier_project
    resp = frontier_client.get("/ui/projects/frontierproj/corpus")
    assert resp.status_code == 200
    text = resp.text
    assert "<strong>1</strong> of <strong>3</strong> provided" in text
    assert f'data-frontier-work-id="{paywalled}"' in text
    assert f'data-frontier-work-id="{oa}"' in text
    assert f'data-frontier-work-id="{provided}"' not in text  # provided rows never park
    assert "Paywalled Paper" in text and "Open Access Paper" in text


def test_frontier_available_open_access_sorts_last(frontier_project, frontier_client):
    _h, _provided, paywalled, oa = frontier_project
    text = frontier_client.get("/ui/projects/frontierproj/corpus").text
    # the OA stub was CREATED before the paywalled work, so created_at order
    # would list it first — the v1 _KLASS_RANK triage sort must move it last.
    assert text.index(f'data-frontier-work-id="{paywalled}"') < text.index(
        f'data-frontier-work-id="{oa}"'
    )
    assert "available_open_access" in text  # the triage badge renders


def test_frontier_upload_bridges_work_and_drops_row(frontier_project, frontier_client):
    from seedgraph.acquisition.service import corpus_rows

    h, _provided, paywalled, oa = frontier_project
    resp = frontier_client.post(
        f"/ui/projects/frontierproj/corpus/{paywalled}/upload",
        data={"_csrf": FRONTIER_CSRF},
        files={"pdf": ("obtained.pdf", FRONTIER_PDF2, "application/pdf")},
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text[:200]
    assert resp.headers["location"] == "/ui/projects/frontierproj/corpus"
    row = next(r for r in corpus_rows(h) if r["work_id"] == paywalled)
    assert row["has_markdown"] is True  # bridged via the existing manual_upload
    text = frontier_client.get("/ui/projects/frontierproj/corpus").text
    assert f'data-frontier-work-id="{paywalled}"' not in text  # the pane dropped it
    assert f'data-frontier-work-id="{oa}"' in text  # the other missing row remains
    assert "<strong>2</strong> of <strong>3</strong> provided" in text


def test_frontier_upload_rejects_missing_csrf_and_unknown_work(
    frontier_project, frontier_client
):
    from seedgraph.acquisition.service import corpus_rows

    h, _provided, paywalled, _oa = frontier_project
    no_csrf = frontier_client.post(
        f"/ui/projects/frontierproj/corpus/{paywalled}/upload",
        files={"pdf": ("x.pdf", FRONTIER_PDF2, "application/pdf")},
        follow_redirects=False,
    )
    assert no_csrf.status_code == 403
    row = next(r for r in corpus_rows(h) if r["work_id"] == paywalled)
    assert row["has_markdown"] is False  # nothing was bridged
    unknown = frontier_client.post(
        "/ui/projects/frontierproj/corpus/work_does_not_exist/upload",
        data={"_csrf": FRONTIER_CSRF},
        files={"pdf": ("x.pdf", FRONTIER_PDF2, "application/pdf")},
        follow_redirects=False,
    )
    assert unknown.status_code == 404  # membership-validated: no orphan bridge


def test_frontier_public_links_only_when_configs_unset(frontier_project, frontier_client):
    text = frontier_client.get("/ui/projects/frontierproj/corpus").text
    assert "https://doi.org/10.1/pay" in text  # public resolver renders
    # neither institutional config is set → nothing beyond the public resolvers.
    assert "OpenURL" not in text
    assert "EZproxy" not in text


def test_frontier_institutional_links_thread_from_effective_config(
    frontier_project, frontier_client
):
    # threaded from the merged inheritance-resolved config (config.loader), so a
    # global setting reaches every project page without a project override.
    write_global_config({"openurl_resolver": "https://resolver.example.edu/openurl"})
    text = frontier_client.get("/ui/projects/frontierproj/corpus").text
    assert "OpenURL" in text
    assert "resolver.example.edu" in text


def test_frontier_titles_are_escaped(frontier_client):
    h = service.create_project("escproj")
    service.add_work(
        h,
        ids={"doi": "10.1/esc"},
        title='<script>alert("x")</script>',
        inclusion_status="included",
    )
    text = frontier_client.get("/ui/projects/escproj/corpus").text
    assert '<script>alert("x")</script>' not in text  # no raw HTML injection
    assert "&lt;script&gt;" in text  # Jinja autoescape holds on the pane
