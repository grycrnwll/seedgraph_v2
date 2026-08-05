"""Track 2 Stage B: mutating ``/ui`` forms + CSRF synchronizer-token enforcement.

Every mutating POST is fail-closed twice: ``require_local_session`` (loopback +
session cookie) AND ``require_csrf`` (``_csrf == app.state.csrf_token``). Each route
runs the SAME validation as its matching CLI command and returns a 303 redirect.

Offline / keyless: the FastAPI surface is exercised via Starlette's ``TestClient``.
The loopback gate is replaced with the ``dependency_overrides`` seam (cross-cutting
#1); the CSRF token is pinned to a fixed value so a posted ``_csrf`` can match. The
gate is left UN-overridden in the no-session test so the real 403 path is exercised.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from seedgraph.acquisition.service import corpus_rows
from seedgraph.api.app import app
from seedgraph.errors import SeedgraphError
from seedgraph.lenses import registry
from seedgraph.project import review as review_mod
from seedgraph.project import service
from seedgraph.web import serve

CSRF = "fixed-csrf-token-for-tests"


@pytest.fixture
def forms_project():
    """A project with one included work, a templated lens, and an open review item."""
    from sqlmodel import Session

    h = service.create_project("formproj")
    w = service.add_work(
        h, ids={"doi": "10.1/x"}, title="Sample Paper", inclusion_status="included"
    )
    project_dir = h.root / "projects" / h.slug
    with Session(h.engine) as session:
        registry.create_lens_from_template(
            session, project_dir, "regularity_conditions_v1", "regularity_conditions_v1"
        )
    item_id = review_mod.enqueue(h, "duplicate_candidate", target_id=w.work_id)
    return {
        "slug": "formproj",
        "work_id": w.work_id,
        "lens_id": "regularity_conditions_v1",
        "item_id": item_id,
        "handle": h,
        "project_dir": project_dir,
    }


@pytest.fixture
def authed():
    """TestClient with the loopback gate overridden + a fixed CSRF token on app.state."""
    app.dependency_overrides[serve.require_local_session] = lambda: None
    prev = getattr(app.state, "csrf_token", None)
    app.state.csrf_token = CSRF
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(serve.require_local_session, None)
        app.state.csrf_token = prev


def _lens_yaml(fx) -> str:
    return registry.lens_yaml_path(fx["project_dir"], fx["lens_id"]).read_text(
        encoding="utf-8"
    )


def _valid_posts(fx) -> list[tuple[str, dict]]:
    """(path, form-data) for every mutating POST with a *valid* payload. ``_csrf`` is
    injected by the caller so the missing-CSRF case can reuse the same list."""
    slug, work_id, lens_id = fx["slug"], fx["work_id"], fx["lens_id"]
    return [
        ("/ui/setup", {"log_level": "INFO"}),
        ("/ui/settings", {"monthly_soft_limit_usd": "5"}),
        (f"/ui/projects/{slug}/settings", {"project_name": "Renamed"}),
        (
            f"/ui/projects/{slug}/corpus/{work_id}/status",
            {"status": "metadata_only", "reason": "ui edit"},
        ),
        (
            f"/ui/projects/{slug}/lenses",
            {"lens_id": "regularity_conditions_v2", "from_template": "regularity_conditions_v1"},
        ),
        (
            f"/ui/projects/{slug}/lenses/{lens_id}/validate",
            {"yaml": _lens_yaml(fx)},
        ),
        # 'reject' — a pure status flip; 'merge' now mutates the graph and refuses
        # this fixture's payload-less item (Build A ch7). The merge lift itself is
        # covered by tests/test_identity_merge_correctness.py.
        (f"/ui/projects/{slug}/review/{fx['item_id']}/resolve", {"action": "reject"}),
    ]


# --------------------------------------------------------------------------
# Each POST redirects (303) with a valid cookie override + matching _csrf.
# --------------------------------------------------------------------------

def test_each_post_redirects_with_session_and_csrf(forms_project, authed):
    for path, data in _valid_posts(forms_project):
        resp = authed.post(path, data={**data, "_csrf": CSRF}, follow_redirects=False)
        assert resp.status_code == 303, (path, resp.status_code, resp.text[:200])
        assert resp.headers["location"]  # 303 See Other carries a redirect target


# --------------------------------------------------------------------------
# 403 with a missing or wrong _csrf (session is present via the override).
# --------------------------------------------------------------------------

def test_post_rejects_missing_csrf(forms_project, authed):
    for path, data in _valid_posts(forms_project):
        resp = authed.post(path, data=data, follow_redirects=False)  # no _csrf
        assert resp.status_code == 403, (path, resp.status_code)
        assert "CSRF" in resp.json()["detail"]


def test_post_rejects_wrong_csrf(forms_project, authed):
    for path, data in _valid_posts(forms_project):
        resp = authed.post(
            path, data={**data, "_csrf": "not-the-token"}, follow_redirects=False
        )
        assert resp.status_code == 403, (path, resp.status_code)


# --------------------------------------------------------------------------
# 403 without the session (the loopback gate is NOT overridden here).
# --------------------------------------------------------------------------

def test_post_refuses_without_session(forms_project):
    client = TestClient(app)  # host "testclient" (non-loopback), no sg_session cookie
    for path, data in _valid_posts(forms_project):
        resp = client.post(
            path, data={**data, "_csrf": CSRF}, follow_redirects=False
        )
        assert resp.status_code == 403, (path, resp.status_code)


# --------------------------------------------------------------------------
# Invalid slug / work_id / payload → rejected with the SAME error the CLI gives.
# --------------------------------------------------------------------------

def test_bad_status_rejected_with_cli_error(forms_project, authed):
    slug, work_id = forms_project["slug"], forms_project["work_id"]
    # The exact message the CLI surfaces (it calls the same service function).
    try:
        service.set_inclusion_status(forms_project["handle"], work_id, "bogus")
        cli_msg = None
    except SeedgraphError as exc:
        cli_msg = str(exc)
    assert cli_msg and "invalid inclusion_status" in cli_msg

    resp = authed.post(
        f"/ui/projects/{slug}/corpus/{work_id}/status",
        data={"_csrf": CSRF, "status": "bogus"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == cli_msg


def test_unknown_work_id_rejected_like_cli(forms_project, authed):
    slug = forms_project["slug"]
    try:
        service.set_inclusion_status(forms_project["handle"], "work_ghost", "excluded")
        cli_msg = None
    except SeedgraphError as exc:
        cli_msg = str(exc)
    assert cli_msg and "no membership row" in cli_msg

    resp = authed.post(
        f"/ui/projects/{slug}/corpus/work_ghost/status",
        data={"_csrf": CSRF, "status": "excluded"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == cli_msg


def test_bad_review_action_rejected_like_cli(forms_project, authed):
    slug, item_id = forms_project["slug"], forms_project["item_id"]
    try:
        review_mod.resolve(forms_project["handle"], item_id, "bogus")
        cli_msg = None
    except SeedgraphError as exc:
        cli_msg = str(exc)
    assert cli_msg and "invalid review action" in cli_msg

    resp = authed.post(
        f"/ui/projects/{slug}/review/{item_id}/resolve",
        data={"_csrf": CSRF, "action": "bogus"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == cli_msg


def test_unknown_lens_template_rejected(forms_project, authed):
    slug = forms_project["slug"]
    resp = authed.post(
        f"/ui/projects/{slug}/lenses",
        data={"_csrf": CSRF, "lens_id": "x_v1", "from_template": "no_such_template"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "unknown template" in resp.json()["detail"]


def test_invalid_lens_yaml_rejected_and_not_written(forms_project, authed):
    slug, lens_id = forms_project["slug"], forms_project["lens_id"]
    before = _lens_yaml(forms_project)
    resp = authed.post(
        f"/ui/projects/{slug}/lenses/{lens_id}/validate",
        data={"_csrf": CSRF, "yaml": "lens_id: x\nname: X\nobject_type: a\noutput_schema: {}\n"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "INVALID" in resp.json()["detail"]
    # Nothing written — the on-disk YAML is unchanged.
    assert _lens_yaml(forms_project) == before


def test_invalid_global_config_patch_rejected(forms_project, authed):
    # A ghost route profile fails the same config validation as the CLI/service.
    resp = authed.post(
        "/ui/setup",
        data={"_csrf": CSRF, "note_extraction_profile": "ghost_profile"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "ghost_profile" in resp.json()["detail"]


def test_bad_slug_project_settings_404(authed):
    resp = authed.post(
        "/ui/projects/Bad__Slug/settings",
        data={"_csrf": CSRF, "project_name": "X"},
        follow_redirects=False,
    )
    assert resp.status_code == 404


# --------------------------------------------------------------------------
# A successful status change is reflected by a follow-up corpus_rows read.
# --------------------------------------------------------------------------

def test_status_change_reflected_in_corpus_rows(forms_project, authed):
    slug, work_id, handle = (
        forms_project["slug"],
        forms_project["work_id"],
        forms_project["handle"],
    )
    assert corpus_rows(handle)[0]["inclusion_status"] == "included"

    resp = authed.post(
        f"/ui/projects/{slug}/corpus/{work_id}/status",
        data={"_csrf": CSRF, "status": "excluded", "reason": "ui edit"},
        follow_redirects=False,
    )
    assert resp.status_code == 303

    rows = corpus_rows(handle)
    assert rows[0]["work_id"] == work_id
    assert rows[0]["inclusion_status"] == "excluded"


# --------------------------------------------------------------------------
# Lens create lift parity: the UI create writes the same YAML the CLI would.
# --------------------------------------------------------------------------

def test_lens_new_writes_yaml_and_registers(forms_project, authed):
    from sqlmodel import Session

    slug = forms_project["slug"]
    resp = authed.post(
        f"/ui/projects/{slug}/lenses",
        data={"_csrf": CSRF, "lens_id": "regularity_conditions_v2", "from_template": "regularity_conditions_v1"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    dest = registry.lens_yaml_path(forms_project["project_dir"], "regularity_conditions_v2")
    assert dest.exists()
    assert "lens_id: regularity_conditions_v2" in dest.read_text(encoding="utf-8")
    with Session(forms_project["handle"].engine) as session:
        ids = {r.lens_id for r in registry.list_lenses(session)}
    assert "regularity_conditions_v2" in ids


def test_review_resolve_lift_marks_resolved(forms_project, authed):
    slug, item_id = forms_project["slug"], forms_project["item_id"]
    resp = authed.post(
        f"/ui/projects/{slug}/review/{item_id}/resolve",
        # 'reject' — the payload-less fixture item cannot 'merge' anymore (Build A
        # ch7 made merge a real graph mutation); this test pins the resolve LIFT.
        data={"_csrf": CSRF, "action": "reject"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    # The item is no longer open (the resolve lift flipped its status).
    open_ids = {i.item_id for i in review_mod.list_open(forms_project["handle"])}
    assert item_id not in open_ids


# --------------------------------------------------------------------------
# Build A ch7/8 applier knobs: the review form exposes ONLY the optional
# fields the resolve handler reads FOR THAT ITEM TYPE (review_action_menu in
# project/cli.py), so a row never shows inputs its item type ignores.
# --------------------------------------------------------------------------

def test_review_form_scopes_duplicate_candidate_inputs(forms_project, authed):
    """duplicate_candidate's menu lists only survivor_id — the citation-only
    knobs (candidate_index/target_work_id) don't apply and shouldn't render."""
    resp = authed.get(f"/ui/projects/{forms_project['slug']}/review")
    assert resp.status_code == 200
    assert 'name="survivor_id"' in resp.text
    for field in ("candidate_index", "target_work_id"):
        assert f'name="{field}"' not in resp.text, field


def test_review_form_scopes_concept_merge_question_and_inputs(authed):
    """A concept_merge_candidate row asks the human question, offers only
    approve/reject with human labels, and renders NONE of the three optional
    applier inputs — it needs none (review_action_menu governance)."""
    h = service.create_project("conceptproj")
    review_mod.enqueue(
        h,
        "concept_merge_candidate",
        payload={
            "kind": "concept_merge_candidate",
            "canonical_label": "average treatment effect",
            "member_label": "weighted average treatment effect",
            "canonical_concept_id": "c1",
            "member_concept_id": "c2",
        },
    )
    resp = authed.get("/ui/projects/conceptproj/review")
    assert resp.status_code == 200
    assert "Merge these two concepts into one, or keep them separate?" in resp.text
    assert "Approve - merge into one concept" in resp.text
    assert "Reject - keep separate" in resp.text
    for field in ("survivor_id", "candidate_index", "target_work_id"):
        assert f'name="{field}"' not in resp.text, field


def test_review_form_candidate_index_resolves_multi_candidate_item(
    forms_project, authed
):
    import sqlite3

    h, slug = forms_project["handle"], forms_project["slug"]
    citing = forms_project["work_id"]
    # A real reference_entries row: the applier's manual_override edge carries
    # reference_id (FK) and its durable UPDATE targets this row.
    conn = sqlite3.connect(str(h.db_path))
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "INSERT INTO reference_entries (reference_id, citing_work_id, "
            "raw_reference_text, resolution_status, created_at) "
            "VALUES ('ref_web', ?, 'Doe 2020', 'ambiguous', "
            "'2026-07-02T00:00:00+00:00')",
            (citing,),
        )
        conn.commit()
    finally:
        conn.close()
    item_id = review_mod.enqueue(
        h,
        "citation_resolution",
        target_type="reference_entry",
        target_id="ref_web",
        payload={
            "kind": "citation_resolution",
            "status": "ambiguous",
            "citing_work_id": citing,
            "reference_id": "ref_web",
            "raw": "Doe 2020",
            "candidates": [{"doi": "10.9/c0"}, {"doi": "10.9/c1"}],
            "run_id": "RW",
        },
    )
    path = f"/ui/projects/{slug}/review/{item_id}/resolve"

    # citation_resolution's menu lists candidate_index/target_work_id (unlike
    # duplicate_candidate's survivor_id) — the scoped review form renders them.
    resp = authed.get(f"/ui/projects/{slug}/review")
    assert 'name="candidate_index"' in resp.text
    assert 'name="target_work_id"' in resp.text

    # Without candidate_index the applier refuses the two-candidate ambiguity.
    resp = authed.post(
        path, data={"_csrf": CSRF, "action": "approve"}, follow_redirects=False
    )
    assert resp.status_code == 400
    assert "candidates" in resp.json()["detail"]

    # Blank optional fields (a browser submits the whole form) keep defaults;
    # candidate_index picks candidate 1 and the resolve goes through.
    resp = authed.post(
        path,
        data={
            "_csrf": CSRF,
            "action": "approve",
            "survivor_id": "",
            "target_work_id": "",
            "candidate_index": "1",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    open_ids = {i.item_id for i in review_mod.list_open(h)}
    assert item_id not in open_ids
    conn = sqlite3.connect(str(h.db_path))
    try:
        chosen = conn.execute(
            "SELECT work_id FROM works WHERE doi='10.9/c1'"
        ).fetchone()[0]
        assert conn.execute(
            "SELECT provenance, confidence FROM citation_edges "
            "WHERE source_work_id=? AND target_work_id=? AND run_id='RW'",
            (citing, chosen),
        ).fetchall() == [("manual_override", 1.0)]
        assert conn.execute(
            "SELECT resolved_work_id, resolution_status, resolution_source "
            "FROM reference_entries WHERE reference_id='ref_web'"
        ).fetchone() == (chosen, "resolved", "manual_override")
    finally:
        conn.close()


def test_review_form_bad_candidate_index_rejected(forms_project, authed):
    slug, item_id = forms_project["slug"], forms_project["item_id"]
    resp = authed.post(
        f"/ui/projects/{slug}/review/{item_id}/resolve",
        data={"_csrf": CSRF, "action": "reject", "candidate_index": "one"},
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "candidate_index must be an integer" in resp.json()["detail"]
