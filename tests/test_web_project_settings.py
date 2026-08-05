"""Track 2: the settings-inheritance UI — global defaults with LIVE per-project
inheritance, sparse project overrides, tighten-only ceilings, and revert-to-global.

Mirrors ``test_web_create_upload.py``: the FastAPI surface is exercised via
Starlette's ``TestClient`` over a throwaway ``SEEDGRAPH_HOME`` (the autouse
``isolated_home`` fixture), with the loopback gate replaced by the
``dependency_overrides`` seam and the CSRF token pinned so a posted ``_csrf`` matches.
Offline / keyless throughout — no network, no LLM.
"""

from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient

from seedgraph.api.app import app
from seedgraph.config.loader import (
    load_project_config as load_runtime_config,
    write_global_config,
)
from seedgraph.paths import project_dir
from seedgraph.project import service
from seedgraph.web import serve

CSRF = "fixed-csrf-token-for-tests"


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


def _raw(slug: str) -> dict:
    """The raw on-disk ``project.yaml`` overlay (the source of truth for sparseness)."""
    path = project_dir(slug) / "project.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# --------------------------------------------------------------------------
# 1. The settings gear link is present in the base chrome on every page.
# --------------------------------------------------------------------------

def test_gear_link_present_in_base_chrome(authed):
    # The landing page extends base.html, so the topnav gear must be reachable
    # even from the ungated entry point.
    resp = authed.get("/ui")
    assert resp.status_code == 200
    assert 'href="/ui/settings"' in resp.text
    assert "Settings" in resp.text


# --------------------------------------------------------------------------
# 2. Project settings page shows inherit vs override + a revert affordance.
# --------------------------------------------------------------------------

def test_project_settings_shows_inherited_then_overridden_and_revert(authed):
    service.create_project("instest")

    # Fresh project: nothing overridden, every field inherits the global default.
    before = authed.get("/ui/projects/instest/settings")
    assert before.status_code == 200
    assert "inheriting" in before.text
    assert 'name="revert"' not in before.text  # no revert control until something is overridden

    # Override a single budget field.
    resp = authed.post(
        "/ui/projects/instest/settings",
        data={
            "_csrf": CSRF,
            "project_name": "Ins Test",
            "override_usd_limit": "on",
            "usd_limit": "3.0",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text[:200]

    after = authed.get("/ui/projects/instest/settings")
    assert after.status_code == 200
    assert "overridden to" in after.text
    assert 'value="budget:usd_limit"' in after.text  # the revert control is wired
    assert "revert to global" in after.text


# --------------------------------------------------------------------------
# 3. POSTing an override writes a SPARSE project.yaml (only the changed key).
# --------------------------------------------------------------------------

def test_override_post_writes_sparse_project_yaml(authed):
    service.create_project("sparse")

    resp = authed.post(
        "/ui/projects/sparse/settings",
        data={
            "_csrf": CSRF,
            "project_name": "Sparse",
            "override_usd_limit": "on",
            "usd_limit": "3.0",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text[:200]

    raw = _raw("sparse")
    # Only the ONE overridden budget field is materialized — the other override
    # groups stay absent (inherited), so the overlay is genuinely sparse.
    assert raw["budget"] == {"usd_limit": 3.0}
    assert "content_policy" not in raw
    assert "answer" not in raw
    assert "llm" not in raw
    # …and it resolves live to the overridden value.
    assert load_runtime_config("sparse").budget.usd_limit == 3.0


# --------------------------------------------------------------------------
# 4. Override groups survive a subsequent identity save (clobber-safety E2E).
# --------------------------------------------------------------------------

def test_override_survives_subsequent_identity_save(authed):
    service.create_project("clobweb")

    # 1) write a budget override.
    r1 = authed.post(
        "/ui/projects/clobweb/settings",
        data={
            "_csrf": CSRF,
            "project_name": "Clob Web",
            "override_monthly_soft_limit_usd": "on",
            "monthly_soft_limit_usd": "4.0",
        },
        follow_redirects=False,
    )
    assert r1.status_code == 303, r1.text[:200]
    assert _raw("clobweb")["budget"] == {"monthly_soft_limit_usd": 4.0}

    # 2) a plain identity rename (no override boxes) must NOT clobber the group.
    r2 = authed.post(
        "/ui/projects/clobweb/settings",
        data={"_csrf": CSRF, "project_name": "Renamed Clob"},
        follow_redirects=False,
    )
    assert r2.status_code == 303, r2.text[:200]

    raw = _raw("clobweb")
    assert raw["project_name"] == "Renamed Clob"           # identity updated
    assert raw["budget"] == {"monthly_soft_limit_usd": 4.0}  # override survived
    assert load_runtime_config("clobweb").budget.monthly_soft_limit_usd == 4.0


# --------------------------------------------------------------------------
# 5. A loosening privacy override is rejected with 400 (ceiling = tighten-only).
# --------------------------------------------------------------------------

def test_loosening_privacy_override_rejected_400(authed):
    service.create_project("privloosen")
    # global external_llm_for_private_full_text defaults to False; the project tries
    # to flip it True (looser) -> rejected, nothing written.
    resp = authed.post(
        "/ui/projects/privloosen/settings",
        data={
            "_csrf": CSRF,
            "project_name": "Priv Loosen",
            "override_external_llm_for_private_full_text": "on",
            "external_llm_for_private_full_text": "on",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400, resp.text[:200]
    assert "loosen" in resp.json()["detail"]
    # nothing was written: no content_policy group, and the rename did NOT persist.
    raw = _raw("privloosen")
    assert "content_policy" not in raw
    assert raw.get("project_name") != "Priv Loosen"


# --------------------------------------------------------------------------
# 6. A loosening budget override (above the global cap) is rejected with 400.
# --------------------------------------------------------------------------

def test_loosening_budget_override_rejected_400(authed):
    write_global_config({"budget": {"usd_limit": 5.0}})
    service.create_project("budloosen")

    resp = authed.post(
        "/ui/projects/budloosen/settings",
        data={
            "_csrf": CSRF,
            "project_name": "Bud Loosen",
            "override_usd_limit": "on",
            "usd_limit": "10.0",  # above the global 5.0 ceiling
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400, resp.text[:200]
    assert "loosen" in resp.json()["detail"]
    assert "budget" not in _raw("budloosen")  # nothing written


# --------------------------------------------------------------------------
# 7. A TIGHTENING budget override is accepted (project may narrow a ceiling).
# --------------------------------------------------------------------------

def test_tightening_budget_override_accepted(authed):
    write_global_config({"budget": {"usd_limit": 5.0}})
    service.create_project("budtighten")

    resp = authed.post(
        "/ui/projects/budtighten/settings",
        data={
            "_csrf": CSRF,
            "project_name": "Bud Tighten",
            "override_usd_limit": "on",
            "usd_limit": "2.0",  # below the global 5.0 ceiling -> allowed
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text[:200]
    assert _raw("budtighten")["budget"] == {"usd_limit": 2.0}
    assert load_runtime_config("budtighten").budget.usd_limit == 2.0


# --------------------------------------------------------------------------
# 8. The revert action removes an override key (field re-inherits global).
# --------------------------------------------------------------------------

def test_revert_removes_override_key(authed):
    service.create_project("revtest")

    authed.post(
        "/ui/projects/revtest/settings",
        data={
            "_csrf": CSRF,
            "project_name": "Rev Test",
            "override_usd_limit": "on",
            "usd_limit": "3.0",
        },
        follow_redirects=False,
    )
    assert _raw("revtest").get("budget") == {"usd_limit": 3.0}

    # Revert that one field back to the global default.
    resp = authed.post(
        "/ui/projects/revtest/settings",
        data={"_csrf": CSRF, "revert": "budget:usd_limit"},
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text[:200]
    raw = _raw("revtest")
    assert "budget" not in raw  # the emptied group was pruned -> back to inherit
    assert load_runtime_config("revtest").budget.usd_limit is None  # global default


# --------------------------------------------------------------------------
# 9. CSRF still gates the reworked project settings POST.
# --------------------------------------------------------------------------

def test_project_settings_post_requires_csrf(authed):
    service.create_project("csrfsettings")
    resp = authed.post(
        "/ui/projects/csrfsettings/settings",
        data={"project_name": "No CSRF", "override_usd_limit": "on", "usd_limit": "1.0"},
        follow_redirects=False,
    )
    assert resp.status_code == 403
    assert "content_policy" not in _raw("csrfsettings")
    assert "budget" not in _raw("csrfsettings")


# --------------------------------------------------------------------------
# 10. The global settings view exposes the routing + answer editors.
# --------------------------------------------------------------------------

def test_global_settings_exposes_routing_and_answer_editor(authed):
    resp = authed.get("/ui/settings")
    assert resp.status_code == 200
    # profile-CHOICE editor per task + the answer/retrieval tunables.
    assert 'name="route_note_extraction_preferred"' in resp.text
    assert 'name="max_evidence_tokens"' in resp.text
    assert 'name="_has_answer_policy"' in resp.text
