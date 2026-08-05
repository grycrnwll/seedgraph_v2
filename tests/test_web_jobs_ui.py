"""Track 2 Stage C: the run planner, job launching for every task type, and the
answer workspace (synchronous no-LLM render + the T1→T2 background-job integration).

Offline / keyless: jobs run on real daemon threads over a throwaway ``SEEDGRAPH_HOME``
(the autouse ``isolated_home`` fixture); the FastAPI surface is driven by Starlette's
``TestClient`` with the ``require_local_session`` dependency override + a fixed CSRF
token (cross-cutting #1 seam). The launched job is the deterministic ``concepts``
overlay build (no network); the real-LLM ask degrades honestly because no backend is
configured — it never crashes.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _phase8_helpers import build_fixture_project  # noqa: E402

from seedgraph.api.app import app  # noqa: E402
from seedgraph.run import _status_from_events, read_events  # noqa: E402
from seedgraph.web import jobs, planner, serve  # noqa: E402

CSRF = "fixed-csrf-token-for-tests"


@pytest.fixture
def jobs_project():
    """A project ('ph8') with works, claims, spans + a citation edge (offline fixture)."""
    return build_fixture_project("ph8")


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


def _wait_idle(slug: str, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not jobs.is_running(slug):
            return
        time.sleep(0.01)
    raise AssertionError(f"job for {slug!r} did not finish within {timeout}s")


def _run_id_from_location(location: str) -> str:
    # /ui/projects/ph8/runs/{run_id}[?...]
    return location.split("/runs/")[1].split("?")[0]


# --------------------------------------------------------------------------
# plan_job preview: affected works + cost + privacy flag
# --------------------------------------------------------------------------

def test_plan_preview_renders_works_cost_and_privacy(jobs_project, authed):
    resp = authed.get("/ui/projects/ph8/plan?task=concepts")
    assert resp.status_code == 200, resp.text[:300]
    body = resp.text
    # affected works (work_a / work_b both have extracted claims) …
    assert "Affected works (2)" in body
    assert "Paper A: Difference-in-Differences" in body
    assert "Paper B: Linear IV" in body
    # … the cost projection …
    assert "Estimated cost" in body
    # … and the privacy flag (local-first note route ⇒ stays on the machine).
    assert "Content may leave this machine" in body
    assert "stays local" in body
    assert "local_ollama_default" in body  # resolved route profile is surfaced


def test_plan_preview_shape_local_first(jobs_project):
    h = build_fixture_project("ph8b")
    plan = planner.plan_job(h, "concepts")
    assert plan.is_llm_task is True
    assert plan.affected_count == 2
    assert plan.external_full_text is False  # content stays local
    assert plan.needs_llm_backend is False  # local profile is usable
    assert plan.route_destination == "local"
    # every affected work resolved to the local route, no leak flag.
    assert all(w.destination == "local" and w.external_full_text is False for w in plan.works)


# --------------------------------------------------------------------------
# Build D ch13 (D-13): the conversion plan carries the closed-form no-network
# acquisition budget; preview and job read the SAME depth/cap constants.
# --------------------------------------------------------------------------

def test_plan_conversion_payload_carries_acquisition_preview():
    h = build_fixture_project("ph8prev")
    plan = planner.plan_job(h, "conversion")
    pv = plan.acquisition_preview
    assert pv is not None
    # Seed count = the conversion worklist (the 2 included fixture works);
    # depth/cap are the SAME module constants _job_conversion passes to
    # run_corpus, so the preview and the launched job cannot disagree.
    assert pv["seed_count"] == plan.affected_count == 2
    assert pv["depth"] == planner._CONVERSION_DEPTH
    assert pv["cap"] == planner._CONVERSION_PER_GEN_CAP
    assert pv["expected_papers"] == (
        plan.affected_count + planner._CONVERSION_PER_GEN_CAP * planner._CONVERSION_DEPTH
    )
    assert pv["api_calls"] == pv["expected_papers"] * 3
    assert pv["est_disk_bytes"] == pv["expected_papers"] * 1_900_000
    # Every other task carries no acquisition preview (template keeps its
    # token/cost lines for those).
    assert planner.plan_job(h, "sections").acquisition_preview is None


def test_plan_conversion_page_renders_budget_not_zero_cost(jobs_project, authed):
    """planner.html renders the three budget lines for the conversion task in
    place of the misleading $0.0000 / 0-token framing."""
    resp = authed.get("/ui/projects/ph8/plan?task=conversion")
    assert resp.status_code == 200, resp.text[:300]
    body = resp.text
    assert "Expected papers" in body
    assert "Provider API calls" in body
    assert "Estimated disk" in body
    # The misleading LLM framing is gone for this task (the monthly-spend line
    # below it is untouched — it reports real LLM spend, not this job's cost)…
    assert "Estimated tokens" not in body
    assert "Estimated cost" not in body
    # …but a non-conversion task still shows the token/cost lines.
    other = authed.get("/ui/projects/ph8/plan?task=sections")
    assert other.status_code == 200
    assert "Estimated tokens" in other.text


# --------------------------------------------------------------------------
# POST plan/{task}: launches a job (fresh run_id + terminal event); run page polls
# --------------------------------------------------------------------------

def test_plan_launch_creates_run_with_terminal_event_and_run_page_polls(jobs_project, authed):
    resp = authed.post(
        "/ui/projects/ph8/plan/concepts",
        data={"_csrf": CSRF},
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text[:300]
    location = resp.headers["location"]
    run_id = _run_id_from_location(location)
    _wait_idle("ph8")

    # The job wrote events to the SAME run it minted, ending in a terminal event.
    events = read_events("ph8", run_id)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "started"
    assert kinds[-1] == "finished"
    assert _status_from_events(events) == "finished"

    # The run progress page renders + wires the events poller.
    page = authed.get(f"/ui/projects/ph8/runs/{run_id}")
    assert page.status_code == 200
    assert run_id in page.text
    assert "/static/js/poll.js" in page.text

    # The polling endpoint reports the terminal status the JS watches for.
    api = authed.get(f"/api/projects/ph8/runs/{run_id}/events")
    assert api.status_code == 200
    assert api.json()["status"] == "finished"


def test_launch_planned_job_rejects_unknown_task(jobs_project, authed):
    resp = authed.post(
        "/ui/projects/ph8/plan/not_a_task", data={"_csrf": CSRF}, follow_redirects=False
    )
    assert resp.status_code == 404


# --------------------------------------------------------------------------
# A per-item planner job's events carry the shared Progress N/M counter
# (data.done / data.total), rendered by the run-page bar.
# --------------------------------------------------------------------------

def test_per_item_job_events_carry_done_total(jobs_project):
    h = build_fixture_project("ph8prog")
    # `sections` is a deterministic per-item job (no LLM). Its worklist here is empty
    # (no bridged markdown), so drive it over explicit work_ids: every iteration steps
    # once (the "no markdown" path also steps), so done climbs 1..N over total==N.
    plan = planner.plan_job(h, "sections", work_ids=["work_a", "work_b"])
    run_id = planner.launch_planned_job(h, plan)
    _wait_idle("ph8prog")

    events = read_events("ph8prog", run_id)
    work = [e for e in events if e["event"] == "work"]
    assert work, [e["event"] for e in events]
    # Every per-item event carries the shared counter.
    assert all(e["data"].get("total") == 2 for e in work)
    assert sorted(e["data"]["done"] for e in work) == [1, 2]

    # And the run page renders the N/M progress bar for those data-bearing events.
    page = authed_get_run("ph8prog", run_id)
    assert "run-bar" in page or "1/2" in page


def authed_get_run(slug: str, run_id: str) -> str:
    app.dependency_overrides[serve.require_local_session] = lambda: None
    try:
        client = TestClient(app)
        resp = client.get(f"/ui/projects/{slug}/runs/{run_id}")
        assert resp.status_code == 200
        return resp.text
    finally:
        app.dependency_overrides.pop(serve.require_local_session, None)


# --------------------------------------------------------------------------
# Answer workspace: a no-LLM ask renders an envelope synchronously
# --------------------------------------------------------------------------

def test_no_llm_ask_renders_envelope_synchronously(jobs_project, authed):
    resp = authed.post(
        "/ui/projects/ph8/ask",
        data={"_csrf": CSRF, "q": "parallel trends", "mode": "project_only"},
        follow_redirects=False,
    )
    # Synchronous render (no background job, no redirect).
    assert resp.status_code == 200, resp.text[:300]
    body = resp.text
    assert "Citations" in body
    assert "source_grounded" in body  # the answer_category badge
    assert "Paper A: Difference-in-Differences" in body  # the grounded citation


# --------------------------------------------------------------------------
# Answer workspace: an LLM ask with no backend degrades to retrieval-only (no crash)
# --------------------------------------------------------------------------

def test_llm_ask_no_backend_degrades_to_retrieval_only(jobs_project, authed):
    resp = authed.post(
        "/ui/projects/ph8/ask",
        data={"_csrf": CSRF, "q": "parallel trends", "mode": "project_only", "use_llm": "on"},
        follow_redirects=False,
    )
    # No usable answer backend offline ⇒ synchronous retrieval-only render, not a 500.
    assert resp.status_code == 200, resp.text[:300]
    body = resp.text
    assert "No usable LLM backend" in body  # the honest degradation banner
    assert "retrieval-only" in body
    # It did NOT launch a job (synchronous degrade), so nothing is running.
    assert not jobs.is_running("ph8")


# --------------------------------------------------------------------------
# A second concurrent job per project → friendly busy message (not a crash)
# --------------------------------------------------------------------------

def test_second_concurrent_job_friendly_busy(jobs_project, authed):
    started = threading.Event()
    gate = threading.Event()

    def slow(emit, handle):
        started.set()
        gate.wait(timeout=10)

    rid = jobs.launch_job("ph8", phase="extract", fn=slow)
    assert started.wait(timeout=5)
    assert jobs.is_running("ph8")
    try:
        resp = authed.post(
            "/ui/projects/ph8/plan/concepts",
            data={"_csrf": CSRF},
            follow_redirects=False,
        )
        # Rejected with a friendly busy page (409), never a crash or a second run.
        assert resp.status_code == 409, resp.text[:300]
        assert "busy" in resp.text.lower()
        assert "already running" in resp.text.lower()
    finally:
        gate.set()
        _wait_idle("ph8")

    # Once the first job clears, a fresh launch succeeds with a NEW run id.
    resp2 = authed.post(
        "/ui/projects/ph8/plan/concepts", data={"_csrf": CSRF}, follow_redirects=False
    )
    assert resp2.status_code == 303
    _wait_idle("ph8")
    assert _run_id_from_location(resp2.headers["location"]) != rid


# --------------------------------------------------------------------------
# Every Stage C mutating POST enforces CSRF + session (fail-closed twice)
# --------------------------------------------------------------------------

_POSTS = [
    ("/ui/projects/ph8/plan/concepts", {}),
    ("/ui/projects/ph8/ask", {"q": "parallel trends", "mode": "project_only"}),
]


def test_stage_c_posts_reject_missing_or_wrong_csrf(jobs_project, authed):
    for path, data in _POSTS:
        # missing _csrf
        r = authed.post(path, data=data, follow_redirects=False)
        assert r.status_code == 403, (path, r.status_code)
        assert "CSRF" in r.json()["detail"]
        # wrong _csrf
        r2 = authed.post(path, data={**data, "_csrf": "nope"}, follow_redirects=False)
        assert r2.status_code == 403, (path, r2.status_code)


def test_stage_c_posts_refuse_without_session(jobs_project):
    client = TestClient(app)  # host "testclient" (non-loopback), no sg_session cookie
    for path, data in _POSTS:
        r = client.post(path, data={**data, "_csrf": CSRF}, follow_redirects=False)
        assert r.status_code == 403, (path, r.status_code)
