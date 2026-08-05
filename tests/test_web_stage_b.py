"""Track 2 Stage B: run progress events, the background job runner, the public run
inventory helpers, the events polling route, and the graph.json public-safe contract.

Offline / keyless: jobs run on real daemon threads but every fixture is a throwaway
``SEEDGRAPH_HOME``; the FastAPI surface is exercised via Starlette's ``TestClient``
with the ``require_local_session`` dependency override (cross-cutting #1 seam).
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from seedgraph.api.app import app
from seedgraph.errors import SeedgraphError
from seedgraph.paths import project_dir
from seedgraph.project import service
from seedgraph.run import (
    append_event,
    ensure_run,
    latest_run_id,
    list_runs,
    read_events,
    read_manifest,
    update_manifest,
)
from seedgraph.web import jobs, serve


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _wait_idle(slug: str, timeout: float = 5.0) -> None:
    """Block until ``slug``'s job is no longer live (terminal event already written)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not jobs.is_running(slug):
            return
        time.sleep(0.01)
    raise AssertionError(f"job for {slug!r} did not finish within {timeout}s")


# --------------------------------------------------------------------------
# append_event / read_events — seq, after-filter, missing file
# --------------------------------------------------------------------------

def test_append_event_seq_monotonic_and_fields_round_trip():
    rid = ensure_run("evproj")
    s0 = append_event("evproj", rid, phase="p", event="a")
    s1 = append_event("evproj", rid, phase="p", event="b", message="hi", data={"k": 1})
    s2 = append_event("evproj", rid, phase="p", event="c", level="error")
    assert (s0, s1, s2) == (0, 1, 2)  # 0-indexed, monotonic from line count

    events = read_events("evproj", rid)
    assert [e["event"] for e in events] == ["a", "b", "c"]
    assert [e["seq"] for e in events] == [0, 1, 2]
    assert events[1]["message"] == "hi"
    assert events[1]["data"] == {"k": 1}
    assert events[1]["level"] == "info"
    assert events[2]["level"] == "error"
    assert all(e["phase"] == "p" and e["ts"] for e in events)


def test_read_events_after_filter():
    rid = ensure_run("evproj_after")
    for name in ("a", "b", "c"):
        append_event("evproj_after", rid, phase="p", event=name)
    assert [e["event"] for e in read_events("evproj_after", rid, after=-1)] == ["a", "b", "c"]
    assert [e["event"] for e in read_events("evproj_after", rid, after=0)] == ["b", "c"]
    assert read_events("evproj_after", rid, after=2) == []


def test_read_events_missing_file_is_empty():
    rid = ensure_run("evproj_missing")
    assert read_events("evproj_missing", rid) == []  # run dir exists, no log yet
    assert read_events("evproj_missing", "run-does-not-exist") == []  # no run dir


def test_append_event_concurrent_seqs_are_unique():
    # The per-run lock must keep seqs unique even under concurrent appenders.
    rid = ensure_run("evproj_race")
    seqs: list[int] = []
    lock = threading.Lock()

    def writer() -> None:
        for _ in range(20):
            s = append_event("evproj_race", rid, phase="p", event="x")
            with lock:
                seqs.append(s)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(seqs) == list(range(80))  # 4*20 unique, gap-free seqs
    logged = read_events("evproj_race", rid)
    assert sorted(e["seq"] for e in logged) == list(range(80))


# --------------------------------------------------------------------------
# latest_run_id / read_manifest / list_runs
# --------------------------------------------------------------------------

def test_latest_run_id_none_when_no_runs():
    service.create_project("lrproj")
    assert latest_run_id("lrproj") is None


def test_latest_run_id_picks_most_recent_by_mtime():
    r1 = ensure_run("lrproj2")
    r2 = ensure_run("lrproj2")
    runs = project_dir("lrproj2") / "runs"
    os.utime(runs / r1, (1000, 1000))
    os.utime(runs / r2, (2000, 2000))
    assert latest_run_id("lrproj2") == r2
    os.utime(runs / r1, (3000, 3000))
    assert latest_run_id("lrproj2") == r1


def test_read_manifest_returns_sections_and_raises_when_missing():
    rid = ensure_run("mproj")
    update_manifest("mproj", rid, {"stage_a": {"n": 1}})
    manifest = read_manifest("mproj", rid)
    assert manifest.run_id == rid
    assert manifest.created_at
    assert manifest.sections["stage_a"] == {"n": 1}

    with pytest.raises(SeedgraphError):
        read_manifest("mproj", "run-does-not-exist")


def test_list_runs_summarizes_most_recent_first():
    r1 = ensure_run("listproj")
    update_manifest("listproj", r1, {"citation": {"edges": 3}})
    append_event("listproj", r1, phase="cite", event="started")
    append_event("listproj", r1, phase="cite", event="finished")
    r2 = ensure_run("listproj")  # newer, no events
    runs = project_dir("listproj") / "runs"
    os.utime(runs / r1, (1000, 1000))
    os.utime(runs / r2, (2000, 2000))

    rows = list_runs("listproj")
    assert [row["run_id"] for row in rows] == [r2, r1]  # most-recent first
    by_id = {row["run_id"]: row for row in rows}
    assert by_id[r1]["sections"]["citation"] == {"edges": 3}
    assert by_id[r1]["status"] == "finished"
    assert by_id[r1]["last_event"]["event"] == "finished"
    assert by_id[r2]["status"] is None
    assert by_id[r2]["last_event"] is None
    assert by_id[r2]["created_at"]  # manifest stamp present even with no events


def test_list_runs_empty_when_no_runs_dir():
    service.create_project("listempty")
    assert list_runs("listempty") == []


# --------------------------------------------------------------------------
# launch_job — terminal events, single-job ceiling, own connection
# --------------------------------------------------------------------------

def test_job_success_emits_started_then_finished():
    service.create_project("okproj")

    def fn(emit, handle):
        emit("progress", "halfway", pct=50)

    rid = jobs.launch_job("okproj", phase="extract", fn=fn)
    _wait_idle("okproj")

    events = read_events("okproj", rid)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "started"
    assert "progress" in kinds
    assert kinds[-1] == "finished"  # guaranteed terminal event
    assert [e["seq"] for e in events] == list(range(len(events)))
    progress = next(e for e in events if e["event"] == "progress")
    assert progress["data"] == {"pct": 50}
    assert not jobs.is_running("okproj")


def test_job_failure_emits_terminal_failed_and_never_re_raises():
    service.create_project("badproj")

    def fn(emit, handle):
        raise RuntimeError("boom-xyz")

    # launch_job must NOT propagate the worker exception into the caller/request.
    rid = jobs.launch_job("badproj", phase="extract", fn=fn)
    _wait_idle("badproj")

    events = read_events("badproj", rid)
    assert events[-1]["event"] == "failed"  # guaranteed terminal event on exception
    assert events[-1]["level"] == "error"
    assert "boom-xyz" in events[-1]["message"]
    assert not jobs.is_running("badproj")  # liveness cleared even after failure


def test_one_active_job_per_project_rejects_second_launch():
    service.create_project("busyproj")
    started = threading.Event()
    gate = threading.Event()

    def slow(emit, handle):
        started.set()
        gate.wait(timeout=5)

    rid = jobs.launch_job("busyproj", phase="extract", fn=slow)
    assert started.wait(timeout=5)
    assert jobs.is_running("busyproj")
    assert jobs.active_run_id("busyproj") == rid

    # A second concurrent launch is rejected with a clear signal (not a crash).
    with pytest.raises(jobs.ProjectBusyError):
        jobs.launch_job("busyproj", phase="extract", fn=lambda e, h: None)

    gate.set()
    _wait_idle("busyproj")
    # Once the first job finishes, a fresh launch succeeds with a NEW run id.
    rid2 = jobs.launch_job("busyproj", phase="extract", fn=lambda e, h: None)
    _wait_idle("busyproj")
    assert rid2 != rid


def test_worker_opens_its_own_handle_in_the_thread():
    service.create_project("connproj")
    captured: dict = {}
    main_ident = threading.get_ident()

    def fn(emit, handle):
        captured["ident"] = threading.get_ident()
        captured["slug"] = handle.slug
        # The handle's engine is a live, worker-thread-owned connection (not the
        # caller's): a real query must succeed from inside the job thread.
        with handle.engine.connect() as conn:
            captured["db_ok"] = conn.execute(text("SELECT 1")).scalar() == 1

    jobs.launch_job("connproj", phase="extract", fn=fn)
    _wait_idle("connproj")

    assert captured["ident"] != main_ident  # ran on a separate (worker) thread
    assert captured["slug"] == "connproj"
    assert captured["db_ok"] is True


# --------------------------------------------------------------------------
# GET /api/projects/{slug}/runs/{run_id}/events
# --------------------------------------------------------------------------

def test_events_route_returns_run_events_and_status():
    service.create_project("routeproj")
    rid = ensure_run("routeproj")
    append_event("routeproj", rid, phase="p", event="started")
    append_event("routeproj", rid, phase="p", event="finished")

    app.dependency_overrides[serve.require_local_session] = lambda: None
    try:
        client = TestClient(app)
        resp = client.get(f"/api/projects/routeproj/runs/{rid}/events")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == rid
        assert body["status"] == "finished"
        assert [e["event"] for e in body["events"]] == ["started", "finished"]

        # ?after slices the events but status stays computed over the FULL log.
        resp2 = client.get(f"/api/projects/routeproj/runs/{rid}/events?after=0")
        body2 = resp2.json()
        assert [e["event"] for e in body2["events"]] == ["finished"]
        assert body2["status"] == "finished"
    finally:
        app.dependency_overrides.pop(serve.require_local_session, None)


def test_events_route_gated_by_local_session():
    service.create_project("routeproj2")
    rid = ensure_run("routeproj2")
    client = TestClient(app)  # host "testclient" (non-loopback), no sg_session cookie
    assert client.get(f"/api/projects/routeproj2/runs/{rid}/events").status_code == 403


# --------------------------------------------------------------------------
# graph.json public-safe regression (cross-cutting #2)
# --------------------------------------------------------------------------

def test_graph_json_always_public_safe_and_private_view_separate(monkeypatch, tmp_path):
    import networkx as nx

    from seedgraph.semantic import export as export_mod

    graph = nx.DiGraph()
    graph.add_node("work::w1", node_type="Work", title="W1")
    graph.add_node(
        "concept::c1",
        node_type="Concept",
        canonical_label="Priv",
        access_class="user_supplied_private",
        definition="secret definition sentence",
    )
    graph.add_edge(
        "work::w1",
        "concept::c1",
        edge_type="discusses",
        epistemic_type="interpretive",
        access_class="user_supplied_private",
    )
    # Route owns the single build_graph call; stub it so no DB fixture is needed.
    monkeypatch.setattr(export_mod, "build_graph", lambda conn, run_id: graph)

    written = export_mod.export_graph(
        None, slug="reg", run_id="r1", allow_private=True, root=tmp_path
    )

    # runs/{run_id}/graph.json is PUBLIC-SAFE even under --allow-private: the private
    # concept (and its full-text-derived definition) is withheld.
    graph_json = next(p for p in written if p.name == "graph.json")
    assert graph_json.parent.name == "r1"
    public_ids = {n["id"] for n in json.loads(graph_json.read_text())["nodes"]}
    assert "work::w1" in public_ids
    assert "concept::c1" not in public_ids

    # --allow-private lands the private view ONLY in a separate exports file.
    private_json = next(p for p in written if p.name == "graph.private.json")
    assert private_json.parent.name == "exports"
    private_ids = {n["id"] for n in json.loads(private_json.read_text())["nodes"]}
    assert "concept::c1" in private_ids
