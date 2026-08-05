"""Build F ops slice (chunks 5, 6, 7, 9) — config snapshot/fingerprint round-trip,
``doctor --json`` shape contract, the fail-loud import guard's symbol list, and the
done-vs-crashed ``interrupted`` sweep.

Offline / keyless throughout: CLI verbs run under Typer's ``CliRunner`` on the
autouse throwaway ``SEEDGRAPH_HOME``; the events route is exercised in-process via
Starlette's ``TestClient`` with the ``require_local_session`` dependency override.
"""

from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from seedgraph.api.app import app as api_app
from seedgraph.cli import app
from seedgraph.config.loader import (
    _FORBIDDEN_SECRET_KEYS,
    config_fingerprint,
    effective_config_snapshot,
    load_project_config,
    write_project_overrides,
)
from seedgraph.project import service
from seedgraph.run import (
    _status_from_events,
    append_event,
    ensure_run,
    latest_run_id,
    list_runs,
    mark_interrupted_runs,
    read_events,
    read_manifest,
)
from seedgraph.web import serve

runner = CliRunner()


# --------------------------------------------------------------------------
# ch7 — fail-loud contract import guard (the guard's real test is CI turning
# red when a symbol vanishes; this pins that the list itself stays importable).
# --------------------------------------------------------------------------

def test_contract_symbol_list_is_importable():
    import conftest

    for module_name, symbol in conftest._CONTRACT_SYMBOLS:
        module = importlib.import_module(module_name)
        assert hasattr(module, symbol), f"{module_name}.{symbol} missing"
    conftest._assert_contract_symbols()  # must not raise on a healthy tree


# --------------------------------------------------------------------------
# ch5 — effective_config_snapshot + config_fingerprint
# --------------------------------------------------------------------------

def _all_keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _all_keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _all_keys(item)


def test_manifest_config_fingerprint_round_trips():
    """v1 criterion 9 / R2-6: reload the citation manifest and rebuild the
    fingerprint from its own config_snapshot."""
    service.create_project("fingproj")
    result = runner.invoke(app, ["cite", "run", "fingproj"])
    assert result.exit_code == 0, result.output

    rid = latest_run_id("fingproj")
    section = read_manifest("fingproj", rid).sections["citation"]
    assert section["config_fingerprint"] == config_fingerprint(section["config_snapshot"])
    # A real digest, not the old 16-char slug hash.
    assert len(section["config_fingerprint"]) == 64
    # The persisted snapshot carries no secret-NAMED key anywhere.
    forbidden = {k.lower() for k in _FORBIDDEN_SECRET_KEYS}
    assert not [
        k for k in _all_keys(section["config_snapshot"])
        if isinstance(k, str) and k.lower() in forbidden
    ]


def test_different_config_knob_gives_different_fingerprint():
    service.create_project("fingknob")
    before = config_fingerprint(effective_config_snapshot(load_project_config("fingknob")))
    write_project_overrides("fingknob", {"answer": {"max_evidence_tokens": 4321}})
    after = config_fingerprint(effective_config_snapshot(load_project_config("fingknob")))
    assert before != after


def test_two_runs_differing_in_one_knob_get_different_manifest_fingerprints():
    """Acceptance criterion 5 end-to-end: the persisted manifest fingerprint tracks
    the EFFECTIVE config, not the project slug (the old slug hash was identical for
    every run of a project)."""
    h = service.create_project("fingruns")
    assert runner.invoke(app, ["cite", "run", "fingruns"]).exit_code == 0
    rid1 = latest_run_id("fingruns")

    write_project_overrides("fingruns", {"answer": {"max_evidence_tokens": 4321}})
    assert runner.invoke(app, ["cite", "run", "fingruns"]).exit_code == 0
    runs_dir = h.db_path.parent / "runs"
    (rid2,) = {p.name for p in runs_dir.iterdir() if p.is_dir()} - {rid1}

    first = read_manifest("fingruns", rid1).sections["citation"]
    second = read_manifest("fingruns", rid2).sections["citation"]
    assert first["config_fingerprint"] != second["config_fingerprint"]
    # Both still round-trip against their OWN persisted snapshot.
    assert first["config_fingerprint"] == config_fingerprint(first["config_snapshot"])
    assert second["config_fingerprint"] == config_fingerprint(second["config_snapshot"])


def test_fingerprint_is_key_order_invariant():
    assert config_fingerprint({"a": 1, "b": [2, 3]}) == config_fingerprint({"b": [2, 3], "a": 1})
    assert config_fingerprint({"a": 1}) != config_fingerprint({"a": 2})


def test_snapshot_strips_secret_named_keys_from_hostile_mapping():
    """Defense-in-depth: even a hostile merged patch never lands a secret-named
    field in the snapshot (values under legitimate keys survive)."""
    hostile = {
        "llm": {"profiles": {"p": {"api_key": "sk-live-oops", "model": "m"}}},
        "token": "raw",
        "nested": [{"password": "x", "keep": 1}],
    }
    snap = effective_config_snapshot(hostile)
    forbidden = {k.lower() for k in _FORBIDDEN_SECRET_KEYS}
    assert not [k for k in _all_keys(snap) if isinstance(k, str) and k.lower() in forbidden]
    assert snap["llm"]["profiles"]["p"]["model"] == "m"
    assert snap["nested"][0]["keep"] == 1


# --------------------------------------------------------------------------
# ch6 — doctor --json
# --------------------------------------------------------------------------

def test_doctor_json_shape_and_exit_zero_on_healthy_home():
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output

    # The CI smoke's contract: the LAST non-empty stdout line is the JSON object.
    last = [line for line in result.output.splitlines() if line.strip()][-1]
    doc = json.loads(last)
    assert set(doc) == {"checks", "ok"}
    assert doc["ok"] is True
    assert doc["checks"]
    for check in doc["checks"]:
        # Keys PER CHECK are pinned; the check LIST is not (later builds' checks
        # must flow through order-free — reconciliation §1 item 9).
        assert set(check) == {"name", "ok", "severity", "detail"}
        assert isinstance(check["name"], str)
        assert isinstance(check["ok"], bool)
        assert check["severity"] in ("error", "warning")
        assert isinstance(check["detail"], str)


def test_doctor_default_render_unchanged():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "RESULT: OK" in result.output
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.output)


# --------------------------------------------------------------------------
# ch9 — done-vs-crashed: `interrupted` terminal + mark_interrupted_runs sweep
# --------------------------------------------------------------------------

def test_status_from_events_recognizes_interrupted_terminal():
    assert _status_from_events([{"event": "started"}, {"event": "interrupted"}]) == "interrupted"
    assert _status_from_events([{"event": "started"}]) == "running"
    assert _status_from_events([]) is None


def test_sweep_marks_stale_running_run_interrupted():
    service.create_project("swproj")
    rid = ensure_run("swproj")
    append_event("swproj", rid, phase="extract", event="started")
    assert list_runs("swproj")[0]["status"] == "running"

    assert mark_interrupted_runs("swproj") == [rid]

    rows = list_runs("swproj")
    assert rows[0]["status"] == "interrupted"
    last = read_events("swproj", rid)[-1]
    assert last["event"] == "interrupted"
    assert last["level"] == "warning"
    assert last["message"] == "process restarted mid-run"


def test_sweep_is_idempotent():
    service.create_project("swidem")
    rid = ensure_run("swidem")
    append_event("swidem", rid, phase="p", event="started")
    assert mark_interrupted_runs("swidem") == [rid]
    n_events = len(read_events("swidem", rid))

    assert mark_interrupted_runs("swidem") == []  # second sweep appends nothing
    assert len(read_events("swidem", rid)) == n_events


def test_sweep_leaves_terminal_and_eventless_runs_untouched():
    service.create_project("swterm")
    rid_fin = ensure_run("swterm")
    append_event("swterm", rid_fin, phase="p", event="started")
    append_event("swterm", rid_fin, phase="p", event="finished")
    rid_fail = ensure_run("swterm")
    append_event("swterm", rid_fail, phase="p", event="started")
    append_event("swterm", rid_fail, phase="p", event="failed")
    # A manifest-only run with NO events (the CLI cite-run shape) must never be
    # stamped — CLI runs are event-silent by invariant (risk 7).
    rid_cli = ensure_run("swterm")

    assert mark_interrupted_runs("swterm") == []

    by_id = {row["run_id"]: row["status"] for row in list_runs("swterm")}
    assert by_id[rid_fin] == "finished"
    assert by_id[rid_fail] == "failed"
    assert by_id[rid_cli] is None
    assert read_events("swterm", rid_cli) == []  # still no events.jsonl


def test_sweep_on_project_without_runs_dir_is_noop():
    service.create_project("swnone")
    assert mark_interrupted_runs("swnone") == []


def test_serve_sweeps_stale_runs_before_uvicorn_binds(monkeypatch):
    """`serve` stamps every project's stale runs BEFORE uvicorn binds, so no
    status request can ever be served inside the restart window (design 9)."""
    import uvicorn

    service.create_project("swserve")
    rid = ensure_run("swserve")
    append_event("swserve", rid, phase="p", event="started")

    seen_at_bind: dict = {}

    def fake_run(app_obj, host, port):
        seen_at_bind["status"] = list_runs("swserve")[0]["status"]

    monkeypatch.setattr(uvicorn, "run", fake_run)
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0, result.output
    assert seen_at_bind["status"] == "interrupted"
    assert f"marked stale run as interrupted: swserve/{rid}" in result.output


def test_swept_stale_run_reports_interrupted_via_events_route():
    """Route-level: the UNCHANGED events route over swept state (poll.js stops on
    any terminal status, so `interrupted` ends the polling loop)."""
    service.create_project("swroute")
    rid = ensure_run("swroute")
    append_event("swroute", rid, phase="p", event="started")
    mark_interrupted_runs("swroute")

    api_app.dependency_overrides[serve.require_local_session] = lambda: None
    try:
        client = TestClient(api_app)
        resp = client.get(f"/api/projects/swroute/runs/{rid}/events")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "interrupted"
        assert body["events"][-1]["event"] == "interrupted"
    finally:
        api_app.dependency_overrides.pop(serve.require_local_session, None)
