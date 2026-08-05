import json

import pytest

from seedgraph.errors import SeedgraphError
from seedgraph.paths import project_dir
from seedgraph.run import ensure_run, update_manifest


def _manifest(slug, run_id):
    path = project_dir(slug) / "runs" / run_id / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_ensure_run_mints_unique_and_creates_dir():
    rid = ensure_run("proj")
    assert rid.startswith("run-")
    assert (project_dir("proj") / "runs" / rid).is_dir()
    assert ensure_run("proj") != rid  # fresh mint each call


def test_ensure_run_explicit_id_is_idempotent():
    rid = ensure_run("proj", run_id="run-fixed-0001")
    assert rid == "run-fixed-0001"
    update_manifest("proj", rid, {"stage_a": {"count": 1}})
    # re-threading the same run does not clobber the manifest
    assert ensure_run("proj", run_id="run-fixed-0001") == "run-fixed-0001"
    assert _manifest("proj", rid)["sections"]["stage_a"] == {"count": 1}


def test_disjoint_sections_persist():
    rid = ensure_run("proj")
    update_manifest("proj", rid, {"stage_a": {"count": 1}})
    update_manifest("proj", rid, {"stage_b": {"count": 2}})
    sections = _manifest("proj", rid)["sections"]
    assert sections["stage_a"] == {"count": 1}
    assert sections["stage_b"] == {"count": 2}


def test_same_section_double_write_raises():
    rid = ensure_run("proj")
    update_manifest("proj", rid, {"stage_a": {"count": 1}})
    with pytest.raises(SeedgraphError):
        update_manifest("proj", rid, {"stage_a": {"count": 2}})


def test_no_secret_written_to_manifest(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret-value")
    rid = ensure_run("proj")
    update_manifest("proj", rid, {"stage": {"note": "ran ok"}})
    raw = (project_dir("proj") / "runs" / rid / "manifest.json").read_text(encoding="utf-8")
    assert "super-secret-value" not in raw
