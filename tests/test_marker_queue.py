"""Process-global marker-conversion queue: concurrency cap, drain, failure isolation,
the gated status endpoint, and the per-GPU subprocess dispatch seam.

The pool is process-global with N=GPU-slots workers; ``marker_queue._reset_for_tests``
retires it between tests so each one re-resolves ``SEEDGRAPH_MARKER_SLOTS``. The
concurrency / drain / failure tests monkeypatch ``_run_conversion`` (the dispatch
seam) so they never spawn a real GPU subprocess; the real dispatch + GPU pinning is
covered by mock-``subprocess.run`` tests, and the child entrypoint is exercised
in-process."""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from seedgraph.acquisition.service import convert_and_bridge, corpus_rows, ingest_upload
from seedgraph.api.app import app
from seedgraph.project import service
from seedgraph.web import marker_queue, serve


@pytest.fixture(autouse=True)
def _reset_queue():
    """Retire the global pool around every test (process-global, no persistence)."""
    marker_queue._reset_for_tests()
    yield
    marker_queue._reset_for_tests()


class _Instrument:
    """A fake ``_run_conversion`` that records the max observed concurrency."""

    def __init__(self, hold: float = 0.2):
        self.hold = hold
        self.lock = threading.Lock()
        self.current = 0
        self.max_seen = 0
        self.ran = 0

    def __call__(
        self, slug, work_id, source_file_id, file_hash, *, root=None, gpu_index=None
    ):
        with self.lock:
            self.current += 1
            self.ran += 1
            self.max_seen = max(self.max_seen, self.current)
        time.sleep(self.hold)
        with self.lock:
            self.current -= 1


def test_concurrency_never_exceeds_two_slots(monkeypatch):
    monkeypatch.setenv("SEEDGRAPH_MARKER_SLOTS", "2")
    marker_queue._reset_for_tests()  # re-resolve slots with the env override
    instr = _Instrument()
    monkeypatch.setattr(marker_queue, "_run_conversion", instr)

    for i in range(5):
        marker_queue.enqueue("proj", f"w{i}", f"sf{i}", f"h{i}")
    assert marker_queue.wait_idle(timeout=30) is True

    assert instr.ran == 5
    assert instr.max_seen == 2
    assert marker_queue.slots() == 2


def test_concurrency_one_slot_is_serial(monkeypatch):
    monkeypatch.setenv("SEEDGRAPH_MARKER_SLOTS", "1")
    marker_queue._reset_for_tests()
    instr = _Instrument(hold=0.05)
    monkeypatch.setattr(marker_queue, "_run_conversion", instr)

    for i in range(5):
        marker_queue.enqueue("proj", f"w{i}", f"sf{i}", f"h{i}")
    assert marker_queue.wait_idle(timeout=30) is True

    assert instr.ran == 5
    assert instr.max_seen == 1
    assert marker_queue.slots() == 1


def test_drain_marks_work_converted(monkeypatch, tmp_path):
    monkeypatch.setenv("SEEDGRAPH_FAKE_MARKER", "1")
    marker_queue._reset_for_tests()

    # In-process shim for the dispatch seam: covers queue<->bridge integration without
    # spawning a real GPU subprocess (that path is tested via mock subprocess.run).
    def _in_process(
        slug, work_id, source_file_id, file_hash, *, root=None, gpu_index=None
    ):
        h2 = service.open_project(slug, root=root)
        convert_and_bridge(
            h2, work_id=work_id, source_file_id=source_file_id,
            file_hash=file_hash, cache_root=root,
        )

    monkeypatch.setattr(marker_queue, "_run_conversion", _in_process)

    h = service.create_project("queueproj")
    work = service.add_work(
        h, title="Queued Paper", is_seed=True, inclusion_status="included"
    )
    pdf = tmp_path / "seed.pdf"
    pdf.write_bytes(b"%PDF-1.4 background queue body text here")
    src = ingest_upload(pdf, cache_root=None)

    marker_queue.enqueue("queueproj", work.work_id, src.source_file_id, src.file_hash)
    assert marker_queue.wait_idle(timeout=60) is True

    rows = corpus_rows(h)
    assert len(rows) == 1
    assert rows[0]["has_markdown"] is True
    assert rows[0]["acquisition_state"] == "already_cached_local"
    assert marker_queue.status("queueproj") == {}


def test_failure_is_isolated_and_worker_survives(monkeypatch):
    marker_queue._reset_for_tests()
    completed: list[str] = []

    def maybe_fail(
        slug, work_id, source_file_id, file_hash, *, root=None, gpu_index=None
    ):
        if work_id == "bad":
            raise RuntimeError("boom convert")
        completed.append(work_id)

    monkeypatch.setattr(marker_queue, "_run_conversion", maybe_fail)
    marker_queue.enqueue("proj", "bad", "sf0", "h0")
    marker_queue.enqueue("proj", "good", "sf1", "h1")
    assert marker_queue.wait_idle(timeout=30) is True

    st = marker_queue.status("proj")
    assert st["bad"]["state"] == "failed"
    assert "boom convert" in st["bad"]["detail"]
    assert "good" not in st
    assert completed == ["good"]


def test_run_conversion_pins_gpu_via_subprocess(monkeypatch):
    """The real dispatch seam pins ``CUDA_VISIBLE_DEVICES`` and targets the entrypoint."""
    captured = {}

    class _FakeProc:
        returncode = 0
        stdout = "OK"
        stderr = ""

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs.get("env")
        return _FakeProc()

    monkeypatch.setattr(marker_queue.subprocess, "run", fake_run)
    marker_queue._run_conversion("proj", "w1", "sf1", "h1", gpu_index=1)

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert captured["argv"][1] == "-m"
    assert "seedgraph.web._convert_worker" in captured["argv"]
    assert captured["argv"][-4:] == ["proj", "w1", "sf1", "h1"]


def test_run_conversion_no_pin_when_gpu_index_none(monkeypatch):
    """CPU / no-GPU degrade: no ``CUDA_VISIBLE_DEVICES`` override is injected."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    captured = {}

    class _FakeProc:
        returncode = 0
        stdout = "OK"
        stderr = ""

    def fake_run(argv, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakeProc()

    monkeypatch.setattr(marker_queue.subprocess, "run", fake_run)
    marker_queue._run_conversion("proj", "w1", "sf1", "h1", gpu_index=None)

    assert "CUDA_VISIBLE_DEVICES" not in captured["env"]


def test_run_conversion_raises_on_nonzero_exit(monkeypatch):
    class _FailProc:
        returncode = 1
        stdout = ""
        stderr = "RuntimeError: value cannot be converted to type c10::Half"

    monkeypatch.setattr(marker_queue.subprocess, "run", lambda argv, **k: _FailProc())
    with pytest.raises(RuntimeError, match="c10::Half"):
        marker_queue._run_conversion("proj", "w1", "sf1", "h1", gpu_index=0)


def test_convert_worker_entrypoint_converts(monkeypatch, tmp_path):
    """The subprocess entrypoint, exercised IN-PROCESS (no real subprocess): a real
    project + ingested file is converted+bridged and exits 0."""
    monkeypatch.setenv("SEEDGRAPH_FAKE_MARKER", "1")
    from seedgraph.web import _convert_worker

    h = service.create_project("entryproj")
    work = service.add_work(
        h, title="Entry Paper", is_seed=True, inclusion_status="included"
    )
    pdf = tmp_path / "entry.pdf"
    pdf.write_bytes(b"%PDF-1.4 entrypoint conversion body text here")
    src = ingest_upload(pdf, cache_root=None)

    rc = _convert_worker.main(
        ["_convert_worker", "entryproj", work.work_id, src.source_file_id, src.file_hash]
    )
    assert rc == 0
    rows = corpus_rows(h)
    assert rows[0]["has_markdown"] is True


def test_convert_worker_entrypoint_reports_errors():
    """Bad args -> usage + exit 2; an open/convert failure -> exit 1 (parent records)."""
    from seedgraph.web import _convert_worker

    assert _convert_worker.main(["_convert_worker"]) == 2
    # Unknown project -> open_project raises -> caught -> exit 1.
    assert _convert_worker.main(["_convert_worker", "ghostproj", "w", "sf", "h"]) == 1


def test_status_endpoint_is_gated_and_returns_json(monkeypatch):
    marker_queue._reset_for_tests()
    gate = threading.Event()

    def block(slug, work_id, source_file_id, file_hash, *, root=None, gpu_index=None):
        gate.wait(5)

    monkeypatch.setattr(marker_queue, "_run_conversion", block)
    marker_queue.enqueue("ep", "w1", "sf", "h")

    nosession = TestClient(app)
    assert nosession.get("/ui/projects/ep/upload/status").status_code == 403

    app.dependency_overrides[serve.require_local_session] = lambda: None
    try:
        client = TestClient(app)
        body: dict = {}
        for _ in range(100):
            r = client.get("/ui/projects/ep/upload/status")
            assert r.status_code == 200
            body = r.json()
            if "w1" in body:
                break
            time.sleep(0.02)
        assert "w1" in body
        assert body["w1"]["state"] in ("queued", "converting")
        assert "detail" in body["w1"]
    finally:
        app.dependency_overrides.pop(serve.require_local_session, None)
        gate.set()
    marker_queue.wait_idle(timeout=5)
