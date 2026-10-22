"""Process-global marker-conversion queue: concurrency cap, drain, failure isolation
and the gated status endpoint.

Queue workers convert IN the app process through the shared warm converter pool
(``seedgraph.conversion``); the app saves the result. ``marker_queue._reset_for_tests``
retires the queue between tests so each one re-reads the pool size. The concurrency /
failure tests monkeypatch ``_run_conversion``; the drain tests run the real path with
a fake-engine pool (tests/support/converter.py)."""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from seedgraph.acquisition.service import corpus_rows, ingest_upload
from seedgraph.conversion import set_pool
from seedgraph.api.app import app
from seedgraph.project import service
from seedgraph.web import marker_queue, serve
from support.converter import fake_pool, synthetic_pdf


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

    def __call__(self, slug, work_id, source_file_id, file_hash, *, root=None):
        with self.lock:
            self.current += 1
            self.ran += 1
            self.max_seen = max(self.max_seen, self.current)
        time.sleep(self.hold)
        with self.lock:
            self.current -= 1


@pytest.fixture
def use_pool():
    """Install a fake-engine pool as the shared pool for one test."""
    made = []

    def install(**kw):
        p = fake_pool(**kw)
        made.append(p)
        set_pool(p)
        marker_queue._reset_for_tests()  # re-read the pool size
        return p

    yield install
    for p in made:
        p.shutdown()


def test_concurrency_never_exceeds_two_slots(monkeypatch, use_pool):
    use_pool(gpus=("0", "1"))  # queue slots follow the pool size (one per GPU)
    instr = _Instrument()
    monkeypatch.setattr(marker_queue, "_run_conversion", instr)

    for i in range(5):
        marker_queue.enqueue("proj", f"w{i}", f"sf{i}", f"h{i}")
    assert marker_queue.wait_idle(timeout=30) is True

    assert instr.ran == 5
    assert instr.max_seen == 2
    assert marker_queue.slots() == 2


def test_concurrency_one_slot_is_serial(monkeypatch, use_pool):
    use_pool(gpus=(None,))
    instr = _Instrument(hold=0.05)
    monkeypatch.setattr(marker_queue, "_run_conversion", instr)

    for i in range(5):
        marker_queue.enqueue("proj", f"w{i}", f"sf{i}", f"h{i}")
    assert marker_queue.wait_idle(timeout=30) is True

    assert instr.ran == 5
    assert instr.max_seen == 1
    assert marker_queue.slots() == 1


def test_drain_converts_in_app_through_the_warm_pool(tmp_path, use_pool):
    """Real path: queue worker -> convert_and_bridge -> WarmMarkerBackend -> shared
    pool -> worker child; the APP process saves markdown + bridge."""
    pool = use_pool()
    h = service.create_project("queueproj")
    work = service.add_work(
        h, title="Queued Paper", is_seed=True, inclusion_status="included"
    )
    src = ingest_upload(synthetic_pdf(tmp_path, "seed.pdf", "queue body"), cache_root=None)

    marker_queue.enqueue("queueproj", work.work_id, src.source_file_id, src.file_hash)
    assert marker_queue.wait_idle(timeout=60) is True

    rows = corpus_rows(h)
    assert len(rows) == 1
    assert rows[0]["has_markdown"] is True
    assert rows[0]["acquisition_state"] == "already_cached_local"
    assert marker_queue.status("queueproj") == {}
    assert pool.running_count() == 1  # the converter stays warm for the next paper


def test_crash_mid_pdf_marks_failed_and_queue_continues(tmp_path, use_pool):
    use_pool()
    h = service.create_project("crashproj")
    bad = service.add_work(h, title="Bad", is_seed=True, inclusion_status="included")
    good = service.add_work(h, title="Good", is_seed=True, inclusion_status="included")
    src_bad = ingest_upload(synthetic_pdf(tmp_path, "bad.pdf", "FAKE:CRASH"), cache_root=None)
    src_good = ingest_upload(synthetic_pdf(tmp_path, "good.pdf", "fine"), cache_root=None)

    marker_queue.enqueue("crashproj", bad.work_id, src_bad.source_file_id, src_bad.file_hash)
    marker_queue.enqueue("crashproj", good.work_id, src_good.source_file_id,
                         src_good.file_hash)
    assert marker_queue.wait_idle(timeout=60) is True

    st = marker_queue.status("crashproj")
    assert st[bad.work_id]["state"] == "failed"
    assert "exited" in st[bad.work_id]["detail"]
    assert good.work_id not in st
    by_id = {r["work_id"]: r for r in corpus_rows(h)}
    assert by_id[good.work_id]["has_markdown"] is True
    assert by_id[bad.work_id]["has_markdown"] is False


def test_failure_is_isolated_and_worker_survives(monkeypatch):
    marker_queue._reset_for_tests()
    completed: list[str] = []

    def maybe_fail(slug, work_id, source_file_id, file_hash, *, root=None):
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


def test_queue_module_never_imports_torch():
    import sys

    assert "torch" not in sys.modules
    assert not hasattr(marker_queue, "_detect_num_gpus")


def test_status_endpoint_is_gated_and_returns_json(monkeypatch):
    marker_queue._reset_for_tests()
    gate = threading.Event()

    def block(slug, work_id, source_file_id, file_hash, *, root=None):
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
