"""WarmMarkerBackend: the warm pool behind the cache's MarkerBackend seam.

``convert_source_file`` still owns dedup / validation / storage in the app process;
the converter only returns markdown. Driven through ``add_document`` /
``convert_source_file`` with the real worker loop (stub engine).
"""

from __future__ import annotations

import json

import pytest

from seedgraph.cache import store
from seedgraph.cache.convert import ConversionError, MarkerConfig, _default_backend, add_document
from seedgraph.cache.marker_backend import FakeMarkerBackend, MarkerResult
from seedgraph.conversion import WarmMarkerBackend
from seedgraph.db.connection import open_cache_db
from support.converter import fake_pool, synthetic_pdf


@pytest.fixture
def pool():
    p = fake_pool()
    yield p
    p.shutdown()


def _runs():
    conn = open_cache_db(None)
    try:
        return [dict(r) for r in conn.execute(
            "SELECT converter_version, run_status, error FROM conversion_runs "
            "ORDER BY created_at, conversion_run_id")]
    finally:
        conn.close()


def test_backend_returns_marker_result_with_page_count_and_timings(pool, tmp_path):
    backend = WarmMarkerBackend(pool)
    res = backend(synthetic_pdf(tmp_path, "a.pdf", pages=4), MarkerConfig())
    assert isinstance(res, MarkerResult)
    assert res.page_count == 4
    assert res.timings["convert_seconds"] >= 0
    assert res.converter_version == "1.10.2"
    assert res.block_json["options"]["paginate_output"] is True


def test_convert_source_file_saves_in_app_and_records_worker_version(pool, tmp_path):
    md = add_document(synthetic_pdf(tmp_path, "a.pdf"), backend=WarmMarkerBackend(pool))
    assert _runs() == [{"converter_version": "1.10.2", "run_status": "success", "error": None}]
    assert store.markdown_blob_path(md.markdown_hash).read_text(encoding="utf-8").count(
        "-" * 48) == 2
    meta = json.loads((store.marker_dir(md.conversion_run_id) / "meta.json").read_text())
    assert meta["engine"] == "fake"
    manifest = json.loads(
        (store.marker_dir(md.conversion_run_id) / "manifest.json").read_text())
    assert manifest["converter"]["version"] == "1.10.2"


def test_dedup_hit_never_starts_a_converter(pool, tmp_path):
    pdf = synthetic_pdf(tmp_path, "a.pdf")
    backend = WarmMarkerBackend(pool)
    add_document(pdf, backend=backend)
    pool.shutdown()
    assert pool.running_count() == 0
    add_document(pdf, backend=backend)  # same bytes, same fingerprint
    assert pool.running_count() == 0


def test_force_ocr_reaches_the_worker(pool, tmp_path):
    backend = WarmMarkerBackend(pool)
    res = backend(synthetic_pdf(tmp_path, "a.pdf"), MarkerConfig(force_ocr=True))
    assert res.block_json["options"]["force_ocr"] is True


def test_crash_mid_pdf_marks_that_pdf_failed_and_next_converts(pool, tmp_path):
    backend = WarmMarkerBackend(pool)
    with pytest.raises(ConversionError, match="exited"):
        add_document(synthetic_pdf(tmp_path, "crash.pdf", "FAKE:CRASH"), backend=backend)
    add_document(synthetic_pdf(tmp_path, "ok.pdf"), backend=backend)
    runs = _runs()
    assert [r["run_status"] for r in runs] == ["failed", "success"]
    assert "exited" in runs[0]["error"]


def test_use_llm_is_refused_not_silently_ignored(pool, tmp_path):
    backend = WarmMarkerBackend(pool)
    with pytest.raises(Exception, match="use_llm"):
        backend(synthetic_pdf(tmp_path, "a.pdf"), MarkerConfig(use_llm=True))


def test_default_backend_is_warm_unless_fake_flag(monkeypatch):
    monkeypatch.delenv("SEEDGRAPH_FAKE_MARKER", raising=False)
    assert isinstance(_default_backend(), WarmMarkerBackend)
    monkeypatch.setenv("SEEDGRAPH_FAKE_MARKER", "1")
    assert isinstance(_default_backend(), FakeMarkerBackend)


def test_marker_result_new_fields_are_optional():
    r = MarkerResult("# x", None, [])
    assert r.page_count is None and r.timings == {} and r.converter_version is None


