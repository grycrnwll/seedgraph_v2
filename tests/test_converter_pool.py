"""Warm converters and the per-GPU pool (seedgraph.conversion).

Every test drives the REAL worker loop (stub engine, app Python) through the pool's
public interface: ``ConverterPool.convert`` / ``converter_version`` / ``shutdown``.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from seedgraph.conversion import ConverterError, ConverterPool, discover_gpus
from seedgraph.conversion import pool as pool_mod
from support.converter import fake_pool, synthetic_pdf


@pytest.fixture
def pools():
    made: list[ConverterPool] = []

    def make(**kw) -> ConverterPool:
        p = fake_pool(**kw)
        made.append(p)
        return p

    yield make
    for p in made:
        p.shutdown()


def test_convert_returns_markdown_pages_timings_and_version(pools, tmp_path):
    pool = pools()
    out = pool.convert(synthetic_pdf(tmp_path, "a.pdf", pages=3))
    assert out.page_count == 3
    assert out.markdown.count("{2}" + "-" * 48) == 1
    assert out.converter_version == "1.10.2"
    assert out.timings["convert_seconds"] >= 0
    assert out.timings["wall_seconds"] >= out.timings["convert_seconds"]
    assert out.metadata["options"] == {
        "paginate_output": True, "force_ocr": False, "output_format": "markdown"
    }


def test_converter_starts_lazily_and_loads_once(pools, tmp_path):
    pool = pools()
    assert pool.running_count() == 0  # nothing spawned at construction
    pool.convert(synthetic_pdf(tmp_path, "a.pdf"))
    pid = pool.converter_pids()[0]
    out = pool.convert(synthetic_pdf(tmp_path, "b.pdf"))
    assert pool.converter_pids()[0] == pid  # same warm process served both
    assert "load_seconds" not in out.timings  # model load not paid again


def test_options_override_defaults(pools, tmp_path):
    pool = pools()
    out = pool.convert(synthetic_pdf(tmp_path, "a.pdf"), {"force_ocr": True})
    assert out.metadata["options"]["force_ocr"] is True
    assert out.metadata["options"]["paginate_output"] is True


def test_convert_error_raises_and_converter_stays_warm(pools, tmp_path):
    pool = pools()
    with pytest.raises(ConverterError, match="fake engine failure"):
        pool.convert(synthetic_pdf(tmp_path, "bad.pdf", "FAKE:ERROR"))
    pid = pool.converter_pids()[0]
    assert pool.convert(synthetic_pdf(tmp_path, "ok.pdf")).page_count == 2
    assert pool.converter_pids()[0] == pid


def test_child_crash_mid_pdf_fails_that_pdf_and_next_job_restarts(pools, tmp_path):
    pool = pools()
    pool.convert(synthetic_pdf(tmp_path, "warm.pdf"))
    first_pid = pool.converter_pids()[0]
    with pytest.raises(ConverterError, match="exited"):
        pool.convert(synthetic_pdf(tmp_path, "crash.pdf", "FAKE:CRASH"))
    assert pool.running_count() == 0
    out = pool.convert(synthetic_pdf(tmp_path, "next.pdf"))
    assert out.page_count == 2
    assert pool.converter_pids()[0] != first_pid


def test_external_kill_mid_pdf_fails_that_pdf_and_queue_continues(pools, tmp_path):
    pool = pools()
    pool.convert(synthetic_pdf(tmp_path, "warm.pdf"))
    errors: list[BaseException] = []

    def run():
        try:
            pool.convert(synthetic_pdf(tmp_path, "hang.pdf", "FAKE:HANG"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.5)
    pool.kill_converter(0)  # what an operator / OOM killer does to the child
    t.join(timeout=20)
    assert not t.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], ConverterError)
    assert pool.convert(synthetic_pdf(tmp_path, "after.pdf")).page_count == 2


def test_timeout_kills_child_fails_pdf_and_next_job_succeeds(pools, tmp_path):
    pool = pools(convert_timeout=1.0)
    pool.convert(synthetic_pdf(tmp_path, "warm.pdf"))
    pid = pool.converter_pids()[0]
    t0 = time.monotonic()
    with pytest.raises(ConverterError, match="timed out"):
        pool.convert(synthetic_pdf(tmp_path, "hang.pdf", "FAKE:HANG"))
    assert time.monotonic() - t0 < 15
    assert not _pid_alive(pid)
    assert pool.convert(synthetic_pdf(tmp_path, "next.pdf")).page_count == 2


def test_per_call_timeout_overrides_default(pools, tmp_path):
    pool = pools(convert_timeout=600)
    with pytest.raises(ConverterError, match="timed out"):
        pool.convert(synthetic_pdf(tmp_path, "hang.pdf", "FAKE:HANG"), timeout=1.0)


def test_ready_timeout_kills_a_child_that_never_becomes_ready(pools, tmp_path):
    pool = pools(ready_timeout=1.0,
                 extra_env={"SG_CONVERTER_FAKE_LOAD_SECONDS": "30"})
    with pytest.raises(ConverterError, match="ready"):
        pool.convert(synthetic_pdf(tmp_path, "a.pdf"))
    assert pool.running_count() == 0


def test_idle_converter_shuts_down_and_restarts_on_next_job(pools, tmp_path):
    pool = pools(idle_timeout=0.5)
    pool.convert(synthetic_pdf(tmp_path, "a.pdf"))
    pid = pool.converter_pids()[0]
    deadline = time.monotonic() + 10
    while pool.running_count() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pool.running_count() == 0
    assert not _pid_alive(pid)
    assert pool.convert(synthetic_pdf(tmp_path, "b.pdf")).page_count == 2


def test_each_converter_is_pinned_to_its_gpu_and_both_are_used(pools, tmp_path):
    pool = pools(gpus=("0", "1"), extra_env={"SG_CONVERTER_FAKE_SLOW_SECONDS": "1.0"})
    outs = []
    lock = threading.Lock()

    def run(i):
        out = pool.convert(synthetic_pdf(tmp_path, f"s{i}.pdf", "FAKE:SLOW"))
        with lock:
            outs.append(out)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert len(outs) == 2
    assert {o.gpu for o in outs} == {"0", "1"}
    # The worker itself saw the pin (stub reports CUDA_VISIBLE_DEVICES as its device).
    assert {o.cuda_device for o in outs} == {"0", "1"}
    assert time.monotonic() - t0 < 2 * 1.0 + 1.5  # ran in parallel (plus spawn)


def test_unpinned_converter_does_not_set_cuda_visible_devices(pools, tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    pool = pools(gpus=(None,))
    out = pool.convert(synthetic_pdf(tmp_path, "a.pdf"))
    assert out.gpu is None and out.cuda_device is None


def test_version_mismatch_with_lock_is_refused(pools, tmp_path):
    pool = pools(extra_env={"SG_CONVERTER_FAKE_VERSION": "9.9.9"})
    with pytest.raises(ConverterError, match="9.9.9"):
        pool.convert(synthetic_pdf(tmp_path, "a.pdf"))


def test_converter_version_needs_no_child_process(pools):
    pool = pools()
    assert pool.converter_version == "1.10.2"
    assert pool.running_count() == 0


def test_lock_version_is_read_from_converter_uv_lock():
    assert pool_mod.locked_marker_version() == "1.10.2"


def test_shutdown_stops_all_children(pools, tmp_path):
    pool = pools(gpus=("0", "1"))
    pool.convert(synthetic_pdf(tmp_path, "a.pdf"))
    pids = [p for p in pool.converter_pids() if p]
    pool.shutdown()
    assert pool.running_count() == 0
    assert all(not _pid_alive(p) for p in pids)


# --- GPU discovery (no torch in the app) -----------------------------------


def test_discover_gpus_env_override(monkeypatch):
    monkeypatch.setenv("SEEDGRAPH_MARKER_SLOTS", "3")
    monkeypatch.setattr(pool_mod, "_nvidia_smi_gpu_count", lambda: 2)
    assert discover_gpus() == ["0", "1", "0"]


def test_discover_gpus_from_nvidia_smi(monkeypatch):
    monkeypatch.delenv("SEEDGRAPH_MARKER_SLOTS", raising=False)
    monkeypatch.setattr(pool_mod, "_nvidia_smi_gpu_count", lambda: 2)
    assert discover_gpus() == ["0", "1"]


def test_discover_gpus_falls_back_to_one_unpinned_slot(monkeypatch):
    monkeypatch.delenv("SEEDGRAPH_MARKER_SLOTS", raising=False)
    monkeypatch.setattr(pool_mod, "_nvidia_smi_gpu_count", lambda: 0)
    assert discover_gpus() == [None]
    monkeypatch.setenv("SEEDGRAPH_MARKER_SLOTS", "garbage")
    assert discover_gpus() == [None]


def test_nvidia_smi_absent_counts_zero(monkeypatch):
    monkeypatch.setattr(pool_mod.shutil, "which", lambda name: None)
    assert pool_mod._nvidia_smi_gpu_count() == 0


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import subprocess

        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_crash_and_timeout_are_converter_crashes_but_error_results_are_not(pools, tmp_path):
    from seedgraph.conversion import ConverterCrashed

    pool = pools(convert_timeout=1.0)
    with pytest.raises(ConverterCrashed):
        pool.convert(synthetic_pdf(tmp_path, "crash.pdf", "FAKE:CRASH"))
    with pytest.raises(ConverterCrashed):
        pool.convert(synthetic_pdf(tmp_path, "hang.pdf", "FAKE:HANG"))
    with pytest.raises(ConverterError) as info:
        pool.convert(synthetic_pdf(tmp_path, "bad.pdf", "FAKE:ERROR"))
    assert not isinstance(info.value, ConverterCrashed)


# --- opt-in: the REAL converter environment on a real GPU (-m marker) --------


def _minimal_pdf(path, text="Hello warm converter. References"):
    """A valid one-page PDF with one line of text (no PDF library needed)."""
    stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


@pytest.mark.marker
def test_real_converter_round_trip(tmp_path):
    from seedgraph.conversion.pool import default_command, locked_marker_version

    pool = ConverterPool(["0"], command=default_command(),
                         expected_version=locked_marker_version())
    try:
        out = pool.convert(_minimal_pdf(tmp_path / "one.pdf"))
    finally:
        pool.shutdown()
    assert out.page_count == 1
    assert out.markdown.count("{0}" + "-" * 48) == 1
    assert "warm converter" in out.markdown
    assert out.converter_version == "1.10.2"
    assert out.ready["load_seconds"] > 0


def test_exit_shutdown_kills_a_busy_converter_instead_of_waiting(tmp_path):
    """Ctrl+C on serve mid-PDF must not block interpreter exit for the PDF timeout."""
    from seedgraph.conversion import set_pool

    pool = fake_pool(convert_timeout=600)
    pool.convert(synthetic_pdf(tmp_path, "warm.pdf"))
    pid = pool.converter_pids()[0]
    t = threading.Thread(
        target=lambda: pytest.raises(ConverterError, pool.convert,
                                     synthetic_pdf(tmp_path, "hang.pdf", "FAKE:HANG")),
        daemon=True)
    t.start()
    time.sleep(0.5)
    old = set_pool(pool)
    try:
        t0 = time.monotonic()
        pool_mod._shutdown_at_exit()
        assert time.monotonic() - t0 < 15
        assert not _pid_alive(pid)
    finally:
        set_pool(old)
        t.join(timeout=15)
