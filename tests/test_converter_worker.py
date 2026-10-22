"""The converter worker's pipe protocol (docs/CONVERTER_PROTOCOL.md).

Runs the REAL worker loop (``converter/src/sg_converter/worker.py``) under the app's
own Python with the stub engine (``SG_CONVERTER_ENGINE=fake``), so no Marker, torch or
GPU is needed. The stub's behaviour is driven by tokens inside the (synthetic) PDF.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONVERTER_SRC = REPO / "converter" / "src"


def _spawn(**extra_env) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CONVERTER_SRC)
    env["SG_CONVERTER_ENGINE"] = "fake"
    env.update(extra_env)
    return subprocess.Popen(
        [sys.executable, "-m", "sg_converter.worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env,
    )


def _send(proc, msg: dict) -> None:
    proc.stdin.write(json.dumps(msg).encode("utf-8") + b"\n")
    proc.stdin.flush()


def _recv(proc) -> dict:
    line = proc.stdout.readline()
    assert line, f"worker closed stdout; stderr={proc.stderr.read()!r}"
    return json.loads(line.decode("utf-8"))


def _pdf(tmp_path: Path, name: str, body: str, pages: int = 2) -> Path:
    page_objs = b"".join(b"<< /Type /Page >>\n" for _ in range(pages))
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4\n" + page_objs + body.encode("utf-8"))
    return p


@pytest.fixture
def worker():
    proc = _spawn()
    yield proc
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)
    for s in (proc.stdin, proc.stdout, proc.stderr):
        s.close()


def test_ready_handshake_reports_protocol_and_versions(worker):
    ready = _recv(worker)
    assert ready["type"] == "ready"
    assert ready["protocol"] == 1
    assert ready["converter"]["name"] == "marker-pdf"
    assert isinstance(ready["converter"]["version"], str) and ready["converter"]["version"]
    assert ready["python"].count(".") == 2
    assert isinstance(ready["load_seconds"], float)
    assert "cuda_device" in ready and "torch" in ready


def test_convert_ok_returns_markdown_page_count_and_timings(worker, tmp_path):
    _recv(worker)
    pdf = _pdf(tmp_path, "a.pdf", "hello body", pages=3)
    _send(worker, {"type": "convert", "id": "r1", "pdf_path": str(pdf),
                   "options": {"paginate_output": True, "force_ocr": False,
                               "output_format": "markdown"}})
    res = _recv(worker)
    assert res["type"] == "result" and res["id"] == "r1" and res["ok"] is True
    assert res["page_count"] == 3
    assert res["markdown"].count("-" * 48) == 3  # one page delimiter per page
    assert isinstance(res["metadata"], dict)
    assert isinstance(res["warnings"], list)
    assert res["timings"]["convert_seconds"] >= 0


def test_options_reach_the_engine(worker, tmp_path):
    _recv(worker)
    pdf = _pdf(tmp_path, "a.pdf", "x")
    _send(worker, {"type": "convert", "id": "r1", "pdf_path": str(pdf),
                   "options": {"paginate_output": True, "force_ocr": True}})
    res = _recv(worker)
    assert res["metadata"]["options"]["force_ocr"] is True


def test_convert_error_is_a_result_not_a_crash(worker, tmp_path):
    _recv(worker)
    bad = _pdf(tmp_path, "bad.pdf", "FAKE:ERROR")
    _send(worker, {"type": "convert", "id": "e1", "pdf_path": str(bad), "options": {}})
    res = _recv(worker)
    assert res == {"type": "result", "id": "e1", "ok": False,
                   "error": "RuntimeError: fake engine failure"}
    # The worker is still alive and serves the next job.
    good = _pdf(tmp_path, "good.pdf", "fine")
    _send(worker, {"type": "convert", "id": "g1", "pdf_path": str(good), "options": {}})
    assert _recv(worker)["ok"] is True


def test_missing_pdf_is_an_error_result(worker, tmp_path):
    _recv(worker)
    _send(worker, {"type": "convert", "id": "m1",
                   "pdf_path": str(tmp_path / "nope.pdf"), "options": {}})
    res = _recv(worker)
    assert res["ok"] is False and res["id"] == "m1"
    assert res["error"].startswith("FileNotFoundError")


def test_library_prints_cannot_corrupt_the_protocol(worker, tmp_path):
    """The stub prints to sys.stdout AND writes to fd 1 mid-convert; the protocol
    stream must still be exactly one JSON object per line."""
    _recv(worker)
    noisy = _pdf(tmp_path, "noisy.pdf", "FAKE:PRINT")
    _send(worker, {"type": "convert", "id": "n1", "pdf_path": str(noisy), "options": {}})
    res = _recv(worker)
    assert res["id"] == "n1" and res["ok"] is True
    _send(worker, {"type": "ping"})
    assert _recv(worker) == {"type": "pong"}
    _send(worker, {"type": "shutdown"})
    worker.wait(timeout=10)
    assert worker.stdout.read() == b""
    assert b"noise from a library" in worker.stderr.read()


def test_non_ascii_markdown_survives_the_pipe(worker, tmp_path):
    _recv(worker)
    pdf = _pdf(tmp_path, "u.pdf", "FAKE:UNICODE")
    _send(worker, {"type": "convert", "id": "u1", "pdf_path": str(pdf), "options": {}})
    res = _recv(worker)
    assert "β̂ ≤ ∞" in res["markdown"]


def test_shutdown_message_exits_cleanly(worker):
    _recv(worker)
    _send(worker, {"type": "shutdown"})
    assert worker.wait(timeout=10) == 0


def test_stdin_eof_exits_cleanly(worker):
    _recv(worker)
    worker.stdin.close()
    assert worker.wait(timeout=10) == 0


def test_unknown_message_type_gets_an_error_reply(worker):
    _recv(worker)
    _send(worker, {"type": "bogus", "id": "b1"})
    res = _recv(worker)
    assert res["type"] == "error" and "bogus" in res["error"]
