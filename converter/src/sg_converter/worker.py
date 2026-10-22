"""Warm converter worker: loads Marker's models once, then converts PDFs over a pipe.

Run as::

    uv run --frozen --project converter python -m sg_converter.worker

Protocol (version 1, see docs/CONVERTER_PROTOCOL.md): newline-delimited UTF-8 JSON.
stdin carries requests from the app, the ORIGINAL stdout carries replies. Everything a
library prints (tqdm bars, warnings, C-level writes to fd 1) is redirected to stderr
before any heavy import, so it can never corrupt the protocol stream.

This module never imports seedgraph and never touches any database: the app saves
whatever it returns.

``SG_CONVERTER_ENGINE=fake`` swaps Marker for a tiny stub engine
(:mod:`sg_converter.fake_engine`) so the app's tests can drive this exact loop with no
Marker, torch or GPU installed.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback

PROTOCOL_VERSION = 1


class _Protocol:
    """Binary, line-oriented JSON channel over the saved real stdout."""

    def __init__(self, out_fd: int) -> None:
        self._out = os.fdopen(out_fd, "wb", buffering=0)

    def send(self, msg: dict) -> None:
        data = json.dumps(msg, ensure_ascii=False, default=str).encode("utf-8") + b"\n"
        self._out.write(data)
        self._out.flush()


def _isolate_stdout() -> int:
    """Keep the real stdout for the protocol; point fd 1 and sys.stdout at stderr."""
    sys.stdout.flush()
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return proto_fd


def _load_engine():
    if os.environ.get("SG_CONVERTER_ENGINE") == "fake":
        from . import fake_engine

        return fake_engine.FakeEngine()
    from . import marker_engine

    return marker_engine.MarkerEngine()


def _convert(engine, msg: dict) -> dict:
    req_id = msg.get("id")
    try:
        pdf_path = msg["pdf_path"]
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"no such PDF: {pdf_path}")
        options = dict(msg.get("options") or {})
        t0 = time.perf_counter()
        out = engine.convert(pdf_path, options)
        seconds = time.perf_counter() - t0
        return {
            "type": "result",
            "id": req_id,
            "ok": True,
            "markdown": out["markdown"],
            "page_count": out["page_count"],
            "metadata": out.get("metadata") or {},
            "warnings": list(out.get("warnings") or []),
            "timings": {"convert_seconds": round(seconds, 4)},
        }
    except Exception as exc:  # noqa: BLE001 - every failure becomes a result message
        traceback.print_exc(file=sys.stderr)
        return {"type": "result", "id": req_id, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    proto = _Protocol(_isolate_stdout())
    stdin = sys.stdin.buffer

    t0 = time.perf_counter()
    engine = _load_engine()
    load_seconds = time.perf_counter() - t0
    info = engine.info()
    proto.send({
        "type": "ready",
        "protocol": PROTOCOL_VERSION,
        "converter": {"name": "marker-pdf", "version": info["version"]},
        "python": platform.python_version(),
        "torch": info.get("torch"),
        "cuda_device": info.get("cuda_device"),
        "load_seconds": round(load_seconds, 4),
    })

    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line.decode("utf-8"))
            kind = msg.get("type")
        except Exception as exc:  # noqa: BLE001
            proto.send({"type": "error", "error": f"bad message: {exc}"})
            continue
        if kind == "convert":
            proto.send(_convert(engine, msg))
        elif kind == "ping":
            proto.send({"type": "pong"})
        elif kind == "shutdown":
            return 0
        else:
            proto.send({"type": "error", "id": msg.get("id"),
                        "error": f"unknown message type: {kind!r}"})
    return 0  # stdin EOF: the app went away


if __name__ == "__main__":
    raise SystemExit(main())
