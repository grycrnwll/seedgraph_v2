"""Stub engine for tests (``SG_CONVERTER_ENGINE=fake``): no Marker, torch or GPU.

Pages are counted as ``/Type /Page`` objects in the file. Behaviour tokens inside the
PDF bytes drive failure modes, so tests exercise the real worker loop:

- ``FAKE:ERROR``   raise inside convert (an error result)
- ``FAKE:PRINT``   print to sys.stdout and write to fd 1 mid-convert
- ``FAKE:UNICODE`` emit non-ASCII markdown
- ``FAKE:CRASH``   kill the process mid-convert
- ``FAKE:HANG``    never return
- ``FAKE:SLOW``    sleep ``SG_CONVERTER_FAKE_SLOW_SECONDS`` (default 0.5) first

``SG_CONVERTER_FAKE_LOAD_SECONDS`` delays "model loading"; ``SG_CONVERTER_FAKE_VERSION``
overrides the reported converter version.
"""

from __future__ import annotations

import os
import re
import sys
import time

_PAGE = re.compile(rb"/Type\s*/Page(?!s)")


class FakeEngine:
    def __init__(self) -> None:
        delay = float(os.environ.get("SG_CONVERTER_FAKE_LOAD_SECONDS", "0") or 0)
        if delay:
            time.sleep(delay)

    def info(self) -> dict:
        return {
            "version": os.environ.get("SG_CONVERTER_FAKE_VERSION", "1.10.2"),
            "torch": None,
            "cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def convert(self, pdf_path: str, options: dict) -> dict:
        data = open(pdf_path, "rb").read()
        if b"FAKE:CRASH" in data:
            os._exit(3)
        if b"FAKE:HANG" in data:
            time.sleep(3600)
        if b"FAKE:SLOW" in data:
            time.sleep(float(os.environ.get("SG_CONVERTER_FAKE_SLOW_SECONDS", "0.5")))
        if b"FAKE:ERROR" in data:
            raise RuntimeError("fake engine failure")
        if b"FAKE:PRINT" in data:
            print("noise from a library")
            sys.stdout.flush()
            os.write(1, b"raw fd1 noise\n")
        pages = max(1, len(_PAGE.findall(data)))
        body = "Body text β̂ ≤ ∞" if b"FAKE:UNICODE" in data else "Body text."
        parts = []
        for i in range(pages):
            if options.get("paginate_output", True):
                parts.append("\n\n{" + str(i) + "}" + "-" * 48 + "\n\n")
            if i == 0:
                parts.append("# Title\n\n## Introduction\n\n")
            parts.append(f"{body} Page {i}.\n")
        parts.append("\n## References\n\n[1] A reference.\n")
        return {
            "markdown": "".join(parts),
            "page_count": pages,
            "metadata": {"options": dict(options), "engine": "fake"},
            "warnings": [],
        }
