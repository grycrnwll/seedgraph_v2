"""Helpers that run the real converter worker loop with its stub engine.

The worker is started with the app's own Python and ``SG_CONVERTER_ENGINE=fake``, so
no Marker, torch or GPU is involved. Behaviour is chosen by tokens written into the
synthetic PDFs (see ``converter/src/sg_converter/fake_engine.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

from seedgraph.conversion import ConverterPool

REPO = Path(__file__).resolve().parents[2]
CONVERTER_SRC = REPO / "converter" / "src"
FAKE_COMMAND = [sys.executable, "-m", "sg_converter.worker"]


def fake_env(**extra: str) -> dict[str, str]:
    env = {"PYTHONPATH": str(CONVERTER_SRC), "SG_CONVERTER_ENGINE": "fake"}
    env.update(extra)
    return env


def fake_pool(gpus=(None,), **kw) -> ConverterPool:
    extra_env = fake_env(**kw.pop("extra_env", {}))
    kw.setdefault("expected_version", "1.10.2")
    return ConverterPool(list(gpus), command=FAKE_COMMAND, extra_env=extra_env, **kw)


def synthetic_pdf(dir_: Path, name: str, body: str = "", pages: int = 2) -> Path:
    page_objs = b"".join(b"<< /Type /Page >>\n" for _ in range(pages))
    p = Path(dir_) / name
    p.write_bytes(b"%PDF-1.4\n" + page_objs + body.encode("utf-8"))
    return p
