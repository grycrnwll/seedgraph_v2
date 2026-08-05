"""Streamed SHA-256 helpers — lowercase hex, the content-address primitive.

All cache content addresses (``sf_``/``md_`` ids, ``file_hash``/``markdown_hash``
columns) derive from these (D1). Hashes are SHA-256, lowercase hex, computed
once and stored bare in indexed columns. These three helpers are pure and
trivial (the only non-stub logic in this file), so they are implemented in full.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB — streaming chunk so large PDFs never load fully in RAM.


def sha256_file(path: Path) -> str:
    """Return the SHA-256 (lowercase hex) of a file's bytes, streamed in 1 MiB
    chunks. Chunk boundaries do not affect the digest (a phase_1 test pins this).
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 (lowercase hex) of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 (lowercase hex) of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
