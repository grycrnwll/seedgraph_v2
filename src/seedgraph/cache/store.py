"""Content-addressed filesystem store (plan §4 on-disk layout).

Flat content-addressed names (no ``{h2}`` sharding — over-engineering for a
single-user corpus; v1 did not shard)::

    ~/.seedgraph/cache/
        pdfs/{file_hash}.pdf
        html/{file_hash}.html
        markdown/{markdown_hash}.md
        marker/{conversion_run_id}/{manifest.json, meta.json}
        logs/

``storage_uri`` is stored **relative** to the cache root (decision 17) so the
cache is relocatable / cross-volume safe. Writes are atomic (temp file ->
``os.replace``); source bytes are **copied**, never symlinked, to survive a
cross-volume cache (C: vs D:).

Pure path derivations are trivial and implemented in full; the load-bearing IO
(atomic write/copy, read-by-uri) is stubbed for the full build.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .. import paths


def cache_root(root: Path | str | None = None) -> Path:
    """Resolve the cache root (``$SEEDGRAPH_HOME``/``--root``/default ``~/.seedgraph/cache``).

    Thin re-export of :func:`seedgraph.paths.cache_root` so cache callers have a
    single import surface.
    """
    return paths.cache_root(root)


def pdf_path(file_hash: str, root: Path | str | None = None) -> Path:
    """Absolute path of the stored PDF blob for ``file_hash`` (``pdfs/{hash}.pdf``)."""
    return cache_root(root) / "pdfs" / f"{file_hash}.pdf"


def html_path(file_hash: str, root: Path | str | None = None) -> Path:
    """Absolute path of the stored HTML blob for ``file_hash`` (``html/{hash}.html``)."""
    return cache_root(root) / "html" / f"{file_hash}.html"


def markdown_blob_path(markdown_hash: str, root: Path | str | None = None) -> Path:
    """Absolute path of the stored markdown blob (``markdown/{markdown_hash}.md``)."""
    return cache_root(root) / "markdown" / f"{markdown_hash}.md"


def marker_dir(conversion_run_id: str, root: Path | str | None = None) -> Path:
    """Absolute per-conversion artifact dir (``marker/{conversion_run_id}/``)
    holding ``manifest.json`` + best-effort ``meta.json``."""
    return cache_root(root) / "marker" / conversion_run_id


def relative_uri(path: Path, root: Path | str | None = None) -> str:
    """Return ``path`` as a POSIX string RELATIVE to the cache root (decision 17)."""
    return Path(path).resolve().relative_to(cache_root(root).resolve()).as_posix()


def resolve_uri(storage_uri: str, root: Path | str | None = None) -> Path:
    """Resolve a relative ``storage_uri`` back to an absolute path under the cache root."""
    return cache_root(root) / storage_uri


def write_bytes_atomic(data: bytes, dest: Path) -> Path:
    """Atomically write ``data`` to ``dest`` (temp file in the same dir then
    ``os.replace``); create parents; leave no temp file on success. Idempotent for
    content-addressed names (same hash -> same bytes). Returns ``dest``."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return dest


def copy_file_atomic(src: Path, dest: Path) -> Path:
    """Atomically COPY (never symlink) ``src`` -> ``dest`` via temp + ``os.replace``
    so the cache survives a cross-volume source. Returns ``dest``."""
    src = Path(src)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    try:
        os.close(fd)
        shutil.copyfile(src, tmp)  # COPY bytes (never symlink) — cross-volume safe
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return dest


def read_uri(storage_uri: str, root: Path | str | None = None) -> bytes:
    """Read the bytes of a stored blob addressed by its relative ``storage_uri``."""
    return resolve_uri(storage_uri, root).read_bytes()
