"""Process-global background marker-conversion queue for ``seedgraph serve``.

Uploaded seed PDFs are ingested synchronously (fast: hash/dedup/store) on the
request thread, then their slow PDF->markdown conversion is ENQUEUED here and drained
by ``N`` daemon worker threads, where ``N`` is the size of the shared warm converter
pool (:func:`seedgraph.conversion.get_pool`: one converter per GPU). Unlike
:mod:`seedgraph.web.jobs` (per-project, single active job), this queue is
process-GLOBAL with ``N`` concurrent workers.

Each queue worker converts IN this process: it calls
:func:`acquisition.service.convert_and_bridge`, whose default backend sends the PDF to
a warm converter child (models loaded once, pinned to one GPU) and saves the returned
markdown here. The converter never touches a database. Marker and torch are never
imported in the serve process.

``N`` is resolved ONCE when the queue starts (pool size; see
:mod:`seedgraph.conversion.pool` for ``SEEDGRAPH_MARKER_SLOTS`` /
``SEEDGRAPH_MARKER_TIMEOUT`` / idle shutdown).

State is in-memory only -- there is no persistence. A restart with items still queued
ORPHANS them; re-upload to recover (content dedup makes that cheap). Per-task
exceptions (including a converter crash or timeout, which the pool turns into an
error after killing and later restarting that converter) are caught and recorded as
``failed`` (with the message); a worker never dies on a task exception and never
re-raises into the request.
"""

from __future__ import annotations

import queue
import threading
import time

#: One conversion task: (slug, work_id, source_file_id, file_hash, root).
#: Process-global FIFO; tasks enqueued before workers spawn wait here.
_QUEUE: "queue.Queue[tuple]" = queue.Queue()

#: Guards the pool lifecycle flags, the status map, and the converting count.
_LOCK = threading.Lock()

#: ``(slug, work_id) -> {"state": str, "detail": str}``. States: queued / converting
#: / failed. A done task is REMOVED (the corpus row's ``has_markdown`` is then truth).
_STATUS: dict[tuple[str, str], dict] = {}

#: Pool lifecycle (all under ``_LOCK``).
_STARTED = False
_STARTING = False
_SLOTS: int | None = None
_POOL: list[threading.Thread] = []
#: Tasks currently mid-conversion (for ``wait_idle``).
_CONVERTING = 0

#: Sentinel pushed onto the queue to retire a worker (test teardown only).
_SENTINEL = object()

def _resolve_slots() -> int:
    """Worker count = the shared converter pool's size (one per GPU; no torch)."""
    from ..conversion import get_pool

    return max(1, get_pool().size)


def _run_conversion(slug, work_id, source_file_id, file_hash, *, root=None) -> None:
    """Convert one already-ingested source file through the warm pool and bridge it,
    in this process (slow). Raises on failure (recorded by the worker). Module-level
    so tests can monkeypatch this dispatch seam."""
    from ..acquisition.service import convert_and_bridge
    from ..project import service as project_service

    h = project_service.open_project(slug, root=root)
    convert_and_bridge(
        h, work_id=work_id, source_file_id=source_file_id, file_hash=file_hash,
        cache_root=root,
    )
    # Best-effort bibliographic backfill: replace the filename-derived title + attach
    # ids from the converted markdown. Guarded so it can NEVER fail an
    # already-bridged conversion (the markdown is already durable).
    try:
        from ..extraction.metadata import best_effort_backfill

        best_effort_backfill(h, work_id)
    except Exception as exc:  # noqa: BLE001 - backfill/import must not fail conversion
        import logging

        logging.getLogger(__name__).warning(
            "metadata backfill skipped for %s: %s: %s", work_id, type(exc).__name__, exc
        )


def _worker(idx: int) -> None:
    """Drain the global queue forever; one task at a time per worker. The pool hands
    each conversion to whichever converter (GPU) is free."""
    global _CONVERTING
    while True:
        task = _QUEUE.get()
        if task is _SENTINEL:
            _QUEUE.task_done()
            return
        slug, work_id, source_file_id, file_hash, root = task
        with _LOCK:
            _CONVERTING += 1
            _STATUS[(slug, work_id)] = {"state": "converting", "detail": ""}
        try:
            _run_conversion(slug, work_id, source_file_id, file_hash, root=root)
            with _LOCK:
                _STATUS.pop((slug, work_id), None)
        except Exception as exc:  # noqa: BLE001 -- record + survive (incl. crash / timeout)
            with _LOCK:
                _STATUS[(slug, work_id)] = {"state": "failed", "detail": str(exc)}
        finally:
            with _LOCK:
                _CONVERTING -= 1
            _QUEUE.task_done()


def _start_pool() -> None:
    """Resolve slots (may run ``nvidia-smi``) and spawn the worker threads -- OFF the
    request thread (run from the one-shot starter in :func:`_ensure_pool_started`)."""
    global _STARTED, _STARTING, _SLOTS, _POOL
    slots = _resolve_slots()
    with _LOCK:
        if _STARTED:
            _STARTING = False
            return
        _SLOTS = slots
        pool = []
        for i in range(slots):
            t = threading.Thread(
                target=_worker, args=(i,), name=f"seedgraph-marker-{i}", daemon=True
            )
            t.start()
            pool.append(t)
        _POOL = pool
        _STARTED = True
        _STARTING = False


def _ensure_pool_started() -> None:
    """Idempotently kick off the pool WITHOUT blocking the caller: the start runs on a
    one-shot daemon starter thread."""
    global _STARTING
    with _LOCK:
        if _STARTED or _STARTING:
            return
        _STARTING = True
    threading.Thread(
        target=_start_pool, name="seedgraph-marker-starter", daemon=True
    ).start()


def enqueue(slug, work_id, source_file_id, file_hash, *, root=None) -> None:
    """Mark ``work_id`` ``queued`` and enqueue its conversion; ensure the pool runs.

    Non-blocking: the task is parked on the global FIFO and the pool is started off
    the request thread. ``root`` is captured now (default ``None`` => the worker
    resolves ``$SEEDGRAPH_HOME``)."""
    with _LOCK:
        _STATUS[(slug, work_id)] = {"state": "queued", "detail": ""}
    _QUEUE.put((slug, work_id, source_file_id, file_hash, root))
    _ensure_pool_started()


def status(slug: str) -> dict:
    """Return ``{work_id: {"state","detail"}}`` for ``slug``'s pending/failed works."""
    with _LOCK:
        return {wid: dict(st) for (s, wid), st in _STATUS.items() if s == slug}


def slots() -> int | None:
    """Resolved worker-slot count, or ``None`` if the pool has not started yet."""
    with _LOCK:
        return _SLOTS


def queue_depth() -> int:
    """Number of tasks waiting in the FIFO (not counting in-flight conversions)."""
    return _QUEUE.qsize()


def wait_idle(timeout: float = 60.0) -> bool:
    """Block until the queue is drained AND nothing is converting; ``True`` if idle,
    ``False`` on timeout (for tests / graceful checks)."""
    deadline = time.monotonic() + timeout
    while True:
        with _LOCK:
            converting = _CONVERTING
        if _QUEUE.unfinished_tasks == 0 and converting == 0:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def _reset_for_tests() -> None:
    """TEST-ONLY: retire the pool and clear all state so the next ``enqueue`` re-reads
    the pool size. Not production."""
    global _STARTED, _STARTING, _SLOTS, _POOL, _QUEUE, _CONVERTING
    for _ in range(1000):  # let any in-flight async start settle first
        with _LOCK:
            if not _STARTING:
                break
        time.sleep(0.005)
    with _LOCK:
        pool = list(_POOL)
    for _ in pool:
        _QUEUE.put(_SENTINEL)
    for t in pool:
        t.join(timeout=2.0)
    with _LOCK:
        _POOL = []
        _STARTED = False
        _STARTING = False
        _SLOTS = None
        _CONVERTING = 0
        _STATUS.clear()
        _QUEUE = queue.Queue()
