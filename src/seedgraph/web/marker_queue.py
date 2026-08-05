"""Process-global background marker-conversion queue for ``seedgraph serve``.

Uploaded seed PDFs are ingested synchronously (fast: hash/dedup/store) on the
request thread, then their slow Marker PDF->markdown conversion is ENQUEUED here and
drained by a pool of ``N = GPU slots`` daemon workers (one paper per GPU; the rest
wait FIFO). Unlike :mod:`seedgraph.web.jobs` (per-project, single active job), this
queue is process-GLOBAL with ``N`` concurrent workers.

Each conversion runs in its OWN subprocess pinned to a single GPU via
``CUDA_VISIBLE_DEVICES``: marker/Surya keep process-global model singletons + a
shared CUDA context and are NOT safe to run concurrently in one process (2-way
in-process concurrency reliably fails with a ``c10::Half`` overflow). A fresh
interpreter per conversion gives each its own CUDA context, so true "1 paper per
GPU" parallelism is safe. Worker ``i`` pins GPU ``i % num_gpus``; with no GPU
detected there is no pin (CPU; the single effective slot is fine).

# ponytail: model weights reload per task (~10-20s each) because every conversion is
# a fresh process. The upgrade is persistent warm worker processes (one per GPU) that
# import marker once and take tasks over a pipe; deferred while the reload overhead
# matters less than the isolation simplicity.

``N`` is resolved ONCE when the pool starts: ``SEEDGRAPH_MARKER_SLOTS`` (int, clamped
>=1) overrides; otherwise ``torch.cuda.device_count()`` if torch is importable
(CPU-only/no-torch => 1); otherwise 1. ``SEEDGRAPH_MARKER_TIMEOUT`` (seconds) caps a
single conversion subprocess (default 1800).

State is in-memory only -- there is no persistence. A restart with items still queued
ORPHANS them; re-upload to recover (content dedup makes that cheap). Per-task
exceptions (including a subprocess non-zero exit / timeout) are caught and recorded as
``failed`` (with the message); a worker never dies on a task exception and never
re-raises into the request.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
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
_NUM_GPUS = 0
_POOL: list[threading.Thread] = []
#: Tasks currently mid-conversion (for ``wait_idle``).
_CONVERTING = 0

#: Sentinel pushed onto the queue to retire a worker (test teardown only).
_SENTINEL = object()

#: Per-conversion subprocess timeout (seconds); ``SEEDGRAPH_MARKER_TIMEOUT`` overrides.
_CONVERT_TIMEOUT = int(os.environ.get("SEEDGRAPH_MARKER_TIMEOUT", "1800"))


def _resolve_slots() -> int:
    """Resolve the worker count ONCE at pool start (see module docstring)."""
    env = os.environ.get("SEEDGRAPH_MARKER_SLOTS")
    if env is not None and env.strip() != "":
        try:
            return max(1, int(env))
        except ValueError:
            return 1
    try:
        import torch  # noqa: PLC0415 -- optional, slow; off the request thread

        n = int(torch.cuda.device_count())
        return n if n >= 1 else 1
    except Exception:  # noqa: BLE001 -- no torch / no CUDA => single slot
        return 1


def _detect_num_gpus() -> int:
    """Number of CUDA devices (0 for CPU-only / no torch). Pins one GPU per worker."""
    try:
        import torch  # noqa: PLC0415

        return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        return 0


def _run_conversion(
    slug, work_id, source_file_id, file_hash, *, root=None, gpu_index=None
) -> None:
    """Convert one already-ingested source file in an ISOLATED subprocess pinned to
    one GPU, then bridge it (slow).

    Marker/Surya are not safe to run concurrently in one process, so each conversion
    gets its own fresh interpreter + CUDA context. ``gpu_index`` is pinned via
    ``CUDA_VISIBLE_DEVICES`` in the child's env (set BEFORE the child imports
    torch/marker); ``None`` => no pin (CPU). The child inherits the serve env (incl.
    ``$SEEDGRAPH_HOME``), so ``root`` is not forwarded. Raises on a non-zero exit or a
    timeout. Module-level so tests can monkeypatch this dispatch seam."""
    env = dict(os.environ)
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    proc = subprocess.run(
        [
            sys.executable, "-m", "seedgraph.web._convert_worker",
            slug, work_id, source_file_id, file_hash,
        ],
        env=env, capture_output=True, text=True, timeout=_CONVERT_TIMEOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            (proc.stderr or proc.stdout or "marker convert subprocess failed").strip()
        )


def _worker(idx: int) -> None:
    """Drain the global queue forever; one task at a time per worker (1 paper/GPU).

    ``idx`` is this worker's stable slot index; its conversions are pinned to GPU
    ``idx % _NUM_GPUS`` (or unpinned when no GPU is detected)."""
    global _CONVERTING
    gpu_index = (idx % _NUM_GPUS) if _NUM_GPUS else None
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
            _run_conversion(
                slug, work_id, source_file_id, file_hash,
                root=root, gpu_index=gpu_index,
            )
            with _LOCK:
                _STATUS.pop((slug, work_id), None)
        except Exception as exc:  # noqa: BLE001 -- record + survive (incl. subprocess timeout)
            with _LOCK:
                _STATUS[(slug, work_id)] = {"state": "failed", "detail": str(exc)}
        finally:
            with _LOCK:
                _CONVERTING -= 1
            _QUEUE.task_done()


def _start_pool() -> None:
    """Resolve slots + GPU count (may ``import torch``) and spawn the worker pool --
    OFF the request thread (run from the one-shot starter in
    :func:`_ensure_pool_started`)."""
    global _STARTED, _STARTING, _SLOTS, _NUM_GPUS, _POOL
    slots = _resolve_slots()
    num_gpus = _detect_num_gpus()
    with _LOCK:
        if _STARTED:
            _STARTING = False
            return
        _SLOTS = slots
        _NUM_GPUS = num_gpus
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
    """Idempotently kick off the pool WITHOUT blocking the caller: the (possibly slow,
    torch-importing) start runs on a one-shot daemon starter thread."""
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
    the request thread (no ``import torch`` on the request path). ``root`` is captured
    now (default ``None`` => the worker / subprocess resolves ``$SEEDGRAPH_HOME``)."""
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
    ``SEEDGRAPH_MARKER_SLOTS`` and re-resolves the slot / GPU counts. Not production."""
    global _STARTED, _STARTING, _SLOTS, _NUM_GPUS, _POOL, _QUEUE, _CONVERTING
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
        _NUM_GPUS = 0
        _CONVERTING = 0
        _STATUS.clear()
        _QUEUE = queue.Queue()
