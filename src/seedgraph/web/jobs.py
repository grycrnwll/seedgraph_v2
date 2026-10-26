"""Background jobs in the ``serve`` process, and the cross-process converter lock
(docs/V3_UI_DESIGN.md §6, §1.2).

No Starlette or Jinja2 import here: ``seedgraph convert`` / ``library reconvert`` import
the lock API from this module.

**JobRunner** - one in-process runner with two kinds of worker thread:

- ``pool.size`` *conversion* threads, each calling :func:`seedgraph.library.files.convert_file`
  through the stage 0 warm converter pool, so every warm converter stays busy;
- ``workers`` *general* threads (default 1) for :meth:`JobRunner.submit` jobs (ingest,
  DOI lookup, Ollama metadata - wired in by the library routes, U7).

Every worker thread opens its own :class:`~seedgraph.library.store.Library` and closes it
when the thread ends; a ``Library`` is never shared across threads. The registry of jobs
is in memory (:class:`Job` snapshots for htmx polling); the durable truth is the
database. :meth:`JobRunner.start` resumes everything left in ``conversion_queue`` - once.
A failed conversion stays queued in the database (``convert_file``'s rule) but is not
retried in the same run: the next start, or an explicit :meth:`JobRunner.convert`,
picks it up. So a broken PDF can never spin the runner.

**ThreadLibraries** - one lazily opened ``Library`` per thread, for request handlers
(Starlette runs sync endpoints in a thread pool).

**Converter lock** - while ``serve`` runs it holds ``<home>/run/converter.lock``
(its PID). ``convert`` / ``reconvert`` call :func:`check_converter_free` and refuse with
:data:`SERVE_RUNNING_MESSAGE` while a live process holds it. A lock whose PID is gone
(or whose content is unreadable) is stale and ignored. Interface::

    lock = acquire_converter_lock(home)   # raises ConverterLockHeld(pid) if held
    ...; lock.release()                   # or: with acquire_converter_lock(home): ...
    check_converter_free(home)            # raises ConverterLockHeld with the message
    converter_lock_holder(home)           # live PID or None
"""

from __future__ import annotations

import itertools
import os
import queue
import sqlite3
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..errors import SeedgraphError
from ..library.store import Library, open_library_at

# --- converter lock ------------------------------------------------------------------------

LOCK_RELPATH = Path("run") / "converter.lock"
SERVE_RUNNING_MESSAGE = ("seedgraph serve is running; add or re-convert PDFs in the UI, "
                         "or stop it first")


class ConverterLockHeld(SeedgraphError):
    """Another live process (``seedgraph serve``) owns the converters."""

    def __init__(self, pid: int, path: Path) -> None:
        super().__init__(f"{SERVE_RUNNING_MESSAGE} (pid {pid}, lock {path})")
        self.pid = pid
        self.path = path


def converter_lock_path(home: Path | str) -> Path:
    return Path(home) / LOCK_RELPATH


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # NOT os.kill(pid, 0): on Windows that calls TerminateProcess.
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: exists, not ours
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def converter_lock_holder(home: Path | str) -> int | None:
    """The PID of the live process holding the lock, or ``None`` (absent or stale)."""
    pid = _read_pid(converter_lock_path(home))
    return pid if pid is not None and _pid_alive(pid) else None


def check_converter_free(home: Path | str) -> None:
    """Raise :class:`ConverterLockHeld` while ``serve`` holds the converters."""
    pid = converter_lock_holder(home)
    if pid is not None:
        raise ConverterLockHeld(pid, converter_lock_path(home))


@dataclass
class ConverterLock:
    path: Path
    pid: int
    released: bool = False

    def release(self) -> None:
        """Remove the lock file - only if it still holds this lock's PID."""
        if self.released:
            return
        self.released = True
        if _read_pid(self.path) == self.pid:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> ConverterLock:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def acquire_converter_lock(home: Path | str) -> ConverterLock:
    """Create ``<home>/run/converter.lock`` holding this PID (atomic ``O_EXCL``).

    A stale lock (dead PID, unreadable content) is replaced; a live holder - this
    process included - raises :class:`ConverterLockHeld`.
    """
    path = converter_lock_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            holder = converter_lock_holder(home)
            if holder is not None:
                raise ConverterLockHeld(holder, path) from None
            try:
                path.unlink()  # stale
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="ascii") as fh:
            fh.write(str(pid))
        return ConverterLock(path=path, pid=pid)
    holder = converter_lock_holder(home) or -1
    raise ConverterLockHeld(holder, path)


# --- per-thread libraries ------------------------------------------------------------------


class ThreadLibraries:
    """One lazily opened :class:`Library` per thread (never shared across threads)."""

    def __init__(self, lib_dir: Path | str) -> None:
        self.lib_dir = Path(lib_dir)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._opened: list[Library] = []

    def get(self) -> Library:
        lib = getattr(self._local, "lib", None)
        if lib is None:
            lib = open_library_at(self.lib_dir)
            self._local.lib = lib
            with self._lock:
                self._opened.append(lib)
        return lib

    def close_current(self) -> None:
        lib = getattr(self._local, "lib", None)
        if lib is not None:
            self._local.lib = None
            with self._lock:
                if lib in self._opened:
                    self._opened.remove(lib)
            lib.close()

    def close_all(self) -> None:
        """Best effort at shutdown: SQLite refuses to close another thread's connection,
        and those die with the process."""
        with self._lock:
            opened, self._opened = self._opened, []
        for lib in opened:
            try:
                lib.close()
            except sqlite3.ProgrammingError:
                pass


# --- jobs ----------------------------------------------------------------------------------

JobState = Literal["queued", "running", "done", "failed"]
CONVERT = "convert"


@dataclass(frozen=True)
class Job:
    """A snapshot of one job, for the jobs strip (htmx polling)."""

    job_id: str
    kind: str
    label: str
    state: JobState
    message: str = ""
    paper_ids: tuple[str, ...] = ()
    submitted_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def active(self) -> bool:
        return self.state in ("queued", "running")


@dataclass(frozen=True)
class JobsSummary:
    queued: int
    running: int
    failed: int
    done: int

    @property
    def active(self) -> int:
        return self.queued + self.running


@dataclass
class _Work:
    job_id: str
    fn: Callable[[Library], Any]
    convert_id: str | None = None


_STOP = object()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobRunner:
    """Background conversion and general jobs for the ``serve`` process (see module doc).

    ``pool`` is a stage 0 :class:`~seedgraph.conversion.ConverterPool` (or anything with
    ``size``, ``converter_version`` and ``convert``); ``None`` means the process-wide
    :func:`~seedgraph.conversion.get_pool`, resolved only when the runner starts.
    """

    def __init__(self, lib_dir: Path | str, *, pool: Any = None, workers: int = 1,
                 clock: Callable[[], datetime] | None = None, keep_finished: int = 200) -> None:
        self.lib_dir = Path(lib_dir)
        self._pool = pool
        self._workers = max(1, workers)
        self._clock = clock or _utc_now
        self._keep_finished = keep_finished
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._active_convert: dict[str, str] = {}  # source_file_id -> job_id
        self._convert_q: queue.Queue = queue.Queue()
        self._general_q: queue.Queue = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._ids = itertools.count(1)
        self._started = False

    # -- public interface ---------------------------------------------------------------

    @property
    def pool(self) -> Any:
        if self._pool is None:
            from ..conversion import get_pool  # local: get_pool probes GPUs on first use

            self._pool = get_pool()
        return self._pool

    def start(self) -> list[str]:
        """Start the worker threads and resume ``conversion_queue`` (once).

        Returns the job ids submitted for the queued files.
        """
        with self._lock:
            if self._started:
                return []
            self._started = True
        n_convert = max(1, int(getattr(self.pool, "size", 1)))
        for i in range(n_convert):
            self._spawn(f"sg-convert-{i}", self._convert_q)
        for i in range(self._workers):
            self._spawn(f"sg-jobs-{i}", self._general_q)
        return self.resume_queue()

    def resume_queue(self) -> list[str]:
        """Submit every file waiting in ``conversion_queue`` that has no active job."""
        from ..library import files

        lib = open_library_at(self.lib_dir)
        try:
            waiting = files.queued_files(lib)
        finally:
            lib.close()
        return [self.convert(sf) for sf in waiting]

    def convert(self, source_file_id: str, *, label: str | None = None) -> str:
        """Queue one conversion; an already active one for the file is returned as is."""
        with self._lock:
            existing = self._active_convert.get(source_file_id)
            if existing is not None:
                return existing
            job_id = self._new_job(CONVERT, label or source_file_id)
            self._active_convert[source_file_id] = job_id
        self._convert_q.put(_Work(job_id, self._convert_fn(source_file_id), convert_id=source_file_id))
        return job_id

    def submit(self, kind: str, fn: Callable[..., Any], *args: Any, label: str | None = None,
               **kwargs: Any) -> str:
        """Run ``fn(lib, *args, **kwargs)`` on a general worker with that thread's own
        ``Library``. A returned :class:`JobResult` (or string) becomes the job's message
        and paper ids; an exception marks the job failed with its message."""
        with self._lock:
            job_id = self._new_job(kind, label or kind)
        self._general_q.put(_Work(job_id, lambda lib: fn(lib, *args, **kwargs)))
        return job_id

    def jobs(self, active_only: bool = False) -> list[Job]:
        """Snapshots, oldest first."""
        with self._lock:
            return [j for j in self._jobs.values() if j.active or not active_only]

    def job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def summary(self) -> JobsSummary:
        with self._lock:
            states = [j.state for j in self._jobs.values()]
        return JobsSummary(queued=states.count("queued"), running=states.count("running"),
                           failed=states.count("failed"), done=states.count("done"))

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until no job is queued or running; ``False`` on timeout."""
        with self._idle:
            return self._idle.wait_for(
                lambda: not any(j.active for j in self._jobs.values()), timeout=timeout)

    def stop(self, timeout: float = 5.0, *, shutdown_pool: bool = False) -> None:
        """Ask the workers to finish their current job and exit, waiting at most
        ``timeout`` seconds in all. ``shutdown_pool`` (server exit) first stops the
        converters, so a conversion in flight fails at once instead of holding Ctrl+C."""
        import time

        if shutdown_pool and self._pool is not None:
            shutdown = getattr(self._pool, "shutdown", None)
            if shutdown is not None:
                shutdown()
        for t in self._threads:
            (self._convert_q if t.name.startswith("sg-convert") else self._general_q).put(_STOP)
        deadline = time.monotonic() + timeout
        for t in self._threads:
            t.join(max(0.0, deadline - time.monotonic()))
        self._threads = [t for t in self._threads if t.is_alive()]

    # -- internals ----------------------------------------------------------------------

    def _now(self) -> str:
        return self._clock().isoformat()

    def _new_job(self, kind: str, label: str) -> str:  # caller holds the lock
        job_id = f"job_{next(self._ids):05d}"
        self._jobs[job_id] = Job(job_id=job_id, kind=kind, label=label, state="queued",
                                 submitted_at=self._now())
        return job_id

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._idle:
            self._jobs[job_id] = replace(self._jobs[job_id], **changes)
            finished = [k for k, j in self._jobs.items() if not j.active]
            for k in finished[: max(0, len(finished) - self._keep_finished)]:
                del self._jobs[k]
            self._idle.notify_all()

    def _convert_fn(self, source_file_id: str) -> Callable[[Library], JobResult]:
        def run(lib: Library) -> JobResult:
            from ..conversion import WarmMarkerBackend
            from ..library import files

            outcome = files.convert_file(lib, source_file_id, backend=WarmMarkerBackend(self.pool))
            if outcome.status == "failed":
                raise JobFailed(outcome.error or "conversion failed", paper_ids=outcome.paper_ids)
            return JobResult(f"converted ({outcome.text_quality or 'text not assessed'})",
                             paper_ids=outcome.paper_ids)
        return run

    def _spawn(self, name: str, q: queue.Queue) -> None:
        t = threading.Thread(target=self._loop, args=(q,), name=name, daemon=True)
        self._threads.append(t)
        t.start()

    def _loop(self, q: queue.Queue) -> None:
        lib = open_library_at(self.lib_dir)  # this thread's own connection
        try:
            while True:
                work = q.get()
                if work is _STOP:
                    return
                self._run(lib, work)
        finally:
            lib.close()

    def _run(self, lib: Library, work: _Work) -> None:
        self._update(work.job_id, state="running", started_at=self._now())
        try:
            result = work.fn(lib)
        except JobFailed as exc:
            changes = dict(state="failed", message=str(exc), paper_ids=exc.paper_ids)
        except Exception as exc:  # a job's failure is reported, never kills the worker
            changes = dict(state="failed", message=f"{type(exc).__name__}: {exc}")
        else:
            if isinstance(result, JobResult):
                changes = dict(state="done", message=result.message, paper_ids=result.paper_ids)
            else:
                changes = dict(state="done", message="" if result is None else str(result))
        finally:
            if work.convert_id is not None:
                with self._lock:
                    self._active_convert.pop(work.convert_id, None)
        self._update(work.job_id, finished_at=self._now(), **changes)


@dataclass(frozen=True)
class JobResult:
    """What a job function may return: a message and the papers it touched."""

    message: str = ""
    paper_ids: tuple[str, ...] = field(default_factory=tuple)


class JobFailed(Exception):
    """Raise from a job function to fail it with a message and paper ids."""

    def __init__(self, message: str, *, paper_ids: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.paper_ids = tuple(paper_ids)
