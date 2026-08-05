"""Background job runner for ``seedgraph serve`` (Track 2, review #7).

``launch_job`` mints a **fresh** run, runs the job body on a daemon thread, and
guarantees exactly one terminal progress event — ``finished`` on a clean return,
``failed`` (level=error) on any exception, which is swallowed so it never re-raises
into the launching request. At most one job runs per project at a time: a second
concurrent launch is rejected with :class:`ProjectBusyError` (a clear "project
busy" signal, not a crash; ponytail — single-user ceiling, queue later if needed).

The worker opens its **own** :class:`ProjectHandle` inside the thread, so a job
never shares the request thread's DB connection. There is no DB table and no
cross-restart persistence: the only job artifacts are the run's ``events.jsonl``
(via :func:`seedgraph.run.append_event`) and the process-local liveness map below.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from ..errors import SeedgraphError
from ..project import service as project_service
from ..run import append_event, ensure_run

#: Process-local liveness map: ``slug -> active run_id`` (the single-job ceiling).
_RUNNING: dict[str, str] = {}
#: Guards ``_RUNNING`` AND the launch admission check so two launches can't both win.
_REGISTRY_LOCK = threading.Lock()


class ProjectBusyError(SeedgraphError):
    """Raised when a launch races an already-running job for the same project."""


def is_running(slug: str) -> bool:
    """Return whether a job is currently active for ``slug`` (liveness probe)."""
    with _REGISTRY_LOCK:
        return slug in _RUNNING


def active_run_id(slug: str) -> str | None:
    """Return the run id of ``slug``'s active job, or ``None`` if idle."""
    with _REGISTRY_LOCK:
        return _RUNNING.get(slug)


def launch_job(
    slug: str,
    *,
    phase: str,
    fn: Callable[..., None],
    root: Path | str | None = None,
) -> str:
    """Mint a fresh run and run ``fn(emit, handle)`` on a daemon thread; return run_id.

    ``emit(event, message="", *, level="info", **data) -> int`` appends a progress
    event to the new run's log. ``fn`` receives a :class:`ProjectHandle` opened
    **inside the worker thread** (never the caller's request-thread connection).

    Exactly one terminal event is guaranteed by a ``try/finally``: ``finished`` on a
    clean return, ``failed`` (level=error) on any exception — the exception is logged
    as that event and swallowed, never re-raised into the launching request.

    At most one job per project: a concurrent launch raises
    :class:`ProjectBusyError` (admission check + liveness mark are atomic under
    ``_REGISTRY_LOCK``, so the run dir is only minted once the slot is won).
    """
    with _REGISTRY_LOCK:
        if slug in _RUNNING:
            raise ProjectBusyError(
                f"a job is already running for project {slug!r} "
                f"(run {_RUNNING[slug]}); only one job per project at a time"
            )
        run_id = ensure_run(slug, root=root)  # run_id=None -> fresh mint
        _RUNNING[slug] = run_id

    def emit(event: str, message: str = "", *, level: str = "info", **data) -> int:
        return append_event(
            slug,
            run_id,
            phase=phase,
            event=event,
            message=message,
            level=level,
            data=data or None,
            root=root,
        )

    # Expose the job's run_id on the emitter so a body that writes a run-scoped
    # artifact (manifest / graph.json / saved answer) targets the SAME run the
    # events stream to — never a second, orphaned run dir.
    emit.run_id = run_id  # type: ignore[attr-defined]

    def _worker() -> None:
        try:
            # Open our OWN handle/connection in this thread — never the caller's.
            handle = project_service.open_project(
                slug, root=Path(root) if root is not None else None
            )
            emit("started", f"{phase} started")
            fn(emit, handle)
            emit("finished", f"{phase} complete")
        except Exception as exc:  # noqa: BLE001 — terminal event, never re-raise into request
            emit("failed", str(exc), level="error")
        finally:
            with _REGISTRY_LOCK:
                _RUNNING.pop(slug, None)

    thread = threading.Thread(
        target=_worker, name=f"seedgraph-job-{slug}", daemon=True
    )
    thread.start()
    return run_id
