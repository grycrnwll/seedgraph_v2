"""Run-manager substrate (decision r2-6 / D5) — filesystem + JSON only.

No ``runs`` table, no model call. ``ensure_run`` mints/threads a ``run_id`` and
creates ``projects/{slug}/runs/{run_id}/``; ``update_manifest`` performs
**shallow, section-keyed atomic replacement** of each TOP-LEVEL section key
(temp file + ``os.replace``). There is NO deep-merge and NO cross-stage
accumulation: each stage owns disjoint top-level sections, and a second write to
an already-present section is a contract violation (raises). The manifest file
is the only run artifact.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from . import paths
from .errors import SeedgraphError


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunManifest:
    run_id: str
    created_at: str
    sections: dict = field(default_factory=dict)


def validate_run_id(run_id: str) -> None:
    """Run IDs are names, never user-controlled relative or absolute paths."""
    if (not isinstance(run_id, str) or len(run_id) > 120
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id)):
        raise SeedgraphError("invalid run ID; use the ID returned by prepare")


@contextmanager
def batch_writer(slug: str, run_id: str, *, root=None):
    """Nonblocking OS lock, automatically released when the host process dies."""
    directory = _run_dir(slug, run_id, root)
    if not directory.is_dir():
        raise SeedgraphError("unknown extraction batch")
    with open(directory / "writer.lock", "a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SeedgraphError("another process is writing this batch; retry later") from exc
        try:
            yield directory
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _run_dir(slug: str, run_id: str, root: Path | str | None) -> Path:
    validate_run_id(run_id)
    return paths.project_dir(slug, root) / "runs" / run_id


def _manifest_path(slug: str, run_id: str, root: Path | str | None) -> Path:
    return _run_dir(slug, run_id, root) / "manifest.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def ensure_run(slug: str, *, run_id: str | None = None, root: Path | str | None = None) -> str:
    """Mint (or thread an explicit) ``run_id`` and create its run directory.

    Threading an explicit ``run_id`` is idempotent — an existing manifest is not
    clobbered.
    """
    paths.validate_slug(slug)
    if run_id is None:
        run_id = f"run-{_utc_stamp()}-{uuid4().hex[:8]}"
    run_dir = _run_dir(slug, run_id, root)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        _atomic_write_json(
            manifest_path,
            {"run_id": run_id, "created_at": _now_iso(), "sections": {}},
        )
    return run_id


def update_manifest(
    slug: str,
    run_id: str,
    sections: dict,
    *,
    root: Path | str | None = None,
) -> None:
    """Shallow, section-keyed atomic replacement of top-level manifest sections (D5).

    Raises :class:`SeedgraphError` if any provided top-level section key is already
    present (the same-section double-write contract violation).
    """
    paths.validate_slug(slug)
    manifest_path = _manifest_path(slug, run_id, root)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"run_id": run_id, "created_at": _now_iso(), "sections": {}}
    current = manifest.setdefault("sections", {})

    for key, value in sections.items():
        if key in current:
            raise SeedgraphError(
                f"manifest section {key!r} is already present; top-level sections are "
                f"owned by a single stage and written once (D5 — no deep-merge, no "
                f"cross-stage accumulation)."
            )
        current[key] = value

    _atomic_write_json(manifest_path, manifest)


# ---------------------------------------------------------------------------
# Run progress events (Track 2, review #7) — append-only ``events.jsonl``.
#
# The only job-progress artifact (no ``runs`` table, no DB). Each event is one
# JSON line; ``seq`` is a 0-indexed monotonic counter taken from the current line
# count. Appends are serialized per run by a process-local lock so the background
# job thread and a concurrent poller never interleave a partial line or duplicate
# a seq.
# ---------------------------------------------------------------------------

#: Per-run append locks, keyed by ``(slug, run_id)``; guarded by ``_EVENT_LOCKS_GUARD``.
_EVENT_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_EVENT_LOCKS_GUARD = threading.Lock()


def _event_lock(slug: str, run_id: str) -> threading.Lock:
    key = (slug, run_id)
    with _EVENT_LOCKS_GUARD:
        lock = _EVENT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _EVENT_LOCKS[key] = lock
        return lock


def _events_path(slug: str, run_id: str, root: Path | str | None) -> Path:
    return _run_dir(slug, run_id, root) / "events.jsonl"


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def append_event(
    slug: str,
    run_id: str,
    *,
    phase: str,
    event: str,
    message: str = "",
    level: str = "info",
    data: dict | None = None,
    root: Path | str | None = None,
) -> int:
    """Atomically append one progress event to ``runs/{run_id}/events.jsonl``.

    Returns the event's monotonic ``seq`` (0-indexed). Under a per-run
    :class:`threading.Lock` the seq is taken from the current line count, one JSON
    line is written in append mode and flushed — so the background job thread and a
    concurrent poller can never interleave a partial line or reuse a seq (review #7).
    """
    paths.validate_slug(slug)
    path = _events_path(slug, run_id, root)
    record: dict = {
        "phase": phase,
        "event": event,
        "level": level,
        "message": message,
        "data": data or {},
        "ts": _now_iso(),
    }
    with _event_lock(slug, run_id):
        path.parent.mkdir(parents=True, exist_ok=True)
        seq = _count_lines(path)
        record["seq"] = seq
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return seq


def read_events(
    slug: str,
    run_id: str,
    *,
    after: int = -1,
    root: Path | str | None = None,
) -> list[dict]:
    """Return run progress events with ``seq > after`` (all of them when ``after=-1``).

    A missing log yields ``[]``. No lock is taken on the read path: appends are
    atomic whole-line writes, so a reader only ever sees complete lines.
    """
    paths.validate_slug(slug)
    path = _events_path(slug, run_id, root)
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if int(record.get("seq", -1)) > after:
                out.append(record)
    return out


def _status_from_events(events: list[dict]) -> str | None:
    """Derive a run's coarse status from its terminal event.

    ``None`` when there are no events yet; ``finished`` / ``failed`` for the two
    guaranteed terminal job events; ``interrupted`` for the synthetic terminal
    :func:`mark_interrupted_runs` stamps on a crash-orphaned run (Build F ch9);
    otherwise ``running`` (a job is mid-flight).
    """
    if not events:
        return None
    last = events[-1].get("event")
    if last in ("finished", "failed", "interrupted", "paused", "awaiting_host"):
        return last
    return "running"


def mark_interrupted_runs(slug: str, *, root: Path | str | None = None) -> list[str]:
    """Append a synthetic ``interrupted`` terminal to every run of ``slug`` whose
    events.jsonl ends non-terminal; return the stamped run ids (Build F ch9).

    Called by ``serve`` for every project BEFORE uvicorn binds, so a run orphaned
    by a crash/restart can never report ``running`` forever. Safe by construction:
    only web jobs write job events today (CLI verbs use ``Progress`` with
    ``emit=None``, stdout-only), and web-job threads die with the serve process —
    so at sweep time any ``running`` log is provably orphaned. Runs with NO
    events (e.g. CLI cite runs, manifest-only) are untouched, as are terminal
    (``finished``/``failed``/``interrupted``) runs — which also makes a second
    sweep a no-op (idempotent). Revisit if CLI event-mirroring lands.
    """
    paths.validate_slug(slug)
    runs_dir = paths.project_dir(slug, root) / "runs"
    if not runs_dir.exists():
        return []
    stamped: list[str] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        run_id = run_dir.name
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {}
            if manifest.get("sections", {}).get("agent_extraction", {}).get("owner") == "external_host":
                continue
        events = read_events(slug, run_id, root=root)
        if _status_from_events(events) != "running":
            continue
        append_event(
            slug,
            run_id,
            phase=events[-1].get("phase", "run"),
            event="interrupted",
            level="warning",
            message="process restarted mid-run",
            root=root,
        )
        stamped.append(run_id)
    return stamped


# ---------------------------------------------------------------------------
# Public run inventory (Track 2) — the single lift of the ``_latest_run_id`` helper
# that web/routes.py and cli.py each duplicated (no behavior change).
# ---------------------------------------------------------------------------


def latest_run_id(slug: str, *, root: Path | str | None = None) -> str | None:
    """Return the most-recent run id (by mtime, then name) or ``None``."""
    paths.validate_slug(slug)
    runs_dir = paths.project_dir(slug, root) / "runs"
    if not runs_dir.exists():
        return None
    dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: (p.stat().st_mtime, p.name)).name


def read_manifest(
    slug: str, run_id: str, *, root: Path | str | None = None
) -> RunManifest:
    """Load ``runs/{run_id}/manifest.json`` into a :class:`RunManifest`.

    Raises :class:`SeedgraphError` when the run has no manifest.
    """
    paths.validate_slug(slug)
    path = _manifest_path(slug, run_id, root)
    if not path.exists():
        raise SeedgraphError(
            f"no manifest for run {run_id!r} of project {slug!r}"
        )
    doc = json.loads(path.read_text(encoding="utf-8"))
    return RunManifest(
        run_id=doc.get("run_id", run_id),
        created_at=doc.get("created_at", ""),
        sections=doc.get("sections", {}) or {},
    )


def list_runs(slug: str, *, root: Path | str | None = None) -> list[dict]:
    """Summarize every run of ``slug``, most-recent first.

    Each row is ``{run_id, created_at, sections, last_event, status}``: manifest
    metadata plus the run's terminal progress event and its derived status (review
    #7). Runs without a manifest yet still appear (with ``created_at=None``).
    """
    paths.validate_slug(slug)
    runs_dir = paths.project_dir(slug, root) / "runs"
    if not runs_dir.exists():
        return []
    dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    rows: list[dict] = []
    for run_dir in dirs:
        run_id = run_dir.name
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            doc = {}
        events = read_events(slug, run_id, root=root)
        rows.append(
            {
                "run_id": run_id,
                "created_at": doc.get("created_at"),
                "sections": doc.get("sections", {}) or {},
                "last_event": events[-1] if events else None,
                "status": _status_from_events(events),
            }
        )
    return rows
