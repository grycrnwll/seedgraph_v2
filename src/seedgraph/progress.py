"""Shared progress abstraction for long per-item pipelines.

One tiny handle, :class:`Progress`, drives both surfaces the codebase already has:

* the **CLI heartbeat** — ``N/M unit — detail  k=v`` lines to stdout, matching the
  house ``typer.echo`` tab/``k=v`` style; and
* the **web run stream** — a ``progress`` event appended to
  ``runs/{run_id}/events.jsonl`` in the *exact* shape
  :func:`seedgraph.run.append_event` already writes, so the UI needs no new reader.
  Inside a background job the wiring is ``Progress(len(work_ids), "works", emit=emit)``
  with no other change.

The only difference between the two is whether an ``emit`` callable is passed:
``emit=None`` (the CLI default) makes a run **stdout-only**, so a CLI-driven run never
writes a non-terminal ``progress`` event that would leave ``_status_from_events`` stuck
on ``running``.

:func:`make_emitter` builds a jobs-style ``emit`` closure over
:func:`run.append_event` for the rare CLI command that *wants* its progress mirrored
into ``events.jsonl`` (opt-in). :func:`conversion_summary` is a **pure** read-model that
folds ``corpus_rows`` + ``marker_queue.status`` into the corpus-page conversion bar.

Zero new dependencies; no new DB table; no new event reader. Imports only ``sys``,
``typing`` and (lazily) :func:`seedgraph.run.append_event`.
"""

from __future__ import annotations

import sys
from typing import Callable, Optional, TextIO


class Progress:
    """A running ``done``/``total`` counter that echoes and/or emits each step.

    ``Progress(total, unit)`` prints ``{done}/{total} {unit} — {detail}  k=v`` to
    ``stream`` (default stdout) on every :meth:`step`. When an ``emit`` callable is
    supplied (the ``web/jobs.py`` closure), each step ALSO appends an event of type
    ``event`` (default ``"progress"``) whose ``data`` carries ``done``/``total`` plus
    the per-step ``**counts`` — the identical shape :func:`run.append_event` writes, so
    the UI reads it through the existing ``/events`` endpoint with no new reader.
    """

    def __init__(
        self,
        total: int,
        unit: str = "items",
        *,
        emit: Optional[Callable[..., int]] = None,
        event: str = "progress",
        echo: bool = True,
        stream: Optional[TextIO] = None,
    ) -> None:
        self.total = total
        self.unit = unit
        self.done = 0
        self.emit = emit
        self.event = event
        self.echo = echo
        self._stream = stream

    def _format(self, detail: str, counts: dict) -> str:
        line = f"{self.done}/{self.total} {self.unit}"
        if detail:
            line += f" — {detail}"
        if counts:
            line += "  " + " ".join(f"{k}={v}" for k, v in counts.items())
        return line

    def step(self, detail: str = "", *, level: str = "info", **counts) -> int:
        """Advance one and surface the heartbeat.

        Increments ``done``; if ``echo`` prints ``{done}/{total} {unit} — {detail}
        k=v`` to ``stream``; if an ``emit`` was supplied, appends a ``progress`` event
        whose ``message`` equals the echoed line and whose ``data`` is
        ``{done, total, **counts}``. Returns the event ``seq`` when emitting, else
        ``-1`` (``emit=None`` never touches the filesystem)."""
        self.done += 1
        line = self._format(detail, counts)
        if self.echo:
            stream = self._stream if self._stream is not None else sys.stdout
            print(line, file=stream)
        if self.emit is not None:
            return self.emit(
                self.event,
                line,
                level=level,
                done=self.done,
                total=self.total,
                **counts,
            )
        return -1

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exc) -> bool:  # never swallows an exception
        return False


def make_emitter(
    slug: str,
    run_id: str,
    *,
    phase: str,
    root=None,
) -> Callable[..., int]:
    """A jobs-style ``emit`` closure over :func:`run.append_event` (identical shape).

    ``emit(event, message="", *, level="info", **data) -> seq``. ``emit.run_id`` is
    stamped so a run-scoped artifact targets the SAME run the events stream to. For the
    rare CLI command that opts into mirroring its progress into ``events.jsonl``."""
    from .run import append_event

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

    emit.run_id = run_id  # type: ignore[attr-defined]
    return emit


def conversion_summary(rows: list[dict], queue_status: dict[str, dict]) -> dict:
    """Pure corpus-bar read-model folded from the two structures ``ui_corpus`` holds.

    ``rows`` = :func:`acquisition.service.corpus_rows`; ``queue_status`` =
    :func:`web.marker_queue.status`. Returns
    ``{converted, pending, queued, converting, failed, total, done_pct}`` where
    ``converted = Σ rows[has_markdown]``; ``queued``/``converting``/``failed`` are
    counted from the queue map; ``pending = queued + converting`` and
    ``total = converted + pending`` (the "share of the acquirable corpus that has
    markdown"). No ``web``/``acquisition`` imports — keeps this module leaf-level."""
    converted = sum(1 for r in rows if r.get("has_markdown"))
    queued = converting = failed = 0
    for st in queue_status.values():
        state = st.get("state")
        if state == "queued":
            queued += 1
        elif state == "converting":
            converting += 1
        elif state == "failed":
            failed += 1
    pending = queued + converting
    total = converted + pending
    done_pct = round(100 * converted / total) if total else 0
    return {
        "converted": converted,
        "pending": pending,
        "queued": queued,
        "converting": converting,
        "failed": failed,
        "total": total,
        "done_pct": done_pct,
    }


__all__ = ["Progress", "make_emitter", "conversion_summary"]
