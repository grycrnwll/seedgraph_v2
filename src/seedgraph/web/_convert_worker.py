"""Isolated single-conversion entrypoint -- one process per GPU.

Marker/Surya keep process-global model singletons + a shared CUDA context and are
NOT safe to run concurrently in one process, so :mod:`seedgraph.web.marker_queue`
spawns a fresh interpreter (its own CUDA context) per conversion. The parent pins the
GPU by setting ``CUDA_VISIBLE_DEVICES`` in this child's env BEFORE it imports
torch/marker; the child otherwise inherits the serve env (incl. ``$SEEDGRAPH_HOME``).

Usage::

    python -m seedgraph.web._convert_worker <slug> <work_id> <source_file_id> <file_hash>

Exits 0 on success (``OK`` on stdout); non-zero with the error on stderr on failure.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> int:
    try:
        slug, work_id, source_file_id, file_hash = argv[1], argv[2], argv[3], argv[4]
    except IndexError:
        print(
            "usage: _convert_worker <slug> <work_id> <source_file_id> <file_hash>",
            file=sys.stderr,
        )
        return 2
    from ..acquisition.service import convert_and_bridge
    from ..project import service as project_service

    try:
        h = project_service.open_project(slug)
        convert_and_bridge(
            h, work_id=work_id, source_file_id=source_file_id, file_hash=file_hash
        )
    except Exception as exc:  # noqa: BLE001 -- report to parent via exit code + stderr
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # Best-effort bibliographic backfill: replace the filename-derived title + attach
    # ids from the converted markdown. The import is INSIDE the guard so even an
    # import-time error (a broken extraction dep) can NEVER fail an already-bridged
    # conversion — logs to stderr, no non-zero exit; the markdown is already durable.
    try:
        from ..extraction.metadata import best_effort_backfill

        best_effort_backfill(h, work_id)
    except Exception as exc:  # noqa: BLE001 - backfill/import must not fail conversion
        print(
            f"metadata backfill skipped for {work_id}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
