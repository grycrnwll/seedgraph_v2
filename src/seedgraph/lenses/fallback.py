"""Deterministic (no-LLM) lens fallback — honest degradation (plan §8, decisions 38/58).

When no LLM profile is permitted for the work's `access_class`,
`deterministic_lens_pass` runs an FTS5/anchor candidate-span search over the
work's markdown using the lens `positive_anchors` (minus `negative_anchors`),
emitting `epistemic_type='deterministic'` saved-search spans as `lens_outputs`.

Honesty invariants (plan §12 "no-LLM fallback honesty"):
- emits saved-search spans, NOT typed/confidence-bearing claims;
- leaves `assertion_status` NULL (never fabricated for these non-claim spans);
- never invents a typed claim;
- not-found is recorded honestly;
- the reduced-capability gap is recorded in the run ``manifest.json``
  (via ``LensRunResult.degraded`` / ``capability_note``, persisted by the CLI).

The per-work writer is shared with the runner (``runner.deterministic_work``) so
there is exactly one no-LLM code path; this module is the public loop that opens
``cache.db`` read-only and iterates works.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .. import cache_access
from ..acquisition.bridge import resolve_work_markdown
from ..db.adapter import raw_conn
from ..extraction.runner import resolve_source_access_class
from .runner import LensRunResult, deterministic_work

if TYPE_CHECKING:
    from sqlmodel import Session

    from .schema import LensDefinition


def deterministic_lens_pass(
    session: "Session",
    lens: "LensDefinition",
    work_ids: list[str],
    *,
    cache_db: Path | str | None = None,
) -> LensRunResult:
    """Anchor/FTS5 candidate-span match over each work's markdown.

    For each work, search the resolved markdown for `positive_anchors` while
    excluding `negative_anchors`, writing matched spans as `lens_outputs` with
    `epistemic_type='deterministic'` and `assertion_status=NULL`; works with no
    anchor match record an honest not-found; works with no markdown are skipped
    (recorded in ``skipped_no_markdown``). Returns a `LensRunResult` whose counts
    feed the manifest's reduced-capability note. Implements decisions 38/58.

    ``cache_db`` is the seedgraph HOME root used to open ``cache.db`` read-only.
    """
    result = LensRunResult(lens_id=lens.lens_id)
    result.degraded = True
    result.capability_note = (
        "no permitted LLM profile; deterministic anchor/FTS saved-search fallback "
        "(epistemic_type=deterministic, assertion_status NULL, no typed claims)"
    )
    definition_hash = lens.definition_hash()
    conn = raw_conn(session)
    cache_conn = cache_access.open_cache_ro(cache_db)
    try:
        for work_id in work_ids:
            resolved = resolve_work_markdown(session, work_id=work_id)
            if resolved is None:
                result.skipped_no_markdown.append(work_id)
                continue
            markdown_id, markdown_hash = resolved
            access_class = resolve_source_access_class(cache_conn, markdown_id)
            deterministic_work(
                session,
                conn,
                cache_conn,
                cache_db,
                lens,
                work_id=work_id,
                markdown_id=markdown_id,
                markdown_hash=markdown_hash,
                definition_hash=definition_hash,
                access_class=access_class,
                result=result,
            )
    finally:
        cache_conn.close()
    return result
