"""Span creation/retrieval — the single public ``ensure_span`` API + offset writer.

Phase 3 — Evidence Spans. ``ensure_span`` is the canonical, cross-phase
span-creation API (decision D7): phase_4 calls it exactly the way the CLI does. All
writes operate on the **raw** ``sqlite3.Connection`` obtained via
``db.adapter.raw_conn(project_session)`` so the span row + ``span_fts`` row commit
atomically inside the caller's ORM transaction (e.g. together with phase_4's claim
INSERT and its ``claim_spans`` junction row) — and roll back together. cache.db is
opened read-only (``cache_conn``) only to resolve lineage / access_class / markdown
bytes.

Span id discipline (idempotency): auto paragraph spans get a **deterministic**
``ids.auto_span_id(markdown_hash, start, end, span_kind)`` so re-indexing yields the
same ids (no duplicate rows / FTS rows); manual/claim spans get an opaque
``ids.new_id('span')``. Offsets are Python ``str`` code-point indices.

access_class is stamped fail-closed (decisions 30/60/76): the source-resolved class
reconciled with any caller-supplied floor via ``AccessClass.most_restrictive(...)`` —
a too-permissive caller value can never widen below the source class; unresolved
lineage yields the most-restrictive default (``user_supplied_private``).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from seedgraph import anchor, cache_access
from seedgraph.ids import new_id
from seedgraph.sections.parser import section_for_offset
from seedgraph.sections.store import load_sections
from seedgraph.segment import page_boundaries, page_for_range
from seedgraph.vocab import AccessClass

# Persisted evidence_spans columns, in table order (the Span dataclass fields up to
# created_at; resolved_text / heading_path are derived, not stored).
_SPAN_COLUMNS = (
    "span_id",
    "markdown_id",
    "markdown_hash",
    "source_file_id",
    "source_file_hash",
    "work_id",
    "section_id",
    "start_char",
    "end_char",
    "exact_quote",
    "quote_hash",
    "norm_version",
    "span_kind",
    "page_start",
    "page_end",
    "access_class",
    "anchor_status",
    "created_at",
)
_SPAN_SELECT = ", ".join(_SPAN_COLUMNS)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Span:
    """A retrieved span — its ``evidence_spans`` row plus optionally resolved context.

    ``resolved_text`` is the live ``markdown[start_char:end_char]`` (equal to
    ``exact_quote`` when anchored); ``heading_path`` is the resolving section's
    breadcrumb; ``page_start`` / ``page_end`` are best-effort.
    """

    span_id: str
    markdown_id: str
    markdown_hash: str
    source_file_id: str
    source_file_hash: str
    work_id: str
    section_id: str | None
    start_char: int
    end_char: int
    exact_quote: str
    quote_hash: str
    norm_version: str
    span_kind: str
    page_start: int | None
    page_end: int | None
    access_class: str
    anchor_status: str
    created_at: str
    resolved_text: str | None = None
    heading_path: str | None = None


def ensure_span(
    project_conn: sqlite3.Connection,
    cache_conn: sqlite3.Connection,
    cache_root: Path,
    *,
    markdown_id: str,
    work_id: str,
    exact_quote: str,
    access_class: str | None = None,
    span_kind: str = "manual",
) -> str | None:
    """Locate-or-create a span for ``exact_quote`` in ``markdown_id``; return its id.

    The single public span-creation API (decision D7); phase_4 consumes it the same
    way. Behavior:

    1. Resolve ``markdown_id`` via ``cache_access.read_markdown`` (read-only cache);
       return ``None`` if the markdown is unresolvable.
    2. Locate the UNIQUE exact (NFC ``str.find``) occurrence of ``exact_quote``;
       return ``None`` on no match or ambiguous (>=2) matches — never guesses.
    3. Dedup on ``(markdown_id, quote_hash)`` via ``ix_spans_quotehash``: if a span
       already exists, reconcile its ``access_class`` to the most-restrictive of the
       existing/source/caller values and return the existing ``span_id``.
    4. Otherwise stamp ``access_class =
       AccessClass.most_restrictive(source_resolved, access_class)`` (fail-closed),
       resolve ``section_id``/pages, and ``INSERT`` into ``evidence_spans`` + ``span_fts``
       within ``project_conn``'s transaction (atomic with the caller's ORM writes).

    ``project_conn`` MUST be the raw connection from ``db.adapter.raw_conn(session)`` —
    never reached through private session internals.
    """
    md = cache_access.read_markdown(cache_conn, cache_root, markdown_id)
    if md is None:
        return None
    located = anchor.locate_quote(md.text, exact_quote)
    if located is None:  # no match or ambiguous (>=2) — never guess
        return None
    start, end = located

    qh = anchor.quote_hash(exact_quote)
    existing = project_conn.execute(
        "SELECT span_id, access_class FROM evidence_spans "
        "WHERE markdown_id = ? AND quote_hash = ?",
        (markdown_id, qh),
    ).fetchone()
    if existing is not None:
        existing_span_id, existing_access = existing[0], existing[1]
        # Reconcile to the most-restrictive of existing / source / caller (fail-closed;
        # a re-call with a wider caller floor can never widen an already-stored class).
        reconciled = AccessClass.most_restrictive(
            existing_access, md.access_class, access_class
        )
        if str(reconciled) != existing_access:
            project_conn.execute(
                "UPDATE evidence_spans SET access_class = ? WHERE span_id = ?",
                (str(reconciled), existing_span_id),
            )
        return existing_span_id

    return _write_span(
        project_conn,
        cache_conn,
        cache_root,
        markdown_id=markdown_id,
        work_id=work_id,
        start=start,
        end=end,
        exact_quote=exact_quote,
        span_kind=span_kind,
        access_class=access_class,
    )


def _write_span(
    project_conn: sqlite3.Connection,
    cache_conn: sqlite3.Connection,
    cache_root: Path,
    *,
    markdown_id: str,
    work_id: str,
    start: int,
    end: int,
    exact_quote: str,
    span_kind: str = "manual",
    span_id: str | None = None,
    access_class: str | None = None,
) -> str:
    """Internal offset writer — write one span at explicit ``[start, end)``; RAISES on bad slice.

    Resolves ``markdown_hash`` / ``source_file_*`` / ``access_class`` / ``section_id`` /
    ``page_*`` from cache, stamps ``quote_hash`` (= ``sha256(NFC(exact_quote))``) and
    ``norm_version``, and enforces the write-time invariant
    ``markdown[start:end] == exact_quote`` — raising ``ValueError`` when the slice does
    not match (unlike ``ensure_span``, which returns ``None``). ``span_id`` is the
    caller-provided deterministic id for auto paragraph spans, else a fresh
    ``ids.new_id('span')``. ``access_class`` (optional) is the caller floor reconciled
    most-restrictive with the source class. Inserts ``evidence_spans`` + ``span_fts``
    in ``project_conn``'s transaction. Returns the written ``span_id``.
    """
    md = cache_access.read_markdown(cache_conn, cache_root, markdown_id)
    if md is None:
        raise ValueError(f"_write_span: markdown {markdown_id!r} is unresolvable")

    # Write-time invariant — RAISE on a bad slice (the offset writer never guesses).
    actual = md.text[start:end]
    if actual != exact_quote:
        raise ValueError(
            "span invariant violated: markdown[%d:%d] != exact_quote "
            "(got %r, expected %r)" % (start, end, actual, exact_quote)
        )

    quote_hash = anchor.quote_hash(exact_quote)
    # Fail-closed access stamp: most-restrictive(source, caller-floor). A too-permissive
    # caller value can NEVER widen below the source class.
    stamped_access = str(AccessClass.most_restrictive(md.access_class, access_class))

    sections = load_sections(project_conn, markdown_id)
    section_id = section_for_offset(sections, start)
    page_start, page_end = page_for_range(page_boundaries(md.text), start, end)

    resolved_span_id = span_id or new_id("span")
    created_at = _now()
    project_conn.execute(
        "INSERT INTO evidence_spans "
        "(span_id, markdown_id, markdown_hash, source_file_id, source_file_hash, "
        " work_id, section_id, start_char, end_char, exact_quote, quote_hash, "
        " norm_version, span_kind, page_start, page_end, access_class, anchor_status, "
        " created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            resolved_span_id,
            markdown_id,
            md.markdown_hash,
            md.source_file_id,
            md.source_file_hash,
            work_id,
            section_id,
            start,
            end,
            exact_quote,
            quote_hash,
            anchor.NORM_VERSION,
            span_kind,
            page_start,
            page_end,
            stamped_access,
            "anchored",
            created_at,
        ),
    )
    project_conn.execute(
        "INSERT INTO span_fts(quote_text, span_id, markdown_id, work_id, section_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (exact_quote, resolved_span_id, markdown_id, work_id, section_id),
    )
    return resolved_span_id


def _row_to_span(row, *, resolved_text=None, heading_path=None) -> Span:
    """Build a :class:`Span` from a ``_SPAN_SELECT`` row tuple."""
    values = dict(zip(_SPAN_COLUMNS, row))
    return Span(resolved_text=resolved_text, heading_path=heading_path, **values)


def get_span(
    conn: sqlite3.Connection,
    cache_conn: sqlite3.Connection,
    cache_root: Path,
    span_id: str,
    *,
    resolve_text: bool = True,
) -> Span:
    """Load a :class:`Span` by id, optionally resolving live text + section breadcrumb.

    When ``resolve_text`` is True, reads the current markdown via ``cache_access`` and
    fills ``resolved_text`` (= ``markdown[start:end]``) and the section ``heading_path``
    — the ``spans get`` surface (exact source text, section, work, page, status). When
    the pinned markdown is gone (cache GC), the row is still returned with
    ``resolved_text=None`` (shadow spans remain retrievable).
    """
    row = conn.execute(
        f"SELECT {_SPAN_SELECT} FROM evidence_spans WHERE span_id = ?", (span_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"no span {span_id!r}")
    span = _row_to_span(tuple(row))
    if not resolve_text:
        return span

    md = cache_access.read_markdown(cache_conn, cache_root, span.markdown_id)
    resolved_text = None
    heading_path = None
    if md is not None:
        resolved_text = md.text[span.start_char:span.end_char]
    if span.section_id is not None:
        for section in load_sections(conn, span.markdown_id):
            if section.section_id == span.section_id:
                heading_path = section.heading_path
                break
    return _row_to_span(tuple(row), resolved_text=resolved_text, heading_path=heading_path)


def verify_span(
    conn: sqlite3.Connection,
    cache_conn: sqlite3.Connection,
    cache_root: Path,
    span_id: str,
) -> str:
    """Verify one span's invariant + staleness; return/persist its ``anchor_status``.

    Re-checks ``markdown[start:end] == exact_quote`` and ``quote_hash`` against the
    current markdown; uses denormalized lineage (``source_file_id`` +
    ``source_file_hash``) to detect that a newer markdown supersedes this content even
    after the old ``markdown_id`` row is GC'd, marking such spans
    ``anchor_status='stale'`` (shadow-don't-delete). Returns the resulting status.
    """
    row = conn.execute(
        f"SELECT {_SPAN_SELECT} FROM evidence_spans WHERE span_id = ?", (span_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"no span {span_id!r}")
    span = _row_to_span(tuple(row))
    if span.anchor_status == "orphaned":
        return "orphaned"  # already routed to review; verify does not resurrect it

    md = cache_access.read_markdown(cache_conn, cache_root, span.markdown_id)
    invariant_ok = (
        md is not None
        and md.text[span.start_char:span.end_char] == span.exact_quote
        and anchor.quote_hash(span.exact_quote) == span.quote_hash
    )
    current = cache_access.current_markdown_for_source(
        cache_conn, span.source_file_id, span.source_file_hash
    )
    superseded = current is not None and current[1] != span.markdown_hash

    if invariant_ok and not superseded:
        new_status = "anchored"
    else:
        # Either the pinned markdown changed/vanished, or a newer markdown exists for
        # this source lineage — both are staleness (shadow-don't-delete; never delete).
        new_status = "stale"

    if new_status != span.anchor_status:
        conn.execute(
            "UPDATE evidence_spans SET anchor_status = ? WHERE span_id = ?",
            (new_status, span_id),
        )
    return new_status


def reanchor_spans(
    conn: sqlite3.Connection,
    cache_conn: sqlite3.Connection,
    cache_root: Path,
    *,
    work_id: str,
) -> int:
    """Relocate stale spans of ``work_id`` into the current markdown; return count moved.

    For each stale span, exact-substring relocate the ``exact_quote`` into the current
    markdown (``anchor.reanchor``); on a unique match a NEW anchored span is written in
    the current markdown (its ``section_id`` re-resolved, ``span_fts`` updated) while
    the stale original is RETAINED as a shadow (shadow-don't-delete — it stays
    retrievable). Quotes that cannot be uniquely relocated become
    ``anchor_status='orphaned'`` and route to ``review_queue``. Returns the number of
    spans relocated.

    Idempotent: the shadow keeps ``anchor_status='stale'`` (no terminal "relocated"
    status exists in the closed ``AnchorStatus`` vocabulary), so a second pass re-selects
    it; a ``(markdown_id, quote_hash)`` dedup on the current markdown then skips it
    without writing a duplicate span/``span_fts`` row, and ``moved`` only counts spans
    actually relocated this pass — so re-running yields the SAME rows and ``moved == 0``.
    """
    from seedgraph.ids import auto_span_id

    rows = conn.execute(
        f"SELECT {_SPAN_SELECT} FROM evidence_spans "
        "WHERE work_id = ? AND anchor_status = 'stale'",
        (work_id,),
    ).fetchall()

    moved = 0
    for raw in rows:
        span = _row_to_span(tuple(raw))
        current = cache_access.current_markdown_for_source(
            cache_conn, span.source_file_id, span.source_file_hash
        )
        if current is None:
            continue  # no newer markdown to relocate into
        new_markdown_id, new_markdown_hash = current
        new_md = cache_access.read_markdown(cache_conn, cache_root, new_markdown_id)
        if new_md is None:
            continue
        located = anchor.reanchor(span.exact_quote, new_md.text)
        if located is None:
            # Unique-miss -> orphan + review_queue (never guess among duplicates).
            conn.execute(
                "UPDATE evidence_spans SET anchor_status = 'orphaned' WHERE span_id = ?",
                (span.span_id,),
            )
            _enqueue_orphan(conn, span)
            continue

        start, end = located
        if span.span_kind == "paragraph":
            new_span_id = auto_span_id(new_markdown_hash, start, end, "paragraph")
        else:
            new_span_id = None
        # Idempotent reanchor for BOTH manual and paragraph spans: skip if this quote
        # is already relocated into the current markdown. Dedup on
        # (new_markdown_id, quote_hash) — the SAME span identity ensure_span uses (a
        # uniquely relocated quote occurs exactly once in the new markdown) — because a
        # relocated span keeps the stale original as a shadow (its anchor_status stays
        # 'stale', shadow-don't-delete), so a second pass re-selects that shadow. Manual
        # spans get a fresh new_id() with no deterministic-id guard, so this check is
        # what stops duplicate evidence_spans / span_fts rows from accumulating. A skip
        # is NOT counted in `moved` (nothing was relocated on this pass).
        if conn.execute(
            "SELECT 1 FROM evidence_spans WHERE markdown_id = ? AND quote_hash = ?",
            (new_markdown_id, span.quote_hash),
        ).fetchone() is not None:
            continue
        _write_span(
            conn,
            cache_conn,
            cache_root,
            markdown_id=new_markdown_id,
            work_id=work_id,
            start=start,
            end=end,
            exact_quote=span.exact_quote,
            span_kind=span.span_kind,
            span_id=new_span_id,
            access_class=span.access_class,
        )
        moved += 1
    return moved


def _enqueue_orphan(conn: sqlite3.Connection, span: Span) -> None:
    """Raw-insert a ``review_queue`` row for an orphaned span (in the caller's txn).

    A direct insert (not ``project.review.enqueue``) because the orphan payload is a
    new ``item_type`` outside the phase_0 discriminated-union validator, and because
    reanchor operates on the raw ``sqlite3.Connection`` (not an ORM session) so the
    review row commits/rolls back atomically with the orphan status update.
    """
    payload = json.dumps(
        {
            "kind": "span_orphan",
            "span_id": span.span_id,
            "markdown_id": span.markdown_id,
            "exact_quote": span.exact_quote,
        }
    )
    conn.execute(
        "INSERT INTO review_queue "
        "(item_id, item_type, target_type, target_id, payload, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, 'open', ?)",
        (
            new_id("rq"),
            "span_reanchor_orphan",
            "evidence_span",
            span.span_id,
            payload,
            _now(),
        ),
    )
