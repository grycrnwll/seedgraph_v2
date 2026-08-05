"""Stdlib paragraph segmentation + in-markdown pagination → page mapping.

Phase 3 — Evidence Spans. Two deterministic, dependency-free concerns:

* ``paragraphs`` — blank-line paragraph segmentation returning half-open code-point
  offsets ``(start_char, end_char)`` into the markdown string. These offsets
  materialize the **paragraph-level** spans that exact-term search returns. The
  offsets must round-trip: ``markdown[s:e]`` reconstructs the paragraph verbatim.
* ``page_boundaries`` / ``page_for_range`` — best-effort page mapping parsed from
  **in-markdown pagination delimiters** (Marker ``paginate_output``) already present
  in the markdown this phase reads (decision 83). Returns ``[]`` when no delimiters
  are present, which yields ``NULL`` ``page_start``/``page_end`` on spans/sections.
  No separate ``pages.py`` module and no Marker block-JSON dependency (block JSON is
  not a settled cache deliverable).

Sentence granularity is deliberately NOT here (YAGNI; lands with claim/answer phases).
"""

from __future__ import annotations

import re

_WS = " \t\r\n\f\v"

#: Marker ``paginate_output`` page delimiter: a line of the form ``{N}-----`` (the
#: page index in braces followed by a run of dashes). The integer is Marker's
#: 0-based page index; we surface it 1-based. Tolerant of surrounding whitespace
#: and any dash-run length >= 3.
_PAGE_DELIM = re.compile(r"(?m)^[ \t]*\{(\d+)\}-{3,}[ \t]*$")


def paragraphs(markdown: str) -> list[tuple[int, int]]:
    """Segment ``markdown`` into paragraphs on blank lines; return half-open offsets.

    Returns ``[(start_char, end_char), ...]`` Python ``str`` code-point offsets such
    that each ``markdown[start:end]`` is a verbatim paragraph (round-trips the
    source). Handles leading/trailing newlines and runs of blank lines without
    emitting empty/whitespace-only paragraphs.
    """
    blocks: list[tuple[int, int]] = []
    pos = 0
    block_start: int | None = None
    for line in markdown.splitlines(keepends=True):
        if line.strip() == "":  # blank line closes the current block
            if block_start is not None:
                blocks.append((block_start, pos))
                block_start = None
        elif block_start is None:
            block_start = pos
        pos += len(line)
    if block_start is not None:
        blocks.append((block_start, pos))

    # Tighten each block to its non-whitespace extent so the stored quote is the
    # paragraph text itself (internal newlines of a multi-line paragraph survive).
    result: list[tuple[int, int]] = []
    for start, end in blocks:
        while end > start and markdown[end - 1] in _WS:
            end -= 1
        while start < end and markdown[start] in _WS:
            start += 1
        if end > start:
            result.append((start, end))
    return result


def page_boundaries(markdown: str) -> list[tuple[int, int]]:
    """Parse Marker pagination delimiters → ``[(char_offset, page_no), ...]``.

    Each tuple marks the code-point offset at which a given 1-based ``page_no``
    begins. Returns ``[]`` when the markdown contains no pagination delimiters (the
    delimiter-free case ⇒ ``NULL`` pages). Best-effort and degradation-safe: never
    raises on malformed/absent delimiters.
    """
    bounds: list[tuple[int, int]] = []
    for match in _PAGE_DELIM.finditer(markdown):
        try:
            page_no = int(match.group(1)) + 1  # Marker index is 0-based -> 1-based
        except (TypeError, ValueError):  # pragma: no cover - regex guarantees digits
            continue
        bounds.append((match.end(), page_no))
    return bounds


def _page_at(bounds: list[tuple[int, int]], offset: int) -> int | None:
    """Page number of the last boundary at or before ``offset`` (None if before all)."""
    page: int | None = None
    for boundary_offset, page_no in bounds:  # bounds are in document order
        if boundary_offset <= offset:
            page = page_no
        else:
            break
    return page


def page_for_range(
    bounds: list[tuple[int, int]], start: int, end: int
) -> tuple[int | None, int | None]:
    """Map a span's ``[start, end)`` offset range onto ``(page_start, page_end)``.

    ``bounds`` is the output of :func:`page_boundaries`. Returns ``(None, None)`` when
    ``bounds`` is empty (no pagination delimiters present), else the page numbers
    containing ``start`` and ``end`` respectively.
    """
    if not bounds:
        return (None, None)
    end_probe = end - 1 if end > start else start
    return (_page_at(bounds, start), _page_at(bounds, end_probe))
