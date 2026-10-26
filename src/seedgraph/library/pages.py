"""Quote → PDF page lookup and the selection anchor (v3 stage 3 design §3.1, §5).

``quote_locators(lib, paper_id)`` gives one :class:`QuoteLocator` per anchored claim of
the paper's current claims. The page comes from, in order:

- ``marker``: the span's stored ``page_start``/``page_end`` (filled by ``ensure_span``
  from Marker's page delimiters), or, when those are NULL, the delimiters of the span's
  own markdown at the span's stored offsets;
- ``reanchored``: the span's markdown has no delimiters, but the paper's current
  markdown (same PDF) does: the quote is relocated there with ``anchor.reanchor``
  (exact and unique only) and paged from it;
- ``none``: no page information. **A page is never guessed.** A page beyond the PDF's
  page count (pypdfium2, cached per file) is also ``none``, with a warning.

``anchor_selection(lib, paper_id, page, text)`` maps text selected in the PDF pane to
the unique verbatim markdown slice it came from, or refuses (§5).

Seam: the claims come from the ``current_claims`` view when the library has it (the
overlay-aware current interpretation, stage 3 §4.1); otherwise from ``note_current``'s
claims with their rank-0 anchor.
"""

from __future__ import annotations

import functools
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .. import anchor
from ..segment import page_boundaries, page_for_range
from .store import Library
from .textfold import fold

HOW_MARKER = "marker"
HOW_REANCHORED = "reanchored"
HOW_NONE = "none"


@dataclass(frozen=True)
class QuoteLocator:
    claim_id: str
    span_id: str
    page: int | None
    page_end: int | None
    how: str  # marker | reanchored | none
    needle: str  # the quote's folded form (textfold.fold)
    warning: str | None = None


@dataclass(frozen=True)
class PageAssignment:
    """Where a span's quote is in the PDF, and which markdown/file said so."""

    page: int | None
    page_end: int | None
    how: str
    quote: str
    markdown_id: str | None  # the markdown that supplied the page
    source_file_id: str | None  # the PDF those pages refer to
    warning: str | None = None


@dataclass(frozen=True)
class AnchorResult:
    """``status``: ok | not_found | ambiguous | no_markers | no_markdown | empty."""

    status: str
    quote: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    markdown_id: str | None = None
    matches: int = 0
    message: str = ""


# --- shared reads ------------------------------------------------------------------------


@functools.lru_cache(maxsize=256)
def _page_count_cached(path: str, mtime_ns: int, size: int) -> int | None:
    try:
        import pypdfium2 as pdfium
    except ImportError:  # pragma: no cover - pypdfium2 is an app dependency
        return None
    try:
        doc = pdfium.PdfDocument(path)
    except Exception:  # noqa: BLE001 - unreadable PDF: no count
        return None
    try:
        return len(doc)
    finally:
        doc.close()


def pdf_path(lib: Library, source_file_id: str) -> Path | None:
    row = lib.conn.execute("SELECT storage_uri FROM source_files WHERE source_file_id = ?",
                           (source_file_id,)).fetchone()
    if row is None or not row[0]:
        return None
    path = lib.root / row[0]
    return path if path.is_file() else None


def pdf_page_count(lib: Library, source_file_id: str) -> int | None:
    """The PDF's page count (pypdfium2), cached per file; ``None`` if unavailable."""
    path = pdf_path(lib, source_file_id)
    if path is None:
        return None
    st = path.stat()
    return _page_count_cached(str(path), st.st_mtime_ns, st.st_size)


def current_markdown(lib: Library, paper_id: str) -> tuple[str | None, str | None]:
    """The paper's primary ``(markdown_id, source_file_id)``."""
    row = lib.conn.execute(
        "SELECT markdown_id, source_file_id FROM paper_files WHERE paper_id = ? AND role = 'primary'",
        (paper_id,)).fetchone()
    return (row[0], row[1]) if row else (None, None)


class PageAssigner:
    """Assigns pages to spans (§3.1), caching markdown text and page bounds per call site."""

    def __init__(self, lib: Library):
        self.lib = lib
        self._text: dict[str, str | None] = {}
        self._bounds: dict[str, list[tuple[int, int]]] = {}
        self._md_file: dict[str, str | None] = {}

    def markdown_text(self, markdown_id: str | None) -> str | None:
        if markdown_id is None:
            return None
        if markdown_id not in self._text:
            row = self.lib.conn.execute(
                "SELECT storage_uri, source_file_id FROM markdown_documents WHERE markdown_id = ?",
                (markdown_id,)).fetchone()
            text = None
            if row is not None:
                path = self.lib.root / row[0]
                if path.is_file():
                    text = path.read_bytes().decode("utf-8")
            self._md_file[markdown_id] = row[1] if row else None
            self._text[markdown_id] = text
        return self._text[markdown_id]

    def bounds(self, markdown_id: str | None) -> list[tuple[int, int]]:
        if markdown_id is None:
            return []
        if markdown_id not in self._bounds:
            text = self.markdown_text(markdown_id)
            self._bounds[markdown_id] = page_boundaries(text) if text else []
        return self._bounds[markdown_id]

    def has_markers(self, markdown_id: str | None) -> bool:
        return bool(self.bounds(markdown_id))

    def assign(self, span_id: str, *, paper_id: str) -> PageAssignment:
        row = self.lib.conn.execute(
            "SELECT markdown_id, source_file_id, start_char, end_char, exact_quote, page_start, page_end "
            "FROM evidence_spans WHERE span_id = ?", (span_id,)).fetchone()
        if row is None:
            return PageAssignment(None, None, HOW_NONE, "", None, None, f"span {span_id} not found")
        md_id, sf_id, start, end, quote, p_start, p_end = row
        page = page_end = None
        how, src_md, src_file = HOW_NONE, None, None
        if p_start is not None:
            page, page_end, how, src_md, src_file = p_start, p_end or p_start, HOW_MARKER, md_id, sf_id
        elif self.has_markers(md_id):
            p, pe = page_for_range(self.bounds(md_id), start, end)
            if p is not None:
                page, page_end, how, src_md, src_file = p, pe or p, HOW_MARKER, md_id, sf_id
        else:
            cur_md, _cur_file = current_markdown(self.lib, paper_id)
            if cur_md and cur_md != md_id and self.has_markers(cur_md):
                located = anchor.reanchor(quote, self.markdown_text(cur_md) or "")
                if located is not None:
                    p, pe = page_for_range(self.bounds(cur_md), *located)
                    if p is not None:
                        page, page_end, how = p, pe or p, HOW_REANCHORED
                        src_md, src_file = cur_md, self._md_file.get(cur_md)
        if page is None:
            return PageAssignment(None, None, HOW_NONE, quote, None, None)
        count = pdf_page_count(self.lib, src_file) if src_file else None
        if count is not None and max(page, page_end or page) > count:
            return PageAssignment(
                None, None, HOW_NONE, quote, src_md, src_file,
                f"page {max(page, page_end or page)} is beyond the PDF's {count} pages: "
                "the markdown and the PDF disagree")
        return PageAssignment(page, page_end, how, quote, src_md, src_file)


# --- quote locators ----------------------------------------------------------------------


def _has_object(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (name,)).fetchone() is not None


def current_anchored_claims(lib: Library, paper_id: str) -> list[tuple[str, str]]:
    """``(claim_id, span_id)`` of the paper's current, anchored, non-lens claims."""
    conn = lib.conn
    if _has_object(conn, "current_claims"):
        rows = conn.execute(
            "SELECT claim_id, span_id FROM current_claims WHERE paper_id = ? AND span_id IS NOT NULL "
            "AND claim_type NOT LIKE 'lens:%' ORDER BY field_key, claim_id", (paper_id,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT c.claim_id, (SELECT cs.span_id FROM claim_spans cs WHERE cs.claim_id = c.claim_id "
            "                    ORDER BY cs.rank, cs.span_id LIMIT 1) AS span_id "
            "FROM extracted_claims c JOIN note_current n ON n.note_id = c.structured_note_id "
            "WHERE c.paper_id = ? AND c.claim_type NOT LIKE 'lens:%' ORDER BY c.field_key, c.claim_id",
            (paper_id,)).fetchall()
    return [(r[0], r[1]) for r in rows if r[1] is not None]


def quote_locators(lib: Library, paper_id: str) -> list[QuoteLocator]:
    """One locator per anchored claim of the paper's current claims (see the module doc)."""
    assigner = PageAssigner(lib)
    out = []
    for claim_id, span_id in current_anchored_claims(lib, paper_id):
        a = assigner.assign(span_id, paper_id=paper_id)
        out.append(QuoteLocator(claim_id, span_id, a.page, a.page_end, a.how, fold(a.quote).text, a.warning))
    return out


# --- selection anchor --------------------------------------------------------------------


def _slice_for_pages(bounds: list[tuple[int, int]], text_len: int, first: int, last: int) -> tuple[int, int]:
    """Raw ``[start, end)`` of pages ``first..last`` (1-based) from the delimiter bounds."""
    start, end = 0, text_len
    for offset, page_no in bounds:  # document order
        if page_no <= first:
            start = offset
        elif page_no > last:
            end = offset  # just after the next page's delimiter line, which folds to nothing
            break
    return start, end


_IN_WORD = "­​‌‍⁠﻿"
_HYPHEN_JOIN_AHEAD = re.compile(r"-[ \t]*\r?\n[ \t]*(?=[^\W\d_])")
_HYPHEN_JOIN_BEHIND = re.compile(r"(?<=[^\W\d_])-[ \t]*\r?\n[ \t]*$")


def _widen_to_words(md: str, start: int, end: int) -> tuple[int, int]:
    """Widen ``[start, end)`` to whole words, across line-end hyphenation and soft hyphens."""
    while True:
        if start > 0 and (md[start - 1].isalnum() or md[start - 1] in _IN_WORD):
            start -= 1
            continue
        m = _HYPHEN_JOIN_BEHIND.search(md, max(0, start - 16), start)
        if m and start < len(md) and md[start].isalpha():
            start = m.start()
            continue
        break
    while True:
        if end < len(md) and (md[end].isalnum() or md[end] in _IN_WORD):
            end += 1
            continue
        m = _HYPHEN_JOIN_AHEAD.match(md, end)
        if m and end > 0 and md[end - 1].isalpha():
            end = m.end()
            continue
        break
    return start, end


def anchor_selection(lib: Library, paper_id: str, page: int, text: str) -> AnchorResult:
    """Map text selected on PDF ``page`` to the unique verbatim markdown slice (§5).

    Folds the selection and the markdown of pages ``page-1 .. page+1``; requires exactly
    one match there, maps it back through the index map, widens it to whole words, and
    requires the slice to be unique in the whole markdown (``anchor.locate_quote``).
    """
    needle = fold(text or "").text
    if not needle:
        return AnchorResult("empty", message="the selection has no letters or digits")
    md_id, _ = current_markdown(lib, paper_id)
    assigner = PageAssigner(lib)
    md = assigner.markdown_text(md_id)
    if md is None:
        return AnchorResult("no_markdown", message="the paper has no converted text")
    bounds = assigner.bounds(md_id)
    if not bounds:
        return AnchorResult("no_markers", markdown_id=md_id,
                            message="no page information: select the passage in the text view")
    lo, hi = _slice_for_pages(bounds, len(md), page - 1, page + 1)
    folded = fold(md[lo:hi])
    hits = []
    pos = folded.text.find(needle)
    while pos != -1:
        hits.append(pos)
        pos = folded.text.find(needle, pos + 1)
    if not hits:
        return AnchorResult("not_found", markdown_id=md_id,
                            message="the selection was not found in the text of this page")
    if len(hits) > 1:
        return AnchorResult("ambiguous", markdown_id=md_id, matches=len(hits),
                            message=f"the selection appears {len(hits)} times on this page; "
                                    "select a longer passage")
    s, e = folded.raw_span(hits[0], hits[0] + len(needle))
    start, end = lo + s, lo + e
    start, end = _widen_to_words(md, start, end)
    quote = md[start:end]
    if anchor.locate_quote(md, quote) is None:
        return AnchorResult("ambiguous", markdown_id=md_id, matches=1,
                            message="the passage also appears elsewhere in the paper; "
                                    "select a longer passage")
    return AnchorResult("ok", quote=quote, start_char=start, end_char=end, markdown_id=md_id, matches=1)
