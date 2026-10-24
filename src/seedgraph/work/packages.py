"""Package text (design §1.3) and the text-quality gate's cache and hold-back (§5.3, §5.4).

Text:
- :func:`strip_references` - the markdown minus its ``section_kind = 'references'``
  ranges, each replaced by a visible ``[references omitted]`` marker. Port of v2
  ``extraction.runner._strip_references`` (fail-open: no ranges, or a cut that would
  leave no text, returns the markdown unchanged).
- :func:`plan_parts` - whole paper by default; over ``split_threshold_tokens`` it is cut
  into ``ceil(tokens / part_target_tokens)`` near-equal parts at section starts,
  falling back to paragraph boundaries inside an oversized section. ``start_char`` /
  ``end_char`` are offsets into the FULL stored markdown, so a part's text is
  ``strip_references`` of that slice and quotes always anchor against the full text.

Gate:
- :func:`text_quality` - the cached verdict for a markdown (``text_quality`` table,
  re-assessed when ``checker_version`` differs from ``textquality.CHECKER_VERSION``).
- :func:`hold_back_reason` - why a paper must not get packages: an open note-hiding
  ``pdf_title_miss`` / ``shared_pdf`` item, or a ``garbled`` / ``no_text`` verdict
  (raising ``text_garbled`` / ``text_missing``, ``hides_notes = 0``) unless the
  researcher resolved that item with ``accept_text``.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..library import attention, textquality
from ..llm.tokens import estimate_tokens
from ..notes.spans import read_markdown

if TYPE_CHECKING:
    from ..library.store import Library

REFERENCES_OMITTED_MARKER = "[references omitted]"
DEFAULT_SPLIT_THRESHOLD_TOKENS = 100_000
DEFAULT_PART_TARGET_TOKENS = 60_000

#: Attention kinds whose open, note-hiding item holds a paper back (design §5.4).
TITLE_HOLD_KINDS: tuple[str, ...] = ("pdf_title_miss", "shared_pdf")
GATE_KINDS = {"garbled": "text_garbled", "no_text": "text_missing"}
ACCEPT_TEXT = "accept_text"

Range = tuple[int, int]


# --- paper text --------------------------------------------------------------------------

@dataclass(frozen=True)
class PaperText:
    paper_id: str
    markdown_id: str
    markdown_hash: str
    text: str
    reference_ranges: tuple[Range, ...]
    section_starts: tuple[int, ...]


def primary_markdown(conn: sqlite3.Connection, paper_id: str) -> tuple[str, str] | None:
    """``(markdown_id, markdown_hash)`` of the paper's primary file, or ``None``."""
    row = conn.execute(
        "SELECT pf.markdown_id, m.markdown_hash FROM paper_files pf "
        "JOIN markdown_documents m ON m.markdown_id = pf.markdown_id "
        "WHERE pf.paper_id = ? AND pf.role = 'primary'", (paper_id,)).fetchone()
    return (row[0], row[1]) if row else None


def reference_ranges(conn: sqlite3.Connection, markdown_id: str) -> tuple[Range, ...]:
    return tuple((int(s), int(e)) for s, e in conn.execute(
        "SELECT start_char, end_char FROM document_sections WHERE markdown_id = ? "
        "AND section_kind = 'references' AND end_char > start_char ORDER BY start_char, ordinal",
        (markdown_id,)))


def section_starts(conn: sqlite3.Connection, markdown_id: str) -> tuple[int, ...]:
    return tuple(sorted({int(r[0]) for r in conn.execute(
        "SELECT start_char FROM document_sections WHERE markdown_id = ?", (markdown_id,))}))


def load_paper_text(conn: sqlite3.Connection, paper_id: str) -> PaperText | None:
    """The paper's current markdown with its reference ranges and section starts, or
    ``None`` when it has no converted primary file or the blob is unreadable."""
    md = primary_markdown(conn, paper_id)
    if md is None:
        return None
    text = read_markdown(conn, md[0])
    if text is None:
        return None
    return PaperText(paper_id, md[0], md[1], text, reference_ranges(conn, md[0]), section_starts(conn, md[0]))


def _merge(ranges: Iterable[Range]) -> list[Range]:
    out: list[Range] = []
    for s, e in sorted((int(s), int(e)) for s, e in ranges if int(e) > int(s)):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def strip_references(markdown: str, ranges: Iterable[Range] | None) -> str:
    """``markdown`` with each reference range replaced by the marker (fail-open)."""
    merged = _merge(ranges or ())
    if not merged:
        return markdown
    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        start = max(start, cursor)
        if start >= end:
            continue
        parts.append(markdown[cursor:start])
        parts.append(f"{REFERENCES_OMITTED_MARKER}\n")
        cursor = end
    parts.append(markdown[cursor:])
    trimmed = "".join(parts)
    if not trimmed.replace(REFERENCES_OMITTED_MARKER, "").strip():
        return markdown  # an all-references document: fail open, as v2
    return trimmed


def _clip(ranges: Sequence[Range], start: int, end: int) -> list[Range]:
    """``ranges`` clipped to ``[start, end)`` and shifted to slice offsets."""
    return [(max(s, start) - start, min(e, end) - start) for s, e in ranges if s < end and e > start]


def slice_text(markdown: str, ranges: Sequence[Range], start: int, end: int) -> str:
    """The package text of ``markdown[start:end]``: its reference ranges cut out."""
    return strip_references(markdown[start:end], _clip(ranges, start, end))


# --- the split ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PartSlice:
    part: int
    part_count: int
    start_char: int
    end_char: int
    text: str

    @property
    def est_tokens(self) -> int:
        return estimate_tokens(self.text)


_PARAGRAPH_RE = re.compile(r"\n[ \t]*\n+")


def _kept_before(pos: int, ranges: Sequence[Range]) -> int:
    """Characters before ``pos`` that are not in a reference range."""
    removed = sum(max(0, min(e, pos) - s) for s, e in ranges if s < pos)
    return pos - removed


def _in_ranges(pos: int, ranges: Sequence[Range]) -> bool:
    return any(s < pos < e for s, e in ranges)


def plan_parts(markdown: str, ref_ranges: Sequence[Range], starts: Sequence[int], *,
               split_threshold_tokens: int = DEFAULT_SPLIT_THRESHOLD_TOKENS,
               part_target_tokens: int = DEFAULT_PART_TARGET_TOKENS) -> list[PartSlice]:
    """The paper's parts (one for a whole paper). See the module doc."""
    merged = _merge(ref_ranges)
    whole = strip_references(markdown, merged)
    tokens = estimate_tokens(whole)
    n = len(markdown)
    if tokens <= split_threshold_tokens or n == 0:
        return [PartSlice(1, 1, 0, n, whole)]
    count = max(2, math.ceil(tokens / max(1, part_target_tokens)))
    total_kept = _kept_before(n, merged)
    part_chars = total_kept / count
    sections = sorted({s for s in starts if 0 < s < n and not _in_ranges(s, merged)})
    paragraphs = sorted({m.end() for m in _PARAGRAPH_RE.finditer(markdown)
                         if 0 < m.end() < n and not _in_ranges(m.end(), merged)})
    cuts: list[int] = []
    for k in range(1, count):
        goal = k * part_chars
        prev = cuts[-1] if cuts else 0
        best: int | None = None
        for pool in (sections, paragraphs):
            cands = [c for c in pool if c > prev]
            if not cands:
                continue
            c = min(cands, key=lambda p: (abs(_kept_before(p, merged) - goal), p))
            # A section start more than half a part from the goal means the section is
            # oversized: fall back to a paragraph boundary inside it.
            if pool is sections and abs(_kept_before(c, merged) - goal) > part_chars / 2:
                best = best if best is not None else c
                continue
            best = c
            break
        if best is not None and best not in cuts:
            cuts.append(best)
    bounds = [0, *sorted(set(cuts)), n]
    pieces = [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
    texts = [(a, b, slice_text(markdown, merged, a, b)) for a, b in pieces]
    texts = [t for t in texts if t[2].replace(REFERENCES_OMITTED_MARKER, "").strip()] or texts
    total = len(texts)
    return [PartSlice(i, total, a, b, t) for i, (a, b, t) in enumerate(texts, start=1)]


# --- the gate ----------------------------------------------------------------------------

@dataclass(frozen=True)
class GateVerdict:
    status: str  # 'pass' | 'garbled' | 'no_text'
    summary: str
    cached: bool


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def text_quality(lib: Library, markdown_id: str, text: str, ranges: Sequence[Range], *,
                 force: bool = False) -> GateVerdict:
    """The cached verdict for ``markdown_id``, assessing (and caching) it when there is
    no result for the current ``checker_version``."""
    row = lib.conn.execute("SELECT status, score_json, checker_version FROM text_quality WHERE markdown_id = ?",
                           (markdown_id,)).fetchone()
    if row is not None and not force and row[2] == textquality.CHECKER_VERSION:
        summary = (json.loads(row[1]) or {}).get("summary", "") if row[1] else ""
        return GateVerdict(row[0], summary, True)
    quality = textquality.assess_text(text, reference_ranges=ranges)
    with lib.write() as conn:
        conn.execute(
            "INSERT INTO text_quality (markdown_id, status, score_json, checker_version, assessed_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(markdown_id) DO UPDATE SET status = excluded.status, "
            "score_json = excluded.score_json, checker_version = excluded.checker_version, "
            "assessed_at = excluded.assessed_at",
            (markdown_id, quality.status, json.dumps(quality.as_dict(), ensure_ascii=False, sort_keys=True),
             quality.checker_version, _now()))
    return GateVerdict(quality.status, quality.summary, False)


@dataclass(frozen=True)
class HoldBack:
    reason: str
    attention_item_id: str | None


def title_hold(conn: sqlite3.Connection, paper_id: str) -> HoldBack | None:
    marks = ", ".join("?" for _ in TITLE_HOLD_KINDS)
    row = conn.execute(
        f"SELECT item_id, kind, detail FROM attention_items WHERE paper_id = ? AND status = 'open' "
        f"AND hides_notes = 1 AND kind IN ({marks}) ORDER BY kind, item_id LIMIT 1",
        (paper_id, *TITLE_HOLD_KINDS)).fetchone()
    if row is None:
        return None
    return HoldBack(f"{row[1]}: {row[2] or 'this PDF may not be this paper'}; resolve the attention item first",
                    row[0])


def gate_hold(lib: Library, paper: PaperText) -> HoldBack | None:
    """Hold back a garbled or text-less paper (raising its attention item) unless the
    researcher accepted this markdown's text."""
    verdict = text_quality(lib, paper.markdown_id, paper.text, paper.reference_ranges)
    if verdict.status == "pass":
        return None
    kind = GATE_KINDS[verdict.status]
    subject = {"paper_id": paper.paper_id, "markdown_id": paper.markdown_id}
    existing = attention.get_item(lib, attention.item_id_for(kind, subject))
    if existing is not None and existing.status == "resolved" and existing.resolution == ACCEPT_TEXT:
        return None
    fix = (f"re-convert with forced OCR (seedgraph library reconvert --ocr {paper.paper_id}), "
           f"or resolve the attention item with {ACCEPT_TEXT} to note it anyway")
    item = existing if existing is not None and existing.status == "resolved" else attention.raise_item(
        lib, kind, subject, paper_id=paper.paper_id, detail=f"{verdict.summary}; {fix}", hides_notes=False)
    return HoldBack(f"{kind}: {verdict.summary}; {fix}", item.item_id)


def hold_back_reason(lib: Library, paper: PaperText, *, title_check: bool = True) -> HoldBack | None:
    if title_check:
        held = title_hold(lib.conn, paper.paper_id)
        if held is not None:
            return held
    return gate_hold(lib, paper)
