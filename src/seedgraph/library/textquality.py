"""Text-quality gate detector (design §5.2). Pure: markdown in, verdict out.

Scores the reference-stripped text (what would be sent to a worker) for lost-ligature
mojibake and missing text layers. Converter garbage (U+FFFD, ``(cid:N)``, private-use
code points) is counted and reported but never decides the verdict. No dictionary lookup: cheap, deterministic and
explainable ("40 ``Ö`` inside words, 13 ``§`` inside words; examples: ...").
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

from seedgraph.library import mojibake

#: Bump whenever the detector or its constants change; cached results with another
#: version are re-assessed (design §5.3).
CHECKER_VERSION = "textquality-2"

DOC_RATE_MIN = 1.0  # strong hits per 1 000 words
DOC_HITS_MIN = 10
PAGE_MIN_WORDS = 50
PAGE_HITS_MIN = 2
PAGE_RATE_MIN = 2.0
BAD_PAGES_MIN = 3
BAD_PAGES_SHARE = 0.20
NO_TEXT_MEDIAN_WORDS = 30
MAX_EXAMPLES = 8
EXAMPLE_CONTEXT = 20

REFERENCES_OMITTED_MARKER = "[references omitted]"

_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_PAGE_MARKER_RE = re.compile(r"^\{\d+\}-{10,}$", re.MULTILINE)


@dataclass(frozen=True)
class TextQuality:
    status: str  # 'pass' | 'garbled' | 'no_text'
    doc_rate: float
    strong_hits: int
    words: int
    pages: int
    bad_pages: int
    median_page_words: float
    hits_by_signal: dict[str, int] = field(default_factory=dict)
    corroborating_hits: int = 0
    examples: tuple[str, ...] = ()
    summary: str = ""
    checker_version: str = CHECKER_VERSION

    def as_dict(self) -> dict:
        d = asdict(self)
        d["examples"] = list(self.examples)
        return d


def strip_reference_ranges(markdown: str, ranges: Iterable[tuple[int, int]] | None) -> str:
    """``markdown`` with each ``(start, end)`` range replaced by a visible marker.

    Fail-open like v2's ``_strip_references``: no ranges, or a cut that would leave no
    text, returns the markdown unchanged.
    """
    spans = sorted((int(s), int(e)) for s, e in (ranges or ()) if int(e) > int(s))
    if not spans:
        return markdown
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        start = max(start, cursor)
        if start >= end:
            continue
        parts.append(markdown[cursor:start])
        parts.append(f"{REFERENCES_OMITTED_MARKER}\n")
        cursor = end
    parts.append(markdown[cursor:])
    trimmed = "".join(parts)
    if not trimmed.replace(REFERENCES_OMITTED_MARKER, "").strip():
        return markdown
    return trimmed


def split_pages(text: str) -> list[str]:
    """Pages by Marker page markers; text before the first marker is dropped when blank.
    Without markers the whole text is one page."""
    pieces = _PAGE_MARKER_RE.split(text)
    if len(pieces) == 1:
        return pieces
    if not pieces[0].strip():
        pieces = pieces[1:]
    return pieces


def _example(text: str, hit: mojibake.Hit) -> str:
    a = max(0, hit.start - EXAMPLE_CONTEXT)
    b = min(len(text), hit.end + EXAMPLE_CONTEXT)
    return " ".join(text[a:b].split())


_GARBAGE_NAMES = {
    mojibake.SIGNAL_REPLACEMENT: "replacement characters",
    mojibake.SIGNAL_CID: "(cid:N) codes",
    mojibake.SIGNAL_PUA: "private-use characters",
}


def _on_table_line(text: str, pos: int) -> bool:
    start = text.rfind("\n", 0, pos) + 1
    return text[start:pos].lstrip().startswith("|")


def _summary(status: str, by_signal: Counter, in_tables: Counter, examples: list[str]) -> str:
    if not by_signal:
        return f"{status}: no mojibake signals"
    parts = []
    for sig, n in by_signal.most_common():
        if sig in _GARBAGE_NAMES:
            part = f"{n} {_GARBAGE_NAMES[sig]}"
            if in_tables[sig] * 2 > n:
                part += ", mostly in tables"
            elif in_tables[sig]:
                part += f" ({in_tables[sig]} in tables)"
            parts.append(part)
        else:
            parts.append(f"{n} {sig!r}")
    ex = "; ".join(examples[:3])
    return f"{status}: " + ", ".join(parts) + (f"; examples: {ex}" if ex else "")


def assess_text(
    markdown: str, *, reference_ranges: Iterable[tuple[int, int]] | None = None
) -> TextQuality:
    """Assess ``markdown`` (references cut out by ``reference_ranges``, if given)."""
    text = strip_reference_ranges(markdown, reference_ranges)
    pages = split_pages(text)
    page_words: list[int] = []
    page_strong: list[int] = []
    by_signal: Counter = Counter()
    strong_examples: list[str] = []
    garbage_examples: list[str] = []
    in_tables: Counter = Counter()
    corroborating = 0
    for page in pages:
        page_words.append(len(_WORD_RE.findall(page)))
        strong = 0
        for hit in mojibake.find_hits(page):
            by_signal[hit.signal] += 1
            if hit.strong:
                strong += 1
                if len(strong_examples) < MAX_EXAMPLES:
                    strong_examples.append(_example(page, hit))
            else:
                corroborating += 1
                if hit.signal in _GARBAGE_NAMES:
                    if _on_table_line(page, hit.start):
                        in_tables[hit.signal] += 1
                    if len(garbage_examples) < MAX_EXAMPLES:
                        garbage_examples.append(_example(page, hit))
        page_strong.append(strong)

    words = sum(page_words)
    strong_hits = sum(page_strong)
    doc_rate = 1000.0 * strong_hits / words if words else 0.0
    bad_pages = sum(
        1
        for w, h in zip(page_words, page_strong)
        if w >= PAGE_MIN_WORDS and h >= PAGE_HITS_MIN and 1000.0 * h / w >= PAGE_RATE_MIN
    )
    median_words = float(statistics.median(page_words)) if page_words else 0.0

    if median_words < NO_TEXT_MEDIAN_WORDS:
        status = "no_text"
    elif (doc_rate >= DOC_RATE_MIN and strong_hits >= DOC_HITS_MIN) or bad_pages >= max(
        BAD_PAGES_MIN, BAD_PAGES_SHARE * len(pages)
    ):
        status = "garbled"
    else:
        status = "pass"

    examples = (strong_examples + garbage_examples)[:MAX_EXAMPLES]
    return TextQuality(
        status=status,
        doc_rate=round(doc_rate, 4),
        strong_hits=strong_hits,
        words=words,
        pages=len(pages),
        bad_pages=bad_pages,
        median_page_words=median_words,
        hits_by_signal=dict(by_signal),
        corroborating_hits=corroborating,
        examples=tuple(examples),
        summary=_summary(status, by_signal, in_tables, examples),
    )
