"""The one text normalisation shared by quote highlight, selection anchoring and the
highlight measurement (v3 stage 3 design §3.2).

``fold(text)`` returns a :class:`Folded`: the folded string plus, for every folded
character, the half-open ``(start, end)`` range of raw ``text`` it came from, so a match
in folded text maps back to verbatim raw offsets (``Folded.raw_span``).

The steps, in order:

1. Drop what the PDF does not print, found on the raw text: Marker page delimiters
   (``{N}`` + dashes on their own line), display and inline ``$…$`` math, link targets
   (``](url)``), HTML-like tags (``<sup>``, ``<span …>``; their contents stay), line-end
   hyphenation (``-``, optional spaces, a newline, then a letter), soft hyphens and
   zero-width characters.
2. Lost-ligature mojibake back to letters, in context (``library.mojibake``, families A
   and B: ``identiÖed`` → ``identified``).
3. Unicode NFKC and lower case, per grapheme cluster (a base character plus its
   combining marks), so a decomposed ``e`` + U+0301 folds like a composed ``é``. Final
   sigma folds to ``σ``.
4. Keep only ``0-9`` and Unicode letters. Whitespace, punctuation and the remaining
   markdown (``*``, ``_``, ``#``, backticks, pipes, brackets) disappear, which absorbs
   PDF.js's per-item spacing, line breaks and curly vs straight quotes.

``find_passage(haystack, needle)`` is the shared finder over folded strings: every
full-needle hit, else (needles of 30+ characters only) three 30-character windows
(start, middle, end) found in order within 1.5x the needle's length.

``web/static/js/fold.js`` implements the same functions; both run the shared vectors in
``tests/fixtures/textfold_vectors.json``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .mojibake import fold_with_map as _mojibake_fold

#: Window length and spread limit of the window fallback in :func:`find_passage`.
WINDOW = 30
SPREAD = 1.5

_REMOVE = re.compile(
    r"^[ \t]*\{[0-9]+\}-{3,}[ \t\r]*$"  # Marker page delimiter line
    r"|\$\$[\s\S]*?\$\$"  # display math
    r"|\$[^$\n]+\$"  # inline math
    r"|\]\([^)\s]*(?:\s+\"[^\"]*\")?\)"  # link / image target: ](url) or ](url "title")
    r"|</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>"  # an HTML-like tag (contents kept)
    r"|-[ \t]*\r?\n[ \t]*(?=[^\W\d_])"  # line-end hyphenation before a letter
    "|[­​-‍⁠﻿]",  # soft hyphen, zero-width characters
    re.M,
)


@dataclass(frozen=True)
class Folded:
    """A folded string and, per folded character, the raw ``[start, end)`` it came from."""

    text: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]

    def raw_span(self, start: int, end: int) -> tuple[int, int]:
        """Raw ``(start, end)`` covering folded characters ``[start, end)``."""
        if not 0 <= start < end <= len(self.text):
            raise ValueError(f"folded range [{start}, {end}) is empty or out of bounds")
        return self.starts[start], self.ends[end - 1]


def _keep(ch: str) -> bool:
    return "0" <= ch <= "9" or ch.isalpha()


def fold(text: str) -> Folded:
    """Fold ``text`` (see the module doc) and keep the index map back to raw offsets."""
    # 1. drop unprinted markup, keeping each surviving character's raw index
    kept: list[str] = []
    kmap: list[int] = []
    cursor = 0
    for m in _REMOVE.finditer(text):
        for i in range(cursor, m.start()):
            kept.append(text[i])
            kmap.append(i)
        cursor = m.end()
    for i in range(cursor, len(text)):
        kept.append(text[i])
        kmap.append(i)
    k_text = "".join(kept)

    # 2. mojibake, composing its spans (over k_text) with kmap
    m_text, m_spans = _mojibake_fold(k_text)
    raw_spans = [(kmap[s], kmap[e - 1] + 1) for s, e in m_spans]

    # 3 + 4. per cluster: NFKC, lower case, filter
    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    i, n = 0, len(m_text)
    while i < n:
        j = i + 1
        while j < n and unicodedata.category(m_text[j]).startswith("M"):
            j += 1
        start, end = raw_spans[i][0], raw_spans[j - 1][1]
        for ch in unicodedata.normalize("NFKC", m_text[i:j]):
            for lc in ch.lower():
                if lc == "ς":  # final sigma
                    lc = "σ"
                if _keep(lc):
                    out.append(lc)
                    starts.append(start)
                    ends.append(end)
        i = j
    return Folded("".join(out), tuple(starts), tuple(ends))


def _all_hits(haystack: str, needle: str) -> list[tuple[int, int]]:
    hits = []
    pos = haystack.find(needle)
    while pos != -1:
        hits.append((pos, pos + len(needle)))
        pos = haystack.find(needle, pos + 1)
    return hits


def find_passage(haystack: str, needle: str) -> list[tuple[int, int]]:
    """Folded ``[start, end)`` ranges of ``needle`` in ``haystack`` (both already folded).

    Every exact hit; if there is none and the needle has at least ``WINDOW`` characters,
    the window fallback: for each start-window hit (not inside an earlier result), the
    first middle-window hit at or after it and the first end-window hit at or after that,
    accepted when the whole range is at most ``SPREAD`` x the needle's length.
    """
    if not needle:
        return []
    hits = _all_hits(haystack, needle)
    n = len(needle)
    if hits or n < WINDOW:
        return hits
    mid = n // 2
    w_start = needle[:WINDOW]
    w_mid = needle[mid - WINDOW // 2: mid - WINDOW // 2 + WINDOW]
    w_end = needle[n - WINDOW:]
    out: list[tuple[int, int]] = []
    a = haystack.find(w_start)
    while a != -1:
        if not out or a >= out[-1][1]:
            b = haystack.find(w_mid, a)
            c = haystack.find(w_end, b) if b != -1 else -1
            if c != -1 and c + WINDOW - a <= SPREAD * n:
                out.append((a, c + WINDOW))
        a = haystack.find(w_start, a + 1)
    return out
