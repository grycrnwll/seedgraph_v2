"""The shared mojibake table (design §5.2), used by the text-quality gate, quote feedback
and quote re-matching.

Lost-ligature mojibake from old PDF fonts turns ``fi``/``ff``/``ffi``/``fl`` into Latin-1
lookalikes. Two families are seen in the library:

- **Family A**: ``Ö``→fi (identiÖed), ``§``→ff (di§erent), ``¢``+space→ffi (e¢ ciency),
  ``á``→fl (reáects), ``í``→apostrophe before ``s`` (authorís).
- **Family B**: ``…``→fi (…rst), ``¤``→ff (di¤erence), ``¢``→ffi.

Stage 1's title matcher keeps its own fold table (``library/titlecheck.py``) unchanged.

Signals:

- **Strong**: ``Ö § ¢ … ¤`` touching a lowercase ASCII letter on both sides (``¢`` may
  be followed by one space before the letter: family A's ``e¢ ciency``), or at word start
  (no space) followed by two or more lowercase letters (``Örst``).
- **Corroborating only**: in-word ``á`` and ``í`` (real letters in names and non-English
  words), and converter garbage: U+FFFD, each ``(cid:N)`` and each private-use-area code
  point (calibration found these in otherwise readable papers, mostly symbol fonts in
  tables). They are counted and reported but never trigger the gate.
- ``Þ`` and ``ß`` are not signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MojibakeGlyph:
    glyph: str
    family: str  # "A", "B" or "A+B"
    stands_for: str
    strength: str  # "strong" | "corroborating"


FAMILIES: tuple[MojibakeGlyph, ...] = (
    MojibakeGlyph("Ö", "A", "fi", "strong"),
    MojibakeGlyph("§", "A", "ff", "strong"),
    MojibakeGlyph("¢", "A+B", "ffi", "strong"),
    MojibakeGlyph("á", "A", "fl", "corroborating"),
    MojibakeGlyph("í", "A", "'", "corroborating"),
    MojibakeGlyph("…", "B", "fi", "strong"),
    MojibakeGlyph("¤", "B", "ff", "strong"),
)

STRONG_GLYPHS = "".join(g.glyph for g in FAMILIES if g.strength == "strong")
CORROBORATING_GLYPHS = "".join(g.glyph for g in FAMILIES if g.strength == "corroborating")

# Signal names used as keys of ``hits_by_signal``.
SIGNAL_REPLACEMENT = "U+FFFD"
SIGNAL_CID = "(cid:N)"
SIGNAL_PUA = "private_use"
CONVERTER_GARBAGE_SIGNALS = (SIGNAL_REPLACEMENT, SIGNAL_CID, SIGNAL_PUA)

_G = re.escape(STRONG_GLYPHS)
# A strong glyph inside a word (``¢`` may take one space before the next letter), or at
# word start followed by >= 2 lowercase letters.
_STRONG_GLYPH_RE = re.compile(
    rf"(?<=[a-z])(?:¢ ?|[{_G}])(?=[a-z])"
    rf"|(?<![A-Za-z])[{_G}](?=[a-z]{{2}})"
)
_CID_RE = re.compile(r"\(cid:\d+\)")
_PUA_RE = re.compile("[-\U000f0000-\U000ffffd\U00100000-\U0010fffd]")
_CORROBORATING_RE = re.compile(rf"(?<=[a-z])[{re.escape(CORROBORATING_GLYPHS)}](?=[a-z])")


@dataclass(frozen=True)
class Hit:
    signal: str  # the glyph, or one of the SIGNAL_* names
    start: int
    end: int
    strong: bool


def find_hits(text: str) -> list[Hit]:
    """Every strong and corroborating signal in ``text``, in text order."""
    hits: list[Hit] = []
    for m in _STRONG_GLYPH_RE.finditer(text):
        hits.append(Hit(m.group(0)[0], m.start(), m.end(), True))
    for m in re.finditer("�", text):
        hits.append(Hit(SIGNAL_REPLACEMENT, m.start(), m.end(), False))
    for m in _CID_RE.finditer(text):
        hits.append(Hit(SIGNAL_CID, m.start(), m.end(), False))
    for m in _PUA_RE.finditer(text):
        hits.append(Hit(SIGNAL_PUA, m.start(), m.end(), False))
    for m in _CORROBORATING_RE.finditer(text):
        hits.append(Hit(m.group(0), m.start(), m.end(), False))
    hits.sort(key=lambda h: h.start)
    return hits


# Fold: each mojibake glyph back to what it stands for, only in context.
_FOLD_RE = re.compile(
    r"(?P<ffi>(?<=[a-z])¢ ?(?=[a-z])|(?<![A-Za-z])¢(?=[a-z]{2}))"
    rf"|(?P<fi>(?<=[a-z])[Ö…](?=[a-z])|(?<![A-Za-z])[Ö…](?=[a-z]{{2}}))"
    r"|(?P<ff>(?<=[a-z])[§¤](?=[a-z])|(?<![A-Za-z])[§¤](?=[a-z]{2}))"
    r"|(?P<fl>(?<=[a-z])á(?=[a-z]))"
    r"|(?P<apos>(?<=[a-z])í(?=s\b))"
)
_FOLD_REPL = {"ffi": "ffi", "fi": "fi", "ff": "ff", "fl": "fl", "apos": "'"}


def fold_with_map(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Fold mojibake in ``text``; return the folded string and, for each folded
    character, the ``(start, end)`` source span in ``text`` it came from."""
    out: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for m in _FOLD_RE.finditer(text):
        for i in range(cursor, m.start()):
            out.append(text[i])
            spans.append((i, i + 1))
        repl = _FOLD_REPL[m.lastgroup or ""]
        for ch in repl:
            out.append(ch)
            spans.append((m.start(), m.end()))
        cursor = m.end()
    for i in range(cursor, len(text)):
        out.append(text[i])
        spans.append((i, i + 1))
    return "".join(out), spans


def fold(text: str) -> str:
    """``text`` with mojibake glyphs replaced by the letters they stand for (in context)."""
    return fold_with_map(text)[0]
