"""Simple-fact counters over Marker markdown for the golden-set check.

All functions are pure and dependency-free (stdlib + seedgraph helpers).

Page markers
    Marker ``paginate_output`` lines ``{N}----`` (0-based page index). Parsed with
    the same delimiter rule as :mod:`seedgraph.segment`.

Equations ("math spans") — the stage-0 definition of "equations still parse"
    Scan order:

    1. Escaped dollars ``\\$`` are removed first (they are currency, not math).
    2. Display math: ``$$`` tokens are paired left to right across the whole
       document (blocks may span lines). Each pair is one *display* span. A final
       unpaired ``$$`` opens an unclosed block running to the end of the document;
       it counts as one display span that does not parse. Display spans are then
       blanked out (newlines kept) before step 3.
    3. Inline math: within each line, single ``$`` tokens are paired left to right.
       Each pair is one *inline* span. A leftover unpaired ``$`` on a line counts as
       one inline span that does not parse.

    A span *parses* when its delimiters are paired (above), its curly braces
    balance (never closing more than opened, ending at zero; escaped ``\\{`` /
    ``\\}`` and ``\\\\`` are ignored), and its ``\\left`` and ``\\right`` counts
    are equal. This is a syntactic sanity check, not a TeX render.

Tables
    A table block is a run of consecutive lines that start with ``|`` (after
    leading whitespace) and contain at least one header-separator row: a line made
    only of ``|``, ``-``, ``:`` and spaces that contains ``---`` (so ``|---|``,
    ``| :--- | ---: |`` all count). Runs of pipe lines without a separator row are
    not counted.

References
    Found when :func:`seedgraph.sections.parser.parse_sections` classifies any
    section as ``references``.

Similarity
    Page markers stripped, each non-blank line whitespace-collapsed, then a
    length-weighted line-level :class:`difflib.SequenceMatcher` ratio
    (``autojunk=False``). Identical normalized text short-circuits to exactly 1.0.
    Inputs beyond :data:`SIMILARITY_MAX_LINES` lines are truncated (``sampled``).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

_PAGE_DELIM = re.compile(r"(?m)^[ \t]*\{(\d+)\}-{3,}[ \t]*$")
_ESCAPED_DOLLAR = re.compile(r"\\\$")
_DISPLAY = re.compile(r"\$\$")
_LEFT = re.compile(r"\\left(?![A-Za-z])")
_RIGHT = re.compile(r"\\right(?![A-Za-z])")
_SEPARATOR_ROW = re.compile(r"^\|[\s|:\-]*$")

SIMILARITY_MAX_LINES = 20000


def page_markers(markdown: str) -> list[int]:
    """Return the 0-based page indices of every Marker page marker, in order."""
    return [int(m.group(1)) for m in _PAGE_DELIM.finditer(markdown)]


@dataclass(frozen=True)
class MathStats:
    display: int
    inline: int
    non_parsing: int

    @property
    def total(self) -> int:
        return self.display + self.inline


def _span_parses(body: str) -> bool:
    if len(_LEFT.findall(body)) != len(_RIGHT.findall(body)):
        return False
    cleaned = body.replace("\\\\", "").replace("\\{", "").replace("\\}", "")
    depth = 0
    for ch in cleaned:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def math_stats(markdown: str) -> MathStats:
    """Count display/inline math spans and those that do not parse (module docstring)."""
    text = _ESCAPED_DOLLAR.sub("", markdown)
    display = inline = bad = 0

    tokens = [m.start() for m in _DISPLAY.finditer(text)]
    pieces: list[str] = []
    cursor = 0
    for i in range(0, len(tokens), 2):
        open_at = tokens[i]
        if i + 1 < len(tokens):
            close_at = tokens[i + 1]
            body = text[open_at + 2 : close_at]
            end = close_at + 2
            ok = _span_parses(body)
        else:  # unclosed block runs to end of document
            body = text[open_at + 2 :]
            end = len(text)
            ok = False
        display += 1
        bad += 0 if ok else 1
        pieces.append(text[cursor:open_at])
        pieces.append("\n" * body.count("\n"))
        cursor = end
    pieces.append(text[cursor:])
    remainder = "".join(pieces)

    for line in remainder.splitlines():
        parts = line.split("$")
        n_dollars = len(parts) - 1
        if n_dollars == 0:
            continue
        for j in range(n_dollars // 2):
            inline += 1
            bad += 0 if _span_parses(parts[2 * j + 1]) else 1
        if n_dollars % 2:
            inline += 1
            bad += 1
    return MathStats(display=display, inline=inline, non_parsing=bad)


def count_tables(markdown: str) -> int:
    """Count markdown table blocks (pipe-line runs containing a separator row)."""
    count = 0
    in_block = False
    has_sep = False
    for raw in markdown.splitlines() + [""]:
        line = raw.strip()
        if line.startswith("|"):
            in_block = True
            if "---" in line and _SEPARATOR_ROW.match(line):
                has_sep = True
            continue
        if in_block and has_sep:
            count += 1
        in_block = has_sep = False
    return count


def has_references(markdown: str) -> bool:
    """True when the section parser classifies any section as ``references``."""
    from seedgraph.sections.parser import parse_sections

    sections = parse_sections(
        markdown,
        markdown_id="md_golden",
        markdown_hash="0",
        source_file_id="sf_golden",
        source_file_hash="0",
        work_id="work_golden",
    )
    return any(s.section_kind == "references" for s in sections)


def _normalized_lines(markdown: str) -> list[str]:
    text = _PAGE_DELIM.sub("", markdown)
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def similarity_detail(old: str, new: str) -> tuple[float, bool]:
    """Return ``(ratio, sampled)`` per the module docstring."""
    a, b = _normalized_lines(old), _normalized_lines(new)
    if a == b:
        return 1.0, False
    sampled = len(a) > SIMILARITY_MAX_LINES or len(b) > SIMILARITY_MAX_LINES
    a, b = a[:SIMILARITY_MAX_LINES], b[:SIMILARITY_MAX_LINES]
    total = sum(map(len, a)) + sum(map(len, b))
    if total == 0:
        return 1.0, sampled
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    matched = sum(sum(len(x) for x in a[blk.a : blk.a + blk.size]) for blk in matcher.get_matching_blocks())
    return 2.0 * matched / total, sampled


def similarity(old: str, new: str) -> float:
    return similarity_detail(old, new)[0]
