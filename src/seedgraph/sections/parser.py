"""Deterministic ATX-heading markdown parser → ordered ``Section`` records.

Phase 3 — Evidence Spans. The required, primary sectioning substrate (decisions
31/52/68: markdown-header sections primary, flat self-nesting, markdown-hash-derived
deterministic ``section_id``). A single forward scan over ATX headings (``#`` ..
``######``) yields ordered sections with:

* half-open ``[start_char, end_char)`` code-point offsets,
* ATX ``level`` (0 == preamble / leading heading-less text),
* 0-based ``ordinal`` (document order),
* ``parent_section_id`` via a heading-level stack (flat self-nesting),
* ``heading_path`` breadcrumb (e.g. ``'2 Identification > 2.1 Assumptions'``),
* ``section_kind`` classified from the heading text (e.g. ``## References`` →
  ``references``) — default ``body``.

``section_id`` is content-deterministic — ``ids.section_id(markdown_hash, ordinal)``
== ``'sec_' + sha256(markdown_hash|ordinal)[:16]`` — keyed on the markdown *hash*, so
re-parsing identical bytes yields identical ids (idempotent ``INSERT OR REPLACE``) and
ids are stable even when the cache ``markdown_id`` row differs but bytes are identical
(rebuild-safety). Setext-only / heading-less markdown falls entirely into a single
level-0 preamble section. ``heading_text`` is restricted full text, so sections are
private-by-default (project.db only).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from seedgraph import ids

#: Staleness key stamped onto every section (column ``section_parser_version``).
#: Bump when sectioning boundaries change; invalidates sections for re-parse.
#: ``secparse-2`` (Build F chunk 3): the references cue is now start-anchored on the
#: prefix-stripped heading text, so stored ``section_kind`` values change for
#: numbered/bold/anchor-span references headings and for prose headings that merely
#: mention "references" — a corpus-wide re-parse of ``document_sections`` (and the
#: downstream re-derivation of ``reference_entries``) via the staleness mechanism.
SECTION_PARSER_VERSION = "secparse-2"

#: ATX heading: 1..6 leading ``#`` then required space then the heading text, with
#: optional trailing ``#`` run (closed ATX). Setext (``===``/``---`` underlines) is
#: deliberately NOT matched — Setext-only documents fall into the level-0 preamble.
_ATX = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")

#: Reference-heading classification (decisions 31/52). Matched against the heading
#: text AFTER its tolerant prefix (marker anchor span, bold/italic emphasis, and a
#: leading ``N.``/``N)``/``N.M`` section number) is stripped by
#: :func:`_strip_heading_prefix` — the shapes marker actually emits
#: (``### 6. REFERENCES``, ``# **References**``, ``# <span id=...></span>**References**``;
#: v1 ``_HEADING_RE`` recorded these numbered/bold/anchor-span prefixes as 7 of 10
#: references sections a naive regex dropped on the live 35-paper corpus). The
#: references cue is START-ANCHORED with one optional leading qualifier word (ported
#: from v1 ``_SUBSECTION_CUE_RE``, ``extract/bib_parser.py:196-201``) so
#: ``Selected References`` / ``Rejoinder References`` classify ``references`` while a
#: prose ``Comparison with previous references`` (cue mid-line) stays ``body``. The
#: other kinds remain plain substring cues.
_KIND_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "references",
        re.compile(
            r"^(?:[A-Za-z][\w'.-]*\s+)?"  # optional single leading qualifier word
            r"(?:references?\b|bibliography\b|works\s+cited\b|literature\s+cited\b"
            r"|reference\s+list\b|\S+\s+cited\b)",
            re.I,
        ),
    ),
    ("abstract", re.compile(r"\babstract\b", re.I)),
    ("appendix", re.compile(r"\b(appendix|appendices)\b", re.I)),
    ("acknowledgments", re.compile(r"\backnowledge?ments?\b", re.I)),
]

#: The tolerant heading-text prefix stripped before kind classification, mirroring
#: v1 ``_ATX_SUBHEADING_RE``'s strip (``extract/bib_parser.py:178-187``): a marker
#: page-anchor span, markdown bold/italic/code markers, then a leading
#: ``N.``/``N)``/``N.M`` section number. LOAD-BEARING under the start-anchored
#: references cue above — a leading digit or ``*`` would otherwise fail the cue's
#: single-qualifier-word alternative and silently drop a whole bibliography (v1's
#: live-caught 7-of-10 scar). Only stripped for MATCHING; ``heading_text`` is stored
#: verbatim.
_ANCHOR_SPAN = re.compile(r"<span[^>]*>\s*</span>")
_EMPHASIS = re.compile(r"[*_`]+")
_LEADING_NUM = re.compile(r"^\s*\d{1,3}(?:\.\d{1,3})*[.)]?\s+")


def _strip_heading_prefix(heading_text: str) -> str:
    """Strip the tolerant prefix (anchor span, emphasis markers, leading number)."""
    text = _ANCHOR_SPAN.sub("", heading_text)
    text = _EMPHASIS.sub("", text)
    text = _LEADING_NUM.sub("", text.strip())
    return text.strip()


def _classify_kind(heading_text: str | None) -> str:
    """Classify a heading's ``section_kind`` (default ``body``)."""
    if not heading_text:
        return "body"
    probe = _strip_heading_prefix(heading_text.strip())
    for kind, pattern in _KIND_PATTERNS:
        if pattern.search(probe):
            return kind
    return "body"


@dataclass(frozen=True)
class Section:
    """One parsed section — mirrors the ``document_sections`` row it will be stored as.

    Offsets are Python ``str`` code-point indices into the decoded markdown. ``level``
    0 marks the preamble; ``parent_section_id`` is a soft (deterministic, non-FK)
    self-reference; ``page_start`` / ``page_end`` are best-effort (None when no
    pagination delimiters).
    """

    section_id: str
    markdown_id: str
    markdown_hash: str
    source_file_id: str
    source_file_hash: str
    work_id: str
    parent_section_id: str | None
    level: int
    ordinal: int
    heading_text: str | None
    heading_path: str | None
    section_kind: str
    start_char: int
    end_char: int
    page_start: int | None
    page_end: int | None


def parse_sections(
    markdown: str,
    *,
    markdown_id: str,
    markdown_hash: str,
    source_file_id: str,
    source_file_hash: str,
    work_id: str,
) -> list[Section]:
    """Parse ``markdown`` into ordered :class:`Section` records (deterministic).

    Single forward ATX-heading scan: assigns ``level`` / ``ordinal`` /
    ``parent_section_id`` (heading-level stack) / ``heading_path`` / ``section_kind``,
    half-open offsets, and a content-deterministic ``section_id`` per ordinal. Leading
    heading-less text and Setext-only documents become a single level-0 preamble. Pages
    are filled best-effort from in-markdown delimiters (else None). Identical bytes →
    identical sections/ids (idempotent).
    """
    from seedgraph.segment import page_boundaries, page_for_range

    total = len(markdown)
    page_bounds = page_boundaries(markdown)

    # 1. Locate every ATX heading: (line_start_offset, level, heading_text).
    headings: list[tuple[int, int, str]] = []
    for match in _ATX.finditer(markdown):
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        headings.append((match.start(), level, heading_text))

    # 2. Determine raw section boundaries as (start, end, level, heading_text).
    #    Sections tile the whole document with no gaps (preamble + headings), so
    #    every offset resolves to exactly one section.
    raw: list[tuple[int, int, int, str | None]] = []
    if not headings:
        # Heading-less / Setext-only: the entire document is one level-0 preamble.
        if markdown.strip():
            raw.append((0, total, 0, None))
    else:
        first_start = headings[0][0]
        # Preamble before the first ATX heading (only when it carries content).
        if first_start > 0 and markdown[:first_start].strip():
            raw.append((0, first_start, 0, None))
        for index, (start, level, heading_text) in enumerate(headings):
            end = headings[index + 1][0] if index + 1 < len(headings) else total
            raw.append((start, end, level, heading_text))

    # 3. Assign ordinals, parent ids (heading-level stack), breadcrumb, kind, pages.
    sections: list[Section] = []
    stack: list[tuple[int, str, str | None]] = []  # (level, section_id, heading_text)
    for ordinal, (start, end, level, heading_text) in enumerate(raw):
        sec_id = ids.section_id(markdown_hash, ordinal)
        if level == 0:
            # Preamble is not part of the heading hierarchy.
            parent_section_id = None
            heading_path = None
        else:
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_section_id = stack[-1][1] if stack else None
            ancestors = [text for _lvl, _sid, text in stack if text]
            heading_path = " > ".join([*ancestors, heading_text]) if heading_text else None
            stack.append((level, sec_id, heading_text))
        page_start, page_end = page_for_range(page_bounds, start, end)
        sections.append(
            Section(
                section_id=sec_id,
                markdown_id=markdown_id,
                markdown_hash=markdown_hash,
                source_file_id=source_file_id,
                source_file_hash=source_file_hash,
                work_id=work_id,
                parent_section_id=parent_section_id,
                level=level,
                ordinal=ordinal,
                heading_text=heading_text,
                heading_path=heading_path,
                section_kind=_classify_kind(heading_text),
                start_char=start,
                end_char=end,
                page_start=page_start,
                page_end=page_end,
            )
        )
    return sections


def section_for_offset(sections: list[Section], start_char: int) -> str | None:
    """Return the ``section_id`` of the deepest section containing ``start_char``.

    Point-in-interval lookup over half-open ``[start_char, end_char)`` ranges; returns
    the deepest (most specific) containing section, or ``None`` if no section contains
    the offset. Used to (re)resolve ``evidence_spans.section_id`` at index/create time.
    """
    best: Section | None = None
    for section in sections:
        if section.start_char <= start_char < section.end_char:
            # Deepest == highest level; ties broken by document order (latest wins).
            if best is None or section.level >= best.level:
                best = section
    return best.section_id if best is not None else None
