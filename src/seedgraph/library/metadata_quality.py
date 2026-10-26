"""Metadata quality rules for titles (docs/V3_UI_DESIGN.md §7). Pure: no DB, no I/O.

    clean_title(s)                         -> str | None
    generic_title_reason(s, *, venue=None) -> str | None

:func:`clean_title` removes what a cover page adds to a title: footnote and affiliation
marks (``†``, ``‡``, ``§``, ``¶``, ``*``), trailing Unicode superscript digits and
``<sup>…</sup>``; it collapses whitespace and trims trailing ``. , ; :``.
:func:`generic_title_reason` says why a (cleaned) title is not a paper's title: a
stoplisted section or front-matter name, ``chapter N``, the venue's own name, or fewer
than three characters. Callers do not save such a title; they raise a
``metadata_generic_title`` attention item instead.

Rules chosen where the design is silent:
- ``†‡§¶`` are removed anywhere (they are never part of a title); ``*`` and superscript
  digits only at the end; ASCII digits are never touched ("COVID-19", "Web 2.0").
- ``?``, ``!`` and closing brackets are kept; only ``. , ; :`` are trimmed.
"""

from __future__ import annotations

import re

_SUP_TAG_RE = re.compile(r"<sup>.*?</sup>", re.IGNORECASE | re.DOTALL)
_MARKS_RE = re.compile("[†‡§¶]")
_TRAILING_RE = re.compile("[\\s*.,;:¹²³⁰⁴-⁹]+$")
_WS_RE = re.compile(r"\s+")

#: Section and front-matter names that are never a paper's title (compared casefolded).
GENERIC_TITLES: frozenset[str] = frozenset({
    "review", "book review", "book reviews", "reviews", "introduction", "contents",
    "table of contents", "abstract", "front matter", "back matter", "preface", "foreword",
    "acknowledgements", "acknowledgments", "editorial", "editorial board", "article",
    "untitled", "working paper", "title page", "index", "references", "bibliography",
    "erratum", "corrigendum", "appendix", "discussion", "comment", "reply",
})
_CHAPTER_RE = re.compile(r"^chapter\s+(\d+|[ivxlcdm]+)$", re.IGNORECASE)
MIN_TITLE_CHARS = 3
#: The cover region: the first characters of a paper's text, the same span the metadata
#: task packages (``work.tasks_metadata.HEAD_CHARS``). A paper's own title, year and DOI
#: are looked for here only - a short paper's first pages can already hold its
#: reference list, where cited works' titles, years and DOIs are printed.
COVER_CHARS = 2600


def cover_region(pages: list[str] | tuple[str, ...]) -> list[str]:
    """The cover region of a paper's first pages, as a one-page list (``[]`` for no text)."""
    text = "\n".join(pages)[:COVER_CHARS]
    return [text] if text.strip() else []


def clean_title(title: str | None) -> str | None:
    """The title without footnote marks, collapsed and trimmed; ``None`` when nothing is left."""
    if title is None:
        return None
    text = _SUP_TAG_RE.sub("", title)
    text = _MARKS_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    text = _TRAILING_RE.sub("", text).strip()
    return text or None


def _key(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().casefold()


def generic_title_reason(title: str | None, *, venue: str | None = None) -> str | None:
    """Why ``title`` is not a paper title, or ``None`` when it may be saved."""
    cleaned = clean_title(title)
    if cleaned is None or len(cleaned) < MIN_TITLE_CHARS:
        return f"shorter than {MIN_TITLE_CHARS} characters"
    key = _key(cleaned)
    if key in GENERIC_TITLES:
        return f"'{cleaned}' is a generic section or front-matter name, not a paper title"
    if _CHAPTER_RE.match(key):
        return f"'{cleaned}' is a chapter label, not a paper title"
    if venue and key == _key(clean_title(venue) or ""):
        return f"'{cleaned}' is the venue's name, not the paper's title"
    return None
