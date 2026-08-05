"""Phase 3b — deterministic, offline references parser.

Ports v1 ``extract/bib_parser.py``'s entry-splitting / reflow /
em-dash-repeated-author / OCR-fix / field-extraction heuristics, **minus** v1's
internal heading scanner (``_split_into_sections`` / ``_HEADING_RE``). The
heading scanner is *deleted*, not maintained: the references region is located
exclusively from ``phase_3``'s ``document_sections`` rows where
``section_kind='references'`` (decisions 31/52/66/68; deviation (a) of r2-2).

This module is **PURE**: no network, no DB, no provider. It turns markdown text
plus a list of references char-ranges into one :class:`BibEntry` per cited work.

Decisions implemented here:
- 31/52/66/68 — markdown-header ``document_sections`` are the substrate; never a
  parallel heading scanner. ``parse_references`` returns ``[]`` for empty ranges.
- doc 04 §10–11 — reference entries extracted from markdown, kept verbatim for
  recall (the raw text is carried to ``reference_entries.raw_reference_text``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

# ===========================================================================
# OCR / ligature fixes (conservative — bib-extractor Step 5)
# ===========================================================================
#: The replacement glyph pdftotext leaves for a lost non-ASCII character.
_REPL = "�"

#: ``�`` BETWEEN digits is a lost dash in a page/issue range → ``-`` (safe).
_REPL_BETWEEN_DIGITS_RE = re.compile(rf"(?<=\d){_REPL}(?=\d)")

#: A small table of UNAMBIGUOUS OCR substitutions (bib-extractor logs these).
_OCR_FIXES: tuple[tuple[str, str], ...] = (
    ("$HORROCKS", "Shorrocks"),
    ("Combridge", "Cambridge"),
)


def _apply_ocr_fixes(text: str) -> str:
    """Apply the conservative OCR/ligature fixes from bib-extractor Step 5.

    Only ``�`` between digits (page ranges) is touched globally; in-word
    ``�`` (lost ligature/accent) is deliberately LEFT ALONE. The small
    substitution table handles a couple of unambiguous garbles.
    """
    text = _REPL_BETWEEN_DIGITS_RE.sub("-", text)
    for bad, good in _OCR_FIXES:
        text = text.replace(bad, good)
    return text


# ===========================================================================
# Page-break debris stripping (bib-extractor Step 3)
# ===========================================================================
_DEBRIS_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)^\s*Downloaded from\b"),
    re.compile(r"(?i)^\s*This content downloaded from\b"),
    re.compile(r"(?i)^\s*All use subject to JSTOR\b"),
    # A bare URL line is debris — EXCEPT a DOI URL, which is part of an entry
    # (its identifier) and is joined onto the entry it follows.
    re.compile(r"(?i)^\s*https?://(?!(?:dx\.)?doi\.org/)\S+\s*$"),
    re.compile(r"^\s*\d{1,4}\s*$"),  # standalone page number (1-4 digits)
    re.compile(r"(?i)^\s*\[?\s*(?:No\.|Vol\.)\s*\d"),  # issue markers
    re.compile(r"(?i)^\s*DISCUSSION OF THE PAPER\b"),
)

#: A running ALL-CAPS author header line (e.g. ``RICHARDSON AND GREEN``): all
#: caps, short, contains ``AND`` or a comma, and — crucially — has NO year (a
#: real entry carries a year), so we do not strip an entry.
_CAPS_HEADER_RE = re.compile(r"^[^a-z0-9]*[A-Z][A-Z .,&-]{1,58}$")
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")

#: A leftover ATX heading line inside a references range. The references region is
#: located via ``document_sections`` char ranges whose ``start_char`` is the
#: heading line itself, so the very first line of a range is the ``## References``
#: heading — and a multi-section range may carry sub-headings. Such heading lines
#: are NOT reference entries and are stripped (they are not author-comma-shredded
#: into spurious entries). This is a per-line skip, NOT a heading SCANNER: it
#: never *opens* a section (decisions 31/52/66/68).
_ATX_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}[ \t]+\S")


def _is_debris(line: str) -> bool:
    """Whether ``line`` is page-break debris to drop (bib-extractor Step 3)."""
    if _ATX_HEADING_LINE_RE.match(line):
        return True
    if any(p.search(line) for p in _DEBRIS_RES):
        return True
    stripped = line.strip()
    if (
        _CAPS_HEADER_RE.match(stripped)
        and ("AND" in stripped or "," in stripped)
        and not _YEAR_RE.search(stripped)
    ):
        return True
    return False


# ===========================================================================
# Entry-start / continuation / em-dash detection (bib-extractor Steps 4 & 5)
# ===========================================================================
#: An enumeration marker opening a line: ``[1]`` / ``(2)`` / ``1.`` / ``12)``.
_ENUM_RE = re.compile(r"^\s*(?:\[\s*\d{1,3}\s*\]|\(\s*\d{1,3}\s*\)|\d{1,3}[.)])\s+")
#: Same, but CAPTURING the number — to test for a sequential ``[1][2][3]…`` run.
_ENUM_NUM_RE = re.compile(r"^\s*(?:\[\s*(\d{1,3})\s*\]|\(\s*(\d{1,3})\s*\)|(\d{1,3})[.)])\s+")


def _leading_enum_number(line: str) -> Optional[int]:
    """The enumerator number a line opens with (``[3] …`` → 3), or ``None``."""
    m = _ENUM_NUM_RE.match(line)
    if not m:
        return None
    return int(next(g for g in m.groups() if g is not None))


def _is_enumerated_section(cleaned: list[str]) -> bool:
    """Whether a section is a NUMBERED bibliography (``[1] … [2] … [3] …``).

    Detected by a mostly-increasing run of >= 3 leading enumerator numbers rather
    than a fraction of all lines — a long reference wraps across many continuation
    lines (which carry no number), so a line-fraction test misclassifies a clean
    numbered section as name-year and lets the author-comma splitter shred it. A
    sequential run survives wrapping and won't fire on a name-year section's stray
    ``[1]``.
    """
    nums = [n for ln in cleaned if (n := _leading_enum_number(ln)) is not None]
    if len(nums) < 3:
        return False
    increasing = sum(1 for i in range(1, len(nums)) if nums[i] > nums[i - 1])
    return increasing >= 0.8 * (len(nums) - 1)


#: Em-dash / repeated-author continuation marker at the START of an entry: a run
#: of hyphens/em-dashes (or the mojibake glyph) optionally followed by ``.``/``:``.
_EMDASH_CONT_RE = re.compile(rf"^\s*(?:-{{2,}}|—+|–+|{_REPL})\s*[.:]?\s*")

#: An author-year entry START: ``Lastname, I.`` … ``(YYYY)`` / ``, YYYY``.
_AUTHOR_YEAR_START_RE = re.compile(
    r"^[\"“]?[A-Z][\w'’.-]+,\s+[^()\n]{0,160}?\(?(?:1[5-9]\d{2}|20\d{2})[)\.,]"
)

#: A mid-line NEW-entry boundary (collapsed multi-entry line).
_COLLAPSE_SPLIT_RE = re.compile(r"(?<=[.’\"”])\s+(?=[A-Z][\w'’-]+,\s+[A-Z])")

#: A continuation line that is clearly NOT a new entry start.
_CONTINUATION_RE = re.compile(r"^\s*(?:[a-z]|[,&-]|pp\.|eds?\.|ch\.|vol\.|no\.)")

#: A DOI anywhere in an entry (extraction + a split boundary signal).
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>,;)\]]+", re.IGNORECASE)

#: An arXiv id anywhere in an entry — a STRONG id for CS/ML references.
_ARXIV_RE = re.compile(
    r"arxiv[:\s]*\s*((?:[a-z][a-z.-]+/)?\d{4}\.\d{4,5}|[a-z][a-z.-]+/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)

# --- Real-marker line cruft (CS/ML arXiv output): list bullets, page-anchor
# spans, and back-reference links that pdf->markdown leaves inside the reference
# list. Stripped per line BEFORE entry detection.
_LIST_MARKER_RE = re.compile(r"^\s*[-*+]\s+")
_ANCHOR_SPAN_RE = re.compile(r"<span[^>]*>\s*</span>\s*")
_BACKREF_LINK_RE = re.compile(r"\s*\[[\d,\s]+\]\(#[^)]*\)")

# --- "Demarkdown" normalization. Marker (PDF->markdown) leaves inline markup
# inside reference lines that defeats entry-start detection
# (``_AUTHOR_YEAR_START_RE``) and the collapse-splitter (``_COLLAPSE_SPLIT_RE``),
# both of which assume a reference line opens with a bare capital letter:
#   * Elsevier/ScienceDirect hyperlink every author/title, so every line opens
#     with ``[`` and the whole list collapses into ONE entry.
#   * scanned papers wrap each reference in ``**...**`` bold, so the leading
#     ``**`` blocks entry-start and the collapse-split.
# Applied per line AFTER ``_BACKREF_LINK_RE`` (so numeric ``[1,2](#page…)``
# backrefs are gone first and a bare ``[n]`` enumerator is never touched — the
# link regex requires ``](…)`` immediately after ``]``).
#: A markdown link ``[text](url)``. Paren-safe: the URL body tolerates ONE level
#: of balanced inner parens (Elsevier backref URLs contain ``(22)`` etc.).
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((?:[^()]|\([^()]*\))*\)")
#: A DOI inside a link URL — kept even when the visible text lacks it (e.g.
#: ``[CrossRef](https://doi.org/10.1234/xyz)``) so ``_extract_doi`` still fires.
_MD_LINK_URL_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s)]+")
#: Bold/italic emphasis markers (``**``/``*``). Asterisks ONLY — underscores are
#: LEFT ALONE (DOIs / arXiv ids / identifiers can contain ``_``).
_EMPHASIS_RE = re.compile(r"\*\*|\*")


def _demarkdown_link(m: re.Match[str]) -> str:
    """Replace a markdown link with its visible text, preserving a URL-only DOI."""
    text = m.group(1)
    url = m.group(0)[m.end(1) - m.start() :]  # the ``](url)`` remainder
    doi = _MD_LINK_URL_DOI_RE.search(url)
    if doi and doi.group(0) not in text:
        return f"{text} {doi.group(0)}"
    return text


def _clean_entry_line(line: str) -> str:
    """Strip real-marker line cruft (anchor spans, list bullet, backref links) and
    normalize Marker's inline markdown (links -> text, ``*``/``**`` emphasis removed)
    so entry-start detection and the collapse-split see a bare-capital line start.
    """
    line = _ANCHOR_SPAN_RE.sub("", line)
    line = _LIST_MARKER_RE.sub("", line)
    line = _BACKREF_LINK_RE.sub("", line)
    line = _MD_LINK_RE.sub(_demarkdown_link, line)
    line = _EMPHASIS_RE.sub("", line)
    return line


#: A quoted title (IEEE style ``Authors, "Title," Venue``).
_QUOTED_TITLE_RE = re.compile(r"[\"“]([^\"“”]{6,})[\"”]")

#: Leading author initials in a numbered (``[n]``) reference: ``Y. Bengio``.
_INITIAL_SURNAME_RE = re.compile(r"^(?:[A-Z]\.-?\s*){1,4}([A-Z][a-zA-Z'’-]{1,})")


def _looks_like_entry_start(line: str) -> bool:
    """Whether ``line`` begins a NEW bibliography entry (bib-extractor Step 4)."""
    if _ENUM_RE.match(line):
        return True
    if _EMDASH_CONT_RE.match(line):
        return True  # a repeated-author entry IS a new entry (author carried fwd)
    if _CONTINUATION_RE.match(line):
        return False
    return bool(_AUTHOR_YEAR_START_RE.match(line.strip()))


# ===========================================================================
# Field extraction
# ===========================================================================
def _extract_year(text: str) -> Optional[int]:
    """First 4-digit year (1500-2099) in ``text``, or ``None``."""
    m = _YEAR_RE.search(text)
    return int(m.group(0)) if m else None


def _extract_doi(text: str) -> Optional[str]:
    """First DOI in ``text`` (bare, lowercased, trailing punctuation trimmed)."""
    m = _DOI_RE.search(text)
    if not m:
        return None
    doi = m.group(0).rstrip(".,;)")
    return doi.lower()


def _extract_arxiv(text: str) -> Optional[str]:
    """First arXiv id in ``text`` (bare id, version suffix dropped), or ``None``."""
    m = _ARXIV_RE.search(text)
    if not m:
        return None
    return m.group(1).lower()


_FIRST_AUTHOR_RE = re.compile(r"^[\"“]?\s*([A-Z][\w'’.-]+)\s*,")


def _extract_first_author(text: str) -> Optional[str]:
    """Surname of the first author, both citation styles.

    Style A (name-year): ``Lastname, I.`` → ``Lastname``. Style B (numbered
    CS/ML): ``[1] I. Lastname`` → ``Lastname``. A leading enumeration marker is
    stripped first.
    """
    body = _ENUM_RE.sub("", text).strip()
    m = _FIRST_AUTHOR_RE.match(body)
    if m:
        return m.group(1)
    m = _INITIAL_SURNAME_RE.match(body)
    if m:
        return m.group(1)
    return None


#: A year token with optional surrounding parens, to locate the title tail.
_YEAR_TOKEN_RE = re.compile(r"\(?\b(?:1[5-9]\d{2}|20\d{2})\b\)?[.,]?\s*")


def _extract_title(text: str) -> Optional[str]:
    """Best-effort title: the clause AFTER the author-year prefix.

    PRECISION GATE (plan §4.2(a), the no-year trigger): when the entry carries NO
    year token, return ``None`` so a no-year author shred never reaches ``by_title``
    and becomes a wrong node (it stays an UNRESOLVED stub instead).
    """
    body = _ENUM_RE.sub("", text).strip()
    m = _YEAR_TOKEN_RE.search(body)
    if m is None:
        return None
    tail = body[m.end():]
    tail = tail.strip().strip("\"“”'")
    if not tail:
        return None
    parts = re.split(r"\.\s+", tail, maxsplit=1)
    title = parts[0].strip(" .")
    return title or None


def _extract_title_numbered(text: str) -> Optional[str]:
    """Best-effort title for a Style-B (numbered ``[n]``) reference.

    Shape: ``[n] I. Author, ... Title. Venue, Year.`` — the title is the clause
    just BEFORE the venue/identifier tail. Cut at the first of: italic venue
    (``*...*``), an arXiv id, or a DOI; then take the last ``. ``-delimited clause
    of the head, skipping pure year/number and trailing ``Venue, Year`` clauses.

    NOTE: the ``*Venue*`` italic cue is stripped upstream by ``_clean_entry_line``'s
    demarkdown, so a bare trailing ``Venue, Year`` clause (e.g. ``Nature, 2015``)
    is skipped here to recover the same title boundary the ``*`` used to mark.
    """
    body = _ENUM_RE.sub("", text).strip()
    qm = _QUOTED_TITLE_RE.search(body)
    if qm:
        q = qm.group(1).strip().strip(",.")
        if len(q) >= 6:
            return q
    cut = len(body)
    star = body.find("*")
    if star > 0:
        cut = min(cut, star)
    for rx in (_ARXIV_RE, _DOI_RE):
        mm = rx.search(body)
        if mm:
            cut = min(cut, mm.start())
    head = re.sub(r"\b[Ii]n:?\s*$", "", body[:cut]).strip(" .,")
    if not head:
        return None
    for seg in reversed(re.split(r"\.\s+", head)):
        seg = seg.strip(" .,")
        if not (seg and re.search(r"[A-Za-z]", seg)):
            continue
        if re.fullmatch(r"(?:1[5-9]|20)\d{2}", seg):
            continue
        if re.search(r",\s*(?:1[5-9]|20)\d{2}\b", seg):
            continue  # a trailing ``Venue, Year`` clause is not the title
        return seg
    return None


# ===========================================================================
# Plain-vs-qualified references-label discriminator (Build E design 10)
# ===========================================================================
#: The CLOSED set of PLAIN references-section labels. A range whose normalized
#: label is exactly one of these keeps the pre-existing reflow behavior
#: byte-for-byte — including the load-bearing collapse-split for name-year
#: entries jammed onto one physical line. Anything ELSE that reached
#: ``section_kind='references'`` (e.g. "Rejoinder References", "References in
#: Discussion by X") is a QUALIFIED discrete sub-list — the v2 mapping of v1's
#: ``opened_by_heading`` flag (all v2 ranges are heading-opened by construction,
#: so the discriminator lives on the label, not on how the range was found).
#: Deliberately closed: the miss direction is asymmetric-benign — an unlisted
#: plain variant (e.g. "Sources") classifies QUALIFIED and merely loses the
#: collapse-split (collapsed lines stay unsplit; it never shreds).
_PLAIN_REFERENCE_LABELS = frozenset(
    {
        "references",
        "bibliography",
        "works cited",
        "literature cited",
        "reference list",
        "references and notes",
        "literature",
    }
)

#: Leading ``N.`` / ``N)`` numbering on a heading label (``6. References``).
_LABEL_NUMBERING_RE = re.compile(r"^\s*\d{1,3}[.)]\s*")
#: Markdown emphasis / code markers inside a heading label.
_LABEL_EMPHASIS_RE = re.compile(r"[*_`]+")


def _is_discrete_sublist_label(section_label: str) -> bool:
    """Whether ``section_label`` marks a QUALIFIED discrete reference sub-list.

    Normalizes the label (strip anchor spans, markdown emphasis, leading
    ``N.``/``N)`` numbering, and a trailing colon; collapse whitespace;
    lowercase) and tests it against the closed ``_PLAIN_REFERENCE_LABELS`` set.
    PLAIN → ``False`` (today's reflow behavior, byte-identical); everything
    else → ``True`` (a discussion/rejoinder-style discrete sub-list whose
    leading enumerators are authoritative and whose collapse-split is
    suppressed — see ``_reflow``).
    """
    label = _ANCHOR_SPAN_RE.sub("", section_label or "")
    label = _LABEL_EMPHASIS_RE.sub("", label)
    label = _LABEL_NUMBERING_RE.sub("", label)
    label = label.strip().rstrip(":").strip()
    label = re.sub(r"\s+", " ", label).lower()
    return label not in _PLAIN_REFERENCE_LABELS


# ===========================================================================
# BibEntry — the parse output
# ===========================================================================
@dataclass
class BibEntry:
    """One parsed bibliography entry (one cited work).

    ``raw`` is the verbatim reflowed one-line entry (PRIVATE / full-text-derived —
    persisted to ``reference_entries.raw_reference_text``). ``section_label`` is
    the originating ``document_sections`` heading text, carried as provenance so
    multi-section documents keep distinct labels. ``ordinal`` is the entry's
    1-based position within its references range. ``continued_author`` is True when
    an em-dash repeated-author convention carried the previous entry's first author
    forward.
    """

    raw: str
    section_label: str
    ordinal: int
    first_author: Optional[str]
    year: Optional[int]
    title: Optional[str]
    doi: Optional[str]
    arxiv: Optional[str]
    continued_author: bool = False

    def to_record(self) -> dict:
        """Project this entry into an identity record for ``canonical_key`` /
        ``upsert_work`` (phase_5) and the resolver. No DB/network here.
        """
        rec: dict[str, Any] = {}
        if self.doi:
            rec["doi"] = self.doi
        if self.arxiv:
            rec["arxiv"] = self.arxiv
        if self.title:
            rec["title"] = self.title
        if self.year:
            rec["year"] = self.year
        if self.first_author:
            rec["authors"] = [self.first_author]
        return rec


# ===========================================================================
# Reflow one references range into one physical line per entry
# ===========================================================================
def _reflow(lines: Iterable[str], *, discrete_list: bool = False) -> list[str]:
    """Reflow wrapped lines into one physical line per entry (bib-extractor Step 4).

    Strips real-marker cruft + debris (incl. stray ATX heading lines), joins
    hanging-indent / continuation lines onto the entry they continue, and starts a
    new entry on an enumeration / author-year / em-dash start. ENUMERATED sections
    (``[1] … [2] … [3]``) split ONLY on the enumerator and suppress the aggressive
    author-comma collapse-split (so an ``Initial. Surname, Initial. Surname`` author
    list is never shredded into fragments).

    ``discrete_list=True`` marks a QUALIFIED discrete sub-list (a
    discussion/rejoinder-style range, per ``_is_discrete_sublist_label``): its
    leading enumerators are authoritative even below ``_is_enumerated_section``'s
    ≥3 sequential-run floor, and the collapse-split is suppressed, so a 1- or
    2-entry sub-list is never shredded (v1 ``opened_by_heading``,
    bib_parser.py:933-936).
    """
    cleaned: list[str] = []
    for raw_line in lines:
        line = _clean_entry_line(_apply_ocr_fixes(raw_line))
        if not line.strip():
            continue
        if _is_debris(line):
            continue
        cleaned.append(line)

    # A qualified discrete sub-list's enumerators are trusted even below the
    # ≥3 sequential-run floor (a 1- or 2-entry discussion section).
    has_enum = discrete_list and any(
        _leading_enum_number(ln) is not None for ln in cleaned
    )
    enumerated = has_enum or _is_enumerated_section(cleaned)

    entries: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            joined = re.sub(r"\s+", " ", " ".join(s.strip() for s in buf if s.strip()))
            joined = joined.strip()
            if joined:
                entries.append(joined)
            buf.clear()

    seen_first_enum = False
    for line in cleaned:
        if enumerated:
            is_start = bool(_ENUM_RE.match(line) or _EMDASH_CONT_RE.match(line))
            if is_start and not seen_first_enum:
                # Drop preamble accumulated before the first enumerator.
                buf.clear()
                seen_first_enum = True
        else:
            is_start = _looks_like_entry_start(line)
        if is_start and buf:
            flush()
        buf.append(line)
    flush()

    if enumerated or discrete_list:
        # Enumerator boundaries are authoritative — never author-comma-split. A
        # discrete (qualified-heading) sub-list likewise suppresses the
        # collapse-split so a short name-year sub-list is not shredded into
        # author fragments. Miss direction is asymmetric-benign (design 10): an
        # unlisted PLAIN variant (e.g. "Sources") lands here and merely keeps a
        # collapsed multi-entry line unsplit — it never shreds.
        return entries

    split_entries: list[str] = []
    for entry in entries:
        for piece in _COLLAPSE_SPLIT_RE.split(entry):
            piece = piece.strip()
            if piece:
                split_entries.append(piece)
    return split_entries


def _entry_from_raw(raw: str, *, section_label: str, ordinal: int, prev_author: Optional[str]) -> BibEntry:
    """Build one :class:`BibEntry` from a reflowed raw line + the previous author."""
    continued = bool(_EMDASH_CONT_RE.match(raw))
    body = _EMDASH_CONT_RE.sub("", raw, count=1) if continued else raw

    first_author = _extract_first_author(body)
    if continued and first_author is None:
        first_author = prev_author  # carry the previous author forward

    nb = _ENUM_RE.sub("", body).strip()
    is_numbered_style = bool(_ENUM_RE.match(body)) and not _AUTHOR_YEAR_START_RE.match(nb)
    title = _extract_title_numbered(body) if is_numbered_style else _extract_title(body)

    return BibEntry(
        raw=raw,
        section_label=section_label,
        ordinal=ordinal,
        first_author=first_author,
        year=_extract_year(body),
        title=title,
        doi=_extract_doi(body),
        arxiv=_extract_arxiv(body),
        continued_author=continued and first_author is not None,
    )


def parse_references(
    markdown: str, ref_ranges: list[tuple[int, int, str]]
) -> list[BibEntry]:
    """Parse the references region(s) of ``markdown`` into a :class:`BibEntry` list.

    ``ref_ranges`` is ``[(start_char, end_char, section_label)]`` taken from the
    ``document_sections`` rows where ``section_kind='references'`` (phase_3). For
    each range this reflows wrapped/collapsed lines, splits into one entry per cited
    work (numbered ``[1]…[n]`` lists are not author-comma-shredded; a QUALIFIED
    discrete sub-list — a non-plain label like "Rejoinder References", per
    ``_is_discrete_sublist_label`` — trusts its enumerators below the ≥3 floor and
    suppresses the collapse-split, so a 1- or 2-entry discussion/rejoinder
    sub-section is never shredded), repairs em-dash
    repeated-author runs (carrying the prior author forward, ``continued_author=True``,
    never across a range boundary), applies v1's OCR fixes, and extracts
    ``(first_author, year, title, doi, arxiv)`` best-effort. ``section_label`` is
    carried onto every entry from its range; ``ordinal`` is the 1-based entry index
    within the range.

    Returns ``[]`` when ``ref_ranges`` is empty (no references section) — this
    function **never** scans for headings itself (decisions 31/52/66/68). PURE: no
    network, no DB.
    """
    out: list[BibEntry] = []
    for start, end, section_label in ref_ranges:
        block = markdown[start:end]
        prev_author: Optional[str] = None  # reset per range (no cross-bleed)
        ordinal = 0
        for raw in _reflow(
            block.splitlines(),
            discrete_list=_is_discrete_sublist_label(section_label),
        ):
            ordinal += 1
            entry = _entry_from_raw(
                raw, section_label=section_label, ordinal=ordinal, prev_author=prev_author
            )
            out.append(entry)
            if entry.first_author is not None:
                prev_author = entry.first_author
    return out
