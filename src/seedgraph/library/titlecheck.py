"""The one shared title matcher (design §7.1–7.2). Pure.

Callers give a title, an optional DOI and the text of a paper's first pages and get a
:class:`PdfCheck` verdict: ``pass`` / ``weak`` / ``miss`` / ``unknown``. The
migration, the standing PDF check, "check these" evidence and later library-wide checks
all use this module, so the thresholds live here as named constants.

Matching never rewrites text. Marker output on some PDFs maps ligature glyphs to
Latin-1 lookalikes (``speciÖcation``, ``e¢ ciency``, ``di¤erence``, ``…rst``); those
characters are also real letters or symbols, so both the page text and the title are
matched as a *variant set* (original + folded copy) and the best score wins.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from seedgraph.citation.resolve import _OVERLAP_STOPWORDS
from seedgraph.project.identity import normalize_id

Status = Literal["pass", "weak", "miss", "unknown"]
Via = Literal["doi", "exact", "tolerant"]

# --- Thresholds (calibrated, see the S2 calibration report in the v3 home under
# migration/calibration/). Scores are the fraction of the title's content tokens found
# inside one window of the first pages.
#: At or above this the title is considered found (``pass`` via ``tolerant``).
PASS_THRESHOLD = 0.80
#: Below this the PDF is a ``miss`` (attention item, notes hidden); between the two it is
#: ``weak`` (reported only). Calibrated on all v2 held works (S2): the two known mix-ups
#: score 0.25 (1 of 4 and 2 of 8 content tokens). Every case scoring 0.25–0.39 was a wrong
#: or partial PDF (an online appendix, a publisher advertisement, a different document),
#: and the lowest right-PDF score was 0.50 (garbled font text). 0.40 sits in that gap: a
#: 4-token title now needs 2 hits to be ``weak``, an 8-token title 4 hits. The design's
#: provisional 0.30 left the mix-ups one token away and let the bad PDFs through as weak.
MISS_THRESHOLD = 0.40
#: A title with this many content tokens or fewer needs every one of them to ``pass``.
SHORT_TITLE_TOKENS = 3
#: Window = WINDOW_FACTOR * |T| + WINDOW_SLACK page tokens.
WINDOW_FACTOR = 2
WINDOW_SLACK = 10
#: Pages taken from the start of a document.
FIRST_PAGES = 3
#: Characters used when markdown carries no Marker page markers.
NO_MARKER_CHARS = 8000

# --- Mojibake fold table. Each entry maps a lookalike glyph to the ligature it stands for
# in v2 Marker output. A fold applies only where the glyph touches a lowercase ASCII letter
# (the glyphs are real symbols elsewhere: names with Ö, an ellipsis, a cent sign). Evidence
# from v2 markdown: ``Ö``→fi (speciÖcation, Örst), ``…``→fi (…rst, in…nity), ``¢ ?``→ffi
# (e¢ ciency, coe¢ cient, di¢ cult), ``¤``→ff (di¤erence, e¤ect). ``Þ`` (a closing paren in
# math) and ``ß`` (German) were checked and are NOT folded. Extend only with evidence.
_FOLDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<=[a-z])¢ ?(?=[a-z])|(?<=[a-z])¢|¢ ?(?=[a-z])"), "ffi"),
    (re.compile(r"(?<=[a-z])¤|¤(?=[a-z])"), "ff"),
    (re.compile(r"(?<=[a-z])Ö|Ö(?=[a-z])"), "fi"),
    (re.compile(r"(?<=[a-z])…|…(?=[a-z])"), "fi"),
)

_SUP_RE = re.compile(r"<sup>.*?</sup>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]{0,200}>")
_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_HYPHEN_BREAK_RE = re.compile(r"-[ \t]*\n\s*")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_INTRA_HYPHEN_RE = re.compile(r"(?<=[A-Za-z])-(?=[A-Za-z])")
#: Marker ``paginate_output`` delimiter, as in :mod:`seedgraph.segment` (0-based index).
_PAGE_DELIM = re.compile(r"(?m)^[ \t]*\{(\d+)\}-{3,}[ \t]*$")


@dataclass(frozen=True)
class PdfCheck:
    """Verdict of :func:`check_pdf_text`.

    ``score`` is 1.0 for a DOI or exact hit, else the best window coverage (0.0 for
    ``unknown``). ``via`` is set for ``pass`` (and holds ``tolerant`` for weak/miss
    scores). ``evidence`` is a short human-readable note (which variant, which DOI,
    the matched tokens).
    """

    status: Status
    score: float
    via: Via | None
    evidence: str


# --- Normalisation ------------------------------------------------------------------


def _strip_markup(text: str) -> str:
    text = _SUP_RE.sub(" ", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _TAG_RE.sub(" ", text)
    return text.replace("**", " ").replace("#", " ")


def normalize_tokens(text: str) -> list[str]:
    """Design §7.2 normalisation steps 1–5: tokens of ``text`` (no variant folding)."""
    text = unicodedata.normalize("NFKC", text)
    text = _strip_markup(text)
    text = _HYPHEN_BREAK_RE.sub("", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _NON_ALNUM_RE.sub(" ", text.lower()).split()


def _fold(text: str) -> str:
    for pattern, repl in _FOLDS:
        text = pattern.sub(repl, text)
    return text


def _labelled_variants(text: str) -> list[tuple[str, str]]:
    folded = _fold(text)
    out = [("original", text)]
    for label, candidate in (("folded", folded), ("hyphen-closed", _INTRA_HYPHEN_RE.sub("", folded))):
        if all(candidate != t for _, t in out):
            out.append((label, candidate))
    return out


def variants(text: str) -> list[str]:
    """The variant set of ``text``: the original, the mojibake-folded copy, and that copy
    with intra-word hyphens closed (``Non-Parametric`` ~ ``Nonparametric``; v2 evidence),
    each only when it differs from the previous ones."""
    return [t for _, t in _labelled_variants(text)]


def content_tokens(tokens: Sequence[str]) -> list[str]:
    """Distinct content tokens in first-seen order (``citation.resolve`` rules: stopwords
    dropped, length > 1, alphabetic-bearing; applied to the normalised stream)."""
    seen: dict[str, None] = {}
    for tok in tokens:
        if len(tok) > 1 and tok not in _OVERLAP_STOPWORDS and not tok.isdigit():
            seen.setdefault(tok, None)
    return list(seen)


# --- Scoring --------------------------------------------------------------------------


def _contains_run(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    first = needle[0]
    for i in range(len(haystack) - n + 1):
        if haystack[i] == first and list(haystack[i : i + n]) == list(needle):
            return True
    return False


def window_score(title_content: Sequence[str], page_tokens: Sequence[str]) -> tuple[float, list[str]]:
    """Best fraction of ``title_content`` covered inside one window of page tokens.

    The window is ``WINDOW_FACTOR * |T| + WINDOW_SLACK`` tokens, so a long first page
    cannot match a title out of scattered words. Returns ``(score, matched tokens)``.
    """
    targets = set(title_content)
    if not targets or not page_tokens:
        return 0.0, []
    size = WINDOW_FACTOR * len(targets) + WINDOW_SLACK
    counts: dict[str, int] = {}
    best: set[str] = set()
    covered = 0
    for i, tok in enumerate(page_tokens):
        if tok in targets:
            counts[tok] = counts.get(tok, 0) + 1
            if counts[tok] == 1:
                covered += 1
        if i >= size:
            old = page_tokens[i - size]
            if old in targets:
                counts[old] -= 1
                if counts[old] == 0:
                    covered -= 1
        if covered > len(best):
            best = {t for t, c in counts.items() if c > 0}
            if covered == len(targets):
                break
    return len(best) / len(targets), sorted(best)


def _flatten(pages: Sequence[str]) -> tuple[str, frozenset[int]]:
    """The pages lower-cased with whitespace removed, and the positions where whitespace was."""
    chars: list[str] = []
    gaps: set[int] = set()
    for ch in "\n".join(pages).lower().replace("\\", ""):
        if ch.isspace():
            gaps.add(len(chars))
        else:
            chars.append(ch)
    return "".join(chars), frozenset(gaps)


def _doi_in_text(doi: str, pages: Sequence[str], flat_gaps: tuple[str, frozenset[int]] | None = None) -> bool:
    # Whitespace is stripped (a DOI may wrap across lines), but a removed-whitespace
    # position still counts as a boundary, so ``10.1/ab.12`` does not hit ``10.1/ab.123``
    # while ``10.1/ab.12 Next`` does.
    flat, gaps = flat_gaps if flat_gaps is not None else _flatten(pages)
    start = flat.find(doi)
    while start != -1:
        end = start + len(doi)
        if end >= len(flat) or not flat[end].isalnum() or end in gaps:
            return True
        start = flat.find(doi, start + 1)
    return False


def _status_for(score: float, n_content: int) -> Status:
    if n_content <= SHORT_TITLE_TOKENS:
        if score >= 1.0:
            return "pass"
    elif score >= PASS_THRESHOLD:
        return "pass"
    return "weak" if score >= MISS_THRESHOLD else "miss"


@dataclass(frozen=True)
class PreparedPages:
    """First pages normalised once (:func:`prepare_pages`), so many titles can be checked
    against one PDF (the "which paper is this PDF?" suggestions) without re-normalising."""

    pages: tuple[str, ...]
    variants: tuple[tuple[str, tuple[str, ...]], ...]  # (variant label, page tokens)
    tokens: frozenset[str]  # every token of every variant
    flat: str = ""  # lower-cased, whitespace removed (the DOI search)
    gaps: frozenset[int] = frozenset()

    def may_pass(self, title: str | None, doi: str | None) -> bool:
        """A cheap necessary condition for :func:`check_prepared` to reach ``weak`` or better:
        the DOI's text is on the pages, or enough of the title's content tokens are (the
        window score can never exceed the fraction found anywhere)."""
        norm_doi = normalize_id("doi", doi) if doi else None
        if norm_doi and norm_doi in self.flat:
            return True
        if not title:
            return False
        for v in variants(title):
            content = set(content_tokens(normalize_tokens(v)))
            if content and len(content & self.tokens) >= MISS_THRESHOLD * len(content):
                return True
        return False


def prepare_pages(pages: Sequence[str]) -> PreparedPages | None:
    """The normalised variants of ``pages``, or ``None`` when there is no text at all."""
    if not pages or not any(p.strip() for p in pages):
        return None
    text = "\n".join(pages)
    variants_ = tuple((label, tuple(normalize_tokens(v))) for label, v in _labelled_variants(text))
    flat, gaps = _flatten(pages)
    return PreparedPages(tuple(pages), variants_, frozenset(t for _, toks in variants_ for t in toks), flat, gaps)


def check_pdf_text(title: str | None, doi: str | None, pages: Sequence[str]) -> PdfCheck:
    """Does the text of these first pages belong to the paper titled ``title``?

    Order: DOI hit → ``pass`` via ``doi``; contiguous normalised title → ``pass`` via
    ``exact``; else window coverage of the title's content tokens (§7.2). No text at all,
    or no title, gives ``unknown`` (never a miss).
    """
    prepared = prepare_pages(pages)
    if prepared is None:
        return PdfCheck("unknown", 0.0, None, "no text")
    return check_prepared(title, doi, prepared)


def check_prepared(title: str | None, doi: str | None, prepared: PreparedPages) -> PdfCheck:
    """:func:`check_pdf_text` on pages already normalised by :func:`prepare_pages`."""
    pages = prepared.pages
    norm_doi = normalize_id("doi", doi) if doi else None
    if norm_doi and _doi_in_text(norm_doi, pages, (prepared.flat, prepared.gaps)):
        return PdfCheck("pass", 1.0, "doi", f"doi {norm_doi} in text")
    if not title or not normalize_tokens(title):
        return PdfCheck("unknown", 0.0, None, "no title")

    page_variants = prepared.variants
    title_variants = [(label, normalize_tokens(v)) for label, v in _labelled_variants(title)]

    for ti, ttoks in title_variants:
        for pi, ptoks in page_variants:
            if _contains_run(ptoks, ttoks):
                return PdfCheck("pass", 1.0, "exact", _variant_note(ti, pi, "exact title"))

    best: tuple[float, str, str, list[str], int] | None = None
    for ti, ttoks in title_variants:
        tcontent = content_tokens(ttoks)
        if not tcontent:
            continue
        for pi, ptoks in page_variants:
            score, hits = window_score(tcontent, ptoks)
            if best is None or score > best[0]:
                best = (score, ti, pi, hits, len(tcontent))
    if best is None:
        return PdfCheck("unknown", 0.0, None, "title has no content tokens")
    score, ti, pi, hits, n_content = best
    status = _status_for(score, n_content)
    note = _variant_note(ti, pi, f"{len(hits)}/{n_content} title tokens: {' '.join(hits)}")
    return PdfCheck(status, round(score, 4), "tolerant", note)


def _variant_note(title_variant: str, page_variant: str, what: str) -> str:
    # Names the non-original variants that produced the match (see :func:`variants`).
    used = [f"{side} {label}" for side, label in (("title", title_variant), ("page", page_variant))
            if label != "original"]
    return what + (f" ({', '.join(used)})" if used else "")


def title_similarity(a: str | None, b: str | None) -> float:
    """Symmetric title-vs-title similarity on the same normalisation and variants:
    the best Jaccard overlap of content tokens (0.0 when either side has none). Used as
    "check these" evidence (§4.3)."""
    if not a or not b:
        return 0.0
    best = 0.0
    for va in variants(a):
        ta = set(content_tokens(normalize_tokens(va)))
        for vb in variants(b):
            tb = set(content_tokens(normalize_tokens(vb)))
            if ta and tb:
                best = max(best, len(ta & tb) / len(ta | tb))
    return round(best, 4)


# --- First pages ----------------------------------------------------------------------


def first_pages_from_markdown(md: str, n: int = FIRST_PAGES) -> list[str]:
    """The first ``n`` pages of Marker markdown, split on page markers.

    Pages are selected by their 0-based marker number (``{0}``…``{n-1}``); any text
    before the first marker belongs to the first page. Without markers (external imports)
    the first ``NO_MARKER_CHARS`` characters are one page. Empty markdown gives ``[]``.
    """
    if not md or not md.strip():
        return []
    marks = list(_PAGE_DELIM.finditer(md))
    if not marks:
        return [md[:NO_MARKER_CHARS]]
    preamble = md[: marks[0].start()]
    pages: list[str] = []
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(md)
        if int(m.group(1)) >= n:
            continue
        body = md[m.end() : end]
        if not pages and preamble.strip():
            body = preamble + "\n" + body
        pages.append(body)
    return pages


def first_pages_from_pdf(pdf_path: Path, n: int = FIRST_PAGES) -> list[str] | None:
    """Text layer of the first ``n`` PDF pages via pypdfium2, or ``None`` if the file
    cannot be opened. A scan with no text layer gives empty strings (→ ``unknown``)."""
    import pypdfium2 as pdfium  # lazy: keep the module import pure and light

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
    except Exception:  # noqa: BLE001 - unreadable/encrypted PDFs are simply "no text"
        return None
    try:
        out: list[str] = []
        for i in range(min(n, len(pdf))):
            page = pdf[i]
            textpage = page.get_textpage()
            try:
                out.append(textpage.get_text_range())
            finally:
                textpage.close()
                page.close()
        return out
    finally:
        pdf.close()
