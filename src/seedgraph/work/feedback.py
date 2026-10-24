"""Failed-quote feedback (design §1.5): explain why a quote did not anchor and hand the
agent the exact markdown text to copy.

Pure. The anchor verdict always comes from the unchanged exact-unique matcher
(``anchor.locate_quote``); nothing here loosens it. A diagnosis only explains a failure:
each fold (lookalike characters, whitespace, mojibake) is tried on both sides, and when the
folded quote occurs exactly once in the folded markdown, the matching *original* markdown
slice is returned. That slice anchors with ``locate_quote`` whenever it is unique in the
markdown.
"""

from __future__ import annotations

import difflib
import functools
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from seedgraph.anchor import locate_quote
from seedgraph.library import mojibake

NOT_FOUND = "not_found"
MULTIPLE_MATCHES = "multiple_matches"
LOOKALIKE_CHARACTERS = "lookalike_characters"
WHITESPACE_OR_LINEBREAK = "whitespace_or_linebreak"
MOJIBAKE = "mojibake"
IN_REFERENCES = "in_references"
EMPTY = "empty"
REASONS = (
    NOT_FOUND,
    MULTIPLE_MATCHES,
    LOOKALIKE_CHARACTERS,
    WHITESPACE_OR_LINEBREAK,
    MOJIBAKE,
    IN_REFERENCES,
    EMPTY,
)

#: Share of the quote's word tokens a passage must contain to be offered as "nearest".
NEAREST_MIN_OVERLAP = 0.6

# Explicit confusables beyond what NFKD folds (NFKD already maps µ→μ, ligatures, NBSP,
# thin and other width spaces).
_CONFUSABLES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
}

Spans = list[tuple[int, int]]


@dataclass(frozen=True)
class CodePointDifference:
    text: str  # "U+03BC" (in the markdown) — empty when the quote has an extra character
    quote: str  # "U+00B5" (in the quote) — empty when the quote lacks a character

    def describe(self) -> str:
        return f"{self.text or 'nothing'} in text, {self.quote or 'nothing'} in quote"


@dataclass(frozen=True)
class QuoteDiagnosis:
    anchored: bool  # True exactly when locate_quote(markdown, quote) is not None
    reason: str | None  # one of REASONS, or None for a clean anchor
    hint: str
    markdown_slice: str | None = None
    span: tuple[int, int] | None = None
    match_count: int = 0
    differences: tuple[CodePointDifference, ...] = field(default_factory=tuple)
    overlap: float | None = None

    def as_dict(self) -> dict:
        return {
            "anchored": self.anchored,
            "reason": self.reason,
            "hint": self.hint,
            "markdown_slice": self.markdown_slice,
            "span": list(self.span) if self.span else None,
            "match_count": self.match_count,
            "differences": [{"text": d.text, "quote": d.quote} for d in self.differences],
            "overlap": self.overlap,
        }


# --- folds -----------------------------------------------------------------------------
# Each fold maps text -> (folded, spans), spans[i] = source (start, end) of folded char i.


def _identity(text: str) -> tuple[str, Spans]:
    return text, [(i, i + 1) for i in range(len(text))]


def _compose(*folds: Callable[[str], tuple[str, Spans]]) -> Callable[[str], tuple[str, Spans]]:
    def run(text: str) -> tuple[str, Spans]:
        cur, spans = _identity(text)
        for f in folds:
            nxt, inner = f(cur)
            spans = [(spans[a][0], spans[b - 1][1]) for a, b in inner]
            cur = nxt
        return cur, spans

    return run


def _lookalike_fold(text: str) -> tuple[str, Spans]:
    out: list[str] = []
    spans: Spans = []
    for i, ch in enumerate(text):
        rep = _CONFUSABLES.get(ch)
        if rep is None:
            rep = unicodedata.normalize("NFKD", ch)
            rep = "".join(_CONFUSABLES.get(c, c) for c in rep)
        for c in rep:
            out.append(c)
            spans.append((i, i + 1))
    return "".join(out), spans


_HYPHEN_BREAK_RE = re.compile(r"-[ \t]*\r?\n\s*")
_WS_RE = re.compile(r"\s+")


def _whitespace_fold(text: str) -> tuple[str, Spans]:
    """Join ``-\\n`` hyphenation, then collapse whitespace runs to one space."""
    out: list[str] = []
    spans: Spans = []
    cursor = 0
    pattern = re.compile(f"(?P<hy>{_HYPHEN_BREAK_RE.pattern})|(?P<ws>{_WS_RE.pattern})")
    for m in pattern.finditer(text):
        for i in range(cursor, m.start()):
            out.append(text[i])
            spans.append((i, i + 1))
        if m.lastgroup == "ws":
            out.append(" ")
            spans.append((m.start(), m.end()))
        elif spans:  # hyphen break: drop it, widen the previous char to cover it
            spans[-1] = (spans[-1][0], m.end())
        cursor = m.end()
    for i in range(cursor, len(text)):
        out.append(text[i])
        spans.append((i, i + 1))
    return "".join(out), spans


def _strip(folded: str, spans: Spans) -> tuple[str, Spans]:
    """Trim folded leading/trailing spaces (a quote's edges are not meaningful)."""
    a, b = 0, len(folded)
    while a < b and folded[a] == " ":
        a += 1
    while b > a and folded[b - 1] == " ":
        b -= 1
    return folded[a:b], spans[a:b]


_FOLDS: dict[str, Callable[[str], tuple[str, Spans]]] = {
    "lookalike": _lookalike_fold,
    "whitespace": _whitespace_fold,
    "both": _compose(_lookalike_fold, _whitespace_fold),
    "mojibake": _compose(mojibake.fold_with_map, _lookalike_fold, _whitespace_fold),
}


@functools.lru_cache(maxsize=16)
def _folded_markdown(fold_name: str, markdown: str) -> tuple[str, Spans]:
    """The folded markdown, memoised: one submit diagnoses many quotes against one text."""
    return _FOLDS[fold_name](markdown)


def _count_all(haystack: str, needle: str) -> int:
    """Occurrences of ``needle``, overlapping ones included (as ``_find_unique`` counts)."""
    n, i = 0, haystack.find(needle)
    while i != -1:
        n += 1
        i = haystack.find(needle, i + 1)
    return n


def _differences(slice_: str, quote: str) -> tuple[CodePointDifference, ...]:
    """Differing non-whitespace code points between the markdown slice and the quote."""
    diffs: list[CodePointDifference] = []
    sm = difflib.SequenceMatcher(a=slice_, b=quote, autojunk=False)
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        if op == "equal":
            continue
        a_chars = [c for c in slice_[a1:a2] if not c.isspace()]
        b_chars = [c for c in quote[b1:b2] if not c.isspace()]
        for k in range(max(len(a_chars), len(b_chars))):
            ta = f"U+{ord(a_chars[k]):04X}" if k < len(a_chars) else ""
            tb = f"U+{ord(b_chars[k]):04X}" if k < len(b_chars) else ""
            if ta != tb:
                diffs.append(CodePointDifference(ta, tb))
    return tuple(diffs)


# --- nearest passage -------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+")


def _nearest_passage(quote: str, markdown: str) -> tuple[tuple[int, int], float] | None:
    """The window of markdown tokens (as long as the quote) sharing the most tokens with
    the quote, as ``((start, end), overlap share)``; None when nothing scores well."""
    q_tokens = [t.lower() for t in _TOKEN_RE.findall(quote)]
    if not q_tokens:
        return None
    need = Counter(q_tokens)
    toks = [(m.group(0).lower(), m.start(), m.end()) for m in _TOKEN_RE.finditer(markdown)]
    n = len(q_tokens)
    if not toks:
        return None
    window: Counter = Counter()
    shared = 0
    best = (-1, 0)
    for i, (tok, _, _) in enumerate(toks):
        window[tok] += 1
        if window[tok] <= need.get(tok, 0):
            shared += 1
        if i >= n:
            old = toks[i - n][0]
            if window[old] <= need.get(old, 0):
                shared -= 1
            window[old] -= 1
        if shared > best[0]:
            best = (shared, max(0, i - n + 1))
    score = best[0] / n
    if score < NEAREST_MIN_OVERLAP:
        return None
    first = best[1]
    last = min(first + n, len(toks)) - 1
    # Trim window edges that are not quote tokens.
    while first < last and toks[first][0] not in need:
        first += 1
    while last > first and toks[last][0] not in need:
        last -= 1
    return (toks[first][1], toks[last][2]), round(score, 3)


# --- the diagnosis ---------------------------------------------------------------------


def _in_ranges(span: tuple[int, int], ranges: Iterable[tuple[int, int]]) -> bool:
    return any(int(s) <= span[0] and span[1] <= int(e) for s, e in ranges)


def _fold_diagnosis(
    reason: str, fold_name: str, quote: str, markdown: str
) -> QuoteDiagnosis | None:
    fq, _ = _strip(*_FOLDS[fold_name](quote))
    if not fq:
        return None
    fm, spans = _folded_markdown(fold_name, markdown)
    first = fm.find(fq)
    if first == -1:
        return None
    count = _count_all(fm, fq)
    label = {
        LOOKALIKE_CHARACTERS: "matches only after folding lookalike characters",
        WHITESPACE_OR_LINEBREAK: "matches only after normalising whitespace and line breaks",
        MOJIBAKE: "matches only the garbled (mojibake) text in the markdown",
    }[reason]
    if count > 1:
        return QuoteDiagnosis(
            anchored=False,
            reason=reason,
            hint=f"The quote {label}, but in {count} places; copy the text exactly from "
            "the markdown and extend it until it is unique.",
            match_count=count,
        )
    start = spans[first][0]
    end = spans[first + len(fq) - 1][1]
    slice_ = markdown[start:end]
    diffs = _differences(slice_, quote) if reason == LOOKALIKE_CHARACTERS else ()
    hint = f"The quote {label}. Copy this exact markdown text instead: {slice_!r}"
    if diffs:
        hint += " (" + "; ".join(d.describe() for d in diffs) + ")"
    span = locate_quote(markdown, slice_)
    return QuoteDiagnosis(
        anchored=False,
        reason=reason,
        hint=hint,
        markdown_slice=slice_ if span else None,
        span=span,
        match_count=1,
        differences=diffs,
    )


def diagnose_quote(
    quote: str | None,
    markdown: str,
    *,
    reference_ranges: Iterable[tuple[int, int]] | None = None,
) -> QuoteDiagnosis:
    """Diagnose one ``found`` claim's ``exact_quote`` against the full markdown.

    ``reference_ranges`` are the ``(start, end)`` character ranges of the stripped
    reference list; a match inside one still anchors but carries ``in_references``.
    """
    if quote is None or not quote.strip():
        return QuoteDiagnosis(
            anchored=False,
            reason=EMPTY,
            hint="The claim is found but has no quote: add a verbatim quote or set the "
            "status to ambiguous.",
        )

    span = locate_quote(markdown, quote)
    if span is not None:
        if reference_ranges and _in_ranges(span, reference_ranges):
            return QuoteDiagnosis(
                anchored=True,
                reason=IN_REFERENCES,
                hint="Warning: the quote lies inside the reference list; it still anchors.",
                markdown_slice=quote,
                span=span,
                match_count=1,
            )
        return QuoteDiagnosis(anchored=True, reason=None, hint="", span=span, match_count=1)

    exact = _count_all(markdown, quote)
    if exact >= 2:
        return QuoteDiagnosis(
            anchored=False,
            reason=MULTIPLE_MATCHES,
            hint=f"The quote occurs {exact} times; extend the quote until it is unique.",
            match_count=exact,
        )

    for reason, fold_name in (
        (LOOKALIKE_CHARACTERS, "lookalike"),
        (WHITESPACE_OR_LINEBREAK, "whitespace"),
        (LOOKALIKE_CHARACTERS, "both"),
        (MOJIBAKE, "mojibake"),
    ):
        found = _fold_diagnosis(reason, fold_name, quote, markdown)
        if found is not None:
            return found

    near = _nearest_passage(quote, markdown)
    if near is None:
        return QuoteDiagnosis(
            anchored=False,
            reason=NOT_FOUND,
            hint="The quote is not in the paper's text. Copy a passage verbatim from the "
            "text, or set the status to ambiguous.",
        )
    (a, b), score = near
    slice_ = markdown[a:b]
    ok = locate_quote(markdown, slice_) is not None
    return QuoteDiagnosis(
        anchored=False,
        reason=NOT_FOUND,
        hint="The quote is not in the paper's text. The nearest passage is: "
        f"{slice_!r}. Copy the text verbatim from it if it says what you meant.",
        markdown_slice=slice_ if ok else None,
        span=(a, b) if ok else None,
        overlap=score,
    )
