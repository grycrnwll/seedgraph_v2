"""Quote normalization, hashing, slicing, and exact-substring (re)location.

Phase 3 — Evidence Spans. The *verbatim* ``exact_quote`` (``markdown[start:end]``)
is the authoritative unit of record for a span; this module owns the integrity
machinery around it:

* ``normalize_quote`` / ``quote_hash`` — NFC normalization (decision 7: sha256 hex,
  NFC) with a stamped ``NORM_VERSION``. The stored ``exact_quote`` is kept *verbatim*
  (the anchor of record); NFC is applied **only inside the hash**.
* ``slice_quote`` / ``assert_invariant`` — the write-time invariant
  ``markdown[start:end] == exact_quote`` and ``quote_hash == sha256(NFC(quote))``
  (success-criterion §1; doc 09 §5). Offsets are Python ``str`` code-point indices
  into the decoded markdown string, NOT UTF-8 byte offsets — the documented
  producer/consumer contract.
* ``locate_quote`` — unique exact-substring locate used by ``ensure_span`` and
  ``spans create --quote`` (None on zero or >=2 occurrences; never guesses).
* ``reanchor`` — exact-substring relocation of an unchanged quote into re-flowed
  markdown after reconversion (decisions 20/53/83). Exact-substring only; no fuzzy
  (difflib/rapidfuzz) matching in Phase 3.
"""

from __future__ import annotations

import hashlib
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from seedgraph.spans.store import Span

#: Versioned normalization scheme stamped onto every ``quote_hash`` (column
#: ``evidence_spans.norm_version``). Bump when the normalization changes.
NORM_VERSION = "nfc-1"


def normalize_quote(text: str) -> str:
    """Return the NFC (canonical-composition) normalization of ``text``.

    Trivial, obviously-correct helper. Used only to compute ``quote_hash``; the
    stored ``exact_quote`` stays verbatim (anchor of record).
    """
    return unicodedata.normalize("NFC", text)


def quote_hash(exact_quote: str) -> str:
    """``sha256(NFC(exact_quote))`` as lowercase hex (decision 7, ``NORM_VERSION``).

    Idempotent under NFC: ``quote_hash(q) == quote_hash(NFC(q))``.
    """
    return hashlib.sha256(normalize_quote(exact_quote).encode("utf-8")).hexdigest()


def slice_quote(markdown: str, start: int, end: int) -> str:
    """Return the half-open verbatim slice ``markdown[start:end]`` (code-point offsets)."""
    return markdown[start:end]


def assert_invariant(markdown: str, span: "Span") -> None:
    """Assert the write-time span invariant, raising ``ValueError`` on violation.

    Checks both ``markdown[span.start_char:span.end_char] == span.exact_quote`` and
    ``span.quote_hash == quote_hash(span.exact_quote)``. This is the integrity gate
    every span create/verify routes through (success-criterion §1; doc 09 §5).
    """
    actual = markdown[span.start_char:span.end_char]
    if actual != span.exact_quote:
        raise ValueError(
            "span invariant violated: markdown[%d:%d] != exact_quote "
            "(got %r, expected %r)"
            % (span.start_char, span.end_char, actual, span.exact_quote)
        )
    expected_hash = quote_hash(span.exact_quote)
    if span.quote_hash != expected_hash:
        raise ValueError(
            "span invariant violated: quote_hash %r != sha256(NFC(exact_quote)) %r"
            % (span.quote_hash, expected_hash)
        )


def _find_unique(markdown: str, needle: str) -> tuple[int, int] | None:
    """Return the half-open offsets of the UNIQUE exact occurrence of ``needle``.

    ``None`` when ``needle`` is empty, absent, or occurs two-or-more times. The
    search is over the verbatim strings (not NFC-folded), so the located slice
    ``markdown[start:end]`` is byte-for-codepoint identical to ``needle`` — this is
    what preserves the write-time invariant.
    """
    if not needle:
        return None
    first = markdown.find(needle)
    if first == -1:
        return None
    # Any further occurrence (incl. overlapping) makes the match ambiguous.
    if markdown.find(needle, first + 1) != -1:
        return None
    return (first, first + len(needle))


def locate_quote(markdown: str, quote: str) -> tuple[int, int] | None:
    """Locate the UNIQUE exact occurrence of ``quote`` in ``markdown``.

    Returns ``(start_char, end_char)`` half-open code-point offsets when the quote
    occurs exactly once; ``None`` when it occurs zero or two-or-more times (never
    guesses among duplicates — the caller must fall back to explicit ``--start/--end``).
    Backs ``ensure_span`` locate + dedup and ``spans create --quote``.
    """
    return _find_unique(markdown, quote)


def reanchor(old_quote: str, new_markdown: str) -> tuple[int, int] | None:
    """Relocate ``old_quote`` into ``new_markdown`` by UNIQUE exact substring match.

    Returns the new ``(start_char, end_char)`` on a unique exact match, else ``None``
    (zero or ambiguous matches route to ``review_queue`` as ``anchor_status='orphaned'``).
    Exact-substring only — no fuzzy matching in Phase 3 (decisions 20/53/83).
    """
    return _find_unique(new_markdown, old_quote)
