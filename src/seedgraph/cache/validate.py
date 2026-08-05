"""Lightweight markdown validation (doc 04 §9 / doc 11 §8).

Runs after every successful backend conversion. The cheap checks ship now; the
deeper semantic audit (equations-as-LaTeX, tables readable, theorems preserved)
is explicitly deferred (plan §2).

- **Fatal** (``ok=False`` -> caller sets ``run_status='failed'`` and writes NO
  markdown row/file): empty / whitespace-only output, output below a tiny minimum
  length, or an obvious-garbage heuristic (near-zero ratio of content tokens —
  word-like runs OR math/LaTeX markup — e.g. an OCR failure).
- **Non-fatal warnings** (appended to ``warnings_json``; run stays ``success``):
  no markdown title, no section header, no references/bibliography section.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# Fatal thresholds.
_MIN_LEN = 8  # below this many non-whitespace chars -> too short to be a real conversion
_MIN_CONTENT_RATIO = 0.20  # content-char ratio below this -> obvious garbage / OCR failure
_GARBAGE_RATIO_ENV = "SEEDGRAPH_GARBAGE_RATIO"  # env override for the ratio floor

_WORDLIKE_RE = re.compile(r"[A-Za-z]{2,}")
# Math/LaTeX markup is CONTENT, not garbage (symbol/math-dense papers legitimately
# carry few word-like runs). Inline/display math spans, \( \) \[ \] delimiters,
# \begin{...}/\end{...} environments, and \command markup all count toward content.
_MATH_RE = re.compile(
    r"\$\$.+?\$\$"  # $$ display math $$
    r"|\$[^$\n]+?\$"  # $ inline math $
    r"|\\[()\[\]]"  # \( \) \[ \]
    r"|\\(?:begin|end)\{[^}]*\}"  # \begin{...} / \end{...}
    r"|\\[A-Za-z]+",  # \command  (\frac, \alpha, \sum, ...)
    re.DOTALL,
)
_TITLE_RE = re.compile(r"(?m)^#\s+\S")  # an ATX H1 with content
_SECTION_RE = re.compile(r"(?m)^#{2,6}\s+\S")  # any H2..H6 heading
_REFERENCES_RE = re.compile(
    r"(?im)(^#{1,6}\s*(references|bibliography|works\s+cited)\b)|(\breferences\b)"
)


def _resolve_content_ratio(override: "float | None") -> float:
    """Resolve the garbage-gate ratio floor: explicit ``override`` wins, else the
    ``SEEDGRAPH_GARBAGE_RATIO`` env var (if a valid float), else the default 0.20."""
    if override is not None:
        return override
    raw = os.environ.get(_GARBAGE_RATIO_ENV)
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return _MIN_CONTENT_RATIO


def _content_ratio(stripped: str) -> float:
    """Fraction of characters that are CONTENT — word-like runs OR math/LaTeX markup.
    Spans are unioned (a ``\\frac`` counts once, not once per overlapping matcher)."""
    covered = bytearray(len(stripped))
    for rx in (_WORDLIKE_RE, _MATH_RE):
        for m in rx.finditer(stripped):
            covered[m.start() : m.end()] = b"\x01" * (m.end() - m.start())
    return sum(covered) / len(stripped)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of :func:`validate_markdown`.

    ``ok=False`` means a FATAL signal (empty/garbage) -> the conversion run is
    failed and no markdown is stored. ``warnings`` are non-fatal quality signals
    merged into the run's ``warnings_json`` regardless of ``ok``.
    """

    ok: bool
    warnings: list[str]


def validate_markdown(text: str, *, min_content_ratio: "float | None" = None) -> ValidationResult:
    """Validate converted markdown; see module docstring for the fatal vs. warning
    split. Implements the doc 04 §9 / doc 11 §8 cheap gate.

    FATAL (``ok=False``): empty / whitespace-only output, output below a tiny
    minimum length, or an obvious-garbage heuristic (near-zero ratio of CONTENT
    tokens, e.g. an OCR failure). Content = word-like runs OR math/LaTeX markup, so
    symbol/math-dense papers are not false-rejected. NON-FATAL warnings (``ok=True``):
    no markdown title, no section header, no references/bibliography section.

    The ratio floor (default ``0.20``) is configurable: ``min_content_ratio`` wins,
    else the ``SEEDGRAPH_GARBAGE_RATIO`` env var, else the default.
    """
    stripped = (text or "").strip()
    if not stripped:
        return ValidationResult(ok=False, warnings=["empty or whitespace-only markdown"])
    if len(stripped) < _MIN_LEN:
        return ValidationResult(
            ok=False,
            warnings=[f"markdown below minimum length ({len(stripped)} < {_MIN_LEN} chars)"],
        )
    threshold = _resolve_content_ratio(min_content_ratio)
    ratio = _content_ratio(stripped)
    if ratio < threshold:
        return ValidationResult(
            ok=False,
            warnings=[
                f"obvious-garbage heuristic: content (word/math) char ratio "
                f"{ratio:.2f} < {threshold:.2f}"
            ],
        )

    warnings: list[str] = []
    if not _TITLE_RE.search(text):
        warnings.append("no markdown title detected (no '# ' H1 heading)")
    if not _SECTION_RE.search(text):
        warnings.append("no section header detected (no '## ' heading)")
    if not _REFERENCES_RE.search(text):
        warnings.append("no references/bibliography section detected")
    return ValidationResult(ok=True, warnings=warnings)
