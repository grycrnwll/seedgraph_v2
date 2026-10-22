"""Per-PDF golden-set comparison of an OLD and a NEW markdown (pure).

Pass bar (SPEC stage 0 / ticket 12 #5); metric definitions live in
:mod:`seedgraph.golden.metrics`.

=============  ==============================================================
check          rule
=============  ==============================================================
page_markers   NEW markers are exactly ``0..n-1`` with ``n`` = PDF page count.
               Page count unknown -> WARN (contiguity still reported).
references     NEW has a section classified ``references``. Missing in NEW but
               present in OLD -> FAIL; missing in both -> WARN (a baseline gap
               of that paper's v2 markdown, which no converter change caused).
quotes         Fraction of the paper's current-note quotes that re-anchor
               uniquely (:func:`seedgraph.anchor.reanchor`). FAIL when NEW is
               more than 1 percentage point below OLD. No quotes -> WARN.
equations      FAIL when NEW has more non-parsing math spans than OLD, or the
               total math-span count falls by more than 2% (skipped when OLD
               has none).
tables         FAIL when NEW table count < OLD.
speed          Seconds per PDF page; FAIL when NEW > 1.25 x OLD. Missing timing
               -> WARN. The ratio NEW/OLD is reported in ``extra``.
similarity     Normalized text similarity; WARN below 0.98, never FAIL.
=============  ==============================================================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from seedgraph.anchor import reanchor

from . import metrics

QUOTE_TOLERANCE = 0.01
MATH_TOTAL_TOLERANCE = 0.02
SPEED_FACTOR = 1.25
SIMILARITY_WARN_BELOW = 0.98


@dataclass
class Rendition:
    """One markdown conversion of a PDF plus whatever timing is known."""

    markdown: str
    seconds: Optional[float] = None
    timing_label: str = ""
    converter_version: Optional[str] = None
    page_count: Optional[int] = None  # as reported by the converter, if any


@dataclass
class CheckResult:
    name: str
    old: Any
    new: Any
    status: str  # pass | fail | warn
    reason: str
    extra: dict = field(default_factory=dict)


@dataclass
class PdfResult:
    golden_id: str
    checks: list[CheckResult]
    meta: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        states = {c.status for c in self.checks}
        if "fail" in states:
            return "fail"
        if "warn" in states:
            return "warn"
        return "pass"

    def check(self, name: str) -> CheckResult:
        return next(c for c in self.checks if c.name == name)

    def to_dict(self) -> dict:
        return {
            "id": self.golden_id,
            "status": self.status,
            "meta": dict(self.meta),
            "checks": [asdict(c) for c in self.checks],
        }


def _page_markers(old: Rendition, new: Rendition, page_count: Optional[int]) -> CheckResult:
    old_m, new_m = metrics.page_markers(old.markdown), metrics.page_markers(new.markdown)
    contiguous = new_m == list(range(len(new_m)))
    extra = {"page_count": page_count, "new_contiguous": contiguous}
    if page_count is None:
        return CheckResult("page_markers", len(old_m), len(new_m), "warn",
                           "PDF page count unknown; contiguity " + ("ok" if contiguous else "BROKEN"), extra)
    if new_m == list(range(page_count)):
        return CheckResult("page_markers", len(old_m), len(new_m), "pass",
                           f"{page_count} markers, contiguous 0..{page_count - 1}", extra)
    why = []
    if len(new_m) != page_count:
        why.append(f"{len(new_m)} markers != {page_count} PDF pages")
    if not contiguous:
        why.append("markers not contiguous 0..n-1")
    return CheckResult("page_markers", len(old_m), len(new_m), "fail", "; ".join(why), extra)


def _references(old: Rendition, new: Rendition) -> CheckResult:
    o, n = metrics.has_references(old.markdown), metrics.has_references(new.markdown)
    if n:
        return CheckResult("references", o, n, "pass", "References section found")
    if not o:
        return CheckResult("references", o, n, "warn",
                           "no References section in NEW nor OLD (baseline gap, not a regression)")
    return CheckResult("references", o, n, "fail", "no References section in NEW (OLD had one)")


def _match_rate(markdown: str, quotes: Sequence[str]) -> float:
    return sum(1 for q in quotes if reanchor(q, markdown) is not None) / len(quotes)


def _quotes(old: Rendition, new: Rendition, quotes: Sequence[str]) -> CheckResult:
    if not quotes:
        return CheckResult("quotes", None, None, "warn", "no current-note quotes for this paper")
    o, n = _match_rate(old.markdown, quotes), _match_rate(new.markdown, quotes)
    extra = {"n_quotes": len(quotes)}
    if n < o - QUOTE_TOLERANCE - 1e-12:
        return CheckResult("quotes", o, n, "fail",
                           f"NEW re-match {n:.1%} is more than 1 pt below OLD {o:.1%}", extra)
    return CheckResult("quotes", o, n, "pass", f"{n:.1%} vs {o:.1%} of {len(quotes)} quotes", extra)


def _equations(old: Rendition, new: Rendition) -> CheckResult:
    o, n = metrics.math_stats(old.markdown), metrics.math_stats(new.markdown)
    extra = {
        "old": {"display": o.display, "inline": o.inline, "non_parsing": o.non_parsing},
        "new": {"display": n.display, "inline": n.inline, "non_parsing": n.non_parsing},
    }
    why = []
    if n.non_parsing > o.non_parsing:
        why.append(f"non-parsing spans rose {o.non_parsing} -> {n.non_parsing}")
    if o.total > 0 and n.total < o.total * (1 - MATH_TOTAL_TOLERANCE):
        why.append(f"math spans fell {o.total} -> {n.total} (>2%)")
    if why:
        return CheckResult("equations", o.total, n.total, "fail", "; ".join(why), extra)
    return CheckResult("equations", o.total, n.total, "pass",
                       f"{n.total} spans ({n.non_parsing} non-parsing) vs {o.total} ({o.non_parsing})", extra)


def _tables(old: Rendition, new: Rendition) -> CheckResult:
    o, n = metrics.count_tables(old.markdown), metrics.count_tables(new.markdown)
    if n < o:
        return CheckResult("tables", o, n, "fail", f"table count fell {o} -> {n}")
    return CheckResult("tables", o, n, "pass", f"{n} tables vs {o}")


def _speed(old: Rendition, new: Rendition, page_count: Optional[int]) -> CheckResult:
    def per_page(r: Rendition) -> Optional[float]:
        pages = page_count or len(metrics.page_markers(r.markdown))
        if r.seconds is None or not pages:
            return None
        return r.seconds / pages

    o, n = per_page(old), per_page(new)
    extra = {"old_timing": old.timing_label, "new_timing": new.timing_label}
    if o is None or n is None:
        return CheckResult("speed", o, n, "warn", "timing unavailable for "
                           + ("OLD" if o is None else "NEW"), extra)
    ratio = n / o if o else float("inf")
    extra["ratio"] = ratio
    if n > SPEED_FACTOR * o:
        return CheckResult("speed", o, n, "fail", f"{n:.2f} s/page is {ratio:.2f}x OLD {o:.2f} (>1.25x)", extra)
    return CheckResult("speed", o, n, "pass", f"{n:.2f} s/page = {ratio:.2f}x OLD {o:.2f}", extra)


def _similarity(old: Rendition, new: Rendition) -> CheckResult:
    ratio, sampled = metrics.similarity_detail(old.markdown, new.markdown)
    extra = {"sampled": sampled}
    note = " (sampled)" if sampled else ""
    if ratio < SIMILARITY_WARN_BELOW:
        return CheckResult("similarity", 1.0, ratio, "warn", f"text similarity {ratio:.4f}{note} (warning only)", extra)
    return CheckResult("similarity", 1.0, ratio, "pass", f"text similarity {ratio:.4f}{note}", extra)


def compare(
    golden_id: str,
    old: Rendition,
    new: Rendition,
    *,
    page_count: Optional[int],
    quotes: Sequence[str],
) -> PdfResult:
    """Run every check for one PDF and return the structured result."""
    checks = [
        _page_markers(old, new, page_count),
        _references(old, new),
        _quotes(old, new, quotes),
        _equations(old, new),
        _tables(old, new),
        _speed(old, new, page_count),
        _similarity(old, new),
    ]
    meta = {
        "page_count": page_count,
        "old_converter_version": old.converter_version,
        "new_converter_version": new.converter_version,
        "old_timing": old.timing_label,
        "new_timing": new.timing_label,
    }
    return PdfResult(golden_id, checks, meta)
