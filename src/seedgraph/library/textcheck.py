"""The text-quality gate on demand (``seedgraph library textcheck``; design §5.3, §5.6).

- :func:`check_papers` runs (or reads the cached) gate verdict for papers' current
  markdown, reference-stripped by its ``references`` sections, and stores it in
  ``text_quality`` (the cache batch creation reads).
- :func:`check_golden` scores each golden-set entry's markdown and compares the verdict
  with its category: the gate's acceptance is "verdict equals category", so a
  ``garbled`` entry should be held back and every other category should pass.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .store import Library


@dataclass(frozen=True)
class TextCheck:
    paper_id: str
    markdown_id: str | None
    status: str  # 'pass' | 'garbled' | 'no_text' | 'no_markdown'
    summary: str
    cached: bool


@dataclass(frozen=True)
class GoldenCheck:
    golden_id: str
    category: str
    markdown_id: str
    status: str  # the gate verdict, or 'missing' when the markdown is not in the library
    expected: str  # 'garbled' for the garbled category, else 'pass'
    agrees: bool
    summary: str


def _verdict(lib: Library, markdown_id: str, *, force: bool):
    from ..notes.spans import read_markdown  # local: work/notes import the library
    from ..work import packages

    text = read_markdown(lib.conn, markdown_id)
    if text is None:
        return None
    return packages.text_quality(lib, markdown_id, text, packages.reference_ranges(lib.conn, markdown_id),
                                 force=force)


def check_papers(lib: Library, paper_ids: Sequence[str] | None = None, *, force: bool = False) -> list[TextCheck]:
    """The gate verdict for each paper's primary markdown (default: every held paper
    with one, by id). ``force`` re-assesses instead of reading the cache."""
    from ..work.packages import primary_markdown

    if not paper_ids:
        paper_ids = [r[0] for r in lib.conn.execute(
            "SELECT DISTINCT pf.paper_id FROM paper_files pf JOIN paper_status s ON s.paper_id = pf.paper_id "
            "WHERE s.status = 'held' AND pf.markdown_id IS NOT NULL ORDER BY pf.paper_id")]
    out = []
    for pid in paper_ids:
        md = primary_markdown(lib.conn, pid)
        verdict = _verdict(lib, md[0], force=force) if md else None
        if verdict is None:
            out.append(TextCheck(pid, md[0] if md else None, "no_markdown",
                                 "no converted text" if md is None else "markdown file unreadable", False))
        else:
            out.append(TextCheck(pid, md[0], verdict.status, verdict.summary, verdict.cached))
    return out


def _expected(category: str) -> str:
    return "garbled" if category == "garbled" else "pass"


def check_golden(lib: Library, manifest: Iterable[Mapping], *, force: bool = False) -> list[GoldenCheck]:
    """Score every golden-set entry (``id``, ``category``, ``v2_markdown_id``) by its
    markdown in this library and compare the verdict with the category."""
    out = []
    for entry in manifest:
        md_id = str(entry["v2_markdown_id"])
        if not md_id.startswith("md_"):
            md_id = f"md_{md_id}"
        category = str(entry.get("category") or "")
        expected = _expected(category)
        verdict = _verdict(lib, md_id, force=force)
        if verdict is None:
            out.append(GoldenCheck(str(entry["id"]), category, md_id, "missing", expected, False,
                                   "markdown not in this library"))
            continue
        out.append(GoldenCheck(str(entry["id"]), category, md_id, verdict.status, expected,
                               verdict.status == expected, verdict.summary))
    return out
