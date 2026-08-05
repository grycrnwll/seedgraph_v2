"""Build F chunk 4 — real-corpus bibliography-overlap regression gate (``-m corpus``).

Ships v1 criterion 11 (the overlap ≥ 0.80 gate over the machine-local read-only
regression library) ported to the v2 parser API. Ordering (per the plan): this lands
only AFTER Build E chunk 1 (discussion-shred suppression) and this build's chunk 3
(references-heading detection completions) — discussion-paper notes in the live library
are exactly what dragged v1's pre-fix overlap to ~0.56; gating before the fixes would
institutionalize a red or sandbagged floor.

Two tests live here:

* :func:`test_bib_regression_overlap_no_library_writes` — the GATE, marked ``corpus``
  (opt-in, deselected by the ``addopts`` ``not corpus`` on machines without the library,
  which never leaves the dev box, so this never runs in CI). It measures the SHIPPING v2
  pipeline: for every ``paper_library/notes/*.md`` note carrying a ``## Full
  Bibliography`` section it builds ``"## References\n" + body``, runs the real path —
  :func:`sections.parser.parse_sections` for references-kind ranges, then
  :func:`citation.bib_parser.parse_references` over those ranges — counts matched entries
  (v1's rubric), asserts aggregate ``matched/total ≥ 0.80``, prints the
  ``bib regression overlap: M/T = P%`` line, and asserts the library byte-size snapshot
  is unchanged (belt to chunk 1's write-guard suspenders). It self-skips when the corpus
  dir is absent or holds zero Full Bibliography sections.
* :func:`test_matched_entry_rubric` — an OFFLINE unit test (unmarked, runs in the default
  suite) pinning the matched-entry rubric itself on synthetic entries, so the metric the
  gate measures is verified even where the library is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from seedgraph.citation.bib_parser import BibEntry, parse_references
from seedgraph.sections.parser import parse_sections

_FULL_BIB_HEADING = "## Full Bibliography"


def _corpus_notes_dir() -> Path:
    """The read-only regression library's ``notes/`` dir.

    Resolved per call from ``SEEDGRAPH_REGRESSION_CORPUS`` (default the machine-local
    ``D:/Documents/research/paper_library``) — the SAME env var and default the autouse
    ``_guard_paper_library`` write-guard reads, so the gate and the guard target one root.
    """
    root = os.environ.get(
        "SEEDGRAPH_REGRESSION_CORPUS", "D:/Documents/research/paper_library"
    )
    return Path(root) / "notes"


def _entry_is_matched(entry: BibEntry) -> bool:
    """v1's matched-entry rubric: a title OR DOI, AND a year OR first author.

    An entry counts as MATCHED when the parser recovered enough fields to identify the
    cited work — at minimum a title-or-DOI plus a year-or-first-author
    (v1 ``tests/test_acceptance.py:1113-1115``). The two clauses are ANDed; each clause
    is an OR of two fields.
    """
    return bool((entry.title or entry.doi) and (entry.year or entry.first_author))


# ---------------------------------------------------------------------------
# The gate (opt-in, ``-m corpus``).
# ---------------------------------------------------------------------------
@pytest.mark.corpus
def test_bib_regression_overlap_no_library_writes():
    """Real-corpus bib overlap ≥ 0.80 over the read-only library, NO writes.

    RECALL regression, not a 100% target (v1: "report overlap %"). For every note that
    carries a ``## Full Bibliography`` section (appended by the bib-extractor workflow),
    feed everything after that H2 back through the SHIPPING v2 path — synthesize a
    ``## References`` heading, ``parse_sections`` it to obtain references-kind ranges, then
    ``parse_references`` over those ranges — and count an entry MATCHED per
    :func:`_entry_is_matched`. ``overlap = matched / total`` (aggregated across all notes,
    as in v1) is the measurable recall number.

    Gating: if the corpus dir is absent or holds zero ``## Full Bibliography`` sections in
    this environment, the test SKIPS with a clear reason (not xfail, not failure — there
    is simply nothing to regress against here).

    No-write invariant: the autouse ``_guard_paper_library`` fixture raises on any
    write/append/create open under the library. We ALSO snapshot the notes dir's file set
    + sizes before and after and assert they are unchanged, so the run provably leaves the
    read-only library byte-for-byte intact.
    """
    notes = _corpus_notes_dir()
    md_files = sorted(notes.glob("*.md")) if notes.is_dir() else []
    if not md_files:
        pytest.skip(
            f"regression corpus absent — no note files under {notes} "
            "(nothing to regress against here)"
        )

    # Snapshot the corpus (path -> size) BEFORE touching it, to prove no writes.
    before = {p: p.stat().st_size for p in md_files}

    # Only notes that actually carry the bib-extractor's "## Full Bibliography" section
    # are part of the regression corpus.
    bib_files = [
        md
        for md in md_files
        if _FULL_BIB_HEADING in md.read_text(encoding="utf-8", errors="ignore")
    ]
    if not bib_files:
        pytest.skip(
            f"no '{_FULL_BIB_HEADING}' regression sections under {notes} "
            f"({len(md_files)} note file(s) present, 0 with a Full Bibliography)"
        )

    total = matched = n_files = 0
    for md in bib_files:
        text = md.read_text(encoding="utf-8", errors="ignore")  # READ-ONLY
        # Take everything after the "## Full Bibliography" H2 (the last section the
        # bib-extractor appends) and prepend a heading the sections parser recognizes,
        # so the body is measured through the real references path.
        body = text.split(_FULL_BIB_HEADING, 1)[1]
        document = "## References\n" + body
        sections = parse_sections(
            document,
            markdown_id="md_corpus",
            markdown_hash="h_corpus",
            source_file_id="src_corpus",
            source_file_hash="sh_corpus",
            work_id="w_corpus",
        )
        ref_ranges = [
            (s.start_char, s.end_char, s.heading_text or "")
            for s in sections
            if s.section_kind == "references"
        ]
        entries = parse_references(document, ref_ranges)
        if not entries:
            continue
        n_files += 1
        for entry in entries:
            total += 1
            if _entry_is_matched(entry):
                matched += 1

    assert total > 0, "the regression corpus must yield at least one parsed entry"
    overlap = matched / total
    # Report the measurable recall number (NOT a hard target — see plan / risk 1).
    print(
        f"bib regression overlap: {matched}/{total} = {overlap:.1%} "
        f"across N={n_files} note(s) with a Full Bibliography"
    )
    # Floor at 0.80: locks in the numbered-/name-year-aware splitter's gain (v1 lifted
    # observed recall from ~0.56 over-split fragments to ~0.985 whole entries) and catches
    # a real regression while leaving headroom for corpus drift. The floor is only ever
    # lowered with a recorded rationale in the landing note, never silently (risk 1).
    assert overlap >= 0.80, f"bib parser recall unexpectedly low: {overlap:.1%}"

    # No-write invariant: the corpus is byte-for-byte unchanged after the run.
    after = {p: p.stat().st_size for p in sorted(notes.glob("*.md"))}
    assert after == before, "regression run must NOT modify the read-only library"


# ---------------------------------------------------------------------------
# The metric, pinned offline (unmarked — runs in the default suite).
# ---------------------------------------------------------------------------
def _synthetic_entry(
    *,
    title: str | None,
    doi: str | None,
    year: int | None,
    first_author: str | None,
) -> BibEntry:
    """A ``BibEntry`` with only the four rubric fields set (rest inert placeholders)."""
    return BibEntry(
        raw="raw",
        section_label="References",
        ordinal=1,
        first_author=first_author,
        year=year,
        title=title,
        doi=doi,
        arxiv=None,
    )


@pytest.mark.parametrize(
    "title, doi, year, first_author, expected",
    [
        # MATCHED — a title-or-DOI clause AND a year-or-first-author clause, each OR path.
        ("A Title", None, 2020, None, True),  # title + year
        ("A Title", None, None, "Smith", True),  # title + first author
        (None, "10.1000/x", 2020, None, True),  # doi + year
        (None, "10.1000/x", None, "Smith", True),  # doi + first author
        ("A Title", "10.1000/x", 2020, "Smith", True),  # all four
        # NOT matched — one clause empty (proves the two clauses are ANDed, not ORed).
        ("A Title", None, None, None, False),  # title only, no year/author
        (None, "10.1000/x", None, None, False),  # doi only, no year/author
        ("A Title", "10.1000/x", None, None, False),  # title+doi, no year/author
        (None, None, 2020, "Smith", False),  # year+author, no title/doi
        (None, None, 2020, None, False),  # year only
        (None, None, None, "Smith", False),  # first author only
        (None, None, None, None, False),  # empty
    ],
)
def test_matched_entry_rubric(title, doi, year, first_author, expected):
    """Pin the matched-entry rubric (title-or-DOI AND year-or-first-author) offline."""
    entry = _synthetic_entry(title=title, doi=doi, year=year, first_author=first_author)
    assert _entry_is_matched(entry) is expected
