"""Phase 3b — parsed-bibliography pipeline acceptance tests (§11), fully OFFLINE.

Every plan §11 acceptance test is implemented with real assertions and runs with
no network and no LLM: providers are pure in-memory mocks (or the offline
``FakeProvider`` behind ``SEEDGRAPH_FAKE_PROVIDERS`` + a pre-seeded
``provider_cache`` for the CLI milestone), identity is real, and the two-stage
lifecycle is exercised on a real cache (ingest + FakeMarkerBackend convert),
bridged work->markdown, sectioned, then parsed/resolved/projected through the
phase_3b surface.

Contracts exercised: bib_parser over document_sections ranges (no internal heading
scanner), the resolve precision guard + strong-id gate, the resolution-confidence
rubric (NOT a hardcoded 1.0), reference_entries population, parsed edges + the
reference_id FK (auto-vivified metadata_only targets, no dangle), union + authority
+ edge_disagreements (confirm vs adds-coverage split), citation_resolution review
routing, Stage-A/Stage-B idempotency (same run + new run), cross-run FK-safe
reparse (must-fix #1), per-work atomicity, the shareability boundary, and the
no_markdown / no_references skip paths.
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import pytest
from sqlmodel import Session
from typer.testing import CliRunner

from seedgraph import cache_access
from seedgraph.acquisition.bridge import backfill_markdown, resolve_work_markdown, write_bridge
from seedgraph.cache.convert import convert_source_file
from seedgraph.cache.db import init_cache_db
from seedgraph.cache.ingest import ingest_file
from seedgraph.cache.marker_backend import FakeMarkerBackend
from seedgraph.cache.provider_cache import cache_put, key_referenced_works
from seedgraph.citation import bib_parser, parsed_bib, resolve  # noqa: F401 (import-clean)
from seedgraph.citation.bib_parser import (
    BibEntry,
    _clean_entry_line,
    _is_discrete_sublist_label,
    parse_references,
)
from seedgraph.citation.edges import (
    authoritative_edges,
    edge_disagreements,
    is_shareable_edge,
    write_edge,
)
from seedgraph.citation.parsed_bib import build_parsed_edges
from seedgraph.citation.resolve import resolve_references
from seedgraph.cli import app
from seedgraph.db.adapter import raw_conn
from seedgraph.db.connection import open_cache_db
from seedgraph.graph.build import build_citation_graph
from seedgraph.graph.export import export_graph, load_graph
from seedgraph.project import identity as identity_mod
from seedgraph.project import review as review_mod
from seedgraph.project import service
from seedgraph.run import ensure_run
from seedgraph.sections.parser import parse_sections
from seedgraph.sections.store import replace_sections
from seedgraph.vocab import AccessClass, AcquisitionMethod

runner = CliRunner()


# ===========================================================================
# Offline fixtures / helpers
# ===========================================================================

REFS_MD = (
    "# Paper A\n\n"
    "This paper builds on prior work in the field.\n\n"
    "## References\n\n"
    "Blue, R. (2018). The blue widget framework. Widget Journal. https://doi.org/10.1111/blue\n\n"
    "Green, S. (2019). Green gadget analysis methods. Gadget Review. https://doi.org/10.2222/green\n"
)


class _Providers:
    """A pure in-memory provider chain double (async surface, offline)."""

    def __init__(self, by_doi=None, by_title=None):
        self._by_doi = {k.lower(): v for k, v in (by_doi or {}).items()}
        self._by_title = dict(by_title or {})

    async def by_doi(self, doi):
        return self._by_doi.get(str(doi).lower())

    async def by_title(self, title, year=None):
        return self._by_title.get(title)


def _refs_providers():
    """Chain double serving the two REFS_MD DOIs. Build A ch5: a by_doi MISS no
    longer self-resolves from the parsed entry (decision 13), so fixtures whose
    references must RESOLVE seed the chain with their DOIs explicitly."""
    return _Providers(by_doi={
        "10.1111/blue": {"doi": "10.1111/blue", "title": "The blue widget framework"},
        "10.2222/green": {"doi": "10.2222/green", "title": "Green gadget analysis methods"},
    })


def _project_conn(h) -> sqlite3.Connection:
    conn = sqlite3.connect(str(h.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _convert_bridge_sections(h, work_id, markdown, *, slug_file, pdf_bytes=None):
    """Ingest a pdf, convert to ``markdown``, bridge work->markdown, build sections."""
    p = Path(os.environ["SEEDGRAPH_HOME"]) / f"{slug_file}.pdf"
    p.write_bytes(pdf_bytes or (b"%PDF-1.4 " + slug_file.encode() + b" body words here and more"))
    src = ingest_file(
        p, access_class=AccessClass.open_access,
        acquisition_method=AcquisitionMethod.open_access_fetch, root=None,
    )
    md = convert_source_file(
        src.source_file_id, backend=FakeMarkerBackend(markdown=markdown), root=None
    )
    with Session(h.engine) as s:
        write_bridge(
            s, work_id=work_id, source_file_id=src.source_file_id, file_hash=src.file_hash,
            markdown_id=md.markdown_id, markdown_hash=md.markdown_hash,
            acquisition_method="open_access_fetch",
        )
        s.commit()
    _build_sections(h, work_id, md)
    return md, src


def _build_sections(h, work_id, md):
    cache_conn = cache_access.open_cache_ro(None)
    try:
        with Session(h.engine) as s:
            conn = raw_conn(s)
            row = cache_access.read_markdown(cache_conn, None, md.markdown_id)
            secs = parse_sections(
                row.text, markdown_id=md.markdown_id, markdown_hash=row.markdown_hash,
                source_file_id=row.source_file_id, source_file_hash=row.source_file_hash,
                work_id=work_id,
            )
            replace_sections(conn, md.markdown_id, secs)
            s.commit()
    finally:
        cache_conn.close()


def _citing_paper(slug, markdown=REFS_MD, *, doi="10.1000/aaa", openalex="W_A"):
    """A project + included citing work A with a bridged, sectioned references markdown."""
    h = service.create_project(slug)
    a = service.add_work(h, ids={"doi": doi, "openalex": openalex}, title="Paper A")
    md, src = _convert_bridge_sections(h, a.work_id, markdown, slug_file="cit")
    return h, a.work_id, md, src


def _wid_by_doi(h, doi: str) -> str | None:
    conn = sqlite3.connect(str(h.db_path))
    try:
        row = conn.execute("SELECT work_id FROM works WHERE doi = ?", (doi.lower(),)).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


# ===========================================================================
# bib_parser over document_sections ranges (PURE)
# ===========================================================================

def test_parse_references_known_section_yields_expected_entries():
    """A markdown with a known ``## References`` section + its document_sections
    range → ``parse_references`` yields the expected entries with correct count
    and ``first_author``/``year``/``title``/``doi``/``arxiv`` fields."""
    md = (
        "# Paper\n\nBody text here.\n\n## References\n\n"
        "Smith, J. (2019). A study of widgets. Journal of Widgets. https://doi.org/10.1234/widgets\n\n"
        "Jones, M. (2020). Theory of gadgets. Gadget Press. arXiv:1706.03762\n"
    )
    start = md.index("## References")
    entries = parse_references(md, [(start, len(md), "References")])
    assert len(entries) == 2
    assert entries[0].first_author == "Smith"
    assert entries[0].year == 2019
    assert entries[0].title == "A study of widgets"
    assert entries[0].doi == "10.1234/widgets"
    assert entries[0].arxiv is None
    assert entries[1].first_author == "Jones"
    assert entries[1].year == 2020
    assert entries[1].arxiv == "1706.03762"
    # The ``## References`` heading line is never surfaced as a spurious entry.
    assert all("References" not in e.raw or e.first_author for e in entries)
    assert [e.ordinal for e in entries] == [1, 2]


def test_numbered_section_not_author_comma_shredded():
    """A numbered ``[1]…[n]`` references section is split one entry per number,
    not shredded on author commas."""
    md = (
        "## References\n\n"
        "[1] Y. Bengio, A. Courville, P. Vincent. Representation learning review. *Nature*, 2015.\n"
        "[2] J. Smith, R. Roe. Another deep paper here. *Science*, 2016.\n"
        "[3] K. Lee, L. Park, M. Kim. A third numbered work. *Cell*, 2017.\n"
    )
    entries = parse_references(md, [(0, len(md), "References")])
    assert len(entries) == 3  # NOT shredded into one-per-author
    assert [e.first_author for e in entries] == ["Bengio", "Smith", "Lee"]
    assert entries[0].title == "Representation learning review"


def test_em_dash_repeated_author_carries_forward():
    """An em-dash / 3-em-dash repeated-author entry carries the prior author
    forward and sets ``continued_author=True``."""
    md = (
        "## References\n\n"
        "Smith, J. (2019). First paper of the run. Journal A.\n\n"
        "———. (2020). Second paper same author. Journal B.\n"
    )
    entries = parse_references(md, [(0, len(md), "References")])
    assert len(entries) == 2
    assert entries[0].first_author == "Smith"
    assert entries[0].continued_author is False
    assert entries[1].first_author == "Smith"  # carried forward
    assert entries[1].continued_author is True
    assert entries[1].year == 2020


def test_empty_ref_ranges_returns_empty():
    """``parse_references(md, [])`` returns ``[]`` — the parser never scans for
    headings itself (decisions 31/52/66/68; no internal heading scanner)."""
    md = "# A\n\n## References\n\nSmith, J. (2019). Title. Venue.\n"
    assert parse_references(md, []) == []


def test_multi_section_keeps_distinct_section_labels():
    """A multi-section doc (discussion + rejoinder, two references ranges) keeps
    distinct ``section_label`` values on its entries."""
    md = (
        "# Paper\n\nBody.\n\n"
        "## References\n\nSmith, J. (2019). Main reference work. Journal A.\n\n"
        "## Rejoinder References\n\nJones, M. (2020). Rejoinder reference work. Journal B.\n"
    )
    secs = parse_sections(
        md, markdown_id="md_x", markdown_hash="x", source_file_id="sf_x",
        source_file_hash="x", work_id="work_1",
    )
    ranges = [
        (s.start_char, s.end_char, s.heading_text)
        for s in secs if s.section_kind == "references"
    ]
    assert len(ranges) == 2  # both reference-kind sections located by document_sections
    entries = parse_references(md, ranges)
    labels = {e.section_label for e in entries}
    assert labels == {"References", "Rejoinder References"}


# ---------------------------------------------------------------------------
# Marker inline-markdown "demarkdown" — hyperlinked (Elsevier) and bold-wrapped
# references. Marker leaves markdown links / ``*``/``**`` emphasis inside
# reference lines, so every line opens with ``[`` or ``**`` and defeats
# entry-start detection + the collapse-split (the whole list collapses to ONE
# entry). ``_clean_entry_line`` now normalizes links -> text and strips ``*``.
# ---------------------------------------------------------------------------

def test_elsevier_hyperlinked_references_split_and_clean():
    """Elsevier/ScienceDirect hyperlink every author/title, so each reference line
    opens with ``[`` — at HEAD the whole list collapses into ONE entry. After
    demarkdown (paren-safe link -> visible text, tolerating ``(22)`` in the URL)
    each line opens with a bare capital again and splits into its own entry."""
    md = (
        "## References\n\n"
        "[Abadie,](http://refhub.elsevier.com/S0304-405X(22)00020-4/sbref0001) A., "
        "2005. [Difference-in-differences](http://refhub.elsevier.com/S0304-405X(22)00020-4/sbref0001) "
        "estimators. Rev. Econ. Stud. 72 (1), 1-19.\n"
        "[Callaway,](http://refhub.elsevier.com/S0304-405X(22)00020-4/sbref0002) B., "
        "Sant'Anna, P., 2021. Difference-in-differences with multiple time periods. "
        "J. Econometrics 225 (2), 200-230.\n"
        "[Goodman-Bacon,](http://refhub.elsevier.com/S0304-405X(22)00020-4/sbref0003) A., "
        "2021. Difference-in-differences with variation in treatment timing. "
        "J. Econometrics 225 (2), 254-277.\n"
    )
    entries = parse_references(md, [(0, len(md), "References")])
    assert len(entries) == 3  # was 1 (whole list collapsed) at HEAD
    assert [e.first_author for e in entries] == ["Abadie", "Callaway", "Goodman-Bacon"]
    assert [e.year for e in entries] == [2005, 2021, 2021]
    # No leftover markdown link/URL cruft anywhere in the reflowed text or fields.
    for e in entries:
        assert "[" not in e.raw and "]" not in e.raw
        assert "refhub.elsevier.com" not in e.raw
        assert "](" not in e.raw
    # The first entry's title is clean (no link fragment).
    assert entries[0].title == "Difference-in-differences estimators"


def test_bold_wrapped_references_split_and_clean():
    """Scanned papers wrap each reference in ``**...**`` bold; the leading ``**``
    blocks entry-start and the collapse-split, collapsing a multi-ref line into ONE
    entry. After emphasis stripping the line splits into its two entries."""
    md = (
        "## References\n\n"
        '**AMEMIYA, T. (1986), "Advanced Econometrics", *Harvard University Press*, '
        'Cambridge.** **ANDERSON, T. W. (1958), "An Introduction to Multivariate '
        'Statistical Analysis", *Wiley*, New York.**\n'
    )
    entries = parse_references(md, [(0, len(md), "References")])
    assert len(entries) == 2  # was 1 (bold-collapsed) at HEAD
    assert [e.first_author for e in entries] == ["AMEMIYA", "ANDERSON"]
    for e in entries:
        assert "*" not in e.raw  # emphasis stripped
    assert entries[0].year == 1986
    assert entries[1].year == 1958


def test_demarkdown_preserves_url_only_doi():
    """A link whose visible text lacks a DOI but whose URL carries one keeps the
    DOI (``[CrossRef](https://doi.org/10.1234/xyz)`` -> ``CrossRef 10.1234/xyz``)
    so ``_extract_doi`` still fires."""
    md = (
        "## References\n\n"
        "Smith, J. (2019). A study of widgets. Journal of Widgets. "
        "[CrossRef](https://doi.org/10.1234/xyz)\n"
    )
    entries = parse_references(md, [(0, len(md), "References")])
    assert len(entries) == 1
    assert entries[0].doi == "10.1234/xyz"
    assert entries[0].first_author == "Smith"


def test_clean_entry_line_leaves_bare_enumerators_untouched():
    """Guard: demarkdown must NOT touch a bare ``[1]`` / ``(2)`` enumerator (the
    link regex requires ``](…)`` immediately after ``]``)."""
    assert _clean_entry_line("[1] Y. Bengio. Deep learning. Nature, 2015.") == (
        "[1] Y. Bengio. Deep learning. Nature, 2015."
    )
    assert _clean_entry_line("(2) J. Smith. Another paper. Science, 2016.") == (
        "(2) J. Smith. Another paper. Science, 2016."
    )
    # And a numbered section with markdown-decorated bodies still splits per number.
    md = (
        "## References\n\n"
        "[1] Y. Bengio. **Deep learning** methods. Nature, 2015.\n"
        "[2] J. Smith. Another paper here. Science, 2016.\n"
        "[3] K. Lee. A third numbered work. Cell, 2017.\n"
    )
    entries = parse_references(md, [(0, len(md), "References")])
    assert len(entries) == 3
    assert [e.first_author for e in entries] == ["Bengio", "Smith", "Lee"]


# ---------------------------------------------------------------------------
# Gap-scan §4.6 — short numbered discussion/rejoinder sub-sections (Chunk 0
# captured the shred as strict xfails; Chunk 1 fixed it via the design-10
# plain-vs-qualified label discriminator: a QUALIFIED label — anything whose
# normalized form is NOT in the closed ``_PLAIN_REFERENCE_LABELS`` set — marks a
# discrete sub-list whose enumerators are authoritative below the >=3 floor and
# whose ``_COLLAPSE_SPLIT_RE`` collapse-split is suppressed (v1's
# ``opened_by_heading``, v1 bib_parser.py:933-936, 966-971).
# ---------------------------------------------------------------------------

_BARE_ENUM_FRAGMENT_RE = re.compile(r"^(?:\[\s*\d{1,3}\s*\]|\(\s*\d{1,3}\s*\)|\d{1,3}[.)])$")


def test_one_entry_numbered_discussion_section_not_shredded():
    """A 1-entry numbered references sub-section opened by a qualified discussion
    heading parses to exactly its listed count — one entry, no bare-enumerator
    fragment (v1 plan §3.3's Yoo scar)."""
    md = (
        "## References in Discussion by Bertrand Clarke\n\n"
        "1. Yoo, K. (2020). Comment on posterior model probabilities. Journal of Statistics.\n"
    )
    entries = parse_references(
        md, [(0, len(md), "References in Discussion by Bertrand Clarke")]
    )
    assert len(entries) == 1
    assert not any(_BARE_ENUM_FRAGMENT_RE.match(e.raw) for e in entries)
    assert entries[0].first_author == "Yoo"
    assert entries[0].year == 2020
    # Per-entry section_label provenance intact.
    assert entries[0].section_label == "References in Discussion by Bertrand Clarke"


def test_two_entry_numbered_rejoinder_section_not_shredded():
    """A 2-entry numbered rejoinder references sub-section parses to exactly two
    entries — its multi-author entry is never collapse-split into author-name
    fragments (v1 plan §3.3's Zhou scar)."""
    md = (
        "## Rejoinder References\n\n"
        "1. Cengiz, D., A. Dube, A. Lindner, and B. Zipperer (2019). The effect of "
        "minimum wages on low-wage jobs. Quarterly Journal of Economics.\n"
        "2. Zhou, W. (2021). Semiparametric estimation under shape constraints. "
        "Annals of Statistics.\n"
    )
    entries = parse_references(md, [(0, len(md), "Rejoinder References")])
    assert len(entries) == 2
    assert not any(_BARE_ENUM_FRAGMENT_RE.match(e.raw) for e in entries)
    assert [e.first_author for e in entries] == ["Cengiz", "Zhou"]
    assert [e.ordinal for e in entries] == [1, 2]
    assert {e.section_label for e in entries} == {"Rejoinder References"}


def test_two_entry_nameyear_discussion_section_not_collapse_shredded():
    """A 2-entry NAME-YEAR (non-numbered) qualified range is a discrete sub-list
    too: the collapse-split is suppressed, so its multi-author entry is never
    shredded into 'Cengiz, D., A.'-style fragments."""
    md = (
        "## References in Discussion by Ana Ramos\n\n"
        "Cengiz, D., A. Dube, A. Lindner, and B. Zipperer (2019). The effect of "
        "minimum wages on low-wage jobs. Quarterly Journal of Economics.\n\n"
        "Zhou, W. (2021). Semiparametric estimation under shape constraints. "
        "Annals of Statistics.\n"
    )
    entries = parse_references(
        md, [(0, len(md), "References in Discussion by Ana Ramos")]
    )
    assert len(entries) == 2
    assert [e.first_author for e in entries] == ["Cengiz", "Zhou"]
    assert {e.section_label for e in entries} == {
        "References in Discussion by Ana Ramos"
    }


def test_shred_discriminator_plain_labels_classify_plain():
    """Design-10 normalization: numbered plain headings (``## 6. References``),
    emphasis/anchor/colon dressing, and every closed-set member classify PLAIN
    (``_is_discrete_sublist_label`` is False → behavior byte-identical)."""
    for label in (
        "References",
        "6. References",
        "6) References",
        "References:",
        "**References**",
        '<span id="page-9-0"></span>References',
        "References and Notes",
        "Literature",
        "Bibliography",
        "Works Cited",
        "Literature Cited",
        "REFERENCE LIST",
    ):
        assert _is_discrete_sublist_label(label) is False, label
    # Qualified discussion/rejoinder labels classify as discrete sub-lists.
    for label in (
        "Rejoinder References",
        "References in Discussion by Bertrand Clarke",
        "Main References (Yao, Vehtari, Simpson, and Gelman)",
        "Sources",  # unlisted plain variant — closed set, benign miss direction
    ):
        assert _is_discrete_sublist_label(label) is True, label


def test_unlisted_plain_variant_sources_benign_no_split_not_shredded():
    """Design-10 asymmetry, pinned: an UNLISTED plain variant ('Sources')
    classifies QUALIFIED, so a collapsed two-work name-year line stays ONE
    unsplit entry — the benign miss (loses the collapse-split) — and is never
    shredded into bare-enumerator/author fragments."""
    md = (
        "## Sources\n\n"
        "Smith, J. (2019). First widget paper. Journal A. "
        "Jones, M. (2020). Second gadget paper. Journal B.\n"
    )
    entries = parse_references(md, [(0, len(md), "Sources")])
    assert len(entries) == 1  # unsplit collapsed line — benign, NOT a shred
    assert "Smith, J. (2019)" in entries[0].raw
    assert "Jones, M. (2020)" in entries[0].raw
    assert not any(_BARE_ENUM_FRAGMENT_RE.match(e.raw) for e in entries)


def test_plain_references_collapsed_line_still_splits_to_two():
    """CONTROL for the §4.6 fix: in a plain (non-numbered) ``References`` range,
    two name-year entries collapsed onto ONE physical line still split into
    exactly two entries — the collapse-split is load-bearing there and must
    survive Chunk 1's suppression (which applies only to short numbered
    sub-sections)."""
    md = (
        "## References\n\n"
        "Smith, J. (2019). First widget paper. Journal A. "
        "Jones, M. (2020). Second gadget paper. Journal B.\n"
    )
    entries = parse_references(md, [(0, len(md), "References")])
    assert len(entries) == 2
    assert [e.first_author for e in entries] == ["Smith", "Jones"]
    assert [e.year for e in entries] == [2019, 2020]


# ===========================================================================
# resolve precision guard (v1 parity; providers mocked)
# ===========================================================================

def _entry(title=None, year=None, doi=None, arxiv=None, first_author="X", ordinal=1):
    return BibEntry(
        raw="raw", section_label="References", ordinal=ordinal, first_author=first_author,
        year=year, title=title, doi=doi, arxiv=arxiv,
    )


def test_doi_entry_resolves_by_doi():
    """A DOI-bearing entry resolves by DOI: ``status='resolved'``,
    ``resolution_source='doi'``, ``confidence≈1.0``."""
    e = _entry(title="Widgets", year=2019, doi="10.1234/widgets")
    prov = _Providers(by_doi={"10.1234/widgets": {"doi": "10.1234/widgets", "openalex_id": "W_X", "title": "Widgets"}})
    [rr] = resolve_references([e], prov, identity_mod, citing_work_id="W_SRC")
    assert rr.status == "resolved"
    assert rr.resolution_source == "doi"
    assert rr.confidence == pytest.approx(1.0)


def test_doi_miss_no_title_routes_ambiguous():
    """Build A ch5 (decision 13): a parsed DOI the chain MISSES with no title to
    corroborate → ``ambiguous`` with the DOI kept as a reviewer-visible candidate
    — never a self-minted ``resolved``/confidence-1.0 identity."""
    e = _entry(doi="10.9999/miss")  # no title
    [rr] = resolve_references([e], _Providers(), identity_mod, citing_work_id="W_SRC")
    assert rr.status == "ambiguous"
    assert rr.resolved_work_id is None
    assert rr.candidates and rr.candidates[0]["doi"] == "10.9999/miss"


def test_doi_miss_pinning_title_resolves_title_year():
    """Build A ch5: a DOI miss FALLS THROUGH to the title(+year) path (previously
    skipped entirely for DOI-bearing entries); a strong-id title hit resolves via
    ``title_year`` with the scored confidence, not the DOI's hardcoded 1.0."""
    e = _entry(doi="10.9999/miss", title="Theory of gadgets and gizmos", year=2020)
    prov = _Providers(by_title={
        "Theory of gadgets and gizmos": {"openalex_id": "W_TY", "title": "Theory of gadgets and gizmos", "year": 2020},
    })
    [rr] = resolve_references([e], prov, identity_mod, citing_work_id="W_SRC")
    assert rr.status == "resolved"
    assert rr.resolution_source == "title_year"
    assert 0.7 <= rr.confidence <= 0.9
    assert rr.candidates[0].get("openalex_id") == "W_TY"


def test_doi_miss_unpinned_title_stays_ambiguous_with_doi_candidate():
    """Build A ch5: DOI miss + title the chain also misses → ``ambiguous``; the
    parsed-DOI candidate is preserved for the reviewer."""
    e = _entry(doi="10.9999/miss", title="An unfindable manuscript title", year=2020)
    [rr] = resolve_references([e], _Providers(), identity_mod, citing_work_id="W_SRC")
    assert rr.status == "ambiguous"
    assert rr.resolved_work_id is None
    assert any(c.get("doi") == "10.9999/miss" for c in rr.candidates)


def test_doi_miss_full_path_enqueues_review_no_edge_no_identity():
    """Build A ch5 end-to-end: with an EMPTY chain the two REFS_MD DOI entries
    route ``ambiguous`` — review items enqueued, ZERO edges, and no work is
    self-minted from the unverified parsed DOIs."""
    h, a, md, _src = _citing_paper("doimiss")
    res = build_parsed_edges(h, None, _Providers(), identity_mod, work_id=a, run_id="R")
    assert res.entries == 2 and res.ambiguous == 2 and res.resolved == 0
    assert res.edges_written == 0
    assert _wid_by_doi(h, "10.1111/blue") is None  # no self-minted identity
    conn = _project_conn(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0] == 0
    finally:
        conn.close()
    items = [i for i in review_mod.list_open(h) if i.item_type == "citation_resolution"]
    assert len(items) == 2
    import json as _json
    dois = {_json.loads(i.payload)["doi"] for i in items}
    assert dois == {"10.1111/blue", "10.2222/green"}


def test_title_only_entry_ambiguous_no_edge():
    """A title-only entry → ``status='ambiguous'`` and produces **no** edge; the
    entry is retained raw."""
    e = _entry(title="An untraceable manuscript title", year=2015)
    prov = _Providers()  # chain returns nothing
    [rr] = resolve_references([e], prov, identity_mod, citing_work_id="W_SRC")
    assert rr.status == "ambiguous"
    assert rr.resolved_work_id is None


def test_denylisted_refwork_doi_suspect():
    """A no-own-DOI entry matched to a denylisted reference-work DOI → ``suspect``
    with ``reject_reason='refwork_doi_denylist'`` and no edge."""
    e = _entry(title="A handbook chapter on something", year=2000)
    prov = _Providers(by_title={
        "A handbook chapter on something": {"doi": "10.4135/refwork123", "title": "A handbook chapter on something"},
    })
    [rr] = resolve_references([e], prov, identity_mod, citing_work_id="W_SRC")
    assert rr.status == "suspect"
    assert rr.reject_reason == "refwork_doi_denylist"
    assert rr.resolved_work_id is None


def test_low_title_overlap_strong_id_suspect():
    """A strong-id hit with ``<2`` content-token title overlap → ``suspect`` with
    ``reject_reason='low_title_overlap'`` and no edge."""
    e = _entry(title="Quantum widgets in deep space", year=2005)
    prov = _Providers(by_title={
        "Quantum widgets in deep space": {"openalex_id": "W_WRONG", "title": "Cooking recipes for beginners"},
    })
    [rr] = resolve_references([e], prov, identity_mod, citing_work_id="W_SRC")
    assert rr.status == "suspect"
    assert rr.reject_reason == "low_title_overlap"
    assert rr.resolved_work_id is None


def test_unresolvable_entry_unresolved_kept_raw():
    """An entry with no usable fields → ``status='unresolved'``, kept raw so
    recall stays measurable (doc 04 §5)."""
    e = BibEntry(raw="garbled line", section_label="References", ordinal=1,
                 first_author=None, year=None, title=None, doi=None, arxiv=None)
    [rr] = resolve_references([e], _refs_providers(), identity_mod, citing_work_id="W_SRC")
    assert rr.status == "unresolved"
    assert rr.entry.raw == "garbled line"


def test_title_hash_canonical_key_no_edge_routes_ambiguous():
    """Strong-id gate: a chain hit whose ``identity.canonical_key`` is a
    ``title_hash`` does NOT produce an edge — it routes ``ambiguous`` (decision 64
    parity)."""

    class _TitleHashIdentity:
        def canonical_key(self, rec):
            # Simulate a chain record that only resolves to a fuzzy title hash.
            return ("title_hash", "deadbeef")

    e = _entry(title="A plausibly matching title here", year=2018)
    prov = _Providers(by_title={
        "A plausibly matching title here": {"openalex_id": "W_TH", "title": "A plausibly matching title here"},
    })
    [rr] = resolve_references([e], prov, _TitleHashIdentity(), citing_work_id="W_SRC")
    assert rr.status == "ambiguous"
    assert rr.resolved_work_id is None


def test_title_year_resolution_confidence_not_hardcoded_one():
    """A title(+year) strong-id hit above threshold resolves with a token-overlap
    SCORED confidence in [0.7, 0.9] — NOT the provider-only hardcoded 1.0 (§7)."""
    e = _entry(title="Theory of gadgets and gizmos", year=2020)
    prov = _Providers(by_title={
        "Theory of gadgets and gizmos": {"openalex_id": "W_TY", "title": "Theory of gadgets and gizmos", "year": 2020},
    })
    [rr] = resolve_references([e], prov, identity_mod, citing_work_id="W_SRC")
    assert rr.status == "resolved"
    assert rr.resolution_source == "title_year"
    assert 0.7 <= rr.confidence <= 0.9
    assert rr.confidence != 1.0


# ===========================================================================
# reference_entries population
# ===========================================================================

def test_reference_entries_population_one_row_per_entry():
    """Every parsed entry yields exactly one ``reference_entries`` row with
    verbatim ``raw_reference_text``, correct ``markdown_id``/``markdown_hash``/
    ``section_label``, and ``resolution_status`` per the §7 rubric."""
    h, a, md, _src = _citing_paper("refpop")
    # B exists (doi 10.1111/blue) -> Blue resolves; Green's target is auto-vivified.
    service.add_work(h, ids={"doi": "10.1111/blue", "openalex": "W_B"}, title="Blue")
    res = build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R")
    assert res.entries == 2
    conn = _project_conn(h)
    try:
        rows = conn.execute(
            "SELECT raw_reference_text, markdown_id, markdown_hash, section_label, resolution_status "
            "FROM reference_entries WHERE citing_work_id = ? ORDER BY created_at", (a,)
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 2
    for raw, mid, mhash, label, status in rows:
        assert mid == md.markdown_id
        assert mhash == md.markdown_hash
        assert label == "References"
        assert status == "resolved"  # both DOI-bearing -> resolved
    raws = {r[0] for r in rows}
    assert any("Blue, R. (2018)" in r for r in raws)  # verbatim raw text retained
    assert any("Green, S. (2019)" in r for r in raws)


# ===========================================================================
# parsed edges + reference_id
# ===========================================================================

def test_parsed_edge_carries_reference_id_and_resolves_target():
    """Each ``resolved`` entry yields one ``parsed_bibliography`` edge carrying its
    ``reference_id``; ``target_work_id`` resolves (auto-vivified ``metadata_only``
    when out-of-corpus); the FK guarantees no dangling edge (criterion 5)."""
    # Reference Blue's DOI exists in corpus; Green's DOI does NOT -> auto-vivified.
    h, a, md, _src = _citing_paper("refid")
    service.add_work(h, ids={"doi": "10.1111/blue", "openalex": "W_B"}, title="Blue")
    res = build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R")
    assert res.resolved == 2 and res.edges_written == 2

    conn = _project_conn(h)
    try:
        edges = conn.execute(
            "SELECT source_work_id, target_work_id, provenance, reference_id, confidence "
            "FROM citation_edges WHERE provenance = 'parsed_bibliography'"
        ).fetchall()
        # Every edge carries a reference_id resolving to a real reference_entries row.
        for _s, tgt, prov, ref_id, conf in edges:
            assert prov == "parsed_bibliography"
            assert ref_id is not None
            assert conn.execute(
                "SELECT 1 FROM reference_entries WHERE reference_id = ?", (ref_id,)
            ).fetchone() is not None
            assert conn.execute(
                "SELECT 1 FROM works WHERE work_id = ?", (tgt,)
            ).fetchone() is not None
        # The out-of-corpus Green target was auto-vivified metadata_only.
        green = _wid_by_doi(h, "10.2222/green")
        assert green is not None
        incl = conn.execute(
            "SELECT inclusion_status FROM project_documents WHERE work_id = ?", (green,)
        ).fetchone()[0]
        assert incl == "metadata_only"
        # No dangling edges / FK violations (foreign_keys=ON).
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()
    assert len(edges) == 2


# ===========================================================================
# union + authority + disagreement (confirm path)
# ===========================================================================

def test_union_authority_disagreement_confirm_path():
    """Seed a ``provider_reference`` edge for pair (A,B) under run R; parsing a
    ``parsed_bibliography`` edge for (A,B) under R → both rows persist;
    ``authoritative_edges(R)`` returns the parsed edge; ``edge_disagreements(R)``
    lists (A,B) with both provenances and winner ``parsed_bibliography``."""
    h, a, md, _src = _citing_paper("confirm")
    b = service.add_work(h, ids={"doi": "10.1111/blue", "openalex": "W_B"}, title="Blue").work_id

    conn = _project_conn(h)
    try:
        # Provider tier already has A->B under R.
        write_edge(conn, source=a, target=b, provenance="provider_reference", confidence=1.0, run_id="R")
        conn.commit()
    finally:
        conn.close()

    build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R")

    conn = _project_conn(h)
    try:
        rows = conn.execute(
            "SELECT provenance FROM citation_edges WHERE source_work_id=? AND target_work_id=? AND run_id='R'",
            (a, b),
        ).fetchall()
        provs = {r[0] for r in rows}
        auth = authoritative_edges(conn, "R")
        dis = edge_disagreements(conn, "R")
    finally:
        conn.close()

    assert provs == {"provider_reference", "parsed_bibliography"}  # both rows persist
    ab_auth = [e for e in auth if e["source_work_id"] == a and e["target_work_id"] == b]
    assert len(ab_auth) == 1 and ab_auth[0]["provenance"] == "parsed_bibliography"  # parsed wins
    ab_dis = [d for d in dis if d["source"] == a and d["target"] == b]
    assert len(ab_dis) == 1
    assert set(ab_dis[0]["provenances"]) == {"provider_reference", "parsed_bibliography"}
    assert ab_dis[0]["authoritative_provenance"] == "parsed_bibliography"


def test_single_provenance_pair_not_listed_in_disagreements():
    """A pair carrying only one provenance is NOT listed by
    ``edge_disagreements`` (criterion 4)."""
    h = service.create_project("single")
    a = service.add_work(h, ids={"openalex": "W_A"}, title="A").work_id
    b = service.add_work(h, ids={"openalex": "W_B"}, title="B").work_id
    conn = _project_conn(h)
    try:
        write_edge(conn, source=a, target=b, provenance="provider_reference", confidence=1.0, run_id="R")
        conn.commit()
        dis = edge_disagreements(conn, "R")
    finally:
        conn.close()
    assert dis == []  # a single-provenance pair is not a disagreement


def test_adds_coverage_single_provenance_absent_from_disagreements():
    """A parsed edge for pair (A,C) with no provider counterpart appears in
    ``authoritative_edges(R)`` / ``cite table`` but is **absent** from
    ``edge_disagreements(R)`` (criterion 3 vs 4 split)."""
    h, a, md, _src = _citing_paper("addscov")
    # Both reference targets exist; NO provider edges are seeded.
    service.add_work(h, ids={"doi": "10.1111/blue", "openalex": "W_B"}, title="Blue")
    c = service.add_work(h, ids={"doi": "10.2222/green", "openalex": "W_C"}, title="Green").work_id
    build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R")

    conn = _project_conn(h)
    try:
        auth = authoritative_edges(conn, "R")
        dis = edge_disagreements(conn, "R")
    finally:
        conn.close()

    ac_auth = [e for e in auth if e["source_work_id"] == a and e["target_work_id"] == c]
    assert len(ac_auth) == 1 and ac_auth[0]["provenance"] == "parsed_bibliography"
    assert dis == []  # single-provenance adds-coverage edges are NOT disagreements


# ===========================================================================
# review routing
# ===========================================================================

def test_review_routing_enqueues_citation_resolution_payload():
    """An ambiguous and a suspect entry each enqueue exactly one
    ``citation_resolution`` ``review_queue`` item whose ``CitationResolutionPayload``
    (discriminator ``kind``) passes the ``ReviewPayload`` union validation."""
    md = (
        "# Paper\n\nBody.\n\n## References\n\n"
        "White, T. (2015). An ambiguous untraceable manuscript title here. Self Published.\n\n"
        "Black, U. (2016). A handbook chapter on something else. Reference Compendium.\n"
    )
    h = service.create_project("review")
    a = service.add_work(h, ids={"doi": "10.1000/aaa", "openalex": "W_A"}, title="Paper A").work_id
    _convert_bridge_sections(h, a, md, slug_file="rev")
    prov = _Providers(by_title={
        # White -> not found -> ambiguous; Black -> refwork denylist -> suspect.
        "A handbook chapter on something else": {"doi": "10.4135/refworkY", "title": "A handbook chapter on something else"},
    })
    res = build_parsed_edges(h, None, prov, identity_mod, work_id=a, run_id="R")
    assert res.ambiguous == 1 and res.suspect == 1 and res.edges_written == 0

    items = review_mod.list_open(h)
    cr = [i for i in items if i.item_type == "citation_resolution"]
    assert len(cr) == 2
    statuses = set()
    import json as _json

    for item in cr:
        # The stored payload round-trips through the union validator (discriminator kind).
        payload = _json.loads(item.payload)
        validated = review_mod._validate_payload(payload)
        assert validated["kind"] == "citation_resolution"
        assert item.target_type == "reference_entry"
        statuses.add(validated["status"])
    assert statuses == {"ambiguous", "suspect"}


def test_invalid_review_payload_rejected():
    """An invalid citation-resolution payload is rejected by ``enqueue``
    (decision 80)."""
    h = service.create_project("badpayload")
    bad = {
        "kind": "citation_resolution",
        "status": "not_a_valid_status",  # not in Literal['ambiguous','suspect']
        "citing_work_id": "W_A",
        "reference_id": "ref_x",
        "raw": "raw",
        "run_id": "R",
    }
    with pytest.raises(Exception):
        review_mod.enqueue(h, "citation_resolution", target_type="reference_entry",
                           target_id="ref_x", payload=bad)
    # Nothing was persisted.
    assert review_mod.list_open(h) == []


# ===========================================================================
# Stage-A / Stage-B idempotency
# ===========================================================================

def test_same_run_unchanged_markdown_idempotent():
    """Running ``cite parse`` twice on unchanged markdown under the same run →
    identical ``reference_entries`` count and no duplicate parsed edges (Stage A
    skipped, Stage B UNIQUE-dedups) (criterion 7a)."""
    h, a, md, _src = _citing_paper("idem")
    service.add_work(h, ids={"doi": "10.1111/blue", "openalex": "W_B"}, title="Blue")
    r1 = build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R")
    assert r1.reparsed is True

    def _counts():
        conn = _project_conn(h)
        try:
            refs = conn.execute("SELECT COUNT(*) FROM reference_entries WHERE citing_work_id=?", (a,)).fetchone()[0]
            edges = conn.execute(
                "SELECT COUNT(*) FROM citation_edges WHERE source_work_id=? AND provenance='parsed_bibliography' AND run_id='R'", (a,)
            ).fetchone()[0]
            return refs, edges
        finally:
            conn.close()

    before = _counts()
    r2 = build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R")
    assert r2.reparsed is False  # Stage A skipped on unchanged markdown
    after = _counts()
    assert before == after  # no new refs, no duplicate edges


def test_new_run_unchanged_markdown_projects_parsed_tier():
    """Regression for must-fix #2: a NEW run R2 over unchanged markdown still gets
    the full parsed tier projected (Stage A skipped, Stage B ALWAYS projects);
    ``authoritative_edges(R2)`` sees it (criterion 7b)."""
    h, a, md, _src = _citing_paper("newrun")
    service.add_work(h, ids={"doi": "10.1111/blue", "openalex": "W_B"}, title="Blue")
    build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R1")

    r2 = build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R2")
    assert r2.reparsed is False  # Stage A skipped (unchanged markdown)
    assert r2.edges_written == 2  # ...but Stage B STILL projected into R2

    conn = _project_conn(h)
    try:
        n_r2 = conn.execute(
            "SELECT COUNT(*) FROM citation_edges WHERE source_work_id=? AND provenance='parsed_bibliography' AND run_id='R2'", (a,)
        ).fetchone()[0]
        auth_r2 = authoritative_edges(conn, "R2")
    finally:
        conn.close()
    assert n_r2 == 2  # the silent-loss bug must NOT recur
    assert any(e["provenance"] == "parsed_bibliography" for e in auth_r2)


# ===========================================================================
# cross-run FK-safe reparse (regression for must-fix #1)
# ===========================================================================

def test_cross_run_fk_safe_reparse():
    """Seed parsed edges for work W under runs R1 AND R2 (both referencing W's
    ``reference_entries``); reconvert (new ``markdown_hash``); reparse deletes W's
    parsed edges in **both** runs before deleting ``reference_entries`` with
    ``PRAGMA foreign_keys=ON`` and no FK violation; refs+edges reinserted; the
    doctor reports the pre-reparse rows ``stale`` against cache (criterion 7c)."""
    from seedgraph.doctor import reference_entries_findings

    h, a, md, src = _citing_paper("fksafe")
    service.add_work(h, ids={"doi": "10.1111/blue", "openalex": "W_B"}, title="Blue")
    # Edges under R1 and R2 (Stage A once, Stage B into both runs) — both reference
    # the same reference_entries rows.
    build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R1")
    build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R2")

    conn = _project_conn(h)
    try:
        before = conn.execute(
            "SELECT COUNT(*) FROM citation_edges WHERE source_work_id=? AND provenance='parsed_bibliography'", (a,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert before >= 4  # >=2 edges in each of R1 and R2

    # Reconvert to NEW markdown (same lineage; new markdown_id/hash), update bridge
    # + rebuild sections so resolve_work_markdown returns the new hash.
    new_md = convert_source_file(
        src.source_file_id,
        backend=FakeMarkerBackend(markdown=REFS_MD + "\nGarvey, P. (2021). Extra appended reference. Press.\n"),
        force=True, root=None,
    )
    assert new_md.markdown_hash != md.markdown_hash
    with Session(h.engine) as s:
        backfill_markdown(s, work_id=a, markdown_id=new_md.markdown_id, markdown_hash=new_md.markdown_hash)
        s.commit()
    _build_sections(h, a, new_md)

    # Pre-reparse: the stored reference_entries (old hash) are STALE against cache.
    pre = reference_entries_findings(h.engine, None)
    assert any(s == "stale" for _w, s in pre)

    # Reparse: Stage A deletes parsed edges across BOTH runs before refs (FK-safe).
    res = build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R1")
    assert res.reparsed is True

    conn = _project_conn(h)
    try:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []  # no FK violation
        # Refs reinserted with the NEW hash.
        hashes = {r[0] for r in conn.execute(
            "SELECT DISTINCT markdown_hash FROM reference_entries WHERE citing_work_id=?", (a,)
        ).fetchall()}
        assert hashes == {new_md.markdown_hash}
        # Edges re-projected into R1 (other runs' stale parsed edges were cleared).
        n_r1 = conn.execute(
            "SELECT COUNT(*) FROM citation_edges WHERE source_work_id=? AND provenance='parsed_bibliography' AND run_id='R1'", (a,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert n_r1 >= 1

    # Post-reparse the soft-refs reconcile ok again.
    post = reference_entries_findings(h.engine, None)
    assert all(s == "ok" for _w, s in post)


# ===========================================================================
# atomicity (§6.4)
# ===========================================================================

def test_per_work_lifecycle_atomic_rollback(monkeypatch):
    """Inject a failure during Stage A (the resolve step); the whole work rolls
    back — no half-written ``reference_entries``, no orphan edges, no stray review
    items (one connection, one transaction)."""
    h, a, md, _src = _citing_paper("atomic")
    service.add_work(h, ids={"doi": "10.1111/blue", "openalex": "W_B"}, title="Blue")

    def _boom(*args, **kwargs):
        raise RuntimeError("injected mid-Stage-A failure")

    # parsed_bib imports resolve_references lazily inside the body, so patching the
    # module attribute takes effect at call time.
    monkeypatch.setattr("seedgraph.citation.resolve.resolve_references", _boom)

    with pytest.raises(RuntimeError):
        build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R")

    conn = _project_conn(h)
    try:
        refs = conn.execute("SELECT COUNT(*) FROM reference_entries WHERE citing_work_id=?", (a,)).fetchone()[0]
        edges = conn.execute("SELECT COUNT(*) FROM citation_edges WHERE source_work_id=?", (a,)).fetchone()[0]
        rq = conn.execute("SELECT COUNT(*) FROM review_queue").fetchone()[0]
    finally:
        conn.close()
    assert refs == 0 and edges == 0 and rq == 0  # whole work rolled back


# ===========================================================================
# shareability boundary
# ===========================================================================

def test_parsed_bibliography_not_shareable_excluded_from_graph_json(tmp_path):
    """``is_shareable_edge('parsed_bibliography') is False``; ``graph.json`` contains
    no parsed edges (only provider_reference), while the full local
    ``authoritative_edges`` view contains them (criterion 8)."""
    assert is_shareable_edge("parsed_bibliography") is False

    h, a, md, _src = _citing_paper("share")
    service.add_work(h, ids={"doi": "10.1111/blue", "openalex": "W_B"}, title="Blue")
    service.add_work(h, ids={"doi": "10.2222/green", "openalex": "W_C"}, title="Green")
    build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R")

    conn = _project_conn(h)
    try:
        auth = authoritative_edges(conn, "R")
        g = build_citation_graph(conn, run_id="R", closed_world=True)
    finally:
        conn.close()

    assert any(e["provenance"] == "parsed_bibliography" for e in auth)  # local view has them
    out = tmp_path / "out"
    export_graph(g, out, manifest={"citation": {"run_id": "R"}})
    reloaded = load_graph(out / "graph.json")
    assert reloaded.number_of_edges() == 0  # parsed edges filtered out of the share export
    for _s, _t, data in reloaded.edges(data=True):
        assert is_shareable_edge(data["provenance"])


# ===========================================================================
# skip paths
# ===========================================================================

def test_skip_no_markdown():
    """A work with no ``work_source_files`` row → ``skipped_reason='no_markdown'``,
    no crash."""
    h = service.create_project("nomd")
    a = service.add_work(h, ids={"doi": "10.1000/aaa"}, title="A").work_id
    res = build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R")
    assert res.skipped_reason == "no_markdown"
    assert res.entries == 0 and res.edges_written == 0


def test_skip_no_references():
    """A markdown with no ``references`` section → ``skipped_reason='no_references'``,
    zero ``reference_entries`` rows."""
    md = "# Paper\n\n## Introduction\n\nNo references section here at all.\n"
    h = service.create_project("noref")
    a = service.add_work(h, ids={"doi": "10.1000/aaa"}, title="A").work_id
    _convert_bridge_sections(h, a, md, slug_file="noref")
    res = build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R")
    assert res.skipped_reason == "no_references"
    assert res.entries == 0
    conn = _project_conn(h)
    try:
        n = conn.execute("SELECT COUNT(*) FROM reference_entries WHERE citing_work_id=?", (a,)).fetchone()[0]
    finally:
        conn.close()
    assert n == 0


def test_parse_auto_builds_missing_sections():
    """Regression: a work with a real ``## References`` section but whose
    ``sections build`` was NEVER run must NOT be mis-skipped as ``no_references``.

    ``cite parse`` locates references from phase_3 ``document_sections``; when that
    prerequisite is absent, parse now builds the sections on demand (same
    parse_sections/replace_sections the ``sections build`` CLI runs) before loading
    the ranges. Set up bridge WITHOUT pre-building sections, then assert the
    bibliography parses and the references section rows were auto-built."""
    # Ingest + convert + bridge, but deliberately OMIT the sections pre-build step
    # (no parse_sections / replace_sections in this setup).
    h = service.create_project("autosections")
    a = service.add_work(h, ids={"doi": "10.1000/aaa", "openalex": "W_A"}, title="Paper A").work_id
    p = Path(os.environ["SEEDGRAPH_HOME"]) / "autosec.pdf"
    p.write_bytes(b"%PDF-1.4 autosec body words here and more")
    src = ingest_file(
        p, access_class=AccessClass.open_access,
        acquisition_method=AcquisitionMethod.open_access_fetch, root=None,
    )
    md = convert_source_file(
        src.source_file_id, backend=FakeMarkerBackend(markdown=REFS_MD), root=None
    )
    with Session(h.engine) as s:
        write_bridge(
            s, work_id=a, source_file_id=src.source_file_id, file_hash=src.file_hash,
            markdown_id=md.markdown_id, markdown_hash=md.markdown_hash,
            acquisition_method="open_access_fetch",
        )
        s.commit()

    # Sanity: no document_sections exist for this markdown yet (prerequisite absent).
    conn = _project_conn(h)
    try:
        pre = conn.execute(
            "SELECT COUNT(*) FROM document_sections WHERE markdown_id=?", (md.markdown_id,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert pre == 0

    # B exists (doi 10.1111/blue) -> Blue resolves by DOI; Green auto-vivified.
    service.add_work(h, ids={"doi": "10.1111/blue", "openalex": "W_B"}, title="Blue")

    res = build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R")

    # NOT silently mis-skipped as no_references, and the bibliography WAS parsed.
    assert res.skipped_reason != "no_references"
    assert not res.skipped_reason
    assert res.entries > 0

    # The references section was auto-built (>=1 section_kind='references' row).
    conn = _project_conn(h)
    try:
        n_refsec = conn.execute(
            "SELECT COUNT(*) FROM document_sections "
            "WHERE markdown_id=? AND section_kind='references'", (md.markdown_id,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert n_refsec >= 1


#: A references section whose ONLY heading is a bold-only ``**References**`` line —
#: never an ATX ``document_sections`` row (the sections parser is ATX-only).
BOLD_REFS_MD = (
    "# Paper A\n\n"
    "This paper builds on prior work in the field.\n\n"
    "**References**\n\n"
    "Blue, R. (2018). The blue widget framework. Widget Journal. https://doi.org/10.1111/blue\n\n"
    "Green, S. (2019). Green gadget analysis methods. Gadget Review. https://doi.org/10.2222/green\n"
)


def test_non_atx_bold_heading_parses_via_fallback():
    """Build F chunk 3 (§5.8a): a bold-only ``**References**`` heading is recovered by
    the ``parsed_bib`` non-ATX fallback.

    The ATX-only sections parser produces NO ``section_kind='references'`` row for a
    bold heading, so at HEAD ``build_parsed_edges`` skips the whole bibliography as
    ``no_references``. The chunk-3 fallback synthesizes a range from the bold heading to
    end-of-document, so the bibliography now parses to its two entries.
    """
    h = service.create_project("boldrefs")
    a = service.add_work(
        h, ids={"doi": "10.1000/aaa", "openalex": "W_A"}, title="Paper A"
    ).work_id
    md, _src = _convert_bridge_sections(h, a, BOLD_REFS_MD, slug_file="boldref")

    # Sanity: the ATX-only sections parser created NO references-kind section — the
    # exact condition (zero references-kind ranges) the parsed_bib fallback recovers.
    conn = _project_conn(h)
    try:
        n_refsec = conn.execute(
            "SELECT COUNT(*) FROM document_sections "
            "WHERE markdown_id=? AND section_kind='references'", (md.markdown_id,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert n_refsec == 0

    res = build_parsed_edges(h, None, _refs_providers(), identity_mod, work_id=a, run_id="R")

    # Recovered: NOT skipped as no_references, and both entries parsed.
    assert res.skipped_reason != "no_references"
    assert not res.skipped_reason
    assert res.entries == 2


# ===========================================================================
# acceptance milestone (closes gap #2) — CLI, offline (FakeProvider + cache)
# ===========================================================================

def _milestone_setup(slug, monkeypatch):
    """Project with A (bridged references), B, C all included; provider edge A->B
    seeded so cite project links only A->B; A's references cite B + C by DOI.

    B/C carry the SAME openalex ids the FakeProvider deterministically mints for
    their DOIs — under Build A's ch3 blocker a disagreeing W id would (correctly)
    refuse the merge and route to review instead of silently over-merging, which
    is not what this milestone exercises."""
    from seedgraph.providers.fake import _wid

    monkeypatch.setenv("SEEDGRAPH_FAKE_PROVIDERS", "1")
    h, a, md, _src = _citing_paper(slug)
    w_blue, w_green = _wid("10.1111/blue"), _wid("10.2222/green")
    b = service.add_work(h, ids={"doi": "10.1111/blue", "openalex": w_blue}, title="Paper B").work_id
    c = service.add_work(h, ids={"doi": "10.2222/green", "openalex": w_green}, title="Paper C").work_id

    # Provider tier: A references ONLY B (so A->C is parsed-only adds-coverage).
    init_cache_db(None)
    cache_conn = open_cache_db(None)
    try:
        cache_put(
            cache_conn, provider="openalex",
            request_key=key_referenced_works({"openalex_id": "W_A"}),
            response=[{"openalex_id": w_blue}],
        )
    finally:
        cache_conn.close()
    return h, a, b, c


def test_milestone_confirm_disagreement(monkeypatch):
    """``seedgraph cite disagreements`` shows ≥1 (source, target) pair where the
    parsed tier confirms the provider tier (both provenances, winner parsed)."""
    h, a, b, c = _milestone_setup("mileconfirm", monkeypatch)
    assert runner.invoke(app, ["cite", "project", "mileconfirm"]).exit_code == 0
    pr = runner.invoke(app, ["cite", "parse", "mileconfirm"])
    assert pr.exit_code == 0, pr.output

    dr = runner.invoke(app, ["cite", "disagreements", "mileconfirm"])
    assert dr.exit_code == 0, dr.output
    assert f"{a}\t{b}\t" in dr.output
    assert "parsed_bibliography,provider_reference" in dr.output
    # The line names parsed_bibliography as the authoritative winner.
    line = next(ln for ln in dr.output.splitlines() if ln.startswith(f"{a}\t{b}\t"))
    assert line.endswith("parsed_bibliography")


def test_milestone_adds_coverage(monkeypatch):
    """``seedgraph cite table`` reflects an included→target citation the provider
    tier missed (a parsed-only edge), verified via ``cite table`` (criterion 3)."""
    h, a, b, c = _milestone_setup("mileadds", monkeypatch)
    assert runner.invoke(app, ["cite", "project", "mileadds"]).exit_code == 0
    assert runner.invoke(app, ["cite", "parse", "mileadds"]).exit_code == 0

    tr = runner.invoke(app, ["cite", "table", "mileadds"])
    assert tr.exit_code == 0, tr.output
    # A->C is the parsed-only adds-coverage edge (both included -> appears in table).
    assert f"{a}\t{c}\t" in tr.output
    # ...and it is NOT a disagreement (single provenance).
    dr = runner.invoke(app, ["cite", "disagreements", "mileadds"])
    assert f"{a}\t{c}\t" not in dr.output


def test_milestone_rerun_noop_same_run(monkeypatch):
    """Re-running ``cite parse`` on unchanged markdown under the same run is a
    no-op (criterion 7a)."""
    h, a, b, c = _milestone_setup("milenoop", monkeypatch)
    runner.invoke(app, ["cite", "project", "milenoop"])
    runner.invoke(app, ["cite", "parse", "milenoop"])

    def _counts():
        conn = _project_conn(h)
        try:
            refs = conn.execute("SELECT COUNT(*) FROM reference_entries").fetchone()[0]
            edges = conn.execute(
                "SELECT COUNT(*) FROM citation_edges WHERE provenance='parsed_bibliography'"
            ).fetchone()[0]
            return refs, edges
        finally:
            conn.close()

    before = _counts()
    r2 = runner.invoke(app, ["cite", "parse", "milenoop"])
    assert r2.exit_code == 0, r2.output
    assert "(current)" in r2.output  # Stage A skipped
    assert _counts() == before  # no new refs, no duplicate edges


def test_milestone_rerun_projects_under_new_run(monkeypatch):
    """Re-running ``cite parse`` under a freshly minted run still projects the full
    parsed tier into that run (criterion 7b)."""
    h, a, b, c = _milestone_setup("milenewrun", monkeypatch)
    runner.invoke(app, ["cite", "project", "milenewrun"])
    runner.invoke(app, ["cite", "parse", "milenewrun"])

    r2 = ensure_run("milenewrun", root=h.root)
    pr = runner.invoke(app, ["cite", "parse", "milenewrun", "--run-id", r2])
    assert pr.exit_code == 0, pr.output

    conn = _project_conn(h)
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM citation_edges WHERE provenance='parsed_bibliography' AND run_id=?", (r2,)
        ).fetchone()[0]
    finally:
        conn.close()
    assert n == 2  # full parsed tier projected into the new run


def test_cite_parse_prints_progress_heartbeat(monkeypatch):
    """``cite parse`` over 2 included+bridged works prints an ``N/M works`` stdout
    heartbeat per work AND the unchanged final summary line (shared Progress; the
    per-work state string — e.g. ``(current)`` — is preserved)."""
    monkeypatch.setenv("SEEDGRAPH_FAKE_PROVIDERS", "1")
    h, a, _md, _src = _citing_paper("progcli")
    b = service.add_work(
        h, ids={"doi": "10.3000/bbb", "openalex": "W_BB"}, title="Paper BB"
    ).work_id
    _convert_bridge_sections(h, b, REFS_MD, slug_file="cit2")

    pr = runner.invoke(app, ["cite", "parse", "progcli"])
    assert pr.exit_code == 0, pr.output
    # Two works → an N/M heartbeat for each.
    assert "1/2 works" in pr.output
    assert "2/2 works" in pr.output
    # The per-work state string still surfaces (backward-compat with the old line).
    assert "(reparsed)" in pr.output or "(current)" in pr.output
    # The final summary line is preserved verbatim (works=2 …).
    assert "works=2" in pr.output
