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

import re
import sqlite3

import pytest
from typer.testing import CliRunner

from seedgraph.citation.bib_parser import (
    BibEntry,
    _clean_entry_line,
    _is_discrete_sublist_label,
    parse_references,
)
from seedgraph.citation.resolve import resolve_references
from seedgraph.project import identity as identity_mod
from seedgraph.sections.parser import parse_sections

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


# ===========================================================================
# parsed edges + reference_id
# ===========================================================================


# ===========================================================================
# union + authority + disagreement (confirm path)
# ===========================================================================


# ===========================================================================
# review routing
# ===========================================================================


# ===========================================================================
# Stage-A / Stage-B idempotency
# ===========================================================================


# ===========================================================================
# cross-run FK-safe reparse (regression for must-fix #1)
# ===========================================================================


# ===========================================================================
# atomicity (§6.4)
# ===========================================================================


# ===========================================================================
# shareability boundary
# ===========================================================================


# ===========================================================================
# skip paths
# ===========================================================================


#: A references section whose ONLY heading is a bold-only ``**References**`` line —
#: never an ATX ``document_sections`` row (the sections parser is ATX-only).
BOLD_REFS_MD = (
    "# Paper A\n\n"
    "This paper builds on prior work in the field.\n\n"
    "**References**\n\n"
    "Blue, R. (2018). The blue widget framework. Widget Journal. https://doi.org/10.1111/blue\n\n"
    "Green, S. (2019). Green gadget analysis methods. Gadget Review. https://doi.org/10.2222/green\n"
)


# ===========================================================================
# acceptance milestone (closes gap #2) — CLI, offline (FakeProvider + cache)
# ===========================================================================


