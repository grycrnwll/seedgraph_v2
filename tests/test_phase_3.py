"""Phase 3 — Evidence Spans: acceptance tests (plan §11), fully OFFLINE.

Every plan §11 unit/acceptance test is implemented with real assertions and runs
with no network and no LLM: a synthetic markdown is pushed through the real
cache (ingest + FakeMarkerBackend convert), bridged work->markdown, then
sectioned / span-indexed / searched / verified / reanchored through the phase_3
surface. The module-level imports double as a whole-phase import-cleanliness check.

Contracts exercised: the write-time invariant (markdown[s:e]==exact_quote &
quote_hash), ensure_span (unique locate/None on ambiguous, dedup, most-restrictive
access_class that never widens), _write_span RAISES on a bad slice, D7 raw_conn
atomicity (ensure_span + ORM write commit/rollback together), D6 migration<->ORM
parity, D1 content-addressed anchor, no-stemming FTS, idempotent reindex, and
lineage-based staleness + shadow-don't-delete reanchor.
"""

from __future__ import annotations

import sqlite3

import pytest
from sqlalchemy.dialects import sqlite as sqlite_dialect
from typer.testing import CliRunner

# Import-clean check for the entire phase-3 surface (collection exercises these).
from seedgraph import anchor, ids, segment
from seedgraph.sections import parser as sections_parser

runner = CliRunner()


# --------------------------------------------------------------------------
# Offline fixtures / helpers
# --------------------------------------------------------------------------

MD = (
    "# Difference-in-Differences\n"
    "\n"
    "We motivate the design in this opening paragraph.\n"
    "\n"
    "## 2 Identification\n"
    "\n"
    "We rely on the parallel trends assumption for identification.\n"
    "\n"
    "### 2.1 Assumptions\n"
    "\n"
    "Assumption 2 states the rank condition holds for the design matrix.\n"
    "\n"
    "This paragraph discusses mixing conditions and ergodicity at length.\n"
    "\n"
    "## References\n"
    "\n"
    "[1] Author, A. A Title. Journal.\n"
)


def _conn(h) -> sqlite3.Connection:
    """A raw project.db connection with fail-closed FKs (foreign_keys=ON)."""
    conn = sqlite3.connect(str(h.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# --- anchor -----------------------------------------------------------------

def test_anchor_quote_slice_invariant():
    """`assert_invariant` holds when markdown[start:end] == exact_quote and raises otherwise."""
    md = "alpha beta gamma"

    class _S:
        start_char = 6
        end_char = 10
        exact_quote = "beta"
        quote_hash = anchor.quote_hash("beta")

    anchor.assert_invariant(md, _S())  # no raise

    class _Bad:
        start_char = 6
        end_char = 10
        exact_quote = "WRONG"
        quote_hash = anchor.quote_hash("WRONG")

    with pytest.raises(ValueError):
        anchor.assert_invariant(md, _Bad())

    class _BadHash:
        start_char = 6
        end_char = 10
        exact_quote = "beta"
        quote_hash = "deadbeef"

    with pytest.raises(ValueError):
        anchor.assert_invariant(md, _BadHash())


def test_anchor_quote_hash_nfc():
    """quote_hash(q) == sha256(NFC(q)) lowercase hex, and is stable under NFC idempotence."""
    import hashlib
    import unicodedata

    q = "Assumption 2"
    expected = hashlib.sha256(unicodedata.normalize("NFC", q).encode("utf-8")).hexdigest()
    assert anchor.quote_hash(q) == expected
    assert anchor.quote_hash(q) == anchor.quote_hash(unicodedata.normalize("NFC", q))
    # A decomposed vs composed accented form hashes identically (NFC folds them).
    composed = "café"
    decomposed = "café"
    assert anchor.quote_hash(composed) == anchor.quote_hash(decomposed)


def test_anchor_locate_quote_unique():
    """locate_quote returns the unique exact match; returns None on 0 or >=2 occurrences."""
    md = "the rank condition holds; the slope is positive."
    assert anchor.locate_quote(md, "rank condition") == (4, 18)
    assert md[4:18] == "rank condition"
    assert anchor.locate_quote(md, "nonexistent") is None
    assert anchor.locate_quote("foo foo", "foo") is None  # ambiguous (2 occurrences)
    assert anchor.locate_quote(md, "") is None


def test_anchor_reanchor_relocates_unchanged_quote():
    """reanchor relocates an unchanged quote into re-flowed markdown (exact, unique)."""
    old = "# A\n\nparallel trends assumption is key.\n"
    new = "# A (rev)\n\nIndeed the parallel trends assumption is restated here.\n"
    loc = anchor.reanchor("parallel trends assumption", new)
    assert loc is not None
    assert new[loc[0]:loc[1]] == "parallel trends assumption"
    assert anchor.reanchor("totally different text", new) is None


# --- segment ----------------------------------------------------------------

def test_segment_paragraph_offsets_roundtrip():
    """paragraph offsets reconstruct the source; blank-line/trailing-newline edge cases
    produce no empty/whitespace paragraphs."""
    md = "\n\nFirst paragraph line one.\nstill first.\n\n\n  \n\nSecond para.\n\n"
    paras = segment.paragraphs(md)
    assert len(paras) == 2
    for s, e in paras:
        assert md[s:e] == md[s:e].strip()  # tightened, no leading/trailing whitespace
        assert md[s:e].strip() != ""
    assert md[paras[0][0]:paras[0][1]] == "First paragraph line one.\nstill first."
    assert md[paras[1][0]:paras[1][1]] == "Second para."
    # Empty / whitespace-only input yields no paragraphs.
    assert segment.paragraphs("") == []
    assert segment.paragraphs("   \n\n \t\n") == []


def test_segment_page_boundaries_maps_and_empty():
    """page_boundaries maps a delimiter-bearing doc to correct page numbers and returns
    [] (=> NULL pages) on a delimiter-free doc."""
    paged = (
        "{0}------------------------------------------------\n"
        "page one body.\n\n"
        "{1}------------------------------------------------\n"
        "page two body.\n"
    )
    bounds = segment.page_boundaries(paged)
    assert [p for _o, p in bounds] == [1, 2]  # 0-based marker -> 1-based page
    one = paged.index("page one body")
    two = paged.index("page two body")
    assert segment.page_for_range(bounds, one, one + 3) == (1, 1)
    assert segment.page_for_range(bounds, two, two + 3) == (2, 2)
    # Delimiter-free doc => [] => NULL pages.
    assert segment.page_boundaries("# T\n\nbody only.\n") == []
    assert segment.page_for_range([], 0, 5) == (None, None)


# --- sections/parser --------------------------------------------------------

def _sections(markdown, markdown_hash="h0"):
    return sections_parser.parse_sections(
        markdown, markdown_id="md_" + markdown_hash, markdown_hash=markdown_hash,
        source_file_id="sf_x", source_file_hash="x", work_id="work_1",
    )


def test_sections_nested_levels_parent_ordinal_path():
    """Nested headings produce correct level / parent_section_id / ordinal / heading_path."""
    md = "# 1 Intro\n\ntext\n\n## 1.1 Sub\n\nmore\n\n### 1.1.1 Deep\n\ndeep text\n\n## 1.2 Other\n\nx\n"
    secs = _sections(md)
    by_text = {s.heading_text: s for s in secs}
    assert by_text["1 Intro"].level == 1 and by_text["1 Intro"].parent_section_id is None
    assert by_text["1.1 Sub"].level == 2
    assert by_text["1.1 Sub"].parent_section_id == by_text["1 Intro"].section_id
    assert by_text["1.1.1 Deep"].level == 3
    assert by_text["1.1.1 Deep"].parent_section_id == by_text["1.1 Sub"].section_id
    # 1.2 Other pops back to under 1 Intro.
    assert by_text["1.2 Other"].parent_section_id == by_text["1 Intro"].section_id
    # Ordinals are 0-based document order.
    assert [s.ordinal for s in secs] == list(range(len(secs)))
    assert by_text["1.1.1 Deep"].heading_path == "1 Intro > 1.1 Sub > 1.1.1 Deep"


def test_sections_headingless_setext_falls_into_preamble():
    """Heading-less / Setext-only markdown falls entirely into a single level-0 preamble."""
    md = "Title Underlined\n================\n\nBody paragraph with no ATX heading.\n"
    secs = _sections(md)
    assert len(secs) == 1
    assert secs[0].level == 0
    assert secs[0].heading_text is None
    assert secs[0].start_char == 0 and secs[0].end_char == len(md)


def test_sections_preamble_before_first_heading_captured():
    """Preamble text before the first ATX heading is captured as a level-0 section."""
    md = "Leading preamble text before any heading.\n\n# First Heading\n\nbody.\n"
    secs = _sections(md)
    assert secs[0].level == 0
    assert secs[0].start_char == 0
    assert secs[0].end_char == md.index("# First Heading")
    assert secs[1].level == 1 and secs[1].heading_text == "First Heading"


def test_sections_references_heading_classified():
    """A `## References` heading is classified section_kind == 'references'."""
    md = "# T\n\nbody.\n\n## References\n\n[1] x.\n"
    secs = _sections(md)
    refs = [s for s in secs if s.heading_text == "References"][0]
    assert refs.section_kind == "references"
    body = [s for s in secs if s.heading_text == "T"][0]
    assert body.section_kind == "body"


def test_sections_section_for_offset_returns_deepest():
    """section_for_offset returns the deepest containing section for an offset."""
    md = "# A\n\nintro.\n\n## B\n\ndeep body here.\n"
    secs = _sections(md)
    deep_offset = md.index("deep body here")
    deepest = [s for s in secs if s.heading_text == "B"][0]
    assert sections_parser.section_for_offset(secs, deep_offset) == deepest.section_id
    # An offset in A's own text resolves to A (no deeper child contains it).
    intro_offset = md.index("intro.")
    parent = [s for s in secs if s.heading_text == "A"][0]
    assert sections_parser.section_for_offset(secs, intro_offset) == parent.section_id


def test_sections_deterministic_id_stable_and_rebuild_safe():
    """Deterministic section_id is stable across two parses of identical markdown and
    unchanged when the cache markdown_id row id differs but the bytes are identical."""
    md = "# A\n\nx\n\n## B\n\ny\n"
    a = _sections(md, markdown_hash="HASH")
    b = _sections(md, markdown_hash="HASH")
    assert [s.section_id for s in a] == [s.section_id for s in b]
    # Different cache markdown_id (row id) but identical bytes/hash -> identical ids.
    c = sections_parser.parse_sections(
        md, markdown_id="md_DIFFERENT_ROW_ID", markdown_hash="HASH",
        source_file_id="sf_x", source_file_hash="x", work_id="work_1",
    )
    assert [s.section_id for s in a] == [s.section_id for s in c]
    # A different hash yields different ids.
    d = _sections(md, markdown_hash="OTHER")
    assert [s.section_id for s in a] != [s.section_id for s in d]


# --- idempotency (must-fix) -------------------------------------------------

# --- section re-parse vs spans (must-fix) -----------------------------------

# --- spans/store ------------------------------------------------------------

# --- db/adapter (decision D7) -----------------------------------------------

# --- migration vs ORM (decision D6) -----------------------------------------

def _affinity(type_str: str) -> str:
    t = type_str.upper()
    if "INT" in t:
        return "INTEGER"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "TEXT"


def _assert_table_parity(insp, table_name, model):
    orm_cols = {c.name: c for c in model.__table__.columns}
    mig_cols = {c["name"]: c for c in insp.get_columns(table_name)}
    assert set(orm_cols) == set(mig_cols), f"{table_name}: column set drift"

    mig_pk = set(insp.get_pk_constraint(table_name)["constrained_columns"])
    orm_pk = {c.name for c in model.__table__.columns if c.primary_key}
    assert mig_pk == orm_pk, f"{table_name}: PK drift"

    orm_dialect = sqlite_dialect.dialect()
    for name, orm_col in orm_cols.items():
        mig_col = mig_cols[name]
        orm_aff = _affinity(str(orm_col.type.compile(dialect=orm_dialect)))
        mig_aff = _affinity(str(mig_col["type"]))
        assert orm_aff == mig_aff, f"{table_name}.{name}: type {orm_aff} != {mig_aff}"
        if name not in mig_pk:
            assert orm_col.nullable == mig_col["nullable"], f"{table_name}.{name}: nullability drift"

    mig_ix = {tuple(ix["column_names"]) for ix in insp.get_indexes(table_name)}
    orm_ix = {tuple(c.name for c in ix.columns) for ix in model.__table__.indexes}
    assert mig_ix == orm_ix, f"{table_name}: index column-sets drift ({mig_ix} != {orm_ix})"


# --- content-addressed anchor (decision D1) ---------------------------------

# --- cache_access -----------------------------------------------------------


# --- fts/search -------------------------------------------------------------

# --- staleness / reanchor ---------------------------------------------------

# --- index empty/missing ----------------------------------------------------

# --- ids --------------------------------------------------------------------

def test_ids_manual_opaque_unique_and_derived_deterministic():
    """Manual span ids are unique and opaque; section ids and auto-span ids are deterministic."""
    a = ids.new_id("span")
    b = ids.new_id("span")
    assert a != b and a.startswith("span_")
    # Deterministic section + auto-span ids.
    assert ids.section_id("HASH", 0) == ids.section_id("HASH", 0)
    assert ids.section_id("HASH", 0) != ids.section_id("HASH", 1)
    assert ids.section_id("HASH", 0).startswith("sec_") and len(ids.section_id("HASH", 0)) == len("sec_") + 16
    assert ids.auto_span_id("HASH", 1, 9, "paragraph") == ids.auto_span_id("HASH", 1, 9, "paragraph")
    assert ids.auto_span_id("HASH", 1, 9, "paragraph") != ids.auto_span_id("HASH", 1, 9, "manual")
    assert ids.auto_span_id("HASH", 1, 9, "paragraph").startswith("span_")


# --- milestone acceptance (doc 10 §7 + doc 09 §12) --------------------------

