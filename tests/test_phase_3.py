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

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects import sqlite as sqlite_dialect
from sqlmodel import Session, select
from typer.testing import CliRunner

# Import-clean check for the entire phase-3 surface (collection exercises these).
from seedgraph import anchor, cache_access, doctor_spans, ids, segment
from seedgraph.acquisition.bridge import resolve_work_markdown, write_bridge
from seedgraph.cache.convert import convert_source_file
from seedgraph.cache.ingest import ingest_file
from seedgraph.cache.marker_backend import FakeMarkerBackend
from seedgraph.cli import app
from seedgraph.db.adapter import raw_conn
from seedgraph.db.connection import open_cache_db
from seedgraph.db.migrations import run_migrations
from seedgraph.db.project_models import DocumentSection, EvidenceSpan, Work
from seedgraph.fts import schema as fts_schema
from seedgraph.fts import search as fts_search
from seedgraph.ids import sha256_hex
from seedgraph.project import service
from seedgraph.sections import parser as sections_parser
from seedgraph.sections import store as sections_store
from seedgraph.spans import index as spans_index
from seedgraph.spans import store as spans_store
from seedgraph.vocab import AccessClass, AcquisitionMethod

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


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_doc(slug, markdown=MD, *, access_class=AccessClass.open_access, doi="10.1/x", force=False):
    """Create a project + one bridged converted markdown; return (handle, work_id, md_doc)."""
    h = service.create_project(slug)
    w = service.add_work(h, ids={"doi": doi}, title="W " + slug)
    method = (
        AcquisitionMethod.open_access_fetch
        if access_class == AccessClass.open_access
        else AcquisitionMethod.upload
    )
    p = Path(os.environ["SEEDGRAPH_HOME"]) / f"{slug}.pdf"
    p.write_bytes(b"%PDF-1.4 " + slug.encode() + b" body content with several words here")
    src = ingest_file(p, access_class=access_class, acquisition_method=method, root=None)
    md = convert_source_file(
        src.source_file_id, backend=FakeMarkerBackend(markdown=markdown), force=force, root=None
    )
    with Session(h.engine) as s:
        write_bridge(
            s, work_id=w.work_id, source_file_id=src.source_file_id, file_hash=src.file_hash,
            markdown_id=md.markdown_id, markdown_hash=md.markdown_hash,
            acquisition_method=method.value,
        )
        s.commit()
    return h, w.work_id, md


def _reconvert(work_id, slug_file, new_markdown):
    """Reconvert the same source file to NEW markdown (same lineage); return md_doc.

    Looks up the source_file_id behind ``slug_file`` (the ``_make_doc`` pdf), then
    forces a fresh conversion — same source_file_id/hash, new markdown_id/hash.
    """
    p = Path(os.environ["SEEDGRAPH_HOME"]) / f"{slug_file}.pdf"
    raw = p.read_bytes()
    source_file_id = "sf_" + sha256_hex(raw)
    return convert_source_file(
        source_file_id, backend=FakeMarkerBackend(markdown=new_markdown), force=True, root=None
    )


def _conn(h) -> sqlite3.Connection:
    """A raw project.db connection with fail-closed FKs (foreign_keys=ON)."""
    conn = sqlite3.connect(str(h.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _counts(h):
    conn = sqlite3.connect(str(h.db_path))
    try:
        spans = conn.execute("SELECT COUNT(*) FROM evidence_spans").fetchone()[0]
        fts = conn.execute("SELECT COUNT(*) FROM span_fts").fetchone()[0]
        secs = conn.execute("SELECT COUNT(*) FROM document_sections").fetchone()[0]
        return spans, fts, secs
    finally:
        conn.close()


def _index(h, work_id, md):
    cache_conn = cache_access.open_cache_ro(None)
    try:
        with Session(h.engine) as s:
            conn = raw_conn(s)
            n = spans_index.index_document(
                conn, cache_conn, None, work_id=work_id, markdown_id=md.markdown_id
            )
            s.commit()
    finally:
        cache_conn.close()
    return n


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

def test_index_document_idempotent_same_ids_count_fts():
    """Running index_document twice on the same markdown yields the same auto-span ids,
    the same row count, and the same number of span_fts rows (no duplicates)."""
    h, wid, md = _make_doc("idem")
    n1 = _index(h, wid, md)
    ids1 = sorted(
        r[0] for r in sqlite3.connect(str(h.db_path)).execute(
            "SELECT span_id FROM evidence_spans WHERE span_kind='paragraph'"
        )
    )
    c1 = _counts(h)
    n2 = _index(h, wid, md)
    ids2 = sorted(
        r[0] for r in sqlite3.connect(str(h.db_path)).execute(
            "SELECT span_id FROM evidence_spans WHERE span_kind='paragraph'"
        )
    )
    c2 = _counts(h)
    assert n1 == n2 and n1 > 0
    assert ids1 == ids2  # deterministic auto-span ids
    assert c1 == c2  # same spans / fts / sections counts (no duplicates)


# --- section re-parse vs spans (must-fix) -----------------------------------

def test_section_reparse_no_fk_violation_spans_resolve(monkeypatch):
    """Bumping section_parser_version and re-parsing under PRAGMA foreign_keys=ON causes
    no FK violation; existing spans' section_id is re-resolved in the index txn and still
    resolves."""
    h, wid, md = _make_doc("reparse")
    _index(h, wid, md)
    before = sqlite3.connect(str(h.db_path)).execute(
        "SELECT span_id, section_id FROM evidence_spans WHERE span_kind='paragraph'"
    ).fetchall()
    assert all(sec is not None for _sid, sec in before)

    # Bump the parser version and re-parse/re-index under foreign_keys=ON.
    monkeypatch.setattr(sections_parser, "SECTION_PARSER_VERSION", "secparse-2")
    monkeypatch.setattr(sections_store, "SECTION_PARSER_VERSION", "secparse-2")
    _index(h, wid, md)  # would raise on an FK violation

    conn = _conn(h)
    try:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        # Every span's section_id still resolves to an existing section (re-resolved).
        dangling = conn.execute(
            "SELECT COUNT(*) FROM evidence_spans e WHERE e.section_id IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM document_sections d WHERE d.section_id = e.section_id)"
        ).fetchone()[0]
        assert dangling == 0
        version = conn.execute(
            "SELECT DISTINCT section_parser_version FROM document_sections"
        ).fetchall()
        assert version == [("secparse-2",)]
    finally:
        conn.close()


# --- spans/store ------------------------------------------------------------

def test_ensure_span_unique_locate_returns_none_on_ambiguous():
    """ensure_span locates the unique exact quote and returns None on no/ambiguous match."""
    md_text = "# T\n\nThe repeated phrase here. And the repeated phrase here again.\n\nA unique sentence only once.\n"
    h, wid, md = _make_doc("ensure_amb", md_text)
    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h)
    try:
        # Unique quote -> a span id.
        sid = spans_store.ensure_span(
            conn, cache_conn, None, markdown_id=md.markdown_id, work_id=wid,
            exact_quote="A unique sentence only once.",
        )
        assert sid is not None
        # Ambiguous (>=2 occurrences) -> None.
        assert spans_store.ensure_span(
            conn, cache_conn, None, markdown_id=md.markdown_id, work_id=wid,
            exact_quote="repeated phrase here",
        ) is None
        # No match -> None.
        assert spans_store.ensure_span(
            conn, cache_conn, None, markdown_id=md.markdown_id, work_id=wid,
            exact_quote="not present at all",
        ) is None
        conn.commit()
    finally:
        conn.close()
        cache_conn.close()


def test_ensure_span_dedups_on_markdown_quotehash():
    """ensure_span dedups on (markdown_id, quote_hash) — re-calling returns the same span_id."""
    h, wid, md = _make_doc("ensure_dedup")
    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h)
    try:
        a = spans_store.ensure_span(
            conn, cache_conn, None, markdown_id=md.markdown_id, work_id=wid,
            exact_quote="parallel trends assumption",
        )
        b = spans_store.ensure_span(
            conn, cache_conn, None, markdown_id=md.markdown_id, work_id=wid,
            exact_quote="parallel trends assumption",
        )
        assert a is not None and a == b
        count = conn.execute(
            "SELECT COUNT(*) FROM evidence_spans WHERE quote_hash = ?",
            (anchor.quote_hash("parallel trends assumption"),),
        ).fetchone()[0]
        assert count == 1
        conn.commit()
    finally:
        conn.close()
        cache_conn.close()


def test_ensure_span_stamps_most_restrictive_access_class():
    """ensure_span stamps AccessClass.most_restrictive(source, caller); a too-permissive
    caller floor never widens below the source class (no leak)."""
    h, wid, md = _make_doc("ensure_ac", access_class=AccessClass.user_supplied_private)
    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h)
    try:
        # Caller hands a too-permissive 'open_access'; the private source must win.
        sid = spans_store.ensure_span(
            conn, cache_conn, None, markdown_id=md.markdown_id, work_id=wid,
            exact_quote="parallel trends assumption", access_class="open_access",
        )
        ac = conn.execute(
            "SELECT access_class FROM evidence_spans WHERE span_id = ?", (sid,)
        ).fetchone()[0]
        assert ac == "user_supplied_private"
        conn.commit()
    finally:
        conn.close()
        cache_conn.close()

    # An OA source with no caller floor stamps open_access (source class flows through).
    # Distinct markdown content => its own content-addressed markdown row (no sharing).
    oa_md = MD.replace("Difference-in-Differences", "DiD (open-access variant)")
    h2, wid2, md2 = _make_doc("ensure_ac_oa", oa_md, access_class=AccessClass.open_access, doi="10.2/oa")
    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h2)
    try:
        sid = spans_store.ensure_span(
            conn, cache_conn, None, markdown_id=md2.markdown_id, work_id=wid2,
            exact_quote="parallel trends assumption",
        )
        ac = conn.execute(
            "SELECT access_class FROM evidence_spans WHERE span_id = ?", (sid,)
        ).fetchone()[0]
        assert ac == "open_access"
        conn.commit()
    finally:
        conn.close()
        cache_conn.close()


def test_write_span_enforces_invariant_raises_on_bad_slice():
    """_write_span enforces the write-time invariant and RAISES on a slice that does not
    match the supplied exact_quote."""
    h, wid, md = _make_doc("write_bad")
    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h)
    try:
        # A good slice writes fine.
        good = md.markdown_id
        text = cache_access.read_markdown(cache_conn, None, good).text
        start = text.index("parallel trends")
        sid = spans_store._write_span(
            conn, cache_conn, None, markdown_id=good, work_id=wid,
            start=start, end=start + len("parallel trends"),
            exact_quote="parallel trends",
        )
        assert sid is not None
        # A slice that doesn't match exact_quote RAISES (the offset writer never guesses).
        with pytest.raises(ValueError):
            spans_store._write_span(
                conn, cache_conn, None, markdown_id=good, work_id=wid,
                start=start, end=start + 5, exact_quote="WRONG QUOTE",
            )
        conn.commit()
    finally:
        conn.close()
        cache_conn.close()


def test_get_span_resolved_text_equals_exact_quote():
    """get_span returns resolved live text equal to exact_quote for an anchored span."""
    h, wid, md = _make_doc("getspan")
    _index(h, wid, md)
    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h)
    try:
        sid = conn.execute(
            "SELECT span_id FROM evidence_spans WHERE exact_quote LIKE '%parallel trends%'"
        ).fetchone()[0]
        span = spans_store.get_span(conn, cache_conn, None, sid)
        assert span.resolved_text == span.exact_quote
        assert "parallel trends" in span.resolved_text
        assert span.heading_path is not None
    finally:
        conn.close()
        cache_conn.close()


def test_access_class_fail_closed_when_lineage_missing():
    """access_class defaults fail-closed to 'user_supplied_private' when lineage/floor is missing."""
    # The vocab floor itself fails closed for None / empty input.
    assert AccessClass.most_restrictive(None) == AccessClass.user_supplied_private
    assert AccessClass.most_restrictive() == AccessClass.user_supplied_private
    assert AccessClass.most_restrictive(None, None) == AccessClass.user_supplied_private
    # And the column default is the most-restrictive class (schema fail-closed).
    h, wid, md = _make_doc("failclosed", access_class=AccessClass.user_supplied_private, doi="10.9/p")
    _index(h, wid, md)
    classes = {
        r[0] for r in sqlite3.connect(str(h.db_path)).execute(
            "SELECT DISTINCT access_class FROM evidence_spans"
        )
    }
    assert classes == {"user_supplied_private"}


# --- db/adapter (decision D7) -----------------------------------------------

def test_raw_conn_ensure_span_and_orm_commit_atomically():
    """raw_conn(session) returns the DBAPI conn bound to the session txn; an ensure_span
    INSERT and an ORM write through the same session commit atomically in one transaction."""
    h, wid, md = _make_doc("d7commit")
    cache_conn = cache_access.open_cache_ro(None)
    try:
        with Session(h.engine) as session:
            conn = raw_conn(session)
            sid = spans_store.ensure_span(
                conn, cache_conn, None, markdown_id=md.markdown_id, work_id=wid,
                exact_quote="parallel trends assumption",
            )
            # An ORM write in the SAME session/txn.
            work = session.exec(select(Work).where(Work.work_id == wid)).one()
            work.canonical_title = "RENAMED ATOMIC"
            session.add(work)
            session.commit()
    finally:
        cache_conn.close()

    check = sqlite3.connect(str(h.db_path))
    try:
        assert check.execute(
            "SELECT COUNT(*) FROM evidence_spans WHERE span_id = ?", (sid,)
        ).fetchone()[0] == 1
        assert check.execute(
            "SELECT canonical_title FROM works WHERE work_id = ?", (wid,)
        ).fetchone()[0] == "RENAMED ATOMIC"
    finally:
        check.close()


def test_raw_conn_rollback_hides_span_and_orm_write():
    """Rolling back the session hides both the span row and the ORM write (no half-write)."""
    h, wid, md = _make_doc("d7rollback")
    cache_conn = cache_access.open_cache_ro(None)
    try:
        with Session(h.engine) as session:
            conn = raw_conn(session)
            sid = spans_store.ensure_span(
                conn, cache_conn, None, markdown_id=md.markdown_id, work_id=wid,
                exact_quote="parallel trends assumption",
            )
            assert sid is not None
            work = session.exec(select(Work).where(Work.work_id == wid)).one()
            work.canonical_title = "SHOULD NOT PERSIST"
            session.add(work)
            session.flush()  # the writes are live in-txn...
            session.rollback()  # ...then rolled back together.
    finally:
        cache_conn.close()

    check = sqlite3.connect(str(h.db_path))
    try:
        assert check.execute(
            "SELECT COUNT(*) FROM evidence_spans WHERE span_id = ?", (sid,)
        ).fetchone()[0] == 0  # span gone
        title = check.execute(
            "SELECT canonical_title FROM works WHERE work_id = ?", (wid,)
        ).fetchone()[0]
        assert title != "SHOULD NOT PERSIST"  # ORM write gone too
    finally:
        check.close()


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


def test_migration_orm_metadata_parity(tmp_path):
    """Applying project/0005_evidence_spans.sql then reflecting the DB schema matches the
    SQLModel ORM metadata (table/column/index parity); create_all is never invoked."""
    db = tmp_path / "parity.db"
    conn = sqlite3.connect(str(db))
    run_migrations(conn, "project")  # the migrate step authors schema (D6)
    conn.close()

    insp = inspect(create_engine(f"sqlite:///{db.as_posix()}"))
    assert "document_sections" in insp.get_table_names()
    assert "evidence_spans" in insp.get_table_names()
    # 0005 ships span_fts ONLY (claim_fts / note_fts are owned by phase_4's 0007).
    assert "span_fts" in insp.get_table_names()

    _assert_table_parity(insp, "document_sections", DocumentSection)
    _assert_table_parity(insp, "evidence_spans", EvidenceSpan)

    # document_sections UNIQUE(markdown_id, ordinal).
    uq = [set(u["column_names"]) for u in insp.get_unique_constraints("document_sections")]
    assert {"markdown_id", "ordinal"} in uq

    # FKs: both tables -> works(work_id) (the only real, intra-project FK).
    for table in ("document_sections", "evidence_spans"):
        fks = insp.get_foreign_keys(table)
        assert all(fk["referred_table"] == "works" for fk in fks)
        assert any(fk["constrained_columns"] == ["work_id"] for fk in fks)


# --- content-addressed anchor (decision D1) ---------------------------------

def test_content_addressed_markdown_id_equals_hash():
    """A span's markdown_id == 'md_' + sha256(markdown_bytes) == its denormalized
    markdown_hash; cross-db join on markdown_id and on markdown_hash resolve the same row."""
    h, wid, md = _make_doc("d1")
    _index(h, wid, md)
    expected_hash = sha256_hex(MD.encode("utf-8"))
    conn = _conn(h)
    try:
        row = conn.execute(
            "SELECT DISTINCT markdown_id, markdown_hash FROM evidence_spans"
        ).fetchone()
    finally:
        conn.close()
    markdown_id, markdown_hash = row
    assert markdown_id == "md_" + expected_hash
    assert markdown_hash == expected_hash
    assert markdown_id == "md_" + markdown_hash  # id IS the prefixed hash

    cache_conn = cache_access.open_cache_ro(None)
    try:
        by_id = cache_conn.execute(
            "SELECT storage_uri FROM markdown_documents WHERE markdown_id = ?", (markdown_id,)
        ).fetchone()[0]
        by_hash = cache_conn.execute(
            "SELECT storage_uri FROM markdown_documents WHERE markdown_hash = ?", (markdown_hash,)
        ).fetchone()[0]
        assert by_id == by_hash  # same cache row resolved either way
    finally:
        cache_conn.close()


# --- cache_access -----------------------------------------------------------

def test_cache_ro_attach_cannot_write():
    """A read-only ATTACH of cache.db cannot write (write attempt raises)."""
    cache_conn = cache_access.open_cache_ro(None)  # migrates then opens mode=ro
    try:
        with pytest.raises(sqlite3.OperationalError):
            cache_conn.execute(
                "INSERT INTO source_files (source_file_id, file_hash, access_class, created_at) "
                "VALUES ('sf_x', 'x', 'open_access', '2020')"
            )
    finally:
        cache_conn.close()


def test_read_markdown_rehash_matches_and_none_on_missing():
    """read_markdown returns bytes whose re-hash equals the stored markdown_hash (and the
    markdown_id minus the 'md_' prefix); returns None on a missing markdown_id."""
    h, wid, md = _make_doc("readmd")
    cache_conn = cache_access.open_cache_ro(None)
    try:
        row = cache_access.read_markdown(cache_conn, None, md.markdown_id)
        assert row is not None
        assert sha256_hex(row.text.encode("utf-8")) == row.markdown_hash
        assert row.markdown_hash == md.markdown_id[len("md_"):]
        assert row.access_class == "open_access"
        assert cache_access.read_markdown(cache_conn, None, "md_does_not_exist") is None
    finally:
        cache_conn.close()


# --- fts/search -------------------------------------------------------------

def test_search_exact_terms_return_correct_spans():
    """Indexing then searching `parallel trends`, `Assumption 2`, `rank condition` returns
    the correct paragraph spans with correct work_id and a resolving section."""
    h, wid, md = _make_doc("ftsterms")
    _index(h, wid, md)
    conn = _conn(h)
    try:
        for term in ("parallel trends", "Assumption 2", "rank condition"):
            hits = fts_search.search_spans(conn, term)
            assert len(hits) >= 1, term
            assert all(hit.work_id == wid for hit in hits)
            assert all(hit.section_id is not None for hit in hits)
            assert any(term.lower() in hit.quote_text.lower() for hit in hits)
    finally:
        conn.close()


def test_search_no_stemming_mixing_not_mix():
    """No-stemming: `mixing` does NOT match `mix` / `mixture` (unicode61, no porter)."""
    h, wid, md = _make_doc("nostem")
    _index(h, wid, md)
    conn = _conn(h)
    try:
        assert len(fts_search.search_spans(conn, "mixing")) >= 1  # the verbatim token
        assert fts_search.search_spans(conn, "mix") == []  # no stemming to the stem
        assert fts_search.search_spans(conn, "mixture") == []
    finally:
        conn.close()


def test_search_phrase_matches_adjacent_tokens():
    """The phrase query "assumption 2" matches adjacent tokens (detail=full phrase query)."""
    h, wid, md = _make_doc("phrase")
    _index(h, wid, md)
    conn = _conn(h)
    try:
        hits = fts_search.search_spans(conn, "assumption 2")
        assert len(hits) >= 1
        assert any("Assumption 2" in hit.quote_text for hit in hits)
        # A non-adjacent pairing that never appears as a phrase returns nothing.
        assert fts_search.search_spans(conn, "assumption ergodicity") == []
    finally:
        conn.close()


def test_search_filters_by_work_and_section_kind():
    """Results filter by work_id and section_kind."""
    h, wid, md = _make_doc("filt1", doi="10.4/a")
    _index(h, wid, md)
    # A second work in the SAME project/home with the same term but DISTINCT content
    # (distinct bytes => distinct content-addressed markdown_id; no span-id collision).
    md_b = MD.replace("Difference-in-Differences", "Second Paper Variant")
    w2 = service.add_work(h, ids={"doi": "10.4/b"}, title="W2")
    p = Path(os.environ["SEEDGRAPH_HOME"]) / "filt2.pdf"
    p.write_bytes(b"%PDF-1.4 filt2 distinct body words here now")
    src2 = ingest_file(p, access_class=AccessClass.open_access, acquisition_method=AcquisitionMethod.open_access_fetch, root=None)
    md2 = convert_source_file(src2.source_file_id, backend=FakeMarkerBackend(markdown=md_b), root=None)
    with Session(h.engine) as s:
        write_bridge(s, work_id=w2.work_id, source_file_id=src2.source_file_id, file_hash=src2.file_hash,
                     markdown_id=md2.markdown_id, markdown_hash=md2.markdown_hash, acquisition_method="open_access_fetch")
        s.commit()
    _index(h, w2.work_id, md2)

    conn = _conn(h)
    try:
        # work filter: only the named work's hits.
        only1 = fts_search.search_spans(conn, "parallel trends", work_id=wid)
        assert only1 and all(hit.work_id == wid for hit in only1)
        # section_kind filter: 'Author' only lives in the references section.
        refs = fts_search.search_spans(conn, "Author", section_kind="references")
        assert len(refs) >= 1
        assert fts_search.search_spans(conn, "Author", section_kind="body") == []
    finally:
        conn.close()


def test_reindex_spans_deletes_prior_rows_no_duplicates():
    """reindex_spans deletes prior rows by markdown_id so no stale duplicate FTS rows remain
    after a re-index (the must-fix)."""
    h, wid, md = _make_doc("reindex")
    _index(h, wid, md)
    conn = _conn(h)
    try:
        spans_n = conn.execute(
            "SELECT COUNT(*) FROM evidence_spans WHERE markdown_id = ?", (md.markdown_id,)
        ).fetchone()[0]
        # Corrupt the FTS with a deliberate duplicate, then reindex by markdown_id.
        conn.execute(
            "INSERT INTO span_fts(quote_text, span_id, markdown_id, work_id, section_id) "
            "SELECT quote_text, span_id, markdown_id, work_id, section_id FROM span_fts "
            "WHERE markdown_id = ? LIMIT 1", (md.markdown_id,),
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM span_fts WHERE markdown_id = ?", (md.markdown_id,)
        ).fetchone()[0] == spans_n + 1
        n = fts_schema.reindex_spans(conn, md.markdown_id)
        assert n == spans_n
        assert conn.execute(
            "SELECT COUNT(*) FROM span_fts WHERE markdown_id = ?", (md.markdown_id,)
        ).fetchone()[0] == spans_n  # no stale duplicate rows
        conn.commit()
    finally:
        conn.close()


# --- staleness / reanchor ---------------------------------------------------

def test_verify_marks_stale_via_lineage_after_md_row_gc():
    """Simulating reconversion (new markdown_id/hash, same source_file_id/hash), verify marks
    spans 'stale' even after the old markdown_id row is removed (lineage-based detection)."""
    h, wid, md = _make_doc("stale")
    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h)
    try:
        sid = spans_store.ensure_span(
            conn, cache_conn, None, markdown_id=md.markdown_id, work_id=wid,
            exact_quote="parallel trends assumption",
        )
        conn.commit()
    finally:
        conn.close()
        cache_conn.close()

    # Reconvert the SAME source to new markdown (new id/hash, same lineage).
    md2 = _reconvert(wid, "stale", MD.replace("Difference-in-Differences", "Diff-in-Diff (rev2)"))
    assert md2.markdown_id != md.markdown_id

    # GC the OLD markdown_documents row so detection must use lineage, not the row id.
    wcache = open_cache_db(None)
    try:
        wcache.execute("DELETE FROM markdown_documents WHERE markdown_id = ?", (md.markdown_id,))
        wcache.commit()
    finally:
        wcache.close()

    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h)
    try:
        assert cache_access.read_markdown(cache_conn, None, md.markdown_id) is None  # GC'd
        status = spans_store.verify_span(conn, cache_conn, None, sid)
        conn.commit()
        assert status == "stale"
        persisted = conn.execute(
            "SELECT anchor_status FROM evidence_spans WHERE span_id = ?", (sid,)
        ).fetchone()[0]
        assert persisted == "stale"  # shadow-don't-delete (row retained, marked stale)
    finally:
        conn.close()
        cache_conn.close()


def test_reanchor_relocates_exact_and_orphans_altered_to_review_queue():
    """reanchor relocates exact matches and routes a deliberately-altered quote to
    review_queue with anchor_status='orphaned'."""
    md_v1 = (
        "# Paper\n\nThe parallel trends assumption holds in this setting.\n\n"
        "The rank condition is satisfied exactly here.\n"
    )
    md_v2 = (
        "# Paper (rev)\n\nIndeed the parallel trends assumption holds throughout.\n\n"
        "The rank requirement is now completely different.\n"
    )
    h, wid, md = _make_doc("reanchor", md_v1)
    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h)
    try:
        keep = spans_store.ensure_span(
            conn, cache_conn, None, markdown_id=md.markdown_id, work_id=wid,
            exact_quote="parallel trends assumption",
        )
        orphan = spans_store.ensure_span(
            conn, cache_conn, None, markdown_id=md.markdown_id, work_id=wid,
            exact_quote="rank condition is satisfied exactly",
        )
        conn.commit()
    finally:
        conn.close()
        cache_conn.close()

    _reconvert(wid, "reanchor", md_v2)  # same lineage, new markdown

    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h)
    try:
        # Mark both stale (superseded by the new markdown), then reanchor.
        spans_store.verify_span(conn, cache_conn, None, keep)
        spans_store.verify_span(conn, cache_conn, None, orphan)
        moved = spans_store.reanchor_spans(conn, cache_conn, None, work_id=wid)
        conn.commit()
        assert moved == 1  # only the surviving quote relocates

        # The altered quote is orphaned and routed to review_queue.
        assert conn.execute(
            "SELECT anchor_status FROM evidence_spans WHERE span_id = ?", (orphan,)
        ).fetchone()[0] == "orphaned"
        rq = conn.execute(
            "SELECT COUNT(*) FROM review_queue WHERE target_id = ? AND status='open'", (orphan,)
        ).fetchone()[0]
        assert rq == 1
        # A NEW anchored span for the surviving quote now exists in the new markdown.
        relocated = conn.execute(
            "SELECT COUNT(*) FROM evidence_spans WHERE exact_quote = ? AND anchor_status='anchored'",
            ("parallel trends assumption",),
        ).fetchone()[0]
        assert relocated == 1
    finally:
        conn.close()
        cache_conn.close()


def test_reanchor_shadow_dont_delete_keeps_original_retrievable():
    """Original-markdown spans remain retrievable after reanchor (shadow-don't-delete)."""
    md_v1 = "# Paper\n\nThe parallel trends assumption holds in this setting.\n"
    md_v2 = "# Paper (rev)\n\nIndeed the parallel trends assumption holds throughout.\n"
    h, wid, md = _make_doc("shadow", md_v1)
    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h)
    try:
        original = spans_store.ensure_span(
            conn, cache_conn, None, markdown_id=md.markdown_id, work_id=wid,
            exact_quote="parallel trends assumption",
        )
        conn.commit()
    finally:
        conn.close()
        cache_conn.close()

    _reconvert(wid, "shadow", md_v2)  # NB: v1 markdown row is NOT GC'd

    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h)
    try:
        spans_store.verify_span(conn, cache_conn, None, original)
        spans_store.reanchor_spans(conn, cache_conn, None, work_id=wid)
        conn.commit()
        # The original (now a shadow) is RETAINED and still retrievable verbatim.
        span = spans_store.get_span(conn, cache_conn, None, original)
        assert span.span_id == original
        assert span.exact_quote == "parallel trends assumption"
        assert span.resolved_text == "parallel trends assumption"  # v1 markdown still present
        assert span.markdown_id == md.markdown_id  # not moved in place
    finally:
        conn.close()
        cache_conn.close()


def test_reanchor_is_idempotent_for_manual_spans_no_duplicates():
    """Re-running reanchor_spans on a relocated MANUAL span writes no duplicate.

    Regression: a manual span gets a fresh new_id() (no deterministic-id guard), and its
    relocated stale original is retained as a shadow (anchor_status stays 'stale'), so a
    second pass re-selects the shadow. Without a (markdown_id, quote_hash) dedup it would
    re-relocate, accumulating duplicate evidence_spans / span_fts rows and inflating
    `moved`. After the fix: second pass moves 0 and counts are unchanged."""
    md_v1 = "# Paper\n\nThe parallel trends assumption holds in this setting.\n"
    md_v2 = "# Paper (rev)\n\nIndeed the parallel trends assumption holds throughout.\n"
    h, wid, md = _make_doc("reidem", md_v1)
    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h)
    try:
        spans_store.ensure_span(
            conn, cache_conn, None, markdown_id=md.markdown_id, work_id=wid,
            exact_quote="parallel trends assumption",
        )
        conn.commit()
    finally:
        conn.close()
        cache_conn.close()

    _reconvert(wid, "reidem", md_v2)  # same lineage, new markdown

    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h)
    try:
        # Mark stale, then reanchor TWICE under PRAGMA foreign_keys=ON.
        for row in conn.execute(
            "SELECT span_id FROM evidence_spans WHERE work_id = ?", (wid,)
        ).fetchall():
            spans_store.verify_span(conn, cache_conn, None, row[0])
        moved_1 = spans_store.reanchor_spans(conn, cache_conn, None, work_id=wid)
        moved_2 = spans_store.reanchor_spans(conn, cache_conn, None, work_id=wid)
        conn.commit()
        assert moved_1 == 1  # the quote relocates once
        assert moved_2 == 0  # idempotent: nothing left to relocate

        # Exactly 2 rows for the quote: 1 anchored relocated + 1 stale shadow (no dup).
        total = conn.execute(
            "SELECT COUNT(*) FROM evidence_spans WHERE exact_quote = ?",
            ("parallel trends assumption",),
        ).fetchone()[0]
        assert total == 2
        anchored = conn.execute(
            "SELECT COUNT(*) FROM evidence_spans "
            "WHERE exact_quote = ? AND anchor_status = 'anchored'",
            ("parallel trends assumption",),
        ).fetchone()[0]
        assert anchored == 1
        fts_rows = conn.execute(
            "SELECT COUNT(*) FROM span_fts WHERE quote_text = ?",
            ("parallel trends assumption",),
        ).fetchone()[0]
        assert fts_rows == 2  # one shadow + one relocated, never 3+
        # Search returns no inflated/duplicate hits (shadow + relocated only).
        assert len(fts_search.search_spans(conn, "parallel trends", work_id=wid)) == 2
    finally:
        conn.close()
        cache_conn.close()


# --- index empty/missing ----------------------------------------------------

def test_index_document_empty_or_missing_returns_zero_no_raise(monkeypatch):
    """index_document on a missing/empty/whitespace markdown returns 0 and flags, never raises."""
    h, wid, md = _make_doc("emptyidx")
    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h)
    try:
        # Missing markdown id -> 0, no raise.
        assert spans_index.index_document(
            conn, cache_conn, None, work_id=wid, markdown_id="md_missing"
        ) == 0
        # Whitespace-only markdown -> 0, no raise (guarded before any write).
        ws = cache_access.MarkdownRow(
            text="   \n\n \t\n", markdown_hash="h", storage_uri="x", source_file_id="sf_x",
            source_file_hash="x", access_class="open_access",
        )
        monkeypatch.setattr(cache_access, "read_markdown", lambda *a, **k: ws)
        assert spans_index.index_document(
            conn, cache_conn, None, work_id=wid, markdown_id="md_whitespace"
        ) == 0
        conn.commit()
    finally:
        conn.close()
        cache_conn.close()


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

def test_milestone_sections_build_and_spans_index_one_paper():
    """`seedgraph sections build` + `seedgraph spans index` for one paper succeed end-to-end."""
    h, wid, md = _make_doc("mile1")
    r1 = runner.invoke(app, ["sections", "build", "mile1", "--work", wid])
    assert r1.exit_code == 0, r1.output
    assert "sections=" in r1.output
    r2 = runner.invoke(app, ["spans", "index", "mile1", "--work", wid])
    assert r2.exit_code == 0, r2.output
    assert "spans=" in r2.output
    spans, fts, secs = _counts(h)
    assert spans > 0 and fts == spans and secs > 0


def test_milestone_search_each_term_returns_span_with_section():
    """search each term returns >=1 span whose get_span text contains the term verbatim,
    with correct work_id and a resolving section."""
    h, wid, md = _make_doc("mile2")
    _index(h, wid, md)
    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h)
    try:
        for term in ("parallel trends", "Assumption 2", "rank condition"):
            hits = fts_search.search_spans(conn, term)
            assert hits, term
            span = spans_store.get_span(conn, cache_conn, None, hits[0].span_id)
            assert term.lower() in span.resolved_text.lower()
            assert span.work_id == wid
            assert span.section_id is not None and span.heading_path is not None
    finally:
        conn.close()
        cache_conn.close()


def test_milestone_spans_get_returns_exact_text_section_pages():
    """`spans get <span_id>` returns exact source text == exact_quote, the section breadcrumb,
    and (if pagination delimiters present) page numbers."""
    paged = (
        "# Paged Paper\n\n"
        "{0}------------------------------------------------\n\n"
        "The parallel trends assumption is discussed on the first page.\n\n"
        "{1}------------------------------------------------\n\n"
        "The rank condition appears later on the second page here.\n"
    )
    h, wid, md = _make_doc("mile3", paged)
    _index(h, wid, md)
    cache_conn = cache_access.open_cache_ro(None)
    conn = _conn(h)
    try:
        hit = fts_search.search_spans(conn, "rank condition")[0]
        span = spans_store.get_span(conn, cache_conn, None, hit.span_id)
        assert span.resolved_text == span.exact_quote
        assert span.heading_path is not None
        assert span.page_start == 2 and span.page_end == 2  # second-page delimiter
        # CLI surface returns the exact text too.
        out = runner.invoke(app, ["spans", "get", "mile3", span.span_id])
        assert out.exit_code == 0, out.output
        assert "rank condition" in out.output
    finally:
        conn.close()
        cache_conn.close()


def test_milestone_spans_verify_all_invariant_holds():
    """`spans verify --all` reports markdown[start:end]==exact_quote and matching quote_hash
    for every span (all anchored on a freshly indexed paper)."""
    h, wid, md = _make_doc("mile4")
    _index(h, wid, md)
    out = runner.invoke(app, ["spans", "verify", "mile4", "--all"])
    assert out.exit_code == 0, out.output
    assert "anchored=" in out.output
    assert "stale" not in out.output and "orphaned" not in out.output


def test_milestone_reindex_changes_no_count_no_duplicate_fts():
    """Re-running `spans index` changes no span count and creates no duplicate FTS rows."""
    h, wid, md = _make_doc("mile5")
    runner.invoke(app, ["spans", "index", "mile5", "--work", wid])
    before = _counts(h)
    runner.invoke(app, ["spans", "index", "mile5", "--work", wid])
    after = _counts(h)
    assert before == after
    assert after[0] == after[1]  # spans == fts rows (no duplicates)


def test_milestone_doctor_reports_fts5_and_zero_dangling():
    """`seedgraph doctor` reports FTS5 present and zero dangling span/section references."""
    h, wid, md = _make_doc("mile6")
    _index(h, wid, md)
    out = runner.invoke(app, ["doctor", "--project", "mile6"])
    assert out.exit_code == 0, out.output
    assert "[PASS] span_fts5_available" in out.output
    assert "[PASS] span_section_fts_reconcile" in out.output
    assert "dangling_section_refs=0" in out.output
    # FTS5 present (the phase_0 baseline check too).
    assert "[PASS] fts5_available" in out.output
