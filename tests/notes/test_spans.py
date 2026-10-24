"""Library port of ensure_span: exact-unique matching, invariant, dedupe, pages and sections."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from seedgraph import anchor
from seedgraph.notes import spans

from .conftest import make_paper


def _span(conn, span_id):
    return conn.execute("SELECT * FROM evidence_spans WHERE span_id = ?", (span_id,)).fetchone()


def test_read_markdown_returns_the_stored_text(lib, paper):
    assert spans.read_markdown(lib.conn, paper.markdown_id) == paper.text


def test_read_markdown_keeps_crlf_so_offsets_match_the_bytes(lib):
    text = "line one" + chr(13) + chr(10) + "line two" + chr(13) + chr(10)
    other = make_paper(lib, paper_id="lp_invented0000000002", n=2, text=text)
    assert spans.read_markdown(lib.conn, other.markdown_id) == text


def test_read_markdown_of_unknown_id_is_none(lib, paper):
    assert spans.read_markdown(lib.conn, "md_" + "0" * 64) is None


def test_anchored_span_keeps_the_invariant_and_finds_page_and_section(lib, paper):
    quote = "Our two-step widget estimator counts widgets twice and averages."
    with lib.write() as conn:
        span_id = spans.ensure_span(conn, markdown_id=paper.markdown_id, exact_quote=quote)
    row = _span(lib.conn, span_id)
    anchor.assert_invariant(paper.text, SimpleNamespace(**dict(row)))  # raises on violation
    assert row["exact_quote"] == quote
    assert row["markdown_hash"] == paper.markdown_hash
    assert row["source_file_id"] == paper.source_file_id
    assert row["section_id"] == "sec_1_method"
    assert row["page_start"] is not None
    assert row["anchor_status"] == "anchored"
    fts = lib.conn.execute("SELECT span_id, markdown_id FROM span_fts WHERE span_fts MATCH 'averages'").fetchall()
    assert [tuple(r) for r in fts] == [(span_id, paper.markdown_id)]


@pytest.mark.parametrize("quote", [
    "Widgets grow by an invented amount.",          # twice in the text
    "a sentence that is not in the invented text",  # absent
    "The micro-sign μ appears once",                # Greek mu, the text has U+00B5
    "",
])
def test_no_unique_exact_match_gives_none_and_writes_nothing(lib, paper, quote):
    with lib.write() as conn:
        assert spans.ensure_span(conn, markdown_id=paper.markdown_id, exact_quote=quote) is None
    assert lib.conn.execute("SELECT COUNT(*) FROM evidence_spans").fetchone()[0] == 0


def test_same_quote_is_deduplicated(lib, paper):
    quote = "Units do not anticipate the invented programme before it starts."
    with lib.write() as conn:
        a = spans.ensure_span(conn, markdown_id=paper.markdown_id, exact_quote=quote)
        b = spans.ensure_span(conn, markdown_id=paper.markdown_id, exact_quote=quote)
    assert a == b
    assert lib.conn.execute("SELECT COUNT(*) FROM evidence_spans").fetchone()[0] == 1
    assert lib.conn.execute("SELECT COUNT(*) FROM span_fts").fetchone()[0] == 1
