"""Library search: v2 match semantics, graph scope, audience hiding, stable order, match sets (§8.1)."""

from __future__ import annotations

import pytest

from seedgraph.library import attention, graphs, search, store

from .test_store import NOW, add_markdown, add_paper, add_source_file, attach_file

LATER = "2026-03-01T00:00:00+00:00"


# --- invented-row builders -------------------------------------------------------------

def add_section(conn, md_id, section_id, kind, ordinal):
    sf, sfh = conn.execute(
        "SELECT source_file_id, substr(source_file_id, 4) FROM markdown_documents WHERE markdown_id = ?",
        (md_id,)).fetchone()
    conn.execute(
        "INSERT INTO document_sections (section_id, markdown_id, markdown_hash, source_file_id, "
        "source_file_hash, level, ordinal, section_kind, start_char, end_char, "
        "section_parser_version, created_at) VALUES (?, ?, ?, ?, ?, 1, ?, ?, 0, 1000, 'v1', ?)",
        (section_id, md_id, md_id[3:], sf, sfh, ordinal, kind, NOW),
    )


def add_span(conn, md_id, span_id, start, text, *, page=None, section_id=None):
    sf = conn.execute("SELECT source_file_id FROM markdown_documents WHERE markdown_id = ?",
                      (md_id,)).fetchone()[0]
    conn.execute(
        "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, source_file_id, "
        "source_file_hash, section_id, start_char, end_char, exact_quote, quote_hash, span_kind, "
        "page_start, page_end, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'qh', 'paragraph', ?, ?, ?)",
        (span_id, md_id, md_id[3:], sf, sf[3:], section_id, start, start + len(text), text,
         page, page, NOW),
    )
    conn.execute(
        "INSERT INTO span_fts (quote_text, span_id, markdown_id, section_id) VALUES (?, ?, ?, ?)",
        (text, span_id, md_id, section_id),
    )


def add_run(conn, run_id, paper_id, *, schema_id="note_v1", lens_id=None, created_at=NOW):
    conn.execute(
        "INSERT INTO extraction_runs (extraction_run_id, paper_id, markdown_id, markdown_hash, "
        "schema_id, lens_id, schema_version, prompt_version, run_status, created_at) "
        "VALUES (?, ?, 'md_x', 'x', ?, ?, '1', '1', 'success', ?)",
        (run_id, paper_id, None if lens_id else schema_id, lens_id, created_at),
    )


def add_note(conn, note_id, paper_id, version_no, text, *, label=None):
    run = "run_" + note_id
    add_run(conn, run, paper_id)
    conn.execute(
        "INSERT INTO structured_notes (note_id, extraction_run_id, paper_id, markdown_id, "
        "markdown_hash, schema_id, schema_version, prompt_version, raw_note_json, note_text, "
        "version_no, version_label, created_at) "
        "VALUES (?, ?, ?, 'md_x', 'x', 'note_v1', '1', '1', '{}', ?, ?, ?, ?)",
        (note_id, run, paper_id, text, version_no, label, NOW),
    )
    conn.execute("INSERT INTO note_fts (note_id, note_text) VALUES (?, ?)", (note_id, text))
    return run


def add_claim(conn, claim_id, paper_id, run_id, text, *, note_id=None, label="invented label"):
    conn.execute(
        "INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, paper_id, "
        "claim_type, field_key, normalized_label, claim_text, status, epistemic_type, created_at) "
        "VALUES (?, ?, ?, ?, 'method', 'k', ?, ?, 'ok', 'stated', ?)",
        (claim_id, note_id, run_id, paper_id, label, text, NOW),
    )
    conn.execute(
        "INSERT INTO claim_fts (claim_id, normalized_label, claim_text) VALUES (?, ?, ?)",
        (claim_id, label, text),
    )


def anchor(conn, claim_id, span_id, rank=0):
    conn.execute(
        "INSERT INTO claim_spans (claim_id, span_id, rank, created_at) VALUES (?, ?, ?, ?)",
        (claim_id, span_id, rank, NOW),
    )


# --- the invented library ----------------------------------------------------------------
#
# md1 is shared by lp_a and lp_s (the shared-PDF case); md2 = lp_b; md3 = lp_c (in no graph);
# md4 = lp_f (held, on the frontier). Graph "diff_in_diff": members lp_a, lp_s, lp_b; frontier lp_f.

@pytest.fixture
def lib(tmp_path):
    library = store.open_library(tmp_path)
    with library.write() as conn:
        mds = {}
        for n, pid in enumerate(("lp_a", "lp_b", "lp_c", "lp_f"), start=1):
            add_paper(conn, pid)
            sf = add_source_file(conn, n)
            mds[pid] = add_markdown(conn, sf, n)
            attach_file(conn, pid, sf, mds[pid])
        add_paper(conn, "lp_s")
        sf1 = conn.execute("SELECT source_file_id FROM paper_files WHERE paper_id='lp_a'").fetchone()[0]
        attach_file(conn, "lp_s", sf1, mds["lp_a"])

        add_section(conn, mds["lp_a"], "sec_a_intro", "introduction", 0)
        add_section(conn, mds["lp_a"], "sec_a_body", "body", 1)
        add_span(conn, mds["lp_a"], "sp_a1", 0, "The parallel trends assumption holds for widgets.",
                 page=2, section_id="sec_a_body")
        add_span(conn, mds["lp_a"], "sp_a2", 200, "Here trends parallel each other only loosely.",
                 section_id="sec_a_intro")
        add_span(conn, mds["lp_b"], "sp_b1", 0, "We test parallel trends with invented gadgets.", page=1)
        add_span(conn, mds["lp_c"], "sp_c1", 0, "Parallel trends again, in an unrelated paper.")
        add_span(conn, mds["lp_f"], "sp_f1", 0, "A frontier note on parallel trends.")

        # notes: lp_a has two versions; lp_b has a current v1 and a labelled v2;
        # lp_s is hidden by an attention item; lp_c is in no graph.
        ra1 = add_note(conn, "note_a1", "lp_a", 1, "Older widget estimator summary.")
        ra2 = add_note(conn, "note_a2", "lp_a", 2, "Newer widget estimator summary.")
        rb1 = add_note(conn, "note_b1", "lp_b", 1, "Gadget widget estimator summary.")
        rb2 = add_note(conn, "note_b2", "lp_b", 2, "Labelled widget estimator summary.",
                       label="other_markdown")
        rs1 = add_note(conn, "note_s1", "lp_s", 1, "Shared widget estimator summary.")
        rc1 = add_note(conn, "note_c1", "lp_c", 1, "Unrelated widget estimator summary.")
        add_claim(conn, "cl_a1", "lp_a", ra1, "widget estimator is biased", note_id="note_a1")
        add_claim(conn, "cl_a2", "lp_a", ra2, "widget estimator is unbiased", note_id="note_a2")
        anchor(conn, "cl_a2", "sp_a1")
        add_claim(conn, "cl_b1", "lp_b", rb1, "widget estimator on gadgets", note_id="note_b1")
        add_claim(conn, "cl_b2", "lp_b", rb2, "widget estimator labelled", note_id="note_b2")
        add_claim(conn, "cl_s1", "lp_s", rs1, "widget estimator shared", note_id="note_s1")
        add_claim(conn, "cl_c1", "lp_c", rc1, "widget estimator unrelated", note_id="note_c1")
        # lens claims: only the latest run per (paper, lens) is current
        add_run(conn, "lens_old", "lp_b", lens_id="L1", created_at=NOW)
        add_run(conn, "lens_new", "lp_b", lens_id="L1", created_at=LATER)
        add_claim(conn, "cl_lens_old", "lp_b", "lens_old", "widget estimator lens old")
        add_claim(conn, "cl_lens_new", "lp_b", "lens_new", "widget estimator lens new")
        # a claim with neither note nor lens is always current
        add_run(conn, "run_loose", "lp_b", schema_id=None)
        add_claim(conn, "cl_loose", "lp_b", "run_loose", "widget estimator loose")

    attention.raise_item(library, "pdf_title_miss", {"paper_id": "lp_s"}, paper_id="lp_s",
                         hides_notes=True, created_at=NOW)
    with graphs.create_graph(library, "diff_in_diff") as g:
        for pid in ("lp_a", "lp_s", "lp_b"):
            graphs.set_membership(g, pid, "member", added_at=NOW)
        graphs.set_membership(g, "lp_f", "frontier", added_at=NOW)
    yield library
    library.close()


def span_ids(hits):
    return {h.span_id for h in hits}


# --- spans -----------------------------------------------------------------------------------

def test_span_search_matches_the_query_as_one_verbatim_phrase(lib):
    hits = search.search(lib, "parallel trends", kind="spans", limit=None)
    # "trends parallel" (reversed order) does not match the phrase
    assert span_ids(hits) == {"sp_a1", "sp_b1", "sp_c1", "sp_f1"}


def test_span_search_uses_v2s_match_builder():
    from seedgraph.fts.search import build_match_query

    for q in ("parallel trends", 'say "hi"', "AR(1) - x*", "  padded  "):
        assert search.match_query(q) == build_match_query(q)


def test_fts_syntax_in_the_query_never_raises(lib):
    assert search.search(lib, "AR(1) OR - * NEAR", kind="spans") == []
    assert search.search(lib, 'an "unbalanced quote', kind="claims") == []


def test_empty_query_returns_nothing(lib):
    for kind in ("spans", "claims", "notes"):
        assert search.search(lib, "   ", kind=kind) == []


def test_graph_scope_keeps_member_markdowns_and_frontier_is_opt_in(lib):
    assert span_ids(search.search(lib, "parallel trends", kind="spans", graph="diff_in_diff")) == {
        "sp_a1", "sp_b1"}
    assert span_ids(search.search(lib, "parallel trends", kind="spans", graph="diff_in_diff",
                                  include_frontier=True)) == {"sp_a1", "sp_b1", "sp_f1"}


def test_a_span_hit_reports_every_member_paper_on_its_markdown_once(lib):
    hits = search.search(lib, "parallel trends", kind="spans", graph="diff_in_diff")
    a1 = [h for h in hits if h.span_id == "sp_a1"]
    assert len(a1) == 1
    hit = a1[0]
    assert hit.paper_ids == ("lp_a", "lp_s")
    assert hit.paper_id == "lp_a"
    md = lib.conn.execute("SELECT markdown_id, markdown_hash FROM evidence_spans "
                          "WHERE span_id = 'sp_a1'").fetchone()
    assert hit.markdown_id == md[0]
    assert hit.page_start == 2 and hit.page_end == 2
    assert hit.section_id == "sec_a_body"
    assert hit.text == "The parallel trends assumption holds for widgets."
    assert hit.key == (md[1], 0, len("The parallel trends assumption holds for widgets."))


def test_span_filters_by_paper_and_section_kind(lib):
    assert span_ids(search.search(lib, "parallel trends", kind="spans", paper_id="lp_b")) == {"sp_b1"}
    assert span_ids(search.search(lib, "trends", kind="spans", section_kind="introduction")) == {
        "sp_a2"}


def test_paper_filter_and_graph_scope_combine(lib):
    with graphs.open_graph(lib, "diff_in_diff") as g:
        graphs.remove_membership(g, "lp_s")
    # lp_s shares lp_a's markdown but is no longer a member: the filters AND together
    assert search.search(lib, "parallel trends", kind="spans", graph="diff_in_diff", paper_id="lp_s") == []
    hit = search.search(lib, "parallel trends", kind="spans", graph="diff_in_diff", paper_id="lp_a")[0]
    assert hit.paper_ids == ("lp_a",)


def test_spans_are_not_hidden_from_agents(lib):
    # lp_s has an open hides_notes item, but text is never hidden (only notes and claims)
    hits = search.search(lib, "parallel trends", kind="spans", paper_id="lp_s")
    assert span_ids(hits) == {"sp_a1"}


def test_equal_scores_are_ordered_by_paper_then_offset_then_span_id(tmp_path):
    library = store.open_library(tmp_path)
    try:
        with library.write() as conn:
            for n, pid in ((1, "lp_z"), (2, "lp_y")):
                add_paper(conn, pid)
                sf = add_source_file(conn, n)
                attach_file(conn, pid, sf, add_markdown(conn, sf, n))
            md_z, md_y = (conn.execute("SELECT markdown_id FROM paper_files WHERE paper_id=?",
                                       (p,)).fetchone()[0] for p in ("lp_z", "lp_y"))
            same = "identical gizmo sentence"
            # inserted out of the expected order on purpose
            add_span(conn, md_y, "sp_y_late", 50, same)
            add_span(conn, md_z, "sp_z", 0, same)
            add_span(conn, md_y, "sp_y_b", 10, same)
            add_span(conn, md_y, "sp_y_a", 10, same)
        hits = search.search(library, "gizmo", kind="spans", limit=None)
        assert [h.span_id for h in hits] == ["sp_y_a", "sp_y_b", "sp_y_late", "sp_z"]
        assert len({h.rank for h in hits}) == 1
        assert [h.span_id for h in search.search(library, "gizmo", kind="spans", limit=2)] == [
            "sp_y_a", "sp_y_b"]
    finally:
        library.close()


# --- notes -----------------------------------------------------------------------------------

def note_ids(hits):
    return {h.note_id for h in hits}


def test_agent_note_search_returns_only_current_unhidden_notes(lib):
    hits = search.search(lib, "widget estimator", kind="notes", limit=None)
    assert note_ids(hits) == {"note_a2", "note_b1", "note_c1"}
    assert all(h.is_current and not h.hidden and h.version_label is None for h in hits)


def test_note_search_scoped_to_a_graph(lib):
    assert note_ids(search.search(lib, "widget estimator", kind="notes", graph="diff_in_diff")) == {
        "note_a2", "note_b1"}


def test_researcher_sees_every_note_labelled(lib):
    hits = {h.note_id: h for h in search.search(lib, "widget estimator", kind="notes",
                                                audience="researcher", limit=None)}
    assert set(hits) == {"note_a1", "note_a2", "note_b1", "note_b2", "note_s1", "note_c1"}
    assert hits["note_a1"].is_current is False and hits["note_a1"].hidden is False
    assert hits["note_b2"].version_label == "other_markdown" and hits["note_b2"].hidden is True
    assert hits["note_s1"].hidden is True and hits["note_s1"].is_current is True
    assert hits["note_a2"].paper_id == "lp_a" and hits["note_a2"].paper_ids == ("lp_a",)


def test_resolving_the_attention_item_unhides_the_note(lib):
    item = attention.list_items(lib, paper_id="lp_s")[0]
    attention.resolve_item(lib, item.item_id, "confirmed", resolved_at=LATER)
    assert "note_s1" in note_ids(search.search(lib, "widget estimator", kind="notes", graph="diff_in_diff"))


# --- claims ----------------------------------------------------------------------------------

def claim_ids(hits):
    return {h.claim_id for h in hits}


def test_agent_claim_search_mirrors_v2s_current_claims(lib):
    hits = search.search(lib, "widget estimator", kind="claims", limit=None)
    assert claim_ids(hits) == {"cl_a2", "cl_b1", "cl_c1", "cl_lens_new", "cl_loose"}


def test_claim_hit_carries_paper_note_and_its_anchor_page(lib):
    hit = next(h for h in search.search(lib, "unbiased", kind="claims"))
    assert hit.claim_id == "cl_a2"
    assert hit.note_id == "note_a2"
    assert hit.paper_id == "lp_a"
    assert hit.span_id == "sp_a1"
    assert hit.page_start == 2
    assert hit.text == "widget estimator is unbiased"
    assert hit.key == "cl_a2"


def test_researcher_claims_include_hidden_and_superseded_labelled(lib):
    hits = {h.claim_id: h for h in search.search(lib, "widget estimator", kind="claims",
                                                 graph="diff_in_diff", audience="researcher", limit=None)}
    assert set(hits) == {"cl_a1", "cl_a2", "cl_b1", "cl_b2", "cl_s1", "cl_lens_old",
                         "cl_lens_new", "cl_loose"}
    assert hits["cl_lens_old"].is_current is False
    assert hits["cl_b2"].version_label == "other_markdown"
    assert hits["cl_s1"].hidden is True


def test_claims_and_notes_order_is_stable_on_ties(lib):
    # every note text has the same length and one occurrence of each term: tied scores
    hits = search.search(lib, "widget estimator", kind="notes", audience="researcher", limit=None)
    assert len({h.rank for h in hits}) == 1
    assert [h.note_id for h in hits] == [
        "note_a1", "note_a2", "note_b1", "note_b2", "note_c1", "note_s1"]


# --- match sets ------------------------------------------------------------------------------

def test_match_set_is_the_unlimited_unordered_key_set(lib):
    md_a, md_b = (lib.conn.execute("SELECT markdown_hash FROM evidence_spans WHERE span_id = ?",
                                   (s,)).fetchone()[0] for s in ("sp_a1", "sp_b1"))
    assert search.match_set(lib, "parallel trends", kind="spans", graph="diff_in_diff") == frozenset({
        (md_a, 0, len("The parallel trends assumption holds for widgets.")),
        (md_b, 0, len("We test parallel trends with invented gadgets.")),
    })
    assert search.match_set(lib, "widget estimator", kind="notes", graph="diff_in_diff") == frozenset({
        "note_a2", "note_b1"})
    assert search.match_set(lib, "widget estimator", kind="claims", graph="diff_in_diff") == frozenset({
        "cl_a2", "cl_b1", "cl_lens_new", "cl_loose"})


def test_invalid_kind_or_audience_is_refused(lib):
    with pytest.raises(search.SearchError):
        search.search(lib, "x", kind="concepts")
    with pytest.raises(search.SearchError):
        search.search(lib, "x", kind="spans", audience="robot")
