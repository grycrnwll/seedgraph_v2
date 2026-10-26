"""Paper merge for duplicate review: plan (pure, fingerprinted) -> merge exactly the plan.

Invented text only; every library is under pytest's tmp.
"""

from __future__ import annotations

import pytest

from seedgraph.library import attention, graphs, merge, search, store
from seedgraph.notes import overlay
from seedgraph.notes.format import SCHEMA_ID

from .test_search import add_claim, add_span, anchor
from .test_store import NOW, add_markdown, add_paper, add_source_file, attach_file

EARLY = "2025-01-01T00:00:00+00:00"
MID = "2025-06-01T00:00:00+00:00"
LATE = "2026-06-01T00:00:00+00:00"


def count(conn, sql, *args):
    return conn.execute(sql, args).fetchone()[0]


def note(conn, note_id, paper_id, version_no, text, *, created_at=NOW, md_id="md_x", label=None,
         schema_id=SCHEMA_ID):
    run = "run_" + note_id
    conn.execute(
        "INSERT INTO extraction_runs (extraction_run_id, paper_id, markdown_id, markdown_hash, schema_id, "
        "schema_version, prompt_version, run_status, created_at) VALUES (?, ?, ?, 'x', ?, '1', '1', 'success', ?)",
        (run, paper_id, md_id, schema_id, created_at))
    conn.execute(
        "INSERT INTO structured_notes (note_id, extraction_run_id, paper_id, markdown_id, markdown_hash, schema_id, "
        "schema_version, prompt_version, raw_note_json, note_text, version_no, version_label, created_at) "
        "VALUES (?, ?, ?, ?, 'x', ?, '1', '1', '{}', ?, ?, ?, ?)",
        (note_id, run, paper_id, md_id, schema_id, text, version_no, label, created_at))
    conn.execute("INSERT INTO note_fts (note_id, note_text) VALUES (?, ?)", (note_id, text))
    return run


def ref(conn, ref_id, citing, text, *, resolved=None, ordinal=1):
    conn.execute(
        "INSERT INTO reference_entries (reference_id, citing_paper_id, raw_reference_text, resolved_paper_id, "
        "resolution_status, ordinal, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ref_id, citing, text, resolved, "resolved" if resolved else "unresolved", ordinal, NOW))


def link(conn, source, target, *, ref_id=None, provenance="parsed_bibliography", confidence=0.9):
    conn.execute(
        "INSERT INTO citation_links (source_paper_id, target_paper_id, provenance, confidence, reference_id, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?)", (source, target, provenance, confidence, ref_id, NOW))


def ident(conn, paper_id, id_type, value):
    conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value) VALUES (?, ?, ?)", (paper_id, id_type, value))


def graph_rows(lib, slug):
    with graphs.open_graph(lib, slug) as g:
        return {p.paper_id: (p.status, p.is_seed) for p in graphs.membership(g)}


# lp_keep:  held on sf 1 / md 1, DOI; notes v1 (EARLY), v2 (LATE); claim cl_k2 anchored in md 1.
# lp_other: held on sf 2 / md 2 (a different PDF), OpenAlex id; note v1 (MID, made on md 2) with a claim;
#           a lens run + output; a reference list; attention items; cited by lp_b.
# lp_t: cited by both; lp_b cites lp_other. lp_cited: a cited-only paper with no file.

@pytest.fixture
def lib(tmp_path):
    library = store.open_library(tmp_path)
    with library.write() as conn:
        for pid in ("lp_keep", "lp_other", "lp_t", "lp_b", "lp_cited"):
            add_paper(conn, pid, title=f"An invented paper {pid}")
        conn.execute("UPDATE papers SET title = NULL, year = 2031 WHERE paper_id = 'lp_keep'")
        ident(conn, "lp_keep", "doi", "10.0000/invented.keep")
        ident(conn, "lp_other", "openalex", "W123456")
        sf1, sf2 = add_source_file(conn, 1), add_source_file(conn, 2)
        md1, md2 = add_markdown(conn, sf1, 1), add_markdown(conn, sf2, 2)
        attach_file(conn, "lp_keep", sf1, md1)
        attach_file(conn, "lp_other", sf2, md2)
        add_span(conn, md1, "sp_k1", 0, "An invented sentence about kept widgets.")
        add_span(conn, md2, "sp_o1", 0, "An invented sentence about other widgets.")

        note(conn, "note_k1", "lp_keep", 1, "First invented keep note.", created_at=EARLY, md_id=md1)
        note(conn, "note_k2", "lp_keep", 2, "Second invented keep note.", created_at=LATE, md_id=md1)
        add_claim(conn, "cl_k2", "lp_keep", "run_note_k2", "kept widgets are invented", note_id="note_k2")
        anchor(conn, "cl_k2", "sp_k1")
        note(conn, "note_o1", "lp_other", 1, "An invented other note.", created_at=MID, md_id=md2)
        add_claim(conn, "cl_o1", "lp_other", "run_note_o1", "other widgets are invented", note_id="note_o1")
        anchor(conn, "cl_o1", "sp_o1")
        conn.execute(
            "INSERT INTO extraction_runs (extraction_run_id, paper_id, markdown_id, markdown_hash, lens_id, "
            "schema_version, prompt_version, run_status, created_at) VALUES "
            "('lens_o', 'lp_other', ?, 'x', 'L1', '1', '1', 'success', ?)", (md2, NOW))
        add_claim(conn, "cl_lens_o", "lp_other", "lens_o", "other widgets under a lens")
        conn.execute(
            "INSERT INTO lens_outputs (lens_output_id, extraction_run_id, lens_id, paper_id, status, claim_id, "
            "fields_json, created_at) VALUES ('lo_o', 'lens_o', 'L1', 'lp_other', 'ok', 'cl_lens_o', '{}', ?)",
            (NOW,))

        ref(conn, "ref_k1", "lp_keep", "Inventor, A. (2001). Invented T.", resolved="lp_t", ordinal=1)
        ref(conn, "ref_k2", "lp_keep", "Maker, B. (2002). Another invention.", ordinal=2)
        link(conn, "lp_keep", "lp_t", ref_id="ref_k1", confidence=0.7)
        ref(conn, "ref_o1", "lp_other", "Inventor,  A. (2001).  Invented T.", resolved="lp_t", ordinal=1)
        ref(conn, "ref_o2", "lp_other", "Dreamer, C. (2003). A third invention.", ordinal=2)
        ref(conn, "ref_o3", "lp_other", "Keeper, D. (2004). The kept paper.", resolved="lp_keep", ordinal=3)
        link(conn, "lp_other", "lp_t", ref_id="ref_o1", confidence=0.95)
        link(conn, "lp_other", "lp_keep", ref_id="ref_o3")
        conn.execute("INSERT INTO citation_review_decisions (reference_id, decision, decided_at) "
                     "VALUES ('ref_o2', 'reject', ?)", (NOW,))
        ref(conn, "ref_b1", "lp_b", "Other, E. (2005). The other paper.", resolved="lp_other")
        link(conn, "lp_b", "lp_other", ref_id="ref_b1")
        conn.execute("INSERT INTO v2_id_map (v2_project, v2_table, v2_id, v3_table, v3_id) "
                     "VALUES ('second_test', 'works', 'w_77', 'papers', 'lp_other')")

    for n in (1, 2):
        for rel in (f"pdfs/{n:064x}.pdf", f"markdown/{n + 100:064x}.md"):
            path = library.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"invented bytes")
    attention.raise_item(library, "identity_check", {"paper_id": "lp_other", "candidates": ["lp_keep"]},
                         paper_id="lp_other", detail="may be the same invented paper")
    attention.raise_item(library, "metadata_needed", {"paper_id": "lp_other"}, paper_id="lp_other")
    attention.raise_item(library, "pdf_title_miss", {"paper_id": "lp_other"}, paper_id="lp_other",
                         hides_notes=True)
    attention.raise_item(library, "malformed_title", {"paper_id": "lp_other"}, paper_id="lp_other")
    attention.raise_item(library, "malformed_title", {"paper_id": "lp_keep"}, paper_id="lp_keep")

    with graphs.create_graph(library, "widgets") as g:
        graphs.set_membership(g, "lp_keep", "frontier")
        graphs.set_membership(g, "lp_other", "member", is_seed=True)
    with graphs.create_graph(library, "land") as g:
        graphs.set_membership(g, "lp_other", "excluded")
    with graphs.create_graph(library, "harbour") as g:
        graphs.set_membership(g, "lp_keep", "excluded")
        graphs.set_membership(g, "lp_other", "frontier")
    yield library
    library.close()


def do_merge(lib, keep="lp_keep", other="lp_other", **kw):
    plan = merge.plan_merge(lib, keep, other, **kw)
    assert plan.blockers == (), plan.blockers
    return plan, merge.merge_papers(lib, plan, confirm=True)


# --- the plan --------------------------------------------------------------------------------


def test_the_plan_is_pure_and_names_what_moves(lib):
    before = lib.conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0]
    plan = merge.plan_merge(lib, "lp_keep", "lp_other")
    assert (plan.keep_id, plan.other_id) == ("lp_keep", "lp_other")
    assert plan.blockers == ()
    assert plan.identifiers == (("openalex", "W123456"),)
    assert plan.primary_file == f"sf_{1:064x}" and plan.detached_file == f"sf_{2:064x}"
    assert plan.counts["notes"] == 1 and plan.counts["lens_outputs"] == 1
    assert plan.fingerprint == merge.plan_merge(lib, "lp_keep", "lp_other").fingerprint
    assert lib.conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == before
    assert store.paper_exists(lib.conn, "lp_other")


def test_merging_a_paper_with_itself_or_an_unknown_paper_is_refused(lib):
    with pytest.raises(merge.MergeError):
        merge.plan_merge(lib, "lp_keep", "lp_keep")
    with pytest.raises(merge.MergeError):
        merge.plan_merge(lib, "lp_keep", "lp_nobody")


def test_conflicting_ids_of_one_type_block_the_merge(lib):
    with lib.write() as conn:
        ident(conn, "lp_other", "doi", "10.0000/invented.other")
    plan = merge.plan_merge(lib, "lp_keep", "lp_other")
    assert any("doi" in b and "10.0000/invented.other" in b for b in plan.blockers)
    with pytest.raises(merge.MergeError):
        merge.merge_papers(lib, plan, confirm=True)
    assert store.paper_exists(lib.conn, "lp_other")


def test_a_stale_plan_is_refused(lib):
    plan = merge.plan_merge(lib, "lp_keep", "lp_other")
    with lib.write() as conn:
        note(conn, "note_o2", "lp_other", 2, "A late invented note.", created_at=LATE)
    with pytest.raises(merge.StalePlanError):
        merge.merge_papers(lib, plan, confirm=True)
    assert store.paper_exists(lib.conn, "lp_other")


def test_merge_needs_confirm(lib):
    plan = merge.plan_merge(lib, "lp_keep", "lp_other")
    with pytest.raises(merge.MergeError):
        merge.merge_papers(lib, plan, confirm=False)


def test_a_paper_with_work_in_flight_blocks_the_merge(lib):
    with lib.write() as conn:
        conn.execute("INSERT INTO batches (batch_id, task_type, task_version, selection_json, created_at) "
                     "VALUES ('b_1', 'notes', '1', '[]', ?)", (NOW,))
        conn.execute("INSERT INTO batch_items (batch_id, ordinal, subject_id, state) "
                     "VALUES ('b_1', 1, 'lp_other', 'queued')")
    plan = merge.plan_merge(lib, "lp_keep", "lp_other")
    assert any("b_1" in b for b in plan.blockers)


# --- what moves ------------------------------------------------------------------------------


def test_identifiers_move_and_missing_metadata_is_filled(lib):
    do_merge(lib)
    rows = lib.conn.execute("SELECT id_type, id_value FROM identifiers WHERE paper_id = 'lp_keep' "
                            "ORDER BY id_type").fetchall()
    assert [tuple(r) for r in rows] == [("doi", "10.0000/invented.keep"), ("openalex", "W123456")]
    title, year = lib.conn.execute("SELECT title, year FROM papers WHERE paper_id = 'lp_keep'").fetchone()
    assert (title, year) == ("An invented paper lp_other", 2031)


def test_the_other_paper_row_is_removed_and_nothing_dangles(lib):
    do_merge(lib)
    assert not store.paper_exists(lib.conn, "lp_other")
    assert lib.conn.execute("PRAGMA foreign_key_check").fetchall() == []
    for table, col in merge.PAPER_REFERENCES:
        assert count(lib.conn, f"SELECT COUNT(*) FROM {table} WHERE {col} = 'lp_other'") == 0, table


def test_every_table_that_references_papers_is_handled(lib):
    fks = set()
    for (table,) in lib.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'"):
        for fk in lib.conn.execute(f"PRAGMA foreign_key_list('{table}')"):
            if fk["table"] == "papers":
                fks.add((table, fk["from"]))
    assert fks <= set(merge.PAPER_REFERENCES)


def test_the_kept_pdf_stays_primary_and_the_other_pdf_is_detached_not_deleted(lib):
    plan, result = do_merge(lib)
    files = lib.conn.execute("SELECT source_file_id, role FROM paper_files WHERE paper_id = 'lp_keep'").fetchall()
    assert [tuple(r) for r in files] == [(f"sf_{1:064x}", "primary")]
    assert count(lib.conn, "SELECT COUNT(*) FROM source_files WHERE source_file_id = ?", plan.detached_file) == 1
    assert (lib.root / f"pdfs/{2:064x}.pdf").exists() and (lib.root / f"markdown/{102:064x}.md").exists()
    assert result.detached_file == plan.detached_file


def test_the_researcher_can_choose_the_other_pdf_as_primary(lib):
    plan, _ = do_merge(lib, primary_file=f"sf_{2:064x}")
    assert plan.detached_file == f"sf_{1:064x}"
    files = lib.conn.execute("SELECT source_file_id FROM paper_files WHERE paper_id = 'lp_keep'").fetchall()
    assert [r[0] for r in files] == [f"sf_{2:064x}"]


def test_a_primary_file_neither_paper_holds_is_refused(lib):
    with pytest.raises(merge.MergeError):
        merge.plan_merge(lib, "lp_keep", "lp_other", primary_file="sf_nobody")


def test_a_cited_only_survivor_takes_the_other_papers_pdf(lib):
    plan, _ = do_merge(lib, keep="lp_cited")
    assert plan.detached_file is None
    assert store.is_held(lib.conn, "lp_cited")
    files = lib.conn.execute("SELECT source_file_id FROM paper_files WHERE paper_id = 'lp_cited'").fetchall()
    assert [r[0] for r in files] == [f"sf_{2:064x}"]


def test_a_shared_pdf_becomes_one_link(lib):
    with lib.write() as conn:
        conn.execute("DELETE FROM paper_files WHERE paper_id = 'lp_other'")
        attach_file(conn, "lp_other", f"sf_{1:064x}", f"md_{101:064x}")
    sf1 = f"sf_{1:064x}"
    sides = '{"papers": [{"paper_id": "lp_keep"}, {"paper_id": "lp_other"}]}'
    shared = attention.raise_item(lib, "shared_pdf", {"source_file_id": sf1}, detail=sides)
    third = attention.raise_item(lib, "shared_pdf", {"source_file_id": sf1, "n": 2},
                                 detail='{"papers": [{"paper_id": "lp_keep"}, {"paper_id": "lp_b"}]}')
    plan, _ = do_merge(lib)
    assert plan.detached_file is None
    assert count(lib.conn, "SELECT COUNT(*) FROM paper_files WHERE paper_id = 'lp_keep'") == 1
    assert attention.get_item(lib, shared.item_id).resolution == "merged"
    assert attention.get_item(lib, third.item_id).status == "open"  # a third paper still shares it


def test_a_migrated_identity_check_naming_both_papers_in_its_evidence_is_resolved(lib):
    item = attention.raise_item(
        lib, "identity_check", {"reason": "no_match", "paper_id": "lp_keep", "key_type": None, "key_value": None},
        paper_id="lp_keep", detail='{"candidates": [{"paper_id": "lp_other", "title": "x"}], "works": []}')
    do_merge(lib)
    assert attention.get_item(lib, item.item_id).resolution == "same"


def test_all_note_versions_are_kept_and_renumbered_by_created_at(lib):
    do_merge(lib, primary_file=None)
    rows = lib.conn.execute("SELECT note_id, version_no FROM structured_notes WHERE paper_id = 'lp_keep' "
                            "ORDER BY version_no").fetchall()
    assert [tuple(r) for r in rows] == [("note_k1", 1), ("note_o1", 2), ("note_k2", 3)]
    assert count(lib.conn, "SELECT COUNT(*) FROM extracted_claims WHERE paper_id = 'lp_keep'") == 3
    assert count(lib.conn, "SELECT COUNT(*) FROM extraction_runs WHERE paper_id = 'lp_keep'") == 4  # + lens run


def test_notes_made_on_the_detached_pdf_are_labelled_other_markdown(lib):
    do_merge(lib)
    label = lib.conn.execute("SELECT version_label FROM structured_notes WHERE note_id = 'note_o1'").fetchone()[0]
    assert label == "other_markdown"
    assert lib.conn.execute("SELECT note_id FROM note_current WHERE paper_id = 'lp_keep'").fetchone()[0] == "note_k2"


def test_lens_outputs_move_with_their_runs(lib):
    do_merge(lib)
    assert lib.conn.execute("SELECT paper_id FROM lens_outputs WHERE lens_output_id = 'lo_o'").fetchone()[0] \
        == "lp_keep"


def test_both_papers_with_researcher_edits_refuse_the_merge(lib):
    overlay.save_notes(lib, "lp_keep", "my invented keep reading", expected_version=0)
    overlay.save_notes(lib, "lp_other", "my invented other reading", expected_version=0)
    plan = merge.plan_merge(lib, "lp_keep", "lp_other", primary_file=f"sf_{2:064x}")
    assert any("researcher edits" in b for b in plan.blockers)


def test_one_overlay_moves_and_stays_pinned_to_its_base_note(lib):
    overlay.save_notes(lib, "lp_other", "my invented other reading", expected_version=0)
    plan = merge.plan_merge(lib, "lp_keep", "lp_other")
    # the overlay's base was made on lp_other's PDF, so detaching that PDF is refused
    assert any("researcher edits" in b for b in plan.blockers)
    do_merge(lib, primary_file=f"sf_{2:064x}")
    h = overlay.head(lib.conn, "lp_keep")
    assert h is not None and h.base_note_id == "note_o1" and h.researcher_notes == "my invented other reading"
    assert lib.conn.execute("SELECT note_id FROM note_current WHERE paper_id = 'lp_keep'").fetchone()[0] == "note_o1"
    assert count(lib.conn, "SELECT COUNT(*) FROM interpretation_claims WHERE paper_id = 'lp_other'") == 0
    assert count(lib.conn, "SELECT COUNT(*) FROM interpretation_notes WHERE paper_id = 'lp_keep'") == 1


def test_graph_memberships_are_unioned(lib):
    do_merge(lib)
    assert graph_rows(lib, "widgets") == {"lp_keep": ("member", True)}
    assert graph_rows(lib, "land") == {"lp_keep": ("excluded", False)}
    assert graph_rows(lib, "harbour") == {"lp_keep": ("frontier", False)}


def test_citation_links_are_repointed_and_deduped(lib):
    do_merge(lib)
    links = {(r[0], r[1]): (r[2], r[3]) for r in lib.conn.execute(
        "SELECT source_paper_id, target_paper_id, confidence, reference_id FROM citation_links")}
    assert links[("lp_keep", "lp_t")][0] == 0.95
    assert links[("lp_b", "lp_keep")] == (0.9, "ref_b1")
    assert ("lp_keep", "lp_keep") not in links
    assert len(links) == 2


def test_reference_entries_are_repointed_and_deduped(lib):
    do_merge(lib)
    rows = lib.conn.execute("SELECT reference_id, ordinal, resolved_paper_id FROM reference_entries "
                            "WHERE citing_paper_id = 'lp_keep' ORDER BY ordinal").fetchall()
    assert [r[0] for r in rows] == ["ref_k1", "ref_k2", "ref_o2", "ref_o3"]
    assert [r[1] for r in rows] == [1, 2, 3, 4]
    assert lib.conn.execute("SELECT resolved_paper_id FROM reference_entries WHERE reference_id = 'ref_b1'"
                            ).fetchone()[0] == "lp_keep"
    assert count(lib.conn, "SELECT COUNT(*) FROM citation_review_decisions WHERE reference_id = 'ref_o2'") == 1


def test_attention_items_move_or_resolve(lib):
    do_merge(lib)
    by_kind = {}
    for it in attention.list_items(lib):
        by_kind.setdefault(it.kind, []).append(it)
    (identity,) = by_kind["identity_check"]
    assert (identity.status, identity.resolution) == ("resolved", "same")
    (meta,) = by_kind["metadata_needed"]
    assert (meta.paper_id, meta.status) == ("lp_keep", "open")
    (miss,) = by_kind["pdf_title_miss"]  # about the detached PDF
    assert (miss.paper_id, miss.status, miss.resolution) == ("lp_keep", "resolved", "merged")
    malformed = sorted((i.paper_id, i.status) for i in by_kind["malformed_title"])
    assert malformed == [("lp_keep", "open"), ("lp_keep", "resolved")]


def test_the_decision_and_traceability_are_recorded(lib):
    do_merge(lib)
    row = lib.conn.execute("SELECT verdict, decided_by, reason FROM duplicate_decisions "
                           "WHERE paper_a = 'lp_keep' AND paper_b = 'lp_other'").fetchone()
    assert tuple(row)[:2] == ("same", "researcher") and "lp_keep" in row[2]
    assert lib.conn.execute("SELECT v3_id FROM v2_id_map WHERE v2_project = 'second_test' AND v2_id = 'w_77'"
                            ).fetchone()[0] == "lp_keep"
    assert merge.resolve_merged(lib.conn, "lp_other") == "lp_keep"
    assert merge.resolve_merged(lib.conn, "lp_keep") == "lp_keep"


def test_concepts_are_flagged_out_of_date_and_the_library_dirty(lib):
    with lib.write() as conn:
        conn.execute("UPDATE library_meta SET value = '0' WHERE key = 'dirty'")
    do_merge(lib)
    assert lib.conn.execute("SELECT reason FROM concept_refresh WHERE paper_id = 'lp_keep'").fetchone()[0] \
        == merge.REFRESH_REASON
    assert store.is_dirty(lib.conn)


def test_search_finds_the_moved_claims_under_the_surviving_paper(lib):
    with lib.write() as conn:  # make lp_other's note the newest, so it becomes current
        conn.execute("UPDATE structured_notes SET created_at = ? WHERE note_id = 'note_o1'", ("2027-01-01T00:00:00",))
        conn.execute("DELETE FROM attention_items WHERE kind = 'pdf_title_miss'")  # it would hide the notes
    do_merge(lib, primary_file=f"sf_{2:064x}")
    hits = search.search(lib, "other widgets", kind="claims", audience="agent")
    assert {h.paper_id for h in hits} == {"lp_keep"} and "cl_o1" in {h.claim_id for h in hits}
    notes = search.search(lib, "invented other note", kind="notes", audience="researcher")
    assert {h.paper_id for h in notes} == {"lp_keep"}
    for kind in ("claims", "notes", "spans"):
        assert all(h.paper_id != "lp_other" for h in search.search(lib, "invented", kind=kind,
                                                                    audience="researcher", limit=None))


# --- different papers ------------------------------------------------------------------------


def test_different_papers_records_the_decision(lib):
    merge.record_different(lib, "lp_other", "lp_keep")
    row = lib.conn.execute("SELECT paper_a, paper_b, verdict, decided_by FROM duplicate_decisions").fetchone()
    assert tuple(row) == ("lp_keep", "lp_other", "different", "researcher")
    assert merge.decided_different(lib.conn, "lp_other", "lp_keep")


def test_a_prior_different_verdict_is_a_warning_in_the_plan(lib):
    merge.record_different(lib, "lp_other", "lp_keep")
    plan = merge.plan_merge(lib, "lp_keep", "lp_other")
    assert any("different" in w for w in plan.warnings)
    do_merge(lib)
    assert lib.conn.execute("SELECT verdict FROM duplicate_decisions").fetchone()[0] == "same"


def test_suggest_survivor_prefers_the_held_paper(lib):
    assert merge.suggest_survivor(lib.conn, "lp_cited", "lp_other") == ("lp_other", "lp_cited")
