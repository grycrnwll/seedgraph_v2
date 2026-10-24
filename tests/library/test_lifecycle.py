"""Lifecycle: remove from a graph, plan a delete, delete exactly the plan (§5)."""

from __future__ import annotations

import pytest

from seedgraph.library import attention, graphs, lifecycle, store

from .test_search import add_claim, add_note, add_run, add_span, anchor
from .test_store import NOW, add_markdown, add_paper, add_source_file, attach_file


def count(conn, sql, *args):
    return conn.execute(sql, args).fetchone()[0]


def add_reference(conn, ref_id, citing, resolved=None, ordinal=1):
    conn.execute(
        "INSERT INTO reference_entries (reference_id, citing_paper_id, raw_reference_text, "
        "resolved_paper_id, resolution_status, ordinal, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ref_id, citing, f"Invented, A. ({ordinal}) An invented reference.", resolved,
         "resolved" if resolved else "unresolved", ordinal, NOW),
    )


def add_link(conn, source, target, ref_id=None, provenance="parsed_bibliography"):
    conn.execute(
        "INSERT INTO citation_links (source_paper_id, target_paper_id, provenance, confidence, "
        "reference_id, created_at) VALUES (?, ?, ?, 0.9, ?, ?)",
        (source, target, provenance, ref_id, NOW),
    )


def write_blob(lib, rel):
    path = lib.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"invented bytes")
    return path


# lp_a: held on its own PDF (sf 1), with notes, claims, anchors, a lens output, a reference
#       list, a review decision, an attention item; member+seed of "diff_in_diff", frontier of "land".
# lp_b: held (sf 2); its reference list cites lp_a.
# lp_c: held (sf 3), cited by nobody.
# lp_x, lp_y: share one PDF (sf 4) and its markdown.
# lp_t: cited-only target of lp_a's references.

@pytest.fixture
def lib(tmp_path):
    library = store.open_library(tmp_path)
    with library.write() as conn:
        for pid in ("lp_a", "lp_b", "lp_c", "lp_x", "lp_y", "lp_t"):
            add_paper(conn, pid)
        conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value) "
                     "VALUES ('lp_a', 'doi', '10.0000/invented.a')")
        conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value) "
                     "VALUES ('lp_c', 'doi', '10.0000/invented.c')")
        mds = {}
        for n, pid in ((1, "lp_a"), (2, "lp_b"), (3, "lp_c"), (4, "lp_x")):
            sf = add_source_file(conn, n)
            mds[pid] = add_markdown(conn, sf, n)
            attach_file(conn, pid, sf, mds[pid])
        sf_shared = conn.execute(
            "SELECT source_file_id FROM paper_files WHERE paper_id='lp_x'").fetchone()[0]
        attach_file(conn, "lp_y", sf_shared, mds["lp_x"])

        add_span(conn, mds["lp_a"], "sp_a1", 0, "An invented sentence about widgets.", page=1)
        add_span(conn, mds["lp_c"], "sp_c1", 0, "An invented sentence about gadgets.")
        add_span(conn, mds["lp_x"], "sp_x1", 0, "An invented shared sentence.")

        ra1 = add_note(conn, "note_a1", "lp_a", 1, "First invented note on widgets.")
        ra2 = add_note(conn, "note_a2", "lp_a", 2, "Researcher-edited invented note on widgets.")
        conn.execute("UPDATE structured_notes SET origin = 'researcher' WHERE note_id = 'note_a2'")
        add_claim(conn, "cl_a1", "lp_a", ra1, "widgets are invented", note_id="note_a1")
        add_claim(conn, "cl_a2", "lp_a", ra2, "widgets are still invented", note_id="note_a2")
        anchor(conn, "cl_a2", "sp_a1")
        add_run(conn, "lens_a", "lp_a", lens_id="L1")
        add_claim(conn, "cl_lens_a", "lp_a", "lens_a", "widgets under a lens")
        conn.execute(
            "INSERT INTO lens_outputs (lens_output_id, extraction_run_id, lens_id, paper_id, status, "
            "claim_id, fields_json, created_at) VALUES ('lo_a', 'lens_a', 'L1', 'lp_a', 'ok', "
            "'cl_lens_a', '{}', ?)", (NOW,))
        rc = add_note(conn, "note_c1", "lp_c", 1, "Invented note on gadgets.")
        add_claim(conn, "cl_c1", "lp_c", rc, "gadgets are invented", note_id="note_c1")
        anchor(conn, "cl_c1", "sp_c1")
        rx = add_note(conn, "note_x1", "lp_x", 1, "Invented note on the shared PDF.")
        add_claim(conn, "cl_x1", "lp_x", rx, "shared claim", note_id="note_x1")
        anchor(conn, "cl_x1", "sp_x1")

        add_reference(conn, "ref_a1", "lp_a", resolved="lp_t", ordinal=1)
        add_reference(conn, "ref_a2", "lp_a", ordinal=2)
        add_link(conn, "lp_a", "lp_t", "ref_a1")
        conn.execute("INSERT INTO citation_review_decisions (reference_id, decision, decided_at) "
                     "VALUES ('ref_a2', 'reject', ?)", (NOW,))
        add_reference(conn, "ref_b1", "lp_b", resolved="lp_a")
        add_link(conn, "lp_b", "lp_a", "ref_b1")
    attention.raise_item(library, "pdf_title_miss", {"paper_id": "lp_a"}, paper_id="lp_a",
                         hides_notes=True, created_at=NOW)
    attention.raise_item(library, "malformed_title", {"paper_id": "lp_c"}, paper_id="lp_c",
                         created_at=NOW)
    for sf_n in (1, 3, 4):
        h = f"{sf_n:064x}"
        write_blob(library, f"pdfs/{h}.pdf")
        write_blob(library, f"markdown/{sf_n + 100:064x}.md")

    with graphs.create_graph(library, "diff_in_diff") as g:
        graphs.set_membership(g, "lp_a", "member", is_seed=True, added_at=NOW)
        graphs.set_membership(g, "lp_x", "member", added_at=NOW)
        graphs.set_membership(g, "lp_c", "member", added_at=NOW)
    with graphs.create_graph(library, "land") as g:
        graphs.set_membership(g, "lp_a", "frontier", added_at=NOW)
    yield library
    library.close()


def graph_status(lib, slug, paper_id):
    with graphs.open_graph(lib, slug) as g:
        rows = {p.paper_id: p.status for p in graphs.membership(g)}
    return rows.get(paper_id)


def fk_clean(conn):
    return conn.execute("PRAGMA foreign_key_check").fetchall() == []


# --- remove_from_graph ------------------------------------------------------------------------

def test_remove_from_graph_drops_only_the_membership_and_reports_a_seed(lib):
    before = count(lib.conn, "SELECT COUNT(*) FROM structured_notes WHERE paper_id='lp_a'")
    result = lifecycle.remove_from_graph(lib, "diff_in_diff", "lp_a")
    assert result.removed is True
    assert result.was_seed is True
    assert result.status == "member"
    assert graph_status(lib, "diff_in_diff", "lp_a") is None
    assert graph_status(lib, "land", "lp_a") == "frontier"
    assert count(lib.conn, "SELECT COUNT(*) FROM structured_notes WHERE paper_id='lp_a'") == before
    assert store.paper_exists(lib.conn, "lp_a")


def test_remove_from_graph_of_a_non_member_reports_nothing_removed(lib):
    result = lifecycle.remove_from_graph(lib, "diff_in_diff", "lp_b")
    assert result.removed is False and result.was_seed is False and result.status is None


# --- plan_delete ------------------------------------------------------------------------------

def test_plan_names_every_graph_using_the_paper_and_warns(lib):
    plan = lifecycle.plan_delete(lib, "lp_a")
    assert [(u.slug, u.status, u.is_seed) for u in plan.graphs] == [
        ("diff_in_diff", "member", True), ("land", "frontier", False)]
    assert any("diff_in_diff" in w and "land" in w for w in plan.warnings)
    assert plan.counts == {
        "notes": 2, "current_notes": 1, "claims": 3, "anchors": 1, "lens_outputs": 1,
        "references": 2, "outgoing_links": 1, "incoming_links": 1,
    }
    assert plan.keeps_cited_only is True  # lp_b's reference list cites it
    assert plan.has_researcher_edits is False  # stage 3 adds overlays
    assert plan.pdf_shared is False


def test_plan_writes_nothing(lib):
    with store.transaction(lib.conn) as c:
        c.execute("UPDATE library_meta SET value = '0' WHERE key = 'dirty'")
    lifecycle.plan_delete(lib, "lp_a")
    assert not store.is_dirty(lib.conn)


def test_plan_reports_a_shared_pdf(lib):
    plan = lifecycle.plan_delete(lib, "lp_x")
    assert plan.pdf_shared is True
    assert any("shared" in w.lower() for w in plan.warnings)


def test_plan_for_an_unknown_paper_is_refused(lib):
    with pytest.raises(lifecycle.LifecycleError):
        lifecycle.plan_delete(lib, "lp_nope")


# --- delete_paper ------------------------------------------------------------------------------

def test_delete_removes_the_held_parts_and_keeps_a_cited_paper_as_cited_only(lib):
    plan = lifecycle.plan_delete(lib, "lp_a")
    result = lifecycle.delete_paper(lib, plan)
    c = lib.conn
    for table, col in (("structured_notes", "paper_id"), ("extracted_claims", "paper_id"),
                       ("extraction_runs", "paper_id"), ("lens_outputs", "paper_id"),
                       ("reference_entries", "citing_paper_id"), ("paper_files", "paper_id"),
                       ("citation_links", "source_paper_id")):
        assert count(c, f"SELECT COUNT(*) FROM {table} WHERE {col} = 'lp_a'") == 0, table
    assert count(c, "SELECT COUNT(*) FROM claim_spans WHERE claim_id LIKE 'cl_a%'") == 0
    assert count(c, "SELECT COUNT(*) FROM note_fts WHERE note_id LIKE 'note_a%'") == 0
    assert count(c, "SELECT COUNT(*) FROM claim_fts WHERE claim_id IN ('cl_a1','cl_a2','cl_lens_a')") == 0
    assert count(c, "SELECT COUNT(*) FROM citation_review_decisions") == 0
    # kept as cited-only, because lp_b's reference list cites it
    assert store.paper_exists(c, "lp_a") and not store.is_held(c, "lp_a")
    assert count(c, "SELECT COUNT(*) FROM identifiers WHERE paper_id = 'lp_a'") == 1
    assert count(c, "SELECT COUNT(*) FROM citation_links WHERE target_paper_id = 'lp_a'") == 1
    # removed from every graph
    assert graph_status(lib, "diff_in_diff", "lp_a") is None
    assert graph_status(lib, "land", "lp_a") is None
    # its own PDF was not shared: rows and blobs go
    h = f"{1:064x}"
    assert count(c, "SELECT COUNT(*) FROM source_files WHERE source_file_id = ?", f"sf_{h}") == 0
    assert count(c, "SELECT COUNT(*) FROM evidence_spans WHERE span_id = 'sp_a1'") == 0
    assert count(c, "SELECT COUNT(*) FROM span_fts WHERE span_id = 'sp_a1'") == 0
    assert not (lib.root / f"pdfs/{h}.pdf").exists()
    assert not (lib.root / f"markdown/{101:064x}.md").exists()
    assert result.paper_row_deleted is False
    # untouched neighbours
    assert count(c, "SELECT COUNT(*) FROM structured_notes WHERE paper_id = 'lp_c'") == 1
    assert fk_clean(c)


def test_delete_of_an_uncited_paper_removes_the_paper_row_too(lib):
    result = lifecycle.delete_paper(lib, lifecycle.plan_delete(lib, "lp_c"))
    c = lib.conn
    assert result.paper_row_deleted is True
    assert not store.paper_exists(c, "lp_c")
    assert count(c, "SELECT COUNT(*) FROM identifiers WHERE paper_id = 'lp_c'") == 0
    assert attention.list_items(lib, paper_id="lp_c") == []
    assert graph_status(lib, "diff_in_diff", "lp_c") is None
    assert fk_clean(c)


def test_delete_leaves_a_pdf_and_markdown_another_paper_uses(lib):
    lifecycle.delete_paper(lib, lifecycle.plan_delete(lib, "lp_x"))
    c = lib.conn
    h = f"{4:064x}"
    assert count(c, "SELECT COUNT(*) FROM source_files WHERE source_file_id = ?", f"sf_{h}") == 1
    assert count(c, "SELECT COUNT(*) FROM evidence_spans WHERE span_id = 'sp_x1'") == 1
    assert count(c, "SELECT COUNT(*) FROM paper_files WHERE paper_id = 'lp_y'") == 1
    assert (lib.root / f"pdfs/{h}.pdf").exists()
    assert (lib.root / f"markdown/{104:064x}.md").exists()
    assert not store.paper_exists(c, "lp_x")
    assert fk_clean(c)


def test_delete_refuses_a_plan_whose_graph_usage_changed(lib):
    plan = lifecycle.plan_delete(lib, "lp_a")
    lifecycle.remove_from_graph(lib, "land", "lp_a")
    with pytest.raises(lifecycle.StalePlanError):
        lifecycle.delete_paper(lib, plan)
    assert store.is_held(lib.conn, "lp_a")
    assert graph_status(lib, "diff_in_diff", "lp_a") == "member"


def test_delete_refuses_a_plan_when_a_note_was_added_since(lib):
    plan = lifecycle.plan_delete(lib, "lp_a")
    with lib.write() as conn:
        add_note(conn, "note_a3", "lp_a", 3, "A third invented note.")
    with pytest.raises(lifecycle.StalePlanError):
        lifecycle.delete_paper(lib, plan)
    assert count(lib.conn, "SELECT COUNT(*) FROM structured_notes WHERE paper_id='lp_a'") == 3


def test_delete_needs_confirmation(lib):
    with pytest.raises(lifecycle.LifecycleError):
        lifecycle.delete_paper(lib, lifecycle.plan_delete(lib, "lp_a"), confirm=False)
    assert store.is_held(lib.conn, "lp_a")


def test_researcher_edited_notes_go_only_with_the_paper(lib):
    """Placeholder for stage 3 overlays: written against note rows for now."""
    edited = "SELECT COUNT(*) FROM structured_notes WHERE paper_id='lp_a' AND origin='researcher'"
    lifecycle.remove_from_graph(lib, "diff_in_diff", "lp_a")
    lifecycle.remove_from_graph(lib, "land", "lp_a")
    assert count(lib.conn, edited) == 1
    lifecycle.delete_paper(lib, lifecycle.plan_delete(lib, "lp_a"))
    assert count(lib.conn, edited) == 0
