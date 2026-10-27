""""This isn't the paper": move a suspect PDF to the library paper it belongs to, or to a new
paper (library.reassign). Plan (pure, fingerprinted) -> reassign exactly the plan; the notes
made on that PDF follow it. Invented text only; every library is under pytest's tmp.
"""

from __future__ import annotations

import pytest
from web.paper_support import FixedBackend, text_pdf

from seedgraph.library import attention, files, graphs, reassign, store
from seedgraph.library.identity import derive_key, mint_paper_id
from seedgraph.notes import overlay

from .test_merge import EARLY, LATE, MID, count, note
from .test_search import add_claim, add_span, anchor
from .test_store import NOW, add_paper

SRC_TITLE = "Seasonal Patterns in Invented Lighthouse Maintenance"
TGT_TITLE = "Harbour Crane Productivity Under Invented Tides"
DECOY_TITLE = "Harbour Crane Scheduling in Invented Ports"
HELD_TITLE = "Ledger Methods for Invented Granaries"
ODD_TITLE = "Quiet Moss Growth on Invented Northern Walls"
COVER_DOI = "10.5555/invented.crane.2029"

# The suspect PDF: its cover shows the target's title (and a DOI no paper holds yet).
WRONG_PAGES = [f"{TGT_TITLE}. Ada Inventor. Journal of Invented Ports 2029. doi:{COVER_DOI}",
               "We measure how invented cranes lift invented boxes when the invented tide is high.",
               "Cranes lift more boxes at high tide in this made up harbour."]
RIGHT_PAGES = [f"{HELD_TITLE}. Bo Madeup. 2030.", "Invented granaries keep ledgers.", "The end."]
MD_WRONG = (f"# {TGT_TITLE}\n\n{{0}}{'-' * 48}\n\n{WRONG_PAGES[0]}\n\n{{1}}{'-' * 48}\n\n{WRONG_PAGES[1]}\n\n"
            f"{{2}}{'-' * 48}\n\n{WRONG_PAGES[2]}\n")


class Fx:
    def __init__(self, lib, sf, md, sf_held):
        self.lib, self.sf, self.md, self.sf_held = lib, sf, md, sf_held


@pytest.fixture
def fx(tmp_path):
    lib = store.open_library(tmp_path / "lib")
    with lib.write() as conn:
        add_paper(conn, "lp_src", title=SRC_TITLE)
        add_paper(conn, "lp_tgt", title=TGT_TITLE)
        add_paper(conn, "lp_decoy", title=DECOY_TITLE)
        add_paper(conn, "lp_held", title=HELD_TITLE)
        add_paper(conn, "lp_odd", title=ODD_TITLE)
        conn.execute("UPDATE papers SET year = 2029")
    res = files.attach_pdf(lib, "lp_src", text_pdf(tmp_path / "wrong.pdf", WRONG_PAGES))
    assert res.check.status == "miss" and res.attention_item_id
    md = files.convert_file(lib, res.source_file_id, backend=FixedBackend(MD_WRONG)).markdown_id
    held = files.attach_pdf(lib, "lp_held", text_pdf(tmp_path / "right.pdf", RIGHT_PAGES))
    with lib.write() as conn:
        # the source's notes: v1, v2 made on the suspect PDF's markdown; the target's own v1 (MID)
        note(conn, "note_s1", "lp_src", 1, "First invented crane note.", created_at=EARLY, md_id=md)
        note(conn, "note_s2", "lp_src", 2, "Second invented crane note.", created_at=LATE, md_id=md)
        add_span(conn, md, "sp_w1", 0, "Cranes lift more boxes at high tide.")
        add_claim(conn, "cl_s2", "lp_src", "run_note_s2", "cranes lift more", note_id="note_s2")
        anchor(conn, "cl_s2", "sp_w1")
        conn.execute(
            "INSERT INTO extraction_runs (extraction_run_id, paper_id, markdown_id, markdown_hash, lens_id, "
            "schema_version, prompt_version, run_status, created_at) VALUES "
            "('lens_s', 'lp_src', ?, 'x', 'L1', '1', '1', 'success', ?)", (md, NOW))
        add_claim(conn, "cl_lens_s", "lp_src", "lens_s", "cranes under a lens")
        conn.execute(
            "INSERT INTO lens_outputs (lens_output_id, extraction_run_id, lens_id, paper_id, status, claim_id, "
            "fields_json, created_at) VALUES ('lo_s', 'lens_s', 'L1', 'lp_src', 'ok', 'cl_lens_s', '{}', ?)",
            (NOW,))
        note(conn, "note_t1", "lp_tgt", 1, "An older invented target note.", created_at=MID, md_id="md_gone")
    with graphs.create_graph(lib, "harbours") as g:
        graphs.set_membership(g, "lp_src", "member", is_seed=True, reason="invented seed")
    yield Fx(lib, res.source_file_id, md, held.source_file_id)
    lib.close()


def miss_item(fx, paper_id="lp_src"):
    return attention.get_item(fx.lib, attention.item_id_for(
        files.MISS_KIND, {"paper_id": paper_id, "source_file_id": fx.sf}))


def versions(lib, paper_id):
    return [tuple(r) for r in lib.conn.execute(
        "SELECT note_id, version_no FROM structured_notes WHERE paper_id = ? ORDER BY version_no", (paper_id,))]


def graph_row(lib, slug, paper_id):
    with graphs.open_graph(lib, slug) as g:
        return next(((p.status, p.is_seed) for p in graphs.membership(g) if p.paper_id == paper_id), None)


# --- suggestions -------------------------------------------------------------------------------


def test_suggestions_rank_the_title_on_the_pdf_cover_first(fx):
    found = reassign.suggest_targets(fx.lib, "lp_src", fx.sf)
    assert found[0].paper_id == "lp_tgt" and found[0].score == 1.0
    ids = [s.paper_id for s in found]
    assert "lp_src" not in ids
    assert ids.index("lp_decoy") > 0 and ids.index("lp_decoy") < len(ids)
    assert found[0].score > next(s.score for s in found if s.paper_id == "lp_decoy")
    assert "lp_odd" not in ids  # one title token of six on the cover: below a weak match


def test_title_search_finds_library_papers(fx):
    found = reassign.find_papers(fx.lib, "harbour crane", exclude=("lp_src",))
    assert {s.paper_id for s in found} == {"lp_tgt", "lp_decoy"}
    assert reassign.find_papers(fx.lib, "  ") == ()


# --- move to a library paper -------------------------------------------------------------------


def test_moving_to_a_cited_only_paper_makes_it_held_and_the_source_cited_only(fx):
    plan = reassign.plan_reassign_pdf(fx.lib, "lp_src", fx.sf, target_paper_id="lp_tgt")
    assert plan.ok and plan.target_was_cited_only and plan.source_becomes_cited_only
    assert plan.counts["notes"] == 2 and plan.counts["claims"] == 2 and plan.counts["lens_outputs"] == 1
    assert [g.slug for g in plan.graphs] == ["harbours"]
    result = reassign.reassign_pdf(fx.lib, plan, confirm=True)
    conn = fx.lib.conn
    assert result.receiving_paper_id == "lp_tgt"
    assert store.is_held(conn, "lp_tgt") and not store.is_held(conn, "lp_src")
    assert conn.execute("SELECT markdown_id FROM paper_files WHERE paper_id = 'lp_tgt'").fetchone()[0] == fx.md
    # the source drops to the frontier (a cited-only paper cannot be a member); the target is not added
    assert graph_row(fx.lib, "harbours", "lp_src") == ("frontier", True)
    assert graph_row(fx.lib, "harbours", "lp_tgt") is None


def test_notes_claims_anchors_and_lens_outputs_follow_the_pdf_in_version_order(fx):
    plan = reassign.plan_reassign_pdf(fx.lib, "lp_src", fx.sf, target_paper_id="lp_tgt")
    reassign.reassign_pdf(fx.lib, plan)
    conn = fx.lib.conn
    # renumbered on the receiving paper by created_at: s1 (EARLY), t1 (MID), s2 (LATE)
    assert versions(fx.lib, "lp_tgt") == [("note_s1", 1), ("note_t1", 2), ("note_s2", 3)]
    assert versions(fx.lib, "lp_src") == []
    for table, col in (("extraction_runs", "extraction_run_id"), ("extracted_claims", "claim_id"),
                       ("lens_outputs", "lens_output_id")):
        assert count(conn, f"SELECT COUNT(*) FROM {table} WHERE paper_id = 'lp_src'") == 0, table
    assert count(conn, "SELECT COUNT(*) FROM extracted_claims WHERE paper_id = 'lp_tgt'") == 2
    assert count(conn, "SELECT COUNT(*) FROM claim_spans WHERE claim_id = 'cl_s2'") == 1
    assert conn.execute("SELECT paper_id FROM note_current WHERE note_id = 'note_s2'").fetchone()[0] == "lp_tgt"
    # FTS rows are keyed by id, so the moved note still finds its paper
    hit = conn.execute("SELECT n.paper_id FROM note_fts f JOIN structured_notes n ON n.note_id = f.note_id "
                       "WHERE note_fts MATCH 'second'").fetchone()
    assert hit[0] == "lp_tgt"
    refresh = {r[0] for r in conn.execute("SELECT paper_id FROM concept_refresh")}
    assert {"lp_src", "lp_tgt"} <= refresh


def test_the_item_is_resolved_and_the_receiving_paper_is_rechecked(fx):
    assert miss_item(fx).status == "open"
    reassign.reassign_pdf(fx.lib, reassign.plan_reassign_pdf(fx.lib, "lp_src", fx.sf, target_paper_id="lp_tgt"))
    item = miss_item(fx)
    assert item.status == "resolved" and item.resolution == "wrong_pdf_moved"
    row = fx.lib.conn.execute("SELECT pdf_check_status FROM paper_files WHERE paper_id = 'lp_tgt'").fetchone()
    assert row[0] == "pass"
    assert miss_item(fx, "lp_tgt") is None
    assert "lp_tgt" not in attention.hidden_paper_ids(fx.lib.conn)
    assert "lp_src" not in attention.hidden_paper_ids(fx.lib.conn)


def test_a_receiving_paper_whose_title_misses_gets_a_fresh_item(fx):
    reassign.reassign_pdf(fx.lib, reassign.plan_reassign_pdf(fx.lib, "lp_src", fx.sf, target_paper_id="lp_odd"))
    item = miss_item(fx, "lp_odd")
    assert item is not None and item.status == "open" and item.hides_notes
    assert "lp_odd" in attention.hidden_paper_ids(fx.lib.conn)


def test_a_target_holding_a_different_pdf_is_refused_with_merge_offered(fx):
    plan = reassign.plan_reassign_pdf(fx.lib, "lp_src", fx.sf, target_paper_id="lp_held")
    assert not plan.ok
    assert any("already has a different PDF" in b and "merge" in b for b in plan.blockers)
    with pytest.raises(reassign.ReassignError):
        reassign.reassign_pdf(fx.lib, plan)
    assert store.is_held(fx.lib.conn, "lp_src")


def test_a_target_already_holding_the_same_pdf_just_drops_the_source_link(fx, tmp_path):
    files.attach_pdf(fx.lib, "lp_tgt", text_pdf(tmp_path / "again.pdf", WRONG_PAGES))
    shared = attention.raise_item(fx.lib, "shared_pdf", {"source_file_id": fx.sf}, detail="{}")
    plan = reassign.plan_reassign_pdf(fx.lib, "lp_src", fx.sf, target_paper_id="lp_tgt")
    assert plan.ok and not plan.target_was_cited_only
    reassign.reassign_pdf(fx.lib, plan)
    holders = [r[0] for r in fx.lib.conn.execute("SELECT paper_id FROM paper_files WHERE source_file_id = ?",
                                                 (fx.sf,))]
    assert holders == ["lp_tgt"]
    assert attention.get_item(fx.lib, shared.item_id).resolution == "wrong_pdf_moved"


def test_researcher_edits_on_a_moving_note_refuse_the_move(fx):
    overlay.save_notes(fx.lib, "lp_src", "my invented reading", expected_version=0)
    plan = reassign.plan_reassign_pdf(fx.lib, "lp_src", fx.sf, target_paper_id="lp_tgt")
    assert any("researcher edits" in b for b in plan.blockers)
    with pytest.raises(reassign.ReassignError):
        reassign.reassign_pdf(fx.lib, plan)


def test_a_stale_plan_is_refused(fx):
    plan = reassign.plan_reassign_pdf(fx.lib, "lp_src", fx.sf, target_paper_id="lp_tgt")
    with fx.lib.write() as conn:
        note(conn, "note_s3", "lp_src", 3, "A third invented note.", created_at=LATE, md_id=fx.md)
    with pytest.raises(reassign.StalePlanError):
        reassign.reassign_pdf(fx.lib, plan)
    assert store.is_held(fx.lib.conn, "lp_src")


def test_bad_requests_are_refused(fx):
    with pytest.raises(reassign.ReassignError):
        reassign.plan_reassign_pdf(fx.lib, "lp_src", fx.sf)  # neither a target nor new_paper
    with pytest.raises(reassign.ReassignError):
        reassign.plan_reassign_pdf(fx.lib, "lp_src", fx.sf, target_paper_id="lp_src")
    with pytest.raises(reassign.ReassignError):
        reassign.plan_reassign_pdf(fx.lib, "lp_src", fx.sf_held, target_paper_id="lp_tgt")  # not its PDF
    with pytest.raises(reassign.ReassignError):
        reassign.reassign_pdf(fx.lib, reassign.plan_reassign_pdf(fx.lib, "lp_src", None, target_paper_id="lp_tgt"),
                              confirm=False)


# --- a paper not in the library ------------------------------------------------------------------


def test_the_new_paper_path_creates_a_paper_that_needs_metadata(fx):
    plan = reassign.plan_reassign_pdf(fx.lib, "lp_src", fx.sf, new_paper=True)
    expected = mint_paper_id(derive_key(pdf_hashes=[fx.sf[3:]]), {"lp_src"}).paper_id
    assert plan.ok and plan.new_paper and plan.receiving_paper_id == expected
    assert plan.cover_identifiers == (("doi", COVER_DOI),)
    result = reassign.reassign_pdf(fx.lib, plan)
    pid = result.receiving_paper_id
    conn = fx.lib.conn
    assert pid == expected and store.is_held(conn, pid) and not store.is_held(conn, "lp_src")
    assert conn.execute("SELECT title FROM papers WHERE paper_id = ?", (pid,)).fetchone()[0] is None
    assert [tuple(r) for r in conn.execute("SELECT id_value, source FROM identifiers WHERE paper_id = ?", (pid,))] == [
        (COVER_DOI, "pdf_scan")]
    items = {i.kind: i for i in attention.list_items(fx.lib, paper_id=pid, status="open")}
    assert "metadata_needed" in items
    assert miss_item(fx).resolution == "wrong_pdf_new_paper"
    assert [n for n, _ in versions(fx.lib, pid)] == ["note_s1", "note_s2"]
    assert graph_row(fx.lib, "harbours", pid) is None
