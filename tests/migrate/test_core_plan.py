"""The pure plan builder (internal seam, design §6.1 step 4): rules that need a variant
of the synthetic v2 home. Each variant is a private copy edited with plain SQL."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from fixtures import build_v2_home
from seedgraph.migrate import run_migration
from seedgraph.migrate.plan import build_plan
from seedgraph.migrate.snapshot import take_snapshot

NOW = "2026-10-23T04:00:00+00:00"


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return build_v2_home(tmp_path_factory.mktemp("v2") / "home")


@pytest.fixture(scope="module")
def base_plan(fx):
    return build_plan(take_snapshot(fx.home, now=NOW))


def variant(fx, tmp_path: Path, edits: list[tuple[str, str, tuple]]) -> Path:
    home = tmp_path / "home"
    shutil.copytree(fx.home, home)
    for slug, sql, args in edits:
        db = home / ("cache/cache.db" if slug == "cache" else f"projects/{slug}/project.db")
        conn = sqlite3.connect(db)
        conn.execute(sql, args)
        conn.commit()
        conn.close()
    return home


def plan_for(home: Path):
    return build_plan(take_snapshot(home, now=NOW))


def paper_of(plan, slug, work_id):
    return next(r["v3_id"] for r in plan.rows("v2_id_map")
                if (r["v2_project"], r["v2_table"], r["v2_id"]) == (slug, "works", work_id))


def versions(plan, pid):
    return [(n["note_id"], n["version_no"], n["version_label"])
            for n in sorted(plan.rows("structured_notes"), key=lambda n: n["version_no"]) if n["paper_id"] == pid]


# --- determinism -------------------------------------------------------------------------

def test_same_snapshot_data_gives_an_equal_plan(fx, base_plan):
    again = build_plan(take_snapshot(fx.home, now=NOW))
    assert again.library == base_plan.library
    assert again.graphs == base_plan.graphs
    assert again.attention == base_plan.attention
    assert again.ledger == base_plan.ledger
    assert again.report == base_plan.report


def test_every_v2_row_has_a_disposition(base_plan):
    assert base_plan.ledger.unaccounted() == []
    summary = base_plan.ledger.summary()
    assert summary["second_test"]["works"]
    assert summary["cache"]["conversion_runs"] == {"carried": 24, "not_carried:pending_run": 1}


# --- note version order ----------------------------------------------------------------------

def test_versions_order_by_created_at_then_precedence_then_note_id(fx, tmp_path):
    st, tp = fx.works["overlap@second_test"], fx.works["overlap@test_proj"]
    item = fx.one("overlap_diverging_notes")
    n_st, n_tp = item.ids["note_ids"]
    # same created_at: second_test (higher precedence) comes first, test_proj's is current
    home = variant(fx, tmp_path, [("test_proj", "UPDATE structured_notes SET created_at = ? WHERE note_id = ?",
                                   ("2026-07-06T00:05:00+00:00", n_tp))])
    plan = plan_for(home)
    pid = paper_of(plan, st.slug, st.work_id)
    assert paper_of(plan, tp.slug, tp.work_id) == pid
    assert versions(plan, pid) == [(n_st, 1, None), (n_tp, 2, None)]


def test_older_note_wins_when_it_is_newer(fx, tmp_path):
    item = fx.one("overlap_diverging_notes")
    n_st, n_tp = item.ids["note_ids"]
    home = variant(fx, tmp_path, [("second_test", "UPDATE structured_notes SET created_at = ? WHERE note_id = ?",
                                   ("2026-07-09T00:00:00+00:00", n_st))])
    plan = plan_for(home)
    st = fx.works["overlap@second_test"]
    assert [v[0] for v in versions(plan, paper_of(plan, st.slug, st.work_id))] == [n_tp, n_st]
    assert not any(x["paper_id"] == paper_of(plan, st.slug, st.work_id) for x in plan.report["current_elsewhere"])


def test_a_labelled_version_is_numbered_but_never_current(base_plan, fx):
    item = fx.one("foreign_markdown")
    w = fx.works[item.ids["work_key"]]
    pid = paper_of(base_plan, w.slug, w.work_id)
    assert versions(base_plan, pid) == [(item.ids["note_id"], 1, "other_markdown")]


# --- reference lists and decision mapping ------------------------------------------------------

def test_decision_maps_by_raw_text_when_the_ordinal_differs(fx, tmp_path):
    item = fx.one("citation_review_decided")
    refs_st, refs_tp = item.ids["reference_ids"]["second_test"], item.ids["reference_ids"]["test_proj"]
    tp_mapped = item.ids["items"][2][1]
    # drop test_proj's first entry: its "ambiguous" entry moves from ordinal 2 to 1
    home = variant(fx, tmp_path, [
        ("test_proj", "DELETE FROM citation_edges WHERE reference_id = ?", (refs_tp[0],)),
        ("test_proj", "DELETE FROM reference_entries WHERE reference_id = ?", (refs_tp[0],)),
    ])
    plan = plan_for(home)
    mapped = {r["v2_id"]: r["v3_id"] for r in plan.rows("v2_id_map") if r["v2_table"] == "review_queue"}
    assert mapped[tp_mapped] == refs_st[1]


def test_ambiguous_raw_text_without_ordinal_match_is_unmapped(fx, tmp_path):
    item = fx.one("citation_review_decided")
    refs_st, refs_tp = item.ids["reference_ids"]["second_test"], item.ids["reference_ids"]["test_proj"]
    tp_mapped = item.ids["items"][2][1]
    home = variant(fx, tmp_path, [
        # second_test's list gets the ambiguous text twice; test_proj's entry sits at another ordinal
        ("second_test", "UPDATE reference_entries SET raw_reference_text = "
                        "(SELECT raw_reference_text FROM reference_entries WHERE reference_id = ?) WHERE reference_id = ?",
         (refs_st[1], refs_st[2])),
        ("test_proj", "DELETE FROM citation_edges WHERE reference_id = ?", (refs_tp[0],)),
        ("test_proj", "DELETE FROM reference_entries WHERE reference_id = ?", (refs_tp[0],)),
    ])
    plan = plan_for(home)
    assert tp_mapped in {u["item_id"] for u in plan.report["unmapped_decisions"]}


def test_the_list_parsed_from_the_current_markdown_beats_precedence(fx, tmp_path):
    item = fx.one("citation_review_decided")
    refs_st = item.ids["reference_ids"]["second_test"]
    # second_test's list now claims another markdown: test_proj's list (current markdown) wins
    home = variant(fx, tmp_path, [("second_test", "UPDATE reference_entries SET markdown_id = 'md_elsewhere' "
                                                  "WHERE reference_id IN (?, ?, ?)", tuple(refs_st))])
    plan = plan_for(home)
    st = fx.works["overlap@second_test"]
    pid = paper_of(plan, st.slug, st.work_id)
    chosen = next(x for x in plan.report["reference_lists"] if x["paper_id"] == pid)
    assert chosen["project"] == "test_proj" and chosen["entries"] == 4


# --- sections (ruling 3) ---------------------------------------------------------------------

def _overlap_md(fx):
    return fx.one("overlap_diverging_notes").ids["markdown_id"]


def test_newer_parser_version_wins_over_precedence(fx, tmp_path):
    md = _overlap_md(fx)
    home = variant(fx, tmp_path, [
        ("test_proj", "UPDATE document_sections SET section_parser_version = 'secparse-3', heading_text = "
                      "heading_text || ' (v3)', created_at = '2026-08-01T00:00:00+00:00' WHERE markdown_id = ?", (md,))])
    plan = plan_for(home)
    rows = [r for r in plan.rows("document_sections") if r["markdown_id"] == md]
    assert rows and {r["section_parser_version"] for r in rows} == {"secparse-3"}
    assert {r["origin"] for r in rows} == {"v2:test_proj"}
    assert plan.errors == []
    assert plan.report["text_counts"]["sections_older_parser"] == 3


def test_equal_versions_take_the_higher_precedence_rows_ignoring_created_at(fx, base_plan):
    md = _overlap_md(fx)
    rows = [r for r in base_plan.rows("document_sections") if r["markdown_id"] == md]
    assert {r["origin"] for r in rows} == {"v2:second_test"} and len(rows) == 3
    assert base_plan.report["text_counts"]["sections_duplicate"] >= 3  # test_proj's copies, other created_at


def test_same_version_different_content_is_an_error(fx, tmp_path):
    md = _overlap_md(fx)
    home = variant(fx, tmp_path, [("test_proj", "UPDATE document_sections SET heading_text = 'Changed' "
                                                "WHERE markdown_id = ? AND ordinal = 1", (md,))])
    plan = plan_for(home)
    assert any("section conflict" in e for e in plan.errors)


# --- anchors and errors ----------------------------------------------------------------------

def test_a_broken_anchor_is_an_error_and_blocks_the_swap(fx, tmp_path):
    item = fx.one("noid_unmatched")
    span = None
    conn = sqlite3.connect(f"file:{(fx.home / 'projects/second_test/project.db').as_posix()}?mode=ro", uri=True)
    span = conn.execute("SELECT cs.span_id FROM claim_spans cs JOIN extracted_claims c USING (claim_id) "
                        "WHERE c.structured_note_id = ? ORDER BY cs.span_id LIMIT 1", (item.ids["note_id"],)).fetchone()[0]
    conn.close()
    home = variant(fx, tmp_path, [("second_test", "UPDATE evidence_spans SET start_char = start_char + 1 "
                                                  "WHERE span_id = ?", (span,))])
    r = run_migration(home, tmp_path / "v3", run_id="mig-broken", now=NOW, process_lister=lambda: [],
                      disk_free=lambda p: 10 ** 15)
    assert any("anchor check failed" in e for e in r.errors)
    assert not r.swapped and not (tmp_path / "v3" / "library").exists()
    assert (r.staging_dir / "library" / "library.db").is_file()


# --- a merged paper with two different PDFs (real-data shape) ------------------------------------

def test_a_merged_paper_with_two_pdfs_keeps_one_primary_file(fx, tmp_path):
    fake = fx.works["fake@oct2026test"]
    home = variant(fx, tmp_path, [
        ("oct2026test", "UPDATE works SET doi = '10.5555/sg.overlap' WHERE work_id = ?", (fake.work_id,)),
        ("oct2026test", "UPDATE identifiers SET id_value = '10.5555/sg.overlap' WHERE work_id = ? AND id_type = 'doi'",
         (fake.work_id,)),
    ])
    plan = plan_for(home)
    st = fx.works["overlap@second_test"]
    pid = paper_of(plan, st.slug, st.work_id)
    assert paper_of(plan, fake.slug, fake.work_id) == pid
    files = [r for r in plan.rows("paper_files") if r["paper_id"] == pid]
    assert len(files) == 1 and files[0]["source_file_id"] == st.source_file_id
    sec = next(x for x in plan.report["secondary_pdfs"] if x["paper_id"] == pid)
    assert sec["secondary"] == [fake.source_file_id[3:]]
    # the fake PDF is not an "unheld" paper: it belongs to a held work
    assert not any(r["source_file_id"] == fake.source_file_id for r in plan.rows("paper_files"))
    assert plan.errors == []


def test_a_list_parsed_from_another_pdfs_markdown_is_not_carried(fx, tmp_path):
    item = fx.one("foreign_markdown")
    foreign, cited = fx.works[item.ids["work_key"]], fx.works["cited@second_test"]
    victim_md = item.ids["pointed_markdown_id"]
    home = variant(fx, tmp_path, [
        ("second_test", "INSERT INTO reference_entries (reference_id, citing_work_id, raw_reference_text, "
                        "resolved_work_id, resolution_status, resolution_source, confidence, markdown_id, markdown_hash, "
                        "created_at) VALUES ('ref_foreign_1', ?, 'Okafor, T. (2008). Survey of irrigation contracts.', "
                        "?, 'resolved', 'doi', 1.0, ?, ?, '2026-07-04T15:00:00+00:00')",
         (foreign.work_id, cited.work_id, victim_md, victim_md[3:])),
        ("second_test", "INSERT INTO citation_edges (source_work_id, target_work_id, provenance, confidence, "
                        "reference_id, run_id, created_at) VALUES (?, ?, 'parsed_bibliography', 1.0, 'ref_foreign_1', "
                        "'run-x', '2026-07-04T15:00:01+00:00')", (foreign.work_id, cited.work_id)),
    ])
    plan = plan_for(home)
    assert not any(r["reference_id"] == "ref_foreign_1" for r in plan.rows("reference_entries"))
    assert plan.ledger.get("second_test", "reference_entries", "ref_foreign_1").reason == "other_markdown_list"
    pid = paper_of(plan, foreign.slug, foreign.work_id)
    assert not any(r["source_paper_id"] == pid for r in plan.rows("citation_links"))
    assert plan.report["citation_edges_dropped"] == {"parsed_from_other_markdown": 1}
    assert plan.errors == [] and plan.ledger.unaccounted() == []


def test_a_v2_merge_that_already_collapsed_the_pair_is_satisfied(fx, tmp_path):
    item = fx.one("duplicate_verdict_same")
    victim = fx.works[item.ids["work_keys"][1]]
    home = variant(fx, tmp_path, [
        ("test_proj", "DELETE FROM identifiers WHERE work_id = ?", (victim.work_id,)),
        ("test_proj", "DELETE FROM project_documents WHERE work_id = ?", (victim.work_id,)),
        ("test_proj", "DELETE FROM works WHERE work_id = ?", (victim.work_id,)),
    ])
    plan = plan_for(home)
    d = plan.ledger.get("test_proj", "review_queue", item.ids["item_id"])
    assert (d.outcome, d.reason) == ("merged", "satisfied_by_v2_merge")
    assert item.ids["item_id"] not in {u["item_id"] for u in plan.report["unmapped_decisions"]}


def test_provider_cache_rows_are_copied(fx, tmp_path):
    r = run_migration(fx.home, tmp_path / "v3", run_id="mig-pc", now=NOW, process_lister=lambda: [],
                      disk_free=lambda p: 10 ** 15)
    conn = sqlite3.connect(f"file:{(tmp_path / 'v3/library/provider_cache.db').as_posix()}?mode=ro", uri=True)
    rows = conn.execute("SELECT provider, request_key FROM provider_cache ORDER BY provider_cache_id").fetchall()
    conn.close()
    assert len(rows) == 4 and rows[0] == ("crossref", "by_doi:doi=10.5555/sg.malformed-1")
    assert r.swapped
