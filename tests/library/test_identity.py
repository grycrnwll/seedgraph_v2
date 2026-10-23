"""Paper identity: key/id derivation and merge grouping (design §4). Invented data only."""

from __future__ import annotations

import hashlib
import random

from seedgraph.library.identity import (
    WorkRecord,
    WorkRef,
    derive_key,
    mint_paper_id,
    paper_id_for_key,
    plan_identity,
)

P1, P2, P3 = "second_test", "land_value", "test_proj"


def w(project, work_id, title=None, year=None, ids=(), pdfs=(), **fields):
    return WorkRecord(project=project, work_id=work_id, title=title, year=year,
                      ids=tuple(ids), pdf_hashes=tuple(pdfs), **fields)


def paper_of(plan, project, work_id):
    ref = WorkRef(project, work_id)
    (hit,) = [p for p in plan.papers if ref in p.members]
    return hit


# --- key and id derivation ------------------------------------------------------------


def test_paper_id_is_lp_plus_20_hex_of_sha256_of_key():
    key = "doi:10.9999/abc"
    assert paper_id_for_key(key) == "lp_" + hashlib.sha256(key.encode()).hexdigest()[:20]


def test_key_uses_normalised_doi_first():
    assert derive_key(ids=[("doi", "https://doi.org/10.9999/ABC"), ("openalex", "w123")]) == "doi:10.9999/abc"


def test_key_precedence_openalex_then_arxiv_then_pdf_then_title_then_v2():
    assert derive_key(ids=[("openalex", "https://openalex.org/w55"), ("arxiv", "2101.00001")]) == "openalex:W55"
    assert derive_key(ids=[("arxiv", "arXiv:2101.00001v3"), ("s2", "abc")]) == "arxiv:2101.00001"
    assert derive_key(pdf_hashes=["bb", "aa"], title="T", year=2001) == "pdf:aa"
    assert derive_key(title="Ovens of Paris!", year=1999) == "title:ovens of paris|1999"
    assert derive_key(title="Ovens of Paris", year=None) == "title:ovens of paris|"
    assert derive_key(title=" ?! ", v2_ref=WorkRef("p", "w9")) == "v2:p:w9"


def test_arxiv_doi_is_folded_to_the_arxiv_key():
    assert derive_key(ids=[("doi", "10.48550/arXiv.2101.00001")]) == "arxiv:2101.00001"


def test_alias_dois_smallest_wins():
    assert derive_key(ids=[("doi", "10.9999/zzz"), ("doi", "10.9999/aaa")]) == "doi:10.9999/aaa"


def test_mint_appends_deterministic_suffix_on_collision():
    base = paper_id_for_key("doi:10.9999/x")
    taken = {base, paper_id_for_key("doi:10.9999/x:2")}
    minted = mint_paper_id("doi:10.9999/x", taken)
    assert minted.key == "doi:10.9999/x:3"
    assert minted.paper_id == paper_id_for_key("doi:10.9999/x:3")
    assert minted.collided
    fresh = mint_paper_id("doi:10.9999/y", taken)
    assert (fresh.key, fresh.collided) == ("doi:10.9999/y", False)


# --- merge grouping -------------------------------------------------------------------


def test_double_import_by_doi_merges_and_records_the_joining_key():
    plan = plan_identity([
        w(P1, "a", "Ovens", 2001, [("doi", "10.9999/ovens")]),
        w(P2, "b", "Ovens", 2001, [("doi", "https://doi.org/10.9999/OVENS")]),
    ])
    assert len(plan.papers) == 1
    paper = plan.papers[0]
    assert paper.members == (WorkRef(P1, "a"), WorkRef(P2, "b"))
    assert paper.paper_id == paper_id_for_key("doi:10.9999/ovens")
    (merge,) = plan.rule_merges
    assert (merge.work_a, merge.work_b, merge.key_type, merge.key_value) == (
        WorkRef(P1, "a"), WorkRef(P2, "b"), "doi", "10.9999/ovens")


def test_preprint_with_arxiv_only_merges_with_published_doi_plus_arxiv():
    plan = plan_identity([
        w(P1, "pub", "Ovens", 2003, [("doi", "10.9999/ovens"), ("arxiv", "2101.00001")]),
        w(P2, "pre", "Ovens", 2001, [("doi", "10.48550/arXiv.2101.00001v2")]),
    ])
    assert len(plan.papers) == 1
    assert plan.rule_merges[0].key_type == "arxiv"


def test_identical_pdf_merges_non_conflicting_works():
    plan = plan_identity([
        w(P1, "a", "Ovens", 2001, [("doi", "10.9999/ovens")], ["h1"]),
        w(P2, "b", "Ovens", 2001, [("openalex", "W7")], ["h1"]),
    ])
    assert len(plan.papers) == 1
    assert plan.rule_merges[0].key_type == "pdf"


def test_conflicting_dois_block_the_identical_pdf_union():
    plan = plan_identity([
        w(P1, "a", "Ovens", 2001, [("doi", "10.9999/ovens")], ["h1"]),
        w(P2, "b", "Kilns", 2005, [("doi", "10.9999/kilns")], ["h1"]),
    ])
    assert len(plan.papers) == 2
    assert plan.rule_merges == ()
    (blocked,) = plan.blocked
    assert (blocked.key_type, blocked.key_value, blocked.conflict_types) == ("pdf", "h1", ("doi",))
    (shared,) = plan.shared_pdfs
    assert shared.pdf_hash == "h1"
    assert set(shared.paper_ids) == {p.paper_id for p in plan.papers}
    # A shared-PDF conflict is an attention item of its own, not a "check these" item.
    assert [c for c in plan.check_items if c.reason == "blocked_conflict"] == []


def test_dois_reached_through_openalex_are_blocked_and_listed_to_check():
    plan = plan_identity([
        w(P1, "a", "Ovens", 2001, [("doi", "10.9999/ovens"), ("openalex", "W1")]),
        w(P2, "b", "Ovens II", 2002, [("doi", "10.9999/ovens2"), ("openalex", "W1")]),
    ])
    assert len(plan.papers) == 2
    (item,) = plan.check_items
    assert item.reason == "blocked_conflict"
    assert item.key_type == "openalex"
    assert {c.paper_id for c in item.candidates} | {item.paper_id} == {p.paper_id for p in plan.papers}


def test_same_doi_but_conflicting_openalex_stays_split_with_suffixed_id():
    plan = plan_identity([
        w(P1, "a", "Ovens", 2001, [("doi", "10.9999/ovens"), ("openalex", "W1")]),
        w(P2, "b", "Ovens", 2001, [("doi", "10.9999/ovens"), ("openalex", "W2")]),
    ])
    assert len(plan.papers) == 2
    assert paper_of(plan, P1, "a").key == "doi:10.9999/ovens"
    assert paper_of(plan, P2, "b").key == "doi:10.9999/ovens:2"
    (collision,) = plan.collisions
    assert collision.base_key == "doi:10.9999/ovens"


def test_alias_dois_on_one_work_do_not_conflict_with_a_subset():
    plan = plan_identity([
        w(P1, "a", "Ovens", 2001, [("doi", "10.9999/ovens"), ("doi", "10.9999/ovens-alias")]),
        w(P2, "b", "Ovens", 2001, [("doi", "10.9999/ovens")]),
    ])
    assert len(plan.papers) == 1
    assert plan.papers[0].key == "doi:10.9999/ovens"


def test_no_id_work_joins_identified_group_through_shared_pdf():
    plan = plan_identity([
        w(P1, "a", "Ovens", 2001, [("doi", "10.9999/ovens")], ["h1"]),
        w(P3, "x", "ovens.pdf", None, [], ["h1"]),
    ])
    assert len(plan.papers) == 1
    assert plan.rule_merges[0].key_type == "pdf"


def test_no_id_work_on_a_pdf_shared_by_conflicting_groups_is_not_merged():
    plan = plan_identity([
        w(P1, "a", "Ovens", 2001, [("doi", "10.9999/ovens")], ["h1"]),
        w(P1, "b", "Kilns", 2005, [("doi", "10.9999/kilns")], ["h1"]),
        w(P3, "x", "Something", None, [], ["h1"]),
    ])
    assert len(plan.papers) == 3
    item = next(c for c in plan.check_items if WorkRef(P3, "x") in c.works)
    assert item.reason == "ambiguous_pdf"
    assert len(item.candidates) == 2


def test_no_id_works_sharing_only_each_others_pdf_stay_separate():
    # Design §4.2: no-id works merge only with an identified group.
    plan = plan_identity([w(P1, "x", "Ovens", 2001, [], ["h9"]), w(P2, "y", "Ovens", 2001, [], ["h9"])])
    assert len(plan.papers) == 2
    assert sorted(p.key for p in plan.papers) == ["pdf:h9", "pdf:h9:2"]


def test_no_id_work_merges_on_unique_exact_title_and_year():
    plan = plan_identity([
        w(P1, "a", "Ovens of Paris", 2001, [("doi", "10.9999/ovens")]),
        w(P3, "x", "OVENS of paris!", 2001, []),
        w(P3, "y", "Ovens of Paris", 2002, []),
    ])
    assert len(paper_of(plan, P1, "a").members) == 2
    assert WorkRef(P3, "x") in paper_of(plan, P1, "a").members
    assert plan.rule_merges[0].key_type == "title_year"
    assert paper_of(plan, P3, "y").members == (WorkRef(P3, "y"),)


def test_no_id_title_match_to_two_papers_is_ambiguous():
    plan = plan_identity([
        w(P1, "a", "Ovens", 2001, [("doi", "10.9999/a")]),
        w(P1, "b", "Ovens", 2001, [("doi", "10.9999/b")]),
        w(P3, "x", "Ovens", 2001, [], ["h1"]),
    ])
    assert len(plan.papers) == 3
    item = next(c for c in plan.check_items if WorkRef(P3, "x") in c.works)
    assert item.reason == "ambiguous_title"
    assert len(item.candidates) == 2


def test_no_id_title_without_year_does_not_merge():
    plan = plan_identity([w(P1, "a", "Ovens", None, [("doi", "10.9999/a")]), w(P3, "x", "Ovens", None, [])])
    assert len(plan.papers) == 2


def test_unmatched_no_id_held_work_is_listed_with_nearest_titles():
    plan = plan_identity([
        w(P1, "a", "Bayesian Thermostats for Coastal Bakeries", 2001, [("doi", "10.9999/a")]),
        w(P1, "b", "Volcanic Soil and Grape Yields", 2001, [("doi", "10.9999/b")]),
        w(P3, "x", "Bayesian Thermostats in Coastal Bakeries", 2019, [], ["h5"]),
        w(P3, "cited", "Unmatched cited work", 2019, []),
    ])
    (item,) = plan.check_items
    assert item.reason == "no_match"
    assert item.works == (WorkRef(P3, "x"),)
    assert item.candidates[0].paper_id == paper_of(plan, P1, "a").paper_id
    assert item.candidates[0].similarity > 0.5


def test_researcher_not_duplicates_is_never_merged_and_the_skip_is_reported():
    works = [
        w(P1, "a", "Ovens", 2001, [("doi", "10.9999/ovens")]),
        w(P2, "b", "Ovens", 2001, [("doi", "10.9999/ovens"), ("openalex", "W1")]),
        w(P3, "c", "Ovens", 2001, [("openalex", "W1")]),
    ]
    # The researcher said a and c are different; c would reach a through b's OpenAlex id.
    plan = plan_identity(works, never_merge=[(WorkRef(P3, "c"), WorkRef(P1, "a"))])
    assert paper_of(plan, P1, "a").members == (WorkRef(P1, "a"), WorkRef(P2, "b"))
    assert paper_of(plan, P3, "c").members == (WorkRef(P3, "c"),)
    (skip,) = plan.skipped_never_merge
    assert skip.key_type == "openalex"
    assert skip.decision == (WorkRef(P1, "a"), WorkRef(P3, "c"))
    assert [c.reason for c in plan.check_items] == ["never_merge_skipped"]


def test_field_winner_is_highest_precedence_project_then_lowest_work_id():
    plan = plan_identity([
        w("oct2026test", "a", "Late Title", 2001, [("doi", "10.9999/x")], venue="V-oct"),
        w(P2, "z", "Land Title", None, [("doi", "10.9999/x")], abstract="abs-land"),
        w(P2, "m", None, 2002, [("doi", "10.9999/x")]),
        w("dbg_probe", "0", "Debug", 1990, [("doi", "10.9999/x")]),
    ])
    (paper,) = plan.papers
    assert paper.members[0] == WorkRef(P2, "m")
    assert paper.members[-1] == WorkRef("dbg_probe", "0")
    # first non-empty value in member order
    assert (paper.title, paper.year, paper.venue, paper.abstract) == ("Land Title", 2002, "V-oct", "abs-land")


def test_plan_is_independent_of_input_order():
    works = [
        w(P1, "a", "Ovens", 2001, [("doi", "10.9999/ovens")], ["h1"]),
        w(P2, "b", "Kilns", 2005, [("doi", "10.9999/kilns")], ["h1"]),
        w(P2, "c", "Ovens", 2001, [("openalex", "W1"), ("doi", "10.9999/ovens")]),
        w(P3, "x", "Ovens", 2001, []),
        w(P3, "y", "Breads of Lyon", 2010, [], ["h2"]),
        w(P1, "d", "Same Doi Other OA", 2001, [("doi", "10.9999/q"), ("openalex", "W8")]),
        w(P2, "e", "Same Doi Other OA", 2001, [("doi", "10.9999/q"), ("openalex", "W9")]),
    ]
    expected = plan_identity(works, never_merge=[(WorkRef(P3, "x"), WorkRef(P2, "c"))])
    for seed in range(5):
        shuffled = works[:]
        random.Random(seed).shuffle(shuffled)
        got = plan_identity(shuffled, never_merge=[(WorkRef(P2, "c"), WorkRef(P3, "x"))])
        assert got == expected
