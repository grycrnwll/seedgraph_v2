"""The highlight measurement (design §3.4): page-assigned, verified against the PDF text
layer, page-only and page-wrong; prototype vs current modes. Synthetic PDFs and
invented text only."""

from __future__ import annotations

import hashlib
import json

import pytest

from seedgraph.library import pagecheck
from seedgraph.library.store import open_library_at

from .test_pages import PAGES, make_paper, paged_md, plain_md, save_quotes

PROJ = "invented_proj"


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


@pytest.fixture
def lib(home):
    library = open_library_at(home / "library")
    try:
        yield library
    finally:
        library.close()


def mark_from_v2(lib, paper_id: str, work_id: str) -> None:
    """Make the paper's notes and id map look carried from v2 project ``PROJ``."""
    with lib.write() as conn:
        conn.execute("UPDATE structured_notes SET origin = ? WHERE paper_id = ?", (f"v2:{PROJ}", paper_id))
        conn.execute("INSERT INTO v2_id_map (v2_project, v2_table, v2_id, v3_table, v3_id) "
                     "VALUES (?, 'works', ?, 'papers', ?)", (PROJ, work_id, paper_id))


def write_sets(home, *, proto5_hashes=(), proto31=()) -> None:
    ui = home / "ui"
    ui.mkdir(parents=True, exist_ok=True)
    (ui / "highlight_sets.json").write_text(json.dumps({
        "proto5": {"v2_project": PROJ, "pdf_sha256": list(proto5_hashes)},
        "proto31": {"v2_project": PROJ, "papers": [
            {"v2_work_id": w, "claim_prefixes": [c[6:14] for c in claims]} for w, claims in proto31]},
    }), encoding="utf-8")


def file_hash(lib, source_file_id: str) -> str:
    return lib.conn.execute("SELECT file_hash FROM source_files WHERE source_file_id = ?",
                            (source_file_id,)).fetchone()[0]


# --- tiers ------------------------------------------------------------------------------


def test_each_quote_gets_a_tier_against_the_pdf_text_layer(lib, home, tmp_path):
    other = "Nobody printed this invented sentence in the PDF."
    misplaced = "Bakers in the story share one long table."
    md_pages = [PAGES[0], PAGES[1] + " " + misplaced + " " + other, "A different third page."]
    pid, _, md = make_paper(lib, tmp_path, paged_md(md_pages), pdf_pages=PAGES)
    ok, wrong, missing = save_quotes(lib, pid, md, ["Its ovens warm slowly.", misplaced, other])

    report = pagecheck.run(lib, home, set_name="current", mode="current")

    tiers = {q.claim_id: q.tier for q in report.quotes}
    assert tiers == {ok: "verified", wrong: "page_wrong", missing: "page_only"}
    [row] = report.papers
    assert (row.quotes, row.page_assigned, row.verified, row.page_only, row.page_wrong) == (3, 3, 1, 1, 1)
    assert report.totals["page_wrong"] == 1
    assert report.verdict == "fail" and any("page-wrong" in r for r in report.reasons)
    found = {q.claim_id: q.found_pages for q in report.quotes}
    assert found[wrong] == (3,)


def test_a_short_needle_found_elsewhere_is_page_only_not_page_wrong(lib, home, tmp_path):
    # "$x$ ovens $y$" folds to "ovens" (5 letters): its hit on PDF page 1 is chance.
    md_pages = [PAGES[0], PAGES[1] + " As $x$ ovens $y$ grow.", PAGES[2]]
    pid, _, md = make_paper(lib, tmp_path, paged_md(md_pages), pdf_pages=PAGES)
    [short] = save_quotes(lib, pid, md, ["$x$ ovens $y$"])

    [q] = pagecheck.run(lib, home, set_name="current", mode="current").quotes

    assert (q.claim_id, q.page, q.tier, q.found_pages) == (short, 2, "page_only", (1,))


def test_a_quote_without_a_page_is_not_verified_and_its_paper_is_listed(lib, home, tmp_path):
    pid, _, md = make_paper(lib, tmp_path, plain_md(PAGES))
    save_quotes(lib, pid, md, ["Its ovens warm slowly."])

    report = pagecheck.run(lib, home, set_name="current", mode="current")

    [q] = report.quotes
    assert (q.how, q.page, q.tier) == ("none", None, "no_page")
    [listed] = report.no_marker_papers
    assert listed["paper_id"] == pid and listed["converter_version"] == "fake-fixed-1"
    assert report.totals["page_assigned"] == 0


def test_verification_can_be_skipped(lib, home, tmp_path):
    pid, _, md = make_paper(lib, tmp_path, paged_md(PAGES))
    save_quotes(lib, pid, md, ["Its ovens warm slowly."])

    report = pagecheck.run(lib, home, set_name="current", mode="current", verify=False)

    assert [q.tier for q in report.quotes] == ["not_checked"]
    assert report.totals["page_assigned"] == 1


# --- sets and modes ----------------------------------------------------------------------


def two_versions(lib, tmp_path):
    """A paged paper whose note has two versions; returns (paper, file, v1 claims, v2 claims)."""
    pid, sf, md = make_paper(lib, tmp_path, paged_md(PAGES))
    v1 = save_quotes(lib, pid, md, ["Its ovens warm slowly.", "The index falls in spring.", None])
    v2 = save_quotes(lib, pid, md, ["Bakers in the story share one long table."])
    return pid, sf, v1, v2


def test_prototype_mode_counts_exactly_the_stored_claims_across_note_versions(lib, home, tmp_path):
    pid, _, v1, v2 = two_versions(lib, tmp_path)
    mark_from_v2(lib, pid, "work_invented_a")
    write_sets(home, proto31=[("work_invented_a", v1[:2] + v2)])

    proto = pagecheck.run(lib, home, set_name="proto31", mode="prototype")
    current = pagecheck.run(lib, home, set_name="proto31", mode="current")

    assert sorted(q.claim_id for q in proto.quotes) == sorted(v1[:2] + v2)
    assert [q.claim_id for q in current.quotes] == v2  # the current note only
    assert proto.totals["quotes"] == 3 and proto.totals["page_assigned"] == 3
    assert proto.unresolved == []


def test_a_stored_prefix_that_matches_no_claim_is_reported_unresolved(lib, home, tmp_path):
    pid, _, v1, _ = two_versions(lib, tmp_path)
    mark_from_v2(lib, pid, "work_invented_a")
    write_sets(home, proto31=[("work_invented_a", [v1[0], "claim_ffffffff"]), ("work_missing", [])])

    report = pagecheck.run(lib, home, set_name="proto31", mode="prototype")

    assert [q.claim_id for q in report.quotes] == [v1[0]]
    assert sorted(report.unresolved) == ["claim prefix ffffffff of work_invented_a", "v2 work work_missing"]


def test_proto5_resolves_papers_by_pdf_hash_and_counts_notes_from_the_prototype_project(lib, home, tmp_path):
    pid, sf, v1, v2 = two_versions(lib, tmp_path)
    mark_from_v2(lib, pid, "work_invented_a")
    other, osf, _ = make_paper(lib, tmp_path, paged_md(PAGES), paper_id="lp_00000000000000000043",
                               name="b.pdf")
    write_sets(home, proto5_hashes=[file_hash(lib, sf), hashlib.sha256(b"absent").hexdigest()])

    report = pagecheck.run(lib, home, set_name="proto5", mode="prototype")

    assert [r.paper_id for r in report.papers] == [pid]
    assert sorted(q.claim_id for q in report.quotes) == sorted(v1[:2] + v2)
    assert report.unresolved == [f"pdf {hashlib.sha256(b'absent').hexdigest()}"]


def test_prototype_mode_needs_a_prototype_set(lib, home):
    with pytest.raises(pagecheck.PagecheckError):
        pagecheck.run(lib, home, set_name="current", mode="prototype")


def test_a_missing_sets_file_is_refused(lib, home):
    with pytest.raises(pagecheck.PagecheckError, match="highlight_sets.json"):
        pagecheck.run(lib, home, set_name="proto5", mode="prototype")


# --- expected figures and the report -------------------------------------------------------


def write_expected(home, set_name: str, quoted: int, assigned: int) -> None:
    (home / "ui" / "highlight_expected.json").write_text(json.dumps(
        {set_name: {"quoted": quoted, "page_assigned": assigned}}), encoding="utf-8")


def test_prototype_figures_are_the_bar(lib, home, tmp_path):
    pid, _, v1, v2 = two_versions(lib, tmp_path)
    mark_from_v2(lib, pid, "work_invented_a")
    write_sets(home, proto31=[("work_invented_a", v1[:2] + v2)])

    write_expected(home, "proto31", 3, 2)
    assert pagecheck.run(lib, home, set_name="proto31", mode="prototype").verdict == "pass"

    write_expected(home, "proto31", 4, 4)
    failed = pagecheck.run(lib, home, set_name="proto31", mode="prototype")
    assert failed.verdict == "fail"
    assert any("quotes" in r for r in failed.reasons) and any("page-assigned" in r for r in failed.reasons)


def test_proto5_must_equal_the_expected_total(lib, home, tmp_path):
    pid, sf, v1, v2 = two_versions(lib, tmp_path)
    mark_from_v2(lib, pid, "work_invented_a")
    write_sets(home, proto5_hashes=[file_hash(lib, sf)])
    write_expected(home, "proto5", 3, 3)

    assert pagecheck.run(lib, home, set_name="proto5", mode="prototype").verdict == "pass"


def test_the_report_is_written_under_the_home(lib, home, tmp_path):
    pid, _, md = make_paper(lib, tmp_path, paged_md(PAGES))
    save_quotes(lib, pid, md, ["Its ovens warm slowly."])
    report = pagecheck.run(lib, home, set_name="current", mode="current")

    out = pagecheck.write_report(report, home, stamp="20261026T120000")

    assert out == home / "ui" / "pagecheck" / "20261026T120000"
    data = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert data["totals"]["verified"] == 1 and data["set"] == "current" and data["mode"] == "current"
    assert "verified" in (out / "report.md").read_text(encoding="utf-8")


def test_read_only_open_never_migrates_or_writes(lib, home, tmp_path):
    pid, _, md = make_paper(lib, tmp_path, paged_md(PAGES))
    save_quotes(lib, pid, md, ["Its ovens warm slowly."])
    before = (home / "library" / "library.db").read_bytes()

    with pagecheck.open_readonly(home / "library") as ro:
        report = pagecheck.run(ro, home, set_name="current", mode="current")
        with pytest.raises(Exception):
            ro.conn.execute("INSERT INTO library_meta (key, value) VALUES ('x', 'y')")

    assert report.totals["verified"] == 1
    assert (home / "library" / "library.db").read_bytes() == before
