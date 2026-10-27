"""Reference parsing for held papers during guided growth (docs/V3_CONCEPTS_DESIGN.md §8.2
step 3): a converted paper's reference list is parsed, resolved through the fake provider
chain, written as ``reference_entries`` and ``citation_links``, and the mid band becomes
``citation_review`` attention items. Synthetic PDFs, a fake converter, invented text."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from seedgraph.cache.marker_backend import MarkerResult
from seedgraph.library import files, graphs, growth
from seedgraph.library.store import open_library_at

from .test_files import text_pdf
from .test_growth import GrowthChain, counts, make_graph, pid_for, rec

TITLE = "Invented Ovens at Dawn"
BODY = " ".join(f"Sentence {k} describes an invented oven schedule in some detail." for k in range(60))
REFS = (
    "Baker, A. (2021). Invented yeast ledgers and their keepers. Journal of Invented Bread, 3, 1-20. "
    "doi:10.9999/yeast.1\n\n"
    "Miller, B. (2019). Crust census methods for imaginary towns. Imaginary Review, 7, 30-41.\n\n"
    "Salter, C. (2018). Salt roads across a pretend coast. Pretend Studies, 2, 5-9.\n"
)
MARKDOWN = f"# {TITLE}\n\n## Introduction\n\n{BODY}\n\n## References\n\n{REFS}"


class Backend:
    version = "fake-growth-1"

    def __init__(self, markdown: str):
        self.markdown = markdown

    def __call__(self, pdf_path, cfg):
        return MarkerResult(markdown=self.markdown, block_json=None, warnings=[])


@pytest.fixture
def lib(tmp_path):
    library = open_library_at(tmp_path / "home" / "library")
    try:
        yield library
    finally:
        library.close()


def converted(lib, tmp_path, paper_id="lp_a", markdown=MARKDOWN) -> str:
    now = datetime.now(timezone.utc).isoformat()
    with lib.write() as conn:
        conn.execute("INSERT INTO papers (paper_id, title, year, created_at) VALUES (?, ?, 2031, ?)",
                     (paper_id, TITLE, now))
        conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value) VALUES (?, 'doi', '10.9999/a')",
                     (paper_id,))
    res = files.attach_pdf(lib, paper_id, text_pdf(tmp_path / f"{paper_id}.pdf", [TITLE, "Body."]))
    files.convert_file(lib, res.source_file_id, backend=Backend(markdown))
    return paper_id


def chain() -> GrowthChain:
    return GrowthChain(
        records={"10.9999/yeast.1": rec("10.9999/yeast.1", "Invented yeast ledgers and their keepers", 2021)},
        titles={
            # a title hit with a strong id: resolved behind the gate
            "Crust census methods for imaginary towns": rec("10.9999/crust.2",
                                                            "Crust Census Methods for Imaginary Towns", 2019),
            # a title hit without any strong id: the mid band (ambiguous)
            "Salt roads across a pretend coast": {"title": "Salt roads across a pretend coast", "year": 2018},
        })


def entries(lib, paper_id="lp_a"):
    return lib.conn.execute(
        "SELECT reference_id, resolution_status, resolution_source, resolved_paper_id FROM reference_entries "
        "WHERE citing_paper_id = ? ORDER BY ordinal", (paper_id,)).fetchall()


def review_items(lib):
    return lib.conn.execute("SELECT item_id, paper_id, subject_json, status FROM attention_items "
                            "WHERE kind = 'citation_review' ORDER BY item_id").fetchall()


def test_a_held_papers_references_become_links_and_the_mid_band_review_items(lib, tmp_path):
    pid = converted(lib, tmp_path)

    res = growth.parse_references_for(lib, pid, chain=chain())

    assert (res.status, res.entries, res.resolved) == ("parsed", 3, 2)
    rows = entries(lib)
    assert [(r[1], r[2], r[3]) for r in rows] == [
        ("resolved", "doi", pid_for("10.9999/yeast.1")),
        ("resolved", "title_year", pid_for("10.9999/crust.2")),
        ("ambiguous", None, None)]
    links = lib.conn.execute("SELECT target_paper_id, provenance, reference_id FROM citation_links "
                             "WHERE source_paper_id = ? ORDER BY target_paper_id", (pid,)).fetchall()
    assert {(r[0], r[1], r[2]) for r in links} == {
        (pid_for("10.9999/yeast.1"), "parsed_bibliography", rows[0][0]),
        (pid_for("10.9999/crust.2"), "parsed_bibliography", rows[1][0])}
    assert set(res.new_papers) == {pid_for("10.9999/yeast.1"), pid_for("10.9999/crust.2")}
    (item,) = review_items(lib)
    assert item[1] == pid and json.loads(item[2]) == {"reference_id": rows[2][0]}
    assert res.review_items == (item[0],)


def test_parsing_the_same_markdown_again_changes_nothing(lib, tmp_path):
    pid = converted(lib, tmp_path)
    growth.parse_references_for(lib, pid, chain=chain())
    before, ids = counts(lib), entries(lib)

    again = growth.parse_references_for(lib, pid, chain=chain())

    assert again.status == "already_parsed"
    assert counts(lib) == before and entries(lib) == ids


def test_a_rejected_reference_is_not_raised_for_review_again(lib, tmp_path):
    pid = converted(lib, tmp_path)
    growth.parse_references_for(lib, pid, chain=chain())
    rid = entries(lib)[2][0]
    (item,) = review_items(lib)
    with lib.write() as conn:
        conn.execute("DELETE FROM attention_items WHERE item_id = ?", (item[0],))
        conn.execute("INSERT INTO citation_review_decisions (reference_id, decision, decided_at) "
                     "VALUES (?, 'reject', '2031-01-01')", (rid,))
        conn.execute("DELETE FROM citation_links WHERE source_paper_id = ?", (pid,))
        conn.execute("UPDATE reference_entries SET markdown_hash = 'older' WHERE citing_paper_id = ?", (pid,))

    growth.parse_references_for(lib, pid, chain=chain())  # the markdown "changed": re-parsed

    assert review_items(lib) == []
    assert entries(lib)[2][0] == rid  # same markdown text, same deterministic id
    assert lib.conn.execute("SELECT COUNT(*) FROM citation_links WHERE source_paper_id = ?", (pid,)).fetchone()[0] == 2


def test_papers_without_markdown_or_references_are_reported(lib, tmp_path):
    no_refs = converted(lib, tmp_path, "lp_n", f"# {TITLE}\n\n## Introduction\n\n{BODY}\n")
    assert growth.parse_references_for(lib, no_refs, chain=chain()).status == "no_references"
    with lib.write() as conn:
        conn.execute("INSERT INTO papers (paper_id, title, created_at) VALUES ('lp_c', 'Cited Only', '2031')")
    assert growth.parse_references_for(lib, "lp_c", chain=chain()).status == "no_markdown"


def test_the_walk_parses_unparsed_held_sources_and_ranks_their_references(lib, tmp_path):
    pid = converted(lib, tmp_path)
    make_graph(lib, "ovens", [pid])

    result = growth.walk(lib, "ovens", chain=chain(), depth=1, cap=10)

    assert [p.paper_id for p in result.parsed] == [pid]
    found = {f.paper_id: f.cited_by_members for f in growth.review(lib, "ovens")}
    assert found == {pid_for("10.9999/yeast.1"): 1, pid_for("10.9999/crust.2"): 1}
    # a link only carries the provenance that reported it: no provider ever named these
    provenances = {r[0] for r in lib.conn.execute(
        "SELECT provenance FROM citation_links WHERE source_paper_id = ?", (pid,))}
    assert provenances == {"parsed_bibliography"}
    with graphs.open_graph(lib, "ovens") as g:
        assert [r.paper_id for r in graphs.membership(g)] == [pid]
