"""Adding papers (docs/V3_UI_DESIGN.md §6 and the researcher's rules of 2026-10-26): dropped
PDFs and folders, matching to cited-only papers, the explicit attach, DOI paste with an
open-access fetch, and the needs-a-PDF list. Synthetic PDFs, a fake provider chain and a
fake fetch; invented text; no network."""

from __future__ import annotations

import json

import httpx
import pytest

from seedgraph.acquisition.fetch import FetchResult, OACandidate, fetch_oa
from seedgraph.library import graphs, ingest
from seedgraph.library.store import LibraryError, open_library_at

from .metadata_support import FakeChain, dois_of, insert_paper, items, openalex_record, paper_row
from .test_files import text_pdf

TITLE = "Invented Ovens and the Morning Loaf"
DOI = "10.9999/oven.2031.1"
OTHER_TITLE = "Gadget Prices in an Invented Harbour Town"


@pytest.fixture
def lib(tmp_path):
    library = open_library_at(tmp_path / "home" / "library")
    try:
        yield library
    finally:
        library.close()


def status_of(lib, paper_id):
    return lib.conn.execute("SELECT status FROM paper_status WHERE paper_id = ?", (paper_id,)).fetchone()[0]


def source_file(lib, source_file_id):
    return lib.conn.execute("SELECT access_class, acquisition_method FROM source_files WHERE source_file_id = ?",
                            (source_file_id,)).fetchone()


def paper_count(lib) -> int:
    return lib.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]


# --- dropped PDFs ----------------------------------------------------------------------------


def test_a_cited_only_paper_becomes_held_when_its_pdf_with_its_doi_arrives(lib, tmp_path):
    pid = insert_paper(lib, "lp_cited1", title=TITLE, year=2031, dois=((DOI, "v2"),))
    pdf = text_pdf(tmp_path / "drop.pdf", [f"{TITLE} Ada Inventor doi:{DOI}", "Body."])

    (res,) = ingest.add_pdfs(lib, [pdf])

    assert (res.status, res.paper_id, res.matched_by) == ("attached", pid, "doi")
    assert status_of(lib, pid) == "held"
    assert source_file(lib, res.source_file_id)["access_class"] == "user_supplied_private"
    assert res.attach is not None and res.attach.queued
    assert paper_count(lib) == 1


def test_a_title_and_year_match_attaches_to_the_cited_only_paper(lib, tmp_path):
    pid = insert_paper(lib, "lp_cited2", title=TITLE, year=2031)
    insert_paper(lib, "lp_cited3", title=OTHER_TITLE, year=2031)
    pdf = text_pdf(tmp_path / "drop.pdf", [f"{TITLE.upper()} Ada Inventor, March 2031", "Body."])

    (res,) = ingest.add_pdfs(lib, [pdf])

    assert (res.status, res.paper_id, res.matched_by) == ("attached", pid, "title_year")
    assert status_of(lib, pid) == "held"


def test_a_title_match_in_a_different_year_does_not_attach(lib, tmp_path):
    insert_paper(lib, "lp_cited4", title=TITLE, year=1990)
    (res,) = ingest.add_pdfs(lib, [text_pdf(tmp_path / "drop.pdf", [f"{TITLE} 2031", "Body."])])
    assert res.status == "new_paper" and res.paper_id != "lp_cited4"


def test_a_cited_work_in_the_dropped_pdfs_reference_list_is_not_matched(lib, tmp_path):
    insert_paper(lib, "lp_cited5", title=OTHER_TITLE, year=2004)
    body = " ".join(f"Sentence {k} about invented ovens." for k in range(120))
    pdf = text_pdf(tmp_path / "drop.pdf", [f"{TITLE} by Ada Inventor. {body}",
                                           f"References. Older, C. (2004). {OTHER_TITLE}. Invented Journal."])
    (res,) = ingest.add_pdfs(lib, [pdf])
    assert res.status == "new_paper" and status_of(lib, "lp_cited5") == "cited_only"


def test_a_cited_only_paper_with_a_generic_title_is_never_matched(lib, tmp_path):
    insert_paper(lib, "lp_generic", title="Review", year=2031)
    (res,) = ingest.add_pdfs(lib, [text_pdf(tmp_path / "drop.pdf", [f"A Review of {TITLE}, 2031", "Body."])])
    assert res.status == "new_paper" and status_of(lib, "lp_generic") == "cited_only"


def test_no_match_makes_a_new_untitled_paper_with_metadata_needed(lib, tmp_path):
    pdf = text_pdf(tmp_path / "drop.pdf", [f"{TITLE} doi:{DOI}", "Body."])

    (res,) = ingest.add_pdfs(lib, [pdf])

    assert res.status == "new_paper" and res.matched_by is None
    assert paper_row(lib, res.paper_id)["title"] is None
    assert status_of(lib, res.paper_id) == "held"
    assert dois_of(lib, res.paper_id) == [(DOI, "pdf_scan")]
    (item,) = items(lib, "metadata_needed")
    assert item["paper_id"] == res.paper_id and item["item_id"] in res.attention_item_ids


def test_the_same_file_twice_is_reported_as_a_duplicate(lib, tmp_path):
    pdf = text_pdf(tmp_path / "drop.pdf", ["An invented cover page", "Body."])
    (first,) = ingest.add_pdfs(lib, [pdf])
    copy = tmp_path / "copy.pdf"
    copy.write_bytes(pdf.read_bytes())

    (again,) = ingest.add_pdfs(lib, [copy])

    assert again.status == "duplicate_file" and again.paper_id == first.paper_id
    assert again.source_file_id == first.source_file_id
    assert paper_count(lib) == 1


def test_a_file_converted_earlier_without_a_paper_is_not_a_duplicate(lib, tmp_path):
    from seedgraph.cache.ingest import ingest_file
    from seedgraph.vocab import AccessClass, AcquisitionMethod

    pdf = text_pdf(tmp_path / "drop.pdf", ["An invented cover page", "Body."])
    ingest_file(pdf, access_class=AccessClass.user_supplied_private,
                acquisition_method=AcquisitionMethod.upload, library=lib)
    (res,) = ingest.add_pdfs(lib, [pdf])
    assert res.status == "new_paper"


def test_two_title_year_candidates_make_a_new_paper_and_an_identity_check(lib, tmp_path):
    insert_paper(lib, "lp_twin1", title=TITLE, year=2031)
    insert_paper(lib, "lp_twin2", title=TITLE + "†", year=2031)
    (res,) = ingest.add_pdfs(lib, [text_pdf(tmp_path / "drop.pdf", [f"{TITLE} 2031", "Body."])])

    assert res.status == "new_paper"
    (item,) = items(lib, "identity_check")
    subject = json.loads(item["subject_json"])
    assert subject["paper_id"] == res.paper_id and subject["candidates"] == ["lp_twin1", "lp_twin2"]
    assert status_of(lib, "lp_twin1") == status_of(lib, "lp_twin2") == "cited_only"


def test_a_doi_held_by_a_paper_with_another_pdf_is_flagged_not_merged(lib, tmp_path):
    insert_paper(lib, "lp_held1", title=TITLE, year=2031, dois=((DOI, "v2"),))
    ingest.attach_to_paper(lib, "lp_held1", text_pdf(tmp_path / "first.pdf", [TITLE, "One."]))

    (res,) = ingest.add_pdfs(lib, [text_pdf(tmp_path / "second.pdf", [f"{TITLE} doi:{DOI}", "Two."])])

    assert res.status == "new_paper" and res.paper_id != "lp_held1"
    (item,) = items(lib, "identity_check")
    assert json.loads(item["subject_json"])["candidates"] == ["lp_held1"]


def test_files_that_are_not_pdfs_are_skipped(lib, tmp_path):
    fake = tmp_path / "notes.pdf"
    fake.write_text("not a pdf at all", encoding="utf-8")
    (res,) = ingest.add_pdfs(lib, [fake, ])
    assert res.status == "skipped" and paper_count(lib) == 0


def test_a_folder_is_walked_for_pdfs_and_read_in_place(lib, tmp_path):
    folder = tmp_path / "inbox"
    (folder / "sub").mkdir(parents=True)
    text_pdf(folder / "a.pdf", ["First invented cover", "x"])
    text_pdf(folder / "sub" / "b.PDF", ["Second invented cover", "y"])
    (folder / "readme.txt").write_text("ignored", encoding="utf-8")

    results = ingest.add_folder(lib, folder)

    assert [r.status for r in results] == ["new_paper", "new_paper"]
    assert {source_file(lib, r.source_file_id)["acquisition_method"] for r in results} == {"folder_import"}
    assert (folder / "a.pdf").is_file()
    with pytest.raises(LibraryError):
        ingest.add_folder(lib, tmp_path / "missing")


# --- explicit attach -------------------------------------------------------------------------


def test_the_explicit_attach_bypasses_matching(lib, tmp_path):
    insert_paper(lib, "lp_other", title=TITLE, year=2031, dois=((DOI, "v2"),))
    chosen = insert_paper(lib, "lp_chosen", title=OTHER_TITLE)
    pdf = text_pdf(tmp_path / "drop.pdf", [f"{TITLE} doi:{DOI}", "Body."])

    res = ingest.attach_to_paper(lib, chosen, pdf)

    assert (res.status, res.paper_id, res.matched_by) == ("attached", chosen, "explicit")
    assert status_of(lib, chosen) == "held" and status_of(lib, "lp_other") == "cited_only"
    assert source_file(lib, res.source_file_id)["access_class"] == "user_supplied_private"
    # its title is not on the PDF: flagged by the standing check, never blocked
    assert res.attach.check.status == "miss"


def test_the_explicit_attach_refuses_an_unknown_paper(lib, tmp_path):
    with pytest.raises(LibraryError):
        ingest.attach_to_paper(lib, "lp_nobody", text_pdf(tmp_path / "a.pdf", ["x"]))


# --- DOI paste -------------------------------------------------------------------------------


def oa_fetch(content: bytes | None, provider: str = "arxiv", url: str = "https://arxiv.org/pdf/2031.00001"):
    calls: list[dict] = []

    async def fetch(record):
        calls.append(record)
        if content is None:
            return FetchResult(status="stub")
        return FetchResult(status="downloaded", content=content,
                           candidate=OACandidate(url=url, provider=provider, is_oa_asserted=True))

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


def test_a_pasted_doi_makes_a_cited_only_paper_from_the_provider(lib):
    chain = FakeChain({DOI: openalex_record(DOI, TITLE + "†", 2031, authors=("Ada Inventor",))})

    (res,) = ingest.add_dois(lib, [f"https://doi.org/{DOI.upper()}"], chain=chain, fetch=oa_fetch(None))

    assert res.status == "created" and res.doi == DOI and res.oa == "none"
    row = paper_row(lib, res.paper_id)
    assert (row["title"], row["title_source"], row["year"]) == (TITLE, "openalex", 2031)
    assert status_of(lib, res.paper_id) == "cited_only"
    assert dois_of(lib, res.paper_id) == [(DOI, "openalex")]


def test_a_pasted_doi_with_an_open_access_copy_becomes_held(lib, tmp_path):
    chain = FakeChain({DOI: openalex_record(DOI, TITLE, 2031)})
    pdf_bytes = text_pdf(tmp_path / "oa.pdf", [f"{TITLE} doi:{DOI}", "Body."]).read_bytes()

    (res,) = ingest.add_dois(lib, [DOI], chain=chain, fetch=oa_fetch(pdf_bytes))

    assert res.status == "created" and res.oa == "held"
    assert res.oa_source == "arxiv: https://arxiv.org/pdf/2031.00001"
    assert status_of(lib, res.paper_id) == "held"
    sf = lib.conn.execute("SELECT source_file_id FROM paper_files WHERE paper_id = ?", (res.paper_id,)).fetchone()[0]
    assert tuple(source_file(lib, sf)) == ("open_access", "open_access_fetch")
    assert not any((lib.root.parent / "tmp").rglob("*.pdf"))


def test_a_generic_provider_title_is_not_saved_at_doi_paste(lib):
    chain = FakeChain({DOI: openalex_record(DOI, "Front matter", 2031)})
    (res,) = ingest.add_dois(lib, [DOI], chain=chain, fetch=oa_fetch(None))
    assert res.status == "created" and res.title is None
    assert paper_row(lib, res.paper_id)["title"] is None
    assert len(items(lib, "metadata_generic_title")) == 1 and len(items(lib, "metadata_needed")) == 1


def test_a_known_doi_reports_the_existing_paper_and_changes_nothing(lib):
    pid = insert_paper(lib, "lp_known", title="My Own Name for It", title_source="user", dois=((DOI, "v2"),))
    chain = FakeChain({DOI: openalex_record(DOI, TITLE, 2031)})
    (res,) = ingest.add_dois(lib, [DOI], chain=chain, fetch=oa_fetch(None))
    assert (res.status, res.paper_id) == ("existing", pid)
    assert tuple(paper_row(lib, pid))[:2] == ("My Own Name for It", "user")


def test_unknown_and_invalid_dois_create_nothing(lib):
    results = ingest.add_dois(lib, ["10.9999/nobody.knows", "not a doi"], chain=FakeChain(), fetch=oa_fetch(None))
    assert [r.status for r in results] == ["not_found", "invalid"]
    assert paper_count(lib) == 0


def test_a_pasted_doi_resembling_an_existing_paper_raises_an_identity_check(lib):
    existing = insert_paper(lib, "lp_similar", title=TITLE, year=2031)
    chain = FakeChain({DOI: openalex_record(DOI, TITLE + ".", 2031)})

    (res,) = ingest.add_dois(lib, [DOI], chain=chain, fetch=oa_fetch(None))

    assert res.status == "created" and res.paper_id != existing
    (item,) = items(lib, "identity_check")
    assert json.loads(item["subject_json"])["candidates"] == [existing]
    assert item["item_id"] in res.attention_item_ids


class PaywalledChain(FakeChain):
    """A provider offering only a publisher copy nobody asserted is open access."""

    async def oa_pdf_candidates(self, record):
        return [OACandidate(url="https://publisher.invalid/paywalled.pdf", provider="crossref",
                            is_oa_asserted=False)]


def test_a_paywalled_copy_is_never_fetched(lib):
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, content=b"%PDF-1.4 should never be fetched")

    chain = PaywalledChain({DOI: openalex_record(DOI, TITLE, 2031)})

    async def fetch(record):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_oa(record, chain=chain, client=client)

    (res,) = ingest.add_dois(lib, [DOI], chain=chain, fetch=fetch)

    assert res.status == "created" and res.oa == "none"
    assert requests == []
    assert status_of(lib, res.paper_id) == "cited_only"


# --- needs a PDF -----------------------------------------------------------------------------


def test_needs_a_pdf_lists_cited_only_papers_with_doi_links(lib, tmp_path):
    insert_paper(lib, "lp_n1", title=OTHER_TITLE, year=2030, dois=((DOI, "v2"),))
    insert_paper(lib, "lp_n2", title=TITLE, year=2031)
    held = insert_paper(lib, "lp_n3", title="A Held Invented Paper")
    ingest.attach_to_paper(lib, held, text_pdf(tmp_path / "h.pdf", ["A Held Invented Paper", "x"]))

    rows = ingest.needs_pdf(lib)

    assert [r.paper_id for r in rows] == ["lp_n1", "lp_n2"]
    assert rows[0].doi == DOI and rows[0].doi_url == f"https://doi.org/{DOI}"
    assert rows[1].doi is None and rows[1].doi_url is None


def test_needs_a_pdf_can_be_scoped_to_a_graph_frontier(lib):
    insert_paper(lib, "lp_f1", title=TITLE, dois=((DOI, "v2"),))
    insert_paper(lib, "lp_f2", title=OTHER_TITLE)
    insert_paper(lib, "lp_f3", title="Excluded Invented Paper")
    with graphs.create_graph(lib, "ovens") as g:
        graphs.set_membership(g, "lp_f1", status="frontier")
        graphs.set_membership(g, "lp_f3", status="excluded")

    scoped = ingest.needs_pdf(lib, graph="ovens")
    assert [r.paper_id for r in scoped] == ["lp_f1"] and scoped[0].graphs == ("ovens",)
    assert [r.paper_id for r in ingest.needs_pdf(lib, frontier=True)] == ["lp_f1"]
    assert {r.paper_id for r in ingest.needs_pdf(lib)} == {"lp_f1", "lp_f2", "lp_f3"}
