"""DOI verification against the providers (docs/V3_UI_DESIGN.md §7). Fake provider chain,
synthetic PDFs, invented text; no network."""

from __future__ import annotations

import json

import pytest

from seedgraph.library import metadata_verify
from seedgraph.library.store import open_library_at

from .metadata_support import (FakeChain, crossref_record, dois_of, held_paper, insert_paper, items,
                               openalex_record, paper_row)

TITLE = "Invented Ovens and the Morning Loaf"
DOI = "10.9999/oven.2031.1"
CITED_TITLE = "A Different Invented Study of Gadget Prices"


@pytest.fixture
def lib(tmp_path):
    library = open_library_at(tmp_path / "home" / "library")
    try:
        yield library
    finally:
        library.close()


def cover(title: str = TITLE) -> list[str]:
    return [f"{title} by Ada Inventor and Bo Example, Journal of Invented Results 2031",
            "Invented body text about ovens."]


def test_a_matching_provider_title_wins_with_its_values(lib, tmp_path):
    pid = held_paper(lib, tmp_path, "lp_v1", cover(), title="Invented ovens and the morning loaf†",
                     title_source="ollama", year=2030, dois=((DOI, "ollama"),))
    chain = FakeChain({DOI: openalex_record(DOI, TITLE, 2031, authors=("Ada Inventor", "Bo Example"),
                                            venue="Journal of Invented Results")})

    out = metadata_verify.verify(lib, pid, chain=chain)

    assert out.status == "verified" and out.provider == "openalex"
    row = paper_row(lib, pid)
    assert (row["title"], row["title_source"], row["year"], row["venue"]) == (
        TITLE, "openalex", 2031, "Journal of Invented Results")
    assert json.loads(row["authors_json"]) == ["Ada Inventor", "Bo Example"]
    assert dois_of(lib, pid) == [(DOI, "ollama")]
    assert chain.calls == [DOI]


def test_a_crossref_answer_is_recorded_as_crossref(lib, tmp_path):
    pid = held_paper(lib, tmp_path, "lp_v2", cover(), title=TITLE, title_source="ollama", dois=((DOI, "ollama"),))
    out = metadata_verify.verify(lib, pid, chain=FakeChain({DOI: crossref_record(DOI, TITLE, 2031)}))
    assert out.status == "verified" and paper_row(lib, pid)["title_source"] == "crossref"


def test_a_misread_title_loses_to_the_provider_title_found_in_the_pdf(lib, tmp_path):
    pid = held_paper(lib, tmp_path, "lp_v3", cover(), title="Journal of Invented Results",
                     title_source="ollama", dois=((DOI, "ollama"),))
    out = metadata_verify.verify(lib, pid, chain=FakeChain({DOI: openalex_record(DOI, TITLE, 2031)}))
    assert out.status == "provider_wins"
    assert tuple(paper_row(lib, pid))[:3] == (TITLE, "openalex", 2031)


def test_a_reference_list_doi_is_removed_with_an_item(lib, tmp_path):
    pid = held_paper(lib, tmp_path, "lp_v4", cover(), title=TITLE, title_source="ollama", year=2031,
                     dois=((DOI, "ollama"),))
    out = metadata_verify.verify(lib, pid, chain=FakeChain({DOI: openalex_record(DOI, CITED_TITLE, 2004)}))

    assert out.status == "doi_removed"
    assert dois_of(lib, pid) == []
    assert tuple(paper_row(lib, pid))[:3] == (TITLE, "ollama", 2031)
    (item,) = items(lib, "metadata_doi_mismatch")
    assert item["paper_id"] == pid and DOI in item["subject_json"] and CITED_TITLE in item["detail"]
    assert out.attention_item_id == item["item_id"]


def test_a_cited_title_in_the_reference_list_does_not_count_as_found(lib, tmp_path):
    # A short paper: its reference list (with the cited work's full title) is on the first pages.
    body = " ".join(f"Sentence {k} about invented ovens." for k in range(120))
    pages = [f"{TITLE} by Ada Inventor. {body}",
             f"References. Older, C. (2004). {CITED_TITLE}. doi:{DOI}"]
    pid = held_paper(lib, tmp_path, "lp_v4b", pages, title=TITLE, title_source="ollama", dois=((DOI, "ollama"),))
    out = metadata_verify.verify(lib, pid, chain=FakeChain({DOI: openalex_record(DOI, CITED_TITLE, 2004)}))
    assert out.status == "doi_removed"
    assert paper_row(lib, pid)["title"] == TITLE


def test_a_mismatched_doi_from_a_trusted_source_is_kept_but_flagged(lib, tmp_path):
    pid = held_paper(lib, tmp_path, "lp_v5", cover(), title=TITLE, title_source="v2", dois=((DOI, "v2"),))
    out = metadata_verify.verify(lib, pid, chain=FakeChain({DOI: openalex_record(DOI, CITED_TITLE, 2004)}))
    assert out.status == "doi_mismatch"
    assert dois_of(lib, pid) == [(DOI, "v2")]
    assert len(items(lib, "metadata_doi_mismatch")) == 1


def test_researcher_values_are_never_overwritten(lib, tmp_path):
    pid = held_paper(lib, tmp_path, "lp_v6", cover(), title="My Own Name for It", title_source="user",
                     year=1999, dois=((DOI, "user"),))
    chain = FakeChain({DOI: openalex_record(DOI, TITLE, 2031)})
    out = metadata_verify.verify(lib, pid, chain=chain)
    assert out.status == "user_locked" and out.changed == ()
    assert tuple(paper_row(lib, pid))[:3] == ("My Own Name for It", "user", 1999)
    assert chain.calls == []


def test_an_untitled_paper_gets_the_provider_title_found_in_its_pdf(lib, tmp_path):
    pid = held_paper(lib, tmp_path, "lp_v7", cover(), dois=((DOI, "pdf_scan"),))
    from seedgraph.library import attention
    attention.raise_item(lib, "metadata_needed", {"paper_id": pid}, paper_id=pid, detail="title unknown")

    out = metadata_verify.verify(lib, pid, chain=FakeChain({DOI: openalex_record(DOI, TITLE, 2031)}))

    assert out.status == "provider_wins"
    assert paper_row(lib, pid)["title"] == TITLE
    assert items(lib, "metadata_needed") == []


def test_no_doi_not_found_and_failures_change_nothing(lib, tmp_path):
    bare = insert_paper(lib, "lp_v8", title=TITLE, title_source="ollama")
    assert metadata_verify.verify(lib, bare, chain=FakeChain()).status == "no_doi"

    pid = insert_paper(lib, "lp_v9", title=TITLE, title_source="ollama", dois=((DOI, "ollama"),))
    assert metadata_verify.verify(lib, pid, chain=FakeChain()).status == "not_found"
    failed = metadata_verify.verify(lib, pid, chain=FakeChain(error=RuntimeError("offline")))
    assert failed.status == "lookup_failed" and "offline" in failed.detail
    assert tuple(paper_row(lib, pid))[:2] == (TITLE, "ollama") and dois_of(lib, pid) == [(DOI, "ollama")]


def test_a_generic_provider_title_is_not_taken(lib, tmp_path):
    pid = held_paper(lib, tmp_path, "lp_v10", cover(), title=TITLE, title_source="ollama", dois=((DOI, "ollama"),))
    out = metadata_verify.verify(lib, pid, chain=FakeChain({DOI: openalex_record(DOI, "Editorial", 2031)}))
    assert out.status == "not_found"
    assert paper_row(lib, pid)["title"] == TITLE


# --- refetch ------------------------------------------------------------------------------


def test_malformed_papers_are_untitled_marked_or_generic_and_never_user_set(lib):
    insert_paper(lib, "lp_m1", title=TITLE, title_source="ollama")
    insert_paper(lib, "lp_m2", title=TITLE + "†", title_source="ollama")
    insert_paper(lib, "lp_m3", title="Contents", title_source="ollama")
    insert_paper(lib, "lp_m4", title=None)
    insert_paper(lib, "lp_m5", title="Preface", title_source="user")
    assert metadata_verify.malformed_papers(lib) == ["lp_m2", "lp_m3", "lp_m4"]


def test_refetch_cleans_marks_locally_and_verifies_dois(lib, tmp_path):
    marked = insert_paper(lib, "lp_r1", title=TITLE + "¹", title_source="ollama")
    pid = held_paper(lib, tmp_path, "lp_r2", cover(), title="Contents", title_source="ollama",
                     dois=((DOI, "ollama"),))
    chain = FakeChain({DOI: openalex_record(DOI, TITLE, 2031)})

    outcomes = metadata_verify.refetch(lib, malformed=True, chain=chain)

    by_id = {o.paper_id: o for o in outcomes}
    assert set(by_id) == {marked, pid}
    assert by_id[marked].status == "no_doi" and by_id[marked].changed == ("title",)
    assert paper_row(lib, marked)["title"] == TITLE
    assert by_id[pid].status == "provider_wins" and paper_row(lib, pid)["title"] == TITLE


def test_refetch_needs_papers_or_malformed(lib):
    with pytest.raises(metadata_verify.LibraryError):
        metadata_verify.refetch(lib, chain=FakeChain())
