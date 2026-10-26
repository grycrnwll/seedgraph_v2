"""The ``metadata`` task and the Ollama worker apply the §7 quality rules: titles are
cleaned, generic titles are not saved, and after the submit commits a post-commit hook
verifies the DOI with the providers. Fake chain and fake Ollama; invented text."""

from __future__ import annotations

import json

import pytest

from llm.fake_ollama import FakeOllama, chat_reply
from seedgraph.library import attention, metadata_verify
from seedgraph.work import ollama_worker, routine
from seedgraph.work.claims import WorkerRef
from work.routine_support import filler, make_paper, open_lib

from .metadata_support import FakeChain, dois_of, items, openalex_record, paper_row

OLLAMA = WorkerRef("ollama", "o1", client="ollama", model="invented-model")
TITLE = "A Made-Up Study of Gadget Markets"
DOI = "10.9999/invented.2019.7"
CITED = "10.9999/cited.1"
COVER = (f"# {TITLE}\n\nAda Inventor and Bo Example\n\nJournal of Invented Results, 2019. doi:{DOI}\n\n"
         + filler("cover", 400) + f"\n\n## References\n\nOther, C. (2000). doi:{CITED}\n")
META = {"title": TITLE, "authors": ["Ada Inventor"], "year": 2019, "venue": "Journal of Invented Results",
        "doi": None, "arxiv": None}


@pytest.fixture
def lib(tmp_path):
    library = open_lib(tmp_path)
    yield library
    library.close()


@pytest.fixture
def chain():
    """A chain for post-commit verification in this test (none is set by default)."""
    fake = FakeChain({DOI: openalex_record(DOI, TITLE, 2019, authors=("Ada Inventor", "Bo Example")),
                      CITED: openalex_record(CITED, "An Older Invented Paper on Harbours", 2000)})
    token = metadata_verify.set_chain(fake)
    yield fake
    metadata_verify.reset_chain(token)


def untitled(lib, n=1):
    paper = make_paper(lib, n, text=COVER, sections=[])
    with lib.write() as conn:
        conn.execute("UPDATE papers SET title = NULL, year = NULL WHERE paper_id = ?", (paper.paper_id,))
    attention.raise_item(lib, "metadata_needed", {"paper_id": paper.paper_id}, paper_id=paper.paper_id,
                         detail="title unknown")
    return paper


def submit(lib, paper, note):
    view = routine.create_batch(lib, "metadata", paper_ids=[paper.paper_id])
    msg = routine.claim_next(lib, view.batch_id, OLLAMA).message
    return routine.submit(lib, msg["packet_id"], msg["claim_id"], note=note)


def test_footnote_marks_are_stripped_before_saving(lib):
    paper = untitled(lib)
    result = submit(lib, paper, {**META, "title": TITLE + "†¹"})
    assert result.status == "accepted"
    assert paper_row(lib, paper.paper_id)["title"] == TITLE


def test_a_generic_title_is_not_saved_and_raises_an_item(lib):
    paper = untitled(lib)
    result = submit(lib, paper, {**META, "title": "Table of Contents"})

    assert result.status == "accepted" and result.paper_status == "done"
    assert paper_row(lib, paper.paper_id)["title"] is None
    (item,) = items(lib, "metadata_generic_title")
    assert item["paper_id"] == paper.paper_id
    assert "Table of Contents" in item["subject_json"] and "ollama" in item["detail"]
    assert len(items(lib, "metadata_needed")) == 1  # still needs its title
    # the other fields are still filled
    assert paper_row(lib, paper.paper_id)["year"] == 2019


def test_a_title_equal_to_the_venue_is_generic(lib):
    paper = untitled(lib)
    submit(lib, paper, {**META, "title": "Journal of Invented Results"})
    assert paper_row(lib, paper.paper_id)["title"] is None
    assert len(items(lib, "metadata_generic_title")) == 1


def test_without_a_chain_the_title_stays_unverified(lib):
    paper = untitled(lib)
    submit(lib, paper, {**META, "title": "A made-up study of gadget markets"})
    assert tuple(paper_row(lib, paper.paper_id))[:2] == ("A made-up study of gadget markets", "ollama")


def test_after_the_commit_the_doi_is_verified_and_provider_values_win(lib, chain):
    paper = untitled(lib)
    seen_in_transaction: list[bool] = []
    original = chain.by_doi

    async def by_doi(doi):
        seen_in_transaction.append(lib.conn.in_transaction)
        return await original(doi)

    chain.by_doi = by_doi
    result = submit(lib, paper, {**META, "title": "A made-up study of gadget markets"})

    assert result.status == "accepted" and result.paper_status == "done"
    assert seen_in_transaction == [False]
    row = paper_row(lib, paper.paper_id)
    assert (row["title"], row["title_source"]) == (TITLE, "openalex")
    assert json.loads(row["authors_json"]) == ["Ada Inventor", "Bo Example"]


def test_a_reference_list_doi_from_the_worker_is_removed_with_an_item(lib, chain):
    paper = untitled(lib)
    submit(lib, paper, {**META, "doi": CITED})

    assert dois_of(lib, paper.paper_id) == []
    assert tuple(paper_row(lib, paper.paper_id))[:2] == (TITLE, "ollama")
    (item,) = items(lib, "metadata_doi_mismatch")
    assert CITED in item["subject_json"]


def test_a_failing_hook_never_fails_an_accepted_submission(lib, chain, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("invented hook failure")

    monkeypatch.setattr(metadata_verify, "verify", boom)
    paper = untitled(lib)
    result = submit(lib, paper, META)
    assert result.status == "accepted" and result.paper_status == "done"
    assert paper_row(lib, paper.paper_id)["title"] == TITLE


def test_the_ollama_worker_cleans_rejects_and_verifies_with_its_chain(lib):
    server = FakeOllama().start()
    try:
        first = untitled(lib, 1)
        second = make_paper(lib, 2, text=COVER.replace(DOI, "10.9999/none"), sections=[])
        with lib.write() as conn:
            conn.execute("UPDATE papers SET title = NULL WHERE paper_id = ?", (second.paper_id,))
        batch = routine.create_batch(lib, "metadata", paper_ids=[first.paper_id, second.paper_id]).batch_id
        server.script(chat_reply(json.dumps({**META, "title": "a made-up study of gadget markets*"})),
                      chat_reply(json.dumps({**META, "title": "Abstract", "doi": None})))
        chain = FakeChain({DOI: openalex_record(DOI, TITLE, 2019)})

        report = ollama_worker.run_ollama_worker(lib, batch, model="invented-model", base_url=server.url,
                                                 chain=chain)
    finally:
        server.close()

    assert report.saved == 2
    assert tuple(paper_row(lib, first.paper_id))[:2] == (TITLE, "openalex")
    assert paper_row(lib, second.paper_id)["title"] is None
    assert len(items(lib, "metadata_generic_title")) == 1
    assert metadata_verify.current_chain() is None  # the worker's chain does not leak
