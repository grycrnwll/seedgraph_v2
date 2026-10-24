"""The ``metadata`` task (design §2.2, §6): the cover page plus deterministic id hits;
fill empty fields only; Ollama may take it (it is the only task Ollama may take)."""

from __future__ import annotations

from pathlib import Path

import pytest

from seedgraph.library import attention
from seedgraph.work import routine
from seedgraph.work.claims import WorkerRef

from .routine_support import filler, make_paper, open_lib

OLLAMA = WorkerRef("ollama", "o1", client="ollama", model="invented-model")
COVER = ("# A Made-Up Study of Gadget Markets\n\nAda Inventor and Bo Example\n\n"
         "Journal of Invented Results, 2019. doi:10.9999/INVENTED.2019.7\n\n"
         + filler("cover", 400) + "\n\n## References\n\nOther, C. (2000). doi:10.9999/cited.1\n")


@pytest.fixture
def lib(tmp_path):
    library = open_lib(tmp_path)
    yield library
    library.close()


def _untitled(lib, n=1, text=COVER):
    paper = make_paper(lib, n, text=text, sections=[])
    with lib.write() as conn:
        conn.execute("UPDATE papers SET title = NULL, year = NULL WHERE paper_id = ?", (paper.paper_id,))
    item = attention.raise_item(lib, "metadata_needed", {"paper_id": paper.paper_id}, paper_id=paper.paper_id,
                                detail="title unknown")
    return paper, item


def test_metadata_package_is_the_cover_with_id_hits(lib):
    paper, _ = _untitled(lib)
    view = routine.create_batch(lib, "metadata", query="held needing metadata")
    assert [i.subject_id for i in view.items] == [paper.paper_id]
    reply = routine.claim_next(lib, view.batch_id, OLLAMA)
    assert reply.status == "claimed"
    text = Path(reply.message["text_path"]).read_text(encoding="utf-8")
    assert text.startswith("# A Made-Up Study")
    assert "10.9999/invented.2019.7" in text
    assert "10.9999/cited.1" not in text and len(text) < 3200


def test_metadata_fills_empty_fields_only_and_resolves_the_item(lib):
    paper, item = _untitled(lib)
    with lib.write() as conn:
        conn.execute("UPDATE papers SET venue = 'Set by the researcher' WHERE paper_id = ?", (paper.paper_id,))
    view = routine.create_batch(lib, "metadata", paper_ids=[paper.paper_id])
    msg = routine.claim_next(lib, view.batch_id, OLLAMA).message
    result = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={
        "title": "A Made-Up Study of Gadget Markets", "authors": ["Ada Inventor", "Bo Example"],
        "year": "2019", "venue": "Journal of Invented Results", "doi": None, "arxiv": None})
    assert result.status == "accepted" and result.paper_status == "done"
    row = lib.conn.execute("SELECT title, title_source, authors_json, year, venue FROM papers WHERE paper_id = ?",
                           (paper.paper_id,)).fetchone()
    assert tuple(row) == ("A Made-Up Study of Gadget Markets", "ollama", '["Ada Inventor", "Bo Example"]', 2019,
                          "Set by the researcher")
    ids = lib.conn.execute("SELECT id_type, id_value, source FROM identifiers WHERE paper_id = ?",
                           (paper.paper_id,)).fetchall()
    assert [tuple(r) for r in ids] == [("doi", "10.9999/invented.2019.7", "ollama")]
    assert attention.get_item(lib, item.item_id).status == "resolved"


def test_metadata_type_errors_are_invalid(lib):
    paper, _ = _untitled(lib)
    view = routine.create_batch(lib, "metadata", paper_ids=[paper.paper_id])
    msg = routine.claim_next(lib, view.batch_id, OLLAMA).message
    bad = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"title": 7, "authors": "Ada"})
    assert bad.status == "invalid" and len(bad.errors) == 2


def test_a_paper_with_metadata_is_preserved(lib):
    paper = make_paper(lib, 1)
    (item,) = routine.create_batch(lib, "metadata", paper_ids=[paper.paper_id]).items
    assert item.state == "preserved"


def test_ollama_is_offered_only_metadata(lib):
    paper = make_paper(lib, 1)
    notes = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id])
    reply = routine.claim_next(lib, notes.batch_id, OLLAMA)
    assert reply.status == "not_allowed"
    assert "notes" in reply.next_action
    assert routine.batch_view(lib, notes.batch_id).state.packages == {"open": 1}
