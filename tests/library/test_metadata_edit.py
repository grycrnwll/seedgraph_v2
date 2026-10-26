"""The researcher's metadata edit (docs/V3_UI_DESIGN.md §2.4, §12 q10). Invented values only."""

from __future__ import annotations

import pytest

from seedgraph.library import attention, metadata_verify
from seedgraph.library.metadata_edit import set_user_metadata
from seedgraph.library.store import LibraryError, open_library_at

PID = "lp_00000000000000000901"


@pytest.fixture
def lib(tmp_path):
    library = open_library_at(tmp_path / "library")
    with library.write() as conn:
        conn.execute("INSERT INTO papers (paper_id, title, title_source, created_at) "
                     "VALUES (?, 'Review', 'ollama', '2026-01-01T00:00:00+00:00')", (PID,))
        conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value, source) "
                     "VALUES (?, 'doi', '10.9999/invented.1', 'ollama')", (PID,))
    attention.raise_item(library, "metadata_needed", {"paper_id": PID}, paper_id=PID)
    yield library
    library.close()


def row(lib):
    return lib.conn.execute("SELECT title, title_source, year, venue, authors_json FROM papers WHERE paper_id = ?",
                            (PID,)).fetchone()


def test_values_are_saved_with_the_user_source_and_the_metadata_item_resolves(lib):
    changed = set_user_metadata(lib, PID, title=" An  Invented Title ", year="2029", venue=" Invented Review ",
                                authors=["Ada Inventor", " ", "Bo Madeup"])
    assert changed == ("title", "year", "venue", "authors")
    assert tuple(row(lib)) == ("An Invented Title", "user", 2029, "Invented Review", '["Ada Inventor", "Bo Madeup"]')
    assert attention.list_items(lib, status="open", paper_id=PID) == []


def test_a_provider_never_overwrites_the_researchers_values(lib):
    set_user_metadata(lib, PID, title="An Invented Title")

    class Chain:
        def by_doi(self, doi):
            return {"title": "A Provider Title", "year": 2001, "openalex_id": "W1"}

    outcome = metadata_verify.verify(lib, PID, chain=Chain(), pages=[])
    assert outcome.status == "user_locked"
    assert row(lib)["title"] == "An Invented Title"


@pytest.mark.parametrize("kw", [{"title": "  "}, {"title": "T", "year": "soon"}, {"title": "T", "year": 99}])
def test_bad_values_are_refused(lib, kw):
    with pytest.raises(LibraryError):
        set_user_metadata(lib, PID, **kw)
    assert row(lib)["title_source"] == "ollama"
