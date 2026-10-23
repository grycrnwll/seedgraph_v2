"""Attention items: deterministic ids, idempotent upsert, resolve, stable listing (§2.1, §7.3)."""

from __future__ import annotations

import pytest

from seedgraph.library import attention, store

from .test_store import NOW, add_paper

LATER = "2026-02-01T00:00:00+00:00"


@pytest.fixture
def lib(tmp_path):
    library = store.open_library(tmp_path)
    with library.write() as conn:
        add_paper(conn, "lp_a")
        add_paper(conn, "lp_b")
    yield library
    library.close()


def test_item_id_is_a_deterministic_hash_of_kind_and_subject():
    a = attention.item_id_for("pdf_title_miss", {"paper_id": "lp_a", "source_file_id": "sf_1"})
    # key order in the subject does not matter
    b = attention.item_id_for("pdf_title_miss", {"source_file_id": "sf_1", "paper_id": "lp_a"})
    assert a == b
    assert a.startswith("att_")
    assert a != attention.item_id_for("malformed_title", {"paper_id": "lp_a", "source_file_id": "sf_1"})
    assert a != attention.item_id_for("pdf_title_miss", {"paper_id": "lp_b", "source_file_id": "sf_1"})


def test_raise_item_stores_every_field(lib):
    item = attention.raise_item(
        lib, "pdf_title_miss", {"paper_id": "lp_a"}, paper_id="lp_a",
        detail="the first pages name a different widget study", hides_notes=True,
        created_at=NOW, origin="migration",
    )
    assert item.item_id == attention.item_id_for("pdf_title_miss", {"paper_id": "lp_a"})
    assert item.kind == "pdf_title_miss"
    assert item.paper_id == "lp_a"
    assert item.subject == {"paper_id": "lp_a"}
    assert item.hides_notes is True
    assert item.status == "open"
    assert item.created_at == NOW
    assert item.resolution is None and item.resolved_at is None
    assert attention.get_item(lib, item.item_id) == item


def test_raise_is_idempotent_and_keeps_created_at(lib):
    first = attention.raise_item(lib, "metadata_needed", {"source_file_id": "sf_9"}, created_at=NOW)
    again = attention.raise_item(lib, "metadata_needed", {"source_file_id": "sf_9"},
                                 detail="updated detail", created_at=LATER)
    assert again.item_id == first.item_id
    assert again.created_at == NOW
    assert again.detail == "updated detail"
    assert len(attention.list_items(lib)) == 1


def test_resolve_closes_the_item_and_reraise_does_not_reopen_it(lib):
    item = attention.raise_item(lib, "pdf_title_miss", {"paper_id": "lp_a"}, paper_id="lp_a",
                                hides_notes=True, created_at=NOW)
    resolved = attention.resolve_item(lib, item.item_id, "confirmed by researcher", resolved_at=LATER)
    assert resolved.status == "resolved"
    assert resolved.resolution == "confirmed by researcher"
    assert resolved.resolved_at == LATER
    # a later raise of the same subject must not undo the researcher's decision
    again = attention.raise_item(lib, "pdf_title_miss", {"paper_id": "lp_a"}, paper_id="lp_a",
                                 hides_notes=True, created_at=LATER)
    assert again.status == "resolved"
    assert attention.hidden_paper_ids(lib.conn) == frozenset()


def test_resolve_unknown_item_is_refused(lib):
    with pytest.raises(attention.AttentionError):
        attention.resolve_item(lib, "att_missing", "nothing")


def test_hidden_papers_are_those_with_an_open_hiding_item(lib):
    attention.raise_item(lib, "pdf_title_miss", {"paper_id": "lp_a"}, paper_id="lp_a",
                         hides_notes=True, created_at=NOW)
    attention.raise_item(lib, "malformed_title", {"paper_id": "lp_b"}, paper_id="lp_b",
                         hides_notes=False, created_at=NOW)
    assert attention.hidden_paper_ids(lib.conn) == frozenset({"lp_a"})


def test_listing_is_stably_ordered_and_filterable(lib):
    attention.raise_item(lib, "shared_pdf", {"source_file_id": "sf_2"}, created_at=NOW)
    attention.raise_item(lib, "pdf_title_miss", {"paper_id": "lp_b"}, paper_id="lp_b", created_at=NOW)
    attention.raise_item(lib, "pdf_title_miss", {"paper_id": "lp_a"}, paper_id="lp_a", created_at=NOW)
    done = attention.raise_item(lib, "identity_check", {"papers": ["lp_a", "lp_b"]}, created_at=NOW)
    attention.resolve_item(lib, done.item_id, "different papers", resolved_at=LATER)

    listed = attention.list_items(lib)
    assert [(i.kind, i.paper_id) for i in listed] == [
        ("identity_check", None),
        ("pdf_title_miss", "lp_a"),
        ("pdf_title_miss", "lp_b"),
        ("shared_pdf", None),
    ]
    assert [i.kind for i in attention.list_items(lib, status="open")] == [
        "pdf_title_miss", "pdf_title_miss", "shared_pdf"]
    assert [i.paper_id for i in attention.list_items(lib, kind="pdf_title_miss")] == ["lp_a", "lp_b"]
    assert [i.kind for i in attention.list_items(lib, paper_id="lp_b")] == ["pdf_title_miss"]


def test_raise_composes_inside_an_outer_write(lib):
    with lib.write():
        attention.raise_item(lib, "metadata_needed", {"source_file_id": "sf_7"}, created_at=NOW)
    assert len(attention.list_items(lib)) == 1


def test_invalid_status_filter_is_refused(lib):
    with pytest.raises(attention.AttentionError):
        attention.list_items(lib, status="closed")
