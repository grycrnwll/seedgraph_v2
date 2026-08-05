"""Post-conversion bibliographic metadata backfill — OFFLINE, LLM MOCKED.

The live LLM (local Ollama) is used only at runtime; every test here either
monkeypatches the extractor (:func:`extract_biblio_metadata`) or exercises the
pure regex / JSON / identity paths. Markdown is pushed through the real cache
(ingest + FakeMarkerBackend convert) and bridged work->markdown exactly as
phase_4 does, so the runner + CLI round-trip with no network and no key.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlmodel import Session, select
from typer.testing import CliRunner

from seedgraph.acquisition.bridge import write_bridge
from seedgraph.cache.convert import convert_source_file
from seedgraph.cache.ingest import ingest_file
from seedgraph.cache.marker_backend import FakeMarkerBackend
from seedgraph.cli import app
from seedgraph.config.models import default_profiles, default_routes
from seedgraph.db.project_models import Identifier, ReviewQueueItem, Work
from seedgraph.extraction import metadata
from seedgraph.extraction.metadata import (
    BiblioMeta,
    deterministic_identifiers,
    parse_biblio_json,
)
from seedgraph.extraction.metadata import _meta_from_dict  # tolerant coercion
from seedgraph.project import identity, service
from seedgraph.vocab import AccessClass, AcquisitionMethod

cli = CliRunner()

MD_PLAIN = (
    "# On the Economics of Minimum Wages\n\n"
    "Jane Q. Researcher and John Doe\n\n"
    "We study the labor-market effects of the minimum wage.\n"
)


def _project_with_markdown(
    slug: str,
    *,
    title: str,
    markdown: str = MD_PLAIN,
    doi: str | None = None,
    access_class: AccessClass = AccessClass.open_access,
):
    """Create a project + one work (optionally with a doi) + bridged converted MD."""
    h = service.create_project(slug)
    ids = {"doi": doi} if doi else None
    w = service.add_work(h, ids=ids, title=title)
    p = Path(os.environ["SEEDGRAPH_HOME"]) / f"{slug}.pdf"
    p.write_bytes(b"%PDF-1.4 " + slug.encode() + b" body content words")
    method = (
        AcquisitionMethod.open_access_fetch
        if access_class == AccessClass.open_access
        else AcquisitionMethod.upload
    )
    src = ingest_file(p, access_class=access_class, acquisition_method=method, root=None)
    md = convert_source_file(src.source_file_id, backend=FakeMarkerBackend(markdown=markdown), root=None)
    with Session(h.engine) as s:
        write_bridge(
            s,
            work_id=w.work_id,
            source_file_id=src.source_file_id,
            file_hash=src.file_hash,
            markdown_id=md.markdown_id,
            markdown_hash=md.markdown_hash,
            acquisition_method=method.value,
        )
        s.commit()
    return h, w.work_id


# ==========================================================================
# 0. routing defaults
# ==========================================================================

def test_metadata_extraction_route_defaults():
    routes = default_routes()
    r = routes["metadata_extraction"]
    assert r.preferred_profile == "local_ollama_default"
    assert r.fallback_profile == "no_llm"
    assert r.deterministic_fallback is True
    profs = default_profiles()
    assert "metadata_extraction" in profs["local_ollama_default"].allowed_tasks
    assert "metadata_extraction" in profs["anthropic_api_default"].allowed_tasks


# ==========================================================================
# 1. tolerant JSON parse
# ==========================================================================

def test_parse_biblio_json_fenced():
    raw = '```json\n{"title": "A Title: A Subtitle", "authors": ["X Y"], "year": 2020}\n```'
    data = parse_biblio_json(raw)
    assert data is not None
    meta = _meta_from_dict(data)
    assert meta.title == "A Title: A Subtitle"
    assert meta.authors == ["X Y"]
    assert meta.year == 2020


def test_parse_biblio_json_plain_and_trailing_prose():
    data = parse_biblio_json('Here you go: {"title": "T"} — done')
    assert data == {"title": "T"}


def test_parse_biblio_json_missing_keys_are_tolerant():
    meta = _meta_from_dict(parse_biblio_json('{"title": "Only Title"}'))
    assert meta.title == "Only Title"
    assert meta.authors == []
    assert meta.year is None
    assert meta.doi is None and meta.arxiv_id is None and meta.venue is None


def test_parse_biblio_json_garbage_returns_none():
    assert parse_biblio_json("no json object here at all") is None
    assert parse_biblio_json("") is None
    assert parse_biblio_json("[1, 2, 3]") is None  # top-level array, not an object


# ==========================================================================
# 2. deterministic identifiers (regex over the whole doc)
# ==========================================================================

def test_deterministic_identifiers_jstor_doi_and_arxiv():
    md = (
        "THE AMERICAN ECONOMIC REVIEW\n"
        "Stable URL: https://www.jstor.org/stable/27086719\n"
        "Your use of the JSTOR archive indicates your acceptance of the Terms.\n\n"
        "# Some Paper Title\n\n"
        "DOI: 10.2307/27086719.\n"
        "A preprint is available as arXiv:2103.00020.\n"
    )
    ids = deterministic_identifiers(md)
    assert ids["doi"] == "10.2307/27086719"
    assert ids["arxiv"] == "2103.00020"


def test_deterministic_identifiers_absent():
    ids = deterministic_identifiers("A document with no identifiers whatsoever.")
    assert ids == {"doi": None, "arxiv": None}


def test_deterministic_identifiers_ignores_reference_list_dois():
    # must-fix #1: a working paper's OWN cover DOI is in the head; a CITED work's DOI
    # lives in a later References section (past HEAD_CHARS) and must NEVER be extracted.
    head = (
        "# A Working Paper With Its Own Cover DOI\n\n"
        "Jane Q. Researcher\n\n"
        "DOI: 10.2307/27086719\n\n"
        "Abstract. We study things at some length here.\n\n"
    )
    filler = "Body paragraph text that pads the document out. " * 100
    refs = (
        "\n\n## References\n\n"
        "Smith, J. (2019). A cited paper. https://doi.org/10.9999/citedpaper\n"
        "Jones, K. (2020). Another cited work. arXiv:1234.56789\n"
    )
    md = head + filler + refs
    assert len(head) < 2600 < len(head + filler)  # the References start beyond the head
    ids = deterministic_identifiers(md)
    assert ids["doi"] == "10.2307/27086719"     # the paper's OWN cover DOI
    assert ids["doi"] != "10.9999/citedpaper"   # the reference-list DOI is ignored
    assert ids["arxiv"] is None                 # the reference-list arXiv id is ignored


def test_deterministic_identifiers_arxiv_requires_explicit_label():
    # must-fix #2: a bare NNNN.NNNNN token (coefficient / table value / cited id) is
    # NOT an arXiv id; only an explicit ``arXiv:<id>`` or ``arxiv.org/abs/<id>`` is.
    assert deterministic_identifiers("The point estimate was 2021.12345 in column 3.")["arxiv"] is None
    assert deterministic_identifiers("Preprint arXiv:2401.01234 available online.")["arxiv"] == "2401.01234"
    assert deterministic_identifiers("See https://arxiv.org/abs/2401.01234 for the PDF.")["arxiv"] == "2401.01234"


# ==========================================================================
# 3. update_work_bibliography
# ==========================================================================

def test_filename_derived_title_is_replaced():
    h = service.create_project("ident_a")
    w = service.add_work(h, title="2023 santanna")  # no ids -> filename-derived
    meta = BiblioMeta(title="The Real Title: A Subtitle", authors=["A. Santanna"], year=2023)
    with Session(h.engine) as s:
        upd = identity.update_work_bibliography(s, w.work_id, meta)
        s.commit()
    assert upd.title_changed is True
    with Session(h.engine) as s:
        work = s.get(Work, w.work_id)
        assert work.canonical_title == "The Real Title: A Subtitle"
        assert work.authors == ["A. Santanna"]
        assert work.year == 2023


def test_authoritative_resolve_title_is_preserved():
    h = service.create_project("ident_b")
    w = service.add_work(h, title="Authoritative Title")  # no scalar id yet
    # Simulate a prior authoritative resolve: an identifier with resolver provenance.
    with Session(h.engine) as s:
        s.add(
            Identifier(
                work_id=w.work_id,
                id_type="doi",
                id_value="10.9999/resolved",
                resolution_source="openalex",
            )
        )
        s.commit()
    meta = BiblioMeta(title="LLM Guessed A Different Title", authors=["Z"], year=1999)
    with Session(h.engine) as s:
        upd = identity.update_work_bibliography(s, w.work_id, meta)
        s.commit()
    assert upd.title_changed is False
    with Session(h.engine) as s:
        work = s.get(Work, w.work_id)
        assert work.canonical_title == "Authoritative Title"  # preserved
        assert work.authors == ["Z"]  # empty scalar fields are still filled


def test_doi_attaches_and_scalar_mirror_is_set():
    h = service.create_project("ident_c")
    w = service.add_work(h, title="filename junk")  # no ids
    meta = BiblioMeta(
        title="Real",
        doi="https://doi.org/10.1234/abc",  # surface form normalized
        arxiv_id="arXiv:2101.00001",
    )
    with Session(h.engine) as s:
        upd = identity.update_work_bibliography(s, w.work_id, meta)
        s.commit()
    assert ("doi", "10.1234/abc") in upd.ids_attached
    assert ("arxiv", "2101.00001") in upd.ids_attached
    with Session(h.engine) as s:
        work = s.get(Work, w.work_id)
        assert work.doi == "10.1234/abc"  # scalar mirror
        assert work.arxiv_id == "2101.00001"
        pairs = {
            (i.id_type, i.id_value)
            for i in s.exec(select(Identifier).where(Identifier.work_id == w.work_id)).all()
        }
        assert ("doi", "10.1234/abc") in pairs
        assert ("arxiv", "2101.00001") in pairs


def test_doi_collision_on_different_work_routes_to_review_no_merge():
    h = service.create_project("ident_d")
    wa = service.add_work(h, title="Paper A", ids={"doi": "10.5555/shared"})
    wb = service.add_work(h, title="filename b")  # no ids

    meta = BiblioMeta(title="B Real Title", doi="10.5555/shared")
    with Session(h.engine) as s:
        upd = identity.update_work_bibliography(s, wb.work_id, meta)
        s.commit()

    assert ("doi", "10.5555/shared") in upd.ids_collided
    assert upd.ids_attached == []
    with Session(h.engine) as s:
        # B did NOT silently claim the colliding id (no merge, no scalar).
        workb = s.get(Work, wb.work_id)
        assert workb.doi is None
        # The identifier still resolves to exactly ONE work (A) — no duplicate row.
        rows = s.exec(
            select(Identifier).where(
                Identifier.id_type == "doi", Identifier.id_value == "10.5555/shared"
            )
        ).all()
        assert len(rows) == 1 and rows[0].work_id == wa.work_id
        # A cross-id-collision review item was enqueued.
        items = s.exec(
            select(ReviewQueueItem).where(ReviewQueueItem.item_type == "duplicate_candidate")
        ).all()
        assert any(i.payload and "cross_id_collision" in i.payload for i in items)
        # No NEW work was created.
        assert len(s.exec(select(Work)).all()) == 2


def test_force_conflicting_doi_routes_to_review_not_second_row():
    # HARDENING: under --force the work already OWNS a doi and the extracted doi
    # differs -> route to review; never append a second, disagreeing identifier row.
    h = service.create_project("ident_force")
    w = service.add_work(h, title="Original", ids={"doi": "10.1/original"})
    meta = BiblioMeta(title="Forced New Title", doi="10.2/different")
    with Session(h.engine) as s:
        upd = identity.update_work_bibliography(
            s, w.work_id, meta, allow_title_overwrite=True
        )
        s.commit()
    assert ("doi", "10.2/different") in upd.ids_collided
    assert upd.ids_attached == []
    with Session(h.engine) as s:
        work = s.get(Work, w.work_id)
        assert work.doi == "10.1/original"  # the owned id is preserved, not overwritten
        rows = s.exec(
            select(Identifier).where(
                Identifier.id_type == "doi", Identifier.work_id == w.work_id
            )
        ).all()
        # Exactly ONE doi row for this work — the disagreeing one was NOT appended.
        assert {r.id_value for r in rows} == {"10.1/original"}
        items = s.exec(
            select(ReviewQueueItem).where(ReviewQueueItem.item_type == "duplicate_candidate")
        ).all()
        assert any(i.payload and "id_disagreement" in i.payload for i in items)


# ==========================================================================
# 4. runner + merge (extractor MOCKED)
# ==========================================================================

def test_backfill_replaces_title_and_merges_regex_doi(monkeypatch):
    md = (
        "# The Converted Real Title\n\n"
        "Ada Lovelace and Alan Turing\n\n"
        "Stable URL: https://www.jstor.org/stable/27086719\n"
        "DOI: 10.2307/27086719\n"
    )
    h, wid = _project_with_markdown("id_run", title="2023 santanna", markdown=md)

    # LLM returns title/authors but no doi -> the regex JSTOR doi must fill in.
    def _fake_extract(markdown_head, **kwargs):
        return BiblioMeta(title="The Converted Real Title", authors=["Ada Lovelace", "Alan Turing"], year=2011)

    monkeypatch.setattr(metadata, "extract_biblio_metadata", _fake_extract)

    res = metadata.backfill_work_metadata(h, wid, cache_root=None)
    assert res.status == "ok"
    assert res.title_changed is True
    assert ("doi", "10.2307/27086719") in res.ids_attached
    with Session(h.engine) as s:
        work = s.get(Work, wid)
        assert work.canonical_title == "The Converted Real Title"
        assert work.authors == ["Ada Lovelace", "Alan Turing"]
        assert work.doi == "10.2307/27086719"  # filled from the deterministic scan


def test_backfill_idempotent_skips_authoritative_unless_forced(monkeypatch):
    # A work carrying a strong id is authoritative -> skipped without --force.
    h, wid = _project_with_markdown("id_idem", title="Good Title", doi="10.1/authoritative")

    calls = {"n": 0}

    def _fake_extract(markdown_head, **kwargs):
        calls["n"] += 1
        return BiblioMeta(title="Should Not Apply")

    monkeypatch.setattr(metadata, "extract_biblio_metadata", _fake_extract)

    res = metadata.backfill_work_metadata(h, wid, cache_root=None)
    assert res.status == "skipped_idempotent"
    assert calls["n"] == 0  # extraction not even attempted

    forced = metadata.backfill_work_metadata(h, wid, cache_root=None, force=True)
    assert forced.status == "ok"
    assert calls["n"] == 1


def test_backfill_best_effort_swallows_a_raising_extractor(monkeypatch):
    h, wid = _project_with_markdown("id_raise", title="2020 junk")

    def _boom(markdown_head, **kwargs):
        raise RuntimeError("extractor exploded")

    monkeypatch.setattr(metadata, "extract_biblio_metadata", _boom)

    # best_effort_backfill must NOT propagate — the conversion never fails.
    assert metadata.best_effort_backfill(h, wid, cache_root=None) is None
    with Session(h.engine) as s:
        work = s.get(Work, wid)
        assert work.canonical_title == "2020 junk"  # untouched


# ==========================================================================
# 5. CLI smoke — `corpus identify`
# ==========================================================================

def test_cli_corpus_identify_reports(monkeypatch):
    h, wid = _project_with_markdown("id_cli", title="2023 santanna")

    def _fake_extract(markdown_head, **kwargs):
        return BiblioMeta(title="Extracted Real Title", authors=["Foo Bar"], year=2021)

    monkeypatch.setattr(metadata, "extract_biblio_metadata", _fake_extract)

    result = cli.invoke(app, ["corpus", "identify", "id_cli"])
    assert result.exit_code == 0, result.output
    assert "Extracted Real Title" in result.output
    assert wid in result.output
    with Session(h.engine) as s:
        assert s.get(Work, wid).canonical_title == "Extracted Real Title"
