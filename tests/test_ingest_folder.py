"""Bulk ``corpus ingest-folder`` auto-match acceptance tests (offline, no LLM).

Every test runs fully offline: identity extraction + matching are deterministic
(regex + title_hash) and conversion uses the in-memory :class:`FakeMarkerBackend`
(or a cfg-sensitive double for the OCR-fallback path). No network, no GPU, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlmodel import Session, select

from seedgraph.acquisition.bridge import resolve_work_markdown
from seedgraph.acquisition.service import ingest_folder, ingest_one_pdf  # noqa: F401
from seedgraph.cache.marker_backend import FakeMarkerBackend, MarkerResult
from seedgraph.db.project_models import ProjectDocument, Work, WorkSourceFile
from seedgraph.project import review, service


def _write_pdf(path: Path, marker: bytes = b"body") -> None:
    path.write_bytes(b"%PDF-1.4\n" + marker + b"\nbody words here now.\n")


def _folder(tmp_path: Path) -> Path:
    d = tmp_path / "drop"
    d.mkdir()
    return d


# --------------------------------------------------------------------------
# match by DOI (strong id first)
# --------------------------------------------------------------------------
def test_match_by_doi(tmp_path):
    h = service.create_project("idoi")
    service.add_work(
        h, ids={"doi": "10.1234/abcde"}, title="DOI Work Alpha",
        inclusion_status="metadata_only",
    )
    folder = _folder(tmp_path)
    _write_pdf(folder / "arbitrary_name.pdf", b"doi-doc")
    # H1 deliberately DIFFERENT from the work title so the match must go via DOI.
    md = "# Completely Different Heading\n\nhttps://doi.org/10.1234/abcde\n\n## Body\n\nreal words here now.\n"

    report = ingest_folder(h, folder, backend=FakeMarkerBackend(markdown=md))

    assert report.matched_doi == 1
    assert report.matched_title == 0
    with Session(h.engine) as s:
        rows = s.exec(select(WorkSourceFile)).all()
        assert len(rows) == 1
        assert rows[0].markdown_id is not None and rows[0].markdown_hash is not None
        doc = s.get(ProjectDocument, rows[0].work_id)
        assert doc.access_status == "user_supplied_private"
        assert doc.inclusion_status == "included"  # promote=True default


# --------------------------------------------------------------------------
# match by title_hash (no strong id)
# --------------------------------------------------------------------------
def test_match_by_title(tmp_path):
    h = service.create_project("ititle")
    service.add_work(h, title="Some Distinct Title", inclusion_status="metadata_only")
    folder = _folder(tmp_path)
    _write_pdf(folder / "paper.pdf", b"title-doc")
    md = "# Some Distinct Title\n\n## Intro\n\nreal words here now for body.\n"

    report = ingest_folder(h, folder, backend=FakeMarkerBackend(markdown=md))

    assert report.matched_title == 1
    with Session(h.engine) as s:
        rows = s.exec(select(WorkSourceFile)).all()
        assert len(rows) == 1
        w = s.get(Work, rows[0].work_id)
        assert w.canonical_title == "Some Distinct Title"


# --------------------------------------------------------------------------
# no match -> review_queue (unmatched_upload)
# --------------------------------------------------------------------------
def test_unmatched_to_review(tmp_path):
    h = service.create_project("iunmatch")
    folder = _folder(tmp_path)
    _write_pdf(folder / "x.pdf", b"nomatch")
    md = "# Nobody Matches This\n\n## Body\n\nreal words present here.\n"

    report = ingest_folder(h, folder, backend=FakeMarkerBackend(markdown=md))

    assert report.unmatched_review == 1
    items = review.list_open(h)
    assert len(items) == 1
    assert items[0].item_type == "unmatched_upload"
    payload = json.loads(items[0].payload)
    assert payload["reason"] == "no_match"
    assert payload["source_file_id"] and payload["markdown_id"]
    assert payload["extracted_title"] == "Nobody Matches This"
    with Session(h.engine) as s:
        assert s.exec(select(Work)).all() == []  # no new work minted


# --------------------------------------------------------------------------
# no match + --promote-unmatched -> new included work + bridge
# --------------------------------------------------------------------------
def test_unmatched_promote(tmp_path):
    h = service.create_project("ipromote")
    folder = _folder(tmp_path)
    _write_pdf(folder / "new_paper_title.pdf", b"promote")
    md = "# Fresh Promoted Work\n\n## Body\n\nreal words present here.\n"

    report = ingest_folder(
        h, folder, backend=FakeMarkerBackend(markdown=md), promote_unmatched=True
    )

    assert report.promoted == 1
    assert review.list_open(h) == []  # no review item
    with Session(h.engine) as s:
        works = s.exec(select(Work)).all()
        assert len(works) == 1
        assert works[0].canonical_title == "Fresh Promoted Work"  # from the H1
        rows = s.exec(select(WorkSourceFile)).all()
        assert len(rows) == 1 and rows[0].markdown_id is not None
        doc = s.get(ProjectDocument, works[0].work_id)
        assert doc.inclusion_status == "included"
        assert doc.access_status == "user_supplied_private"


# --------------------------------------------------------------------------
# ambiguous title match -> review with both candidates (NEVER auto-merge)
# --------------------------------------------------------------------------
def test_ambiguous_match(tmp_path):
    h = service.create_project("iambig")
    service.add_work(h, title="Shared Ambiguous Title", inclusion_status="metadata_only")
    service.add_work(h, title="Shared Ambiguous Title", inclusion_status="metadata_only")
    folder = _folder(tmp_path)
    _write_pdf(folder / "amb.pdf", b"ambiguous")
    md = "# Shared Ambiguous Title\n\n## Body\n\nreal words present here.\n"

    report = ingest_folder(h, folder, backend=FakeMarkerBackend(markdown=md))

    assert report.ambiguous_review == 1
    # The 2nd same-title add_work already enqueued a title_collision duplicate_candidate;
    # pick out the unmatched_upload item.
    up = [i for i in review.list_open(h) if i.item_type == "unmatched_upload"]
    assert len(up) == 1
    payload = json.loads(up[0].payload)
    assert payload["reason"] == "ambiguous_match"
    assert len(payload["candidates"]) == 2


# --------------------------------------------------------------------------
# OCR fallback: garbage on default cfg -> ConversionError -> force_ocr retry
# --------------------------------------------------------------------------
class _OcrSensitiveBackend:
    """Fails the validate garbage-gate on the default cfg; good markdown under OCR."""

    version = "ocr-fake-0.0.0"

    def __init__(self, good_markdown: str) -> None:
        self._good = good_markdown
        self.call_count = 0
        self.force_ocr_calls = 0

    def __call__(self, pdf_path, cfg) -> MarkerResult:
        self.call_count += 1
        if cfg.force_ocr:
            self.force_ocr_calls += 1
            return MarkerResult(markdown=self._good, block_json=None, warnings=[])
        # No word-like tokens AND no math/LaTeX markup -> content ratio 0 < 0.20 ->
        # fatal garbage gate. (Avoid '$': a '$...$' pair reads as inline math to the
        # math-aware validator and would count as content.)
        return MarkerResult(
            markdown="1234567890 !@#%^&*() 0987654321 )(*&^%#@!",
            block_json=None,
            warnings=[],
        )


def test_ocr_fallback(tmp_path):
    h = service.create_project("iocr")
    folder = _folder(tmp_path)
    _write_pdf(folder / "scanned.pdf", b"scan")
    good = "# Recovered By OCR\n\n## Body\n\nreal words recovered here now.\n"
    backend = _OcrSensitiveBackend(good)

    report = ingest_folder(h, folder, backend=backend, promote_unmatched=True)

    assert report.ocr_recovered == 1
    assert report.promoted == 1
    assert backend.force_ocr_calls == 1  # OCR retry was exercised
    assert backend.call_count == 2  # default (fail) + force_ocr (success)
    with Session(h.engine) as s:
        rows = s.exec(select(WorkSourceFile)).all()
        assert len(rows) == 1 and rows[0].markdown_id is not None


# --------------------------------------------------------------------------
# idempotent re-drop of the same bytes -> already_bridged, no re-convert
# --------------------------------------------------------------------------
def test_idempotent_rerun(tmp_path):
    h = service.create_project("idem")
    service.add_work(
        h, ids={"doi": "10.5555/idem"}, title="Idem Work",
        inclusion_status="metadata_only",
    )
    folder = _folder(tmp_path)
    _write_pdf(folder / "p.pdf", b"idem")
    md = "# Whatever Heading\n\nhttps://doi.org/10.5555/idem\n\n## Body\n\nreal words here now.\n"
    backend = FakeMarkerBackend(markdown=md)

    r1 = ingest_folder(h, folder, backend=backend)
    assert r1.matched_doi == 1
    calls = backend.call_count

    r2 = ingest_folder(h, folder, backend=backend)
    assert r2.already_bridged == 1
    assert r2.matched_doi == 0
    assert backend.call_count == calls  # Marker NOT re-invoked (pre-convert short-circuit)

    with Session(h.engine) as s:
        assert len(s.exec(select(WorkSourceFile)).all()) == 1  # still exactly one bridge
    assert [i for i in review.list_open(h) if i.item_type == "unmatched_upload"] == []


# --------------------------------------------------------------------------
# non-PDF file in the folder is skipped
# --------------------------------------------------------------------------
def test_skip_non_pdf(tmp_path):
    h = service.create_project("iskip")
    folder = _folder(tmp_path)
    (folder / "notes.txt").write_bytes(b"this is not a pdf")

    report = ingest_folder(h, folder, backend=FakeMarkerBackend())

    assert report.skipped_non_pdf == 1
    assert report.matched_doi == report.matched_title == 0
    with Session(h.engine) as s:
        assert s.exec(select(Work)).all() == []


# --------------------------------------------------------------------------
# missing / empty folder -> empty report (no error)
# --------------------------------------------------------------------------
def test_missing_folder_is_empty_report(tmp_path):
    h = service.create_project("iempty")
    report = ingest_folder(h, tmp_path / "does_not_exist", backend=FakeMarkerBackend())
    assert report.per_pdf == []
    assert report.skipped_non_pdf == 0


# --------------------------------------------------------------------------
# approve an unmatched_upload review item -> new work + bridge (decision 2)
# --------------------------------------------------------------------------
def test_unmatched_review_approve_creates_work(tmp_path):
    h = service.create_project("iapprove")
    folder = _folder(tmp_path)
    _write_pdf(folder / "adopt_me.pdf", b"adopt")
    md = "# Adopted Paper\n\n## Body\n\nreal words present here now.\n"

    report = ingest_folder(h, folder, backend=FakeMarkerBackend(markdown=md))
    assert report.unmatched_review == 1

    item = review.list_open(h)[0]
    review.resolve(h, item.item_id, "approve")

    with Session(h.engine) as s:
        works = s.exec(select(Work)).all()
        assert len(works) == 1
        wid = works[0].work_id
        assert works[0].canonical_title == "Adopted Paper"
        doc = s.get(ProjectDocument, wid)
        assert doc.inclusion_status == "included"
        assert doc.access_status == "user_supplied_private"
        assert resolve_work_markdown(s, work_id=wid) is not None  # bridged w/ markdown
    assert review.list_open(h) == []  # the item is resolved
