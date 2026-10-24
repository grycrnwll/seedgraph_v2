"""Section indexing after a conversion, and which markdown a later attach links
(stage 2 P8, gaps found by P5). All PDFs are synthetic and all text is invented."""

from __future__ import annotations

import pytest

from seedgraph.cache.convert import MarkerConfig
from seedgraph.library import files
from seedgraph.library.reconvert import reconvert_ocr
from seedgraph.library.store import open_library_at
from seedgraph.work import packages

from .test_files import add_paper, paper_file, text_pdf
from .test_reconvert import TITLE, OcrBackend, _filler, garbled_md


@pytest.fixture
def lib(tmp_path):
    library = open_library_at(tmp_path / "home" / "library")
    try:
        yield library
    finally:
        library.close()


BODY = " ".join(f"Sentence {k} describes an invented oven schedule." for k in range(60))


def references_garbled_md() -> str:
    """Clean body text; the garbled run sits only under the References heading."""
    return (f"# {TITLE}\n\n## Introduction\n\n{BODY}\n\n## Results\n\n{BODY}\n\n"
            f"## References\n\n{_filler(True)}\n")


def sections_of(lib, md_id: str):
    return lib.conn.execute(
        "SELECT section_kind, heading_text, start_char, end_char, section_id FROM document_sections "
        "WHERE markdown_id = ? ORDER BY ordinal", (md_id,)).fetchall()


def test_a_new_conversion_indexes_its_sections_with_reference_ranges(lib, tmp_path):
    pid = add_paper(lib, title=TITLE)
    res = files.attach_pdf(lib, pid, text_pdf(tmp_path / "a.pdf", [TITLE]))
    outcome = files.convert_file(lib, res.source_file_id, backend=OcrBackend(references_garbled_md(), ""))

    rows = sections_of(lib, outcome.markdown_id)
    assert [r["heading_text"] for r in rows] == [TITLE, "Introduction", "Results", "References"]
    assert rows[-1]["section_kind"] == "references"
    assert packages.reference_ranges(lib.conn, outcome.markdown_id) == (
        (rows[-1]["start_char"], rows[-1]["end_char"]),)
    # The gate ran after indexing, so the garbled references were stripped before it judged.
    assert outcome.text_quality == "pass"


def test_section_ids_are_deterministic_and_reconversion_does_not_duplicate(lib, tmp_path):
    pid = add_paper(lib, title=TITLE)
    res = files.attach_pdf(lib, pid, text_pdf(tmp_path / "a.pdf", [TITLE]))
    backend = OcrBackend(references_garbled_md(), "")
    first = files.convert_file(lib, res.source_file_id, backend=backend)
    ids = [r["section_id"] for r in sections_of(lib, first.markdown_id)]
    again = files.convert_file(lib, res.source_file_id, backend=backend, force=True)

    assert again.markdown_id == first.markdown_id
    assert [r["section_id"] for r in sections_of(lib, again.markdown_id)] == ids
    assert all(i.startswith("sec_") for i in ids)


def test_a_later_attach_keeps_the_markdown_other_papers_use_not_a_failed_ocr_run(lib, tmp_path):
    backend = OcrBackend(garbled_md(), garbled_md(with_lost=False))
    first = add_paper(lib, "lp_00000000000000000001", title=TITLE)
    pdf = text_pdf(tmp_path / "a.pdf", [TITLE])
    res = files.attach_pdf(lib, first, pdf)
    old_md = files.convert_file(lib, res.source_file_id, backend=backend).markdown_id
    result = reconvert_ocr(lib, first, backend=backend)
    assert result.status == "ocr_did_not_help" and result.new_markdown_id != old_md

    second = add_paper(lib, "lp_00000000000000000002", title=TITLE)
    files.attach_pdf(lib, second, pdf)

    assert paper_file(lib, second)["markdown_id"] == old_md


def test_with_no_paper_linked_attach_prefers_the_last_gate_passing_run(lib, tmp_path):
    backend = OcrBackend(references_garbled_md(), garbled_md(with_lost=False))
    pdf = text_pdf(tmp_path / "a.pdf", [TITLE])
    [good] = files.convert_files(lib, pdfs=[pdf], backend=backend)
    assert good.text_quality == "pass"
    # A later forced-OCR run that fails the gate, with no paper linked to the file.
    bad = files.convert_file(lib, good.source_file_id, backend=backend, config=MarkerConfig(force_ocr=True),
                             force=True)
    assert bad.text_quality == "garbled" and bad.markdown_id != good.markdown_id

    pid = add_paper(lib, title=TITLE)
    files.attach_pdf(lib, pid, pdf)

    assert paper_file(lib, pid)["markdown_id"] == good.markdown_id
