"""Two conversions that produce identical markdown index its sections at the same time
(found by U4: the job showed "failed" although the data was consistent). Indexing must
be idempotent. All PDFs are synthetic and all text is invented."""

from __future__ import annotations

import threading

import pytest

from seedgraph.library import files
from seedgraph.library.store import open_library_at
from seedgraph.sections import parser as section_parser

from .test_files import add_paper, text_pdf
from .test_files_sections import references_garbled_md, sections_of
from .test_reconvert import TITLE, OcrBackend


@pytest.fixture
def lib(tmp_path):
    library = open_library_at(tmp_path / "home" / "library")
    try:
        yield library
    finally:
        library.close()


def test_concurrent_section_indexing_of_one_markdown_is_idempotent(lib, tmp_path, monkeypatch):
    pid = add_paper(lib, title=TITLE)
    res = files.attach_pdf(lib, pid, text_pdf(tmp_path / "a.pdf", [TITLE]))
    md_id = files.convert_file(lib, res.source_file_id, backend=OcrBackend(references_garbled_md(), "")).markdown_id
    expected = [tuple(r) for r in sections_of(lib, md_id)]
    with lib.write() as conn:
        conn.execute("DELETE FROM document_sections WHERE markdown_id = ?", (md_id,))

    # Both threads pass the "already indexed?" pre-check before either writes.
    barrier = threading.Barrier(2, timeout=10)
    real_parse = section_parser.parse_sections

    def parse_then_wait(*args, **kwargs):
        out = real_parse(*args, **kwargs)
        barrier.wait()
        return out

    monkeypatch.setattr(section_parser, "parse_sections", parse_then_wait)
    errors: list[BaseException] = []

    def run() -> None:
        other = open_library_at(lib.root)
        try:
            files.index_sections(other, md_id)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)
        finally:
            other.close()

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert errors == []
    assert [tuple(r) for r in sections_of(lib, md_id)] == expected
