"""The fixture PDF writer: tiny, deterministic PDFs with real pages and a text layer."""

from __future__ import annotations

import pypdfium2 as pdfium

from fixtures.pdf import minimal_pdf


def _pages_text(data: bytes) -> list[str]:
    doc = pdfium.PdfDocument(data)
    try:
        out = []
        for i in range(len(doc)):
            page = doc[i]
            textpage = page.get_textpage()
            out.append(textpage.get_text_range())
            textpage.close()
            page.close()
        return out
    finally:
        doc.close()


def test_pages_and_text_layer_round_trip():
    data = minimal_pdf([["Invented Title of a Paper", "doi:10.5555/sg.example"], ["Second page (body)."]])
    texts = _pages_text(data)
    assert len(texts) == 2
    assert "Invented Title of a Paper" in texts[0]
    assert "10.5555/sg.example" in texts[0]
    assert "Second page (body)." in texts[1]


def test_latin1_lookalike_glyphs_survive():
    texts = _pages_text(minimal_pdf([["SpeciÖcation and e¢ cient di¤erence"]]))
    assert "SpeciÖcation and e¢ cient di¤erence" in texts[0]


def test_page_without_text_layer():
    texts = _pages_text(minimal_pdf([None, None]))
    assert texts == ["", ""]


def test_output_is_deterministic():
    pages = [["Same text"], None]
    assert minimal_pdf(pages) == minimal_pdf(pages)
    assert b"CreationDate" not in minimal_pdf(pages)
