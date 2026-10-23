"""A minimal hand-written PDF writer for the synthetic v2 fixtures.

Each page is a list of text lines (drawn in Helvetica, WinAnsiEncoding, so
Latin-1 lookalike glyphs such as ``Ö``/``¢``/``¤`` survive) or ``None`` for a page
with no text layer (a "scan": it draws one line, no text). The output has no
``/CreationDate`` and no ``/ID``, so identical input gives identical bytes.
"""

from __future__ import annotations

from collections.abc import Sequence


def _pdf_string(text: str) -> bytes:
    out = bytearray(b"(")
    for byte in text.encode("cp1252"):
        if byte in (0x28, 0x29, 0x5C):  # ( ) \
            out += b"\\" + bytes([byte])
        elif 0x20 <= byte < 0x7F:
            out.append(byte)
        else:
            out += b"\\%03o" % byte
    out += b")"
    return bytes(out)


def _content(lines: Sequence[str] | None) -> bytes:
    if lines is None:
        return b"0.5 w 72 700 m 540 700 l S\n"
    parts = [b"BT /F1 11 Tf 14 TL 72 740 Td"]
    for i, line in enumerate(lines):
        parts.append((b"" if i == 0 else b"T* ") + _pdf_string(line) + b" Tj")
    parts.append(b"ET")
    return b"\n".join(parts) + b"\n"


def minimal_pdf(pages: Sequence[Sequence[str] | None]) -> bytes:
    """Return the bytes of a PDF with one page per entry of ``pages``."""
    if not pages:
        raise ValueError("a PDF needs at least one page")
    n = len(pages)
    # Object numbers: 1 catalog, 2 pages, 3 font, then (page, content) pairs.
    page_nums = [4 + 2 * i for i in range(n)]
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % p for p in page_nums)
        + b"] /Count %d >>" % n,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    for i, lines in enumerate(pages):
        stream = _content(lines)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % (page_nums[i] + 1)
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream")
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % num + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref_at)
    return bytes(out)
