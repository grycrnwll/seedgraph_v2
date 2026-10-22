"""PDF page count for the page-marker check (pypdfium2, the engine Marker uses)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def page_count(pdf_path: Path | str) -> Optional[int]:
    """Number of pages in ``pdf_path``; ``None`` when missing/unreadable or pypdfium2 is absent."""
    path = Path(pdf_path)
    if not path.is_file():
        return None
    try:
        import pypdfium2 as pdfium
    except ImportError:  # pragma: no cover - pypdfium2 is an app dependency
        return None
    try:
        doc = pdfium.PdfDocument(str(path))
    except Exception:
        return None
    try:
        return len(doc)
    finally:
        doc.close()
