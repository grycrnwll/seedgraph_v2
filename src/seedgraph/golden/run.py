"""Run the golden-set comparison over manifest entries."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional

from . import pdf, v2
from .checks import CheckResult, PdfResult, compare
from .sources import Source


def pdf_page_count(entry: dict) -> Optional[int]:
    """Page count of the entry's cached PDF in the v2 home (None if unavailable)."""
    return pdf.page_count(v2.pdf_path(entry["quote_source"]["v2_home"], entry["pdf_sha256"]))


def _concurrency(source: Source) -> int:
    """How many PDFs a source can render at once (a converter pool's ``size``)."""
    return max(1, int(getattr(source, "concurrency", 1) or 1))


def run_check(
    entries: Iterable[dict],
    old: Source,
    new: Source,
    *,
    page_counter: Callable[[dict], Optional[int]] = pdf_page_count,
) -> list[PdfResult]:
    """Compare OLD vs NEW for every entry; page count from the PDF, else the converter's.

    Entries run concurrently up to the sources' concurrency (one per pool converter,
    so every GPU is busy); results keep manifest order."""

    def one(entry: dict) -> PdfResult:
        try:
            old_r, new_r = old.rendition(entry), new.rendition(entry)
        except Exception as exc:  # one PDF failing (e.g. converter crash) must not stop the run
            return PdfResult(entry["id"], [CheckResult(
                "source", None, None, "fail", f"{type(exc).__name__}: {exc}")],
                {"category": entry.get("category"), "old_source": old.label, "new_source": new.label})
        pages = page_counter(entry)
        source = "pdf"
        if pages is None:
            pages = new_r.page_count or old_r.page_count
            source = "converter" if pages is not None else None
        result = compare(entry["id"], old_r, new_r, page_count=pages, quotes=v2.note_quotes(entry["quote_source"]))
        result.meta.update({"category": entry.get("category"), "page_count_source": source,
                            "old_source": old.label, "new_source": new.label})
        if new_r.page_count is not None and pages is not None and new_r.page_count != pages:
            result.meta["converter_page_count_mismatch"] = new_r.page_count
        return result

    entries = list(entries)
    workers = max(_concurrency(old), _concurrency(new))
    if workers == 1:
        return [one(e) for e in entries]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, entries))
