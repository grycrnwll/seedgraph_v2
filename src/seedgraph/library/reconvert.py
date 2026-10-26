"""Forced-OCR re-conversion of a paper (design §5.5; ``seedgraph library reconvert --ocr``).

:func:`reconvert_ocr`:
1. converts the paper's primary PDF again through :func:`files.convert_file` with
   ``MarkerConfig(force_ocr=True)`` and ``force=True`` (the stage 0 warm pool when
   ``backend`` is ``None``); the gate runs on the new markdown inside ``convert_file``;
2. if the conversion fails, or the new text still fails the gate, or forced OCR gave the
   very same text, every paper on that PDF keeps its old markdown (``convert_file``
   links the new one, so the links are put back) and the old markdown's
   ``text_garbled`` / ``text_missing`` item stays open (raised if it did not exist yet)
   with "forced OCR did not help";
3. if it passes, the papers on that PDF keep the new markdown, their old gate items are
   resolved (``reconverted_ocr``) and :func:`rematch.rematch` runs for each of them.

Rules chosen where the design is silent:
- A PDF shared by several papers switches for all of them, so every one is re-matched.
- A gate item the researcher already resolved (e.g. ``accept_text``) is never touched.
- A failed conversion leaves attention items as they were; the result carries the error.

Converter lock (UI design §6): while another process (``seedgraph serve``) holds
``<home>/run/converter.lock``, :func:`reconvert_ocr` refuses with
:class:`~seedgraph.web.jobs.ConverterLockHeld` before touching anything. The process that
holds the lock (serve's own jobs) is not refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..cache.convert import MarkerConfig
from ..cache.marker_backend import MarkerBackend
from . import attention, files
from .rematch import RematchResult, rematch
from .store import Library, LibraryError

RESOLUTION = "reconverted_ocr"
DID_NOT_HELP = "forced OCR did not help"
_GATE_KINDS = {"garbled": "text_garbled", "no_text": "text_missing"}

ReconvertStatus = Literal["switched", "ocr_did_not_help", "conversion_failed"]


@dataclass(frozen=True)
class ReconvertResult:
    paper_id: str
    source_file_id: str
    status: ReconvertStatus
    old_markdown_id: str | None
    new_markdown_id: str | None
    text_quality: str | None  # the gate's verdict on the new markdown
    detail: str
    attention_item_id: str | None  # the gate item kept open or resolved
    rematches: tuple[RematchResult, ...] = ()

    def as_dict(self) -> dict:
        return {
            "paper_id": self.paper_id, "source_file_id": self.source_file_id,
            "status": self.status, "old_markdown_id": self.old_markdown_id,
            "new_markdown_id": self.new_markdown_id, "text_quality": self.text_quality,
            "detail": self.detail, "attention_item_id": self.attention_item_id,
            "rematches": [r.as_dict() for r in self.rematches],
        }


def _verdict(lib: Library, markdown_id: str):
    from ..notes.spans import read_markdown  # local: work/notes import the library
    from ..work import packages

    text = read_markdown(lib.conn, markdown_id)
    if text is None:
        return None
    return packages.text_quality(lib, markdown_id, text, packages.reference_ranges(lib.conn, markdown_id))


def _gate_items(lib: Library, paper_id: str, markdown_id: str) -> list[attention.AttentionItem]:
    subject = {"paper_id": paper_id, "markdown_id": markdown_id}
    found = (attention.get_item(lib, attention.item_id_for(kind, subject)) for kind in _GATE_KINDS.values())
    return [item for item in found if item is not None]


def _keep_open(lib: Library, paper_id: str, old_md: str, new_summary: str) -> str | None:
    """Raise or refresh the old markdown's gate item with "forced OCR did not help"."""
    detail = (f"{DID_NOT_HELP} ({new_summary}); the paper keeps its old markdown; resolve this item "
              f"with accept_text to note it anyway, or replace the PDF")
    items = _gate_items(lib, paper_id, old_md)
    subject = {"paper_id": paper_id, "markdown_id": old_md}
    for item in items:
        if item.status == "open":
            return attention.raise_item(lib, item.kind, subject, paper_id=paper_id, detail=detail,
                                        hides_notes=item.hides_notes).item_id
    if items:  # resolved by the researcher: their decision stands
        return items[0].item_id
    old = _verdict(lib, old_md)
    if old is None or old.status not in _GATE_KINDS:
        return None  # the old text passes the gate: nothing to hold back
    return attention.raise_item(lib, _GATE_KINDS[old.status], subject, paper_id=paper_id,
                                detail=f"{old.summary}; {detail}", hides_notes=False).item_id


def refuse_if_converters_busy(lib: Library) -> None:
    """Raise :class:`ConverterLockHeld` while a process other than this one holds the
    converter lock of the library's home."""
    import os

    from ..web.jobs import ConverterLockHeld, converter_lock_holder, converter_lock_path  # no web framework

    home = lib.root.parent
    pid = converter_lock_holder(home)
    if pid is not None and pid != os.getpid():
        raise ConverterLockHeld(pid, converter_lock_path(home))


def reconvert_ocr(lib: Library, paper_id: str, *, backend: MarkerBackend | None = None) -> ReconvertResult:
    """Re-convert the paper's primary PDF with forced OCR (see the module doc).

    Must not be called inside an open library transaction.
    """
    refuse_if_converters_busy(lib)
    row = lib.conn.execute(
        "SELECT source_file_id, markdown_id FROM paper_files WHERE paper_id = ? AND role = 'primary'",
        (paper_id,)).fetchone()
    if row is None:
        raise LibraryError(f"reconvert: paper {paper_id!r} has no PDF attached")
    source_file_id, old_md = row[0], row[1]
    before = {r[0]: r[1] for r in lib.conn.execute(
        "SELECT paper_id, markdown_id FROM paper_files WHERE source_file_id = ? ORDER BY paper_id",
        (source_file_id,))}

    outcome = files.convert_file(lib, source_file_id, backend=backend,
                                 config=MarkerConfig(force_ocr=True), force=True)
    base = dict(paper_id=paper_id, source_file_id=source_file_id, old_markdown_id=old_md)
    if outcome.status != "converted" or outcome.markdown_id is None:
        return ReconvertResult(status="conversion_failed", new_markdown_id=None, text_quality=None,
                               detail=f"forced-OCR conversion failed: {outcome.error}",
                               attention_item_id=None, **base)
    new_md = outcome.markdown_id
    verdict = _verdict(lib, new_md)
    status = verdict.status if verdict is not None else outcome.text_quality

    if new_md == old_md or status != "pass":
        with lib.write() as conn:
            for pid, md in before.items():
                conn.execute("UPDATE paper_files SET markdown_id = ? WHERE paper_id = ? AND source_file_id = ?",
                             (md, pid, source_file_id))
        summary = "the same text as before" if new_md == old_md else (
            verdict.summary if verdict is not None else f"text quality {status}")
        item_id = _keep_open(lib, paper_id, old_md, summary) if old_md is not None else None
        return ReconvertResult(status="ocr_did_not_help", new_markdown_id=new_md, text_quality=status,
                               detail=f"{DID_NOT_HELP}: {summary}; kept markdown {old_md}",
                               attention_item_id=item_id, **base)

    item_id: str | None = None
    rematches: list[RematchResult] = []
    for pid, md in before.items():
        if md is None or md == new_md:
            continue
        for item in _gate_items(lib, pid, md):
            if item.status == "open":
                attention.resolve_item(lib, item.item_id, RESOLUTION)
            if pid == paper_id:
                item_id = item.item_id
        rematches.append(rematch(lib, pid, md, new_md))
    rematches.sort(key=lambda r: (r.paper_id != paper_id, r.paper_id))
    return ReconvertResult(status="switched", new_markdown_id=new_md, text_quality=status,
                           detail=f"switched to markdown {new_md}", attention_item_id=item_id,
                           rematches=tuple(rematches), **base)
