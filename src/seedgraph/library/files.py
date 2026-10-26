"""PDFs and their conversion in the library (v3 design §1.1, §2.1 files, §7.4).

- :func:`attach_pdf` ingests a PDF into ``library/pdfs/<sha>.pdf``, links it to a paper
  (``paper_files``), runs the **standing PDF check** on the PDF's text layer and stores
  it, raises a ``pdf_title_miss`` attention item on a miss (a flag, never a block) and
  queues the file for conversion - all in one library transaction.
- :func:`convert_files` converts named PDFs, the queue, or one paper's files through the
  stage 0 converter pool into ``library.db`` / ``library/markdown/``, links
  ``paper_files.markdown_id``, empties the queue entry and re-checks an ``unknown``
  check on the new markdown. These are the functions ``seedgraph convert`` calls.

Transactions: attach is one ``library.write()`` (a SAVEPOINT inside a caller's own
transaction). A conversion never runs inside a transaction: the cache layer records the
pending run in one short write, converts with nothing open, and stores the result in a
second short write; the paper links follow in a third.

Rules chosen where the design is silent:
- Re-attaching the PDF a paper already has is a no-op that returns the stored state
  (use :func:`recheck_pdf` to re-run the check). Attaching a *different* PDF to a paper
  that already has one is refused: detach first (§7.3 "detach, attach the right one").
- A PDF that already has a successful conversion is linked to that markdown at attach
  and not queued.
- :func:`recheck_pdf` checks the markdown when there is one and falls back to the PDF
  text when that is not a pass; the better result wins (as the migration does, §7.2).
  A later pass does not resolve an open ``pdf_title_miss`` item - the researcher does.
- A PDF given to ``convert`` by path is ingested into the library unattached.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ..cache.convert import CacheError, MarkerConfig, convert_source_file
from ..cache.hashing import sha256_file
from ..cache.ingest import ingest_file
from ..cache.marker_backend import MarkerBackend
from ..vocab import AccessClass, AcquisitionMethod
from .attention import item_id_for, raise_item
from .store import Library, LibraryError, paper_exists
from .titlecheck import PdfCheck, check_pdf_text, first_pages_from_markdown, first_pages_from_pdf

MISS_KIND = "pdf_title_miss"
MISS_DETAIL = "this PDF may not be this paper"

CheckSource = Literal["pdf", "markdown"]


@dataclass(frozen=True)
class AttachResult:
    paper_id: str
    source_file_id: str
    check: PdfCheck
    check_source: CheckSource
    markdown_id: str | None  # set when the PDF was already converted
    queued: bool  # True iff the file is waiting in conversion_queue
    attention_item_id: str | None  # the pdf_title_miss item, on a miss
    already_attached: bool


@dataclass(frozen=True)
class ConvertOutcome:
    source_file_id: str
    status: Literal["converted", "failed"]  # 'converted' includes reuse of an earlier run
    markdown_id: str | None
    error: str | None
    paper_ids: tuple[str, ...]  # papers holding this file
    text_quality: str | None = None  # the gate's verdict on the markdown (design §5.3)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- reads -----------------------------------------------------------------------------


def _title_and_doi(lib: Library, paper_id: str) -> tuple[str | None, str | None]:
    row = lib.conn.execute("SELECT title FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    doi = lib.conn.execute(
        "SELECT id_value FROM identifiers WHERE paper_id = ? AND id_type = 'doi' "
        "ORDER BY id_value LIMIT 1",
        (paper_id,),
    ).fetchone()
    return (row[0] if row else None), (doi[0] if doi else None)


def current_markdown_id(conn, source_file_id: str) -> str | None:
    """The markdown a newly attached paper should link for this file, or ``None``.

    In order: the markdown other papers on this file already link (a failed forced-OCR
    run must not reach a later attach, design §5.5), else the latest successful run
    whose text passed the gate, else the latest successful run.
    """
    linked = conn.execute(
        "SELECT markdown_id FROM paper_files WHERE source_file_id = ? AND markdown_id IS NOT NULL "
        "GROUP BY markdown_id ORDER BY COUNT(*) DESC, MAX(attached_at) DESC LIMIT 1",
        (source_file_id,),
    ).fetchone()
    if linked is not None:
        return linked[0]
    rows = conn.execute(
        "SELECT r.markdown_hash, q.status FROM conversion_runs r "
        "JOIN markdown_documents m ON m.markdown_id = 'md_' || r.markdown_hash "
        "LEFT JOIN text_quality q ON q.markdown_id = m.markdown_id "
        "WHERE r.source_file_id = ? AND r.run_status = 'success' AND r.markdown_hash IS NOT NULL "
        "ORDER BY r.created_at DESC, r.conversion_run_id DESC",
        (source_file_id,),
    ).fetchall()
    if not rows:
        return None
    for md_hash, status in rows:
        if status == "pass":
            return f"md_{md_hash}"
    return f"md_{rows[0][0]}"


def _markdown_pages(lib: Library, markdown_id: str) -> list[str]:
    row = lib.conn.execute(
        "SELECT storage_uri FROM markdown_documents WHERE markdown_id = ?", (markdown_id,)
    ).fetchone()
    if row is None:
        return []
    path = lib.root / row[0]
    if not path.is_file():
        return []
    return first_pages_from_markdown(path.read_text(encoding="utf-8"))


def _pdf_pages_of(lib: Library, source_file_id: str) -> list[str]:
    row = lib.conn.execute(
        "SELECT storage_uri FROM source_files WHERE source_file_id = ?", (source_file_id,)
    ).fetchone()
    if row is None or row[0] is None:
        return []
    return first_pages_from_pdf(lib.root / row[0]) or []


def _papers_of(lib: Library, source_file_id: str) -> tuple[str, ...]:
    return tuple(r[0] for r in lib.conn.execute(
        "SELECT paper_id FROM paper_files WHERE source_file_id = ? ORDER BY paper_id",
        (source_file_id,),
    ))


def queued_files(lib: Library) -> list[str]:
    """Source files waiting for conversion, oldest first."""
    return [r[0] for r in lib.conn.execute(
        "SELECT source_file_id FROM conversion_queue ORDER BY queued_at, source_file_id")]


# --- check storage ---------------------------------------------------------------------


def _check_json(check: PdfCheck, source: CheckSource) -> str:
    return json.dumps(
        {"status": check.status, "score": check.score, "via": check.via,
         "evidence": check.evidence, "source": source},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )


def _store_check(lib: Library, conn, paper_id: str, source_file_id: str, check: PdfCheck,
                 source: CheckSource) -> str | None:
    """Write the check onto the paper_files row; raise the miss item. Returns its id."""
    conn.execute(
        "UPDATE paper_files SET pdf_check_status = ?, pdf_check_json = ? "
        "WHERE paper_id = ? AND source_file_id = ?",
        (check.status, _check_json(check, source), paper_id, source_file_id),
    )
    return _flag_miss(lib, paper_id, source_file_id, check, source)


def _flag_miss(lib: Library, paper_id: str, source_file_id: str, check: PdfCheck,
               source: CheckSource) -> str | None:
    if check.status != "miss":
        return None
    item = raise_item(
        lib,
        MISS_KIND,
        {"paper_id": paper_id, "source_file_id": source_file_id},
        paper_id=paper_id,
        detail=f"{MISS_DETAIL}: its title check missed (score {check.score:.2f}) "
               f"on the {source} text",
        hides_notes=True,  # §7.3: notes hidden from agents until the researcher confirms
    )
    return item.item_id


_RANK = {"pass": 3, "weak": 2, "miss": 1, "unknown": 0}


def _better(a: tuple[PdfCheck, CheckSource], b: tuple[PdfCheck, CheckSource]):
    """The better of two checks (ties keep ``a``): status rank, then score."""
    ka = (_RANK[a[0].status], a[0].score)
    kb = (_RANK[b[0].status], b[0].score)
    return b if kb > ka else a


# --- attach ----------------------------------------------------------------------------


def attach_pdf(
    lib: Library,
    paper_id: str,
    pdf_path: Path | str,
    *,
    access_class: AccessClass = AccessClass.user_supplied_private,
    acquisition_method: AcquisitionMethod = AcquisitionMethod.upload,
) -> AttachResult:
    """Attach a PDF to a paper and run the standing PDF check (§7.4).

    The check reads the PDF's own text layer before any conversion. A ``miss`` raises a
    ``pdf_title_miss`` attention item but never blocks: the file is attached and queued
    either way. One library transaction covers ingest, link, check, item and queue.
    """
    pdf_path = Path(pdf_path)
    if not paper_exists(lib.conn, paper_id):
        raise LibraryError(f"attach_pdf: unknown paper {paper_id!r}")
    if not pdf_path.is_file():
        raise LibraryError(f"attach_pdf: not a file: {pdf_path}")

    source_file_id = f"sf_{sha256_file(pdf_path)}"
    held = lib.conn.execute(
        "SELECT * FROM paper_files WHERE paper_id = ? AND role = 'primary'", (paper_id,)
    ).fetchone()
    if held is not None:
        if held["source_file_id"] != source_file_id:
            raise LibraryError(
                f"paper {paper_id} already has a PDF ({held['source_file_id']}); "
                "detach it first to attach a different one"
            )
        return _existing_result(lib, held)

    title, doi = _title_and_doi(lib, paper_id)
    check = check_pdf_text(title, doi, first_pages_from_pdf(pdf_path) or [])
    source: CheckSource = "pdf"

    with lib.write() as conn:
        src = ingest_file(pdf_path, access_class=access_class,
                          acquisition_method=acquisition_method, library=lib)
        markdown_id = current_markdown_id(conn, src.source_file_id)
        if check.status == "unknown" and markdown_id is not None:
            check = check_pdf_text(title, doi, _markdown_pages(lib, markdown_id))
            source = "markdown"
        conn.execute(
            "INSERT INTO paper_files (paper_id, source_file_id, markdown_id, role, "
            "pdf_check_status, pdf_check_json, attached_at) VALUES (?, ?, ?, 'primary', ?, ?, ?)",
            (paper_id, src.source_file_id, markdown_id, check.status,
             _check_json(check, source), _now()),
        )
        queued = markdown_id is None
        if queued:
            queue_conversion(lib, src.source_file_id)  # a no-op if another paper queued it
        item_id = _flag_miss(lib, paper_id, src.source_file_id, check, source)
    return AttachResult(paper_id=paper_id, source_file_id=src.source_file_id, check=check,
                        check_source=source, markdown_id=markdown_id, queued=queued,
                        attention_item_id=item_id, already_attached=False)


def _existing_result(lib: Library, row) -> AttachResult:
    stored = json.loads(row["pdf_check_json"]) if row["pdf_check_json"] else {}
    check = PdfCheck(status=row["pdf_check_status"], score=stored.get("score", 0.0),
                     via=stored.get("via"), evidence=stored.get("evidence", ""))
    is_queued = lib.conn.execute(
        "SELECT 1 FROM conversion_queue WHERE source_file_id = ?", (row["source_file_id"],)
    ).fetchone() is not None
    item_id = None
    if check.status == "miss":
        item_id = item_id_for(MISS_KIND, {"paper_id": row["paper_id"],
                                          "source_file_id": row["source_file_id"]})
    return AttachResult(paper_id=row["paper_id"], source_file_id=row["source_file_id"],
                        check=check, check_source=stored.get("source", "pdf"),
                        markdown_id=row["markdown_id"], queued=is_queued,
                        attention_item_id=item_id, already_attached=True)


def queue_conversion(lib: Library, source_file_id: str, reason: str = "not_converted") -> bool:
    """Put a source file in ``conversion_queue``; ``False`` if it was already there."""
    with lib.write() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO conversion_queue (source_file_id, reason, queued_at) "
            "VALUES (?, ?, ?)",
            (source_file_id, reason, _now()),
        )
    return cur.rowcount == 1


# --- re-check --------------------------------------------------------------------------


def recheck_pdf(lib: Library, paper_id: str) -> list[PdfCheck]:
    """Re-run the PDF check for each of the paper's files and store the result.

    Uses the markdown when the file is converted, falling back to the PDF text when the
    markdown check is not a pass; the better result wins.
    """
    if not paper_exists(lib.conn, paper_id):
        raise LibraryError(f"recheck_pdf: unknown paper {paper_id!r}")
    title, doi = _title_and_doi(lib, paper_id)
    rows = lib.conn.execute(
        "SELECT source_file_id, markdown_id FROM paper_files WHERE paper_id = ? "
        "ORDER BY source_file_id",
        (paper_id,),
    ).fetchall()
    results: list[PdfCheck] = []
    for sf, md in rows:
        best: tuple[PdfCheck, CheckSource] | None = None
        if md is not None:
            best = (check_pdf_text(title, doi, _markdown_pages(lib, md)), "markdown")
        if best is None or best[0].status != "pass":
            from_pdf = (check_pdf_text(title, doi, _pdf_pages_of(lib, sf)), "pdf")
            best = from_pdf if best is None else _better(best, from_pdf)
        with lib.write() as conn:
            _store_check(lib, conn, paper_id, sf, best[0], best[1])
        results.append(best[0])
    return results


# --- convert ---------------------------------------------------------------------------


def convert_file(
    lib: Library,
    source_file_id: str,
    *,
    backend: MarkerBackend | None = None,
    config: MarkerConfig = MarkerConfig(),
    force: bool = False,
) -> ConvertOutcome:
    """Convert one library source file (or reuse its earlier conversion) and link it.

    A failure is recorded (a ``failed`` run) and returned, not raised; the file stays
    queued. Must not be called inside an open library transaction.
    """
    try:
        md = convert_source_file(source_file_id, config=config, backend=backend, force=force,
                                 library=lib)
    except CacheError as exc:
        return ConvertOutcome(source_file_id=source_file_id, status="failed", markdown_id=None,
                              error=str(exc), paper_ids=_papers_of(lib, source_file_id))

    with lib.write() as conn:
        conn.execute("UPDATE paper_files SET markdown_id = ? WHERE source_file_id = ?",
                     (md.markdown_id, source_file_id))
        conn.execute("DELETE FROM conversion_queue WHERE source_file_id = ?", (source_file_id,))
        unknown = conn.execute(
            "SELECT paper_id FROM paper_files WHERE source_file_id = ? "
            "AND pdf_check_status = 'unknown' ORDER BY paper_id",
            (source_file_id,),
        ).fetchall()
        if unknown:
            pages = _markdown_pages(lib, md.markdown_id)
            for (pid,) in unknown:
                title, doi = _title_and_doi(lib, pid)
                _store_check(lib, conn, pid, source_file_id,
                             check_pdf_text(title, doi, pages), "markdown")
    index_sections(lib, md.markdown_id)
    return ConvertOutcome(source_file_id=source_file_id, status="converted",
                          markdown_id=md.markdown_id, error=None,
                          paper_ids=_papers_of(lib, source_file_id),
                          text_quality=_assess_text(lib, md.markdown_id))


def index_sections(lib: Library, markdown_id: str) -> int:
    """Parse and store ``document_sections`` for a converted markdown (v2's
    ``parse_sections``, deterministic ``sec_`` ids), so its reference ranges exist for
    packaging and the gate. A markdown that already has sections (migrated, or an
    earlier conversion of the same bytes) is left as it is - checked again inside the
    write, so two concurrent conversions of identical markdown cannot collide. Returns
    rows written."""
    from ..notes.spans import read_markdown  # local: notes imports the library
    from ..sections.parser import SECTION_PARSER_VERSION, parse_sections

    conn = lib.conn
    if conn.execute("SELECT 1 FROM document_sections WHERE markdown_id = ? LIMIT 1",
                    (markdown_id,)).fetchone() is not None:
        return 0
    row = conn.execute(
        "SELECT m.markdown_hash, m.source_file_id, s.file_hash FROM markdown_documents m "
        "JOIN source_files s ON s.source_file_id = m.source_file_id WHERE m.markdown_id = ?",
        (markdown_id,)).fetchone()
    text = read_markdown(conn, markdown_id) if row is not None else None
    if text is None:
        return 0
    sections = parse_sections(text, markdown_id=markdown_id, markdown_hash=row[0], source_file_id=row[1],
                              source_file_hash=row[2], work_id="")
    now = _now()
    with lib.write() as wconn:
        # Re-checked inside the write: another conversion of the same bytes (two PDFs with
        # identical markdown converting at once) may have indexed it since the check above.
        if wconn.execute("SELECT 1 FROM document_sections WHERE markdown_id = ? LIMIT 1",
                         (markdown_id,)).fetchone() is not None:
            return 0
        for sec in sections:
            wconn.execute(
                "INSERT OR IGNORE INTO document_sections (section_id, markdown_id, markdown_hash, source_file_id, "
                "source_file_hash, parent_section_id, level, ordinal, heading_text, heading_path, section_kind, "
                "start_char, end_char, page_start, page_end, section_parser_version, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (sec.section_id, sec.markdown_id, sec.markdown_hash, sec.source_file_id, sec.source_file_hash,
                 sec.parent_section_id, sec.level, sec.ordinal, sec.heading_text, sec.heading_path,
                 sec.section_kind, sec.start_char, sec.end_char, sec.page_start, sec.page_end,
                 SECTION_PARSER_VERSION, now))
    return len(sections)


def _assess_text(lib: Library, markdown_id: str) -> str | None:
    """The text-quality gate after a conversion (design §5.3): assess and cache the
    verdict; raising a hold-back item is left to batch creation."""
    from ..notes.spans import read_markdown  # local: work/notes import the library
    from ..work import packages

    text = read_markdown(lib.conn, markdown_id)
    if text is None:
        return None
    ranges = packages.reference_ranges(lib.conn, markdown_id)
    return packages.text_quality(lib, markdown_id, text, ranges, force=True).status


def convert_files(
    lib: Library,
    *,
    pdfs: Sequence[Path | str] = (),
    queued: bool = False,
    paper_id: str | None = None,
    backend: MarkerBackend | None = None,
    config: MarkerConfig = MarkerConfig(),
    force: bool = False,
) -> list[ConvertOutcome]:
    """``seedgraph convert [PDF…] [--queued] [--paper ID]``: convert the chosen files.

    Files are taken in this order, each once: the named PDFs (ingested into the library,
    unattached, if new), then the queue (oldest first), then the paper's files. One
    outcome per file; a failure does not stop the batch.
    """
    if not pdfs and not queued and paper_id is None:
        raise LibraryError("convert: nothing to convert (give PDFs, queued=True or paper_id)")
    chosen: list[str] = []
    for path in pdfs:
        src = ingest_file(Path(path), access_class=AccessClass.user_supplied_private,
                          acquisition_method=AcquisitionMethod.upload, library=lib)
        chosen.append(src.source_file_id)
    if queued:
        chosen.extend(queued_files(lib))
    if paper_id is not None:
        chosen.extend(_paper_files_or_refuse(lib, paper_id))
    return [convert_file(lib, sf, backend=backend, config=config, force=force)
            for sf in _unique(chosen)]


def _paper_files_or_refuse(lib: Library, paper_id: str) -> list[str]:
    if not paper_exists(lib.conn, paper_id):
        raise LibraryError(f"convert: unknown paper {paper_id!r}")
    files = [r[0] for r in lib.conn.execute(
        "SELECT source_file_id FROM paper_files WHERE paper_id = ? ORDER BY source_file_id",
        (paper_id,),
    )]
    if not files:
        raise LibraryError(f"convert: paper {paper_id} has no PDF attached")
    return files


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))
