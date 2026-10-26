"""Adding papers to the library (docs/V3_UI_DESIGN.md §6; researcher rules of 2026-10-26).

    add_pdfs(lib, paths, *, source="upload", chain=None)  -> list[IngestResult]
    add_folder(lib, folder, *, chain=None)                 -> list[IngestResult]
    attach_to_paper(lib, paper_id, pdf_path, ...)          -> IngestResult
    add_dois(lib, dois, *, chain, fetch=None)              -> list[DoiResult]
    needs_pdf(lib, *, graph=None, frontier=False)          -> list[NeedsPdf]

**Dropped PDFs** (:func:`add_pdfs`). Seedgraph never fetches a paywalled PDF, but a
researcher may supply any PDF they have access to; its text is labelled
``user_supplied_private`` (provenance only). Per file:

1. only a file that starts with ``%PDF`` is taken (else ``skipped``);
2. the file is hashed; a file already attached to a paper reports that paper
   (``duplicate_file``) and stops. A file that is in the library but attached to no paper
   (``seedgraph convert PDF``) is not a duplicate;
3. the cover region of the text layer (``metadata_quality.COVER_CHARS``; never the
   whole first pages, which in a short paper already hold the reference list) is scanned
   for the paper's own DOI / arXiv id (``tasks_metadata.deterministic_identifiers``). An
   id held by a **cited-only** paper attaches the PDF to it, so it becomes held
   (``matched_by='doi'`` / ``'arxiv'``);
4. otherwise a tolerant **title + year** match on the same cover region: papers with a
   non-generic title and a year whose year is printed there and whose title passes the
   standing title check (:func:`titlecheck.check_pdf_text`). Exactly one cited-only
   candidate attaches (``matched_by='title_year'``);
5. otherwise a new paper with no title is created (keyed on its DOI, else the PDF hash),
   with a ``metadata_needed`` attention item; an unmatched DOI from the cover is stored
   with ``source='pdf_scan'``. When step 3 or 4 found candidates that cannot take the
   PDF (an id on a paper that holds another PDF, several title matches, a held paper's
   title) an ``identity_check`` item lists them: papers are never merged here;
6. :func:`files.attach_pdf` runs the standing check and queues the conversion (one
   transaction). With a ``chain``, a new paper's DOI is verified at once
   (:func:`metadata_verify.verify`).

**Explicit attach** (:func:`attach_to_paper`, the paper page's "Add PDF"): no matching
at all; the PDF goes to the named paper (a title miss is flagged, never blocked).

**DOI paste** (:func:`add_dois`): a known DOI reports its paper and changes nothing.
Otherwise ``chain.by_doi`` (no transaction open) makes a **cited-only** paper with
``title_source`` = the provider; a tolerant title + year duplicate raises an
``identity_check`` item instead of merging. Then ``fetch(record)`` (default
:func:`acquisition.fetch.fetch_oa`, open-access copies only, never paywalled ones) runs;
a downloaded PDF is attached with ``access_class='open_access'`` /
``acquisition_method='open_access_fetch'`` and the source is reported. A cited-only
paper that was already known gets the same open-access attempt.

**Needs a PDF** (:func:`needs_pdf`): cited-only papers with a ``https://doi.org/<doi>``
link so the researcher can download through their institution and drop the file in;
optionally only one graph's frontier, or every graph's frontier.

Rules chosen where the design is silent:
- §6 step 3 says "strong id only"; the researcher's title + year rule (2026-10-26) adds
  step 4, which only ever attaches to a single cited-only candidate.
- A title + year candidate whose stored DOI differs from a DOI on the PDF is skipped.
- Duplicate titles at DOI paste: same year and :func:`titlecheck.title_similarity` at
  or above the matcher's pass threshold (0.80).
- A DOI that is not ``10.<digits>/<suffix>`` after normalisation is ``invalid``.
- Fetched bytes go to ``<home>/tmp/oa/`` and are deleted once ingested.
- A generic provider title at DOI paste is not saved: ``metadata_generic_title`` and
  ``metadata_needed`` items are raised instead.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..cache.hashing import sha256_file
from ..project.identity import normalize_id
from ..vocab import AccessClass, AcquisitionMethod
from . import attention, files, graphs
from .identity import derive_key, mint_paper_id, normalize_ids
from .metadata_quality import clean_title, cover_region, generic_title_reason
from .metadata_verify import SAME_TITLE, VerifyOutcome, lookup_doi, provider_of, record_values, run_sync, verify
from .store import Library, LibraryError, paper_exists
from .titlecheck import check_pdf_text, first_pages_from_pdf, title_similarity

METADATA_NEEDED = "metadata_needed"
IDENTITY_CHECK = "identity_check"
GENERIC_TITLE = "metadata_generic_title"
DOI_URL = "https://doi.org/{}"
_DOI_SHAPE = re.compile(r"^10\.\d{4,9}/\S+$")
_YEAR_RE = re.compile(r"(?<!\d)(1[5-9]\d\d|2[0-9]\d\d)(?!\d)")

IngestStatus = Literal["attached", "new_paper", "duplicate_file", "skipped"]
Source = Literal["upload", "folder"]
DoiStatus = Literal["created", "existing", "not_found", "invalid", "lookup_failed"]
OaStatus = Literal["held", "none", "failed", "already_held"]


@dataclass(frozen=True)
class IngestResult:
    path: str
    status: IngestStatus
    paper_id: str | None = None
    source_file_id: str | None = None
    matched_by: str | None = None  # 'doi' | 'arxiv' | 'title_year' | 'explicit' | None
    attach: files.AttachResult | None = None
    attention_item_ids: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()  # papers an identity_check item names
    verify: VerifyOutcome | None = None
    detail: str = ""


@dataclass(frozen=True)
class DoiResult:
    doi: str
    status: DoiStatus
    paper_id: str | None = None
    title: str | None = None
    title_source: str | None = None
    oa: OaStatus | None = None
    oa_source: str | None = None  # "<provider>: <url>" of a fetched open-access copy
    attention_item_ids: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class NeedsPdf:
    paper_id: str
    title: str | None
    year: int | None
    authors: tuple[str, ...]
    doi: str | None
    doi_url: str | None
    graphs: tuple[str, ...] = field(default=())  # graphs whose frontier holds the paper


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"%PDF"
    except OSError:
        return False


# --- dropped PDFs -------------------------------------------------------------------------------


def add_pdfs(lib: Library, paths: Iterable[Path | str], *, source: Source = "upload",
             chain: Any = None) -> list[IngestResult]:
    """Add researcher-supplied PDFs, matching each to a library paper (module doc)."""
    method = AcquisitionMethod.folder_import if source == "folder" else AcquisitionMethod.upload
    return [_add_one(lib, Path(p), method, chain) for p in paths]


def add_folder(lib: Library, folder: Path | str, *, chain: Any = None) -> list[IngestResult]:
    """Add every ``*.pdf`` under an existing folder, read in place (sorted by path)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise LibraryError(f"not a folder: {folder}")
    pdfs = sorted((p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"),
                  key=lambda p: str(p).lower())
    return add_pdfs(lib, pdfs, source="folder", chain=chain)


def _add_one(lib: Library, path: Path, method: AcquisitionMethod, chain: Any) -> IngestResult:
    if not path.is_file() or not _is_pdf(path):
        return IngestResult(str(path), "skipped", detail="not a PDF file (it must start with %PDF)")
    sha = sha256_file(path)
    sf_id = f"sf_{sha}"
    holder = lib.conn.execute(
        "SELECT paper_id FROM paper_files WHERE source_file_id = ? ORDER BY paper_id LIMIT 1", (sf_id,)).fetchone()
    if holder is not None:
        return IngestResult(str(path), "duplicate_file", paper_id=holder[0], source_file_id=sf_id,
                            detail=f"this file is already attached to {holder[0]}")

    cover = cover_region(first_pages_from_pdf(path) or [])
    from ..work.tasks_metadata import deterministic_identifiers  # local: work imports the library

    found = deterministic_identifiers("\n".join(cover))
    ids = [(t, v) for t, v in normalize_ids([(k, v) for k, v in found.items() if v])]

    candidates: list[str] = []
    for id_type, value in ids:
        owner = lib.conn.execute("SELECT paper_id FROM identifiers WHERE id_type = ? AND id_value = ?",
                                 (id_type, value)).fetchone()
        if owner is None:
            continue
        if not _held(lib, owner[0]):
            return _attach_matched(lib, path, owner[0], id_type, method)
        candidates.append(owner[0])
    if not candidates:
        title_hits = _title_year_candidates(lib, cover, dict(ids).get("doi"))
        if len(title_hits) == 1 and not _held(lib, title_hits[0]):
            return _attach_matched(lib, path, title_hits[0], "title_year", method)
        candidates = title_hits
    return _new_paper(lib, path, sha, ids, candidates, method, chain)


def _held(lib: Library, paper_id: str) -> bool:
    return lib.conn.execute("SELECT 1 FROM paper_files WHERE paper_id = ? LIMIT 1", (paper_id,)).fetchone() is not None


def _title_year_candidates(lib: Library, cover: Sequence[str], pdf_doi: str | None) -> list[str]:
    """Papers whose year is printed on the cover and whose (non-generic) title passes the check."""
    text = "\n".join(cover)
    years = sorted({int(y) for y in _YEAR_RE.findall(text)})
    if not years:
        return []
    marks = ",".join("?" * len(years))
    rows = lib.conn.execute(
        f"SELECT paper_id, title FROM papers WHERE year IN ({marks}) AND COALESCE(TRIM(title), '') <> '' "
        "ORDER BY paper_id", years).fetchall()
    hits: list[str] = []
    for pid, title in rows:
        if pdf_doi is not None:
            other = [r[0] for r in lib.conn.execute(
                "SELECT id_value FROM identifiers WHERE paper_id = ? AND id_type = 'doi'", (pid,))]
            if other and pdf_doi not in other:
                continue
        if generic_title_reason(title) is not None:
            continue
        if check_pdf_text(clean_title(title) or title, None, cover).status == "pass":
            hits.append(pid)
    return hits


def _attach_matched(lib: Library, path: Path, paper_id: str, matched_by: str,
                    method: AcquisitionMethod) -> IngestResult:
    res = files.attach_pdf(lib, paper_id, path, access_class=AccessClass.user_supplied_private,
                           acquisition_method=method)
    items = tuple(i for i in (res.attention_item_id,) if i)
    return IngestResult(str(path), "attached", paper_id=paper_id, source_file_id=res.source_file_id,
                        matched_by=matched_by, attach=res, attention_item_ids=items,
                        detail=f"attached to {paper_id} (matched by {matched_by.replace('_', ' + ')})")


def _new_paper(lib: Library, path: Path, sha: str, ids: list[tuple[str, str]], candidates: list[str],
               method: AcquisitionMethod, chain: Any) -> IngestResult:
    free_ids = [(t, v) for t, v in ids if lib.conn.execute(
        "SELECT 1 FROM identifiers WHERE id_type = ? AND id_value = ?", (t, v)).fetchone() is None]
    taken = {r[0] for r in lib.conn.execute("SELECT paper_id FROM papers")}
    minted = mint_paper_id(derive_key(ids=free_ids, pdf_hashes=[sha]), taken)
    pid = minted.paper_id
    item_ids: list[str] = []
    with lib.write() as conn:
        conn.execute("INSERT INTO papers (paper_id, created_at) VALUES (?, ?)", (pid, _now()))
        for id_type, value in free_ids:
            conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value, source) VALUES (?, ?, ?, 'pdf_scan')",
                         (pid, id_type, value))
        item_ids.append(attention.raise_item(
            lib, METADATA_NEEDED, {"paper_id": pid}, paper_id=pid,
            detail="added from a PDF that matched no library paper; its title is not known yet").item_id)
        if candidates:
            item_ids.append(_identity_item(lib, pid, candidates, "the added PDF may be the same paper as"))
        res = files.attach_pdf(lib, pid, path, access_class=AccessClass.user_supplied_private,
                               acquisition_method=method)
    if res.attention_item_id:
        item_ids.append(res.attention_item_id)
    outcome = verify(lib, pid, chain=chain) if chain is not None and any(t == "doi" for t, _ in free_ids) else None
    return IngestResult(str(path), "new_paper", paper_id=pid, source_file_id=res.source_file_id, attach=res,
                        attention_item_ids=tuple(item_ids), candidates=tuple(candidates), verify=outcome,
                        detail="new paper; its metadata is needed" + (
                            f"; check whether it duplicates {', '.join(candidates)}" if candidates else ""))


def _identity_item(lib: Library, paper_id: str, candidates: Sequence[str], why: str) -> str:
    cands = sorted(dict.fromkeys(candidates))
    return attention.raise_item(
        lib, IDENTITY_CHECK, {"paper_id": paper_id, "candidates": cands}, paper_id=paper_id,
        detail=f"{why} {', '.join(cands)}; merge them or mark them as different papers").item_id


# --- explicit attach ---------------------------------------------------------------------------


def attach_to_paper(lib: Library, paper_id: str, pdf_path: Path | str, *,
                    access_class: AccessClass = AccessClass.user_supplied_private,
                    acquisition_method: AcquisitionMethod = AcquisitionMethod.upload) -> IngestResult:
    """Attach a PDF to the named paper with no matching (the paper page's "Add PDF").

    Refuses an unknown paper, a non-PDF file, and a paper that already holds another PDF
    (:class:`LibraryError`). The standing title check runs; a miss is flagged, never blocked.
    """
    path = Path(pdf_path)
    if not paper_exists(lib.conn, paper_id):
        raise LibraryError(f"unknown paper {paper_id!r}")
    if not path.is_file() or not _is_pdf(path):
        raise LibraryError(f"not a PDF file: {path}")
    res = files.attach_pdf(lib, paper_id, path, access_class=access_class, acquisition_method=acquisition_method)
    items = tuple(i for i in (res.attention_item_id,) if i)
    detail = "already attached" if res.already_attached else f"attached to {paper_id}"
    return IngestResult(str(path), "attached", paper_id=paper_id, source_file_id=res.source_file_id,
                        matched_by="explicit", attach=res, attention_item_ids=items, detail=detail)


# --- DOI paste ---------------------------------------------------------------------------------


def normalize_doi(raw: str) -> str | None:
    """A DOI from a pasted string (``doi:`` / ``https://doi.org/`` prefixes allowed), or ``None``."""
    doi = normalize_id("doi", raw)
    return doi if doi and _DOI_SHAPE.match(doi) else None


def add_dois(lib: Library, dois: Iterable[str], *, chain: Any,
             fetch: Callable[[dict], Any] | None = None) -> list[DoiResult]:
    """Add papers by DOI: provider metadata -> cited-only paper -> open-access fetch (module doc)."""
    if fetch is None:
        from ..acquisition.fetch import fetch_oa

        def fetch(record: dict) -> Any:  # noqa: F811 - the default open-access fetch
            return fetch_oa(record, chain=chain)
    return [_add_doi(lib, raw, chain, fetch) for raw in dois]


def _add_doi(lib: Library, raw: str, chain: Any, fetch: Callable[[dict], Any]) -> DoiResult:
    doi = normalize_doi(raw)
    if doi is None:
        return DoiResult(str(raw).strip(), "invalid", detail="not a DOI (expected 10.<digits>/<suffix>)")
    owner = lib.conn.execute("SELECT paper_id FROM identifiers WHERE id_type = 'doi' AND id_value = ?",
                             (doi,)).fetchone()
    try:
        record = lookup_doi(chain, doi) if owner is None or not _held(lib, owner[0]) else None
    except Exception as exc:  # noqa: BLE001 - reported, nothing written
        if owner is None:
            return DoiResult(doi, "lookup_failed", detail=f"provider lookup failed: {exc}")
        record = None
    if owner is not None:
        pid = owner[0]
        row = lib.conn.execute("SELECT title, title_source FROM papers WHERE paper_id = ?", (pid,)).fetchone()
        oa, oa_source = ("already_held", None) if _held(lib, pid) else _fetch_oa(lib, pid, record, fetch)
        return DoiResult(doi, "existing", paper_id=pid, title=row[0], title_source=row[1], oa=oa,
                         oa_source=oa_source, detail=f"already in the library as {pid}")
    if not record:
        return DoiResult(doi, "not_found", detail="no provider knows this DOI")

    provider = provider_of(record)
    values = record_values(record)
    generic = generic_title_reason(values["title"], venue=values["venue"]) if values["title"] else None
    rejected_title = values["title"] if generic else None
    if generic:
        values["title"] = None  # never saved; a metadata_generic_title item records it
    taken = {r[0] for r in lib.conn.execute("SELECT paper_id FROM papers")}
    pid = mint_paper_id(derive_key(ids=[("doi", doi)]), taken).paper_id
    item_ids: list[str] = []
    with lib.write() as conn:
        conn.execute(
            "INSERT INTO papers (paper_id, title, title_source, authors_json, year, venue, abstract, oa_status, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, values["title"], provider if values["title"] else None,
             json.dumps(values["authors"], ensure_ascii=False) if values["authors"] else None,
             values["year"], values["venue"], record.get("abstract") if isinstance(record.get("abstract"), str)
             else None, record.get("oa_status") if isinstance(record.get("oa_status"), str) else None, _now()))
        conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value, source) VALUES (?, 'doi', ?, ?)",
                     (pid, doi, provider))
        oa_id = normalize_id("openalex", record.get("openalex_id"))
        if oa_id and conn.execute("SELECT 1 FROM identifiers WHERE id_type = 'openalex' AND id_value = ?",
                                  (oa_id,)).fetchone() is None:
            conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value, source) VALUES (?, 'openalex', ?, ?)",
                         (pid, oa_id, provider))
        if rejected_title:
            item_ids.append(attention.raise_item(
                lib, GENERIC_TITLE, {"paper_id": pid, "title": rejected_title}, paper_id=pid,
                detail=f"{provider} gave the title '{rejected_title}' for {doi}, which was not saved: {generic}"
            ).item_id)
        if not values["title"]:
            item_ids.append(attention.raise_item(lib, METADATA_NEEDED, {"paper_id": pid}, paper_id=pid,
                                                 detail=f"{provider} gave no usable title for {doi}").item_id)
        similar = _similar_papers(lib, pid, values["title"], values["year"])
        if similar:
            item_ids.append(_identity_item(lib, pid, similar, "the pasted DOI's paper may be the same paper as"))
    oa, oa_source = _fetch_oa(lib, pid, record, fetch)
    return DoiResult(doi, "created", paper_id=pid, title=values["title"],
                     title_source=provider if values["title"] else None, oa=oa, oa_source=oa_source,
                     attention_item_ids=tuple(item_ids),
                     detail=f"cited-only paper from {provider}" + (
                         f"; open-access PDF from {oa_source}" if oa == "held" else ""))


def _similar_papers(lib: Library, paper_id: str, title: str | None, year: int | None) -> list[str]:
    if not title or year is None:
        return []
    return [pid for pid, other in lib.conn.execute(
        "SELECT paper_id, title FROM papers WHERE year = ? AND paper_id <> ? AND title IS NOT NULL "
        "ORDER BY paper_id", (year, paper_id)) if title_similarity(title, other) >= SAME_TITLE]


def _fetch_oa(lib: Library, paper_id: str, record: dict | None, fetch: Callable[[dict], Any]) -> tuple[OaStatus, str | None]:
    """Try an open-access copy; attach it when one is downloaded. Never raises."""
    if not record:
        return "none", None
    try:
        result = run_sync(fetch(dict(record)))
    except Exception:  # noqa: BLE001 - an open-access miss never fails the DOI paste
        return "failed", None
    if getattr(result, "status", None) != "downloaded" or not getattr(result, "content", None):
        return ("failed" if getattr(result, "status", None) == "failed" else "none"), None
    cand = result.candidate
    source = f"{cand.provider or 'open access'}: {cand.url}" if cand is not None else "open access"
    tmp = lib.root.parent / "tmp" / "oa"
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / f"{uuid.uuid4().hex}.pdf"
    path.write_bytes(result.content)
    try:
        if not _is_pdf(path):
            return "none", None
        attach_to_paper(lib, paper_id, path, access_class=AccessClass.open_access,
                        acquisition_method=AcquisitionMethod.open_access_fetch)
    finally:
        path.unlink(missing_ok=True)
    return "held", source


# --- needs a PDF -------------------------------------------------------------------------------


def _frontiers(lib: Library) -> dict[str, list[str]]:
    """paper_id -> slugs of the graphs whose frontier holds it."""
    out: dict[str, list[str]] = {}
    for info in graphs.list_graphs(lib):
        with graphs.open_graph(lib, info.slug) as g:
            for gp in graphs.membership(g, status="frontier"):
                out.setdefault(gp.paper_id, []).append(info.slug)
    return out


def needs_pdf(lib: Library, *, graph: str | None = None, frontier: bool = False) -> list[NeedsPdf]:
    """Cited-only papers with their DOI link, ordered by title then ``paper_id``.

    ``graph``: only that graph's frontier. ``frontier``: only papers on some graph's frontier.
    """
    fronts = _frontiers(lib)
    if graph is not None:
        graphs.open_graph(lib, graph).close()  # refuses an unknown graph
    rows = lib.conn.execute(
        "SELECT p.paper_id, p.title, p.year, p.authors_json, "
        "(SELECT MIN(i.id_value) FROM identifiers i WHERE i.paper_id = p.paper_id AND i.id_type = 'doi') AS doi "
        "FROM papers p JOIN paper_status s ON s.paper_id = p.paper_id WHERE s.status = 'cited_only' "
        "ORDER BY p.title IS NULL, p.title COLLATE NOCASE, p.paper_id").fetchall()
    out: list[NeedsPdf] = []
    for pid, title, year, authors_json, doi in rows:
        slugs = tuple(fronts.get(pid, ()))
        if graph is not None and graph not in slugs:
            continue
        if frontier and not slugs:
            continue
        try:
            authors = tuple(json.loads(authors_json)) if authors_json else ()
        except (TypeError, ValueError):
            authors = ()
        out.append(NeedsPdf(pid, title, year, authors, doi, DOI_URL.format(doi) if doi else None, slugs))
    return out
