"""The Library page (design §2.2, §6): list views, adding papers (dropped PDFs, a folder
path, pasted DOIs) as background jobs with a polled job strip, Ollama metadata, and
"Write notes" (estimate -> ``work.routine.create_batch`` -> the batch page).

Every POST makes one deep call: the add endpoints submit one job to the
:class:`~seedgraph.web.jobs.JobRunner` (the job calls ``library.ingest``), Write notes
calls ``work.routine.create_batch``. The job functions below are adapters: each builds
the provider chain in its own worker thread (``web.chain`` when the app was given one,
else :func:`~seedgraph.library.metadata_verify.default_chain`, closed afterwards).

Rules chosen where the design is silent:
- Uploads are read with Starlette's form parser (file parts spool to the system temp
  folder), then copied in chunks to ``<home>/tmp/upload/<job>/<n>.pdf`` under generated
  names, at most 200 MB each and only when they start with ``%PDF``; that folder is
  deleted when the ingest job ends. A typed folder is read in place and never deleted.
- Ollama metadata is a button ("Run Ollama metadata"): one job that creates a
  ``metadata`` batch for the held papers needing metadata and works it with Ollama.
  It is not chained automatically after each conversion.
"""

from __future__ import annotations

import secrets
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from starlette.datastructures import UploadFile
from starlette.routing import Route

from ...errors import SeedgraphError
from ...library import ingest
from ...work import routine
from ..app import Flash, get_web
from ..jobs import JobFailed, JobResult
from ..views import library as library_view
from ._pages import form, is_htmx, not_found, render

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_FILES = 500
CHUNK = 1024 * 1024
JOB_KINDS = frozenset({"ingest", "folder", "doi", "metadata", "convert"})
CHECK_LABELS = {"pass": "PDF matches the paper", "weak": "PDF probably matches the paper",
                "miss": "this PDF may not be this paper", "unknown": "could not check"}


# --- job functions (run on a JobRunner worker thread with that thread's Library) --------------


@contextmanager
def provider_chain(lib, given: Any):
    if given is not None:
        yield given
        return
    from ...library import metadata_verify

    chain = metadata_verify.default_chain(lib)
    try:
        yield chain
    finally:
        chain.cache_engine.close()


def _ingest_lines(results, names: dict[str, str], runner) -> tuple[list[str], list[str]]:
    lines, papers = [], []
    for r in results:
        name = names.get(r.path, Path(r.path).name)
        if r.paper_id:
            papers.append(r.paper_id)
        if r.status == "skipped":
            lines.append(f"{name}: skipped ({r.detail or 'not a PDF'})")
            continue
        if r.status == "duplicate_file":
            lines.append(f"{name}: already in the library ({r.paper_id})")
            continue
        what = "new paper" if r.status == "new_paper" else f"added to {r.paper_id}"
        if r.matched_by and r.status != "new_paper":
            what += f" (matched by {r.matched_by.replace('_', ' + ')})"
        check = CHECK_LABELS.get(r.attach.check.status, r.attach.check.status) if r.attach else "could not check"
        tail = ""
        if r.attach is not None and r.attach.queued:
            runner.convert(r.attach.source_file_id, label=f"convert {name}")
            tail = "; converting"
        if r.candidates:
            tail += "; may duplicate another paper (see Review)"
        lines.append(f"{name}: {what}; {check}{tail}")
    return lines, papers


def ingest_job(lib, runner, chain, paths: list[str], names: dict[str, str], cleanup: str | None) -> JobResult:
    """Dropped PDFs -> :func:`seedgraph.library.ingest.add_pdfs`, then queue conversions."""
    try:
        with provider_chain(lib, chain) as ch:
            results = ingest.add_pdfs(lib, paths, source="upload", chain=ch)
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
    lines, papers = _ingest_lines(results, names, runner)
    return JobResult("\n".join(lines) or "no PDFs", paper_ids=tuple(dict.fromkeys(papers)))


def folder_job(lib, runner, chain, folder: str) -> JobResult:
    """A typed folder, read in place -> :func:`seedgraph.library.ingest.add_folder`."""
    with provider_chain(lib, chain) as ch:
        results = ingest.add_folder(lib, folder, chain=ch)
    lines, papers = _ingest_lines(results, {}, runner)
    return JobResult("\n".join(lines) or "no PDFs in that folder", paper_ids=tuple(dict.fromkeys(papers)))


DOI_WORDS = {"created": "added as cited-only", "existing": "already in the library", "not_found": "not found",
             "invalid": "not a DOI", "lookup_failed": "lookup failed"}
OA_WORDS = {"held": "open-access PDF added", "none": "no open-access copy; add the PDF yourself",
            "failed": "open-access download failed", "already_held": "PDF already held"}


def doi_job(lib, runner, chain, dois: list[str]) -> JobResult:
    """Pasted DOIs -> :func:`seedgraph.library.ingest.add_dois` (open-access copies only)."""
    with provider_chain(lib, chain) as ch:
        results = ingest.add_dois(lib, dois, chain=ch)
    lines = []
    for r in results:
        line = f"{r.doi}: {DOI_WORDS.get(r.status, r.status)}"
        if r.title:
            line += f" ({r.title})"
        if r.oa:
            line += f"; {OA_WORDS.get(r.oa, r.oa)}"
            if r.oa_source:
                line += f" from {r.oa_source}"
        lines.append(line)
    if any(r.oa == "held" for r in results):
        runner.resume_queue()  # convert the fetched open-access PDFs
    return JobResult("\n".join(lines), paper_ids=tuple(r.paper_id for r in results if r.paper_id))


def ollama_job(lib, chain, model: str | None) -> JobResult:
    """A ``metadata`` batch for the held papers needing metadata, worked by Ollama with
    title verification by DOI."""
    from ...llm import ollama
    from ...work import ollama_worker

    reach = ollama.check_reachable()
    if not reach.ok:
        raise JobFailed("Ollama is not reachable; the papers keep their 'metadata needed' items "
                        f"({reach.error or reach.base_url})")
    if not model:
        raise JobFailed("no Ollama model is set (SEEDGRAPH_OLLAMA_MODEL)")
    batch = routine.create_batch(lib, "metadata", query="held needing metadata", client="web")
    with provider_chain(lib, chain) as ch:
        report = ollama_worker.run_ollama_worker(lib, batch.batch_id, model=model, chain=ch)
    text = f"Ollama metadata batch {batch.batch_id}: {report.saved} saved, stopped ({report.stopped})"
    if report.error:
        text += f": {report.error}"
    return JobResult(text, paper_ids=tuple(i.subject_id for i in batch.items))


# --- pages -------------------------------------------------------------------------------------


def _jobs(web):
    return library_view.jobs_strip(web.jobs.jobs(), kinds=JOB_KINDS)


def library_page(request):
    web = get_web(request)
    q = request.query_params
    try:
        page_no = int(q.get("page") or 1)
    except ValueError:
        page_no = 1
    try:
        view = library_view.page(web.library(), view=q.get("view") or "held", graph=q.get("graph") or None,
                                 q=q.get("q") or "", page_no=page_no, frontier=bool(q.get("frontier")))
    except library_view.UnknownGraph:
        return not_found("no such graph")
    return render(request, "library/library.html", {
        "page_title": "Library", "view": view, "jobs": _jobs(web),
        "metadata_needed": library_view.metadata_needed_count(web.library()),
        "back": request.url.path + (f"?{request.url.query}" if request.url.query else ""),
    }, nav="library")


def jobs_partial(request):
    return get_web(request).render(request, "library/_jobs.html", {"jobs": _jobs(get_web(request))})


def _added(request, web, text: str):
    if is_htmx(request):
        return web.render(request, "library/_jobs.html", {"jobs": _jobs(web), "note": text})
    return web.redirect("/library#sg-add-papers", flash=[Flash(text, "success")])


def _refused(request, web, text: str):
    if is_htmx(request):
        return web.render(request, "library/_jobs.html", {"jobs": _jobs(web), "note": text, "error": True},
                          status_code=200)
    return web.redirect("/library#sg-add-papers", flash=[Flash(text, "error")])


def _copy_upload(upload: UploadFile, target: Path) -> str | None:
    """Copy one uploaded part; returns a refusal reason or ``None``."""
    upload.file.seek(0)
    if upload.file.read(5) != b"%PDF-":
        return "not a PDF"
    upload.file.seek(0)
    size = 0
    with target.open("wb") as out:
        while chunk := upload.file.read(CHUNK):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                return "larger than 200 MB"
            out.write(chunk)
    return None


def add_upload(request):
    web = get_web(request)
    data = form(request, max_files=MAX_FILES)
    uploads = [u for u in data.getlist("files") if isinstance(u, UploadFile) and u.filename]
    if not uploads:
        return _refused(request, web, "choose PDF files to add")
    folder = web.home / "tmp" / "upload" / secrets.token_hex(8)
    folder.mkdir(parents=True, exist_ok=False)
    paths, names, refused = [], {}, []
    try:
        for n, upload in enumerate(uploads, start=1):
            target = folder / f"{n:04d}.pdf"  # the uploaded filename is never used as a path
            reason = _copy_upload(upload, target)
            if reason:
                target.unlink(missing_ok=True)
                refused.append(f"{upload.filename}: {reason}")
                continue
            paths.append(str(target))
            names[str(target)] = Path(upload.filename).name
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    if not paths:
        shutil.rmtree(folder, ignore_errors=True)
        return _refused(request, web, "; ".join(refused) or "no PDFs")
    web.jobs.submit("ingest", ingest_job, web.jobs, web.chain, paths, names, str(folder),
                    label=f"add {len(paths)} PDF{'s' if len(paths) != 1 else ''}")
    note = f"adding {len(paths)} PDF{'s' if len(paths) != 1 else ''}"
    if refused:
        note += " (refused: " + "; ".join(refused) + ")"
    return _added(request, web, note)


def add_folder(request):
    web = get_web(request)
    raw = str(form(request).get("folder") or "").strip().strip('"')
    if not raw or not Path(raw).is_dir():
        return _refused(request, web, f"not a folder: {raw}" if raw else "type a folder path")
    web.jobs.submit("folder", folder_job, web.jobs, web.chain, str(Path(raw)), label=f"add folder {raw}")
    return _added(request, web, f"adding the PDFs in {raw} (read in place)")


def add_dois(request):
    web = get_web(request)
    raw = str(form(request).get("dois") or "")
    dois = [line.strip() for line in raw.replace(",", "\n").splitlines() if line.strip()]
    if not dois:
        return _refused(request, web, "paste one DOI per line")
    web.jobs.submit("doi", doi_job, web.jobs, web.chain, dois,
                    label=f"look up {len(dois)} DOI{'s' if len(dois) != 1 else ''}")
    return _added(request, web, f"looking up {len(dois)} DOI{'s' if len(dois) != 1 else ''}")


def ollama_metadata(request):
    from ...llm import ollama

    web = get_web(request)
    form(request)
    web.jobs.submit("metadata", ollama_job, web.chain, ollama.default_model(), label="Ollama metadata")
    return _added(request, web, "Ollama metadata started")


# --- Write notes -------------------------------------------------------------------------------


def write_notes_page(request):
    web = get_web(request)
    q = request.query_params
    lib = web.library()
    ids = q.getlist("paper")
    if not ids and q.get("view"):
        try:
            ids = library_view.view_ids(lib, q.get("view") or "held", graph=q.get("graph") or None,
                                        q=q.get("q") or "")
        except SeedgraphError:
            return not_found("no such graph")
    view = library_view.write_notes(lib, ids, replace=bool(q.get("replace"))) if ids else None
    return render(request, "library/write_notes.html", {"page_title": "Write notes", "view": view},
                  nav="library")


def write_notes_confirm(request):
    web = get_web(request)
    data = form(request)
    ids = [str(p) for p in data.getlist("paper") if str(p).strip()]
    if not ids:
        return web.redirect("/library", flash=[Flash("select papers first", "warning")])
    try:
        batch = routine.create_batch(web.library(), "notes", paper_ids=ids,
                                     options={"replace": bool(data.get("replace"))}, client="web")
    except SeedgraphError as exc:
        return web.redirect("/library", flash=[Flash(f"no batch created: {exc}", "error")])
    return web.redirect(f"/batches/{batch.batch_id}",
                        flash=[Flash("notes batch created; run it with one of the commands below", "success")])


ROUTES = [
    Route("/library", library_page, methods=["GET"]),
    Route("/library/jobs", jobs_partial, methods=["GET"]),
    Route("/library/add", add_upload, methods=["POST"]),
    Route("/library/add-folder", add_folder, methods=["POST"]),
    Route("/library/add-dois", add_dois, methods=["POST"]),
    Route("/library/ollama-metadata", ollama_metadata, methods=["POST"]),
    Route("/library/write-notes", write_notes_page, methods=["GET"]),
    Route("/library/write-notes", write_notes_confirm, methods=["POST"]),
]
