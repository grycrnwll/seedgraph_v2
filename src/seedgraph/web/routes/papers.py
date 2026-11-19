"""The paper page and its header actions (docs/V3_UI_DESIGN.md §2.4). Thin adapters: a GET
renders :func:`views.paper.page`; each POST calls one deep function and redirects back.

    GET  /papers/{id}?layout=split|pdf|notes&show_hidden=1   the page
         &edit={claim_id}  that claim's edit form open (and "show hidden" for a hidden claim)
         &from=spot-check:{sample_id}  adds a "back to the spot-check" link
    POST /papers/{id}/metadata      library.metadata_edit.set_user_metadata (title_source='user')
    POST /papers/{id}/graphs        library.graphs.set_membership
    POST /papers/{id}/write-notes   work.routine.create_batch (one paper) -> /batches/{batch_id}
    POST /papers/{id}/add-pdf       library.ingest.attach_to_paper (multipart, via htmx)
"""

from __future__ import annotations

import secrets
import shutil

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from ...errors import SeedgraphError
from ..app import Flash, get_web
from ..views import paper as paper_view
from ..views import spotcheck as spotcheck_view
from ._paperkit import done, form, not_found, paper_id_or_none

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
GRAPH_STATUSES = ("member", "frontier")


def paper_page(request: Request) -> Response:
    web = get_web(request)
    pid = request.path_params["paper_id"]
    layout = request.query_params.get("layout", "split")
    show_hidden = request.query_params.get("show_hidden") == "1"
    view = paper_view.page(web.library(), pid, layout=layout, show_hidden=show_hidden,
                           edit=request.query_params.get("edit") or None,
                           back_sample=spotcheck_view.back_sample(request.query_params.get("from")))
    if view is None:
        return not_found()
    return web.render(request, "paper/page.html", {"view": view, "page_title": view.header.display_title},
                      nav="library")


def edit_metadata(request: Request) -> Response:
    from ...library.metadata_edit import set_user_metadata

    web = get_web(request)
    pid = paper_id_or_none(request)
    if pid is None:
        return not_found()
    data = form(request)
    back = f"/papers/{pid}"
    try:
        changed = set_user_metadata(web.library(), pid, title=data.get("title"), year=data.get("year"),
                                    venue=data.get("venue"),
                                    authors=str(data.get("authors") or "").splitlines())
    except SeedgraphError as exc:
        return done(request, web, back, [Flash(f"metadata not saved: {exc}", "error")])
    text = f"metadata saved ({', '.join(changed)})" if changed else "metadata saved (no changes)"
    return done(request, web, back, [Flash(text, "success")])


def add_to_graph(request: Request) -> Response:
    from ...library import graphs

    web = get_web(request)
    pid = paper_id_or_none(request)
    if pid is None:
        return not_found()
    data = form(request)
    slug = str(data.get("slug") or "")
    status = str(data.get("status") or "member")
    back = f"/papers/{pid}"
    if status not in GRAPH_STATUSES:
        return done(request, web, back, [Flash(f"unknown graph status {status!r}", "error")])
    try:
        with graphs.open_graph(web.library(), slug) as g:
            graphs.set_membership(g, pid, status)  # type: ignore[arg-type]
    except SeedgraphError as exc:
        return done(request, web, back, [Flash(f"not added: {exc}", "error")])
    return done(request, web, back, [Flash(f"added to {slug} as {status}", "success")])


def write_notes(request: Request) -> Response:
    from ...notes import interpretation
    from ...work import routine

    web = get_web(request)
    pid = paper_id_or_none(request)
    if pid is None:
        return not_found()
    form(request)  # consumed so the CSRF-checked body is read before responding
    lib = web.library()
    replace = interpretation.current(lib, pid, "researcher") is not None
    try:
        batch = routine.create_batch(lib, "notes", paper_ids=[pid], options={"replace": replace},
                                     client="web")
    except SeedgraphError as exc:
        return done(request, web, f"/papers/{pid}", [Flash(f"no batch created: {exc}", "error")])
    return done(request, web, f"/batches/{batch.batch_id}",
                [Flash("notes batch created; run it with one of the commands below", "success")])


def attach_paper_pdf(lib, paper_id, path):
    """Adapter onto U5's deep function (the paper page's "Add PDF")."""
    from ...library import ingest

    return ingest.attach_to_paper(lib, paper_id, path)


def add_pdf(request: Request) -> Response:
    web = get_web(request)
    pid = paper_id_or_none(request)
    if pid is None:
        return not_found()
    back = f"/papers/{pid}"
    data = form(request, max_files=1, max_fields=8, max_part_size=MAX_UPLOAD_BYTES)
    upload = data.get("pdf")
    if upload is None or isinstance(upload, str):
        return done(request, web, back, [Flash("choose a PDF file to add", "error")])
    folder = web.home / "tmp" / "upload" / secrets.token_hex(8)
    folder.mkdir(parents=True, exist_ok=False)
    target = folder / "upload.pdf"  # the uploaded filename is never used as a path
    try:
        size = 0
        upload.file.seek(0)
        head = upload.file.read(5)
        if head != b"%PDF-":
            return done(request, web, back, [Flash("that file is not a PDF", "error")])
        upload.file.seek(0)
        with target.open("wb") as out:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    return done(request, web, back, [Flash("the PDF is larger than 200 MB", "error")])
                out.write(chunk)
        lib = web.library()
        try:
            result = attach_paper_pdf(lib, pid, target)
        except SeedgraphError as exc:
            return done(request, web, back, [Flash(f"PDF not added: {exc}", "error")])
        attach = result.attach
        if attach is not None and attach.queued and attach.source_file_id:
            web.jobs.convert(attach.source_file_id, label=f"convert {pid}")
        check = paper_view.PDF_CHECK_LABELS.get(attach.check.status if attach else "unknown", "")
        return done(request, web, back, [Flash(f"PDF added; {check}", "success")])
    finally:
        shutil.rmtree(folder, ignore_errors=True)


ROUTES = [
    Route("/papers/{paper_id}", paper_page, methods=["GET"], name="paper"),
    Route("/papers/{paper_id}/metadata", edit_metadata, methods=["POST"]),
    Route("/papers/{paper_id}/graphs", add_to_graph, methods=["POST"]),
    Route("/papers/{paper_id}/write-notes", write_notes, methods=["POST"]),
    Route("/papers/{paper_id}/add-pdf", add_pdf, methods=["POST"]),
]
