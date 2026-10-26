"""The PDF route (docs/V3_UI_DESIGN.md §8 "Content"): serves only library blobs looked up by
``paper_id``. No route takes a file path; a cited-only paper, an unknown id or a missing
file is 404.

    GET /papers/{id}/pdf   application/pdf, inline (Range requests are honoured)
"""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import FileResponse, Response
from starlette.routing import Route

from ..app import get_web
from ..views import paper as paper_view
from ._paperkit import not_found


def paper_pdf(request: Request) -> Response:
    pid = request.path_params["paper_id"]
    path = paper_view.pdf_file(get_web(request).library(), pid)
    if path is None:
        return not_found("no PDF for this paper")
    return FileResponse(path, media_type="application/pdf", filename=f"{pid}.pdf",
                        content_disposition_type="inline", headers={"Cache-Control": "no-store"})


ROUTES = [Route("/papers/{paper_id}/pdf", paper_pdf, methods=["GET"])]
