"""``pdfpane.js`` text-layer handling in a real Chromium, on invented text (no home needed).

The page is a stub served from ``web/static`` through Playwright routing: ``pdfpane.js``
loads (its ``init`` returns early without a PDF container) and its finder runs over
hand-built PDF.js-style text layers (spans, ``<br>`` at line ends).

Opt-in: ``uv run --group browser pytest -m browser tests/acceptance/test_pdfpane_browser.py``.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

import pytest

from . import browser_support as bs

pytestmark = pytest.mark.browser

STATIC = Path(__file__).resolve().parents[2] / "src" / "seedgraph" / "web" / "static"
ORIGIN = "http://sg.test"

# Two dollar amounts on different lines: joined without the line break, the fold's
# inline-math rule ($...$) would swallow the prose between them.
LAYER = (  # no whitespace between elements, as PDF.js builds it
    '<div class="page" data-page-number="1"><div class="textLayer">'
    '<span>The orchard grant paid out $4 million</span><br role="presentation">'
    '<span>to the river towns, and the lantern fund later</span><br role="presentation">'
    '<span>added a further $2 million.</span>'
    '</div></div>'
)

STUB = f"""<!doctype html><html><head><meta charset="utf-8"></head><body>
{LAYER}
<script type="module" src="/static/js/pdfpane.js"></script>
</body></html>"""


def _serve(route) -> None:
    url = route.request.url
    if url in (f"{ORIGIN}/", f"{ORIGIN}/stub"):
        route.fulfill(status=200, content_type="text/html", body=STUB)
        return
    rel = url[len(ORIGIN):].split("?", 1)[0]
    if rel.startswith("/static/"):
        path = STATIC / rel[len("/static/"):]
        if path.is_file():
            ctype = "text/javascript" if path.suffix in (".js", ".mjs") else (
                mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            route.fulfill(status=200, content_type=ctype, body=path.read_bytes())
            return
    route.fulfill(status=404, body="")


def test_finder_keeps_line_breaks_between_text_layer_lines():
    with bs.chromium() as browser:
        page = browser.new_page()
        page.route(f"{ORIGIN}/**", _serve)
        page.goto(f"{ORIGIN}/stub")
        page.wait_for_function("() => !!(window.seedgraph && window.seedgraph.pdfPane)")
        got = page.evaluate("""() => {
            const layers = Array.from(document.querySelectorAll('.textLayer'));
            return window.seedgraph.pdfPane.locate(layers, 'totherivertownsandthelanternfundlater');
        }""")
        assert got is not None, "the passage between two dollar amounts was folded away"
        marked = page.evaluate("""() => window.seedgraph.pdfPane.highlightIn(
            Array.from(document.querySelectorAll('.textLayer')), 'totherivertownsandthelanternfundlater')""")
        assert marked == "to the river towns, and the lantern fund later"
