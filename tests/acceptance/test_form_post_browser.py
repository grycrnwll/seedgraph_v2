"""A plain (non-htmx) form POST from the app's own page passes the CSRF guard in Chromium.

The guard requires ``Origin`` to equal the app's origin when present. A browser sends
``Origin: null`` on a form navigation POST when the page's ``Referrer-Policy`` is
``no-referrer`` (Fetch: "append a request Origin header"), which refused every plain
form (paper page "Add to graph", "Write notes", ...) with 403. Synthetic: a throwaway
home and a test-only echo route; no real home needed.

Opt-in: ``uv run --group browser pytest -m browser tests/acceptance/test_form_post_browser.py``.
"""

from __future__ import annotations

import threading
import time

import pytest
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from . import browser_support as bs
from . import harness as h

pytestmark = pytest.mark.browser


async def _echo(request: Request):
    form = await request.form()
    return PlainTextResponse(f"note={form.get('note', '')}")


@pytest.fixture
def server(tmp_path):
    import uvicorn

    from seedgraph.commands.serve import bind_loopback
    from seedgraph.library.store import open_library_at
    from seedgraph.web.app import create_app

    home = tmp_path / "home"
    open_library_at(home / "library").close()
    port = h.free_port()
    app = create_app(home, port=port, routes=[Route("/test/echo", _echo, methods=["POST"])])
    sock = bind_loopback(port)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off",
                                        proxy_headers=False, server_header=False))
    thread = threading.Thread(target=srv.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not srv.started and time.monotonic() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=15)


def test_plain_form_post_from_the_page_is_accepted(server):
    with bs.chromium() as browser:
        page = browser.new_page()
        page.goto(f"{server}/")
        # Build the form through the DevTools protocol (no inline script, so the CSP is untouched).
        page.evaluate("""() => {
            const token = document.querySelector('meta[name="csrf-token"]').content;
            const f = document.createElement('form');
            f.method = 'post'; f.action = '/test/echo';
            for (const [k, v] of [['csrf_token', token], ['note', 'kept']]) {
              const i = document.createElement('input'); i.type = 'hidden'; i.name = k; i.value = v;
              f.appendChild(i);
            }
            const b = document.createElement('button'); b.type = 'submit'; f.appendChild(b);
            document.body.appendChild(f);
        }""")
        with page.expect_response(lambda r: r.url.endswith("/test/echo")) as resp:
            page.click("form[action='/test/echo'] button")
        assert resp.value.status == 200, resp.value.text()
        assert resp.value.text() == "note=kept"
