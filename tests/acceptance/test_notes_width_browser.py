"""The claim edit fields fill the notes pane in a real Chromium (researcher request after the
stage 3 walk-through: they rendered as small boxes that did not follow the panel width).

Synthetic: a throwaway home built by ``tests/web/paper_support.build`` (invented text), served
by uvicorn on a loopback port as in test_form_post_browser.py; no real home needed.

Opt-in: ``uv run --group browser pytest -m browser tests/acceptance/test_notes_width_browser.py``.
"""

from __future__ import annotations

import threading
import time

import pytest
from web import paper_support as ps  # tests/ is on sys.path (rootdir package layout)

from . import browser_support as bs
from . import harness as h

pytestmark = pytest.mark.browser


@pytest.fixture
def server(tmp_path):
    import uvicorn

    from seedgraph.commands.serve import bind_loopback
    from seedgraph.web.app import create_app

    home = tmp_path / "home"
    fx = ps.build(home, tmp_path)
    port = h.free_port()
    app = create_app(home, port=port)
    sock = bind_loopback(port)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off",
                                        proxy_headers=False, server_header=False))
    thread = threading.Thread(target=srv.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not srv.started and time.monotonic() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", fx
    srv.should_exit = True
    thread.join(timeout=15)


def _ratio(page, claim_id: str) -> tuple[float, float, float]:
    page.click(f"#claim-{claim_id} .sg-claim-edit > summary")
    field = page.locator(f"#claim-{claim_id} textarea[name='claim_text']")
    field.wait_for(state="visible")
    pane = page.locator(".sg-pane-notes").bounding_box()
    box = field.bounding_box()
    return box["width"] / pane["width"], box["width"], pane["width"]


@pytest.mark.parametrize("layout", ["split", "notes"])
def test_the_claim_text_field_fills_the_notes_pane(server, layout):
    base, fx = server
    cid = fx.claim("method_or_model[0]")
    with bs.chromium() as browser:
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        watch = bs.watch_console(page)
        page.goto(f"{base}/papers/{fx.held}?layout={layout}")
        ratio, field_w, pane_w = _ratio(page, cid)
        assert ratio >= 0.9, f"claim text field {field_w:.0f}px of a {pane_w:.0f}px notes pane ({layout})"
        label = page.locator(f"#claim-{cid} input[name='normalized_label']").bounding_box()
        assert label["width"] / pane_w >= 0.9
        csp = watch.summary()["csp_count"]
        assert csp == 0, watch.summary()["csp_violations"]
