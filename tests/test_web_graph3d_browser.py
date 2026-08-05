"""Track 3 Stage C — opt-in Playwright smoke for the 3D graph workspace.

**Deselected by default** (``@pytest.mark.browser``, like ``marker`` / ``network``)
and **self-skips when Playwright is not installed** — the offline suite never
drives a real browser. The heavy imports (``playwright``) live inside the test body
behind :func:`pytest.importorskip`, so collecting this module is free and the
default ``pytest -q`` run neither imports nor runs it.

To run for real::

    pip install -e .[browser] && playwright install chromium
    python -m pytest -m browser

It stands up the real FastAPI app under uvicorn on a loopback free port, performs
the ``/auth?t=`` cookie bootstrap, then asserts the task's acceptance checks:
the canvas renders, a work's source PDF/markdown is reachable from the session,
clicking a concept opens the provenance drawer with >=1 paper/claim/span, the
filters do not collapse the layout, and the 2d/table fallback is reachable.
"""

from __future__ import annotations

import threading
import time

import pytest

# Reuse the route test's project + cache fixture builder (no tests/__init__.py, so
# sibling test modules import as top-level under pytest's prepend import mode).
from test_web_graph3d_routes import SLUG, _setup

pytestmark = pytest.mark.browser


def _serve(app, host: str, port: int):
    """Run the app under uvicorn on a daemon thread; return (server, thread)."""
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("uvicorn did not start in time")
    return server, thread


def test_graph3d_browser_smoke():
    # self-skip cleanly when the opt-in browser extra is not installed.
    sync_api = pytest.importorskip("playwright.sync_api")

    from seedgraph.api.app import app
    from seedgraph.web import serve as serve_mod

    run_id = _setup()

    host = "127.0.0.1"
    port = serve_mod.find_free_port(host, 8900)
    # mint the session/csrf tokens the gate + /auth bootstrap expect.
    app.state.session_token = "browser-smoke-token"
    app.state.csrf_token = "browser-smoke-csrf"
    app.state.remote_bind = False

    server, _thread = _serve(app, host, port)
    base = f"http://{host}:{port}"
    try:
        with sync_api.sync_playwright() as p:
            # The package may be importable while the browser binary is not
            # installed (we deliberately never run `playwright install` in the
            # offline suite). Treat an un-launchable browser as a SELF-SKIP, not a
            # failure — same spirit as "self-skips when Playwright is not installed".
            try:
                browser = p.chromium.launch(timeout=8000)
            except Exception as exc:  # missing binary -> Error / TimeoutError / OSError
                # e.g. "Executable doesn't exist … run `playwright install`", or a
                # connection TimeoutError when no browser is provisioned. pytest.skip
                # raises BaseException (not Exception), so it is not re-caught here.
                pytest.skip(f"Playwright browser not installed/launchable: {exc}")
            context = browser.new_context()
            page = context.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))

            # /auth?t= -> sets the HttpOnly cookie and 302s to /ui (no query token after).
            page.goto(f"{base}/auth?t={app.state.session_token}")
            assert page.url.endswith("/ui")

            # ---- 3D view: canvas renders (non-blank) ----
            page.goto(f"{base}/ui/projects/{SLUG}/runs/{run_id}/graph?view=3d")
            page.wait_for_selector("#graph canvas", timeout=15000)
            box = page.eval_on_selector("#graph canvas", "c => ({w:c.width, h:c.height})")
            assert box["w"] > 0 and box["h"] > 0
            page.wait_for_timeout(1500)  # let the force layout settle a few ticks
            assert not errors, f"page errors: {errors}"

            # ---- work open-source reaches pdf + markdown (from the session) ----
            pdf = context.request.get(f"{base}/api/projects/{SLUG}/works/work_a/pdf")
            assert pdf.status == 200
            assert "application/pdf" in pdf.headers.get("content-type", "")
            md = context.request.get(f"{base}/api/projects/{SLUG}/works/work_a/markdown")
            assert md.status == 200

            # ---- concept click opens the drawer with >=1 paper/claim/span ----
            # the right pane groups concepts; rows start collapsed, so expand the
            # first group, then click a concept name to open its provenance drawer.
            page.wait_for_selector("#pane-list .grp-head", timeout=10000)
            first_group = page.locator("#pane-list .grp").first
            first_group.locator(".grp-head .caret").click()
            name = first_group.locator(".crow .nm").first
            name.wait_for(state="visible", timeout=10000)
            name.click()
            page.wait_for_selector("#drawer", state="visible", timeout=10000)
            page.wait_for_function(
                "() => { const b = document.querySelector('#drawer-body');"
                " return b && (b.querySelector('.dwork') || b.querySelector('.dclaim')"
                " || b.querySelector('.dspan')); }",
                timeout=10000,
            )

            # ---- filters do not collapse the layout (no crash, canvas stays sized) ----
            # close the drawer first (it overlays the bottom of the control panel).
            page.click("#drawer-close")
            page.wait_for_selector("#drawer", state="hidden", timeout=5000)
            page.fill("#search", "ols")
            page.check("#t-evidence")
            page.wait_for_timeout(500)
            page.uncheck("#t-evidence")
            page.fill("#search", "")
            box2 = page.eval_on_selector("#graph canvas", "c => ({w:c.width, h:c.height})")
            assert box2["w"] > 0 and box2["h"] > 0
            assert not errors, f"page errors after filtering: {errors}"

            # ---- 2d / table fallback reachable ----
            page.goto(f"{base}/ui/projects/{SLUG}/runs/{run_id}/graph?view=table")
            assert "graph (table view)" in page.content()

            context.close()
            browser.close()
    finally:
        server.should_exit = True
        _thread.join(timeout=10)
