"""Playwright helpers for the opt-in ``browser`` acceptance tests.

The tests skip cleanly when the Playwright package (the ``browser`` dependency group) or
its Chromium build (``uv run --group browser playwright install chromium``) is missing,
or when ``SEEDGRAPH_ACCEPTANCE_HOME`` does not name the v3 home to test (the suite's
own conftest gives every test a throwaway home, so the real one must be named
explicitly).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

HOME_ENV = "SEEDGRAPH_ACCEPTANCE_HOME"
HEADED_ENV = "SEEDGRAPH_ACCEPTANCE_HEADED"


def acceptance_home() -> Path:
    raw = os.environ.get(HOME_ENV)
    if not raw:
        pytest.skip(f"set {HOME_ENV} to the v3 home to run the browser acceptance tests")
    home = Path(raw)
    if not (home / "library").is_dir():
        pytest.skip(f"{HOME_ENV}={home} has no library")
    return home


@contextmanager
def chromium() -> Iterator[object]:
    """A launched Chromium, or ``pytest.skip`` when Playwright or the browser is missing."""
    sync_api = pytest.importorskip("playwright.sync_api", reason="the `browser` dependency group is not installed")
    pw = sync_api.sync_playwright().start()
    try:
        exe = Path(pw.chromium.executable_path)
        if not exe.exists():
            pytest.skip("Chromium for Playwright is not installed "
                        "(`uv run --group browser playwright install chromium`)")
        try:
            browser = pw.chromium.launch(headless=not os.environ.get(HEADED_ENV))
        except sync_api.Error as exc:  # pragma: no cover - depends on the machine
            pytest.skip(f"Chromium did not launch: {exc}")
        try:
            yield browser
        finally:
            browser.close()
    finally:
        pw.stop()


# The page-side probe for one quote button: run pdfpane.js's own showQuote, then report
# where (if anywhere) mark.sg-hl landed and whether the page-only outline was used.
SHOW_QUOTE_JS = """
async (el) => {
  await window.seedgraph.pdfPane.showQuote(el);
  const marks = Array.from(document.querySelectorAll("mark.sg-hl"));
  const pages = Array.from(new Set(marks.map((m) => {
    const p = m.closest(".page[data-page-number]");
    return p ? parseInt(p.getAttribute("data-page-number"), 10) : null;
  })));
  const outlined = Array.from(document.querySelectorAll(".page.sg-hl-page"))
    .map((p) => parseInt(p.getAttribute("data-page-number"), 10));
  return { marks: marks.length, pages, outlined,
           text: marks.map((m) => m.textContent).join("").slice(0, 200) };
}
"""

PDF_READY_JS = "() => !!(window.seedgraph && window.seedgraph.pdfPane && window.seedgraph.pdfPane.state.viewer " \
               "&& window.seedgraph.pdfPane.state.viewer.pagesCount)"


def wait_pdf_ready(page, timeout_ms: int = 60_000) -> None:
    page.wait_for_function(PDF_READY_JS, timeout=timeout_ms)


def quote_buttons(page) -> list:
    """The paper page's quote buttons (one per anchored quote; the page chips are skipped)."""
    return page.query_selector_all("article[data-claim-id] button.sg-quote-link[data-quote]")
