"""Browser highlight pass over proto5 (docs/V3_UI_DESIGN.md §10.2 #3, last bullet).

For every anchored quote on the five prototype test papers, run ``pdfpane.js``'s own
``showQuote`` in Chromium and record where ``mark.sg-hl`` lands, then compare with the
server-side tier from ``library.pagecheck`` (``current`` mode, the claims the page
shows): ``verified`` should agree with "highlighted on the assigned page", ``page_only``
with the page-only outline. The agreement table is a report, not a gate (§12 #4); the
test fails only when the pass did not cover every quote or a highlight landed outside
the assigned page range and its neighbours.

Opt-in: ``SEEDGRAPH_ACCEPTANCE_HOME=<v3 home> uv run --group browser pytest -m browser
tests/acceptance/test_highlight_browser.py``. Served without the lifespan
(:class:`harness.QuietServer`), so the home's conversion queue is not resumed.
Writes ``<home>/ui/acceptance/<run>/highlight_browser.json``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from . import browser_support as bs
from . import harness as h

pytestmark = pytest.mark.browser


def server_tiers(home, set_name: str = "proto5") -> tuple[list[str], dict[str, dict[str, Any]]]:
    from seedgraph.library import pagecheck
    from seedgraph.library.store import library_dir

    with pagecheck.open_readonly(library_dir(home)) as lib:
        report = pagecheck.run(lib, home, set_name=set_name, mode="current", verify=True)
    tiers = {q.claim_id: {"paper_id": q.paper_id, "page": q.page, "page_end": q.page_end or q.page,
                          "how": q.how, "tier": q.tier} for q in report.quotes}
    return [p.paper_id for p in report.papers], tiers


def classify(result: dict[str, Any], page: int | None, page_end: int | None) -> str:
    if page is None:
        return "no_page"
    pages = [p for p in result["pages"] if p is not None]
    if result["marks"] and pages and all(page <= p <= (page_end or page) for p in pages):
        return "on_page"
    if result["marks"] and pages and all(page - 1 <= p <= (page_end or page) + 1 for p in pages):
        return "neighbour"
    if result["marks"]:
        return "elsewhere"
    if result["outlined"]:
        return "page_only"
    return "nothing"


AGREES = {("verified", "on_page"), ("page_only", "page_only"), ("page_only", "neighbour"),
          ("no_page", "no_page")}


def test_proto5_finder_agrees_with_server_tiers():
    home = bs.acceptance_home()
    papers, tiers = server_tiers(home)
    rows: list[dict[str, Any]] = []
    with bs.chromium() as browser, h.QuietServer(home) as server:
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        for pid in papers:
            page.goto(f"{server.base_url}/papers/{pid}?layout=split")
            bs.wait_pdf_ready(page)
            for btn in bs.quote_buttons(page):
                cid = btn.evaluate("(el) => el.closest('article').getAttribute('data-claim-id')")
                if cid not in tiers:
                    continue
                t = tiers[cid]
                result = page.evaluate(bs.SHOW_QUOTE_JS, btn)
                got = classify(result, t["page"], t["page_end"])
                rows.append({"paper_id": pid, "claim_id": cid, "server_tier": t["tier"], "how": t["how"],
                             "page": t["page"], "page_end": t["page_end"], "browser": got,
                             "mark_pages": result["pages"], "agrees": (t["tier"], got) in AGREES})
    matrix = Counter(f"{r['server_tier']} / {r['browser']}" for r in rows)
    summary = {"quotes_server": len(tiers), "quotes_browser": len(rows),
               "agree": sum(r["agrees"] for r in rows), "matrix": dict(matrix),
               "elsewhere": [r for r in rows if r["browser"] == "elsewhere"]}
    run = h.new_run(home, f"highlight-{h.stamp()}")
    h.write_json(run / "highlight_browser.json", {"summary": summary, "rows": rows})
    print(summary)
    assert summary["quotes_browser"] == summary["quotes_server"], summary
    assert not summary["elsewhere"], summary["elsewhere"][:5]
