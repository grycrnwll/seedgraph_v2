"""The scripted walk-through in a real browser (docs/V3_UI_DESIGN.md §10.2 #1).

    SEEDGRAPH_ACCEPTANCE_HOME=<v3 home> SEEDGRAPH_ACCEPTANCE_WORKER=agent \\
        uv run --group browser pytest -m browser tests/acceptance/test_walkthrough_browser.py -s

Needs ``<home>/ui/acceptance/inputs.json`` ``walkthrough``: ``pdf_path`` and
``pdf_sha256`` of one open-access PDF the researcher chose, not yet in the library, and
``graph`` (``diff_in_diff``). Starts the real ``seedgraph serve`` (its JobRunner
converts the dropped PDF; note that it also resumes the home's conversion queue).

1. Library -> drop the PDF -> the job strip goes from converting to converted, the
   standing check result shows.
2. Paper page -> add to the graph -> Write notes -> the batch page shows the copyable
   commands.
3. **Worker hook** (:func:`wait_for_worker`): writes ``WORKER_NEEDED.json`` (batch id,
   commands, CLI mirror) into the run folder and polls the batch page until the paper is
   done. ``SEEDGRAPH_ACCEPTANCE_WORKER=agent`` (the default) waits up to
   ``worker_timeout_s`` for an agent working the batch through the CLI mirror
   (``seedgraph-v3 work claim/submit``); ``none`` stops the walk-through here (skip).
4. Paper page -> click three quotes: each scrolls to its page and a ``mark.sg-hl``
   appears on that page.
5. Re-word, re-label and re-type one claim; hide one; add an assertion; add a
   quote-backed claim by selecting text in the PDF pane.
6. Search finds the edited text, the hidden claim is gone, MCP ``paper_show`` labels the
   researcher content.
7. Revert to the version before the hide: the claim is back; History lists every version.
8. Library list -> the PDF button opens the PDF.

Writes ``walkthrough.json`` into ``<home>/ui/acceptance/<run>/``.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

import pytest

from . import browser_support as bs
from . import harness as h

pytestmark = pytest.mark.browser

WORKER_ENV = "SEEDGRAPH_ACCEPTANCE_WORKER"
EDIT_TOKEN = "vellumgrove"


def _inputs(home: Path) -> dict[str, Any]:
    w = h.load_inputs(home).get("walkthrough") or {}
    pdf = Path(w.get("pdf_path") or "")
    if not w.get("pdf_path") or not pdf.is_file():
        pytest.skip("inputs.json walkthrough.pdf_path is not set to an existing PDF")
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    if w.get("pdf_sha256") and w["pdf_sha256"].lower() != sha:
        pytest.fail(f"{pdf} does not match inputs.json pdf_sha256")
    return {**w, "pdf": pdf, "sha": sha}


def _paper_for_hash(home: Path, sha: str, *, converted: bool = False) -> str | None:
    """The paper holding the PDF with this hash (with ``converted``: only once its
    markdown exists)."""
    from seedgraph.library.store import library_dir, open_library_at

    with open_library_at(library_dir(home)) as lib:
        row = lib.conn.execute("SELECT pf.paper_id, pf.markdown_id FROM paper_files pf JOIN source_files s "
                               "ON s.source_file_id = pf.source_file_id WHERE s.file_hash = ?", (sha,)).fetchone()
    if row is None or (converted and row[1] is None):
        return None
    return row[0]


def wait_for_worker(page, base: str, batch_id: str, run: Path, timeout_s: float) -> str:
    """The hook where an agent works the batch through the CLI mirror; returns the items
    partial's final text."""
    from seedgraph.work.commands import work_commands

    cmds = work_commands(batch_id)
    h.write_json(run / "WORKER_NEEDED.json", {
        "batch_id": batch_id, "commands": cmds.as_dict(),
        "cli_mirror": [f"seedgraph-v3 work claim {batch_id} --worker <name> --client claude",
                       f"seedgraph-v3 work submit {batch_id} ... (see the claim's reply)"]})
    print(f"\nWORKER NEEDED: batch {batch_id}\n  {cmds.claude_code}\n")
    deadline = time.monotonic() + timeout_s
    text = ""
    while time.monotonic() < deadline:
        page.goto(f"{base}/batches/{batch_id}")
        text = page.inner_text("main") if page.query_selector("main") else page.content()
        if re.search(r"\bdone\b", text, re.I) and not re.search(r"\b(queued|claimed)\b", text, re.I):
            return text
        time.sleep(15)
    pytest.fail(f"the batch {batch_id} was not worked within {timeout_s}s")


def _saved(page) -> str:
    page.wait_for_selector("#sg-notes [data-notes-message]", timeout=30_000)
    msg = page.inner_text("#sg-notes [data-notes-message]")
    cls = page.get_attribute("#sg-notes [data-notes-message]", "class") or ""
    assert "sg-flash-success" in cls, msg
    return msg


def _select_in_pdf(page) -> bool:
    """Select a long text-layer span on a rendered page and ask for the anchor; True when
    the add form came back with an exact unique slice."""
    spans = page.query_selector_all(".page[data-page-number] .textLayer span")
    tried = 0
    for span in spans:
        text = (span.text_content() or "").strip()
        if len(text) < 40 or tried > 25:
            continue
        tried += 1
        span.evaluate("""(el) => { const r = document.createRange(); r.selectNodeContents(el);
                         const s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
                         document.dispatchEvent(new MouseEvent('mouseup', {bubbles: true})); }""")
        page.wait_for_selector("#sg-selection-bar:not([hidden])", timeout=5_000)
        page.click("[data-add-from-selection]")
        page.wait_for_selector("#sg-add-claim .sg-anchor", timeout=15_000)
        if page.get_attribute("#sg-add-claim .sg-anchor", "data-anchor-status") == "ok":
            return True
    return False


def test_walkthrough():
    home = bs.acceptance_home()
    inp = _inputs(home)
    graph = inp.get("graph") or "diff_in_diff"
    worker = os.environ.get(WORKER_ENV, inp.get("worker") or "agent")
    if _paper_for_hash(home, inp["sha"]):
        pytest.skip("the walk-through PDF is already in the library; choose another (inputs.json)")
    run = h.new_run(home, f"walkthrough-{h.stamp()}")
    log: dict[str, Any] = {"steps": []}

    def step(name: str, **detail: Any) -> None:
        log["steps"].append({"step": name, **detail})
        h.write_json(run / "walkthrough.json", log)

    with bs.chromium() as browser, h.LiveServer(home) as server:
        base = server.base_url
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, accept_downloads=True)
        page = ctx.new_page()

        # 1. drop the PDF; converting -> converted; the standing check shows
        page.goto(f"{base}/library")
        page.set_input_files("#sg-files", str(inp["pdf"]))
        page.click("form[action='/library/add'] button[type=submit]")
        page.wait_for_selector("#sg-jobs .sg-job", timeout=60_000)
        deadline = time.monotonic() + float(inp.get("convert_timeout_s", 3600))
        strip = ""
        seen_converting = False
        while time.monotonic() < deadline:
            strip = page.inner_text("#sg-jobs")
            seen_converting = seen_converting or "convert" in strip.lower()
            # the strip may stay active for other queued conversions: wait for this PDF's text
            if _paper_for_hash(home, inp["sha"], converted=True):
                break
            time.sleep(3)
        pid = _paper_for_hash(home, inp["sha"], converted=True)
        assert pid, strip
        page.goto(f"{base}/library")
        strip = page.inner_text("#sg-jobs")
        check_shown = any(t in strip for t in ("PDF matches the paper", "PDF probably matches", "may not be this paper",
                                                "could not check"))
        step("drop PDF", paper_id=pid, saw_converting=seen_converting, standing_check_shown=check_shown,
             strip=strip[-1500:])
        assert check_shown, strip

        # 2. add to the graph, Write notes, the batch page's copyable commands
        page.goto(f"{base}/papers/{pid}")
        if page.query_selector("form[action$='/graphs'] select[name=slug]"):
            page.select_option("form[action$='/graphs'] select[name=slug]", graph)
            page.select_option("form[action$='/graphs'] select[name=status]", "member")
            page.click("form[action$='/graphs'] button[type=submit]")
        page.goto(f"{base}/papers/{pid}")
        page.click("form[action$='/write-notes'] button[type=submit]")
        page.wait_for_url(re.compile(r".*/batches/b_[0-9a-f]+.*"), timeout=30_000)
        batch_id = re.search(r"/batches/(b_[0-9a-f]+)", page.url).group(1)
        cmd = page.inner_text("#sg-cmd-claude")
        assert batch_id in cmd
        step("write notes", batch_id=batch_id, command=cmd)

        # 3. the worker hook
        if worker == "none":
            step("worker skipped")
            pytest.skip(f"{WORKER_ENV}=none: batch {batch_id} left for an agent; rerun to continue")
        items = wait_for_worker(page, base, batch_id, run, float(inp.get("worker_timeout_s", 3600)))
        step("worker done", items=items[-800:])

        # 4. three quotes, each highlighted on its own page
        page.goto(f"{base}/papers/{pid}?layout=split")
        bs.wait_pdf_ready(page)
        clicked = []
        for btn in bs.quote_buttons(page):
            p, pe = btn.get_attribute("data-page"), btn.get_attribute("data-page-end")
            if not p:
                continue
            result = page.evaluate(bs.SHOW_QUOTE_JS, btn)
            ok = result["marks"] > 0 and all(int(p) <= x <= int(pe or p) for x in result["pages"])
            clicked.append({"page": int(p), "page_end": int(pe or p), **result, "ok": ok})
            if len([c for c in clicked if c["ok"]]) >= 3:
                break
        step("three quotes", clicked=clicked)
        assert len([c for c in clicked if c["ok"]]) >= 3, clicked

        # 5. edit (re-word, re-label, re-type), hide, add assertion, add quote-backed by PDF selection
        page.goto(f"{base}/papers/{pid}?layout=split")
        bs.wait_pdf_ready(page)
        articles = page.query_selector_all("article[data-claim-id]")
        edit_art, hide_art = articles[0], articles[1]
        edit_id, hide_id = edit_art.get_attribute("data-claim-id"), hide_art.get_attribute("data-claim-id")
        edit_art.query_selector("details.sg-claim-edit summary").click()
        old_type = edit_art.query_selector("select[name=claim_type]").input_value()
        options = [o.get_attribute("value") for o in edit_art.query_selector_all("select[name=claim_type] option")]
        new_type = next(o for o in options if o != old_type)
        edit_art.query_selector("select[name=claim_type]").select_option(new_type)
        edit_art.query_selector("input[name=normalized_label]").fill(f"relabelled {EDIT_TOKEN}")
        edit_art.query_selector("textarea[name=claim_text]").fill(f"Researcher re-wording {EDIT_TOKEN}.")
        edit_art.query_selector("button[type=submit]").click()
        _saved(page)
        page.click(f"#claim-{hide_id} form[action$='/notes/hide'] button")
        _saved(page)
        hide_version = int(page.get_attribute("#sg-notes", "data-expected-version"))
        page.fill("form[action$='/notes/add'] input[name=normalized_label]", f"assertion {EDIT_TOKEN}")
        page.fill("form[action$='/notes/add'] textarea[name=claim_text]", f"An invented assertion {EDIT_TOKEN}.")
        page.click("form[action$='/notes/add'] button[type=submit]")
        _saved(page)
        assert _select_in_pdf(page), "no PDF selection anchored uniquely"
        page.fill("[data-anchor-form] input[name=normalized_label]", f"selected {EDIT_TOKEN}")
        page.fill("[data-anchor-form] textarea[name=claim_text]", f"A claim from a PDF selection {EDIT_TOKEN}.")
        page.click("[data-anchor-form] button[type=submit]")
        _saved(page)
        step("edits", edit=edit_id, hide=hide_id, new_type=new_type, hide_version=hide_version)

        # 6. search, hidden gone, MCP paper_show labels
        from seedgraph.library import search
        from seedgraph.library.store import library_dir, open_library_at
        from seedgraph.mcp.tools_library import _paper_show

        with open_library_at(library_dir(home)) as lib:
            hits = search.search(lib, EDIT_TOKEN, kind="claims", limit=None)
            hit_ids = {x.claim_id for x in hits}
            show = {c["claim_id"]: c for c in (_paper_show(lib, pid)["current_note"] or {}).get("claims", [])}
        researcher = sorted(cid for cid, c in show.items() if c["trust"] == "researcher")
        step("search and MCP", hits=sorted(hit_ids), researcher_in_paper_show=researcher)
        assert edit_id in hit_ids and hide_id not in show and len(researcher) >= 3

        # 7. revert to the version before the hide; History lists every version
        page.click("details[data-history] summary")
        page.click(f"details[data-history] li[data-version='{hide_version - 1}'] form button")
        _saved(page)
        assert page.query_selector(f"#claim-{hide_id}") is not None
        page.click("details[data-history] summary")
        versions = [int(li.get_attribute("data-version")) for li in page.query_selector_all("details[data-history] li[data-version]")]
        head = int(page.get_attribute("#sg-notes", "data-expected-version"))
        step("revert and history", versions=versions, head=head)
        assert sorted(versions) == list(range(1, head + 1)), versions

        # 8. the Library list's PDF button opens the PDF
        page.goto(f"{base}/library?view=held&graph={graph}")
        link = page.query_selector(f"tr[data-paper='{pid}'] a.sg-pdf-button")
        if link is None:
            page.goto(f"{base}/library?view=all")
            link = page.query_selector(f"tr[data-paper='{pid}'] a.sg-pdf-button")
        assert link is not None
        href = link.get_attribute("href")
        resp = ctx.request.get(f"{base}{href}")
        with ctx.expect_page() as new:
            link.click()
        pdf_page = new.value
        pdf_page.wait_for_load_state()
        step("PDF button", href=href, status=resp.status, content_type=resp.headers.get("content-type"),
             opened=pdf_page.url)
        assert resp.status == 200 and resp.headers.get("content-type", "").startswith("application/pdf")
        assert pdf_page.url.endswith(href)
