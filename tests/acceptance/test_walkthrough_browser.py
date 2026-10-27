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
3. **Worker hook**: writes ``WORKER_NEEDED.json`` (batch id, paper id, commands, CLI
   mirror, how to resume) and the batch page into the run folder.
   ``SEEDGRAPH_ACCEPTANCE_WORKER=agent`` (the default) then polls the batch (its
   ``batch_items`` state in the database) for up to ``worker_timeout_s`` while an agent
   works it through the CLI mirror (``seedgraph-v3 work claim/submit``) and carries on;
   ``none`` stops here (skip) and serve is stopped.
   **Resume** after the batch is worked: ``SEEDGRAPH_ACCEPTANCE_RESUME=<run folder name>``
   (e.g. ``walkthrough-20261027T130000Z``) reruns the same test in that run folder: it
   starts a fresh serve, checks once that the batch is done (fails clearly if not) and
   does steps 4-8, appending to the same ``walkthrough.json`` (a run that stopped after
   step 1 resumes at step 2). No serve process has to
   survive the hand-off.
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
import json
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


RESUME_ENV = "SEEDGRAPH_ACCEPTANCE_RESUME"
DONE = ("done",)
PENDING = ("queued", "in_progress", "held_back")


def batch_states(home: Path, batch_id: str) -> dict[str, str]:
    """``{subject_id: state}`` of the batch's items, read-only from the database."""
    from seedgraph.library import pagecheck
    from seedgraph.library.store import library_dir

    with pagecheck.open_readonly(library_dir(home)) as lib:
        return dict(lib.conn.execute("SELECT subject_id, state FROM batch_items WHERE batch_id = ?",
                                     (batch_id,)).fetchall())


def worker_done(home: Path, batch_id: str, pid: str) -> bool:
    return batch_states(home, batch_id).get(pid) in DONE


def worker_needed(page, base: str, run: Path, batch_id: str, pid: str, graph: str) -> Path:
    """Write the hand-off for the agent (``WORKER_NEEDED.json``) and the batch page."""
    from seedgraph.work.commands import work_commands

    cmds = work_commands(batch_id)
    page.goto(f"{base}/batches/{batch_id}")
    (run / "batch_page.html").write_text(page.content(), encoding="utf-8")
    resume = f"{RESUME_ENV}={run.name}"
    path = h.write_json(run / "WORKER_NEEDED.json", {
        "batch_id": batch_id, "paper_id": pid, "graph": graph, "commands": cmds.as_dict(),
        "cli_mirror": [f"seedgraph-v3 work claim {batch_id} --worker <name> --client claude",
                       f"seedgraph-v3 work submit {batch_id} ... (see the claim's reply)"],
        "resume": (f"{bs.HOME_ENV}=<home> {resume} uv run --group browser pytest -m browser "
                   "tests/acceptance/test_walkthrough_browser.py -s")})
    print(f"\nWORKER NEEDED: batch {batch_id}\n  {cmds.claude_code}\n  resume with {resume}\n")
    return path


def wait_for_worker(home: Path, batch_id: str, pid: str, timeout_s: float) -> dict[str, str]:
    """Poll the batch's items until the paper is done (every 15 s, up to ``timeout_s``)."""
    deadline = time.monotonic() + timeout_s
    while True:
        states = batch_states(home, batch_id)
        if states.get(pid) in DONE:
            return states
        if states.get(pid) not in PENDING:
            pytest.fail(f"batch {batch_id}: the paper ended in state {states.get(pid)!r}")
        if time.monotonic() >= deadline:
            pytest.fail(f"the batch {batch_id} was not worked within {timeout_s}s")
        time.sleep(15)


def _version(page) -> int:
    return int(page.get_attribute("#sg-notes", "data-expected-version"))


def _saved(page, before: int | None = None) -> str:
    """Wait for the notes pane swap that answers a save (its version moves past
    ``before`` when given; otherwise the old pane's message could be read) and check the
    success message."""
    if before is not None:
        page.wait_for_function(
            "(v) => { const n = document.querySelector('#sg-notes');"
            " return n && parseInt(n.getAttribute('data-expected-version'), 10) > v; }",
            arg=before, timeout=30_000)
    page.wait_for_selector("#sg-notes [data-notes-message]", timeout=30_000)
    msg = page.inner_text("#sg-notes [data-notes-message]")
    cls = page.get_attribute("#sg-notes [data-notes-message]", "class") or ""
    assert "sg-flash-success" in cls, msg
    return msg


def _select_in_pdf(page) -> bool:
    """Select a long text-layer span on a rendered page and ask for the anchor; True when
    the add form came back with an exact unique slice."""
    page.wait_for_selector(".page[data-page-number] .textLayer span", timeout=30_000)
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
        # wait for this attempt's answer, not the previous attempt's box
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith("/anchor")):
            page.click("[data-add-from-selection]")
        page.wait_for_function("() => { const a = document.querySelector('#sg-add-claim .sg-anchor');"
                               " return a && !a.hasAttribute('data-sg-seen'); }", timeout=15_000)
        status = page.get_attribute("#sg-add-claim .sg-anchor", "data-anchor-status")
        page.eval_on_selector("#sg-add-claim .sg-anchor", "(a) => a.setAttribute('data-sg-seen', '')")
        if status == "ok":
            page.wait_for_selector("[data-anchor-form] input[name=normalized_label]", state="visible",
                                   timeout=15_000)
            return True
    return False


def _drop_pdf(page, base: str, home: Path, inp: dict[str, Any], step) -> str:
    """Step 1: drop the PDF and wait for its conversion. Returns the paper id."""
    # 1. drop the PDF; converting -> converted; the standing check shows
    page.goto(f"{base}/library")
    page.set_input_files("#sg-files", str(inp["pdf"]))
    page.click("form[action='/library/add'] button[type=submit]")
    page.wait_for_selector("#sg-jobs .sg-job", timeout=60_000)
    deadline = time.monotonic() + float(inp.get("convert_timeout_s", 3600))
    strip = ""
    seen_converting = False
    own_line = None
    while time.monotonic() < deadline:
        strip = page.inner_text("#sg-jobs")
        seen_converting = seen_converting or "convert" in strip.lower()
        own_line = own_line or _own_check_line(page, inp["pdf"].name)
        # the strip may stay active for other queued conversions: wait for this PDF's text
        if _paper_for_hash(home, inp["sha"], converted=True):
            break
        time.sleep(10)
    pid = _paper_for_hash(home, inp["sha"], converted=True)
    assert pid, strip
    page.goto(f"{base}/library")
    own_line = own_line or _own_check_line(page, inp["pdf"].name)
    step("drop PDF", paper_id=pid, saw_converting=seen_converting, standing_check=own_line)
    assert own_line, f"no job-strip line for {inp['pdf'].name} with its standing check: {strip[-1500:]}"
    return pid


CHECK_PHRASES = ("PDF matches the paper", "PDF probably matches the paper", "this PDF may not be this paper",
                 "could not check")


def _own_check_line(page, filename: str) -> str | None:
    """The job-strip line the ingest job wrote for this PDF (``<file>: new paper; <check>...``),
    when it carries a standing-check result; else None."""
    for li in page.query_selector_all("#sg-jobs .sg-job-lines li"):
        text = (li.text_content() or "").strip()
        if text.startswith(f"{filename}:") and any(p in text for p in CHECK_PHRASES):
            return text
    return None


def _write_notes(page, base: str, pid: str, graph: str, step) -> str:
    """Step 2: add the paper to the graph, Write notes, the batch page's copyable
    commands. Returns the batch id."""
    page.goto(f"{base}/papers/{pid}")
    if page.query_selector("form[action$='/graphs'] select[name=slug]"):
        page.select_option("form[action$='/graphs'] select[name=slug]", graph)
        page.select_option("form[action$='/graphs'] select[name=status]", "member")
        with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith("/graphs")) as resp:
            page.click("form[action$='/graphs'] button[type=submit]")
        assert resp.value.status < 400, f"add to graph: {resp.value.status} {resp.value.text()[:200]}"
    page.goto(f"{base}/papers/{pid}")
    in_graphs = page.inner_text(".sg-paper-graphs") if page.query_selector(".sg-paper-graphs") else ""
    assert graph in in_graphs, f"the paper is not in {graph}: {in_graphs!r}"
    page.click("form[action$='/write-notes'] button[type=submit]")
    page.wait_for_url(re.compile(r".*/batches/b_[0-9a-f]+.*"), timeout=30_000)
    batch_id = re.search(r"/batches/(b_[0-9a-f]+)", page.url).group(1)
    cmd = page.inner_text("#sg-cmd-claude")
    assert batch_id in cmd
    step("write notes", paper_id=pid, batch_id=batch_id, command=cmd, in_graphs=in_graphs)
    return batch_id


def _resume_state(home: Path, name: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """The run folder, its log, and how far it got: ``{"paper_id", "batch_id"}`` (from
    ``WORKER_NEEDED.json``), or just ``{"paper_id"}`` when only step 1 passed."""
    run = h.acceptance_dir(home) / name
    if not (run / "walkthrough.json").is_file():
        pytest.fail(f"{RESUME_ENV}={name}: {run / 'walkthrough.json'} is missing (run the walk-through first)")
    log = json.loads((run / "walkthrough.json").read_text(encoding="utf-8"))
    need = run / "WORKER_NEEDED.json"
    if need.is_file():
        return run, log, json.loads(need.read_text(encoding="utf-8"))
    dropped = [s for s in log["steps"] if s.get("step") == "drop PDF" and s.get("paper_id")]
    if not dropped:
        pytest.fail(f"{RESUME_ENV}={name}: step 1 never passed there; start a new walk-through")
    return run, log, {"paper_id": dropped[-1]["paper_id"]}


def test_walkthrough():
    home = bs.acceptance_home()
    inp = _inputs(home)
    graph = inp.get("graph") or "diff_in_diff"
    worker = os.environ.get(WORKER_ENV, inp.get("worker") or "agent")
    resume = os.environ.get(RESUME_ENV)
    pid = batch_id = None
    if resume:
        run, log, need = _resume_state(home, resume)
        pid, batch_id = need["paper_id"], need.get("batch_id")
        if _paper_for_hash(home, inp["sha"]) != pid:
            pytest.fail(f"{RESUME_ENV}={resume}: the walk-through PDF is no longer paper {pid}")
        if batch_id and not worker_done(home, batch_id, pid):
            pytest.fail(f"{RESUME_ENV}={resume}: batch {batch_id} is not done yet "
                        f"(items: {batch_states(home, batch_id)}); work it, then rerun")
    else:
        if _paper_for_hash(home, inp["sha"]):
            pytest.skip("the walk-through PDF is already in the library; choose another (inputs.json)")
        run = h.new_run(home, f"walkthrough-{h.stamp()}")
        log = {"steps": []}

    def step(name: str, **detail: Any) -> None:
        log["steps"].append({"step": name, **detail})
        h.write_json(run / "walkthrough.json", log)

    with bs.chromium() as browser, h.LiveServer(home) as server:
        base = server.base_url
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, accept_downloads=True)
        page = ctx.new_page()
        if resume:
            step("resumed", run=run.name, paper_id=pid, batch_id=batch_id,
                 batch_items=batch_states(home, batch_id) if batch_id else None)
        if pid is None:
            pid = _drop_pdf(page, base, home, inp, step)
        if batch_id is None:
            batch_id = _write_notes(page, base, pid, graph, step)
            worker_needed(page, base, run, batch_id, pid, graph)
            if worker == "none":
                step("worker needed", batch_id=batch_id, resume=f"{RESUME_ENV}={run.name}")
                pytest.skip(f"{WORKER_ENV}=none: batch {batch_id} left for an agent; "
                            f"resume with {RESUME_ENV}={run.name}")
            states = wait_for_worker(home, batch_id, pid, float(inp.get("worker_timeout_s", 3600)))
            step("worker done", batch_items=states)

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
        # Hidden claims are not listed, so on a resume after a partial run the first two listed
        # claims are still visible ones; the step works from the current head version.
        head_before = _version(page)
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
        v = _version(page)
        edit_art.query_selector("details.sg-claim-edit button[type=submit]").click()
        _saved(page, v)
        v = _version(page)
        page.click(f"#claim-{hide_id} form[action$='/notes/hide'] button")
        _saved(page, v)
        hide_version = _version(page)
        assert page.query_selector(f"#claim-{hide_id}") is None, "the hidden claim is still listed"
        # the add-assertion form sits in a closed <details>: open it the way a person would
        v = _version(page)
        page.click("details.sg-add-assertion summary")
        page.fill("details.sg-add-assertion input[name=normalized_label]", f"assertion {EDIT_TOKEN}")
        page.fill("details.sg-add-assertion textarea[name=claim_text]", f"An invented assertion {EDIT_TOKEN}.")
        page.click("details.sg-add-assertion button[type=submit]")
        _saved(page, v)
        assert _select_in_pdf(page), "no PDF selection anchored uniquely"
        page.fill("[data-anchor-form] input[name=normalized_label]", f"selected {EDIT_TOKEN}")
        page.fill("[data-anchor-form] textarea[name=claim_text]", f"A claim from a PDF selection {EDIT_TOKEN}.")
        v = _version(page)
        page.click("[data-anchor-form] button[type=submit]")
        _saved(page, v)
        step("edits", head_before=head_before, edit=edit_id, hide=hide_id, new_type=new_type,
             hide_version=hide_version, head_after=_version(page))

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
        v = _version(page)
        page.click(f"details[data-history] li[data-version='{hide_version - 1}'] form button")
        _saved(page, v)
        assert page.query_selector(f"#claim-{hide_id}") is not None
        page.click("details[data-history] summary")
        versions = [int(li.get_attribute("data-version")) for li in page.query_selector_all("details[data-history] li[data-version]")]
        head = int(page.get_attribute("#sg-notes", "data-expected-version"))
        step("revert and history", versions=versions, head=head)
        assert sorted(versions) == list(range(1, head + 1)), versions

        # 8. the Library list's PDF button opens the PDF
        # The list is paged (100 rows) and ordered by title, untitled papers last: page
        # through the graph's held view the way a person would.
        link = None
        for page_no in range(1, 50):
            page.goto(f"{base}/library?view=held&graph={graph}&page={page_no}")
            link = page.query_selector(f"tr[data-paper='{pid}'] a.sg-pdf-button")
            rows = page.query_selector_all("tr[data-paper]")
            if link is not None or not rows or f"page={page_no + 1}" not in page.content():
                break
        assert link is not None, f"{pid} has no PDF button in the {graph} held list"
        href = link.get_attribute("href")
        resp = ctx.request.get(f"{base}{href}")
        # The headless shell has no PDF viewer: the new tab hands the PDF to a download instead
        # of showing it (no "load", empty URL). So the check is: a new tab opens, it requests
        # the PDF URL, and that URL answers 200 application/pdf.
        with ctx.expect_event("response", predicate=lambda r: r.url.endswith(href)) as tab_resp:
            with ctx.expect_page() as new:
                link.click()
        pdf_page = new.value
        step("PDF button", href=href, status=resp.status, content_type=resp.headers.get("content-type"),
             new_tab=True, tab_request=tab_resp.value.url, tab_status=tab_resp.value.status,
             tab_url=pdf_page.url)
        assert resp.status == 200 and resp.headers.get("content-type", "").startswith("application/pdf")
        assert tab_resp.value.frame.page == pdf_page or pdf_page.url in ("", tab_resp.value.url)
        assert tab_resp.value.status == 200
