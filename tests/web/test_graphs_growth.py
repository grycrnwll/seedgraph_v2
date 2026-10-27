"""Guided growth wizard on the graph page (stage 4 design §8.2): seeds -> depth and cap with the
estimate -> the walk as a JobRunner job (htmx polling) -> review found papers with reasons and
OA status -> approve as a job -> results, with the needs-a-PDF list and DOI links. An approved
open-access paper's references are parsed once it converts. Fake provider chain, fake
converter, stubbed open-access fetch: no network. Invented text only."""

from __future__ import annotations

import re
import threading

import pytest
from library.test_files import text_pdf
from library.test_growth import GrowthChain, held, make_graph, pid_for, rec

from seedgraph.acquisition.fetch import FetchResult, OACandidate
from seedgraph.conversion import ConvertOutput
from seedgraph.library import graphs, growth
from seedgraph.library.store import open_library_at
from seedgraph.web.jobs import JobRunner

from .pages_support import no_inline_script_or_style, post_forms_lead_with_csrf, token

SLUG = "ovens"
X, Y, Z = "10.9999/x.1", "10.9999/y.2", "10.9999/z.3"
YEAST = "10.9999/yeast.1"
REFS_MD = (
    "# Invented Yeast Ledger\n\n## Introduction\n\n"
    + " ".join(f"Sentence {k} describes an invented ledger in some detail." for k in range(60))
    + "\n\n## References\n\nBaker, A. (2021). Invented yeast ledgers and their keepers. Journal of Invented "
      "Bread, 3, 1-20. doi:10.9999/yeast.1\n"
)


class RefsPool:
    """A fake converter pool whose markdown carries a reference list."""

    size = 1
    converter_version = "fake-refs-1"

    def __init__(self):
        self._lock = threading.Lock()
        self.n = 0

    def convert(self, pdf_path, options=None, *, timeout=None) -> ConvertOutput:
        with self._lock:
            self.n += 1
            n = self.n
        return ConvertOutput(markdown=REFS_MD + f"\nInvented note {n}.\n", page_count=1, metadata={}, warnings=[],
                             timings={}, converter_version=self.converter_version)


@pytest.fixture
def world(home, tmp_path):
    lib = open_library_at(home / "library")
    try:
        a = held(lib, tmp_path, "lp_a", "Invented Ovens at Dawn", "10.9999/a")
        b = held(lib, tmp_path, "lp_b", "Invented Bread Prices", "10.9999/b")
        make_graph(lib, SLUG, [a, b], seeds=[a])
    finally:
        lib.close()
    chain = GrowthChain(
        records={YEAST: rec(YEAST, "Invented yeast ledgers and their keepers", 2021)},
        refs={"10.9999/a": [rec(X, "Invented Yeast Ledger", oa_status="gold"), rec(Y, "Invented Crust Census",
                                                                                  oa_status="closed")],
              "10.9999/b": [rec(X, "Invented Yeast Ledger", oa_status="gold"), rec(Z, "Invented Salt Roads")]})
    return home, chain


@pytest.fixture
def oa_fetch(monkeypatch, tmp_path):
    """Open-access copies exist only for X; every request is recorded."""
    import seedgraph.acquisition.fetch as fetch_mod

    pdf = text_pdf(tmp_path / "oa.pdf", ["Invented Yeast Ledger doi:" + X, "Body."]).read_bytes()
    asked = []

    async def stub(record, *, chain=None, **kw):
        asked.append(record.get("doi"))
        if record.get("doi") == X:
            return FetchResult(status="downloaded", content=pdf,
                               candidate=OACandidate(url="https://repo.invalid/x.pdf", provider="unpaywall",
                                                     is_oa_asserted=True))
        return FetchResult(status="stub")

    monkeypatch.setattr(fetch_mod, "fetch_oa", stub)
    return asked


def membership(home):
    lib = open_library_at(home / "library")
    try:
        with graphs.open_graph(lib, SLUG) as g:
            return {r.paper_id: r.status for r in graphs.membership(g)}
    finally:
        lib.close()


def references_of(home, paper_id):
    lib = open_library_at(home / "library")
    try:
        return [tuple(r) for r in lib.conn.execute(
            "SELECT resolution_status, resolved_paper_id FROM reference_entries WHERE citing_paper_id = ? "
            "ORDER BY ordinal", (paper_id,))]
    finally:
        lib.close()


def test_step_one_and_two_pick_seeds_and_show_the_estimate_without_network(world, make_app, client_for):
    _, chain = world
    c = client_for(make_app(chain=chain))
    html = c.get(f"/graphs/{SLUG}/grow").text
    assert re.search(r'name="seed" value="lp_a" checked', html) and not re.search(r'value="lp_b" checked', html)
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)
    est = c.get(f"/graphs/{SLUG}/grow?seed=lp_a&seed=lp_b&depth=2&cap=10&estimate=1").text
    assert "22 papers at most" in est and "66 provider calls" in est
    assert chain.calls == []  # the estimate is closed-form


def test_the_walk_runs_as_a_job_then_found_papers_show_reasons_and_oa(world, make_app, client_for, monkeypatch):
    home, chain = world
    calls = []
    real = growth.walk

    def spy(lib, slug, **kw):
        calls.append((slug, kw["seeds"], kw["depth"], kw["cap"]))
        return real(lib, slug, **kw)

    monkeypatch.setattr(growth, "walk", spy)
    app = make_app(chain=chain, pool=RefsPool())
    with client_for(app) as c:
        assert app.state.web.jobs.wait_idle(timeout=30)  # the members' own conversions
        data = {"seed": ["lp_a", "lp_b"], "depth": "1", "cap": "10"}
        assert c.post(f"/graphs/{SLUG}/grow/walk", data=data).status_code == 403
        r = c.post(f"/graphs/{SLUG}/grow/walk", data={"csrf_token": token(c), **data})
        assert r.status_code == 303
        polling = c.get(f"/graphs/{SLUG}/grow/jobs").text
        assert app.state.web.jobs.wait_idle(timeout=30)
        html = c.get(f"/graphs/{SLUG}/grow").text
    assert calls == [(SLUG, ["lp_a", "lp_b"], 1, 10)]
    assert "growth walk" in polling
    rows = {pid: re.search(rf'<tr>(?:(?!</tr>).)*name="choice:{pid}"(?:(?!</tr>).)*</tr>', html, re.S)
            for pid in (pid_for(X), pid_for(Y), pid_for(Z))}
    assert all(rows.values())
    x_row = rows[pid_for(X)].group(0)
    assert "cited by 2 members" in x_row and ">yes<" in x_row and 'value="member"' not in x_row  # cited-only
    assert ">no<" in rows[pid_for(Y)].group(0) and ">unknown<" in rows[pid_for(Z)].group(0)
    assert membership(home) == {"lp_a": "member", "lp_b": "member"}  # nothing joins without approval
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)


def test_approval_adds_the_ticked_papers_fetches_oa_and_lists_the_rest_as_needing_a_pdf(
        world, make_app, client_for, oa_fetch):
    home, chain = world
    app = make_app(chain=chain, pool=RefsPool())
    with client_for(app) as c:
        jobs = app.state.web.jobs
        assert jobs.wait_idle(timeout=30)
        c.post(f"/graphs/{SLUG}/grow/walk", data={"csrf_token": token(c), "seed": ["lp_a", "lp_b"], "depth": "1",
                                                  "cap": "10"})
        assert jobs.wait_idle(timeout=30)
        choices = {f"choice:{pid_for(X)}": "frontier", f"choice:{pid_for(Y)}": "frontier",
                   f"choice:{pid_for(Z)}": "skip"}
        r = c.post(f"/graphs/{SLUG}/grow/approve", data={"csrf_token": token(c), **choices})
        assert r.status_code == 303
        assert jobs.wait_idle(timeout=30)
        html = c.get(f"/graphs/{SLUG}/grow").text
        strip = c.get(f"/graphs/{SLUG}/grow/jobs").text
    assert membership(home) == {"lp_a": "member", "lp_b": "member", pid_for(X): "frontier",
                                pid_for(Y): "frontier"}
    assert sorted(oa_fetch) == [X, Y]  # only ticked papers were tried, open access only
    assert "open-access PDF from unpaywall" in strip and "needs-a-PDF" in strip
    needs = re.search(r"<h2[^>]*>Needs a PDF</h2>(.*?)(?=<h2|$)", html, re.S).group(1)
    assert f'href="https://doi.org/{Y}"' in needs and pid_for(X) not in needs


def test_an_approved_open_access_paper_has_its_references_parsed_after_it_converts(
        world, make_app, client_for, oa_fetch):
    home, chain = world
    app = make_app(chain=chain, pool=RefsPool())
    with client_for(app) as c:
        jobs = app.state.web.jobs
        assert jobs.wait_idle(timeout=30)
        c.post(f"/graphs/{SLUG}/grow/walk", data={"csrf_token": token(c), "seed": ["lp_a"], "depth": "1",
                                                  "cap": "10"})
        assert jobs.wait_idle(timeout=30)
        assert references_of(home, pid_for(X)) == []
        c.post(f"/graphs/{SLUG}/grow/approve", data={"csrf_token": token(c), f"choice:{pid_for(X)}": "frontier"})
        assert jobs.wait_idle(timeout=30)
        assert jobs.wait_idle(timeout=30)  # the conversion queued by the approval job
        messages = [j.message for j in jobs.jobs() if j.kind == "convert"]
    assert references_of(home, pid_for(X)) == [("resolved", pid_for(YEAST))]
    assert any("references of" in m for m in messages)


def test_member_is_refused_for_a_cited_only_paper_before_any_job_starts(world, make_app, client_for):
    home, chain = world
    app = make_app(chain=chain, pool=RefsPool())
    with client_for(app) as c:
        jobs = app.state.web.jobs
        assert jobs.wait_idle(timeout=30)
        c.post(f"/graphs/{SLUG}/grow/walk", data={"csrf_token": token(c), "seed": ["lp_a"], "depth": "1",
                                                  "cap": "10"})
        assert jobs.wait_idle(timeout=30)
        before = len(jobs.jobs())
        r = c.post(f"/graphs/{SLUG}/grow/approve", data={"csrf_token": token(c), f"choice:{pid_for(X)}": "member"})
        assert "sg-flash-error" in c.get(r.headers["location"]).text
        assert len(jobs.jobs()) == before
    assert pid_for(X) not in membership(home)


def test_a_bad_depth_is_refused_without_a_job(world, make_app, client_for):
    _, chain = world
    app = make_app(chain=chain)
    c = client_for(app)
    r = c.post(f"/graphs/{SLUG}/grow/walk", data={"csrf_token": token(c), "seed": ["lp_a"], "depth": "3",
                                                  "cap": "10"})
    assert "sg-flash-error" in c.get(r.headers["location"]).text
    assert app.state.web.jobs.jobs() == []


def test_after_convert_hooks_run_once_and_a_failing_hook_leaves_the_conversion_done(tmp_path):
    from library.test_files import TITLE, add_paper

    from seedgraph.library import files

    lib_dir = tmp_path / "home" / "library"
    lib = open_library_at(lib_dir)
    try:
        pid = add_paper(lib, paper_id="lp_00000000000000000001", title=TITLE)
        sf = files.attach_pdf(lib, pid, text_pdf(tmp_path / "p.pdf", [TITLE])).source_file_id
    finally:
        lib.close()
    seen = []

    def ok(lib, paper_ids):
        seen.append(paper_ids)
        return "followed up"

    def boom(lib, paper_ids):
        raise RuntimeError("invented follow-up failure")

    runner = JobRunner(lib_dir, pool=RefsPool())
    runner.convert(sf, then=ok)
    runner.convert(sf, then=boom)  # joins the queued job's hooks
    runner.start()
    try:
        assert runner.wait_idle(timeout=30)
    finally:
        runner.stop()
    (job,) = [j for j in runner.jobs() if j.kind == "convert"]
    assert job.state == "done" and seen == [(pid,)]
    assert "followed up" in job.message and "follow-up failed" in job.message
