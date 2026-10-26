"""Adding papers in the Library (design §6): dropped PDFs, a typed folder and pasted DOIs
run as JobRunner jobs (fake converter pool, fake provider chain), the job strip polls
while they run, and the upload path never fetches anything."""

from __future__ import annotations

import pytest

from library.metadata_support import FakeChain, openalex_record
from library.test_files import text_pdf
from seedgraph.library import ingest
from seedgraph.library.store import open_library_at
from seedgraph.llm.ollama import Reachability

from .pages_support import token
from .test_jobs import FakePool

TITLE = "Invented Ovens and the Morning Loaf"
DOI = "10.9999/oven.2031.1"


@pytest.fixture
def no_fetch(monkeypatch):
    """Any open-access fetch fails the test (the upload path must never fetch)."""
    import seedgraph.acquisition.fetch as fetch_mod

    async def boom(*a, **k):
        raise AssertionError("the upload path fetched something")

    monkeypatch.setattr(fetch_mod, "fetch_oa", boom)


def papers(home):
    lib = open_library_at(home / "library")
    try:
        return {r[0]: r[1] for r in lib.conn.execute(
            "SELECT p.paper_id, s.status FROM papers p JOIN paper_status s ON s.paper_id = p.paper_id")}
    finally:
        lib.close()


def spy_on(monkeypatch, name):
    calls = []
    real = getattr(ingest, name)

    def wrapped(*a, **k):
        calls.append((a, k))
        return real(*a, **k)

    monkeypatch.setattr(ingest, name, wrapped)
    return calls


def test_dropped_pdfs_are_ingested_as_a_job_converted_and_the_temp_copy_removed(
        home, tmp_path, make_app, client_for, monkeypatch, no_fetch):
    calls = spy_on(monkeypatch, "add_pdfs")
    pool = FakePool(size=1)
    chain = FakeChain()
    app = make_app(pool=pool, chain=chain)
    pdf = text_pdf(tmp_path / "loaf.pdf", [f"{TITLE}\nAda Inventor\n2031", "Invented body text."]).read_bytes()
    with client_for(app) as c:
        t = token(c)
        r = c.post("/library/add", headers={"X-CSRF-Token": t, "HX-Request": "true"},
                   files=[("files", ("my loaf.pdf", pdf, "application/pdf")),
                          ("files", ("notes.txt", b"not a pdf at all", "text/plain"))])
        assert r.status_code == 200 and 'id="sg-jobs"' in r.text and "<html" not in r.text
        assert "notes.txt: not a PDF" in r.text
        assert app.state.web.jobs.wait_idle(timeout=30)
        strip = c.get("/library/jobs").text
    assert len(calls) == 1
    (paths,), kw = calls[0][0][1:], calls[0][1]
    assert kw["source"] == "upload" and kw["chain"] is chain
    assert all(str(home / "tmp" / "upload") in p for p in paths)
    assert not any((home / "tmp" / "upload").rglob("*.pdf"))  # the upload copy is gone
    assert list(papers(home).values()) == ["held"]
    assert len(pool.calls) == 1  # the conversion ran
    assert "my loaf.pdf: new paper" in strip and "converted" in strip
    assert "hx-trigger" not in strip  # idle: polling stops


def test_an_upload_needs_the_csrf_header(make_app, client_for):
    c = client_for(make_app())
    r = c.post("/library/add", files=[("files", ("a.pdf", b"%PDF-1.4 x", "application/pdf"))])
    assert r.status_code == 403


def test_a_typed_folder_is_read_in_place(home, tmp_path, make_app, client_for, monkeypatch, no_fetch):
    calls = spy_on(monkeypatch, "add_folder")
    folder = tmp_path / "to-add"
    folder.mkdir()
    text_pdf(folder / "one.pdf", ["An invented folder paper", "x"])
    app = make_app(pool=FakePool(size=1), chain=FakeChain())
    with client_for(app) as c:
        r = c.post("/library/add-folder", data={"csrf_token": token(c), "folder": str(folder)})
        assert r.status_code == 303
        assert app.state.web.jobs.wait_idle(timeout=30)
    assert len(calls) == 1 and str(calls[0][0][1]) == str(folder)
    assert (folder / "one.pdf").is_file()  # read in place, never moved or deleted
    assert len(papers(home)) == 1


def test_a_folder_that_does_not_exist_starts_no_job(tmp_path, make_app, client_for):
    app = make_app()
    c = client_for(app)
    r = c.post("/library/add-folder", data={"csrf_token": token(c), "folder": str(tmp_path / "nope")})
    assert "not a folder" in c.get(r.headers["location"]).text
    assert app.state.web.jobs.jobs() == []


def test_pasted_dois_make_cited_only_papers_through_the_chain(home, make_app, client_for, monkeypatch):
    import seedgraph.acquisition.fetch as fetch_mod
    from seedgraph.acquisition.fetch import FetchResult

    fetched = []

    async def stub_fetch(record, *, chain=None, **kw):
        fetched.append(record.get("doi"))
        return FetchResult(status="stub")

    monkeypatch.setattr(fetch_mod, "fetch_oa", stub_fetch)
    calls = spy_on(monkeypatch, "add_dois")
    chain = FakeChain({DOI: openalex_record(DOI, TITLE, 2031)})
    app = make_app(pool=FakePool(size=1), chain=chain)
    with client_for(app) as c:
        r = c.post("/library/add-dois", data={"csrf_token": token(c), "dois": f"{DOI}\n\nnot a doi\n"})
        assert r.status_code == 303
        assert app.state.web.jobs.wait_idle(timeout=30)
        strip = c.get("/library/jobs").text
    assert calls[0][0][1] == [DOI, "not a doi"] and calls[0][1]["chain"] is chain
    assert fetched == [DOI]
    assert list(papers(home).values()) == ["cited_only"]
    assert "added as cited-only" in strip and "not a DOI" in strip


def test_a_ui_job_without_an_injected_chain_uses_the_offline_default_chain(home, make_app, client_for,
                                                                         monkeypatch):
    import seedgraph.acquisition.fetch as fetch_mod
    from seedgraph.acquisition.fetch import FetchResult
    from seedgraph.providers.fake import FakeProvider

    monkeypatch.setenv("SEEDGRAPH_FAKE_PROVIDERS", "1")

    async def stub_fetch(record, **kw):
        return FetchResult(status="stub")

    monkeypatch.setattr(fetch_mod, "fetch_oa", stub_fetch)
    calls = spy_on(monkeypatch, "add_dois")
    app = make_app(pool=FakePool(size=1))
    with client_for(app) as c:
        c.post("/library/add-dois", data={"csrf_token": token(c), "dois": DOI})
        assert app.state.web.jobs.wait_idle(timeout=30)
    chain = calls[0][1]["chain"]
    assert all(isinstance(p, FakeProvider) for p in chain.providers)
    with pytest.raises(Exception):
        chain.cache_engine.execute("SELECT 1")  # closed after the job


def test_the_job_strip_polls_every_two_seconds_while_a_job_runs(make_app, client_for):
    import threading

    app = make_app()
    gate = threading.Event()
    app.state.web.jobs.submit("ingest", lambda lib: gate.wait(5), label="invented slow job")
    c = client_for(app)  # the runner is not started: the job stays queued
    html = c.get("/library/jobs").text
    assert 'hx-trigger="every 2s"' in html and 'hx-get="/library/jobs"' in html
    assert "invented slow job" in html
    gate.set()


def test_ollama_metadata_runs_a_metadata_batch_with_the_chain(home, tmp_path, make_app, client_for, monkeypatch):
    import seedgraph.llm.ollama as ollama
    from seedgraph.work import ollama_worker

    monkeypatch.setenv("SEEDGRAPH_OLLAMA_MODEL", "invented:1b")
    monkeypatch.setattr(ollama, "check_reachable",
                        lambda base_url=None, **kw: Reachability(ok=True, base_url="http://localhost:11434"))
    seen = {}

    def fake_run(lib, batch_id, *, model, chain=None, **kw):
        seen.update(batch_id=batch_id, model=model, chain=chain)
        return ollama_worker.OllamaRunReport(batch_id=batch_id, model=model, worker_id="ollama-test")

    monkeypatch.setattr(ollama_worker, "run_ollama_worker", fake_run)
    chain = FakeChain()
    pdf = text_pdf(tmp_path / "u.pdf", ["Nothing identifiable here", "x"])
    lib = open_library_at(home / "library")
    try:
        ingest.add_pdfs(lib, [pdf])  # a new untitled paper with metadata_needed
    finally:
        lib.close()
    app = make_app(pool=FakePool(size=1), chain=chain)
    with client_for(app) as c:
        app.state.web.jobs.wait_idle(timeout=30)  # its conversion first
        c.post("/library/ollama-metadata", data={"csrf_token": token(c)})
        assert app.state.web.jobs.wait_idle(timeout=30)
        strip = c.get("/library/jobs").text
    assert seen["model"] == "invented:1b" and seen["chain"] is chain and seen["batch_id"].startswith("b_")
    assert "Ollama metadata" in strip and "failed" not in strip


def test_ollama_metadata_says_when_ollama_is_not_reachable(make_app, client_for, monkeypatch):
    import seedgraph.llm.ollama as ollama

    monkeypatch.setenv("SEEDGRAPH_OLLAMA_MODEL", "invented:1b")
    monkeypatch.setattr(ollama, "check_reachable", lambda base_url=None, **kw: Reachability(
        ok=False, base_url="http://localhost:11434", error="invented refusal"))
    app = make_app(pool=FakePool(size=1), chain=FakeChain())
    with client_for(app) as c:
        c.post("/library/ollama-metadata", data={"csrf_token": token(c)})
        assert app.state.web.jobs.wait_idle(timeout=30)
        strip = c.get("/library/jobs").text
    assert "Ollama is not reachable" in strip
